"""Helpers for the CRDNN VAD compression study (imported by the notebook and tests)."""
import contextlib
import copy
import glob
import json
import os
import platform
import random
import tempfile
import time

import numpy as np
import scipy.stats
import torch
import torch.ao.quantization as tq
import torch.nn as nn
import torchaudio

DATA_ROOT = os.path.expanduser(os.environ.get("VAD_DATA", "~/vad_data"))
LIBRIPARTY_DIR = os.path.join(DATA_ROOT, "LibriParty")
PRETRAINED_DIR = os.path.join(DATA_ROOT, "pretrained_vad")

torch.backends.quantized.engine = (
    "qnnpack" if platform.machine() in ("arm64", "aarch64") else "fbgemm")

FPS = 100            # label / model frame rate (10 ms hop)
FRAME_STRIDE = 160   # samples per frame at 16 kHz
CLIP_SAMPLES = 48000  # 3 s training clips
CLIP_FRAMES = 301     # model frames for a 3 s clip


# ---------- data ----------

def load_libriparty_gt(json_path, fps=FPS):
    """Frame labels (1 = any speaker talking) and frame count for one session JSON.

    Every top-level key of a LibriParty session JSON is a speaker ID; anything
    else would be silently counted as speech, so it is rejected.
    """
    with open(json_path) as f:
        data = json.load(f)
    for key in data:
        if not key.isdigit():
            raise ValueError(f"non-speaker key {key!r} in {json_path}")
    segments = [(u["start"], u["stop"]) for utts in data.values() for u in utts]
    total = int(max(s[1] for s in segments) * fps) + 1
    labels = np.zeros(total, dtype=int)
    for start, stop in segments:
        labels[round(start * fps):min(round(stop * fps), total)] = 1
    return labels, total


def load_sessions(root, split, n=None):
    """[(wav_path, labels, total_frames)] for <root>/dataset/<split>/session_*, sorted by index."""
    dirs = glob.glob(os.path.join(root, "dataset", split, "session_*"))
    dirs.sort(key=lambda d: int(d.rsplit("_", 1)[1]))
    out = []
    for d in dirs[:n]:
        name = os.path.basename(d)
        labels, total = load_libriparty_gt(os.path.join(d, f"{name}.json"))
        out.append((os.path.join(d, f"{name}_mixture.wav"), labels, total))
    return out


def sample_train_batch(sessions, batch, rng, clip_samples=CLIP_SAMPLES,
                       target_frames=CLIP_FRAMES, frame_stride=FRAME_STRIDE):
    """(wav [B, clip], labels [B, F]) of random 3 s clips, preferring speech ratio > 0.2.

    `rng` is a random.Random, so every caller's sampling is seeded independently.
    """
    xs, ys = [], []
    for _ in range(batch):
        wav_path, gt, _ = rng.choice(sessions)
        wav = torchaudio.load(wav_path)[0][0]
        start = 0
        if wav.numel() > clip_samples:
            for _try in range(8):
                start = rng.randint(0, wav.numel() - clip_samples)
                f0 = start // frame_stride
                if np.mean(gt[f0:f0 + target_frames]) > 0.2:
                    break
        x = wav[start:start + clip_samples]
        x = torch.cat([x, torch.zeros(clip_samples - x.numel())])
        f0 = start // frame_stride
        seg = gt[f0:f0 + target_frames]
        lab = np.pad(seg, (0, target_frames - len(seg)))
        xs.append(x)
        ys.append(torch.tensor(lab, dtype=torch.float32))
    return torch.stack(xs), torch.stack(ys)


# ---------- metrics ----------

def boundaries_to_frame_labels(boundaries, total_frames, fps=FPS):
    """Frame labels from [N, 2] speech boundaries in seconds (rounded like the reference labels)."""
    labels = np.zeros(total_frames, dtype=int)
    for start, stop in boundaries.tolist():
        labels[round(start * fps):min(round(stop * fps), total_frames)] = 1
    return labels


def compute_f1(pred, ref):
    tp = int(np.sum((pred == 1) & (ref == 1)))
    fp = int(np.sum((pred == 1) & (ref == 0)))
    fn = int(np.sum((pred == 0) & (ref == 1)))
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return f1, prec, rec


def paired_delta(a, b):
    """Per-session paired comparison of b against a: mean(b - a), 95% t-CI, Wilcoxon p."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = b - a
    n = len(d)
    mean = float(d.mean())
    if np.all(d == 0):
        return dict(mean=0.0, ci_low=0.0, ci_high=0.0, p_wilcoxon=1.0, n=n)
    lo, hi = scipy.stats.t.interval(0.95, n - 1, loc=mean, scale=scipy.stats.sem(d))
    p = float(scipy.stats.wilcoxon(b, a).pvalue)
    return dict(mean=mean, ci_low=float(lo), ci_high=float(hi), p_wilcoxon=p, n=n)


# ---------- model and evaluation path ----------

def load_vad(savedir=PRETRAINED_DIR):
    from speechbrain.inference.VAD import VAD
    vad = VAD.from_hparams(source="speechbrain/vad-crdnn-libriparty", savedir=savedir)
    vad.mods.eval()
    return vad


def crdnn_logits(cnn, rnn, dnn):
    """feats [B, T, 40] -> logits [B, T, 1] through the given CRDNN sub-modules."""
    def fn(feats):
        o = cnn(feats)
        o = o.reshape(o.shape[0], o.shape[1], o.shape[2] * o.shape[3])
        o, _ = rnn(o)
        return dnn(o)
    return fn


class CRDNNWrapper(nn.Module):
    """wav [B, T] -> logits [B, T_frames, 1]: SpeechBrain front-end + (optionally swapped) CRDNN."""

    def __init__(self, vad, cnn=None, rnn=None, dnn=None):
        super().__init__()
        self.compute_features = vad.mods.compute_features
        self.mean_var_norm = vad.mods.mean_var_norm
        self.cnn = cnn if cnn is not None else vad.mods.cnn
        self.rnn = rnn if rnn is not None else vad.mods.rnn
        self.dnn = dnn if dnn is not None else vad.mods.dnn

    def forward(self, wav):
        feats = self.mean_var_norm(self.compute_features(wav), torch.ones(wav.shape[0]))
        return crdnn_logits(self.cnn, self.rnn, self.dnn)(feats)


class FeatsToLogitsWav(nn.Module):
    """wav -> logits for any feats->logits module (the distilled student)."""

    def __init__(self, vad, model):
        super().__init__()
        self.compute_features = vad.mods.compute_features
        self.mean_var_norm = vad.mods.mean_var_norm
        self.model = model

    def forward(self, wav):
        return self.model(self.mean_var_norm(self.compute_features(wav), torch.ones(wav.shape[0])))


@contextlib.contextmanager
def swapped_chunk_fn(vad, feats_to_logits):
    """Route SpeechBrain's per-chunk neural call through `feats_to_logits`.

    Mirrors VAD.get_speech_prob_chunk line for line (front-end, wav_lens, sigmoid),
    so get_speech_prob_file's chunking, thresholding and boundaries stay SpeechBrain's.
    """
    def chunk(wavs, wav_lens=None):
        if wavs.dim() == 1:
            wavs = wavs.unsqueeze(0)
        if wav_lens is None:
            wav_lens = torch.ones(wavs.shape[0])
        feats = vad.mods.compute_features(wavs.float())
        feats = vad.mods.mean_var_norm(feats, wav_lens)
        return torch.sigmoid(feats_to_logits(feats))
    vad.get_speech_prob_chunk = chunk
    try:
        yield
    finally:
        del vad.get_speech_prob_chunk


def _f1_over_sessions(vad, sessions):
    f1s, ps, rs = [], [], []
    for wav_path, ref, total in sessions:
        with torch.no_grad():
            prob = vad.get_speech_prob_file(wav_path)
            boundaries = vad.get_boundaries(vad.apply_threshold(prob).float(),
                                            output_value="seconds")
        pred = boundaries_to_frame_labels(boundaries, total)
        n = min(len(pred), len(ref))
        f1, p, r = compute_f1(pred[:n], ref[:n])
        f1s.append(f1); ps.append(p); rs.append(r)
    return dict(f1=float(np.mean(f1s)), prec=float(np.mean(ps)), rec=float(np.mean(rs)),
                per_session=f1s)


def eval_native_f1(vad, sessions):
    """Frame-level F1 through SpeechBrain's own pipeline, unmodified."""
    return _f1_over_sessions(vad, sessions)


def eval_f1(vad, feats_to_logits, sessions):
    """Frame-level F1 with the neural part swapped for `feats_to_logits`."""
    with swapped_chunk_fn(vad, feats_to_logits):
        return _f1_over_sessions(vad, sessions)


# ---------- quantization, size, latency ----------

class _StaticQuantWrapper(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.quant = tq.QuantStub()
        self.module = module
        self.dequant = tq.DeQuantStub()

    def forward(self, x):
        return self.dequant(self.module(self.quant(x)))


def static_ptq_module(module, calib_inputs):
    """Eager-mode static INT8 PTQ (prepare -> calibrate -> convert); float in, float out."""
    m = _StaticQuantWrapper(copy.deepcopy(module)).eval()
    m.qconfig = tq.get_default_qconfig(torch.backends.quantized.engine)
    tq.prepare(m, inplace=True)
    with torch.no_grad():
        for x in calib_inputs:
            m(x)
    tq.convert(m, inplace=True)
    return m


def dynamic_quantize_linear(module):
    """Dynamic INT8 quantization of every nn.Linear inside `module` (copy)."""
    return tq.quantize_dynamic(copy.deepcopy(module), {nn.Linear}, dtype=torch.qint8)


def dynamic_quantize_rnn(rnn):
    """Dynamic INT8 quantization of a bare nn.LSTM.

    quantize_dynamic only swaps matching *children*, so the RNN is wrapped first.
    """
    holder = nn.Module()
    holder.rnn = copy.deepcopy(rnn)
    return tq.quantize_dynamic(holder.eval(), {nn.LSTM}, dtype=torch.qint8).rnn


def build_calib_feats(vad, sessions, n_clips, seed):
    """Normalised mel features of `n_clips` random 3 s clips (calibration inputs for a CNN)."""
    rng = random.Random(seed)
    feats = []
    for _ in range(n_clips):
        wav = torchaudio.load(rng.choice(sessions)[0])[0][0]
        start = rng.randint(0, max(0, wav.numel() - CLIP_SAMPLES))
        wav = wav[start:start + CLIP_SAMPLES].unsqueeze(0)
        with torch.no_grad():
            feats.append(vad.mods.mean_var_norm(vad.mods.compute_features(wav), torch.ones(1)))
    return feats


def modules_size_mb(**modules):
    """On-disk size (MiB) of the given modules' state_dicts saved together."""
    with tempfile.NamedTemporaryFile(suffix=".pt") as f:
        torch.save({k: m.state_dict() for k, m in modules.items()}, f.name)
        return os.path.getsize(f.name) / 1024 ** 2


def measure_latency(model, x, n_warmup=30, n_runs=100):
    """Single-thread CPU forward latency in ms: median, std, p95 (thread count restored)."""
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    model.eval()
    try:
        with torch.no_grad():
            for _ in range(n_warmup):
                model(x)
            times = []
            for _ in range(n_runs):
                t0 = time.perf_counter()
                model(x)
                times.append((time.perf_counter() - t0) * 1000)
    finally:
        torch.set_num_threads(threads)
    a = np.array(times)
    return dict(median=float(np.median(a)), std=float(a.std()), p95=float(np.percentile(a, 95)))
