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
import soundfile as sf
import torch
import torch.ao.quantization as tq
import torch.nn as nn
import torchaudio

DATA_ROOT = os.path.expanduser(os.environ.get("VAD_DATA", "~/vad_data"))
LIBRIPARTY_DIR = os.path.join(DATA_ROOT, "LibriParty")
PRETRAINED_DIR = os.path.join(DATA_ROOT, "pretrained_vad")
CKPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")

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
    """[(wav_path, labels, total_frames)] for <root>/dataset/<split>/session_*, sorted by index.

    Labels are padded with non-speech to the full length of the wav, so the
    silence after the last utterance is scored too.
    """
    dirs = glob.glob(os.path.join(root, "dataset", split, "session_*"))
    dirs.sort(key=lambda d: int(d.rsplit("_", 1)[1]))
    out = []
    for d in dirs[:n]:
        name = os.path.basename(d)
        wav_path = os.path.join(d, f"{name}_mixture.wav")
        labels, _ = load_libriparty_gt(os.path.join(d, f"{name}.json"))
        total = sf.info(wav_path).frames // FRAME_STRIDE + 1
        labels = np.pad(labels, (0, max(0, total - len(labels))))[:total]
        out.append((wav_path, labels, total))
    return out


def sample_train_batch(sessions, batch, rng, clip_samples=CLIP_SAMPLES,
                       target_frames=CLIP_FRAMES, frame_stride=FRAME_STRIDE):
    """(wav [B, clip], labels [B, F]) of uniformly random 3 s clips.

    Uniform start positions keep the label prior of the data. (Preferring clips
    with speech, as the first pass did, made 93% of training frames speech against
    62% in the train split, and every model trained on it over-predicted speech.)
    `rng` is a random.Random, so every caller's sampling is seeded independently.
    """
    xs, ys = [], []
    for _ in range(batch):
        wav_path, gt, _ = rng.choice(sessions)
        wav = torchaudio.load(wav_path)[0][0]
        start = rng.randint(0, max(0, wav.numel() - clip_samples))
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


# ---------- QAT and fine-tuning ----------

class FakeQuantLinear(nn.Linear):
    """nn.Linear that sees symmetric per-tensor INT8-rounded weights (STE backward)."""

    def forward(self, x):
        w = self.weight
        scale = w.abs().max() / 127.0
        w_q = (w / scale).round().clamp(-128, 127) * scale
        return nn.functional.linear(x, w + (w_q - w).detach(), self.bias)


def replace_linear_with_fakequant(module):
    """Swap every nn.Linear under `module` for FakeQuantLinear in place; return the count."""
    n = 0
    for name, child in module.named_children():
        if type(child) is nn.Linear:
            fq = FakeQuantLinear(child.in_features, child.out_features, bias=child.bias is not None)
            fq.load_state_dict(child.state_dict())
            setattr(module, name, fq)
            n += 1
        else:
            n += replace_linear_with_fakequant(child)
    return n


def fakequant_dnn(dnn):
    """Copy of the DNN head with its nn.Linear layers fake-quantized (count in .n_fakequant)."""
    dnn = copy.deepcopy(dnn)
    dnn.n_fakequant = replace_linear_with_fakequant(dnn)
    return dnn


def finetune(vad, cnn, rnn, dnn, sessions, steps, lr, batch, seed, freeze_norm=True):
    """Fine-tune copies of (cnn, rnn, dnn) end to end with BCE; return them plus the loss log.

    freeze_norm=True keeps the stack in eval mode while gradients flow, so BatchNorm
    running statistics and dropout stay as pretrained. freeze_norm=False trains in
    train mode, where BatchNorm normalises with the statistics of the current batch.
    """
    cnn, rnn, dnn = (copy.deepcopy(m) for m in (cnn, rnn, dnn))
    params = [p for m in (cnn, rnn, dnn) for p in m.parameters()]
    for p in params:
        p.requires_grad_(True)
    for m in (cnn, rnn, dnn):
        m.train(not freeze_norm)
    rng = random.Random(seed)
    torch.manual_seed(seed)
    opt = torch.optim.Adam(params, lr=lr)
    forward = crdnn_logits(cnn, rnn, dnn)
    log = []
    for _ in range(steps):
        x, y = sample_train_batch(sessions, batch, rng)
        with torch.no_grad():
            feats = vad.mods.mean_var_norm(vad.mods.compute_features(x), torch.ones(x.shape[0]))
        loss = nn.functional.binary_cross_entropy_with_logits(forward(feats).squeeze(-1), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        log.append(loss.item())
    for m in (cnn, rnn, dnn):
        m.eval()
    return cnn, rnn, dnn, log


# ---------- training with early stopping, committed checkpoints ----------

def load_or_train(name, seed, args, train_fn, ckpt_dir=None):
    """Load <name>_seed<seed>.pt/.json if present, else run train_fn() and save both.

    The JSON log stores the recipe `args`; loading a checkpoint trained with a
    different recipe raises instead of silently reusing it.
    """
    ckpt_dir = ckpt_dir or CKPT_DIR
    pt = os.path.join(ckpt_dir, f"{name}_seed{seed}.pt")
    js = os.path.join(ckpt_dir, f"{name}_seed{seed}.json")
    args = json.loads(json.dumps(args))
    if os.path.exists(pt):
        with open(js) as f:
            log = json.load(f)
        if log["args"] != args:
            raise ValueError(f"{pt} was trained with recipe {log['args']}, not {args}")
        return torch.load(pt), log
    state, log = train_fn()
    log["args"] = args
    os.makedirs(ckpt_dir, exist_ok=True)
    torch.save(state, pt)
    with open(js, "w") as f:
        json.dump(log, f, indent=1)
    return state, log


def _train_early_stop(module, loss_fn, dev_f1_fn, rng, max_steps, eval_every, lr, patience):
    """Adam + cosine LR + grad-clip 1.0; keep the weights with the best dev F1."""
    params = [p for p in module.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_steps)
    best, best_sd, best_step, wait, curve = -1.0, None, 0, 0, []
    for step in range(1, max_steps + 1):
        module.train()
        loss = loss_fn(rng)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        if step % eval_every == 0:
            module.eval()
            f1 = dev_f1_fn(module)
            curve.append(dict(step=step, loss=loss.item(), dev_f1=f1))
            if f1 > best:
                best, best_sd, best_step, wait = f1, copy.deepcopy(module.state_dict()), step, 0
            else:
                wait += 1
                if wait >= patience:
                    break
    module.load_state_dict(best_sd)
    module.eval()
    return best_sd, dict(curve=curve, best_dev_f1=best, best_step=best_step, steps_run=step)


def _feats(vad, x):
    with torch.no_grad():
        return vad.mods.mean_var_norm(vad.mods.compute_features(x), torch.ones(x.shape[0]))


def build_lstm_like(rnn):
    """Fresh nn.LSTM with the dimensions of the GRU inside SpeechBrain's RNN wrapper."""
    gru = next(m for m in rnn.modules() if isinstance(m, nn.GRU))
    return nn.LSTM(gru.input_size, gru.hidden_size, num_layers=gru.num_layers,
                   bidirectional=gru.bidirectional, dropout=gru.dropout, batch_first=True)


def train_lstm_only(vad, train, dev, seed, max_steps=3000, eval_every=250, batch=16,
                    lr=5e-4, patience=6):
    """GRU -> fresh LSTM, trained with BCE while the pretrained CNN and DNN stay frozen."""
    torch.manual_seed(seed)
    lstm = build_lstm_like(vad.mods.rnn)
    cnn, dnn = vad.mods.cnn, vad.mods.dnn

    def loss_fn(rng):
        x, y = sample_train_batch(train, batch, rng)
        with torch.no_grad():
            o = cnn(_feats(vad, x))
        o, _ = lstm(o.reshape(o.shape[0], o.shape[1], -1))
        return nn.functional.binary_cross_entropy_with_logits(dnn(o).squeeze(-1), y)

    return _train_early_stop(lstm, loss_fn,
                             lambda m: eval_f1(vad, crdnn_logits(cnn, m, dnn), dev)["f1"],
                             random.Random(seed), max_steps, eval_every, lr, patience)


class Student(nn.Module):
    """Compact CRDNN on mel feats: 2 conv blocks (8, 16 ch) + 2-layer biLSTM(16) + linear head."""

    def __init__(self, n_mels=40, hidden=16):
        super().__init__()

        def blk(ci, co):
            return nn.Sequential(
                nn.Conv2d(ci, co, 3, padding=1), nn.LeakyReLU(0.01),
                nn.Conv2d(co, co, 3, padding=1), nn.LeakyReLU(0.01),
                nn.MaxPool2d((1, 2)))
        self.cnn = nn.Sequential(blk(1, 8), blk(8, 16))           # freq 40 -> 20 -> 10
        self.rnn = nn.LSTM(16 * (n_mels // 4), hidden, num_layers=2,
                           bidirectional=True, batch_first=True)
        self.head = nn.Linear(2 * hidden, 1)

    def forward(self, feats):                                      # [B, T, 40]
        x = self.cnn(feats.unsqueeze(1))                           # [B, C, T, F]
        B, C, T, Fq = x.shape
        x, _ = self.rnn(x.permute(0, 2, 1, 3).reshape(B, T, C * Fq))
        return self.head(x)


def distill_loss(student_logits, teacher_logits, labels, alpha, temp):
    """alpha * T^2 * BCE(s/T, sigmoid(t/T)) + (1 - alpha) * BCE(s, labels)."""
    soft = nn.functional.binary_cross_entropy_with_logits(
        student_logits / temp, torch.sigmoid(teacher_logits / temp)) * temp ** 2
    hard = nn.functional.binary_cross_entropy_with_logits(student_logits, labels)
    return alpha * soft + (1 - alpha) * hard


def train_student(vad, train, dev, seed, alpha, temp=2.0, max_steps=2500, eval_every=250,
                  batch=16, lr=5e-4, patience=6):
    """Train a Student from scratch; alpha=0.5 distils from the FP32 teacher, alpha=0 is hard labels only."""
    torch.manual_seed(seed)
    student = Student()
    teacher = crdnn_logits(vad.mods.cnn, vad.mods.rnn, vad.mods.dnn)

    def loss_fn(rng):
        x, y = sample_train_batch(train, batch, rng)
        f = _feats(vad, x)
        with torch.no_grad():
            t = teacher(f).squeeze(-1)
        return distill_loss(student(f).squeeze(-1), t, y, alpha, temp)

    return _train_early_stop(student, loss_fn, lambda m: eval_f1(vad, m, dev)["f1"],
                             random.Random(seed), max_steps, eval_every, lr, patience)


# ---------- recipes shared by the notebook and any pre-training run ----------

SAMPLER = "uniform-3s"   # recorded in checkpoint recipes; see sample_train_batch
DEV_LABELS = "full-wav"  # recorded in checkpoint recipes; see load_sessions
SEEDS = (0, 1, 2)
N_DEV = 10   # dev sessions used for early stopping
LSTM_RECIPE = dict(max_steps=3000, eval_every=250, batch=16, lr=5e-4, patience=6)
STUDENT_RECIPE = dict(temp=2.0, max_steps=2500, eval_every=250, batch=16, lr=5e-4, patience=6)


def get_lstm(vad, train, dev, seed):
    """E3 LSTM for `seed` (trained once, then loaded from checkpoints/)."""
    sd, log = load_or_train("e3_lstm", seed, {**LSTM_RECIPE, "n_dev": len(dev), "sampler": SAMPLER, "dev_labels": DEV_LABELS},
                            lambda: train_lstm_only(vad, train, dev, seed, **LSTM_RECIPE))
    lstm = build_lstm_like(vad.mods.rnn)
    lstm.load_state_dict(sd)
    return lstm.eval(), log


def get_student(vad, train, dev, seed, alpha):
    """E5 student (alpha=0.5, distilled) or hard-label control (alpha=0) for `seed`."""
    name = "e5_student" if alpha > 0 else "e5_hard"
    sd, log = load_or_train(name, seed, {**STUDENT_RECIPE, "alpha": alpha, "n_dev": len(dev), "sampler": SAMPLER, "dev_labels": DEV_LABELS},
                            lambda: train_student(vad, train, dev, seed, alpha, **STUDENT_RECIPE))
    st = Student()
    st.load_state_dict(sd)
    return st.eval(), log


def strip_fakequant(module):
    """Copy with every FakeQuantLinear turned back into a plain nn.Linear (same weights).

    quantize_dynamic matches exact types, so a FakeQuantLinear would be silently
    left in FP32.
    """
    module = copy.deepcopy(module)

    def walk(m):
        for name, child in m.named_children():
            if isinstance(child, FakeQuantLinear):
                lin = nn.Linear(child.in_features, child.out_features, bias=child.bias is not None)
                lin.load_state_dict(child.state_dict())
                setattr(m, name, lin)
            else:
                walk(child)
    walk(module)
    return module


QAT_RECIPE = dict(steps=60, lr=1e-5, batch=16)           # fixed recipe: norm frozen
QAT_ORIGINAL_RECIPE = dict(steps=60, lr=1e-5, batch=1)   # first-pass recipe: train mode
