"""Helpers for the CRDNN VAD compression study (imported by the notebook and tests)."""
import glob
import json
import os

import numpy as np
import scipy.stats
import torch
import torchaudio

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
