import json
import random

import numpy as np
import pytest
import scipy.stats
import soundfile as sf
import torch

import vadlib


# ---------- labels, sessions, metrics, stats ----------

def test_labels_union_of_speaker_utterances(tmp_path):
    p = tmp_path / "s.json"
    p.write_text(json.dumps({
        "1": [{"start": 0.10, "stop": 0.20}, {"start": 0.50, "stop": 0.60}],
        "2": [{"start": 0.15, "stop": 0.30}],
    }))
    labels, total = vadlib.load_libriparty_gt(p)
    assert total == 61
    assert labels[10:30].all() and labels[50:60].all()
    assert labels[:10].sum() == 0 and labels[30:50].sum() == 0 and labels[60:].sum() == 0


def test_labels_reject_non_speaker_key(tmp_path):
    p = tmp_path / "s.json"
    p.write_text(json.dumps({"1": [{"start": 0.1, "stop": 0.2}],
                             "noises": [{"start": 0.0, "stop": 5.0}]}))
    with pytest.raises(ValueError, match="noises"):
        vadlib.load_libriparty_gt(p)


def test_compute_f1_hand_case():
    pred = np.array([1, 1, 0, 0, 1])
    ref = np.array([1, 0, 0, 1, 1])
    f1, prec, rec = vadlib.compute_f1(pred, ref)       # tp=2 fp=1 fn=1
    assert (prec, rec, f1) == (pytest.approx(2 / 3), pytest.approx(2 / 3), pytest.approx(2 / 3))


def test_boundaries_to_frame_labels():
    b = torch.tensor([[0.10, 0.20], [0.95, 2.00]])
    lab = vadlib.boundaries_to_frame_labels(b, total_frames=100)
    assert lab[10:20].all() and lab[95:].all()
    assert lab[:10].sum() == 0 and lab[20:95].sum() == 0


def test_sample_train_batch_labels_align_with_clip(tmp_path):
    n = 16000 * 20
    wav = (np.arange(n) / 1e6).astype(np.float32)        # sample value encodes its index
    wav_path = tmp_path / "w.wav"
    sf.write(wav_path, wav, 16000, subtype="FLOAT")
    gt = (np.arange(n // 160 + 1) % 7 == 0).astype(int)  # irregular pattern
    gt[::3] = 1
    x, y = vadlib.sample_train_batch([(str(wav_path), gt, len(gt))], batch=8,
                                     rng=random.Random(0))
    assert x.shape == (8, 48000) and y.shape == (8, 301)
    for i in range(8):
        start = round(float(x[i, 0]) * 1e6)
        np.testing.assert_array_equal(y[i].numpy(), gt[start // 160:start // 160 + 301])


def test_sample_train_batch_is_seeded(tmp_path):
    wav_path = tmp_path / "w.wav"
    sf.write(wav_path, np.random.RandomState(0).randn(16000 * 10).astype(np.float32) * 0.1, 16000)
    gt = np.ones(1001, dtype=int)
    s = [(str(wav_path), gt, len(gt))]
    a = vadlib.sample_train_batch(s, 4, random.Random(3))
    b = vadlib.sample_train_batch(s, 4, random.Random(3))
    assert torch.equal(a[0], b[0])


def test_paired_delta_matches_scipy():
    rng = np.random.RandomState(0)
    a = rng.rand(30)
    b = a + 0.01 + rng.randn(30) * 0.005
    d = vadlib.paired_delta(a, b)
    lo, hi = scipy.stats.t.interval(0.95, 29, loc=np.mean(b - a), scale=scipy.stats.sem(b - a))
    assert d["mean"] == pytest.approx(np.mean(b - a))
    assert (d["ci_low"], d["ci_high"]) == (pytest.approx(lo), pytest.approx(hi))
    assert d["p_wilcoxon"] == pytest.approx(scipy.stats.wilcoxon(b, a).pvalue)
    assert d["n"] == 30


def test_paired_delta_identical_inputs():
    a = np.array([0.9, 0.8, 0.95])
    d = vadlib.paired_delta(a, a)
    assert d["mean"] == 0.0 and d["p_wilcoxon"] == 1.0
