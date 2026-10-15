import copy
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


# ---------- model wrapper, eval path, quant helpers (pretrained model + eval data) ----------

DATA_ROOT = vadlib.DATA_ROOT
needs_data = pytest.mark.skipif(
    not (vadlib.os.path.isdir(vadlib.os.path.join(DATA_ROOT, "pretrained_vad"))
         and vadlib.os.path.isdir(vadlib.os.path.join(DATA_ROOT, "LibriParty", "dataset", "eval"))),
    reason="pretrained model or LibriParty eval split not under ~/vad_data")


@pytest.fixture(scope="module")
def vad():
    return vadlib.load_vad()


@pytest.fixture(scope="module")
def eval3():
    return vadlib.load_sessions(vadlib.LIBRIPARTY_DIR, "eval", 3)


@needs_data
def test_wrapper_matches_native_chunk(vad, eval3):
    wav = vadlib.torchaudio.load(eval3[0][0])[0][:, :48000]
    native = vad.get_speech_prob_chunk(wav)
    ours = torch.sigmoid(vadlib.CRDNNWrapper(vad)(wav))
    assert torch.equal(native, ours)


@needs_data
def test_swapped_eval_path_reproduces_native_probs(vad, eval3):
    native = vad.get_speech_prob_file(eval3[0][0])
    with vadlib.swapped_chunk_fn(vad, vadlib.crdnn_logits(vad.mods.cnn, vad.mods.rnn, vad.mods.dnn)):
        ours = vad.get_speech_prob_file(eval3[0][0])
    assert (native - ours).abs().max().item() == 0.0
    assert "get_speech_prob_chunk" not in vars(vad)          # patch removed


@needs_data
def test_swapped_eval_f1_equals_native_f1(vad, eval3):
    native = vadlib.eval_native_f1(vad, eval3)
    ours = vadlib.eval_f1(vad, vadlib.crdnn_logits(vad.mods.cnn, vad.mods.rnn, vad.mods.dnn), eval3)
    assert native["per_session"] == ours["per_session"]


@needs_data
def test_static_ptq_cnn_is_int8_drop_in(vad, eval3):
    feats = vadlib.build_calib_feats(vad, eval3, n_clips=2, seed=0)
    q = vadlib.static_ptq_module(vad.mods.cnn, feats)
    assert any("Quantized" in type(m).__name__ or type(m).__module__.startswith("torch.ao.nn.quantized")
               for m in q.modules())
    with torch.no_grad():
        assert q(feats[0]).shape == vad.mods.cnn(feats[0]).shape
    assert vadlib.modules_size_mb(cnn=q) < vadlib.modules_size_mb(cnn=vad.mods.cnn)


def test_dynamic_quantize_lstm():
    lstm = torch.nn.LSTM(8, 4, num_layers=2, bidirectional=True, batch_first=True)
    q = vadlib.dynamic_quantize_rnn(lstm)
    assert type(q).__module__.startswith("torch.ao.nn.quantized.dynamic")
    out, _ = q(torch.randn(1, 5, 8))
    assert out.shape == (1, 5, 8)


def test_measure_latency_restores_threads():
    before = torch.get_num_threads()
    r = vadlib.measure_latency(torch.nn.Linear(4, 4), torch.randn(1, 4), n_warmup=2, n_runs=5)
    assert set(r) == {"median", "std", "p95"} and torch.get_num_threads() == before


# ---------- QAT / fine-tuning ----------

def _bn_buffers(module):
    return {k: v.clone() for k, v in module.state_dict().items() if "running" in k}


@needs_data
def test_fakequant_scope_is_the_three_dnn_linears(vad):
    assert vadlib.fakequant_dnn(vad.mods.dnn).n_fakequant == 3
    whole = copy.deepcopy(vadlib.CRDNNWrapper(vad))
    assert vadlib.replace_linear_with_fakequant(whole) == 3   # CNN/GRU hold no nn.Linear


@needs_data
def test_finetune_frozen_norm_keeps_bn_stats(vad, eval3):
    before = _bn_buffers(vad.mods.dnn)
    w_before = copy.deepcopy(vad.mods.dnn.state_dict())
    cnn, rnn, dnn, log = vadlib.finetune(vad, vad.mods.cnn, vad.mods.rnn, vad.mods.dnn, eval3,
                                         steps=2, lr=1e-3, batch=2, seed=0, freeze_norm=True)
    after = _bn_buffers(dnn)
    assert all(torch.equal(before[k], after[k]) for k in before)
    assert any(not torch.equal(w_before[k], v) for k, v in dnn.state_dict().items() if "running" not in k)
    assert all(torch.equal(w_before[k], v) for k, v in vad.mods.dnn.state_dict().items())  # original untouched
    assert len(log) == 2


@needs_data
def test_finetune_train_mode_moves_bn_stats(vad, eval3):
    before = _bn_buffers(vad.mods.dnn)
    _, _, dnn, _ = vadlib.finetune(vad, vad.mods.cnn, vad.mods.rnn, vad.mods.dnn, eval3,
                                   steps=2, lr=1e-3, batch=1, seed=0, freeze_norm=False)
    after = _bn_buffers(dnn)
    assert any(not torch.equal(before[k], after[k]) for k in before)
    assert not vad.mods.dnn.training


# ---------- training: checkpoints, distillation loss ----------

def test_load_or_train_trains_once_then_loads(tmp_path):
    calls = []

    def train_fn():
        calls.append(1)
        return torch.nn.Linear(2, 1).state_dict(), {"best_dev_f1": 0.5}

    args = {"lr": 1e-3, "steps": 10}
    sd1, log1 = vadlib.load_or_train("m", 0, args, train_fn, ckpt_dir=tmp_path)
    sd2, log2 = vadlib.load_or_train("m", 0, args, train_fn, ckpt_dir=tmp_path)
    assert len(calls) == 1 and log2 == log1 and log1["args"] == args
    assert all(torch.equal(sd1[k], sd2[k]) for k in sd1)
    assert (tmp_path / "m_seed0.pt").exists() and (tmp_path / "m_seed0.json").exists()


def test_load_or_train_rejects_recipe_mismatch(tmp_path):
    train_fn = lambda: (torch.nn.Linear(2, 1).state_dict(), {})
    vadlib.load_or_train("m", 0, {"lr": 1e-3}, train_fn, ckpt_dir=tmp_path)
    with pytest.raises(ValueError, match="recipe"):
        vadlib.load_or_train("m", 0, {"lr": 5e-4}, train_fn, ckpt_dir=tmp_path)


def test_distill_loss_alpha0_is_plain_bce():
    torch.manual_seed(0)
    s, t, y = torch.randn(2, 9), torch.randn(2, 9), (torch.rand(2, 9) > 0.5).float()
    assert torch.equal(vadlib.distill_loss(s, t, y, alpha=0.0, temp=2.0),
                       torch.nn.functional.binary_cross_entropy_with_logits(s, y))


def test_distill_loss_soft_term_minimised_at_teacher():
    t, y = torch.randn(1, 50), torch.zeros(1, 50)
    at_teacher = vadlib.distill_loss(t.clone(), t, y, alpha=1.0, temp=2.0)
    off = vadlib.distill_loss(t + 1.0, t, y, alpha=1.0, temp=2.0)
    assert at_teacher < off


def test_student_shape_and_size():
    st = vadlib.Student()
    assert st(torch.randn(2, 301, 40)).shape == (2, 301, 1)
    assert sum(p.numel() for p in st.parameters()) == 33369


def test_fakequant_must_be_stripped_before_dynamic_quant():
    head = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.ReLU(), torch.nn.Linear(4, 1))
    fq = copy.deepcopy(head)
    assert vadlib.replace_linear_with_fakequant(fq) == 2
    n_q = lambda m: sum(type(x).__module__.startswith("torch.ao.nn.quantized.dynamic") for x in m.modules())
    assert n_q(vadlib.dynamic_quantize_linear(fq)) == 0          # subclass is silently skipped
    stripped = vadlib.strip_fakequant(fq)
    assert n_q(vadlib.dynamic_quantize_linear(stripped)) == 2
    x = torch.randn(3, 4)
    assert torch.equal(stripped(x), head(x))                      # same weights, plain forward


def test_sample_train_batch_has_no_speech_preference(tmp_path):
    wav_path = tmp_path / "w.wav"
    sf.write(wav_path, np.zeros(16000 * 60, dtype=np.float32), 16000)
    gt = np.zeros(6001, dtype=int)
    gt[:1200] = 1                                        # 20% of the session is speech
    _, y = vadlib.sample_train_batch([(str(wav_path), gt, len(gt))], 400, random.Random(0))
    assert abs(y.mean().item() - 0.2) < 0.05             # label prior matches the data


def test_old_sampler_checkpoints_are_rejected(tmp_path):
    train_fn = lambda: (torch.nn.Linear(2, 1).state_dict(), {})
    vadlib.load_or_train("m", 0, {"lr": 1e-3}, train_fn, ckpt_dir=tmp_path)
    with pytest.raises(ValueError, match="recipe"):
        vadlib.load_or_train("m", 0, {"lr": 1e-3, "sampler": vadlib.SAMPLER}, train_fn, ckpt_dir=tmp_path)


def test_load_sessions_labels_cover_the_whole_wav(tmp_path):
    d = tmp_path / "dataset" / "eval" / "session_0"
    d.mkdir(parents=True)
    sf.write(d / "session_0_mixture.wav", np.zeros(16000 * 10, dtype=np.float32), 16000)
    (d / "session_0.json").write_text(json.dumps({"1": [{"start": 1.0, "stop": 4.0}]}))
    (wav, labels, total), = vadlib.load_sessions(str(tmp_path), "eval")
    assert total == len(labels) == 16000 * 10 // 160 + 1    # frames of the whole wav
    assert labels[100:400].all() and labels[400:].sum() == 0  # silent tail after the last utterance is scored
