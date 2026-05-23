# -*- coding: utf-8 -*-
"""
VAD Compression Experiment — Colab Version
===========================================
Benchmarks four compression strategies for the SpeechBrain CRDNN-based
Voice Activity Detection model (speechbrain/vad-crdnn-libriparty) on the
LibriParty evaluation set.

Compression strategies evaluated
---------------------------------
- FP32 Baseline : original pre-trained weights, no compression.
- PTQ           : post-training dynamic quantization of all nn.Linear layers
                  to INT8 using torch.quantization.quantize_dynamic.
- QAT           : quantization-aware training — FakeQuant (STE) is injected
                  into the DNN sub-module, the model is fine-tuned for 60
                  steps, then converted to INT8.
- PTQ+QAT       : more aggressive variant — FakeQuant is injected into every
                  sub-module (CNN, RNN, DNN), fine-tuned for 60 steps, then
                  dynamic-quantized.

Metrics reported per strategy
------------------------------
- On-disk model size (MB) — serialised state_dict of mods['model'] only.
- Inference latency (ms)  — median and P95 over 100 runs, single-threaded CPU.
- F1 score                — frame-level speech/non-speech on 20 LibriParty
                            eval sessions, plus per-session std and SEM.

Dependencies
------------
    pip install git+https://github.com/speechbrain/speechbrain.git@develop
    torch, torchaudio, numpy, matplotlib, pandas

Google Drive
------------
Results (CSV, plot) and cached datasets are written to
    /content/drive/MyDrive/VAD_Compression/
"""

# ---------------------------------------------------------------------------
# Setup — mount Google Drive and install SpeechBrain
# ---------------------------------------------------------------------------
import subprocess
import os
from google.colab import drive

drive.mount('/content/drive')

# All artefacts (model checkpoints, results, cached dataset) persist here.
SAVE_DIR = "/content/drive/MyDrive/VAD_Compression"
os.makedirs(SAVE_DIR, exist_ok=True)

BRANCH = 'develop'
subprocess.run(
    f"pip install -q git+https://github.com/speechbrain/speechbrain.git@{BRANCH}",
    shell=True,
)
print(f"Results will be saved at: {SAVE_DIR}")

# ---------------------------------------------------------------------------
# Data Configuration
# ---------------------------------------------------------------------------
# DATA_MODE controls which audio source is used for evaluation:
#   "demo"  — single example WAV file (no dataset download, quick smoke test)
#   "small" — LibriParty dataset, first 20 eval sessions (recommended)
#   "full"  — full LibriParty archive (~10 GB)
DATA_MODE = "small"

# Dataset resolution order (avoids redundant downloads):
#   1. Drive path  — persists across Colab sessions.
#   2. Local /content — fast disk, available within the current session.
#   3. tar.gz in Drive — extracted to /content to avoid re-downloading.
#   4. Fresh download from Dropbox — last resort.
DRIVE_LIBRIPARTY = os.path.join(SAVE_DIR, "LibriParty")
LOCAL_LIBRIPARTY = "/content/LibriParty"


def libriparty_is_valid(path):
    """Check whether a directory contains a usable LibriParty evaluation set.

    A valid LibriParty root must contain at least one session directory under
    ``<path>/dataset/eval/session_*/``.

    Parameters
    ----------
    path : str
        Absolute path to the candidate LibriParty root directory.

    Returns
    -------
    bool
        True if the directory exists and contains at least one
        ``dataset/eval/session_*`` sub-directory; False otherwise.
    """
    import glob as _glob
    return (
        os.path.isdir(os.path.join(path, "dataset", "eval"))
        and len(_glob.glob(os.path.join(path, "dataset", "eval", "session_*"))) > 0
    )


if DATA_MODE == "demo":
    print("Mode: Demo — example audio only")
    # Download a single example WAV; no LibriParty needed.
    get_ipython().system(
        'wget -q -O /content/vad_test1.wav '
        '"https://www.dropbox.com/scl/fi/vvffxbkkuv79g0d4c7so3/'
        'example_vad_music.wav?rlkey=q5m5wc6y9fsfvt43x5yy8ohrf&dl=1"'
    )
    AUDIO_FILES = ["/content/vad_test1.wav"]
    LIBRIPARTY_DIR = None

elif DATA_MODE in ("small", "full"):
    import glob
    print(f"Mode: {DATA_MODE.capitalize()} — LibriParty")

    if libriparty_is_valid(DRIVE_LIBRIPARTY):
        LIBRIPARTY_DIR = DRIVE_LIBRIPARTY
        print(f"Found existing LibriParty in Drive: {LIBRIPARTY_DIR}")
    elif libriparty_is_valid(LOCAL_LIBRIPARTY):
        LIBRIPARTY_DIR = LOCAL_LIBRIPARTY
        print(f"Found existing LibriParty in /content: {LIBRIPARTY_DIR}")
    else:
        drive_tar = os.path.join(SAVE_DIR, "LibriParty.tar.gz")
        local_tar = "/content/LibriParty.tar.gz"
        if os.path.exists(drive_tar):
            print("Found LibriParty.tar.gz in Drive, extracting to /content ...")
            get_ipython().system('tar -xzf "$drive_tar" -C /content/')
            LIBRIPARTY_DIR = LOCAL_LIBRIPARTY
        elif os.path.exists(local_tar):
            print("Found LibriParty.tar.gz in /content, extracting ...")
            get_ipython().system('tar -xzf "$local_tar" -C /content/')
            LIBRIPARTY_DIR = LOCAL_LIBRIPARTY
        else:
            print("No cached copy found. Downloading from Dropbox ...")
            get_ipython().system(
                'wget -O /content/LibriParty.tar.gz '
                '"https://www.dropbox.com/s/ebo987wu3hie3zm/LibriParty.tar.gz?dl=1"'
            )
            get_ipython().system('tar -xzf /content/LibriParty.tar.gz -C /content/')
            LIBRIPARTY_DIR = LOCAL_LIBRIPARTY

    assert libriparty_is_valid(LIBRIPARTY_DIR), (
        f"LibriParty not valid at {LIBRIPARTY_DIR} "
        "— verify that dataset/eval/session_* exists."
    )
    AUDIO_FILES = None
    print(f"LibriParty ready at: {LIBRIPARTY_DIR}")

print(f"\nCurrent data mode: {DATA_MODE}")

# ---------------------------------------------------------------------------
# Cell 1 — Imports, Quantization Backend, Helper Functions
# ---------------------------------------------------------------------------
import time
import copy
import json
import random
import platform

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from speechbrain.inference.VAD import VAD

# Select the quantization backend based on the host CPU architecture.
# fbgemm is optimised for x86 (standard Colab instances).
# qnnpack is used on ARM (e.g., Colab with ARM runtime or Apple Silicon).
# This must be set before any quantized operator is executed.
if platform.machine() in ("arm64", "aarch64"):
    torch.backends.quantized.engine = "qnnpack"
else:
    torch.backends.quantized.engine = "fbgemm"
print(
    f"Quantized engine: {torch.backends.quantized.engine} "
    f"(machine={platform.machine()})"
)


def get_wrapper_size_mb(vad_obj, verbose=False):
    """Return the on-disk size of the VAD model's canonical state dict in MB.

    Serialises only ``vad_obj.mods['model']`` (the full CRDNN graph) to a
    temporary file and measures the resulting file size.  Serialising only
    the top-level model avoids double-counting when sub-module references
    (``mods['cnn']``, ``mods['rnn']``, ``mods['dnn']``) co-exist alongside
    the composite ``mods['model']`` — particularly important after dynamic
    quantization, when sub-modules carry INT8 packed params and the composite
    model must reflect the same weights.

    Parameters
    ----------
    vad_obj : speechbrain.inference.VAD
        A loaded SpeechBrain VAD object whose ``mods`` dict contains at least
        the key ``'model'``.
    verbose : bool, optional
        If True, prints a per-dtype parameter count breakdown derived from the
        saved state dict.  Non-tensor entries (scale/zero_point scalars,
        dtype objects, packed params common in quantized state dicts) are
        counted separately and excluded from the dtype breakdown.
        Default is False.

    Returns
    -------
    float
        Rounded model file size in megabytes (3 decimal places).
    """
    tmp = "/tmp/_tmp_vad_model.pt"
    torch.save(vad_obj.mods['model'].state_dict(), tmp)
    size_mb = os.path.getsize(tmp) / (1024 ** 2)

    if verbose:
        sd = vad_obj.mods['model'].state_dict()
        dtypes, skipped = {}, 0
        for v in sd.values():
            # Quantized state dicts contain non-tensor entries such as
            # dtype objects, scale/zero_point scalars, and packed weight
            # params. Only plain tensors contribute to parameter counts.
            if not torch.is_tensor(v):
                skipped += 1
                continue
            key = str(v.dtype)
            dtypes[key] = dtypes.get(key, 0) + v.numel()
        print(
            f"  [size debug] mods['model'] state_dict: {len(sd)} keys "
            f"({skipped} non-tensor entries skipped)"
        )
        for dt, n in sorted(dtypes.items(), key=lambda kv: -kv[1]):
            print(f"    {dt}: {n} params")

    os.remove(tmp)
    return round(size_mb, 3)


def measure_latency(model, input_tensor, n_warmup=30, n_runs=100, device="cpu"):
    """Measure single-forward-pass inference latency of a PyTorch model.

    Runs ``n_warmup`` un-timed passes to stabilise OS/hardware caches and
    JIT state, then records wall-clock time for ``n_runs`` passes.  Median
    and P95 are returned because inference latency distributions are
    right-skewed — a single outlier can inflate the mean.

    Single-threaded mode (``torch.set_num_threads(1)``) is enforced on CPU to
    eliminate run-to-run variance caused by dynamic thread-pool scheduling.

    Parameters
    ----------
    model : torch.nn.Module
        The model to benchmark.  It is moved to ``device`` before timing and
        returned to CPU afterwards.
    input_tensor : torch.Tensor
        A representative input tensor (e.g. shape ``[1, 48000]`` for 3 s at
        16 kHz).  It is moved to ``device`` before timing.
    n_warmup : int, optional
        Number of un-timed warm-up forward passes.  Default is 30.
    n_runs : int, optional
        Number of timed forward passes used to compute statistics.
        Default is 100.
    device : str, optional
        PyTorch device string (``"cpu"`` or ``"cuda"``).  Default is
        ``"cpu"``.

    Returns
    -------
    tuple[float, float, float]
        ``(median_ms, std_ms, p95_ms)`` — median latency, standard deviation,
        and 95th-percentile latency, all in milliseconds, each rounded to
        3 decimal places.
    """
    model.eval()
    model = model.to(device)
    x = input_tensor.to(device)
    if device == "cpu":
        torch.set_num_threads(1)

    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(x)
        times = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            _ = model(x)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)

    model = model.to("cpu")
    arr = np.array(times)
    return (
        round(float(np.median(arr)), 3),
        round(float(arr.std()), 3),
        round(float(np.percentile(arr, 95)), 3),
    )


def global_warmup(model, input_tensor, n_iters=50):
    """Pre-warm shared runtime caches before any latency measurement begins.

    The first few forward passes through a model trigger one-time initialisation
    costs: FFTW plan caching (used by mel-filterbank), torchaudio resampler
    kernel compilation, and thread-pool spin-up.  Running this function on the
    first model before any ``measure_latency`` call ensures all models are
    measured under equal, steady-state conditions.

    Parameters
    ----------
    model : torch.nn.Module
        The model to warm up.  Called in eval + no_grad mode.
    input_tensor : torch.Tensor
        Input tensor passed to the model's ``forward`` method.
    n_iters : int, optional
        Number of warm-up forward passes.  Default is 50.

    Returns
    -------
    None
    """
    model.eval()
    with torch.no_grad():
        for _ in range(n_iters):
            _ = model(input_tensor)


def quantize_all_paths(vad_obj, quant_spec, dtype=torch.qint8):
    """Apply dynamic quantization to both the composite model and its sub-modules.

    SpeechBrain's VAD object exposes the full CRDNN network at
    ``vad_obj.mods['model']`` and also keeps individual references at
    ``vad_obj.mods['cnn']``, ``vad_obj.mods['rnn']``, and
    ``vad_obj.mods['dnn']``.

    ``get_speech_prob_file`` (used by F1 evaluation) runs inference through
    ``mods['model']``.  ``CRDNNWrapper`` (used for latency benchmarking) reads
    directly from the sub-module references.  Quantizing both paths ensures
    that F1 evaluation and latency measurement operate on the same INT8
    weights.

    Parameters
    ----------
    vad_obj : speechbrain.inference.VAD
        A SpeechBrain VAD object with ``mods`` keys ``'model'``, ``'cnn'``,
        ``'rnn'``, and ``'dnn'``.
    quant_spec : set[type]
        Set of ``torch.nn.Module`` subclasses to quantize (e.g.
        ``{torch.nn.Linear}``).  Passed directly as ``qconfig_spec`` to
        ``torch.quantization.quantize_dynamic``.
    dtype : torch.dtype, optional
        Target quantized integer dtype.  Default is ``torch.qint8``.

    Returns
    -------
    speechbrain.inference.VAD
        The same ``vad_obj`` with all relevant sub-modules replaced by their
        dynamically-quantized counterparts.
    """
    vad_obj.mods['model'] = torch.quantization.quantize_dynamic(
        vad_obj.mods['model'], qconfig_spec=quant_spec, dtype=dtype
    )
    for name in ('cnn', 'rnn', 'dnn'):
        vad_obj.mods[name] = torch.quantization.quantize_dynamic(
            vad_obj.mods[name], qconfig_spec=quant_spec, dtype=dtype
        )
    return vad_obj


# ---------------------------------------------------------------------------
# Cell 2 — Load FP32 Baseline
# ---------------------------------------------------------------------------
# Download (first run) or load from cache the pre-trained CRDNN VAD model.
vad_fp32 = VAD.from_hparams(
    source="speechbrain/vad-crdnn-libriparty",
    savedir=os.path.join(SAVE_DIR, "pretrained_vad"),
)
model_fp32 = vad_fp32.mods['model']
model_fp32.eval()
print("FP32 model loaded.")
print(f"Submodules: {list(vad_fp32.mods.keys())}")

# ---------------------------------------------------------------------------
# Cell 3 — CRDNNWrapper
# ---------------------------------------------------------------------------

class CRDNNWrapper(nn.Module):
    """Unified PyTorch Module wrapping the SpeechBrain CRDNN VAD pipeline.

    SpeechBrain stores the VAD pipeline as a collection of named sub-modules
    inside ``vad_obj.mods``.  This wrapper assembles them into a single
    ``nn.Module`` with a standard ``forward(wav)`` signature, which enables:

    - Direct latency measurement via ``measure_latency``.
    - Gradient-based fine-tuning (QAT) with a single ``loss.backward()`` call.
    - Clean integration with ``torch.quantization`` APIs.

    The forward pass implements the sequence:
        raw waveform → mel features → mean-variance normalisation
        → CNN → RNN → DNN → per-frame log-odds (pre-sigmoid)

    Parameters (constructor)
    ------------------------
    vad_obj : speechbrain.inference.VAD
        A loaded SpeechBrain VAD object.  The following keys must exist in
        ``vad_obj.mods``: ``'compute_features'``, ``'mean_var_norm'``,
        ``'cnn'``, ``'rnn'``, ``'dnn'``.
    """

    def __init__(self, vad_obj):
        """Initialise wrapper by extracting sub-modules from ``vad_obj.mods``.

        Parameters
        ----------
        vad_obj : speechbrain.inference.VAD
            Source VAD object.  Sub-modules are shared (not deep-copied) so
            that quantization or fine-tuning applied to the wrapper affects the
            original ``vad_obj.mods`` references in place.
        """
        super().__init__()
        self.compute_features = vad_obj.mods['compute_features']
        self.mean_var_norm    = vad_obj.mods['mean_var_norm']
        self.cnn              = vad_obj.mods['cnn']
        self.rnn              = vad_obj.mods['rnn']
        self.dnn              = vad_obj.mods['dnn']

    def forward(self, wav):
        """Run a full CRDNN forward pass from raw waveform to per-frame logits.

        Parameters
        ----------
        wav : torch.Tensor
            Raw waveform tensor of shape ``[B, T_samples]`` sampled at 16 kHz.
            ``B`` is the batch size (typically 1 during evaluation).

        Returns
        -------
        torch.Tensor
            Per-frame log-odds tensor of shape ``[B, T_frames, 1]``.  Apply
            ``torch.sigmoid`` to obtain speech probabilities.  The temporal
            resolution is determined by the mel-feature hop length (~10 ms per
            frame at 16 kHz).
        """
        feats = self.compute_features(wav)
        # compute_features may return a 4-D tensor [B, T, F, 1] when channel
        # dimension is included; collapse the trailing singleton by averaging.
        if feats.dim() == 4:
            feats = feats.mean(-1)
        feats = self.mean_var_norm(
            feats, torch.ones(feats.shape[0], device=feats.device)
        )
        cnn_out = self.cnn(feats)
        B, T = cnn_out.shape[0], cnn_out.shape[1]
        rnn_in  = cnn_out.reshape(B, T, -1)
        rnn_out, _ = self.rnn(rnn_in)
        out = self.dnn(rnn_out)
        return out


# Sanity-check the wrapper with a synthetic 3-second clip at 16 kHz.
wrapper_fp32 = CRDNNWrapper(vad_fp32)
wrapper_fp32.eval()
DUMMY_WAV = torch.randn(1, 48000)  # shape: [batch=1, samples=48000]
with torch.no_grad():
    test_out = wrapper_fp32(DUMMY_WAV)
print(f"Wrapper test OK — output shape: {test_out.shape}")

# ---------------------------------------------------------------------------
# Cell 4 — FP32 Baseline Benchmarking
# ---------------------------------------------------------------------------
# Warm up shared caches (mel filterbank, FFT plan, thread pool) once before
# any timing begins so that all models start from the same steady state.
print("Global warmup (shared torch/torchaudio caches) ...")
global_warmup(wrapper_fp32, DUMMY_WAV, n_iters=50)

fp32_size = get_wrapper_size_mb(vad_fp32, verbose=True)
fp32_latency_median, fp32_latency_std, fp32_latency_p95 = measure_latency(
    wrapper_fp32, DUMMY_WAV, device="cpu"
)

print("=" * 50)
print("FP32 Baseline Results")
print("=" * 50)
print(f"  Model Size    : {fp32_size} MB")
print(f"  Latency median: {fp32_latency_median} ms  "
      f"(std {fp32_latency_std}, p95 {fp32_latency_p95})")

# ---------------------------------------------------------------------------
# Cell 5 — PTQ: Post-Training Dynamic Quantization
# ---------------------------------------------------------------------------
# Dynamic quantization converts nn.Linear weight matrices to INT8 at load
# time and dequantizes activations on the fly.
#
# Scope limitation: only nn.Linear is quantized.
#   - GRU (inside SpeechBrain's RNN wrapper) cannot be dynamic-quantized
#     because the wrapper calls flatten_parameters() which the quantized
#     variant does not support.
#   - Conv2d is not supported by quantize_dynamic at all.
# As a result, only DNN layers are fully INT8; the CNN and RNN remain FP32.

from torch.ao.nn.quantized.dynamic import Linear as QDynLinear

QUANT_SPEC = {nn.Linear}


def count_linear_params(module):
    """Count the number of parameters that reside in Linear layers.

    Distinguishes between already-quantized dynamic linear layers
    (``QDynLinear``) and plain ``nn.Linear`` layers, and sums their parameter
    counts separately.  The total parameter count across the entire module is
    also returned for context.

    Parameters
    ----------
    module : torch.nn.Module
        Any PyTorch module.  The function recurses through all sub-modules.

    Returns
    -------
    tuple[int, int]
        ``(linear_params, total_params)`` where ``linear_params`` is the
        number of parameters in all Linear (including quantized) layers and
        ``total_params`` is the total number of parameters in the module.
    """
    total = sum(p.numel() for p in module.parameters())
    linear_params = 0
    for m in module.modules():
        if isinstance(m, QDynLinear):
            w, b = m._packed_params._weight_bias()
            linear_params += w.numel() + (b.numel() if b is not None else 0)
        elif isinstance(m, nn.Linear):
            linear_params += sum(p.numel() for p in m.parameters(recurse=False))
    return linear_params, total


vad_ptq = copy.deepcopy(vad_fp32)
vad_ptq = quantize_all_paths(vad_ptq, QUANT_SPEC)

wrapper_ptq = CRDNNWrapper(vad_ptq)
wrapper_ptq.eval()

q_params, t_params = count_linear_params(vad_ptq.mods['model'])
pct = 100 * q_params / max(t_params, 1)
print(
    f"PTQ coverage (in mods['model']): {q_params}/{t_params} Linear params "
    f"quantized ({pct:.1f}%)"
)

ptq_size = get_wrapper_size_mb(vad_ptq, verbose=True)
ptq_latency_median, ptq_latency_std, ptq_latency_p95 = measure_latency(
    wrapper_ptq, DUMMY_WAV, device="cpu"
)

print("=" * 50)
print("PTQ Results")
print("=" * 50)
print(f"  Model Size    : {ptq_size} MB  (FP32: {fp32_size} MB)")
print(f"  Size ratio    : {round(ptq_size / fp32_size, 2)}x  "
      f"({round((1 - ptq_size / fp32_size) * 100, 1)}% reduction)")
print(f"  Latency median: {ptq_latency_median} ms  "
      f"(std {ptq_latency_std}, p95 {ptq_latency_p95})")
print(f"  Speedup       : {round(fp32_latency_median / ptq_latency_median, 2)}x")

# ---------------------------------------------------------------------------
# Cell 6 — Prepare Evaluation Sessions
# ---------------------------------------------------------------------------

def load_libriparty_gt(json_path, fps=100):
    """Load ground-truth speech/non-speech frame labels from a LibriParty JSON.

    LibriParty annotation files contain per-speaker utterance time-stamps.
    This function merges all speaker segments into a single binary label
    array at the specified frame rate.

    Parameters
    ----------
    json_path : str
        Path to a LibriParty session JSON file (e.g.
        ``session_00/session_00.json``).  The JSON maps speaker IDs to lists
        of dicts with ``'start'`` and ``'stop'`` keys (both in seconds).
    fps : int, optional
        Frame rate in frames per second used to convert time-stamps to frame
        indices.  Must match the VAD model's output frame rate (100 fps for
        the SpeechBrain CRDNN model).  Default is 100.

    Returns
    -------
    tuple[numpy.ndarray or None, int]
        ``(labels, total_frames)`` where ``labels`` is a 1-D int array of
        shape ``[total_frames]`` with 1 for speech frames and 0 for
        non-speech.  Returns ``(None, 0)`` if the JSON contains no speech
        segments.
    """
    with open(json_path) as f:
        data = json.load(f)

    segments = []
    for _spk_id, utterances in data.items():
        for utt in utterances:
            segments.append((utt['start'], utt['stop']))

    if not segments:
        return None, 0

    total_frames = int(max(s[1] for s in segments) * fps) + 1
    labels = np.zeros(total_frames, dtype=int)
    for start, stop in segments:
        s = int(start * fps)
        e = int(stop * fps)
        labels[s:min(e, total_frames)] = 1

    return labels, total_frames


# Use 20 sessions so that F1 deltas between compression strategies exceed
# per-session variance and are statistically interpretable.
N_SESSIONS = 20
session_dirs = sorted(
    glob.glob(f"{LIBRIPARTY_DIR}/dataset/eval/session_*/")
)[:N_SESSIONS]

eval_sessions = []
for sd in session_dirs:
    sname     = os.path.basename(sd.rstrip('/'))
    wav_path  = os.path.join(sd, f"{sname}_mixture.wav")
    json_path = os.path.join(sd, f"{sname}.json")
    if not os.path.exists(wav_path) or not os.path.exists(json_path):
        continue
    gt_labels, total_frames = load_libriparty_gt(json_path)
    if gt_labels is not None:
        eval_sessions.append((wav_path, gt_labels, total_frames))

print(f"eval_sessions ready: {len(eval_sessions)} sessions")

# ---------------------------------------------------------------------------
# Cell 7 — QAT Preparation: FakeQuant with Straight-Through Estimator
# ---------------------------------------------------------------------------

class FakeQuantLinear(nn.Linear):
    """Linear layer with simulated INT8 weight quantization during training.

    During the forward pass, weights are symmetrically quantized to 8-bit
    integers using a per-tensor absolute-max scale, then dequantized back to
    FP32 before the matrix multiply.  The straight-through estimator (STE) is
    applied so that gradients flow through the rounding operation unchanged —
    enabling the upstream layers to adapt their weight distributions to be
    more quantization-friendly.

    This layer is a drop-in replacement for ``torch.nn.Linear`` and carries
    no additional persistent parameters.
    """

    def forward(self, x):
        """Forward pass with quantization-aware weight simulation.

        Parameters
        ----------
        x : torch.Tensor
            Input activation tensor of shape ``[*, in_features]``.

        Returns
        -------
        torch.Tensor
            Output tensor of shape ``[*, out_features]``, identical in dtype
            to ``x``.  The computation is equivalent to a standard linear
            layer but uses weights that have been rounded to the nearest INT8
            representable value and then scaled back to FP32.
        """
        w = self.weight
        # Per-tensor symmetric scale: maps the weight range to [-127, 127].
        scale = w.abs().max() / 127.0
        w_fq  = (w / scale).round().clamp(-128, 127) * scale
        # STE: forward uses quantized weights; backward ignores rounding.
        w_ste = w + (w_fq - w).detach()
        return F.linear(x, w_ste, self.bias)


def replace_linear_with_fakequant(module):
    """Recursively replace all ``nn.Linear`` layers with ``FakeQuantLinear``.

    Traverses the module tree depth-first and substitutes every direct
    ``nn.Linear`` instance with a ``FakeQuantLinear`` that shares the same
    weight and bias data.  Only exact ``nn.Linear`` instances are replaced;
    subclasses (including existing ``FakeQuantLinear``) are left unchanged.

    Parameters
    ----------
    module : torch.nn.Module
        Root module to patch in-place.  All ``nn.Linear`` descendants
        (at any depth) are replaced.

    Returns
    -------
    None
        The replacement is performed in-place; the original module reference
        remains valid.
    """
    for name, child in module.named_children():
        if type(child) is nn.Linear:
            fq = FakeQuantLinear(
                child.in_features,
                child.out_features,
                bias=child.bias is not None,
            )
            fq.weight.data = child.weight.data.clone()
            if child.bias is not None:
                fq.bias.data = child.bias.data.clone()
            setattr(module, name, fq)
        else:
            replace_linear_with_fakequant(child)


def sample_clip_with_speech(
    wav_path,
    gt_labels,
    clip_samples=48000,
    frame_stride_samples=160,
    target_frames=301,
    max_tries=10,
):
    """Sample a fixed-length audio clip that contains a meaningful amount of speech.

    Randomly draws up to ``max_tries`` candidate start positions and selects
    the one whose ground-truth speech ratio exceeds 20 %.  If no candidate
    meets the threshold, the one with the highest speech ratio is used.  This
    avoids training on clips that are almost entirely silence, which would
    bias the model towards predicting non-speech.

    Parameters
    ----------
    wav_path : str
        Path to a mono or multi-channel WAV file.  Only the first channel is
        used.
    gt_labels : numpy.ndarray or torch.Tensor
        Frame-level binary labels (1 = speech, 0 = non-speech) at 100 fps,
        covering the full session audio.
    clip_samples : int, optional
        Length of the extracted clip in samples.  At 16 kHz, 48 000 samples
        corresponds to 3 seconds.  Default is 48 000.
    frame_stride_samples : int, optional
        Number of audio samples per label frame.  Used to convert a sample
        offset to a frame index.  Default is 160 (10 ms at 16 kHz).
    target_frames : int, optional
        Expected number of label frames corresponding to ``clip_samples``.
        Used to slice the label array.  Default is 301.
    max_tries : int, optional
        Maximum number of random start positions to evaluate before accepting
        the best candidate found so far.  Default is 10.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        ``(x, labels)`` where:

        - ``x``      — waveform clip, shape ``[1, clip_samples]``, dtype float32.
        - ``labels`` — frame-level binary labels, shape ``[target_frames]``,
          dtype float32.  May be shorter than ``target_frames`` if the clip
          extends past the end of the session.
    """
    wav, _sr = torchaudio.load(wav_path)
    total_samples = wav.shape[1]

    if total_samples <= clip_samples:
        start = 0
    else:
        best_start = 0
        best_ratio = -1.0
        for _ in range(max_tries):
            start = random.randint(0, total_samples - clip_samples)
            label_start = start // frame_stride_samples
            label_end   = label_start + target_frames
            seg_labels  = gt_labels[label_start:label_end]
            speech_ratio = (
                seg_labels.float().mean().item()
                if isinstance(seg_labels, torch.Tensor)
                else float(np.mean(seg_labels))
            )
            if speech_ratio > 0.2:
                best_start = start
                best_ratio = speech_ratio
                break
            if speech_ratio > best_ratio:
                best_start = start
                best_ratio = speech_ratio
        start = best_start

    x = wav[:1, start:start + clip_samples]

    label_start = start // frame_stride_samples
    label_end   = label_start + target_frames
    labels = (
        gt_labels[label_start:label_end].float()
        if isinstance(gt_labels, torch.Tensor)
        else torch.tensor(gt_labels[label_start:label_end], dtype=torch.float32)
    )
    return x, labels


# Prepare QAT wrapper: inject FakeQuant only into the DNN sub-module so that
# the CNN and RNN feature extractors remain stable during fine-tuning.
vad_qat_obj = copy.deepcopy(vad_fp32)
wrapper_qat  = CRDNNWrapper(vad_qat_obj)
replace_linear_with_fakequant(wrapper_qat.dnn)
wrapper_qat.train()
for p in wrapper_qat.parameters():
    p.requires_grad_(True)
print("FakeQuant QAT wrapper ready.")

# ---------------------------------------------------------------------------
# Cell 8 — QAT Fine-tuning and Conversion to INT8
# ---------------------------------------------------------------------------
optimizer = torch.optim.Adam(
    [p for p in wrapper_qat.parameters() if p.requires_grad], lr=1e-5
)
criterion = nn.BCEWithLogitsLoss()

# Cap at 60 steps to keep runtime manageable; cycle through sessions if
# there are fewer than 20 (i.e., sessions * 3 gives at most 60 steps).
N_CALIB_STEPS = min(60, len(eval_sessions) * 3)
wrapper_qat.train()

for step in range(N_CALIB_STEPS):
    wav_path, gt_labels, _total_frames = eval_sessions[step % len(eval_sessions)]
    x, labels = sample_clip_with_speech(wav_path, gt_labels)

    out = wrapper_qat(x).squeeze(0).squeeze(-1)
    T   = out.shape[0]

    # Align label length to model output length (label array may be shorter
    # at the end of a session or longer due to floating-point stride rounding).
    if labels.shape[0] < T:
        labels = torch.cat([labels, torch.zeros(T - labels.shape[0])])
    labels = labels[:T]

    loss = criterion(out, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"  Step {step + 1}/{N_CALIB_STEPS}  "
          f"loss={loss.item():.4f}  "
          f"speech_ratio={labels.mean().item():.2f}")

# Copy fine-tuned weights from the wrapper back to a fresh FP32 clone, then
# apply dynamic quantization to produce the final INT8 QAT model.
wrapper_qat.eval()
vad_qat_final = copy.deepcopy(vad_fp32)

for name in ('cnn', 'rnn', 'dnn'):
    src_sd = {
        k: v.detach().clone()
        for k, v in getattr(wrapper_qat, name).state_dict().items()
    }
    if name in vad_qat_final.mods:
        vad_qat_final.mods[name].load_state_dict(src_sd, strict=False)
    if hasattr(vad_qat_final.mods['model'], name):
        try:
            getattr(vad_qat_final.mods['model'], name).load_state_dict(
                src_sd, strict=False
            )
        except Exception as e:
            print(f"  Warning: could not write to mods['model'].{name}: {e}")

vad_qat_final  = quantize_all_paths(vad_qat_final, QUANT_SPEC)
wrapper_qat_final = CRDNNWrapper(vad_qat_final)
wrapper_qat_final.eval()

qat_size = get_wrapper_size_mb(vad_qat_final, verbose=True)
qat_latency_median, qat_latency_std, qat_latency_p95 = measure_latency(
    wrapper_qat_final, DUMMY_WAV, device="cpu"
)
vad_qat_eval = vad_qat_final

print("=" * 50)
print("QAT Size + Latency")
print("=" * 50)
print(f"  Model Size    : {qat_size} MB")
print(f"  Size ratio    : {round(qat_size / fp32_size, 2)}x")
print(f"  Latency median: {qat_latency_median} ms  "
      f"(std {qat_latency_std}, p95 {qat_latency_p95})")

# ---------------------------------------------------------------------------
# Cell 8b — PTQ+QAT Combined: FakeQuant Across All Sub-modules
# ---------------------------------------------------------------------------
# A more aggressive variant: FakeQuant is injected into every sub-module
# (CNN, RNN, DNN) rather than DNN only, exposing more of the network to
# quantization noise during fine-tuning.

vad_ptq_qat_obj  = copy.deepcopy(vad_fp32)
wrapper_ptq_qat  = CRDNNWrapper(vad_ptq_qat_obj)
replace_linear_with_fakequant(wrapper_ptq_qat)  # apply to all sub-modules
wrapper_ptq_qat.train()
for p in wrapper_ptq_qat.parameters():
    p.requires_grad_(True)

optimizer_comb  = torch.optim.Adam(
    [p for p in wrapper_ptq_qat.parameters() if p.requires_grad], lr=1e-5
)
criterion_comb  = nn.BCEWithLogitsLoss()
N_CALIB_STEPS_COMB = min(60, len(eval_sessions) * 3)

for step in range(N_CALIB_STEPS_COMB):
    wav_path, gt_labels, _total_frames = eval_sessions[step % len(eval_sessions)]
    x, labels = sample_clip_with_speech(wav_path, gt_labels)

    out = wrapper_ptq_qat(x).squeeze(0).squeeze(-1)
    T   = out.shape[0]

    if labels.shape[0] < T:
        labels = torch.cat([labels, torch.zeros(T - labels.shape[0])])
    labels = labels[:T]

    loss = criterion_comb(out, labels)
    optimizer_comb.zero_grad()
    loss.backward()
    optimizer_comb.step()

    print(f"  Step {step + 1}/{N_CALIB_STEPS_COMB}  "
          f"loss={loss.item():.4f}  "
          f"speech_ratio={labels.mean().item():.2f}")

# Copy fine-tuned weights and apply dynamic quantization.
wrapper_ptq_qat.eval()
vad_ptq_qat_final = copy.deepcopy(vad_fp32)

for name in ('cnn', 'rnn', 'dnn'):
    src_sd = {
        k: v.detach().clone()
        for k, v in getattr(wrapper_ptq_qat, name).state_dict().items()
    }
    vad_ptq_qat_final.mods[name].load_state_dict(src_sd, strict=False)
    # Also propagate to the composite model if the attribute name matches.
    for attr_name in dir(vad_ptq_qat_final.mods['model']):
        if attr_name.lower() == name:
            try:
                getattr(vad_ptq_qat_final.mods['model'], attr_name).load_state_dict(
                    src_sd, strict=False
                )
            except Exception:
                pass

vad_ptq_qat_final = quantize_all_paths(vad_ptq_qat_final, QUANT_SPEC)
wrapper_ptq_qat_converted = CRDNNWrapper(vad_ptq_qat_final)
wrapper_ptq_qat_converted.eval()

ptq_qat_size = get_wrapper_size_mb(vad_ptq_qat_final, verbose=True)
ptq_qat_latency_median, ptq_qat_latency_std, ptq_qat_latency_p95 = measure_latency(
    wrapper_ptq_qat_converted, DUMMY_WAV, device="cpu"
)

print("=" * 50)
print("PTQ+QAT Combined")
print("=" * 50)
print(f"  Model Size    : {ptq_qat_size} MB")
print(f"  Size ratio    : {round(ptq_qat_size / fp32_size, 2)}x")
print(f"  Latency median: {ptq_qat_latency_median} ms  "
      f"(std {ptq_qat_latency_std}, p95 {ptq_qat_latency_p95})")

# ---------------------------------------------------------------------------
# Cell 9 — F1 Evaluation (20 Sessions + SEM)
# ---------------------------------------------------------------------------

def vad_get_boundaries(vad_obj, wav_path):
    """Run VAD inference and return detected speech boundaries in seconds.

    Uses SpeechBrain's built-in pipeline: chunk-level speech probability
    estimation followed by threshold application and boundary extraction.

    Parameters
    ----------
    vad_obj : speechbrain.inference.VAD
        A SpeechBrain VAD object (any compression variant).
    wav_path : str
        Path to the input WAV file.

    Returns
    -------
    torch.Tensor
        2-D tensor of shape ``[N_segments, 2]`` where each row is
        ``[start_seconds, end_seconds]`` for a detected speech segment.
    """
    prob_chunks = vad_obj.get_speech_prob_file(wav_path)
    prob_th     = vad_obj.apply_threshold(prob_chunks).float()
    return vad_obj.get_boundaries(prob_th, output_value="seconds")


def boundaries_to_frame_labels(boundaries, total_frames, fps=100):
    """Convert a list of time-boundary pairs to a binary frame-label array.

    Parameters
    ----------
    boundaries : torch.Tensor
        Tensor of shape ``[N, 2]`` with ``(start_s, end_s)`` rows in seconds,
        as returned by ``vad_get_boundaries``.
    total_frames : int
        Total number of frames in the sequence.  Determines the length of the
        output array; frame indices beyond this limit are clipped.
    fps : int, optional
        Frames per second used to convert seconds to frame indices.
        Must match the rate used when building ground-truth labels.
        Default is 100.

    Returns
    -------
    numpy.ndarray
        1-D int array of shape ``[total_frames]`` with 1 for speech frames and
        0 for non-speech frames.
    """
    labels = np.zeros(total_frames, dtype=int)
    for row in boundaries:
        s = int(row[0].item() * fps)
        e = int(row[1].item() * fps)
        labels[s:min(e, total_frames)] = 1
    return labels


def compute_f1(pred, ref):
    """Compute frame-level binary F1, precision, and recall.

    Treats each frame independently as a binary speech/non-speech prediction.
    Returns 0.0 for any metric whose denominator is zero (e.g. when the
    reference contains no speech).

    Parameters
    ----------
    pred : numpy.ndarray
        1-D int array of predicted labels (0 or 1).
    ref : numpy.ndarray
        1-D int array of reference labels (0 or 1).  Must have the same
        length as ``pred``.

    Returns
    -------
    tuple[float, float, float]
        ``(f1, precision, recall)`` each rounded to 4 decimal places.
    """
    tp = int(np.sum((pred == 1) & (ref == 1)))
    fp = int(np.sum((pred == 1) & (ref == 0)))
    fn = int(np.sum((pred == 0) & (ref == 1)))

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    return round(f1, 4), round(prec, 4), round(rec, 4)


def evaluate_vad_on_session(vad_obj, wav_path, ref_labels, total_frames, fps=100):
    """Evaluate a VAD model on a single LibriParty session.

    Runs the full VAD pipeline on the session audio, converts the detected
    boundaries to a frame-label array, and computes F1 against the
    ground-truth labels.

    Parameters
    ----------
    vad_obj : speechbrain.inference.VAD
        VAD model to evaluate.
    wav_path : str
        Path to the session mixture WAV file.
    ref_labels : numpy.ndarray
        Ground-truth binary frame labels from ``load_libriparty_gt``.
    total_frames : int
        Total frame count for the session.
    fps : int, optional
        Frame rate (frames per second).  Default is 100.

    Returns
    -------
    tuple[float, float, float]
        ``(f1, precision, recall)`` for the session.
    """
    boundaries  = vad_get_boundaries(vad_obj, wav_path)
    pred_labels = boundaries_to_frame_labels(boundaries, total_frames, fps)
    min_len     = min(len(pred_labels), len(ref_labels))
    return compute_f1(pred_labels[:min_len], ref_labels[:min_len])


def eval_model_on_sessions(vad_obj, sessions):
    """Evaluate a VAD model across multiple LibriParty sessions and aggregate results.

    Computes per-session F1, precision, and recall, then returns macro-averaged
    statistics together with the per-session F1 standard deviation.  The
    standard deviation is reported alongside the mean so callers can assess
    whether differences between models exceed the per-session noise level.

    Parameters
    ----------
    vad_obj : speechbrain.inference.VAD
        VAD model to evaluate.
    sessions : list[tuple[str, numpy.ndarray, int]]
        List of ``(wav_path, gt_labels, total_frames)`` tuples as produced
        by the session preparation cell.

    Returns
    -------
    tuple[float, float, float, float]
        ``(mean_f1, mean_precision, mean_recall, f1_std)`` each rounded to
        4 decimal places.
    """
    all_f1, all_prec, all_rec = [], [], []
    for wav_path, ref_labels, total_frames in sessions:
        f1, prec, rec = evaluate_vad_on_session(
            vad_obj, wav_path, ref_labels, total_frames
        )
        all_f1.append(f1)
        all_prec.append(prec)
        all_rec.append(rec)
    return (
        round(float(np.mean(all_f1)),  4),
        round(float(np.mean(all_prec)), 4),
        round(float(np.mean(all_rec)),  4),
        round(float(np.std(all_f1)),    4),
    )


print("Evaluating FP32 ...")
fp32_f1,    fp32_prec,    fp32_rec,    fp32_f1_std    = eval_model_on_sessions(vad_fp32,          eval_sessions)
print("Evaluating PTQ ...")
ptq_f1,     ptq_prec,     ptq_rec,     ptq_f1_std     = eval_model_on_sessions(vad_ptq,           eval_sessions)
print("Evaluating QAT ...")
qat_f1,     qat_prec,     qat_rec,     qat_f1_std     = eval_model_on_sessions(vad_qat_eval,      eval_sessions)
print("Evaluating PTQ+QAT ...")
ptq_qat_f1, ptq_qat_prec, ptq_qat_rec, ptq_qat_f1_std = eval_model_on_sessions(vad_ptq_qat_final, eval_sessions)

sem = round(fp32_f1_std / np.sqrt(len(eval_sessions)), 4)

print("\n" + "=" * 50)
print("F1 Evaluation Results")
print("=" * 50)
print(f"  FP32    — F1: {fp32_f1}    (±{fp32_f1_std})  "
      f"Prec: {fp32_prec}   Rec: {fp32_rec}")
print(f"  PTQ     — F1: {ptq_f1}     (±{ptq_f1_std})  "
      f"Prec: {ptq_prec}    Rec: {ptq_rec}")
print(f"  QAT     — F1: {qat_f1}     (±{qat_f1_std})  "
      f"Prec: {qat_prec}    Rec: {qat_rec}")
print(f"  PTQ+QAT — F1: {ptq_qat_f1} (±{ptq_qat_f1_std})  "
      f"Prec: {ptq_qat_prec} Rec: {ptq_qat_rec}")
print(f"\n  Per-session F1 std ≈ {fp32_f1_std};  SEM ≈ {sem}.")
print(f"  Deltas smaller than ≈{sem} are within measurement noise.")

# ---------------------------------------------------------------------------
# Cell 10 — Final Results Table and Latency vs. F1 Plot
# ---------------------------------------------------------------------------
import matplotlib.pyplot as plt
import pandas as pd

results = {
    'Model':        ['FP32 Baseline', 'PTQ (Dynamic)', 'QAT',        'PTQ+QAT'],
    'Format':       ['FP32',          'Mixed/INT8',    'Mixed/INT8', 'Mixed/INT8'],
    'Size (MB)':    [fp32_size,        ptq_size,        qat_size,     ptq_qat_size],
    'Latency (ms)': [fp32_latency_median, ptq_latency_median,
                     qat_latency_median,  ptq_qat_latency_median],
    'F1 Score':     [fp32_f1,          ptq_f1,          qat_f1,       ptq_qat_f1],
    'Precision':    [fp32_prec,        ptq_prec,        qat_prec,     ptq_qat_prec],
    'Recall':       [fp32_rec,         ptq_rec,         qat_rec,      ptq_qat_rec],
}
df = pd.DataFrame(results)

print("=" * 70)
print("FINAL RESULTS TABLE")
print("=" * 70)
print(df.to_string(index=False))

csv_path = os.path.join(SAVE_DIR, "vad_compression_results.csv")
df.to_csv(csv_path, index=False)
print(f"\nSaved CSV to {csv_path}")

# Scatter plot: inference latency (x-axis) vs. F1 score (y-axis).
# Each point represents one compression strategy; the Pareto-optimal
# strategy is the one closest to the top-left corner (low latency, high F1).
fig, ax = plt.subplots(figsize=(7, 5))
colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52']
for lat, f1, label, color in zip(
    results['Latency (ms)'], results['F1 Score'], results['Model'], colors
):
    ax.scatter(lat, f1, s=180, color=color, zorder=5, label=label)
    ax.annotate(label, (lat, f1),
                textcoords="offset points", xytext=(8, 4), fontsize=10)

ax.set_xlabel("Inference Latency (ms)", fontsize=12)
ax.set_ylabel("F1 Score", fontsize=12)
ax.set_title("Latency vs F1 Score — VAD Compression Trade-off", fontsize=13)
ax.legend()
ax.grid(True, linestyle='--', alpha=0.5)

plot_path = os.path.join(SAVE_DIR, "latency_vs_f1.png")
plt.tight_layout()
plt.savefig(plot_path, dpi=150)
plt.show()
print(f"Plot saved to {plot_path}")
