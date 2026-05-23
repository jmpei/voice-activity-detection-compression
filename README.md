# Voice Activity Detection — Compression Benchmark

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TomatoesSuck/Voice-Activity-Detection-Compression/blob/main/vad_experiment_colab.ipynb)

Benchmarks four model-compression strategies on **SpeechBrain's CRDNN-based VAD**
([`speechbrain/vad-crdnn-libriparty`](https://huggingface.co/speechbrain/vad-crdnn-libriparty))
evaluated on the LibriParty dataset.

---

## Compression Strategies

| Strategy | Method | Scope |
|---|---|---|
| **FP32 Baseline** | No compression | — |
| **PTQ** | Post-training dynamic quantization (`torch.quantization.quantize_dynamic`) | `nn.Linear` → INT8 |
| **QAT** | FakeQuant fine-tuning (STE) → dynamic quantization | DNN sub-module only |
| **PTQ+QAT** | FakeQuant fine-tuning (STE) → dynamic quantization | All sub-modules (CNN + RNN + DNN) |

---

## Results

Evaluated on 20 LibriParty eval sessions. Latency: median over 100 runs, single-threaded CPU (x86 Colab).

| Model | Size (MB) | Latency median (ms) | Latency P95 (ms) | F1 | Precision | Recall |
|---|---|---|---|---|---|---|
| FP32 Baseline | 0.435 | 49.761 | 80.693 | **0.9587** | 0.9574 | 0.9606 |
| PTQ (Dynamic) | 0.434 | 50.619 | 82.124 | **0.9594** | 0.9591 | 0.9603 |
| QAT | 0.434 | 54.214 | 62.260 | 0.9287 | 0.9886 | 0.8771 |
| PTQ+QAT | 0.434 | 54.365 | 326.701 | 0.8815 | 0.9965 | 0.7930 |

Per-session F1 std ≈ 0.0156 · SEM ≈ 0.0035. Deltas below the SEM are within measurement noise.

**Key finding**: PTQ matches the FP32 baseline in both F1 and latency (Δ within SEM). QAT and PTQ+QAT degrade F1 by 3 and 8 points respectively with no size benefit, because only 1.2% of parameters reside in quantizable `nn.Linear` layers—the dominant CNN and RNN weights cannot be dynamically quantized.

---

## Project Structure

```
.
├── vad_experiment_colab.ipynb   # Main experiment notebook (run on Google Colab)
├── archive/
│   └── vad_experiment_colab.py  # Archived Python export (no longer maintained)
├── report.md                    # Detailed technical report with analysis
└── README.md
```

Results (CSV + plot) are written to Google Drive at `MyDrive/VAD_Compression/` during the Colab run.

---

## Quick Start

### Run on Google Colab (recommended)

Click the **Open in Colab** badge above. The notebook will:
1. Mount Google Drive and install SpeechBrain from the `develop` branch.
2. Download the LibriParty evaluation set (~4.75 GB) or use a cached copy from Drive.
3. Load the pre-trained CRDNN VAD from HuggingFace Hub.
4. Run all four compression strategies and evaluate each on 20 sessions.
5. Save `vad_compression_results.csv` and `latency_vs_f1.png` to Drive.

### Data modes

Set `DATA_MODE` in the second cell:

| Value | Description |
|---|---|
| `"demo"` | Downloads a single example WAV; no LibriParty needed. Quick smoke test. |
| `"small"` | First 20 LibriParty eval sessions (default, ~4.75 GB download). |
| `"full"` | Full LibriParty archive (~10 GB). |

The dataset is cached to Drive so subsequent runs skip the download.

---

## Dependencies

```
torch
torchaudio
numpy
matplotlib
pandas
speechbrain @ git+https://github.com/speechbrain/speechbrain.git@develop
```

Install in Colab (handled automatically by the notebook):

```bash
pip install git+https://github.com/speechbrain/speechbrain.git@develop
```

> **Note**: `torch.quantization.quantize_dynamic` is deprecated in PyTorch ≥ 2.10.
> The recommended migration path is [`torchao`](https://github.com/pytorch/ao).

---

## Architecture

The CRDNN model processes raw 16 kHz waveforms in three stages:

```
wav [B, T] → Mel features → Mean-Var Norm → CNN → RNN (GRU) → DNN → logits [B, T_frames, 1]
```

`CRDNNWrapper` assembles the SpeechBrain sub-modules (`compute_features`, `mean_var_norm`,
`cnn`, `rnn`, `dnn`) into a single `nn.Module` that supports unified latency benchmarking
and gradient-based fine-tuning.

---

## Quantization Constraints

Dynamic quantization via `quantize_dynamic` supports only `nn.Linear`.
The CRDNN model's dominant layers (Conv2d, GRU) fall outside this scope:

- **GRU**: calling `flatten_parameters()` inside SpeechBrain's RNN wrapper is incompatible with the quantized variant.
- **Conv2d**: not supported by `quantize_dynamic` at all.

As a result, only the DNN head (1.2% of total parameters) is converted to INT8, yielding negligible size reduction.


---

## Update — RNN swap + static PTQ helpers (not yet run in Colab)

Added scaffolding to `vad_experiment_colab.ipynb` for the next conditions (targets the 98.8% of weights dynamic PTQ couldn't reach). **Written but not yet executed in Colab — needs one run to confirm.**

- `CRDNNWrapper(vad, rnn=...)` — optional RNN override; default still uses the pretrained GRU, so existing cells are unchanged. Feature front-end exposed as `wrapper.features(wav)`.
- `build_lstm_like(rnn)` — fresh `nn.LSTM` matching the GRU's dims for a one-line GRU→LSTM swap (random weights; must be trained before its F1 means anything).
- `static_ptq_module(module, calib_inputs)` — `prepare → calibrate → convert`; quantizes Conv2d (the CNN), which `quantize_dynamic` could not. GRU stays FP32.
- `load_sessions("train")` + `build_calibration_set(...)` — calibration clips from the **train** split, disjoint from eval. Needs the train split present (DATA_MODE='full' or dataset/train/ in place).

`vad_experiment_colab.py` moved to `archive/` and is no longer maintained — the notebook is the single source of truth.
