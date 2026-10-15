# Voice Activity Detection — Compression Benchmark

Compresses **SpeechBrain's CRDNN-based VAD**
([`speechbrain/vad-crdnn-libriparty`](https://huggingface.co/speechbrain/vad-crdnn-libriparty),
~108k params, 0.435 MB FP32) for iPhone deployment, evaluated frame-level on the
LibriParty eval split. The goal is a smaller on-disk footprint than the FP32
baseline with F1 that stays close to it.

Full write-up with analysis: [`report.md`](report.md).

---

## Conditions

| ID  | Method                                       |
| --- | -------------------------------------------- |
| E0  | FP32 baseline (GRU)                          |
| E1a | Dynamic PTQ (`quantize_dynamic`, `nn.Linear` → INT8) |
| E1b | QAT — FakeQuant (STE) on the DNN head only   |
| E1c | PTQ + QAT — FakeQuant on all sub-modules     |
| E2  | Static PTQ (CNN Conv2d → INT8)               |
| E3  | GRU → LSTM, FP32 (trained, frozen CNN+DNN)   |
| E4  | LSTM + static-PTQ CNN + dynamic-quant LSTM   |
| E5  | Distilled student (~33k params, FP32)        |
| E6  | Distilled student + PTQ                      |

The first pass (E0–E1c) establishes that dynamic PTQ is a no-op on this
architecture; the second pass (E2–E6) actually shrinks the dominant Conv2d/RNN
weights via static quantization, a GRU→LSTM swap, and distillation.

---

## Results

Frame-level F1 over 20 LibriParty eval sessions. Latency: median over 100
single-threaded CPU runs after warm-up, on a local Apple-Silicon (arm64) Mac.
Per-session F1 SEM ≈ 0.0035; deltas below that are within noise.

| ID  | Method                          | Size (MB) | Latency (ms) | F1     | Precision | Recall |
| --- | ------------------------------- | --------- | ------------ | ------ | --------- | ------ |
| E0  | FP32 baseline (GRU)             | 0.435     | 12.9         | **0.9587** | 0.957 | 0.961  |
| E1a | Dynamic PTQ                     | 0.434     | 13.4         | **0.9594** | 0.959 | 0.960  |
| E1b | QAT (DNN only)                  | 0.434     | 14.1         | 0.8518 | 0.997     | 0.747  |
| E1c | PTQ + QAT (all sub-modules)     | 0.434     | 12.2         | 0.8726 | 0.996     | 0.779  |
| E2  | Static PTQ (CNN → INT8)         | 0.397     | 13.3         | 0.9599 | 0.960     | 0.960  |
| E3  | GRU → LSTM, FP32 (trained)      | 0.545     | 11.9         | 0.9381 | 0.909     | 0.972  |
| E4  | LSTM + PTQ                      | **0.185** | 17.4         | 0.9426 | 0.921     | 0.968  |
| E5  | Distilled student (FP32)        | 0.134     | 13.5         | 0.8671 | 0.790     | 0.973  |
| E6  | Distilled student + PTQ         | **0.050** | 13.8         | 0.8704 | 0.795     | 0.973  |

**Key findings**

- **Dynamic PTQ is a no-op here** (E1a): only 1.2% of params are in quantizable
  `nn.Linear`; the dominant Conv2d/GRU weights are out of scope, so the file
  barely shrinks. QAT scoped to that 1.2% (E1b/E1c) only hurts F1.
- **E4 is the sweet spot** — 0.185 MB at F1 0.9426, under the 200 KB target and
  within ~1.6 points of baseline. Reached by swapping the un-quantizable GRU for
  an LSTM (which dynamic quant *does* support) and then quantizing.
- **E6 is the smallest** — 0.050 MB (8.7× smaller than baseline), but F1 falls
  ~9 points. Below ~100 KB the size–F1 trade-off steepens faster than
  distillation absorbs it.

Size vs F1 Pareto front: [`results/pareto_e0_e6.png`](results/pareto_e0_e6.png).

---

## Project Structure

```
.
├── vad_compression_local.ipynb   # Main notebook — E0–E6 on local Apple Silicon (source of truth)
├── vad_experiment_colab.ipynb    # First-pass notebook (E0–E1c) as run on x86 Colab
├── report.md                     # Full technical report with analysis
├── results/                      # Committed CSVs + plots
└── archive/
    └── vad_experiment_colab.py   # Archived Python export (no longer maintained)
```

Heavy artefacts (dataset, pretrained and trained checkpoints) live
outside the repo under `~/vad_data/`.

---

## Setup & run

```bash
pip install torch torchaudio numpy matplotlib pandas speechbrain
```

Open `vad_compression_local.ipynb` and run top to bottom. `DATA_MODE` (cell 2)
controls how much LibriParty is pulled into `~/vad_data/`:

| Value     | Description                                                      |
| --------- | --------------------------------------------------------------- |
| `"demo"`  | Single example WAV, no dataset download (smoke test, no F1).     |
| `"small"` | First 20 eval sessions (~4.75 GB) — enough for E0–E1c F1.        |
| `"full"`  | Full archive (~10 GB), includes the train split needed by E2–E6.|

> **Note**: `torch.quantization.quantize_dynamic` is deprecated in PyTorch ≥ 2.10;
> the recommended migration path is [`torchao`](https://github.com/pytorch/ao).

---

## Architecture

The CRDNN model processes raw 16 kHz waveforms in stages:

```
wav [B, T] → Mel features → Mean-Var Norm → CNN → RNN (GRU) → DNN → logits [B, T_frames, 1]
```

`CRDNNWrapper` assembles the SpeechBrain sub-modules (`compute_features`,
`mean_var_norm`, `cnn`, `rnn`, `dnn`) into a single `nn.Module` for unified
latency benchmarking and gradient-based fine-tuning. An optional `rnn=` override
enables the E3/E4 GRU→LSTM swap without touching the other sub-modules.

---

## Why dynamic quantization alone fails

`quantize_dynamic` supports only `nn.Linear`. The CRDNN model's dominant layers
fall outside its scope:

- **GRU**: `flatten_parameters()` inside SpeechBrain's RNN wrapper is
  incompatible with the dynamic-quantized variant. (Swapping to `nn.LSTM`, which
  *is* supported, is the E3/E4 fix.)
- **Conv2d**: not supported by `quantize_dynamic` at all. (Static PTQ reaches it
  — the E2 fix.)

So dynamic PTQ converts only the DNN head (1.2% of params) to INT8, yielding
negligible size reduction — which is the whole reason the second pass (E2–E6)
exists.
