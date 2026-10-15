# Voice Activity Detection — Compression Benchmark

Compresses **SpeechBrain's CRDNN VAD**
([`speechbrain/vad-crdnn-libriparty`](https://huggingface.co/speechbrain/vad-crdnn-libriparty),
109,744 params, 0.435 MB FP32) with three methods — post-training quantization
(PTQ), quantization-aware training (QAT) and knowledge distillation — and measures
on-disk size, CPU latency and frame-level F1 on all 50 LibriParty eval sessions.

Full write-up: [`report.md`](report.md).

---

## Results

F1 is the mean per-session frame-level F1 over the 50 eval sessions; trained
conditions are mean ± std over 3 seeds. Δ is paired on the same sessions, with a
95% confidence interval. Latency: median of 100 single-thread CPU runs on a 3 s
input, Apple-Silicon Mac.

| ID | Method | Size (MB) | Latency (ms) | F1 | Δ F1 vs E0 [95% CI] |
|---|---|---|---|---|---|
| E0 | FP32 baseline (GRU) | 0.435 | 13.2 | 0.9566 | — |
| E1a | Dynamic PTQ (`nn.Linear` → INT8) | 0.434 | 13.9 | 0.9570 | 0.0004 [0.0001, 0.0008] |
| FT | Fine-tune FP32 + dynamic PTQ (control for E1b) | 0.434 | 14.7 | 0.9577 ± 0.0006 | 0.0011 [−0.0001, 0.0023] |
| E1b | QAT (FakeQuant on DNN Linear) + dynamic PTQ | 0.434 | 15.4 | 0.9574 ± 0.0008 | 0.0009 [−0.0001, 0.0018] |
| E2 | Static PTQ of the CNN | 0.397 | 16.8 | 0.9572 | 0.0006 [−0.0004, 0.0016] |
| E3 | GRU → LSTM, FP32 | 0.545 | 12.8 | 0.9421 ± 0.0086 | −0.0145 [−0.0219, −0.0070] |
| E4 | E3 + PTQ (CNN static, LSTM/DNN dynamic) | **0.185** | 17.2 | **0.9425 ± 0.0103** | −0.0141 [−0.0221, −0.0061] |
| E5-hard | Student, hard labels only (control for E5) | 0.136 | 9.3 | 0.9249 ± 0.0066 | −0.0316 [−0.0399, −0.0234] |
| E5 | Student distilled from E0 | 0.136 | 9.4 | 0.9297 ± 0.0072 | −0.0269 [−0.0342, −0.0195] |
| E6 | E5 + PTQ | **0.053** | 10.7 | 0.9305 ± 0.0063 | −0.0261 [−0.0335, −0.0186] |

**Findings**

- **Dynamic PTQ is a no-op here.** It converts only `nn.Linear`, which is 1.2% of
  this model's parameters; the Conv2d and GRU weights stay FP32.
- **E4 is the size/F1 sweet spot: 0.185 MB (−57%), 1.4 F1 points below E0.**
  `quantize_dynamic` supports LSTM but not GRU, so the GRU is swapped for a
  trained LSTM and then quantized. Quantization itself costs nothing (E3 → E4);
  the gap comes from the GRU → LSTM swap.
- **QAT adds nothing measurable** over the same fine-tuning without FakeQuant
  (E1b − FT = −0.0003, CI spans 0). The QAT loss in earlier runs (up to 11
  points) came from BatchNorm statistics drifting under train-mode, batch-1
  fine-tuning, not from quantization: restoring the pretrained BatchNorm
  statistics into the same weights recovers F1 (report §4.2).
- **Distillation helps a little** (E5 − E5-hard = +0.0047). E6 is the smallest
  model (0.053 MB, 8.3× smaller) at 2.6 points below E0.
- INT8 is slower than FP32 on this CPU (E4 17.2 ms vs E3 12.8 ms): the gain is
  size, not speed.

Size vs F1: [`results/pareto.png`](results/pareto.png).

---

## Reproduce

```bash
uv venv .venv && uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/pytest
.venv/bin/jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=-1 vad_compression_local.ipynb   # ~25 min on CPU
.venv/bin/python tools/check_numbers.py README.md report.md
```

LibriParty (~5 GB download, 6 GB extracted) is expected under `~/vad_data/LibriParty`
(override with `$VAD_DATA`); the pretrained SpeechBrain model is fetched into
`~/vad_data/pretrained_vad`. The notebook loads trained weights from
`checkpoints/`; a missing checkpoint is retrained with the recipe in `vadlib.py`
(about 15 min per model on CPU).

---

## Project structure

```
.
├── vadlib.py                     # data, eval path, quantization, training helpers
├── vad_compression_local.ipynb   # runs every condition and writes results/
├── tests/                        # pytest: unit tests + integration tests on the real model/data
├── checkpoints/                  # trained LSTM / student weights + training logs (recipe, dev-F1 curve)
├── results/                      # summary.csv, pairwise.csv, conditions.csv, per_session.csv, *.json, pareto.png
├── tools/check_numbers.py        # every table number in README/report must trace to results/
├── report.md
├── vad_experiment_colab.ipynb    # first pass on x86 Colab (superseded; see report §7)
└── archive/vad_experiment_colab.py
```
