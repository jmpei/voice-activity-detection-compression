# Compressing a CRDNN Voice Activity Detector

**Model**: `speechbrain/vad-crdnn-libriparty` (CRDNN, 109,744 params, 0.435 MB FP32)
**Data**: LibriParty — 250 train sessions, 10 dev sessions (early stopping), 50 eval sessions (all reported F1)
**Hardware**: Apple-Silicon Mac, single-thread CPU, PyTorch 2.8 `qnnpack`
**Source of every number**: `results/`, written by one top-to-bottom run of `vad_compression_local.ipynb`

---

## 1. Goal

Make the VAD smaller on disk for on-device (iPhone) use while keeping frame-level F1 close to the FP32 model. Two targets: a model under ~200 KB, and F1 within ~2 absolute points of FP32. Three compression methods are compared: post-training quantization (PTQ), quantization-aware training (QAT), and knowledge distillation.

## 2. Setup

**Model.** SpeechBrain's pipeline turns a 16 kHz waveform into one speech probability per 10 ms frame:

```
wav → Fbank (40 mel) → mean-var norm → CNN → RNN → DNN → sigmoid
```

| Block | Layers | Weight params |
|---|---|---|
| CNN | 2 blocks of Conv2d + LayerNorm + LeakyReLU, Dropout2d | Conv2d: 16,368 |
| RNN | 2-layer bidirectional GRU, hidden 32 | GRU: 86,784 |
| DNN | 2 × (Linear → BatchNorm1d → LeakyReLU → Dropout) + Linear | Linear: 1,328 |

The GRU holds 79% of the parameters, the DNN's `nn.Linear` layers 1.2% (`results/param_breakdown.json`).

**F1.** Each eval session is scored over its whole wav: predicted speech boundaries from SpeechBrain's own post-processing (`get_speech_prob_file → apply_threshold → get_boundaries`, default thresholds) are turned into frame labels and compared with the union of all speakers' utterances. Reported F1 is the mean of the 50 per-session F1s. Conditions that swap sub-modules route only the neural part through the swapped modules; `tests/` checks that with the original modules this path gives the same chunk probabilities (max-abs-diff 0.0) and the same per-session F1 as SpeechBrain's native call.

**Size.** File size of the saved `state_dict`s of the neural modules (`cnn`, `rnn`, `dnn`, or the student). The mel front-end has no weights to compress and is not counted.

**Latency.** Median of 100 single-thread CPU forward passes on a 3 s input (waveform in, logits out, front-end included) after 30 warm-up passes.

**Seeds and statistics.** Every trained condition is run with seeds 0, 1, 2. "± std" is across seeds. Comparisons are paired on the same 50 eval sessions, using per-session F1 averaged over seeds: mean difference, 95% t-interval, and Wilcoxon signed-rank p.

**Training data.** Fine-tuning, LSTM training and distillation draw uniformly random 3 s clips from the train split, which keeps the label prior of the data (speech share: train split 62%, sampled clips 62%, eval split 47%; `results/data_stats.json`). Static PTQ is calibrated on 64 random 3 s train clips. No eval session is used for training, calibration or model selection.

## 3. Results

| ID | Method | Params | Size (MB) | Latency (ms) | F1 | Δ F1 vs E0 [95% CI] |
|---|---|---|---|---|---|---|
| E0 | FP32 baseline | 109,744 | 0.435 | 13.2 | 0.9566 | — |
| E1a | Dynamic PTQ (`nn.Linear` → INT8) | 109,744 | 0.434 | 13.9 | 0.9570 | 0.0004 [0.0001, 0.0008] |
| FT | Fine-tune FP32, then dynamic PTQ (control) | 109,744 | 0.434 | 14.7 | 0.9577 ± 0.0006 | 0.0011 [−0.0001, 0.0023] |
| E1b | QAT: FakeQuant on DNN Linear, then dynamic PTQ | 109,744 | 0.434 | 15.4 | 0.9574 ± 0.0008 | 0.0009 [−0.0001, 0.0018] |
| E2 | Static PTQ of the CNN | 109,744 | 0.397 | 16.8 | 0.9572 | 0.0006 [−0.0004, 0.0016] |
| E3 | GRU → LSTM, FP32 | 138,672 | 0.545 | 12.8 | 0.9421 ± 0.0086 | −0.0145 [−0.0219, −0.0070] |
| E4 | E3 + static PTQ CNN + dynamic PTQ LSTM/DNN | 138,672 | **0.185** | 17.2 | **0.9425 ± 0.0103** | −0.0141 [−0.0221, −0.0061] |
| E5-hard | Student, hard labels only (control) | 33,369 | 0.136 | 9.3 | 0.9249 ± 0.0066 | −0.0316 [−0.0399, −0.0234] |
| E5 | Student, distilled from E0 | 33,369 | 0.136 | 9.4 | 0.9297 ± 0.0072 | −0.0269 [−0.0342, −0.0195] |
| E6 | E5 + PTQ | 33,369 | **0.053** | 10.7 | 0.9305 ± 0.0063 | −0.0261 [−0.0335, −0.0186] |

Precision, recall, p-values and the 8-clip calibration row are in `results/summary.csv`; per-seed rows in `results/conditions.csv`; per-session F1 in `results/per_session.csv`. Size vs F1: `results/pareto.png`.

Paired comparisons that isolate one change each (`results/pairwise.csv`):

| Change | From → to | Δ F1 [95% CI] | Wilcoxon p |
|---|---|---|---|
| FakeQuant during fine-tuning | FT → E1b | −0.0003 [−0.0008, 0.0003] | 0.082 |
| QAT vs plain dynamic PTQ | E1a → E1b | 0.0004 [−0.0004, 0.0012] | 0.25 |
| Quantizing the LSTM model | E3 → E4 | 0.0004 [−0.0011, 0.0018] | 0.016 |
| Distillation vs hard labels | E5-hard → E5 | 0.0047 [0.0017, 0.0078] | 0.0039 |
| Quantizing the student | E5 → E6 | 0.0008 [0.0001, 0.0015] | 0.0078 |

## 4. Findings

### 4.1 PTQ

**Dynamic PTQ is a no-op on this architecture (E1a).** `quantize_dynamic` converts `nn.Linear` and `nn.LSTM` only. Here that is the DNN head, 1.2% of the parameters; the Conv2d and GRU weights stay FP32, so the file shrinks by 0.3%. F1 is unchanged.

**Static PTQ reaches the CNN at no F1 cost (E2).** Eager-mode prepare → calibrate → convert quantizes the Conv2d layers (the CNN uses LayerNorm, so there is no BatchNorm to fuse). The model shrinks 8.7%, to 0.397 MB, and F1 does not move. Calibrating on 8 clips instead of 64 gives practically the same F1 (0.9576 vs 0.9572, `E2-8clip` in `results/summary.csv`). The GRU, 79% of the weights, is still FP32.

**Swapping the GRU for an LSTM makes the recurrent layer quantizable (E3 → E4).** `quantize_dynamic` supports `nn.LSTM` but not `nn.GRU`. E3 replaces the GRU with a fresh LSTM of the same shape and trains only the LSTM, with the pretrained CNN and DNN frozen. An LSTM has four gates to the GRU's three, so E3 is larger than E0 (0.545 MB). Quantizing it (E4) brings the model to 0.185 MB, 57% below E0, and changes F1 by 0.0004 relative to E3. So quantization itself is lossless here. The 1.4-point gap to E0 comes from the GRU → LSTM swap: a fresh LSTM trained under a DNN head that was fitted to the GRU's outputs. The swap also makes results less stable across seeds (E3 std 0.0086; seed 0 has precision 0.971 and recall 0.904, seeds 1 and 2 the reverse balance, see `results/conditions.csv`).

**INT8 is slower on this CPU, not faster.** E4 runs at 17.2 ms against 12.8 ms for E3. Dynamic quantization re-quantizes activations on every call, and on matrices this small that overhead outweighs the cheaper INT8 math. The gain from PTQ here is size, not speed.

### 4.2 QAT

**FakeQuant has no measurable effect (FT vs E1b).** E1b fine-tunes the whole CRDNN with FakeQuant (symmetric per-tensor INT8 weights, straight-through gradient) on the three DNN `nn.Linear` layers. Those are the only layers dynamic PTQ converts, and the only `nn.Linear` layers in the model. Then E1b applies the same dynamic PTQ as E1a. The control, FT, runs the same fine-tuning without FakeQuant. Both stay at the baseline, and E1b − FT is −0.0003 with a CI that spans zero. This is expected: dynamic PTQ of these layers is already lossless (E1a), so QAT has nothing to recover.

**Why earlier runs reported QAT losing up to 11 points.** They fine-tuned in train mode with batch size 1. The DNN's two BatchNorm1d layers then normalise each frame with the statistics of a single 3 s clip. On a clip that is all speech, this subtracts the "speech" signal itself. The pretrained model's BCE on all-speech train clips is 0.0248 in eval mode and 2.7686 with BatchNorm in train mode. Train mode also overwrites the running statistics that inference uses. The notebook re-runs that recipe (`results/qat_diagnostic.json`): F1 0.9281, with precision 0.986 and recall 0.878. Then it puts the pretrained BatchNorm running statistics back into the same fine-tuned weights, and F1 returns to 0.9549 (E0: 0.9566). So the loss comes from BatchNorm statistics, not from quantization. Re-run here, the recipe costs 2.9 points rather than up to 11, because the earlier runs also used a clip sampler that fed it mostly all-speech clips (§7), which is exactly the case shown above where batch statistics do the most damage. The fixed recipe keeps the network in eval mode while gradients flow, with batch size 16.

**E1c is dropped.** The first pass had a "PTQ + QAT on all sub-modules" condition. The CNN and GRU contain no `nn.Linear`, so FakeQuant on all sub-modules touches the same three layers as E1b (`tests/test_vadlib.py::test_fakequant_scope_is_the_three_dnn_linears`). The F1 difference between the two first-pass runs was seed noise.

### 4.3 Distillation

**A third of the parameters costs about 3 points (E5-hard, E5).** The student has 33,369 parameters: two Conv2d blocks with 8 and 16 channels, a 2-layer bidirectional LSTM with hidden 16, and a linear head. It is trained from scratch twice, with the same architecture, steps and learning rate: once on hard labels only (E5-hard), and once with `0.5 · T² · BCE(s/T, σ(t/T)) + 0.5 · BCE(s, y)` at T = 2, using E0 as the teacher (E5).

**Distillation helps, but only a little.** E5 beats E5-hard by 0.0047 F1 (CI 0.0017–0.0078). That is about half a point, small next to the 2.7-point gap that remains to E0.

**Quantizing the student is lossless (E6).** Static PTQ on its CNN and dynamic PTQ on its LSTM and head give the smallest model in the study: 0.053 MB, 8.3× smaller than E0. F1 is unchanged relative to E5.

## 5. Assessment

| Target | E4 | E6 |
|---|---|---|
| Size < ~200 KB | 0.185 MB — met | 0.053 MB — met |
| F1 within ~2 points (0.02) of E0 | Δ −0.0141, 95% CI [−0.0221, −0.0061] | Δ −0.0261 — not met |

E4 meets the size target. Its F1 point estimate is 1.4 points below E0, but the confidence interval reaches 2.2 points, so "within 2 points" holds on average and is not guaranteed. E6 is 3.5× smaller again and gives up another 1.2 points.

## 6. Limitations

- Latency is measured on a Mac CPU with PyTorch eager mode, not on an iPhone or with an iOS runtime.
- One dataset (LibriParty), 50 eval sessions, 3 seeds per trained condition.
- Early stopping uses 10 dev sessions.
- E3/E4 train only the LSTM, so the DNN head stays the one fitted to the GRU. Fine-tuning the head with the LSTM might close part of the 1.4-point gap; it was not tried.
- The detection thresholds are SpeechBrain's defaults for every condition and are not tuned per model.

## 7. What changed from the first pass

The first pass (`vad_experiment_colab.ipynb`, x86 Colab) and an earlier local run produced the numbers this report used to carry. Each item below is fixed in `vadlib.py` and pinned by a test where it can be.

| Issue in the earlier runs | Effect | Fix |
|---|---|---|
| QAT fine-tuned in train mode with batch 1 | BatchNorm statistics drifted; F1 fell up to 11 points and was blamed on QAT | Eval-mode fine-tuning; BatchNorm diagnostic in the QAT findings |
| No fine-tuning control for QAT, no hard-label control for distillation | Effects of FakeQuant and of distillation could not be isolated | FT and E5-hard conditions |
| Training clips chosen to contain speech | 91% of training frames were speech, against 62% in the train split; every trained model over-predicted speech | Uniformly random clips |
| Labels ended at the last utterance | 1.9% of eval audio (up to 53 s in one session) was never scored | Labels cover the whole wav |
| First 20 eval sessions in string order (0, 1, 10, 11, …) | Undocumented subset | All 50 eval sessions |
| "PTQ + QAT on all sub-modules" (E1c) | Same three layers as E1b | Dropped |
| Single seed; noise judged against 1 × SEM | Seed effects and noise not separable | 3 seeds; paired CI and Wilcoxon |
| Notebook run in pieces across kernels; summary cell never executed | One table cell (E3 latency) matched no run | Headless top-to-bottom run; `tools/check_numbers.py` |
| Trained checkpoints kept outside the repo | Results could not be regenerated after they were deleted | `checkpoints/` with recipe-checked training logs |

Colab latencies were also unusable as measurements: the PTQ + QAT run showed a 54.4 ms median with a 326.7 ms P95, a tail from the shared virtual CPU. That is why all timing moved to a local machine.

## Appendix: reproducing

```bash
uv venv .venv && uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/pytest                                   # unit + integration tests
.venv/bin/jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=-1 vad_compression_local.ipynb
.venv/bin/python tools/check_numbers.py README.md report.md
```

The notebook expects LibriParty under `~/vad_data/LibriParty` (or `$VAD_DATA`). It loads the trained weights from `checkpoints/`; deleting a checkpoint retrains it with the recipe in `vadlib.py` (about 15 minutes per model on CPU).
