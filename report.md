# Compressing a CRDNN Voice Activity Detector for iPhone Deployment

**Model**: `speechbrain/vad-crdnn-libriparty` (CRDNN, ~108k params, 0.435 MB FP32)
**Dataset**: LibriParty (20 evaluation sessions, frame-level F1)
**Target hardware**: iPhone, via Core ML
**Last updated**: 2026-05-18

---

## 1. What we want and why

The end goal is a CRDNN-based voice activity detector that runs on iPhone with a smaller on-disk footprint than the FP32 baseline and an F1 score on LibriParty that stays close to it. Size matters more than raw throughput here — the model is already fast on CPU; what hurts is bundling a half-megabyte audio model into an iOS app, and what helps is a small `.mlpackage` that loads quickly and lives comfortably on the Apple Neural Engine.

The path from "PyTorch FP32 checkpoint" to "small Core ML model that performs as well as the baseline" is not a single conversion call. The first attempt — applying `torch.quantization.quantize_dynamic` — turned out to be the wrong tool for this architecture, for reasons that are specific to how the model is built. That failure is the most informative part of the study so far, and it sets up the next phase.

---

## 2. Model and evaluation setup

The CRDNN pipeline ingests a 16 kHz raw waveform and produces one speech probability per 10 ms frame:

```
wav [B, T_samples]
  → compute_features   (Mel filterbank, 80 bins, 10 ms hop)
  → mean_var_norm      (per-utterance mean-variance normalisation)
  → CNN                (Conv2d blocks; feature extraction)
  → RNN                (GRU; temporal modelling)
  → DNN                (Linear + ReLU; classification head)
  → sigmoid → per-frame speech probability
```

`CRDNNWrapper` wraps this as a single `nn.Module`. The wrapper exists for two reasons: end-to-end gradient flow so QAT can fine-tune the whole stack, and a single forward pass that can be timed as one unit. Without it the SpeechBrain pipeline holds the sub-modules in a dict and there is no single object to call.

Evaluation is frame-level F1 against the 20-session LibriParty eval split. Per-session F1 has SEM ≈ 0.0035 (n = 20); deltas smaller than that are within noise. Latency is the median of 100 timed runs on a single CPU thread after 30 warm-up forward passes on a synthetic 3-second `randn(1, 48000)` input, with a shared `global_warmup` (50 passes on the FP32 wrapper) executed first so every condition starts from the same steady state (FFTW plan, mel filterbank cache, thread pool).

Size is measured by serializing `mods['model'].state_dict()` to a temp file and reading the file size. Only the composite `mods['model']` is serialized — the sub-module handles share weights with it.

---

## 3. First pass: dynamic quantization on the existing model

Four conditions were run end-to-end on the pre-trained checkpoint:

| ID  | Method                       | Size (MB) | Latency median (ms) | F1     |
| --- | ---------------------------- | --------- | ------------------- | ------ |
| E0  | FP32 baseline                | 0.435     | 49.8                | 0.9587 |
| E1a | Dynamic PTQ                  | 0.434     | 50.6                | 0.9594 |
| E1b | QAT (FakeQuant on DNN only)  | 0.434     | 54.2                | 0.9287 |
| E1c | PTQ + QAT (FakeQuant on all) | 0.434     | 54.4                | 0.8815 |

Two things stand out. First, all four conditions weigh **0.434–0.435 MB**: a 0.2 % difference. Second, F1 only varies meaningfully in the QAT conditions, and in the wrong direction — both E1b and E1c are worse than E0, with E1c losing nearly 8 absolute points (significantly above the 0.0035 SEM).

### 3.1 Why size barely moved

`torch.quantization.quantize_dynamic` supports `nn.Linear` and `nn.LSTM`, and nothing else. SpeechBrain's RNN block uses `GRU`, not `LSTM`, and additionally calls `flatten_parameters()`, which is incompatible with the dynamic-quantized RNN variant. Conv2d is also out of scope. Counting parameters by layer type:

- `nn.Linear` (the DNN head): **1,328 params** — 1.2 %
- `Conv2d` + `GRU` (everything else): **107,088 params** — 98.8 %

The dynamic PTQ pass converted exactly that 1.2 % to INT8, which is why the on-disk file shrank by ~0.001 MB instead of the ~75 % a "real" 4× INT8 conversion would have produced. The compression regime never matched the model's weight distribution.

### 3.2 Why QAT made F1 worse, not better

QAT (E1b) inserts `FakeQuant` only in the DNN head — the only layers that are actually quantized at inference. STE (straight-through estimator) lets gradients flow through the non-differentiable rounding. The intent is to let the head adapt to the quantization noise it will see at inference time. With 60 fine-tuning steps and lr = 1 × 10⁻⁵, recall dropped from 0.961 to 0.877 while precision climbed slightly — the head learned to predict speech more conservatively.

PTQ+QAT (E1c) made this worse by injecting FakeQuant into CNN and GRU as well. Those layers are not actually quantized at inference (dynamic PTQ can't touch them), so the model trains against weight noise it will never see again at test time. The result is a systematic train/test distribution gap, and F1 falls another 4.7 points.

### 3.3 What the first pass settles

Dynamic PTQ is a no-op on this architecture. QAT, scoped to where dynamic PTQ can reach, makes things worse. Any meaningful compression has to actually shrink the Conv2d and recurrent layers, which means either changing how those layers are quantized (static instead of dynamic), changing the layer types themselves (LSTM in place of GRU), or making the network smaller in the first place (distillation). The next phase does all three.

---

## 4. Second pass: a plan that targets the 98.8 % we missed

The next phase is organised as two tracks — quantization and distillation — that combine into one final condition. Every new condition keeps the same E0 evaluation protocol so the table at the end is directly comparable.

### 4.1 Phase 0 — prerequisites

Before any new condition runs, three pieces of infrastructure need to be in place:

1. `CRDNNWrapper` is refactored to take the RNN type as a parameter (`gru` or `lstm`) so the architecture swap in §4.2 is a one-line change, not a separate model class.
2. A static PTQ helper using `torch.ao.quantization` — the standard `prepare → calibrate → convert` flow with `fbgemm` on x86 and `qnnpack` on ARM.
3. A calibration set: 100–200 frames sampled from the **train** split. Using eval frames for calibration is a common mistake that quietly inflates F1; the calibration data must be disjoint from the data used for the final F1 measurement.

### 4.2 Phase 1 — quantization track

| ID  | Setup                              | What it tests                                                                                  |
| --- | ---------------------------------- | ---------------------------------------------------------------------------------------------- |
| E2  | Static PTQ on the GRU model        | Whether quantizing Conv2d activations gives a partial size win while GRU still blocks the rest. |
| E3  | GRU → LSTM, FP32 (no quantization) | Isolation — the only way to attribute E4's F1 change to quantization rather than the LSTM swap. |
| E4  | LSTM + static PTQ                  | The track's endpoint. Both Conv2d and LSTM are now quantizable. Expected ~100 KB on disk.       |

E3 is non-negotiable: without an LSTM-FP32 reading, any F1 movement in E4 is unattributable. Skipping it would leave the report with a result it cannot defend.

### 4.3 Phase 2 — distillation track

A smaller student network is trained to match the FP32 teacher (E0), then quantized.

| ID  | Setup                                                                                                                                                          |
| --- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| E5  | LSTM student with CNN channels × 0.5 and RNN hidden × 0.5. Loss: `α · KL(student/T, teacher/T) · T² + (1 − α) · BCE`. Starting hyperparameters: α = 0.5, T = 2. |
| E6  | E5 + static PTQ. Expected the smallest condition in the table, on the order of 25–50 KB.                                                                       |

α and T are starting points, not commitments — both will be tuned on a validation slice before locking in E5's final F1 number.

### 4.4 Phase 3 — analysis

When E2–E6 are in, the report's results section is rewritten against the combined table (E0–E6 plus the E1 ablations) and a Pareto plot of size on a log axis against F1. The current §3 narrative becomes "what we tried first and why it didn't work"; the new results section becomes "what worked".

### 4.5 Phase 4 — iPhone deployment

Core ML is the runtime, chosen because the deployment target is iOS — not because it is the best inference engine in general. The conversion path is:

```
trained PyTorch model
  → torch.jit.trace (on a representative input)
  → coremltools.convert  → .mlpackage
```

Both `compute_units = CPU_ONLY` and `compute_units = CPU_AND_NE` are measured. The metrics reported on iPhone are:

- parameter count,
- `.mlpackage` size on disk,
- cold-start latency (first inference, includes ANE compile),
- steady-state latency (median of runs ≥ 11).

Energy is intentionally out of scope. Cold-start matters because on iPhone the first inference after app launch dominates user-perceived latency; steady-state matters for sustained use.

### 4.6 Minimum viable version

If time runs out, E2 is the first cut: it is a useful diagnostic but not load-bearing for the narrative. The minimum path that still produces a defensible result is **E3 → E4 → E5 → E6**, with Phase 4 optional. The story is then "we couldn't shrink the GRU model with dynamic PTQ, so we changed the architecture and distilled it; here is the result."

---

## 5. What "done" looks like

The study is finished when:

1. At least one condition (likely E4 or E6) has on-disk size below ~200 KB.
2. That condition's F1 on LibriParty stays within ~2 absolute points of E0.
3. The same condition runs end-to-end as a Core ML model on iPhone from a notebook, with documented cold-start and steady-state latency.

If criterion 1 is met but criterion 2 is not, that is still a valid outcome of the study: it would say the size–F1 trade-off on this model is sharper than knowledge distillation can absorb, and the report ends with that as its finding rather than a success claim it cannot back up.

---

## Appendix: implementation notes from the first pass

**Quantization backend.** Must be set before any quantized operator runs:

```python
if platform.machine() in ("arm64", "aarch64"):
    torch.backends.quantized.engine = "qnnpack"
else:
    torch.backends.quantized.engine = "fbgemm"
```

**FakeQuantLinear (E1b / E1c).** Symmetric per-tensor INT8 with STE on the backward pass:

```python
scale = w.abs().max() / 127.0
w_fq  = (w / scale).round().clamp(-128, 127) * scale
w_ste = w + (w_fq - w).detach()       # forward: quantized; backward: identity
return F.linear(x, w_ste, self.bias)
```

**Dual-path quantization.** The SpeechBrain inference call (`get_speech_prob_file`) runs through `mods['model']`; `CRDNNWrapper` reads the sub-module handles (`mods['cnn']`, `mods['rnn']`, `mods['dnn']`) directly. Quantizing only one path leaves the two evaluation routes disagreeing about weights, so both are quantized in one call.

**Size measurement.**

```python
torch.save(vad_obj.mods['model'].state_dict(), tmp_path)
size_mb = os.path.getsize(tmp_path) / (1024 ** 2)
```
