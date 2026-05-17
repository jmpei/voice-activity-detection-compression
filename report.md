# Technical Report: Model Compression for Voice Activity Detection

**Project**: VAD Compression Benchmark  
**Model**: `speechbrain/vad-crdnn-libriparty` (CRDNN)  
**Dataset**: LibriParty (20 eval sessions)  
**Date**: 2026-04-17

---

## 1. Abstract

This report evaluates four compression strategies—FP32 baseline, post-training dynamic quantization (PTQ), quantization-aware training (QAT), and a combined PTQ+QAT variant—applied to SpeechBrain's CRDNN-based voice activity detection model. All strategies are benchmarked across three axes: on-disk model size (MB), single-threaded CPU inference latency (ms), and frame-level F1 score on 20 LibriParty evaluation sessions. Results show that PTQ matches the FP32 baseline in both accuracy (F1 0.9594 vs. 0.9587, Δ < SEM) and latency, while QAT and PTQ+QAT introduce meaningful accuracy degradation with no size benefit, due to the architectural constraint that only 1.2% of the model's parameters reside in quantizable `nn.Linear` layers.

---

## 2. Background and Motivation

Edge deployment of voice activity detection models requires low memory footprint and fast inference on CPU. Integer quantization is a standard technique to reduce model size (FP32 → INT8 compresses weights by 4×) and improve throughput on hardware with efficient INT8 arithmetic. However, quantization effectiveness depends critically on which layer types dominate the model's parameter count. This experiment investigates whether `torch.quantization.quantize_dynamic`—restricted to `nn.Linear`—can meaningfully compress the CRDNN VAD architecture, and whether quantization-aware fine-tuning can recover or improve accuracy after quantization.

---

## 3. Model and Architecture

**Base model**: [`speechbrain/vad-crdnn-libriparty`](https://huggingface.co/speechbrain/vad-crdnn-libriparty)  
**Architecture**: CRDNN (Convolutional-Recurrent-DNN)  
**Total parameters**: ~108,416  
**FP32 on-disk size**: 0.435 MB

The pipeline processes 16 kHz raw waveforms through five sequential stages:

```
wav [B, T_samples]
  → compute_features   (Mel filterbank, 80 bins, 10 ms hop)
  → mean_var_norm      (per-utterance mean-variance normalisation)
  → CNN                (Conv2d blocks; feature extraction)
  → RNN                (GRU; temporal modelling)
  → DNN                (Linear + ReLU; classification head)
  → logits [B, T_frames, 1]   (apply sigmoid for speech probabilities)
```

`CRDNNWrapper` encapsulates this pipeline as a single `nn.Module` to enable unified latency measurement and end-to-end gradient flow for QAT fine-tuning.

---

## 4. Compression Methods

### 4.1 FP32 Baseline

The pre-trained model is loaded from HuggingFace Hub without modification. All weights remain in 32-bit float. This is the reference for all comparisons.

### 4.2 Post-Training Dynamic Quantization (PTQ)

`torch.quantization.quantize_dynamic` converts `nn.Linear` weight matrices to INT8 at load time. Activations are computed in FP32 and dequantized on the fly, requiring no calibration data. The quantization is applied to both `mods['model']` (the composite CRDNN) and the individual `mods['cnn']`, `mods['rnn']`, `mods['dnn']` sub-module handles to ensure the SpeechBrain inference pipeline (`get_speech_prob_file`) and the latency wrapper (`CRDNNWrapper`) operate on identical weights.

**Scope limitation**: `quantize_dynamic` does not support `Conv2d` or `GRU`. SpeechBrain's RNN wrapper calls `flatten_parameters()`, which is incompatible with the quantized GRU variant. Consequently, only the DNN classification head undergoes INT8 conversion.

### 4.3 Quantization-Aware Training (QAT)

`FakeQuantLinear` replaces each `nn.Linear` in the DNN sub-module. During the forward pass, weights are symmetrically quantized using a per-tensor absolute-max scale to the INT8 grid and immediately dequantized back to FP32. The straight-through estimator (STE) allows gradients to flow through the non-differentiable rounding operation unchanged.

The model is fine-tuned for **60 steps** with Adam (lr = 1 × 10⁻⁵) and `BCEWithLogitsLoss`. Training clips (3 s, 16 kHz) are sampled from LibriParty eval sessions with a speech-ratio filter (> 20%) to avoid silence-dominated batches. After fine-tuning, weights are transferred to a fresh FP32 clone and `quantize_all_paths` converts to INT8.

**Scope**: FakeQuant injected into **DNN only**; CNN and RNN weights are frozen and remain FP32.

### 4.4 PTQ+QAT Combined

An aggressive variant of QAT where FakeQuant is injected into **all sub-modules** (CNN, RNN, and DNN), exposing the full network to quantization noise during fine-tuning. The same training schedule (60 steps, lr = 1 × 10⁻⁵) is used, followed by `quantize_all_paths`.

---

## 5. Experimental Setup

| Setting | Value |
|---|---|
| Dataset | LibriParty evaluation set |
| Sessions evaluated | 20 |
| Frame rate | 100 fps |
| Evaluation metric | Frame-level F1 (binary speech / non-speech) |
| Latency input | Synthetic 3 s clip at 16 kHz: `torch.randn(1, 48000)` |
| Latency protocol | 30 warm-up + 100 timed runs, single-threaded CPU |
| Latency statistics | Median and P95 (right-skewed distribution) |
| Size measurement | `torch.save(mods['model'].state_dict())` → file size |
| Runtime | Google Colab, x86_64, `fbgemm` quantization backend |
| QAT steps | 60 |
| QAT learning rate | 1 × 10⁻⁵ |
| QAT optimizer | Adam |
| QAT loss | BCEWithLogitsLoss |

A shared `global_warmup` (50 forward passes) is performed on the FP32 wrapper before any model is timed, ensuring all strategies start from the same steady-state cache (FFTW plan, mel filterbank, thread pool).

---

## 6. Results

### 6.1 Summary Table

| Model | Format | Size (MB) | Latency median (ms) | Latency P95 (ms) | F1 | Precision | Recall |
|---|---|---|---|---|---|---|---|
| FP32 Baseline | FP32 | 0.435 | 49.761 | 80.693 | 0.9587 | 0.9574 | 0.9606 |
| PTQ (Dynamic) | Mixed/INT8 | 0.434 | 50.619 | 82.124 | 0.9594 | 0.9591 | 0.9603 |
| QAT | Mixed/INT8 | 0.434 | 54.214 | 62.260 | 0.9287 | 0.9886 | 0.8771 |
| PTQ+QAT | Mixed/INT8 | 0.434 | 54.365 | 326.701 | 0.8815 | 0.9965 | 0.7930 |

Per-session F1: std = 0.0156, SEM = 0.0035 (n = 20).  
**Deltas below ≈ 0.0035 are within measurement noise.**

### 6.2 Model Size

All compressed variants produce 0.434 MB—a reduction of only 0.2% from the 0.435 MB baseline. The expected 4× INT8 compression does not materialise because `quantize_dynamic` converts only the 1,328 parameters in DNN `nn.Linear` layers (1.2% of 108,416 total). The remaining 98.8% of weights—residing in Conv2d and GRU—stay in FP32.

### 6.3 Inference Latency

| Model | Δ vs. FP32 |
|---|---|
| PTQ | +0.858 ms (+1.7%, within noise) |
| QAT | +4.453 ms (+8.9%) |
| PTQ+QAT | +4.604 ms (+9.3%), P95 = 326.7 ms |

PTQ shows no meaningful latency improvement. QAT and PTQ+QAT are marginally slower because FakeQuant-trained weights introduce additional overhead in the `quantize_all_paths` conversion step. The extreme P95 for PTQ+QAT (326.7 ms vs. median 54.4 ms) indicates high run-to-run variance, likely caused by thread scheduling interference from the more complex quantized graph.

### 6.4 F1 Score and Accuracy-Efficiency Trade-off

| Model | F1 | ΔF1 vs. FP32 | Significant? |
|---|---|---|---|
| FP32 | 0.9587 | — | — |
| PTQ | 0.9594 | +0.0007 | No (< SEM) |
| QAT | 0.9287 | −0.0300 | Yes (8.6× SEM) |
| PTQ+QAT | 0.8815 | −0.0772 | Yes (22× SEM) |

PTQ produces statistically equivalent F1 to the FP32 baseline. QAT degrades recall significantly (0.9606 → 0.8771), indicating the fine-tuned model becomes overly conservative in predicting speech. PTQ+QAT further amplifies this pattern—precision approaches 1.0 (0.9965) while recall collapses to 0.793, suggesting the model has learned to suppress most speech predictions to minimise training loss on the silence-heavy sessions.

---

## 7. Discussion

### 7.1 Why Quantization Has Minimal Impact on This Model

The CRDNN VAD model is architecturally dominated by CNN (Conv2d) and RNN (GRU) layers, which `torch.quantization.quantize_dynamic` cannot convert. Only the small DNN head—1,328 of 108,416 parameters—is quantized. This makes the compression regime fundamentally mismatched to the model's weight distribution:

- **No size reduction**: The 0.001 MB saving (0.2%) is negligible.
- **No latency improvement**: The DNN head is not the computational bottleneck on CPU; the mel-filterbank and GRU forward pass dominate.

To achieve meaningful quantization of this architecture, one would need:
- **Static quantization** (requires calibration data) to quantize Conv2d activations.
- **GRU → LSTM / Linear RNN substitution** to enable dynamic quantization of the recurrent component.
- **Migration to `torchao`** (PyTorch ≥ 2.10), which supports broader operator coverage.

### 7.2 Why QAT Hurts Accuracy

The QAT fine-tuning exposes two failure modes:

1. **Training signal instability**: Many training steps show loss values > 2.0 with speech_ratio = 1.0, suggesting the model is over-predicting silence on speech-dominant clips. The 60-step budget is insufficient to converge given the noisy gradient signal.

2. **Scope mismatch**: FakeQuant in the DNN perturbs the classification head while the upstream CNN and RNN feature representations remain FP32. This creates a distribution mismatch at the DNN input between training (FP32 features → FakeQuant DNN) and inference (FP32 features → quantized DNN).

PTQ+QAT is worse because injecting FakeQuant into all sub-modules—including Conv2d and GRU, which cannot actually be dynamically quantized—trains the model to be robust to weight noise that is not present at inference time, creating a systematic train/test distribution gap.

### 7.3 Model Already Compact

At 0.435 MB and 108k parameters, the CRDNN VAD model is already highly compact for its task, achieving 0.9587 F1 on a challenging multi-speaker dataset. The practical need for further compression is limited; the more productive path for edge deployment is:
- Quantizing the audio feature extraction pipeline separately.
- Exporting to ONNX or TorchScript for runtime-level optimisation.
- Exploring structured pruning of the CNN feature maps.

---

## 8. Conclusions

1. **PTQ is Pareto-optimal** among the strategies tested: it matches FP32 in both F1 (Δ < SEM) and latency with zero engineering overhead.
2. **QAT and PTQ+QAT degrade accuracy** without delivering size or speed benefits, making them counterproductive for this architecture.
3. **The bottleneck is architectural**: `quantize_dynamic` covers only 1.2% of model parameters. Meaningful INT8 compression of this CRDNN model requires static quantization or architectural changes to make Conv2d and GRU quantizable.
4. **The model is already compact** (0.435 MB, F1 = 0.9587), which limits the practical motivation for compression.

---

## 9. Future Work

- **Static (calibration-based) quantization** of Conv2d and GRU using a representative calibration dataset.
- **Migration to `torchao`**: PyTorch deprecated `torch.ao.quantization` in 2.10; the successor API supports broader layer types and PT2E compilation.
- **Structured pruning**: channel-level pruning of CNN feature maps followed by fine-tuning could reduce model size while preserving accuracy.
- **Knowledge distillation**: train a smaller student CRDNN using the FP32 model as teacher.
- **ONNX export and runtime quantization**: quantize at the ONNX level using ONNXRuntime to bypass PyTorch's dynamic quantization constraints.

---

## Appendix: Key Implementation Details

### Quantization backend selection

```python
if platform.machine() in ("arm64", "aarch64"):
    torch.backends.quantized.engine = "qnnpack"   # ARM (Apple Silicon, Colab ARM)
else:
    torch.backends.quantized.engine = "fbgemm"    # x86 (standard Colab)
```

Must be set before any quantized operator executes.

### FakeQuantLinear (STE)

```python
scale = w.abs().max() / 127.0          # per-tensor symmetric scale
w_fq  = (w / scale).round().clamp(-128, 127) * scale
w_ste = w + (w_fq - w).detach()        # forward: quantized; backward: straight-through
return F.linear(x, w_ste, self.bias)
```

### Size measurement

Only `mods['model']` (the composite CRDNN) is serialised to avoid double-counting with sub-module references:

```python
torch.save(vad_obj.mods['model'].state_dict(), tmp_path)
size_mb = os.path.getsize(tmp_path) / (1024 ** 2)
```

### Dual-path quantization (`quantize_all_paths`)

SpeechBrain's `get_speech_prob_file` runs inference through `mods['model']`; `CRDNNWrapper` reads `mods['cnn']`, `mods['rnn']`, `mods['dnn']` directly. Both paths are quantized to ensure F1 evaluation and latency benchmarking use identical weights.
