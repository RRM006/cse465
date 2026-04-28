# QUANTIZATION ANALYSIS — Step-by-Step Guide

## Overview

This guide walks you through compressing and analyzing the Moshiko model. You will run experiments that measure quality, speed, and memory across different quantization methods.

**Total time:** 2-4 hours on T4 GPU

**What you will produce:**
- SNR quality numbers for bf16, q8, manual int8, int4, mixed precision
- Layer sensitivity rankings (which layers matter most)
- Comparison charts and plots
- A final report JSON with all metrics

---

## Prerequisites

### What You Must Have Done First

1. **Models downloaded on Google Drive** — from running `moshiko2.0.ipynb` previously
2. **Google Drive mounted** — Models at `/content/drive/MyDrive/Moshiko2.0_Project/`
3. **T4 GPU enabled** — Runtime → Change runtime type → GPU (T4)

### What Models You Need

| Model | Location | Size | Purpose |
|-------|----------|------|---------|
| Mimi bf16 | `models/bf16/tokenizer-e351c8d8-checkpoint125.safetensors` | ~1.4 GB | Full precision codec (baseline) |
| Mimi q8 | `models/q8/tokenizer-e351c8d8-checkpoint125.safetensors` | ~700 MB | Kyutai's INT8 codec |
| Moshiko LM q8 | `models/q8/model.q8.safetensors` | ~8.4 GB | 7B language model (INT8) |

---

## Quick Start

### Step 1: Upload the Notebook

1. Go to https://colab.research.google.com
2. Click **Upload** tab
3. Drag `quantization_analysis.ipynb` into the upload area

### Step 2: Change Runtime Type

1. Click: **Runtime → Change runtime type**
2. Set Hardware accelerator to **GPU**, GPU type to **T4**
3. Click **Save**

### Step 3: Run Cells in Order

Run Cells 1 → 18 from top to bottom.

**Cell 1** installs PyTorch 2.4.0 + moshi + dependencies, then tells you to restart.
**After restart:** Run Cell 2 → Cell 3 → Cell 4 → Cell 5 → then experiments.

### Cell Map

| Cells | Section | What It Does | Time |
|-------|---------|--------------|------|
| 1 | Dependencies | Install PyTorch 2.4.0, moshi, bitsandbytes | 3 min + restart |
| 2-5 | Setup | Folders, manifest, GPU check, load models | 5 min |
| 6-7 | Experiment 1 | Compare bf16 vs q8 (Kyutai's official) | 3 min |
| 8-9 | Experiment 2a | Manual INT8 weight quantization + comparison | 5 min |
| 10 | Experiment 2b | INT4 PTQ via bitsandbytes NF4 | 5 min |
| 11-14 | Experiment 3 | Layer-by-layer sensitivity + mixed precision | 30-60 min |
| 15-16 | Experiment 5 | Moshiko LM (7B) quantization | 15 min |
| 17-18 | Experiment 6 | Final report + dashboard | 2 min |

---

## Detailed Step-by-Step

### Cell 1: Dependencies

Installs PyTorch 2.4.0 (CUDA 12.1), moshi library, bitsandbytes, and other dependencies.

**After Cell 1 finishes:** Click **Runtime → Restart runtime**, then run Cell 2.

### Cell 2: Create Drive Folder Structure

Creates these folders on your Drive (existing folders are NOT touched):
```
Moshiko2.0_Project/
├── quantization/
│   ├── models/
│   │   ├── int8/
│   │   ├── int4/
│   │   └── mixed/
│   ├── results/
│   └── plots/
└── finetuning/
    ├── dataset/
    ├── checkpoints/
    ├── logs/
    └── outputs/
```

### Cell 3: Initialize Checkpoint Manifest

Creates `quantization/results/checkpoint_manifest.json` — tracks which experiments are complete. If Colab disconnects, re-run this cell to see progress.

### Cell 4: GPU Check

Verifies T4 GPU is available. Disables TF32 (T4 compatibility).

### Cell 5: Load Models + Generate Test Signals

Loads Mimi bf16, Mimi q8, Moshiko LM q8. Generates 4 synthetic test signals.

**To add your own audio:** Edit `custom_audio_files = []` in this cell:
```python
custom_audio_files = ["my_speech_sample.wav"]
```
Place WAV files at: `outputs/audio_input/` on Drive.

---

### EXPERIMENT 1: Compare Existing Checkpoints (Cells 6-7)

**What:** Measure bf16 vs q8 quality using Kyutai's official models.

**Cell 6:** For each test signal, encode+decode with bf16 and q8, measure SNR, time, VRAM.

**Cell 7:** Generates 3 charts — SNR comparison, latency, quality loss.

---

### EXPERIMENT 2: Manual Weight Quantization (Cells 8-10)

**What:** Apply your own INT8 and INT4 quantization.

**Cell 8: Manual INT8**
- Rounds each weight tensor to 8-bit precision, then dequantizes back to float
- This simulates INT8 while keeping the model runnable on GPU
- **Why not `torch.quantization.quantize_dynamic`?** — It's incompatible with moshi's custom transformer architecture (uses `weights_per_step` scheduling that quantized ops don't support)
- Saves model to: `quantization/models/int8/mimi_manual_int8.pt`

**Cell 9: Compare Manual INT8 vs Kyutai Q8 vs BF16**
- Three-way comparison on all test signals

**Cell 10: INT4 PTQ**
- Uses bitsandbytes `quantize_4bit()` with NF4
- Storage-only — model cannot run inference with these weights
- Saves to: `quantization/models/int4/mimi_int4.safetensors`

---

### EXPERIMENT 3: Layer-wise Sensitivity Analysis (Cells 11-14)

**What:** Quantize ONE layer at a time, measure quality drop. This is the most research-worthy experiment.

**Cell 11: Identify Mimi Layers**
- Scans model structure, categorizes into encoder/decoder/bottleneck/embedding/other

**Cell 12: Quantize One Layer at a Time**
- For each layer: copy model, quantize that one layer to INT8 (manual rounding), measure SNR drop
- Takes 30-60 minutes
- Results sorted by sensitivity (most sensitive first)

**Cell 13: Plot Layer Sensitivity**
- Two charts: top 20 most sensitive layers + sensitivity by category

**Cell 14: Mixed Precision Strategy**
- Keep top 20% most sensitive layers at bf16, quantize rest to INT8
- Compare quality vs full bf16 and Kyutai's q8

---

### EXPERIMENT 5: Moshiko LM Quantization (Cells 15-16)

**Cell 15: LM Layer Sensitivity**
- Analyzes weight distributions (std/max ratio) to estimate sensitivity
- No bf16 baseline on T4 — uses q8 weight stats

**Cell 16: LM INT4 with bitsandbytes**
- Attempts to quantize Linear layers with NF4
- May fail with OOM — if so, restart runtime and run only this cell

---

### EXPERIMENT 6: Final Report (Cells 17-18)

**Cell 17:** Collects all metrics, saves summary JSON.

**Cell 18:** Generates 2x2 dashboard plot with all comparisons.

---

## Checkpoint/Resume System

Every experiment cell checks `checkpoint_manifest.json` before running. If already completed, it skips.

**If Colab disconnects:**
1. Re-upload notebook
2. Run Cell 1 (restart after)
3. Run Cell 2 → Cell 3 (shows what's done)
4. Skip to where you left off

**To reset:** Delete `quantization/results/checkpoint_manifest.json` and re-run Cell 3.

---

## Output Files

### Results (JSON)
| File | Experiment |
|------|------------|
| `exp1_mimi_bf16_vs_q8.json` | BF16 vs Q8 comparison |
| `exp2_ptq_int8.json` | Manual INT8 metadata |
| `exp2_ptq_int8_comparison.json` | Three-way SNR comparison |
| `exp2_ptq_int4_metadata.json` | INT4 quantization metadata |
| `exp3_layerwise_sensitivity.json` | Per-layer SNR drop |
| `exp3_mixed_precision.json` | Mixed precision results |
| `exp5_lm_layer_sensitivity.json` | LM weight distribution analysis |
| `exp5_lm_int4.json` | LM INT4 quantization results |
| `exp6_final_report.json` | Summary of all experiments |

### Models
| File | Type |
|------|------|
| `int8/mimi_manual_int8.pt` | Manual INT8 (runnable on GPU) |
| `int4/mimi_int4.safetensors` | INT4 (storage-only) |
| `mixed/mimi_mixed_precision.pt` | Mixed precision (20% bf16 + 80% int8) |

### Plots
| File | Content |
|------|---------|
| `exp1_quality_comparison.png` | BF16 vs Q8: SNR, latency, quality loss |
| `exp3_layerwise_sensitivity.png` | Top 20 sensitive layers + category summary |
| `final_dashboard.png` | 2x2 grid of all comparisons |

---

## Troubleshooting

### CUDA Out of Memory
1. Restart runtime
2. Run only the cells you need
3. `torch.cuda.empty_cache()` before heavy operations

### ModuleNotFoundError: No module named 'moshi'
Run Cell 1 again (installs moshi). Then restart runtime.

### RuntimeError: quantized::linear_dynamic not available on CUDA
This is expected — the notebook no longer uses `quantize_dynamic`. It uses manual weight quantization instead. If you see this error, you're using an old version of the notebook.

### Cell 16 (LM INT4) fails
bitsandbytes Linear4bit may not work with moshi's architecture. If it errors, the experiment is skipped. Other experiments are unaffected.

### SNR is very low (< 10dB)
Check that the model is in `.eval()` mode and you're using `torch.no_grad()`.

---

## Quick Reference

| I want to... | Run Cell(s) |
|--------------|-------------|
| Set up everything | 1 (restart), then 2, 3, 4, 5 |
| Compare bf16 vs q8 | 6, 7 |
| Quantize Mimi to INT8 myself | 8, 9 |
| Quantize Mimi to INT4 myself | 10 |
| Find which layers matter most | 11, 12, 13 |
| Try mixed precision | 14 |
| Analyze the 7B LM | 15, 16 |
| Get final report | 17, 18 |
| Check what's already done | 3 |
| Add my own audio | Edit Cell 5, then re-run from Cell 6 |
