# MOSHIKO 2.0 — Project Knowledge Base

## 1. Current Project State

### What Works
| Component | Status | Notebook |
|-----------|--------|----------|
| PyTorch 2.4.0 + CUDA 12.1 install | Working | moshiko2.0.ipynb Cell 1 |
| Dependencies (moshi, bitsandbytes, etc.) | Working | moshiko2.0.ipynb Cell 2 |
| Google Drive mount + folder structure | Working | moshiko2.0.ipynb Cell 3 |
| GPU check + auto-select model | Working | moshiko2.0.ipynb Cell 4 |
| Model download (checkpoint resume-safe) | Working | moshiko2.0.ipynb Cell 5 |
| Mimi codec load (q8) | Working | moshiko2.0.ipynb Cell 6 |
| Moshiko LM load (q8, full config) | Working | moshiko2.0.ipynb Cell 6 |
| Test 1: Mimi encode/decode (synthetic) | Working | test.ipynb Cell 2 |
| Test 2: Mimi encode/decode (real audio) | Working | test.ipynb Cell 3 |
| Test 4: Latency benchmark | Working | test.ipynb Cell 4 |
| Test 5: BF16 vs Q8 comparison | Working | test.ipynb Cell 5 |
| Test 6: Session report | Working | test.ipynb Cell 6 |

### What Is Built (Ready to Run)
| Component | Status | Notebook | Guide |
|-----------|--------|----------|-------|
| Quantization analysis (6 experiments) | Built, not yet run | quantization_analysis.ipynb | quantizationguide.md |
| New Drive folder structure (quantization/, finetuning/) | Defined in notebook | quantization_analysis.ipynb Cell 2 | quantizationguide.md |
| Checkpoint/resume system for experiments | Built into notebook | quantization_analysis.ipynb Cell 3 | quantizationguide.md |

### What Needs Fix
| Component | Issue | Fix Location |
|-----------|-------|-------------|
| Test 3: Speech-to-speech | bfloat16 crash on T4 | test3_OptionB.ipynb or test3_OptionC.ipynb |

### Model Files Downloaded (on Google Drive)
- `models/q8/model.q8.safetensors` — Mimi Codec q8 + Moshiko LM q8 (combined file, ~8.4 GB)
- `models/q8/tokenizer-e351c8d8-checkpoint125.safetensors` — Text tokenizer
- `models/bf16/tokenizer-e351c8d8-checkpoint125.safetensors` — Mimi Codec bf16 (for comparison)

### Key Objects in Memory (after moshiko2.0.ipynb)
- `mimi` — MimiModel (audio codec, encode/decode)
- `moshi_lm` — LMModel (7B speech-text language model)
- `lm_gen` — LMGen (streaming generation wrapper)
- `DEVICE` — "cuda" or "cpu"
- `FOLDERS` — dict of all project folder paths
- `BASE_DIR` — "/content/drive/MyDrive/Moshiko_Project"

### Important: Running quantization_analysis.ipynb
- This notebook is **self-contained** — do NOT run moshiko2.0.ipynb in the same session
- If you ran moshiko2.0.ipynb previously: **Runtime → Restart runtime** first, then open quantization_analysis.ipynb
- The notebook loads all models itself (Cell 5) — models are already on Drive from previous sessions
- No code copying needed between notebooks
- Only manual input (optional): edit `custom_audio_files = []` in Cell 5 to add your own WAV files

---

## 2. Test 3: Speech-to-Speech (Two Options)

### Option B: LMGen with Float32 Fix
**File:** `test3_OptionB.ipynb`

**How it works:**
1. Reloads Mimi and Moshiko LM with `.float()` to force float32 dtype
2. Disables autocast and TF32 (T4 doesn't support bfloat16)
3. Creates fresh LMGen for each generation
4. Feeds input audio tokens frame by frame via `lm_gen.step(frame)`
5. Generates response tokens via `lm_gen.step(None)`
6. Decodes response through Mimi

**How to run:**
1. Run `moshiko2.0.ipynb` completely first
2. Run `test3_OptionB.ipynb` Cell 1 (reloads models, ~2-3 min)
3. Run `test3_OptionB.ipynb` Cell 2 (generation)

**Pros:** Direct control over generation loop, easy to modify
**Cons:** Uses more VRAM (float32), may be tight on T4 (14GB+), manual frame-by-frame feeding

**What you need:**
- Speech WAV file (24kHz, mono) in `outputs/audio_input/`
- Edit `input_path` in Cell 2 to point to your file

**Output:**
- Response WAV file in `outputs/audio_output/`
- Results JSON with generation metrics

---

### Option C: InferenceState Official API (Recommended)
**File:** `test3_OptionC.ipynb`

**How it works:**
1. Uses `moshi.run_inference.InferenceState` — the officially supported pipeline
2. Loads models via `CheckpointInfo` with `dtype=torch.float16` (T4 compatible)
3. Calls `inference_state.run(in_pcm)` — handles framing, encoding, LM generation, decoding internally
4. Returns `(text_tokens, audio_tokens)` tuples per batch item

**How to run:**
1. Run `moshiko2.0.ipynb` completely first
2. Run `test3_OptionC.ipynb` Cell 1 (sets up InferenceState, ~2-3 min)
3. Run `test3_OptionC.ipynb` Cell 2 (inference)

**Pros:** Official API, handles edge cases, returns text transcriptions too, uses float16 (less VRAM)
**Cons:** Less control over generation loop, requires `sphn` library for audio loading

**What you need:**
- Speech WAV file (24kHz, mono) in `outputs/audio_input/`
- Edit `input_path` in Cell 2 to point to your file

**Output:**
- Response WAV file in `outputs/audio_output/`
- Decoded text transcription
- Results JSON with generation metrics

---

## 3. Fine-Tuning: Speech-to-Speech Dialogue + Language Identification

### 3.1 Speech-to-Speech Dialogue Fine-Tuning

**Goal:** Improve the model's ability to generate relevant spoken responses to input speech.

**What you need:**
- Paired audio dataset: (input speech, expected response speech)
- Examples: conversational speech datasets, dialogue corpora with audio
- Format: WAV files, 24kHz, mono
- Minimum: 100+ hours for meaningful fine-tuning
- Storage: ~50 GB+ for dataset + checkpoints

**Approach:**
1. **Prepare dataset:**
   - Convert all audio to 24kHz mono WAV
   - Create a manifest file: `{"input": "path/to/input.wav", "output": "path/to/output.wav"}`
   - Split into train/validation sets (90/10)

2. **Encode audio to tokens:**
   ```python
   with torch.no_grad():
       input_tokens = mimi.encode(input_wav)    # [B, 8, T]
       output_tokens = mimi.encode(output_wav)  # [B, 8, T]
   ```

3. **Fine-tune the LM:**
   - Freeze Mimi (don't train the codec)
   - Train `moshi_lm` with input tokens as conditioning, output tokens as targets
   - Use the same `lm_kwargs` config as loading
   - Loss: cross-entropy on output audio tokens
   - Optimizer: AdamW with learning rate ~1e-5
   - Batch size: as large as VRAM allows (1-2 on T4)

4. **Save checkpoints:**
   - Save `moshi_lm.state_dict()` after each epoch
   - Load with same `loaders.get_moshi_lm()` + `load_state_dict()`

**Key files to modify:**
- Training loop: new notebook `finetune_dialogue.ipynb`
- Data loader: custom PyTorch Dataset that encodes audio to tokens
- Config: same `lm_kwargs` dict from moshiko2.0.ipynb Cell 6

**Estimated time on T4:**
- 100 hours of audio: ~3-5 days for 1 epoch
- Recommended: use A100 or multi-GPU if available

---

### 3.2 Language Identification Fine-Tuning

**Goal:** Add a classification head on top of Moshiko to detect the language of input speech.

**What you need:**
- Labeled speech dataset: (audio file, language label)
- Examples: CommonVoice, VoxLingua107, FLEURS
- Format: WAV files, any sample rate (resample to 24kHz)
- Minimum: 1000+ samples per language, 5+ languages

**Approach:**
1. **Encode audio to tokens:**
   ```python
   with torch.no_grad():
       tokens = mimi.encode(wav)  # [B, 8, T]
   ```

2. **Extract features from LM:**
   - Feed tokens through `moshi_lm` (don't generate, just get hidden states)
   - Take the last hidden state or pool across time
   - Shape: `[B, T, dim]` where dim=4096

3. **Add classification head:**
   ```python
   class LanguageClassifier(torch.nn.Module):
       def __init__(self, lm_model, num_languages):
           super().__init__()
           self.lm = lm_model
           self.head = torch.nn.Sequential(
               torch.nn.Linear(4096, 512),
               torch.nn.ReLU(),
               torch.nn.Dropout(0.3),
               torch.nn.Linear(512, num_languages),
           )
       
       def forward(self, audio_tokens):
           # Get LM hidden states
           hidden = self.lm(audio_tokens)  # depends on LM API
           # Pool and classify
           pooled = hidden.mean(dim=1)  # mean pool over time
           return self.head(pooled)
   ```

4. **Train:**
   - Freeze `moshi_lm` weights (only train the classification head)
   - Loss: cross-entropy
   - Optimizer: AdamW, lr=1e-4
   - Epochs: 5-10

5. **Evaluate:**
   - Accuracy per language
   - Confusion matrix
   - Inference latency

**Key files to create:**
- `finetune_language_id.ipynb` — training notebook
- `language_id_inference.ipynb` — inference notebook for testing

**Estimated time on T4:**
- 10 languages, 10k samples: ~2-4 hours (head only, LM frozen)
- Much faster than dialogue fine-tuning since only the head is trained

---

## 4. Quantization Analysis (Built — Ready to Run)

### 4.1 What's Implemented in quantization_analysis.ipynb

| Experiment | Cell(s) | Type | What It Does | Output File |
|------------|---------|------|--------------|-------------|
| 1: BF16 vs Q8 comparison | 6-7 | Compare existing | Measure Kyutai's official bf16 vs q8 Mimi codec | `quantization/results/exp1_mimi_bf16_vs_q8.json` |
| 2a: Your INT8 PTQ | 8-9 | Post-Training Quantization | Apply `torch.quantization.quantize_dynamic()` to Mimi bf16, compare with Kyutai's q8 | `quantization/results/exp2_ptq_int8_comparison.json` |
| 2b: Your INT4 PTQ | 10 | Post-Training Quantization | Apply bitsandbytes NF4 quantization to Mimi bf16 | `quantization/models/int4/mimi_int4.safetensors` |
| 3a: Layer-wise sensitivity | 11-13 | Sensitivity analysis | Quantize ONE layer at a time, measure SNR drop per layer | `quantization/results/exp3_layerwise_sensitivity.json` |
| 3b: Mixed precision | 14 | Strategy | Keep top 20% sensitive layers at bf16, quantize rest to INT8 | `quantization/results/exp3_mixed_precision.json` |
| 4: Dynamic quantization | 15-16 | Runtime quantization | Weights stay float, activations quantized at runtime | `quantization/results/exp4_dynamic_comparison.json` |
| 5a: LM layer sensitivity | 17 | Sensitivity analysis | Analyze Moshiko 7B LM weight distributions (no bf16 baseline on T4) | `quantization/results/exp5_lm_layer_sensitivity.json` |
| 5b: LM INT4 with bitsandbytes | 18 | Post-Training Quantization | Replace LM Linear layers with `Linear4bit` (NF4) | `quantization/results/exp5_lm_int4.json` |
| 6: Final report | 19-20 | Summary | Collect all metrics, generate dashboard plot | `quantization/results/exp6_final_report.json` |

### 4.2 Checkpoint/Resume System

- Manifest file: `quantization/results/checkpoint_manifest.json`
- Each experiment cell checks manifest before running — auto-skips if already done
- If Colab disconnects: re-run Cells 1-3, then skip to where you left off
- All results saved to Drive immediately — no data loss on disconnect
- To reset: delete `checkpoint_manifest.json` and re-run Cell 3

### 4.3 How to Add Custom Audio

Edit Cell 5 in `quantization_analysis.ipynb`:
```python
custom_audio_files = ["my_speech.wav"]
```
Place WAV files at: `Moshiko_Project/outputs/audio_input/` on Drive.
Requirements: WAV format, 24kHz, mono, max 10 seconds.

### 4.4 Known Limitations

- **LM bf16 cannot load on T4** (15.4 GB > 16 GB VRAM) — LM sensitivity is estimated from weight distributions, not from bf16 comparison
- **INT4 Mimi model cannot run inference directly** — it stores quantized weights + metadata; needs dequantization step before use
- **bitsandbytes Linear4bit** (Cell 18) may fail if library version < 0.41.0 or CUDA incompatibility — skip if it errors, other experiments unaffected

### 4.5 Future Quantization Work (Not Yet Implemented)

- **GPTQ** (Generative Pre-Trained Quantization) — requires `auto-gptq` library, calibration on audio tokens
- **AWQ** (Activation-aware Weight Quantization) — requires `llm-awq` library
- **INT2 quantization** — extreme compression, likely significant quality loss
- **Fine-tuning quantized models** — train the INT4/mixed models to recover quality loss

---

## 5. Troubleshooting

### Common Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `RuntimeError: Expected weight_scb to have type float, but got bfloat16` | T4 GPU doesn't support bfloat16 | Use `.float()` on model, or `dtype=torch.float16` |
| `AttributeError: 'LMGen' object has no attribute 'reset'` | LMGen doesn't have reset() | Create new LMGen instance instead |
| `CUDA out of memory` | T4 has 16GB, model uses ~14GB | Close other processes, reduce batch size, restart runtime |
| `FileNotFoundError: Missing model file` | Download incomplete or Drive not mounted | Re-run download cell, check Drive mount |
| `ImportError: No module named 'sphn'` | sphn not installed | `pip install sphn` (needed for InferenceState) |
| `Model loading hangs` | HuggingFace download slow | Check internet, use local files |

### VRAM Management
- Mimi codec: ~200 MB
- Moshiko LM (q8): ~8.4 GB
- Moshiko LM (bf16): ~15.4 GB (won't fit on T4)
- Activation memory during inference: ~2-4 GB
- Total for q8: ~12-14 GB (fits on T4 with 16 GB)
- Total for bf16: ~18+ GB (won't fit on T4)

### Tips
- Always run `torch.cuda.empty_cache()` after deleting models
- Use `torch.no_grad()` for all inference
- Restart runtime if VRAM usage is unexpectedly high
- Save checkpoints frequently during fine-tuning

---

## 6. File Structure

```
MyDrive/Moshiko_Project/
├── models/
│   ├── bf16/
│   │   └── tokenizer-e351c8d8-checkpoint125.safetensors  (Mimi bf16)
│   └── q8/
│       ├── model.q8.safetensors                          (Mimi + LM q8)
│       └── tokenizer-e351c8d8-checkpoint125.safetensors  (text tokenizer)
│
├── quantization/                          ← NEW (created by quantization_analysis.ipynb Cell 2)
│   ├── models/
│   │   ├── int8/
│   │   │   ├── mimi_your_int8.pt
│   │   │   └── mimi_dynamic_int8.pt
│   │   ├── int4/
│   │   │   ├── mimi_int4.safetensors
│   │   │   └── mimi_int4_metadata.json
│   │   └── mixed/
│   │       └── mimi_mixed_precision.pt
│   ├── results/
│   │   ├── checkpoint_manifest.json
│   │   ├── exp1_mimi_bf16_vs_q8.json
│   │   ├── exp2_ptq_int8_comparison.json
│   │   ├── exp2_ptq_int4_metadata.json
│   │   ├── exp3_layerwise_sensitivity.json
│   │   ├── exp3_mixed_precision.json
│   │   ├── exp4_dynamic_comparison.json
│   │   ├── exp5_lm_layer_sensitivity.json
│   │   ├── exp5_lm_int4.json
│   │   └── exp6_final_report.json
│   └── plots/
│       ├── exp1_quality_comparison.png
│       ├── exp3_layerwise_sensitivity.png
│       └── final_dashboard.png
│
├── finetuning/                            ← NEW (created by quantization_analysis.ipynb Cell 2, empty for future use)
│   ├── dataset/
│   ├── checkpoints/
│   ├── logs/
│   └── outputs/
│
├── checkpoints/                           ← Existing
│   └── download_checkpoint.json
├── outputs/                               ← Existing
│   ├── audio_input/
│   ├── audio_output/
│   ├── benchmarks/
│   └── comparisons/
├── logs/                                  ← Existing
│   ├── session_log.txt
│   └── environment_info.json
└── README.txt
```

### Local Files (E:\workspace\moshika\)

| File | Purpose |
|------|---------|
| `moshiko2.0.ipynb` | Main setup notebook (install, download, load models) |
| `quantization_analysis.ipynb` | Quantization experiments (6 experiments, self-contained) |
| `quantizationguide.md` | Step-by-step guide for running quantization analysis |
| `knowledge.md` | This file — project knowledge base |
| `happened.md` | Historical analysis of moshiko.ipynb iterations |
| `junk/` | Old iterations, test notebooks, test audio files |

---

## 7. Next Steps Priority

1. **Run Test 3** (speech-to-speech) — try Option C first, fall back to Option B
2. **Quantization analysis** — layer sensitivity (Section 4.3) is the most actionable
3. **Language ID fine-tuning** (Section 3.2) — faster to implement than dialogue
4. **Dialogue fine-tuning** (Section 3.1) — requires large dataset, plan for multi-GPU
5. **INT4/INT2 quantization** (Section 4.1) — depends on library support
