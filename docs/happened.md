# Notebook Analysis: moshiko.ipynb — What Happened in Each Cell

This is a Google Colab notebook for setting up and running the **Moshiko** speech-text model (by Kyutai) on a T4 GPU. The notebook went through many iterations of trial-and-error, especially around model loading. Below is a cell-by-cell breakdown.

---

## Cell 1 — Markdown: "Mount Google Drive & create folder structure"
**Purpose:** Section header for the first logical block.

---

## Cell 2 — Code: Mount Google Drive
**What it does:** Mounts Google Drive at `/content/drive` so model files and outputs can be persisted across Colab session disconnects.

---

## Cell 3 — Code: Create Folder Structure + README
**What it does:**
- Defines a full folder hierarchy under `/content/drive/MyDrive/Moshiko_Project/` (models, checkpoints, outputs, logs, notebooks, etc.)
- Creates all folders with `os.makedirs`
- Writes a session log entry to `logs/session_log.txt`
- Creates a `README.txt` with folder structure and resume instructions
- Stores `FOLDERS` and `BASE_DIR` in `builtins` so all other cells can access them globally

---

## Cell 4 — Markdown: "CELL 2 — Check GPU & Environment"
**Purpose:** Section header.

---

## Cell 5 — Code: Check GPU & Environment
**What it does:**
- Checks if PyTorch/CUDA is available, prints GPU name and VRAM
- Checks Python version (requires 3.10+)
- Auto-selects model precision based on VRAM:
  - >= 35GB → bf16 (A100)
  - >= 14GB → q8 (T4)
  - < 14GB → q8 (warning)
- Saves `MODEL_PRECISION` and `MODEL_REPO` to `builtins`
- Writes environment info JSON to Drive logs

---

## Cell 6 — Markdown: "Install All Dependencies"
**Purpose:** Section header.

---

## Cell 7 — Code: Install Dependencies
**What it does:**
- Installs packages: `moshi`, `huggingface_hub`, `torchaudio`, `soundfile`, `numpy`, `scipy`, `matplotlib`, `IPython`, `tqdm`
- Skips already-installed packages
- Verifies critical imports and prints versions

---

## Cell 8 — Markdown: "Download Models"
**Purpose:** Section header.

---

## Cell 9 — Code: Download Models (FIRST ATTEMPT — WRONG FILENAMES)
**What it does:**
- Downloads model files from HuggingFace Hub with a checkpoint system (resume-safe)
- **Used incorrect filenames:** `mimi_weight.pt` and `moshiko-q8.safetensors`
- These filenames did not match the actual repo contents, so this cell would have failed

---

## Cell 10 — Code: Recreate Folders (Ad-hoc fix)
**What it does:**
- Redefines `BASE_DIR` and a subset of `FOLDERS`
- Creates `models_q8`, `models_bf16`, `checkpoints` directories
- Likely run after a session reset when `builtins.FOLDERS` was lost

---

## Cell 11 — Code: Download Models (FIXED FILENAMES)
**What it does:**
- Same checkpoint-based download system as Cell 9
- **Corrected filenames:**
  - `model.q8.safetensors` (Mimi Codec q8)
  - `tokenizer-e351c8d8-checkpoint125.safetensors` (Moshiko LM q8)
  - `tokenizer-e351c8d8-checkpoint125.safetensors` (Mimi Codec bf16)
- BF16 Moshiko LM (~15.4 GB) left commented out

---

## Cell 12 — Markdown: "Load Moshiko Model Into Memory"
**Purpose:** Section header.

---

## Cell 13 — Code: Inspect moshi Package
**What it does:**
- Prints `moshi.__version__`
- Lists available functions in `moshi.models.loaders` to discover the correct API
- Debug cell to figure out which loader function to use

---

## Cell 14 — Code: Load Models (FIRST ATTEMPT — Auto-Detect)
**What it does:**
- Tries multiple loader function names (`load_moshi`, `get_moshi`, `load_model`)
- Falls back to direct class instantiation if no loader found
- This was a guessing approach — the correct function name wasn't known yet

---

## Cell 15 — Code: Reinstall moshi from GitHub
**What it does:**
- Uninstalls existing `moshi` package
- Installs latest from `git+https://github.com/kyutai-labs/moshi.git`
- Prompts user to restart runtime

---

## Cell 16 — Code: Load Models (USING get_moshi_lm)
**What it does:**
- Uses the correct loader: `loaders.get_moshi_lm()`
- Loads Mimi codec and Moshiko LM (q8)
- Sets models to eval mode
- Stores in `builtins` for global access
- This was a cleaner version after discovering the right API

---

## Cell 17 — Markdown: "Day-4/4/26"
**Purpose:** Date marker indicating a new session on April 4, 2026.

---

## Cell 18 — Code: `!pwd`
**What it does:** Prints current working directory. Debug/navigation cell.

---

## Cell 19 — Code: Mount Google Drive (re-run)
**What it does:** Remounts Drive after session restart.

---

## Cell 20 — Code: `!pwd`
**What it does:** Verify working directory again.

---

## Cell 21 — Code: `!ls`
**What it does:** List files in current directory. Debug cell.

---

## Cell 22 — Code: Empty
**What it does:** Nothing. Placeholder.

---

## Cell 23 — Code: Recreate Folder Structure (Duplicate of Cell 3)
**What it does:** Same as Cell 3 — redefines `FOLDERS`, creates directories, writes session log and README. Run after session restart.

---

## Cell 24 — Code: Empty
**What it does:** Nothing. Placeholder.

---

## Cell 25 — Code: Check GPU & Environment (Duplicate of Cell 5)
**What it does:** Same as Cell 5. Re-run after session restart.

---

## Cell 26 — Code: Empty
**What it does:** Nothing. Placeholder.

---

## Cell 27 — Code: Install Dependencies (Duplicate of Cell 7)
**What it does:** Same as Cell 7. Re-run after session restart.

---

## Cell 28 — Code: Empty
**What it does:** Nothing. Placeholder.

---

## Cell 29 — Code: Download Models (Duplicate of Cell 11)
**What it does:** Same as Cell 11 — downloads models with corrected filenames and checkpoint system.

---

## Cell 30 — Code: Verify CUDA
**What it does:**
- Checks `torch.cuda.is_available()`
- Prints GPU name (expects Tesla T4)
- Prints CUDA version
- Quick sanity check before loading models

---

## Cell 31 — Code: Load Models with LMGen
**What it does:**
- Loads Mimi codec via `loaders.get_mimi()`
- Loads Moshiko LM via `loaders.get_moshi_lm()`
- Wraps model in `LMGen` (streaming generator with temp=0.8, temp_text=0.7)
- Reports VRAM usage
- Stores `mimi`, `moshi_lm`, `lm_gen`, `DEVICE` in `builtins`

---

## Cell 32 — Code: Fix PyTorch/Moshi Version Conflict (Cell 3b)
**What it does:**
- Uninstalls old `moshi`
- Downgrades PyTorch to 2.4.0 with CUDA 12.1 (compatibility fix)
- Reinstalls `moshi` from GitHub
- Reinstalls `safetensors`
- Prompts user to restart runtime

---

## Cell 33 — Markdown: "Restart Runtime"
**Purpose:** Reminder to restart the Colab runtime after the PyTorch downgrade.

---

## Cell 34 — Code: Full Setup After Restart (Drive + Folders + README)
**What it does:** Combined Cell 1 + Cell 3 — mounts Drive, creates folders, writes session log and README. Post-restart re-run.

---

## Cell 35 — Code: Check GPU & Environment (Post-restart)
**What it does:** Same as Cell 5/25. Re-run after restart to verify GPU and select model.

---

## Cell 36 — Code: Install Dependencies (Post-restart, labeled "CELL 3")
**What it does:** Same as Cell 7/27. Re-run after restart. Better formatted output messages.

---

## Cell 37 — Code: Empty
**What it does:** Nothing. Placeholder.

---

## Cell 38 — Code: Download Models (Post-restart)
**What it does:** Same as Cell 11/29. Re-run after restart with checkpoint system.

---

## Cell 39 — Code: PyTorch Version Check + Load Models
**What it does:**
- Asserts PyTorch 2.4 is installed
- Loads Mimi + Moshiko LM + LMGen (same as Cell 31)
- Post-restart model loading verification

---

## Cell 40 — Code: Install bitsandbytes
**What it does:**
- Installs `bitsandbytes>=0.41.0` for INT8 quantized model support
- Verifies import works
- Prompts runtime restart

---

## Cell 41 — Code: Empty
**What it does:** Nothing. Placeholder.

---

## Cell 42 — Code: Load Models with `quantize=True`
**What it does:**
- PyTorch 2.4 assertion
- Loads Moshiko LM with `lm_kwargs={'quantize': True}` to handle INT8 weights correctly
- Wraps in LMGen
- This was an attempt to fix the `_scb` key error from quantized weights

---

## Cell 43 — Code: Load Models via config.json
**What it does:**
- Downloads `config.json` from HuggingFace if not present
- Reads config to get model parameters (delays, n_q, dep_q, quantize)
- Passes full config as `lm_kwargs` to `get_moshi_lm()`
- Approach: let the config file drive the model architecture

---

## Cell 44 — Code: Load Models with Filtered config Keys
**What it does:**
- Instead of passing full config (which has extra keys causing TypeError), passes only the 4 keys the LMModel constructor accepts:
  - `delays`, `n_q`, `dep_q`, `quantize`
- This was a refinement of Cell 43's approach

---

## Cell 45 — Code: Read Full config.json
**What it does:**
- Reads and prints all key-value pairs from `config.json`
- Debug cell to discover ALL parameters needed for model loading

---

## Cell 46 — Code: Load Models with ALL config Parameters (FINAL VERSION)
**What it does:**
- Passes the complete set of ~30 parameters from config.json as `lm_kwargs`:
  - Core architecture: delays, n_q, dep_q, card, text_card, dim, num_heads, num_layers, hidden_scale, etc.
  - Depth transformer: depformer_dim, depformer_num_layers, depformer_multi_linear, etc.
  - Quantization: quantize=True
- This is the final working version that successfully loads the q8 model

---

## Cell 47 — Code: Test 1 — Basic Audio Encode-Decode
**What it does:**
- Creates a synthetic 440Hz sine wave (3 seconds, 24kHz)
- Encodes audio through Mimi codec → tokens
- Decodes tokens back → audio
- Saves input and output WAV files to Drive
- Plays both audio clips in the notebook
- Calculates SNR (Signal-to-Noise Ratio) as quality metric
- Saves results JSON to Drive
- Reports encode/decode times in milliseconds

---

## Summary of the Workflow

| Phase | Cells | What Happened |
|-------|-------|---------------|
| **Setup** | 1-7 | Mount Drive, create folders, check GPU, install packages |
| **Download (broken)** | 9 | Wrong filenames — would fail |
| **Download (fixed)** | 10-11 | Correct filenames, checkpoint system |
| **Load (guessing)** | 13-14 | Inspecting API, trying multiple loader names |
| **Fix moshi** | 15 | Reinstall from GitHub |
| **Load (clean)** | 16 | Using `get_moshi_lm()` |
| **New session** | 17-29 | Date marker, remount Drive, rerun setup cells |
| **CUDA check** | 30 | Verify GPU |
| **Load with LMGen** | 31 | First successful load with streaming generator |
| **Version fix** | 32-33 | Downgrade PyTorch to 2.4, restart |
| **Post-restart setup** | 34-38 | Rerun all setup cells |
| **Load + quantize fix** | 39-42 | Multiple attempts: quantize=True, bitsandbytes |
| **config.json approach** | 43-44 | Pass config params to loader |
| **Final load** | 45-46 | All ~30 config parameters passed — working |
| **First test** | 47 | Synthetic audio encode→decode test with SNR metric |

**Key pattern:** The notebook shows extensive iteration on getting the model to load correctly. The user went through at least 8 different attempts (Cells 14, 16, 31, 39, 42, 43, 44, 46) adjusting PyTorch versions, installing bitsandbytes, discovering the right loader function, and finally passing the complete config.json parameters to `get_moshi_lm()`.
