# MOSHIKO 2.0 — Test Documentation

Prerequisite: Run `moshiko2.0.ipynb` completely before running `test.ipynb`. All models (`mimi`, `moshi_lm`, `lm_gen`, `DEVICE`, `FOLDERS`, `BASE_DIR`) must be loaded in memory.

---

## Test 1: Mimi Encode/Decode (Synthetic Audio)

### What it tests
Whether the Mimi audio codec can encode audio to tokens and decode them back without errors. Measures reconstruction quality (SNR) and speed.

### What you need (Input)
Nothing. This test generates a synthetic 440Hz sine wave (3 seconds, 24kHz sample rate) internally.

### What happens (Process)
1. Creates a 440Hz tone: `wav = 0.3 * sin(2 * pi * 440 * t)` with shape `[1, 1, 72000]`
2. Saves the input WAV to `outputs/audio_input/test1_input_440hz.wav`
3. Encodes: `mimi.encode(wav)` → produces discrete tokens with shape `[1, 8, T_codes]` (8 codebooks)
4. Decodes: `mimi.decode(codes)` → reconstructs waveform with shape `[1, 1, T']`
5. Saves the output WAV to `outputs/audio_output/test1_output_TIMESTAMP.wav`
6. Computes SNR (Signal-to-Noise Ratio) by comparing input vs output waveforms
7. Plays both audio clips in the notebook

### What you get (Output)
- **Input file:** `outputs/audio_input/test1_input_440hz.wav`
- **Output file:** `outputs/audio_output/test1_output_TIMESTAMP.wav`
- **Results JSON:** `outputs/test1_results_TIMESTAMP.json` containing:
  - `encode_ms`: Time to encode (milliseconds)
  - `decode_ms`: Time to decode (milliseconds)
  - `snr_db`: Signal-to-Noise Ratio in decibels
  - `input_shape`, `output_shape`: Tensor shapes
- **Audio playback:** Input and output audio players in the notebook

### How to interpret results
- **SNR > 20 dB:** Good reconstruction. The codec preserved the signal well.
- **SNR 10-20 dB:** Acceptable. Some quality loss from quantization.
- **SNR < 10 dB:** Poor reconstruction. Something may be wrong with the codec.
- **Encode/Decode time:** Should be well under the audio duration (RTF < 1.0) for real-time use.
- **Why you get this:** Mimi is a neural audio codec that compresses audio into discrete tokens. The SNR measures how much information is lost during this compression-decompression cycle. A pure 440Hz tone is easy to reconstruct, so expect high SNR.

### Where files are saved
`MyDrive/Moshiko_Project/outputs/` and `MyDrive/Moshiko_Project/outputs/audio_input/` and `MyDrive/Moshiko_Project/outputs/audio_output/`

---

## Test 2: Mimi Encode/Decode (Real Audio File)

### What it tests
Same as Test 1 but with a real audio file instead of synthetic audio. Tests the codec with real-world audio characteristics.

### What you need (Input)
- A WAV file: 24kHz sample rate, mono channel
- Place it in `outputs/audio_input/` on your Google Drive, or upload it to Colab
- Edit the `audio_file` variable in the cell to point to your file path
- If your file has a different sample rate, the cell will auto-resample

### What happens (Process)
1. Loads the WAV file using `soundfile.read()`
2. Resamples to 24kHz if needed (using `torchaudio.transforms.Resample`)
3. Encodes through Mimi → tokens
4. Decodes tokens → reconstructed audio
5. Saves output WAV to `outputs/audio_output/test2_output_TIMESTAMP.wav`
6. Computes SNR between original and reconstructed audio
7. Plays both audio clips in the notebook

### What you get (Output)
- **Output file:** `outputs/audio_output/test2_output_TIMESTAMP.wav`
- **Results JSON:** `outputs/test2_results_TIMESTAMP.json` containing:
  - `input_file`, `output_file`: File paths
  - `duration_s`: Audio duration in seconds
  - `encode_ms`, `decode_ms`: Processing times
  - `snr_db`: Reconstruction quality
- **Audio playback:** Input and output audio players

### How to interpret results
- **Speech audio:** Expect lower SNR than pure tones (speech has complex harmonics). SNR > 15 dB is good.
- **Music audio:** Expect even lower SNR. Music has wider frequency range.
- **Compare with Test 1:** If Test 1 has high SNR but Test 2 has low SNR, the codec is working but struggles with complex audio — this is expected behavior for neural codecs.
- **Why you get this:** Real audio contains many frequencies, transients, and noise. The Mimi codec compresses this into 8 codebooks of discrete tokens, which loses some detail. The SNR quantifies this loss.

### Where files are saved
`MyDrive/Moshiko_Project/outputs/audio_output/`

---

## Test 3: Speech-to-Speech (Audio In → LM Response → Audio Out)

### What it tests
The full pipeline: input speech → Mimi encode → Moshiko LM generates response tokens → Mimi decode → output speech. Tests whether the language model can process audio input and generate audio output.

### What you need (Input)
- A speech audio file (WAV, 24kHz, mono) placed in `outputs/audio_input/`
- The default uses `test_input_440hz.wav` but **a pure tone will not produce meaningful output** — use real speech for meaningful results
- Edit `input_path` in the cell to point to your speech file

### What happens (Process)
1. Loads input speech audio
2. Encodes through Mimi: `mimi.encode(wav)` → audio tokens `[B, 8, T]`
3. Resets `lm_gen` streaming state
4. Feeds input audio tokens into `lm_gen.step()` frame by frame
5. After input is consumed, calls `lm_gen.step(None)` repeatedly to generate response tokens
6. Collects up to `MAX_RESPONSE_FRAMES` (200 frames ≈ 16 seconds) of response tokens
7. Decodes response tokens through Mimi: `mimi.decode(response_tokens)` → audio waveform
8. Saves output WAV and plays both input and response

### What you get (Output)
- **Output file:** `outputs/audio_output/test3_response_TIMESTAMP.wav`
- **Results JSON:** `outputs/test3_results_TIMESTAMP.json` containing:
  - `input_frames`: Number of input audio frames
  - `response_frames`: Number of generated response frames
  - `response_duration_s`: Length of generated speech
  - `gen_time_s`: Total generation time
  - `ms_per_frame`: Generation speed per frame
- **Audio playback:** Input speech and LM response audio players

### How to interpret results
- **With real speech input:** The model should generate a spoken response. Quality depends on the model's training.
- **With synthetic tone input:** The model may generate silence, noise, or nothing. This is expected — the model was trained on speech, not tones.
- **ms_per_frame:** Lower is faster. Moshi's target is ~80ms per frame (80ms latency per 80ms of audio).
- **Why you get this:** The Moshiko LM is a dialogue model. It takes audio tokens as input and generates audio tokens as output, like a conversation. The quality of the response depends on the input being recognizable speech.
- **If no response is generated:** The LM may need proper text conditioning or the input may not be recognized as speech. Try with clear, English speech audio.

### Where files are saved
`MyDrive/Moshiko_Project/outputs/audio_output/`

---

## Test 4: Latency Benchmark

### What it tests
How fast the Mimi codec processes audio at different durations. Measures whether the codec runs faster than real-time (RTF < 1.0).

### What you need (Input)
Nothing. Generates synthetic 440Hz tones at 4 durations internally: 1s, 3s, 5s, 10s.

### What happens (Process)
1. For each duration (1s, 3s, 5s, 10s):
   - Generates a 440Hz tone
   - Runs 3 warmup iterations (GPU kernel compilation)
   - Runs 5 benchmark iterations, measuring encode and decode times separately
   - Uses `torch.cuda.synchronize()` for accurate GPU timing
   - Computes mean encode time, decode time, total time, and RTF
2. Prints a results table
3. Generates two bar charts:
   - Encode vs Decode latency per duration
   - Real-Time Factor (RTF) per duration (green = faster than real-time, red = slower)
4. Saves chart PNG and JSON data

### What you get (Output)
- **Chart:** `outputs/benchmarks/latency_TIMESTAMP.png`
- **Data:** `outputs/benchmarks/latency_TIMESTAMP.json` containing per-duration:
  - `encode_ms`: Mean encode time
  - `decode_ms`: Mean decode time
  - `total_ms`: Combined time
  - `rtf`: Real-Time Factor (processing time / audio duration)
  - `gpu_mem_mb`: GPU memory usage
- **Console table:** Human-readable benchmark table

### How to interpret results
- **RTF < 1.0:** The codec processes audio faster than the audio's duration. Good for real-time applications.
- **RTF > 1.0:** The codec is slower than real-time. Not suitable for live applications.
- **Encode vs Decode:** Encode is typically faster than decode because encoding is a single forward pass while decoding reconstructs a longer waveform.
- **Scaling with duration:** Latency should scale linearly with audio duration. If it doesn't, there may be a bottleneck.
- **Why you get this:** GPU operations have fixed overhead (kernel launch, memory transfer) plus per-sample computation. Shorter clips have higher overhead ratio. The benchmark shows this relationship.
- **For your report:** This data demonstrates the codec's suitability for real-time applications. Compare against Moshi's stated 200ms latency target.

### Where files are saved
`MyDrive/Moshiko_Project/outputs/benchmarks/`

---

## Test 5: BF16 vs Q8 Quality Comparison

### What it tests
Compares audio reconstruction quality between the full-precision (BF16) Mimi codec and the INT8 quantized (Q8) version. Quantifies the quality loss from quantization.

### What you need (Input)
Nothing. Generates 4 synthetic test signals internally:
- Pure tone 440Hz (2 seconds)
- Mixed tones (220Hz + 440Hz + 880Hz, 2 seconds)
- White noise (2 seconds)
- Chirp sweep (200Hz → 2000Hz, 2 seconds)

Also requires the BF16 Mimi file to be downloaded (done by `moshiko2.0.ipynb`).

### What happens (Process)
1. Loads the BF16 Mimi codec alongside the already-loaded Q8 Mimi
2. For each of the 4 test signals:
   - Runs the signal through Q8 Mimi: encode → decode
   - Runs the same signal through BF16 Mimi: encode → decode
   - Computes SNR for both outputs against the original
   - Measures total processing time for both
3. Prints a comparison table showing SNR for each codec and the difference
4. Generates two side-by-side bar charts:
   - SNR comparison (Q8 vs BF16) per signal type
   - Speed comparison (Q8 vs BF16) per signal type
5. Saves charts and JSON data
6. Unloads BF16 Mimi to free VRAM

### What you get (Output)
- **Chart:** `outputs/comparisons/bf16_vs_q8_TIMESTAMP.png`
- **Data:** `outputs/comparisons/bf16_vs_q8_TIMESTAMP.json` containing per-signal:
  - `snr_q8_db`: Q8 reconstruction SNR
  - `snr_bf16_db`: BF16 reconstruction SNR
  - `snr_diff_db`: Difference (BF16 - Q8)
  - `time_q8_ms`, `time_bf16_ms`: Processing times
- **Console table:** Side-by-side comparison

### How to interpret results
- **snr_diff_db close to 0:** Quantization caused negligible quality loss. Q8 is as good as BF16.
- **snr_diff_db > 0:** BF16 has higher SNR (expected). The magnitude tells you how much quality was lost.
- **snr_diff_db < 0:** Q8 somehow has higher SNR (rare, may happen with noise signals due to quantization noise masking).
- **Q8 faster?:** Q8 should be faster or equal to BF16 since INT8 operations are cheaper.
- **Per-signal patterns:**
  - Pure tones: Both codecs should perform well (simple signal).
  - Mixed tones: Slightly more complex, small SNR difference expected.
  - White noise: Lowest SNR for both (random signal is hard to compress).
  - Chirp sweep: Moderate SNR, tests frequency response.
- **Why you get this:** INT8 quantization reduces weight precision from 16-bit to 8-bit, cutting memory usage in half. This introduces small numerical errors that accumulate during encode/decode. The comparison quantifies whether this tradeoff is acceptable for your use case.
- **For your report:** This is direct evidence for your quantization analysis section. Include the SNR difference numbers and charts.

### Where files are saved
`MyDrive/Moshiko_Project/outputs/comparisons/`

---

## Test 6: Session Report Generator

### What it does
Collects all results from the current session and generates a single human-readable summary report.

### What happens (Process)
1. Counts files in each output folder (audio inputs, outputs, benchmarks, comparisons)
2. Reads the latest benchmark JSON and comparison JSON
3. Reads environment info (GPU, VRAM, model, Python version)
4. Formats everything into a structured text report with sections:
   - Environment (GPU, VRAM, model, Python)
   - File counts (inputs, outputs, benchmarks, comparisons)
   - Latest benchmark results (duration, encode, decode, RTF table)
   - BF16 vs Q8 comparison (signal, Q8 SNR, BF16 SNR, difference table)
   - Resume instructions for next session
5. Saves the report to `outputs/session_report_TIMESTAMP.txt`
6. Appends a completion entry to `logs/session_log.txt`

### What you get (Output)
- **Report file:** `outputs/session_report_TIMESTAMP.txt` — a single text file containing all session results
- **Console output:** The full report is also printed to the notebook

### How to interpret results
- This is a summary, not a test. It aggregates data from Tests 1-5.
- Use this file as an appendix for your university report or to share with group members.
- The report shows whether your tests ran successfully and what the key metrics were.
- **Why you get this:** After running multiple tests, results are scattered across many JSON files. This report consolidates everything into one readable document for easy review and sharing.

### Where files are saved
`MyDrive/Moshiko_Project/outputs/session_report_TIMESTAMP.txt`

---

## Running Order

Run `test.ipynb` cells in order after `moshiko2.0.ipynb` is complete:

1. **Test 1** — Always works (synthetic audio, no dependencies)
2. **Test 2** — Upload a real audio file first
3. **Test 3** — Upload speech audio for meaningful LM response
4. **Test 4** — Always works (synthetic benchmark)
5. **Test 5** — Always works (requires BF16 Mimi downloaded)
6. **Test 6** — Run last to generate the session report

## Troubleshooting

| Problem | Cause | Fix |
|---------|-------|-----|
| CUDA out of memory | Too many models in VRAM | Restart runtime, re-run moshiko2.0.ipynb |
| File not found | Models not downloaded | Re-run download cell in moshiko2.0.ipynb |
| Test 3 no response | Input is not speech | Use real speech audio, not tones |
| BF16 comparison fails | BF16 file not downloaded | Re-run download in moshiko2.0.ipynb |
| Drive not mounted | Session disconnected | Re-run mount cell |

## What is NOT included

**Text-to-Speech (TTS)** is not in this notebook. It requires:
- `TTSModel` from `moshi.models.tts`
- Voice conditioning files (speaker embeddings or audio prefixes)
- Different model loading than the dialogue model
- The `simple_generate()` API with voice names

If you need TTS, it should be a separate notebook using `TTSModel.from_checkpoint_info()` with voice files from the HuggingFace repo.
