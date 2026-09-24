# NanoTranslate-S3

A fully offline **Russian -> English** neural translator for **ESP32-S3 N16R8** (16 MB flash + 8 MB octal PSRAM). You type a short Russian sentence on a 240x240 display using a five-way joystick; the board translates it locally and shows the English result. No Wi-Fi, phone, API, or cloud is required.

This repository is deliberately hardware-native. It does **not** try to squeeze a desktop translation checkpoint onto a microcontroller. The tokenizer, student Transformer, quantization format, output head, runtime, flash layout and UI are co-designed for one job: translate one normal sentence, usually 5-20 words.

## What is already verified

The repository includes a real end-to-end smoke test:

```text
PyTorch tiny RU->EN model
        -> row-wise INT8 exporter
        -> .s3mt model with CRC32
        -> portable C++ runtime
        -> "привет" => "hello"
```

The smoke model overfits a 20-pair corpus from loss ~5.77 to ~0.04, the quantized C++ runtime gives the same `hello` output, and a deliberately corrupted model is rejected by CRC. The host C++ runtime builds cleanly with CMake/GCC.

**Not claimed yet:** real ESP32-S3 BLEU/chrF or latency. This environment does not contain ESP-IDF or physical hardware, so board numbers are intentionally left blank until measured. See [`docs/STATUS.md`](docs/STATUS.md).

## Default model

`base` is approximately **999,040 parameters**:

| item | value |
|---|---:|
| architecture | pre-norm encoder-decoder Transformer |
| encoder / decoder | 3 / 2 layers |
| d_model | 128 |
| attention heads | 4 |
| FFN | 256 |
| vocabulary | 1536 FlashPiece tokens |
| max source | 64 tokens |
| max target | 48 tokens |
| weight format | per-row symmetric INT8 + FP32 scale |
| decoding | greedy autoregressive + incremental self-KV cache |
| output projection | tied token embedding + English ASCII lexical constraint |
| estimated packed model | ~1008 KiB |
| encoder state | ~32 KiB |
| decoder self/cross cache | ~224 KiB |

Run the estimator yourself:

```bash
python scripts/estimate_memory.py
```

## Why not just use OPUS-MT directly?

The current `Helsinki-NLP/opus-mt-ru-en` configuration is a Marian encoder-decoder with 6 encoder layers, 6 decoder layers, `d_model=512`, FFN 2048 and a vocabulary of about 62.5k; its published PyTorch checkpoint is roughly 307 MB. That is the wrong architecture shape for an N16R8 MCU.

NanoTranslate instead reduces **width + depth + context + vocabulary + decoding cost together**, then uses sequence-level knowledge distillation from a stronger teacher.

References:

- OPUS-MT RU->EN: https://huggingface.co/Helsinki-NLP/opus-mt-ru-en
- Kim & Rush, *Sequence-Level Knowledge Distillation*: https://aclanthology.org/D16-1139/
- Libovicky & Helcl, CTC/NAR MT: https://aclanthology.org/D18-1336/
- Helcl et al., realistic NAR speed evaluation: https://aclanthology.org/2022.naacl-main.129/
- ESP-DL: https://github.com/espressif/esp-dl
- ESP-NN: https://github.com/espressif/esp-nn
- ESP32-S3 LLM SIMD reference: https://github.com/doryiii/esp32-llm

## Architecture

```text
5-way joystick
      |
      v
+----------------------+         16 MB flash
| 240x240 ST7789 UI    |     +------------------+
| Russian keyboard     |     | .s3mt Q8 model   |
+----------+-----------+     | + FlashPiece     |
           |                 +---------+--------+
           v                           |
      UTF-8 Russian                    | mmap
           |                           v
           +----------------> +--------------------+
                              | TinyMT C++ runtime |
                              | 3 enc + 2 dec      |
                              | PIE INT8 matvec    |
                              +----------+---------+
                                         |
                                         v
                                   English text
                                         |
                                         v
                                  ST7789 display
```

### FlashPiece tokenizer

FlashPiece always includes all 256 byte values, so UTF-8 is lossless without an ICU/SentencePiece runtime. Training adds frequent byte n-grams until the vocabulary budget is reached. Encoding is greedy longest-match and is reproduced directly in C++.

### Selective English output head

The encoder needs Russian pieces, but the decoder is RU->EN only. During generation the runtime scores only:

- EOS;
- vocabulary pieces containing printable ASCII English/punctuation.

It skips Cyrillic/source-only rows entirely. Training targets are normalized to the same ASCII subset. **This is constrained decoding, not mathematically identical to an unconstrained full-vocabulary argmax:** if an excluded token would otherwise win, the prediction is deliberately changed. The constraint is a hardware-aware product choice and its quality cost must be reported against raw references.

### Q8 first, Q4 later

Version 1 uses row-wise INT8 weights and dynamically quantized INT8 activations for dense matvecs. On ESP32-S3 the hot dot product uses PIE signed-int8 SIMD. Q4 is in the optimization roadmap, but should only replace Q8 after a real quality/latency comparison on the board.

## Repository layout

```text
model/              PyTorch model, FlashPiece, training, distillation, exporter
runtime/            portable C++ model loader/tokenizer/inference core
host/               desktop CLI using the exact same runtime
firmware/           ESP-IDF firmware, ST7789 + joystick UI
scripts/            data download, model checker, flash helper, memory estimator
research/           LaTeX engineering research note + bibliography
docs/               hardware, training, status, optimization roadmap
tests/              tokenizer + quantized end-to-end smoke tests
```

# Quick start: host first

## 1. Python environment

```bash
python -m venv .venv
source .venv/bin/activate                 # Windows: .venv\Scripts\activate
pip install -r model/requirements.txt
```

## 2. Run the verified smoke test

```bash
cmake -S host -B host/build
cmake --build host/build -j
PYTHONPATH=model python tests/smoke_e2e.py
```

Expected key lines:

```text
smoke loss [~5.7, ~0.04]
hello
CRC rejection: OK
```

# Train a real model

## 3. Download RU/EN data

Recommended starting corpus: OPUS-100 using its provided train/validation/test splits.

```bash
python scripts/download_opus100.py \
  --out-dir data/opus100 \
  --limit-train 500000
```

For the first experiment, 100k-200k pairs are enough to debug the pipeline. For serious quality experiments increase the training set and keep the test split frozen.

## 4. Distill a teacher

```bash
python model/distill_teacher.py \
  --input data/opus100/train.tsv \
  --output data/distilled.tsv \
  --teacher Helsinki-NLP/opus-mt-ru-en \
  --batch-size 32
```

Teacher generation is a PC/GPU preprocessing step. The teacher never runs on the ESP32.

## 5. Build FlashPiece from the distilled training distribution

```bash
python model/tokenizer.py build \
  --input data/distilled.tsv \
  --output artifacts/tokenizer.json \
  --vocab-size 1536
```

Building the tokenizer **after distillation** spends the finite vocabulary budget on the Russian source plus the teacher-English distribution the student actually sees. Validation/test data is never used to build the tokenizer.

## 6. Train the student

```bash
python model/train.py \
  --train data/distilled.tsv \
  --val data/opus100/validation.tsv \
  --tokenizer artifacts/tokenizer.json \
  --preset base \
  --out artifacts/base \
  --epochs 20
```

Presets:

- `tiny`: 96-d, 2 encoder + 2 decoder layers;
- `base`: 128-d, 3 + 2;
- `quality`: 160-d, 4 + 2.

## 7. Evaluate

```bash
python model/evaluate.py \
  --checkpoint artifacts/base/best.pt \
  --tokenizer artifacts/tokenizer.json \
  --data data/opus100/test.tsv \
  --raw-data data/opus100/test.raw.tsv \
  --json-out artifacts/test.json
```

The evaluator reports example translations, exact match, BLEU and chrF when `sacrebleu` is installed. It can score both the device-normalized ASCII references and the preserved raw English references so preprocessing does not hide deployment quality loss.

## 8. Export Q8 model

```bash
python model/export_s3mt.py \
  --checkpoint artifacts/base/best.pt \
  --tokenizer artifacts/tokenizer.json \
  --output artifacts/nanotranslate.s3mt

python scripts/check_model.py artifacts/nanotranslate.s3mt
```

The `.s3mt` format contains a fixed header, FlashPiece vocabulary, aligned row-wise Q8 tensors, FP32 norms/scales and CRC32. The runtime maps weight data in place rather than copying a megabyte of weights into RAM.

## 9. Test the exact MCU runtime on your PC

```bash
./host/build/nanomt_cli artifacts/nanotranslate.s3mt
```

Example:

```text
ru> сегодня я иду в школу
I am going to school today.
```

The sentence above is only an interface example; actual wording depends on the trained checkpoint.

# ESP32-S3 firmware

Hardware assumptions and example wiring are in [`docs/HARDWARE.md`](docs/HARDWARE.md). Pins are configurable in menuconfig.

```bash
cd firmware
idf.py set-target esp32s3
idf.py menuconfig
idf.py build
idf.py flash monitor
```

The defaults enable:

- CPU 240 MHz;
- 16 MB QIO flash @ 80 MHz;
- 8 MB octal PSRAM @ 80 MHz;
- 64 KB data cache / 64-byte cache line;
- speed compiler optimization.

These settings match the general high-performance ESP32-S3 configuration used by current ESP-DL examples.

Flash the model into the dedicated 2 MiB partition:

```bash
python ../scripts/flash_model.py \
  --port COM5 \
  --model ../artifacts/nanotranslate.s3mt
```

Reset the board after flashing.

## Joystick UI

The default keyboard has Russian lowercase letters plus punctuation and controls:

```text
й ц у к е н г
ш щ з х ъ ф ы
в а п р о л д
ж э я ч с м и
т ь б ю ё , .
BK SP ? ! CLR GO
```

- move: joystick directions;
- select character: press;
- `BK`: delete one UTF-8 character;
- `SP`: space;
- `CLR`: clear input/output;
- `GO`: translate;
- long press: translate.

The UI renders to a framebuffer in PSRAM, copies 20-row strips to a DMA-capable internal-RAM buffer, waits for the SPI transfer-complete callback, and only then reuses that DMA buffer.

# What to optimize next

Do not start by making the model larger. Profile this order on actual silicon:

1. vocabulary sweep: 768 / 1024 / 1536 / 2048;
2. `tiny` vs `base` vs `quality` at fixed data;
3. Q4xQ8 PIE kernel + packed Q4 layout;
4. INT8/INT4 self/cross KV caches;
5. RMSNorm -> quantize -> matvec fusion;
6. dual-core only if bandwidth is not already saturated;
7. CTC/NAR only if it wins **batch=1 wall-clock latency** at acceptable BLEU/chrF.

See [`docs/OPTIMIZATION.md`](docs/OPTIMIZATION.md) and [`research/paper.tex`](research/paper.tex).

## Research note

`research/paper.tex` is a detailed NeurIPS-style preprint by **Baktiyar Ablaikhan**. It includes exact parameter/storage/MAC derivations, quantization math, model-format details, hardware diagrams, reproducible figures, literature citations, a measured PyTorch→Q8→C++ parity experiment, and an intentionally blank physical-board result table until real ESP32 measurements exist. Run `make paper` to regenerate the figures and produce `research/paper.pdf` locally.

## License

Code: MIT. Dataset/teacher/model licenses are separate; see [`NOTICE.md`](NOTICE.md) before distributing trained weights.
