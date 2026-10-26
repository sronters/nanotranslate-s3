# NanoTranslate-S3

**An experimental, offline Russian-to-English neural translation system for the ESP32-S3.**

NanoTranslate-S3 explores **hardware–software co-design** for a tiny encoder–decoder Transformer on an ESP32-S3 N16R8 (16 MB flash, 8 MB PSRAM). The model, byte-level tokenizer, quantization format, C++ inference runtime, and joystick-driven display firmware are designed together for short sentences.

> **Research prototype, not a finished translator.** The complete PyTorch → INT8 export → portable C++ path has passed a **20-pair memorization test**. That demonstrates end-to-end correctness on the toy set, **not** general translation quality. A full model has not yet been benchmarked on physical ESP32-S3 hardware, and there are no measured board latency or held-out BLEU/chrF results.

## Current evidence

| Component | Status |
| --- | --- |
| FlashPiece tokenizer (UTF-8 with byte fallback) | Implemented; round-trip tests |
| Tiny Transformer and row-wise INT8 export | Implemented |
| Portable C++ tokenizer, model loader and inference | Host build and smoke test |
| CRC32 corruption rejection | Tested on host |
| ESP-IDF firmware and ST7789/joystick interface | Compiles in CI; **no physical-board verification** |
| General RU→EN translation quality | **Not yet measured** |
| ESP32-S3 latency, memory high-water marks and power | **Not yet measured** |

For the precise separation of verified, estimated, and unverified claims, see [Status](docs/STATUS.md).

## Design

The default `base` configuration is a **999,040-parameter** pre-norm Transformer (three encoder layers, two decoder layers, width 128, four attention heads, 1,536 vocabulary entries). Its exported matrices use row-wise symmetric INT8 weights and FP32 row scales; the C++ decoder uses incremental KV caching. The tokenizer includes all 256 raw byte values. These are architecture specifications, **not evidence of translation accuracy on a device**.

The ESP32-S3 runtime includes a PIE signed-int8 dot-product path, with a portable host implementation for parity testing. It uses constrained ASCII-English output tokens, so deployment decoding is not identical to unrestricted PyTorch greedy decoding.

## Reproduce the verified smoke test

Python 3.11+, CMake and a C++17 compiler are recommended. The host smoke test needs PyTorch and NumPy, **not an ESP32 board**.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install torch numpy pytest
cmake -S host -B host/build
cmake --build host/build -j
python -m pytest -q
PYTHONPATH=model python tests/smoke_e2e.py
```

Expected qualitative results: the toy model learns the training examples, C++ translates `привет` to `hello`, and corrupt model bytes are rejected. Exact timing and loss can vary across environments.

The fuller reproducible 20-pair parity experiment (which is still *not* a general translation benchmark):

```bash
make research
```

Generated measurements and analytical budget tables are in [`research/data/`](research/data/). See [Research artifacts](docs/RESEARCH.md) for the distinction between measured and calculated values.

## Training and evaluation (not a verified quality result)

The proposed pipeline uses OPUS-100 RU/EN, optional sequence-level distillation from a larger teacher, FlashPiece tokenization, student training, and evaluation against both normalized and raw English references:

```bash
python scripts/download_opus100.py --out-dir data/opus100 --limit-train 500000
python model/distill_teacher.py --input data/opus100/train.tsv --output data/distilled.tsv
python model/tokenizer.py build --input data/distilled.tsv --output artifacts/tokenizer.json
python model/train.py --train data/distilled.tsv --val data/opus100/validation.tsv \
  --tokenizer artifacts/tokenizer.json --preset base --out artifacts/base
python model/evaluate.py --checkpoint artifacts/base/best.pt \
  --tokenizer artifacts/tokenizer.json --data data/opus100/test.tsv \
  --raw-data data/opus100/test.raw.tsv --json-out artifacts/test.json
python model/export_s3mt.py --checkpoint artifacts/base/best.pt \
  --tokenizer artifacts/tokenizer.json --output artifacts/nanotranslate.s3mt
python scripts/check_model.py artifacts/nanotranslate.s3mt
```

This recipe is provided for reproduction and future experiments, **not as a claim that a validated production checkpoint is included**. Downloaded data, generated checkpoints, and binaries are ignored by Git. See [Training](docs/TRAINING.md).

## Firmware (experimental)

Hardware target: **ESP32-S3-WROOM-1 N16R8**, 240×240 ST7789 display, and five-way joystick. Example GPIO assignments and cautions are documented in [Hardware](docs/HARDWARE.md); wiring depends on your board.

```bash
cd firmware
idf.py set-target esp32s3
idf.py build
```

With the appropriate ESP-IDF toolchain and compatible board, flashing instructions are in [Hardware](docs/HARDWARE.md). **The published CI build does not validate physical translation, wiring, or hardware performance.**

## Paper and project map

- [Research note (PDF)](research/paper.pdf) · [LaTeX source](research/paper.tex) · [Research details](docs/RESEARCH.md)
- [`model/`](model/) — PyTorch model, tokenization, training and exporter
- [`runtime/`](runtime/) — portable C++ inference library
- [`host/`](host/) — desktop CLI for model inspection and translation
- [`firmware/`](firmware/) — ESP-IDF app and device UI
- [`experiments/`](experiments/) — analytical budgets and toy correctness experiments
- [`tests/`](tests/) — regression and smoke tests

The paper is a **research note/preprint** in NeurIPS-style formatting; it is **not evidence of NeurIPS acceptance or peer review**.

## Limitations and next milestones

The next scientific milestones are a reproducible held-out translation benchmark, PyTorch-versus-device decoding parity on representative inputs, and latency/peak-memory measurements on real hardware. Only then is it meaningful to compare Q4, smaller vocabularies, or alternative decoders. See [Optimization](docs/OPTIMIZATION.md).

## License and responsible reuse

Project-owned source code is provided under the [MIT License](LICENSE). Dataset and teacher-model licenses are separate, and the bundled conference style and generated glyph table require independent provenance review before wider redistribution. Read [Third-party notices](NOTICE.md) and [Public release checklist](docs/PUBLIC_RELEASE.md). No dataset or trained production checkpoint is distributed.
