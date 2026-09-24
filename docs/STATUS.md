# Status and evidence level

## Verified in this repository

- Python FlashPiece tokenizer round-trip.
- Tiny encoder-decoder Transformer can overfit a 20-pair RU->EN smoke set.
- Exporter writes a row-wise INT8 `.s3mt` file with CRC32.
- Portable C++ loader parses that file and rejects a deliberately corrupted payload.
- On the smoke model, PyTorch and the quantized C++ runtime both produce `привет -> hello`.
- Host C++ project builds with CMake/GCC.

## Implemented but not physically verified in this environment

- ESP-IDF firmware build on an actual ESP32-S3 toolchain.
- ST7789 rendering on a particular display module.
- joystick pin wiring/debounce on a particular board.
- PIE SIMD path on physical silicon.
- full model flash-partition mapping on physical silicon.

## Not claimed yet

There are **no invented ESP32 latency, BLEU or chrF numbers** in this repository. The final performance table must be filled only after training a real checkpoint and flashing a board.

A useful first board experiment is:

1. train `tiny`, `base`, and `quality` presets on the same distilled corpus;
2. evaluate BLEU/chrF on the same frozen test set;
3. flash each exported model;
4. record encoder time, decoder time, output tokens/s, RAM and model size;
5. only then decide whether Q4, CTC/NAR, dual-core, or a smaller vocabulary is worth the complexity.
