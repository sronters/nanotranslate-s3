# Status and evidence

**NanoTranslate-S3 is an experimental prototype, not yet a validated offline translator.** The evidence levels below should be preserved in any publication, portfolio, or demo description.

## Verified by code / existing repository tests

- FlashPiece tokenizer round-trip on the test strings.
- Small PyTorch Transformer memorizes 20 training pairs.
- Q8 export and portable host C++ inference reproduce those memorized translations.
- The host loader rejects a deliberately modified model with a CRC32 mismatch.
- Host C++ project builds with CMake.
- ESP-IDF project **compiles in GitHub Actions CI** using the specified toolchain image. This is a compilation check only.

These claims are about the checked-in smoke test and automated build; they do not establish real-world translation quality.

## Implemented but unverified on physical hardware

- ST7789 display and five-way joystick operation on a specific assembled board.
- Flash partition mapping and model inference on the ESP32-S3.
- ESP32-S3 PIE SIMD execution and end-to-end output parity.
- Measured inference latency, memory usage under load, energy and power.

## Results not yet available

- Held-out BLEU/chrF (including both normalized and raw English references).
- Translation quality for the full student model.
- Board tokens/sec, median/p95 latency, and memory high-water marks.
- Statistical results for architecture, vocabulary, and quantization comparisons.

The 20-pair overfit experiment is a **memorization/parity test**, not a translation benchmark. Analytical MAC/storage figures describe architecture budgets, **not hardware measurements**.

## Before claiming a working standalone translator

1. Build and flash a valid model on the target N16R8 board; verify display and joystick.
2. Measure device output against the same frozen examples used for host and PyTorch comparisons; document decoding constraints.
3. Evaluate held-out translation quality against raw and normalized references.
4. Measure latency, RAM and power on repeated inputs, including long and short sequences.
5. Report hardware revision, ESP-IDF and compiler versions, model checksum and exact test methodology.

See [Public release checklist](PUBLIC_RELEASE.md) for license and privacy checks before switching the repository to public.
