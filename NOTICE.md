# Third-party components and licensing

Project-owned source code is MIT-licensed; that statement **does not automatically relicense** third-party code, fonts, datasets, model weights, or document templates.

- **ESP32-S3 assembly techniques:** the signed-INT8 PIE dot-product approach in `runtime/src/nanomt.cpp` was informed by MIT-licensed [doryiii/esp32-llm](https://github.com/doryiii/esp32-llm) and Espressif's ESP-NN implementation conventions. Check upstream notices when reusing the relevant ideas or code.
- **NeurIPS LaTeX style:** `research/neurips_2026.sty` is a bundled third-party conference style. Its presence reflects a paper layout choice and does **not** indicate acceptance, affiliation, or endorsement by NeurIPS. Preserve upstream notices and verify its redistribution terms.
- **UI font glyphs:** `firmware/main/font8x12.h` was generated from a local TrueType font via `scripts/make_font.py`. The exact originating font/license is **not documented in the repository**. Verify and document the source/license, or regenerate from a suitably licensed font before a public release.
- **Data and models:** OPUS-100 datasets, machine-translation teacher checkpoints, and outputs from them can have separate terms. Verify all upstream licenses before redistributing corpora, teacher outputs, or trained weights. None are included in this repository.

See [the public release checklist](docs/PUBLIC_RELEASE.md).
