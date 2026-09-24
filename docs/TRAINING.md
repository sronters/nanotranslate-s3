# Training recipe

## Objective

Translate one ordinary Russian sentence (normally 5-20 words) to English. This deliberately narrow product requirement lets the student be much smaller than general machine-translation systems.

## Recommended pipeline

1. Download OPUS-100 RU/EN with its train/validation/test split.
2. Preserve both raw English references and a device-normalized printable-ASCII view; the latter matches the MCU target alphabet.
3. Generate teacher translations on the **training split only** with a stronger RU->EN model.
4. Build FlashPiece **only from the distilled training file** (Russian source + teacher English). This aligns the finite vocabulary budget with the distribution the student actually learns.
5. Train the student against teacher outputs (sequence-level distillation).
6. Tune architecture/vocabulary only on validation.
7. Report final BLEU/chrF once on the frozen test split, against both normalized and raw references.
8. Export Q8 and test host-runtime parity before flashing.

## Why teacher targets

A small student benefits from the teacher choosing one simpler translation among many acceptable references. This reduces target multimodality and is the classic reason sequence-level knowledge distillation is effective for compact NMT.

## Suggested experiment grid

| run | vocab | d | enc/dec | FFN | purpose |
|---|---:|---:|---:|---:|---|
| T1 | 768 | 96 | 2/2 | 192 | speed floor |
| B1 | 1024 | 128 | 3/2 | 256 | balanced |
| B2 | 1536 | 128 | 3/2 | 256 | default |
| B3 | 2048 | 128 | 3/2 | 256 | tokenizer/head tradeoff |
| Q1 | 1536 | 160 | 4/2 | 320 | quality ceiling |

Keep training data, teacher, seed and evaluation split fixed when comparing these.
