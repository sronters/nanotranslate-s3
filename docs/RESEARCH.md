# Research artifacts and evidence

- [Paper PDF](../research/paper.pdf)
- [Editable LaTeX source](../research/paper.tex)
- [Research data tables](../research/data/)
- [Status and evidence level](STATUS.md)

The PDF is a **research-note/preprint snapshot**, formatted using a NeurIPS-style LaTeX template. It is not a peer-reviewed or accepted NeurIPS publication.

## Reproduction

From the repository root with Python, PyTorch, CMake/GCC, and a suitable LaTeX installation (`latexmk`, TikZ/PGFPlots and bibliography packages):

```bash
make research
make paper
```

The first command reruns analytical tables and the toy 20-pair PyTorch → row-wise INT8 → C++ parity experiment. The second rebuilds the paper from `research/paper.tex` and `research/refs.bib`.

## How to interpret numbers

| Type | Example | What it establishes |
| --- | --- | --- |
| Measured on host | 20-pair memorization, INT8/C++ parity, CRC rejection | Toy-pipeline correctness on the tested inputs |
| Exact analytical | Parameter count, dense MAC formula, logical cache sizes | Resource budgets for a specified architecture |
| Derived estimate | Prospective Q4 storage | Hypothesis, not an implemented or benchmarked optimization |
| Unmeasured | ESP32-S3 latency, BLEU, chrF, power | **No experimental result yet** |

**Do not reuse host timing as an MCU benchmark.** Device performance claims require actual hardware measurements and clear provenance.
