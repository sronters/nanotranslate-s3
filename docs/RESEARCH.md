# Research paper and reproducibility

**[Read the full NanoTranslate-S3 research paper (PDF)](../research/paper.pdf)**

The repository distributes the compiled research paper only, not the LaTeX project or bibliography source. It is a **research note/preprint**, not a peer-reviewed or accepted NeurIPS publication. Conference-style formatting does not imply endorsement.

## Experimental artifacts

- [Analytical budgets and smoke-test results](../research/data/)
- [Experimental scripts](../experiments/)
- [Status and evidence levels](STATUS.md)

To regenerate the analytical data tables and rerun the host PyTorch → INT8 → C++ memorization/parity smoke experiment:

```bash
make research
```

This command reproduces the experimental inputs and recorded statistics; **it does not rebuild the PDF**, which is intentionally distributed as a finished document.

## Interpreting results

| Evidence | Example | Meaning |
| --- | --- | --- |
| Host measurement | 20-pair memorization, INT8/C++ parity, CRC rejection | Pipeline correctness on a toy training set |
| Exact analytical calculation | Parameter count, dense MAC formula, logical cache sizes | Architecture/resource budgets |
| Derived estimate | Potential Q4 storage size | Proposed optimization, not measured |
| Not yet measured | ESP32-S3 translation speed, BLEU, chrF, power | No device benchmark or held-out quality claim |

Host timing cannot be used to infer MCU latency. Device performance requires physical measurement and reproducible evaluation.
