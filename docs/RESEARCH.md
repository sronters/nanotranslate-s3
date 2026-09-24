# Research artifacts

The detailed paper source is [`research/paper.tex`](../research/paper.tex). Run `make paper` from the repository root to regenerate analytical data, re-run the measured parity smoke experiment, and compile `research/paper.pdf` locally.

Evidence is deliberately separated into four classes:

- **Measured:** host build/tests, toy memorization, PyTorch -> Q8 -> C++ parity, CRC rejection.
- **Exact analytical:** parameter count, Q8 tensor payload, cache bytes, framebuffer/DMA bytes, dense MAC equations.
- **Derived estimate:** prospective row-wise Q4 packed payload before a Q4 kernel exists.
- **Planned/unverified:** full-model BLEU/chrF/COMET, real ESP32-S3 latency/tokens-per-second/power.

Reproduce the research inputs with:

```bash
python experiments/analytic_sweep.py --out-dir research/data
python experiments/smoke_quantization.py --out-dir research/data
cd research && latexmk -pdf -interaction=nonstopmode -halt-on-error paper.tex
```

Do not fill the physical-board result table from host timings. The host runtime exists for correctness/parity debugging, not MCU performance extrapolation.
