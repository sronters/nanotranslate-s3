# Optimization roadmap

The order matters. Optimize after profiling on real N16R8 hardware.

1. **Q8 baseline first.** It is simple, robust and maps naturally to S3 signed-int8 SIMD.
2. **Vocabulary sweep:** 768 / 1024 / 1536 / 2048. The tied output head is evaluated every generated token, so vocabulary size directly affects latency.
3. **Width/depth sweep:** compare `tiny`, `base`, `quality` at fixed data and tokenizer.
4. **Selective output head:** current runtime scores only EOS plus printable-ASCII pieces, because output is English-only.
5. **Q4 weights:** pack two weights per byte and use a Q4xQ8 PIE kernel. Measure flash bandwidth and quality before adopting it.
6. **Quantized KV/cross cache:** decoder cache is FP32 in v1 for simplicity. INT8/INT4 caches can cut PSRAM traffic.
7. **Fused RMSNorm -> activation quantization -> projection:** avoid extra passes over hot vectors.
8. **Cross-attention cache compression:** precomputed decoder cross K/V is an obvious PSRAM target.
9. **Dual-core only after profiling.** If model-weight reads already saturate memory bandwidth, a second core can make little difference.
10. **CTC/NAR experiment:** compare true wall-clock batch=1 latency; do not assume one-pass decoding is automatically faster on a microcontroller.

The fastest useful model is not necessarily the model with the fewest FLOPs. On S3, model layout, flash/PSRAM traffic, alignment, output-vocabulary projection and cache location can dominate.
