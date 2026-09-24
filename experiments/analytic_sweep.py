#!/usr/bin/env python3
"""Deterministic analytical budget sweep for NanoTranslate-S3.

No device latency is inferred here. The script reports exact parameter counts,
serialized-tensor byte estimates, cache sizes, and multiply-accumulate counts
from the model equations used in model/model.py and runtime/src/nanomt.cpp.
"""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path

PRESETS = {
    "tiny": dict(d=96, ff=192, enc=2, dec=2),
    "base": dict(d=128, ff=256, enc=3, dec=2),
    "quality": dict(d=160, ff=320, enc=4, dec=2),
}


def budget(vocab: int, d: int, ff: int, enc: int, dec: int,
           max_src: int = 64, max_tgt: int = 48,
           src_len: int = 32, tgt_len: int = 24,
           target_candidates: int | None = None):
    # Trainable parameters.
    token = vocab * d
    pos = (max_src + max_tgt) * d
    enc_matrix = enc * (4 * d * d + 2 * d * ff)
    dec_matrix = dec * (8 * d * d + 2 * d * ff)
    norms = (2 * enc + 3 * dec + 1) * d
    params = token + pos + enc_matrix + dec_matrix + norms

    # Exporter stores all matrix entries in int8, one float32 scale per row,
    # and all RMSNorm vectors in float32. This excludes the variable-length
    # tokenizer strings, 128-byte header, and <=15-byte alignment padding.
    q_values = token + pos + enc_matrix + dec_matrix
    scale_rows = (
        vocab + max_src + max_tgt
        + enc * (5 * d + ff)
        + dec * (9 * d + ff)
    )
    q8_tensor_bytes = q_values + 4 * scale_rows + 4 * norms
    # A prospective row-wise Q4 representation would halve packed values while
    # retaining row scales/norms; this is an analytical lower-level format size,
    # not a measured implementation result.
    q4_tensor_bytes = (q_values + 1) // 2 + 4 * scale_rows + 4 * norms

    # Logical states used by the current float-cache runtime.
    encoder_state = max_src * d * 4
    decoder_cache = dec * (2 * max_tgt * d + 2 * max_src * d) * 4
    framebuffer = 240 * 240 * 2
    dma_strip = 240 * 20 * 2

    S, T = src_len, tgt_len
    # MAC counts for incremental AR inference. Softmax/normalization/exponentials
    # are excluded; this is dense-dot arithmetic only.
    enc_macs = enc * (4*S*d*d + 2*S*S*d + 2*S*d*ff)
    cross_precompute_macs = dec * (2*S*d*d)
    decoder_core_macs = 0
    for t in range(T):
        decoder_core_macs += dec * (6*d*d + 2*d*ff + 2*d*((t+1)+S))
    C = vocab if target_candidates is None else min(vocab, target_candidates)
    output_head_macs = T * C * d
    total_macs = enc_macs + cross_precompute_macs + decoder_core_macs + output_head_macs

    return {
        "vocab": vocab, "d": d, "ff": ff, "enc": enc, "dec": dec,
        "max_src": max_src, "max_tgt": max_tgt,
        "src_len_for_macs": S, "tgt_len_for_macs": T,
        "token_params": token, "position_params": pos,
        "encoder_matrix_params": enc_matrix, "decoder_matrix_params": dec_matrix,
        "norm_params": norms, "params": params,
        "q8_tensor_bytes": q8_tensor_bytes, "q4_tensor_bytes_est": q4_tensor_bytes,
        "encoder_state_bytes": encoder_state, "decoder_cache_bytes": decoder_cache,
        "framebuffer_bytes": framebuffer, "dma_strip_bytes": dma_strip,
        "target_candidates": C,
        "enc_macs": enc_macs, "cross_precompute_macs": cross_precompute_macs,
        "decoder_core_macs": decoder_core_macs, "output_head_macs": output_head_macs,
        "total_macs": total_macs,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="research/data")
    ap.add_argument("--vocabs", default="512,768,1024,1536,2048,3072")
    ap.add_argument("--src-len", type=int, default=32)
    ap.add_argument("--tgt-len", type=int, default=24)
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    vocabs = [int(x) for x in args.vocabs.split(",")]

    rows = []
    for name, p in PRESETS.items():
        b = budget(1536, **p, src_len=args.src_len, tgt_len=args.tgt_len)
        b["preset"] = name; rows.append(b)
    with (out / "preset_budgets.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    sweep = []
    for v in vocabs:
        b = budget(v, **PRESETS["base"], src_len=args.src_len, tgt_len=args.tgt_len)
        b["preset"] = "base"; sweep.append(b)
    with (out / "vocab_sweep.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(sweep[0])); w.writeheader(); w.writerows(sweep)

    base = budget(1536, **PRESETS["base"], src_len=args.src_len, tgt_len=args.tgt_len)
    (out / "base_budget.json").write_text(json.dumps(base, indent=2), encoding="utf-8")
    print(json.dumps({
        "base_params": base["params"],
        "base_q8_tensor_kib": base["q8_tensor_bytes"] / 1024,
        "decoder_cache_kib": base["decoder_cache_bytes"] / 1024,
        "macs_at_32x24_m": base["total_macs"] / 1e6,
    }, indent=2))

if __name__ == "__main__":
    main()
