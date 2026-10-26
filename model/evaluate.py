#!/usr/bin/env python3
"""Evaluate translation against aligned normalized and optional raw references."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
from pathlib import Path

import torch

from tokenizer import FlashPiece
from model import TinyMT, Config


def load(path, device):
    # Load only tensor/state-dict data. Do not unpickle arbitrary model objects.
    ck = torch.load(path, map_location=device, weights_only=True)
    model = TinyMT(Config(**ck["config"]))
    model.load_state_dict(ck["state_dict"])
    return model.to(device).eval()


def read_parallel_pairs(data_path, raw_path=None, limit=1000):
    """Yield (ru, normalized_ref, raw_ref) while verifying row alignment.

    Training-data scripts write normalized and raw TSVs in identical row order.
    A mismatched row must fail loudly rather than skew reported BLEU/chrF.
    """
    with ExitStack() as stack:
        normalized = stack.enter_context(open(data_path, encoding="utf-8"))
        raw = stack.enter_context(open(raw_path, encoding="utf-8")) if raw_path else None
        seen = 0
        for row_number, line in enumerate(normalized, start=1):
            if seen >= limit:
                break
            parts = line.rstrip("\n").split("\t", 1)
            if len(parts) != 2:
                raise ValueError(f"Malformed normalized TSV at row {row_number}")
            ru, reference = parts
            raw_reference = None
            if raw is not None:
                raw_line = next(raw, None)
                if raw_line is None:
                    raise ValueError(f"Raw reference TSV ends before row {row_number}")
                raw_parts = raw_line.rstrip("\n").split("\t", 1)
                if len(raw_parts) != 2 or raw_parts[0] != ru:
                    raise ValueError(f"Raw/normalized source mismatch at row {row_number}")
                raw_reference = raw_parts[1]
            seen += 1
            yield ru, reference, raw_reference
        if raw is not None and seen < limit and next(raw, None) is not None:
            raise ValueError("Raw reference TSV contains additional rows")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--raw-data")
    parser.add_argument("--json-out")
    parser.add_argument("--limit", type=int, default=1000)
    args = parser.parse_args()

    if args.limit < 1:
        parser.error("--limit must be positive")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = FlashPiece.load(args.tokenizer)
    model = load(args.checkpoint, device)

    references, hypotheses, raw_references = [], [], []
    for ru, reference, raw_reference in read_parallel_pairs(args.data, args.raw_data, args.limit):
        ids = tok.encode(ru)
        if len(ids) > model.config.max_src:
            raise ValueError("Evaluation sentence exceeds model max_src; curate evaluation data or report exclusions")
        src = torch.tensor([ids], device=device)
        output = model.greedy(src)[0].tolist()
        hypothesis = tok.decode(output)
        references.append(reference)
        hypotheses.append(hypothesis)
        if raw_reference is not None:
            raw_references.append(raw_reference)
        if len(hypotheses) <= 10:
            print(f"RU: {ru}\nREF: {reference}\nHYP: {hypothesis}\n")

    if not references:
        raise ValueError("No evaluation pairs found")
    exact = sum(h.strip() == r.strip() for h, r in zip(hypotheses, references)) / len(references)
    result = {"n": len(references), "exact_normalized": exact}
    print(f"N={len(references)} exact={exact:.4f}")

    try:
        import sacrebleu
    except ImportError:
        print("install sacrebleu for BLEU/chrF")
    else:
        result["bleu_normalized"] = sacrebleu.corpus_bleu(hypotheses, [references]).score
        result["chrf_normalized"] = sacrebleu.corpus_chrf(hypotheses, [references]).score
        if raw_references:
            result["bleu_raw_reference"] = sacrebleu.corpus_bleu(hypotheses, [raw_references]).score
            result["chrf_raw_reference"] = sacrebleu.corpus_chrf(hypotheses, [raw_references]).score
        for name, value in result.items():
            if name.startswith(("bleu_", "chrf_")):
                print(name, value)

    if args.json_out:
        output_path = Path(args.json_out)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
