#!/usr/bin/env python3
from __future__ import annotations
import argparse, sys
from contextlib import nullcontext
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from model.tokenizer import normalize_text, normalize_english

def clean_field(s: str) -> str:
    return s.replace("\t", " ").replace("\r", " ").replace("\n", " ").strip()

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out-dir',required=True); ap.add_argument('--limit-train',type=int); a=ap.parse_args()
    from datasets import load_dataset
    ds=load_dataset('Helsinki-NLP/opus-100','en-ru')
    out=Path(a.out_dir);out.mkdir(parents=True,exist_ok=True)
    for split in ('train','validation','test'):
        if split not in ds: continue
        lim=a.limit_train if split=='train' else None; n=0
        raw_path=out/f'{split}.raw.tsv' if split in ('validation','test') else None
        raw_ctx=raw_path.open('w',encoding='utf-8') if raw_path else nullcontext(None)
        with (out/f'{split}.tsv').open('w',encoding='utf-8') as f, raw_ctx as rawf:
            for ex in ds[split]:
                tr=ex.get('translation',ex)
                ru=clean_field(normalize_text(tr['ru']))
                raw_en=clean_field(normalize_text(tr['en']))
                en=clean_field(normalize_english(raw_en))
                if not ru or not en or not raw_en: continue
                if len(ru)>240 or len(en)>240: continue
                f.write(ru+'\t'+en+'\n')
                if rawf is not None: rawf.write(ru+'\t'+raw_en+'\n')
                n+=1
                if lim and n>=lim: break
        print(split,n, f'raw={raw_path}' if raw_path else '')
if __name__=='__main__': main()
