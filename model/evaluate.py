#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, torch
from tokenizer import FlashPiece
from model import TinyMT, Config

def load(path,device):
    ck=torch.load(path,map_location=device); m=TinyMT(Config(**ck["config"])); m.load_state_dict(ck["state_dict"]); m.to(device).eval(); return m

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--checkpoint",required=True); ap.add_argument("--tokenizer",required=True); ap.add_argument("--data",required=True); ap.add_argument("--raw-data"); ap.add_argument("--json-out"); ap.add_argument("--limit",type=int,default=1000)
    a=ap.parse_args(); device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); tok=FlashPiece.load(a.tokenizer); m=load(a.checkpoint,device)
    refs=[]; hyps=[]
    with open(a.data,encoding="utf-8") as f:
        for i,line in enumerate(f):
            if i>=a.limit: break
            p=line.rstrip().split("\t");
            if len(p)<2: continue
            ids=tok.encode(p[0],max_tokens=m.config.max_src); src=torch.tensor([ids],device=device); out=m.greedy(src)[0].tolist(); h=tok.decode(out)
            refs.append(p[1]); hyps.append(h)
            if len(hyps)<=10: print(f"RU: {p[0]}\nREF: {p[1]}\nHYP: {h}\n")
    exact=sum(h.strip()==r.strip() for h,r in zip(hyps,refs))/max(1,len(refs))
    result={"n":len(refs),"exact_normalized":exact}
    print(f"N={len(refs)} exact={exact:.4f}")
    raw_refs=None
    if a.raw_data:
        raw_refs=[]
        with open(a.raw_data,encoding="utf-8") as rf:
            for i,line in enumerate(rf):
                if i>=len(hyps): break
                p=line.rstrip().split("\t",1)
                if len(p)==2: raw_refs.append(p[1])
        if len(raw_refs)!=len(hyps): raise SystemExit("raw-data row count does not match evaluated hypotheses")
    try:
        import sacrebleu
        result["bleu_normalized"]=sacrebleu.corpus_bleu(hyps,[refs]).score
        result["chrf_normalized"]=sacrebleu.corpus_chrf(hyps,[refs]).score
        print("BLEU(normalized)",result["bleu_normalized"]); print("chrF(normalized)",result["chrf_normalized"])
        if raw_refs is not None:
            result["bleu_raw_reference"]=sacrebleu.corpus_bleu(hyps,[raw_refs]).score
            result["chrf_raw_reference"]=sacrebleu.corpus_chrf(hyps,[raw_refs]).score
            print("BLEU(raw reference)",result["bleu_raw_reference"]); print("chrF(raw reference)",result["chrf_raw_reference"])
    except ImportError: print("install sacrebleu for BLEU/chrF")
    if a.json_out:
        from pathlib import Path
        Path(a.json_out).write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
if __name__=="__main__": main()
