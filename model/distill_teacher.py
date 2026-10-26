#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
from tokenizer import normalize_text, normalize_english

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--input",required=True); ap.add_argument("--output",required=True); ap.add_argument("--teacher",default="Helsinki-NLP/opus-mt-ru-en"); ap.add_argument("--batch-size",type=int,default=32); ap.add_argument("--limit",type=int)
    a=ap.parse_args()
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    import torch
    dev="cuda" if torch.cuda.is_available() else "cpu"; tok=AutoTokenizer.from_pretrained(a.teacher); model=AutoModelForSeq2SeqLM.from_pretrained(a.teacher).to(dev).eval()
    rows=[]
    with open(a.input,encoding="utf-8") as f:
        for line in f:
            p=line.rstrip("\n").split("\t")
            if len(p)>=2: rows.append((p[0],p[1]))
            if a.limit and len(rows)>=a.limit: break
    Path(a.output).parent.mkdir(parents=True,exist_ok=True)
    with open(a.output,"w",encoding="utf-8") as out:
        for i in range(0,len(rows),a.batch_size):
            src=[normalize_text(x[0]) for x in rows[i:i+a.batch_size]]
            enc=tok(src,return_tensors="pt",padding=True,truncation=True,max_length=128).to(dev)
            with torch.no_grad(): gen=model.generate(**enc,num_beams=4,max_new_tokens=80,early_stopping=True)
            pred=tok.batch_decode(gen,skip_special_tokens=True)
            for s,t in zip(src,pred): out.write(s.replace("\t"," ")+"\t"+normalize_english(t).replace("\t"," ")+"\n")
            if i%(a.batch_size*20)==0: print(f"{i}/{len(rows)}")
if __name__=="__main__": main()
