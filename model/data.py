from __future__ import annotations
import random
import torch
from torch.utils.data import Dataset
from tokenizer import normalize_text, normalize_english

class TSVPairs(Dataset):
    def __init__(self,path,tok,max_src=64,max_tgt=48,max_pairs=None):
        self.rows=[]; self.tok=tok; self.max_src=max_src; self.max_tgt=max_tgt
        with open(path,"r",encoding="utf-8") as f:
            for line in f:
                p=line.rstrip("\n").split("\t")
                if len(p)<2: continue
                ru,en=normalize_text(p[0]),normalize_english(p[1])
                if not ru or not en: continue
                # Check *untruncated* sequences. Tokenizer.encode(max_tokens=...)
                # would silently shorten overlong examples, contaminating targets.
                s = tok.encode(ru)
                t = tok.encode(en)
                if len(s) <= max_src and len(t) <= max_tgt:
                    self.rows.append((s, t))
                if max_pairs and len(self.rows)>=max_pairs: break
    def __len__(self): return len(self.rows)
    def __getitem__(self,i): return self.rows[i]

def collate(batch):
    bs=len(batch); ms=max(len(x[0]) for x in batch); mt=max(len(x[1]) for x in batch)
    src=torch.zeros(bs,ms,dtype=torch.long); tgt=torch.zeros(bs,mt,dtype=torch.long)
    for i,(s,t) in enumerate(batch): src[i,:len(s)]=torch.tensor(s); tgt[i,:len(t)]=torch.tensor(t)
    return src,tgt
