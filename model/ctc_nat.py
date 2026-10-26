"""Experimental CTC/NAR RU->EN baseline.

This is deliberately not the production firmware path. It exists so the same
training data can test whether one-pass translation is actually faster/better on
ESP32-S3 after wall-clock measurement, rather than assuming NAR wins.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import RMSNorm, EncLayer, Config

class CTCNAT(nn.Module):
    def __init__(self, c: Config, upsample: int = 2):
        super().__init__(); self.c=c; self.upsample=upsample
        self.tok=nn.Embedding(c.vocab_size,c.d_model); self.pos=nn.Embedding(c.max_src,c.d_model)
        self.enc=nn.ModuleList([EncLayer(c) for _ in range(c.n_encoder)])
        self.proj=nn.Linear(c.d_model,c.vocab_size,bias=False); self.proj.weight=self.tok.weight
    def forward(self,src):
        B,S=src.shape; p=torch.arange(S,device=src.device)[None,:]; x=self.tok(src)+self.pos(p); pad=src.eq(0)
        for l in self.enc:x=l(x,pad)
        x=x.repeat_interleave(self.upsample,dim=1)
        return self.proj(x)/(self.c.d_model**0.5)
    @staticmethod
    def collapse(ids,blank=0,eos=2):
        out=[]; prev=None
        for i in ids:
            i=int(i)
            if i!=blank and i!=prev: out.append(i)
            prev=i
            if i==eos: break
        return out
