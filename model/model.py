from __future__ import annotations
from dataclasses import dataclass, asdict
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class Config:
    vocab_size: int
    d_model: int = 128
    d_ff: int = 256
    n_heads: int = 4
    n_encoder: int = 3
    n_decoder: int = 2
    max_src: int = 64
    max_tgt: int = 48
    dropout: float = 0.1

PRESETS = {
    "tiny": dict(d_model=96, d_ff=192, n_heads=4, n_encoder=2, n_decoder=2),
    "base": dict(d_model=128, d_ff=256, n_heads=4, n_encoder=3, n_decoder=2),
    "quality": dict(d_model=160, d_ff=320, n_heads=5, n_encoder=4, n_decoder=2),
}

class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__(); self.weight = nn.Parameter(torch.ones(d)); self.eps=eps
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

class Attention(nn.Module):
    def __init__(self, d, heads, dropout=0.0):
        super().__init__(); assert d % heads == 0
        self.d=d; self.h=heads; self.hd=d//heads
        self.q=nn.Linear(d,d,bias=False); self.k=nn.Linear(d,d,bias=False)
        self.v=nn.Linear(d,d,bias=False); self.o=nn.Linear(d,d,bias=False)
        self.dropout=dropout
    def forward(self, qx, kvx, mask=None, key_padding_mask=None):
        B,T,_=qx.shape; S=kvx.shape[1]
        q=self.q(qx).view(B,T,self.h,self.hd).transpose(1,2)
        k=self.k(kvx).view(B,S,self.h,self.hd).transpose(1,2)
        v=self.v(kvx).view(B,S,self.h,self.hd).transpose(1,2)
        score=(q @ k.transpose(-1,-2)) / math.sqrt(self.hd)
        if mask is not None: score = score.masked_fill(mask, -1e9)
        if key_padding_mask is not None:
            score = score.masked_fill(key_padding_mask[:,None,None,:], -1e9)
        a=F.softmax(score,dim=-1); a=F.dropout(a,self.dropout,self.training)
        y=(a@v).transpose(1,2).contiguous().view(B,T,self.d)
        return self.o(y)

class FFN(nn.Module):
    def __init__(self,d,ff,dropout):
        super().__init__(); self.w1=nn.Linear(d,ff,bias=False); self.w2=nn.Linear(ff,d,bias=False); self.dropout=dropout
    def forward(self,x): return self.w2(F.dropout(F.silu(self.w1(x)),self.dropout,self.training))

class EncLayer(nn.Module):
    def __init__(self,c):
        super().__init__(); self.n1=RMSNorm(c.d_model); self.attn=Attention(c.d_model,c.n_heads,c.dropout); self.n2=RMSNorm(c.d_model); self.ffn=FFN(c.d_model,c.d_ff,c.dropout)
    def forward(self,x,pad):
        z=self.n1(x); x=x+self.attn(z,z,key_padding_mask=pad); x=x+self.ffn(self.n2(x)); return x

class DecLayer(nn.Module):
    def __init__(self,c):
        super().__init__(); self.n1=RMSNorm(c.d_model); self.self_attn=Attention(c.d_model,c.n_heads,c.dropout); self.n2=RMSNorm(c.d_model); self.cross_attn=Attention(c.d_model,c.n_heads,c.dropout); self.n3=RMSNorm(c.d_model); self.ffn=FFN(c.d_model,c.d_ff,c.dropout)
    def forward(self,x,enc,tgt_pad,src_pad):
        T=x.shape[1]; causal=torch.triu(torch.ones(T,T,dtype=torch.bool,device=x.device),1)[None,None,:,:]
        z=self.n1(x); x=x+self.self_attn(z,z,mask=causal,key_padding_mask=tgt_pad)
        x=x+self.cross_attn(self.n2(x),enc,key_padding_mask=src_pad)
        x=x+self.ffn(self.n3(x)); return x

class TinyMT(nn.Module):
    def __init__(self,c:Config):
        super().__init__(); self.config=c
        self.tok=nn.Embedding(c.vocab_size,c.d_model)
        self.src_pos=nn.Embedding(c.max_src,c.d_model); self.tgt_pos=nn.Embedding(c.max_tgt,c.d_model)
        self.enc=nn.ModuleList([EncLayer(c) for _ in range(c.n_encoder)])
        self.dec=nn.ModuleList([DecLayer(c) for _ in range(c.n_decoder)])
        self.final=RMSNorm(c.d_model)
        self.reset_parameters()
    def reset_parameters(self):
        # Small initialization is important with tied output embeddings.
        for m in self.modules():
            if isinstance(m,(nn.Linear,nn.Embedding)): nn.init.normal_(m.weight,mean=0.0,std=0.02)
        for m in self.modules():
            if isinstance(m,RMSNorm): nn.init.ones_(m.weight)
    def encode(self,src):
        B,S=src.shape; pos=torch.arange(S,device=src.device)[None,:]
        x=self.tok(src)+self.src_pos(pos); pad=src.eq(0)
        for l in self.enc: x=l(x,pad)
        return x,pad
    def decode(self,tgt,enc,src_pad):
        B,T=tgt.shape; pos=torch.arange(T,device=tgt.device)[None,:]
        x=self.tok(tgt)+self.tgt_pos(pos); tpad=tgt.eq(0)
        for l in self.dec: x=l(x,enc,tpad,src_pad)
        x=self.final(x)
        # Tied output head. Scale prevents early-training saturation.
        return F.linear(x,self.tok.weight) / math.sqrt(self.config.d_model)
    def forward(self,src,tgt_in):
        enc,sp=self.encode(src); return self.decode(tgt_in,enc,sp)
    @torch.no_grad()
    def greedy(self,src,bos=1,eos=2,max_new=None):
        enc,sp=self.encode(src); B=src.shape[0]; assert B==1
        out=torch.tensor([[bos]],device=src.device)
        limit=max_new or self.config.max_tgt-1
        for _ in range(limit):
            logits=self.decode(out,enc,sp)[:,-1]
            nxt=logits.argmax(-1,keepdim=True)
            out=torch.cat([out,nxt],1)
            if nxt.item()==eos: break
        return out
    def checkpoint(self): return {"config":asdict(self.config),"state_dict":self.state_dict()}
