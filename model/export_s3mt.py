#!/usr/bin/env python3
from __future__ import annotations
import argparse, math, struct, zlib
from pathlib import Path
import numpy as np
import torch
from tokenizer import FlashPiece

MAGIC=b"S3MT0001"; VERSION=1; HEADER_BYTES=128

def align16(buf: bytearray):
    while len(buf)%16: buf.append(0)

def put_f32vec(buf:bytearray, arr):
    a=np.asarray(arr,dtype=np.float32).reshape(-1)
    buf += struct.pack("<4I", len(a),0,0,0); buf += a.tobytes(order="C"); align16(buf)

def quant_rows(a):
    a=np.asarray(a,dtype=np.float32)
    assert a.ndim==2
    mx=np.max(np.abs(a),axis=1)
    scales=np.where(mx>0,mx/127.0,1.0).astype(np.float32)
    q=np.clip(np.rint(a/scales[:,None]),-127,127).astype(np.int8)
    return q,scales

def put_q8(buf:bytearray, arr):
    q,s=quant_rows(arr); rows,cols=q.shape
    buf += struct.pack("<4I",rows,cols,0,0); buf += s.tobytes(order="C"); align16(buf); buf += q.tobytes(order="C"); align16(buf)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--checkpoint",required=True); ap.add_argument("--tokenizer",required=True); ap.add_argument("--output",required=True)
    a=ap.parse_args(); ck=torch.load(a.checkpoint,map_location="cpu"); cfg=ck["config"]; sd={k:v.detach().cpu().numpy() for k,v in ck["state_dict"].items()}; tok=FlashPiece.load(a.tokenizer)
    if tok.vocab_size != cfg["vocab_size"]: raise SystemExit("tokenizer/model vocab mismatch")
    payload=bytearray()
    # Tokenizer section.
    payload += struct.pack("<I",tok.vocab_size)
    for p in tok.pieces:
        if len(p)>65535: raise ValueError("piece too long")
        payload += struct.pack("<H",len(p)); payload += p
    align16(payload); tokenizer_bytes=len(payload)
    # Fixed-order model section.
    put_q8(payload,sd["tok.weight"]); put_q8(payload,sd["src_pos.weight"]); put_q8(payload,sd["tgt_pos.weight"])
    for i in range(cfg["n_encoder"]):
        p=f"enc.{i}."; put_f32vec(payload,sd[p+"n1.weight"])
        for n in ("q","k","v","o"): put_q8(payload,sd[p+f"attn.{n}.weight"])
        put_f32vec(payload,sd[p+"n2.weight"]); put_q8(payload,sd[p+"ffn.w1.weight"]); put_q8(payload,sd[p+"ffn.w2.weight"])
    for i in range(cfg["n_decoder"]):
        p=f"dec.{i}."; put_f32vec(payload,sd[p+"n1.weight"])
        for n in ("q","k","v","o"): put_q8(payload,sd[p+f"self_attn.{n}.weight"])
        put_f32vec(payload,sd[p+"n2.weight"])
        for n in ("q","k","v","o"): put_q8(payload,sd[p+f"cross_attn.{n}.weight"])
        put_f32vec(payload,sd[p+"n3.weight"]); put_q8(payload,sd[p+"ffn.w1.weight"]); put_q8(payload,sd[p+"ffn.w2.weight"])
    put_f32vec(payload,sd["final.weight"])
    crc=zlib.crc32(payload)&0xffffffff; total=HEADER_BYTES+len(payload)
    vals=[VERSION,0,cfg["vocab_size"],cfg["d_model"],cfg["d_ff"],cfg["n_heads"],cfg["n_encoder"],cfg["n_decoder"],cfg["max_src"],cfg["max_tgt"],0,1,2,3,tokenizer_bytes,len(payload),crc,total]
    header=struct.pack("<8s18I48s",MAGIC,*vals,b"\0"*48); assert len(header)==HEADER_BYTES
    out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True); out.write_bytes(header+payload)
    print(f"wrote {out}: {total/1024:.1f} KiB, crc=0x{crc:08x}, vocab={tok.vocab_size}")
if __name__=="__main__": main()
