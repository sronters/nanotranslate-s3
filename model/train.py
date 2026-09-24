#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, math, random
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tokenizer import FlashPiece
from model import TinyMT, Config, PRESETS
from data import TSVPairs, collate

def evaluate_loss(model,loader,device,limit=100):
    model.eval(); total=n=0.0
    with torch.no_grad():
        for i,(src,tgt) in enumerate(loader):
            if i>=limit: break
            src,tgt=src.to(device),tgt.to(device); logits=model(src,tgt[:,:-1]); y=tgt[:,1:]
            loss=F.cross_entropy(logits.reshape(-1,logits.size(-1)),y.reshape(-1),ignore_index=0)
            total+=loss.item(); n+=1
    return total/max(n,1)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--train",required=True); ap.add_argument("--val")
    ap.add_argument("--tokenizer",required=True); ap.add_argument("--preset",choices=PRESETS,default="base")
    ap.add_argument("--out",required=True); ap.add_argument("--epochs",type=int,default=15); ap.add_argument("--batch-size",type=int,default=128)
    ap.add_argument("--lr",type=float,default=3e-4); ap.add_argument("--max-pairs",type=int); ap.add_argument("--seed",type=int,default=42)
    args=ap.parse_args(); random.seed(args.seed); torch.manual_seed(args.seed)
    device=torch.device("cuda" if torch.cuda.is_available() else "mps" if hasattr(torch.backends,"mps") and torch.backends.mps.is_available() else "cpu")
    tok=FlashPiece.load(args.tokenizer); cfg=Config(vocab_size=tok.vocab_size,**PRESETS[args.preset])
    tr=TSVPairs(args.train,tok,cfg.max_src,cfg.max_tgt,args.max_pairs); va=TSVPairs(args.val,tok,cfg.max_src,cfg.max_tgt,10000) if args.val else None
    dl=DataLoader(tr,batch_size=args.batch_size,shuffle=True,collate_fn=collate,num_workers=0); vdl=DataLoader(va,batch_size=args.batch_size,shuffle=False,collate_fn=collate) if va else None
    model=TinyMT(cfg).to(device); opt=torch.optim.AdamW(model.parameters(),lr=args.lr,betas=(0.9,0.98),weight_decay=0.01)
    total_steps=max(1,args.epochs*len(dl)); warm=max(10,min(1000,total_steps//20))
    def lr_mult(step):
        if step<warm: return (step+1)/warm
        p=(step-warm)/max(1,total_steps-warm); return 0.1+0.9*0.5*(1+math.cos(math.pi*p))
    sch=torch.optim.lr_scheduler.LambdaLR(opt,lr_mult)
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True); best=1e9; step=0
    print(f"device={device} pairs={len(tr)} params={sum(p.numel() for p in model.parameters()):,}")
    for ep in range(1,args.epochs+1):
        model.train(); running=seen=0.0
        for src,tgt in dl:
            src,tgt=src.to(device),tgt.to(device); opt.zero_grad(set_to_none=True)
            logits=model(src,tgt[:,:-1]); y=tgt[:,1:]
            loss=F.cross_entropy(logits.reshape(-1,logits.size(-1)),y.reshape(-1),ignore_index=0,label_smoothing=0.05)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); sch.step(); step+=1
            running+=loss.item(); seen+=1
        val=evaluate_loss(model,vdl,device) if vdl else running/max(seen,1)
        print(f"epoch={ep:02d} train={running/max(seen,1):.4f} val={val:.4f} lr={sch.get_last_lr()[0]:.2e}")
        torch.save(model.checkpoint(),out/"last.pt")
        if val<best:
            best=val; torch.save(model.checkpoint(),out/"best.pt")
    (out/"metrics.json").write_text(json.dumps({"best_val_loss":best,"params":sum(p.numel() for p in model.parameters()),"config":model.checkpoint()["config"]},indent=2))

if __name__=="__main__":
    main()
