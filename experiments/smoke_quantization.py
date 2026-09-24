#!/usr/bin/env python3
"""Reproducible toy memorization/parity experiment.

This is deliberately NOT a translation-quality benchmark. It checks that a
small PyTorch model can memorize a tiny RU->EN set and that row-wise Q8 export
plus the portable C++ runtime preserves the memorized greedy translations.
"""
from __future__ import annotations
import argparse, csv, json, os, subprocess, sys, tempfile
from pathlib import Path
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "model"))
from tokenizer import FlashPiece
from model import TinyMT, Config

PAIRS = [
    ('привет','hello'),('доброе утро','good morning'),('спасибо','thank you'),('пожалуйста','please'),
    ('как дела','how are you'),('я дома','i am home'),('я в школе','i am at school'),('это моя книга','this is my book'),
    ('где автобус','where is the bus'),('мне нужна вода','i need water'),('я люблю музыку','i love music'),
    ('сегодня холодно','it is cold today'),('завтра будет тепло','it will be warm tomorrow'),('открой дверь','open the door'),
    ('закрой окно','close the window'),('который час','what time is it'),('я не понимаю','i do not understand'),
    ('повтори пожалуйста','please repeat'),('это очень интересно','this is very interesting'),('до свидания','goodbye')]


def padded(rows, pad=0):
    m=max(map(len,rows)); x=torch.full((len(rows),m),pad,dtype=torch.long)
    for i,r in enumerate(rows): x[i,:len(r)] = torch.tensor(r)
    return x


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out-dir',default='research/data'); ap.add_argument('--steps',type=int,default=600)
    a=ap.parse_args(); out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(0); torch.set_num_threads(1)
    tok=FlashPiece.build([x for p in PAIRS for x in p],vocab_size=320,min_count=1,max_piece_bytes=6)
    cfg=Config(tok.vocab_size,d_model=32,d_ff=64,n_heads=4,n_encoder=1,n_decoder=1,max_src=24,max_tgt=32,dropout=0.0)
    m=TinyMT(cfg); opt=torch.optim.AdamW(m.parameters(),lr=3e-3,weight_decay=0)
    src=padded([tok.encode(x,max_tokens=cfg.max_src) for x,_ in PAIRS]); tgt=padded([tok.encode(y,max_tokens=cfg.max_tgt) for _,y in PAIRS])
    history=[]
    for step in range(a.steps+1):
        if step % 10 == 0 or step == a.steps:
            m.eval()
            with torch.no_grad():
                logits=m(src,tgt[:,:-1]); loss=F.cross_entropy(logits.reshape(-1,logits.size(-1)),tgt[:,1:].reshape(-1),ignore_index=0)
            history.append((step,float(loss)))
        if step == a.steps: break
        m.train(); opt.zero_grad(set_to_none=True); logits=m(src,tgt[:,:-1])
        loss=F.cross_entropy(logits.reshape(-1,logits.size(-1)),tgt[:,1:].reshape(-1),ignore_index=0)
        loss.backward(); opt.step()
    with (out/'smoke_loss.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f); w.writerow(['step','cross_entropy']); w.writerows(history)

    m.eval(); py=[]
    for i,(ru,ref) in enumerate(PAIRS):
        hyp=tok.decode(m.greedy(src[i:i+1])[0].tolist()); py.append((ru,ref,hyp))
    py_exact=sum(ref==hyp for _,ref,hyp in py)

    with tempfile.TemporaryDirectory() as td_s:
        td=Path(td_s); tok.save(td/'tok.json'); torch.save(m.checkpoint(),td/'model.pt')
        env={**os.environ,'PYTHONPATH':str(ROOT/'model')}
        subprocess.check_call([sys.executable,str(ROOT/'model/export_s3mt.py'),'--checkpoint',str(td/'model.pt'),'--tokenizer',str(td/'tok.json'),'--output',str(td/'model.s3mt')],env=env)
        cli=ROOT/'host/build/nanomt_cli'
        if not cli.exists():
            subprocess.check_call(['cmake','-S',str(ROOT/'host'),'-B',str(ROOT/'host/build')])
            subprocess.check_call(['cmake','--build',str(ROOT/'host/build'),'-j2'])
        q8=[]
        for ru,ref in PAIRS:
            text=subprocess.check_output([str(cli),str(td/'model.s3mt'),ru],text=True)
            hyp=text.splitlines()[0].strip(); q8.append((ru,ref,hyp))
        q8_exact=sum(ref==hyp for _,ref,hyp in q8)
        model_bytes=(td/'model.s3mt').stat().st_size

    with (out/'smoke_outputs.tsv').open('w',encoding='utf-8') as f:
        f.write('ru\tref\tpytorch\tq8_cpp\n')
        for p,q in zip(py,q8): f.write(f'{p[0]}\t{p[1]}\t{p[2]}\t{q[2]}\n')
    summary={
        'seed':0,'pairs':len(PAIRS),'steps':a.steps,'initial_loss':history[0][1],'final_loss':history[-1][1],
        'pytorch_train_exact':py_exact,'q8_cpp_train_exact':q8_exact,'q8_model_bytes':model_bytes,
        'note':'Memorization/parity smoke test; not held-out MT quality.'
    }
    (out/'smoke_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))
    if py_exact != len(PAIRS) or q8_exact != len(PAIRS): raise SystemExit('parity smoke failed')

if __name__=='__main__': main()
