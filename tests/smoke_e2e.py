#!/usr/bin/env python3
from pathlib import Path
import os, subprocess, sys, tempfile
import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'model'))
from tokenizer import FlashPiece
from model import TinyMT,Config

PAIRS=[
('привет','hello'),('доброе утро','good morning'),('спасибо','thank you'),('пожалуйста','please'),
('как дела','how are you'),('я дома','i am home'),('я в школе','i am at school'),('это моя книга','this is my book'),
('где автобус','where is the bus'),('мне нужна вода','i need water'),('я люблю музыку','i love music'),
('сегодня холодно','it is cold today'),('завтра будет тепло','it will be warm tomorrow'),('открой дверь','open the door'),
('закрой окно','close the window'),('который час','what time is it'),('я не понимаю','i do not understand'),
('повтори пожалуйста','please repeat'),('это очень интересно','this is very interesting'),('до свидания','goodbye')]

def padded(rows,pad=0):
    m=max(map(len,rows)); x=torch.full((len(rows),m),pad,dtype=torch.long)
    for i,r in enumerate(rows):x[i,:len(r)]=torch.tensor(r)
    return x

def main():
    torch.manual_seed(0); torch.set_num_threads(1)
    with tempfile.TemporaryDirectory() as td:
        td=Path(td); tok=FlashPiece.build([x for p in PAIRS for x in p],vocab_size=320,min_count=1,max_piece_bytes=6); tok.save(td/'tok.json')
        cfg=Config(tok.vocab_size,d_model=32,d_ff=64,n_heads=4,n_encoder=1,n_decoder=1,max_src=24,max_tgt=20,dropout=0.0)
        m=TinyMT(cfg); opt=torch.optim.AdamW(m.parameters(),lr=3e-3,weight_decay=0)
        src=padded([tok.encode(a,max_tokens=cfg.max_src) for a,b in PAIRS]); tgt=padded([tok.encode(b,max_tokens=cfg.max_tgt) for a,b in PAIRS])
        losses=[]
        for step in range(600):
            opt.zero_grad(); logits=m(src,tgt[:,:-1]); loss=F.cross_entropy(logits.reshape(-1,logits.size(-1)),tgt[:,1:].reshape(-1),ignore_index=0); loss.backward(); opt.step()
            if step in (0,599): losses.append(loss.item())
        print('smoke loss',losses)
        assert losses[-1] < 0.08, losses
        assert tok.decode(m.greedy(src[0:1])[0].tolist()) == 'hello'
        torch.save(m.checkpoint(),td/'model.pt')
        subprocess.check_call([sys.executable,str(ROOT/'model/export_s3mt.py'),'--checkpoint',str(td/'model.pt'),'--tokenizer',str(td/'tok.json'),'--output',str(td/'m.s3mt')],env={**os.environ,'PYTHONPATH':str(ROOT/'model')})
        cli=ROOT/'host/build/nanomt_cli'
        if not cli.exists(): subprocess.check_call(['cmake','-S',str(ROOT/'host'),'-B',str(ROOT/'host/build')]); subprocess.check_call(['cmake','--build',str(ROOT/'host/build'),'-j2'])
        out=subprocess.check_output([str(cli),str(td/'m.s3mt'),'привет'],text=True)
        print(out.strip())
        assert out.splitlines()[0].strip() == 'hello'
        bad=bytearray((td/'m.s3mt').read_bytes()); bad[-17]^=0x55; (td/'bad.s3mt').write_bytes(bad)
        p=subprocess.run([str(cli),str(td/'bad.s3mt'),'привет'],text=True,capture_output=True)
        assert p.returncode!=0 and 'CRC32 mismatch' in p.stderr
        print('CRC rejection: OK')

if __name__=='__main__': main()
