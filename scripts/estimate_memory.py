#!/usr/bin/env python3
import argparse
ap=argparse.ArgumentParser();ap.add_argument('--vocab',type=int,default=1536);ap.add_argument('--d',type=int,default=128);ap.add_argument('--ff',type=int,default=256);ap.add_argument('--enc',type=int,default=3);ap.add_argument('--dec',type=int,default=2);ap.add_argument('--max-src',type=int,default=64);ap.add_argument('--max-tgt',type=int,default=48);a=ap.parse_args()
emb=a.vocab*a.d+(a.max_src+a.max_tgt)*a.d; enc=a.enc*(4*a.d*a.d+2*a.d*a.ff); dec=a.dec*(8*a.d*a.d+2*a.d*a.ff); norms=(2*a.enc+3*a.dec+1)*a.d; params=emb+enc+dec+norms
q8=emb+enc+dec; scale_rows=(a.vocab+a.max_src+a.max_tgt)+a.enc*(4*a.d+a.ff+a.d)+a.dec*(8*a.d+a.ff+a.d); model=q8+4*scale_rows+4*norms
enc_state=a.max_src*a.d*4; dec_cache=a.dec*(2*a.max_tgt*a.d+2*a.max_src*a.d)*4
print(f'parameters: {params:,}\napprox Q8 model: {model/1024:.1f} KiB\nencoder state: {enc_state/1024:.1f} KiB\ndecoder KV/cross cache: {dec_cache/1024:.1f} KiB')
