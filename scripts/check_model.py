#!/usr/bin/env python3
import argparse,struct,zlib
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('model');a=p.parse_args();b=Path(a.model).read_bytes()
if len(b)<128 or b[:8]!=b'S3MT0001':raise SystemExit('bad magic/size')
h=struct.unpack_from('<18I',b,8); names=['version','flags','vocab','d_model','d_ff','heads','enc','dec','max_src','max_tgt','pad','bos','eos','unk','tokenizer_bytes','payload_bytes','crc32','total_bytes'];d=dict(zip(names,h))
calc=zlib.crc32(b[128:])&0xffffffff
print('\n'.join(f'{k}: {v}' if k!='crc32' else f'{k}: 0x{v:08x}' for k,v in d.items()));print(f'calculated_crc32: 0x{calc:08x}');print('OK' if calc==d['crc32'] and len(b)==d['total_bytes'] else 'INVALID')
