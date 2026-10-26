#!/usr/bin/env python3
"""Generate an 8x12 monochrome bitmap header from a local TTF.
The repository stores only rasterized glyph bits, not the font file.
"""
from PIL import Image,ImageDraw,ImageFont
import argparse
from pathlib import Path

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--font',required=True); ap.add_argument('--out',required=True); ap.add_argument('--size',type=int,default=11); a=ap.parse_args()
    chars=''.join(chr(i) for i in range(32,127))+'абвгдеёжзийклмнопрстуфхцчшщъыьэюя'
    font=ImageFont.truetype(a.font,a.size)
    glyphs=[]
    for ch in chars:
        im=Image.new('1',(8,12),0); d=ImageDraw.Draw(im)
        box=d.textbbox((0,0),ch,font=font)
        w=box[2]-box[0]; h=box[3]-box[1]
        x=max(0,(8-w)//2-box[0]); y=max(0,(12-h)//2-box[1])
        d.text((x,y),ch,font=font,fill=1)
        rows=[]
        for yy in range(12):
            v=0
            for xx in range(8):
                if im.getpixel((xx,yy)): v|=1<<(7-xx)
            rows.append(v)
        glyphs.append((ord(ch),rows))
    p=Path(a.out); p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('w',encoding='utf-8') as f:
        f.write('#pragma once\n#include <stdint.h>\n#include <stddef.h>\n')
        f.write('typedef struct { uint32_t cp; uint8_t row[12]; } nt_glyph_t;\n')
        f.write('static const nt_glyph_t NT_FONT[] = {\n')
        for cp,rows in glyphs:
            f.write('  {%d,{%s}},\n'%(cp,','.join('0x%02x'%x for x in rows)))
        f.write('};\nstatic const size_t NT_FONT_COUNT=sizeof(NT_FONT)/sizeof(NT_FONT[0]);\n')
if __name__=='__main__':main()
