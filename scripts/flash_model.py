#!/usr/bin/env python3
import argparse,subprocess,sys
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--port',required=True);ap.add_argument('--model',required=True);ap.add_argument('--baud',default='921600');a=ap.parse_args();p=Path(a.model)
if p.stat().st_size>0x200000:raise SystemExit('model exceeds 2 MiB model partition')
cmd=[sys.executable,'-m','esptool','--chip','esp32s3','--port',a.port,'--baud',a.baud,'write_flash','0x210000',str(p)]
print(' '.join(cmd));subprocess.check_call(cmd)
