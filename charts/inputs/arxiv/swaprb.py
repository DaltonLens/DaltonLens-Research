#!/usr/bin/env python3

import sys
import re
import shutil
from pathlib import Path

json_file = sys.argv[1]
out_file = json_file + '.swaprb'
bak_file = json_file + '.bak'
if Path(bak_file).exists():
    print ("Already a backup file, aborting to avoid applying twice.")
    sys.exit (1)

with open(json_file, 'r') as in_f, open(out_file, 'w') as out_f:
    for l in in_f:
        print (l)
        l = re.sub(r'"rgb_color":\[(\d+),(\d+),(\d+)\]',
                   r'"rgb_color":[\3,\2,\1]',
                   l)
        out_f.write(l)
        print (l)

shutil.move (json_file, bak_file)
shutil.move (out_file, json_file)
