#!/usr/bin/env python3
from pathlib import Path
import sys
from collections import defaultdict

if __name__ == "__main__":
    if (len(sys.argv) < 2):
        print ("Usage: keep_only_last_checkpoints folder1 [folder2] ... [folderN]")
        sys.exit (1)
    pt_per_folder = defaultdict(list)
    for p in sys.argv[1:]:
        files = Path(p).glob('**/*.pt')
        for f in files:
            pt_per_folder[f.parent].append (f)

    for p, files in pt_per_folder.items():
        files.sort()
        print (f"Keeping {files[-1]}, removing {len(files)-1} files.")
        for f in files[:-1]:
            f.unlink({f})
