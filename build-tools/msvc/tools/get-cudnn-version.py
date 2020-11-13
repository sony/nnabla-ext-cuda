import glob
import os
import re
import sys

with open(os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), 'cudnn_version.bat'), 'w') as w:
    for inc in glob.glob(os.path.join(sys.argv[1], 'include', '*.h')):
        with open(inc) as f:
            for l in f.readlines():
                m = re.match('^#define\s+CUDNN_(\S+)\s+(\d+)', l)
                if m and m.group(1) in ['MAJOR', 'MINOR', 'PATCHLEVEL']:
                    print(f'SET CUDNN_{m.group(1)}={m.group(2)}', file=w)
