import os
import re
import sys

basedir = os.path.abspath(os.path.dirname(__file__))
input_filename = os.path.join(basedir, 'Dockerfile.build.in')
output_filename = os.path.join(
    basedir, 'Dockerfile.build.{}'.format(sys.argv[1]))

m = re.match('^py(\d)(\d)-cuda(\d)(\d)-cudnn(\d)$', sys.argv[1])
if m:
    pyver1, pyver2, cudaver1, cudaver2, cudnnver = list(m.groups())

    print('Generating {}.'.format(os.path.basename(output_filename)))
    with open(input_filename, 'r') as f_in:
        with open(output_filename, 'w') as f_out:
            basetag = '{}.{}-cudnn{}-devel-centos6'.format(
                cudaver1, cudaver2, cudnnver)
            pyenvname = 'py{}{}'.format(pyver1, pyver2)
            pyversion = '{}.{}'.format(pyver1, pyver2)
            f_out.write(f_in.read().format(BASETAG=basetag,
                                           PYENVNAME=pyenvname,
                                           PYENVVERSION=pyversion))
