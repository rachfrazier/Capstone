from glob import glob
import os
from subprocess import call

for folder in sorted(glob('20*')):
    os.chdir(folder)
    call(['cp ../read* .'],shell=True)
    call(['./read_ud > ud_%s.txt' %folder],shell=True)
    call(['./read_vd > vd_%s.txt' %folder],shell=True)
    call(['./read_dd > dd_%s.txt' %folder],shell=True)
    call(['./read_qd > qd_%s.txt' %folder],shell=True)
    os.chdir('..')


