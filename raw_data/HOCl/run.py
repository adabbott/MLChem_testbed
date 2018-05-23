import os

os.chdir("PES_data")

for i in range(1, 1057):
    os.system("~/Git/PESs/pypes-lib/PyPES_Library/code/pypes_lib.py {}/input.dat {}/output.dat".format(i, i))


