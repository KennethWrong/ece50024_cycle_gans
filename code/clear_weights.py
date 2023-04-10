import os, shutil
import glob

cwd = os.path.dirname(os.path.realpath(__file__))

d_m = os.path.join(cwd, "weights", "discriminator_monet", "*")
d_n = os.path.join(cwd, "weights", "discriminator_nature", "*")
g_m = os.path.join(cwd, "weights", "generator_monet", "*")
g_n = os.path.join(cwd, "weights", "generator_nature", "*")

paths = [d_m, d_n, g_m, g_n]


for p in paths:
    for f in glob.glob(p):
        os.remove(f)