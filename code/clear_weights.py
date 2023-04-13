import os, shutil
import glob

cwd = os.path.dirname(os.path.realpath(__file__))

d_m = os.path.join(cwd, "weights", "discriminator_monet", "*")
d_n = os.path.join(cwd, "weights", "discriminator_nature", "*")
g_m = os.path.join(cwd, "weights", "generator_monet", "*")
g_n = os.path.join(cwd, "weights", "generator_nature", "*")

paths = [d_m, d_n, g_m, g_n]


exclude = ["12-04-2023-23-33", "12-04-2023-23-00", "13-04-2023-00-07"]

exclude_file_name = ["weights-"+e for e in exclude]

print(exclude_file_name)

for p in paths:
    for f in glob.glob(p):
        f_split = f.split("/")
        if f_split[-1] in exclude_file_name:
            continue
        else:
            os.remove(f)