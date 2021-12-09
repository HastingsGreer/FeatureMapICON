import tqdm
import subprocess
import os

in_dir = "/playpen-nvme/tgreer/MannequinChallenge/train/"
out_dir = "/playpen-nvme/tgreer/MannequinChallenge/train_512/"

vnames = os.listdir(in_dir)

for vname in vnames:
    subprocess.run(["ffmpeg", "-i", in_dir + vname, "-s", "512x512", out_dir + vname])
    
