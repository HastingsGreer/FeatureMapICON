import subprocess
import os

for f in os.listdir("."):
    if ".txt" in f:
        print(f)
        with open(f, "r") as ff:
            lines = ff.readlines()
        url = lines[0][:-1]
        subprocess.call(["youtube-dl", "-o", f.split(".")[0], url]) 

