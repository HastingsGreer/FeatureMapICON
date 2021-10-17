import footsteps
import os
import glob
def DavisEval(model, name):
    davis_path = "data_storage/DAVIS/"
    output_path = footsteps.output_dir + name + "/"

    for sequence in glob.glob(davis_path + "/*"):
        print(sequence)


