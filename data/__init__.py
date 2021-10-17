import subprocess
import footsteps
import os
import glob
from PIL import Image
import numpy as np


def DavisEval(model, name):
    davis_path = "data_storage/DAVIS/"
    output_path = footsteps.output_dir + name + "/"
    os.mkdir(output_path)
    with open(davis_path + "/ImageSets/2017/val.txt", "r") as f:
        sequences = f.readlines()
    for sequence in sequences:
        sequence = sequence[:-1]  # strip newline
        sequence_out_path = output_path + sequence + "/"
        sequence_img_path = davis_path + "JPEGImages/480p/" + sequence + "/"
        with open(davis_path + "Annotations/480p/" + sequence + "/00000.png", "rb") as handle:
          first_annotation = np.array(
            Image.open(handle)
          )

        os.mkdir(output_path + sequence)
        with open(sequence_img_path + "00000.jpg", "rb") as handle:
          first_image = np.array(Image.open(handle))

        prev_image = first_image
        prev_annotation = first_annotation

        for i in range(1, len(os.listdir(sequence_img_path))):
            with open(sequence_img_path + f"{i:05}.jpg", "rb") as handle:
              curr_image = np.array(Image.open(handle))
            annotation = model(
                first_image, first_annotation, prev_image, prev_annotation, curr_image
            )
            Image.fromarray(annotation).save(sequence_out_path + f"{i:05}.png")

            prev_image = curr_image
            prev_annoration = first_annotation

    subprocess.run(
        [
            "python",
            "davis2017-evaluation/evaluation_method.py",
            "--davis_path",
            "data_storage/DAVIS",
            "--results_path",
            output_path,
            "--task",
            "semi-supervised",
        ]
    )
