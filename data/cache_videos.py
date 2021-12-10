import fmapicon.threaded_video_dataset
import torch

gen = fmapicon.threaded_video_dataset.threadedProvide()

res = [next(gen) for _ in range(512)]

torch.save(res, "kinetics_cache64.pth")
