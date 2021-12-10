import random
import time
import os
import cv2
import numpy as np
import torch
#import torch.multiprocessing as multiprocessing
import multiprocessing

SHUFFLE_SIZE = 600

BATCH_SIZE=64
"""
try:
    video_dir = "/playpen-nvme/tgreer/MannequinChallenge/train_512/"
    available_videos = os.listdir(video_dir)
    available_videos = [x for x in available_videos if ("mkv" in x) or ("mp4" in x)]
except:
    try:
        video_dir = "/playpen1/tgreer/MannequinChallenge/train_512/"
        available_videos = os.listdir(video_dir)
        available_videos = [x for x in available_videos if ("mkv" in x) or ("mp4" in x)]
    except:
        print("falling back to slow disk")
        video_dir = "/playpen-raid2/tgreer/Mannequin_Challenge/MannequinChallenge/train/"
        available_videos = os.listdir(video_dir)
        available_videos = [x for x in available_videos if ("mkv" in x) or ("mp4" in x)]
"""

import glob

video_dir = "/playpen1/tgreer/Kinetics/kinetics_256/*/*.mp4"
available_videos = glob.glob(video_dir)
def framePacket():       

    max_gap = 30
    images = torch.zeros((64 + max_gap + 1, 3, 120, 120), dtype=torch.uint8)
    # Create a VideoCapture object and read from input file
    fname = random.choice(available_videos)

    cap = cv2.VideoCapture(fname)

       
    # Check if camera opened successfully
    if (cap.isOpened()== False): 
      print("Error opening video  file")
       
    length = cap.get(cv2.CAP_PROP_FRAME_COUNT) 
    if length < 64 + max_gap + 1:
        return framePacket()
    cap.set(cv2.CAP_PROP_POS_FRAMES, random.randint(0, length - 64 - max_gap - 1))

    for i in range(64 + max_gap + 1):
        ret, frame = cap.read()
        if not ret:
            # try again.
            return framePacket()
        #if frame.shape != (512, 512, 3):
        frame = frame[:240:2, :240:2] #cv2.resize(frame, (120, 120))
        images[i] = torch.tensor(frame).permute((2, 0, 1))
    output_index = (torch.arange(1, 65) + torch.randint(max_gap, (64,))).long() 
    x = torch.cat([images[:64], images[output_index]], 1)
    # When everything done, release 
    # the video capture object
    cap.release()
    return x

def putFramePacketsOnQueue(q):
    while True:
        q.put(framePacket())
def grabShufflePutStep(inp, out):
        bigPacket = [inp.get() for _ in range(SHUFFLE_SIZE)]
        x = torch.cat(bigPacket, 0)
        indices = torch.randperm(SHUFFLE_SIZE * 64)
        for i in range(SHUFFLE_SIZE):
            out.put(x[indices[i * 64 : (i + 1) * 64]])

def grabShufflePut(inp, out):
    while True:
        grabShufflePutStep(inp, out)
def threadedProvide():
    
    packetQueue = multiprocessing.Queue(SHUFFLE_SIZE)
    packetProcesses = [None for _ in range(4)]
    for i in range(4):
        packetProcesses[i] = multiprocessing.Process(target=putFramePacketsOnQueue, args=(packetQueue,), daemon=True)
        packetProcesses[i].start()
    shuffledQueue = multiprocessing.Queue(SHUFFLE_SIZE // 6)
    #cached_batches = torch.load("videocache64.pth")
    cached_batches = torch.load("kinetics_cache64.pth")
    process = multiprocessing.Process(target = grabShufflePut, args = (packetQueue, shuffledQueue), daemon=True)
    process.start()
    for c in cached_batches:
        yield c
    while True:
        yield shuffledQueue.get().clone()


if __name__ == "__main__":

    x = time.perf_counter()
    threadedProvide()
    y = time.perf_counter()
    print(y - x)
   
