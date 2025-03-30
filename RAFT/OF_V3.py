import os
import sys

import math
import torch
import cv2
import numpy as np
import argparse
import glob
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

from RAFT.core.raft import RAFT 
from RAFT.core.update import *
from RAFT.core.utils.utils import InputPadder
from RAFT.core.utils import flow_viz
from torchvision.transforms import ToTensor
from collections import OrderedDict

import tensorrt as trt
from torch2trt import torch2trt

#DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_trt_model(args):
    # Load raw checkpoint
    state_dict = torch.load(args.model, map_location="cuda", weights_only=True)

    # Fix for DataParallel issue: Remove "module." prefix
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_key = k.replace("module.", "")  # Remove "module." prefix
        new_state_dict[new_key] = v

    # Initialize RAFT model
    model = RAFT(args)
    model.load_state_dict(new_state_dict)  # Load modified state dict
    model.eval().to("cuda")

    # Convert to TensorRT (FP16)
    dummy_input1 = torch.randn(1, 3, 320, 448).cuda()
    dummy_input2 = torch.randn(1, 3, 320, 448).cuda()
    dummy_input1 = torch.tensor(dummy_input1, dtype=torch.float32).cuda()
    dummy_input2 = torch.tensor(dummy_input2, dtype=torch.float32).cuda() 
    model_trt = torch2trt(model, [dummy_input1, dummy_input2], fp16_mode=True)

    return model_trt

def load_frame(frame):
    if len(frame.shape) == 2:  # Convert grayscale to 3-channel
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)

    img = torch.from_numpy(frame).permute(2, 0, 1).float()
    return img[None].to("cuda")


def compute_optical_flow(model_trt, frame1, frame2):
    frame1 = load_frame(frame1)  # Convert to tensor
    frame2 = load_frame(frame2)

    with torch.no_grad():
        flow_low, flow_up = model_trt(frame1, frame2, iters=5, test_mode=True)  # Lower iters for speed

    return flow_up[0].permute(1, 2, 0).cpu().numpy()

def process_video(args):
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)

    ret, prev_frame = cap.read()
    if not ret or prev_frame is None:
        print("Error: Could not read first frame.")
        return

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    model = load_trt_model(args)

    # Camera parameters (modify if needed)
    fx, fy = 335.97, 336.41
    depth = 5.0  # Assumed depth in meters
    sum_vx, sum_vy, count = 0.0, 0.0, 0
    scaleFactor, step = 2.0, 20  # Scale factor for drawing motion vectors

    #velocity_file = open("velocityRAFT.txt", "w")

    while True:
        ret, curr_frame = cap.read()
        if not ret:
            break

        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        flow = compute_optical_flow(model, prev_gray, curr_gray)

        if flow is None:  # Handle errors
            print("Skipping frame due to optical flow error.")
            prev_gray = curr_gray
            continue

        # Display frame
        display_frame = curr_frame.copy()

        for y in range(0, flow.shape[0], step):
            for x in range(0, flow.shape[1], step):
                dx, dy = flow[y, x]

                # Convert pixel motion to real-world motion
                Vx, Vy = (dx * depth * fps) / fx, (dy * depth * fps) / fy
                sum_vx += Vx
                sum_vy += Vy
                count += 1

                # Draw motion vectors
                start = (x * 2, y * 2)
                end = (x * 2 + int(dx * scaleFactor), y * 2 + int(dy * scaleFactor))
                cv2.arrowedLine(display_frame, start, end, (0, 0, 255), 1, cv2.LINE_AA)

        # Compute and log velocity
        velocity = math.sqrt((sum_vx / count) ** 2 + (sum_vy / count) ** 2) if count > 0 else 0.0
        #velocity_file.write(f"{velocity:.4f}\n")

        # Display speed text
        speed_text = f"Speed: {velocity:.2f} m/s"
        cv2.putText(display_frame, speed_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        # Resize & show the frame
        resized_frame = cv2.resize(display_frame, (800, 600))
        cv2.imshow("Optical Flow", resized_frame)

        prev_gray = curr_gray

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # velocity_file.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to RAFT model checkpoint")
    parser.add_argument("--video", type=str, required=True, help="Path to input video file")
    parser.add_argument("--small", action="store_true", help="Use small RAFT model")
    parser.add_argument("--mixed_precision", action="store_true", help="Use mixed precision")
    parser.add_argument("--alternate_corr", action="store_true", help="Use efficient correlation implementation")

    args = parser.parse_args()
    process_video(args)