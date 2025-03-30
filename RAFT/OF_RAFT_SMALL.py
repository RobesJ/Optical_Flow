from torchvision.models.optical_flow import raft_small
import os
import sys

import math
import torch
import torch.onnx
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

import tensorrt as trt


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
'''
def load_model(args):
    model = torchvision.models.optical_flow.raft_small(weights=("pretrained", Raft_Small_Weights.C_T_V2)
    model.load_state_dict(torch.load(model, map_location=DEVICE))
    model = model.module
    model.to(DEVICE)
    model.eval()
    return model
'''

model = raft_small(pretrained=True)
model.to(DEVICE)
model.eval

def load_frame(frame):
    # Resize the frame so that both dimensions are divisible by 8
    h, w = frame.shape[:2]
    new_h = (h // 8) * 8  # Make height divisible by 8
    new_w = (w // 8) * 8  # Make width divisible by 8

    frame_resized = cv2.resize(frame, (new_w, new_h))  # Resize the frame
    
    # Check if the frame is grayscale (2D) and convert it to RGB (3D)
    if len(frame_resized.shape) == 2:  # Grayscale image
        frame_resized = np.expand_dims(frame_resized, axis=-1)  # Add the channel dimension
        frame_resized = np.repeat(frame_resized, 3, axis=-1)  # Convert grayscale to 3-channel (RGB)

    # Convert frame to torch tensor, permute dimensions, normalize, and add batch dimension
    img = torch.from_numpy(frame_resized).permute(2, 0, 1).float() / 255.0
    return img[None].to(DEVICE)  
'''
def compute_optical_flow(model, frame1, frame2):
    frame1 = load_frame(frame1)
    frame2 = load_frame(frame2)

    padder = InputPadder(frame1.shape)
    frame1, frame2 = padder.pad(frame1, frame2)
    with torch.cuda.amp.autocast():
        with torch.no_grad():
            flow_low, flow_up = model(frame1, frame2, iters=10, test_mode=True)
    return flow_up[0].permute(1, 2, 0).cpu().numpy()


def export_raft_to_onnx(args, onnx_path="raft_model.onnx"):
    model = load_model(args)  # Load RAFT model
    model.eval()

    # Ensure the model is on CUDA if available
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(DEVICE)

    # Create dummy inputs and move them to the same device
    dummy_input1 = torch.randn(1, 3, 384, 512, device=DEVICE)  # Move to CUDA or CPU
    dummy_input2 = torch.randn(1, 3, 384, 512, device=DEVICE)  # Move to CUDA or CPU

    # If mixed precision is enabled, convert model and inputs to FP16
    if args.mixed_precision:
        model.half()
        dummy_input1 = dummy_input1.half()
        dummy_input2 = dummy_input2.half()

    torch.onnx.export(
        model, 
        (dummy_input1, dummy_input2), 
        onnx_path, 
        opset_version=12, 
        input_names=["frame1", "frame2"], 
        output_names=["flow"], 
        dynamic_axes={"frame1": {0: "batch"}, "frame2": {0: "batch"}, "flow": {0: "batch"}}
    )

    print(f"Model exported to {onnx_path}")
'''
'''
def compute_optical_flow(model, frame1, frame2):
    frame1 = load_frame(frame1)
    frame2 = load_frame(frame2)
    
    with torch.no_grad():
        flow = model(frame1, frame2)
    return flow[0].permute(1, 2, 0).cpu().numpy()
'''
def compute_optical_flow(model, frame1, frame2):
    frame1 = load_frame(frame1)
    frame2 = load_frame(frame2)

    with torch.no_grad():
        flow = model(frame1, frame2)
    
    # Print the shape of the flow tensor to debug
    #print("Flow shape:", len(flow))

    # If flow is a 3D tensor, permute it
    if flow.shape[1] == 2:
        flow_up = flow[0]
        return flow_up.permute(1, 2, 0).cpu().numpy()
    else:
        # If flow is 2D, just return it as is
        return None


def process_video(args):
    cap = cv2.VideoCapture(args.video)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    ret, prev_frame = cap.read()
    if not ret:
        print("Error: Could not read first frame.")
        return

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    fx, fy = 335.970824, 336.411475
    cx, cy = 313.369979, 201.104536
    depth = 5.0 
    fps = cap.get(cv2.CAP_PROP_FPS)
    delta_t = 1.0 / fps 

    velocity_file = open("velocityRAFT.txt", "w")

    while True:
        ret, curr_frame = cap.read()
        if not ret:
            break

        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        flow = compute_optical_flow(model, prev_gray, curr_gray)

        if flow is None:
            continue  # Skip this frame if flow is not valid

        display_frame = curr_frame.copy()

        sum_vx, sum_vy = 0.0, 0.0
        count = 0
        scaleFactor = 2.0  # For visualization
        step = 20  # Step size for drawing motion vectors

        # Ensure flow has the correct shape (number of rows and columns)
        flow_height, flow_width, _ = flow.shape  # Now flow is a 3D array [H, W, 2]
        
        for y in range(0, flow_height, step):
            for x in range(0, flow_width, step):
                dx, dy = flow[y, x]  # Access the flow in x and y directions
                
                # Convert pixel motion to real-world motion
                Vx = (dx * depth * fps) / fx
                Vy = (dy * depth * fps) / fy

                sum_vx += Vx
                sum_vy += Vy
                count += 1

                # Draw motion vectors
                start = (x * 2, y * 2)
                end = (x * 2 + int(dx * scaleFactor), y * 2 + int(dy * scaleFactor))
                cv2.arrowedLine(display_frame, start, end, (0, 0, 255), 1, cv2.LINE_AA)

        # Compute average velocity
        if count > 0:
            avg_vx = sum_vx / count
            avg_vy = sum_vy / count
            velocity = math.sqrt(avg_vx ** 2 + avg_vy ** 2)
        else:
            velocity = 0.0

        velocity_file.write(f"{velocity:.4f}\n")

        # Display velocity on screen
        speed_text = f"Speed: {velocity:.2f} m/s"
        cv2.putText(display_frame, speed_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        resized_frame = cv2.resize(display_frame, (800,600))
        cv2.imshow("Optical Flow", resized_frame)

        prev_gray = curr_gray

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    velocity_file.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #parser.add_argument("--model", type=str, required=True, help="Path to RAFT model checkpoint")
    parser.add_argument("--video", type=str, required=True, help="Path to input video file")
    #parser.add_argument("--output", type=str, default="output.avi", help="Path to save output video")
    #parser.add_argument("--small", action="store_true", help="Use small RAFT model")
    #parser.add_argument("--mixed_precision", action="store_true", help="Use mixed precision")
    #parser.add_argument("--alternate_corr", action="store_true", help="Use efficient correlation implementation")
    
    args = parser.parse_args()
    #export_raft_to_onnx(args)
    process_video(args)