import torch
import torchvision.transforms as T
import cv2
import numpy as np
from torchvision.models.optical_flow import raft_small
from torchvision.utils import flow_to_image

# Video processing function
def process_video(video_path, device):
    # Open the video
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    ret, frame1 = cap.read()
    if not ret or frame1 is None:
        print("Error: Could not read first frame.")
        return

    # Get video info
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"FPS of the video: {fps}")

    # Preprocessing transformations
    transforms = T.Compose([
        T.ToTensor(),  
        T.ConvertImageDtype(torch.float32),
        T.Normalize(mean=0.5, std=0.5),  
        T.Resize(size=(240, 424)),  
    ])

    # Load the RAFT model (small variant)
    try:
        model = raft_small(weights="Raft_Small_Weights.DEFAULT", progress=False).to(device)
        model.eval()
    except Exception as e:
        print(f"Error loading RAFT model: {e}")
        return

    prev_frame_rgb = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)

    while True:
        # Read the next frame
        ret, frame2 = cap.read()
        if not ret:
            print("End of video.")
            break
        
        frame2_rgb = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)

        # Convert frames to tensors
        frame1_tensor = transforms(prev_frame_rgb).unsqueeze(0).to(device)
        frame2_tensor = transforms(frame2_rgb).unsqueeze(0).to(device)

        # Calculate optical flow using RAFT model
        with torch.no_grad():
            try:
                flow_list = model(frame1_tensor, frame2_tensor)
                predicted_flow = flow_list[-1]  
                flow_img = flow_to_image(predicted_flow)
                flow_img = flow_img.squeeze(0).permute(1, 2, 0).cpu().numpy()
            except Exception as e:
                print(f"Error in model inference: {e}")
                break

        # Convert flow image to BGR for OpenCV display
        flow_img = cv2.cvtColor(flow_img, cv2.COLOR_RGB2BGR)

        # Show optical flow output
        cv2.imshow("Optical Flow", flow_img)

        # Exit condition
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting.")
            break

        # Update previous frame
        prev_frame_rgb = frame2_rgb

    # Release the video capture object and close OpenCV windows
    cap.release()
    cv2.destroyAllWindows()


# Initialize device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Path to your video
video_path = "/home/rootroot/cpp_test/gray_output_1.mkv"

# Process the video
process_video(video_path, device)