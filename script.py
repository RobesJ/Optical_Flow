import matplotlib.pyplot as plt
import cv2


video_path = "RAFT/output_video.avi"
cap = cv2.VideoCapture(video_path)

fps = cap.get(cv2.CAP_PROP_FPS)
print(f"FPS: {fps}")

cap.release()


# List to store velocity values
velocities = []
velocities2 = []
velocitiesRAFT = []

# Read the velocity data from the file
with open("velocity.txt", "r") as file:
    for line in file:
        try:
            velocity = float(line.strip())  # Convert the line to a float
            velocities.append(velocity)
        except ValueError:
            continue  # Skip invalid lines

# Generate x-axis values (assuming velocity is recorded per frame)

with open("velocity1.txt", "r") as file:
    for line in file:
        try:
            velocity = float(line.strip())  # Convert the line to a float
            velocities2.append(velocity)
        except ValueError:
            continue  # Skip invalid lines

with open("RAFT/velocityRAFT.txt", "r") as file:
    for line in file:
        try:
            velocity = float(line.strip())  # Convert the line to a float
            velocitiesRAFT.append(velocity)
        except ValueError:
            continue  # Skip invalid lines

frames = list(range(1, len(velocitiesRAFT) + 1))

# Plot the velocity data
plt.figure(figsize=(10, 6))
plt.plot(frames, velocities, marker='o', linestyle='-', color='b', label="Velocity (m/s)")
plt.plot(frames, velocities2, marker='o', linestyle='-', color='r', label="Velocity2 (m/s)")
plt.plot(frames, velocitiesRAFT, marker='o', linestyle='-', color='g', label="VelocityRAFT (m/s)")
plt.xlabel("Frame Number")
plt.ylabel("Velocity (m/s)")
plt.title("Drone Velocity Over Time")
plt.legend()
plt.grid(True)
plt.tight_layout()


# Show the plot
plt.show()