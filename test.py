# load libraries
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
from supervision import Detections
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# download model
model_path = hf_hub_download(repo_id="arnabdhar/YOLOv8-Face-Detection", filename="model.pt")

# load model
model = YOLO(model_path)

# inference
image_path1 = "the-rock.jpg"
image_path2 = "operating-room.jpg"
image_path3 = "operating-room2.jpg"
image_path4 = "operating-room3.jpg"
output = model(Image.open(image_path4))
results = Detections.from_ultralytics(output[0])

# Convert image to numpy array for visualization
img_np = output[0].orig_img  # Original image as a numpy array

# Fix color format by converting BGR to RGB
img_np_rgb = img_np[..., ::-1]  # Flip the last channel

# Create a plot to visualize the image and detections
fig, ax = plt.subplots(1)
ax.imshow(img_np_rgb)  # Use the corrected RGB image

# Loop through detected faces and draw bounding boxes
for box in results.xyxy:
    x1, y1, x2, y2 = box[:4]  # Coordinates of the bounding box
    width = x2 - x1
    height = y2 - y1
    rect = patches.Rectangle((x1, y1), width, height, linewidth=2, edgecolor='r', facecolor='none')
    ax.add_patch(rect)

# Show the image with bounding boxes
plt.axis('off')
plt.show()
