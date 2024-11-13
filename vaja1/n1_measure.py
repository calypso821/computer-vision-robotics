import matplotlib.pyplot as plt
import cv2
import glob
import os

# Path to your images
folder_path = "resources/vaja1/n1"

# Get sorted list of image files
image_files = sorted(glob.glob(os.path.join(folder_path, "frame_*.jpg")))

# Load and display each image
for i, img_path in enumerate(image_files):
    # Read image using cv2
    img = cv2.imread(img_path)
    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Create new figure for each image
    plt.figure(figsize=(10, 8))
    plt.imshow(img)
    plt.title(f'Frame {i:03d}')
    
plt.show()