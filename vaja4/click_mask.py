import cv2
import numpy as np
import numpy as np
import matplotlib.pyplot as plt

from skimage import draw

from common_func import (
    poly2mask,
    apply_mask
)

A_camera = cv2.imread('vaja4/material/camera2.jpg')
A_camera = cv2.cvtColor(A_camera, cv2.COLOR_BGR2RGB)

# Create the figure and all subplots first
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
plt.tight_layout()

# Plot the camera image in the first subplot
ax1.imshow(A_camera)
ax1.set_title('Camera Space (Image)')

print("Click points in Camera Image (left), press Enter when done")
clicked_points = np.array(plt.ginput(-1, timeout=-1))

# Plot the points on the first subplot to keep them visible
if len(clicked_points) > 0:
    ax1.plot(clicked_points[:, 0], clicked_points[:, 1], 'r+')  # Red dots and lines

# Create and display mask
mask = poly2mask(
    y_coords=clicked_points[:, 1],
    x_coords=clicked_points[:, 0],
    shape=(A_camera.shape[0], A_camera.shape[1])
)

# Plot mask and applied mask
ax2.imshow(mask, cmap='gray')
ax2.set_title("Mask")

ax3.imshow(apply_mask(A_camera, mask))
ax3.set_title("Applied mask")

plt.tight_layout()
plt.show()