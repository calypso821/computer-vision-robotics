import cv2
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from skimage import draw
from matplotlib.patches import Polygon

from common_func import (
    poly2mask,
    get_components_info,
    get_color_mask,
    clean_binary_mask
)

I_camera2 = cv2.imread("vaja4/material/camera2.jpg")
I_camera2 = cv2.cvtColor(I_camera2, cv2.COLOR_BGR2RGB)

# Hue [0, 359] (circle) opencv -> [0, 179] 8bit optimization
# Low saturation - white colors
blue_comp_mask = get_color_mask(I_camera2, 95, 120, 150)

# First close to fill gaps, then open to remove noise
blue_comp_clean = clean_binary_mask(blue_comp_mask, 3)

# Create the figure and all subplots first
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
plt.tight_layout()

# Plot the camera image in the first subplot
ax1.imshow(I_camera2)
ax1.set_title('Camera Space (Image)')

print("Click points in Camera Image (left), press Enter when done")
clicked_points = np.array(plt.ginput(-1, timeout=-1))

# Plot the points on the first subplot to keep them visible
if len(clicked_points) > 0:
    ax1.plot(clicked_points[:, 0], clicked_points[:, 1], 'r+')  # Red dots

# Create and display mask
mask = poly2mask(
    y_coords=clicked_points[:, 1],
    x_coords=clicked_points[:, 0],
    shape=(I_camera2.shape[0], I_camera2.shape[1])
)

combined_mask = mask & blue_comp_clean
components = get_components_info(combined_mask)

# Plot mask and applied mask
ax2.imshow(combined_mask, cmap='gray')
ax2.set_title("Mask")

ax3.imshow(I_camera2)
ax3.set_title("Centroids")

for comp in components:
    centroid = comp['centroid']
    box = comp['bounding_box']
    ax3.scatter(centroid[0], centroid[1], color='red', s=3)
    polygon = Polygon(box, fill=False, color='red', linewidth=1)
    ax3.add_patch(polygon)

plt.tight_layout()
plt.show()