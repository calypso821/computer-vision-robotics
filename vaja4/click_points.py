import cv2
import numpy as np
import numpy as np
import matplotlib.pyplot as plt

from common_func import (
    load_h
)


A_camera = cv2.imread('vaja4/material/camera1.jpg')
A_camera = cv2.cvtColor(A_camera, cv2.COLOR_BGR2RGB)
h_camera = load_h("vaja4/material/camera1.txt")

h_camera_inv = np.linalg.inv(h_camera)

# Now the plotting code
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

# Show camera image on left (Camera Space)
ax1.imshow(A_camera)
ax1.set_title('Camera Space (Image)')

# Create right plot (World/Global Space)
ax2.set_ylim(0, 400)
ax2.set_xlim(-200, 200)
ax2.grid(True)
ax2.set_title('World/Global Space')

print("Click points in Camera Image (left), press Enter when done")
clicked_points = np.array(plt.ginput(-1, timeout=-1))

# Convert clicked points to homogeneous coordinates
clicked_homogeneous = np.zeros((len(clicked_points), 3))  # Create empty array
clicked_homogeneous[:, 0] = clicked_points[:, 0]  # Copy x coordinates
clicked_homogeneous[:, 1] = clicked_points[:, 1]  # Copy y coordinates
clicked_homogeneous[:, 2] = 1  # Set last column to ones

# Transform from Camera to World using h_camera_inv
world_points_homogeneous = (h_camera_inv @ clicked_homogeneous.T).T
print(h_camera_inv)
world_points = world_points_homogeneous[:, :2] / world_points_homogeneous[:, 2].reshape(-1, 1)

# Plot points
ax1.plot(clicked_points[:, 0], clicked_points[:, 1], 'r+', markersize=10, label='Camera Points')
ax2.plot(world_points[:, 1], world_points[:, 0], 'r+', markersize=10, label='World Points')

ax1.legend()
ax2.legend()
plt.show()