import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import draw
from matplotlib.patches import Polygon

def load_h(txt_file_path):
    """Load camera homografic matrix"""
    with open(txt_file_path, 'r') as f:
        l = [[float(num) for num in line.split(',')] for line in f]    
    return np.array(l)


def poly2mask(y_coords, x_coords, shape):
    """Create binary mask from polygon coordinates."""
    fill_row_coords, fill_col_coords = draw.polygon(y_coords, x_coords, shape)
    mask = np.zeros(shape, dtype=np.bool)
    mask[fill_row_coords, fill_col_coords] = True
    return mask

def apply_mask(image, mask):
    """Apply binary mask to RGB image."""
    mask = mask.astype("uint8")
    mask_3d = np.zeros_like(image)
    for i in range(3):
        mask_3d[:,:,i] = mask
    return image * mask_3d

def get_color_mask(image, hue_min, hue_max, sat_min):
    """Create binary mask based on HSV color thresholds."""
    image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    mask = (image_hsv[:,:,0] > hue_min) & (image_hsv[:,:,0] < hue_max) & (image_hsv[:,:,1] > sat_min)
    return np.where(mask, 1, 0).astype(np.uint8)

def clean_binary_mask(mask, kernel_size=3):
    """Clean binary mask using morphological operations (close then open)."""
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    # First close to fill gaps
    closed = cv2.erode(cv2.dilate(mask, kernel), kernel)
    # Then open to remove noise
    cleaned = cv2.dilate(cv2.erode(closed, kernel), kernel)
    return cleaned

def get_components_info(binary_mask):
    """Get connected components information including labels, centroids, and bounding boxes."""
    num_labels, labels = cv2.connectedComponents(binary_mask)
    components = []
    
    for label in range(1, num_labels):  # Skip background (label 0)
        mask = (labels == label)
        y, x = np.nonzero(mask)
        
        # Calculate centroid
        centroid = (np.mean(x), np.mean(y))
        
        # Get rotated bounding box
        points = np.column_stack((x, y))
        rect = cv2.minAreaRect(points.astype(np.float32))
        box = cv2.boxPoints(rect)
        
        components.append({
            'label': label,
            'centroid': centroid,
            'bounding_box': box
        })
    
    return components