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

def get_homography_matrix(camera):
    camera_position = camera.position()
    intrinsics = camera.intrinsics
    rotation_matrix = camera_position[0]
    translation_vector = camera_position[1]
    
    transform = np.hstack((rotation_matrix, np.transpose(translation_vector)))
    projective = np.matmul(intrinsics, transform);
    homography = projective[:, [0, 1, 3]];
    homography /= homography[2, 2];
    
    return homography

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

# def get_color_mask(image, hue_min, hue_max, sat_min):
#     """Create binary mask based on HSV color thresholds."""
#     image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
#     mask = (image_hsv[:,:,0] > hue_min) & (image_hsv[:,:,0] < hue_max) & (image_hsv[:,:,1] > sat_min)
#     return np.where(mask, 1, 0).astype(np.uint8)

def get_color_mask(image, color):
    """Create binary mask for specified color in HSV space."""
    image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    
    # Simple HSV ranges for each color
    if color == 'red':
        mask1 = (image_hsv[:,:,0] < 10) & (image_hsv[:,:,1] > 100)
        mask2 = (image_hsv[:,:,0] > 160) & (image_hsv[:,:,1] > 100)
        mask = mask1 | mask2
    
    elif color == 'green':
        mask = (image_hsv[:,:,0] > 35) & (image_hsv[:,:,0] < 85) & (image_hsv[:,:,1] > 100)
    
    elif color == 'blue':
        mask = (image_hsv[:,:,0] > 95) & (image_hsv[:,:,0] < 120) & (image_hsv[:,:,1] > 100)
    
    elif color == 'black':
        mask = (image_hsv[:,:,2] < 50)
    
    else:
        raise ValueError("Color must be 'red', 'green', 'blue', or 'black'")

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


def transform_points(points, h_camera):
    """Transform points using a homography matrix."""
    # Convert points list to numpy array
    points = np.array(points)

    # 3x3 matrix -> homogenus tranfromation (roation, scale, translation)
    # 1. Converte global/world coordiantes to homogenus [x, y, 1]
    # 2. Mulltiplay with homographic matrix -> Image coordinates!
    # [x']   [h11 h12 h13] [x]
    # [y'] = [h21 h22 h23] [y]
    # [w']   [h31 h32 h33] [1]
    # 3. Image coordiantes (homogenus) -> [x'/w', y'/w']

    # p_image = H @ P_world 
    # where:
    # p_image - point in image coordinates (where it appears in the picture)
    # P_world - point in world/global coordinates (real world position)
    # H - homography matrix (3x3) that maps from world to image
    
    # Create homogeneous coordinates
    p_world_homogeneous = np.zeros((len(points), 3))
    p_world_homogeneous[:, 0] = points[:, 0]
    p_world_homogeneous[:, 1] = points[:, 1]
    p_world_homogeneous[:, 2] = 1
    
    # Multiply by homography matrix
    p_image_homogeneous_t = h_camera @ p_world_homogeneous.T
    p_image_homogeneous = p_image_homogeneous_t.T
    
    # Convert back from homogeneous coordinates
    p_image_x = p_image_homogeneous[:, 0] / p_image_homogeneous[:, 2]
    p_image_y = p_image_homogeneous[:, 1] / p_image_homogeneous[:, 2]
    
    return p_image_x, p_image_y