import cv2
import matplotlib.pyplot as plt

def read_and_display_image(image_path):
    # Read image using OpenCV
    # OpenCV reads images in BGR format
    image = cv2.imread(image_path)
    print(image.shape)
    
    if image is None:
        raise ValueError("Image could not be read. Check if the path is correct.")
    
    # Convert BGR to RGB for correct color display in Matplotlib
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Numpy
    # 1. Flip vertically (order of rows 1. dimension)
    # image[::-1]
    # 2. Flip horizontall (order of colums 2. dimension)
    # image[:, ::-1] 
    # :,  <-- leave dimension as it is 
    # 3. Order element components (3. dimension) - BGR -> RGB
    # image[:, :, ::-1]
    # :, :,  <-- leave 1., 2. (rows, columns) as it is
    # Also works: 
    # image[:, :, 2::-1]
    # image[:, :, [2,1,0]]
    img_rgb_numpy = image[:, :, ::-1]
    
    # Create figure and display image
    plt.figure(figsize=(10, 8))
    plt.imshow(img_rgb_numpy)
    plt.title('Umbrellas')
    plt.show()

# Example usage
image_path = "resources/umbrellas.jpg"
read_and_display_image(image_path)