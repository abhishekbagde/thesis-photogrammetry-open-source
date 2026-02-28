import cv2
import numpy as np
import os

def resize_image(input_path, output_path, target_width, quality=95):
    # Read the image
    img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
    
    # Calculate new height maintaining aspect ratio
    aspect_ratio = img.shape[1] / img.shape[0]
    target_height = int(target_width / aspect_ratio)
    
    # Check if the image has an alpha channel
    if img.shape[2] == 4:
        # Split the image into color channels and alpha channel
        bgr = img[:,:,0:3]
        alpha = img[:,:,3]
        
        # Resize color channels
        resized_bgr = cv2.resize(bgr, (target_width, target_height), interpolation=cv2.INTER_LANCZOS4)
        
        # Resize alpha channel
        resized_alpha = cv2.resize(alpha, (target_width, target_height), interpolation=cv2.INTER_LANCZOS4)
        
        # Merge the resized color and alpha channels
        resized_img = cv2.merge((resized_bgr, resized_alpha))
    else:
        # Resize the image without alpha channel
        resized_img = cv2.resize(img, (target_width, target_height), interpolation=cv2.INTER_LANCZOS4)
    
    # Save the resized image with specified quality
    cv2.imwrite(output_path, resized_img, [cv2.IMWRITE_PNG_COMPRESSION, 9])

def process_directory(input_dir, output_dir, target_width, quality=95):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    for filename in os.listdir(input_dir):
        if filename.lower().endswith('.png'):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            
            # Resize the image
            resize_image(input_path, output_path, target_width, quality)
            
            # Get the size of the output file
            output_size = os.path.getsize(output_path) / (1024 * 1024)  # Size in MB
            
            print(f"Resized {filename} to width {target_width}. Output size: {output_size:.2f} MB")

# Usage
input_directory = 'Datasets/3D_Photogrammetry_Data/cuneiform930/pngs'
output_directory = 'Datasets/3D_Photogrammetry_Data/cuneiform930/reduced_pngs'
target_width = 3000  # Increased from 2000 to 3000 for better quality
quality = 95  # High quality setting

process_directory(input_directory, output_directory, target_width, quality)