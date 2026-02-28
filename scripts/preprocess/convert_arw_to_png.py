import os
import rawpy
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor

def convert_arw_to_png(input_file, output_file):
    with rawpy.imread(input_file) as raw:
        rgb = raw.postprocess()
    
    # Convert to BGR for OpenCV
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    
    # Move the image to GPU memory
    gpu_image = cv2.UMat(bgr)
    
    # Apply some GPU-accelerated operations (example: brightness adjustment)
    adjusted = cv2.convertScaleAbs(gpu_image, alpha=1.1, beta=10)
    
    # Move the result back to CPU memory
    result = adjusted.get()
    
    # Save as PNG
    cv2.imwrite(output_file, result)

def process_images(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    arw_files = [f for f in os.listdir(input_folder) if f.lower().endswith('.arw')]
    
    with ThreadPoolExecutor() as executor:
        futures = []
        for arw_file in arw_files:
            input_path = os.path.join(input_folder, arw_file)
            output_path = os.path.join(output_folder, os.path.splitext(arw_file)[0] + '.png')
            futures.append(executor.submit(convert_arw_to_png, input_path, output_path))
        
        for i, future in enumerate(futures):
            future.result()
            print(f"Processed image {i+1}/{len(arw_files)}")

if __name__ == "__main__":
    input_folder = 'Datasets/3D_Photogrammetry_Data/cuneiform930/June28th/RAWS'
    output_folder = 'Datasets/3D_Photogrammetry_Data/cuneiform930/pngs'
    process_images(input_folder, output_folder)