import os
import cv2
import numpy as np

def convert_tif_to_png(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    tif_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.tif', '.tiff'))]

    for tif_file in tif_files:
        image_path = os.path.join(input_folder, tif_file)
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

        # Check the bit depth of the image
        if img.dtype == np.uint16:
            # Convert 16-bit to 8-bit
            img = (img / 256).astype(np.uint8)
        
        # Check if the image is grayscale
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
        # Ensure the image is in BGR format for PNG
        if img.shape[2] == 4:  # If it has an alpha channel
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        # Save as PNG
        output_path = os.path.join(output_folder, os.path.splitext(tif_file)[0] + '.png')
        cv2.imwrite(output_path, img)
        print(f"Converted {tif_file} to {output_path}")

# Example usage
input_folder = 'Datasets/3D_Photogrammetry_Data/GasterAmulet49D/Output/Scarab1'
output_folder = 'Datasets/3D_Photogrammetry_Data/GasterAmulet49D/pngs'
convert_tif_to_png(input_folder, output_folder)