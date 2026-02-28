import cv2
import numpy as np
import matplotlib.pyplot as plt

def extract_features(image_path):
    # Read image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Initialize SIFT detector
    sift = cv2.SIFT_create()

    # Find keypoints and descriptors
    keypoints, descriptors = sift.detectAndCompute(img, None)

    # Draw keypoints on the image
    img_keypoints = cv2.drawKeypoints(img, keypoints, None)

    return img, keypoints, descriptors, img_keypoints

def display_keypoints(img_keypoints, title):
    plt.figure(figsize=(10, 8))
    plt.imshow(cv2.cvtColor(img_keypoints, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')
    plt.show()


# Usage
img1_path = '/Users/abhishekbagde/Documents/Manchester/CS/Project/Datasets/3D_Photogrammetry_Data/wesley/3D190614_VFA61_0069.jpg'
img2_path = '/Users/abhishekbagde/Documents/Manchester/CS/Project/Datasets/3D_Photogrammetry_Data/wesley/3D190614_VFA61_0151.jpg'

# Extract features from both images
img1, kp1, des1, img_kp1 = extract_features(img1_path)
img2, kp2, des2, img_kp2 = extract_features(img2_path)

# Display keypoints for both images
display_keypoints(img_kp1, 'Keypoints in Image 1')
display_keypoints(img_kp2, 'Keypoints in Image 2')

# Print information about extracted features
print(f"Image 1: {len(kp1)} keypoints, Descriptor shape: {des1.shape}")
print(f"Image 2: {len(kp2)} keypoints, Descriptor shape: {des2.shape}")