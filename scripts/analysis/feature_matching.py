import cv2
import numpy as np
import matplotlib.pyplot as plt

def extract_features(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(img, None)
    return img, keypoints, descriptors

def match_features(descriptors1, descriptors2):
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
    return good_matches

def display_matches(img1, kp1, img2, kp2, matches):
    # Create a new output image that concatenates the two images
    rows1, cols1 = img1.shape
    rows2, cols2 = img2.shape
    out = np.zeros((max([rows1, rows2]), cols1 + cols2), dtype='uint8')
    out[:rows1, :cols1] = img1
    out[:rows2, cols1:cols1+cols2] = img2

    # Draw the matches
    for mat in matches:
        img1_idx = mat.queryIdx
        img2_idx = mat.trainIdx
        (x1, y1) = kp1[img1_idx].pt
        (x2, y2) = kp2[img2_idx].pt

        # Draw a line with circles on both ends
        cv2.line(out, (int(x1), int(y1)), (int(x2)+cols1, int(y2)), (0, 0, 0), 1)
        cv2.circle(out, (int(x1), int(y1)), 4, (0, 0, 0), 1)
        cv2.circle(out, (int(x2)+cols1, int(y2)), 4, (0, 0, 0), 1)

    # Display the image
    plt.figure(figsize=(12, 6))
    plt.imshow(out, cmap='gray')
    plt.axis('off')
    plt.tight_layout()
    plt.show()


# Usage
image_path1 = '/Users/abhishekbagde/Documents/Manchester/CS/Project/Datasets/3D_Photogrammetry_Data/wesley/3D190614_VFA61_0069.jpg'
image_path2 = '/Users/abhishekbagde/Documents/Manchester/CS/Project/Datasets/3D_Photogrammetry_Data/wesley/3D190614_VFA61_0151.jpg'

# Extract features from both images
img1, keypoints1, descriptors1 = extract_features(image_path1)
img2, keypoints2, descriptors2 = extract_features(image_path2)

# Match features
matches = match_features(descriptors1, descriptors2)

# Display matches
display_matches(img1, keypoints1, img2, keypoints2, matches[:50])  # Display top 50 matches

# Print some information about the extracted features and matches
print(f"Image 1: {len(keypoints1)} keypoints, Descriptor shape: {descriptors1.shape}")
print(f"Image 2: {len(keypoints2)} keypoints, Descriptor shape: {descriptors2.shape}")
print(f"Number of good matches: {len(matches)}")
