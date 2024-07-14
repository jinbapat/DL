import cv2
import numpy as np
from matplotlib import pyplot as plt
def process_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    equalized = cv2.equalizeHist(img)
    _, thresholded = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)
    edges = cv2.Canny(img, 100, 200)
    flipped = cv2.flip(img, 1)
    kernel = np.ones((5,5), np.uint8)
    morphed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

    images = [img, equalized, thresholded, edges, flipped, morphed]
    titles = ["Original", "Equalized", "Thresholded", "Edges", "Flipped", "Morphed"]

    plt.figure(figsize=(15,10))
    for i in range(6):
        plt.subplot(2,3,i+1)
        plt.imshow(images[i], cmap='gray')
        plt.title(titles[i])
        plt.axis('off')
    plt.tight_layout()
    plt.show()
process_image("IMG_4131.JPG")
