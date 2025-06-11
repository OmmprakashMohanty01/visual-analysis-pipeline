import math
import logging

logger = logging.getLogger(__name__)


import cv2
import numpy as np


def compare_images(image1, image2):
    # Convert images to grayscale
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # Calculate the structural similarity
    ssim = cv2.SSIM(gray1, gray2)

    # Calculate the pixel difference in terms of mean absolute difference
    mad = np.mean(np.abs(gray1 - gray2))

    return ssim, mad


# Example usage
image1 = cv2.imread("image1.jpg")
image2 = cv2.imread("image2.jpg")

ssim, mad = compare_images(image1, image2)
print("Structural Similarity: ", ssim)
print("Mean Absolute Difference: ", mad)
