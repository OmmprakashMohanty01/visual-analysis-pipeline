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


def compare_images_by_size(image1, image2):
    return abs(image1.shape[0] - image2.shape[0]) + abs(
        image1.shape[1] - image2.shape[1]
    )


def compare_images_by_size_difference(image1, image2):
    return abs(image1.shape[0] - image2.shape[0]) * abs(
        image1.shape[1] - image2.shape[1]
    )


def compare_images_by_aspect_ratio(image1, image2):
    return abs(image1.shape[1] / image1.shape[0] - image2.shape[1] / image2.shape[0])


def compare_images_by_color_dominance(image1, image2):
    # Convert images to HSV color space
    hsv1 = cv2.cvtColor(image1, cv2.COLOR_BGR2HSV)
    hsv2 = cv2.cvtColor(image2, cv2.COLOR_BGR2HSV)

    # Calculate color dominance
    h1 = np.mean(hsv1[:, :, 0])
    s1 = np.mean(hsv1[:, :, 1])
    v1 = np.mean(hsv1[:, :, 2])
    h2 = np.mean(hsv2[:, :, 0])
    s2 = np.mean(hsv2[:, :, 1])
    v2 = np.mean(hsv2[:, :, 2])

    dominance1 = max(h1, s1, v1)
    dominance2 = max(h2, s2, v2)

    return abs(dominance1 - dominance2)


def compare_images_by_texture(image1, image2):
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    texture_diff = np.mean(np.abs(cv2.blur(gray1, (5, 5)) - cv2.blur(gray2, (5, 5))))

    return texture_diff


def compare_images_by_edge_detection(image1, image2):
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    edge1 = cv2.Canny(gray1, 100, 200)
    edge2 = cv2.Canny(gray2, 100, 200)

    edge_diff = np.mean(np.abs(edge1 - edge2))

    return edge_diff
