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


def compare_images_by_entropy(image1, image2):
    hist1 = cv2.calcHist([image1], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([image2], [0], None, [256], [0, 256])

    entropy1 = 0
    for i in range(len(hist1)):
        p1 = hist1[i] / (image1.shape[0] * image1.shape[1])
        if p1 > 0:
            entropy1 -= p1 * np.log2(p1)

    entropy2 = 0
    for i in range(len(hist2)):
        p2 = hist2[i] / (image2.shape[0] * image2.shape[1])
        if p2 > 0:
            entropy2 -= p2 * np.log2(p2)

    return abs(entropy1 - entropy2)


def compare_images_by_morphology(image1, image2):
    erosion1 = cv2.erode(image1, cv2.getStructElement(cv2.MORPH_ELLIPSE, (3, 3)))
    dilation1 = cv2.dilate(image1, cv2.getStructElement(cv2.MORPH_ELLIPSE, (3, 3)))
    erosion2 = cv2.erode(image2, cv2.getStructElement(cv2.MORPH_ELLIPSE, (3, 3)))
    dilation2 = cv2.dilate(image2, cv2.getStructElement(cv2.MORPH_ELLIPSE, (3, 3)))

    morph_diff1 = np.sum(np.abs(image1 - erosion1)) + np.sum(np.abs(image1 - dilation1))
    morph_diff2 = np.sum(np.abs(image2 - erosion2)) + np.sum(np.abs(image2 - dilation2))

    return morph_diff1 / (image1.shape[0] * image1.shape[1]) + morph_diff2 / (
        image2.shape[0] * image2.shape[1]
    )


def compare_images_by_edge_intensity(image1, image2):
    edges1 = cv2.Canny(image1, 50, 150)
    edges2 = cv2.Canny(image2, 50, 150)
    edge_diff = np.sum(np.abs(edges1 - edges2)) / (
        image1.shape[0] * image1.shape[1]
    ) + np.sum(np.abs(edges2 - edges1)) / (image2.shape[0] * image2.shape[1])
    return edge_diff


def compare_images_by_color_histogram(image1, image2):
    hist1 = cv2.calcHist([image1], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([image2], [0], None, [256], [0, 256])
    color_diff = np.sum(np.abs(hist1 - hist2)) / max(np.sum(hist1), np.sum(hist2))
    return color_diff


def compare_images_by_textural_feature(image1, image2):
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    feature1 = cv2.describeImage(gray1)
    feature2 = cv2.describeImage(gray2)
    textural_diff = np.sum(np.abs(feature1 - feature2)) / (
        feature1.shape[0] * feature1.shape[1]
    )
    return textural_diff
