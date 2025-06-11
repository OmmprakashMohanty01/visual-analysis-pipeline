import os
import logging

logger = logging.getLogger(__name__)


def process_image_array(
    image_array, size=None, crop=False, x1=0, y1=0, x2=None, y2=None, normalize=False
):
    import cv2
    import numpy as np

    if size is not None:
        if len(size) != 2:
            raise ValueError("Size must be a tuple of two integers.")
        size = (size[1], size[0])
        if isinstance(image_array, str):
            image = cv2.imread(image_array)
        else:
            image = image_array
        image = cv2.resize(image, size)

    if crop:
        if x2 is None:
            x2 = image.shape[1]
        if y2 is None:
            y2 = image.shape[0]
        image = image[y1:y2, x1:x2]

    if isinstance(image_array, str):
        if normalize:
            image = cv2.normalize(
                image,
                None,
                alpha=0,
                beta=1,
                norm_type=cv2.NORM_MINMAX,
                dtype=cv2.CV_32F,
            )
    elif isinstance(image_array, np.ndarray) and normalize:
        image = cv2.normalize(
            image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F
        )

    image = image / 255.0

    if isinstance(image_array, str):
        cv2.imwrite("processed_image.jpg", image)
    else:
        return image
