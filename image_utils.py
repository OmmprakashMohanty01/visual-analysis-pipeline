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


def process_image_batch(
    image_batch, size=None, crop=False, normalize=False, save=True, output_dir="."
):
    import numpy as np

    if not isinstance(image_batch, (list, tuple)):
        raise ValueError("Image batch must be a list or tuple of image arrays.")

    processed_batch = []
    for image_array in image_batch:
        processed_image = process_image_array(
            image_array, size, crop, normalize=normalize
        )
        if save:
            cv2.imwrite(
                os.path.join(output_dir, f"processed_image_{len(processed_batch)}.jpg"),
                processed_image,
            )
        processed_batch.append(processed_image)

    return np.stack(processed_batch)


def process_image_array_with_mask(
    image_array, size=None, crop=False, normalize=False, mask=None
):
    if not isinstance(mask, (np.ndarray, type(None))):
        raise ValueError("Mask must be a numpy array or None.")

    processed_image = process_image_array(image_array, size, crop, normalize=normalize)

    if mask is not None:
        alpha_channel = cv2.split(mask)[0]
        processed_image = cv2.addWeighted(processed_image, 0.6, alpha_channel, 0.4, 0)

    return processed_image


def save_images_in_dir(
    images, output_dir=".", prefix="", extension=".jpg", overwrite=False
):
    for i, image in enumerate(images):
        filename = f"{prefix}{i}{extension}"
        filepath = os.path.join(output_dir, filename)
        if os.path.exists(filepath) and not overwrite:
            raise FileExistsError(f"File '{filepath}' already exists.")
        cv2.imwrite(filepath, image)


def generate_image_grid(images, rows, cols, save_path="grid.jpg", extension=".jpg"):
    for i in range(rows * cols):
        filename = f"input_{i}{extension}"
        cv2.imwrite(os.path.join(save_path, filename), images[i])

    grid_image = process_image_array_with_mask(
        np.zeros((rows * 128, cols * 128, 3)), size=((rows * 128, cols * 128))
    )
    for i in range(rows * cols):
        x = i % cols
        y = i // cols
        grid_image[y * 128 : y * 128 + 128, x * 128 : x * 128 + 128] = images[i]
    cv2.imwrite("grid.jpg", grid_image)


def resize_images(images, new_size):
    return [cv2.resize(image, new_size) for image in images]
