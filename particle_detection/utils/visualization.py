import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

def resize_and_convert(image, target_shape, to_rgb=False):
    """
    Resizes an image to the target shape and optionally converts it to RGB.

    Args:
        image (np.ndarray): The input image as a NumPy array.
        target_shape (tuple[int, int]): The desired image dimensions (width, height).
        to_rgb (bool): Whether to convert the image to RGB. Defaults to False.

    Returns:
        np.ndarray: The resized (and optionally RGB-converted) image.
    """
    resized = cv2.resize(image, target_shape, interpolation=cv2.INTER_NEAREST)
    if to_rgb:
        return cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
    return resized

def get_contours(mask):
    """
    Extracts contours from a binary mask.

    Args:
        mask (np.ndarray): Binary mask from which to extract contours.

    Returns:
        list[np.ndarray]: A list of contours, where each contour is a NumPy array.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def filter_contours_by_area_and_shape(contours, min_area, max_area, min_circularity):
    """
    Filters contours based on area and circularity.

    Args:
        contours (list[np.ndarray]): List of contours to filter.
        min_area (float): Minimum area for a contour to be considered valid.
        max_area (float): Maximum area for a contour to be considered valid.
        min_circularity (float): Minimum circularity for a contour to be considered valid.

    Returns:
        list[np.ndarray]: List of contours that satisfy the area and circularity criteria.
    """
    valid_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area <= area <= max_area:
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                circularity = 4 * np.pi * (area / (perimeter ** 2))
                if circularity >= min_circularity:
                    valid_contours.append(contour)
    return valid_contours

def draw_and_display(image, contours, title=None, color=(0, 255, 0), thickness=2, save_path=None):
    """
    Draws contours on an image, displays the result, and optionally saves it.

    Args:
        image (np.ndarray): The input image on which to draw contours.
        contours (list[np.ndarray]): List of contours to draw.
        title (str, optional): Title of the visualization. Defaults to None.
        color (tuple[int, int, int]): Color of the drawn contours in BGR format. Defaults to (0, 255, 0).
        thickness (int): Thickness of the contour lines. Defaults to 2.
        save_path (str, optional): Path to save the visualized image. Defaults to None.

    Returns:
        None
    """
    image_copy = image.copy()
    cv2.drawContours(image_copy, contours, -1, color, thickness)
    
    plt.figure(figsize=(12, 12))
    plt.imshow(image_copy)
    plt.title(title if title is not None else "Visualization")
    plt.axis("off")
    plt.show()

    if save_path:
        if image_copy.dtype != np.uint8:
            image_copy = cv2.normalize(image_copy, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        cv2.imwrite(save_path, cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB))