def resize_and_convert(image, target_shape, to_rgb=False):
    resized = cv2.resize(image, target_shape, interpolation=cv2.INTER_NEAREST)
    if to_rgb:
        return cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
    return resized

def get_contours(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def filter_contours_by_area_and_shape(contours, min_area, max_area, min_circularity):
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
    image_copy = image.copy()
    cv2.drawContours(image_copy, contours, -1, color, thickness)
    
    plt.figure(figsize=(12, 12))
    plt.imshow(image_copy)
    plt.title(title if title is not None else "Visualization")
    plt.axis("off")
    plt.show()

    if save_path:
        # Ensure image is in 8-bit format before saving
        if image_copy.dtype != np.uint8:
            image_copy = cv2.normalize(image_copy, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        cv2.imwrite(save_path, cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB))
