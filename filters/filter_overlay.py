import cv2
import numpy as np
from filters.face_detector import get_face_landmarks

def load_filter(image_path):
    """
    Load an image with transparency (RGBA). Convert to BGRA if missing alpha channel.
    """
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    
    # Convert BGR to BGRA if missing alpha channel
    if img.shape[-1] == 3:  
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)

    return img

# Load filter images
filter_images = {
    "hat": load_filter("assets/hat.png"),
    "glasses": load_filter("assets/glasses.png"),
    "mask": load_filter("assets/mask.png"),
    "bald": load_filter("assets/bald.png"),
    "full_face": load_filter("assets/full_face.png"),
}

def overlay_filter(image, filter_img, face_landmarks, points, scale_factor=1.0, y_offset=0, x_offset=0):
    """
    Overlay filter on face using alpha blending, ensuring correct dimensions.
    """
    (x1, y1), (x2, y2) = face_landmarks[points[0]], face_landmarks[points[1]]
    width = int((x2 - x1) * scale_factor)  # Scale width
    height = int(width * filter_img.shape[0] / filter_img.shape[1])  # Maintain aspect ratio

    # Resize the filter image
    resized_filter = cv2.resize(filter_img, (width, height), interpolation=cv2.INTER_AREA)

    # Adjust position with offsets
    y1 = max(0, y1 - height // 2 + y_offset)
    y2 = min(image.shape[0], y1 + height)
    x1 = max(0, x1 + x_offset)  # Apply x_offset for left/right adjustments
    x2 = min(image.shape[1], x1 + width)

    # Ensure dimensions match before blending
    region = image[y1:y2, x1:x2]
    region_height, region_width = region.shape[:2]
    resized_filter = resized_filter[:region_height, :region_width]  # Crop to match

    # Extract the filter and mask (alpha channel)
    filter_rgb = resized_filter[:, :, :3]  # First 3 channels: BGR
    alpha_mask = resized_filter[:, :, 3] / 255.0  # Normalize alpha (0-1)

    # Blend the images using the alpha mask
    for c in range(3):
        region[:, :, c] = (1 - alpha_mask) * region[:, :, c] + alpha_mask * filter_rgb[:, :, c]

    # Apply the blended image back to the main image
    image[y1:y2, x1:x2] = region


def apply_filters(image, filter_type):
    """
    Apply the selected filter based on user input.
    """
    landmarks_list, faces = get_face_landmarks(image)

    for landmarks in landmarks_list:
        if filter_type == "hat":
            overlay_filter(image, filter_images["hat"], landmarks, [19, 24], scale_factor=2.9, y_offset=-100, x_offset=-200)
        elif filter_type == "glasses":
            overlay_filter(image, filter_images["glasses"], landmarks, [36, 45], scale_factor=1.6, y_offset=20, x_offset=-50)  # Glasses over eyes
        elif filter_type == "mask":
            overlay_filter(image, filter_images["mask"], landmarks, [3, 13], scale_factor=1.3, y_offset=30, x_offset=-50)
        elif filter_type == "bald":
            overlay_filter(image, filter_images["bald"], landmarks, [19, 24], scale_factor=3.5, y_offset=-60, x_offset=-220)
        elif filter_type == "full_face":
            overlay_filter(image, filter_images["full_face"], landmarks, [36, 45], scale_factor=1.3, y_offset=30, x_offset=-50)

    return image
