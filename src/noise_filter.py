from pathlib import Path
import cv2
import numpy as np

def load_image(img_path: Path):
    img = cv2.imread(str(img_path))
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def filter_img(src_img, algorithm: str):
    if algorithm == "median":
        return cv2.medianBlur(src_img, 5)
    elif algorithm == "gaussian":
        return cv2.GaussianBlur(src_img, (5, 5), 0)
    else:
        return src_img

def binarization(src_img, threshold: int):
    _, thresh = cv2.threshold(src_img, threshold, 255, cv2.THRESH_BINARY)
    return thresh

def save_image(src_img, output_path: Path):
    cv2.imwrite(str(output_path), src_img)

def dice_coefficient(img1, img2):
    img1 = img1.astype(np.float32) / 255.0
    img2 = img2.astype(np.float32) / 255.0
    intersection = np.sum(img1 * img2)
    union = np.sum(img1) + np.sum(img2)
    if union == 0:
        return 1.0
    return 2 * intersection / union