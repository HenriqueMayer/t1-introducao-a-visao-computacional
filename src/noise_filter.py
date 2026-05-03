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
    elif algorithm == "bilateral":
        return cv2.bilateralFilter(src_img, 9, 75, 75)
    elif algorithm == "blur":
        return cv2.blur(src_img, (5, 5))
    else:
        return src_img

def binarization(src_img, threshold: int, inverse=False):
    mode = cv2.THRESH_BINARY_INV if inverse else cv2.THRESH_BINARY
    _, thresh = cv2.threshold(src_img, threshold, 255, mode)
    return thresh

def save_image(src_img, output_path: Path):
    cv2.imwrite(str(output_path), src_img)

def _normalize(img):
    img = img.astype(np.float32)
    if img.max() > 1.0:
        img /= 255.0
    return img

def dice_coefficient(img1, img2):
    img1 = _normalize(img1)
    img2 = _normalize(img2)
    intersection = np.sum(img1 * img2)
    union = np.sum(img1) + np.sum(img2)
    if union == 0:
        return 1.0
    return 2 * intersection / union

def mean_squared_error(img1, img2):
    img1 = _normalize(img1)
    img2 = _normalize(img2)
    return np.mean((img1 - img2) ** 2)

def iou_score(img1, img2):
    img1 = (img1 > 0).astype(np.uint8)
    img2 = (img2 > 0).astype(np.uint8)
    intersection = np.logical_and(img1, img2).sum()
    union = np.logical_or(img1, img2).sum()
    if union == 0:
        return 1.0
    return intersection / union