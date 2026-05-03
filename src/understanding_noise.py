import utils
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np


def understanding_noise():
    raw_img_list = []

    masks_comparison_dice = []
    datadir = Path("data/images/images")
    output_dir = datadir / "output"
    output_dir.mkdir(exist_ok=True)
    mask_base = Path("data/masks/masks")

    for image_dir in datadir.rglob("*.png"):
        if "output" in str(image_dir):
            continue
        image = utils.load_image(image_dir)
        if image is None:
            continue
        raw_img_list.append(image.astype(np.float64))

    all_pixels = np.concatenate(raw_img_list)
    utils.variance_vs_intensity_from_pixels(all_pixels)
    
    
  
understanding_noise()
  




