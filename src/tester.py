import utils
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np


def main():
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
    



    # print(sum(masks_comparison_dice)/len(masks_comparison_dice))



main()
