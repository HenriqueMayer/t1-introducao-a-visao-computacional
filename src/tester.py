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
        image_median_appl = utils.filter_img(image, "median")
        image_gaussian_appl = utils.filter_img(image, "gaussian")
        image_poisson_denois_appl = utils.poisson_denoise(image)

    
        image2 = utils.binarization(image, threshold= np.median(image))
        # image2= utils.filter_img(image, "median")
        # output_path = output_dir / image_dir.name
        # utils.save_image(image2, output_path)


        # image2 = utils.binarization(denoised, threshold= np.median(image))
        
        # comparing to mask available on the dataset
        
        # mask_path = mask_base / image_dir.parent.name / image_dir.name
        # if mask_path.exists():
            
        #     mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        #     if mask is not None:
        #         dice = utils.dice_coefficient(image2, mask)
        #         masks_comparison_dice.append(dice)
        #         print(f"Dice coefficient for {image_dir.name}: {dice:.4f}")
        # break

    # print(sum(masks_comparison_dice)/len(masks_comparison_dice))



main()
