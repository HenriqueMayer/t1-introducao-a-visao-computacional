import utils
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np
import noise_filter


def plotting_hist():
    raw_img_list = []
    img_bilateral_applied_list= []
    img_blur_applied_list = []
    img_gaussian_applied_list= []
    img_poisson_denoiser_applied_list = []

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
      
        # image_median_appl = utils.filter_img(image, "median")
        # image_gaussian_appl = utils.filter_img(image, "gaussian")
        image_bilateral_appl = noise_filter.filter_img(image, "bilateral")
        image_blur_appl = noise_filter.filter_img(image, "blur")
        # image_poisson_denois_appl = utils.poisson_denoise(image)

        # raw_img_list.append(image.flatten())
        img_bilateral_applied_list.append(image_bilateral_appl.flatten())
        img_blur_applied_list.append(image_blur_appl.flatten())
    
    utils.get_all_images_hist(np.concatenate(img_bilateral_applied_list))
    utils.get_all_images_hist(np.concatenate(img_blur_applied_list))


    
  


plotting_hist()
