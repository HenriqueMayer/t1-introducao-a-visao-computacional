import utils
import noise_filter
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np


def understanding_noise():
    raw_img_list = []
    img_median_applied_list= []
    img_gaussian_applied_list= []
    img_bilateral_applied_list = []
    img_blur_applied_list =[]

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
        img_median_applied_list.append(noise_filter.filter_img(image, "median"))
        img_gaussian_applied_list.append(noise_filter.filter_img(image, "gaussian"))
        img_bilateral_applied_list.append(noise_filter.filter_img(image, "bilateral"))
        img_blur_applied_list.append(noise_filter.filter_img(image, "blur"))

    
     
    
    utils.immerkaer_calc(np.concatenate(raw_img_list))

    #utils.variance_vs_intensity_from_image(np.concatenate(raw_img_list))
    # utils.variance_vs_intensity_from_pixels(np.concatenate(raw_img_list), np.concatenate(img_median_applied_list))
    # utils.variance_vs_intensity_from_pixels(np.concatenate(raw_img_list), np.concatenate(img_gaussian_applied_list))
    # utils.variance_vs_intensity_from_pixels(np.concatenate(raw_img_list), np.concatenate(img_bilateral_applied_list))
    # utils.variance_vs_intensity_from_pixels(np.concatenate(raw_img_list), np.concatenate(img_blur_applied_list))


    
    
  
understanding_noise()
  




