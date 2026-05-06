import utils
import noise_filter
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np

def getting_dataset_var(datadir_, data_dir_name):
        raw_img_list = []
        raw_raw_img_list = []
        datadir = Path(datadir_)
        output_dir = datadir / "output"
        output_dir.mkdir(exist_ok=True)

        count = 0
        for image_dir in datadir.rglob("*.png"):
            if "output" in str(image_dir):
                continue

            image = utils.load_image(image_dir)
            if image is None:
                continue
            
            raw_img = image.flatten()
            image = image.astype(np.float64) / 255.0

            raw_img_list.append(image.ravel())
            raw_raw_img_list.append(raw_img.astype(np.float64))
            count += 1

        print("Images used:", count)

        all_pixels = np.concatenate(raw_img_list)
        print(data_dir_name, np.var(all_pixels)* 255)

        return raw_img_list, raw_raw_img_list

def understanding_noise():
    raw_img_list = []
    img_median_applied_list= []
    img_gaussian_applied_list= []
    img_bilateral_applied_list = []
    img_blur_applied_list =[]
    img_poisson_applied_list=[]

            # img_blur_applied_list.append(noise_filter.filter_img(image, "blur"))
            # img_poisson_applied_list.append(utils.poisson_denoise(image))

 

    #utils.variance_vs_intensity_from_image(np.concatenate(raw_img_list))
    # utils.variance_vs_intensity_from_pixels(np.concatenate(raw_img_list), np.concatenate(img_poisson_applied_list))
    # utils.variance_vs_intensity_from_pixels(np.concatenate(raw_img_list), np.concatenate(img_gaussian_applied_list))
    # utils.variance_vs_intensity_from_pixels(np.concatenate(raw_img_list), np.concatenate(img_bilateral_applied_list))
    # utils.variance_vs_intensity_from_pixels(np.concatenate(raw_img_list), np.concatenate(img_blur_applied_list))


    parsar_list, raw_parsar_list = getting_dataset_var(datadir_= "data_segraggated/train/Palsar",data_dir_name ="Palsar")
    sentinel_list, raw_sentinel_list = getting_dataset_var(datadir_= "data_segraggated/train/Sentinel",data_dir_name ="Sentinel")
    # concatenated = np.concatenate(parsar_list + sentinel_list)
    # overall_variance = np.var(concatenated)
    # print("Total Variance",overall_variance * 255)
    # utils.variance_vs_intensity_from_image(np.concatenate(raw_parsar_list))
    # utils.variance_vs_intensity_from_image(np.concatenate(raw_sentinel_list))
    # utils.get_all_images_hist(np.concatenate(raw_parsar_list))
    utils.get_all_images_hist(np.concatenate(raw_sentinel_list))

  
understanding_noise()
  




