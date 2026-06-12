import utils
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np
import noise_filter


def main():
    masks_comparison_dice = []
    path = Path("data_segraggated/train/Palsar/palsar_10.png")
    path2 = Path("data_segraggated/train/Palsar/palsar_3094.png")

    image1 = utils.load_image(path)
    image2 = utils.load_image(path2)
 
    utils.plot_images(image2, noise_filter.filter_img(image2, "bilateral"))



    # print(sum(masks_comparison_dice)/len(masks_comparison_dice))



main()
