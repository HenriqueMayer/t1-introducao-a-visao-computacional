import noise_filter
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np
import numpy as np

def adaptive_threshold(image):
    adaptive_threshold_mean = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 6)
    adaptive_threshold_gaussian = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 9, 6)


    plt.figure(figsize = (10,10))
    titles = ["Adaptive Threshold - Mean", "Adaptive Threshold - Gaussian"]
    for idx, thres in enumerate([adaptive_threshold_mean, adaptive_threshold_gaussian]):
        plt.subplot(1,2,idx+1)
        plt.imshow(thres, 'gray')
        plt.title(titles[idx])
        plt.xticks([]), plt.yticks([])
                
    plt.show()
   
        
def anscombe_transform(img):
    return 2.0 * np.sqrt(img + 3.0/8.0)

def inverse_anscombe(y):
    return (y / 2.0)**2 - 3.0/8.0

def poisson_denoise(image):
    image = image.astype(np.float32)

    # Step 1: stabilize variance
    transformed = anscombe_transform(image)

    # Step 2: denoise (Non-Local Means)
    denoised = cv2.fastNlMeansDenoising(
        transformed.astype(np.uint8),
        None,
        h=1
    )

    # Step 3: inverse transform
    result = inverse_anscombe(denoised.astype(np.float32))

    return np.clip(result, 0, 255).astype(np.uint8)

def plot_images(image1, image2, name=""):
    if isinstance(image1, str) and isinstance(image2, str):
        image1 = cv2.imread(image1, 0) 
        image2 = cv2.imread(image2,0)
        name = Path(image1)
    
    fig, ax = plt.subplots(1, 2, figsize=(10,5))

    ax[0].imshow(image1, cmap='gray')
    ax[0].set_title(f"Median Filter Applied to: {name}")

    ax[1].imshow(image2, cmap='gray')
    ax[1].set_title(f"Binarizatiom Applied to: {name}")
    plt.show()

def calculate_image_histogram(image1, image2, name=None):
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    ax[0].hist(image1.flatten(), bins=50, color='blue')
    ax[0].set_title(f"Image {name} Histogram")

    ax[1].hist(image2.flatten(), bins=50, color='blue')
    ax[1].set_title(f"Image 2 {name} Histogram")

    plt.show()


def main():
    masks_comparison_dice = []
    datadir = Path("data/images/images")
    output_dir = datadir / "output"
    output_dir.mkdir(exist_ok=True)
    mask_base = Path("data/masks/masks")
    
    for image_dir in datadir.rglob("*.png"):
        if "output" in str(image_dir):
            continue
        image = noise_filter.load_image(image_dir)
        if image is None:
            continue
        image = noise_filter.filter_img(image, "median")
        
        # I defined this arbitrarily, maybe we should try other values
        image2 = noise_filter.binarization(image, threshold= np.median(image))
        output_path = output_dir / image_dir.name
        noise_filter.save_image(image2, output_path)

        plot_images(image1=image, image2=image2, name=image_dir.name)
        calculate_image_histogram(image1=image, image2=image2, name=image_dir.name)
        
        denoised = poisson_denoise(image)
        plot_images(image1=image, image2=denoised, name=image_dir.name)

        image2 = noise_filter.binarization(denoised, threshold= np.median(image))
        plot_images(image1=image, image2=image2, name=image_dir.name)
        
        break

        # comparing to mask available on the dataset
        mask_path = mask_base / image_dir.parent.name / image_dir.name
        if mask_path.exists():
            
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                dice = noise_filter.dice_coefficient(image2, mask)
                masks_comparison_dice.append(dice)
                print(f"Dice coefficient for {image_dir.name}: {dice:.4f}")
        break

    # print(sum(masks_comparison_dice)/len(masks_comparison_dice))



main()
