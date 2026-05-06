from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt

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

    # 1. Anscombe
    transformed = anscombe_transform(image)

    # 2. Clip to reasonable range (important)
    transformed = np.clip(transformed, 0, 50)

    # Scale to 0–255 WITHOUT destroying contrast
    transformed_8u = cv2.normalize(transformed, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # 3. NLM
    denoised_8u = cv2.fastNlMeansDenoising(
        transformed_8u,
        None,
        h=17,  # we'll tune this next
        templateWindowSize=7,
        searchWindowSize=21
    )

    # Back to float scale
    denoised = denoised_8u.astype(np.float32) / 255.0
    denoised = denoised * transformed.max()

    # 4. Inverse Anscombe
    result = inverse_anscombe(denoised)

    return np.clip(result, 0, 255).astype(np.uint8)

def plot_images(image1, image2, name=""):
    if isinstance(image1, str) and isinstance(image2, str):
        image1 = cv2.imread(image1, 0) 
        image2 = cv2.imread(image2,0)
        name = Path(image1)
    
    fig, ax = plt.subplots(1, 2, figsize=(10,5))

    ax[0].imshow(image1, cmap='gray')
    # ax[0].set_title(f"Median Filter Applied to: {name}")

    ax[1].imshow(image2, cmap='gray')
    # ax[1].set_title(f"Binarizatiom Applied to: {name}")
    plt.show()

def calculate_image_histogram(image1, image2, name=None):
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    ax[0].hist(image1.flatten(), bins=50, color='blue')
    # ax[0].set_title(f"Image {name} Histogram")

    ax[1].hist(image2.flatten(), bins=50, color='blue')
    # ax[1].set_title(f"Image 2 {name} Histogram")

    plt.show()
def get_all_images_hist(images_list):
    plt.figure(figsize=(8,5))
    plt.hist(images_list, bins=256)
    plt.title("Global Pixel Intensity Histogram")
    plt.xlabel("Pixel value (0–255)")
    plt.ylabel("Frequency")
    plt.show()

def variance_vs_intensity_from_pixels(pixels, filtered_image):
    bins = np.linspace(0, 255, 20)
    noise = pixels - filtered_image

    pixels_flat = pixels.ravel()
    noise_flat = noise.ravel()

    digitized = np.digitize(pixels_flat, bins)

    means, variances = [], []

    for i in range(1, len(bins)):
        values = noise_flat[digitized == i]
        if len(values) > 10:
            means.append(np.mean(pixels_flat[digitized == i]))
            variances.append(np.var(values))



    plt.scatter(means, variances, label="Data")
    m, b = np.polyfit(means, variances, 1)

# Create line values
    x_line = np.linspace(min(means), max(means), 100)
    y_line = m * x_line + b

# Plot the line
    plt.plot(x_line, y_line, color='red', label=f"Fit: y={m:.2f}x + {b:.2f}")

    plt.xlabel("Mean intensity")
    plt.ylabel("Variance")
    plt.legend()
    plt.show()

def variance_vs_intensity_from_image(pixels):
    bins = np.linspace(0, 255, 20)

    # Flatten once
    pixels_flat = pixels.ravel()

    # Bin by intensity
    digitized = np.digitize(pixels_flat, bins)

    means, variances = [], []

    for i in range(1, len(bins)):
        values = pixels_flat[digitized == i]

        if len(values) > 10:
            means.append(np.mean(values))    # mean intensity in bin
            variances.append(np.var(values))  # variance of intensity in bin
            # print(f"BIN NUMBER {i} MEAN {np.mean(values)} VARIANCE {np.var(values)}") 
            print(values)
            print("-" *20)

    # Plot
    plt.scatter(means, variances, label="Data")

    # Fit line
    m, b = np.polyfit(means, variances, 1)
    x_line = np.linspace(min(means), max(means), 100)
    y_line = m * x_line + b

    plt.plot(x_line, y_line, color='red', label=f"Fit: y={m:.2f}x + {b:.2f}")

    plt.xlabel("Mean intensity")
    plt.ylabel("Variance (signal + noise)")
    plt.legend()
    plt.show()   

def immerkaer_calc(image):
    def anscombe_transform(image):
        return 2.0 * np.sqrt(image + 3.0/8.0)
    def immerkaer_sigma(image):
        image = image.astype(np.float64)

        kernel = np.array([[1, -2, 1],
                        [-2, 4, -2],
                        [1, -2, 1]])

        filtered = cv2.filter2D(image, -1, kernel)

        sigma = np.sum(np.abs(filtered)) * np.sqrt(np.pi / 2) / (6 * (image.shape[0] - 2) * (image.shape[1] - 2))

        return sigma
    transformed = anscombe_transform(image)
    sigma = immerkaer_sigma(transformed)
    print(sigma)
    return sigma