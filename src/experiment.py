import optuna
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import noise_filter
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data(img_dir: Path, mask_dir: Path):
    images = []
    masks = []
    names = []
    img_paths = sorted(list(img_dir.glob("*.png")))
    
    for img_path in img_paths:
        mask_path = mask_dir / img_path.name
        if mask_path.exists():
            img = noise_filter.load_image(img_path)
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if img is not None and mask is not None:
                if img.shape != mask.shape:
                    mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
                images.append(img)
                masks.append(mask)
                names.append(img_path.name)
    
    return images, masks, names

def objective(trial, images, masks):
    filter_type = trial.suggest_categorical("filter_type", ["none", "median", "gaussian", "bilateral", "blur"])
    threshold = trial.suggest_int("threshold", 0, 254)
    inverse = trial.suggest_categorical("inverse", [True, False])
    
    mses = []
    for img, mask in zip(images, masks):
        filtered = noise_filter.filter_img(img, filter_type)
        binarized = noise_filter.binarization(filtered, threshold, inverse=inverse)
        mse = noise_filter.mean_squared_error(binarized, mask)
        mses.append(mse)
    
    return np.mean(mses)

def run_experiment():
    img_train_dir = Path("data/images/images/train")
    mask_train_dir = Path("data/masks/masks/train")
    output_csv = Path("experiment_results.csv")
    
    logger.info("Loading data...")
    images, masks, names = load_data(img_train_dir, mask_train_dir)
    
    if not images:
        logger.error("No images found. Please check the data paths.")
        return

    logger.info(f"Loaded {len(images)} image-mask pairs.")

    filters_to_test = ["none", "median", "gaussian", "bilateral", "blur"]
    all_results = []

    for f_type in filters_to_test:
        logger.info(f"Optimizing threshold for filter: {f_type}")
        study = optuna.create_study(direction="minimize")
        study.optimize(lambda trial: objective_fixed_filter(trial, images, masks, f_type), n_trials=30)
        
        best_threshold = study.best_trial.params["threshold"]
        best_inverse = study.best_trial.params["inverse"]
        logger.info(f"Best params for {f_type}: threshold={best_threshold}, inverse={best_inverse}")

        for img, mask, name in zip(images, masks, names):
            filtered = noise_filter.filter_img(img, f_type)
            binarized = noise_filter.binarization(filtered, best_threshold, inverse=best_inverse)
            
            mse = noise_filter.mean_squared_error(binarized, mask)
            dice = noise_filter.dice_coefficient(binarized, mask)
            iou = noise_filter.iou_score(binarized, mask)
            
            all_results.append({
                "image_name": name,
                "filter_type": f_type,
                "threshold": best_threshold,
                "inverse": best_inverse,
                "mse": mse,
                "dice": dice,
                "iou": iou
            })

    df = pd.DataFrame(all_results)
    df.to_csv(output_csv, index=False)
    logger.info(f"Results saved to {output_csv}")

    summary = df.groupby("filter_type")[["mse", "dice", "iou"]].agg(["mean", "std"])
    print("\nSummary Statistics by Filter Type:")
    print(summary)

def objective_fixed_filter(trial, images, masks, filter_type):
    threshold = trial.suggest_int("threshold", 0, 254)
    inverse = trial.suggest_categorical("inverse", [True, False])
    mses = []
    for img, mask in zip(images, masks):
        filtered = noise_filter.filter_img(img, filter_type)
        binarized = noise_filter.binarization(filtered, threshold, inverse=inverse)
        mse = noise_filter.mean_squared_error(binarized, mask)
        mses.append(mse)
    return np.mean(mses)

if __name__ == "__main__":
    run_experiment()