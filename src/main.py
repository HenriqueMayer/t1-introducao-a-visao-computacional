import noise_filter
from pathlib import Path
import cv2

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
        image = noise_filter.binarization(image, 120)
        output_path = output_dir / image_dir.name
        noise_filter.save_image(image, output_path)
        
        

        
        # comparing to mask available on the dataset
        mask_path = mask_base / image_dir.parent.name / image_dir.name
        if mask_path.exists():
            
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                dice = noise_filter.dice_coefficient(image, mask)
                masks_comparison_dice.append(dice)
                print(f"Dice coefficient for {image_dir.name}: {dice:.4f}")
            else:
                print(f"Could not load mask for {image_dir.name}")
        else:
            print(f"No mask found for {image_dir.name}")

    print(f"AVERAGE DICE {sum(masks_comparison_dice)/len(masks_comparison_dice)}")


if __name__ == "__main__":
    main()