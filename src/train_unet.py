import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import pandas as pd
import numpy as np
import argparse
import sys
import os
from datetime import datetime

# Ensure src path is available
sys.path.append(str(Path(__file__).parent.resolve()))
from unet import UNet
from dataset import OilSpillDataset
import noise_filter

import random

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def train_filter(filter_type, epochs=3, batch_size=16, lr=1e-3, img_size=256, device="cuda", save_model=True, seed=None):
    if seed is not None:
        print(f"Setting seed to {seed} for training U-Net with filter: {filter_type.upper()}")
        set_seed(seed)
    else:
        print(f"Training U-Net for filter: {filter_type.upper()}")
    
    img_train_dir = Path("data/images/images/train")
    mask_train_dir = Path("data/masks/masks/train")
    img_val_dir = Path("data/images/images/val")
    mask_val_dir = Path("data/masks/masks/val")
    
    # Initialize datasets
    train_dataset = OilSpillDataset(img_train_dir, mask_train_dir, filter_type=filter_type, img_size=img_size)
    val_dataset = OilSpillDataset(img_val_dir, mask_val_dir, filter_type=filter_type, img_size=img_size)
    
    print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        print("Error: Empty dataset")
        return None
        
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    
    # Initialize model
    model = UNet(n_channels=1, n_classes=1, bilinear=True).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    best_dice = -1.0
    best_metrics = {}
    
    # Create output directory and unique model path with timestamp
    Path("models").mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if seed is not None:
        model_path = Path("models") / f"unet_{filter_type}_seed{seed}_{timestamp}.pth"
    else:
        model_path = Path("models") / f"unet_{filter_type}_{timestamp}.pth"
    
    # Early stopping config
    patience = 3
    epochs_no_improve = 0
    
    # Track training history
    history = []
    
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for batch_idx, (imgs, masks) in enumerate(train_loader):
            imgs = imgs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * imgs.size(0)
            if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == len(train_loader):
                print(f"Epoch [{epoch}/{epochs}] Batch [{batch_idx+1}/{len(train_loader)}] Loss: {loss.item():.4f}")
                
        train_loss /= len(train_dataset)
        
        # Validation
        model.eval()
        val_loss = 0.0
        dices = []
        ious = []
        mses = []
        
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs = imgs.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)
                
                outputs = model(imgs)
                loss = criterion(outputs, masks)
                val_loss += loss.item() * imgs.size(0)
                
                # Apply Sigmoid to logits
                preds = torch.sigmoid(outputs)
                
                # Move to CPU for metrics
                preds_np = preds.cpu().numpy()
                masks_np = masks.cpu().numpy()
                
                # Calculate metrics for each image in batch
                for pred, mask in zip(preds_np, masks_np):
                    pred_bin = (pred > 0.5).astype(np.uint8) * 255
                    mask_uint = (mask * 255).astype(np.uint8)
                    
                    dice = noise_filter.dice_coefficient(pred_bin, mask_uint)
                    mse = noise_filter.mean_squared_error(pred_bin, mask_uint)
                    iou = noise_filter.iou_score(pred_bin, mask_uint)
                    
                    dices.append(dice)
                    mses.append(mse)
                    ious.append(iou)
                    
        val_loss /= len(val_dataset)
        mean_dice = np.mean(dices)
        mean_iou = np.mean(ious)
        mean_mse = np.mean(mses)
        
        print(f"Epoch [{epoch}/{epochs}] - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Dice: {mean_dice:.4f} | Val IoU: {mean_iou:.4f} | Val MSE: {mean_mse:.4f}")
        
        # Record training history
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_dice": mean_dice,
            "val_iou": mean_iou,
            "val_mse": mean_mse
        })
        
        if mean_dice > best_dice:
            best_dice = mean_dice
            best_metrics = {
                "epoch": epoch,
                "val_loss": val_loss,
                "dice": mean_dice,
                "iou": mean_iou,
                "mse": mean_mse,
                "model_path": str(model_path)
            }
            if save_model:
                torch.save(model.state_dict(), model_path)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered at epoch {epoch} (no improvement in Dice score for {patience} epochs)")
                break
            
    # Save training history
    if history and save_model:
        df_hist = pd.DataFrame(history)
        if seed is not None:
            history_path = Path("models") / f"unet_{filter_type}_seed{seed}_{timestamp}_history.csv"
            stable_history_path = Path("models") / f"unet_{filter_type}_seed{seed}_history.csv"
        else:
            history_path = Path("models") / f"unet_{filter_type}_{timestamp}_history.csv"
            stable_history_path = Path("models") / f"unet_{filter_type}_history.csv"
        df_hist.to_csv(history_path, index=False)
        # Also save a stable version for easy loading in notebooks
        df_hist.to_csv(stable_history_path, index=False)
            
    if best_metrics:
        if save_model:
            print(f"Best model for {filter_type} saved to {best_metrics['model_path']} at epoch {best_metrics['epoch']} with Dice: {best_metrics['dice']:.4f}")
        else:
            print(f"Finished training for {filter_type} at epoch {best_metrics['epoch']} with Dice: {best_metrics['dice']:.4f}")
    else:
        print(f"Training finished, but no model was saved.")
    return best_metrics

def main():
    parser = argparse.ArgumentParser(description="Train U-Net for Oil Spill Semantic Segmentation")
    parser.add_argument("--filter", type=str, default="gaussian", choices=["all", "none", "median", "gaussian", "bilateral", "blur"], help="Filter to train on")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=0.00011013017365962953, help="Learning rate")
    parser.add_argument("--img-size", type=int, default=256, help="Image resize dimension")
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    filters_to_run = ["none", "median", "gaussian", "bilateral", "blur"] if args.filter == "all" else [args.filter]
    
    results = []
    for f in filters_to_run:
        metrics = train_filter(f, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, img_size=args.img_size, device=device)
        if metrics:
            results.append({
                "filter_type": f,
                "best_epoch": metrics["epoch"],
                "val_loss": metrics["val_loss"],
                "dice": metrics["dice"],
                "iou": metrics["iou"],
                "mse": metrics["mse"],
                "model_path": metrics.get("model_path", "")
            })
            
    # Save comparison results
    df = pd.DataFrame(results)
    output_file = "unet_results.csv"
    df.to_csv(output_file, index=False)
    print(f"\nTraining complete. Results saved to {output_file}")
    print(df)

if __name__ == "__main__":
    main()
