import optuna
import torch
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import logging
import argparse

# Ensure src path is available for imports
sys.path.append(str(Path(__file__).parent.resolve()))

from train_unet import train_filter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def objective_for_filter(trial, filter_type, opt_epochs, device):
    # Suggest hyperparameters
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
    img_size = trial.suggest_categorical("img_size", [128, 256])
    
    logger.info(f"[Trial {trial.number}] Testing filter={filter_type}, lr={lr:.6f}, batch_size={batch_size}, img_size={img_size}")
    
    try:
        # We do NOT save checkpoints/history files during search trials
        metrics = train_filter(
            filter_type=filter_type,
            epochs=opt_epochs,
            batch_size=batch_size,
            lr=lr,
            img_size=img_size,
            device=device,
            save_model=False
        )
        
        if metrics is None or "mse" not in metrics:
            return 0.0
            
        return metrics["mse"]
    except Exception as e:
        logger.error(f"[Trial {trial.number}] Failed for filter {filter_type} with error: {e}")
        return 0.0

def run_experiment(n_trials=15, opt_epochs=5, final_epochs=20):
    filters = ["none", "median", "gaussian", "bilateral", "blur"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    best_configs = {}
    
    # Step 1: Optimize hyperparameters per filter
    for filter_type in filters:
        logger.info(f"STARTING OPTIMIZATION FOR FILTER: {filter_type.upper()}")
        
        study = optuna.create_study(direction="minimize")
        study.optimize(
            lambda trial: objective_for_filter(trial, filter_type, opt_epochs, device),
            n_trials=n_trials
        )
        
        logger.info(f"Optimization finished for {filter_type}.")
        logger.info(f"Best Trial: {study.best_trial.number}")
        logger.info(f"Best Value (MSE): {study.best_value:.4f}")
        logger.info(f"Best Params: {study.best_params}")
        
        # Save study results to CSV specifically for this filter
        df_study = study.trials_dataframe()
        opt_csv = f"unet_optuna_optimization_{filter_type}.csv"
        df_study.to_csv(opt_csv, index=False)
        logger.info(f"Saved trials to {opt_csv}")
        
        best_configs[filter_type] = study.best_params
        
    # Step 2: Train & Test (evaluate) using the optimal hyperparameters per filter across 4 seeds
    logger.info(f"TRAINING FINAL MODELS WITH OPTIMAL HYPERPARAMETERS (4 SEEDS)")
    
    seeds = [42, 1903, 2003, 8]
    final_results = []
    
    for filter_type, params in best_configs.items():
        lr = params["lr"]
        batch_size = params["batch_size"]
        img_size = params["img_size"]
        
        logger.info(f"\nTraining final models for filter: {filter_type.upper()} with lr={lr:.6f}, batch_size={batch_size}, img_size={img_size}")
        
        for seed in seeds:
            logger.info(f"Running Seed {seed}...")
            
            # Train fully with optimal hyperparameters and specific seed, save_model=True
            metrics = train_filter(
                filter_type=filter_type,
                epochs=final_epochs,
                batch_size=batch_size,
                lr=lr,
                img_size=img_size,
                device=device,
                save_model=True,
                seed=seed
            )
            
            if metrics:
                result_row = {
                    "filter_type": filter_type,
                    "seed": seed,
                    "best_lr": lr,
                    "best_batch_size": batch_size,
                    "best_img_size": img_size,
                    "best_epoch": metrics["epoch"],
                    "val_loss": metrics["val_loss"],
                    "dice": metrics["dice"],
                    "iou": metrics["iou"],
                    "mse": metrics["mse"],
                    "model_path": metrics.get("model_path", "")
                }
                final_results.append(result_row)
            else:
                logger.error(f"Failed to train final model for filter {filter_type} with seed {seed}")
            
    # Save detailed final results to CSV
    df_results = pd.DataFrame(final_results)
    output_file = "unet_optimal_results.csv"
    df_results.to_csv(output_file, index=False)
    logger.info(f"\nAll final models trained. Detailed results saved to {output_file}")
    
    # Calculate summary statistics (mean +/- std) per filter
    if not df_results.empty:
        summary_stats = []
        for filter_type in filters:
            df_filter = df_results[df_results["filter_type"] == filter_type]
            if not df_filter.empty:
                summary_stats.append({
                    "filter_type": filter_type,
                    "best_lr": df_filter.iloc[0]["best_lr"],
                    "best_batch_size": df_filter.iloc[0]["best_batch_size"],
                    "best_img_size": df_filter.iloc[0]["best_img_size"],
                    "dice_mean": df_filter["dice"].mean(),
                    "dice_std": df_filter["dice"].std(),
                    "iou_mean": df_filter["iou"].mean(),
                    "iou_std": df_filter["iou"].std(),
                    "mse_mean": df_filter["mse"].mean(),
                    "mse_std": df_filter["mse"].std()
                })
        df_summary = pd.DataFrame(summary_stats)
        df_summary.to_csv("unet_optimal_summary.csv", index=False)
        logger.info("Summary statistics saved to unet_optimal_summary.csv")
        
        # Print summary comparison table
        print("SUMMARY STATISTICS BY FILTER (MEAN +/- STD OVER 4 SEEDS)")
        df_print = df_summary.copy()
        df_print["Dice"] = df_print.apply(lambda r: f"{r['dice_mean']:.4f} ± {r['dice_std']:.4f}", axis=1)
        df_print["IoU"] = df_print.apply(lambda r: f"{r['iou_mean']:.4f} ± {r['iou_std']:.4f}", axis=1)
        df_print["MSE"] = df_print.apply(lambda r: f"{r['mse_mean']:.4f} ± {r['mse_std']:.4f}", axis=1)
        df_print = df_print[["filter_type", "best_lr", "best_batch_size", "best_img_size", "Dice", "IoU", "MSE"]]
        
        try:
            print(df_print.to_markdown(index=False))
        except Exception:
            print(df_print)
        print("="*80)
        
        # Print detailed per-seed runs table
        print("\n" + "="*80)
        print("DETAILED PER-SEED RUNS")
        print("="*80)
        try:
            print(df_results[["filter_type", "seed", "dice", "iou", "mse", "best_epoch"]].to_markdown(index=False))
        except Exception:
            print(df_results[["filter_type", "seed", "dice", "iou", "mse", "best_epoch"]])
        print("="*80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimize U-Net hyperparameters per filter and run final evaluation")
    parser.add_argument("--trials", type=int, default=15, help="Number of Optuna trials per filter")
    parser.add_argument("--opt-epochs", type=int, default=5, help="Number of training epochs per trial during optimization")
    parser.add_argument("--final-epochs", type=int, default=20, help="Number of training epochs for final optimal training")
    args = parser.parse_args()
    
    run_experiment(n_trials=args.trials, opt_epochs=args.opt_epochs, final_epochs=args.final_epochs)
