import optuna
import torch
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import logging

# Ensure src path is available for imports
sys.path.append(str(Path(__file__).parent.resolve()))

from train_unet import train_filter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def objective(trial):
    # Hyperparameters to optimize
    filter_type = trial.suggest_categorical("filter", ["none", "median", "gaussian", "bilateral", "blur"])
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
    img_size = trial.suggest_categorical("img_size", [128, 256])
    
    # keep epochs relatively low for optimization to save time
    epochs = 5 
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    logger.info(f"Trial {trial.number}: Testing filter={filter_type}, lr={lr:.6f}, batch_size={batch_size}, img_size={img_size}")
    
    try:
        metrics = train_filter(
            filter_type=filter_type,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            img_size=img_size,
            device=device
        )
        
        if metrics is None:
            return 0.0
            
        return metrics["dice"]
    except Exception as e:
        logger.error(f"Trial {trial.number} failed with error: {e}")
        return 0.0

def run_optimization(n_trials=20):
    logger.info("Starting Optuna study for UNet hyperparameters...")
    
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    
    logger.info("Optimization finished.")
    logger.info(f"Best Trial: {study.best_trial.number}")
    logger.info(f"Best Value (Dice): {study.best_value:.4f}")
    logger.info(f"Best Params: {study.best_params}")
    
    # Save study results to CSV
    df = study.trials_dataframe()
    df.to_csv("unet_optuna_optimization.csv", index=False)
    logger.info("Results saved to unet_optuna_optimization.csv")

if __name__ == "__main__":
    run_optimization(n_trials=15)
