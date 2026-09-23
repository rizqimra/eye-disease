import optuna
import torch
import os
import pandas as pd
from train import main as train_main, parse_args

def objective(trial):
    # Suggest hyperparameters
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
    
    # Generator hyperparameters
    lr_g = trial.suggest_float('lr_g', 1e-5, 1e-2)
    optimizer_g = trial.suggest_categorical('optimizer_g', ['adam', 'sgd'])
    beta1_g = trial.suggest_float('beta1_g', 0.0, 0.99)
    beta2_g = trial.suggest_float('beta2_g', 0.5, 0.999)

    # Discriminator hyperparameters
    lr_d = trial.suggest_float('lr_d', 1e-5, 1e-2)
    optimizer_d = trial.suggest_categorical('optimizer_d', ['adam', 'sgd'])
    beta1_d = trial.suggest_float('beta1_d', 0.0, 0.99)
    beta2_d = trial.suggest_float('beta2_d', 0.5, 0.999)

    # Set up experiment name for this trial
    exp_name = f"256_new3_optuna_trial_{trial.number}"

    # Prepare args for train.py
    args = [
        '--exp_name', exp_name,
        '--num_epochs', '10',
        '--batch_size', str(batch_size),
        '--image_size', '256',
        '--feature_mapG', '256',
        '--feature_mapD', '256',
        '--device', 'cuda',
        '--lr_g', str(lr_g),
        '--optimizer_g', str(optimizer_g),
        '--beta1_g', str(beta1_g),
        '--beta2_g', str(beta2_g),
        '--lr_d', str(lr_d),
        '--optimizer_d', str(optimizer_d),
        '--beta1_d', str(beta1_d),
        '--beta2_d', str(beta2_d),
    ]

    # Patch sys.argv for train.py
    import sys
    sys.argv = ['train.py'] + args

    fid = train_main()
    return fid


if __name__ == "__main__":
    # Use SQLite storage to persist the study across runs
    storage_url = "sqlite:///optuna_study.db"
    study = optuna.create_study(
        direction="minimize",
        storage=storage_url,
        study_name="dcgan_tuning_256_new3",
        load_if_exists=True  # Load existing study if it exists
    )
    study.optimize(objective, n_trials=60)

    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    # --- Save all trials as a CSV table ---
    df = study.trials_dataframe(attrs=("number", "value", "state", "params"))
    df.to_csv("256_optuna_trials_new3.csv", index=False)
    print("\nAll trial results saved to 256_optuna_trials_new3.csv")
