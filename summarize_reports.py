import os
import glob
import re
import pandas as pd

# Path to all training_report.txt files
report_files = glob.glob("checkpoints/new_optuna_trial_*/training_report.txt")

results = []

for report_path in report_files:
    with open(report_path, "r") as f:
        text = f.read()

    # Extract values using regex
    exp = re.search(r"Experiment:\s*(.*)", text)
    epochs = re.search(r"Epochs:\s*(\d+)", text)
    batch_size = re.search(r"Batch size:\s*(\d+)", text)
    lr_g = re.search(r"Learning rate \(G\):\s*([0-9.eE+-]+)", text)
    optimizer_g = re.search(r"Optimizer \(G\):\s*(\w+)", text)
    beta1_g = re.search(r"Beta1 \(G\):\s*([0-9.eE+-]+)", text)
    beta2_g = re.search(r"Beta2 \(G\):\s*([0-9.eE+-]+)", text)
    lr_d = re.search(r"Learning rate \(D\):\s*([0-9.eE+-]+)", text)
    optimizer_d = re.search(r"Optimizer \(D\):\s*(\w+)", text)
    beta1_d = re.search(r"Beta1 \(D\):\s*([0-9.eE+-]+)", text)
    beta2_d = re.search(r"Beta2 \(D\):\s*([0-9.eE+-]+)", text)
    best_fid = re.search(r"Best FID:\s*([0-9.eE+-]+)", text)
    best_is = re.search(r"Best Inception Score:\s*([0-9.eE+-]+)", text)

    results.append({
        "experiment": exp.group(1) if exp else "",
        "epochs": int(epochs.group(1)) if epochs else None,
        "batch_size": int(batch_size.group(1)) if batch_size else None,
        "lr_g": float(lr_g.group(1)) if lr_g else None,
        "optimizer_g": optimizer_g.group(1) if optimizer_g else "",
        "beta1_g": float(beta1_g.group(1)) if beta1_g else None,
        "beta2_g": float(beta2_g.group(1)) if beta2_g else None,
        "lr_d": float(lr_d.group(1)) if lr_d else None,
        "optimizer_d": optimizer_d.group(1) if optimizer_d else "",
        "beta1_d": float(beta1_d.group(1)) if beta1_d else None,
        "beta2_d": float(beta2_d.group(1)) if beta2_d else None,
        "best_fid": float(best_fid.group(1)) if best_fid else None,
        "best_is": float(best_is.group(1)) if best_is else None,
    })

# Save to CSV
df = pd.DataFrame(results)
df = df.sort_values("experiment")
df.to_csv("optuna_training_reports_summary.csv", index=False)
print("Saved summary to optuna_training_reports_summary.csv")