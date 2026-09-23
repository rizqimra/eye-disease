import re
import pandas as pd
import matplotlib.pyplot as plt

def extract_metrics(report_path, class_label):
    with open(report_path) as f:
        text = f.read()
    # Extract FID section
    fid_section = re.search(r"--- Per-Epoch FID ---\n(.*?)\n--- Per-Epoch Inception Score ---", text, re.DOTALL)
    fid_lines = fid_section.group(1).strip().split('\n') if fid_section else []
    # Extract IS section
    is_section = re.search(r"--- Per-Epoch Inception Score ---\n(.*)", text, re.DOTALL)
    is_lines = is_section.group(1).strip().split('\n') if is_section else []
    # Parse
    fids = [float(re.search(r": ([0-9.]+)", l).group(1)) for l in fid_lines if ":" in l]
    iss = [float(re.search(r": ([0-9.]+)", l).group(1)) for l in is_lines if ":" in l]
    epochs = list(range(1, min(len(fids), len(iss)) + 1))
    df = pd.DataFrame({"epoch": epochs, "FID": fids[:len(epochs)], "IS": iss[:len(epochs)], "class_label": class_label})
    return df

# Paths to your reports
reports = [
    ("checkpoints/tuned_dgcan_normal/training_report.txt", "normal"),
    ("checkpoints/tuned_dcgan_cataract/training_report.txt", "cataract"),
    ("checkpoints/tuned_dcgan_glaucoma/training_report.txt", "glaucoma"),
]

dfs = [extract_metrics(path, label) for path, label in reports]
df_all = pd.concat(dfs, ignore_index=True)

# Plot FID
plt.figure(figsize=(10,5))
for label in df_all["class_label"].unique():
    plt.plot(df_all[df_all["class_label"] == label]["epoch"],
             df_all[df_all["class_label"] == label]["FID"],
             label=f"FID {label}")
plt.xlabel("Epoch")
plt.ylabel("FID")
plt.title("FID per Epoch untuk Tiap Kelas")
plt.legend()
plt.grid(True)
plt.show()

# Plot IS
plt.figure(figsize=(10,5))
for label in df_all["class_label"].unique():
    plt.plot(df_all[df_all["class_label"] == label]["epoch"],
             df_all[df_all["class_label"] == label]["IS"],
             label=f"IS {label}")
plt.xlabel("Epoch")
plt.ylabel("IS")
plt.title("IS per Epoch for Each Class")
plt.legend()
plt.grid(True)
plt.show()
