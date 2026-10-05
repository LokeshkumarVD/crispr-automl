import pickle
import numpy as np

# Load dataset
with open("encoded_data_charlier_et_al.pkl", "rb") as f:
    data = pickle.load(f)

# ==========================
# CIRCLE-seq / CRISPR DATA
# ==========================

crispr = data["encodedDataCrispr"]

print("===== CIRCLE-seq / CRISPR DATA =====")

print("Number of samples:", len(crispr["target"]))
print("Images shape:", crispr["images"].shape)
print("Target shape:", crispr["target"].shape)
print("Target names shape:", crispr["target_names"].shape)

print("\nFirst 10 target names:")
print(crispr["target_names"][:10])

print("\nFirst 10 labels:")
print(crispr["target"][:10])

# Count classes
unique, counts = np.unique(crispr["target"], return_counts=True)

print("\nClass distribution:")
for label, count in zip(unique, counts):
    print("Class", label, ":", count)


# ==========================
# GUIDE-seq DATA
# ==========================

guide = data["encodedDataGuideSeq"]

print("\n\n===== GUIDE-seq DATA =====")

print("Number of samples:", len(guide["target"]))
print("Images shape:", guide["images"].shape)
print("Target shape:", guide["target"].shape)
print("Target names shape:", guide["target_names"].shape)

print("\nFirst 10 target names:")
print(guide["target_names"][:10])

print("\nFirst 10 labels:")
print(guide["target"][:10])

# Count classes
unique, counts = np.unique(guide["target"], return_counts=True)

print("\nClass distribution:")
for label, count in zip(unique, counts):
    print("Class", label, ":", count)