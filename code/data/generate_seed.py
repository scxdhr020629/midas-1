import pandas as pd
import numpy as np
import json
import random

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

# Load the matrix
ass = pd.read_excel('resistant_matrix.xlsx', sheet_name='Sheet1', index_col=0)

# Find positive interactions (where == 1)
pos = np.where(ass.values == 1)
po_rows = pos[0]
po_cols = pos[1]
num_pos = len(po_rows)

# Shuffle indices for positives
pos_indices = np.arange(num_pos)
np.random.shuffle(pos_indices)

# Split into 5 folds for positives
fold_size_pos = num_pos // 5
Positive = []
for i in range(5):
    start = i * fold_size_pos
    end = start + fold_size_pos if i < 4 else num_pos
    # Convert to Python int explicitly
    fold = [int(x) for x in pos_indices[start:end]]
    Positive.append(fold)

# Find negative interactions (where == 0)
neg = np.where(ass.values == 0)
ne_rows = neg[0]
ne_cols = neg[1]
num_neg = len(ne_rows)

# Downsample negatives
balance_ratio = 1
sampled_num_neg = num_pos * balance_ratio

# Randomly sample negatives
sampled_neg_indices = np.random.choice(num_neg, sampled_num_neg, replace=False)

# Split sampled negatives into 5 folds
fold_size_neg = sampled_num_neg // 5
Negative = []
for i in range(5):
    start = i * fold_size_neg
    end = start + fold_size_neg if i < 4 else sampled_num_neg
    # Convert to Python int explicitly
    fold = [int(x) for x in sampled_neg_indices[start:end]]
    Negative.append(fold)

# Save to JSON files
with open('resistant_Positive.txt', 'w') as f:
    json.dump(Positive, f)

with open('resistant_Negative.txt', 'w') as f:
    json.dump(Negative, f)

print(f"Generated Positive.txt with {num_pos} positives split into 5 folds.")
print(f"Generated Negative.txt with {sampled_num_neg} negatives (downsampled from {num_neg}) split into 5 folds.")
print("Balance ratio (neg:pos) per fold ≈", balance_ratio)