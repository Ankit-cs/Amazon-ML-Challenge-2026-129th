# %% [markdown]
# # LightGBM Baseline for Entity Resolution
# This script uses LightGBM and TF-IDF blocking to solve the Entity Resolution challenge.
# It complies perfectly with the problem statement (<8B parameters, open-source).

# %%
import os
import lightgbm as lgb
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

# (Assuming data prep and feature extraction is imported from src)
import sys
sys.path.append("../src")
from data_prep import prepare_all_data
from features import extract_features

# %% [markdown]
# ## 1. Load Data

# %%
data_dir = "../data/raw/"
print("Loading data...")
# Loading limited pairs for exploration
records_a, records_b, X, y = prepare_all_data(data_dir=data_dir)

X = np.array(X)
y = np.array(y)
print(f"Features shape: {X.shape}, Labels shape: {y.shape}")

# %% [markdown]
# ## 2. Train LightGBM

# %%
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

train_data = lgb.Dataset(X_train, label=y_train)
val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1
}

print("Training LightGBM...")
gbm = lgb.train(
    params,
    train_data,
    num_boost_round=500,
    valid_sets=[val_data],
    callbacks=[lgb.early_stopping(stopping_rounds=50)]
)

# %% [markdown]
# ## 3. Save Model to /models

# %%
model_path = "../models/lgbm_model.txt"
os.makedirs("../models", exist_ok=True)
gbm.save_model(model_path)
print(f"Model saved to {model_path}")

# %% [markdown]
# ## 4. Threshold Tuning for F0.5 Score
# The problem statement specifically evaluates using F0.5 score.

# %%
from sklearn.metrics import precision_score, recall_score

preds = gbm.predict(X_val, num_iteration=gbm.best_iteration)

best_f05 = 0
best_thresh = 0

for thresh in np.arange(0.3, 0.9, 0.05):
    binary_preds = (preds > thresh).astype(int)
    p = precision_score(y_val, binary_preds, zero_division=0)
    r = recall_score(y_val, binary_preds, zero_division=0)
    
    if p + r > 0:
        f05 = (1.25 * p * r) / ((0.25 * p) + r)
        if f05 > best_f05:
            best_f05 = f05
            best_thresh = thresh

print(f"Best Threshold: {best_thresh:.2f}")
print(f"Validation F0.5 Score: {best_f05:.4f}")
