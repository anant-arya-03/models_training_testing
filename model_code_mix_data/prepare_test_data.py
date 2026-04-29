"""
prepare_test_data.py — Create a held-out test set from the original training CSVs.

The train_pipeline.py uses train_test_split with test_size=0.2, random_state=42
on the *balanced* data. Here we replicate the same cleaning but carve out a proper
test split from the *original* (unbalanced) data BEFORE any oversampling, so the
test set has never been seen by the model during training.

Steps:
  1. Load & clean data identically to train_pipeline.py
  2. Stratified split → 80 % train / 20 % TEST  (random_state=99 — different seed
     so it is NOT the same split used for validation)
  3. Save the 20 % test portion to  test/test_data/test_set.csv
"""

import os, re
import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MISINFO_CSV   = os.path.join(PROJECT_ROOT, "model_traning", "training data", "misinfo_train.csv")
NONMISINFO_CSV = os.path.join(PROJECT_ROOT, "model_traning", "training data", "nonmisinfo_train.csv")
OUT_PATH      = os.path.join(PROJECT_ROOT, "test", "test_data", "test_set.csv")

# ---- Load ----
df1 = pd.read_csv(MISINFO_CSV, dtype=str)
df2 = pd.read_csv(NONMISINFO_CSV, dtype=str)
df  = pd.concat([df1, df2], ignore_index=True)
df["text"] = df["text"].astype(str)
df["label"] = df["label"].map({"misinfo": 1, "nonmisinfo": 0})
df = df.dropna(subset=["label"])
df["label"] = df["label"].astype(int)

# ---- Clean (same as train_pipeline.py) ----
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"www\.\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#(\w+)", r"\1", text)
    text = re.sub(r"rt\s+", "", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["text"] = df["text"].apply(clean_text)
df = df[df["text"].str.split().str.len() >= 3]

print(f"Total cleaned samples: {len(df)}")
print(f"Label distribution:\n{df['label'].value_counts().to_string()}\n")

# ---- Stratified split — use a DIFFERENT seed so test ≠ val ----
_, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=99,   # different from training seed (42)
    stratify=df["label"],
)

test_df = test_df.reset_index(drop=True)

# Map labels back to strings for readability
test_df["label"] = test_df["label"].map({1: "misinfo", 0: "nonmisinfo"})

print(f"Test set size: {len(test_df)}")
print(f"Test label distribution:\n{test_df['label'].value_counts().to_string()}\n")

test_df[["text", "label"]].to_csv(OUT_PATH, index=False)
print(f"Saved → {OUT_PATH}")
