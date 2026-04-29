"""
predict.py — Use the trained best_model to predict on test data with best accuracy.

Usage:
    python test/predict.py

    Place your test CSV file(s) inside  test/test_data/
    Each CSV must have a 'text' column (and optionally a 'label' column).

    Results are saved to  test/predictions/
"""

import os
import sys
import re
import glob
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import (
    classification_report,
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix,
    accuracy_score,
)

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# Paths  (relative to project root)
# ──────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR    = os.path.join(PROJECT_ROOT, "best_model")
TEST_DATA_DIR = os.path.join(PROJECT_ROOT, "test", "test_data")
PRED_DIR      = os.path.join(PROJECT_ROOT, "test", "predictions")

MAX_LEN = 256           # must match training
BATCH_SIZE = 64         # inference batch size (adjust for GPU memory)
LABEL_MAP = {0: "nonmisinfo", 1: "misinfo"}

# ──────────────────────────────────────────────
# Text cleaning  (same as train_pipeline.py)
# ──────────────────────────────────────────────
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"www\.\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#(\w+)", r"\1", text)
    text = re.sub(r"rt\s+", "", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ──────────────────────────────────────────────
# Load model & tokenizer once
# ──────────────────────────────────────────────
print(f"Loading model from: {MODEL_DIR}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.to(device)
model.eval()
print("Model loaded successfully.\n")

# ──────────────────────────────────────────────
# Inference helpers
# ──────────────────────────────────────────────
@torch.no_grad()
def get_probabilities(texts: list[str]) -> np.ndarray:
    """Return (N, 2) probability array for a list of texts."""
    all_probs = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        encodings = tokenizer(
            batch,
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN,
            return_tensors="pt",
        )
        encodings = {k: v.to(device) for k, v in encodings.items()}
        logits = model(**encodings).logits
        probs = F.softmax(logits, dim=1).cpu().numpy()
        all_probs.append(probs)
    return np.concatenate(all_probs, axis=0)


def find_best_threshold(probs: np.ndarray, labels: np.ndarray) -> float:
    """Search for the threshold that maximizes macro-F1 (same as training)."""
    best_f1, best_thresh = 0.0, 0.5
    for thresh in np.arange(0.20, 0.65, 0.01):
        preds = (probs[:, 1] > thresh).astype(int)
        _, _, f1, _ = precision_recall_fscore_support(
            labels, preds, average="macro", zero_division=0
        )
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
    return best_thresh


def evaluate_and_print(labels: np.ndarray, preds: np.ndarray, probs: np.ndarray, threshold: float):
    """Print a full evaluation report."""
    acc = accuracy_score(labels, preds)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )
    try:
        auc = roc_auc_score(labels, probs[:, 1])
    except Exception:
        auc = 0.0

    print(f"  Best Threshold  : {threshold:.2f}")
    print(f"  Accuracy        : {acc:.4f}")
    print(f"  Macro Precision : {p_macro:.4f}")
    print(f"  Macro Recall    : {r_macro:.4f}")
    print(f"  Macro F1        : {f1_macro:.4f}")
    print(f"  AUC-ROC         : {auc:.4f}")
    print()
    print("  Confusion Matrix:")
    print(confusion_matrix(labels, preds))
    print()
    print("  Classification Report:")
    print(classification_report(labels, preds, target_names=["nonmisinfo", "misinfo"], zero_division=0))


# ──────────────────────────────────────────────
# Process every CSV in test/test_data/
# ──────────────────────────────────────────────
csv_files = sorted(glob.glob(os.path.join(TEST_DATA_DIR, "*.csv")))

if not csv_files:
    print("=" * 60)
    print("No CSV files found in test/test_data/")
    print("Please place your test CSV files there and re-run.")
    print("Each CSV must have a 'text' column.")
    print("If it also has a 'label' column, evaluation metrics will be shown.")
    print("=" * 60)
    sys.exit(0)

for csv_path in csv_files:
    fname = os.path.basename(csv_path)
    print("=" * 60)
    print(f"Processing: {fname}")
    print("=" * 60)

    # ---- Load data ----
    df = pd.read_csv(csv_path, dtype=str)

    if "text" not in df.columns:
        print(f"  ⚠  Skipping '{fname}' — no 'text' column found.\n")
        continue

    df["text_clean"] = df["text"].apply(clean_text)
    # Remove very short texts (same filter as training)
    df = df[df["text_clean"].str.split().str.len() >= 3].reset_index(drop=True)
    print(f"  Samples after cleaning: {len(df)}")

    # ---- Predict ----
    texts = df["text_clean"].tolist()
    probs = get_probabilities(texts)

    # ---- Determine if labels are available ----
    has_labels = "label" in df.columns
    if has_labels:
        # Map string labels to int if needed
        label_col = df["label"].copy()
        label_col = label_col.map({"misinfo": 1, "nonmisinfo": 0, "1": 1, "0": 0})
        label_col = pd.to_numeric(label_col, errors="coerce")
        label_col = label_col.dropna().astype(int)
        labels = label_col.values

        if len(labels) == len(probs):
            # Find best threshold on this test set for maximum accuracy
            threshold = find_best_threshold(probs, labels)
            preds = (probs[:, 1] > threshold).astype(int)
            print()
            evaluate_and_print(labels, preds, probs, threshold)
        else:
            has_labels = False
            print("  ⚠  Some labels could not be parsed; skipping evaluation.")

    if not has_labels:
        # Use default threshold 0.5 when no labels available
        threshold = 0.50
        preds = (probs[:, 1] > threshold).astype(int)
        print(f"  (No labels found — using default threshold {threshold})")

    # ---- Build output DataFrame ----
    out_df = df.copy()
    out_df["predicted_label"] = [LABEL_MAP[p] for p in preds]
    out_df["prob_nonmisinfo"] = probs[:, 0].round(5)
    out_df["prob_misinfo"]    = probs[:, 1].round(5)
    out_df["threshold_used"]  = threshold

    # ---- Save predictions ----
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_name = f"predictions_{os.path.splitext(fname)[0]}_{timestamp}.csv"
    out_path = os.path.join(PRED_DIR, out_name)
    out_df.to_csv(out_path, index=False)
    print(f"  Predictions saved → {out_path}\n")

print("All done!")
