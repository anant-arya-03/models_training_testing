import os
import re
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, f1_score,
    classification_report, confusion_matrix
)
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer

# ================================================
# 0. PATHS
# ================================================
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR   = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

# ================================================
# 1. CLEAN TEXT  (same as training)
# ================================================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+",      "",      text)
    text = re.sub(r"@\w+",         "",      text)
    text = re.sub(r"(.)\1{2,}",   r"\1\1", text)
    text = re.sub(r"[^a-z0-9\s]", " ",     text)
    text = re.sub(r"\s+",          " ",     text)
    return text.strip()

# ================================================
# 2. LOAD SAVED TEST SPLIT  (completely unseen during training)
# ================================================
print("=" * 55)
print("           LOADING UNSEEN TEST DATA            ")
print("=" * 55)

test_split_path = os.path.join(MODEL_DIR, "test_split.csv")
if not os.path.exists(test_split_path):
    raise FileNotFoundError(
        "test_split.csv not found in models/. "
        "Please run train_model.py first to generate it."
    )

test_df = pd.read_csv(test_split_path)
print(f"Test samples  : {len(test_df)}")
print(f"Label distribution:\n{test_df['label'].value_counts()}")

# ================================================
# 3. LOAD LABEL CLASSES
# ================================================
le = LabelEncoder()
le.classes_ = np.load(os.path.join(MODEL_DIR, "labels.npy"), allow_pickle=True)
label_names  = le.classes_.tolist()
num_labels   = len(label_names)

print(f"\nLabel classes : {label_names}")

# ================================================
# 4. LOAD SAVED MODEL & TOKENIZER
# ================================================
print("\n" + "=" * 55)
print("           LOADING SAVED MODEL                 ")
print("=" * 55)

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model     = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

print(f"Model loaded from : {MODEL_DIR}")

# ================================================
# 5. TOKENIZE TEST DATA
# ================================================
test_dataset = Dataset.from_pandas(test_df).rename_column("label", "labels")

def tokenize(batch):
    return tokenizer(
        batch["text"],
        padding="max_length",
        truncation=True,
        max_length=512
    )

test_dataset = test_dataset.map(tokenize, batched=True)
test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# ================================================
# 6. RUN PREDICTIONS
# ================================================
print("\n" + "=" * 55)
print("           RUNNING PREDICTIONS                 ")
print("=" * 55)

trainer     = Trainer(model=model)
predictions = trainer.predict(test_dataset)
preds       = np.argmax(predictions.predictions, axis=1)
true_labels = predictions.label_ids
probs       = torch.softmax(torch.tensor(predictions.predictions), dim=1).numpy()

# ================================================
# 7. METRICS
# ================================================
acc = accuracy_score(true_labels, preds)
f1  = f1_score(true_labels, preds, average="macro")

print(f"\nAccuracy  : {acc:.4f}")
print(f"F1 (macro): {f1:.4f}")

print("\n========== CLASSIFICATION REPORT (TEST SET) ==========")
print(classification_report(true_labels, preds, target_names=label_names))

# ================================================
# 8. SAVE PREDICTIONS TO CSV
# ================================================
results_df = test_df.copy().reset_index(drop=True)
results_df["true_label"]      = le.inverse_transform(true_labels)
results_df["predicted_label"] = le.inverse_transform(preds)
results_df["correct"]         = results_df["true_label"] == results_df["predicted_label"]

for i, cls in enumerate(label_names):
    results_df[f"prob_{cls}"] = probs[:, i]

results_df["confidence"] = probs.max(axis=1)

out_csv = os.path.join(RESULTS_DIR, "test_predictions.csv")
results_df.to_csv(out_csv, index=False)
print(f"\nPredictions saved to : {out_csv}")

# ================================================
# 9. CONFUSION MATRIX
# ================================================
cm = confusion_matrix(true_labels, preds)

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=label_names,
    yticklabels=label_names,
    linewidths=0.5
)
plt.title("Confusion Matrix — Unseen Test Set", fontsize=16, fontweight="bold")
plt.ylabel("Actual Label",    fontsize=12)
plt.xlabel("Predicted Label", fontsize=12)
plt.tight_layout()
cm_path = os.path.join(RESULTS_DIR, "test_confusion_matrix.png")
plt.savefig(cm_path, dpi=150)
plt.close()
print(f"Confusion matrix saved to : {cm_path}")

# ================================================
# 10. WRONG PREDICTIONS
# ================================================
wrong_df   = results_df[~results_df["correct"]]
wrong_path = os.path.join(RESULTS_DIR, "wrong_predictions.csv")
wrong_df.to_csv(wrong_path, index=False)
print(f"Wrong predictions saved to : {wrong_path}  ({len(wrong_df)} samples)")

# ================================================
# 11. FINAL SUMMARY
# ================================================
print("\n" + "=" * 55)
print("          FINAL SUMMARY (UNSEEN TEST SET)      ")
print("=" * 55)
print(f"Total Test Samples : {len(test_df)}")
print(f"Correct            : {results_df['correct'].sum()}")
print(f"Wrong              : {(~results_df['correct']).sum()}")
print(f"Accuracy           : {acc:.4f}")
print(f"F1 Score (macro)   : {f1:.4f}")
print(f"Avg Confidence     : {results_df['confidence'].mean():.4f}")
print("=" * 55)
print("          TESTING COMPLETE!                    ")
print("=" * 55)
