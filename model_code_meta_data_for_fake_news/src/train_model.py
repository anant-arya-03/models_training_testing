import os
import re
import sys
import numpy as np
import pandas as pd
import torch
import logging
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from imblearn.over_sampling import RandomOverSampler
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
from torch.nn import CrossEntropyLoss

# ================================================
# 0. SUPPRESS WARNINGS
# ================================================
logging.getLogger("transformers").setLevel(logging.ERROR)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ================================================
# 1. PATHS
# ================================================
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH  = os.path.join(BASE_DIR, "data", "fake_news_project.csv")
MODEL_DIR  = os.path.join(BASE_DIR, "models")
MODEL_NAME = "roberta-large"

os.makedirs(MODEL_DIR, exist_ok=True)

# ================================================
# 2. GPU CHECK
# ================================================
print("=" * 55)
print("               SYSTEM CHECK                    ")
print("=" * 55)
print(f"PyTorch Version : {torch.__version__}")
print(f"GPU Available   : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Name        : {torch.cuda.get_device_name(0)}")
    print(f"VRAM            : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("WARNING: No GPU found, training will be slow!")
print("=" * 55)

# ================================================
# 3. CLEAN TEXT
# ================================================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+",       "",      text)  # remove URLs
    text = re.sub(r"@\w+",          "",      text)  # remove mentions
    text = re.sub(r"(.)\1{2,}",    r"\1\1", text)  # remove repeated chars
    text = re.sub(r"[^a-z0-9\s]",  " ",     text)  # remove special chars
    text = re.sub(r"\s+",           " ",     text)  # collapse whitespace
    return text.strip()

# ================================================
# 4. LOAD DATA
# ================================================
print("\n" + "=" * 55)
print("               LOADING DATA                    ")
print("=" * 55)

try:
    df = pd.read_csv(DATA_PATH, encoding="latin1")
    print(f"Loaded successfully!")
    print(f"Shape   : {df.shape}")
    print(f"Columns : {df.columns.tolist()}")
except Exception as e:
    print(f"ERROR loading CSV: {e}")
    raise

# Validate required columns
required_cols = {"content", "raw_label", "title"}
if not required_cols.issubset(df.columns):
    raise ValueError(f"Missing columns. Required: {required_cols}. Found: {df.columns.tolist()}")

# ================================================
# 5. PROCESS DATA
# ================================================
print("\n" + "=" * 55)
print("               PROCESSING DATA                 ")
print("=" * 55)

label_map = {
    "fake"        : "fake",
    "misleading"  : "fake",
    "mostly fake" : "fake",
    "half true"   : "neutral",
    "mostly true" : "real",
    "true"        : "real"
}

# Combine title + content for richer input
df["text"]  = df["title"].fillna("") + " [SEP] " + df["content"].fillna("")
df["label"] = df["raw_label"].map(label_map)
df = df[["text", "label"]].dropna()
df["text"]  = df["text"].apply(clean_text)

print(f"Shape after processing : {df.shape}")
print(f"Label distribution:\n{df['label'].value_counts()}")

# ================================================
# 6. ENCODE LABELS
# ================================================
print("\n" + "=" * 55)
print("               ENCODING LABELS                 ")
print("=" * 55)

le          = LabelEncoder()
df["label"] = le.fit_transform(df["label"])
label_names = le.classes_.tolist()
num_labels  = len(label_names)

np.save(os.path.join(MODEL_DIR, "labels.npy"), le.classes_)
print(f"Label classes : {label_names}")
print(f"Num labels    : {num_labels}")

# ================================================
# 7. OVERSAMPLE MINORITY CLASSES
# ================================================
print("\n" + "=" * 55)
print("          FIXING CLASS IMBALANCE               ")
print("=" * 55)

print(f"Before oversampling:\n{df['label'].value_counts()}")

ros          = RandomOverSampler(random_state=42)
X_res, y_res = ros.fit_resample(df[["text"]], df["label"])
df           = pd.DataFrame({"text": X_res["text"], "label": y_res})

print(f"\nAfter oversampling:\n{df['label'].value_counts()}")

# ================================================
# 8. CLASS WEIGHTS
# ================================================
label_counts  = df["label"].value_counts().sort_index()
class_weights = torch.tensor(
    [len(df) / (num_labels * label_counts[i]) for i in range(num_labels)],
    dtype=torch.float
)
print(f"\nClass weights : {class_weights}")

# ================================================
# 9. TRAIN / VALIDATION / TEST SPLIT  (70 / 15 / 15)
# ================================================
print("\n" + "=" * 55)
print("             SPLITTING DATASET                 ")
print("=" * 55)

# Step 1: split off 15% test → remaining 85%
train_val_df, test_df = train_test_split(
    df,
    test_size=0.15,
    random_state=42,
    stratify=df["label"]
)

# Step 2: split remaining 85% into 70% train + 15% val
# 15/85 ≈ 0.1765 gives exactly 15% of total for val
train_df, val_df = train_test_split(
    train_val_df,
    test_size=0.1765,
    random_state=42,
    stratify=train_val_df["label"]
)

print(f"Total samples : {len(df)}")
print(f"Train size    : {len(train_df)}  (~70%)")
print(f"Val size      : {len(val_df)}    (~15%)")
print(f"Test size     : {len(test_df)}   (~15%)")

# Save test split so test_write.py can load it directly
test_df.to_csv(os.path.join(MODEL_DIR, "test_split.csv"), index=False)
print(f"Test split saved to models/test_split.csv")

# ================================================
# 10. HUGGINGFACE DATASETS
# ================================================
train_dataset = Dataset.from_pandas(train_df).rename_column("label", "labels")
val_dataset   = Dataset.from_pandas(val_df).rename_column("label", "labels")
test_dataset  = Dataset.from_pandas(test_df).rename_column("label", "labels")

# ================================================
# 11. TOKENIZER
# ================================================
print("\n" + "=" * 55)
print("             LOADING TOKENIZER                 ")
print("=" * 55)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(batch):
    return tokenizer(
        batch["text"],
        padding="max_length",
        truncation=True,
        max_length=512
    )

train_dataset = train_dataset.map(tokenize, batched=True)
val_dataset   = val_dataset.map(tokenize, batched=True)
test_dataset  = test_dataset.map(tokenize, batched=True)

# Set PyTorch format
train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
val_dataset.set_format(type="torch",   columns=["input_ids", "attention_mask", "labels"])
test_dataset.set_format(type="torch",  columns=["input_ids", "attention_mask", "labels"])

print("Tokenization complete!")

# ================================================
# 12. LOAD MODEL
# ================================================
print("\n" + "=" * 55)
print("               LOADING MODEL                   ")
print("=" * 55)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=num_labels
)
print(f"Model loaded : {MODEL_NAME}")
print(f"Num labels   : {num_labels}")

# ================================================
# 13. WEIGHTED LOSS TRAINER
# ================================================
class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels  = inputs.get("labels")
        outputs = model(**inputs)
        logits  = outputs.get("logits")
        loss_fn = CrossEntropyLoss(
            weight=class_weights.to(model.device),
            label_smoothing=0.1        # NEW: prevents overconfidence, better generalization
        )
        loss    = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss

# ================================================
# 14. METRICS
# ================================================
def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    return {
        "accuracy" : accuracy_score(p.label_ids, preds),
        "f1"       : f1_score(p.label_ids, preds, average="macro")
    }

# ================================================
# 15. TRAINING ARGUMENTS  — tuned for higher accuracy
# ================================================
training_args = TrainingArguments(
    output_dir                  = MODEL_DIR,

    # ── Learning rate (CHANGED: 5e-6 → 2e-5) ────────────────────
    learning_rate               = 2e-5,        # sweet spot for RoBERTa fine-tuning

    # ── Batch size (CHANGED: 16 → 8) ────────────────────────────
    per_device_train_batch_size = 8,           # smaller = better generalization
    per_device_eval_batch_size  = 16,

    # ── Epochs (CHANGED: 10 → 15) ────────────────────────────────
    num_train_epochs            = 15,          # more room to converge

    # ── LR Scheduler (CHANGED: linear → cosine) ──────────────────
    lr_scheduler_type           = "cosine",    # smoother decay, better final acc

    # ── Evaluation & saving ───────────────────────────────────────
    eval_strategy               = "epoch",
    save_strategy               = "epoch",
    logging_steps               = 25,
    report_to                   = "none",

    # ── Warmup (CHANGED: 0.15 → 0.1) ────────────────────────────
    warmup_ratio                = 0.1,         # less warmup, faster useful training

    # ── Regularization (CHANGED: 0.01 → 0.05) ───────────────────
    weight_decay                = 0.05,        # stronger regularization

    # ── Best model selection ──────────────────────────────────────
    load_best_model_at_end      = True,
    metric_for_best_model       = "f1",
    greater_is_better           = True,

    # ── Hardware ──────────────────────────────────────────────────
    fp16                        = True,
    dataloader_num_workers      = 8,
    dataloader_pin_memory       = True,

    # ── Gradient accumulation (CHANGED: 4 → 8) ──────────────────
    gradient_accumulation_steps = 8,           # effective batch = 8×8 = 64
    gradient_checkpointing      = True,
)

# ================================================
# 16. TRAINER  (val_dataset for eval, test stays unseen)
# ================================================
trainer = WeightedTrainer(
    model           = model,
    args            = training_args,
    train_dataset   = train_dataset,
    eval_dataset    = val_dataset,
    compute_metrics = compute_metrics,
    callbacks       = [EarlyStoppingCallback(early_stopping_patience=5)]  # CHANGED: 3 → 5
)

# ================================================
# 17. TRAIN
# ================================================
print("\n" + "=" * 55)
print("             TRAINING STARTED                  ")
print("=" * 55)
print("  Changes vs previous run:")
print("  learning_rate               : 5e-6   → 2e-5")
print("  per_device_train_batch_size : 16     → 8")
print("  num_train_epochs            : 10     → 15")
print("  lr_scheduler_type           : linear → cosine")
print("  warmup_ratio                : 0.15   → 0.1")
print("  weight_decay                : 0.01   → 0.05")
print("  gradient_accumulation_steps : 4      → 8")
print("  early_stopping_patience     : 3      → 5")
print("  label_smoothing             : none   → 0.1")
print("=" * 55)

trainer.train()

# ================================================
# 18. EVALUATE  (on validation set)
# ================================================
print("\n" + "=" * 55)
print("         EVALUATING ON VALIDATION SET          ")
print("=" * 55)

results = trainer.evaluate()
print(f"Validation Results : {results}")

# ================================================
# 19. PREDICTIONS ON VALIDATION SET
# ================================================
print("\n" + "=" * 55)
print("      GENERATING PREDICTIONS (VALIDATION)      ")
print("=" * 55)

predictions = trainer.predict(val_dataset)
preds       = np.argmax(predictions.predictions, axis=1)
true_labels = predictions.label_ids

# ================================================
# 20. CLASSIFICATION REPORT
# ================================================
print("\n========== CLASSIFICATION REPORT (VALIDATION) ==========")
print(classification_report(true_labels, preds, target_names=label_names))

# ================================================
# 21. CONFUSION MATRIX PLOT
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
plt.title("Confusion Matrix (Validation)",  fontsize=16, fontweight="bold")
plt.ylabel("Actual Label",     fontsize=12)
plt.xlabel("Predicted Label",  fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, "confusion_matrix.png"), dpi=150)
plt.close()
print("Confusion matrix saved!")

# ================================================
# 22. TRAINING CURVES PLOT
# ================================================
history   = trainer.state.log_history
epochs    = []
eval_loss = []
eval_acc  = []
eval_f1   = []

for log in history:
    if "eval_loss" in log:
        epochs.append(log.get("epoch"))
        eval_loss.append(log.get("eval_loss"))
        eval_acc.append(log.get("eval_accuracy"))
        eval_f1.append(log.get("eval_f1"))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(epochs, eval_acc, marker="o", color="blue",  linewidth=2, label="Accuracy")
axes[0].plot(epochs, eval_f1,  marker="s", color="green", linewidth=2, label="F1 Score")
axes[0].set_title("Accuracy & F1 per Epoch", fontsize=14, fontweight="bold")
axes[0].set_xlabel("Epoch",  fontsize=11)
axes[0].set_ylabel("Score",  fontsize=11)
axes[0].set_ylim(0, 1)
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)

axes[1].plot(epochs, eval_loss, marker="o", color="red", linewidth=2, label="Eval Loss")
axes[1].set_title("Evaluation Loss per Epoch", fontsize=14, fontweight="bold")
axes[1].set_xlabel("Epoch", fontsize=11)
axes[1].set_ylabel("Loss",  fontsize=11)
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, "training_curves.png"), dpi=150)
plt.close()
print("Training curves saved!")

# ================================================
# 23. SAVE MODEL
# ================================================
print("\n" + "=" * 55)
print("               SAVING MODEL                    ")
print("=" * 55)

trainer.save_model(MODEL_DIR)
tokenizer.save_pretrained(MODEL_DIR)
print(f"Model saved to : {MODEL_DIR}")

# ================================================
# 24. FINAL SUMMARY
# ================================================
print("\n" + "=" * 55)
print("               FINAL SUMMARY                   ")
print("=" * 55)
print(f"Label Classes  : {label_names}")
print(f"Best Accuracy  : {max(eval_acc):.4f}")
print(f"Best F1 Score  : {max(eval_f1):.4f}")
print(f"Final Eval Loss: {eval_loss[-1]:.4f}")
print(f"Model saved to : {MODEL_DIR}")
print(f"Plots saved to : {MODEL_DIR}")
print("=" * 55)
print("          TRAINING COMPLETE!                   ")
print("=" * 55)
print(f"\nNOTE: Test set (15%) saved to models/test_split.csv")
print(f"      Run test_write.py to evaluate on unseen test data.")
print("=" * 55)
