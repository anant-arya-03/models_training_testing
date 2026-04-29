"""
train_model.py
==============
Training pipeline for 3-class sentiment classification on the EmoSen
code-mixed Hinglish dataset using XLM-RoBERTa-base and HuggingFace Trainer.

Classes:  negative (0) | neutral (1) | positive (2)
"""

import os
import re

import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)


# ---------------------------------------------------------------------------
# Paths (resolved relative to this script so it works from any working dir)
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")

TRAIN_CSV = os.path.join(DATA_DIR, "sentimix_train.csv")
VAL_CSV = os.path.join(DATA_DIR, "sentimix_val.csv")

MODEL_NAME = "l3cube-pune/hing-roberta"
MAX_LENGTH = 128


# ---------------------------------------------------------------------------
# Text preprocessing
# ---------------------------------------------------------------------------
def preprocess_text(text):
    """Apply minimal preprocessing suitable for code-mixed Hinglish text."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+", "", text)          # remove URLs
    text = re.sub(r"@\w+", "", text)              # remove @usernames
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)   # normalise elongated words
    text = text.strip()
    return text


# ---------------------------------------------------------------------------
# Data loading & preparation
# ---------------------------------------------------------------------------
def load_and_preprocess_data(file_path):
    """Load CSV, keep only tweet & sentiment columns, apply preprocessing."""
    df = pd.read_csv(file_path, usecols=["tweet", "sentiment"])
    df = df.dropna(subset=["tweet", "sentiment"]).reset_index(drop=True)
    df["tweet"] = df["tweet"].apply(preprocess_text)
    return df


def encode_labels(df, label_encoder=None):
    """
    Encode sentiment labels to integers.

    If *label_encoder* is None a new one is created with a fixed class order:
        negative -> 0, neutral -> 1, positive -> 2
    """
    if label_encoder is None:
        label_encoder = LabelEncoder()
        # Fix the class order explicitly
        label_encoder.fit(["negative", "neutral", "positive"])

    df["label"] = label_encoder.transform(df["sentiment"])
    return df, label_encoder


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------
def tokenize_function(examples, tokenizer):
    """Tokenize a batch of examples."""
    return tokenizer(
        examples["tweet"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
    )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_metrics(eval_pred):
    """Compute accuracy, precision, recall, and macro-F1."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(labels, predictions, average="macro", zero_division=0),
        "recall": recall_score(labels, predictions, average="macro", zero_division=0),
        "f1_macro": f1_score(labels, predictions, average="macro", zero_division=0),
    }


# ---------------------------------------------------------------------------
# Custom Trainer with class weights for imbalanced data
# ---------------------------------------------------------------------------
class WeightedTrainer(Trainer):
    """Trainer that applies class weights to handle imbalanced datasets."""

    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if class_weights is not None:
            self.class_weights = torch.tensor(class_weights, dtype=torch.float32)
        else:
            self.class_weights = None

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        if self.class_weights is not None:
            weight = self.class_weights.to(logits.device)
            loss_fn = torch.nn.CrossEntropyLoss(weight=weight, label_smoothing=0.1)
        else:
            loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0.1)

        loss = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    # 1. Load & preprocess data ---------------------------------------------
    print("Loading and preprocessing data...")
    train_df = load_and_preprocess_data(TRAIN_CSV)
    val_df = load_and_preprocess_data(VAL_CSV)

    # 2. Encode labels ------------------------------------------------------
    print("Encoding labels...")
    train_df, label_encoder = encode_labels(train_df)
    val_df, _ = encode_labels(val_df, label_encoder)

    print(f"  Train samples : {len(train_df)}")
    print(f"  Val   samples : {len(val_df)}")
    print(f"  Label mapping : {dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}")

    # Compute class weights to handle imbalanced data
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.array([0, 1, 2]),
        y=train_df["label"].values
    )
    print(f"  Class weights : {dict(zip(['negative', 'neutral', 'positive'], class_weights))}")

    # 3. Convert to HuggingFace Dataset -------------------------------------
    print("Converting to HuggingFace Dataset...")
    train_dataset = Dataset.from_pandas(train_df[["tweet", "label"]], preserve_index=False)
    val_dataset = Dataset.from_pandas(val_df[["tweet", "label"]], preserve_index=False)

    # 4. Tokenize -----------------------------------------------------------
    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("Tokenizing datasets...")
    train_dataset = train_dataset.map(
        lambda x: tokenize_function(x, tokenizer), batched=True
    )
    val_dataset = val_dataset.map(
        lambda x: tokenize_function(x, tokenizer), batched=True
    )

    # 5. Load model ---------------------------------------------------------
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=3
    )

    # 6. Training arguments -------------------------------------------------
    training_args = TrainingArguments(
        output_dir=MODEL_DIR,
        learning_rate=2e-5,                      # Slightly higher for domain model
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=2,           # Effective batch size = 16
        num_train_epochs=5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        report_to="none",
    )

    # 7. Trainer with class weights -----------------------------------------
    print("Configuring Trainer with class weights...")
    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )

    # 8. Train --------------------------------------------------------------
    print("Training model...")
    trainer.train()

    # 9. Evaluate on validation set -----------------------------------------
    print("Evaluating on validation set...")
    eval_results = trainer.evaluate()
    print("Validation Results:")
    for key, value in eval_results.items():
        print(f"  {key}: {value:.4f}")

    # 10. Save best model, tokenizer & label encoder ------------------------
    best_model_path = os.path.join(MODEL_DIR, "best_model")
    os.makedirs(best_model_path, exist_ok=True)

    print("Saving best model...")
    trainer.save_model(best_model_path)
    tokenizer.save_pretrained(best_model_path)

    # Persist label encoder classes so predict.py can reconstruct the mapping
    np.save(os.path.join(best_model_path, "label_classes.npy"), label_encoder.classes_)
    print(f"Model and tokenizer saved to {best_model_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
        raise