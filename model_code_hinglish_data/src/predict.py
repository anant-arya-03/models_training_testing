"""
predict.py
==========
Prediction pipeline for the EmoSen sentiment classification model.
"""

import os
import re
import numpy as np
import pandas as pd
import torch

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

from transformers import AutoTokenizer, AutoModelForSequenceClassification


# ----------------------------------------------------------
# Paths
# ----------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models", "best_model")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

TEST_CSV = os.path.join(DATA_DIR, "sentimix_test.csv")
OUTPUT_CSV = os.path.join(RESULTS_DIR, "predictions.csv")

MAX_LENGTH = 128
BATCH_SIZE = 32


# ----------------------------------------------------------
# Text preprocessing
# ----------------------------------------------------------
def preprocess_text(text):

    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)
    text = text.strip()

    return text


# ----------------------------------------------------------
# Load data
# ----------------------------------------------------------
def load_and_preprocess_data(file_path):

    df = pd.read_csv(file_path, usecols=["tweet", "sentiment"])
    df = df.dropna(subset=["tweet", "sentiment"]).reset_index(drop=True)
    df["tweet"] = df["tweet"].apply(preprocess_text)

    return df


# ----------------------------------------------------------
# Load label encoder
# ----------------------------------------------------------
def load_label_encoder(model_dir):

    classes_path = os.path.join(model_dir, "label_classes.npy")

    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.load(classes_path, allow_pickle=True)

    return label_encoder


# ----------------------------------------------------------
# Batch prediction
# ----------------------------------------------------------
def predict_sentiment(model, tokenizer, tweets, device):

    all_preds = []
    all_conf = []

    for i in range(0, len(tweets), BATCH_SIZE):

        batch = tweets[i:i+BATCH_SIZE]

        inputs = tokenizer(
            batch,
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )

        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():

            outputs = model(**inputs)

            probs = torch.nn.functional.softmax(outputs.logits, dim=1)

            preds = torch.argmax(probs, dim=1)

            conf = probs.max(dim=1).values

        all_preds.append(preds.cpu().numpy())
        all_conf.append(conf.cpu().numpy())

    return np.concatenate(all_preds), np.concatenate(all_conf)


# ----------------------------------------------------------
# MAIN
# ----------------------------------------------------------
def main():

    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("Loading model...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.eval()

    print("Loading label encoder...")
    label_encoder = load_label_encoder(MODEL_DIR)

    print("Loading test data...")
    test_df = load_and_preprocess_data(TEST_CSV)

    tweets = test_df["tweet"].tolist()
    true_labels = test_df["sentiment"].tolist()

    print("Running inference...")

    pred_ids, confidences = predict_sentiment(model, tokenizer, tweets, device)

    pred_labels = label_encoder.inverse_transform(pred_ids)

    # ----------------------------------------------------------
    # Evaluation Metrics
    # ----------------------------------------------------------

    true_ids = label_encoder.transform(true_labels)

    accuracy = accuracy_score(true_ids, pred_ids)
    precision = precision_score(true_ids, pred_ids, average="macro", zero_division=0)
    recall = recall_score(true_ids, pred_ids, average="macro", zero_division=0)
    f1 = f1_score(true_ids, pred_ids, average="macro", zero_division=0)

    print("\nTest Evaluation Metrics")
    print("----------------------")

    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Macro  : {f1:.4f}")

    print("\nClassification Report:")
    print(classification_report(true_ids, pred_ids, target_names=label_encoder.classes_))

    print("\nConfusion Matrix:")
    print(confusion_matrix(true_ids, pred_ids))

    # ----------------------------------------------------------
    # Save predictions
    # ----------------------------------------------------------

    output_df = pd.DataFrame({

        "tweet": tweets,
        "true_label": true_labels,
        "predicted_label": pred_labels,
        "confidence_score": np.round(confidences, 4)

    })

    output_df.to_csv(OUTPUT_CSV, index=False)

    print(f"\nPredictions saved to {OUTPUT_CSV}")


if __name__ == "__main__":

    main()