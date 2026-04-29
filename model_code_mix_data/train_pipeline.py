import pandas as pd
import torch
import re
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, precision_recall_fscore_support, roc_auc_score
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)

# ---- Data Loading ----
df1 = pd.read_csv("/home/projectwork/data_project/model_traning/training data/misinfo_train.csv", dtype=str)
df2 = pd.read_csv("/home/projectwork/data_project/model_traning/training data/nonmisinfo_train.csv", dtype=str)

df = pd.concat([df1, df2], ignore_index=True)
df['text'] = df['text'].astype(str)

df['label'] = df['label'].map({'misinfo': 1, 'nonmisinfo': 0})
df = df.dropna(subset=['label'])
df['label'] = df['label'].astype(int)

# ---- Text Cleaning ----
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

df['text'] = df['text'].apply(clean_text)
df = df[df['text'].str.split().str.len() >= 3]

print("Original Label Distribution:")
print(df['label'].value_counts())

# ================================================================
# KEY FIX 1: OVERSAMPLE minority class to reduce imbalance
# ================================================================
from imblearn.over_sampling import RandomOverSampler

minority_count = df['label'].value_counts()[1]
majority_count = df['label'].value_counts()[0]

# Oversample minority to at least 1:5 ratio with majority
target_minority = max(minority_count, majority_count // 5)

df_majority = df[df['label'] == 0]
df_minority = df[df['label'] == 1]

# Oversample minority with replacement
df_minority_upsampled = df_minority.sample(
    n=target_minority,
    replace=True,
    random_state=42
)

df_balanced = pd.concat([df_majority, df_minority_upsampled], ignore_index=True)
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

print("\nBalanced Label Distribution:")
print(df_balanced['label'].value_counts())

# ---- Split ----
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df_balanced['text'],
    df_balanced['label'],
    test_size=0.2,
    random_state=42,
    stratify=df_balanced['label']
)

# ---- Tokenization ----
model_name = "xlm-roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
MAX_LEN = 256

train_encodings = tokenizer(
    train_texts.tolist(),
    truncation=True,
    padding="max_length",
    max_length=MAX_LEN
)

val_encodings = tokenizer(
    val_texts.tolist(),
    truncation=True,
    padding="max_length",
    max_length=MAX_LEN
)

# ---- Dataset ----
class TweetDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels.tolist()

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = TweetDataset(train_encodings, train_labels)
val_dataset = TweetDataset(val_encodings, val_labels)

# ---- Model ----
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2,
    hidden_dropout_prob=0.15,
    attention_probs_dropout_prob=0.15
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# ================================================================
# KEY FIX 2: Heavily weighted loss for minority class
# ================================================================
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_labels),
    y=train_labels
)

# Further boost minority weight
class_weights[1] = class_weights[1] * 2.0
alpha = torch.tensor(class_weights, dtype=torch.float).to(device)
print("Alpha (boosted):", alpha)

# ---- Focal Loss with label smoothing ----
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, label_smoothing=0.03):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(
            logits, targets,
            reduction='none',
            label_smoothing=self.label_smoothing
        )
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.alpha is not None:
            alpha_t = self.alpha.gather(0, targets)
            focal_loss = alpha_t * focal_loss

        return focal_loss.mean()

# ---- Custom Trainer ----
class FocalTrainer(Trainer):
    def __init__(self, alpha, gamma=2, label_smoothing=0.03, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.focal_loss = FocalLoss(alpha=alpha, gamma=gamma, label_smoothing=label_smoothing)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss = self.focal_loss(logits, labels)
        return (loss, outputs) if return_outputs else loss

# ================================================================
# KEY FIX 3: Use macro F1 + threshold search + AUC
# ================================================================
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = F.softmax(torch.tensor(logits), dim=1).numpy()

    # Search best threshold
    best_f1 = 0
    best_thresh = 0.5
    for thresh in np.arange(0.20, 0.65, 0.01):
        preds_t = (probs[:, 1] > thresh).astype(int)
        _, _, f1_t, _ = precision_recall_fscore_support(
            labels, preds_t, average='macro', zero_division=0
        )
        if f1_t > best_f1:
            best_f1 = f1_t
            best_thresh = thresh

    preds = (probs[:, 1] > best_thresh).astype(int)

    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        labels, preds, average='macro', zero_division=0
    )
    precision_c1, recall_c1, f1_c1, _ = precision_recall_fscore_support(
        labels, preds, average='binary', zero_division=0
    )

    try:
        auc = roc_auc_score(labels, probs[:, 1])
    except:
        auc = 0.0

    print(f"\n--- Best Threshold: {best_thresh:.2f} ---")
    print(f"AUC-ROC: {auc:.4f}")
    print("\nClassification Report:\n")
    print(classification_report(labels, preds, zero_division=0))

    return {
        "f1_macro": f1_macro,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_class1": f1_c1,
        "precision_class1": precision_c1,
        "recall_class1": recall_c1,
        "auc_roc": auc,
        "best_threshold": best_thresh
    }

# ---- Training Args ----
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=15,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    learning_rate=1e-5,
    weight_decay=0.01,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    eval_strategy="epoch",
    save_strategy="epoch",
    fp16=True,
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",      # macro F1 balances BOTH classes
    greater_is_better=True,
    logging_steps=50,
    save_total_limit=3,
    gradient_accumulation_steps=2,
    report_to="none",
    dataloader_num_workers=4,
)

# ---- Train ----
trainer = FocalTrainer(
    alpha=alpha,
    gamma=2,
    label_smoothing=0.03,
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=4)]
)

trainer.train()

# ---- Save ----
trainer.save_model("./best_model")
tokenizer.save_pretrained("./best_model")
print("\nBest model saved to ./best_model")