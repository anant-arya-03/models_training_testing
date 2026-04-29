# models_training_testing

This repository holds the complete training and testing codebase for three machine learning models built to understand human language the way it is actually written — not just clean, textbook English, but the messy, beautiful, code-mixed language that people in India use every day online.

The core motivation behind this project was straightforward: most existing NLP models are trained on structured, formal text. But if you look at any Indian social media post, WhatsApp message, or news comment section, you will find a completely different linguistic reality — people switching between Hindi and English mid-sentence, writing in Roman script, mixing emotions and sarcasm in ways that standard models simply fail to understand. These three models were built to fix that.

All three are fine-tuned on top of **XLM-RoBERTa**, a multilingual transformer model from Facebook AI Research that was pre-trained on 100 languages and is particularly well-suited for code-mixed and low-resource language tasks.

---

## Live Demo

A frontend interface for all three models is live and open to use. You can paste any text — Hinglish, English, mixed, or even a news headline — and see the model's prediction in real time.

🔗 **[https://frontend-psi-eight-67.vercel.app/](https://frontend-psi-eight-67.vercel.app/)**

---

## Models on Hugging Face

All three trained models are publicly hosted on Hugging Face. You can use them directly via the Inference API or load them with the `transformers` library without needing to clone this repo or run any training yourself.

| Model | Hugging Face Link | Purpose |
|-------|------------------|---------|
| EmoSense | [Ansh1419/xlm-eno-sence-model](https://huggingface.co/Ansh1419/xlm-eno-sence-model) | Sentiment and emotion detection in Hinglish text |
| Fake News Detector | [Ansh1419/xlm-fakenews](https://huggingface.co/Ansh1419/xlm-fakenews) | Misinformation and fake news classification |
| CodeMix Classifier | [Ansh1419/xlm-roberta-codemix](https://huggingface.co/Ansh1419/xlm-roberta-codemix) | Code-mixed and mixed-script text classification |

> **Why are the models on Hugging Face and not here?**
> Model weight files like `model.safetensors` are extremely large — often several hundred MBs — and storing them directly in a GitHub repository is not practical. GitHub has strict file size limits, and large binaries bloat the repository history permanently. Hugging Face is purpose-built for hosting model weights and handles versioning, large file storage, and inference serving cleanly. So the full trained weights live there, while this repository holds the code used to produce them.

---

## A Note on Checkpoints

Inside the `results/` folders you will find training checkpoints saved at different steps. However, these checkpoints are **not complete model snapshots**. Each checkpoint folder contains only three files:

- `config.json` — the model architecture configuration
- `trainer_state.json` — training progress, loss curves, evaluation metrics logged at that step
- `training_args.bin` — the exact hyperparameters and training arguments used

The actual model weights (`model.safetensors` or `pytorch_model.bin`) are **not included** in the checkpoints here for the same reason as above — they are too large for GitHub. The fully trained and final model weights are available on Hugging Face (links above). These checkpoint files are kept here primarily for reproducibility and reference — so anyone looking at the training can see exactly how it progressed, what the loss looked like at each stage, and what arguments were used.

---

## Repository Structure

```
models_training_testing/
│
├── model_code_hinglish_data/             # EmoSense — Hinglish sentiment & emotion model
│   ├── src/
│   │   ├── train_model.py                # Full training script
│   │   ├── predict.py                    # Run inference on new text
│   │   └── new.ipynb                     # Exploratory notebook used during development
│   ├── results/
│   │   └── predictions.csv               # Prediction outputs from model evaluation
│   └── output_images/
│       ├── confusion_matrix_sentiment_emosence.png   # Confusion matrix from test run
│       └── traning_analysis_of_hinglish_data.png     # Loss and accuracy curves
│
├── model_code_meta_data_for_fake_news/   # Fake News Detection model
│   ├── src/
│   │   ├── train_model.py                # Training script
│   │   ├── test_model.py                 # Evaluation script
│   │   └── test_write.py                 # Writes test results to file
│   └── results/
│       ├── test_confusion_matrix.png     # Visual breakdown of predictions vs ground truth
│       ├── test_predictions.csv          # All test predictions
│       └── wrong_predictions.csv         # Cases the model got wrong — useful for error analysis
│
└── model_code_mix_data/                  # CodeMix classification model
    ├── train_pipeline.py                 # End-to-end training pipeline
    ├── predict.py                        # Inference on new input
    ├── prepare_test_data.py              # Data cleaning and preprocessing
    ├── an.py                             # Utility / analysis script
    ├── requirements.txt                  # All Python dependencies
    ├── output_images/                    # Training visualisations
    └── results/
        ├── checkpoint-4825/              # Saved at step 4825
        │   ├── config.json
        │   ├── trainer_state.json
        │   └── training_args.bin
        ├── checkpoint-7720/              # Saved at step 7720
        │   ├── config.json
        │   ├── trainer_state.json
        │   └── training_args.bin
        └── checkpoint-8685/             # Final checkpoint
            ├── config.json
            ├── trainer_state.json
            └── training_args.bin
```

---

## The Three Models — In Depth

### EmoSense — Hinglish Sentiment and Emotion Detection

Most sentiment analysis tools are built for English and fail the moment someone writes something like *"yaar ye toh zyada hi accha tha!"* or *"bro I'm so done with this honestly."* EmoSense was built specifically for that gap.

The model is trained on Hinglish text — the blend of Hindi and English that is native to the way millions of people in India actually communicate. It does not just detect positive or negative sentiment; it goes deeper into emotion categories, making it useful for applications like social media monitoring, product feedback analysis, and mental health tooling in Indian language contexts.

- **Base model:** XLM-RoBERTa
- **Language:** Hinglish (Hindi-English code-mixed, Roman script)
- **Task:** Sentiment classification and emotion detection
- **Hugging Face:** [Ansh1419/xlm-eno-sence-model](https://huggingface.co/Ansh1419/xlm-eno-sence-model)

---

### Fake News Detector — Misinformation and Fake News Classification

Misinformation is a real and growing problem, particularly in regional and multilingual contexts where fact-checking infrastructure is weaker. This model was trained to classify whether a given piece of news content or metadata is real or fake.

The approach uses XLM-RoBERTa's multilingual understanding to go beyond surface-level keyword matching. The `wrong_predictions.csv` file saved in results is particularly useful here — studying where the model fails gives a much clearer picture of the kinds of fake news it struggles with, which in turn guides future improvements.

- **Base model:** XLM-RoBERTa
- **Task:** Binary classification — Real vs Fake
- **Input:** News article text or metadata
- **Hugging Face:** [Ansh1419/xlm-fakenews](https://huggingface.co/Ansh1419/xlm-fakenews)

---

### CodeMix Classifier — Mixed-Script Text Classification

Code-mixing is not an accident or a mistake — it is how a huge population of internet users naturally communicates. This model is designed to classify and understand text where languages and scripts are freely combined. It handles the kind of content that breaks most standard NLP pipelines: sentences that start in one language and end in another, transliterated words, and hybrid vocabulary.

The training pipeline for this model is the most structured of the three, with a full `train_pipeline.py` and a separate `prepare_test_data.py` for preprocessing. The checkpoints saved at steps 4825, 7720, and 8685 track the model's learning progression — each one contains only the configuration files (not the weights), and the final model is hosted on Hugging Face.

- **Base model:** XLM-RoBERTa
- **Task:** Code-mixed language classification
- **Hugging Face:** [Ansh1419/xlm-roberta-codemix](https://huggingface.co/Ansh1419/xlm-roberta-codemix)

---

## Getting Started

### Install Dependencies

```bash
pip install -r model_code_mix_data/requirements.txt
```

The core libraries across all three models are:

- `transformers` — for loading XLM-RoBERTa and running fine-tuning
- `torch` — PyTorch as the deep learning backend
- `datasets` — HuggingFace datasets library for data handling
- `pandas` — data manipulation and CSV output
- `scikit-learn` — evaluation metrics like F1, precision, recall
- `matplotlib` / `seaborn` — for generating confusion matrices and training curves

### Running Inference

Each model has a `predict.py` file. Pass your input text and get a prediction:

```bash
# Hinglish emotion detection
python model_code_hinglish_data/src/predict.py

# Fake news classification
python model_code_meta_data_for_fake_news/src/test_model.py

# CodeMix classification
python model_code_mix_data/predict.py
```

### Training from Scratch

If you want to retrain any of these models on your own dataset:

```bash
# EmoSense
python model_code_hinglish_data/src/train_model.py

# Fake News
python model_code_meta_data_for_fake_news/src/train_model.py

# CodeMix
python model_code_mix_data/train_pipeline.py
```

Make sure your data is formatted correctly before running. For the CodeMix model, run the preprocessing step first:

```bash
python model_code_mix_data/prepare_test_data.py
```

---

## Using the Models Directly from Hugging Face

You do not need to clone this repo or run any training to use the models. Simply load them using the `transformers` library:

```python
from transformers import pipeline

# EmoSense — Hinglish sentiment
classifier = pipeline("text-classification", model="Ansh1419/xlm-eno-sence-model")
result = classifier("yaar ye toh bahut bura hua")
print(result)

# Fake News
classifier = pipeline("text-classification", model="Ansh1419/xlm-fakenews")
result = classifier("Government announces new policy on digital infrastructure")
print(result)

# CodeMix
classifier = pipeline("text-classification", model="Ansh1419/xlm-roberta-codemix")
result = classifier("bhai ye scene toh next level tha seriously")
print(result)
```

---

## Why XLM-RoBERTa?

XLM-RoBERTa is pre-trained on text from 100 languages using a masked language modelling objective — the same idea as BERT, but massively scaled across languages. This makes it particularly powerful for tasks involving non-English or mixed-language text, because it has genuinely learned cross-lingual representations rather than just translating everything to English internally.

For Hinglish and code-mixed tasks specifically, XLM-RoBERTa handles Roman-script Hindi naturally, understands context across language switches, and fine-tunes well even with relatively small domain-specific datasets. That made it the natural backbone choice for all three models here.

---

## Author

**Anant Arya**
GitHub: [@anant-arya-03](https://github.com/anant-arya-03)
Hugging Face: [Ansh1419](https://huggingface.co/Ansh1419)
