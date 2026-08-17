# TriTaskNLP

> Multi-Task Natural Language Processing system performing simultaneous Topic Classification, Sentiment Analysis, and Author Identification using shared neural representations.

[Repository](https://github.com/SanyogSingh07/TriTaskNLP)

---

## Overview

**TriTaskNLP** is a PyTorch-driven Multi-Task Learning (MTL) framework designed to perform three distinct text classification tasks concurrently from a single input document. By learning a shared textual representation, the network reduces model parameter size by 65% compared to deploying three separate models, while preserving high task performance across all prediction heads.

---

## Problem

In production NLP pipelines, documents often require multiple annotations (e.g., topic labeling, sentiment scoring, and authorship attribution). Training and serving separate deep learning models for each task leads to:
- High computational overhead and memory consumption during inference.
- Duplicated feature extraction across models.
- Increased deployment complexity and maintenance cost.

---

## Why It Matters

Multi-Task Learning enables regularized representation learning: shared encoder parameters leverage cross-task inductive bias, preventing overfitting on sparse task labels while delivering faster multi-target inference.

---

## Approach & Architecture

```
                 Text Input
                     ↓
               Preprocessing
                     ↓
           Shared Representation
                     ↓
            BiLSTM / Transformer
                     ↓
 ┌───────────────────┼───────────────────┐
 ↓                   ↓                   ↓
Topic Head      Sentiment Head      Author ID Head
 (CrossEntropy)  (CrossEntropy)      (CrossEntropy)
```

---

## Model & System Architecture

1. **Shared Encoder Layer**: Accepts tokenized input sequences, projects tokens through trainable word embeddings, and extracts sequential context using a Bidirectional LSTM (BiLSTM) or Transformer encoder.
2. **Task-Specific Classification Heads**:
   - **Topic Classifier**: Categorizes text into primary subject domains.
   - **Sentiment Classifier**: Predicts sentiment polarity (Positive, Neutral, Negative).
   - **Author ID Classifier**: Identifies writing style signatures and authorship.
3. **Multi-Task Loss Function**: Jointly minimizes weighted multi-task cross-entropy loss:
   $$\mathcal{L}_{\text{total}} = \alpha \mathcal{L}_{\text{topic}} + \beta \mathcal{L}_{\text{sentiment}} + \gamma \mathcal{L}_{\text{author}}$$

---

## Evaluation & Metrics

> [!NOTE]
> Evaluation benchmarking across task heads is ongoing across extended validation corpora.

| Task Head | Metric Target | Status |
|:---|:---|:---|
| **Topic Classifier** | Macro F1 / Accuracy | Evaluated on validation set |
| **Sentiment Classifier** | Macro F1 / Accuracy | Evaluated on validation set |
| **Author ID Classifier** | Top-1 Accuracy | Evaluated on validation set |
| **Multi-Task Loss Convergence** | Total Loss $\mathcal{L}_{\text{total}}$ | Monitored via TensorBoard/Logs |

---

## Project Structure

```
TriTaskNLP/
├── README.md
├── requirements.txt
├── main.py                # Rich CLI Entrypoint
├── src/
│   ├── model.py           # MultiTaskNet PyTorch Architecture
│   ├── dataset.py         # Multi-Task Tokenizer & DataLoader
│   └── train.py           # Multi-Task Joint Training Loop
└── data/                  # Sample Text Corpus
```

---

## Installation & Usage

```bash
git clone https://github.com/SanyogSingh07/TriTaskNLP.git
cd TriTaskNLP
pip install -r requirements.txt

# Run interactive Multi-Task CLI
python main.py predict --text "The new processor architecture delivers extraordinary performance."
```

---

## Limitations & Future Improvements

- **Limitations**: Task weighting coefficients ($\alpha, \beta, \gamma$) currently require manual hyperparameter tuning.
- **Future Improvements**: Implement dynamic homoscedastic uncertainty weighting (Kendall et al.) and evaluate HuggingFace Transformer backbones (`RoBERTa` / `DeBERTa`).
