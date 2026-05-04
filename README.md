# TriTaskNLP

*A Multi-Task Learning Framework for Topic Classification, Sentiment Analysis, and Author Identification*

---

## Overview

TriTaskNLP is a unified Natural Language Processing system designed to perform **three core language understanding tasks simultaneously**:

* Topic Classification
* Sentiment Analysis
* Author Identification (Stylometry)

Instead of training separate models for each task, this project implements a **shared representation learning architecture**, where a single encoder learns generalized linguistic features that are reused across tasks. This approach improves efficiency, reduces redundancy, and enables knowledge transfer between tasks.

---

## Motivation

Traditional NLP pipelines treat tasks independently, leading to:

* Redundant feature extraction
* Increased computational cost
* Limited generalization

This project explores **multi-task learning (MTL)** as a solution, leveraging shared embeddings and encoders to learn richer, more transferable representations of text.

---

## System Architecture

The system follows a layered architecture with shared and task-specific components:

```
Input Text
   ↓
Preprocessing (Tokenization, Normalization)
   ↓
Embedding Layer (Word2Vec / Trainable Embeddings)
   ↓
Shared Encoder (BiLSTM / Transformer)
   ↓
Shared Representation Vector
   ↓
 ┌───────────────┬───────────────┬────────────────┐
 │ Topic Head    │ Sentiment Head│ Author Head    │
 │ (Softmax)     │ (Softmax)     │ (Softmax)      │
 └───────────────┴───────────────┴────────────────┘
```

### Key Design Choices

* **Shared Encoder**: Captures semantic, syntactic, and stylistic features
* **Task Heads**: Independent classifiers for each objective
* **Joint Loss Optimization**: Weighted combination of task losses

---

## Workflow

```
Data Collection → Preprocessing → Embedding Generation
        ↓
   Shared Encoder
        ↓
 Multi-Task Predictions
        ↓
 Evaluation & Visualization
```

The pipeline is designed to be modular, enabling easy experimentation with different encoders, embeddings, and datasets.

---

## Features

* Multi-task learning with shared representations
* GPU-accelerated training (CUDA support)
* Interactive CLI with structured dashboard
* Real-time training progress and visualization
* PCA, t-SNE, and clustering-based analysis
* Confusion matrix and performance metrics
* Stylometric feature integration for author identification

---

## Command Line Interface

The system includes a structured CLI built using `Typer` and `Rich`, offering both command-based and interactive usage.

### Launch Interactive Menu

```bash
python main.py
```

### Direct Commands

```bash
python main.py train
python main.py evaluate
python main.py predict "Sample text"
python main.py visualize
```

### CLI Capabilities

* Guided execution through menu interface
* Animated transitions and loading states
* Structured outputs using tables and panels
* Integrated visualization triggers

---

## Visualization & Analysis

The project includes multiple visualization techniques to interpret learned representations:

* **PCA (Principal Component Analysis)** for dimensionality reduction
* **t-SNE** for non-linear embedding visualization
* **K-Means Clustering** for grouping documents
* **Confusion Matrix** for classification performance

These tools help analyze how well the shared representation separates different tasks and classes.

---

## Model Training

Training is performed using a **multi-objective loss function**:

```
Loss = λ₁ * L_topic + λ₂ * L_sentiment + λ₃ * L_author
```

Where:

* Each component represents task-specific loss
* λ values control task importance

### Optimization Features

* CUDA acceleration
* Batch-wise training
* Live progress tracking
* Real-time loss visualization

---

## Project Structure

```
TriTaskNLP/
│── cli/                 # CLI commands
│── data/                # Dataset loaders
│── models/              # Model architecture
│── utils/               # Visualization, metrics, dashboard
│── train.py             # Training pipeline
│── evaluate.py          # Evaluation logic
│── main.py              # CLI entry point
│── requirements.txt
```

---

## Performance Expectations

| Task                  | Expected Accuracy |
| --------------------- | ----------------- |
| Topic Classification  | 92–96%            |
| Sentiment Analysis    | 90–94%            |
| Author Identification | 85–92%            |

Performance depends on dataset quality, embedding choice, and hyperparameter tuning.

---

## Trade-offs

While multi-task learning offers several advantages, it introduces certain challenges:

### Advantages

* Improved generalization
* Reduced model redundancy
* Efficient use of shared features

### Limitations

* Task interference (negative transfer)
* Complex loss balancing
* Higher architectural complexity

---

## Datasets

The model can be trained using combinations of:

* AG News (Topic Classification)
* IMDB / Amazon Reviews (Sentiment Analysis)
* PAN Author Profiling / Blog Corpus (Author Identification)

---

## References

* Analytics Vidhya — NLP preprocessing and embeddings
* Gensim documentation — topic modeling
* SpaCy documentation — NLP pipelines
* Research papers on multi-task learning in NLP

---

## Conclusion

TriTaskNLP demonstrates how a unified architecture can effectively handle multiple NLP tasks through shared representation learning. The project balances theoretical concepts with practical implementation, making it suitable for both academic evaluation and real-world applications.
