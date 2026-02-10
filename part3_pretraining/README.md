# Part 3 – Task 4.1: Pretraining a Small Language Model

## Assignment Reference

- **Section:** 4.1 – Pretraining a Small Language Model  
- **Marks:** 20  

This module implements **Task 4.1** as specified in the assignment. A small Transformer-based language model is pretrained on cleaned retail conversation data generated in **Part 1 (DQ/DDQ)**.

---

## Execution Dependency (Mandatory)

> ⚠️ **Part 1 must be executed before running Part 3**

This module depends on the cleaned dataset produced in Part 1.  
Ensure the following file exists before execution:

```
part1_dq_ddq/cleaned_conv.csv
```

If this file is missing, Part 3 will not run successfully.

---

## Environment Setup

All required dependencies are listed in `requirements.txt`.

```bash
pip install -r requirements.txt
```

---

## Code Structure (As Required)

```
part3_pretraining/
├── pretraining.py          # Main pretraining pipeline
├── config.py               # Configuration parameters required for training
├── pretrained_model.pt     # Saved model checkpoint
├── loss_curves.png         # Training & validation loss curves
├── README.md               # Documentation
```

---

## 4.1.2 Dataset Preparation

### Data Source

- Cleaned retail conversation data from **Part 1 (DQ/DDQ)**

### Dataset Description

- Total conversations: ~1M  
- Approximate token count: ~250M  

### Train / Validation / Test Split

| Split | Percentage |
|------|------------|
| Train | 80% |
| Validation | 10% |
| Test | 10% |

A fixed random seed is used to ensure reproducibility.

---

### Tokenization

- **Type:** Word-level tokenization  
- **Method:** `text.split()`  
- Subword tokenization is intentionally not used  

### Vocabulary

- `<PAD>` – Padding token  
- `<UNK>` – Unknown token  

---

### Padding and Truncation

- **Maximum sequence length:** 256 tokens  
- Longer sequences are truncated  
- Shorter sequences are padded dynamically per batch  

---

## 4.2 Task 4.1 – Model Architecture

| Component | Value |
|----------|-------|
| Embedding dimension | 128 |
| Number of attention heads | 4 |
| Number of Transformer layers | 2 |

---

## Training Procedure

### Objective

- Next-token prediction  
- `x = tokens[:-1]`, `y = tokens[1:]`

---

### Training Configuration

| Parameter | Value |
|----------|-------|
| Epochs | 5 |
| Batch size | 32 |
| Optimizer | Adam |
| Learning rate | 1e-3 |

---

### Loss Function

- CrossEntropyLoss  
- Padding tokens ignored using `ignore_index`

---

## Validation and Evaluation

- Validation loss computed per epoch  
- Evaluation metric: **Perplexity**  

```
Perplexity = exp(Validation Loss)
```

---

## Loss Tracking and Visualization

- Training and validation loss tracked  
- Saved as `loss_curves.png`

---

## Model Checkpointing

- Saved as `pretrained_model.pt`

---

## How to Run

```bash
python -m part3_pretraining.pretraining
```

---

## Rubric Compliance Summary

| Requirement | Status |
|------------|--------|
| Cleaned data from Part 1 | ✅ |
| 80/10/10 split | ✅ |
| Word-level tokenization | ✅ |
| Max sequence length = 256 | ✅ |
| Padding & truncation | ✅ |
| 2-layer Transformer | ✅ |
| Train + validation | ✅ |
| Loss curves | ✅ |
| Perplexity | ✅ |
| Model saved | ✅ |

---

