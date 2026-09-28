<div align="center">

# Deep Learning Entity Resolution (Amazon ML Challenge 2026)

**Team VA | Secured 129th Position **

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

*An advanced Business Entity Resolution architecture built for extreme memory efficiency and numeric stability on consumer hardware.*

</div>

---

## Team Information
* **Vishal Singh**
* **Ankit Kumar**

---

## Executive Summary

Our team developed a **hybrid deep learning approach** combining **DeBERTa-v3** with engineered features through a **cross-attention fusion mechanism**.
This architecture is fully adapted for identifying matching entities (Business Entity Resolution tasks), resulting in robust and generalizable **matching prediction performance**.

---

## Methodology Overview

### Problem Analysis & Key Observations

Through extensive exploratory data analysis and systems profiling, we observed that **entity resolution** is influenced by both semantic similarity and string matching. 
We identified and resolved several critical systems-level bottlenecks:
- **Windows Deadlocks:** PyTorch DataLoader deadlocks caused by the Rust-based Hugging Face Fast Tokenizer.
- **Data Processing Overhead:** The data preparation and hard-negative mining step took excessive time (35+ minutes) initially.
- **VRAM Limitations:** Massive models like `DeBERTa-v3-large` easily run out of memory on consumer GPUs during training.
- **Numeric Instability:** PyTorch tensor mismatch bugs (`Half` and `Float`) frequently occur during cross-attention computations.

### Solution Strategy

**Approach Type:** Hybrid (Pretrained LM + Cross-Attention)

**Core Innovation:**
A robust architecture that fuses `DeBERTa-v3` embeddings via **cross-attention** for matching entities, optimized by data caching and mixed precision training.

**Dataset Insight:**
The 35+ minute data preparation and hard-negative mining step is cached to a `.pkl` file. Subsequent runs instantly load the pre-processed data in **under 10 seconds**!

### Design Tradeoffs & Precision Enhancements

To maximize model performance on consumer hardware, we navigated several critical design tradeoffs and implementation choices:
* **Precision over Speed (Gradient Accumulation):** By reducing our physical batch size to 4 and accumulating gradients to reach an effective batch size of 16, we traded off raw iteration speed to enable training the much more capable `DeBERTa-v3-large` architecture, drastically enhancing our final precision.
* **Mixed Precision (AMP) vs Numeric Stability:** Utilizing `torch.amp.autocast("cuda")` provided significant memory savings, allowing longer context lengths. However, this required custom `CrossAttentionBlock` casting to strictly prevent precision loss and `Half / Float` mismatches during our precision-sensitive cross-attention fusion steps.
* **Cached Hard Negative Mining:** Offloading complex string matching and hard-negative sampling to a preprocessed `.pkl` file traded upfront disk space for an immense acceleration in dataloading, freeing up the CPU completely so it wouldn't bottleneck the GPU.

**Loss Function:** We optimized the network using **BCEWithLogitsLoss** for binary classification of matching entities.

---

## Model Architecture

### Architecture Overview

| Stage                 | Model/Component                  | Target Metric |
| --------------------- | -------------------------------- | ------------- |
| Final Training        | DeBERTa-v3 + Cross-Attention     | F1-Score      |

### Model Components

**Text Processing Pipeline**
* **Preprocessing:** Data caching for hard-negative mining
* **Model:** DeBERTa-v3
* **Environment:** Windows Deadlock Fix implemented (`os.environ["TOKENIZERS_PARALLELISM"] = "false"`)

**Feature Engineering & Training Pipeline**
* **Optimization:** Gradient Accumulation reduces physical batch size to 4 but accumulates gradients over 4 steps (effective batch size 16).
* **Mixed Precision (AMP):** Uses the latest `torch.amp.autocast("cuda")` and `GradScaler`.
* **Dtype Stability:** Custom `CrossAttentionBlock` dynamically casts inputs to match weight dtypes.

---

## Model Performance

Our rigorous validation strategy ensured high confidence in our final submissions.

| Model                       | Validation F1-Score | Notes       |
| --------------------------- | ------------------- | ----------- |
| Pretrained DeBERTa Baseline | -                   | Text-only   |
| Final Hybrid Model          | **0.98812**         | 80/20 hold-out |
| Challenge Submission        | **0.98812**         | Final Entry |

---

## Repository Structure

```text
AmazonML/
├── data/
│   ├── raw/             # Training & Test TSV data & Cached .pkl file
│   └── processed/       # Any intermediate generated data
├── models/              # Checkpoints for the PyTorch models
├── src/                 # Main source code
│   ├── data_prep.py     # Hard negative mining, caching & candidate generation
│   ├── dataset.py       # PyTorch DataLoader logic
│   ├── features.py      # Fuzzy string metrics & Jaccard similarities
│   ├── model.py         # PyTorch DeBERTa + Cross Attention Hybrid Model
│   └── train.py         # Training loop with AMP, Evaluation & Accumulation
├── requirements.txt     # Python dependencies
├── setup.md             # Step-by-step setup and installation instructions
└── README.md
```

---

## How to Run the Training Pipeline

**1. Environment Setup & Dependencies**
Follow instructions in [setup.md](setup.md) to install requirements (`protobuf`, `sentencepiece`).
   
**2. Execute the Trainer**
```bash
cd src
python train.py
```

**3. Checkpoints**
Best validation weights are saved automatically as `hybrid_deberta_model_best.pt` in the `models/` directory.

---

<div align="center">
  <i>Developed during the Amazon ML Challenge 2026.</i>
</div>
