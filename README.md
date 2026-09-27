# Amazon ML Challenge 2026: Business Entity Resolution

This repository contains the advanced Deep Learning architecture built to solve the 2026 Amazon ML Business Entity Resolution Challenge. It implements the exact `DeBERTa-v3` + `Cross-Attention Fusion` architecture that secured 3rd place in the 2025 challenge, fully adapted for Entity Resolution (using `BCEWithLogitsLoss`).

## 🌟 Key Features & Updates
- **Local F1-Score Validation:** Automatically splits the training data (80/20) and calculates the **F1-Score, Precision, and Recall** at the end of every epoch.
- **Data Caching:** The 35+ minute data preparation and hard-negative mining step is cached to a `.pkl` file. Subsequent runs instantly load the pre-processed data in under 10 seconds!
- **Gradient Accumulation & Memory Optimization:** Reduces the physical batch size to 4 but accumulates gradients over 4 steps (effective batch size 16). This allows massive models like `DeBERTa-v3-large` to train on consumer GPUs without running Out of Memory.
- **Modern Mixed Precision (AMP):** Uses the latest `torch.amp.autocast("cuda")` and `GradScaler` to prevent float16 underflow/NaN errors.
- **Dtype Stability:** The custom `CrossAttentionBlock` dynamically casts inputs to match weight dtypes, permanently fixing PyTorch `Half and Float` tensor mismatch bugs.
- **Windows Deadlock Fix:** Implements `os.environ["TOKENIZERS_PARALLELISM"] = "false"` to prevent the Rust-based Hugging Face Fast Tokenizer from deadlocking the PyTorch DataLoader on Windows.

## 📁 Repository Structure
```
AmazonML/
│
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
└── README.md            # You are here
```

## 🚀 How to Run the Training Pipeline

1. **Environment Setup & Dependencies:**
   Follow the detailed instructions in [setup.md](setup.md) to set up your virtual environment and install the required dependencies (including the essential `protobuf` and `sentencepiece` packages).

2. **Execute the Trainer:**
   The training pipeline automatically loads the datasets (or the cache if it exists), splits the data, and begins training.
   ```bash
   cd src
   python train.py
   ```

3. **Checkpoints:**
   The model evaluates itself at the end of every epoch. The weights that achieve the highest Validation F1-Score will be saved automatically in the `models/` directory as `hybrid_deberta_model_best.pt`.
