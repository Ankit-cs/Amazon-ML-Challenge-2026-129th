# Amazon ML Challenge - Entity Resolution Setup

Follow these instructions to set up the environment and run the Hybrid Entity Resolution model locally.

## 1. Prerequisites
Ensure you have Python 3.10+ installed on your system. 

## 2. Environment Setup

Open your terminal (PowerShell) and navigate to the project directory:
```powershell
cd C:\.amazon\AmazonML
```

Create a virtual environment (if you haven't already):
```powershell
python -m venv .venv
```

Activate the virtual environment:
```powershell
# On Windows
.\.venv\Scripts\Activate
```

## 3. Install Dependencies

Install the required packages from the requirements file. We also specifically ensure `protobuf` and `sentencepiece` are installed, as they are strictly required for the `DeBERTa-v3` tokenizer.

```powershell
pip install -r requirements.txt
pip install protobuf sentencepiece
```

*(Note: If you run into CUDA issues, install the specific PyTorch version for your GPU from the [official PyTorch website](https://pytorch.org/get-started/locally/))*

## 4. Run the Training Pipeline

Navigate to the `src` directory and run the training script. 
The script is configured with caching, so the 35+ minute data processing step is instantly bypassed on subsequent runs!

```powershell
cd src
python train.py
```

### What happens during training:
1. **Data Prep**: Loads the raw data from `data/raw`, generates candidate pairs using token matching, and mines "Hard Negatives".
2. **Local Validation**: Splits the dataset into 80% Training and 20% Validation.
3. **Training**: Trains the Hybrid DeBERTa-v3 Model (with manual feature cross-attention) using Mixed Precision (AMP) to prevent GPU memory overflows.
4. **Evaluation**: At the end of every epoch, it prints the exact **F1-Score, Precision, and Recall** on the validation set.
5. **Saving**: Saves the highest-scoring model to `models/hybrid_deberta_model_best.pt`.
