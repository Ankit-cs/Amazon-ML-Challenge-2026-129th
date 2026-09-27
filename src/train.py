import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import gc
import logging
from typing import Tuple, List

import torch
import torch.nn as nn
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score

from model import HybridEntityResolutionModel
from dataset import EntityMatchDataset
from data_prep import prepare_all_data

# Set up professional logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def train_epoch(model, loader, optimizer, scheduler, loss_fn, device, accumulation_steps=4):
    model.train()
    total_loss = 0
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())
    
    optimizer.zero_grad() # Initialize zero before loop
    
    for i, batch in enumerate(tqdm(loader, desc="Training")):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        features = batch['features'].to(device)
        targets = batch['label'].to(device)
        
        with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
            preds = model(input_ids, attention_mask, features)
            # Scale the loss by accumulation steps
            loss = loss_fn(preds, targets) / accumulation_steps
            
        scaler.scale(loss).backward()
        
        # Step optimizer only when we hit the accumulation steps
        if (i + 1) % accumulation_steps == 0 or (i + 1) == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()
            
        total_loss += loss.item() * accumulation_steps # Keep tracking original loss scale
        
    return total_loss / len(loader)

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_logits = []
    all_targets = []
    
    for batch in tqdm(loader, desc="Evaluating"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        features = batch['features'].to(device)
        targets = batch['label'].cpu().numpy()
        
        with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
            logits = model(input_ids, attention_mask, features)
            
        all_logits.extend(torch.sigmoid(logits).cpu().numpy())
        all_targets.extend(targets)
        
    all_logits = np.array(all_logits)
    all_targets = np.array(all_targets)
    
    # OPTIMIZATION: Threshold Tuning
    # Instead of blindly using 0.5, we scan thresholds to find the exact one that maximizes the F1-Score!
    best_f1 = 0
    best_thresh = 0.5
    for thresh in np.arange(0.1, 0.9, 0.05):
        preds = (all_logits > thresh).astype(int)
        f1 = f1_score(all_targets, preds)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
            
    # Calculate Precision & Recall at the best threshold
    final_preds = (all_logits > best_thresh).astype(int)
    precision = precision_score(all_targets, final_preds, zero_division=0)
    recall = recall_score(all_targets, final_preds, zero_division=0)
    
    logger.info(f"Optimal F1 Threshold found at: {best_thresh:.2f}")
    return best_f1, precision, recall

def main():
    MODEL_NAME = "microsoft/deberta-v3-large"
    BATCH_SIZE = 4
    EPOCHS = 3
    LR = 2e-5
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data", "raw")
    logger.info(f"Loading datasets and generating candidates from: {data_dir}")
    records_a, records_b, features_list, labels = prepare_all_data(data_dir=data_dir)
    
    logger.info("Splitting data into Training and Validation sets (80/20)...")
    train_idx, val_idx = train_test_split(range(len(labels)), test_size=0.2, random_state=42, stratify=labels)
    
    def get_subset(indices):
        return [records_a[i] for i in indices], [records_b[i] for i in indices], [features_list[i] for i in indices], [labels[i] for i in indices]
        
    tr_a, tr_b, tr_f, tr_l = get_subset(train_idx)
    vl_a, vl_b, vl_f, vl_l = get_subset(val_idx)
    
    logger.info("Initializing exact 2026 logic model for Entity Resolution...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    train_dataset = EntityMatchDataset(tr_a, tr_b, tr_f, tr_l, tokenizer, max_length=128)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    val_dataset = EntityMatchDataset(vl_a, vl_b, vl_f, vl_l, tokenizer, max_length=128)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    model = HybridEntityResolutionModel(model_name=MODEL_NAME).to(DEVICE)
    
    # OPTIMIZATION: Differential Learning Rates
    # DeBERTa is already pretrained, so it needs a very small learning rate (LR) so it doesn't "forget" its knowledge.
    # The CrossAttention and Classifier are newly initialized, so they need a much higher LR to learn quickly.
    backbone_params = [p for n, p in model.named_parameters() if "backbone" in n]
    custom_params = [p for n, p in model.named_parameters() if "backbone" not in n]
    
    optimizer = AdamW([
        {"params": backbone_params, "lr": LR},         # 2e-5 for DeBERTa
        {"params": custom_params, "lr": LR * 10}       # 2e-4 for the new layers
    ])
    loss_fn = nn.BCEWithLogitsLoss()
    
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps)
    
    logger.info("Beginning Training...")
    best_f1 = 0
    for epoch in range(EPOCHS):
        logger.info(f"Epoch {epoch+1}/{EPOCHS}")
        loss = train_epoch(model, train_loader, optimizer, scheduler, loss_fn, DEVICE)
        logger.info(f"Epoch Loss: {loss:.4f}")
        
        f1, prec, rec = evaluate(model, val_loader, DEVICE)
        logger.info(f"Validation -> F1-Score: {f1:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f}")
        
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), "../models/hybrid_deberta_model_best.pt")
            logger.info("New best model saved!")
        
    logger.info(f"Training Complete. Best Validation F1-Score: {best_f1:.4f}")

if __name__ == "__main__":
    main()
