import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import gc
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

def train_epoch(model, loader, optimizer, scheduler, loss_fn, device):
    model.train()
    total_loss = 0
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())
    for batch in tqdm(loader, desc="Training"):
        optimizer.zero_grad()
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        features = batch['features'].to(device)
        targets = batch['label'].to(device)
        
        with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
            preds = model(input_ids, attention_mask, features)
            loss = loss_fn(preds, targets)
            
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        total_loss += loss.item()
        
    return total_loss / len(loader)

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds = []
    all_targets = []
    
    for batch in tqdm(loader, desc="Evaluating"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        features = batch['features'].to(device)
        targets = batch['label'].cpu().numpy()
        
        with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
            logits = model(input_ids, attention_mask, features)
            preds = (torch.sigmoid(logits) > 0.5).int().cpu().numpy()
            
        all_preds.extend(preds)
        all_targets.extend(targets)
        
    f1 = f1_score(all_targets, all_preds)
    precision = precision_score(all_targets, all_preds, zero_division=0)
    recall = recall_score(all_targets, all_preds, zero_division=0)
    return f1, precision, recall

def main():
    MODEL_NAME = "microsoft/deberta-v3-large"
    BATCH_SIZE = 4
    EPOCHS = 3
    LR = 2e-5
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data", "raw")
    print("Loading datasets and generating candidates from:", data_dir)
    records_a, records_b, features_list, labels = prepare_all_data(data_dir=data_dir)
    
    print("Splitting data into Training and Validation sets (80/20)...")
    train_idx, val_idx = train_test_split(range(len(labels)), test_size=0.2, random_state=42, stratify=labels)
    
    def get_subset(indices):
        return [records_a[i] for i in indices], [records_b[i] for i in indices], [features_list[i] for i in indices], [labels[i] for i in indices]
        
    tr_a, tr_b, tr_f, tr_l = get_subset(train_idx)
    vl_a, vl_b, vl_f, vl_l = get_subset(val_idx)
    
    print("Initializing exact 2026  logic model for Entity Resolution...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    train_dataset = EntityMatchDataset(tr_a, tr_b, tr_f, tr_l, tokenizer, max_length=128)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    val_dataset = EntityMatchDataset(vl_a, vl_b, vl_f, vl_l, tokenizer, max_length=128)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    model = HybridEntityResolutionModel(model_name=MODEL_NAME).to(DEVICE)
    
    optimizer = AdamW(model.parameters(), lr=LR)
    loss_fn = nn.BCEWithLogitsLoss()
    
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps)
    
    print("Beginning Training...")
    best_f1 = 0
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch+1}/{EPOCHS}")
        loss = train_epoch(model, train_loader, optimizer, scheduler, loss_fn, DEVICE)
        print(f"Epoch Loss: {loss:.4f}")
        
        f1, prec, rec = evaluate(model, val_loader, DEVICE)
        print(f"Validation -> F1-Score: {f1:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f}")
        
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), "../models/hybrid_deberta_model_best.pt")
            print("New best model saved!")
        
    print(f"Training Complete. Best Validation F1-Score: {best_f1:.4f}")

if __name__ == "__main__":
    main()
