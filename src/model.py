import torch
import torch.nn as nn
from transformers import AutoModel

class CrossAttentionBlock(nn.Module):
    def __init__(self, text_dim, feat_dim, num_heads=4):
        super().__init__()
        self.query_proj = nn.Linear(text_dim, text_dim)
        self.key_proj = nn.Linear(feat_dim, text_dim)
        self.value_proj = nn.Linear(feat_dim, text_dim)
        self.attn = nn.MultiheadAttention(embed_dim=text_dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(text_dim)
        self.ff = nn.Sequential(
            nn.Linear(text_dim, text_dim), 
            nn.ReLU(), 
            nn.Linear(text_dim, text_dim)
        )

    def forward(self, text_emb, feat_emb):
        # Prevent Half/Float mismatches by ensuring inputs match the weights
        dtype = self.query_proj.weight.dtype
        text_emb = text_emb.to(dtype)
        feat_emb = feat_emb.to(dtype)
        
        q = self.query_proj(text_emb).unsqueeze(1)
        k = self.key_proj(feat_emb).unsqueeze(1)
        v = self.value_proj(feat_emb).unsqueeze(1)
        attn_output, _ = self.attn(q, k, v)
        out = self.norm(text_emb + attn_output.squeeze(1))
        return out + self.ff(out)

class HybridEntityResolutionModel(nn.Module):
    def __init__(self, model_name="microsoft/deberta-v3-large", num_features=13):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        
        # 1. OPTIMIZATION: Gradient Checkpointing
        # Saves huge amounts of GPU VRAM, allowing larger effective batch sizes
        if hasattr(self.backbone, "gradient_checkpointing_enable"):
            self.backbone.gradient_checkpointing_enable()
            
        hidden_size = self.backbone.config.hidden_size
        
        self.feat_embed = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        self.cross_attn = CrossAttentionBlock(hidden_size, 128)
        
        # 2. OPTIMIZATION: Multi-Sample Dropout
        # We create 5 different dropouts. Averaging their predictions makes the model vastly more robust.
        self.dropouts = nn.ModuleList([nn.Dropout(0.1 + i * 0.05) for i in range(5)])
        self.classifier = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask, features):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        
        # 3. OPTIMIZATION: Mean Pooling (Superior to [CLS] token)
        # Instead of just taking the first token, we average all tokens weighted by the attention mask
        token_embeddings = out.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        pooled_emb = sum_embeddings / sum_mask
        
        feat_emb = self.feat_embed(features)
        
        # Cross Attention Fusion
        fused = self.cross_attn(pooled_emb, feat_emb)
        
        # Apply Multi-Sample Dropout and average the logits
        logits = torch.mean(torch.stack([self.classifier(dropout(fused)) for dropout in self.dropouts], dim=0), dim=0)
        
        return logits.squeeze(-1)
