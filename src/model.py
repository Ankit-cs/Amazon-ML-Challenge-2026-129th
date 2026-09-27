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
        hidden_size = self.backbone.config.hidden_size
        
        self.feat_embed = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Exact 3rd place architecture cross-attention
        self.cross_attn = CrossAttentionBlock(hidden_size, 128)
        self.dropout = nn.Dropout(0.2)
        
        # Classification head for Binary Match (0 or 1)
        self.classifier = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask, features):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        cls_emb = out.last_hidden_state[:, 0, :]
        
        feat_emb = self.feat_embed(features)
        
        fused = self.cross_attn(cls_emb, feat_emb)
        fused = self.dropout(fused)
        
        return self.classifier(fused).squeeze(-1)
