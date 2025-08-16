"""
Transformer Model for Mathematical Symbol Recognition

This module implements a Transformer architecture optimized for stroke-based
mathematical symbol recognition from handwriting data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Optional, Tuple


class PositionalEncoding(nn.Module):
    """Positional encoding for stroke sequences."""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [seq_len, batch_size, d_model]
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class StrokeEmbedding(nn.Module):
    """Embedding layer for stroke features."""
    
    def __init__(self, input_dim: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.input_projection = nn.Linear(input_dim, d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Layer normalization for input stabilization
        self.layer_norm = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [batch_size, seq_len, input_dim]
        Returns:
            Tensor of shape [seq_len, batch_size, d_model]
        """
        # Project to model dimension
        x = self.input_projection(x) * math.sqrt(self.d_model)
        
        # Apply layer norm and dropout
        x = self.layer_norm(x)
        x = self.dropout(x)
        
        # Transpose for transformer (seq_len first)
        return x.transpose(0, 1)


class StrokeTransformerEncoder(nn.Module):
    """Multi-layer Transformer encoder for stroke sequences."""
    
    def __init__(self, 
                 d_model: int = 256,
                 nhead: int = 8,
                 num_layers: int = 6,
                 dim_feedforward: int = 1024,
                 dropout: float = 0.1,
                 activation: str = 'gelu'):
        super().__init__()
        
        # Single transformer encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=False,
            norm_first=True  # Pre-norm for better training stability
        )
        
        # Stack multiple layers
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers
        )
        
        self.d_model = d_model
    
    def forward(self, 
                src: torch.Tensor, 
                src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            src: Tensor of shape [seq_len, batch_size, d_model]
            src_key_padding_mask: Boolean mask [batch_size, seq_len] where True = ignore
        Returns:
            Tensor of shape [seq_len, batch_size, d_model]
        """
        return self.transformer_encoder(src, src_key_padding_mask=src_key_padding_mask)


class GlobalPooling(nn.Module):
    """Global pooling strategies for sequence aggregation."""
    
    def __init__(self, pooling_type: str = 'attention', d_model: int = 256):
        super().__init__()
        self.pooling_type = pooling_type
        
        if pooling_type == 'attention':
            # Learnable attention pooling
            self.attention = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.ReLU(),
                nn.Linear(d_model // 2, 1)
            )
        elif pooling_type == 'multihead_attention':
            # Multi-head attention pooling
            self.query = nn.Parameter(torch.randn(1, d_model))
            self.multihead_attn = nn.MultiheadAttention(d_model, num_heads=8, batch_first=False)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [seq_len, batch_size, d_model]
            mask: Optional padding mask [batch_size, seq_len]
        Returns:
            Tensor of shape [batch_size, d_model]
        """
        if self.pooling_type == 'mean':
            # Simple mean pooling
            if mask is not None:
                # Masked mean
                mask_expanded = mask.unsqueeze(-1).transpose(0, 1)  # [seq_len, batch_size, 1]
                x_masked = x.masked_fill(mask_expanded, 0.0)
                lengths = (~mask).sum(dim=1, keepdim=True).float()  # [batch_size, 1]
                return x_masked.sum(dim=0) / lengths.transpose(0, 1)
            else:
                return x.mean(dim=0)
        
        elif self.pooling_type == 'max':
            # Max pooling
            if mask is not None:
                mask_expanded = mask.unsqueeze(-1).transpose(0, 1)  # [seq_len, batch_size, 1]
                x_masked = x.masked_fill(mask_expanded, float('-inf'))
                return x_masked.max(dim=0)[0]
            else:
                return x.max(dim=0)[0]
        
        elif self.pooling_type == 'attention':
            # Attention-based pooling
            # x: [seq_len, batch_size, d_model] -> [batch_size, seq_len, d_model]
            x_transposed = x.transpose(0, 1)
            
            # Compute attention weights
            attention_weights = self.attention(x_transposed)  # [batch_size, seq_len, 1]
            
            if mask is not None:
                attention_weights = attention_weights.masked_fill(mask.unsqueeze(-1), float('-inf'))
            
            attention_weights = F.softmax(attention_weights, dim=1)
            
            # Weighted sum
            return (x_transposed * attention_weights).sum(dim=1)  # [batch_size, d_model]
        
        elif self.pooling_type == 'multihead_attention':
            # Multi-head attention pooling with learnable query
            batch_size = x.size(1)
            query = self.query.unsqueeze(1).repeat(1, batch_size, 1)  # [1, batch_size, d_model]
            
            # Apply multi-head attention
            attn_output, _ = self.multihead_attn(
                query, x, x, 
                key_padding_mask=mask
            )
            
            return attn_output.squeeze(0)  # [batch_size, d_model]
        
        else:
            raise ValueError(f"Unknown pooling type: {self.pooling_type}")


class ClassificationHead(nn.Module):
    """Classification head with dropout and residual connections."""
    
    def __init__(self, d_model: int, num_classes: int, dropout: float = 0.3):
        super().__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, num_classes)
        )
        
        # Alternative: simpler head
        # self.classifier = nn.Linear(d_model, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


class StrokeTransformer(nn.Module):
    """Complete Transformer model for mathematical symbol recognition."""
    
    def __init__(self,
                 input_dim: int = 8,  # [x, y, vx, vy, speed, direction, curvature, pen_state]
                 num_classes: int = 229,  # Number of symbol classes
                 d_model: int = 256,
                 nhead: int = 8,
                 num_encoder_layers: int = 6,
                 dim_feedforward: int = 1024,
                 dropout: float = 0.1,
                 max_len: int = 128,
                 pooling_type: str = 'attention'):
        super().__init__()
        
        self.d_model = d_model
        self.num_classes = num_classes
        
        # Components
        self.embedding = StrokeEmbedding(input_dim, d_model, dropout)
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)
        
        self.transformer = StrokeTransformerEncoder(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        
        self.pooling = GlobalPooling(pooling_type, d_model)
        self.classifier = ClassificationHead(d_model, num_classes, dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.constant_(module.bias, 0)
    
    def create_padding_mask(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Create padding mask for variable-length sequences.
        
        Args:
            x: Input tensor [batch_size, seq_len, input_dim]
            lengths: Optional sequence lengths [batch_size]
        
        Returns:
            Boolean mask [batch_size, seq_len] where True = padding
        """
        batch_size, seq_len = x.size(0), x.size(1)
        
        if lengths is not None:
            # Use provided lengths
            mask = torch.arange(seq_len, device=x.device).expand(batch_size, seq_len) >= lengths.unsqueeze(1)
        else:
            # Detect padding by checking if all features are zero
            # This assumes padding is done with zeros
            padding_mask = (x.abs().sum(dim=-1) == 0)  # [batch_size, seq_len]
            mask = padding_mask
        
        return mask
    
    def forward(self, 
                x: torch.Tensor, 
                lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor [batch_size, seq_len, input_dim]
            lengths: Optional sequence lengths [batch_size]
        
        Returns:
            Logits tensor [batch_size, num_classes]
        """
        # Create padding mask
        padding_mask = self.create_padding_mask(x, lengths)
        
        # Embedding and positional encoding
        embedded = self.embedding(x)  # [seq_len, batch_size, d_model]
        encoded = self.pos_encoding(embedded)
        
        # Transformer encoding
        transformer_out = self.transformer(encoded, src_key_padding_mask=padding_mask)
        
        # Global pooling
        pooled = self.pooling(transformer_out, padding_mask)
        
        # Classification
        logits = self.classifier(pooled)
        
        return logits
    
    def predict(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Make predictions with softmax.
        
        Args:
            x: Input tensor [batch_size, seq_len, input_dim]
            lengths: Optional sequence lengths [batch_size]
        
        Returns:
            Probabilities tensor [batch_size, num_classes]
        """
        with torch.no_grad():
            logits = self.forward(x, lengths)
            return F.softmax(logits, dim=-1)
    
    def get_attention_weights(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> dict:
        """
        Extract attention weights for visualization.
        
        Args:
            x: Input tensor [batch_size, seq_len, input_dim]
            lengths: Optional sequence lengths [batch_size]
        
        Returns:
            Dictionary with attention weights from each layer
        """
        attention_weights = {}
        
        # Create padding mask
        padding_mask = self.create_padding_mask(x, lengths)
        
        # Forward through embedding
        embedded = self.embedding(x)
        encoded = self.pos_encoding(embedded)
        
        # Extract attention weights from each transformer layer
        x_layer = encoded
        for i, layer in enumerate(self.transformer.transformer_encoder.layers):
            # Get self-attention weights
            # Note: This requires modifying the transformer layer to return attention weights
            # For now, we'll skip this implementation
            attention_weights[f'layer_{i}'] = None
        
        return attention_weights


def count_parameters(model: nn.Module) -> int:
    """Count total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_summary(model: nn.Module, input_size: Tuple[int, ...]) -> str:
    """Generate model summary string."""
    total_params = count_parameters(model)
    
    summary = f"""
StrokeTransformer Model Summary:
================================
Input shape: {input_size}
Total parameters: {total_params:,}
Model size: ~{total_params * 4 / 1024 / 1024:.2f} MB (float32)

Architecture:
- Embedding: {model.embedding.input_projection.in_features} -> {model.d_model}
- Transformer: {len(model.transformer.transformer_encoder.layers)} layers
- Model dimension: {model.d_model}
- Attention heads: {model.transformer.transformer_encoder.layers[0].self_attn.num_heads}
- Feed-forward dim: {model.transformer.transformer_encoder.layers[0].linear1.out_features}
- Output classes: {model.num_classes}
- Pooling: {model.pooling.pooling_type}
"""
    return summary


# Example usage and testing
if __name__ == "__main__":
    # Test model creation
    model = StrokeTransformer(
        input_dim=8,
        num_classes=229,
        d_model=256,
        nhead=8,
        num_encoder_layers=6
    )
    
    print(model_summary(model, (1, 128, 8)))
    
    # Test forward pass
    batch_size, seq_len, input_dim = 4, 128, 8
    x = torch.randn(batch_size, seq_len, input_dim)
    
    with torch.no_grad():
        output = model(x)
        print(f"Output shape: {output.shape}")
        print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")