import math
import logging
from bot_core.logger import get_logger

logger = get_logger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    # Dummy classes to prevent ImportErrors if torch is missing but this file is imported
    class nn:
        class Module: pass
        class LayerNorm: pass
    class F:
        @staticmethod
        def softmax(x, dim=1): return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        if not TORCH_AVAILABLE:
            return
            
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register as buffer (not a learnable parameter, but part of state_dict)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        # pe slice: (seq_len, d_model)
        # We unsqueeze pe to (1, seq_len, d_model) for broadcasting
        return x + self.pe[:x.size(1), :].unsqueeze(0)

class LSTMPredictor(nn.Module):
    """
    Enhanced LSTM with Layer Normalization for better stability.
    """
    def __init__(self, input_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is not available.")
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, 
                            dropout=dropout, batch_first=True)
        self.ln = nn.LayerNorm(hidden_dim)
        self.fc = nn.Linear(hidden_dim, 3) # 0: sell, 1: hold, 2: buy

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        _, (hn, _) = self.lstm(x)
        # Use the hidden state of the last layer: (batch, hidden_dim)
        out = hn[-1]
        out = self.ln(out)
        # Return LOGITS (CrossEntropyLoss expects logits, not probabilities)
        return self.fc(out)

class AttentionNetwork(nn.Module):
    """
    Residual Transformer Block Architecture.
    Embedding -> PosEnc -> [TransformerEncoderLayer] -> GlobalAvgPool -> FC
    """
    def __init__(self, input_dim, hidden_dim, num_layers, nhead, dropout):
        super().__init__()
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is not available.")
            
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim)
        
        # Transformer Encoder Layer includes Self-Attention, FeedForward, LayerNorm, and Residuals internally
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=nhead, batch_first=True, dropout=dropout
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.fc = nn.Linear(hidden_dim, 3) # 0: sell, 1: hold, 2: buy

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        x = self.embedding(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        # Global Average Pooling over sequence dimension
        x = x.mean(dim=1)
        # Return LOGITS
        return self.fc(x)
