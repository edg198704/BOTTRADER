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
    class F:
        @staticmethod
        def softmax(x, dim=1): return x

class LSTMPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is not available.")
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, 
                            dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 3) # 0: sell, 1: hold, 2: buy

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        # Use the hidden state of the last layer
        return F.softmax(self.fc(hn[-1]), dim=1)

class AttentionNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, nhead, dropout):
        super().__init__()
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is not available.")
        self.embedding = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=nhead, batch_first=True, dropout=dropout
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(hidden_dim, 3) # 0: sell, 1: hold, 2: buy

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        x = x.mean(dim=1)
        return F.softmax(self.fc(x), dim=1)
