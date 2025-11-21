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
        class Conv1d: pass
        class ReLU: pass
        class Dropout: pass
        class Sequential: pass
        class Linear: pass
        class TransformerEncoderLayer: pass
        class TransformerEncoder: pass
        class utils:
            @staticmethod
            def weight_norm(x): return x
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

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.utils.weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.utils.weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TCNPredictor(nn.Module):
    """
    Temporal Convolutional Network (TCN) for Time Series Classification.
    Replaces standard LSTM for better gradient flow and parallelization.
    """
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TCNPredictor, self).__init__()
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is not available.")
            
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], 3) # 3 classes: Sell, Hold, Buy

    def forward(self, x):
        # x shape: (batch, seq_len, features) -> (batch, features, seq_len) for Conv1d
        x = x.transpose(1, 2)
        y = self.network(x)
        # Take last time step: (batch, channels, seq_len) -> (batch, channels)
        y = y[:, :, -1]
        return self.fc(y)

class AttentionNetwork(nn.Module):
    """
    Residual Transformer Block Architecture optimized for Time Series.
    Uses the last sequence element's embedding for prediction instead of global averaging.
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
        
        self.ln = nn.LayerNorm(hidden_dim)
        self.fc = nn.Linear(hidden_dim, 3) # 0: sell, 1: hold, 2: buy

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        x = self.embedding(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        
        # Take the last time step's embedding as the context for prediction
        # Shape: (batch, hidden_dim)
        last_step = x[:, -1, :]
        
        out = self.ln(last_step)
        return self.fc(out)
