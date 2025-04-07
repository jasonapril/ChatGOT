import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional

# Import base class, config class, and registration decorator
from .base import LanguageModel
from ..config.schemas import SimpleRNNConfig # Config location changed

# Register this model implementation
# @register_model(name="simple_rnn", config_cls=SimpleRNNConfig)
class SimpleRNN(LanguageModel):
    """
    A simple RNN-based language model.
    """
    def __init__(self, config: SimpleRNNConfig):
        super().__init__(config) # Pass the specific config type
        self.config = config # Store config for easy access in forward

        # Embedding layer
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        
        # RNN layer - input size is the embedding dimension (d_model)
        self.rnn = nn.RNN(
            input_size=config.d_model, 
            hidden_size=config.hidden_size, 
            num_layers=config.num_layers, 
            batch_first=True,
            dropout=config.dropout if config.num_layers > 1 else 0 # Add dropout if multiple layers
        )
        
        # Output layer - maps hidden state to vocabulary size
        self.fc = nn.Linear(config.hidden_size, config.vocab_size)
        
        # Apply weight initialization (optional but good practice)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.xavier_uniform_(module.weight)
        elif isinstance(module, nn.RNN):
            for name, param in module.named_parameters():
                if 'bias' in name:
                    torch.nn.init.zeros_(param)
                elif 'weight' in name:
                    torch.nn.init.xavier_uniform_(param)

    def forward(self, x: torch.Tensor, targets: Optional[torch.Tensor] = None):
        """
        Forward pass for the RNN model.
        Args:
            x: Input token indices of shape [batch_size, seq_len]
            targets: Optional target token indices for loss calculation [batch_size, seq_len]
        Returns:
            Logits [batch_size, seq_len, vocab_size], or (Logits, Loss) if targets are provided.
        """
        batch_size, seq_len = x.shape
        
        # 1. Get embeddings
        emb = self.embedding(x) # [batch_size, seq_len, d_model]
        
        # 2. Pass through RNN
        # Initialize hidden state
        h0 = torch.zeros(self.config.num_layers, batch_size, self.config.hidden_size, device=x.device)
        # Get RNN outputs
        rnn_out, _ = self.rnn(emb, h0) # rnn_out shape: [batch_size, seq_len, hidden_size]
        
        # 3. Pass through final linear layer to get logits
        logits = self.fc(rnn_out) # [batch_size, seq_len, vocab_size]
        
        # 4. Calculate loss if targets are provided
        loss = None
        if targets is not None:
            # Reshape logits and targets for cross_entropy
            # Logits: [batch_size * seq_len, vocab_size]
            # Targets: [batch_size * seq_len]
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1) # Assuming -1 is padding index
            return logits, loss
            
        return logits 