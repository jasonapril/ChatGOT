"""
Transformer model implementation for character-level language modeling.
"""
import math
import logging
from typing import Optional, Tuple, Union, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import base class, new config type, and registration decorator
from .base import LanguageModel, LanguageModelConfig 
from .factory import register_model


class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer models.
    
    This applies a sinusoidal encoding based on position in the sequence.
    """
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        """
        Initialize the positional encoding.
        
        Args:
            d_model: Dimension of the model
            dropout: Dropout probability
            max_len: Maximum sequence length
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply positional encoding to input.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, d_model]
            
        Returns:
            Tensor with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# Register this model implementation
@register_model("language", architecture_name="transformer")
class TransformerModel(LanguageModel):
    """
    Transformer model for character-level language modeling.
    Uses the standard nn.TransformerDecoderLayer and nn.TransformerDecoder.
    Configured via Pydantic LanguageModelConfig.
    """
    
    def __init__(self, config: LanguageModelConfig):
        """
        Initialize the transformer model using a LanguageModelConfig.
        """
        super().__init__(config)
        
        # Extract parameters from the validated Pydantic config
        self.vocab_size = config.vocab_size
        self.d_model = getattr(config, 'd_model', 768)
        self.n_head = getattr(config, 'n_head', 12)
        # Get d_hid, applying default only if it's missing OR explicitly None
        d_hid_from_config = getattr(config, 'd_hid', None)
        self.d_hid = d_hid_from_config if d_hid_from_config is not None else self.d_model * 4
        self.n_layers = getattr(config, 'n_layers', 12)
        self.dropout_rate = getattr(config, 'dropout', 0.1)
        self.max_seq_length = getattr(config, 'max_seq_length', 1024)
        self.layer_norm_eps = getattr(config, 'layer_norm_eps', 1e-5)
        self.activation = getattr(config, 'activation', 'gelu')
        self.bias = getattr(config, 'bias', True)
        self.norm_first = getattr(config, 'norm_first', True) # Ensure consistent param
        
        # --- Model Layers --- #
        # Token embedding
        self.token_embedding = nn.Embedding(self.vocab_size, self.d_model)
        
        # Position embedding (learnable)
        self.position_embedding = nn.Embedding(self.max_seq_length, self.d_model)

        # Dropout for embeddings
        self.dropout = nn.Dropout(self.dropout_rate)
        
        # Create standard transformer decoder layer
        decoder_layers = nn.TransformerDecoderLayer(
            d_model=self.d_model,
            nhead=self.n_head,
            dim_feedforward=self.d_hid,
            dropout=self.dropout_rate,
            activation=self.activation,
            batch_first=True, # Important: ensure batch dimension is first
            norm_first=self.norm_first, # Use pre-norm or post-norm based on config
            bias=self.bias,
            layer_norm_eps=self.layer_norm_eps
        )
        
        # Create standard transformer decoder
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layers,
            num_layers=self.n_layers,
            norm=nn.LayerNorm(self.d_model, eps=self.layer_norm_eps) # Final norm layer
        )
        
        # Output layer (ties weights with token embedding if configured)
        self.output_layer = nn.Linear(self.d_model, self.vocab_size, bias=False)
        # self.token_embedding.weight = self.output_layer.weight # Weight tying
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Log model size is handled by factory
        # self._log_model_size()
    
    def _init_weights(self, module):
        """Initialize the weights of the model."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            # Initialize bias only if it exists
            if module.bias is not None:
                module.bias.data.zero_()
            # Initialize weight (LayerNorm usually has weight if affine)
            if module.weight is not None: 
                module.weight.data.fill_(1.0)
    
    def _generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """
        Generate a square mask for the sequence (causal mask).
        Ensures attention is only paid to previous tokens.
        """
        # Use float('-inf') for positions to be masked
        mask = torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1)
        return mask
    
    def forward(self, x: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass of the TransformerModel.
        
        Args:
            x: Input token indices of shape [batch_size, seq_len]
            targets: Optional target token indices for loss calculation [batch_size, seq_len]
            
        Returns:
            Logits [batch_size, seq_len, vocab_size], or (Logits, Loss) if targets are provided.
        """
        batch_size, seq_len = x.size()
        if seq_len > self.max_seq_length:
            raise ValueError(f"Input sequence length ({seq_len}) exceeds model's max sequence length ({self.max_seq_length})")
        
        # Get token embeddings
        tok_emb = self.token_embedding(x) * math.sqrt(self.d_model) # Scale embedding
        
        # Get position embeddings
        positions = torch.arange(0, seq_len, dtype=torch.long, device=x.device)
        # positions = torch.clamp(positions, max=self.max_seq_length-1) # Clamp if using fixed PE
        pos_emb = self.position_embedding(positions) # [seq_len, d_model]
        
        # Combine embeddings and apply dropout
        x = self.dropout(tok_emb + pos_emb) # [batch_size, seq_len, d_model]
        
        # Create attention mask for the decoder
        # Shape should be [seq_len, seq_len]
        tgt_mask = self._generate_square_subsequent_mask(seq_len).to(x.device)
        
        # The standard TransformerDecoder expects no memory for decoder-only setup
        # It doesn't explicitly take a memory_mask or memory_key_padding_mask in this case.
        # tgt_key_padding_mask could be added if needed based on input padding.
        output = self.transformer_decoder(
            tgt=x,
            memory=torch.zeros((batch_size, 0, self.d_model), device=x.device), # No memory needed
            tgt_mask=tgt_mask,
            # memory_mask=None, # Not used without memory
            # tgt_key_padding_mask=None, # Optional: Add if input can be padded
            # memory_key_padding_mask=None # Not used without memory
        )
        
        # Generate logits
        logits = self.output_layer(output) # [batch_size, seq_len, vocab_size]
        
        # Calculate loss if targets provided
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1) # Ignore padding
            return logits, loss
        
        return logits
    
    # --- Remove generate method --- #
    # The generate method is now inherited from the GenerativeModel base class
    # def generate(...): 
    #    ... 