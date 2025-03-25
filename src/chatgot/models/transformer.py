"""
Transformer Model Implementation for ChatGoT.

This module defines the transformer-based model architecture for character-level
text generation. The architecture features:

1. Efficient transformer implementation with multi-head attention
2. Memory-optimized embedding and position encoding
3. Customizable hyperparameters for different GPU constraints
4. Support for mixed precision and performance optimizations
"""

import logging
import math
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from chatgot.utils.logging import get_logger

logger = get_logger(__name__)


class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer models.
    
    This implementation can use either learned positional embeddings or
    fixed sinusoidal embeddings as in the original "Attention Is All You Need" paper.
    """
    def __init__(
        self,
        d_model: int,
        dropout: float = 0.1,
        max_len: int = 5000,
        encoding_type: str = "learned"
    ):
        """
        Initialize the positional encoding.
        
        Args:
            d_model: The model dimension
            dropout: Dropout rate
            max_len: Maximum sequence length
            encoding_type: Type of positional encoding, either "learned" or "sinusoidal"
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.encoding_type = encoding_type
        
        if encoding_type == "learned":
            self.position_embeddings = nn.Embedding(max_len, d_model)
            self.register_buffer("position_ids", torch.arange(max_len).expand((1, -1)))
        else:  # sinusoidal
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
            )
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0)
            self.register_buffer("pe", pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply positional encoding to the input.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, d_model]
            
        Returns:
            Tensor with positional encoding applied
        """
        if self.encoding_type == "learned":
            position_ids = self.position_ids[:, :x.size(1)]
            position_embeddings = self.position_embeddings(position_ids)
            x = x + position_embeddings
        else:  # sinusoidal
            x = x + self.pe[:, :x.size(1)]
            
        return self.dropout(x)


class TransformerModel(nn.Module):
    """
    Character-level transformer model for text generation.
    
    This model implements a standard transformer architecture with customizable
    hyperparameters to fit different GPU memory constraints. It includes options
    for attention masking, dropout rates, and activation functions.
    """
    def __init__(
        self, 
        vocab_size: int, 
        d_model: int = 768,     # Standard GPT-2 Small
        n_head: int = 12,       # Standard GPT-2 Small
        d_hid: int = 3072,      # Standard GPT-2 Small
        n_layers: int = 12,     # Standard GPT-2 Small
        dropout: float = 0.1,   # Standard GPT-2 dropout
        max_seq_length: int = 1024,  # Standard GPT-2 context window
        layer_norm_eps: float = 1e-5,
        memory_efficient: bool = False,
        use_activation_checkpointing: bool = False,
        pos_encoding_type: str = "learned"
    ):
        """
        Initialize the transformer model using GPT-2 Small architecture.
        
        Args:
            vocab_size: Size of the character vocabulary
            d_model: Dimension of the model (embedding dimension)
            n_head: Number of attention heads
            d_hid: Dimension of the feedforward hidden layer
            n_layers: Number of transformer encoder layers
            dropout: Dropout probability
            max_seq_length: Maximum sequence length
            layer_norm_eps: Layer normalization epsilon value
            memory_efficient: Whether to use memory-efficient attention
            use_activation_checkpointing: Whether to use activation checkpointing
            pos_encoding_type: Type of positional encoding ("learned" or "sinusoidal")
        """
        super().__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_seq_length = max_seq_length
        self.use_activation_checkpointing = use_activation_checkpointing
        
        # Initialize embedding layer
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(
            d_model, 
            dropout, 
            max_seq_length,
            encoding_type=pos_encoding_type
        )
        
        # Create transformer encoder layer
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=n_head, 
            dim_feedforward=d_hid, 
            dropout=dropout,
            activation="gelu",  # GELU tends to work better than ReLU for language models
            batch_first=True,   # Use batch-first convention
            norm_first=True,    # Use Pre-LN for better training stability
            layer_norm_eps=layer_norm_eps
        )
        
        # Create transformer encoder
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers, 
            num_layers=n_layers,
            norm=nn.LayerNorm(d_model, eps=layer_norm_eps)
        )
        
        # Output layer
        self.decoder = nn.Linear(d_model, vocab_size)
        
        # Initialize parameters with small values
        self._init_weights()
        
        # No mask is needed for training (self-attention handles this automatically)
        self.src_mask = None
        
        # Log model size at initialization
        self._log_model_size()
        
        # Log if using activation checkpointing
        if self.use_activation_checkpointing:
            logger.info("Using activation checkpointing to reduce memory usage")
    
    def _init_weights(self):
        """Initialize the weights with small values for better training stability."""
        init_range = 0.02
        self.embedding.weight.data.normal_(mean=0.0, std=init_range)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.normal_(mean=0.0, std=init_range)
    
    def _log_model_size(self):
        """Log the model size in terms of parameter count."""
        num_params = sum(p.numel() for p in self.parameters())
        logger.info(f"Initialized transformer model with {num_params:,} parameters")
        logger.info(f"Model config: d_model={self.d_model}, vocab_size={self.vocab_size}")
    
    def _generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """Generate a square attention mask for sequence data.
        
        The mask ensures that predictions for position i can only depend on known
        outputs at positions less than i (for auto-regressive generation).
        
        Args:
            sz: Sequence length
            
        Returns:
            A lower triangular matrix with -inf in the upper triangular part
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    def forward(
        self, 
        src: torch.Tensor, 
        src_key_padding_mask: Optional[torch.Tensor] = None, 
        is_causal: bool = False
    ) -> torch.Tensor:
        """
        Forward pass for the transformer model.
        
        Args:
            src: Input tensor of token indices [batch_size, seq_len]
            src_key_padding_mask: Mask for padding tokens (1 for padding)
            is_causal: Whether to use causal masking for autoregressive generation
            
        Returns:
            Output tensor of logits [batch_size, seq_len, vocab_size]
        """
        # Create causal mask if needed
        if is_causal:
            src_mask = self._generate_square_subsequent_mask(src.size(1)).to(src.device)
        else:
            src_mask = None
        
        # Pass input through embedding and positional encoding layers
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.positional_encoding(src)
        
        # Pass through transformer encoder
        output = self.transformer_encoder(
            src, 
            mask=src_mask, 
            src_key_padding_mask=src_key_padding_mask
        )
        
        # Pass through final linear layer
        output = self.decoder(output)
        
        return output
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 0.9,
        repetition_penalty: float = 1.0,
        verbose: bool = False
    ) -> torch.Tensor:
        """
        Generate text using the model with various sampling strategies.
        
        This function implements an auto-regressive generation loop with
        several decoding strategies:
        - Temperature sampling
        - Top-k sampling (if top_k > 0)
        - Nucleus/top-p sampling (if top_p < 1.0)
        - Repetition penalty to discourage repeating the same tokens
        
        Args:
            input_ids: Input tensor of token indices
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature (1.0 = no change, <1.0 = less random, >1.0 = more random)
            top_k: Keep only the top k tokens with highest probability (0 = disabled)
            top_p: Keep tokens comprising the top p probability mass (1.0 = disabled)
            repetition_penalty: Penalty for repeating tokens (1.0 = no penalty)
            verbose: Whether to log generation progress
            
        Returns:
            Tensor containing the generated token indices
        """
        # Make sure model is in eval mode
        self.eval()
        
        # Store the original input for returning later
        original_input = input_ids.clone()
        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        # Ensure we don't modify the original input
        input_ids = input_ids.clone()
        
        # Track unique tokens for repetition penalty
        prev_tokens = [[] for _ in range(batch_size)]
        
        # Generate tokens one by one
        for i in range(max_new_tokens):
            # Limit the sequence length to avoid out-of-memory issues
            if input_ids.shape[1] > self.max_seq_length:
                input_ids = input_ids[:, -self.max_seq_length:]
            
            # Forward pass with causal masking
            with torch.no_grad():
                outputs = self.forward(input_ids, is_causal=True)
                
                # Get the next token logits (last position)
                next_token_logits = outputs[:, -1, :]
                
                # Apply temperature scaling
                if temperature > 0:
                    next_token_logits = next_token_logits / temperature
                
                # Apply repetition penalty
                if repetition_penalty > 1.0:
                    for b in range(batch_size):
                        for prev_token in prev_tokens[b]:
                            next_token_logits[b, prev_token] /= repetition_penalty
                
                # Apply top-k sampling
                if top_k > 0:
                    # Zero out all logits below the top k
                    values, _ = torch.topk(next_token_logits, top_k)
                    min_values = values[:, -1].unsqueeze(-1).expand_as(next_token_logits)
                    next_token_logits = torch.where(
                        next_token_logits < min_values,
                        torch.ones_like(next_token_logits) * float('-inf'),
                        next_token_logits
                    )
                
                # Apply top-p (nucleus) sampling
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True, dim=-1)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    
                    # Shift the indices to the right to keep the first token above threshold
                    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                    sorted_indices_to_remove[:, 0] = 0
                    
                    # Scatter sorted tensors to original indexing
                    indices_to_remove = sorted_indices_to_remove.scatter(
                        1, sorted_indices, sorted_indices_to_remove
                    )
                    next_token_logits = next_token_logits.masked_fill(indices_to_remove, float('-inf'))
                
                # Apply softmax to get probabilities
                probs = F.softmax(next_token_logits, dim=-1)
                
                # Sample from the distribution
                next_tokens = torch.multinomial(probs, num_samples=1)
            
            # Update inputs with the new token (auto-regressive)
            input_ids = torch.cat([input_ids, next_tokens], dim=-1)
            
            # Add to previous tokens for repetition penalty
            for b in range(batch_size):
                prev_tokens[b].append(next_tokens[b, 0].item())
                
                # Limit the history to the last 25 tokens for efficiency
                if len(prev_tokens[b]) > 25:
                    prev_tokens[b] = prev_tokens[b][-25:]
            
            if verbose and (i+1) % 10 == 0:
                logger.info(f"Generated {i+1}/{max_new_tokens} tokens")
        
        return input_ids


def create_transformer_model(
    vocab_size: int,
    max_seq_length: int = 1024,  # Standard GPT-2 context window
    d_model: int = 768,     # Standard GPT-2 Small embedding dimension
    n_head: int = 12,        # Standard GPT-2 Small head count
    d_hid: int = 3072,      # Standard GPT-2 Small feedforward dimension
    n_layers: int = 12,     # Standard GPT-2 Small layer count
    dropout: float = 0.1,   # Standard GPT-2 dropout
    layer_norm_eps: float = 1e-5,
    memory_efficient: bool = True,
    use_activation_checkpointing: bool = False,
    pos_encoding_type: str = "learned"
) -> TransformerModel:
    """
    Create a transformer model following the standard GPT-2 Small architecture.
    
    Args:
        vocab_size: Size of the character vocabulary
        max_seq_length: Maximum sequence length
        d_model: Dimension of the model (embedding dimension)
        n_head: Number of heads in the multi-head attention
        d_hid: Dimension of the feedforward layer
        n_layers: Number of transformer layers
        dropout: Dropout probability
        layer_norm_eps: Layer normalization epsilon
        memory_efficient: Whether to use memory-efficient attention
        use_activation_checkpointing: Whether to use activation checkpointing to save memory
        pos_encoding_type: Type of positional encoding ("learned" or "sinusoidal")
        
    Returns:
        A TransformerModel instance
    """
    model = TransformerModel(
        vocab_size=vocab_size,
        d_model=d_model,
        n_head=n_head,
        d_hid=d_hid,
        n_layers=n_layers,
        dropout=dropout,
        max_seq_length=max_seq_length,
        layer_norm_eps=layer_norm_eps,
        memory_efficient=memory_efficient,
        use_activation_checkpointing=use_activation_checkpointing,
        pos_encoding_type=pos_encoding_type
    )
    
    # Log parameters
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Created transformer model with {num_params:,} parameters")
    logger.info(f"Model config: d_model={d_model}, n_head={n_head}, d_hid={d_hid}, n_layers={n_layers}")
    logger.info(f"Using {'memory-efficient' if memory_efficient else 'standard'} implementation")
    
    return model 