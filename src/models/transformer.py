"""
Transformer model implementation for character-level language modeling.
"""
import math
import logging
from typing import Optional, Tuple, Union, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import LanguageModel, ModelConfig


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


class TransformerModel(LanguageModel):
    """
    Transformer model for character-level language modeling.
    
    This implements a decoder-only transformer (similar to GPT).
    """
    
    def __init__(
        self, 
        config: Optional[ModelConfig] = None,
        vocab_size: Optional[int] = None,
        d_model: int = 768,     # Standard GPT-2 Small
        n_head: int = 12,       # Standard GPT-2 Small
        d_hid: int = 3072,      # Standard GPT-2 Small
        n_layers: int = 12,     # Standard GPT-2 Small
        dropout: float = 0.1,   # Standard GPT-2 dropout
        max_seq_length: int = 1024,  # Standard GPT-2 context window
        layer_norm_eps: float = 1e-5,
        activation: str = 'gelu',
        bias: bool = True
    ):
        """
        Initialize the transformer model.
        
        Args:
            config: Model configuration object (takes precedence over other args)
            vocab_size: Size of the vocabulary
            d_model: Dimension of the model (embedding dimension)
            n_head: Number of attention heads
            d_hid: Dimension of the feedforward layer
            n_layers: Number of transformer layers
            dropout: Dropout probability
            max_seq_length: Maximum sequence length
            layer_norm_eps: Layer normalization epsilon
            activation: Activation function ('gelu' or 'relu')
            bias: Whether to use bias in linear layers
        """
        # Process configuration
        if config is None:
            # If vocab_size is not provided, raise an error
            if vocab_size is None:
                raise ValueError("Either config or vocab_size must be provided")
                
            config = ModelConfig(
                vocab_size=vocab_size,
                d_model=d_model,
                n_head=n_head,
                d_hid=d_hid,
                n_layers=n_layers,
                dropout=dropout,
                max_seq_length=max_seq_length,
                layer_norm_eps=layer_norm_eps,
                activation=activation,
                bias=bias,
                architecture="transformer"
            )
        elif vocab_size is not None:
            # If both config and vocab_size are provided, use vocab_size
            config.vocab_size = vocab_size
        
        super().__init__(config)
        
        # Extract parameters from config for easy access
        self.vocab_size = self.config.vocab_size
        self.d_model = self.config.d_model
        self.n_head = self.config.n_head
        self.d_hid = self.config.d_hid
        self.n_layers = self.config.n_layers
        self.max_seq_length = self.config.max_seq_length
        self.layer_norm_eps = self.config.layer_norm_eps
        self.activation = self.config.activation
        self.bias = self.config.bias
        self.dropout_rate = self.config.dropout
        
        # Token embedding
        self.token_embedding = nn.Embedding(self.vocab_size, self.d_model)
        
        # Position embedding
        self.position_embedding = nn.Embedding(self.max_seq_length, self.d_model)
        
        # Create transformer decoder layer
        decoder_layers = nn.TransformerDecoderLayer(
            d_model=self.d_model,
            nhead=self.n_head,
            dim_feedforward=self.d_hid,
            dropout=self.dropout_rate,
            activation=self.activation,
            batch_first=True,
            norm_first=True,
            bias=self.bias,
            layer_norm_eps=self.layer_norm_eps
        )
        
        # Create transformer decoder
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layers,
            num_layers=self.n_layers,
            norm=nn.LayerNorm(self.d_model, eps=self.layer_norm_eps)
        )
        
        # Output layer
        self.output_layer = nn.Linear(self.d_model, self.vocab_size, bias=self.bias)
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Log model size
        self._log_model_size()
    
    def _init_weights(self, module):
        """Initialize the weights of the model."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def _log_model_size(self):
        """Log the number of parameters in the model."""
        n_params = sum(p.numel() for p in self.parameters())
        logging.info(f"Model initialized with {n_params:,} parameters")
    
    def _generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """
        Generate a square mask for the sequence.
        
        The mask ensures that the prediction for position i
        can only depend on known elements in positions 0:i-1.
        
        Args:
            sz: Sequence length
            
        Returns:
            Mask tensor
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    def forward(self, x: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass of the model.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len]
            targets: Target tensor of shape [batch_size, seq_len] or None
            
        Returns:
            Output logits and optionally loss
        """
        batch_size, seq_len = x.size()
        
        # Get token embeddings
        token_embeddings = self.token_embedding(x)
        
        # Get position embeddings
        positions = torch.arange(0, seq_len, dtype=torch.long, device=x.device)
        # Clamp positions to max_seq_length-1 to prevent out-of-bounds issues
        positions = torch.clamp(positions, max=self.max_seq_length-1)
        positions = positions.unsqueeze(0).expand(batch_size, -1)
        position_embeddings = self.position_embedding(positions)
        
        # Combine embeddings
        x = token_embeddings + position_embeddings
        
        # Create attention mask to prevent attending to future tokens
        mask = self._generate_square_subsequent_mask(seq_len).to(x.device)
        
        # Pass through transformer
        output = self.transformer_decoder(
            tgt=x,
            memory=torch.zeros((batch_size, 1, self.d_model), device=x.device),
            tgt_mask=mask
        )
        
        # Generate logits
        logits = self.output_layer(output)
        
        # Calculate loss if targets provided
        if targets is not None:
            # Reshape for cross-entropy
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            return logits, loss
        
        return logits
    
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
        Generate text from the model.
        
        Args:
            input_ids: Input token ids of shape [batch_size, seq_len]
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature
            top_k: Number of highest probability tokens to keep for top-k sampling
            top_p: Probability threshold for top-p sampling
            repetition_penalty: Penalty for repeating tokens
            verbose: Whether to log progress
            
        Returns:
            Generated token ids
        """
        self.eval()
        batch_size = input_ids.shape[0]
        
        # Create list to store generated tokens
        generated_tokens = input_ids.clone()
        
        # Set past to None
        past = None
        
        # Generate tokens one by one
        for i in range(max_new_tokens):
            # Forward pass
            with torch.no_grad():
                logits = self(generated_tokens)
                
                # Get logits for the next token (last token in the sequence)
                next_token_logits = logits[:, -1, :]  # [batch_size, vocab_size]
                
                # Apply temperature
                next_token_logits = next_token_logits / temperature
                
                # Apply repetition penalty
                if repetition_penalty != 1.0:
                    for b in range(batch_size):
                        for token_id in generated_tokens[b]:
                            next_token_logits[b, token_id] /= repetition_penalty
                
                # Apply top-k filtering
                if top_k > 0:
                    # Get top-k values and indices
                    topk_values, topk_indices = torch.topk(next_token_logits, top_k)
                    
                    # Create filter mask
                    filter_mask = torch.zeros_like(next_token_logits, dtype=torch.bool)
                    
                    # Set top-k indices to True
                    for b in range(batch_size):
                        filter_mask[b, topk_indices[b]] = True
                    
                    # Set non-top-k values to -inf
                    next_token_logits = torch.where(filter_mask, next_token_logits, 
                                                  torch.tensor(-float("inf"), device=next_token_logits.device))
                
                # Apply top-p (nucleus) filtering
                if top_p < 1.0:
                    # Convert logits to probabilities
                    probs = F.softmax(next_token_logits, dim=-1)
                    
                    # Sort probabilities in descending order
                    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
                    
                    # Compute cumulative probabilities
                    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                    
                    # Create mask for probabilities to keep
                    keep_mask = cumulative_probs < top_p
                    
                    # Always keep at least one token
                    keep_mask[:, 0] = True
                    
                    # Create filter mask
                    filter_mask = torch.zeros_like(next_token_logits, dtype=torch.bool)
                    
                    # Populate filter mask
                    for b in range(batch_size):
                        filter_mask[b, sorted_indices[b, keep_mask[b]]] = True
                    
                    # Apply filter
                    next_token_logits = torch.where(filter_mask, next_token_logits, 
                                                  torch.tensor(-float("inf"), device=next_token_logits.device))
                
                # Convert logits to probabilities for sampling
                probs = F.softmax(next_token_logits, dim=-1)
                
                # Sample next token
                next_token = torch.multinomial(probs, num_samples=1)  # [batch_size, 1]
                
                # Add token to generated tokens
                generated_tokens = torch.cat([generated_tokens, next_token], dim=1)
                
                if verbose and (i + 1) % 10 == 0:
                    logging.info(f"Generated {i + 1}/{max_new_tokens} tokens")
        
        return generated_tokens
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get model configuration.
        
        Returns:
            Dictionary with model configuration
        """
        return {
            "model_type": self.model_type,
            "architecture": "transformer",
            "vocab_size": self.vocab_size,
            "d_model": self.d_model,
            "n_head": self.n_head,
            "d_hid": self.d_hid,
            "n_layers": self.n_layers,
            "dropout": self.dropout_rate,
            "max_seq_length": self.max_seq_length,
            "layer_norm_eps": self.layer_norm_eps,
            "activation": self.activation,
            "bias": self.bias
        }


def create_transformer_model(
    config: Optional[ModelConfig] = None,
    vocab_size: Optional[int] = None, 
    d_model: int = 768, 
    n_head: int = 12, 
    d_hid: int = 3072, 
    n_layers: int = 12, 
    dropout: float = 0.1, 
    max_seq_length: int = 1024, 
    layer_norm_eps: float = 1e-5, 
    activation: str = 'gelu', 
    bias: bool = True
) -> TransformerModel:
    """
    Create a transformer model.
    
    Args:
        config: Model configuration object (takes precedence over other args)
        vocab_size: Size of the vocabulary
        d_model: Dimension of the model (embedding dimension)
        n_head: Number of attention heads
        d_hid: Dimension of the feedforward layer
        n_layers: Number of transformer layers
        dropout: Dropout probability
        max_seq_length: Maximum sequence length
        layer_norm_eps: Layer normalization epsilon
        activation: Activation function ('gelu' or 'relu')
        bias: Whether to use bias in linear layers
        
    Returns:
        TransformerModel instance
    """
    return TransformerModel(
        config=config,
        vocab_size=vocab_size,
        d_model=d_model,
        n_head=n_head,
        d_hid=d_hid,
        n_layers=n_layers,
        dropout=dropout,
        max_seq_length=max_seq_length,
        layer_norm_eps=layer_norm_eps,
        activation=activation,
        bias=bias
    ) 