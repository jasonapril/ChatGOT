"""
Transformer model implementation for character-level language modeling.
"""
import math
import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


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


class TransformerModel(nn.Module):
    """
    Transformer model for character-level language modeling.
    
    This implements a decoder-only transformer (similar to GPT).
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
        activation: str = 'gelu',
        bias: bool = True
    ):
        """
        Initialize the transformer model.
        
        Args:
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
        super().__init__()
        
        # Store parameters
        self.d_model = d_model
        self.n_head = n_head
        self.max_seq_length = max_seq_length
        
        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # Position embedding
        self.position_embedding = nn.Embedding(max_seq_length, d_model)
        
        # Create transformer decoder layer
        decoder_layers = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=d_hid,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=True,
            bias=bias,
            layer_norm_eps=layer_norm_eps
        )
        
        # Create transformer decoder
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layers,
            num_layers=n_layers,
            norm=nn.LayerNorm(d_model, eps=layer_norm_eps)
        )
        
        # Output layer
        self.output_layer = nn.Linear(d_model, vocab_size, bias=bias)
        
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
    
    def forward(self, x: torch.Tensor, targets: Optional[torch.Tensor] = None) -> torch.Tensor:
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
        
        for i in range(max_new_tokens):
            # If input_ids exceeds max length, truncate it
            if input_ids.size(1) > self.max_seq_length:
                input_ids = input_ids[:, -self.max_seq_length:]
            
            # Forward pass to get logits
            with torch.no_grad():
                logits = self(input_ids)
                if isinstance(logits, tuple):
                    logits = logits[0]  # Extract logits from (logits, loss) tuple
                logits = logits[:, -1, :]  # Get logits for the last token
            
            # Apply temperature
            logits = logits / temperature
            
            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for b in range(batch_size):
                    for token_id in set(input_ids[b].tolist()):
                        logits[b, token_id] /= repetition_penalty
            
            # Apply top-k filtering
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float('-inf')
            
            # Apply top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift the indices to the right to keep also the first token above the threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                # Scatter sorted tensors to original indexing
                indices_to_remove = sorted_indices_to_remove.scatter(
                    dim=1,
                    index=sorted_indices,
                    src=sorted_indices_to_remove
                )
                logits[indices_to_remove] = float('-inf')
            
            # Sample from the distribution
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append new token to the sequence
            input_ids = torch.cat((input_ids, next_token), dim=1)
            
            # Log progress if verbose
            if verbose and (i + 1) % 10 == 0:
                logging.info(f"Generated {i+1}/{max_new_tokens} tokens")
        
        # Return the full sequence
        return input_ids


def create_transformer_model(
    vocab_size, 
    d_model=768, 
    n_head=12, 
    d_hid=3072, 
    n_layers=12, 
    dropout=0.1, 
    max_seq_length=1024, 
    layer_norm_eps=1e-5, 
    activation='gelu', 
    bias=True
):
    """
    Create a transformer model with the specified parameters.
    
    Args:
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
        A TransformerModel instance
    """
    # Log the model configuration
    logging.info(f"Creating transformer model with configuration:")
    logging.info(f"  vocab_size: {vocab_size}")
    logging.info(f"  d_model: {d_model}")
    logging.info(f"  n_head: {n_head}")
    logging.info(f"  d_hid: {d_hid}")
    logging.info(f"  n_layers: {n_layers}")
    logging.info(f"  dropout: {dropout}")
    logging.info(f"  max_seq_length: {max_seq_length}")
    logging.info(f"  layer_norm_eps: {layer_norm_eps}")
    logging.info(f"  activation: {activation}")
    logging.info(f"  bias: {bias}")
    
    # Create and return the model
    model = TransformerModel(
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
    
    return model 