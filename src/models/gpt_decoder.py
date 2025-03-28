"""
Optimized decoder-only transformer model (GPT style) without cross-attention.

This is a more efficient implementation compared to the standard transformer.py model,
removing ~28M parameters by eliminating the unused cross-attention mechanism.
"""
import math
import logging
from typing import Optional, Tuple, Dict, List, Any, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import LanguageModel, ModelConfig


class CausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention layer.
    
    This is similar to the standard PyTorch implementation but without cross-attention,
    saving ~28M parameters in the 12-layer model.
    """
    
    def __init__(
        self,
        d_model: int,
        n_head: int,
        dropout: float = 0.1,
        bias: bool = True,
        layer_norm_eps: float = 1e-5,
    ):
        super().__init__()
        assert d_model % n_head == 0, "d_model must be divisible by n_head"
        
        # Save parameters
        self.d_model = d_model
        self.n_head = n_head
        self.head_dim = d_model // n_head
        
        # Create projection layers
        self.query = nn.Linear(d_model, d_model, bias=bias)
        self.key = nn.Linear(d_model, d_model, bias=bias)
        self.value = nn.Linear(d_model, d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        
        # Regularization
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        
        # Register buffer for causal mask
        self.register_buffer("causal_mask", None)
    
    def _prepare_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Create a causal mask for self-attention."""
        if self.causal_mask is None or self.causal_mask.size(0) < seq_len:
            mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
            mask = mask.masked_fill(mask == 1, float("-inf"))
            if self.causal_mask is None or self.causal_mask.size(0) < seq_len:
                self.causal_mask = mask
        
        return self.causal_mask[:seq_len, :seq_len]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for causal self-attention.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, d_model]
            
        Returns:
            Output tensor of shape [batch_size, seq_len, d_model]
        """
        # Get batch size and sequence length
        batch_size, seq_len, _ = x.size()
        
        # Project queries, keys, and values
        q = self.query(x)  # [batch_size, seq_len, d_model]
        k = self.key(x)    # [batch_size, seq_len, d_model]
        v = self.value(x)  # [batch_size, seq_len, d_model]
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)  # [batch_size, n_head, seq_len, head_dim]
        k = k.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)  # [batch_size, n_head, seq_len, head_dim]
        v = v.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)  # [batch_size, n_head, seq_len, head_dim]
        
        # Compute scaled dot-product attention
        # (batch_size, n_head, seq_len, head_dim) @ (batch_size, n_head, head_dim, seq_len)
        # = (batch_size, n_head, seq_len, seq_len)
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply causal mask to prevent attending to future tokens
        causal_mask = self._prepare_causal_mask(seq_len, x.device)
        attn = attn + causal_mask
        
        # Apply softmax and dropout
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)
        
        # Apply attention to values
        # (batch_size, n_head, seq_len, seq_len) @ (batch_size, n_head, seq_len, head_dim)
        # = (batch_size, n_head, seq_len, head_dim)
        out = attn @ v
        
        # Reshape and project
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        out = self.out_proj(out)
        out = self.resid_dropout(out)
        
        return out


class FeedForward(nn.Module):
    """
    Feed-forward network with GELU activation.
    """
    
    def __init__(
        self,
        d_model: int,
        d_hid: int,
        dropout: float = 0.1,
        activation: str = "gelu",
        bias: bool = True,
    ):
        super().__init__()
        
        self.fc1 = nn.Linear(d_model, d_hid, bias=bias)
        self.fc2 = nn.Linear(d_hid, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)
        
        # Set activation function
        if activation == "gelu":
            self.act = nn.GELU()
        elif activation == "relu":
            self.act = nn.ReLU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for feed-forward network.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, d_model]
            
        Returns:
            Output tensor of shape [batch_size, seq_len, d_model]
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        
        return x


class DecoderLayer(nn.Module):
    """
    Decoder layer with self-attention and feed-forward network.
    """
    
    def __init__(
        self,
        d_model: int,
        n_head: int,
        d_hid: int,
        dropout: float = 0.1,
        activation: str = "gelu",
        norm_first: bool = True,
        layer_norm_eps: float = 1e-5,
        bias: bool = True,
    ):
        super().__init__()
        
        # Self-attention block
        self.self_attn = CausalSelfAttention(
            d_model=d_model,
            n_head=n_head,
            dropout=dropout,
            bias=bias,
            layer_norm_eps=layer_norm_eps,
        )
        
        # Feed-forward block
        self.ff = FeedForward(
            d_model=d_model,
            d_hid=d_hid,
            dropout=dropout,
            activation=activation,
            bias=bias,
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Normalization order
        self.norm_first = norm_first
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for decoder layer.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, d_model]
            
        Returns:
            Output tensor of shape [batch_size, seq_len, d_model]
        """
        # Apply normalization before or after attention
        if self.norm_first:
            # Pre-LayerNorm architecture (more stable)
            x = x + self.self_attn(self.norm1(x))
            x = x + self.ff(self.norm2(x))
        else:
            # Post-LayerNorm architecture
            x = self.norm1(x + self.self_attn(x))
            x = self.norm2(x + self.ff(x))
        
        return x


class GPTDecoder(LanguageModel):
    """
    Optimized decoder-only transformer model (GPT style) without cross-attention.
    
    This architecture is more efficient for autoregressive text generation.
    """
    def __init__(
        self,
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
    ):
        """
        Initialize GPT decoder model.
        
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
                architecture="gpt"
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
        
        # Embeddings
        self.token_embedding = nn.Embedding(self.vocab_size, self.d_model)
        self.position_embedding = nn.Embedding(self.max_seq_length, self.d_model)
        self.dropout = nn.Dropout(self.dropout_rate)
        
        # Create decoder layers
        self.layers = nn.ModuleList([
            DecoderLayer(
                d_model=self.d_model,
                n_head=self.n_head,
                d_hid=self.d_hid,
                dropout=self.dropout_rate,
                activation=self.activation,
                norm_first=True,
                layer_norm_eps=self.layer_norm_eps,
                bias=self.bias,
            )
            for _ in range(self.n_layers)
        ])
        
        # Final layer normalization
        self.ln_f = nn.LayerNorm(self.d_model, eps=self.layer_norm_eps)
        
        # Output projection
        self.out_proj = nn.Linear(self.d_model, self.vocab_size, bias=self.bias)
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Log model size
        self._log_model_size()
    
    def _init_weights(self, module):
        """
        Initialize the weights of the model.
        
        Args:
            module: Model module to initialize
        """
        if isinstance(module, nn.Linear):
            # Standard initialization for linear layers
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            # Standard initialization for embedding layers
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            # Standard initialization for layer normalization
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
    
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
        
        # Create position indices
        positions = torch.arange(0, seq_len, dtype=torch.long, device=x.device)
        positions = torch.clamp(positions, max=self.max_seq_length-1)  # Clamp to prevent out-of-bounds
        positions = positions.unsqueeze(0).expand(batch_size, -1)  # [batch_size, seq_len]
        
        # Get token and position embeddings
        token_embeds = self.token_embedding(x)  # [batch_size, seq_len, d_model]
        pos_embeds = self.position_embedding(positions)  # [batch_size, seq_len, d_model]
        
        # Combine embeddings
        hidden_states = token_embeds + pos_embeds  # [batch_size, seq_len, d_model]
        hidden_states = self.dropout(hidden_states)
        
        # Pass through decoder layers
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        
        # Apply final layer normalization
        hidden_states = self.ln_f(hidden_states)
        
        # Project to vocabulary space
        logits = self.out_proj(hidden_states)  # [batch_size, seq_len, vocab_size]
        
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
        
        # Generate tokens one by one
        for i in range(max_new_tokens):
            # Forward pass
            with torch.no_grad():
                # If sequence is too long, truncate from the beginning
                curr_input = generated_tokens
                if curr_input.size(1) > self.max_seq_length:
                    curr_input = curr_input[:, -self.max_seq_length:]
                
                logits = self(curr_input)
                
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
            "architecture": "gpt",
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


def create_gpt_model(
    config: Optional[ModelConfig] = None,
    vocab_size: Optional[int] = None,
    d_model: int = 768,
    n_head: int = 12,
    d_hid: int = 3072,
    n_layers: int = 12,
    dropout: float = 0.1,
    max_seq_length: int = 1024,
    layer_norm_eps: float = 1e-5,
    activation: str = "gelu",
    bias: bool = True
) -> GPTDecoder:
    """
    Create a GPT model.
    
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
        GPTDecoder instance
    """
    return GPTDecoder(
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