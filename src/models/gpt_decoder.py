"""
Optimized decoder-only transformer model (GPT style) without cross-attention.

This is a more efficient implementation compared to the standard transformer.py model,
removing ~28M parameters by eliminating the unused cross-attention mechanism.
"""
import math
import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


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


class GPTDecoder(nn.Module):
    """
    Optimized decoder-only transformer model (GPT style) without cross-attention.
    
    This architecture is more efficient for autoregressive text generation.
    """
    def __init__(
        self,
        vocab_size: int,
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
        super().__init__()
        
        # Store important configuration parameters
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_head = n_head
        self.n_layers = n_layers
        self.d_hid = d_hid
        self.max_seq_length = max_seq_length
        self.activation = activation  # Store the activation function name
        
        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # Position embedding
        self.position_embedding = nn.Embedding(max_seq_length, d_model)
        
        # Create decoder layers
        self.layers = nn.ModuleList([
            DecoderLayer(
                d_model=d_model,
                n_head=n_head,
                d_hid=d_hid,
                dropout=dropout,
                activation=activation,
                norm_first=True,
                layer_norm_eps=layer_norm_eps,
                bias=bias,
            )
            for _ in range(n_layers)
        ])
        
        # Final normalization
        self.norm = nn.LayerNorm(d_model, eps=layer_norm_eps)
        
        # Output layer
        self.output_layer = nn.Linear(d_model, vocab_size, bias=bias)
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Log model size
        self._log_model_size()
    
    def _init_weights(self, module):
        """Initialize the weights of the model."""
        if isinstance(module, nn.Embedding):
            # Embeddings benefit from normal initialization with slightly higher std dev
            module.weight.data.normal_(mean=0.0, std=0.05)
        elif isinstance(module, nn.Linear):
            # Linear layers use Kaiming initialization for ReLU-like activations
            if hasattr(self, 'activation') and (self.activation == 'gelu' or self.activation == 'relu'):
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
            else:
                # Fall back to standard initialization
                module.weight.data.normal_(mean=0.0, std=0.02)
                
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def _log_model_size(self):
        """Log the number of parameters in the model."""
        n_params = sum(p.numel() for p in self.parameters())
        logging.info(f"Model initialized with {n_params:,} parameters")
    
    def forward(self, x: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
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
        
        # Apply embedding dropout for regularization
        token_embeddings = F.dropout(token_embeddings, p=0.05, training=self.training)
        
        # Get position embeddings
        positions = torch.arange(0, seq_len, dtype=torch.long, device=x.device)
        # Clamp positions to max_seq_length-1 to prevent out-of-bounds issues
        positions = torch.clamp(positions, max=self.max_seq_length-1)
        positions = positions.unsqueeze(0).expand(batch_size, -1)
        position_embeddings = self.position_embedding(positions)
        
        # Combine embeddings
        x = token_embeddings + position_embeddings
        
        # Pass through decoder layers
        for layer in self.layers:
            x = layer(x)
        
        # Final normalization
        x = self.norm(x)
        
        # Generate logits
        logits = self.output_layer(x)
        
        # Calculate loss if targets provided
        if targets is not None:
            # Reshape for cross-entropy
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            # Log loss statistics periodically for debugging
            if torch.rand(1).item() < 0.01:  # Log ~1% of the time
                with torch.no_grad():
                    # Check if NaN
                    if torch.isnan(loss).any():
                        logging.warning(f"NaN loss detected")
                    
                    # Log additional loss diagnostics
                    vocab_size = logits.size(-1)
                    theoretical_min = math.log(vocab_size)
                    ratio_to_theoretical = loss.item() / theoretical_min
                    logging.debug(f"Loss: {loss.item():.4f}, Ratio to theoretical min: {ratio_to_theoretical:.2f}x")
            
            return logits, loss
        
        return logits, None
    
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
                logits, _ = self(input_ids)
                logits = logits[:, -1, :] / temperature  # Only need the last token's logits
            
            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for b in range(batch_size):
                    for token_id in set(input_ids[b].tolist()):
                        logits[b, token_id] /= repetition_penalty
            
            # Apply top-k sampling
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float('-inf')
            
            # Apply top-p (nucleus) sampling
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift the indices to the right to keep also the first token above the threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                for b in range(batch_size):
                    indices_to_remove = sorted_indices[b][sorted_indices_to_remove[b]]
                    logits[b, indices_to_remove] = float('-inf')
            
            # Sample next token
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to input_ids
            input_ids = torch.cat([input_ids, next_token], dim=1)
            
            if verbose and (i+1) % 10 == 0:
                logging.info(f"Generated {i+1}/{max_new_tokens} tokens")
        
        return input_ids


def create_gpt_decoder(
    vocab_size: int,
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
    Create an optimized GPT decoder model with the specified parameters.
    
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
        A GPTDecoder instance
    """
    # Log the model configuration
    logging.info(f"Creating optimized GPT decoder model with configuration:")
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
    model = GPTDecoder(
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