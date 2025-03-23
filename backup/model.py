import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

@dataclass
class TransformerConfig:
    """Configuration for the Transformer model."""
    vocab_size: int = 100  # Size of vocabulary
    context_size: int = 256  # Maximum context length
    n_layer: int = 8  # Number of transformer layers
    n_head: int = 8  # Number of attention heads
    n_embd: int = 384  # Embedding dimension
    dropout: float = 0.2  # Dropout probability
    bias: bool = True  # Use bias in layernorm and linear layers

class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention."""
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0, "Embedding dimension must be divisible by number of heads"
        
        # Key, query, value projections
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # Output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # Regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        
        # Create causal mask once at initialization
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.context_size, config.context_size)).view(
                1, 1, config.context_size, config.context_size
            )
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, embd_dim = x.size()  # (B, T, C)
        
        # Calculate query, key, value for all heads in batch
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        
        # Reshape to (B, nh, T, hs)
        head_size = self.n_embd // self.n_head
        q = q.view(batch_size, seq_len, self.n_head, head_size).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_head, head_size).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_head, head_size).transpose(1, 2)
        
        # Causal self-attention: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        # Compute attention scores
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # Apply mask to prevent attending to future tokens
        mask = self.mask[:, :, :seq_len, :seq_len]
        att = att.masked_fill(mask == 0, float("-inf"))
        # Apply softmax to get attention weights
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        
        # Apply attention weights to values
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        
        # Re-assemble all head outputs side by side
        y = y.transpose(1, 2).contiguous().view(batch_size, seq_len, embd_dim)
        
        # Output projection
        y = self.resid_dropout(self.c_proj(y))
        
        return y

class MLP(nn.Module):
    """Multi-layer perceptron after attention."""
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    """Transformer block: communication followed by computation."""
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-normalization for more stable training
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class CharTransformer(nn.Module):
    """Character-level transformer language model with causal self-attention."""
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        
        # Token embedding
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        # Position embedding
        self.wpe = nn.Embedding(config.context_size, config.n_embd)
        # Transformer blocks
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        # Final layer normalization
        self.ln_f = nn.LayerNorm(config.n_embd, bias=config.bias)
        # Language modeling head
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # Weight tying between token embedding and LM head
        self.wte.weight = self.lm_head.weight
        
        # Initialize parameters
        self.apply(self._init_weights)
        
        # Report number of parameters
        print(f"Number of parameters: {self.get_num_params():,}")
        
    def _init_weights(self, module: nn.Module) -> None:
        """Initialize the weights based on module type."""
        if isinstance(module, nn.Linear):
            # Initialize linear layers with small random values
            # and appropriate scaling for stability
            std = 0.02
            if hasattr(module, "weight") and module.weight is not None:
                torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if hasattr(module, "bias") and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            # Initialize embedding weights with small random values
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            # Initialize LayerNorm weights to ones and biases to zeros
            if hasattr(module, "weight") and module.weight is not None:
                torch.nn.init.ones_(module.weight)
            if hasattr(module, "bias") and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
                
    def get_num_params(self) -> int:
        """Calculate total number of trainable parameters."""
        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return n_params
        
    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            idx: Input token indices of shape (B, T)
            targets: Optional target token indices (not used in this interface)
            
        Returns:
            logits: Output token logits of shape (B, T, vocab_size)
        """
        device = idx.device
        batch_size, seq_len = idx.size()
        
        # Check if input is within context size
        assert seq_len <= self.config.context_size, f"Input sequence length ({seq_len}) exceeds context size ({self.config.context_size})"
        
        # Get token and position embeddings
        token_embeddings = self.wte(idx)  # (B, T, C)
        position_ids = torch.arange(0, seq_len, dtype=torch.long, device=device).unsqueeze(0)  # (1, T)
        position_embeddings = self.wpe(position_ids)  # (1, T, C)
        
        # Combine token and position embeddings
        x = token_embeddings + position_embeddings  # (B, T, C)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
            
        # Apply final layer norm
        x = self.ln_f(x)
        
        # Calculate logits
        logits = self.lm_head(x)  # (B, T, vocab_size)
        
        return logits
    
    def configure_optimizers(self, weight_decay: float, learning_rate: float) -> torch.optim.Optimizer:
        """
        Configure optimizer with weight decay.
        
        Args:
            weight_decay: Weight decay coefficient
            learning_rate: Learning rate
            
        Returns:
            Configured optimizer
        """
        # Create two parameter groups: one with weight decay and one without
        decay_params = []
        no_decay_params = []
        seen_params = set()
        
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            
            # Get the parameter ID to avoid duplicates
            param_id = id(param)
            if param_id in seen_params:
                continue
            seen_params.add(param_id)
            
            # Determine if this parameter should have weight decay
            # Bias terms and LayerNorm/Embedding weights should not have weight decay
            if name.endswith("bias") or ".bias" in name:
                no_decay_params.append(param)
            elif name.endswith("ln") or ".ln" in name or "layernorm" in name.lower():
                no_decay_params.append(param)
            elif "embedding" in name.lower() or ".wpe." in name or ".wte." in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        
        # Create optimizer with the two parameter groups
        optimizer_grouped_parameters = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]
        
        # Print optimizer configuration
        print(f"Optimizer config: {len(decay_params)} params with weight decay, "
              f"{len(no_decay_params)} params without weight decay")
        
        # Use AdamW optimizer
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=learning_rate)
        
        return optimizer
    
    @torch.no_grad()
    def generate(self, 
                 idx: torch.Tensor, 
                 max_new_tokens: int, 
                 temperature: float = 1.0, 
                 top_k: Optional[int] = None,
                 verbose: bool = False) -> torch.Tensor:
        """
        Generate text given a starting sequence of token indices.
        
        Args:
            idx: Starting token indices of shape (B, T)
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature (higher means more random)
            top_k: If specified, limits sampling to top k most likely tokens
            verbose: Whether to print token probabilities during generation
            
        Returns:
            Generated token indices of shape (B, T+max_new_tokens)
        """
        self.eval()  # Set to evaluation mode
        
        for i in range(max_new_tokens):
            # Crop context to block_size if it's too long
            idx_cond = idx if idx.size(1) <= self.config.context_size else idx[:, -self.config.context_size:]
            
            # Forward pass to get logits for the next token
            logits = self.forward(idx_cond)
            
            # Focus on the last token position to predict the next
            logits = logits[:, -1, :]  # (B, vocab_size)
            
            # Apply temperature
            logits = logits / (temperature if temperature > 0 else 1.0)
            
            # Optionally apply top-k sampling
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                # Set all logits below top_k to negative infinity
                logits[logits < v[:, [-1]]] = -float('Inf')
                
            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)
            
            # Sample from the probability distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            
            if verbose:
                # Print top 5 most likely tokens
                top_probs, top_indices = torch.topk(probs, 5)
                tokens = [f"{i.item()}={p.item():.3f}" for i, p in zip(top_indices[0], top_probs[0])]
                print(f"Step {i}: Token={idx_next[0].item()}, Top tokens: {', '.join(tokens)}")
            
            # Append the new token to the sequence
            idx = torch.cat((idx, idx_next), dim=1)
            
        return idx
    
    def save_checkpoint(self, path: str, optimizer=None, scheduler=None, epoch=0, loss=None, char_to_idx=None, idx_to_char=None) -> None:
        """
        Save model checkpoint.
        
        Args:
            path: Path to save checkpoint
            optimizer: Optional optimizer state
            scheduler: Optional scheduler state
            epoch: Current epoch number
            loss: Current loss value
            char_to_idx: Character to index mapping
            idx_to_char: Index to character mapping
        """
        # Create directory if needed
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Prepare checkpoint data
        checkpoint = {
            "model_state_dict": self.state_dict(),
            "config": self.config.__dict__,
        }
        
        # Add optional data if provided
        if optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()
        if scheduler is not None:
            checkpoint["scheduler_state_dict"] = scheduler.state_dict()
        if epoch is not None:
            checkpoint["epoch"] = epoch
        if loss is not None:
            checkpoint["loss"] = loss
        if char_to_idx is not None:
            checkpoint["char_to_idx"] = char_to_idx
        if idx_to_char is not None:
            checkpoint["idx_to_char"] = idx_to_char
        
        # Save checkpoint
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")
    
    @classmethod
    def load_checkpoint(cls, path: str, device="cpu", optimizer=None, scheduler=None) -> Tuple[Any, ...]:
        """
        Load model from checkpoint.
        
        Args:
            path: Path to checkpoint
            device: Device to load model on
            optimizer: Optional optimizer to load state into
            scheduler: Optional scheduler to load state into
            
        Returns:
            Tuple containing model and optionally optimizer, scheduler, epoch, loss, char_to_idx, idx_to_char
        """
        checkpoint = torch.load(path, map_location=device)
        
        # Load config and create model
        config_dict = checkpoint.get("config", {})
        config = TransformerConfig(**config_dict)
        model = cls(config)
        
        # Load model weights
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(device)
        
        # Return variables to unpack
        result = [model]
        
        # Add optimizer if requested and available
        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            result.append(optimizer)
            
        # Add scheduler if requested and available
        if scheduler is not None and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            result.append(scheduler)
            
        # Add other data if available
        for key in ["epoch", "loss", "char_to_idx", "idx_to_char"]:
            if key in checkpoint:
                result.append(checkpoint[key])
                
        return tuple(result)

def create_char_transformer(
    vocab_size,
    n_layer=12,
    n_head=12,
    n_embd=768,
    dropout=0.1,
    context_size=256,
):
    """
    Create a standard character-level Transformer model with approximately 85M parameters.
    Using fixed configuration for consistency and speed.
    """
    config = TransformerConfig(
        vocab_size=vocab_size,
        context_size=context_size,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        dropout=dropout,
    )
    
    model = CharTransformer(config)
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
    return model 