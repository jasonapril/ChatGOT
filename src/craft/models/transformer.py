# mypy: ignore-errors
"""
Transformer model implementation for character-level language modeling.
"""
import math
import logging
from typing import Optional, Tuple, Union, Dict, Any, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoderLayer # <-- ADD IMPORT

# Import base class, new config type, and registration decorator
from .base import LanguageModel # Import only the base model class
from ..config.schemas import LanguageModelConfig # Config location changed
from pydantic import ValidationError # For error handling
# from .registry import register_model # REMOVE registry import
# Import the generation utility
from ..utils.generation import autoregressive_generate

logger = logging.getLogger(__name__)

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
        # Explicitly type x after addition
        x: torch.Tensor = x + self.pe[:, :x.size(1), :] # type: ignore[index]
        # Cast return value to Tensor
        return cast(torch.Tensor, self.dropout(x)) # type: ignore

class TransformerModel(LanguageModel):
    """
    Transformer model for character-level language modeling.
    Uses the standard nn.TransformerDecoderLayer and nn.TransformerDecoder.
    Configured via Pydantic LanguageModelConfig.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize the transformer model using keyword arguments parsed into LanguageModelConfig.
        """
        try:
            # Parse kwargs into the expected Pydantic model
            config = LanguageModelConfig(**kwargs)
        except ValidationError as e:
            logger.error(f"Configuration validation failed for TransformerModel: {e}")
            # Re-raise or handle as appropriate for the application context
            raise ValueError(f"Invalid configuration provided to TransformerModel: {e}") from e

        # Pass the validated Pydantic config object to the parent class
        super().__init__(config)

        # Store config if needed elsewhere in this class
        self.config = config

        # Layer definitions using validated config attributes
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.max_seq_length, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        # Ensure d_hid is calculated or provided
        d_hid = config.d_hid if config.d_hid is not None else config.d_model * 4

        # Use nn.TransformerEncoderLayer
        self.transformer_layers = nn.ModuleList([
            TransformerEncoderLayer( # <-- USE TORCH LAYER
                d_model=config.d_model,
                nhead=config.n_head,
                dim_feedforward=d_hid,
                dropout=config.dropout,
                activation=config.activation,
                layer_norm_eps=config.layer_norm_eps,
                batch_first=True,
                norm_first=config.norm_first,
                bias=config.bias
            )
            for _ in range(config.n_layers)
        ])
        self.layer_norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps, bias=config.bias)
        self.output_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Optional weight tying
        self.token_embedding.weight = self.output_head.weight

        # Initialize weights
        self.apply(self._init_weights)
        # Apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layers))

        logger.info(f"TransformerModel initialized with d_model={config.d_model}, n_layers={config.n_layers}, n_head={config.n_head}")
    
    def _init_weights(self, module: nn.Module) -> None:
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
    
    def _generate_square_subsequent_mask(self, sz: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """
        Generate a square mask for the sequence (causal mask).
        Ensures attention is only paid to previous tokens.
        """
        # Use float('-inf') for positions to be masked
        mask = torch.full((sz, sz), float('-inf'), device=device, dtype=dtype)
        mask = torch.triu(mask, diagonal=1)
        # mask = torch.triu(torch.ones((sz, sz), device=device, dtype=torch.bool), diagonal=1)
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
        if seq_len > self.config.max_seq_length:
            raise ValueError(f"Input sequence length ({seq_len}) exceeds model's max sequence length ({self.config.max_seq_length})")
        
        # Get token embeddings
        tok_emb = self.token_embedding(x) * math.sqrt(self.config.d_model) # Scale embedding
        
        # Get position embeddings
        positions = torch.arange(0, seq_len, dtype=torch.long, device=x.device)
        # positions = torch.clamp(positions, max=self.config.max_seq_length-1) # Clamp if using fixed PE
        pos_emb = self.position_embedding(positions) # [seq_len, d_model]
        
        # Combine embeddings and apply dropout
        x = self.dropout(tok_emb + pos_emb) # type: ignore[index] # [batch_size, seq_len, d_model]
        
        # Create attention mask for the decoder (used as src_mask for EncoderLayer)
        attn_mask = self._generate_square_subsequent_mask(sz=seq_len, device=x.device, dtype=x.dtype)

        # Pass input through the transformer layers
        output = x
        for layer in self.transformer_layers:
             # Pass attn_mask to src_mask argument
             output = layer(output, src_mask=attn_mask) # <-- APPLY EACH LAYER

        # Apply final layer norm
        output = self.layer_norm(output) # <-- APPLY FINAL NORM

        # Generate logits
        logits: torch.Tensor = self.output_head(output) # [batch_size, seq_len, vocab_size]
        
        # Calculate loss if targets provided
        if targets is not None:
            # Explicitly type loss
            loss: torch.Tensor = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1) # Ignore padding
            return logits, loss
        
        return logits
    
    # --- ADD generate method implementation --- #
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: float = 1.0,
        eos_token_id: Optional[int] = None,
        verbose: bool = False
    ) -> torch.Tensor:
        """
        Generates sequences using the autoregressive_generate utility function.

        Args:
            input_ids: Tensor of starting token IDs (batch_size, seq_len).
            max_new_tokens: Maximum number of new tokens to generate.
            temperature: Softmax temperature (0 for greedy).
            top_k: Keep only top_k tokens for sampling.
            top_p: Keep smallest set of tokens with cumulative probability >= top_p.
            repetition_penalty: Penalty applied to repeated tokens (1.0 = no penalty).
            eos_token_id: ID of the end-of-sequence token to stop generation.
            verbose: Log progress within the generation utility.

        Returns:
            Tensor containing the input_ids plus the generated tokens.
        """
        # Ensure model is in evaluation mode for generation
        self.eval()
        
        # Delegate the actual generation logic to the utility function
        return autoregressive_generate(
            model=self,
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            eos_token_id=eos_token_id,
            verbose=verbose
        )