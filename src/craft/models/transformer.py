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

# Import base class, new config type, and registration decorator
from .base import LanguageModel # Import only the base model class
from ..config.schemas import LanguageModelConfig # Config location changed
# from .registry import register_model # REMOVE registry import
# Import the generation utility
from ..utils.generation import autoregressive_generate


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

@register_model(name="craft.models.transformer.TransformerModel", config_cls=LanguageModelConfig) # type: ignore[index]
class TransformerModel(LanguageModel): # type: ignore[no-any-return]
    """
    Transformer model for character-level language modeling.
    Uses the standard nn.TransformerDecoderLayer and nn.TransformerDecoder.
    Configured via Pydantic LanguageModelConfig.
    """
    
    def __init__(self, config: LanguageModelConfig):
        """
        Initialize the transformer model using a LanguageModelConfig.
        """
        # Ensure config is the correct Pydantic type
        if not isinstance(config, LanguageModelConfig):
             raise TypeError(f"Expected config to be LanguageModelConfig, got {type(config)}")

        super().__init__(config)
        
        # Access parameters directly from the validated config object
        self.vocab_size = config.vocab_size
        if self.vocab_size is None:
            raise ValueError("TransformerModel requires config.vocab_size to be set.")
        self.d_model = config.d_model
        self.n_head = config.n_head
        self.d_hid = config.d_hid if config.d_hid is not None else config.d_model * 4
        self.n_layers = config.n_layers
        self.dropout_rate = config.dropout
        # Ensure max_seq_length is set on the instance, required by autoregressive_generate
        self.max_seq_length = config.max_seq_length 
        self.layer_norm_eps = config.layer_norm_eps
        self.activation = config.activation
        self.bias = config.bias
        self.norm_first = config.norm_first
        
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
    
    def _generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """
        Generate a square mask for the sequence (causal mask).
        Ensures attention is only paid to previous tokens.
        """
        # Use float('-inf') for positions to be masked
        mask: torch.Tensor = torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1)
        return mask # type: ignore[no-any-return]
    
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
        x = self.dropout(tok_emb + pos_emb) # type: ignore[index] # [batch_size, seq_len, d_model]
        
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
        logits: torch.Tensor = self.output_layer(output) # [batch_size, seq_len, vocab_size]
        
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