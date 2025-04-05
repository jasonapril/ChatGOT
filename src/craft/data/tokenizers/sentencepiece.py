import logging
import sentencepiece as spm
import os
from typing import List, Optional, Dict, Any
from .base import BaseTokenizer

logger = logging.getLogger(__name__)

class SentencePieceTokenizer(BaseTokenizer):
    def __init__(self, 
                 # Make vocab_size and model_prefix optional for loading scenario
                 vocab_size: Optional[int] = None, # Allow None when loading
                 model_prefix: Optional[str] = None, # Allow None when loading
                 model_path: Optional[str] = None, # Path PREIFX for loading existing model
                 character_coverage: float = 1.0,
                 pad_token: str = '<pad>',
                 unk_token: str = '<unk>',
                 bos_token: str = '<s>',
                 eos_token: str = '</s>',
                 pad_id: int = 0,
                 unk_id: int = 1,
                 bos_id: int = 2,
                 eos_id: int = 3,
                 **kwargs # Capture any extra args for BaseTokenizer/logging
                 ):
        """
        Initializes the SentencePieceTokenizer.
        Can be initialized with config for training or directly load a model if model_path is provided.
        
        Args:
            vocab_size (Optional[int]): Target vocabulary size (required for training, 
                                      determined from loaded model if None and model_path is given).
            model_prefix (Optional[str]): Prefix for saving model files (required for training).
            model_path (Optional[str]): Path prefix to load an existing .model file.
                                        If provided, the tokenizer attempts to load immediately.
            character_coverage (float): Character coverage ratio for training.
            pad_token (str): Padding token string.
            unk_token (str): Unknown token string.
            bos_token (str): Beginning-of-sentence token string.
            eos_token (str): End-of-sentence token string.
            pad_id (int): Token ID for padding.
            unk_id (int): Token ID for unknown.
            bos_id (int): Token ID for BOS.
            eos_id (int): Token ID for EOS.
            **kwargs: Additional arguments (not used by this class directly but captured).
        """
        # Reconstruct config dict for BaseTokenizer and potential saving
        # Filter out None values before creating dict unless they are explicitly part of the config intention
        # For simplicity here, we pass all locals.
        config_dict = {k: v for k, v in locals().items() if k not in ['self', 'kwargs', 'config_dict']}
        config_dict.update(kwargs)
        super().__init__(config_dict) 

        # Assign attributes directly (may be None initially if loading)
        self._initial_vocab_size = vocab_size # Store initial config value if provided
        self._initial_model_prefix = model_prefix # Store initial config value if provided
        # Other attributes remain the same
        self.character_coverage = character_coverage
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.pad_id = pad_id
        self.unk_id = unk_id
        self.bos_id = bos_id
        self.eos_id = eos_id

        self.sp = None # SentencePieceProcessor instance
        self._loaded_model_path = None # Store the path prefix used for loading

        # Load model immediately if model_path is provided
        if model_path:
            try:
                self._load_model(model_path)
                # Note: _load_model now updates self.vocab_size based on the loaded model
            except Exception as e:
                logger.error(f"Failed to load model during __init__ from path: {model_path}", exc_info=True)
                pass 
        elif vocab_size is None or model_prefix is None:
             # If not loading and required training args are missing
             logger.warning("SentencePieceTokenizer initialized without model_path and missing vocab_size or model_prefix. Training will likely fail.")

    def _load_model(self, model_path_prefix: str):
        """Internal helper to load the SentencePiece model from a prefix."""
        self._loaded_model_path = model_path_prefix # Store the prefix used
        model_file = f"{model_path_prefix}.model"
        logger.info(f"Attempting to load SentencePiece model from: {model_file}")
        if not os.path.exists(model_file):
             logger.error(f"SentencePiece model file not found: {model_file}")
             raise FileNotFoundError(f"SentencePiece model file not found: {model_file}")
        try:
            self.sp = spm.SentencePieceProcessor()
            self.sp.load(model_file)
            # GET vocab_size FROM loaded model
            loaded_vocab_size = self.sp.get_piece_size()
            if self._initial_vocab_size is not None and loaded_vocab_size != self._initial_vocab_size:
                logger.warning(
                    f"Loaded SentencePiece model vocab size ({loaded_vocab_size}) differs "
                    f"from config vocab_size ({self._initial_vocab_size}). Using loaded size."
                )
            # Update the effective vocab_size based on the loaded model
            # self.vocab_size = loaded_vocab_size # No! Use the property/getter method instead
            logger.info(f"Successfully loaded SentencePiece model from {model_file} with vocab size {loaded_vocab_size}")
        except Exception as e:
            logger.error(f"Failed to load SentencePiece model from {model_file}: {e}", exc_info=True)
            self.sp = None # Ensure sp is None on failure
            raise RuntimeError(f"Failed to load SentencePiece model from {model_file}: {e}") from e

    def train(self, files: List[str], output_dir: str) -> None:
        """Train the SentencePiece model on a list of input text files."""
        # Ensure required args for training are present
        if self._initial_vocab_size is None or self._initial_model_prefix is None:
            raise ValueError("Cannot train SentencePieceTokenizer without 'vocab_size' and 'model_prefix' being set during initialization.")
        if not files:
            raise ValueError("Input file list cannot be empty for training.")
            
        os.makedirs(output_dir, exist_ok=True)
        
        # Use model_prefix attribute set during init
        model_save_prefix = os.path.join(output_dir, self._initial_model_prefix) 
        logger.info(f"Training SentencePiece model, saving to prefix: {model_save_prefix}")
        
        input_files_str = ",".join(files)
        
        try:
            spm.SentencePieceTrainer.train(
                input=input_files_str, 
                model_prefix=model_save_prefix,
                vocab_size=self._initial_vocab_size, # Use initial config value for training
                character_coverage=self.character_coverage,
                model_type='bpe',
                pad_id=self.pad_id,
                unk_id=self.unk_id,
                bos_id=self.bos_id,
                eos_id=self.eos_id,
                pad_piece=self.pad_token,
                unk_piece=self.unk_token,
                bos_piece=self.bos_token,
                eos_piece=self.eos_token,
            )
            self._load_model(model_save_prefix)
            logger.info(f"Successfully trained and loaded SentencePiece model: {model_save_prefix}.model")
        except Exception as e:
            logger.error(f"SentencePiece training failed: {e}", exc_info=True)
            raise RuntimeError(f"SentencePiece training failed: {e}") from e
        
    def load(self, model_path_prefix: str) -> None:
        """Load or reload a pre-trained SentencePiece model from a prefix."""
        # Public method simply calls the internal one
        self._load_model(model_path_prefix)
        
    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        if self.sp is None:
            raise RuntimeError("Tokenizer not initialized or failed to load. Call train() or ensure model_path was valid during init.")
        return self.sp.encode_as_ids(text)
    
    def decode(self, ids: List[int]) -> str:
        """Decode token IDs to text."""
        if self.sp is None:
            raise RuntimeError("Tokenizer not initialized or failed to load. Call train() or ensure model_path was valid during init.")
        return self.sp.decode_ids(ids)
    
    def get_vocab_size(self) -> int:
         """Return the vocabulary size."""
         if self.sp:
             return self.sp.get_piece_size()
         # Fallback to initial config value if model not loaded
         return self._initial_vocab_size if self._initial_vocab_size is not None else 0
    
    def save(self, output_dir: str) -> None:
        """Save the tokenizer configuration (not the model files themselves)."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Create a clean dictionary with only serializable config values
        config_to_save = {
            # Use get_vocab_size() to get the potentially updated size
            'vocab_size': self.get_vocab_size(), 
            # Use the stored initial prefix if available, otherwise None or a default?
            # For saving, model_prefix is mostly informational if model is already trained.
            'model_prefix': self._initial_model_prefix, 
            'character_coverage': self.character_coverage,
            'pad_token': self.pad_token,
            'unk_token': self.unk_token,
            'bos_token': self.bos_token,
            'eos_token': self.eos_token,
            'pad_id': self.pad_id,
            'unk_id': self.unk_id,
            'bos_id': self.bos_id,
            'eos_id': self.eos_id,
            # Add the target class explicitly for potential re-instantiation
            '_target_': f"{self.__class__.__module__}.{self.__class__.__name__}"
        }
        # Filter out None values that are not essential for reloading
        config_to_save = {k: v for k, v in config_to_save.items() if v is not None or k == '_target_'}

        import json
        save_path = os.path.join(output_dir, 'tokenizer_config.json')
        logger.info(f"Saving tokenizer configuration to: {save_path}")
        try:
            with open(save_path, 'w', encoding="utf-8") as f:
                json.dump(config_to_save, f, indent=4)
        except Exception as e:
            logger.error(f"Failed to save tokenizer config: {e}", exc_info=True)
            # Optionally re-raise if saving config is critical
            # raise IOError(f"Failed to save tokenizer config to {save_path}") from e

    # Add __getstate__ and __setstate__ for pickling if needed?
    # For now, assume Hydra handles instantiation via config. 