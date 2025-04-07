import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import shutil

import sentencepiece as spm

from .base import Tokenizer

logger = logging.getLogger(__name__)


class SentencePieceTokenizer(Tokenizer):
    """
    A tokenizer using Google's SentencePiece library.

    Handles training, loading, encoding, decoding, and vocabulary management
    for SentencePiece models (Unigram, BPE, Char, Word).
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        vocab_size: Optional[int] = None,
        model_prefix: Optional[str] = None,
        model_type: str = "unigram",
        character_coverage: float = 0.9995,
        special_tokens: Optional[Dict[str, str]] = None,
        add_bos_token: bool = False, # Added for consistency, maps to spm option
        add_eos_token: bool = False, # Added for consistency, maps to spm option
        **kwargs,
    ):
        """
        Initializes the SentencePieceTokenizer.

        Can either load an existing model or be configured for training.

        Args:
            model_path: Path prefix to the pre-trained SentencePiece model
                        (e.g., 'my_sp_model' for 'my_sp_model.model' and
                        'my_sp_model.vocab'). If provided, attempts to load.
            vocab_size: Vocabulary size required for training.
            model_prefix: Prefix for saving the model during training (e.g., 'sp_model').
                          Required for training if model_path is not given.
            model_type: SentencePiece model type ('unigram', 'bpe', 'char', 'word').
            character_coverage: Character coverage for training (for BPE/Unigram).
            special_tokens: Dictionary mapping special token roles
                             (e.g., 'pad', 'unk') to their string representation.
            add_bos_token: Whether to add BOS token during encoding (passed to spm).
            add_eos_token: Whether to add EOS token during encoding (passed to spm).
            **kwargs: Additional arguments passed to the base Tokenizer.
        """
        # Store training/config parameters before calling super().__init__
        self.model_path_prefix = model_path # Store original path prefix if loading
        self.vocab_size_config = vocab_size # Store configured vocab size for training/metadata
        self.model_prefix_config = model_prefix # Store configured prefix for training/metadata
        self.model_type = model_type
        self.character_coverage = character_coverage
        # We handle special tokens after potential loading

        self.sp_model: Optional[spm.SentencePieceProcessor] = None
        self.metadata: Dict[str, Any] = {} # Store loaded/saved metadata

        loaded_config = {}
        loaded_special_tokens = {}

        if model_path:
            logger.info(f"Attempting to load SentencePiece model from prefix: {model_path}")
            model_file = Path(f"{model_path}.model")
            metadata_file = Path(f"{model_path}.json")
            if model_file.exists() and metadata_file.exists():
                try:
                    # Load metadata first
                    with open(metadata_file, "r", encoding="utf-8") as f:
                        self.metadata = json.load(f)
                        logger.info(f"Loaded SentencePiece metadata from: {metadata_file}")
                        # Extract relevant info for Tokenizer base class
                        loaded_config['vocab_size'] = self.metadata.get('vocab_size')
                        loaded_special_tokens = self.metadata.get('special_tokens', {})
                        # Update init args from metadata if they weren't provided explicitly
                        # (explicit args take precedence)
                        self.vocab_size_config = vocab_size if vocab_size is not None else loaded_config.get('vocab_size')
                        self.model_prefix_config = model_prefix if model_prefix is not None else self.metadata.get('model_prefix')
                        self.model_type = model_type if model_type else self.metadata.get('model_type', 'unigram')
                        self.character_coverage = character_coverage if character_coverage else self.metadata.get('character_coverage', 0.9995)
                        # special_tokens handled later

                    # Load SentencePiece model
                    self.sp_model = spm.SentencePieceProcessor(str(model_file))
                    logger.info(f"Loaded SentencePiece model from: {model_file}")
                    # Set add_bos/add_eos based on loaded model/metadata or init args
                    # SPM doesn't directly expose these after loading, rely on metadata or args
                    self.add_bos_token = add_bos_token # Use init arg if provided
                    self.add_eos_token = add_eos_token # Use init arg if provided


                except Exception as e:
                    logger.error(f"Failed to load SentencePiece model or metadata from {model_path}: {e}")
                    self.sp_model = None
                    self.metadata = {} # Reset metadata on load failure
            else:
                 missing = []
                 if not model_file.exists():
                     missing.append(str(model_file))
                 if not metadata_file.exists():
                    missing.append(str(metadata_file))
                 logger.warning(f"SentencePiece model/metadata file(s) not found: {', '.join(missing)}. Initializing empty tokenizer.")
                 self.model_path_prefix = None # Reset prefix if files are missing


        # Combine special tokens: explicit args override loaded ones
        # Start with an empty map; base class will handle defaults if needed.
        final_special_tokens_map = {}
        final_special_tokens_map.update(loaded_special_tokens) # Update with loaded tokens first
        if special_tokens: # Then update with explicit args, overriding loaded ones
            final_special_tokens_map.update(special_tokens)


        # Call super AFTER loading or determining config
        super().__init__(
            vocab_size=self.sp_model.get_piece_size() if self.sp_model else self.vocab_size_config,
            special_tokens=final_special_tokens_map,
            add_bos_token=add_bos_token,
            add_eos_token=add_eos_token,
             **kwargs,
        )

        # Sync vocab/ids from loaded SentencePiece model if available
        if self.sp_model:
             self._sync_vocab_from_sp_model()
        else:
            if not self.model_path_prefix and (not self.vocab_size_config or not self.model_prefix_config):
                 logger.warning(
                     "SentencePieceTokenizer initialized without model_path and missing "
                     "vocab_size or model_prefix. Training is not possible unless "
                     "these are set."
                 )

    def _sync_vocab_from_sp_model(self):
        """Updates internal vocab mappings from the loaded sp_model."""
        if not self.sp_model:
            return
        self.vocab_size = self.sp_model.get_piece_size()
        # Use base class's special token strings if available, otherwise SP's defaults
        unk = self.unk_token if self.unk_token is not None else "<unk>" # Default SP unk
        bos = self.bos_token if self.bos_token is not None else "<s>"   # Default SP bos
        eos = self.eos_token if self.eos_token is not None else "</s>"   # Default SP eos
        pad = self.pad_token # No clear SP default, use ours or None

        self.unk_token_id = self.sp_model.unk_id() if self.sp_model.unk_id() != -1 else None # Map SP's -1 to None
        self.bos_token_id = self.sp_model.bos_id() if self.sp_model.bos_id() != -1 else None
        self.eos_token_id = self.sp_model.eos_id() if self.sp_model.eos_id() != -1 else None
        self.pad_token_id = self.sp_model.pad_id() if self.sp_model.pad_id() != -1 else None

        # Rebuild vocab mappings
        self.idx_to_token = {i: self.sp_model.id_to_piece(i) for i in range(self.vocab_size)}
        # Ensure special tokens from config are mapped correctly, even if SP model used different IDs internally
        # Base class __init__ already sets self.special_tokens_map
        for role, token in self.special_tokens_map.items():
            if token is not None:
                try:
                    token_id = self.sp_model.piece_to_id(token)
                    self.idx_to_token[token_id] = token # Ensure our string is used
                    setattr(self, f"{role}_token_id", token_id)
                except ValueError: # Should not happen if token is in vocab
                     logger.warning(f"Special token '{token}' (role: {role}) not found in loaded SentencePiece model vocab.")


        self.token_to_idx = {v: k for k, v in self.idx_to_token.items()}
        logger.debug(f"Synced vocab from SentencePiece model. Size: {self.vocab_size}")
        logger.debug(f"  UNK ID: {self.unk_token_id} ('{self.unk_token}')")
        logger.debug(f"  BOS ID: {self.bos_token_id} ('{self.bos_token}')")
        logger.debug(f"  EOS ID: {self.eos_token_id} ('{self.eos_token}')")
        logger.debug(f"  PAD ID: {self.pad_token_id} ('{self.pad_token}')")



    @classmethod
    def load(cls, model_path_prefix: Union[str, Path], **kwargs) -> "SentencePieceTokenizer":
        """Loads a tokenizer from a saved SentencePiece model prefix."""
        model_path_prefix = str(model_path_prefix) # Ensure string path
        model_file = Path(f"{model_path_prefix}.model")
        metadata_file = Path(f"{model_path_prefix}.json")

        if not model_file.is_file():
            raise FileNotFoundError(f"SentencePiece model file not found: {model_file}")
        if not metadata_file.is_file():
             raise FileNotFoundError(f"SentencePiece metadata file not found: {metadata_file}") # Now required

        # Pass the prefix to __init__ for loading
        return cls(model_path=model_path_prefix, **kwargs)


    def train(
        self,
        input_file: Union[str, Path],
        output_dir: Union[str, Path],
        vocab_size: Optional[int] = None,
        model_prefix: Optional[str] = None,
        model_type: Optional[str] = None,
        character_coverage: Optional[float] = None,
        special_tokens_override: Optional[Dict[str, str]] = None,
        add_bos_token: Optional[bool] = None, # Allow override during train call
        add_eos_token: Optional[bool] = None, # Allow override during train call
    ):
        """
        Trains a SentencePiece model.

        Args:
            input_file: Path to the input text corpus for training.
            output_dir: Directory to save the trained model and vocab.
            vocab_size: Override the instance's vocabulary size.
            model_prefix: Override the instance's model prefix.
            model_type: Override the instance's model type.
            character_coverage: Override the instance's character coverage.
            special_tokens_override: Override the instance's special tokens for this training run.
            add_bos_token: Override whether to treat BOS as control symbol.
            add_eos_token: Override whether to treat EOS as control symbol.
        """
        input_file = str(input_file)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True) # Use Path.mkdir instead

        # Use overrides or instance attributes, defaulting to False if instance attribute is None or not set
        vs = vocab_size if vocab_size is not None else self.vocab_size_config
        prefix = model_prefix if model_prefix is not None else self.model_prefix_config
        mt = model_type if model_type is not None else self.model_type
        cc = character_coverage if character_coverage is not None else self.character_coverage
        add_bos = add_bos_token if add_bos_token is not None else (getattr(self, 'add_bos_token', False) if getattr(self, 'add_bos_token', False) is not None else False)
        add_eos = add_eos_token if add_eos_token is not None else (getattr(self, 'add_eos_token', False) if getattr(self, 'add_eos_token', False) is not None else False)


        if vs is None or prefix is None:
            raise ValueError(
                "Cannot train SentencePieceTokenizer without 'vocab_size' and "
                "'model_prefix' being set during initialization or configured."
            )

        # Combine special tokens: override -> instance -> base defaults handled by self.special_tokens_map
        # Start with the current instance's map (which includes defaults from base class)
        train_special_tokens = self.special_tokens_map.copy() # Make a copy to avoid modifying the instance's map
        # No need to update from defaults again, as self.special_tokens_map already has them.
        if special_tokens_override: # Apply overrides specific to this training run
            train_special_tokens.update(special_tokens_override)


        model_prefix_path = output_dir / prefix
        logger.info(f"Starting SentencePiece training...")
        logger.info(f"  Input: {input_file}")
        logger.info(f"  Output Prefix: {model_prefix_path}")
        logger.info(f"  Vocab Size: {vs}")
        logger.info(f"  Model Type: {mt}")
        logger.info(f"  Character Coverage: {cc}")
        logger.info(f"  Special Tokens: {train_special_tokens}")
        logger.info(f"  Add BOS: {add_bos}")
        logger.info(f"  Add EOS: {add_eos}")


        # Prepare special tokens for SentencePiece Trainer
        # Needs comma-separated strings and flags for control/user defined
        control_symbols = []
        user_defined_symbols = []
        token_to_role = {v: k for k, v in train_special_tokens.items()}

        # Standard SP control symbols if present
        if train_special_tokens.get("bos"): control_symbols.append(train_special_tokens["bos"])
        if train_special_tokens.get("eos"): control_symbols.append(train_special_tokens["eos"])
        if train_special_tokens.get("pad"): control_symbols.append(train_special_tokens["pad"])
        # UNK is handled by unk_piece/unk_id, not directly here? Let's assume unk is user-defined if not handled by spm
        # All others are user-defined
        for token in train_special_tokens.values():
             if token and token not in control_symbols:
                 # Check if it's the UNK piece - SP handles this via unk_piece
                 role = token_to_role.get(token)
                 if role != 'unk':
                    user_defined_symbols.append(token)


        # Build command line arguments for SentencePieceTrainer.train
        spm_args = {
            "input": input_file,
            "model_prefix": str(model_prefix_path),
            "vocab_size": vs,
            "model_type": mt,
            "character_coverage": cc,
            "unk_piece": train_special_tokens.get("unk", "<unk>"), # Corrected default
            "bos_piece": train_special_tokens.get("bos", "<s>"),   # Corrected default
            "eos_piece": train_special_tokens.get("eos", "</s>"),   # Corrected default
            "pad_piece": train_special_tokens.get("pad"),           # Corrected default (None if not in map)
             # SPM uses add_dummy_prefix=False by default which seems safer
             # Avoid issues with leading spaces being tokenized differently
             "add_dummy_prefix": False,
             "remove_extra_whitespaces": True, # Generally useful
             "normalization_rule_name": "nmt_nfkc_cf", # Recommended default
             "split_digits": True,
            # Let SPM handle IDs based on pieces/symbols, but specify UNK ID
            # Use self.unk_id from base class, which should be set during __init__
            "unk_id": self.unk_id if self.unk_token and self.unk_id is not None else 0, # Default to 0 if not set, SPM requires non -1 unk_id
            "bos_id": -1, # Let SPM assign based on piece/control symbol
            "eos_id": -1, # Let SPM assign based on piece/control symbol
            "pad_id": -1, # Let SPM assign based on piece/control symbol
        }

        # Handle control/user symbols carefully based on SentencePiece expectations
        # Only add flags if the corresponding list is not empty
        if control_symbols:
            spm_args["control_symbols"] = ",".join(control_symbols)
        if user_defined_symbols:
             spm_args["user_defined_symbols"] = ",".join(user_defined_symbols)


        # Handle BOS/EOS IDs - map boolean flags to SP arguments
        # bos_id=-1 and eos_id=-1 means don't add them automatically
        spm_args["bos_id"] = -1 # Disable automatic addition by default
        spm_args["eos_id"] = -1 # Disable automatic addition by default
        if add_bos and train_special_tokens.get("bos"):
             # We let the user handle adding BOS via encode options later
             # but ensure the token exists if requested
             logger.debug("BOS token will be included in vocab if 'add_bos_token' is True.")
        if add_eos and train_special_tokens.get("eos"):
             logger.debug("EOS token will be included in vocab if 'add_eos_token' is True.")


        # Filter out None values before passing to train
        spm_args_filtered = {k: v for k, v in spm_args.items() if v is not None}
        spm_args_str = " ".join(f"--{k}={v}" for k, v in spm_args_filtered.items())
        logger.debug(f"SPM Train Args: {spm_args_str}")


        try:
            spm.SentencePieceTrainer.train(**spm_args_filtered)
            logger.info(f"SentencePiece model trained successfully: {model_prefix_path}.model")

            # --- Save Metadata ---
            self.metadata = {
                "vocab_size": vs, # Save the *requested* vocab size
                "model_prefix": prefix,
                "model_type": mt,
                "character_coverage": cc,
                "special_tokens": train_special_tokens, # Save the actual tokens used
                "add_bos_token": add_bos, # Save the config used for training
                "add_eos_token": add_eos, # Save the config used for training
                "spm_args": spm_args_filtered, # Save args for reference
            }
            metadata_path = Path(f"{model_prefix_path}.json")
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(self.metadata, f, indent=2)
            logger.info(f"Saved SentencePiece metadata to: {metadata_path}")

            # --- Load the trained model into this instance ---
            self.sp_model = spm.SentencePieceProcessor(str(f"{model_prefix_path}.model"))
            self.model_path_prefix = str(model_prefix_path) # Update instance path
            # Update base class attributes based on loaded model & saved metadata
            self.vocab_size_config = self.metadata.get("vocab_size")
            self.special_tokens_map = self.metadata.get("special_tokens", {})
            self.add_bos_token = self.metadata.get("add_bos_token", False)
            self.add_eos_token = self.metadata.get("add_eos_token", False)
            self._sync_vocab_from_sp_model() # Resync vocab/ids


        except Exception as e:
             logger.exception(f"SentencePiece training failed: {e}")
             raise RuntimeError("SentencePiece training failed.") from e


    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Encodes text into a list of token IDs."""
        if not self.sp_model:
            raise RuntimeError("SentencePiece model not loaded or trained.")

        # SentencePiece handles add_bos/eos via encode options if model configured for it
        # Our add_bos/add_eos flags control whether we *request* this behaviour.
        # The actual IDs come from the model.
        encode_options = []
        if add_special_tokens and self.add_bos_token and self.bos_token_id is not None:
            encode_options.append("bos")
        if add_special_tokens and self.add_eos_token and self.eos_token_id is not None:
            encode_options.append("eos")

        # Construct the option string if needed
        encode_extra_options = f"enable_sampling=false,alpha=0.1,nbest_size=-1:{','.join(encode_options)}" if encode_options else ""


        return self.sp_model.encode(text, out_type=int, add_bos=False, add_eos=False, extra_options=encode_extra_options)
        # return self.sp_model.encode(
        #     text,
        #     out_type=int,
        #     add_bos=add_special_tokens and self.add_bos_token,
        #     add_eos=add_special_tokens and self.add_eos_token,
        # )


    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decodes a list of token IDs back into text."""
        if not self.sp_model:
            raise RuntimeError("SentencePiece model not loaded or trained.")

        # Filter out special tokens *before* decoding if requested
        if skip_special_tokens:
            special_ids_to_skip = {
                self.pad_token_id,
                self.bos_token_id,
                self.eos_token_id,
                # Don't usually skip UNK during decode unless explicitly told?
                # self.unk_token_id
            }
            # Also skip any other user-defined special tokens if they have IDs
            for role, token_id in [
                ('cls_token_id', getattr(self, 'cls_token_id', None)),
                ('sep_token_id', getattr(self, 'sep_token_id', None)),
                ('mask_token_id', getattr(self, 'mask_token_id', None))
             ]:
                 if token_id is not None:
                     special_ids_to_skip.add(token_id)


            filtered_ids = [id_ for id_ in token_ids if id_ is not None and id_ not in special_ids_to_skip]
            # Handle case where all tokens are special
            if not filtered_ids and token_ids:
                 # If original list wasn't empty but filtered is, maybe return empty string?
                 # Or decode the original list? Let's decode original for now.
                 logger.debug("All tokens were special tokens; decoding original list.")
                 filtered_ids = token_ids # Revert to original if filter result is empty
            elif not filtered_ids and not token_ids:
                 return "" # Return empty if input was empty


        else:
            filtered_ids = token_ids

        # SPM decode handles removal of underlying sentencepiece formatting marks like ' '
        return self.sp_model.decode(filtered_ids)


    def save(self, output_dir: Union[str, Path], model_prefix: Optional[str] = None):
        """Saves the tokenizer model and configuration metadata."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        prefix = model_prefix if model_prefix is not None else self.model_prefix_config
        if not prefix:
            logger.error("Cannot save tokenizer without a model_prefix.")
            return

        output_model_path = output_dir / f"{prefix}.model"
        output_metadata_path = output_dir / f"{prefix}.json"

        # Handle the .model file
        # The .model file is expected to be generated by SentencePieceTrainer.train
        # If the tokenizer was loaded, self.model_path_prefix should point to the original.
        model_file_saved_or_copied = False
        if self.model_path_prefix:
            original_model_path = Path(f"{self.model_path_prefix}.model")
            if original_model_path.exists():
                logger.info(f"Copying existing model file from {original_model_path} to {output_model_path}")
                try:
                    shutil.copy2(original_model_path, output_model_path)
                    model_file_saved_or_copied = True
                except Exception as e:
                    logger.error(f"Failed to copy existing model file: {e}")
            else:
                logger.warning(f"Original model path {original_model_path} does not exist. Cannot copy.")

        # Check if the model file exists at the target path (either copied or created by training)
        if not output_model_path.exists():
            logger.warning(
                 f"SentencePiece model file {output_model_path} was not found. "
                 f"It should have been created by training or copied from the load path."
            )
        elif not model_file_saved_or_copied:
             logger.info(f"SentencePiece model file {output_model_path} found (presumably created by training). Skipping copy.")
             model_file_saved_or_copied = True # Mark as handled

        # Always save the metadata (config)
        logger.info(f"Saving tokenizer metadata to: {output_metadata_path}")
        metadata_to_save = {
            "model_type": self.model_type,
            "vocab_size": self.vocab_size_config, # Save the configured vocab size
            "character_coverage": self.character_coverage,
            "special_tokens": self.special_tokens_map, # Save the final map used
            "add_bos_token": self.add_bos_token,
            "add_eos_token": self.add_eos_token,
            "model_prefix": prefix # Save the prefix used for saving
            # Add any other relevant config attributes here
        }
        try:
            with open(output_metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata_to_save, f, indent=4)
            logger.info(f"Successfully saved metadata to {output_metadata_path}")
        except Exception as e:
            logger.error(f"Failed to save tokenizer metadata to {output_metadata_path}: {e}")

    # -------------------
    # Core Methods
    # -------------------

    def get_vocab(self) -> Dict[str, int]:
        """Returns the vocabulary mapping tokens to IDs."""
        if not self.sp_model:
             # Return mapping based on config if not loaded/trained yet
             # This might be incomplete or just special tokens
             logger.warning("SentencePiece model not loaded; returning vocab based on configuration.")
             # Rebuild token_to_idx based on current special tokens and potentially vocab size
             temp_vocab = {token: i for i, token in enumerate(self.special_tokens_map.values()) if token is not None}
             # Add placeholder indices if vocab size is known but model not loaded? Risky.
             return temp_vocab
        # Rebuild from idx_to_token which was synced from sp_model
        return {token: idx for idx, token in self.idx_to_token.items()}

    def get_vocab_size(self) -> int:
        """Returns the effective vocabulary size."""
        if self.sp_model:
            return self.sp_model.get_piece_size()
        elif self.vocab_size_config:
            return self.vocab_size_config
        else:
             logger.warning("Cannot determine vocab size: model not loaded and vocab_size not configured.")
             return 0

    def token_to_id(self, token: str) -> Optional[int]:
        """Converts a token string to its ID."""
        if not self.sp_model:
            # Fallback to configured vocab if possible
             return self.token_to_idx.get(token) # Uses map built by base class
        try:
            # Use SP model directly for consistency
            return self.sp_model.piece_to_id(token)
        except ValueError: # SentencePiece raises ValueError for OOV tokens
            return self.unk_token_id # Return configured UNK id


    def id_to_token(self, token_id: int) -> Optional[str]:
        """Converts a token ID to its string representation."""
        if not self.sp_model:
             # Fallback to configured vocab if possible
             return self.idx_to_token.get(token_id) # Uses map built by base class
        if 0 <= token_id < self.sp_model.get_piece_size():
             # Use SP model directly
             return self.sp_model.id_to_piece(token_id)
        else:
             # Handle out-of-range IDs - potentially return UNK string?
             logger.warning(f"ID {token_id} is out of range for the SentencePiece vocabulary (Size: {self.sp_model.get_piece_size()}).")
             # Returning UNK string might be better than None if an ID was provided
             return self.unk_token # Return configured UNK token string

    @property
    def config(self) -> Dict[str, Any]:
        """Returns the tokenizer's configuration."""
        base_config = super().config
        sp_config = {
            "model_path": self.model_path_prefix,
            "vocab_size": self.vocab_size_config, # Configured size
            "model_prefix": self.model_prefix_config, # Configured prefix
            "model_type": self.model_type,
            "character_coverage": self.character_coverage,
            # "metadata": self.metadata # Maybe too verbose?
        }
        # Merge, ensuring base config (like special tokens map) is preserved
        base_config.update(sp_config)
        # Ensure vocab_size in config reflects loaded model if available
        if self.sp_model:
             base_config['vocab_size'] = self.sp_model.get_piece_size()
        return base_config