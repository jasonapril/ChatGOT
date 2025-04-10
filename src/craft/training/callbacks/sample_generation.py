import logging
import torch
from typing import List, Optional, Dict, Any
import sys

from .base import Callback, TYPE_CHECKING
# Assume TextGenerator lives elsewhere, e.g., in craft.training.generation
# Adjust the import path as necessary
from ..generation import TextGenerator

# Added TYPE_CHECKING block
if TYPE_CHECKING:
    from ..data.base import BaseDataset # type: ignore[import-not-found]
    from ..trainer import Trainer

logger = logging.getLogger(__name__)

class SampleGenerationCallback(Callback):
    """
    Generates text samples using the model during training.

    Can generate samples at the end of specified steps and/or epochs.
    Logs the generated samples to the console.
    """
    def __init__(self, step_interval: Optional[int] = None,
                 epoch_interval: Optional[int] = None,
                 start_prompt: str = "The meaning of life is ",
                 max_new_tokens: int = 50,
                 temperature: float = 0.8,
                 # Add other generation params if needed (top_k, top_p, etc.)
                 ) -> None:
        """
        Args:
            step_interval (Optional[int]): Generate samples every N steps. If None, disabled.
            epoch_interval (Optional[int]): Generate samples every N epochs. If None, disabled.
            start_prompt (str): The prompt to use for generation.
            max_new_tokens (int): Maximum number of new tokens to generate.
            temperature (float): Sampling temperature.
        """
        super().__init__()
        self.step_interval: Optional[int] = step_interval
        self.epoch_interval: Optional[int] = epoch_interval
        self.start_prompt: str = start_prompt
        self.max_new_tokens: int = max_new_tokens
        self.temperature: float = temperature
        self.generator: Optional[TextGenerator] = None # Initialize generator as None
        self.initialized: bool = False # Flag to track if generator is initialized
        self.logger = logging.getLogger(self.__class__.__name__)

        if self.step_interval is None and self.epoch_interval is None:
            # Clarified warning message
            self.logger.warning("SampleGenerationCallback initialized with step_interval=None and epoch_interval=None. "
                              "Callback will not trigger based on steps or epochs. "
                              "(Time-based triggering might still be active via TrainingLoop)." )

    def on_train_begin(self, **kwargs: Any) -> None:
        # Defer generator creation until we have the trainer and its device/model/dataset
        # Use self.trainer now, assuming set_trainer was called
        if self.trainer and hasattr(self.trainer, 'model') and hasattr(self.trainer, 'device') and hasattr(self.trainer, 'train_dataloader') and hasattr(self.trainer.train_dataloader, 'dataset'):
             try:
                 # Pass the necessary components to TextGenerator
                 # TextGenerator expects model, device, config (dict), dataset (with tokenizer/decode)
                 # Also explicitly pass the trainer's tokenizer
                 self.generator = TextGenerator(
                     model=self.trainer.model,
                     device=self.trainer.device,
                     dataset=self.trainer.train_dataloader.dataset, # Pass dataset object
                     config={}, # Pass empty config for now,
                     tokenizer=getattr(self.trainer, 'tokenizer', None) # Pass tokenizer from trainer
                 )
                 self.logger.info("TextGenerator initialized within SampleGenerationCallback.")
                 self.initialized = True
             except AttributeError as e:
                 self.logger.error(f"Failed to initialize TextGenerator: Missing required trainer/dataloader/dataset/tokenizer attribute ({e})")
                 self.initialized = False
             except Exception as e:
                 self.logger.error(f"Failed to initialize TextGenerator: {e}", exc_info=True)
                 self.initialized = False
        else:
             self.logger.error("Cannot initialize TextGenerator: Trainer or its model/device/train_dataloader/dataset is not available.")
             self.initialized = False

    def _initialize_generator(self, trainer: Optional['Trainer']) -> bool:
        """Attempts to initialize the TextGenerator."""
        self.logger.info("Initializing TextGenerator for sample generation.")
        if not self.trainer or not hasattr(self.trainer, 'model') or not hasattr(self.trainer, 'device') or \
           not hasattr(self.trainer, 'train_dataloader') or not self.trainer.train_dataloader or \
           not hasattr(self.trainer.train_dataloader, 'dataset') or not self.trainer.train_dataloader.dataset or \
           not hasattr(self.trainer, 'tokenizer'): # Check for tokenizer explicitly
            self.logger.error("Trainer is missing required attributes (model, device, train_dataloader with dataset, tokenizer) for TextGenerator initialization.")
            return False
        
        # --- Get Components from Trainer --- # 
        model = self.trainer.model
        device = self.trainer.device
        dataset: Optional['BaseDataset'] = self.trainer.train_dataloader.dataset # Use Optional[BaseDataset] hint
        explicit_tokenizer = getattr(self.trainer, 'tokenizer', None) # Get tokenizer directly from trainer

        self.logger.debug(f"_initialize_generator: Received dataset object ID: {id(dataset)}")
        if dataset is None:
            self.logger.error("Trainer dataloader dataset is None. Cannot initialize TextGenerator.")
            return False
        self.logger.debug(f"_initialize_generator: Dataset object type: {type(dataset)}")
        if explicit_tokenizer:
            self.logger.debug(f"_initialize_generator: Received explicit tokenizer from trainer: {type(explicit_tokenizer)} (ID: {id(explicit_tokenizer)})")
        else:
            self.logger.debug("_initialize_generator: No explicit tokenizer found on trainer.")
        # --------------------------------- #
            
        # --- Add check for explicit tokenizer --- #
        if explicit_tokenizer is None:
            self.logger.error("Trainer does not have an explicit 'tokenizer' attribute needed for TextGenerator.")
            return False # FAIL if tokenizer is missing
        # Ensure dataset is not None here before passing
        if dataset is None:
             self.logger.error("Dataset became None unexpectedly before TextGenerator init.")
             return False # FAIL if tokenizer is missing
        # --------------------------------------- #
            
        # Ensure dataset is not None before passing (runtime check + mypy assertion)
        if dataset is None:
            self.logger.error("Dataset became None unexpectedly before TextGenerator init.")
            return False
        assert dataset is not None # Assert for mypy

        try:
            # Pass the trainer's model, device, dataset, and *explicit tokenizer*.
            self.generator = TextGenerator(
                model=model,
                device=device,
                dataset=dataset, # type: ignore[arg-type]
                tokenizer=explicit_tokenizer,
                config={} # Pass empty config for now
            )
            self.logger.info(f"TextGenerator initialized within SampleGenerationCallback. Explicit tokenizer used: {bool(explicit_tokenizer)}")
            self.initialized = True
            return True
        except AttributeError as e:
            self.logger.error(
                f"Failed to initialize TextGenerator: Missing required attribute in Trainer (e.g., model, device, train_dataloader): {e}",
                exc_info=True
            )
            self.generator = None # Ensure generator is None if init fails
            self.initialized = False
        except Exception as e: # Catch other potential errors during init
            self.logger.error(f"Failed to initialize TextGenerator: {e}", exc_info=True)
            self.generator = None
            self.initialized = False

        return False

    def on_epoch_begin(self, epoch: int, **kwargs: Any) -> None:
        """Called at the start of each training epoch."""
        # Optionally reset any epoch-specific state here
        global_step = kwargs.get('global_step', -1)
        self.logger.debug(
            f"[Epoch Start] Callback {self.__class__.__name__}: Epoch {epoch+1} begin at step {global_step}."
        )

    def on_epoch_end(self, epoch: int, global_step: int, metrics: Dict[str, Any], **kwargs: Any) -> None:
        """Generate samples at the end of an epoch if interval matches."""
        # Check interval
        if self.epoch_interval and (epoch + 1) % self.epoch_interval == 0:
            self.generate_samples(f"Epoch {epoch + 1}") # Pass only trigger event
            self.logger.info("Finished generating samples.")

    def on_step_begin(self, step: int, **kwargs: Any) -> None:
        """Called at the beginning of a training step."""
        pass # No action needed

    def on_step_end(self, step: int, global_step: int, metrics: Dict[str, Any], **kwargs: Any) -> None:
        """Generate samples at the end of a step if interval matches."""
        # Use global_step passed from Trainer
        if self.step_interval and (global_step + 1) % self.step_interval == 0:
            self.generate_samples(f"Step {global_step + 1}") # Pass only trigger event
            self.logger.info("Finished generating samples.")

    def generate_samples(self, trigger_event: str) -> None:
        """Generates and logs sample text."""
        # Attempt initialization if generator is None or not initialized
        if not self.generator or not self.initialized: # Check both flags
            self.logger.debug("Generator not ready, attempting initialization in generate_samples.")
            if not self._initialize_generator(self.trainer): # Pass self.trainer here
                 self.logger.error("TextGenerator could not be initialized. Cannot generate samples.")
                 return
            # Check again after attempt
            if not self.generator:
                 self.logger.error("TextGenerator is still None after initialization attempt. Cannot generate samples.")
                 return
            else:
                 self.logger.debug("TextGenerator successfully initialized by generate_samples.")

        # Proceed only if generator is valid
        self.logger.info(f"--- Generating Sample ({trigger_event}) ---")
        self.logger.info(f"Prompt: {self.start_prompt}")
        generated_texts: List[str] # Added type hint
        try:
            # Use parameters stored in the callback instance
            # Make sure self.generator is not None before calling
            if self.generator is None:
                 self.logger.error("Generator is None in generate_samples despite initialization checks.")
                 return
                 
            generated_texts = self.generator.generate_text(
                prompt=self.start_prompt, # Changed from start_prompt
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                # Pass other params if added to __init__
            )

            if not generated_texts:
                self.logger.warning("Sample generation produced no output.")
                generated_texts = ["[No text generated]"]
            # Removed the explicit decoding block, assuming generate_text returns decoded strings

        except AttributeError as e:
             # Catch specific errors like missing tokenizer methods
             self.logger.error(f"Error during sample generation (AttributeError): {e}. Check if tokenizer is correctly initialized and has required methods.", exc_info=True)
             generated_texts = [f"[Generation Error: {e}]"]
        except Exception as e:
            self.logger.error(f"Error during text generation: {e}", exc_info=True)
            generated_texts = [f"[Generation Error: {e}]"]

        # Log the generated text, handling potential console encoding issues
        raw_text = generated_texts[0] if generated_texts else "[No text generated]"
        try:
            # Directly print UTF-8 encoded bytes to the standard output buffer
            # This bypasses the logger and potential console encoding issues for this specific output
            output_bytes = f"Generated: {raw_text}\n".encode('utf-8', errors='replace')
            sys.stdout.buffer.write(output_bytes)
            sys.stdout.flush() # Ensure it gets written immediately
        except Exception as print_e:
            # Fallback if direct printing fails
            self.logger.error(f"Error directly printing generated text to console: {print_e}. Raw text snippet: {repr(raw_text[:100])}")

        self.logger.info(f"--- End Sample Generation ({trigger_event}) ---")

    def on_train_end(self, metrics: Optional[Dict[str, Any]] = None, **kwargs: Any) -> None:
        """Called at the end of training."""
        pass

    # Removed duplicate on_step_begin definition
    # Corrected signature
    # def on_step_begin(self, step: int, logs=None):
    #     pass 