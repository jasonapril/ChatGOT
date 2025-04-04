import logging
import torch
from typing import List, Optional

from .base import Callback
# Assume TextGenerator lives elsewhere, e.g., in craft.training.generation
# Adjust the import path as necessary
from ..generation import TextGenerator

logger = logging.getLogger(__name__)

class SampleGenerationCallback(Callback):
    """
    Generates text samples using the model during training.

    Can generate samples at the end of specified steps and/or epochs.
    Logs the generated samples to the console.
    """
    def __init__(self, step_interval: Optional[int] = None,
                 epoch_interval: Optional[int] = None,
                 prompt: str = "The meaning of life is",
                 max_new_tokens: int = 50,
                 temperature: float = 0.7):
        """
        Args:
            step_interval (Optional[int]): Generate samples every N steps. If None, disabled.
            epoch_interval (Optional[int]): Generate samples every N epochs. If None, disabled.
            prompt (str): The prompt to use for generation.
            max_new_tokens (int): Maximum number of new tokens to generate.
            temperature (float): Sampling temperature.
        """
        super().__init__()
        if step_interval is None and epoch_interval is None:
            self.logger.warning("SampleGenerationCallback initialized but both step_interval and epoch_interval are None. Callback will be inactive.")
        self.step_interval = step_interval
        self.epoch_interval = epoch_interval
        self.prompt = prompt
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.generator = None

    def set_trainer(self, trainer):
        """Set the trainer and initialize the TextGenerator."""
        super().set_trainer(trainer)
        # Initialize the TextGenerator only if the model supports generation
        if hasattr(trainer.model, 'generate') and callable(getattr(trainer.model, 'generate')):
            # Pass the whole dataset object, TextGenerator will look for tokenizer/decode on it
            self.generator = TextGenerator(
                model=trainer.model,
                device=trainer.device,
                config=getattr(trainer, 'config', {}), # Pass trainer config (or empty dict)
                dataset=getattr(trainer, 'dataset', None) # Pass trainer dataset (or None)
            )
            if self.generator.dataset is None:
                 self.logger.warning(
                    "SampleGenerationCallback: Trainer has no 'dataset' attribute. "
                    "Encoding/Decoding for sample generation will fail."
                 )
            # Warning for missing tokenizer is now handled within TextGenerator's __init__ or generate_text
        else:
            self.generator = None
            self.logger.warning(
                f"SampleGenerationCallback: Model {type(trainer.model).__name__} does not have a .generate() method. "
                "Sample generation will be disabled."
            )

    def _generate_samples(self, step_or_epoch_info: str):
        """Internal method to generate and log samples."""
        if not self.generator or not self.prompt:
            # Log a warning only if intervals are set, otherwise it's expected
            if self.step_interval is not None or self.epoch_interval is not None:
                 self.logger.warning(f"Sample generation requested ({step_or_epoch_info}), but generator is not initialized or prompt is missing.")
            return

        self.logger.info(f"--- Generating Sample ({step_or_epoch_info}) ---")
        self.logger.info(f"Prompt: {self.prompt}")

        model_training_state = None # Initialize to handle potential errors before assignment
        try:
            # Ensure model is in eval mode for generation
            model_training_state = self.trainer.model.training
            self.trainer.model.eval()

            with torch.no_grad():
                generated_texts = self.generator.generate_text(
                    prompt=self.prompt,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    num_return_sequences=1 # Only generate one sample for logging
                )

            # Log the generated sample (assuming generate_text returns a list)
            if generated_texts:
                self.logger.info(f"Generated: {generated_texts[0]}")
            else:
                self.logger.warning("Generation produced no output.")

        except Exception as e:
            self.logger.error(f"Error during sample generation: {e}", exc_info=True)
        finally:
            # Restore model training state only if it was successfully retrieved
            if model_training_state is not None and self.trainer.model:
                self.trainer.model.train(mode=model_training_state)
        self.logger.info(f"--- End Sample Generation ({step_or_epoch_info}) ---")


    def on_step_end(self, step, logs=None):
        """Generate samples at specified step intervals."""
        if self.step_interval is not None and (step + 1) % self.step_interval == 0:
            self._generate_samples(f"Step {step + 1}")

    def on_epoch_end(self, trainer, epoch, logs=None):
        """Generate samples at specified epoch intervals."""
        if self.epoch_interval is not None and (epoch + 1) % self.epoch_interval == 0:
            self._generate_samples(f"Epoch {epoch + 1}")

    # Implement other abstract methods as no-ops
    def on_train_begin(self, trainer, logs=None):
        pass # Generator is initialized in set_trainer

    def on_train_end(self, trainer, logs=None):
        pass

    def on_epoch_begin(self, trainer, epoch, logs=None):
        pass

    def on_step_begin(self, step: int, logs=None):
        pass 