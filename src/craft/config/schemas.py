from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any, Literal
import logging

# Configure Pydantic to be compatible with OmegaConf/Hydra resolutions
# See: https://github.com/omry/omegaconf/issues/1162
BaseModel.model_config['validate_assignment'] = True
BaseModel.model_config['extra'] = 'ignore' # Ignore extra fields from Hydra like _target_ by default

logger = logging.getLogger(__name__)

# --- Individual Component Schemas ---

class OptimizerConfig(BaseModel):
    target: str = Field(..., validation_alias='_target_', description="Target class for the optimizer (e.g., torch.optim.AdamW)")
    lr: float = Field(..., gt=0, description="Learning rate")
    weight_decay: float = Field(0.0, ge=0, description="Weight decay")
    # Allow other optimizer-specific parameters
    # Using model_extra allows Pydantic to capture these without explicit definition
    model_config = {'extra': 'allow'}

class SchedulerConfig(BaseModel):
    target: str = Field(..., validation_alias='_target_', description="Target class for the scheduler (e.g., torch.optim.lr_scheduler.CosineAnnealingLR)")
    # Allow other scheduler-specific parameters
    model_config = {'extra': 'allow'}

# More flexible Callback config initially - Hydra instantiates these
class CallbackConfigEntry(BaseModel):
    target: str = Field(..., validation_alias='_target_', description="Target class for the callback")
    # Allow any other parameters for the specific callback
    model_config = {'extra': 'allow'}

# Type hint for the callbacks section in the main config
CallbacksConfig = Optional[Dict[str, CallbackConfigEntry]]


class TrainingConfig(BaseModel):
    batch_size: int = Field(..., gt=0, description="Batch size for training")
    num_epochs: Optional[int] = Field(1, gt=0, description="Number of training epochs (alternative to max_steps)")
    max_steps: Optional[int] = Field(None, gt=0, description="Maximum number of training steps (alternative to num_epochs)")
    use_amp: bool = Field(False, description="Whether to use Automatic Mixed Precision")
    gradient_accumulation_steps: int = Field(1, ge=1, description="Number of steps to accumulate gradients over")
    max_grad_norm: Optional[float] = Field(None, gt=0, description="Maximum gradient norm for clipping")
    log_interval: int = Field(50, gt=0, description="Log training metrics every N steps")
    eval_interval: int = Field(1000, ge=0, description="Evaluate on validation set every N steps (0 to disable)")
    save_interval: int = Field(5000, ge=0, description="Save checkpoint every N steps (0 to disable)")
    # Add time-based intervals if needed later
    time_save_interval_seconds: Optional[int] = Field(None, ge=0, description="Save checkpoint every N seconds")
    time_eval_interval_seconds: Optional[int] = Field(None, ge=0, description="Evaluate every N seconds")
    warmup_steps: Optional[int] = 100
    weight_decay: Optional[float] = 0.1
    save_steps_interval: Optional[int] = 0 # Keep every N steps
    compile_model: Optional[bool] = False # Placeholder
    activation_checkpointing: Optional[bool] = False
    torch_compile: Optional[bool] = False
    # generation settings within training for sampling
    sample_max_new_tokens: Optional[int] = 100
    sample_temperature: Optional[float] = 0.8
    sample_start_text: Optional[str] = "The meaning of life is"


# Placeholder DataConfig - needs details based on specific data loaders
class DataConfig(BaseModel):
    # Define the structure for individual dataset splits (train/val/test)
    class DatasetSplitConfig(BaseModel):
        dataset: Dict[str, Any] # Expects a dictionary, target is inside
        # We might want to make this more specific later, e.g.:
        # class DatasetEntry(BaseModel):
        #     target: str = Field(..., validation_alias='_target_')
        #     file_path: str
        #     block_size: int
        #     model_config = {'extra': 'allow'}
        # dataset: DatasetEntry

    batch_size: int = Field(..., gt=0, description="Data loading batch size")
    num_workers: int = Field(0, ge=0, description="Number of workers for data loading")
    block_size: int = Field(..., gt=0, description="Sequence length for model input")
    train: Optional[DatasetSplitConfig] = None
    val: Optional[DatasetSplitConfig] = None
    test: Optional[DatasetSplitConfig] = None
    # Specific fields depend heavily on the chosen dataset (e.g., file paths, block_size)
    # Allow other top-level data fields if necessary, but restrict if possible
    model_config = {'extra': 'allow'}


# Placeholder ModelConfig - needs details based on specific models
class ModelConfig(BaseModel):
    target: str = Field(..., validation_alias='_target_', description="Target function/class for creating the model")
    # Fields like n_layer, n_head, n_embd, dropout, bias will go here
    # Vocab size is handled separately in train.py for now
    config: Optional[Dict[str, Any]] = Field(None, description="Nested config block specific to the model type")
    vocab_size: Optional[int] = Field(None, description="Vocabulary size (often inferred from data)")

    # Allow other model-specific fields
    model_config = {'extra': 'allow'}

class GenerationConfig(BaseModel):
    start_prompt: str = "\n"
    max_new_tokens: int = Field(200, gt=0)
    temperature: float = Field(0.8, ge=0)
    top_k: Optional[int] = Field(None, ge=1)


# --- Main Application Schema ---

class AppConfig(BaseModel):
    # Config groups loaded by Hydra
    model: ModelConfig
    training: TrainingConfig
    data: DataConfig
    optimizer: OptimizerConfig
    scheduler: Optional[SchedulerConfig] = None # Scheduler is optional
    callbacks: CallbacksConfig = None # Callbacks are optional
    generation: GenerationConfig

    # Top-level parameters
    project_name: str = "Craft"
    experiment_name: Optional[str] = "default_experiment"
    seed: int = 42
    device: Literal['auto', 'cpu', 'cuda'] = 'auto'
    log_level: str = "INFO"
    force_cpu: bool = False # Potentially redundant with device?
    resume_from: Optional[str] = None # Path or "latest"

    # Allow other top-level fields if necessary, but maybe restrict later
    model_config = {'extra': 'allow'}


# Example usage (for testing schemas, not used in runtime directly here):
if __name__ == '__main__':
    # Example raw dictionary mimicking Hydra output
    test_conf_dict = {
        'model': {'_target_': 'craft.models.create_model', 'n_layer': 2, 'config': {'vocab_size': 50}},
        'training': {'batch_size': 4, 'max_steps': 100, 'use_amp': True},
        'data': {'_target_': 'craft.data.create_loaders', 'batch_size': 4, 'path': '/data'},
        'optimizer': {'_target_': 'torch.optim.AdamW', 'lr': 1e-3},
        'scheduler': {'_target_': 'torch.optim.lr_scheduler.CosineAnnealingLR', 'T_max': 100},
        'callbacks': {'checkpoint': {'_target_': 'craft.callbacks.CheckpointCallback'}},
        'generation': {'max_new_tokens': 50},
        'seed': 123,
        'device': 'cuda'
    }
    try:
        validated_config = AppConfig(**test_conf_dict)
        print("Validation Successful:")
        print(validated_config.model_dump_json(indent=2))

        # Test missing required field
        invalid_conf = test_conf_dict.copy()
        del invalid_conf['training']['batch_size']
        # AppConfig(**invalid_conf) # Uncomment to test validation error

    except Exception as e:
        print(f"Validation Failed: {e}") 