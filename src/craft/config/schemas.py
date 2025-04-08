from pydantic import BaseModel, Field, validator, ConfigDict, field_validator, model_validator
from typing import Optional, List, Dict, Any, Literal, Union, Tuple, Type, cast
from typing_extensions import Annotated
from pydantic_core.core_schema import ValidationInfo
from pydantic import FieldValidationInfo
import logging

# Configure Pydantic to be compatible with OmegaConf/Hydra resolutions
# See: https://github.com/omry/omegaconf/issues/1162
# Use model_config for Pydantic v2
_model_config_shared = ConfigDict(
    validate_assignment=True,
    extra='ignore' # Ignore extra fields like _target_ by default
)

logger = logging.getLogger(__name__)

# --- Specific Model Configuration Schemas (Moved from models/configs.py) --- #

class BaseModelConfig(BaseModel):
    """
    Base Pydantic configuration class for all models.
    Provides automatic validation and type hints.
    """
    model_config = ConfigDict(extra='allow') # Keep allow for flexibility at specific model level if needed later
    # model_type: str = Field("base", description="The type of the model (e.g., language, vision).") # Maybe remove if architecture is discriminator
    architecture: str = Field(..., description="Name for the specific model architecture (e.g., transformer, simple_rnn). Used as discriminator.")

class GenerativeModelConfig(BaseModelConfig):
    """Config for Generative Models"""
    # model_type: str = Field("generative", description="Model type set to generative.")
    max_seq_length: int = Field(1024, description="Maximum sequence length the model can handle")

class LanguageModelConfig(GenerativeModelConfig):
    """Base Config for most Language Models (e.g., Transformer)"""
    architecture: Literal["transformer"] = Field("transformer", description="Architecture set to transformer.") # Discriminator value
    vocab_size: Optional[int] = Field(None, description="Size of the vocabulary (often inferred).")
    d_model: int = Field(768, description="Model dimension.")
    n_head: int = Field(12, description="Number of attention heads.")
    d_hid: Optional[int] = Field(None, description="Hidden dimension in feed-forward layers.")
    n_layers: int = Field(12, description="Number of transformer layers.")
    dropout: float = Field(0.1, description="Dropout probability.")
    bias: bool = Field(True, description="Whether to use bias in linear layers.")
    layer_norm_eps: float = Field(1e-5, description="Epsilon for layer normalization.")
    activation: str = Field('gelu', description="Activation function.")
    norm_first: bool = Field(True, description="Apply layer norm before attention/FFN.")

    @field_validator('d_hid', mode='before')
    @classmethod
    def set_d_hid_default(cls, v: Any, info: ValidationInfo) -> Optional[int]:
        """Set default d_hid = d_model * 4 if not provided."""
        # Access validation context data via info.data
        d_model_val: Optional[int] = None

        if v is None and info.data and ('d_model' in info.data):
            d_model_val = info.data.get('d_model')
        elif v is None and 'd_model' in cls.model_fields and cls.model_fields['d_model'].default is not None:
             d_model_val = cls.model_fields['d_model'].default

        if d_model_val is not None and isinstance(d_model_val, int):
             return d_model_val * 4

        # If v was not None initially, or d_model couldn't be found/used, return original v
        # We need to ensure v is Optional[int] or None if it was None initially
        if isinstance(v, int) or v is None:
             return v
        else:
             # If v is something else (e.g., str from config), try converting or raise error
             # For now, returning None if conversion isn't straightforward
             logger.warning(f"Could not determine integer value for d_hid based on input {v} or d_model. Defaulting to None.")
             return None

class SimpleRNNConfig(GenerativeModelConfig):
    """Config for Simple RNN Models"""
    architecture: Literal["simple_rnn"] = Field("simple_rnn", description="Architecture set to simple_rnn.") # Discriminator value
    # Add back fields previously inherited from LanguageModelConfig that are needed
    vocab_size: Optional[int] = Field(None, description="Size of the vocabulary (often inferred).")
    d_model: int = Field(512, description="Model dimension (used for embedding in RNN).")
    dropout: float = Field(0.1, description="Dropout probability.")
    # RNN specific fields
    hidden_size: int = Field(512, description="Size of the RNN hidden state.")
    num_layers: int = Field(2, description="Number of RNN layers.")

    # Need to tell Pydantic which fields from parent NOT to use if overridden by name/purpose
    # For now, rely on factory logic to use hidden_size/num_layers correctly.

    @model_validator(mode='before')
    @classmethod # Need classmethod for model_validator
    # Cast return value from model_dump and add type hint for empty dict
    def extract_targets(cls, values: Any) -> Dict[str, Any]:
        if isinstance(values, dict):
            values_copy = values.copy()
            if 'dataset' in values_copy and isinstance(values_copy['dataset'], dict):
                dataset_params = values_copy['dataset'].copy()
                values_copy['dataset_target'] = dataset_params.pop('_target_', None)
                values_copy['dataset_params'] = dataset_params
            if 'dataloader' in values_copy and isinstance(values_copy['dataloader'], dict):
                dataloader_params = values_copy['dataloader'].copy()
                values_copy['dataloader_target'] = dataloader_params.pop('_target_', None)
                values_copy['dataloader_params'] = dataloader_params
            return values_copy
        elif isinstance(values, BaseModel):
             return cast(Dict[str, Any], values.model_dump())
        else:
            logger.warning(f"Unexpected input type {type(values)} for DatasetSplitConfig validator, expected dict or BaseModel. Returning empty dict.")
            # Explicitly type the empty dictionary
            return {}

class VisionModelConfig(BaseModelConfig):
    """Placeholder Config for Vision Models"""
    architecture: Literal["vision_transformer"] = Field("vision_transformer", description="Example vision architecture.") # Discriminator value
    # model_type: str = Field("vision", description="Model type set to vision.")
    image_size: Tuple[int, int] = Field((224, 224), description="Input image dimensions (height, width).")
    patch_size: int = Field(16, description="Size of image patches.")
    num_channels: int = Field(3, description="Number of input image channels.")
    # ... other vision-specific fields (e.g., d_model, n_layers for ViT)

class MultiModalModelConfig(BaseModelConfig):
    """Placeholder Config for Multi-Modal Models"""
    architecture: Literal["clip_style"] = Field("clip_style", description="Example multimodal architecture.") # Discriminator value
    # model_type: str = Field("multimodal", description="Model type set to multimodal.")
    # References to language and vision configs, or combined fields
    # Using Dict for now, could refine to specific Union types if needed
    language_config: Optional[Dict[str, Any]] = None
    vision_config: Optional[Dict[str, Any]] = None
    # ... other multimodal-specific fields

# --- Union of Specific Model Configs ---
# Create a Union type alias first
_ModelConfigUnion = Union[
    LanguageModelConfig,
    SimpleRNNConfig,
    VisionModelConfig,
    MultiModalModelConfig
    # Add other specific model configs here as they are created
]

# Use the 'architecture' field as the discriminator via Annotated
AnyModelConfig = Annotated[
    _ModelConfigUnion,
    Field(discriminator="architecture")
]


# --- Individual Component Schemas (Existing) ---

class OptimizerConfig(BaseModel):
    model_config = _model_config_shared.copy()
    # _target_ becomes target in Pydantic v2 field names unless aliased
    target: str = Field(..., validation_alias='_target_', description="Target class for the optimizer (e.g., torch.optim.AdamW)")
    lr: float = Field(..., gt=0, description="Learning rate")
    weight_decay: float = Field(0.0, ge=0, description="Weight decay")
    # Allow other optimizer-specific parameters
    model_config['extra'] = 'allow'


class SchedulerConfig(BaseModel):
    model_config = _model_config_shared.copy()
    target: str = Field(..., validation_alias='_target_', description="Target class for the scheduler (e.g., torch.optim.lr_scheduler.CosineAnnealingLR)")
    # Allow other scheduler-specific parameters
    model_config['extra'] = 'allow'

# More flexible Callback config initially - Hydra instantiates these
class CallbackConfigEntry(BaseModel):
    model_config = _model_config_shared.copy()
    target: str = Field(..., validation_alias='_target_', description="Target class for the callback")
    # Allow any other parameters for the specific callback
    model_config['extra'] = 'allow'

# Type hint for the callbacks section in the main config
CallbacksConfig = Optional[Dict[str, CallbackConfigEntry]]

# --- Data Configuration (Existing, minor adjustments maybe needed) ---

class DataConfig(BaseModel):
    model_config = _model_config_shared.copy()
    # Define the structure for individual dataset splits (train/val/test)
    class DatasetSplitConfig(BaseModel):
        # This inner model also needs the shared config
        model_config = _model_config_shared.copy()
        # dataset: Dict[str, Any] # Expects a dictionary, target is inside -> Let's make target explicit
        dataset_target: Optional[str] = Field(None, validation_alias='dataset._target_', description="Target class/function for the dataset")
        dataset_params: Dict[str, Any] = Field(default_factory=dict, validation_alias='dataset', description="Parameters for the dataset target (excluding _target_)")

        dataloader_target: Optional[str] = Field(None, validation_alias='dataloader._target_', description="Optional target class for the dataloader")
        dataloader_params: Dict[str, Any] = Field(default_factory=dict, validation_alias='dataloader', description="Parameters for the dataloader target (excluding _target_)")

        @model_validator(mode='before')
        @classmethod # Need classmethod for model_validator
        def extract_targets(cls, values: Any) -> Dict[str, Any]:
            # Helper to separate _target_ from params if nested dicts are provided directly
            if isinstance(values, dict):
                # Make a copy to avoid modifying the original dict during iteration/validation if necessary
                values_copy = values.copy()
                if 'dataset' in values_copy and isinstance(values_copy['dataset'], dict):
                    # Use pop to remove _target_ while getting its value, modify the copy
                    dataset_params = values_copy['dataset'].copy()
                    values_copy['dataset_target'] = dataset_params.pop('_target_', None)
                    values_copy['dataset_params'] = dataset_params
                    # Remove the original dataset key if we replace it with target/params
                    # values_copy.pop('dataset') # Or keep it if alias handles it?

                if 'dataloader' in values_copy and isinstance(values_copy['dataloader'], dict):
                    dataloader_params = values_copy['dataloader'].copy()
                    values_copy['dataloader_target'] = dataloader_params.pop('_target_', None)
                    values_copy['dataloader_params'] = dataloader_params
                    # values_copy.pop('dataloader')
                return values_copy # Return the modified copy
            elif isinstance(values, BaseModel): # Handle case where it might already be parsed?
                 # If it's already a BaseModel instance, return its dict representation
                 # This might not be needed depending on when the validator runs
                 return cast(Dict[str, Any], values.model_dump())
            else:
                # Or raise an error if the input type is unexpected
                # raise TypeError(f"Expected dict for DatasetSplitConfig, got {type(values)}")
                # For now, return original value if not dict
                logger.warning(f"Unexpected input type {type(values)} for DatasetSplitConfig validator, expected dict or BaseModel. Returning empty dict.")
                # Explicitly type the empty dictionary
                return {}
        # Allow other fields? Maybe not needed now with explicit target/params
        # model_config['extra'] = 'allow'

    # Common data parameters now at this level
    batch_size: int = Field(..., gt=0, description="Data loading batch size")
    num_workers: int = Field(0, ge=0, description="Number of workers for data loading")
    block_size: int = Field(..., gt=0, description="Sequence length for model input")

    # Expects a 'datasets' dictionary containing the splits
    datasets: Dict[Literal['train', 'val', 'test'], Optional[DatasetSplitConfig]]

    # Allow other top-level data fields if necessary? Keep ignore for now.
    # model_config['extra'] = 'allow'


# --- Generation Configuration (Existing) ---
class GenerationConfig(BaseModel):
    model_config = _model_config_shared.copy()
    start_prompt: str = "\\n"
    max_new_tokens: int = Field(200, gt=0)
    temperature: float = Field(0.8, ge=0)
    top_k: Optional[int] = Field(None, ge=1)


# --- Training Configuration (Existing) ---
class TrainingConfig(BaseModel):
    model_config = _model_config_shared.copy()
    batch_size: int = Field(..., gt=0, description="Batch size for training")
    num_epochs: Optional[int] = Field(1, gt=0, description="Number of training epochs (alternative to max_steps)")
    max_steps: Optional[int] = Field(None, gt=0, description="Maximum number of training steps (alternative to num_epochs)")
    use_amp: bool = Field(False, description="Whether to use Automatic Mixed Precision")
    gradient_accumulation_steps: int = Field(1, ge=1, description="Number of steps to accumulate gradients over")
    max_grad_norm: Optional[float] = Field(None, gt=0, description="Maximum gradient norm for clipping")
    log_interval: int = Field(50, gt=0, description="Log training metrics every N steps")
    eval_interval: int = Field(1000, ge=0, description="Evaluate on validation set every N steps (0 to disable)")
    save_interval: int = Field(5000, ge=0, description="Save checkpoint every N steps (0 to disable)")
    time_save_interval_seconds: int = Field(0, description="Save checkpoint every N seconds (0 to disable). Prioritized over step/epoch intervals if non-zero.")
    time_eval_interval_seconds: Optional[int] = Field(None, ge=0, description="Evaluate every N seconds")
    warmup_steps: Optional[int] = 100
    save_steps_interval: Optional[int] = 0 # Keep every N steps
    compile_model: Optional[bool] = False
    activation_checkpointing: Optional[bool] = False
    torch_compile: Optional[bool] = False
    # Revert default_factory back to Type and add type ignore
    generation: GenerationConfig = Field(default_factory=GenerationConfig) # type: ignore [arg-type]
    keep_last: Optional[int] = Field(None, description="Number of recent checkpoints to keep")
    resume_from_checkpoint: Optional[str] = Field(None, description="Path to resume from")
    checkpoint_dir: Optional[str] = Field(None, description="Directory to save checkpoints")


# --- Experiment Schema (Refactored ModelConfig) ---

class ExperimentConfig(BaseModel):
    """Schema for the core experiment parameters nested under 'experiment' key."""
    model_config = _model_config_shared.copy()
    # Use the discriminated union for the model config
    model: AnyModelConfig
    training: TrainingConfig
    data: DataConfig
    optimizer: OptimizerConfig
    scheduler: Optional[SchedulerConfig] = None # Scheduler is optional
    callbacks: CallbacksConfig = None # Callbacks are optional
    # Add experiment_name here
    experiment_name: Optional[str] = Field("default_experiment", description="Name of the experiment, used for logging and checkpointing.")
    # Allow extra fields? Keep ignore for now.
    # model_config['extra'] = 'allow'

# --- Main Application Schema (Unchanged Structure) ---

class AppConfig(BaseModel):
    """Main application configuration schema, expecting experiment settings under 'experiment'."""
    model_config = _model_config_shared.copy()
    experiment: ExperimentConfig

    # Top-level parameters (remain at the top)
    project_name: str = "Craft"
    # Remove experiment_name from here, it now lives in ExperimentConfig
    # experiment_name: Optional[str] = "default_experiment"
    seed: int = 42
    device: Literal['auto', 'cpu', 'cuda'] = 'auto'
    log_level: str = "INFO"
    force_cpu: bool = False # Potentially redundant with device?
    resume_from: Optional[str] = None # Path or "latest"

    # Allow other top-level fields? Keep ignore for now.
    # model_config['extra'] = 'allow'


# Example usage (for testing schemas) - Update example dict structure
# REMOVE THIS ENTIRE BLOCK
# if __name__ == '__main__':
#    # Example raw dictionary mimicking Hydra output for a Transformer
#    test_conf_dict_transformer = {
#        'experiment': {
#            'model': {
#                'architecture': 'transformer', # Discriminator
#                '_target_': 'craft.models.transformer.TransformerLanguageModel', # Example target
#                'vocab_size': 50257, # Example value
#                'max_seq_length': 1024,
#                'd_model': 768,
#                'n_head': 12,
#                # d_hid should be automatically calculated
#                'n_layers': 12,
#                # other LanguageModelConfig fields...
#            },
#            'training': {
#                '_target_': 'craft.some.training.target', # Example placeholder
#                'batch_size': 32,
#                'num_epochs': 5,
#                'use_amp': True,
#                # ... other training params
#            },
#            'data': {
#                'batch_size': 32, # Redundant?
#                'num_workers': 4,
#                'block_size': 1024,
#                'datasets': {
#                    'train': {
#                        'dataset': {
#                            '_target_': 'craft.data.datasets.TextDataset',
#                            'file_path': '/path/to/train.txt',
#                            'tokenizer_name': 'gpt2',
#                        },
#                        'dataloader': {
#                            'shuffle': True
#                        }
#                    },
#                    'val': {
#                        'dataset': {
#                            '_target_': 'craft.data.datasets.TextDataset',
#                            'file_path': '/path/to/val.txt',
#                            'tokenizer_name': 'gpt2',
#                        },
#                        'dataloader': {
#                            'shuffle': False
#                        }
#                    }
#                }
#            },
#            'optimizer': {
#                '_target_': 'torch.optim.AdamW',
#                'lr': 1e-4,
#                'weight_decay': 0.01
#            },
#            'scheduler': {
#                '_target_': 'torch.optim.lr_scheduler.CosineAnnealingLR',
#                'T_max': 10000 # Example
#            },
#            'callbacks': {
#                'tensorboard': {
#                    '_target_': 'craft.training.callbacks.TensorBoardLogger'
#                },
#                'checkpointing': {
#                    '_target_': 'craft.training.callbacks.ModelCheckpoint',
#                    'monitor': 'val_loss',
#                    'save_top_k': 1
#                }
#            }
#        },
#        # Top level app config
#        'project_name': 'TestCraft',
#        'experiment_name': 'transformer_test',
#        'device': 'cuda'
#    }
#    try:
#        app_config = AppConfig(**test_conf_dict_transformer)
#        print("Transformer Config Parsed Successfully:")
#        print(app_config.model_dump_json(indent=2))
#        # Access nested fields
#        print(f"Model Architecture: {app_config.experiment.model.architecture}")
#        print(f"Training batch size: {app_config.experiment.training.batch_size}")
#        print(f"Train dataset target: {app_config.experiment.data.datasets['train'].dataset_target}")
#        print(f"Optimizer LR: {app_config.experiment.optimizer.lr}")
#    except Exception as e:
#        print(f"Error parsing Transformer config: {e}")
#    print("-"*20)
#
#    # Example for SimpleRNN
#    test_conf_dict_rnn = test_conf_dict_transformer.copy() # Start from transformer example
#    test_conf_dict_rnn['experiment']['model'] = {
#        'architecture': 'simple_rnn',
#        '_target_': 'craft.models.simple_rnn.SimpleRNNLanguageModel',
#        'vocab_size': 50257,
#        'max_seq_length': 512,
#        'd_model': 256, # Input embedding size
#        'hidden_size': 512, # RNN hidden size
#        'num_layers': 2, # RNN layers
#        'dropout': 0.1
#    }
#    try:
#        app_config_rnn = AppConfig(**test_conf_dict_rnn)
#        print("SimpleRNN Config Parsed Successfully:")
#        print(app_config_rnn.model_dump_json(indent=2))
#        print(f"Model Architecture: {app_config_rnn.experiment.model.architecture}")
#        if isinstance(app_config_rnn.experiment.model, SimpleRNNConfig):
#             print(f"RNN Hidden Size: {app_config_rnn.experiment.model.hidden_size}")
#    except Exception as e:
#        print(f"Error parsing SimpleRNN config: {e}") 