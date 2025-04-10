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
    # Revert to allow/ignore extra fields to bypass validation issue with unexpected keys
    # model_config = ConfigDict(extra='forbid') # Forbid extra fields <-- REMOVE/COMMENT OUT
    architecture: Literal["transformer"] = Field("transformer", description="Architecture set to transformer.") # Discriminator value
    vocab_size: Optional[int] = Field(None, description="Size of the vocabulary (often inferred).")
    d_model: int = Field(768, description="Model dimension.")
    n_head: int = Field(12, description="Number of attention heads.")
    d_hid: Optional[int] = Field(None, description="Hidden dimension in feed-forward layers.")
    n_layers: int = Field(12, description="Number of transformer layers.")
    dropout: float = Field(0.1, ge=0.0, le=1.0, description="Dropout probability.")
    bias: bool = Field(True, description="Whether to use bias in linear layers.")
    layer_norm_eps: float = Field(1e-5, description="Epsilon for layer normalization.")
    activation: str = Field('gelu', description="Activation function.")
    norm_first: bool = Field(True, description="Apply layer norm before attention/FFN.")

    @field_validator('d_hid', mode='before')
    @classmethod
    def set_d_hid_default(cls, v: Any, info: ValidationInfo) -> Optional[int]:
        """Set default d_hid = d_model * 4 if not provided."""
        d_model_val: Optional[int] = None
        if v is None and info.data and ('d_model' in info.data):
            d_model_val = info.data.get('d_model')
        elif v is None and 'd_model' in cls.model_fields and cls.model_fields['d_model'].default is not None:
             d_model_val = cls.model_fields['d_model'].default

        if d_model_val is not None and isinstance(d_model_val, int):
             return d_model_val * 4

        if isinstance(v, int) or v is None:
             return v
        else:
             logger.warning(f"Could not determine integer value for d_hid based on input {v} or d_model. Defaulting to None.")
             return None

    @model_validator(mode='after') # Run after individual fields are validated
    def check_d_model_divisible_by_n_head(self) -> 'LanguageModelConfig':
        """Check if d_model is divisible by n_head."""
        if self.n_head > 0 and self.d_model % self.n_head != 0:
            logger.warning(
                f"Model dimension d_model ({self.d_model}) is not divisible by the number of heads n_head ({self.n_head}). "
                f"This is often required for multi-head attention implementations."
            )
            # Optionally raise ValueError if it's a strict requirement
            # raise ValueError(f"d_model ({self.d_model}) must be divisible by n_head ({self.n_head})")
        return self

class SimpleRNNConfig(GenerativeModelConfig):
    """Config for Simple RNN Models"""
    # Revert to allow/ignore extra fields
    # model_config = ConfigDict(extra='forbid') # Forbid extra fields <-- REMOVE/COMMENT OUT
    architecture: Literal["simple_rnn"] = Field("simple_rnn", description="Architecture set to simple_rnn.") # Discriminator value
    # Add back fields previously inherited from LanguageModelConfig that are needed
    vocab_size: Optional[int] = Field(None, description="Size of the vocabulary (often inferred).")
    d_model: int = Field(512, description="Model dimension (used for embedding in RNN).")
    dropout: float = Field(0.1, description="Dropout probability.")
    # RNN specific fields
    hidden_size: int = Field(512, description="Size of the RNN hidden state.")
    num_layers: int = Field(2, description="Number of RNN layers.")
    top_k: Optional[int] = Field(None, ge=1)

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
    # Revert to allow/ignore extra fields
    # model_config = ConfigDict(extra='forbid') # Forbid extra fields <-- REMOVE/COMMENT OUT
    architecture: Literal["vision_transformer"] = Field("vision_transformer", description="Example vision architecture.") # Discriminator value
    # model_type: str = Field("vision", description="Model type set to vision.")
    image_size: Tuple[int, int] = Field((224, 224), description="Input image dimensions (height, width).")
    patch_size: int = Field(16, description="Size of image patches.")
    num_channels: int = Field(3, description="Number of input image channels.")
    # ... other vision-specific fields (e.g., d_model, n_layers for ViT)

class MultiModalModelConfig(BaseModelConfig):
    """Placeholder Config for Multi-Modal Models"""
    # Revert to allow/ignore extra fields
    # model_config = ConfigDict(extra='forbid') # Forbid extra fields <-- REMOVE/COMMENT OUT
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
    type: Optional[str] = None # Add type field to match YAML

    # Define the structure for individual dataset splits (train/val/test)
    class DatasetSplitConfig(BaseModel):
        # This inner model also needs the shared config
        model_config = _model_config_shared.copy()
        # Remove validation_alias, rely on the validator to create these fields
        dataset_target: Optional[str] = Field(None, description="Target class/function for the dataset")
        dataset_params: Dict[str, Any] = Field(default_factory=dict, description="Parameters for the dataset target (excluding _target_)")

        dataloader_target: Optional[str] = Field(None, description="Optional target class for the dataloader")
        dataloader_params: Dict[str, Any] = Field(default_factory=dict, description="Parameters for the dataloader target (excluding _target_)")

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


# --- Checkpointing Configuration --- #

class CheckpointingConfig(BaseModel):
    """Configuration specific to checkpointing behavior."""
    model_config = _model_config_shared.copy()
    checkpoint_dir: Optional[str] = Field(None, description="Base directory to save checkpoints. If None, derived from output_dir.")
    keep_last: Optional[int] = Field(3, ge=1, description="Number of recent checkpoints to keep.")
    keep_best: Optional[int] = Field(1, ge=0, description="Number of best checkpoints to keep based on metric.")
    save_best_only: bool = Field(False, description="If True, only saves checkpoints that improve the monitored metric.")
    save_weights_only: bool = Field(False, description="If True, save only the model weights.")
    val_metric: Optional[str] = Field("val_loss", description="Metric name used to determine the 'best' checkpoint.")
    mode: Literal['min', 'max'] = Field('min', description="Mode for comparing the validation metric ('min' or 'max').")
    time_save_interval_seconds: Optional[int] = Field(None, ge=0, description="Save checkpoint every N seconds (0 or None to disable). Prioritized over step/epoch intervals if non-zero.")
    checkpoint_prefix: str = Field("checkpoint", description="Prefix for checkpoint filenames.")


# --- Generation Configuration (Existing) ---
class GenerationConfig(BaseModel):
    model_config = _model_config_shared.copy()
    start_prompt: str = "\\\\n"
    max_new_tokens: int = Field(200, gt=0)
    temperature: float = Field(0.8, ge=0)
    top_k: Optional[int] = Field(None, ge=1)
    top_p: Optional[float] = Field(None, ge=0.0, le=1.0, description="Nucleus sampling probability.")
    repetition_penalty: Optional[float] = Field(1.0, ge=1.0, description="Penalty for repeating tokens (1.0 means no penalty).")
    use_beam_search: bool = Field(False, description="Whether to use beam search for generation.")
    num_beams: Optional[int] = Field(None, ge=1, description="Number of beams for beam search.")
    length_penalty: Optional[float] = Field(1.0, description="Length penalty for beam search.")
    early_stopping: Optional[bool] = Field(False, description="Whether to stop beam search early.")


# --- Training Configuration (Refined) ---
class TrainingConfig(BaseModel):
    model_config = _model_config_shared.copy()
    # batch_size removed - defined in DataConfig
    # seed removed - defined in AppConfig
    # torch_compile removed - defined in AppConfig
    # resume_from_checkpoint removed - handled by AppConfig/CLI
    # checkpoint_dir removed - handled by CheckpointingConfig

    num_epochs: Optional[int] = Field(None, ge=1, description="Number of training epochs (set this or max_steps)")
    max_steps: Optional[int] = Field(None, ge=1, description="Maximum training steps (set this or num_epochs)")
    use_amp: bool = Field(False, description="Whether to use Automatic Mixed Precision")
    gradient_accumulation_steps: int = Field(1, ge=1, description="Number of steps to accumulate gradients before optimizer step")
    max_grad_norm: Optional[float] = Field(None, gt=0, description="Maximum gradient norm for clipping")
    log_interval: int = Field(50, gt=0, description="Log training metrics every N steps")
    eval_interval: int = Field(1000, ge=0, description="Evaluate on validation set every N steps (0 to disable)")
    save_interval: Optional[int] = Field(None, ge=0, description="Save checkpoint every N steps (0 to disable)") # Default None
    # time_save_interval_seconds: int = Field(0, description="Save checkpoint every N seconds (0 to disable). Prioritized over step/epoch intervals if non-zero.") # Moved to CheckpointingConfig
    # time_eval_interval_seconds: Optional[int] = Field(None, ge=0, description="Evaluate every N seconds") # Revisit if needed
    warmup_steps: Optional[int] = Field(100, ge=0, description="Number of warmup steps for learning rate scheduler")
    activation_checkpointing: Optional[bool] = Field(False, description="Enable activation checkpointing")
    generation: GenerationConfig = Field(default_factory=GenerationConfig, description="Default generation parameters for sampling callbacks") # type: ignore [arg-type]
    log_throughput_interval_batches: int = Field(
        100, description="Log throughput every N batches."
    )

    @model_validator(mode='after')
    def check_epochs_or_steps(self) -> 'TrainingConfig':
        if self.num_epochs is None and self.max_steps is None:
            raise ValueError("Either 'num_epochs' or 'max_steps' must be specified in training config.")
        # Optional: Warn if both are set?
        # if self.num_epochs is not None and self.max_steps is not None:
        #     logger.warning("Both 'num_epochs' and 'max_steps' are set. Training will stop when the first limit is reached.")
        return self

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
    # Add the checkpointing config here
    checkpointing: Optional[CheckpointingConfig] = Field(None, description="Checkpointing configuration.")
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