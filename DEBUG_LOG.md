# ChatGoT Technical Debug Log

## Code Modifications

### 2024-03-24: Checkpoint System

#### Fixed checkpoint loading (scripts/train_with_samples.py)
```python
# Added gradient checkpointing reset before loading checkpoint
if hasattr(model, 'layers'):
    logger.info("Resetting all layer forward functions to avoid checkpoint loading issues")
    for layer in model.layers:
        if hasattr(layer, 'forward_original'):
            logger.info(f"  - Restoring original forward function for layer")
            layer_cls = layer.__class__
            original_forward = layer_cls.forward
            layer.forward = types.MethodType(original_forward, layer)
            delattr(layer, 'forward_original')

# More robust state dict loading with filtering
model_state_dict = checkpoint['model_state_dict']
model_dict = model.state_dict()
filtered_dict = {k: v for k, v in model_state_dict.items() if k in model_dict}
model_dict.update(filtered_dict)
model.load_state_dict(model_dict)
```

#### Updated checkpoint saving frequency
```python
# Changed from:
save_checkpoint_steps = 500  
checkpoint_interval_minutes = 5

# To:
save_checkpoint_steps = 100  # Save checkpoints every 100 steps
checkpoint_interval_minutes = 2  # Save checkpoint every 2 minutes
```

#### Fixed indentation error in training loop
```python
# Fixed indentation for model.train() line in the epoch loop
for epoch in range(start_epoch, epochs):
    model.train()  # This line had incorrect indentation
```

### 2024-03-24: Re-enabled Gradient Checkpointing

#### Added safe gradient checkpointing implementation
```python
def enable_gradient_checkpointing(model):
    """Safely enable gradient checkpointing with error handling."""
    if not hasattr(model, 'layers'):
        logger.warning("Model doesn't have 'layers' attribute, skipping gradient checkpointing")
        return False
    
    try:
        for i, layer in enumerate(model.layers):
            # Store the original forward function if not already stored
            if not hasattr(layer, 'forward_original'):
                layer.forward_original = layer.forward
                # Create safe wrapper with error handling
                def checkpoint_wrapper(fn, *args, **kwargs):
                    try:
                        return torch_checkpoint(fn, *args, **kwargs)
                    except Exception as e:
                        logger.error(f"Error in gradient checkpointing: {e}")
                        # Fall back to standard forward pass
                        return fn(*args, **kwargs)
                
                layer.forward = functools.partial(checkpoint_wrapper, layer.forward_original)
                logger.info(f"Gradient checkpointing enabled for layer {i}")
        return True
    except Exception as e:
        logger.error(f"Failed to enable gradient checkpointing: {e}")
        # Restore original forward functions if any error occurs
        disable_gradient_checkpointing(model)
        return False

def disable_gradient_checkpointing(model):
    """Safely disable gradient checkpointing and restore original forward functions."""
    if not hasattr(model, 'layers'):
        return
    
    try:
        for i, layer in enumerate(model.layers):
            if hasattr(layer, 'forward_original'):
                layer.forward = layer.forward_original
                delattr(layer, 'forward_original')
                logger.info(f"Gradient checkpointing disabled for layer {i}")
        return True
    except Exception as e:
        logger.error(f"Error disabling gradient checkpointing: {e}")
        return False
```

#### Updated gradient checkpointing initialization
```python
# Changed from:
if device.type == 'cuda' and torch_checkpoint is not None and gradient_checkpointing:
    logger.warning("Gradient checkpointing is DISABLED due to compatibility issues")
    # Skip the actual checkpoint enabling code

# To:
if device.type == 'cuda' and torch_checkpoint is not None and gradient_checkpointing:
    logger.info("Attempting to enable gradient checkpointing for memory efficiency")
    if enable_gradient_checkpointing(model):
        logger.info("Successfully enabled gradient checkpointing")
    else:
        logger.warning("Failed to enable gradient checkpointing, continuing without it")
```

### 2024-03-24: Checkpoint Management and Log File Fixes

#### Added automatic cleanup of old checkpoints
```python
def cleanup_old_checkpoints(checkpoint_dir, config_name, timestamp, max_to_keep=5):
    """Keep only the specified number of recent checkpoints to save disk space."""
    # Get list of checkpoints for this specific model and run
    checkpoint_pattern = f"{config_name}_{timestamp}_step_*.pt"
    checkpoints = []
    
    try:
        for file in os.listdir(checkpoint_dir):
            if file.startswith(f"{config_name}_{timestamp}_step_") and file.endswith(".pt"):
                step = int(file.split("_step_")[1].split(".pt")[0])
                checkpoints.append((step, os.path.join(checkpoint_dir, file)))
        
        # Sort by step number (descending)
        checkpoints.sort(reverse=True)
        
        # Keep only max_to_keep checkpoints
        if len(checkpoints) > max_to_keep:
            for _, checkpoint_path in checkpoints[max_to_keep:]:
                try:
                    os.remove(checkpoint_path)
                    logger.info(f"Removed old checkpoint: {os.path.basename(checkpoint_path)}")
                except Exception as e:
                    logger.warning(f"Failed to remove checkpoint {checkpoint_path}: {e}")
    except Exception as e:
        logger.warning(f"Error during checkpoint cleanup: {e}")
```

#### Added configuration parameter for checkpoint retention
```yaml
training:
  # existing parameters...
  max_checkpoints_to_keep: 3    # Keep only 3 most recent checkpoints
```

#### Created utility script for fixing log encoding issues
```python
# scripts/fix_log_encoding.py
def fix_log_file(input_path, output_path=None, remove_ansi=True):
    """Fix encoding issues in a log file."""
    # Detect encoding
    encoding, confidence = detect_encoding(input_path)
    
    # Read the file with detected encoding
    with open(input_path, 'r', encoding=encoding, errors='replace') as f:
        content = f.read()
    
    # Clean up ANSI escape sequences if requested
    if remove_ansi:
        content = clean_ansi_escape_sequences(content)
    
    # Write with UTF-8 encoding
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(content)
```

### 2024-03-24: Improved Logging System

#### Implemented dual logging to console and file
```python
def setup_logging(config_name=None):
    """Set up logging with both console (colored) and file (clean) output."""
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    # Create a formatter without color codes for file logging
    file_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Create a detailed file name with timestamp
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_filename = f"logs/train_{config_name}_{timestamp}.log" if config_name else f"logs/train_{timestamp}.log"
    
    # Set up file handler with rotation (10 MB max size, keep 5 backups)
    file_handler = logging.handlers.RotatingFileHandler(
        log_filename, 
        maxBytes=10*1024*1024,  # 10 MB
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(logging.INFO)
    
    # Setup console handler with colors for terminal
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Add both handlers to logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
```

#### Integrated log files with TensorBoard
```python
# Create a copy of the log file in the TensorBoard directory
if log_file and os.path.exists(log_file):
    try:
        symlink_path = os.path.join(log_dir, "training.log")
        # On Windows, use file copy instead of symlink
        if os.name == 'nt':
            import shutil
            shutil.copy2(log_file, symlink_path)
        else:
            os.symlink(os.path.abspath(log_file), symlink_path)
    except Exception as e:
        logger.warning(f"Could not create log file link: {e}")
```

### 2024-03-24: Safe Mixed Precision Training

#### Added SafeGradScaler for robust mixed precision
```python
class SafeGradScaler(GradScaler):
    """Enhanced GradScaler with NaN detection and fallback to full precision."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.nan_counter = 0
        self.max_nan_before_fallback = 3
        self.fallback_triggered = False
        self.warning_logged = False
        
    def scale(self, loss):
        # Check for NaN before scaling
        if torch.isnan(loss).any() or torch.isinf(loss).any():
            self.nan_counter += 1
            logger.warning(f"NaN/Inf detected in loss before scaling: {loss.item() if not torch.isinf(loss).all() else 'inf'} (occurrence {self.nan_counter}/{self.max_nan_before_fallback})")
            
            # If we've seen too many NaNs, disable mixed precision
            if self.nan_counter >= self.max_nan_before_fallback and not self.fallback_triggered:
                logger.error(f"Too many NaN/Inf values detected, disabling mixed precision")
                self._enabled = False
                self.fallback_triggered = True
                
                # Return unscaled loss to continue training in full precision
                return loss
        else:
            # Reset counter if loss is normal
            if self.nan_counter > 0:
                self.nan_counter = max(0, self.nan_counter - 1)  # Gradually decrease counter
                
        if not self._enabled and not self.warning_logged:
            logger.warning("Mixed precision is disabled. Using full precision for forward/backward passes.")
            self.warning_logged = True
            return loss
            
        return super().scale(loss)
    
    def step(self, optimizer, *args, **kwargs):
        # Additional safety around optimizer step
        if not self._enabled:
            # In full precision mode, just do a normal step
            return optimizer.step(*args, **kwargs)
            
        try:
            return super().step(optimizer, *args, **kwargs)
        except RuntimeError as e:
            if "inf or nan" in str(e).lower():
                logger.error(f"NaN/Inf detected during optimizer step, disabling mixed precision: {e}")
                self._enabled = False
                self.fallback_triggered = True
                
                # Attempt normal optimizer step as fallback
                try:
                    optimizer.zero_grad()
                    return optimizer.step(*args, **kwargs)
                except Exception as e2:
                    logger.error(f"Fallback optimizer step also failed: {e2}")
                    raise
            else:
                raise
```

#### Enhanced training loop for NaN/Inf detection
```python
# Enhanced loss checks to detect NaN/Inf
if torch.isnan(unscaled_loss).any() or torch.isinf(unscaled_loss).any():
    logger.warning(f"NaN/Inf loss detected after forward pass: {unscaled_loss.item() if not torch.isinf(unscaled_loss).all() else 'inf'}")
    
    # Try to diagnose the issue
    logits_info = f"Logits: min={outputs.min().item():.4f}, max={outputs.max().item():.4f}, mean={outputs.mean().item():.4f}, std={outputs.std().item():.4f}"
    logger.warning(f"Abnormal outputs detected: {logits_info}")
    
    # If we're accumulating gradients, skip this batch but continue training
    if is_accumulating:
        logger.info("Skipping gradient accumulation for this batch")
        continue
```

#### Created dedicated mixed precision test configuration
```yaml
# configs/models/debug/chatgot_small_14M_mixed_precision.yaml
training:
  mixed_precision: true   # ENABLED mixed precision training with enhanced safety
  gradient_checkpointing: false  # Disabled for isolated testing of mixed precision
```

## Configuration Changes

### Created 14M parameter clean version
Created `configs/models/debug/chatgot_small_14M_clean.yaml` with:
- d_model: 512 (reduced from 768)
- n_layers: 6 (reduced from 12)
- n_head: 8 (reduced from 12)
- max_seq_length: 256 (reduced from 1024)
- Removed gradient checkpointing

### Created gradient checkpointing test configuration
Created `configs/models/debug/chatgot_small_14M_gradient_checkpoint.yaml` with:
- Same parameters as clean version
- Enabled gradient checkpointing
- Disabled mixed precision for isolated testing

### Updated configurations with checkpoint management
Added to both configuration files:
```yaml
training:
  # existing parameters...
  max_checkpoints_to_keep: 3  # Keep only 3 most recent checkpoints
```

### Created mixed precision test configuration
Created `configs/models/debug/chatgot_small_14M_mixed_precision.yaml` with:
- Same parameters as clean version
- Enabled mixed precision with enhanced safety
- Disabled gradient checkpointing for isolated testing

## Current Behavior
- Training starts and runs without errors
- Checkpoints are saved every 100 steps and every 2 minutes
- Only the most recent N checkpoints are kept (configurable)
- Checkpoints can be loaded with `--resume_from` argument
- Gradient checkpointing is enabled with safe error handling
- Mixed precision training is enabled with NaN detection and fallback
- Logs are saved to both console (colored) and files (clean text) simultaneously
- Log files are automatically rotated when they reach 10 MB
- Log files are integrated with TensorBoard for easy access

## Next Implementation Tasks
1. ✅ Restore gradient checkpointing with improved error handling
2. ✅ Re-enable mixed precision with safety checks
3. ⬜ Re-enable dynamic batch size adjustment with validation 