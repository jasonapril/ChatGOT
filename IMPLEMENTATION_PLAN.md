# ChatGoT Implementation Plan

## Priority 1: Basic Stability & Correctness
- ✅ Fix indentation error in training loop
- ✅ Ensure checkpoint saving/loading works correctly
- ⬜ Validate training metrics and convergence

## Priority 2: Memory Management
- ✅ Re-enable gradient checkpointing
  - [x] Implement safe forward function wrapping
  - [x] Add proper cleanup on exception
  - [x] Test with checkpoint loading
  - [x] Add configuration parameter validation

## Priority 3: Training Performance
- ✅ Re-enable mixed precision training
  - [x] Add safety checks for NaN/inf values
  - [x] Implement fallback to full precision
  - [x] Add detailed logging for debugging
  - [x] Test with different batch sizes

- ⬜ Restore dynamic batch size adjustment
  - [x] Implement batch reduction on OOM
  - [ ] Add batch recovery attempt after stable training
  - [ ] Add memory tracking throughout training
  - [ ] Test with different sequence lengths

## Priority 4: Training Robustness
- ⬜ Improve loss trend tracking
  - [ ] Add moving average window
  - [ ] Implement anomaly detection
  - [ ] Add automatic LR adjustment
  
- ⬜ Enhance ETA calculation
  - [ ] Implement median filtering
  - [ ] Add adaptive smoothing
  - [ ] Fix progress reporting

## Priority 5: Generation Quality
- ⬜ Improve sampling during training
  - [ ] Use dynamic temperature based on loss
  - [ ] Add proper handling of sentence boundaries
  - [ ] Implement better repetition penalty
  - [ ] Add diversity metrics

## Detailed Implementation Steps

### Gradient Checkpointing
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
                def checkpoint_wrapper(layer_fn, *args, **kwargs):
                    return torch_checkpoint(layer_fn, *args, **kwargs)
                
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
    
    for i, layer in enumerate(model.layers):
        if hasattr(layer, 'forward_original'):
            layer.forward = layer.forward_original
            delattr(layer, 'forward_original')
            logger.info(f"Gradient checkpointing disabled for layer {i}")
```

### Mixed Precision Implementation (COMPLETED)
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
        # Rest of implementation...
```

### Batch Size Recovery Implementation (NEXT PRIORITY)
```python
def try_increase_batch_size(batch_size, original_batch_size, loss_history, stable_epochs=2):
    """Attempt to increase batch size if training has been stable for a while."""
    if batch_size < original_batch_size and len(loss_history) >= stable_epochs:
        # Check loss stability over recent epochs
        recent_losses = loss_history[-stable_epochs:]
        variance = torch.var(torch.tensor(recent_losses))
        if variance < 0.01:  # Loss has been stable
            new_batch_size = min(batch_size + 1, original_batch_size)
            logger.info(f"Training stable for {stable_epochs} epochs, increasing batch size from {batch_size} to {new_batch_size}")
            return new_batch_size
    return batch_size
```

### Testing Methodology
For each feature:
1. Implement in isolation on the clean 14M model
2. Run short training (100 steps) to validate
3. Test checkpoint saving/loading
4. Document any issues in DEBUG_LOG.md
5. Update DEBUGGING_PROGRESS.md with completion status 