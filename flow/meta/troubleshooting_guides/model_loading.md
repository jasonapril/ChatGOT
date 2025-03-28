# Troubleshooting Guide: Model Loading

This document captures lessons learned from debugging model loading issues, particularly when loading pre-trained model checkpoints into slightly different architectures.

## Common Model Loading Issues

1. **Dataset Mismatches**
   - Vocabulary size differences between training and inference
   - Missing character maps or tokenizers
   - Incompatible dataset processing methods

2. **Architecture Mismatches**
   - Parameter dimension differences (e.g., feedforward layer size)
   - Layer naming differences between checkpoint and code
   - Missing or extra layers
   - Different configuration parameters

3. **State Dictionary Issues**
   - Key naming differences
   - Shape mismatches in tensors
   - Missing or unexpected keys

## Diagnostic Checklist

When troubleshooting model loading issues, follow this systematic approach:

1. **Isolate the Problem**
   - First, separate dataset loading from model loading
   - Create minimal test cases for each component
   - Log each step clearly

2. **Check Warning Messages**
   - Pay close attention to "Missing keys" and "Unexpected keys" warnings
   - Look for shape mismatch errors
   - Examine tensor dimension information in error messages

3. **Inspect Checkpoint Structure**
   ```python
   checkpoint = torch.load('path/to/checkpoint.pt', map_location='cpu')
   print(checkpoint.keys())  # Look at top-level structure
   
   # If model_state_dict is a key
   state_dict = checkpoint.get('model_state_dict', checkpoint)
   
   # Print keys and shapes
   for key, tensor in state_dict.items():
       print(f"{key}: {tensor.shape}")
   ```

4. **Compare Model Parameters**
   ```python
   # Create model with expected architecture
   model = create_model()
   
   # Print model's expected keys
   for name, param in model.named_parameters():
       print(f"{name}: {param.shape}")
   ```

5. **Test Incremental Loading**
   - Try loading only a subset of parameters first
   - Check if certain layers load correctly while others fail

## Solutions to Common Issues

### Key Mapping for Different Naming Conventions

When checkpoint keys don't match model keys:

```python
# Create a new state dict with adapted keys
new_state_dict = {}

# Map key names from checkpoint format to model format
key_mapping = {
    'old_name.weight': 'new_name.weight',
    'old_name.bias': 'new_name.bias',
    # Add more mappings as needed
}

for key, value in model_state_dict.items():
    # Check if this is a key we need to rename
    new_key = key
    for old_pattern, new_pattern in key_mapping.items():
        if old_pattern in key:
            new_key = key.replace(old_pattern, new_pattern)
            break
            
    # Skip certain keys if needed
    if 'key_to_skip' in key:
        continue
        
    new_state_dict[new_key] = value

# Try to load with the adapted keys
missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
```

### Handling Dimension Mismatches

When model dimensions don't match checkpoint dimensions:

1. **Extract Configuration from Checkpoint**
   ```python
   # Get configuration from checkpoint
   config = checkpoint.get("config", {})
   d_model = config.get("d_model", 512)
   n_head = config.get("n_head", 8)
   d_hid = config.get("d_hid", 2048)  # Use checkpoint values
   ```

2. **Create Model with Matching Dimensions**
   ```python
   # Create model with extracted dimensions
   model = create_model(
       d_model=d_model,
       n_head=n_head,
       d_hid=d_hid
   )
   ```

### Dataset Loading Issues

For character-level datasets:

```python
# Manual dataset creation (more reliable than using load_data)
with open(data_path, 'r', encoding='utf-8') as f:
    text = f.read()

# Create dataset directly
dataset = CharDataset(text, block_size)

# Access attributes directly
vocab_size = len(dataset.char_to_idx)
```

## Utilities for Debugging

### Model Structure Comparison

```python
def compare_model_with_checkpoint(model, checkpoint_state_dict):
    """Compare model structure with checkpoint."""
    model_keys = set(k for k, _ in model.named_parameters())
    checkpoint_keys = set(checkpoint_state_dict.keys())
    
    print(f"Model has {len(model_keys)} parameter tensors")
    print(f"Checkpoint has {len(checkpoint_keys)} parameter tensors")
    
    missing = model_keys - checkpoint_keys
    unexpected = checkpoint_keys - model_keys
    
    if missing:
        print(f"Missing keys: {missing}")
    if unexpected:
        print(f"Unexpected keys: {unexpected}")
    
    # Check for shape mismatches
    common_keys = model_keys.intersection(checkpoint_keys)
    for key in common_keys:
        model_shape = next(p.shape for n, p in model.named_parameters() if n == key)
        checkpoint_shape = checkpoint_state_dict[key].shape
        
        if model_shape != checkpoint_shape:
            print(f"Shape mismatch for {key}: model {model_shape} vs checkpoint {checkpoint_shape}")
```

## Lessons Learned

1. **Always Use Explicit Dimension Parameters**
   - Don't rely on defaults for model dimensions
   - Extract and log all configuration from checkpoints

2. **Implement Robust Key Mapping**
   - Develop reusable utilities for key mapping
   - Document naming conventions for models

3. **Separate Data and Model Loading**
   - Test dataset loading independently
   - Verify dataset attributes before model loading

4. **Thorough Logging**
   - Log model structure, parameter counts, and dimensions
   - Log warnings and errors with context

By following these guidelines, you can significantly reduce debugging time for model loading issues. 