# PyTorch Debugging Notes

## Performance Issues

*   **AMP on CPU:** Using Automatic Mixed Precision (`torch.amp`) with `use_amp=True` can lead to extreme slowdowns when running on CPU. The `torch.amp.autocast("cpu", ...)` context manager adds significant overhead. If running or testing on CPU, ensure `use_amp` is set to `False` or automatically disabled based on the device type.
    *   Relevant Log Message: `[__main__][WARNING] - AMP is enabled but running on CPU. Disabling AMP for CPU execution.`

## Silent Crashes / Halts

*   **Check GPU Resources:** Silent halts after setup might indicate CUDA errors (OOM, illegal memory access). Monitor `nvidia-smi` during the run.
*   **DataLoader Workers:** If `num_workers > 0` in DataLoader, worker processes can hang, especially on Windows. Set `num_workers=0` for debugging.
*   **Basic `print()` Debugging:** Use simple `print("DEBUG: Step X", flush=True)` statements at key points (start/end of loops, before/after model forward/backward, before/after optimizer step) if standard logging doesn't appear.

## Configuration/Instantiation

*   **Hydra & Pydantic:** Validate configurations *after* Hydra composition using Pydantic models, but be mindful that Hydra's instantiation utils (`hydra.utils.instantiate`) often work best with raw `DictConfig` objects rather than the validated Pydantic models.
*   **Callback Instantiation:** Errors like `ModuleNotFoundError` or `AttributeError` during callback loops often mean the `_target_` path in the config (either default or experiment override) is incorrect or points to a non-existent class.
*   **Config Path Resolution:** When accessing paths from config (e.g., checkpoint paths, data paths), be aware of the current working directory (Hydra changes it to the output directory). Use `hydra.utils.get_original_cwd()` to resolve paths relative to the launch directory if needed. 