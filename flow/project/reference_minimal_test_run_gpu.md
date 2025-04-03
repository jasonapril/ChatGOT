# Baseline: Working Minimal Test Configuration (GPU)

*   **Date:** 2025-04-03
*   **Goal:** Verify basic training loop runs for 2 steps on GPU without errors after resolving environment issues (persistent OOM errors).
*   **Command:** `python -m src.craft.cli.run train language training=minimal_test data.block_size=256`
*   **Outcome:** Success, 2 steps completed (~16s/it).

## Environment Configuration

*   **Python:** 3.13.2
*   **PyTorch:** `torch-2.6.0+cu118`
*   **TorchVision:** `torchvision-0.21.0+cu118`
*   **Torchaudio:** `torchaudio-2.6.0+cu118`
*   **Installation Method:** `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`
*   **CUDA Driver:** 546.33 (Supports CUDA 12.3)
*   **GPU:** NVIDIA GeForce GTX 1650 Ti (4GB VRAM)

## Key Training Configuration (Resolved)

*   **Hydra Config Group:** `training=minimal_test`
*   **File (`conf/training/minimal_test.yaml`):**
    *   `epochs: 1`
    *   `max_steps: 2`
    *   *(Other settings inherited from defaults)*
*   **Command Line Overrides:**
    *   `data.block_size=256`
*   **Effective Settings:**
    *   Model: Default (`transformer_model` - `d_model=768`, `n_layers=10`, etc.)
    *   Block Size (Sequence Length): 256
    *   Batch Size: 32 (from `conf/data/got_char_level.yaml`)
    *   AMP: Enabled (`use_amp: True` from `conf/training/default.yaml`)

## Notes

*   This configuration successfully ran after resolving persistent CUDA OOM errors that were likely due to a corrupted PyTorch/CUDA state.
*   The key fixes were reinstalling PyTorch `cu118` and using a command-line override for `data.block_size`.
*   Attempts to use `conf/training/minimal_test.yaml` to override `data.block_size` failed due to Hydra's override precedence.
*   This baseline uses the default **large** model but with a **small** `block_size` (256) to fit on the 4GB GPU and run relatively quickly. 