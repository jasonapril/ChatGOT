#!/usr/bin/env python
"""
Evaluation Module
================

This module contains the evaluation logic for validating model performance.
"""

import torch
import torch.nn.functional as F
import logging
import time
from typing import Dict, Any, Optional, List
from tqdm import tqdm

class Evaluator:
    """Handles model evaluation on validation data."""
    
    def __init__(
        self,
        model: torch.nn.Module,
        val_dataloader: torch.utils.data.DataLoader,
        device: torch.device,
        config: Dict[str, Any],
        use_amp: bool = False,
        callbacks: Optional[List[Any]] = None
    ):
        self.model = model
        self.val_dataloader = val_dataloader
        self.device = device
        self.config = config
        self.use_amp = use_amp
        self.callbacks = callbacks if callbacks is not None else []
        self.logger = logging.getLogger(self.__class__.__name__)

    def evaluate(self) -> Dict[str, float]:
        """Evaluates the model on the validation set."""
        if self.val_dataloader is None:
            self.logger.info("No validation dataloader provided, skipping evaluation.")
            return {}
            
        self.logger.info("Starting evaluation...")
        self.model.eval()
        total_loss = 0.0
        total_batches = len(self.val_dataloader)
        eval_start_time = time.time()
        total_tokens = 0

        progress_bar = tqdm(self.val_dataloader, total=total_batches, desc="Evaluating")

        with torch.no_grad():
            for batch in progress_bar:
                # Unpack batch
                inputs = batch['input_ids'].to(self.device, non_blocking=True)
                targets = batch['labels'].to(self.device, non_blocking=True)

                # Forward pass with AMP
                with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
                    outputs = self.model(inputs)
                    loss = F.cross_entropy(outputs.view(-1, outputs.size(-1)), targets.view(-1))

                if not (torch.isnan(loss).any() or torch.isinf(loss).any()):
                    total_loss += loss.item()
                    total_tokens += inputs.numel()
                else:
                    self.logger.warning("NaN/Inf detected during evaluation. Skipping batch.")

        avg_loss = total_loss / total_batches if total_batches > 0 else 0.0
        eval_time = time.time() - eval_start_time
        tokens_per_sec = total_tokens / eval_time if eval_time > 0 else 0

        self.logger.info(f"Evaluation finished in {eval_time:.2f}s. Avg Loss: {avg_loss:.4f}, Tokens/sec: {tokens_per_sec:.2f}")

        return {'loss': avg_loss, 'tokens_per_sec': tokens_per_sec} 