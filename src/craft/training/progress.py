"""
Progress Tracking Module
=======================

This module handles progress tracking, logging, and metrics reporting during training.
"""

import logging
import time
from typing import Dict, Any, Optional
from tqdm import tqdm
import numpy as np

class ProgressTracker:
    """Tracks and reports training progress and metrics."""
    
    def __init__(
        self,
        total_steps: int,
        log_interval: int = 10,
        desc: str = "Training",
        position: int = 0,
        leave: bool = True,
        initial_step: int = 0
    ):
        self.total_steps = total_steps
        self.log_interval = log_interval
        self.desc = desc
        self.position = position
        self.leave = leave
        self.initial_step = initial_step
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize metrics tracking
        self.step_times = []
        self.losses = []
        self.learning_rates = []
        self.tokens_per_second = []
        self.start_time = None
        self.last_log_time = None
        
        # Initialize progress bar
        self.progress_bar = None
        self._setup_progress_bar()

    def _setup_progress_bar(self):
        """Initialize the progress bar with tqdm."""
        try:
            self.progress_bar = tqdm(
                total=self.total_steps,
                desc=self.desc,
                position=self.position,
                leave=self.leave,
                dynamic_ncols=True,
                mininterval=1.0,
                initial=self.initial_step
            )
        except Exception as e:
            self.logger.warning(f"Failed to initialize tqdm progress bar: {e}. Using simple logging.")
            self.progress_bar = None

    def start(self):
        """Start tracking progress."""
        self.start_time = time.time()
        self.last_log_time = self.start_time

    def update(
        self,
        step: int,
        loss: float,
        learning_rate: Optional[float] = None,
        tokens_per_second: Optional[float] = None,
        additional_metrics: Optional[Dict[str, Any]] = None
    ):
        """Update progress with current metrics."""
        current_time = time.time()
        
        # Update metrics
        self.step_times.append(current_time - self.last_log_time)
        self.losses.append(loss)
        if learning_rate is not None:
            self.learning_rates.append(learning_rate)
        if tokens_per_second is not None:
            self.tokens_per_second.append(tokens_per_second)
            
        self.last_log_time = current_time
        
        # Update progress bar
        if self.progress_bar is not None:
            metrics = {'loss': f'{loss:.4f}'}
            if learning_rate is not None:
                metrics['lr'] = f'{learning_rate:.2e}'
            if tokens_per_second is not None:
                metrics['T/s'] = f'{tokens_per_second:.0f}'
            if additional_metrics:
                metrics.update(additional_metrics)
            self.progress_bar.set_postfix(metrics)
            self.progress_bar.update(1)
        
        # Log metrics at intervals
        if step % self.log_interval == 0:
            self._log_metrics(step, loss, learning_rate, tokens_per_second, additional_metrics)

    def _log_metrics(
        self,
        step: int,
        loss: float,
        learning_rate: Optional[float],
        tokens_per_second: Optional[float],
        additional_metrics: Optional[Dict[str, Any]]
    ):
        """Log current metrics."""
        elapsed_time = time.time() - self.start_time
        avg_step_time = np.mean(self.step_times[-self.log_interval:]) if self.step_times else 0
        
        log_msg = f"Step: {step}/{self.total_steps} | "
        log_msg += f"Loss: {loss:.4f} | "
        if learning_rate is not None:
            log_msg += f"LR: {learning_rate:.2e} | "
        if tokens_per_second is not None:
            log_msg += f"T/s: {tokens_per_second:.0f} | "
        log_msg += f"Time: {elapsed_time:.1f}s | "
        log_msg += f"Step Time: {avg_step_time:.3f}s"
        
        if additional_metrics:
            for key, value in additional_metrics.items():
                log_msg += f" | {key}: {value}"
                
        self.logger.info(log_msg)

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of tracked metrics."""
        return {
            'total_steps': self.total_steps,
            'total_time': time.time() - self.start_time if self.start_time else 0,
            'avg_loss': np.mean(self.losses) if self.losses else 0,
            'avg_learning_rate': np.mean(self.learning_rates) if self.learning_rates else 0,
            'avg_tokens_per_second': np.mean(self.tokens_per_second) if self.tokens_per_second else 0,
            'avg_step_time': np.mean(self.step_times) if self.step_times else 0
        }

    def close(self):
        """Clean up progress tracking."""
        if self.progress_bar is not None:
            self.progress_bar.close()
            self.progress_bar = None 