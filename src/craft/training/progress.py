"""
Progress Tracking Module
=======================

This module handles progress tracking, logging, and metrics reporting during training.
"""

import logging
import time
from typing import Dict, Any, Optional, List, Type
from tqdm import tqdm
import numpy as np
from types import TracebackType

class ProgressTracker:
    """Tracks and displays training progress using tqdm and logging."""
    
    def __init__(
        self,
        total_steps: Optional[int] = None,
        desc: str = "Training",
        log_interval: int = 100,
        disable_progress_bar: bool = False,
    ) -> None:
        """Initialize the ProgressTracker.

        Args:
            total_steps (Optional[int]): The total number of steps for the progress bar.
                                         If None, the bar will run indefinitely until manually closed.
            desc (str): Description prefix for the progress bar and logs.
            log_interval (int): Frequency (in steps) for logging metrics.
            disable_progress_bar (bool): If True, disables the tqdm progress bar.
        """
        self.total_steps = total_steps
        self.desc = desc
        self.log_interval = log_interval
        self.disable_progress_bar = disable_progress_bar
        self.logger = logging.getLogger(f"{__name__}.ProgressTracker")
        
        self.current_step: int = 0
        self.current_epoch: int = 0
        self.total_epochs: int = 0
        self.epoch_steps: int = 0
        self.step_times: List[float] = []
        self.losses: List[float] = []
        self.learning_rates: List[float] = []
        self.steps_per_second: List[float] = []
        self.start_time: Optional[float] = None
        self.last_log_time: Optional[float] = None
        
        # Initialize progress bar with type hint
        self._pbar: Optional[tqdm] = None
        self._setup_progress_bar()

    def _setup_progress_bar(self) -> None:
        """Initialize the progress bar with tqdm."""
        if self.disable_progress_bar:
            self._pbar = None
            self.logger.info("Progress bar disabled.")
            return

        try:
            self._pbar = tqdm(
                total=self.total_steps,
                desc=self.desc,
                unit="step",
                dynamic_ncols=True,
                disable=self.disable_progress_bar,
            )
        except Exception as e:
            self.logger.warning(f"Failed to initialize tqdm progress bar: {e}. Progress bar disabled.")
            self._pbar = None

    def start(self) -> None:
        """Start tracking time and reset progress."""
        self.start_time = time.monotonic()
        self.last_log_time = self.start_time
        self.current_step = 0
        self.current_epoch = 0
        self.epoch_steps = 0
        self.step_times = []
        self.losses = []
        self.learning_rates = []
        self.steps_per_second = []
        self.logger.info(f"Starting {self.desc}...")
        if self.pbar:
            self.pbar.reset(total=self.total_steps)

    def update(
        self,
        step: int,
        loss: float,
        learning_rate: Optional[float] = None,
        steps_per_second: Optional[float] = None,
        additional_metrics: Optional[Dict[str, Any]] = None
    ) -> None:
        """Update progress with new metrics for a completed step."""
        if self.start_time is None or self.last_log_time is None:
            self.logger.warning("ProgressTracker.update called before start(). Auto-starting.")
            self.start()

        current_time = time.monotonic()
        step_time = current_time - (self.last_log_time or current_time)
        self.step_times.append(step_time)
        self.losses.append(loss)
        if learning_rate is not None:
            self.learning_rates.append(learning_rate)
        if steps_per_second is not None:
            self.steps_per_second.append(steps_per_second)

        self.last_log_time = current_time
        self.current_step = step
        self.epoch_steps += 1

        # Update progress bar
        if self.pbar:
            pbar_current_n = self.pbar.n if hasattr(self.pbar, 'n') else 0
            update_n = max(0, step - pbar_current_n)

            metrics = {'loss': f'{loss:.4f}'}
            if learning_rate is not None:
                metrics['lr'] = f'{learning_rate:.2e}'
            if steps_per_second is not None:
                metrics['S/s'] = f'{steps_per_second:.1f}'
            if additional_metrics:
                formatted_additional = {
                    k: (f"{v:.3f}" if isinstance(v, (float, np.floating)) else v)
                    for k, v in additional_metrics.items()
                }
                metrics.update(formatted_additional)

            self.pbar.set_postfix(metrics, refresh=False)
            if update_n > 0:
                self.pbar.update(update_n)

        # Log metrics at intervals
        if step > 0 and self.log_interval > 0 and step % self.log_interval == 0:
            self._log_metrics(step, loss, learning_rate, steps_per_second, additional_metrics)

    def _log_metrics(
        self,
        step: int,
        current_loss: float,
        current_learning_rate: Optional[float],
        current_steps_per_second: Optional[float],
        current_additional_metrics: Optional[Dict[str, Any]]
    ) -> None:
        """Log current progress metrics, averaging over the log interval."""
        if not self.step_times or self.start_time is None: return

        interval = min(len(self.losses), self.log_interval)
        avg_loss = np.mean(self.losses[-interval:]) if interval > 0 else current_loss
        avg_step_time = np.mean(self.step_times[-interval:]) if interval > 0 else 0
        avg_sps = np.mean(self.steps_per_second[-interval:]) if self.steps_per_second and interval > 0 else current_steps_per_second

        elapsed_time = time.monotonic() - self.start_time

        log_msg = f"{self.desc} Step: {step}"
        if self.total_steps: log_msg += f"/{self.total_steps}"
        if self.total_epochs > 0: log_msg += f" | Epoch: {self.current_epoch + 1}/{self.total_epochs}"
        log_msg += f" | Loss: {avg_loss:.4f}"
        if current_learning_rate is not None:
            log_msg += f" | LR: {current_learning_rate:.2e}"
        if avg_sps is not None and avg_sps > 0:
            log_msg += f" | S/s: {avg_sps:.1f}"
        elif current_steps_per_second is not None:
            log_msg += f" | S/s: {current_steps_per_second:.1f}"

        log_msg += f" | Step Time: {avg_step_time:.3f}s"
        log_msg += f" | Elapsed: {elapsed_time:.1f}s"

        if current_additional_metrics:
            metrics_str = " | ".join(f"{k}: {(f'{v:.3f}' if isinstance(v, (float, np.floating)) else v)}"
                                     for k, v in current_additional_metrics.items())
            log_msg += f" | {metrics_str}"

        self.logger.info(log_msg)

    def get_average_metrics(self) -> Dict[str, float]:
        """Calculate average metrics over the entire run so far."""
        return {
            'avg_loss': float(np.mean(self.losses)) if self.losses else 0.0,
            'avg_learning_rate': float(np.mean(self.learning_rates)) if self.learning_rates else 0.0,
            'avg_steps_per_second': float(np.mean(self.steps_per_second)) if self.steps_per_second else 0.0,
            'avg_step_time': float(np.mean(self.step_times)) if self.step_times else 0.0
        }

    def close(self) -> None:
        """Clean up progress tracking and log final summary."""
        if self.pbar:
            self.pbar.close()
        self._pbar = None

        if self.start_time:
            total_time = time.monotonic() - self.start_time
            self.logger.info(f"{self.desc} finished in {total_time:.2f} seconds.")
            avg_metrics = self.get_average_metrics()
            self.logger.info(f"Final Average Loss: {avg_metrics['avg_loss']:.4f}")
            if avg_metrics['avg_learning_rate'] > 0:
                 self.logger.info(f"Final Average LR: {avg_metrics['avg_learning_rate']:.2e}")
            if avg_metrics['avg_steps_per_second'] > 0:
                self.logger.info(f"Final Average S/s: {avg_metrics['avg_steps_per_second']:.1f}")
            self.logger.info(f"Final Average Step Time: {avg_metrics['avg_step_time']:.3f}s")

    @property
    def pbar(self) -> Optional[tqdm]:
        """Return the tqdm progress bar instance if it's initialized and not disabled."""
        if hasattr(self, "_pbar") and self._pbar is not None:
            return self._pbar
        return None

    def update_step(self, new_step: int) -> None:
        """Directly update the progress bar's displayed step count.

        Note: This is mainly for synchronization if the external loop counter
              differs from the number of `update()` calls. Prefer using `update()`
              which handles metrics and bar updates together.
        """
        if self.pbar is None:
            return

        self.current_step = new_step

        update_n = new_step - self.pbar.n
        if update_n != 0:
            self.pbar.update(update_n)

    def set_epoch(self, current_epoch: int, total_epochs: int) -> None:
        """Set the current and total epochs for display."""
        self.current_epoch = current_epoch
        self.total_epochs = total_epochs
        self.epoch_steps = 0

        if self.pbar:
            postfix_metrics = self.pbar.postfix if isinstance(self.pbar.postfix, dict) else {}
            postfix_metrics["epoch"] = f"{self.current_epoch + 1}/{self.total_epochs}"
            self.pbar.set_postfix(postfix_metrics, refresh=True)

    # Context manager methods
    def __enter__(self) -> "ProgressTracker":
        """Enter the runtime context related to this object."""
        self.start()
        return self

    def __exit__(self, exc_type: Optional[Type[BaseException]], exc_val: Optional[BaseException], exc_tb: Optional[TracebackType]) -> None:
        """Exit the runtime context related to this object."""
        self.close() 