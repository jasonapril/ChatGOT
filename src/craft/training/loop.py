import time

class TrainingLoop:
    def _run_epoch(self):
        # ... existing code ...
        # Checkpoint saving based on steps or time
        current_time = time.time()
        should_save_step = self.training_config.save_steps_interval > 0 and self.global_step % self.training_config.save_steps_interval == 0
        should_save_time = self.training_config.time_save_interval_seconds > 0 and (current_time - self.last_save_time) >= self.training_config.time_save_interval_seconds

        if (should_save_step or should_save_time) and self.checkpoint_manager:
            logger.info(f"Triggering checkpoint save at step {self.global_step}. Step interval met: {should_save_step}, Time interval met: {should_save_time}")
            self.checkpoint_manager.save_checkpoint(
                step=self.global_step,
                model=self.model,
                optimizer=self.optimizer,
                # ... existing code ...
            )
        # ... existing code ... 