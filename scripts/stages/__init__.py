"""
Pipeline Stages Module
=====================

Implements the individual pipeline stages:
- Process: Data preprocessing
- Optimize: Model optimization
- Train: Model training
- Generate: Text generation
"""

from pipeline.stages.pipeline_stages import (
    run_process_stage,
    run_optimize_stage,
    run_train_stage,
    run_generate_stage
) 