"""
Visualization utilities for ballbot environments and trained models.

This module provides tools to visualize:
- Environment configurations before training
- Trained models during/after training
- Training progress from CSV logs

Usage:
    # As CLI commands (after installation):
    ballbot-visualize-env --env_config configs/env/perlin_directional.yaml
    ballbot-visualize-model --model_path outputs/.../best_model.zip
    ballbot-plot-training --csv outputs/.../progress.csv --config outputs/.../config.yaml
    
    # As Python modules:
    python -m ballbot_rl.visualization.visualize_env --env_config ...
    python -m ballbot_rl.visualization.visualize_model --model_path ...
    python -m ballbot_rl.visualization.plot_training --csv ... --config ...
"""

from ballbot_rl.visualization.visualize_env import visualize_environment, main as visualize_env_main
from ballbot_rl.visualization.visualize_model import visualize_model, main as visualize_model_main
from ballbot_rl.visualization.plot_training import plot_train_val_progress, main as plot_training_main

__all__ = [
    "visualize_environment",
    "visualize_model",
    "plot_train_val_progress",
]

