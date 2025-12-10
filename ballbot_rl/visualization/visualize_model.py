#!/usr/bin/env python3
"""
Visualize a trained ballbot model in MuJoCo.

Usage:
    # As CLI command (after installation):
    ballbot-visualize-model --model_path outputs/experiments/runs/.../best_model/best_model.zip
    
    # As Python module:
    python -m ballbot_rl.visualization.visualize_model --model_path outputs/experiments/runs/.../best_model/best_model.zip
    
    # With more episodes:
    ballbot-visualize-model --model_path .../best_model.zip --n_episodes 5
"""

import argparse
import numpy as np
import torch
from pathlib import Path
from termcolor import colored

from stable_baselines3 import PPO, SAC
from ballbot_rl.training.utils import make_ballbot_env
from ballbot_gym.core.config import load_config, get_component_config


def visualize_model(model_path, n_episodes=3, gui=True, seed=42):
    """
    Load a trained model and visualize it in MuJoCo.
    
    Args:
        model_path: Path to the trained model (.zip file)
        n_episodes: Number of episodes to visualize
        gui: Whether to show MuJoCo GUI
        seed: Random seed for reproducibility
    """
    model_path = Path(model_path).resolve()
    
    if not model_path.exists():
        print(colored(f"❌ Error: Model file not found: {model_path}", "red", attrs=["bold"]))
        return
    
    print(colored(f"📦 Loading model from: {model_path}", "cyan", attrs=["bold"]))
    
    # Load model
    try:
        # Try to detect algorithm from file or default to PPO
        model = PPO.load(str(model_path))
        print(colored("✓ Model loaded successfully (PPO)", "green"))
    except Exception as e:
        print(colored(f"⚠️  Failed to load as PPO, trying SAC: {e}", "yellow"))
        try:
            model = SAC.load(str(model_path))
            print(colored("✓ Model loaded successfully (SAC)", "green"))
        except Exception as e2:
            print(colored(f"❌ Failed to load model: {e2}", "red", attrs=["bold"]))
            return
    
    # Try to load training config if available
    config_path = model_path.parent.parent / "config.yaml"
    if config_path.exists():
        config = load_config(str(config_path))
        terrain_config = get_component_config(config, "terrain")
        reward_config = get_component_config(config, "reward")
        env_config = {
            "camera": config.get("camera", {}),
            "env": config.get("env", {}),
            "logging": config.get("logging", {})
        }
        print(colored(f"✓ Loaded training config from: {config_path}", "green"))
    else:
        # Use defaults
        terrain_config = {"type": "perlin", "config": {}}
        reward_config = {"type": "directional", "config": {"target_direction": [0.0, 1.0]}}
        env_config = None
        print(colored("⚠️  No config.yaml found, using defaults", "yellow"))
    
    # Create environment with GUI
    print(colored(f"\n🎮 Creating environment with GUI={gui}...", "cyan", attrs=["bold"]))
    env_factory = make_ballbot_env(
        terrain_config=terrain_config,
        reward_config=reward_config,
        env_config=env_config,
        gui=gui,
        log_options={"cams": False, "reward_terms": False},
        seed=seed,
        eval_env=True
    )
    env = env_factory()
    
    if gui and hasattr(env, 'passive_viewer') and env.passive_viewer is None:
        print(colored(
            "⚠️  Warning: MuJoCo viewer not available.\n"
            "   On macOS, you may need to use 'mjpython' instead of 'python':\n"
            "   mjpython visualize_model.py --model_path ...",
            "yellow", attrs=["bold"]
        ))
    
    print(colored(f"\n🎬 Running {n_episodes} episode(s) with deterministic policy...\n", "cyan", attrs=["bold"]))
    
    # Run episodes
    total_reward = 0.0
    for episode in range(n_episodes):
        obs, info = env.reset()
        episode_reward = 0.0
        step_count = 0
        done = False
        
        print(colored(f"Episode {episode + 1}/{n_episodes} starting...", "yellow"))
        
        while not done and step_count < 4000:  # Max episode length
            # Use deterministic policy (no exploration)
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            
            episode_reward += reward
            step_count += 1
            done = terminated or truncated
            
            # Print progress every 100 steps
            if step_count % 100 == 0:
                print(f"  Step {step_count}: Reward={episode_reward:.2f}, Current reward={reward:.4f}")
        
        total_reward += episode_reward
        print(colored(
            f"✓ Episode {episode + 1} completed: {step_count} steps, Total reward: {episode_reward:.2f}",
            "green", attrs=["bold"]
        ))
        
        if episode < n_episodes - 1:
            input(colored("\nPress Enter to continue to next episode...", "cyan"))
    
    avg_reward = total_reward / n_episodes
    print(colored(
        f"\n📊 Summary: {n_episodes} episodes, Average reward: {avg_reward:.2f}",
        "cyan", attrs=["bold"]
    ))
    
    print(colored("\n✓ Visualization complete. Closing environment...", "green"))
    env.close()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize a trained ballbot model in MuJoCo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Visualize best model from a training run
  ballbot-visualize-model --model_path outputs/experiments/runs/.../best_model/best_model.zip
  
  # Or as Python module
  python -m ballbot_rl.visualization.visualize_model --model_path outputs/experiments/runs/.../best_model/best_model.zip
  
  # Visualize with 5 episodes
  ballbot-visualize-model --model_path outputs/experiments/runs/.../best_model/best_model.zip --n_episodes 5
  
  # Visualize without GUI (headless)
  ballbot-visualize-model --model_path outputs/experiments/runs/.../best_model/best_model.zip --no_gui
        """
    )
    
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the trained model (.zip file)"
    )
    parser.add_argument(
        "--n_episodes",
        type=int,
        default=3,
        help="Number of episodes to visualize (default: 3)"
    )
    parser.add_argument(
        "--no_gui",
        action="store_true",
        help="Disable MuJoCo GUI (headless mode)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    
    args = parser.parse_args()
    
    visualize_model(
        model_path=args.model_path,
        n_episodes=args.n_episodes,
        gui=not args.no_gui,
        seed=args.seed
    )


if __name__ == "__main__":
    main()

