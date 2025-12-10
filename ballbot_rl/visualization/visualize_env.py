#!/usr/bin/env python3
"""
Visualize ballbot environment before training.

This script helps you inspect your environment configuration:
- Terrain generation (Perlin, flat, etc.)
- Robot spawn position
- Camera setup
- Reward configuration
- General environment behavior

Usage:
    # As CLI command (after installation):
    ballbot-visualize-env --env_config configs/env/perlin_directional.yaml
    
    # As Python module:
    python -m ballbot_rl.visualization.visualize_env --env_config configs/env/perlin_directional.yaml
    
    # Visualize from training config (uses env_config from training config)
    ballbot-visualize-env --train_config configs/train/ppo_directional.yaml
    
    # Visualize with custom terrain type
    ballbot-visualize-env --terrain_type flat --n_episodes 2
"""

import argparse
import numpy as np
import platform
from pathlib import Path
from termcolor import colored

from ballbot_rl.training.utils import make_ballbot_env
from ballbot_gym.core.config import load_config, get_component_config


def visualize_environment(
    env_config_path=None,
    train_config_path=None,
    terrain_type=None,
    n_episodes=2,
    n_steps_per_episode=500,
    gui=True,
    seed=42,
    action_mode="random"
):
    """
    Visualize a ballbot environment configuration.
    
    Args:
        env_config_path: Path to environment config YAML file
        train_config_path: Path to training config YAML file (uses env_config from it)
        terrain_type: Override terrain type (e.g., "flat", "perlin")
        n_episodes: Number of episodes to visualize
        n_steps_per_episode: Steps per episode
        gui: Whether to show MuJoCo GUI
        seed: Random seed
        action_mode: "random" (random actions) or "zero" (no actions, just observe)
    """
    print(colored("=" * 80, "cyan", attrs=["bold"]))
    print(colored("🔍 Ballbot Environment Visualizer", "cyan", attrs=["bold"]))
    print(colored("=" * 80, "cyan", attrs=["bold"]))
    
    # Load configuration
    terrain_config = None
    reward_config = None
    env_config = None
    
    if train_config_path:
        print(colored(f"\n📋 Loading training config: {train_config_path}", "cyan"))
        train_config = load_config(str(train_config_path))
        
        # Extract env_config path from training config
        env_config_rel_path = train_config.get("env_config")
        if env_config_rel_path:
            env_config_path = Path(env_config_rel_path)
            if not env_config_path.is_absolute():
                env_config_path = Path("configs/env") / env_config_path
            print(colored(f"   → Using env config: {env_config_path}", "cyan"))
        else:
            print(colored("   ⚠️  No env_config in training config, using defaults", "yellow"))
    
    if env_config_path:
        print(colored(f"\n📋 Loading environment config: {env_config_path}", "cyan"))
        env_config = load_config(str(env_config_path))
        terrain_config = get_component_config(env_config, "terrain")
        reward_config = get_component_config(env_config, "reward")
        
        # Extract camera/env/logging settings
        env_config = {
            "camera": env_config.get("camera", {}),
            "env": env_config.get("env", {}),
            "logging": env_config.get("logging", {})
        }
        
        print(colored(f"   ✓ Terrain: {terrain_config.get('type', 'unknown')}", "green"))
        print(colored(f"   ✓ Reward: {reward_config.get('type', 'unknown')}", "green"))
    elif terrain_type:
        # Use simple terrain type override
        terrain_config = {"type": terrain_type, "config": {}}
        reward_config = {"type": "directional", "config": {"target_direction": [0.0, 1.0]}}
        env_config = None
        print(colored(f"\n📋 Using terrain type: {terrain_type}", "cyan"))
    else:
        # Use defaults
        terrain_config = {"type": "perlin", "config": {}}
        reward_config = {"type": "directional", "config": {"target_direction": [0.0, 1.0]}}
        env_config = None
        print(colored("\n📋 Using default configuration", "cyan"))
    
    # Print configuration summary
    print(colored("\n📊 Configuration Summary:", "cyan", attrs=["bold"]))
    print(f"   Terrain Type: {terrain_config.get('type', 'unknown')}")
    if terrain_config.get('config'):
        print(f"   Terrain Config: {terrain_config['config']}")
    print(f"   Reward Type: {reward_config.get('type', 'unknown')}")
    if reward_config.get('config'):
        print(f"   Reward Config: {reward_config['config']}")
    if env_config:
        camera_config = env_config.get("camera", {})
        if camera_config:
            print(f"   Camera: {camera_config.get('height', '?')}x{camera_config.get('width', '?')}, "
                  f"RGB disabled: {camera_config.get('disable_rgb', False)}")
        env_settings = env_config.get("env", {})
        if env_settings:
            print(f"   Max Episode Steps: {env_settings.get('max_ep_steps', 'default')}")
    
    # Create environment with GUI
    print(colored(f"\n🎮 Creating environment with GUI={gui}...", "cyan", attrs=["bold"]))
    is_macos = platform.system() == 'Darwin'
    
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
    
    # Check if GUI is available
    if gui:
        if hasattr(env, 'passive_viewer') and env.passive_viewer is None:
            print(colored(
                "\n⚠️  Warning: MuJoCo viewer not available.\n"
                "   On macOS, you may need to use 'mjpython' instead of 'python':\n"
                "   mjpython scripts/visualization/visualize_env.py --env_config ...",
                "yellow", attrs=["bold"]
            ))
        else:
            print(colored("   ✓ MuJoCo viewer opened", "green"))
    
    # Print environment info
    print(colored("\n📐 Environment Information:", "cyan", attrs=["bold"]))
    print(f"   Action Space: {env.action_space}")
    print(f"   Observation Space Keys: {list(env.observation_space.spaces.keys())}")
    
    # Run visualization episodes
    print(colored(
        f"\n🎬 Running {n_episodes} episode(s) with {action_mode} actions...\n",
        "cyan", attrs=["bold"]
    ))
    
    total_reward = 0.0
    for episode in range(n_episodes):
        obs, info = env.reset(seed=seed + episode)
        episode_reward = 0.0
        step_count = 0
        
        print(colored(f"Episode {episode + 1}/{n_episodes} starting...", "yellow"))
        print(f"   Initial position: {info.get('pos2d', 'N/A')}")
        print(f"   Observation keys: {list(obs.keys())}")
        
        while step_count < n_steps_per_episode:
            # Choose action based on mode
            if action_mode == "random":
                action = env.action_space.sample()
            elif action_mode == "zero":
                action = np.zeros(env.action_space.shape, dtype=env.action_space.dtype)
            else:
                raise ValueError(f"Unknown action_mode: {action_mode}")
            
            obs, reward, terminated, truncated, info = env.step(action)
            
            episode_reward += reward
            step_count += 1
            
            if terminated or truncated:
                print(colored(
                    f"   Episode ended at step {step_count}: "
                    f"{'Terminated' if terminated else 'Truncated'}",
                    "yellow"
                ))
                break
            
            # Print progress every 100 steps
            if step_count % 100 == 0:
                pos = info.get('pos2d', [0, 0])
                print(f"   Step {step_count}: Reward={episode_reward:.2f}, Position=({pos[0]:.2f}, {pos[1]:.2f})")
        
        total_reward += episode_reward
        final_pos = info.get('pos2d', [0, 0])
        print(colored(
            f"✓ Episode {episode + 1} completed: {step_count} steps, "
            f"Total reward: {episode_reward:.2f}, "
            f"Final position: ({final_pos[0]:.2f}, {final_pos[1]:.2f})",
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
    
    print(colored("\n💡 Tips:", "cyan", attrs=["bold"]))
    print("   - Check terrain shape and difficulty")
    print("   - Verify robot spawns correctly")
    print("   - Observe reward behavior")
    print("   - Adjust config if needed before training")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize ballbot environment configuration before training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Visualize from environment config
  ballbot-visualize-env --env_config configs/env/perlin_directional.yaml
  
  # Or as Python module
  python -m ballbot_rl.visualization.visualize_env --env_config configs/env/perlin_directional.yaml
  
  # Visualize from training config (uses env_config from training config)
  ballbot-visualize-env --train_config configs/train/ppo_directional.yaml
  
  # Quick test with flat terrain
  ballbot-visualize-env --terrain_type flat --n_episodes 1
  
  # Test with zero actions (just observe terrain)
  ballbot-visualize-env --env_config configs/env/perlin_directional.yaml --action_mode zero
        """
    )
    
    config_group = parser.add_mutually_exclusive_group()
    config_group.add_argument(
        "--env_config",
        type=str,
        help="Path to environment config YAML file (configs/env/*.yaml)"
    )
    config_group.add_argument(
        "--train_config",
        type=str,
        help="Path to training config YAML file (uses env_config from it)"
    )
    config_group.add_argument(
        "--terrain_type",
        type=str,
        choices=["flat", "perlin", "ramp", "sinusoidal", "ridge_valley", "hills", 
                 "bowl", "gradient", "terraced", "wavy", "spiral", "mixed"],
        help="Override terrain type (uses default reward config)"
    )
    
    parser.add_argument(
        "--n_episodes",
        type=int,
        default=2,
        help="Number of episodes to visualize (default: 2)"
    )
    parser.add_argument(
        "--n_steps",
        type=int,
        default=500,
        help="Maximum steps per episode (default: 500)"
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
    parser.add_argument(
        "--action_mode",
        type=str,
        choices=["random", "zero"],
        default="random",
        help="Action mode: 'random' (random actions) or 'zero' (no actions, just observe) (default: random)"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.env_config and not args.train_config and not args.terrain_type:
        parser.error("Must specify one of: --env_config, --train_config, or --terrain_type")
    
    visualize_environment(
        env_config_path=args.env_config,
        train_config_path=args.train_config,
        terrain_type=args.terrain_type,
        n_episodes=args.n_episodes,
        n_steps_per_episode=args.n_steps,
        gui=not args.no_gui,
        seed=args.seed,
        action_mode=args.action_mode
    )


if __name__ == "__main__":
    main()

