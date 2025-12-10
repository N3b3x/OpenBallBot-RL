#!/usr/bin/env python3
"""
Interactive Environment Browser for Ballbot RL

This script provides an interactive interface to browse and visualize all available
ballbot environments. It allows you to:
- Browse predefined environment configurations
- Browse training configurations (which reference env configs)
- Select and configure terrain types interactively
- Select and configure reward types interactively
- Create fully custom environments

The script automatically detects macOS and uses mjpython when available.

Usage:
    # Interactive mode (recommended)
    ballbot-browse-env
    
    # Or as Python module
    python -m ballbot_rl.visualization.browse_environments
    
    # Direct selection (non-interactive)
    ballbot-browse-env --env_config configs/env/perlin_directional.yaml
"""

import argparse
import os
import sys
import platform
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional, List
import yaml

from termcolor import colored

# Import ballbot modules to trigger registration
import ballbot_gym  # noqa: F401 - Registers environment
from ballbot_gym.core.registry import ComponentRegistry
from ballbot_gym.core.config import load_config, get_component_config
from ballbot_rl.training.utils import make_ballbot_env
from ballbot_rl.visualization.visualize_env import visualize_environment


def check_environment_activation():
    """Check if virtual environment is activated and provide guidance."""
    venv_path = os.environ.get('VIRTUAL_ENV')
    conda_env = os.environ.get('CONDA_DEFAULT_ENV')
    
    if venv_path:
        venv_name = Path(venv_path).name
        print(colored(f"✓ Virtual environment activated: {venv_name}", "green"))
        return True
    elif conda_env:
        print(colored(f"✓ Conda environment activated: {conda_env}", "green"))
        return True
    else:
        print(colored("⚠️  No virtual environment detected", "yellow"))
        print(colored("   For best results, activate your environment:", "yellow"))
        print(colored("   source rl_lab_env/bin/activate  # or your venv path", "cyan"))
        print()
        return False


def check_mjpython():
    """Check if mjpython is available (required on macOS for GUI)."""
    is_macos = platform.system() == 'Darwin'
    
    if is_macos:
        import shutil
        mjpython_path = shutil.which('mjpython')
        if mjpython_path:
            print(colored(f"✓ mjpython found: {mjpython_path}", "green"))
            return True
        else:
            print(colored("⚠️  mjpython not found in PATH", "yellow"))
            print(colored("   On macOS, GUI visualization requires mjpython", "yellow"))
            print(colored("   Install MuJoCo and ensure mjpython is in PATH", "yellow"))
            print(colored("   Or use --no_gui flag for headless mode", "cyan"))
            print()
            return False
    return True  # Not macOS, regular python is fine


def find_config_files(config_dir: str, pattern: str = "*.yaml") -> List[Path]:
    """Find all config files in a directory."""
    config_path = Path(config_dir)
    if not config_path.exists():
        return []
    
    configs = sorted(config_path.glob(pattern))
    return [c for c in configs if c.is_file()]


def get_repo_root() -> Path:
    """Get the repository root directory."""
    # Try to find repo root by looking for configs directory
    current = Path(__file__).parent
    for _ in range(5):  # Go up max 5 levels
        if (current / "configs").exists():
            return current
        current = current.parent
    # Fallback: assume we're in ballbot_rl/visualization/
    return Path(__file__).parent.parent.parent.parent


def list_predefined_environments() -> List[Dict[str, Any]]:
    """List all predefined environment configurations."""
    repo_root = get_repo_root()
    env_config_dir = repo_root / "configs" / "env"
    
    envs = []
    for config_file in find_config_files(str(env_config_dir)):
        try:
            config = load_config(str(config_file))
            terrain_type = get_component_config(config, "terrain").get("type", "unknown")
            reward_type = get_component_config(config, "reward").get("type", "unknown")
            
            envs.append({
                "name": config_file.stem,
                "path": str(config_file.relative_to(repo_root)),
                "full_path": str(config_file),
                "terrain": terrain_type,
                "reward": reward_type,
                "description": config.get("description", "")
            })
        except Exception as e:
            print(colored(f"⚠️  Error loading {config_file.name}: {e}", "yellow"))
    
    return envs


def list_training_configs() -> List[Dict[str, Any]]:
    """List all training configurations (which reference env configs)."""
    repo_root = get_repo_root()
    train_config_dir = repo_root / "configs" / "train"
    
    configs = []
    for config_file in find_config_files(str(train_config_dir)):
        try:
            config = load_config(str(config_file))
            env_config_ref = config.get("env_config", "")
            
            configs.append({
                "name": config_file.stem,
                "path": str(config_file.relative_to(repo_root)),
                "full_path": str(config_file),
                "env_config": env_config_ref,
                "algo": config.get("algo", {}).get("name", "unknown")
            })
        except Exception as e:
            print(colored(f"⚠️  Error loading {config_file.name}: {e}", "yellow"))
    
    return configs


def get_terrain_config_interactive(terrain_type: str) -> Dict[str, Any]:
    """Interactively configure terrain parameters."""
    config = {}
    
    print(colored(f"\n📐 Configuring {terrain_type} terrain:", "cyan", attrs=["bold"]))
    print(colored("   (Press Enter to use default values)", "yellow"))
    
    # Common parameters
    if terrain_type in ["perlin", "hills", "mixed"]:
        seed_input = input("   Seed (default: None for random): ").strip()
        if seed_input:
            try:
                config["seed"] = int(seed_input) if seed_input.lower() != "none" else None
            except ValueError:
                config["seed"] = None
    
    # Terrain-specific parameters
    if terrain_type == "perlin":
        scale = input("   Scale (default: 25.0): ").strip()
        config["scale"] = float(scale) if scale else 25.0
        
        octaves = input("   Octaves (default: 4): ").strip()
        config["octaves"] = int(octaves) if octaves else 4
        
        persistence = input("   Persistence (default: 0.2): ").strip()
        config["persistence"] = float(persistence) if persistence else 0.2
        
        lacunarity = input("   Lacunarity (default: 2.0): ").strip()
        config["lacunarity"] = float(lacunarity) if lacunarity else 2.0
    
    elif terrain_type == "ramp":
        angle = input("   Ramp angle in degrees (default: 15.0): ").strip()
        config["ramp_angle"] = float(angle) if angle else 15.0
        
        direction = input("   Direction: x, y, or radial (default: x): ").strip()
        config["ramp_direction"] = direction if direction else "x"
        
        flat_ratio = input("   Flat ratio (default: 0.3): ").strip()
        config["flat_ratio"] = float(flat_ratio) if flat_ratio else 0.3
    
    elif terrain_type == "hills":
        num_hills = input("   Number of hills (default: 5): ").strip()
        config["num_hills"] = int(num_hills) if num_hills else 5
        
        hill_height = input("   Hill height (default: 0.7): ").strip()
        config["hill_height"] = float(hill_height) if hill_height else 0.7
        
        hill_radius = input("   Hill radius (default: 0.15): ").strip()
        config["hill_radius"] = float(hill_radius) if hill_radius else 0.15
    
    elif terrain_type == "sinusoidal":
        amplitude = input("   Amplitude (default: 0.5): ").strip()
        config["amplitude"] = float(amplitude) if amplitude else 0.5
        
        frequency = input("   Frequency (default: 0.1): ").strip()
        config["frequency"] = float(frequency) if frequency else 0.1
        
        direction = input("   Direction: x, y, or both (default: both): ").strip()
        config["direction"] = direction if direction else "both"
    
    elif terrain_type == "ridge_valley":
        ridge_height = input("   Ridge height (default: 0.6): ").strip()
        config["ridge_height"] = float(ridge_height) if ridge_height else 0.6
        
        valley_depth = input("   Valley depth (default: 0.4): ").strip()
        config["valley_depth"] = float(valley_depth) if valley_depth else 0.4
        
        spacing = input("   Spacing (default: 0.2): ").strip()
        config["spacing"] = float(spacing) if spacing else 0.2
        
        orientation = input("   Orientation: x or y (default: x): ").strip()
        config["orientation"] = orientation if orientation else "x"
    
    elif terrain_type == "bowl":
        depth = input("   Depth (default: 0.6): ").strip()
        config["depth"] = float(depth) if depth else 0.6
        
        radius = input("   Radius (default: 0.4): ").strip()
        config["radius"] = float(radius) if radius else 0.4
        
        center_x = input("   Center X (default: 0.5): ").strip()
        config["center_x"] = float(center_x) if center_x else 0.5
        
        center_y = input("   Center Y (default: 0.5): ").strip()
        config["center_y"] = float(center_y) if center_y else 0.5
    
    elif terrain_type == "gradient":
        max_slope = input("   Max slope in degrees (default: 20.0): ").strip()
        config["max_slope"] = float(max_slope) if max_slope else 20.0
        
        grad_type = input("   Gradient type: linear or exponential (default: linear): ").strip()
        config["gradient_type"] = grad_type if grad_type else "linear"
        
        direction = input("   Direction: x or y (default: x): ").strip()
        config["direction"] = direction if direction else "x"
    
    elif terrain_type == "terraced":
        num_terraces = input("   Number of terraces (default: 5): ").strip()
        config["num_terraces"] = int(num_terraces) if num_terraces else 5
        
        terrace_height = input("   Terrace height (default: 0.15): ").strip()
        config["terrace_height"] = float(terrace_height) if terrace_height else 0.15
        
        direction = input("   Direction: x or y (default: x): ").strip()
        config["direction"] = direction if direction else "x"
    
    elif terrain_type == "wavy":
        print("   Wave amplitudes (comma-separated, default: 0.3,0.2,0.1):")
        amplitudes = input("   ").strip()
        config["wave_amplitudes"] = [float(x) for x in amplitudes.split(",")] if amplitudes else [0.3, 0.2, 0.1]
        
        print("   Wave frequencies (comma-separated, default: 0.05,0.1,0.2):")
        frequencies = input("   ").strip()
        config["wave_frequencies"] = [float(x) for x in frequencies.split(",")] if frequencies else [0.05, 0.1, 0.2]
        
        print("   Wave directions in degrees (comma-separated, default: 0.0,45.0,90.0):")
        directions = input("   ").strip()
        config["wave_directions"] = [float(x) for x in directions.split(",")] if directions else [0.0, 45.0, 90.0]
    
    elif terrain_type == "spiral":
        tightness = input("   Spiral tightness (default: 0.1): ").strip()
        config["spiral_tightness"] = float(tightness) if tightness else 0.1
        
        height_var = input("   Height variation (default: 0.5): ").strip()
        config["height_variation"] = float(height_var) if height_var else 0.5
        
        direction = input("   Direction: cw or ccw (default: cw): ").strip()
        config["direction"] = direction if direction else "cw"
    
    elif terrain_type == "mixed":
        print(colored("   Mixed terrain requires multiple components.", "yellow"))
        print(colored("   Using default mixed configuration.", "yellow"))
        config = {
            "components": [
                {"type": "hills", "weight": 0.4, "config": {"num_hills": 3, "hill_height": 0.5}},
                {"type": "ramp", "weight": 0.3, "config": {"ramp_angle": 10.0, "ramp_direction": "x"}},
                {"type": "perlin", "weight": 0.3, "config": {"scale": 30.0, "octaves": 3}}
            ],
            "blend_mode": "additive"
        }
    
    # flat terrain has no config
    if terrain_type == "flat":
        print(colored("   Flat terrain has no configuration parameters.", "cyan"))
    
    return config


def get_reward_config_interactive(reward_type: str) -> Dict[str, Any]:
    """Interactively configure reward parameters."""
    config = {}
    
    print(colored(f"\n🎯 Configuring {reward_type} reward:", "cyan", attrs=["bold"]))
    print(colored("   (Press Enter to use default values)", "yellow"))
    
    if reward_type == "directional":
        print("   Target direction as [x, y] (default: [0.0, 1.0] for forward):")
        direction_input = input("   ").strip()
        if direction_input:
            try:
                # Parse list format
                direction_input = direction_input.strip("[]")
                parts = [float(x.strip()) for x in direction_input.split(",")]
                if len(parts) == 2:
                    config["target_direction"] = parts
                else:
                    print(colored("   Invalid format, using default [0.0, 1.0]", "yellow"))
                    config["target_direction"] = [0.0, 1.0]
            except:
                print(colored("   Invalid format, using default [0.0, 1.0]", "yellow"))
                config["target_direction"] = [0.0, 1.0]
        else:
            config["target_direction"] = [0.0, 1.0]
        
        scale = input("   Scale (default: 0.01): ").strip()
        config["scale"] = float(scale) if scale else 0.01
        
        action_reg = input("   Action regularization coefficient (default: -0.0001): ").strip()
        config["action_reg_coef"] = float(action_reg) if action_reg else -0.0001
        
        survival = input("   Survival bonus (default: 0.02): ").strip()
        config["survival_bonus"] = float(survival) if survival else 0.02
    
    elif reward_type == "distance":
        print("   Goal position as [x, y] (default: [5.0, 5.0]):")
        goal_input = input("   ").strip()
        if goal_input:
            try:
                goal_input = goal_input.strip("[]")
                parts = [float(x.strip()) for x in goal_input.split(",")]
                if len(parts) == 2:
                    config["goal_position"] = parts
                else:
                    print(colored("   Invalid format, using default [5.0, 5.0]", "yellow"))
                    config["goal_position"] = [5.0, 5.0]
            except:
                print(colored("   Invalid format, using default [5.0, 5.0]", "yellow"))
                config["goal_position"] = [5.0, 5.0]
        else:
            config["goal_position"] = [5.0, 5.0]
        
        scale = input("   Scale (default: 1.0): ").strip()
        config["scale"] = float(scale) if scale else 1.0
    
    return config


def get_env_config_interactive() -> Dict[str, Any]:
    """Interactively configure environment settings."""
    config = {}
    
    print(colored("\n⚙️  Configuring environment settings:", "cyan", attrs=["bold"]))
    print(colored("   (Press Enter to use default values)", "yellow"))
    
    # Camera settings
    print("\n📷 Camera settings:")
    height = input("   Image height (default: 64): ").strip()
    width = input("   Image width (default: 64): ").strip()
    
    camera_config = {}
    if height:
        camera_config["height"] = int(height)
    if width:
        camera_config["width"] = int(width)
    
    disable_rgb = input("   Disable RGB (depth only)? [y/N]: ").strip().lower()
    camera_config["disable_rgb"] = disable_rgb == "y"
    
    if camera_config:
        config["camera"] = camera_config
    
    # Environment settings
    print("\n🌍 Environment settings:")
    max_steps = input("   Max episode steps (default: 4000): ").strip()
    if max_steps:
        if "env" not in config:
            config["env"] = {}
        config["env"]["max_ep_steps"] = int(max_steps)
    
    max_tilt = input("   Max allowed tilt in degrees (default: 20): ").strip()
    if max_tilt:
        if "env" not in config:
            config["env"] = {}
        config["env"]["max_allowed_tilt"] = float(max_tilt)
    
    return config


def interactive_menu():
    """Display interactive menu and handle user selection."""
    print(colored("=" * 80, "cyan", attrs=["bold"]))
    print(colored("🌍 Ballbot Environment Browser", "cyan", attrs=["bold"]))
    print(colored("=" * 80, "cyan", attrs=["bold"]))
    print()
    
    # Check environment
    check_environment_activation()
    check_mjpython()
    
    # Get available components
    available_terrains = ComponentRegistry.list_terrains()
    available_rewards = ComponentRegistry.list_rewards()
    
    print(colored("Available Components:", "cyan", attrs=["bold"]))
    print(f"   Terrains ({len(available_terrains)}): {', '.join(available_terrains)}")
    print(f"   Rewards ({len(available_rewards)}): {', '.join(available_rewards)}")
    print()
    
    # Menu options
    print(colored("Select visualization mode:", "cyan", attrs=["bold"]))
    print("   1. Browse predefined environment configurations")
    print("   2. Browse training configurations (uses env configs)")
    print("   3. Create custom environment (select terrain & reward)")
    print("   4. Quick test (flat terrain, default settings)")
    print("   0. Exit")
    print()
    
    choice = input(colored("Enter choice [0-4]: ", "cyan")).strip()
    
    env_config_path = None
    train_config_path = None
    terrain_config = None
    reward_config = None
    env_config = None
    
    if choice == "1":
        # Browse predefined env configs
        envs = list_predefined_environments()
        if not envs:
            print(colored("⚠️  No environment configurations found", "yellow"))
            return
        
        print(colored("\n📋 Predefined Environment Configurations:", "cyan", attrs=["bold"]))
        for i, env in enumerate(envs, 1):
            print(f"   {i}. {env['name']}")
            print(f"      Terrain: {env['terrain']}, Reward: {env['reward']}")
            if env['description']:
                print(f"      {env['description']}")
        
        env_choice = input(colored(f"\nSelect environment [1-{len(envs)}]: ", "cyan")).strip()
        try:
            idx = int(env_choice) - 1
            if 0 <= idx < len(envs):
                env_config_path = envs[idx]['full_path']
                print(colored(f"✓ Selected: {envs[idx]['name']}", "green"))
            else:
                print(colored("Invalid selection", "red"))
                return
        except ValueError:
            print(colored("Invalid selection", "red"))
            return
    
    elif choice == "2":
        # Browse training configs
        configs = list_training_configs()
        if not configs:
            print(colored("⚠️  No training configurations found", "yellow"))
            return
        
        print(colored("\n📋 Training Configurations:", "cyan", attrs=["bold"]))
        for i, cfg in enumerate(configs, 1):
            print(f"   {i}. {cfg['name']}")
            print(f"      Algorithm: {cfg['algo']}, Env Config: {cfg['env_config']}")
        
        cfg_choice = input(colored(f"\nSelect training config [1-{len(configs)}]: ", "cyan")).strip()
        try:
            idx = int(cfg_choice) - 1
            if 0 <= idx < len(configs):
                train_config_path = configs[idx]['full_path']
                print(colored(f"✓ Selected: {configs[idx]['name']}", "green"))
            else:
                print(colored("Invalid selection", "red"))
                return
        except ValueError:
            print(colored("Invalid selection", "red"))
            return
    
    elif choice == "3":
        # Custom environment
        print(colored("\n🏔️  Select Terrain Type:", "cyan", attrs=["bold"]))
        for i, terrain in enumerate(available_terrains, 1):
            print(f"   {i}. {terrain}")
        
        terrain_choice = input(colored(f"\nSelect terrain [1-{len(available_terrains)}]: ", "cyan")).strip()
        try:
            idx = int(terrain_choice) - 1
            if 0 <= idx < len(available_terrains):
                terrain_type = available_terrains[idx]
                terrain_config_dict = get_terrain_config_interactive(terrain_type)
                terrain_config = {"type": terrain_type, "config": terrain_config_dict}
                print(colored(f"✓ Selected terrain: {terrain_type}", "green"))
            else:
                print(colored("Invalid selection", "red"))
                return
        except ValueError:
            print(colored("Invalid selection", "red"))
            return
        
        print(colored("\n🎯 Select Reward Type:", "cyan", attrs=["bold"]))
        for i, reward in enumerate(available_rewards, 1):
            print(f"   {i}. {reward}")
        
        reward_choice = input(colored(f"\nSelect reward [1-{len(available_rewards)}]: ", "cyan")).strip()
        try:
            idx = int(reward_choice) - 1
            if 0 <= idx < len(available_rewards):
                reward_type = available_rewards[idx]
                reward_config_dict = get_reward_config_interactive(reward_type)
                reward_config = {"type": reward_type, "config": reward_config_dict}
                print(colored(f"✓ Selected reward: {reward_type}", "green"))
            else:
                print(colored("Invalid selection", "red"))
                return
        except ValueError:
            print(colored("Invalid selection", "red"))
            return
        
        # Optional: configure environment settings
        configure_env = input(colored("\nConfigure environment settings? [y/N]: ", "cyan")).strip().lower()
        if configure_env == "y":
            env_config = get_env_config_interactive()
    
    elif choice == "4":
        # Quick test
        terrain_config = {"type": "flat", "config": {}}
        reward_config = {"type": "directional", "config": {"target_direction": [0.0, 1.0]}}
        print(colored("✓ Using flat terrain with directional reward", "green"))
    
    elif choice == "0":
        print(colored("Exiting...", "cyan"))
        return
    
    else:
        print(colored("Invalid choice", "red"))
        return
    
    # Visualization settings
    print(colored("\n🎬 Visualization Settings:", "cyan", attrs=["bold"]))
    n_episodes = input("   Number of episodes (default: 2): ").strip()
    n_episodes = int(n_episodes) if n_episodes else 2
    
    n_steps = input("   Steps per episode (default: 500): ").strip()
    n_steps = int(n_steps) if n_steps else 500
    
    use_gui = input("   Show GUI? [Y/n]: ").strip().lower()
    gui = use_gui != "n"
    
    seed = input("   Random seed (default: 42): ").strip()
    seed = int(seed) if seed else 42
    
    action_mode = input("   Action mode: random or zero [random]: ").strip()
    action_mode = action_mode if action_mode in ["random", "zero"] else "random"
    
    # Visualize
    print(colored("\n🚀 Starting visualization...", "cyan", attrs=["bold"]))
    print()
    
    # For custom environments with full config, create a temporary config file
    temp_config_path = None
    if terrain_config and reward_config and not env_config_path and not train_config_path:
        # Create temporary config file for custom environment
        temp_config = {
            "terrain": terrain_config,
            "reward": reward_config
        }
        
        # Add env_config if provided
        if env_config:
            temp_config.update(env_config)
        
        # Write to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(temp_config, f)
            temp_config_path = f.name
        
        print(colored(f"   Created temporary config: {temp_config_path}", "cyan"))
        env_config_path = temp_config_path
    
    try:
        # Visualize environment
        visualize_environment(
            env_config_path=env_config_path,
            train_config_path=train_config_path,
            terrain_type=None,
            n_episodes=n_episodes,
            n_steps_per_episode=n_steps,
            gui=gui,
            seed=seed,
            action_mode=action_mode
        )
    finally:
        # Clean up temporary config file
        if temp_config_path and os.path.exists(temp_config_path):
            os.unlink(temp_config_path)
            print(colored(f"   Cleaned up temporary config", "cyan"))


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Interactive environment browser for Ballbot RL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode (recommended)
  ballbot-browse-env
  
  # Direct visualization from config
  ballbot-browse-env --env_config configs/env/perlin_directional.yaml
  
  # From training config
  ballbot-browse-env --train_config configs/train/ppo_directional.yaml
        """
    )
    
    config_group = parser.add_mutually_exclusive_group()
    config_group.add_argument(
        "--env_config",
        type=str,
        help="Path to environment config YAML file (skips interactive menu)"
    )
    config_group.add_argument(
        "--train_config",
        type=str,
        help="Path to training config YAML file (skips interactive menu)"
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
        help="Action mode: 'random' or 'zero' (default: random)"
    )
    
    args = parser.parse_args()
    
    # If config provided, skip interactive menu
    if args.env_config or args.train_config:
        visualize_environment(
            env_config_path=args.env_config,
            train_config_path=args.train_config,
            n_episodes=args.n_episodes,
            n_steps_per_episode=args.n_steps,
            gui=not args.no_gui,
            seed=args.seed,
            action_mode=args.action_mode
        )
    else:
        # Interactive mode
        interactive_menu()


if __name__ == "__main__":
    main()
