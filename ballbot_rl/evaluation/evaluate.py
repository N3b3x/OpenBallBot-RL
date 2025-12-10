import numpy as np
import json
import pdb
import torch
import argparse
import random
from pathlib import Path
import yaml

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from termcolor import colored

from ballbot_rl.training.utils import make_ballbot_env
from ballbot_gym.core.config import load_config, get_component_config


def main(args, seed=None, eval_config=None):

    with torch.no_grad():
        model_path = Path(args.path).resolve()
        
        # Auto-detect algorithm (model files contain metadata)
        # Try PPO first (most common), then SAC
        model = None
        if args.algo:
            # User specified algorithm explicitly
            if args.algo == "ppo":
                model = PPO.load(str(model_path))
            elif args.algo == "sac":
                model = SAC.load(str(model_path))
            else:
                raise Exception(f"Unknown algorithm: {args.algo}. Supported: ppo, sac")
        else:
            # Auto-detect: try PPO first, then SAC
            try:
                model = PPO.load(str(model_path))
                print(colored("✓ Auto-detected algorithm: PPO", "green"))
            except Exception as e:
                try:
                    model = SAC.load(str(model_path))
                    print(colored("✓ Auto-detected algorithm: SAC", "green"))
                except Exception as e2:
                    raise Exception(
                        f"Failed to load model. Tried PPO and SAC.\n"
                        f"PPO error: {e}\nSAC error: {e2}\n"
                        f"You can specify --algo ppo or --algo sac explicitly."
                    )

        # Determine terrain/reward configs
        # Priority: CLI override > eval config > model's training config
        if args.override_terrain_type:
            # CLI override: use simple terrain_type string (backward compat)
            terrain_config = {"type": args.override_terrain_type, "config": {}}
            reward_config = {"type": "directional", "config": {"target_direction": [0.0, 1.0]}}
            env_config = None
        elif eval_config and eval_config.get("env_config"):
            # Load env config from eval config
            env_config_path = eval_config["env_config"]
            if not Path(env_config_path).is_absolute():
                env_config_path = Path("configs/env") / env_config_path
            env_config = load_config(str(env_config_path))
            terrain_config = get_component_config(env_config, "terrain")
            reward_config = get_component_config(env_config, "reward")
            # Extract camera/env/logging settings
            env_config = {
                "camera": env_config.get("camera", {}),
                "env": env_config.get("env", {}),
                "logging": env_config.get("logging", {})
            }
        else:
            # Use model's training config (if available) or defaults
            terrain_config = {"type": getattr(model, "terrain_type", "perlin"), "config": {}}
            reward_config = {"type": "directional", "config": {"target_direction": [0.0, 1.0]}}
            env_config = None

        log_options = {
            "cams": False,
            "reward_terms": True
        }
        if eval_config:
            log_options.update(eval_config.get("logging", {}))

        env = make_ballbot_env(
            gui=eval_config.get("render", True) if eval_config else True,
            terrain_config=terrain_config,
            reward_config=reward_config,
            env_config=env_config,
            log_options=log_options,
            seed=seed,
            eval_env=True)()

        if args.override_terrain_type:
            print(
                colored(
                    f"Policy was trained on terrain type {getattr(model, 'terrain_type', 'unknown')}, "
                    f"but it will be tested on terrain type {args.override_terrain_type}",
                    "yellow",
                    attrs=["bold"]))
        elif eval_config and eval_config.get("env_config"):
            print(
                colored(
                    f"Using environment config: {eval_config['env_config']}",
                    "yellow",
                    attrs=["bold"]))
        else:
            print(
                colored(
                    f"Policy was trained on terrain type {getattr(model, 'terrain_type', 'unknown')} "
                    f"and will be tested on the same terrain type.",
                    "yellow",
                    attrs=["bold"]))

        p_sum = sum([
            param.abs().sum().item() for param in model.policy.parameters()
            if param.requires_grad
        ])
        print(colored(f"sum_of_model_params=={p_sum}", "yellow"))

        n_test = eval_config.get("n_test_episodes", args.n_test) if eval_config else args.n_test
        deterministic = eval_config.get("deterministic", True) if eval_config else True
        
        print(colored(f"\n🧪 Running {n_test} evaluation episode(s)...", "cyan", attrs=["bold"]))
        print(colored(f"   Deterministic: {deterministic}", "cyan"))
        
        total_reward = 0.0
        episode_lengths = []
        
        for test_i in range(n_test):
            obs, _ = env.reset(seed=seed + test_i)
            done = False

            G_tau = 0  # Discounted return
            gamma = 0.99999
            episode_reward = 0.0
            count = 0
            
            while not done:
                action, _ = model.predict(obs, deterministic=deterministic)
                obs, reward, terminated, truncated, info = env.step(action)

                G_tau += gamma**count * reward
                episode_reward += reward
                count += 1
                done = terminated or truncated

            total_reward += episode_reward
            episode_lengths.append(count)
            print(colored(
                f"  Episode {test_i + 1}/{n_test}: "
                f"Reward={episode_reward:.2f}, "
                f"Length={count} steps, "
                f"Discounted Return (G_tau)={G_tau:.2f}",
                "green"
            ))
        
        avg_reward = total_reward / n_test
        avg_length = np.mean(episode_lengths)
        print(colored(
            f"\n📊 Summary: "
            f"Avg Reward={avg_reward:.2f} ± {np.std([total_reward/n_test]):.2f}, "
            f"Avg Length={avg_length:.1f} ± {np.std(episode_lengths):.1f}",
            "cyan", attrs=["bold"]
        ))

    env.close()

    return model


def cli_main():
    """CLI entry point for evaluation."""
    _parser = argparse.ArgumentParser(description="Test a policy.")
    _parser.add_argument(
        "--algo", 
        type=str, 
        default=None,
        help="Algorithm type (ppo, sac). If not specified, will auto-detect from model file."
    )
    _parser.add_argument("--path", type=str, required=True, help="path to policy")
    _parser.add_argument("--n_test",
                         type=int,
                         help="How many times to test policy",
                         default=1)
    _parser.add_argument(
        "--seed",
        type=int,
        help="For repeatablility. If not set, will be chosen randomly",
        default=-1)
    _parser.add_argument(
        "--override_terrain_type",
        type=str,
        default="",
        help=
        "Can be used to run the policy on a terrain type it hasn't been trained on. See the ballbot env for options."
    )
    _parser.add_argument(
        "--eval_config",
        type=str,
        default="",
        help="Path to evaluation config YAML file (default: configs/eval/default.yaml)"
    )

    _args = _parser.parse_args()

    # Load eval config if provided
    eval_config = None
    if _args.eval_config:
        eval_config = load_config(_args.eval_config)
    elif Path("configs/eval/default.yaml").exists():
        eval_config = load_config("configs/eval/default.yaml")

    _seed = _args.seed if _args.seed != -1 else np.random.randint(10000)
    random.seed(_seed)
    np.random.seed(_seed)
    torch.manual_seed(_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if torch.cuda.is_available():
        torch.cuda.manual_seed(_seed)
    torch.cuda.manual_seed_all(_seed)

    _model = main(_args, seed=_seed, eval_config=eval_config)
    return _model


if __name__ == "__main__":
    cli_main()
