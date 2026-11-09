"""Train two independent DQN agents on the CityFlow corridor."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from agents.dqn import DQNAgent
from environment import (
    ACTION_COUNT,
    OBSERVATION_SIZE,
    CorridorEnv,
)

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = PROJECT_ROOT / "data" / "corridor_2x1" / "cityflow_config.json"
DEFAULT_CHECKPOINT = PROJECT_ROOT / "models" / "dqn_traffic_2x1.pt"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--max-seconds", type=int, default=1800)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--metrics",
        type=Path,
        default=PROJECT_ROOT / "results" / "training_metrics.jsonl",
    )
    args = parser.parse_args()
    if args.episodes < 1 or args.max_seconds < 1:
        parser.error("--episodes and --max-seconds must be positive.")
    return args


def main() -> None:
    args = _parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for DQN training. Install a CUDA-enabled PyTorch "
            "build and verify GPU access from WSL2."
        )
    device = torch.device("cuda")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    print(f"Training on {torch.cuda.get_device_name(device)}")

    env = CorridorEnv(args.config, max_episode_seconds=args.max_seconds)
    agents = {
        signal_id: DQNAgent(
            OBSERVATION_SIZE,
            ACTION_COUNT,
            device=device,
            seed=args.seed + index,
        )
        for index, signal_id in enumerate(env.intersections)
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    total_training_steps = args.episodes * (
        args.max_seconds // env.decision_interval_seconds + 1
    )
    global_step = 0
    epsilon_start = 1.0
    epsilon_end = 0.05
    epsilon_decay_steps = max(1, int(total_training_steps * 0.8))

    try:
        with args.metrics.open("w", encoding="utf-8") as metrics_file:
            for episode in range(args.episodes):
                states, reset_info = env.reset(seed=args.seed + episode)
                action_masks = reset_info["action_masks"]
                totals = {
                    "reward": 0.0,
                    "mean_queue": 0.0,
                    "throughput": 0,
                    "decisions": 0,
                }
                terminated = truncated = False

                while not (terminated or truncated):
                    fraction = min(global_step / epsilon_decay_steps, 1.0)
                    epsilon = epsilon_start + fraction * (epsilon_end - epsilon_start)
                    actions = {
                        signal_id: agent.act(
                            states[signal_id],
                            action_masks[signal_id],
                            epsilon,
                        )
                        for signal_id, agent in agents.items()
                    }
                    next_states, rewards, terminated, truncated, info = env.step(
                        actions
                    )
                    next_masks = info["action_masks"]
                    for signal_id, agent in agents.items():
                        agent.remember(
                            states[signal_id],
                            actions[signal_id],
                            rewards[signal_id],
                            next_states[signal_id],
                            next_masks[signal_id],
                            terminated or truncated,
                        )
                        agent.optimize()
                        totals["reward"] += rewards[signal_id]
                    totals["mean_queue"] += info["mean_queue"]
                    totals["throughput"] = info["throughput"]
                    totals["decisions"] += 1
                    states = next_states
                    action_masks = next_masks
                    global_step += 1

                denominator = max(1, totals["decisions"])
                record = {
                    "episode": episode + 1,
                    "seed": args.seed + episode,
                    "epsilon": epsilon,
                    "reward_per_agent_step": totals["reward"]
                    / (denominator * len(agents)),
                    "mean_queue": totals["mean_queue"] / denominator,
                    "throughput": totals["throughput"],
                    "decisions": totals["decisions"],
                }
                metrics_file.write(json.dumps(record) + "\n")
                metrics_file.flush()
                print(
                    f"Episode {episode + 1}/{args.episodes}: "
                    f"reward={record['reward_per_agent_step']:.3f}, "
                    f"mean_queue={record['mean_queue']:.2f}, "
                    f"throughput={record['throughput']}"
                )

        checkpoint = {
            "format_version": 1,
            "observation_size": OBSERVATION_SIZE,
            "action_count": ACTION_COUNT,
            "signal_ids": list(agents),
            "episodes": args.episodes,
            "seed": args.seed,
            "agents": {
                signal_id: agent.state_dict()
                for signal_id, agent in agents.items()
            },
        }
        temporary_path = args.output.with_suffix(args.output.suffix + ".tmp")
        torch.save(checkpoint, temporary_path)
        temporary_path.replace(args.output)
        print(f"Saved independent J1/J2 agents to {args.output.resolve()}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
