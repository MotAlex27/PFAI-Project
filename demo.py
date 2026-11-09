"""Run one CityFlow policy and export a visualizer-compatible replay."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import torch

from agents.dqn import DQNAgent
from environment import (
    ACTION_COUNT,
    OBSERVATION_SIZE,
    SIGNAL_IDS,
    CorridorEnv,
)
from train import DEFAULT_CHECKPOINT, DEFAULT_CONFIG

PROJECT_ROOT = Path(__file__).resolve().parent


def _load_agents(
    checkpoint_path: Path, device: torch.device
) -> dict[str, DQNAgent]:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"DQN checkpoint not found: {checkpoint_path}. Run train.py first."
        )
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=True
    )
    if checkpoint.get("observation_size") != OBSERVATION_SIZE:
        raise ValueError("Checkpoint observation size does not match this environment.")
    if checkpoint.get("action_count") != ACTION_COUNT:
        raise ValueError("Checkpoint action count does not match this environment.")
    if set(checkpoint.get("signal_ids", [])) != set(SIGNAL_IDS):
        raise ValueError("Checkpoint must contain independent agents for J1 and J2.")
    agents = {}
    for index, signal_id in enumerate(SIGNAL_IDS):
        agent = DQNAgent(
            OBSERVATION_SIZE,
            ACTION_COUNT,
            device=device,
            seed=index,
        )
        agent.load_state_dict(checkpoint["agents"][signal_id])
        agents[signal_id] = agent
    return agents


def _actions(
    policy: str,
    env: CorridorEnv,
    observations: dict[str, np.ndarray],
    masks: dict[str, np.ndarray],
    agents: dict[str, DQNAgent],
) -> dict[str, int]:
    if policy == "dqn":
        return {
            signal_id: agent.act(observations[signal_id], masks[signal_id])
            for signal_id, agent in agents.items()
        }
    actions = {}
    for signal_id in SIGNAL_IDS:
        if policy == "fixed":
            switch = env.green_elapsed[signal_id] >= 30
        else:
            pressures = env.phase_pressures(signal_id)
            target_phase = int(pressures[1] > pressures[0])
            switch = target_phase != env.phase[signal_id]
        if not masks[signal_id][0]:
            actions[signal_id] = 1
        else:
            actions[signal_id] = int(switch and masks[signal_id][1])
    return actions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--policy", choices=("dqn", "fixed", "max-pressure"), default="dqn"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--max-seconds", type=int, default=1800)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument(
        "--replay-output",
        type=Path,
        default=PROJECT_ROOT / "data" / "corridor_2x1" / "replay.txt",
    )
    args = parser.parse_args()
    if args.max_seconds <= 0:
        parser.error("--max-seconds must be positive.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agents = _load_agents(args.checkpoint, device) if args.policy == "dqn" else {}
    print(f"Inference device: {device}")
    env = CorridorEnv(
        args.config, max_episode_seconds=args.max_seconds, replay=True
    )
    try:
        observations, reset_info = env.reset(seed=args.seed)
        masks = reset_info["action_masks"]
        terminated = truncated = False
        while not (terminated or truncated):
            actions = _actions(args.policy, env, observations, masks, agents)
            observations, _, terminated, truncated, info = env.step(actions)
            masks = info["action_masks"]
        replay_path = env.base_dir / env.config.get("replayLogFile", "replay.txt")
    finally:
        env.close()

    if not replay_path.is_file():
        raise FileNotFoundError(
            f"CityFlow did not create its replay file at {replay_path}."
        )
    args.replay_output.parent.mkdir(parents=True, exist_ok=True)
    if replay_path.resolve() != args.replay_output.resolve():
        shutil.copy2(replay_path, args.replay_output)
    print(f"Exported CityFlow replay to {args.replay_output.resolve()}")
    print(
        f"Episode finished at {info['simulation_seconds']}s; "
        f"throughput={info['throughput']}, mean queue={info['mean_queue']:.2f}"
    )


if __name__ == "__main__":
    main()
