"""Benchmark fixed-time, Max-Pressure, and trained DQN control."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from demo import _actions, _load_agents
from environment import (
    SIGNAL_IDS,
    CorridorEnv,
)
from train import DEFAULT_CHECKPOINT, DEFAULT_CONFIG

PROJECT_ROOT = Path(__file__).resolve().parent
POLICIES = ("fixed", "max-pressure", "dqn")


def _run_policy(
    policy: str,
    config_path: Path,
    checkpoint: Path,
    seed: int,
    max_seconds: int,
    device: torch.device,
) -> dict[str, float | int | str]:
    with tempfile.TemporaryDirectory(prefix="corridor-eval-") as temp_dir:
        seeded_config = json.loads(config_path.read_text(encoding="utf-8"))
        seeded_config["seed"] = seed
        seeded_config["dir"] = str(config_path.resolve().parent)
        temporary_config = Path(temp_dir) / "cityflow_config.json"
        temporary_config.write_text(
            json.dumps(seeded_config), encoding="utf-8"
        )
        env = CorridorEnv(
            temporary_config, max_episode_seconds=max_seconds
        )
        try:
            agents = _load_agents(checkpoint, device) if policy == "dqn" else {}
            observations, reset_info = env.reset(seed=seed)
            masks = reset_info["action_masks"]
            queue_sum = 0.0
            decisions = 0
            terminated = truncated = False
            while not (terminated or truncated):
                actions = _actions(policy, env, observations, masks, agents)
                observations, _, terminated, truncated, info = env.step(actions)
                masks = info["action_masks"]
                queue_sum += info["mean_queue"]
                decisions += 1

            return {
                "policy": policy,
                "seed": seed,
                "average_travel_time": float(
                    env.engine.get_average_travel_time()
                ),
                "mean_queue": queue_sum / max(decisions, 1),
                "throughput": int(info["throughput"]),
                "simulation_seconds": int(info["simulation_seconds"]),
            }
        finally:
            env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max-seconds", type=int, default=1800)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "results"
    )
    parser.add_argument(
        "--inference-device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    args = parser.parse_args()
    if args.episodes < 1 or args.max_seconds <= 0:
        parser.error("--episodes and --max-seconds must be positive.")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(
            f"DQN checkpoint not found: {args.checkpoint}. Run train.py first."
        )
    if not args.config.is_file():
        raise FileNotFoundError(
            f"CityFlow config not found: {args.config}. Run generate_scenario.py first."
        )
    if args.inference_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA inference was requested, but CUDA is unavailable.")
    device = (
        torch.device("cuda")
        if args.inference_device == "cuda"
        or (args.inference_device == "auto" and torch.cuda.is_available())
        else torch.device("cpu")
    )

    results = [
        _run_policy(
            policy,
            args.config.resolve(),
            args.checkpoint.resolve(),
            args.seed + episode,
            args.max_seconds,
            device,
        )
        for policy in POLICIES
        for episode in range(args.episodes)
    ]
    summary = {}
    for policy in POLICIES:
        policy_results = [row for row in results if row["policy"] == policy]
        summary[policy] = {
            metric: {
                "mean": float(np.mean([row[metric] for row in policy_results])),
                "std": float(np.std([row[metric] for row in policy_results])),
            }
            for metric in ("average_travel_time", "mean_queue", "throughput")
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.output_dir / "evaluation.json"
    results_path.write_text(
        json.dumps({"runs": results, "summary": summary}, indent=2),
        encoding="utf-8",
    )
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    labels = [policy.replace("-", "\n").title() for policy in POLICIES]
    for axis, metric, title, ylabel in (
        (
            axes[0],
            "average_travel_time",
            "Average Travel Time",
            "Seconds",
        ),
        (axes[1], "mean_queue", "Mean Queue Length", "Vehicles"),
    ):
        means = [summary[policy][metric]["mean"] for policy in POLICIES]
        deviations = [summary[policy][metric]["std"] for policy in POLICIES]
        axis.bar(labels, means, yerr=deviations, capsize=5)
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle(f"CityFlow corridor benchmark ({args.episodes} seeds)")
    figure.tight_layout()
    plot_path = args.output_dir / "evaluation.png"
    figure.savefig(plot_path, dpi=160)
    plt.close(figure)
    print(json.dumps(summary, indent=2))
    print(f"Saved benchmark data to {results_path.resolve()}")
    print(f"Saved benchmark plots to {plot_path.resolve()}")


if __name__ == "__main__":
    main()
