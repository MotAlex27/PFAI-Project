"""Gymnasium-style multi-agent wrapper around the CityFlow engine."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import cityflow
import gymnasium as gym
import numpy as np
from gymnasium import spaces

OBSERVATION_SIZE = 11
ACTION_COUNT = 2
SIGNAL_IDS = ("J1", "J2")
MIN_GREEN_SECONDS = 15
MAX_GREEN_SECONDS = 60
YELLOW_SECONDS = 3
DECISION_INTERVAL_SECONDS = 5
QUEUE_NORMALIZER = 20.0


class CorridorEnv(gym.Env):
    """Two-agent environment with shared arterial-neighbor queue features."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        config_path: str | Path,
        *,
        max_episode_seconds: int = 1800,
        decision_interval_seconds: int = DECISION_INTERVAL_SECONDS,
        replay: bool = False,
    ) -> None:
        super().__init__()
        if max_episode_seconds <= 0:
            raise ValueError("Episode duration must be positive.")
        if decision_interval_seconds <= 0:
            raise ValueError("Decision interval must be positive.")
        if MIN_GREEN_SECONDS % decision_interval_seconds:
            raise ValueError("Minimum green must be divisible by the decision interval.")
        if MAX_GREEN_SECONDS % decision_interval_seconds:
            raise ValueError("Maximum green must be divisible by the decision interval.")

        self.config_path = Path(config_path).resolve()
        if not self.config_path.is_file():
            raise FileNotFoundError(f"CityFlow config not found: {self.config_path}")
        self.config = json.loads(self.config_path.read_text(encoding="utf-8"))
        base_dir = Path(self.config["dir"])
        if not base_dir.is_absolute():
            base_dir = (self.config_path.parent / base_dir).resolve()
        self.base_dir = base_dir
        roadnet_path = base_dir / self.config["roadnetFile"]
        self.roadnet = json.loads(roadnet_path.read_text(encoding="utf-8"))
        self.max_episode_seconds = max_episode_seconds
        self.decision_interval_seconds = decision_interval_seconds
        self.min_green_seconds = MIN_GREEN_SECONDS
        self.max_green_seconds = MAX_GREEN_SECONDS
        self.yellow_seconds = YELLOW_SECONDS
        self.replay = replay
        self.action_space = spaces.Dict(
            {
                signal_id: spaces.Discrete(ACTION_COUNT)
                for signal_id in SIGNAL_IDS
            }
        )
        self.observation_space = spaces.Dict(
            {
                signal_id: spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(OBSERVATION_SIZE,),
                    dtype=np.float32,
                )
                for signal_id in SIGNAL_IDS
            }
        )
        self._index_roadnet()
        self.engine: Any = None
        self.simulation_seconds = 0
        self.phase = {signal_id: 0 for signal_id in SIGNAL_IDS}
        self.green_elapsed = {signal_id: 0 for signal_id in SIGNAL_IDS}
        self.previous_vehicle_ids: set[str] = set()
        self.total_throughput = 0
        self._config_temp_dir: tempfile.TemporaryDirectory | None = None
        self._make_engine()

    def _make_engine(self) -> None:
        if self.engine is not None:
            self.engine = None
        if self._config_temp_dir is not None:
            self._config_temp_dir.cleanup()
        self._config_temp_dir = tempfile.TemporaryDirectory(
            prefix="cityflow-corridor-"
        )
        engine_config = dict(self.config)
        engine_config["seed"] = int(self.config.get("seed", 0))
        engine_config["saveReplay"] = self.replay
        engine_config["dir"] = f"{self.base_dir}{os.sep}"
        config_file = Path(self._config_temp_dir.name) / "cityflow_config.json"
        config_file.write_text(json.dumps(engine_config), encoding="utf-8")
        self.engine = cityflow.Engine(str(config_file), thread_num=1)

    def _index_roadnet(self) -> None:
        intersections = {
            item["id"]: item
            for item in self.roadnet["intersections"]
            if item["id"] in SIGNAL_IDS and not item["virtual"]
        }
        if set(intersections) != set(SIGNAL_IDS):
            raise ValueError("Road network must contain signalized intersections J1 and J2.")
        self.intersections = intersections
        self.road_by_id = {road["id"]: road for road in self.roadnet["roads"]}
        self.lane_ids = {
            road["id"]: [
                f'{road["id"]}_{index}' for index in range(len(road["lanes"]))
            ]
            for road in self.roadnet["roads"]
        }
        self.incoming_roads = {
            signal_id: [
                road["id"]
                for road in self.roadnet["roads"]
                if road["endIntersection"] == signal_id
            ]
            for signal_id in SIGNAL_IDS
        }
        self.neighbor = {"J1": "J2", "J2": "J1"}
        self.phase_links: dict[str, list[list[dict]]] = {}

        for signal_id, intersection in intersections.items():
            phase_links = [[], []]
            for phase_index, phase in enumerate(
                intersection["trafficLight"]["lightphases"][:ACTION_COUNT]
            ):
                for link_index in phase["availableRoadLinks"]:
                    link = intersection["roadLinks"][link_index]
                    phase_links[phase_index].append(link)
            self.phase_links[signal_id] = phase_links
            if any(not links for links in phase_links):
                raise ValueError(f"{signal_id} has a signal phase with no movements.")

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None and seed != self.config.get("seed"):
            self.config["seed"] = int(seed)
            self._make_engine()
        self.engine.reset()
        self.simulation_seconds = 0
        self.phase = {signal_id: 0 for signal_id in SIGNAL_IDS}
        self.green_elapsed = {signal_id: 0 for signal_id in SIGNAL_IDS}
        for signal_id in SIGNAL_IDS:
            self.engine.set_tl_phase(signal_id, 0)
        self.previous_vehicle_ids = set(self.engine.get_vehicles())
        self.total_throughput = 0
        return self._observations(), {"action_masks": self.action_masks()}

    def _queue_for_road(
        self, road_id: str, counts: dict[str, int]
    ) -> float:
        return float(
            sum(counts.get(lane_id, 0) for lane_id in self.lane_ids[road_id])
        )

    def _approach_queues(self, signal_id: str, counts: dict[str, int]) -> list[float]:
        point = self.intersections[signal_id]["point"]
        queues = {"north": 0.0, "south": 0.0, "east": 0.0, "west": 0.0}
        for road_id in self.incoming_roads[signal_id]:
            start = self.road_by_id[road_id]["points"][0]
            if start["x"] == point["x"]:
                direction = "north" if start["y"] > point["y"] else "south"
            else:
                direction = "east" if start["x"] > point["x"] else "west"
            queues[direction] += self._queue_for_road(road_id, counts)
        return [queues[name] for name in ("north", "south", "east", "west")]

    def _observations(self) -> dict[str, np.ndarray]:
        waiting_counts = self.engine.get_lane_waiting_vehicle_count()
        local = {
            signal_id: self._approach_queues(signal_id, waiting_counts)
            for signal_id in SIGNAL_IDS
        }
        observations = {}
        for signal_id in SIGNAL_IDS:
            phase_encoding = [
                float(self.phase[signal_id] == phase_index)
                for phase_index in range(ACTION_COUNT)
            ]
            observations[signal_id] = np.asarray(
                [
                    *(
                        min(queue / QUEUE_NORMALIZER, 1.0)
                        for queue in local[signal_id]
                    ),
                    *(
                        min(queue / QUEUE_NORMALIZER, 1.0)
                        for queue in local[self.neighbor[signal_id]]
                    ),
                    *phase_encoding,
                    min(
                        self.green_elapsed[signal_id] / self.max_green_seconds,
                        1.0,
                    ),
                ],
                dtype=np.float32,
            )
        return observations

    def action_masks(self) -> dict[str, np.ndarray]:
        masks = {}
        for signal_id in SIGNAL_IDS:
            can_switch = self.green_elapsed[signal_id] >= self.min_green_seconds
            must_switch = self.green_elapsed[signal_id] >= self.max_green_seconds
            masks[signal_id] = np.asarray(
                [not must_switch, can_switch], dtype=np.bool_
            )
        return masks

    def _vehicle_ids(self) -> set[str]:
        return set(self.engine.get_vehicles(include_waiting=True))

    def _reward_and_info(
        self,
    ) -> tuple[dict[str, float], dict[str, Any]]:
        waiting_counts = self.engine.get_lane_waiting_vehicle_count()
        approach_queues = {
            signal_id: self._approach_queues(signal_id, waiting_counts)
            for signal_id in SIGNAL_IDS
        }
        current_vehicle_ids = self._vehicle_ids()
        departed = self.previous_vehicle_ids - current_vehicle_ids
        self.previous_vehicle_ids = current_vehicle_ids
        self.total_throughput += len(departed)

        rewards = {}
        for signal_id in SIGNAL_IDS:
            queues = approach_queues[signal_id]
            maximum_queue = max(queues, default=0.0) / QUEUE_NORMALIZER
            total_queue = sum(queues) / QUEUE_NORMALIZER
            rewards[signal_id] = (
                -(maximum_queue**2)
                - 0.05 * total_queue
                + 0.5 * len(departed) / len(SIGNAL_IDS)
            )
        info = {
            "approach_queues": approach_queues,
            "mean_queue": float(
                np.mean([sum(queues) for queues in approach_queues.values()])
            ),
            "throughput": self.total_throughput,
            "departed_this_step": len(departed),
            "average_travel_time": float(
                self.engine.get_average_travel_time()
            ),
            "simulation_seconds": self.simulation_seconds,
        }
        return rewards, info

    def phase_pressures(self, signal_id: str) -> list[float]:
        counts = self.engine.get_lane_vehicle_count()
        pressure_by_phase = []
        for links in self.phase_links[signal_id]:
            pressure = 0.0
            for link in links:
                upstream = self._queue_for_road(link["startRoad"], counts)
                downstream = self._queue_for_road(link["endRoad"], counts)
                pressure += upstream - downstream
            pressure_by_phase.append(pressure)
        return pressure_by_phase

    def step(
        self, actions: dict[str, int]
    ) -> tuple[
        dict[str, np.ndarray],
        dict[str, float],
        bool,
        bool,
        dict[str, Any],
    ]:
        if set(actions) != set(SIGNAL_IDS):
            raise ValueError(f"Actions must be supplied for exactly {SIGNAL_IDS}.")
        masks = self.action_masks()
        for signal_id, action in actions.items():
            if not isinstance(action, (int, np.integer)) or action not in (0, 1):
                raise ValueError(f"Invalid action for {signal_id}: {action!r}")
            if not masks[signal_id][action]:
                raise ValueError(
                    f"Action {action} for {signal_id} is masked at "
                    f"{self.green_elapsed[signal_id]}s green."
                )

        switched = []
        for signal_id, action in actions.items():
            if action == 1:
                self.engine.set_tl_phase(signal_id, 2)
                switched.append(signal_id)

        if switched:
            for _ in range(self.yellow_seconds):
                self.engine.next_step()
            self.simulation_seconds += self.yellow_seconds
            for signal_id in switched:
                self.phase[signal_id] = 1 - self.phase[signal_id]
                self.green_elapsed[signal_id] = 0
                self.engine.set_tl_phase(signal_id, self.phase[signal_id])
            for signal_id in SIGNAL_IDS:
                if signal_id not in switched:
                    self.green_elapsed[signal_id] += self.yellow_seconds

        for _ in range(self.decision_interval_seconds):
            self.engine.next_step()
        self.simulation_seconds += self.decision_interval_seconds
        for signal_id in SIGNAL_IDS:
            self.green_elapsed[signal_id] += self.decision_interval_seconds

        observations = self._observations()
        rewards, info = self._reward_and_info()
        terminated = False
        truncated = self.simulation_seconds >= self.max_episode_seconds
        info["action_masks"] = self.action_masks()
        return observations, rewards, terminated, truncated, info

    def close(self) -> None:
        self.engine = None
        if self._config_temp_dir is not None:
            self._config_temp_dir.cleanup()
            self._config_temp_dir = None
