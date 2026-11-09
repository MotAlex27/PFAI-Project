"""Independent DQN agent with masked actions and a bounded replay buffer."""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    next_action_mask: np.ndarray
    done: bool


class ReplayBuffer:
    def __init__(self, capacity: int, seed: int) -> None:
        if capacity <= 0:
            raise ValueError("Replay-buffer capacity must be positive.")
        self._items: deque[Transition] = deque(maxlen=capacity)
        self._random = random.Random(seed)

    def __len__(self) -> int:
        return len(self._items)

    def add(self, transition: Transition) -> None:
        self._items.append(transition)

    def sample(self, size: int) -> list[Transition]:
        if size > len(self._items):
            raise ValueError("Cannot sample more transitions than the buffer contains.")
        return self._random.sample(self._items, size)


class QNetwork(nn.Module):
    def __init__(self, observation_size: int, action_count: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(observation_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_count),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.layers(observations)


class DQNAgent:
    def __init__(
        self,
        observation_size: int,
        action_count: int = 2,
        *,
        device: torch.device | str,
        seed: int,
        learning_rate: float = 5e-4,
        gamma: float = 0.99,
        buffer_capacity: int = 50_000,
        batch_size: int = 128,
        target_update_steps: int = 500,
    ) -> None:
        if observation_size <= 0 or action_count <= 0:
            raise ValueError("Observation and action dimensions must be positive.")
        if batch_size <= 0 or target_update_steps <= 0:
            raise ValueError("Batch size and target update interval must be positive.")
        self.device = torch.device(device)
        self.action_count = action_count
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_steps = target_update_steps
        self.online = QNetwork(observation_size, action_count).to(self.device)
        self.target = QNetwork(observation_size, action_count).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        self.optimizer = torch.optim.Adam(
            self.online.parameters(), lr=learning_rate
        )
        self.replay = ReplayBuffer(buffer_capacity, seed)
        self._random = random.Random(seed)
        self.optimization_steps = 0

    @torch.no_grad()
    def act(
        self,
        observation: np.ndarray,
        action_mask: np.ndarray,
        epsilon: float = 0.0,
    ) -> int:
        legal = np.flatnonzero(action_mask)
        if legal.size == 0:
            raise ValueError("At least one action must be legal.")
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("Epsilon must be between zero and one.")
        if self._random.random() < epsilon:
            return int(self._random.choice(legal.tolist()))
        tensor = torch.as_tensor(
            observation, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        q_values = self.online(tensor).squeeze(0)
        mask = torch.as_tensor(action_mask, dtype=torch.bool, device=self.device)
        return int(q_values.masked_fill(~mask, -torch.inf).argmax().item())

    def remember(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        next_action_mask: np.ndarray,
        done: bool,
    ) -> None:
        self.replay.add(
            Transition(
                state=np.array(state, dtype=np.float32, copy=True),
                action=int(action),
                reward=float(reward),
                next_state=np.array(next_state, dtype=np.float32, copy=True),
                next_action_mask=np.array(next_action_mask, dtype=np.bool_, copy=True),
                done=bool(done),
            )
        )

    def optimize(self) -> float | None:
        if len(self.replay) < self.batch_size:
            return None

        transitions = self.replay.sample(self.batch_size)
        states = torch.as_tensor(
            np.stack([item.state for item in transitions]),
            dtype=torch.float32,
            device=self.device,
        )
        actions = torch.as_tensor(
            [item.action for item in transitions],
            dtype=torch.int64,
            device=self.device,
        ).unsqueeze(1)
        rewards = torch.as_tensor(
            [item.reward for item in transitions],
            dtype=torch.float32,
            device=self.device,
        )
        next_states = torch.as_tensor(
            np.stack([item.next_state for item in transitions]),
            dtype=torch.float32,
            device=self.device,
        )
        next_masks = torch.as_tensor(
            np.stack([item.next_action_mask for item in transitions]),
            dtype=torch.bool,
            device=self.device,
        )
        dones = torch.as_tensor(
            [item.done for item in transitions],
            dtype=torch.float32,
            device=self.device,
        )

        predicted = self.online(states).gather(1, actions).squeeze(1)
        with torch.no_grad():
            next_values = self.target(next_states).masked_fill(
                ~next_masks, -torch.inf
            )
            max_next = next_values.max(dim=1).values
            max_next = torch.where(dones.bool(), 0.0, max_next)
            targets = rewards + self.gamma * max_next

        loss = nn.functional.smooth_l1_loss(predicted, targets)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), max_norm=10.0)
        self.optimizer.step()
        self.optimization_steps += 1
        if self.optimization_steps % self.target_update_steps == 0:
            self.target.load_state_dict(self.online.state_dict())
        return float(loss.item())

    def state_dict(self) -> dict:
        return {
            "online": self.online.state_dict(),
            "target": self.target.state_dict(),
        }

    def load_state_dict(self, state: dict) -> None:
        self.online.load_state_dict(state["online"])
        self.target.load_state_dict(state.get("target", state["online"]))
        self.target.eval()
