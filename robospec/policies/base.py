"""Batch policy interface used by robospec.run."""
import numpy as np


class Policy:
    """One action row per environment; keep per-env state keyed by env index."""

    # Local runs stop after this many steps (None = run until the evaluator stops us).
    max_steps = None

    def reset(self, env_index: int) -> None:
        """Clear state for one environment (called at start and after each episode end)."""

    def act(self, observations: list[dict]) -> np.ndarray:
        """observations[i] is env i's decoded obs; return float32 (N, 19)."""
        raise NotImplementedError
