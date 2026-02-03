"""Lightweight Gym wrappers for specific Dominion phases."""

from src.agent_rl.dominion_env import DominionBuyPhaseEnv


class BuyPhaseEnv(DominionBuyPhaseEnv):
    """Alias for clarity – no behaviour changes yet."""
    pass

# TODO: ActionPhaseEnv once we start training the action phase.
