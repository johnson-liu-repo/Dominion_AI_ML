"""Replay normalization and frame navigation helpers for diagnostics tooling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional


SPECIAL_BUYS = {"PASS", "ILLEGAL"}


@dataclass(frozen=True)
class ReplayEvent:
    index: int
    turn: int
    player_id: Optional[str]
    buys: List[str]
    readable_description: str
    phase: Optional[str] = None
    reward: Optional[float] = None
    legal_actions: Optional[List[str]] = None
    state_diff: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class ReplayFrame:
    frame_index: int
    event: Optional[ReplayEvent]
    turn: int


@dataclass
class ReplayModel:
    events: List[ReplayEvent]
    frames: List[ReplayFrame]

    @property
    def frame_count(self) -> int:
        return len(self.frames)

    def frame_at(self, idx: int) -> ReplayFrame:
        idx = max(0, min(idx, len(self.frames) - 1))
        return self.frames[idx]

    def next_index(self, idx: int) -> int:
        return min(idx + 1, len(self.frames) - 1)

    def prev_index(self, idx: int) -> int:
        return max(idx - 1, 0)

    def first_index_for_turn(self, turn: int) -> int:
        for frame in self.frames:
            if frame.turn == turn and frame.event is not None:
                return frame.frame_index
        return 0


def _as_list(v: Any) -> List[Any]:
    return list(v) if isinstance(v, list) else []


def _describe(raw: Dict[str, Any]) -> str:
    if raw.get("readable_description"):
        return str(raw["readable_description"])
    who = raw.get("player") or raw.get("player_id") or "Unknown"
    buys = [b for b in _as_list(raw.get("buys")) if b not in SPECIAL_BUYS]
    if buys:
        return f"{who} bought {', '.join(str(b) for b in buys)}"
    if "PASS" in _as_list(raw.get("buys")):
        return f"{who} passed"
    if "ILLEGAL" in _as_list(raw.get("buys")):
        return f"{who} illegal action"
    return f"{who} action"


def normalize_events(raw_events: Iterable[Dict[str, Any]]) -> List[ReplayEvent]:
    normalized: List[ReplayEvent] = []
    for i, raw in enumerate(raw_events):
        try:
            turn = int(raw.get("turn"))
        except (TypeError, ValueError):
            continue
        idx = int(raw.get("index", i))
        legal = raw.get("legal_actions")
        legal_actions = list(legal) if isinstance(legal, list) else None
        reward_val = raw.get("reward")
        reward = float(reward_val) if isinstance(reward_val, (int, float)) else None
        state_diff = raw.get("state_diff") if isinstance(raw.get("state_diff"), dict) else None
        normalized.append(
            ReplayEvent(
                index=idx,
                turn=turn,
                player_id=raw.get("player_id") or raw.get("player"),
                buys=[str(b) for b in _as_list(raw.get("buys"))],
                readable_description=_describe(raw),
                phase=raw.get("phase") if isinstance(raw.get("phase"), str) else None,
                reward=reward,
                legal_actions=legal_actions,
                state_diff=state_diff,
            )
        )
    normalized.sort(key=lambda e: (e.turn, e.index))
    return normalized


def build_replay_model(raw_events: Iterable[Dict[str, Any]]) -> ReplayModel:
    events = normalize_events(raw_events)
    frames = [ReplayFrame(frame_index=0, event=None, turn=0)]
    for i, ev in enumerate(events, start=1):
        frames.append(ReplayFrame(frame_index=i, event=ev, turn=ev.turn))
    return ReplayModel(events=events, frames=frames)
