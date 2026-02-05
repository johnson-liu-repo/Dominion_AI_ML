"""Training IO utilities for logging episodes, turns, and checkpoints."""

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import torch


# Consistent CSV schema for per-episode summaries.
EPISODE_CSV_HEADER = [
    "episode",
    "steps",
    "reward",
    "score_diff",
    "epsilon",
    "timestamp",
]


def _safe_int_suffix(name: str, prefix: str) -> Optional[int]:
    """Return the numeric suffix if `name` matches `prefix` + digits, else None."""
    if not name.startswith(prefix):
        return None
    suffix = name[len(prefix):]
    return int(suffix) if suffix.isdigit() else None


def _count_card_names(cards: Iterable[Any]) -> Dict[str, int]:
    counts: Counter = Counter()
    for card in cards:
        name = getattr(card, "name", None)
        if name is None:
            name = str(card)
        counts[str(name)] += 1
    return dict(sorted(counts.items(), key=lambda item: item[0]))


def resolve_run_dir(
    output_dir: Path,
    run_dir: Optional[Path] = None,
    resume_from: Optional[Path] = None,
) -> Path:
    """
    Decide which run directory to use for training outputs.

    Priority:
    1) `run_dir` if provided (created if missing),
    2) `resume_from` (file or dir),
    3) auto-create next `training_XXX` folder inside `output_dir`.
    """
    if run_dir is not None:
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    if resume_from is not None:
        resume_from = Path(resume_from)
        if resume_from.is_file():
            if resume_from.parent.name == "checkpoints":
                return resume_from.parent.parent
            return resume_from.parent
        if resume_from.is_dir():
            return resume_from

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find existing `training_###` folders to pick the next id.
    existing = []
    for path in output_dir.iterdir():
        if not path.is_dir():
            continue
        suffix = _safe_int_suffix(path.name, "training_")
        if suffix is not None:
            existing.append(suffix)
    next_id = (max(existing) + 1) if existing else 1
    run_dir = output_dir / f"training_{next_id:03d}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


class TrainingRunWriter:
    """Write training artifacts (CSVs, per-episode JSONL, checkpoints) for a run."""

    def __init__(self, run_dir: Path):
        """Prepare run folders and ensure CSV headers exist."""
        self.run_dir = Path(run_dir)
        self.episodes_dir = self.run_dir / "episodes"
        self.checkpoints_dir = self.run_dir / "checkpoints"
        self.episodes_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)

        # Initialize the episode summary CSV if it does not exist yet.
        self.episode_csv = self.run_dir / "episode_data_over_time.csv"
        if not self.episode_csv.exists():
            with self.episode_csv.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=EPISODE_CSV_HEADER)
                writer.writeheader()

        # Track checkpoint paths per episode in a separate CSV.
        self.weights_index = self.run_dir / "model_weights_over_time.csv"
        if not self.weights_index.exists():
            with self.weights_index.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["episode", "checkpoint"])
                writer.writeheader()

        # Initialize the final decks JSON file if it does not exist yet.
        self.final_decks_json = self.run_dir / "final_decks.json"
        if not self.final_decks_json.exists():
            with self.final_decks_json.open("w", encoding="utf-8") as f:
                json.dump([], f, ensure_ascii=True, indent=2)

    def log_episode(self, row: Dict[str, Any]) -> None:
        """Append a single episode summary row to the CSV."""
        with self.episode_csv.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=EPISODE_CSV_HEADER)
            writer.writerow(row)

    def write_turns(self, episode_idx: int, events: List[Dict[str, Any]]) -> Path:
        """Write a JSONL file containing per-turn events for an episode."""
        path = self.episodes_dir / f"episode_{episode_idx:06d}_turns.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for event in events:
                f.write(json.dumps(event, ensure_ascii=True) + "\n")
        return path

    def save_checkpoint(self, payload: Dict[str, Any], name: str) -> Path:
        """Save a torch checkpoint payload under the checkpoints directory."""
        path = self.checkpoints_dir / f"{name}.pt"
        torch.save(payload, path)
        return path

    def log_weights_checkpoint(self, episode_idx: int, checkpoint_path: Path) -> None:
        """Append the checkpoint path used for a given episode."""
        with self.weights_index.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["episode", "checkpoint"])
            writer.writerow({"episode": episode_idx, "checkpoint": str(checkpoint_path)})

    def log_final_decks(self, episode_idx: int, players: Iterable[Any]) -> Path:
        """Append final deck contents for all players to final_decks.json."""
        entry = {
            "episode": int(episode_idx),
            "players": [],
        }
        for idx, player in enumerate(players):
            player_id = getattr(player, "player_id", "") or f"player_{idx}"
            cards_iter = player.get_all_cards() if hasattr(player, "get_all_cards") else []
            counts = _count_card_names(cards_iter)
            entry["players"].append({
                "player_id": player_id,
                "total_cards": int(sum(counts.values())),
                "cards": counts,
            })

        data: List[Dict[str, Any]] = []
        if self.final_decks_json.exists():
            try:
                with self.final_decks_json.open("r", encoding="utf-8") as f:
                    loaded = json.load(f)
                if isinstance(loaded, list):
                    data = loaded
            except json.JSONDecodeError:
                data = []

        data.append(entry)
        with self.final_decks_json.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=True, indent=2)
        return self.final_decks_json


def load_checkpoint(
    path: Path,
    policy_net: torch.nn.Module,
    target_net: Optional[torch.nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Load a training checkpoint and restore network/optimizer state.

    Falls back to `policy_state` for `target_net` if `target_state` is absent.
    """
    payload = torch.load(path, map_location=device or "cpu")
    policy_net.load_state_dict(payload["policy_state"])
    if target_net is not None:
        target_state = payload.get("target_state", payload["policy_state"])
        target_net.load_state_dict(target_state)
    if optimizer is not None and "opt_state" in payload:
        optimizer.load_state_dict(payload["opt_state"])
    return payload
