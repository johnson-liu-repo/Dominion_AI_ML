"""Training IO utilities for logging episodes, turns, and checkpoints."""

import csv
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import torch


# Consistent CSV schema for per-episode summaries.
EPISODE_CSV_HEADER = [
    "episode",
    "global_episode",
    "source_episode",
    "steps",
    "reward",
    "score_diff",
    "epsilon",
    "timestamp",
]

WEIGHTS_INDEX_HEADER = [
    "episode",
    "global_episode",
    "source_episode",
    "checkpoint",
]


def _read_existing_header(path: Path) -> Optional[List[str]]:
    if not path.exists():
        return None
    with path.open("r", newline="") as f:
        reader = csv.reader(f)
        return next(reader, None) or None

def _tiered_episode_dir(root: Path, prefix: str, episode_idx: int) -> Path:
    """
    Return a tiered directory for artifacts keyed by episode number.

    Layout:
      root/
        {prefix}{000000,100000,200000,...}/
          {prefix}{010000,020000,...,090000}/

    Episodes in the first 10k of a 100k block live directly under the 100k folder.
    """
    episode_idx = int(episode_idx)
    bucket_100k = (episode_idx // 100_000) * 100_000
    bucket_10k = (episode_idx // 10_000) * 10_000

    top = root / f"{prefix}{bucket_100k:06d}"
    if bucket_10k == bucket_100k:
        return top
    return top / f"{prefix}{bucket_10k:06d}"


def _safe_int_suffix(name: str, prefix: str) -> Optional[int]:
    """Return the numeric suffix if `name` matches `prefix` + digits, else None."""
    if not name.startswith(prefix):
        return None
    suffix = name[len(prefix):]
    return int(suffix) if suffix.isdigit() else None


def infer_source_run_dir(path: Optional[Path]) -> Optional[Path]:
    """Infer the training run directory from a checkpoint or run path."""
    if path is None:
        return None

    candidate = Path(path)
    if candidate.is_file():
        for parent in candidate.parents:
            if parent.name == "checkpoints":
                return parent.parent
        return candidate.parent
    if candidate.is_dir():
        if candidate.name == "checkpoints":
            return candidate.parent
        return candidate
    return None


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
    force_new_run: bool = False,
) -> Path:
    """
    Decide which run directory to use for training outputs.

    Priority:
    1) `run_dir` if provided (created if missing),
    2) `resume_from` (file or dir) when not forcing a new run,
    3) auto-create next `training_XXX` folder inside `output_dir`.
    """
    if run_dir is not None:
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    if resume_from is not None and not force_new_run:
        inferred = infer_source_run_dir(resume_from)
        if inferred is not None:
            return inferred

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
        self.continuation_metadata_json = self.run_dir / "continuation_metadata.json"
        self.episodes_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)

        # Initialize the episode summary CSV if it does not exist yet.
        self.episode_csv = self.run_dir / "episode_data_over_time.csv"
        self.episode_csv_fieldnames = _read_existing_header(self.episode_csv) or EPISODE_CSV_HEADER
        if not self.episode_csv.exists():
            with self.episode_csv.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.episode_csv_fieldnames)
                writer.writeheader()

        # Track checkpoint paths per episode in a separate CSV.
        self.weights_index = self.run_dir / "model_weights_over_time.csv"
        self.weights_index_fieldnames = _read_existing_header(self.weights_index) or WEIGHTS_INDEX_HEADER
        if not self.weights_index.exists():
            with self.weights_index.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.weights_index_fieldnames)
                writer.writeheader()

        # Initialize the final decks JSON file if it does not exist yet.
        self.final_decks_json = self.run_dir / "final_decks.json"
        if not self.final_decks_json.exists():
            with self.final_decks_json.open("w", encoding="utf-8") as f:
                json.dump([], f, ensure_ascii=True, indent=2)

    def log_episode(self, row: Dict[str, Any]) -> None:
        """Append a single episode summary row to the CSV."""
        with self.episode_csv.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.episode_csv_fieldnames)
            writer.writerow({key: row.get(key, "") for key in self.episode_csv_fieldnames})

    def write_continuation_metadata(self, metadata: Dict[str, Any]) -> Path:
        """Write continuation provenance metadata for forked runs."""
        payload = dict(metadata)
        payload.setdefault("created_at", time.time())
        with self.continuation_metadata_json.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=True, indent=2)
        return self.continuation_metadata_json

    def write_turns(self, episode_idx: int, events: List[Dict[str, Any]]) -> Path:
        """Write a JSONL file containing per-turn events for an episode."""
        turns_dir = _tiered_episode_dir(self.episodes_dir, "episode_ep_", episode_idx)
        turns_dir.mkdir(parents=True, exist_ok=True)
        path = turns_dir / f"episode_{episode_idx:06d}_turns.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for event in events:
                f.write(json.dumps(event, ensure_ascii=True) + "\n")
        return path

    def save_checkpoint(self, payload: Dict[str, Any], name: str) -> Path:
        """Save a torch checkpoint payload under the checkpoints directory."""
        episode_idx = _safe_int_suffix(name, "checkpoint_ep_")
        if episode_idx is not None:
            ckpt_dir = _tiered_episode_dir(self.checkpoints_dir, "checkpoint_ep_", episode_idx)
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            path = ckpt_dir / f"{name}.pt"
        else:
            path = self.checkpoints_dir / f"{name}.pt"
        torch.save(payload, path)
        return path

    def log_weights_checkpoint(
        self,
        episode_idx: int,
        checkpoint_path: Path,
        *,
        global_episode: Optional[int] = None,
        source_episode: Optional[int] = None,
    ) -> None:
        """Append the checkpoint path used for a given episode."""
        with self.weights_index.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.weights_index_fieldnames)
            row = {
                "episode": episode_idx,
                "global_episode": "" if global_episode is None else int(global_episode),
                "source_episode": "" if source_episode is None else int(source_episode),
                "checkpoint": str(checkpoint_path),
            }
            writer.writerow({key: row.get(key, "") for key in self.weights_index_fieldnames})

    def log_final_decks(
        self,
        episode_idx: int,
        players: Iterable[Any],
        *,
        global_episode: Optional[int] = None,
        source_episode: Optional[int] = None,
    ) -> Path:
        """Append final deck contents for all players to final_decks.json."""
        entry = {
            "episode": int(episode_idx),
            "global_episode": None if global_episode is None else int(global_episode),
            "source_episode": None if source_episode is None else int(source_episode),
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
