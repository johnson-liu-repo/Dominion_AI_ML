#!/usr/bin/env python
"""
Plot moving-average final deck card counts for RL_Agent vs big_money.

Reads a `final_decks.json` produced during training (see `src/agent_rl/training_io.py`)
and plots a moving average (default: 10 episodes) for each card in the final decks.

Examples:
  python scripts/plot_final_deck_card_trends.py data/training/training_006/final_decks.json
  python scripts/plot_final_deck_card_trends.py data/training/training_006 --out plots/final_decks_ma10.png
  python scripts/plot_final_deck_card_trends.py 6 --window 25 --show
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D


def _safe_int_suffix(name: str, prefix: str) -> int | None:
    if not name.startswith(prefix):
        return None
    suffix = name[len(prefix):]
    return int(suffix) if suffix.isdigit() else None


def _resolve_final_decks_path(arg: str | None, training_root: Path) -> Path:
    """
    Resolve a user-provided argument to a `final_decks.json` path.

    Supported forms:
      - None: pick latest `training_###` in `training_root`
      - integer string: treat as training run id (e.g. "6" -> training_006)
      - path to a run directory (contains final_decks.json)
      - path to final_decks.json directly
      - "training_###" (resolved under training_root if it exists)
    """
    if arg is None:
        candidates: list[tuple[int, Path]] = []
        if training_root.exists():
            for path in training_root.iterdir():
                if not path.is_dir():
                    continue
                suffix = _safe_int_suffix(path.name, "training_")
                if suffix is None:
                    continue
                final_decks = path / "final_decks.json"
                if final_decks.exists():
                    candidates.append((suffix, final_decks))
        if not candidates:
            raise FileNotFoundError(
                f"No final_decks.json found under: {training_root} (expected training_### folders)"
            )
        candidates.sort(key=lambda item: item[0])
        return candidates[-1][1]

    if arg.isdigit():
        run_dir = training_root / f"training_{int(arg):03d}"
        path = run_dir / "final_decks.json"
        if not path.exists():
            raise FileNotFoundError(f"final_decks.json not found for run id {arg}: {path}")
        return path

    raw = Path(arg)
    if not raw.exists():
        # Allow "training_###" as shorthand under training_root.
        candidate = training_root / arg
        if candidate.exists():
            raw = candidate

    if raw.is_dir():
        path = raw / "final_decks.json"
        if not path.exists():
            raise FileNotFoundError(f"final_decks.json not found in directory: {raw}")
        return path

    if raw.is_file():
        return raw

    raise FileNotFoundError(f"Path not found: {arg}")


def _load_final_decks(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        loaded = json.load(f)
    if not isinstance(loaded, list):
        raise ValueError(f"Expected a JSON array at top-level: {path}")
    return loaded


def _extract_episode_player_cards(
    data: list[dict[str, Any]],
    players_filter: set[str] | None,
) -> tuple[list[int], dict[int, dict[str, dict[str, int]]], set[str], set[str]]:
    """
    Return:
      - sorted episode ids
      - episode -> player_id -> card_name -> count
      - card names observed (post-filter)
      - player ids observed (post-filter)
    """
    episode_map: dict[int, dict[str, dict[str, int]]] = {}
    cards_seen: set[str] = set()
    players_seen: set[str] = set()

    for entry in data:
        episode_raw = entry.get("episode")
        try:
            episode = int(episode_raw)
        except Exception:
            continue

        players = entry.get("players", [])
        if not isinstance(players, list):
            continue

        ep_players = episode_map.setdefault(episode, {})
        for player in players:
            if not isinstance(player, dict):
                continue
            player_id = str(player.get("player_id", "")).strip()
            if not player_id:
                continue
            if players_filter is not None and player_id not in players_filter:
                continue

            cards_obj = player.get("cards", {}) or {}
            if not isinstance(cards_obj, dict):
                continue

            counts: dict[str, int] = {}
            for k, v in cards_obj.items():
                if v is None:
                    continue
                try:
                    counts[str(k)] = int(v)
                except Exception:
                    continue

            ep_players[player_id] = counts
            players_seen.add(player_id)
            cards_seen.update(counts.keys())

    episodes = sorted(episode_map.keys())
    return episodes, episode_map, cards_seen, players_seen


def _counts_dataframe(
    episodes: list[int],
    episode_map: dict[int, dict[str, dict[str, int]]],
    player_id: str,
    cards: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, int]] = []
    for episode in episodes:
        counts = episode_map.get(episode, {}).get(player_id, {})
        rows.append({card: int(counts.get(card, 0)) for card in cards})

    df = pd.DataFrame(rows, index=episodes, columns=cards)
    df.index.name = "episode"
    return df


def _pick_cards(
    cards_seen: set[str],
    episode_map: dict[int, dict[str, dict[str, int]]],
    players: list[str],
    max_cards: int | None,
) -> list[str]:
    cards = sorted(cards_seen)
    if max_cards is None or max_cards <= 0 or len(cards) <= max_cards:
        return cards

    totals: Counter[str] = Counter()
    for ep_players in episode_map.values():
        for player_id in players:
            for card, count in (ep_players.get(player_id, {}) or {}).items():
                try:
                    totals[str(card)] += int(count)
                except Exception:
                    continue

    ranked = [card for card, _ in totals.most_common()]
    # Keep deterministic ordering for ties/unseen cards.
    ranked_set = set(ranked)
    ranked.extend([c for c in cards if c not in ranked_set])
    return ranked[:max_cards]


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    training_root = repo_root / "data" / "training"

    parser = argparse.ArgumentParser(
        description=(
            "Plot moving-average final deck card counts for RL_Agent vs big_money from final_decks.json."
        )
    )
    parser.add_argument(
        "path",
        nargs="?",
        default=None,
        help=(
            "Path to final_decks.json, a training_### run directory, a run id (e.g. 6), "
            "or omit to use the latest run under data/training."
        ),
    )
    parser.add_argument(
        "--window",
        type=int,
        default=10,
        help="Moving average window in episodes (default: 10).",
    )
    parser.add_argument(
        "--players",
        nargs="+",
        default=["RL_Agent", "big_money"],
        help="Player ids to plot (default: RL_Agent big_money).",
    )
    parser.add_argument(
        "--max-cards",
        type=int,
        default=None,
        help="Optional: only plot the top N cards by total count across selected players.",
    )
    parser.add_argument(
        "--cols",
        type=int,
        default=4,
        help="Number of subplot columns (default: 4).",
    )
    parser.add_argument("--out", default=None, help="Optional output image path.")
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show the plot window (useful when --out is also provided).",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Optional plot title. Default is based on the run directory name.",
    )
    args = parser.parse_args()

    final_decks_path = _resolve_final_decks_path(args.path, training_root)
    data = _load_final_decks(final_decks_path)

    requested_players = [str(p) for p in args.players]
    episodes, episode_map, cards_seen, players_seen = _extract_episode_player_cards(
        data, players_filter=set(requested_players)
    )

    if not episodes:
        raise ValueError(f"No episodes found in: {final_decks_path}")

    missing_players = [p for p in requested_players if p not in players_seen]
    if missing_players:
        print(
            f"Warning: players not found in {final_decks_path.name}: {', '.join(missing_players)}",
            file=sys.stderr,
        )

    players_to_plot = [p for p in requested_players if p in players_seen]
    if not players_to_plot:
        raise ValueError(
            f"None of the requested players were found. Requested: {requested_players}. Found: {sorted(players_seen)}"
        )

    cards = _pick_cards(cards_seen, episode_map, players_to_plot, max_cards=args.max_cards)
    if not cards:
        raise ValueError(f"No cards found for players {players_to_plot} in: {final_decks_path}")

    ma_by_player: dict[str, pd.DataFrame] = {}
    for player_id in players_to_plot:
        df = _counts_dataframe(episodes, episode_map, player_id, cards)
        ma_by_player[player_id] = df.rolling(window=max(1, int(args.window)), min_periods=1).mean()

    cols = max(1, int(args.cols))
    cols = min(cols, len(cards))
    rows = int(math.ceil(len(cards) / cols))

    # Add a bit of width for an external legend.
    fig, axes = plt.subplots(
        nrows=rows,
        ncols=cols,
        figsize=(cols * 4.0 + 2.5, rows * 2.8),
        sharex=True,
        sharey=False,
    )
    if isinstance(axes, plt.Axes):
        axes_list = [axes]
    else:
        axes_list = list(axes.flatten())

    cycle = plt.rcParams.get("axes.prop_cycle", None)
    colors = (cycle.by_key().get("color", []) if cycle is not None else []) or ["C0", "C1", "C2", "C3"]
    player_colors = {p: colors[i % len(colors)] for i, p in enumerate(players_to_plot)}

    for i, card in enumerate(cards):
        ax = axes_list[i]
        for player_id in players_to_plot:
            ax.plot(
                episodes,
                ma_by_player[player_id][card],
                label=player_id,
                color=player_colors[player_id],
                linewidth=1.5,
            )
        ax.set_title(card, fontsize=10)
        ax.grid(True, alpha=0.3)

    for ax in axes_list[len(cards):]:
        ax.axis("off")

    run_name = final_decks_path.parent.name
    title = args.title or f"Final Deck Card Moving Averages (ma{args.window}) - {run_name}"
    fig.suptitle(title)
    fig.supxlabel("Episode")
    fig.supylabel("Final deck card count (moving average)")

    legend_handles = [
        Line2D([0], [0], color=player_colors[p], linewidth=2.0, label=p) for p in players_to_plot
    ]
    fig.legend(
        handles=legend_handles,
        loc="center left",
        bbox_to_anchor=(1.0, 0.5),
        borderaxespad=0.0,
    )

    fig.tight_layout(rect=[0.0, 0.0, 0.86, 0.95])

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")

    if args.show or not args.out:
        plt.show()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

