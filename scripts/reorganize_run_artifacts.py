import argparse
import csv
import re
import shutil
from pathlib import Path


_CHECKPOINT_RE = re.compile(r"^checkpoint_ep_(\d+)\.pt$")
_EPISODE_TURNS_RE = re.compile(r"^episode_(\d+)_turns\.jsonl$")


DEFAULT_TIERS = (100_000, 10_000, 1_000, 100, 10)


def _tiered_dir(root: Path, prefix: str, episode_idx: int, *, tiers: tuple[int, ...]) -> Path:
    """
    Return a tiered directory for artifacts keyed by episode number.

    `tiers` are bucket sizes (largest -> smallest). Example:
      tiers=(100_000, 10_000, 1_000, 100, 10) yields:
        root/{prefix}000000/{prefix}000000/{prefix}001000/{prefix}001000/{prefix}001020/

    Buckets are *not* deduplicated: when a smaller tier falls in the same bucket as a larger tier,
    we still create the nested directory. This keeps the directory depth consistent and avoids
    mixing files and subfolders in the same tier directory.
    """
    episode_idx = int(episode_idx)
    buckets = []
    for tier in tiers:
        tier = int(tier)
        if tier <= 0:
            continue
        bucket = (episode_idx // tier) * tier
        buckets.append(bucket)

    path = root
    for bucket in buckets:
        path = path / f"{prefix}{bucket:06d}"
    return path


def _move_file(src: Path, dst: Path, dry_run: bool) -> bool:
    if src.resolve() == dst.resolve():
        return False
    if dst.exists():
        print(f"SKIP (exists): {dst}")
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dry_run:
        print(f"DRYRUN: move {src} -> {dst}")
        return True
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))
    print(f"MOVED: {src} -> {dst}")
    return True


def _merge_dir(src: Path, dst: Path, dry_run: bool) -> int:
    moved = 0
    if dry_run:
        print(f"DRYRUN: merge-dir {src} -> {dst}")
    dst.mkdir(parents=True, exist_ok=True)

    for child in list(src.iterdir()):
        child_dst = dst / child.name
        if child.is_file():
            if _move_file(child, child_dst, dry_run=dry_run):
                moved += 1
            continue
        if child.is_dir():
            if _move_dir(child, child_dst, dry_run=dry_run):
                moved += 1
            continue

    if not dry_run:
        try:
            next(src.iterdir())
        except StopIteration:
            src.rmdir()
            print(f"RMDIR: {src}")
    return moved


def _move_dir(src: Path, dst: Path, dry_run: bool) -> bool:
    if src.resolve() == dst.resolve():
        return False

    if dst.exists():
        return _merge_dir(src, dst, dry_run=dry_run) > 0

    dst.parent.mkdir(parents=True, exist_ok=True)
    if dry_run:
        print(f"DRYRUN: move-dir {src} -> {dst}")
        return True

    shutil.move(str(src), str(dst))
    print(f"MOVEDDIR: {src} -> {dst}")
    return True


def reorganize_tier_dirs(root: Path, prefix: str, *, tiers: tuple[int, ...], dry_run: bool) -> int:
    """
    Move existing tier directories into their correct location.

    This is useful when migrating between tier layouts (e.g., from a flat list of
    `{prefix}010000/` siblings into `{prefix}000000/{prefix}010000/`).
    """
    moved = 0
    dir_re = re.compile(rf"^{re.escape(prefix)}(\d+)$")

    candidates = []
    for path in root.rglob(f"{prefix}*"):
        if not path.is_dir():
            continue
        if dir_re.match(path.name):
            candidates.append(path)

    # Process deepest paths first so we can lift incorrectly nested buckets in one pass.
    candidates.sort(key=lambda p: len(p.parts), reverse=True)
    for path in candidates:
        if not path.exists() or not path.is_dir():
            continue
        match = dir_re.match(path.name)
        if not match:
            continue
        episode_idx = int(match.group(1))
        dst = _tiered_dir(root, prefix, episode_idx, tiers=tiers)
        if path.resolve() == dst.resolve():
            continue
        try:
            dst.relative_to(path)
            print(f"SKIP (dst inside src): {path} -> {dst}")
            continue
        except ValueError:
            pass
        if _move_dir(path, dst, dry_run=dry_run):
            moved += 1
    return moved


def prune_empty_tier_dirs(root: Path, prefix: str, dry_run: bool) -> int:
    """Remove empty tier directories under `root` (only dirs matching `{prefix}\\d+`)."""
    removed = 0
    dir_re = re.compile(rf"^{re.escape(prefix)}(\d+)$")
    candidates = []
    for path in root.rglob(f"{prefix}*"):
        if path.is_dir() and dir_re.match(path.name):
            candidates.append(path)

    candidates.sort(key=lambda p: len(p.parts), reverse=True)
    for path in candidates:
        if not path.exists() or not path.is_dir():
            continue
        try:
            next(path.iterdir())
            continue
        except StopIteration:
            if dry_run:
                print(f"DRYRUN: rmdir {path}")
            else:
                path.rmdir()
                print(f"RMDIR: {path}")
            removed += 1
    return removed


def reorganize_checkpoints(checkpoints_dir: Path, *, tiers: tuple[int, ...], dry_run: bool) -> int:
    moved = 0
    for path in checkpoints_dir.rglob("checkpoint_ep_*.pt"):
        if not path.is_file():
            continue
        match = _CHECKPOINT_RE.match(path.name)
        if not match:
            continue
        episode_idx = int(match.group(1))
        dst_dir = _tiered_dir(checkpoints_dir, "checkpoint_ep_", episode_idx, tiers=tiers)
        dst = dst_dir / path.name
        if _move_file(path, dst, dry_run=dry_run):
            moved += 1
    return moved


def reorganize_episodes(episodes_dir: Path, *, tiers: tuple[int, ...], dry_run: bool) -> int:
    moved = 0
    for path in episodes_dir.rglob("episode_*_turns.jsonl"):
        if not path.is_file():
            continue
        match = _EPISODE_TURNS_RE.match(path.name)
        if not match:
            continue
        episode_idx = int(match.group(1))
        dst_dir = _tiered_dir(episodes_dir, "episode_ep_", episode_idx, tiers=tiers)
        dst = dst_dir / path.name
        if _move_file(path, dst, dry_run=dry_run):
            moved += 1
    return moved


def rewrite_weights_index(run_dir: Path, *, tiers: tuple[int, ...], dry_run: bool) -> int:
    index_path = run_dir / "model_weights_over_time.csv"
    if not index_path.exists():
        return 0

    updated = 0
    rows = []
    with index_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return 0
        for row in reader:
            ckpt_val = (row.get("checkpoint") or "").strip()
            ckpt_name = Path(ckpt_val).name
            match = _CHECKPOINT_RE.match(ckpt_name)
            if match:
                episode_idx = int(match.group(1))
                dst_dir = _tiered_dir(run_dir / "checkpoints", "checkpoint_ep_", episode_idx, tiers=tiers)
                new_val = str((dst_dir / ckpt_name).resolve())
                if new_val != ckpt_val:
                    row["checkpoint"] = new_val
                    updated += 1
            rows.append(row)

    if updated and not dry_run:
        with index_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["episode", "checkpoint"])
            writer.writeheader()
            writer.writerows(rows)
    elif updated and dry_run:
        print(f"DRYRUN: would rewrite {index_path} with {updated} updated rows")
    return updated


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Reorganize a training run's checkpoints/ and episodes/ into tiered subdirectories."
    )
    parser.add_argument("run_dir", type=Path, help="Path to a run directory (e.g. data/training/training_007)")
    parser.add_argument(
        "--tiers",
        nargs="+",
        type=int,
        default=list(DEFAULT_TIERS),
        help="Tier bucket sizes (largest->smallest). Default: 100000 10000 1000 100 10",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print actions without modifying files")
    parser.add_argument(
        "--prune-empty",
        action="store_true",
        help="Remove empty tier dirs after moving files/directories",
    )
    args = parser.parse_args()

    run_dir = args.run_dir
    checkpoints_dir = run_dir / "checkpoints"
    episodes_dir = run_dir / "episodes"

    if not run_dir.exists():
        raise SystemExit(f"Run dir not found: {run_dir}")
    if not checkpoints_dir.exists():
        raise SystemExit(f"Missing checkpoints dir: {checkpoints_dir}")
    if not episodes_dir.exists():
        raise SystemExit(f"Missing episodes dir: {episodes_dir}")

    tiers = tuple(sorted({int(v) for v in args.tiers if int(v) > 0}, reverse=True)) or DEFAULT_TIERS

    moved_ckpt_dirs = reorganize_tier_dirs(
        checkpoints_dir,
        "checkpoint_ep_",
        tiers=tiers,
        dry_run=args.dry_run,
    )
    moved_ep_dirs = reorganize_tier_dirs(
        episodes_dir,
        "episode_ep_",
        tiers=tiers,
        dry_run=args.dry_run,
    )

    moved_ckpts = reorganize_checkpoints(checkpoints_dir, tiers=tiers, dry_run=args.dry_run)
    moved_eps = reorganize_episodes(episodes_dir, tiers=tiers, dry_run=args.dry_run)
    updated_index = rewrite_weights_index(run_dir, tiers=tiers, dry_run=args.dry_run)

    removed_dirs = 0
    if args.prune_empty:
        removed_dirs += prune_empty_tier_dirs(checkpoints_dir, "checkpoint_ep_", dry_run=args.dry_run)
        removed_dirs += prune_empty_tier_dirs(episodes_dir, "episode_ep_", dry_run=args.dry_run)

    print(
        "Done. "
        f"moved_checkpoint_dirs={moved_ckpt_dirs} moved_episode_dirs={moved_ep_dirs} "
        f"moved_checkpoints={moved_ckpts} moved_episodes={moved_eps} "
        f"updated_weights_index_rows={updated_index} removed_empty_tier_dirs={removed_dirs}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
