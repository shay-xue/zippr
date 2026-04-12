"""
copy_h5_from_manraj.py
----------------------
Recursively copies EVERY .h5 file tracked anywhere under data/recordings/
in the `origin/manraj` branch into the working-tree's data/recordings/,
resolving Git-LFS pointers from the local LFS object store.

Handles the nested layout on manraj:
  data/recordings/broadband_data_*.h5          <- top-level broadband files
  data/recordings/mapping/broadband_data_*.h5  <- mapping-run files
  data/recordings/train_NNN_<label>.h5/        <- per-session directories
      broadband_data_*.h5                          (dir name ends in .h5!)

All files are copied FLAT into dest_dir (data/recordings/) using just their
basename.  Filenames are unique (distinct timestamps) so no collisions occur.

Usage
-----
    python scripts/copy_h5_from_manraj.py             # live run
    python scripts/copy_h5_from_manraj.py --dry-run   # preview only
    python scripts/copy_h5_from_manraj.py --src-branch origin/manraj
    python scripts/copy_h5_from_manraj.py --dest-dir data/recordings
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


# ── git helpers ────────────────────────────────────────────────────────────────

def git_run(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git"] + list(args),
        cwd=REPO_ROOT,
        capture_output=True,
        check=check,
    )


def list_h5_paths(branch: str) -> list[str]:
    """Return every .h5 path tracked under data/recordings/ on *branch*."""
    result = git_run("ls-tree", "-r", "--name-only", branch)
    all_paths = result.stdout.decode().splitlines()
    return [p for p in all_paths if p.endswith(".h5")]


def read_blob_bytes(branch: str, path: str) -> bytes:
    """Return raw bytes of a blob at branch:path."""
    result = subprocess.run(
        ["git", "show", f"{branch}:{path}"],
        cwd=REPO_ROOT,
        capture_output=True,
        check=True,
    )
    return result.stdout


# ── LFS helpers ───────────────────────────────────────────────────────────────

def parse_lfs_oid(blob: bytes) -> str | None:
    """
    If *blob* is an LFS pointer text, return the sha256 OID.
    Returns None for real file content.
    """
    try:
        text = blob.decode("utf-8", errors="replace")
    except Exception:
        return None
    if not text.startswith("version https://git-lfs.github.com/spec/v1"):
        return None
    for line in text.splitlines():
        if line.startswith("oid sha256:"):
            return line.split(":", 1)[1].strip()
    return None


def find_lfs_object(oid: str) -> Path | None:
    """
    Locate a cached LFS object by its sha256 OID.
    Checks the main repo .git/lfs/objects and any worktree gitdir.
    """
    stores = [REPO_ROOT / ".git" / "lfs" / "objects"]

    gitfile = REPO_ROOT / ".git"
    if gitfile.is_file():
        line = gitfile.read_text().strip()
        if line.startswith("gitdir:"):
            worktree_gitdir = Path(line.split(":", 1)[1].strip())
            stores.append(worktree_gitdir / "lfs" / "objects")

    for store in stores:
        candidate = store / oid[:2] / oid[2:4] / oid
        if candidate.exists():
            return candidate
    return None


# ── main ──────────────────────────────────────────────────────────────────────

def copy_h5_files(
    src_branch: str = "origin/manraj",
    dest_dir: str = "data/recordings",
    dry_run: bool = False,
) -> None:
    dest = REPO_ROOT / dest_dir
    dest.mkdir(parents=True, exist_ok=True)

    all_h5 = list_h5_paths(src_branch)
    if not all_h5:
        print(f"No .h5 files found in branch '{src_branch}'.")
        return

    print(f"Found {len(all_h5)} .h5 file(s) in '{src_branch}':\n")

    # Group by category for cleaner output
    for p in all_h5:
        print(f"  {p}")
    print()

    copied = skipped = missing = 0

    for src_path in all_h5:
        filename = Path(src_path).name          # just the basename
        out_path = dest / filename

        blob = read_blob_bytes(src_branch, src_path)
        oid  = parse_lfs_oid(blob)

        if oid:
            lfs_src = find_lfs_object(oid)
            if lfs_src is None:
                print(f"  [MISSING LFS]  {src_path}  (oid={oid[:16]}...)")
                missing += 1
                continue

            src_size = lfs_src.stat().st_size
            # Skip if destination already exists and is the same size
            if out_path.exists() and out_path.stat().st_size == src_size:
                print(f"  [skip]   {filename}  ({src_size/1e6:.1f} MB, already up-to-date)")
                skipped += 1
                continue

            if dry_run:
                print(f"  [dry-run] {filename}  <- {src_path}  ({src_size/1e6:.1f} MB)")
            else:
                shutil.copy2(lfs_src, out_path)
                print(f"  [copied] {filename}  ({src_size/1e6:.1f} MB)  [LFS]")
            copied += 1

        else:
            # Actual blob content (rare for large .h5 files)
            src_size = len(blob)
            if out_path.exists() and out_path.stat().st_size == src_size:
                print(f"  [skip]   {filename}  ({src_size/1e6:.1f} MB, already up-to-date)")
                skipped += 1
                continue

            if dry_run:
                print(f"  [dry-run] {filename}  <- {src_path}  ({src_size/1e6:.1f} MB)")
            else:
                out_path.write_bytes(blob)
                print(f"  [copied] {filename}  ({src_size/1e6:.1f} MB)  [blob]")
            copied += 1

    print()
    action = "Would copy" if dry_run else "Copied"
    print(f"{'='*55}")
    print(f"  {action}: {copied}  |  Skipped (up-to-date): {skipped}  |  Missing LFS: {missing}")
    print(f"{'='*55}")
    if missing:
        print("  Run `git lfs fetch --all origin` to pull missing objects.")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Copy all .h5 files from a git branch into data/recordings/."
    )
    p.add_argument("--src-branch", default="origin/manraj",
                   help="Source git branch (default: origin/manraj).")
    p.add_argument("--dest-dir", default="data/recordings",
                   help="Destination directory relative to repo root.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print what would be done without copying anything.")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    copy_h5_files(
        src_branch=args.src_branch,
        dest_dir=args.dest_dir,
        dry_run=args.dry_run,
    )
