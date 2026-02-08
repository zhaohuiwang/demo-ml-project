
# I. Most situations, this is enough
# scripts/train.py
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]          # scripts/ → project_root/



# II. This is the simplest, most readable, least magical solution used by most people today.
# demo_ml_package/utils/paths.py
from pathlib import Path

def get_project_root() -> Path:
    """Look for pyproject.toml or .git — very common heuristic"""
    path = Path.cwd().resolve()
    for parent in [path, *path.parents]:
        if (parent / "pyproject.toml").exists() or (parent / ".git").exists():
            return parent
    return path  # fallback to cwd


# III. You do not need this
import numpy as np
from pathlib import Path
import torch


def find_project_root(
    start: Path | None = None,
    *, # everything after this must be passed as a keyword argument
    markers: tuple[str, ...] = (),
    dirname: str | None = None,
) -> Path:
    """
    Find the project root directory by searching upward in the filesystem.

    Starting from `start` (or the current working directory if not provided),
    this function walks up through parent directories until it finds a
    directory that matches one of the following conditions:

    - Contains at least one file or directory listed in `markers`
      (e.g. "pyproject.toml", ".git")
    - Has a directory name equal to `dirname`

    At least one of `markers` or `dirname` must be provided.

    Parameters
    ----------
    start : Path | None, optional
        The directory to start searching from. If None, the current working
        directory is used. The path is resolved to an absolute path before
        searching.

    markers : tuple[str, ...], keyword-only, optional
        A tuple of file or directory names that indicate the project root.
        If any marker exists in a directory, that directory is considered
        the project root.

    dirname : str | None, keyword-only, optional
        The expected name of the project root directory. If a directory with
        this name is encountered while searching upward, it is returned as
        the project root.

    Returns
    -------
    Path - The resolved absolute path of the detected project root directory.

    Raises
    ------
    ValueError - If neither `markers` nor `dirname` is provided.

    FileNotFoundError - If no matching project root is found before reaching the filesystem root.

    Examples
    --------
    Find project root using marker files:
    >>> find_project_root(markers=("pyproject.toml", ".git"))

    Find project root using directory name:
    >>> find_project_root(dirname="my_project")

    Hybrid search (recommended):
    >>> find_project_root(
            start=Path(__file__).parent,
            dirname="my_project",
            markers=("pyproject.toml", ".git")
        )
    """
    if not markers and not dirname:
        raise ValueError("Provide at least one of `markers` or `dirname`")

    if start is None:
        start = Path.cwd()

    start = start.resolve()

    for path in [start, *start.parents]:
        if dirname and path.name == dirname:
            return path

        if markers and any((path / m).exists() for m in markers):
            return path

    raise FileNotFoundError(
        f"Project root not found starting from {start} "
        f"(dirname={dirname}, markers={markers})"
    )


# There are several good ways to find and replace strings (including regex patterns) across all files in a Python/ML project.

# 1. VS Code:
# Press Ctrl+Shift+H (Replace in Files)
# Supports regex (click .* icon)
# Use files to include/exclude patterns: **/*.py, **/*.yaml, !**/venv/**, !**/.git/**, !**/__pycache__/**
# Very fast even in large ML repos

# 2. Quick command-line one-liners (very fast if you're comfortable with terminal)
                                  
# # macOS / Linux (using sed)
# find . -type f \( -name "*.py" -o -name "*.yaml" -o -name "*.yml" -o -name "*.json" \) \
#   -not -path "*/venv/*" -not -path "*/.git/*" -not -path "*/__pycache__/*" \
#   -exec sed -i '' 's/old_pattern/new_pattern/g' {} \;

# # Linux only (GNU sed allows -i without backup suffix)
# find . -type f -name "*.py" -exec sed -i 's/old_pattern/new_pattern/g' {} +


# # 3. Pure Python script (most flexible – good for complex patterns)
# # Here's a safe, recursive version that supports regex and has a dry-run mode:

import os
import re
from pathlib import Path

def replace_in_project(
    root_dir: str = ".",
    pattern: str = r"old_pattern",      # can be regex
    replacement: str = "new_value",
    file_patterns: tuple = ("*.py", "*.yaml", "*.yml", "*.json", "*.ipynb"),
    exclude_dirs: tuple = ("venv", ".git", "__pycache__", ".ipynb_checkpoints", "data", "models", "logs"),
    dry_run: bool = True,
    verbose: bool = True
):
    root = Path(root_dir).resolve()
    regex = re.compile(pattern)

    changed_files = 0
    total_replacements = 0

    for path in root.rglob("*"):
        if path.is_dir():
            if any(excl in path.parts for excl in exclude_dirs):
                continue
            continue

        if not any(path.match(pat) for pat in file_patterns):
            continue

        try:
            original_text = path.read_text(encoding="utf-8")
            new_text, count = regex.subn(replacement, original_text)

            if count > 0:
                changed_files += 1
                total_replacements += count

                if verbose:
                    print(f"Found {count} occurrence(s) in: {path.relative_to(root)}")

                if not dry_run:
                    path.write_text(new_text, encoding="utf-8")

        except Exception as e:
            print(f"Skipped {path}: {e}")

    print(f"\nSummary:")
    print(f"  Files changed : {changed_files}")
    print(f"  Total replacements: {total_replacements}")
    if dry_run:
        print("  (dry run — nothing was actually changed)")

# # ────────────────────────────────────────────────
# # Example usage:

# # Simple string replace
# replace_in_project(
#     pattern="print\\(",
#     replacement="logger.info(",
#     dry_run=True                    # Change to False when you're sure
# )

# # Regex example: change device('cuda:0') → device('cuda')
# replace_in_project(
#     pattern=r"device\(['\"]cuda:\d+['\"]\)",
#     replacement="device('cuda')",
#     dry_run=True
# )

# 4. Jupyter notebooks specifically (important for ML projects)

# Notebooks are JSON, so naïve tools can mess them up.
# Safe way
# pip install nbstripout
# jupyter nbconvert --to notebook --inplace --RegexRemovePreprocessor.patterns="old_pattern" *.ipynb
