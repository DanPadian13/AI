#!/usr/bin/env python3
"""Quick NeuroGym smoke test derived from the Lab 3 notebook instructions.

Usage (once Gym/NeuroGym are installed):
    python3 code/neuro_gym_testing.py --task PerceptualDecisionMaking-v0

This script intentionally mirrors the lab workflow:
1. Use Gym 0.23.1 for compatibility with NeuroGym.
2. Clone https://github.com/neurogym/neurogym.git and `pip install -e .`.
3. Instantiate a small Dataset to confirm everything works.
"""

from __future__ import annotations

import argparse
import importlib
import logging
import os
import sys
import warnings
from typing import Iterable, Sequence, Tuple, List, Optional
from pathlib import Path
import tempfile

GYM_INSTALL_HINT = (
    "Gym is missing. Install the lab-tested version via:\n"
    "  pip3 uninstall -y gym\n"
    "  pip3 install gym==0.23.1"
)

NEUROGYM_INSTALL_HINT = (
    "NeuroGym is missing. Follow the lab steps:\n"
    "  git clone https://github.com/neurogym/neurogym.git\n"
    "  cd neurogym && pip3 install -e ."
)


def require_module(module_name: str, hint: str):
    """Import `module_name`, or exit with the provided hint."""
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:  # pragma: no cover - purely defensive messaging.
        print(f"\n[Missing dependency] Could not import '{module_name}'.", file=sys.stderr)
        print(hint, file=sys.stderr)
        print("\nAfter installing, re-run this script.", file=sys.stderr)
        raise SystemExit(1) from exc


def discover_tasks(ngym_module) -> Sequence[str]:
    """Best-effort extraction of the registered NeuroGym environments."""
    candidate_attrs = ("all_envs", "ALL_ENVS", "all_tasks", "ALL_TASKS")
    for attr in candidate_attrs:
        value = getattr(ngym_module, attr, None)
        if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
            return sorted(set(value))

    return []


def build_dataset(ngym_module, task: str, seq_len: int, batch_size: int, dt: float):
    """Instantiate a NeuroGym Dataset and fetch a single batch."""
    kwargs = {"dt": dt}
    dataset = ngym_module.Dataset(
        task, env_kwargs=kwargs, batch_size=batch_size, seq_len=seq_len
    )
    env = dataset.env
    inputs, targets = dataset()
    return dataset, env, inputs, targets


def load_registry() -> Optional[Tuple]:
    """Access neurogym.envs.registration if available."""
    try:
        registration = importlib.import_module("neurogym.envs.registration")
    except ImportError:
        return None
    all_envs = getattr(registration, "all_envs", None)
    make = getattr(registration, "make", None)
    if callable(all_envs) and callable(make):
        return all_envs, make
    return None


def task_metadata(all_envs_fn, make_fn, include_extras: bool) -> List[dict]:
    """Collect docstrings/metadata for each registered environment."""
    kwargs = {}
    if include_extras:
        kwargs = {"contrib": True, "collections": True, "psychopy": True}
    env_ids = sorted(all_envs_fn(**kwargs))
    results = []
    for env_id in env_ids:
        summary = ""
        paper = ""
        tags = ""
        note = ""
        try:
            env = make_fn(env_id)
            base_env = getattr(env, "unwrapped", env)
            doc = (getattr(base_env, "__doc__", "") or "").strip()
            if doc:
                summary = doc.splitlines()[0]
            metadata = getattr(base_env, "metadata", {}) or {}
            summary = metadata.get("description", summary)
            paper = metadata.get("paper_name") or ""
            tags_list = metadata.get("tags")
            if isinstance(tags_list, (list, tuple)) and tags_list:
                tags = ", ".join(tags_list)
            spec = getattr(env, "spec", None)
            if not paper and spec is not None:
                paper = getattr(spec, "kwargs", {}).get("paper_name", "") or ""
        except Exception as exc:  # pragma: no cover - best effort
            note = f"Failed to instantiate: {exc}"
        else:
            try:  # pragma: no cover
                env.close()
            except Exception:
                pass
        results.append(
            {
                "id": env_id,
                "summary": summary,
                "paper": paper,
                "tags": tags,
                "note": note,
            }
        )
    return results


def configure_warnings(suppress: bool) -> None:
    """Optionally silence the noisy Gym/NeuroGym deprecation chatter."""
    if not suppress:
        return

    warnings.filterwarnings(
        "ignore", category=UserWarning, module="gymnasium.core"
    )
    warnings.filterwarnings(
        "ignore", category=UserWarning, module="gymnasium.envs.registration"
    )
    logging.getLogger("gymnasium").setLevel(logging.ERROR)


def main() -> None:
    # Ensure Matplotlib (used indirectly by NeuroGym) has somewhere writable.
    tmp = Path(tempfile.gettempdir()) / "matplotlib"
    tmp.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(tmp))
    fontcache = tmp / "fontconfig"
    fontcache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("XDG_CACHE_HOME", str(tmp))

    parser = argparse.ArgumentParser(
        description="Validate Gym/NeuroGym installation and inspect available tasks."
    )
    parser.add_argument(
        "--task",
        default="PerceptualDecisionMaking-v0",
        help="NeuroGym task id to sample (default: %(default)s)",
    )
    parser.add_argument(
        "--seq-len", type=int, default=100, help="Sequence length passed to Dataset()"
    )
    parser.add_argument(
        "--batch-size", type=int, default=16, help="Batch size passed to Dataset()"
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=20.0,
        help="Task discretization step (matching lab defaults).",
    )
    parser.add_argument(
        "--show-warnings",
        action="store_true",
        help="Display Gym/NeuroGym warnings (suppressed by default).",
    )
    parser.add_argument(
        "--no-task-catalog",
        action="store_true",
        help="Skip printing the full task/description catalog.",
    )
    parser.add_argument(
        "--include-extra-tasks",
        action="store_true",
        help="Include contrib/psychopy/collection envs when listing tasks.",
    )
    args = parser.parse_args()

    configure_warnings(suppress=not args.show_warnings)

    gym = require_module("gym", GYM_INSTALL_HINT)
    ngym = require_module("neurogym", NEUROGYM_INSTALL_HINT)

    print(f"Gym {getattr(gym, '__version__', '?')} successfully imported.")
    print(f"NeuroGym package path: {ngym.__file__}")

    registry = load_registry()
    if registry is not None:
        all_envs_fn, make_fn = registry
        tasks = task_metadata(
            all_envs_fn,
            make_fn,
            include_extras=args.include_extra_tasks,
        )
        print(
            f"Discovered {len(tasks)} NeuroGym tasks "
            f"({'core only' if not args.include_extra_tasks else 'with extras'})."
        )
        if not args.no_task_catalog:
            for task in tasks:
                summary = task["summary"] or "(no docstring)"
                paper = f" | Paper: {task['paper']}" if task["paper"] else ""
                tags = f" | Tags: {task['tags']}" if task["tags"] else ""
                note = f" | NOTE: {task['note']}" if task["note"] else ""
                print(f"- {task['id']}: {summary}{paper}{tags}{note}")
    else:
        print("Unable to import neurogym.envs.registration; task listing skipped.")

    try:
        dataset, env, inputs, targets = build_dataset(
            ngym, args.task, args.seq_len, args.batch_size, args.dt
        )
    except Exception as exc:  # pragma: no cover - runtime diagnostics only.
        print(
            "\n[Dataset error] Failed to instantiate the NeuroGym dataset. "
            "Double-check the task id and dependency versions.",
            file=sys.stderr,
        )
        raise

    obs_dim = getattr(env.observation_space, "shape", ("?",))
    act_dim = getattr(env.action_space, "n", "?")
    print(f"\nDataset ready for task '{args.task}':")
    print(f"  Observation dim: {obs_dim}")
    print(f"  Action count:    {act_dim}")
    print(f"  Input batch shape:  {inputs.shape}")
    print(f"  Target batch shape: {targets.shape}")


if __name__ == "__main__":
    main()
