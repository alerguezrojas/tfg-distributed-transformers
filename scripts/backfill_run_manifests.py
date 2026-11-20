"""Write ``run_<ts>.json`` for the runs that predate the manifest (docs/eventos.md).

For every ``train_<ts>.log`` under ``logs/`` without a manifest, the manifest is built
from the first source available:

1. its ``events_<ts>.jsonl`` (the ``run_start`` event: complete), looked up next to the
   log and in every ``--events-from`` folder (event streams are usually not committed,
   so they may live in another checkout);
2. otherwise the ``Configuración:`` line of the log (seed, loss, precision, batch, lr,
   epochs, train/val sizes), the folder layout (env, dataset, strategy, model) and the
   task and classes of the registered dataset.

Nothing is invented: what a run did not record is left out of its manifest.

Usage:
    python scripts/backfill_run_manifests.py                       # logs/ of this checkout
    python scripts/backfill_run_manifests.py --events-from /path/to/other/logs --dry-run
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.run_manifest import (  # noqa: E402
    build_manifest, manifest_from_log, manifest_path, read_run_start, write_manifest)
from src.web.run_registry import _env_mode_model_from_path  # noqa: E402

_LOG_TS = re.compile(r"^train_(?:deep_)?(\d{8}_\d{6})\.log$")


def dataset_facts(dataset: str) -> dict:
    """Task, ordered-classes flag and class names of a registered dataset (the dataset
    definition, not a guess); {} when it is not registered or the registry cannot load."""
    try:
        from src.data.registry import get_dataset_spec
        spec = get_dataset_spec(dataset)
    except Exception:       # unknown dataset, or no torch in this environment
        return {}
    return {"task": spec.task, "ordinal": bool(spec.ordinal), "classes": list(spec.classes)}


def find_events(directory: Path, run_id: str, events_roots: list[Path]) -> Path | None:
    """``events_<run_id>.jsonl`` next to the log or under any of ``events_roots``."""
    name = f"events_{run_id}.jsonl"
    if (directory / name).is_file():
        return directory / name
    for root in events_roots:
        found = next(iter(sorted(root.rglob(name))), None)
        if found is not None:
            return found
    return None


def backfill(logs_root: Path, events_roots: list[Path] | None = None, *,
             force: bool = False, dry_run: bool = False) -> dict[str, list[Path]]:
    """Create the missing manifests; returns the manifests by source (``events``/``log``)
    plus ``skipped`` (already had one)."""
    done: dict[str, list[Path]] = {"events": [], "log": [], "skipped": []}
    for log in sorted(logs_root.rglob("train_*.log")):
        m = _LOG_TS.match(log.name)
        if not m:
            continue
        run_id = m.group(1)
        target = manifest_path(log.parent, run_id)
        if target.exists() and not force:
            done["skipped"].append(target)
            continue
        run_start = read_run_start(find_events(log.parent, run_id, events_roots or []))
        env, mode, model, dataset = _env_mode_model_from_path(log, logs_root)
        if run_start:
            manifest, source = build_manifest(run_start, source="events"), "events"
        else:
            facts = dataset_facts(dataset)
            manifest = manifest_from_log(
                log, dataset=dataset, model=model, mode=mode, run_id=run_id, env=env,
                task=facts.get("task"), classes=facts.get("classes"))
            if "ordinal" in facts:
                manifest["ordinal"] = facts["ordinal"]
            source = "log"
        if not dry_run:
            write_manifest(target, manifest)
        done[source].append(target)
    return done


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--logs", type=Path, default=Path("logs"), help="logs/ folder")
    parser.add_argument("--events-from", type=Path, action="append", default=[],
                        help="folder to look for events_*.jsonl in (repeatable)")
    parser.add_argument("--force", action="store_true", help="overwrite existing manifests")
    parser.add_argument("--dry-run", action="store_true", help="only report")
    args = parser.parse_args()
    done = backfill(args.logs, args.events_from, force=args.force, dry_run=args.dry_run)
    verb = "would write" if args.dry_run else "wrote"
    print(f"{verb} {len(done['events'])} manifests from events, {len(done['log'])} from "
          f"the log; {len(done['skipped'])} already had one")


if __name__ == "__main__":
    main()
