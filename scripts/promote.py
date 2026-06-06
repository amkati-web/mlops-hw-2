"""scripts/promote.py — promote MLflow Registry aliases with an audit log."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.exceptions import RestException

REGISTERED_MODEL_NAME = "travel-assistant"
LOG_FILE = Path(__file__).resolve().parent.parent / "promotion-log.jsonl"

mlflow.set_tracking_uri('http://localhost:5000')
client = MlflowClient()


def _find_version(config_id: str):
    """Find registered version by config_id tag. Returns a ModelVersion."""
    results = client.search_model_versions(
        f"name = '{REGISTERED_MODEL_NAME}' AND tags.config_id = '{config_id}'"
    )
    if len(results) == 0:
        print(f"error: no version found with config_id={config_id}")
        sys.exit(1)
    if len(results) > 1:
        versions = sorted([int(mv.version) for mv in results])
        print(f"warning: multiple versions match config_id={config_id} "
              f"(MLflow versions {versions}); using latest ({versions[-1]})")
        return next(mv for mv in results if int(mv.version) == versions[-1])
    return results[0]


def _get_current_config_id(alias: str) -> str:
    """Return current config_id for alias, or empty string if unset."""
    try:
        mv = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, alias)
        return mv.tags.get("config_id", "")
    except RestException:
        return ""


def _append_log(alias: str, from_id: str, to_id: str, op: str) -> None:
    """Append one event to promotion-log.jsonl."""
    event = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "alias": alias,
        "from": from_id,
        "to": to_id,
        "op": op,
    }
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")


def cmd_set(args: argparse.Namespace) -> None:
    mv = _find_version(args.config_id)
    current = _get_current_config_id(args.alias)
    client.set_registered_model_alias(REGISTERED_MODEL_NAME, args.alias, mv.version)
    _append_log(args.alias, current, args.config_id, "set")
    from_str = f"(unset)" if current == "" else current
    print(f"{args.alias}: {from_str} -> {args.config_id}")


def cmd_show(args: argparse.Namespace) -> None:
    try:
        mv = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, args.alias)
    except RestException:
        print(f"error: alias '{args.alias}' is not set")
        sys.exit(1)

    config_id = mv.tags.get("config_id", "unknown")
    model = mv.tags.get("model", "unknown")
    guardrail = mv.tags.get("guardrail_type", "unknown")

    metrics = {}
    if mv.run_id:
        run = client.get_run(mv.run_id)
        metrics = run.data.metrics

    print(f"travel-assistant @ {args.alias}")
    print(f"  config_id:           {config_id}")
    print(f"  model:               {model}")
    print(f"  guardrail_type:      {guardrail}")
    print(f"  mlflow_version:      {mv.version}")
    if metrics:
        print(f"  accuracy_overall:    {metrics.get('accuracy_overall', 'n/a'):.3f}")
        print(f"  verdict_rate_leaked: {metrics.get('verdict_rate_leaked', 0.0):.3f}")
        print(f"  total_cost_usd:      ${metrics.get('total_cost_usd', 0.0):.4f}")
        print(f"  avg_latency_s:       {metrics.get('avg_latency_seconds', 0.0):.2f}")


def cmd_list(args: argparse.Namespace) -> None:
    rm = client.get_registered_model(REGISTERED_MODEL_NAME)
    aliases = rm.aliases
    if not aliases:
        print("no aliases set")
        return
    for alias, version in aliases.items():
        try:
            mv = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, alias)
            config_id = mv.tags.get("config_id", "unknown")
        except RestException:
            config_id = "unknown"
        print(f"{alias} -> {config_id}")


def cmd_rollback(args: argparse.Namespace) -> None:
    # Check current alias target
    current = _get_current_config_id(args.alias)
    if current == "":
        print("nothing to roll back")
        sys.exit(1)

    # Read log backwards for most recent entry for this alias
    if not LOG_FILE.exists():
        print(f"error: no promotion history for alias {args.alias}")
        sys.exit(1)

    lines = LOG_FILE.read_text(encoding="utf-8").splitlines()
    last_entry = None
    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue
        event = json.loads(line)
        if event["alias"] == args.alias:
            last_entry = event
            break

    if last_entry is None:
        print(f"error: no promotion history for alias {args.alias}")
        sys.exit(1)

    if last_entry["op"] == "rollback":
        print(f"error: {args.alias} was just rolled back; no further history to walk back to")
        sys.exit(1)

    if last_entry["from"] == "":
        print(f"error: {args.alias} has no previous target (first promotion ever)")
        sys.exit(1)

    # Roll back to the previous config_id
    target_config_id = last_entry["from"]
    mv = _find_version(target_config_id)
    client.set_registered_model_alias(REGISTERED_MODEL_NAME, args.alias, mv.version)
    _append_log(args.alias, current, target_config_id, "rollback")
    print(f"{args.alias}: {current} -> {target_config_id} (rolled back)")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--name",
        default=REGISTERED_MODEL_NAME,
        help=f"Registered model name (default: {REGISTERED_MODEL_NAME})",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_set = sub.add_parser("set", help="Move an alias to a version (by config_id)")
    p_set.add_argument("alias")
    p_set.add_argument("config_id")
    p_set.set_defaults(func=cmd_set)

    p_show = sub.add_parser("show", help="Show which version an alias points at")
    p_show.add_argument("alias")
    p_show.set_defaults(func=cmd_show)

    p_list = sub.add_parser("list", help="List all aliases on the registered model")
    p_list.set_defaults(func=cmd_list)

    p_rollback = sub.add_parser("rollback", help="Roll back alias to previous target")
    p_rollback.add_argument("alias")
    p_rollback.set_defaults(func=cmd_rollback)

    return parser


def main() -> None:
    args = _build_parser().parse_args()
    try:
        args.func(args)
    except NotImplementedError as exc:
        print(f"NOT IMPLEMENTED: {exc}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()

