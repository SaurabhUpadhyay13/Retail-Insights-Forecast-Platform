"""Reliable one-shot refresh entry point for an external daily scheduler."""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from contextlib import contextmanager
from datetime import date, datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Iterator, Sequence

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
OUTPUT_DIR = ROOT_DIR / "outputs"
LOG_DIR = ROOT_DIR / "logs"
STATUS_PATH = OUTPUT_DIR / "refresh_status.json"
HEALTHCHECK_STATUS_PATH = OUTPUT_DIR / "refresh_healthcheck.json"
LOCK_PATH = OUTPUT_DIR / ".daily_refresh.lock"
PIPELINE_STEPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("01_data_cleaning.py", ("--csv-only",)),
    ("02_streaming_pipeline.py", ()),
    ("03_export_for_powerbi.py", ()),
)
PUBLISHED_FILES = (
    "fact_forecast_anomaly.csv",
    "dim_date.csv",
    "kpi_summary_by_category.csv",
    "model_performance.csv",
)
DEFAULT_TIMEOUT_SECONDS = 12 * 60 * 60
DEFAULT_RETRIES = 1
STALE_LOCK_SECONDS = 24 * 60 * 60


def configure_logging(verbose: bool = False) -> logging.Logger:
    """Log to the scheduler console and a bounded rotating file."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("retail_refresh")
    # Keep detailed stage output in the rotating file while the console stays
    # concise unless --verbose is requested.
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(message)s", datefmt="%Y-%m-%dT%H:%M:%S%z"
    )
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.DEBUG if verbose else logging.INFO)
    console.setFormatter(formatter)
    file_handler = RotatingFileHandler(
        LOG_DIR / "daily_refresh.log",
        maxBytes=10 * 1024 * 1024,
        backupCount=10,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(console)
    logger.addHandler(file_handler)
    return logger


def write_json_atomic(path: Path, payload: dict) -> None:
    """Publish JSON without exposing readers to a partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def exclusive_job_lock(run_id: str) -> Iterator[None]:
    """Prevent overlap and recover locks abandoned for more than a day."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if LOCK_PATH.exists() and time.time() - LOCK_PATH.stat().st_mtime > STALE_LOCK_SECONDS:
        LOCK_PATH.unlink(missing_ok=True)
    try:
        descriptor = os.open(LOCK_PATH, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        owner = LOCK_PATH.read_text(encoding="utf-8", errors="replace").strip()
        raise RuntimeError(
            "Another daily refresh is already running. "
            f"Lock details: {owner or 'unavailable'}"
        ) from exc
    try:
        os.write(
            descriptor,
            json.dumps(
                {
                    "pid": os.getpid(),
                    "run_id": run_id,
                    "started_at_utc": datetime.now(timezone.utc).isoformat(),
                }
            ).encode("utf-8"),
        )
        os.close(descriptor)
        yield
    finally:
        try:
            os.close(descriptor)
        except OSError:
            pass
        LOCK_PATH.unlink(missing_ok=True)


@contextmanager
def published_output_rollback(logger: logging.Logger) -> Iterator[None]:
    """Restore the last good Power BI tables when any stage fails."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="refresh-backup-", dir=OUTPUT_DIR) as raw:
        backup_dir = Path(raw)
        existing: set[str] = set()
        for filename in PUBLISHED_FILES:
            source = OUTPUT_DIR / filename
            if source.exists():
                shutil.copy2(source, backup_dir / filename)
                existing.add(filename)
        try:
            yield
        except BaseException:
            logger.error("Refresh failed; restoring the last published Power BI tables")
            for filename in PUBLISHED_FILES:
                destination = OUTPUT_DIR / filename
                if filename in existing:
                    restore_tmp = destination.with_suffix(destination.suffix + ".restore")
                    shutil.copy2(backup_dir / filename, restore_tmp)
                    os.replace(restore_tmp, destination)
                else:
                    destination.unlink(missing_ok=True)
            raise


def run_step(
    script_name: str,
    script_args: Sequence[str],
    timeout_seconds: int,
    retries: int,
    logger: logging.Logger,
) -> None:
    """Run one idempotent stage with timeout and bounded retries."""
    script_path = SCRIPTS_DIR / script_name
    if not script_path.is_file():
        raise FileNotFoundError(f"Pipeline stage is missing: {script_path}")
    command = [sys.executable, "-u", str(script_path), *script_args]
    for attempt in range(1, retries + 2):
        logger.info("Starting %s (attempt %d/%d)", script_name, attempt, retries + 1)
        started = time.monotonic()
        process = subprocess.Popen(
                command,
                cwd=ROOT_DIR,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )

        def log_output() -> None:
            assert process.stdout is not None
            for line in process.stdout:
                logger.debug("[%s] %s", script_name, line.rstrip())

        output_thread = threading.Thread(target=log_output, daemon=True)
        output_thread.start()
        try:
            return_code = process.wait(timeout=timeout_seconds)
            output_thread.join(timeout=10)
        except subprocess.TimeoutExpired as exc:
            process.kill()
            process.wait()
            output_thread.join(timeout=10)
            logger.error("%s exceeded its %d-second timeout", script_name, timeout_seconds)
            failure: Exception = exc
        else:
            if return_code == 0:
                logger.info(
                    "Completed %s in %.1f seconds", script_name, time.monotonic() - started
                )
                return
            failure = RuntimeError(
                f"{script_name} exited with status {return_code}. "
                f"See {LOG_DIR / 'daily_refresh.log'} for details."
            )
            logger.error("%s", failure)
        if attempt <= retries:
            delay = min(30 * attempt, 120)
            logger.warning("Retrying %s in %d seconds", script_name, delay)
            time.sleep(delay)
        else:
            raise failure


def validate_published_outputs() -> dict[str, object]:
    """Require a coherent, nonempty exported star schema."""
    paths = {name: OUTPUT_DIR / name for name in PUBLISHED_FILES}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing published output(s): " + ", ".join(missing))
    fact = pd.read_csv(paths["fact_forecast_anomaly.csv"], parse_dates=["date"])
    dates = pd.read_csv(paths["dim_date.csv"], parse_dates=["date"])
    kpi = pd.read_csv(paths["kpi_summary_by_category.csv"])
    performance = pd.read_csv(paths["model_performance.csv"])
    if fact.empty or dates.empty or kpi.empty or performance.empty:
        raise ValueError("Published fact, date, KPI, and performance tables must be nonempty")
    required = {
        "date", "category", "actual_units", "revenue",
        "online_model_forecast", "production_forecast", "production_model",
        "any_anomaly",
    }
    missing_columns = sorted(required.difference(fact.columns))
    if missing_columns:
        raise ValueError("Fact table is missing columns: " + ", ".join(missing_columns))
    if fact[["date", "category"]].isna().any().any():
        raise ValueError("Fact table contains null date/category keys")
    if fact.duplicated(["date", "category"]).any():
        raise ValueError("Fact table contains duplicate date/category keys")
    if (fact["actual_units"] < 0).any() or (fact["revenue"] < 0).any():
        raise ValueError("Fact table contains negative units or revenue")
    min_date = fact["date"].min()
    max_date = fact["date"].max()
    expected_dates = pd.date_range(min_date, max_date, freq="D")
    actual_dates = pd.DatetimeIndex(dates["date"].dropna().sort_values().unique())
    if not actual_dates.equals(expected_dates):
        raise ValueError("Date dimension does not continuously cover the fact table")
    fact_categories = set(fact["category"].astype(str))
    kpi_categories = set(kpi["category"].astype(str))
    if fact_categories != kpi_categories:
        raise ValueError("KPI categories do not match fact-table categories")
    required_models = {"online_sgd", "seasonal_naive_7d", "production_selection"}
    if not required_models.issubset(set(performance["model"].astype(str))):
        raise ValueError("Model performance table is missing required benchmarks")
    return {
        "fact_rows": int(len(fact)),
        "category_count": len(fact_categories),
        "min_data_date": min_date.date().isoformat(),
        "max_data_date": max_date.date().isoformat(),
        "anomaly_rows": int(fact["any_anomaly"].fillna(False).astype(bool).sum()),
    }


def parse_run_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("run date must use YYYY-MM-DD") from exc


def run_daily_job(
    run_date: date,
    *,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    retries: int = DEFAULT_RETRIES,
    skip_clean: bool = False,
    dry_run: bool = False,
    verbose: bool = False,
) -> dict[str, object]:
    """Execute, validate, and record one complete refresh."""
    logger = configure_logging(verbose)
    run_id = uuid.uuid4().hex
    started_at = datetime.now(timezone.utc)
    status: dict[str, object] = {
        "run_id": run_id,
        "run_date": run_date.isoformat(),
        "mode": "dry_run" if dry_run else "refresh",
        "status": "running",
        "started_at_utc": started_at.isoformat(),
    }
    status_path = HEALTHCHECK_STATUS_PATH if dry_run else STATUS_PATH
    with exclusive_job_lock(run_id):
        write_json_atomic(status_path, status)
        logger.info("Refresh %s started for business date %s", run_id, run_date)
        try:
            if dry_run:
                metrics = validate_published_outputs()
                logger.info("Dry run complete; existing published outputs are healthy")
            else:
                steps = PIPELINE_STEPS[1:] if skip_clean else PIPELINE_STEPS
                with published_output_rollback(logger):
                    for script_name, args in steps:
                        run_step(script_name, args, timeout_seconds, retries, logger)
                    metrics = validate_published_outputs()
            finished_at = datetime.now(timezone.utc)
            status.update(
                status="succeeded",
                finished_at_utc=finished_at.isoformat(),
                duration_seconds=round((finished_at - started_at).total_seconds(), 3),
                metrics=metrics,
            )
            write_json_atomic(status_path, status)
            logger.info("Refresh %s succeeded: %s", run_id, metrics)
            return status
        except BaseException as exc:
            finished_at = datetime.now(timezone.utc)
            status.update(
                status="failed",
                finished_at_utc=finished_at.isoformat(),
                duration_seconds=round((finished_at - started_at).total_seconds(), 3),
                error_type=type(exc).__name__,
                error=str(exc),
            )
            write_json_atomic(status_path, status)
            logger.exception("Refresh %s failed", run_id)
            raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the retail forecast refresh once")
    parser.add_argument(
        "--run-date", type=parse_run_date, default=date.today(),
        help="Business date for observability (YYYY-MM-DD; default: today)",
    )
    parser.add_argument(
        "--timeout-seconds", type=int, default=DEFAULT_TIMEOUT_SECONDS,
        help="Maximum runtime for each stage",
    )
    parser.add_argument("--retries", type=int, default=DEFAULT_RETRIES)
    parser.add_argument(
        "--skip-clean", action="store_true",
        help="Reuse the current cleaned CSV and rebuild forecasts/exports only",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Validate current published outputs without changing pipeline data",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.timeout_seconds <= 0:
        raise SystemExit("--timeout-seconds must be positive")
    if args.retries < 0:
        raise SystemExit("--retries cannot be negative")
    try:
        run_daily_job(
            args.run_date,
            timeout_seconds=args.timeout_seconds,
            retries=args.retries,
            skip_clean=args.skip_clean,
            dry_run=args.dry_run,
            verbose=args.verbose,
        )
    except Exception:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
