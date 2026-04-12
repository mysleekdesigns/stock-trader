#!/usr/bin/env python3
"""CLI for running the walk-forward model training pipeline.

Usage examples::

    # Train all models from CSV data
    python -m scripts.train_models --config config/training.yaml --models all --data-source csv

    # Train only tree models, 50 hyperopt trials
    python -m scripts.train_models --models lightgbm xgboost --data-source db --n-trials 50

    # Train LSTM with custom output directory
    python -m scripts.train_models --models lstm --data-source csv --output-dir ./trained_models
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import structlog
import yaml

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_MODELS = ("lightgbm", "xgboost", "lstm", "tft", "all")
DEFAULT_OUTPUT_DIR = "models/trained"
DEFAULT_CONFIG = "config/training.yaml"


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train ML models using walk-forward validation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG,
        help="Path to a YAML training configuration file.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["all"],
        choices=VALID_MODELS,
        help="Which models to train.",
    )
    parser.add_argument(
        "--data-source",
        type=str,
        choices=["db", "csv"],
        default="csv",
        help="Load data from TimescaleDB ('db') or local CSV files ('csv').",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to write trained model artefacts.",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=0,
        help="Number of Optuna hyperopt trials.  0 disables hyperopt.",
    )
    parser.add_argument(
        "--csv-path",
        type=str,
        default=None,
        help="Path to CSV file (required when --data-source=csv).",
    )
    parser.add_argument(
        "--db-url",
        type=str,
        default=None,
        help="TimescaleDB connection URL (required when --data-source=db).",
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["SPY"],
        help="Symbols to train on (used for DB source).",
    )
    return parser


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> dict[str, Any]:
    """Load YAML training configuration; return empty dict on missing file."""
    path = Path(config_path)
    if not path.exists():
        logger.warning("config_not_found", path=str(path))
        return {}
    with open(path) as fh:
        cfg = yaml.safe_load(fh) or {}
    logger.info("config_loaded", path=str(path), keys=list(cfg.keys()))
    return cfg


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data_csv(csv_path: str) -> tuple["pd.DataFrame", "pd.DataFrame"]:
    """Load features and targets from a CSV file.

    Expected layout: DatetimeIndex, feature columns, then three target columns
    (``direction``, ``returns``, ``volatility``) as the last three columns.
    """
    import pandas as pd

    logger.info("loading_csv", path=csv_path)
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    df.index.name = "timestamp"

    target_cols = ["direction", "returns", "volatility"]
    present_targets = [c for c in target_cols if c in df.columns]

    if len(present_targets) < 3:
        logger.warning(
            "missing_target_columns",
            expected=target_cols,
            found=present_targets,
        )
        # Fall back: derive targets from the close column if available.
        if "close" in df.columns:
            df["returns"] = df["close"].pct_change().fillna(0)
            df["direction"] = (df["returns"] > 0).astype(int)
            df["volatility"] = df["returns"].rolling(21).std().fillna(0)
        else:
            raise SystemExit("CSV must contain target columns or a 'close' column.")

    y = df[["direction", "returns", "volatility"]]
    X = df.drop(columns=["direction", "returns", "volatility"], errors="ignore")

    logger.info("data_loaded", n_rows=len(X), n_features=X.shape[1])
    return X, y


async def load_data_db(
    db_url: str,
    symbols: list[str],
) -> tuple["pd.DataFrame", "pd.DataFrame"]:
    """Load data from TimescaleDB, compute features and derive targets."""
    import pandas as pd
    from datetime import datetime

    from src.core.types import TimeFrame
    from src.data.storage.timeseries_store import TimeseriesStore
    from src.features.technical import registry as tech_registry
    import src.features.price  # noqa: F401
    from src.features.pipeline import FeaturePipeline

    store = TimeseriesStore(db_url)
    all_frames: list[pd.DataFrame] = []

    try:
        for symbol in symbols:
            logger.info("fetching_from_db", symbol=symbol)
            bars = await store.get_bars(
                symbol,
                TimeFrame.DAILY,
                datetime(2015, 1, 1),
                datetime.utcnow(),
            )
            if not bars:
                logger.warning("no_bars", symbol=symbol)
                continue

            records = [
                {
                    "open": b.open,
                    "high": b.high,
                    "low": b.low,
                    "close": b.close,
                    "volume": b.volume,
                }
                for b in bars
            ]
            df = pd.DataFrame(records, index=pd.DatetimeIndex([b.timestamp for b in bars]))
            df.index.name = "timestamp"
            all_frames.append(df)
    finally:
        await store.close()

    if not all_frames:
        raise SystemExit("No data retrieved from database.")

    raw = pd.concat(all_frames).sort_index()

    # Compute features.
    pipeline = FeaturePipeline(registry=tech_registry, normalize=False)
    featured = pipeline.fit_transform(raw)

    # Derive targets.
    featured["returns"] = raw["close"].pct_change().fillna(0)
    featured["direction"] = (featured["returns"] > 0).astype(int)
    featured["volatility"] = featured["returns"].rolling(21).std().fillna(0)

    y = featured[["direction", "returns", "volatility"]]
    X = featured.drop(columns=["direction", "returns", "volatility"])

    # Drop warm-up NaN rows.
    mask = X.notna().all(axis=1) & y.notna().all(axis=1)
    X, y = X[mask], y[mask]

    logger.info("db_data_loaded", n_rows=len(X), n_features=X.shape[1])
    return X, y


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def resolve_model_names(requested: list[str]) -> list[str]:
    """Expand 'all' and return deduplicated model name list."""
    if "all" in requested:
        return ["lightgbm", "xgboost", "lstm", "tft"]
    return list(dict.fromkeys(requested))  # preserve order, deduplicate


def create_model(name: str, config: dict[str, Any]) -> Any:
    """Instantiate a model by name, pulling hyper-params from config."""
    model_params = config.get("models", {}).get(name, {})

    if name == "lightgbm":
        from src.models.tree.lightgbm_model import LightGBMPredictor

        return LightGBMPredictor(params=model_params)

    if name == "xgboost":
        from src.models.tree.xgboost_model import XGBoostPredictor

        return XGBoostPredictor(params=model_params)

    if name == "lstm":
        from src.models.deep.lstm_attention import LSTMAttentionPredictor

        return LSTMAttentionPredictor(params=model_params)

    if name == "tft":
        from src.models.transformer.temporal_fusion import TFTPredictor

        return TFTPredictor(params=model_params)

    raise ValueError(f"Unknown model: {name}")


# ---------------------------------------------------------------------------
# Feature pipeline
# ---------------------------------------------------------------------------

def build_feature_pipeline() -> Any:
    """Construct the feature engineering pipeline."""
    try:
        from src.features.technical import registry as tech_registry
        import src.features.price  # noqa: F401
        from src.features.pipeline import FeaturePipeline

        return FeaturePipeline(registry=tech_registry, normalize=True)
    except ImportError:
        logger.warning("feature_pipeline_not_available")
        return None


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def run_training(
    models: list[Any],
    X: "pd.DataFrame",
    y: "pd.DataFrame",
    config: dict[str, Any],
    output_dir: str,
    n_trials: int,
) -> dict[str, dict[str, Any]]:
    """Run walk-forward training for each model.

    Returns a summary dict mapping model name -> metrics.
    """
    from src.models.training.walk_forward import WalkForwardSplitter
    from src.models.training.trainer import ModelTrainer

    # Splitter params from config or defaults (PRD: 504 / 63 / 21 / 21).
    split_cfg = config.get("walk_forward", {})
    splitter = WalkForwardSplitter(
        train_size=split_cfg.get("train_size", 504),
        val_size=split_cfg.get("val_size", 63),
        test_size=split_cfg.get("test_size", 21),
        step_size=split_cfg.get("step_size", 21),
    )

    trainer = ModelTrainer(
        splitter=splitter,
        output_dir=Path(output_dir),
    )

    summary: dict[str, dict[str, Any]] = {}

    for model in models:
        model_name = getattr(model, "name", str(model))
        logger.info("training_model", model=model_name)

        try:
            metrics = trainer.train(model, X, y)
            summary[model_name] = metrics
            logger.info("model_trained", model=model_name, metrics=metrics)

            # Persist model artefacts.
            model_path = Path(output_dir) / model_name
            model.save(model_path)
            logger.info("model_saved", model=model_name, path=str(model_path))
        except Exception:
            logger.exception("training_failed", model=model_name)
            summary[model_name] = {"error": "training_failed"}

    # Optionally run hyperopt.
    if n_trials > 0:
        _run_hyperopt(models, X, y, splitter, config, n_trials, output_dir, summary)

    return summary


def _run_hyperopt(
    models: list[Any],
    X: "pd.DataFrame",
    y: "pd.DataFrame",
    splitter: Any,
    config: dict[str, Any],
    n_trials: int,
    output_dir: str,
    summary: dict[str, dict[str, Any]],
) -> None:
    """Run Optuna hyperparameter optimisation if available."""
    try:
        from src.models.training.hyperopt import run_hyperopt

        for model in models:
            model_name = getattr(model, "name", str(model))
            logger.info("hyperopt_start", model=model_name, n_trials=n_trials)
            try:
                best = run_hyperopt(
                    model=model,
                    X=X,
                    y=y,
                    splitter=splitter,
                    n_trials=n_trials,
                )
                summary[f"{model_name}_hyperopt"] = best
                logger.info("hyperopt_complete", model=model_name, best=best)
            except Exception:
                logger.exception("hyperopt_failed", model=model_name)
    except ImportError:
        logger.warning("hyperopt_module_not_available")


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

def register_models(
    summary: dict[str, dict[str, Any]],
    output_dir: str,
) -> None:
    """Register trained models in the model registry (MLflow or local)."""
    try:
        from src.models.training.registry import ModelRegistry

        registry = ModelRegistry()
        for model_name, metrics in summary.items():
            if "error" in metrics:
                continue
            logger.info("registering_model", model=model_name)
            registry.register(
                name=model_name,
                path=str(Path(output_dir) / model_name),
                metrics=metrics,
            )
    except ImportError:
        logger.warning("model_registry_not_available_skipping_registration")


# ---------------------------------------------------------------------------
# Summary printing
# ---------------------------------------------------------------------------

def print_summary(summary: dict[str, dict[str, Any]]) -> None:
    """Print a formatted summary table to stdout."""
    print("\n" + "=" * 70)
    print("  MODEL TRAINING SUMMARY")
    print("=" * 70)

    for model_name, metrics in summary.items():
        print(f"\n  {model_name}")
        print("  " + "-" * 40)
        if isinstance(metrics, dict):
            for key, value in metrics.items():
                if isinstance(value, float):
                    print(f"    {key:30s}: {value:>12.6f}")
                else:
                    print(f"    {key:30s}: {value!s:>12s}")
        else:
            print(f"    {metrics}")

    print("\n" + "=" * 70 + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    """Entry point for the training CLI."""
    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.BoundLogger,
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    parser = build_parser()
    args = parser.parse_args()

    # Load config.
    config = load_config(args.config)

    # Load data.
    if args.data_source == "csv":
        csv_path = args.csv_path or config.get("csv_path")
        if not csv_path:
            logger.error("csv_path_required")
            print("ERROR: --csv-path is required when --data-source=csv")
            return 1
        X, y = load_data_csv(csv_path)
    else:
        import asyncio

        db_url = args.db_url or config.get("db_url")
        if not db_url:
            logger.error("db_url_required")
            print("ERROR: --db-url is required when --data-source=db")
            return 1
        X, y = asyncio.run(load_data_db(db_url, args.symbols))

    # Resolve model names.
    model_names = resolve_model_names(args.models)
    logger.info("models_selected", models=model_names)

    # Initialize feature pipeline (optional pre-processing).
    feature_pipeline = build_feature_pipeline()
    if feature_pipeline is not None:
        logger.info("applying_feature_pipeline")
        try:
            X = feature_pipeline.fit_transform(X)
        except Exception:
            logger.warning("feature_pipeline_failed_using_raw_data")

    # Create model instances.
    models: list[Any] = []
    for name in model_names:
        try:
            model = create_model(name, config)
            models.append(model)
            logger.info("model_created", model=name)
        except (ImportError, ValueError):
            logger.warning("model_unavailable", model=name)

    if not models:
        logger.error("no_models_available")
        print("ERROR: No models could be initialised.")
        return 1

    # Ensure output directory exists.
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Train.
    summary = run_training(
        models=models,
        X=X,
        y=y,
        config=config,
        output_dir=args.output_dir,
        n_trials=args.n_trials,
    )

    # Register in model registry.
    register_models(summary, args.output_dir)

    # Print summary.
    print_summary(summary)

    logger.info("training_pipeline_complete", n_models=len(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
