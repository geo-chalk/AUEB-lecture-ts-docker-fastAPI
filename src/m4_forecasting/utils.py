import argparse
import logging
import sys

import optuna

from m4_forecasting import LOGGER_NAME
from src.m4_forecasting.config import PipelineConfig


def setup_logging(name: str = LOGGER_NAME) -> logging.Logger:
    """
    Configures the Root Logger to handle ALL logs from all libraries.
    Clears existing handlers to prevent duplicates.
    """
    root_logger = logging.getLogger()

    # 1. CLEAR THE SLATE: Remove any existing handlers (Fixes accumulation)
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    # 2. Add ONE clean handler to the root
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        fmt='[%(name)s - %(asctime)s] %(levelname)s [%(filename)s:%(lineno)d - %(funcName)s] - %(message)s',
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    handler.setFormatter(formatter)
    root_logger.addHandler(handler)
    root_logger.setLevel(logging.INFO)

    # 3. FIX OPTUNA: Disable its private handler so it only uses our Root handler
    optuna.logging.disable_default_handler()
    optuna.logging.enable_propagation()

    return logging.getLogger(name)


def keep_loggers(kept_loggers: list):
    """
    Sets all loggers to WARNING except the ones in the whitelist.
    """
    # Force an update of the manager's dictionary to find all loggers
    logging.root.manager.loggerDict.keys()

    for name, logger in logging.root.manager.loggerDict.items():
        if isinstance(logger, logging.PlaceHolder):
            continue

        # Check if this logger is in the 'keep' list OR is a child/parent
        is_kept = any(name == k or name.startswith(k + ".") for k in kept_loggers)

        if is_kept:
            logger.setLevel(logging.INFO)
            # Ensure it propagates to our nice Root handler
            logger.propagate = True
        else:
            logger.setLevel(logging.WARNING)


def parse_args() -> argparse.Namespace:
    """
    Parses CLI arguments.
    IMPORTANT: All defaults are None so we know if the user
    explicitly set them. Real defaults live in PipelineConfig.
    """
    parser = argparse.ArgumentParser(description="M4 Forecasting Pipeline CLI")

    # Note: No default=... here.
    parser.add_argument("--horizon", type=int, help="Forecast horizon in hours")
    parser.add_argument("--group", type=str, choices=["Hourly", "Daily", "Weekly"], help="M4 Data Group")

    # For debug-series, we accept 0 to mean 'All Data'
    parser.add_argument("--n-series-debug", type=int, dest="n_series_debug",
                        help="Number of series (0 for all, default defined in Config)")

    parser.add_argument("--n-trials", type=int, dest="n_trials", help="Optuna trials")
    parser.add_argument("--n-windows", type=int, dest="n_windows", help="CV windows")

    return parser.parse_args()


def update_config_from_args(config: PipelineConfig, args: argparse.Namespace) -> PipelineConfig:
    """
    Iterates through CLI args and updates the Config object
    only if the argument was explicitly provided (is not None).
    """
    args_dict = vars(args)

    for key, value in args_dict.items():
        # 1. If the user provided a value (value is not None)
        # 2. And that key actually exists in our Config class
        if value is not None and hasattr(config, key):
            setattr(config, key, value)

    # Special logic: If user passed 0 for debug, set config to None (load all data)
    if args.n_series_debug == 0:
        config.n_series_debug = None
    return config
