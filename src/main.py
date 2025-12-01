from typing import Dict, Any
import pickle

import pandas as pd
# --- NEW IMPORT ---
from statsforecast import StatsForecast
from statsforecast.models import SeasonalNaive
from utilsforecast.evaluation import evaluate

from src.m4_forecasting.config import PipelineConfig
from src.m4_forecasting.data import M4DataLoader
from src.m4_forecasting.trainer import ForecastingEngine
from src.m4_forecasting.utils import setup_logging, keep_loggers, parse_args, update_config_from_args


def main():
    # Setup loggers
    logger = setup_logging()
    keep_loggers(["m4_forecasting", "optuna"])

    # Load config and parse input arguments
    config = PipelineConfig()
    args = parse_args()
    config: PipelineConfig = update_config_from_args(config, args)

    # print config
    logger.info(f"Config: \n{config}")

    # ---------------------------------------------------------
    # Load the data
    # ---------------------------------------------------------
    loader = M4DataLoader(config)
    df = loader.load()
    train_df, test_df = loader.split(df)

    # ---------------------------------------------------------
    # Train & Predict: Advanced ML Model (AutoML)
    # ---------------------------------------------------------
    engine = ForecastingEngine(config)
    engine.train(train_df)
    engine.visualize_optimization()  # Optional

    # Returns DataFrame with columns: [unique_id, ds, AutoLightGBM]
    y_pred_ml = engine.predict()

    # ---------------------------------------------------------
    # Train & Predict: Naive Baselines
    # ---------------------------------------------------------
    logger.info("Generating Baseline forecasts...")

    # We use StatsForecast for classical/statistical baselines
    sf = StatsForecast(
        models=[
            SeasonalNaive(season_length=24)  # Last day (Strong baseline)
        ],
        freq=config.freq
    )

    sf.fit(train_df)
    y_pred_baseline: pd.DataFrame = sf.predict(h=config.horizon)

    # ---------------------------------------------------------
    # Merge & Evaluate
    # ---------------------------------------------------------
    # Merge ML predictions with Baseline predictions
    # y_pred_ml: [id, ds, AutoLightGBM]
    # y_pred_baseline: [id, ds, Naive, SeasonalNaive]
    all_preds: pd.DataFrame = y_pred_ml.merge(y_pred_baseline, on=['unique_id', 'ds'], how='left')

    # Get Actuals
    y_true: pd.DataFrame = test_df.groupby('unique_id').head(config.horizon).reset_index(drop=True)

    # Merge everything for evaluation
    eval_df: pd.DataFrame = all_preds.merge(
        y_true[['unique_id', 'ds', 'y']],
        on=['unique_id', 'ds'],
        how='inner'
    )

    logger.info("Evaluating models...")
    results: pd.DataFrame = evaluate(df=eval_df, metrics=config.metrics)

    # Show the Comparison Table
    summary: pd.DataFrame = results.drop(columns=['unique_id']).groupby('metric').mean().reset_index()

    logger.info("\n" + "=" * 40)
    logger.info(" FINAL LEADERBOARD (Lower is Better)")
    logger.info("=" * 40)
    logger.info(f"\n{summary}")
    logger.info("=" * 40)

    # ---------------------------------------------------------
    # PRODUCTION RETRAINING
    # ---------------------------------------------------------
    logger.info("\n" + "=" * 40)
    logger.info(" FINALIZING FOR PRODUCTION")
    logger.info("=" * 40)

    # Print the specific 'winning' recipe
    best_params: Dict[str, Any] = engine.get_best_params()
    logger.info(f"Winning Hyperparameters:\n{best_params}")

    # Retrain on the FULL dataset (df) - combining Train + Test
    # This ensures the model knows the most recent trends before going live.
    production_model = engine.train_production_model(df)

    # ---------------------------------------------------------
    # Save Production Artifact
    # ---------------------------------------------------------
    # We save the MLForecast object, not the AutoML wrapper
    with open(config.model_path, 'wb') as f:
        pickle.dump(production_model, f)

    logger.info(f"Production model saved to: {config.model_path}")
    logger.info("--- Pipeline Completed Successfully ---")


if __name__ == "__main__":
    main()
