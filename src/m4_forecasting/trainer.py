import logging
import pickle

import lightgbm as lgb
import optuna
import optuna.visualization as ov
import pandas as pd
from mlforecast import MLForecast
from mlforecast.auto import AutoMLForecast, AutoLightGBM
from mlforecast.lag_transforms import RollingMean, ExponentiallyWeightedMean
from mlforecast.target_transforms import Differences

from m4_forecasting import LOGGER_NAME
from m4_forecasting.config import PipelineConfig

logger = logging.getLogger(LOGGER_NAME)


class ForecastingEngine:
    """
    Orchestrates the time-series forecasting lifecycle including hyperparameter optimization,
    model training, production retraining, and visualization.

    This engine utilizes `mlforecast.AutoMLForecast` to search for the optimal
    feature engineering configurations (lags, transforms) and LightGBM hyperparameters.

    Attributes:
        config (PipelineConfig): Configuration object containing paths, frequency, and training parameters.
        model (AutoMLForecast or MLForecast): The internal model object. Starts as None, populates after training.
    Example:
            >>> from m4_forecasting.config import PipelineConfig
            >>> import pandas as pd
            >>>
            >>> # 1. Setup Configuration
            >>> config = PipelineConfig(
            ...     freq='H',
            ...     n_trials=20,
            ...     horizon=24,
            ...     n_windows=2,
            ...     model_dir='artifacts',
            ...     model_path='artifacts/model.pkl',
            ...     num_threads=4
            ... )
            >>>
            >>> # 2. Prepare Data (unique_id, ds, y columns required)
            >>> df = pd.DataFrame({
            ...     'unique_id': ['ts_1'] * 100,
            ...     'ds': pd.date_range(start='2023-01-01', periods=100, freq='H'),
            ...     'y': range(100)
            ... })
            >>>
            >>> # 3. Initialize and Train
            >>> engine = ForecastingEngine(config)
            >>> engine.train(df)  # Runs Optuna optimization
            >>>
            >>> # 4. Generate Forecast
            >>> forecast = engine.predict(horizon=24)
            >>> print(forecast.head())
    """

    def __init__(self, config: PipelineConfig):
        """
        Initializes the ForecastingEngine with a specific configuration.

        Args:
            config (PipelineConfig): Configuration dataclass containing trial counts,
                                     file paths, and time series settings.
        """
        self.config = config
        self.model = None

    def get_best_params(self) -> dict:
        """
        Retrieves the optimal hyperparameters identified during the optimization phase.

        Returns:
            dict: A dictionary of best parameters (e.g., {'learning_rate': 0.01, 'lag_type': 'short'}).

        Raises:
            ValueError: If the `train()` method has not been executed yet.
        """
        if not self.model or not hasattr(self.model, 'results_'):
            raise ValueError("Run train() first.")

        # We assume 'lgb' is the model name we used
        return self.model.results_['lgb'].best_params

    def train_production_model(self, full_df: pd.DataFrame) -> MLForecast:
        """
        Trains the final production model using the best parameters found during optimization
        on the entire available dataset (Train + Test).

        This reconstruction step is necessary because `AutoMLForecast` splits data for validation,
        whereas the production model must learn from the most recent data points available.

        Args:
            full_df (pd.DataFrame): The complete historical dataset (ID, Date, Target).

        Returns:
            MLForecast: A fitted `MLForecast` object ready for inference.
        """
        logger.info("--- Starting Production Retraining (Full Dataset) ---")

        # 1. Get the Best Params
        params = self.get_best_params()
        logger.info(f"Best Params Found: {params}")

        # 2. Reconstruct Model Params (LightGBM)
        # We filter out params that belong to feature engineering (lag_type, roll_window, etc)
        # to leave only the LightGBM specific ones.
        lgb_params = {
            k: v for k, v in params.items()
            if k not in ['lag_type', 'roll_window', 'ewm_alpha']
        }

        # Instantiate the final regressor
        final_model = lgb.LGBMRegressor(**lgb_params, verbosity=-1)

        # 3. Reconstruct Feature Engineering Params
        # We must replicate the logic from _feature_search_space
        if params['lag_type'] == 'short':
            lags = [1, 24]
        else:
            lags = [1, 24, 168]

        lag_transforms = {
            1: [RollingMean(window_size=params['roll_window'])],
            24: [ExponentiallyWeightedMean(alpha=params['ewm_alpha'])],
        }

        # 4. Create the Production Pipeline
        fcst = MLForecast(
            models=[final_model],
            freq=self.config.freq,
            lags=lags,
            lag_transforms=lag_transforms,
            date_features=['hour', 'dayofweek'],
            target_transforms=[Differences([24])]
        )

        # 5. Fit on ALL data
        fcst.fit(full_df)
        logger.info("Production model trained on full history.")

        return fcst

    @staticmethod
    def _feature_search_space(trial: optuna.Trial) -> dict:
        """
        Defines the Optuna search space for Feature Engineering parameters.

        This includes:
        - Lag selection strategies.
        - Window sizes for rolling means.
        - Alpha decay rates for Exponentially Weighted Means.

        Args:
            trial (optuna.Trial): The current Optuna trial object.

        Returns:
            dict: Configuration dictionary compatible with MLForecast's auto-config.
        """
        lag_type = trial.suggest_categorical('lag_type', ['short', 'long'])
        lags = [1, 24] if lag_type == 'short' else [1, 24, 168]

        roll_window = trial.suggest_int('roll_window', 24, 168, step=24)
        alpha = trial.suggest_float('ewm_alpha', 0.1, 0.9)

        return {
            "lags": lags,
            "lag_transforms": {
                1: [RollingMean(window_size=roll_window)],
                24: [ExponentiallyWeightedMean(alpha=alpha)],
            },
            "date_features": ['hour', 'dayofweek'],
            "target_transforms": [Differences([24])]
        }

    @staticmethod
    def _model_search_space(trial: optuna.Trial) -> dict:
        """
        Defines the Optuna search space for LightGBM hyperparameters.

        Args:
            trial (optuna.Trial): The current Optuna trial object.

        Returns:
            dict: Dictionary of LightGBM hyperparameters to test.
        """
        return {
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.5, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 31, 512),
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=50),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 100),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "verbose": -1  # Silence LightGBM internal logs
        }

    def train(self, train_df: pd.DataFrame):
        """
        Executes the AutoML optimization loop.

        This process:
        1. Defines the search spaces for both features and the model.
        2. Uses `AutoMLForecast` to run cross-validation over `n_windows`.
        3. Optimizes using Optuna to minimize the loss (defaulting here to RMSE/default unless specified).

        Args:
            train_df (pd.DataFrame): The training subset of the data.
        """
        logger.info(f"Initializing AutoML (Trials: {self.config.n_trials})...")

        # We pass our custom model search space into AutoLightGBM
        # This overwrites the default 'fixed' learning rate logic
        models = {
            'lgb': AutoLightGBM(config=self._model_search_space)
        }

        self.model = AutoMLForecast(
            models=models,
            freq=self.config.freq,
            init_config=self._feature_search_space,  # This configures the features (lags)
            num_threads=self.config.num_threads,
        )

        self.model.fit(
            train_df,
            n_windows=self.config.n_windows,
            h=self.config.horizon,
            num_samples=self.config.n_trials,
            # loss=smape
        )
        logger.info("AutoML Optimization and Training finished.")

    def predict(self, horizon: int = None) -> pd.DataFrame:
        """
        Generates forecasts using the trained model.

        Args:
            horizon (int, optional): The number of steps to predict into the future.
                                     Defaults to the horizon specified in the config.

        Returns:
            pd.DataFrame: DataFrame containing IDs, dates, and predicted values.
        """
        h = horizon if horizon else self.config.horizon
        return self.model.predict(h=h)

    def visualize_optimization(self):
        """
        Generates and saves interactive Plotly HTML visualizations of the optimization process.

        Plots generated:
        - Optimization History: How the objective value improved over trials.
        - Parameter Importance: Which hyperparameters had the biggest impact.
        - Interaction Heatmap: Specifically for `num_leaves` vs `n_estimators`.

        Outputs are saved to `{config.model_dir}/plots`.
        """
        if not self.model or not hasattr(self.model, 'results_'):
            logger.warning("No optimization results found. Did you run train()?")
            return

        plots_dir = self.config.model_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Generating optimization plots in {plots_dir}...")

        logger.info(f"Available params: {self.model.results_['lgb'].best_trial.params.keys()}")
        for model_name, study in self.model.results_.items():
            try:
                # Standard Plots
                ov.plot_optimization_history(study).write_html(plots_dir / f"{model_name}_history.html")
                ov.plot_param_importances(study).write_html(plots_dir / f"{model_name}_importance.html")

                # Heatmap (Contour)
                try:
                    target_params = ['num_leaves', 'n_estimators']

                    # --- FIX START ---
                    # Ensure trials exist
                    if len(study.trials) == 0:
                        logger.warning(f"No trials found for {model_name}, skipping heatmap.")
                        continue

                    # study.trials[0].params is a dict {'param_name': value}
                    # We just need to check if our targets are in the keys
                    available_params = study.trials[0].params.keys()

                    if all(p in available_params for p in target_params):
                        fig_heat = ov.plot_contour(study, params=target_params)
                        fig_heat.update_layout(title=f"Interaction: Rolling Window vs Alpha ({model_name})")
                        fig_heat.write_html(plots_dir / f"{model_name}_heatmap_features.html")
                        logger.info(f"Generated feature interaction heatmap for {model_name}")
                    else:
                        logger.warning(
                            f"Params {target_params} not found in study. Available: {list(available_params)}")
                    # --- FIX END ---

                except Exception as e:
                    logger.warning(f"Failed to generate heatmap for {model_name}: {e}")

            except Exception as e:
                logger.error(f"Failed to generate generic plots for {model_name}: {e}")

    def save(self):
        """
        Serializes the best performing model artifact to the path specified in the config.
        """
        # ... (Save logic remains the same) ...
        best_artifact = self.model.models_['lgb']
        with open(self.config.model_path, 'wb') as f:
            pickle.dump(best_artifact, f)
        logger.info(f"Best model saved to: {self.config.model_path}")