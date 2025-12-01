from typing import Tuple

import pandas as pd
from datasetsforecast.m4 import M4
from m4_forecasting.config import PipelineConfig
from m4_forecasting import LOGGER_NAME
import logging

logger = logging.getLogger(LOGGER_NAME)

class M4DataLoader:
    def __init__(self, config: PipelineConfig):
        self.config = config

    def load(self) -> pd.DataFrame:
        """Downloads and loads the M4 data."""
        logger.info(f"Loading M4 data (Group: {self.config.group})...")

        # load() returns a tuple (df, group_info, ...), we just need df
        df, *_ = M4.load(directory=str(self.config.data_dir), group=self.config.group)

        # --- FIX START ---
        # Convert 'ds' from object/int to datetime to allow frequency operations
        # The M4 loader might sometimes return integers or strings depending on version
        if df['ds'].dtype == 'object' or pd.api.types.is_integer_dtype(df['ds']):
            # M4 is relative time, but for this demo we often map it to arbitrary dates
            start = pd.Timestamp("2000-01-01 00:00:00")

            df["ds"] = start + pd.to_timedelta(df["ds"], unit="h")

        # Ensure correct sorting for temporal splitting
        df = df.sort_values(by=['unique_id', 'ds']).reset_index(drop=True)

        if self.config.n_series_debug:
            logger.info(f"Subsetting to first {self.config.n_series_debug} series for debugging.")
            uids = df['unique_id'].unique()[:self.config.n_series_debug]
            df = df[df['unique_id'].isin(uids)].reset_index(drop=True)

        logger.info(f"Data loaded successfully. Total rows: {len(df)}")
        return df

    def split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Splits data into training and hold-out test sets using group-aware indexing.
        Returns: (train_df, test_df)
        """
        logger.info(f"Splitting data (Test size: {self.config.test_size})...")
        # Take the last `test_size` rows per group for testing
        test_df = df.groupby('unique_id').tail(self.config.test_size).copy()

        # Take everything except the last `test_size` rows for training
        # FAST / MODERN PANDAS
        train_df = df.groupby('unique_id').head(-self.config.test_size).reset_index(drop=True)

        logger.info(f"Split complete. Train rows: {len(train_df)}, Test rows: {len(test_df)}")
        return train_df, test_df