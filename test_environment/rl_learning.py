import numpy as np
import pandas as pd
from collections import deque
import features as features
import scalers as scaler
import joblib
from sb3_contrib import RecurrentPPO
from sklearn.linear_model import LinearRegression
import logging
import os

logger = logging.getLogger()
    
class RLlearningCloserLong():

    def __init__(self):
        self.sequence_length = 32
        self.seq_buffer = deque(maxlen=self.sequence_length)
        self.seq_buffer_scaled = deque(maxlen=self.sequence_length)

        self.scale_attributes, self.non_scale_attributes, self.all_attributes = features.execution_attributes_long()
        self.fitted_scaler = joblib.load(os.getenv("GENERAL_SCALER_TEST"))
        self.fitted_trend_scaler = joblib.load(os.getenv("TREND_SCALER_TEST"))

        model_path = os.getenv("RL_MODEL_CLOSER_LONG_TEST")

        self.seq_rl_model = RecurrentPPO.load(model_path, device="cpu")
        self.seq_rl_model.policy.set_training_mode(False)

        self.rnn_state = None
        self.episode_start = np.array([True], dtype=bool)

        logger.info(f"Starting to test model: {model_path}")

        self.holding_bars = 0

    def initialize_observation_space(self, feature_data):

        self.rnn_state = None
        self.episode_start[:] = True
        self.holding_bars = 0
        self.seq_buffer.clear()
        self.seq_buffer_scaled.clear()

        trade_info_chunk = pd.DataFrame({
                                    "relative_profit_long": np.zeros((self.sequence_length,), dtype=np.float32), 
                                    "relative_profit_short": np.zeros((self.sequence_length,), dtype=np.float32), 
                                    "bet_type": np.zeros((self.sequence_length,), dtype=np.float32),
                                    "holding_bars": np.zeros((self.sequence_length,), dtype=np.float32)
                                    })
        
        extend_data = self.add_long_and_short_slopes(feature_data.copy())
        
        feature_data_seq = extend_data.iloc[-self.sequence_length:]

        raw_trade_state_features_chunk = trade_info_chunk[self.all_attributes["trade_state"].keys()]

        scaled_data = scaler.transform_data(feature_data_seq, self.fitted_scaler)

        trend_featrues_for_scaling = [
            attribute for attribute, option in self.all_attributes["trend_features"].items()
            if option.get("scale") is True
        ]

        trend_featrues_for_scaling = list(self.fitted_trend_scaler.feature_names_in_)
        scaled_data[trend_featrues_for_scaling] = self.fitted_trend_scaler.transform(scaled_data[trend_featrues_for_scaling])

        self.data_dict = {
                "trade_state": np.array(raw_trade_state_features_chunk, dtype=np.float32),
                "market_dynamics": np.array(scaled_data[self.all_attributes["market_dynamics"].keys()], dtype=np.float32), 
                "momentum_features": np.array(scaled_data[self.all_attributes["momentum_features"].keys()], dtype=np.float32),
                "market_condition": np.array(scaled_data[self.all_attributes["market_condition"].keys()], dtype=np.float32), 
                "trend_features": np.array(scaled_data[self.all_attributes["trend_features"].keys()], dtype=np.float32), 
            }
        
        to_scaled_buffer = pd.DataFrame()
    
        for chunk, columns in zip(self.data_dict.keys(), [list(self.all_attributes["trade_state"].keys()), list(self.all_attributes["market_dynamics"].keys()), 
                                                    list(self.all_attributes["momentum_features"].keys()), list(self.all_attributes["market_condition"].keys()), 
                                                    list(self.all_attributes["trend_features"].keys())
                                                    ]):
            to_scaled_buffer[columns] = self.data_dict[chunk]

        for index, row in to_scaled_buffer.iterrows():
            self.seq_buffer_scaled.append(row)

        print("RL model initialized.")

    def make_rl_prediction(self, feature_data, bet_type, entry_price):
        feature_data = self.add_long_and_short_slopes(feature_data.copy())
        current_close = feature_data["close"].iloc[-2]

        if bet_type == 1:
            relative_profit_long = (current_close - entry_price) / entry_price
            relative_profit_short = 0
            bet_type_encoded = 1
        else:
            relative_profit_short = -((current_close - entry_price) / entry_price)
            relative_profit_long = 0
            bet_type_encoded = 0

        scaled_data = scaler.transform_data(feature_data, self.fitted_scaler)

        trend_featrues_for_scaling = list(self.fitted_trend_scaler.feature_names_in_)
        scaled_data[trend_featrues_for_scaling] = self.fitted_trend_scaler.transform(scaled_data[trend_featrues_for_scaling])

        latest = scaled_data.iloc[-1]

        new_row = latest[self.scale_attributes + ["hour_sin", "hour_cos", "trend_agreement"]].copy()
        new_row["relative_profit_long"] = np.float32(relative_profit_long)
        new_row["relative_profit_short"] = np.float32(relative_profit_short)
        new_row["bet_type"] = np.float32(bet_type_encoded)
        new_row["holding_bars"] = np.float32(np.log1p(self.holding_bars) / np.log1p(100))

        self.seq_buffer_scaled.append(new_row)

        if len(self.seq_buffer_scaled) < self.sequence_length:
            print("Sequence buffer not full yet.")
            return 0

        scaled_obs_df = pd.DataFrame(self.seq_buffer_scaled)

        obs_dict = {
            "trade_state": np.array(scaled_obs_df[list(self.all_attributes["trade_state"].keys())], dtype=np.float32),
            "market_dynamics": np.array(scaled_obs_df[list(self.all_attributes["market_dynamics"].keys())], dtype=np.float32),
            "momentum_features": np.array(scaled_obs_df[list(self.all_attributes["momentum_features"].keys())], dtype=np.float32),
            "market_condition": np.array(scaled_obs_df[list(self.all_attributes["market_condition"].keys())], dtype=np.float32),
            "trend_features": np.array(scaled_obs_df[list(self.all_attributes["trend_features"].keys())], dtype=np.float32),
        }

        obs_batched = {k: v[None, ...] for k, v in obs_dict.items()}

        action, self.rnn_state = self.seq_rl_model.predict(
            obs_batched,
            state=self.rnn_state,
            episode_start=self.episode_start,
            deterministic=True
        )
        self.episode_start[:] = False

        prediction = int(action)

        print(f"RL prediction: {prediction}")
        return prediction

    def add_long_and_short_slopes(self, df):

        def calc_slope(series):
            X = np.arange(len(series)).reshape(-1, 1)
            y = series.values
            return LinearRegression().fit(X, y).coef_[0]

        # Parameters
        long_window = 20
        short_window = 5

        long_slopes = []
        short_slopes = []

        close_series = df["close"].reset_index(drop=True)

        for i in range(len(close_series)):
            if i >= long_window:
                long_slope = calc_slope(close_series[i - long_window:i])
            else:
                long_slope = 0.0

            if i >= short_window:
                short_slope = calc_slope(close_series[i - short_window:i])
            else:
                short_slope = 0.0

            long_slopes.append(long_slope)
            short_slopes.append(short_slope)

        df["long_term_slope"] = long_slopes
        df["short_term_slope"] = short_slopes

        # Δ slope (delta from previous row)
        df["long_term_slope_delta"] = df["long_term_slope"].diff().fillna(0.0)
        df["short_term_slope_delta"] = df["short_term_slope"].diff().fillna(0.0)

        # slope_ema_diff: slope of EMA difference (assumes ema_10 and ema_50 exist)
        if "ema_10" in df.columns and "ema_50" in df.columns:
            df["ema_diff_2"] = df["ema_10"] - df["ema_50"]
            df["slope_ema_diff_2"] = df["ema_diff_2"].rolling(window=5, min_periods=1).apply(calc_slope, raw=False)
        else:
            df["slope_ema_diff_2"] = 0.0  # Fallback if EMAs missing

        # trend_strength_std: rolling std of long-term slope
        df["trend_strength_std"] = df["long_term_slope"].rolling(window=5, min_periods=1).std().fillna(0.0)

        # trend_agreement: binary flag (1 if long and short slopes agree in sign)
        df["trend_agreement"] = (np.sign(df["long_term_slope"]) == np.sign(df["short_term_slope"])).astype(int)

        # slope angles (bounded versions of slope)
        df["long_slope_angle"] = np.arctan(df["long_term_slope"])
        df["short_slope_angle"] = np.arctan(df["short_term_slope"])

        df["momentum_composite_2"] = (
            0.4 * df["vol_weighted_momentum"].fillna(0) +
            0.2 * df["mach_hist"].fillna(0) +
            0.2 * df["roc_30m"].fillna(0) +
            0.2 * df["slope_ema_diff_2"].fillna(0)
        )
                
        df["momentum_composite_scaled"] = (
                df["momentum_composite_2"] - df["momentum_composite_2"].rolling(20).mean()
            ) / (df["momentum_composite_2"].rolling(20).std() + 1e-6)
        df["momentum_positive"] = (df["momentum_composite_scaled"] > 0).astype(int)
        df["momentum_negative"] = (df["momentum_composite_scaled"] < 0).astype(int)

        df["di_plus"] = df["di_plus"] / 100
        df["di_minus"] = df["di_minus"] / 100

        df["di_ratio"] = df["di_plus"] / (df["di_minus"] + 1e-6)

        return df
    
class RLlearningCloserShort():

    def __init__(self):
        self.sequence_length = 32
        self.seq_buffer = deque(maxlen=self.sequence_length)
        self.seq_buffer_scaled = deque(maxlen=self.sequence_length)

        self.scale_attributes, self.non_scale_attributes, self.all_attributes = features.execution_attributes_short()
        self.fitted_scaler = joblib.load(os.getenv("GENERAL_SCALER_TEST"))
        self.fitted_trend_scaler = joblib.load(os.getenv("TREND_SCALER_TEST"))

        model_path = os.getenv("RL_MODEL_CLOSER_SHORT_TEST")

        self.seq_rl_model = RecurrentPPO.load(model_path, device="cpu")
        self.seq_rl_model.policy.set_training_mode(False)

        self.rnn_state = None
        self.episode_start = np.array([True], dtype=bool)

        logger.info(f"Starting to test model: {model_path}")

        self.holding_bars = 0

    def initialize_observation_space(self, feature_data):

        self.rnn_state = None
        self.episode_start[:] = True
        self.holding_bars = 0
        self.seq_buffer.clear()
        self.seq_buffer_scaled.clear()

        trade_info_chunk = pd.DataFrame({
                                    "relative_profit_long": np.zeros((self.sequence_length,), dtype=np.float32), 
                                    "relative_profit_short": np.zeros((self.sequence_length,), dtype=np.float32), 
                                    "bet_type": np.zeros((self.sequence_length,), dtype=np.float32),
                                    "holding_bars": np.zeros((self.sequence_length,), dtype=np.float32)
                                    })
        
        extend_data = self.add_long_and_short_slopes(feature_data.copy())
        
        feature_data_seq = extend_data.iloc[-self.sequence_length:]

        raw_trade_state_features_chunk = trade_info_chunk[self.all_attributes["trade_state"].keys()]

        scaled_data = scaler.transform_data(feature_data_seq, self.fitted_scaler)

        trend_featrues_for_scaling = [
            attribute for attribute, option in self.all_attributes["trend_features"].items()
            if option.get("scale") is True
        ]

        trend_featrues_for_scaling = list(self.fitted_trend_scaler.feature_names_in_)
        scaled_data[trend_featrues_for_scaling] = self.fitted_trend_scaler.transform(scaled_data[trend_featrues_for_scaling])

        self.data_dict = {
                "trade_state": np.array(raw_trade_state_features_chunk, dtype=np.float32),
                "market_dynamics": np.array(scaled_data[self.all_attributes["market_dynamics"].keys()], dtype=np.float32), 
                "momentum_features": np.array(scaled_data[self.all_attributes["momentum_features"].keys()], dtype=np.float32),
                "market_condition": np.array(scaled_data[self.all_attributes["market_condition"].keys()], dtype=np.float32), 
                "trend_features": np.array(scaled_data[self.all_attributes["trend_features"].keys()], dtype=np.float32), 
            }
        
        to_scaled_buffer = pd.DataFrame()
    
        for chunk, columns in zip(self.data_dict.keys(), [list(self.all_attributes["trade_state"].keys()), list(self.all_attributes["market_dynamics"].keys()), 
                                                    list(self.all_attributes["momentum_features"].keys()), list(self.all_attributes["market_condition"].keys()), 
                                                    list(self.all_attributes["trend_features"].keys())
                                                    ]):
            to_scaled_buffer[columns] = self.data_dict[chunk]

        for index, row in to_scaled_buffer.iterrows():
            self.seq_buffer_scaled.append(row)

        print("RL model initialized.")

    def make_rl_prediction(self, feature_data, bet_type, entry_price):
        feature_data = self.add_long_and_short_slopes(feature_data.copy())
        current_close = feature_data["close"].iloc[-2]

        if bet_type == 1:
            relative_profit_long = (current_close - entry_price) / entry_price
            relative_profit_short = 0
            bet_type_encoded = 1
        else:
            relative_profit_short = -((current_close - entry_price) / entry_price)
            relative_profit_long = 0
            bet_type_encoded = 0

        scaled_data = scaler.transform_data(feature_data, self.fitted_scaler)

        trend_featrues_for_scaling = list(self.fitted_trend_scaler.feature_names_in_)
        scaled_data[trend_featrues_for_scaling] = self.fitted_trend_scaler.transform(scaled_data[trend_featrues_for_scaling])

        latest = scaled_data.iloc[-1]

        new_row = latest[self.scale_attributes + ["hour_sin", "hour_cos", "trend_agreement"]].copy()
        new_row["relative_profit_long"] = np.float32(relative_profit_long)
        new_row["relative_profit_short"] = np.float32(relative_profit_short)
        new_row["bet_type"] = np.float32(bet_type_encoded)
        new_row["holding_bars"] = np.float32(np.log1p(self.holding_bars) / np.log1p(100))

        self.seq_buffer_scaled.append(new_row)

        if len(self.seq_buffer_scaled) < self.sequence_length:
            print("Sequence buffer not full yet.")
            return 0

        scaled_obs_df = pd.DataFrame(self.seq_buffer_scaled)

        obs_dict = {
            "trade_state": np.array(scaled_obs_df[list(self.all_attributes["trade_state"].keys())], dtype=np.float32),
            "market_dynamics": np.array(scaled_obs_df[list(self.all_attributes["market_dynamics"].keys())], dtype=np.float32),
            "momentum_features": np.array(scaled_obs_df[list(self.all_attributes["momentum_features"].keys())], dtype=np.float32),
            "market_condition": np.array(scaled_obs_df[list(self.all_attributes["market_condition"].keys())], dtype=np.float32),
            "trend_features": np.array(scaled_obs_df[list(self.all_attributes["trend_features"].keys())], dtype=np.float32),
        }

        obs_batched = {k: v[None, ...] for k, v in obs_dict.items()}

        action, self.rnn_state = self.seq_rl_model.predict(
            obs_batched,
            state=self.rnn_state,
            episode_start=self.episode_start,
            deterministic=True
        )
        self.episode_start[:] = False

        prediction = int(action)

        print(f"RL prediction: {prediction}")
        return prediction

    def add_long_and_short_slopes(self, df):

        def calc_slope(series):
            X = np.arange(len(series)).reshape(-1, 1)
            y = series.values
            return LinearRegression().fit(X, y).coef_[0]

        # Parameters
        long_window = 20
        short_window = 5

        long_slopes = []
        short_slopes = []

        close_series = df["close"].reset_index(drop=True)

        for i in range(len(close_series)):
            if i >= long_window:
                long_slope = calc_slope(close_series[i - long_window:i])
            else:
                long_slope = 0.0

            if i >= short_window:
                short_slope = calc_slope(close_series[i - short_window:i])
            else:
                short_slope = 0.0

            long_slopes.append(long_slope)
            short_slopes.append(short_slope)

        df["long_term_slope"] = long_slopes
        df["short_term_slope"] = short_slopes

        # Δ slope (delta from previous row)
        df["long_term_slope_delta"] = df["long_term_slope"].diff().fillna(0.0)
        df["short_term_slope_delta"] = df["short_term_slope"].diff().fillna(0.0)

        # slope_ema_diff: slope of EMA difference (assumes ema_10 and ema_50 exist)
        if "ema_10" in df.columns and "ema_50" in df.columns:
            df["ema_diff_2"] = df["ema_10"] - df["ema_50"]
            df["slope_ema_diff_2"] = df["ema_diff_2"].rolling(window=5, min_periods=1).apply(calc_slope, raw=False)
        else:
            df["slope_ema_diff_2"] = 0.0  # Fallback if EMAs missing

        # trend_strength_std: rolling std of long-term slope
        df["trend_strength_std"] = df["long_term_slope"].rolling(window=5, min_periods=1).std().fillna(0.0)

        # trend_agreement: binary flag (1 if long and short slopes agree in sign)
        df["trend_agreement"] = (np.sign(df["long_term_slope"]) == np.sign(df["short_term_slope"])).astype(int)

        # slope angles (bounded versions of slope)
        df["long_slope_angle"] = np.arctan(df["long_term_slope"])
        df["short_slope_angle"] = np.arctan(df["short_term_slope"])

        df["momentum_composite_2"] = (
            0.4 * df["vol_weighted_momentum"].fillna(0) +
            0.2 * df["mach_hist"].fillna(0) +
            0.2 * df["roc_30m"].fillna(0) +
            0.2 * df["slope_ema_diff_2"].fillna(0)
        )
                
        df["momentum_composite_scaled"] = (
                df["momentum_composite_2"] - df["momentum_composite_2"].rolling(20).mean()
            ) / (df["momentum_composite_2"].rolling(20).std() + 1e-6)
        df["momentum_positive"] = (df["momentum_composite_scaled"] > 0).astype(int)
        df["momentum_negative"] = (df["momentum_composite_scaled"] < 0).astype(int)

        df["di_plus"] = df["di_plus"] / 100
        df["di_minus"] = df["di_minus"] / 100

        df["di_ratio"] = df["di_plus"] / (df["di_minus"] + 1e-6)

        return df