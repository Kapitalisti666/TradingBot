import numpy as np
import tensorflow as tf
from tensorflow import keras
import scalers as scaler
import joblib
from sklearn.linear_model import LinearRegression
import os
import logging
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger()

class SupervisedLearning():

    def __init__(self):

        self.fitted_scaler = joblib.load(os.getenv("GENERAL_SCALER_TEST"))
        self.fitted_slope_scaler = joblib.load(os.getenv("SLOPE_SCALER_TEST"))
        self.fitted_trend_scaler = joblib.load(os.getenv("TREND_SCALER_TEST"))

        self.stage_1_long = keras.models.load_model(os.getenv("SL_MODEL_BINARY_LONG_TEST"), compile = False)
        self.stage_1_short = keras.models.load_model(os.getenv("SL_MODEL_BINARY_SHORT_TEST"), compile = False)
        self.stage_2_direction = keras.models.load_model(os.getenv("SL_MODEL_DIRECTION_TEST"), compile = False)
    
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
            raise("Missing values!!")

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

        df[[
            "long_term_slope",
            "short_term_slope",
            "long_term_slope_delta",
            "short_term_slope_delta",
            "ema_diff_2",
            "slope_ema_diff_2",
            "trend_strength_std",
            "long_slope_angle",
            "short_slope_angle",
            "momentum_composite_2",
            "momentum_composite_scaled",
            "MA_Crossover"
        ]] = self.fitted_trend_scaler.transform(df[[
            "long_term_slope",
            "short_term_slope",
            "long_term_slope_delta",
            "short_term_slope_delta",
            "ema_diff_2",
            "slope_ema_diff_2",
            "trend_strength_std",
            "long_slope_angle",
            "short_slope_angle",
            "momentum_composite_2",
            "momentum_composite_scaled",
            "MA_Crossover"
        ]])

        return df

    def single_stage_prediction(self):

        market_dynamics_micro = ['spread_pct', 'tick_volume', 'slope_ema_10', 'close']
        momentum_features_micro = ['macd_diff', 'momentum_composite_2', "MA_Crossover", "roc_30m"]
        market_condition_micro = ['atr', 'bb_%b', 'volatility_rolling_std', 'bb_width']
        trend_features_trend_entry = ['adosc', 'vwap', 'obv', "long_term_slope", "long_slope_angle"]

        market_dynamics_micro = self.scaled_data[market_dynamics_micro].iloc[-32:].to_numpy(dtype=np.float32)
        market_dynamics_micro = tf.convert_to_tensor(np.expand_dims(market_dynamics_micro, axis=0), dtype=tf.float32)

        momentum_features_micro = self.scaled_data[momentum_features_micro].iloc[-1:].to_numpy(dtype=np.float32)
        momentum_features_micro = tf.convert_to_tensor(momentum_features_micro, dtype=tf.float32)

        market_condition_micro = self.scaled_data[market_condition_micro].iloc[-1:].to_numpy(dtype=np.float32)
        market_condition_micro = tf.convert_to_tensor(market_condition_micro, dtype=tf.float32)

        trend_features_trend_entry = self.scaled_data[trend_features_trend_entry].iloc[-1:].to_numpy(dtype=np.float32)
        trend_features_trend_entry = tf.convert_to_tensor(trend_features_trend_entry, dtype=tf.float32)

        prediction_long = self.stage_1_long.predict([market_dynamics_micro, momentum_features_micro, market_condition_micro, trend_features_trend_entry]).item()
        prediction_short = self.stage_1_short.predict([market_dynamics_micro, momentum_features_micro, market_condition_micro, trend_features_trend_entry]).item()

        direction_pred = self.stage_2_direction.predict([market_dynamics_micro, momentum_features_micro, market_condition_micro, trend_features_trend_entry])

        predictions = {
            "long": prediction_long,
            "short": prediction_short,
            "long_dir": direction_pred[0][1],
            "short_dir": direction_pred[0][0]
        }

        return predictions

    def make_prediction(self, data):

        self.data = self.add_long_and_short_slopes(data.copy())
        self.scaled_data = scaler.transform_data(self.data.copy(), self.fitted_scaler)

        if not hasattr(self, "previous_meta_bid"):
            self.previous_meta_bid = 0
            self.probs_long = 0
            self.probs_short = 0

        bid = self.single_stage_predictions()
        #bid = self.single_stage_predictions_with_direction()

        probs = {
            "long": self.probs_long,
            "short": self.probs_short,
        }

        return bid, probs
        
    def single_stage_predictions(self):

        predictions = self.single_stage_prediction()

        self.probs_long = predictions["long"]
        self.probs_short = predictions["short"]

        if predictions["long"] < 0.5 and self.previous_meta_bid == 1:
            self.previous_meta_bid = 0

        if predictions["short"] < 0.5 and self.previous_meta_bid == -1:
            self.previous_meta_bid = 0
        
        if predictions["long"] > 0.7 and predictions["short"] < 0.5 and self.previous_meta_bid != 1:
            self.previous_meta_bid = 1
            return 1
        elif predictions["short"] > 0.7 and predictions["long"] < 0.5 and self.previous_meta_bid != -1:
            self.previous_meta_bid = -1
            return -1
        else:
            return 0
        
    def single_stage_predictions_with_direction(self):

        predictions = self.single_stage_prediction()

        self.probs_long = predictions["long"]
        self.probs_short = predictions["short"]

        if predictions["long"] < 0.5 and self.previous_meta_bid == 1:
            self.previous_meta_bid = 0

        if predictions["short"] < 0.5 and self.previous_meta_bid == -1:
            self.previous_meta_bid = 0
        
        if predictions["long"] > 0.7 and predictions["long_dir"] > 0.7 and self.previous_meta_bid != 1:
            self.previous_meta_bid = 1
            return 1
        elif predictions["short"] > 0.7 and predictions["short_dir"] > 0.7 and self.previous_meta_bid != -1:
            self.previous_meta_bid = -1
            return -1
        else:
            return 0
    