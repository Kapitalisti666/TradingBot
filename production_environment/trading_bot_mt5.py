import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import datetime
import tensorflow as tf
import logging
import features as features
import joblib
import scalers as scaler
from collections import deque, Counter
from sb3_contrib import RecurrentPPO
import torch.nn as nn
import torch as th
from tensorflow import keras
from sklearn.linear_model import LinearRegression
import time
import os
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger()

class Trading_bot:

    def __init__(self, login: int, server: str, password: str, symbol: str, volume_bet = 1.0, closing_time = None) -> None:

        self.mt5 = mt5

        self.mt5.initialize()

        if not self.mt5.initialize():
            print("initialize() failed, error code =", self.mt5.last_error())
            self.mt5.shutdown()

        self.mt5.login(login, password, server)

        self.symbol = symbol
        self.volume_bet = volume_bet

        self.buy = False
        self.sell = False

        self.tp = 0
        self.sl = 0

        self.stop_los = 0
        self.take_profit = 0

        self.bet_placed = False
        self.direction = None

        self.sell_price = None
        self.buy_price = None

        self.existing_position_for_console_print = None

        self.stop = False
        self.closing_time = closing_time
        self.counter = 0
        
        self.transaction_error= None

        self.make_ai_prediction = False

        self.env = None
        self.first_rl_run = True
        self.rl_model = None
        self.rl_observation = None
        self.rl_action = None
        self.rl_prediction_to_console = None
        self.reward = 0
        self.entry_price = 0

        self.rl_model_closer_long = RecurrentPPO.load(os.getenv("RL_MODEL_CLOSER_LONG"), device="cpu")
        self.rl_model_closer_long.policy.set_training_mode(False)
        self.episode_start_long = np.array([True], dtype=bool)

        self.rl_model_closer_short = RecurrentPPO.load(os.getenv("RL_MODEL_CLOSER_SHORT"), device="cpu")
        self.rl_model_closer_short.policy.set_training_mode(False)
        self.episode_start_short = np.array([True], dtype=bool)

        self.feature_data = None
        self.fitted_scaler = joblib.load(os.getenv("GENERAL_SCALER"))
        self.fitted_slope_scaler = joblib.load(os.getenv("SLOPE_SCALER"))
        self.fitted_trend_scaler = joblib.load(os.getenv("TREND_SCALER"))

        self.single_stage_long = keras.models.load_model(os.getenv("SL_MODEL_BINARY_LONG"), compile = False)
        self.single_stage_short = keras.models.load_model(os.getenv("SL_MODEL_BINARY_SHORT"), compile = False)
        self.single_stage_2_direction = keras.models.load_model(os.getenv("SL_MODEL_DIRECTION"), compile = False)

        self.order_comment = None
        self.permission_to_trade = False
        self.penalty_time = None

        self.long_slope = 0
        self.short_slope = 0
        self.long_trend = None
        self.short_trend = None
        self.increasing_trend = True
        self.decreasing_trend = True

        self.first_run = True
        self.previous_order = None
        self.check_existing_positions_start = True

        self.initialize_rl = True

        self.permissions = {
            "long": False,
            "short": False
        }

        self.profits = []

    def check_existing_position(self):

        account_info = self.mt5.account_info()
        self.trade_allowed = account_info.trade_allowed
        positions = self.mt5.positions_get(symbol=self.symbol)
    
        if positions:
            positions_dataframe = pd.DataFrame(positions, columns=positions[0]._asdict().keys())
            existing_position = positions_dataframe['volume'].iloc[0]
            position_placed_price = positions_dataframe["price_open"].iloc[0]
            order_type = positions_dataframe["type"].iloc[0]

            if order_type == 0:
                self.buy = True
            elif order_type == 1:
                self.sell = True

            self.entry_price = position_placed_price
            self.order_entry_price = position_placed_price

            if self.check_existing_positions_start:
                self.initialize_rl = True
                self.initialize_closer_rl = True
                logger.info(f"Existing position found from the start, data will be initialized.")

            self.check_existing_positions_start = False

            return True
        else:
            self.check_existing_positions_start = False
            return False
        
    def additional_rows(self, dataframe, first_row, amount_of_additonal_rows):
        find_first_row = dataframe[dataframe["time"] == first_row["time"]]
        target_index = find_first_row.index[0]
        start_index = target_index - amount_of_additonal_rows  
        additional_rows = dataframe.iloc[start_index:target_index].copy()

        return additional_rows
        
    def trim_data(self, data):

        df = pd.DataFrame(data)
        df['time']=pd.to_datetime(df['time'], unit='s')

        current_date = df["time"].iloc[-1].date()
        df_today = df[df["time"].dt.date == current_date].copy()

        extension = self.additional_rows(df, df_today.iloc[0], 220)

        final_df = pd.concat([extension, df_today])

        return final_df, current_date
    
    def get_features(self):
        data = self.mt5.copy_rates_from_pos(self.symbol, self.mt5.TIMEFRAME_M5, 1, 10000)
        df, current_date = self.trim_data(data)
        expanded_df = features.execution_features(df)
        expanded_df.dropna(inplace=True)

        final_dataframe_filtered = expanded_df[(expanded_df['time'].dt.date == current_date) & (expanded_df['time'].dt.time >= pd.to_datetime('02:00').time())]
        final_dataframe_filtered.reset_index(drop=True, inplace=True)
        return final_dataframe_filtered
    
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
    
    def rl_closer_long(self):
        
        scale_attributes, non_scale_attributes, all_attributes = features.execution_attributes_long_new()
        current_close = self.feature_data["close"].iloc[-1]

        if self.buy:
            relative_profit_long = (current_close - self.entry_price) / self.entry_price
            relative_profit_short = 0
        else:
            relative_profit_short = -((current_close - self.entry_price) / self.entry_price)
            relative_profit_long = 0
   
        logger.info(f"RL model activated")

        sequence_length = 32

        if not hasattr(self, "seq_buffer_long"):
            self.seq_buffer_long = deque(maxlen=sequence_length)
        
        if self.initialize_closer_rl:

            logger.info(f"First round, the data will be initialized.")

            self.initialize_closer_rl = False
            self.rnn_state = None
            self.episode_start_long[:] = True
            self.holding_bars = 0
            self.seq_buffer_long.clear()

            trade_info_chunk = pd.DataFrame({"relative_profit_long": np.zeros((sequence_length,), dtype=np.float32), 
                                             "holding_bars": np.zeros((sequence_length,), dtype=np.float32)
                                             })
            
            trade_info_chunk.iloc[-1] = {
            "relative_profit_long": np.float32(relative_profit_long),
            "holding_bars": np.float32(np.array(np.log1p(self.holding_bars) / np.log1p(100))),
            }
            
            feature_data_seq = self.scaled_data.iloc[-sequence_length:]
            feature_data_seq.reset_index(drop=True, inplace=True)

            scaled_data = feature_data_seq

            self.data_dict_long = {
                    "trade_state": np.array(trade_info_chunk, dtype=np.float32),
                    "market_dynamics": np.array(scaled_data[all_attributes["market_dynamics"].keys()], dtype=np.float32), 
                    "momentum_features": np.array(scaled_data[all_attributes["momentum_features"].keys()], dtype=np.float32),
                    "market_condition": np.array(scaled_data[all_attributes["market_condition"].keys()], dtype=np.float32), 
                    "trend_features": np.array(scaled_data[all_attributes["trend_features"].keys()], dtype=np.float32), 
                   }
            
            to_scaled_buffer = pd.DataFrame()
        
            for chunk, columns in zip(self.data_dict_long.keys(), [list(all_attributes["trade_state"].keys()), list(all_attributes["market_dynamics"].keys()), 
                                                        list(all_attributes["momentum_features"].keys()), list(all_attributes["market_condition"].keys()), 
                                                        list(all_attributes["trend_features"].keys())
                                                        ]):
                to_scaled_buffer[columns] = self.data_dict_long[chunk]

            for index, row in to_scaled_buffer.iterrows():
                self.seq_buffer_long.append(row)

        else:

            self.holding_bars += 1

            scaled_data = self.scaled_data

            latest_values = scaled_data.iloc[-1]

            new_row = latest_values[scale_attributes + ["trend_agreement"]].copy().to_frame().T
            
            new_row["relative_profit_long"] = np.array(relative_profit_long, dtype=np.float32)
            new_row["holding_bars"] = np.float32(np.array(np.log1p(self.holding_bars) / np.log1p(100)))
        
            self.seq_buffer_long.append(new_row.squeeze(axis=0))

            scaled_obs_dataframe = pd.DataFrame(self.seq_buffer_long)

            self.data_dict_long = {
                    "trade_state": np.array(scaled_obs_dataframe[all_attributes["trade_state"].keys()], dtype=np.float32),
                    "market_dynamics": np.array(scaled_obs_dataframe[all_attributes["market_dynamics"].keys()], dtype=np.float32), 
                    "momentum_features": np.array(scaled_obs_dataframe[all_attributes["momentum_features"].keys()], dtype=np.float32),
                    "market_condition": np.array(scaled_obs_dataframe[all_attributes["market_condition"].keys()], dtype=np.float32), 
                    "trend_features": np.array(scaled_obs_dataframe[all_attributes["trend_features"].keys()], dtype=np.float32), 
                    }

        obs_batched = {k: v[None, ...] for k, v in self.data_dict_long.items()}

        action, self.rnn_state = self.rl_model_closer_long.predict(
            obs_batched,
            state=self.rnn_state,
            episode_start=self.episode_start_long,
            deterministic=True
        )

        self.episode_start_long[:] = False

        prediction = int(action)

        if prediction == 1:
            logger.info(f"Rl model prediction is {prediction}. Bet will be closed")
            if self.buy:
                self.direction = 'sell'
                self.take_profit = 1
            elif self.sell:
                self.direction = 'buy'
                self.take_profit = 1
        else:
            logger.info(f"Rl model prediction is {prediction}. It is not time to close")

        self.make_ai_prediction = False
        logger.info(f"make_ai_prediction is changed to {self.make_ai_prediction}")
       
        return prediction
    
    def rl_closer_short(self):
        
        scale_attributes, non_scale_attributes, all_attributes = features.execution_attributes_short_new()
        current_close = self.feature_data["close"].iloc[-1]

        if self.buy:
            relative_profit_long = (current_close - self.entry_price) / self.entry_price
            relative_profit_short = 0
        else:
            relative_profit_short = -((current_close - self.entry_price) / self.entry_price)
            relative_profit_long = 0
   
        logger.info(f"RL model activated")

        sequence_length = 32

        if not hasattr(self, "seq_buffer_short"):
            self.seq_buffer_short = deque(maxlen=sequence_length)
        
        if self.initialize_closer_rl:

            logger.info(f"First round, the data will be initialized.")

            self.initialize_closer_rl = False
            self.rnn_state = None
            self.episode_start_short[:] = True
            self.holding_bars = 0
            self.seq_buffer_short.clear()

            trade_info_chunk = pd.DataFrame({"relative_profit_short": np.zeros((sequence_length,), dtype=np.float32), 
                                             "holding_bars": np.zeros((sequence_length,), dtype=np.float32)
                                             })
            
            trade_info_chunk.iloc[-1] = {
            "relative_profit_short": np.float32(relative_profit_short),
            "holding_bars": np.float32(np.array(np.log1p(self.holding_bars) / np.log1p(100))),
            }
            
            feature_data_seq = self.scaled_data.iloc[-sequence_length:]
            feature_data_seq.reset_index(drop=True, inplace=True)

            scaled_data = feature_data_seq

            self.data_dict_short = {
                    "trade_state": np.array(trade_info_chunk, dtype=np.float32),
                    "market_dynamics": np.array(scaled_data[all_attributes["market_dynamics"].keys()], dtype=np.float32), 
                    "momentum_features": np.array(scaled_data[all_attributes["momentum_features"].keys()], dtype=np.float32),
                    "market_condition": np.array(scaled_data[all_attributes["market_condition"].keys()], dtype=np.float32), 
                    "trend_features": np.array(scaled_data[all_attributes["trend_features"].keys()], dtype=np.float32), 
                   }
            
            to_scaled_buffer = pd.DataFrame()
        
            for chunk, columns in zip(self.data_dict_short.keys(), [list(all_attributes["trade_state"].keys()), list(all_attributes["market_dynamics"].keys()), 
                                                        list(all_attributes["momentum_features"].keys()), list(all_attributes["market_condition"].keys()), 
                                                        list(all_attributes["trend_features"].keys())
                                                        ]):
                to_scaled_buffer[columns] = self.data_dict_short[chunk]

            for index, row in to_scaled_buffer.iterrows():
                self.seq_buffer_short.append(row)
        else:

            self.holding_bars += 1

            scaled_data = self.scaled_data

            latest_values = scaled_data.iloc[-1]

            new_row = latest_values[scale_attributes + ["trend_agreement"]].copy().to_frame().T
            
            new_row["relative_profit_short"] = np.array(relative_profit_short, dtype=np.float32)
            new_row["holding_bars"] = np.float32(np.array(np.log1p(self.holding_bars) / np.log1p(100)))
        
            self.seq_buffer_short.append(new_row.squeeze(axis=0))

            scaled_obs_dataframe = pd.DataFrame(self.seq_buffer_short)

            self.data_dict_short = {
                    "trade_state": np.array(scaled_obs_dataframe[all_attributes["trade_state"].keys()], dtype=np.float32),
                    "market_dynamics": np.array(scaled_obs_dataframe[all_attributes["market_dynamics"].keys()], dtype=np.float32), 
                    "momentum_features": np.array(scaled_obs_dataframe[all_attributes["momentum_features"].keys()], dtype=np.float32),
                    "market_condition": np.array(scaled_obs_dataframe[all_attributes["market_condition"].keys()], dtype=np.float32), 
                    "trend_features": np.array(scaled_obs_dataframe[all_attributes["trend_features"].keys()], dtype=np.float32), 
                    }

        obs_batched = {k: v[None, ...] for k, v in self.data_dict_short.items()}

        action, self.rnn_state = self.rl_model_closer_short.predict(
            obs_batched,
            state=self.rnn_state,
            episode_start=self.episode_start_short,
            deterministic=True
        )

        self.episode_start_short[:] = False

        prediction = int(action)

        if prediction == 1:
            logger.info(f"Rl model prediction is {prediction}. Bet will be closed")
            if self.buy:
                self.direction = 'sell'
                self.take_profit = 1
            elif self.sell:
                self.direction = 'buy'
                self.take_profit = 1
        else:
            logger.info(f"Rl model prediction is {prediction}. It is not time to close")

        self.make_ai_prediction = False
        logger.info(f"make_ai_prediction is changed to {self.make_ai_prediction}")
       
        return prediction
    
    def closer_logic(self):

        if self.buy:
            self.rl_closer_long()
        elif self.sell:
            self.rl_closer_short()

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

        prediction_long = self.single_stage_long.predict([market_dynamics_micro, momentum_features_micro, market_condition_micro, trend_features_trend_entry]).item()
        prediction_short = self.single_stage_short.predict([market_dynamics_micro, momentum_features_micro, market_condition_micro, trend_features_trend_entry]).item()

        direction_pred = self.single_stage_2_direction.predict([market_dynamics_micro, momentum_features_micro, market_condition_micro, trend_features_trend_entry])

        predictions = {
            "long": prediction_long,
            "short": prediction_short,
            "long_dir": direction_pred[0][1],
            "short_dir": direction_pred[0][0]
        }

        return predictions
    
    def single_stage_predictions(self):

        predictions = self.single_stage_prediction()

        long = predictions["long"]
        short = predictions["short"]

        if long < 0.7:
            self.permissions["long"] = True
        if short < 0.7:
            self.permissions["short"] = True

        if predictions["long"] < 0.5 and self.previous_meta_bid == 1:
            self.previous_meta_bid = 0

        if predictions["short"] < 0.5 and self.previous_meta_bid == -1:
            self.previous_meta_bid = 0
        
        if predictions["long"] > 0.7 and predictions["short"] < 0.5 and self.previous_meta_bid != 1 and self.permissions["long"]:
            self.previous_meta_bid = 1
            logger.info(f"Long prediction: {long}")
            logger.info(f"Short prediction: {short}")
            logger.info(f"A long bid will be placed.")
            return 1
        elif predictions["short"] > 0.7 and predictions["long"] < 0.5 and self.previous_meta_bid != -1 and self.permissions["short"]:
            self.previous_meta_bid = -1
            logger.info(f"Long prediction: {long}")
            logger.info(f"Short prediction: {short}")
            logger.info(f"A short bid will be placed.")
            return -1
        else:
            logger.info(f"Long prediction: {long}")
            logger.info(f"Short prediction: {short}")
            logger.info(f"No bids will be placed.")
            return 0

    def sl_prediction(self):

        if not self.bet_placed:

            self.entry_price = self.feature_data["close"].iloc[-1]

            if not hasattr(self, "long_betting_zone"):
                self.previous_meta_bid = 0

            bid = self.single_stage_predictions()

            if bid == 1:
                self.direction = "buy"
            if bid == -1:
                self.direction = "sell"

        self.make_ai_prediction = False
        logger.info(f"make_ai_prediction is changed to {self.make_ai_prediction}")
        
    def check_prediction_time(self):

        if not hasattr(self, "last_prediction_time"):
            self.last_prediction_time = self.date

        elapsed = self.date - self.last_prediction_time
        if self.date.minute % 5 == 0 and self.date.second == 0:
            self.last_prediction_time = self.date
            return True
        elif elapsed >= datetime.timedelta(minutes=5):
            self.last_prediction_time = self.date
            return True
        else:
            return False 
        
    def check_stop_loss_by_candle(self):

        threshold = 0.005
        threshold_real = ((self.volume_bet * 2000) / 2) * -1

        if self.buy:
            loss_percent = 1 - (self.feature_data["close"].iloc[-1]/self.entry_price) 
            real_loss = (self.feature_data["close"].iloc[-1] - self.entry_price) * self.volume_bet * 25
            if loss_percent > threshold:
                logger.info(f"Loss percentage is {loss_percent}, the bid will be closed")
                self.direction = 'sell'
                self.stop_los = 1
            if real_loss < threshold_real:
                logger.info(f"Current loss is {real_loss}, the bid will be closed")
                self.direction = 'sell'
                self.stop_los = 1
        elif self.sell:
            loss_percent = 1 - (self.entry_price/self.feature_data["close"].iloc[-1]) 
            real_loss = (self.entry_price - self.feature_data["close"].iloc[-1]) * self.volume_bet * 25
            if loss_percent > threshold:
                logger.info(f"Loss percentage is {loss_percent}, the bid will be closed")
                self.direction = 'buy'
                self.stop_los = 1
            if real_loss < threshold_real:
                logger.info(f"Current loss is {real_loss}, the bid will be closed")
                self.direction = 'buy'
                self.stop_los = 1
    
    def check_take_profit_by_candle(self):

        if not hasattr(self, "profit_limit"):
            self.profit_limit = False

        threshold = 0.005
        threshold_real = 12

        if self.buy:
            profit_percent = (self.feature_data["close"].iloc[-1]/self.entry_price) - 1 
            real_profit = (self.feature_data["close"].iloc[-1] - self.entry_price) * self.volume_bet * 25
            if profit_percent > threshold:
                logger.info(f"Profit limit exceeded {profit_percent}")
                self.direction = 'sell'
                self.take_profit = 1
            if real_profit > threshold_real:
                logger.info(f"Profit limit exceeded {real_profit}")
                self.direction = 'sell'
                self.take_profit = 1
        elif self.sell:
            profit_percent = (self.entry_price/self.feature_data["close"].iloc[-1]) - 1 
            real_profit = (self.entry_price - self.feature_data["close"].iloc[-1]) * self.volume_bet * 25
            if profit_percent > threshold:
                logger.info(f"Profit limit exceeded {profit_percent}")
                self.direction = 'buy'
                self.take_profit = 1
            if real_profit > threshold_real:
                logger.info(f"Profit limit exceeded {real_profit}")
                self.direction = 'buy'
                self.take_profit = 1

    def predict_trend(self, window):

        close_prices = self.feature_data["close"].iloc[-window:].values

        X = np.arange(len(close_prices)).reshape(-1, 1)
        y = np.array(close_prices)
        model = LinearRegression().fit(X, y)

        return model.coef_[0]

    def check_trend(self):

        self.long_slope = self.predict_trend(20)
        self.short_slope = self.predict_trend(5)

        if hasattr(self, 'previous_short_slope'):
            if self.short_slope >= self.previous_short_slope:
                self.increasing_trend = True
            else:
                self.increasing_trend = False
            if self.short_slope <= self.previous_short_slope:
                self.decreasing_trend = True
            else:
                self.decreasing_trend = False

        self.previous_short_slope = self.short_slope

        if self.long_slope < -0.1:
            self.long_trend = "falling"
        elif self.long_slope > 0.1:
            self.long_trend = "rising"
        else:
            self.long_trend = "flat"

        if self.short_slope < -0.1:
            self.short_trend = "falling"
        elif self.short_slope > 0.1:
            self.short_trend = "rising"
        else:
            self.short_trend = "flat"

    def check_new_data(self):
        
        if not hasattr(self, "previous_data"):
            self.previous_data = self.feature_data[["open", "high", "low", "close"]].iloc[-1]

        new_data = False

        logger.info("Retrieving new data...")

        while not new_data:
            self.feature_data = self.get_features()
            self.current_data = self.feature_data[["open", "high", "low", "close"]].iloc[-1]
            if not self.current_data.equals(self.previous_data):
                new_data = True
                self.previous_data = self.feature_data[["open", "high", "low", "close"]].iloc[-1]
                logger.info("New data received...")

    def compute_latest_dynamic_trend_label_tick(self, df_live, window=20):
        if len(df_live) < max(window, 6):
            return None  # not enough data

        df = df_live.copy()

        # EMA calculations
        ema_5 = df["close"].ewm(span=5).mean().iloc[-1]
        ema_20 = df["close"].ewm(span=20).mean().iloc[-1]

        # VWAP using tick_volume
        vwap = (df["close"] * df["tick_volume"]).cumsum() / df["tick_volume"].cumsum()
        price_above_vwap_ratio = (df["close"] > vwap).mean()

        # Slope over the last `window` closes
        x = np.arange(window).reshape(-1, 1)
        y = df["close"].iloc[-window:].values
        slope = LinearRegression().fit(x, y).coef_[0]

        # Recent gain over last 5 bars
        recent_gain = df["close"].iloc[-1] - df["close"].iloc[-6]

        # Scoring
        score = (
            int(slope > 0.1) +
            int(ema_5 > ema_20) +
            int(price_above_vwap_ratio > 0.6) +
            int(recent_gain > 0)
        )

        if score >= 3:
            return score, "Bullish"
        elif score <= 1:
            return score, "Bearish"
        else:
            return score, "Sideways"
        
    def compute_trend_scores_over_rows(self, df, window=20, lookback=5):
        scores = []
        labels = []

        trend_scores = deque(maxlen=lookback+1)

        for i in range(window, min(len(df), lookback + window)):
            sub_df = df.iloc[:i].copy()

            # EMA calculations
            ema_5 = sub_df["close"].ewm(span=5).mean().iloc[-1]
            ema_20 = sub_df["close"].ewm(span=20).mean().iloc[-1]

            # VWAP using tick_volume
            vwap = (sub_df["close"] * sub_df["tick_volume"]).cumsum() / sub_df["tick_volume"].cumsum()
            price_above_vwap_ratio = (sub_df["close"] > vwap).mean()

            # Slope over the last `window` closes
            x_vals = np.arange(window).reshape(-1, 1)
            y_vals = sub_df["close"].iloc[-window:].values
            slope = LinearRegression().fit(x_vals, y_vals).coef_[0]

            # Recent gain over last 5 bars
            if len(sub_df) < 6:
                recent_gain = 0
            else:
                recent_gain = sub_df["close"].iloc[-1] - sub_df["close"].iloc[-6]

            # Scoring
            score = (
                int(slope > 0.1) +
                int(ema_5 > ema_20) +
                int(price_above_vwap_ratio > 0.6) +
                int(recent_gain > 0)
            )

            # Assign label
            if score >= 3:
                label = "Bullish"
            elif score <= 1:
                label = "Bearish"
            else:
                label = "Sideways"

            scores.append(score)
            labels.append(label)
            trend_scores.append(score)

        # Return DataFrame aligned to input
        result_df = df.iloc[window:window + len(scores)].copy()
        result_df["trend_score"] = scores
        result_df["trend_label"] = labels

        return trend_scores
    
    def check_score_and_trend(self):

        score, trend = self.compute_latest_dynamic_trend_label_tick(self.feature_data)
        self.trend_scores.append(score)
        self.trend = trend

        average_trend_score = sum(self.trend_scores) / len(self.trend_scores)
        logger.info(f"Trend score: {average_trend_score}")
        logger.info(f"Trend is {trend}")

    def test_data(self):

        data = pd.read_excel("test_data/combined_2025_06_17.xlsx")

        return data

    def signals(self):

        self.date = datetime.datetime.now()
        self.time_now = self.date.strftime("%Y_%m_%d_%H_%M_%S")

        self.feature_data = self.get_features()

        if self.first_run:
            if self.check_existing_position():
                self.bet_placed = True
                self.permission_to_trade = True
            else:
                self.bet_placed = False


        if not hasattr(self, "trend_scores"):
            self.trend_scores = self.compute_trend_scores_over_rows(self.feature_data)
            _, self.trend = self.compute_latest_dynamic_trend_label_tick(self.feature_data)

        self.direction = None

        prices = self.mt5.symbol_info(self.symbol)
        self.sell_price = prices.ask
        self.buy_price = prices.bid

        if not self.permission_to_trade and not self.penalty_time:
            if self.date.minute % 5 == 0 and self.date.second == 0:
                logger.info(f"Time is {self.time_now}. Trade permission granted.")
                self.permission_to_trade = True

        if self.penalty_time:
            self.permission_to_trade = False
            elapsed = self.date - self.penalty_time
            if elapsed >= datetime.timedelta(minutes=self.penalty_minutes):
                self.permission_to_trade = True
                self.penalty_time = None

        if self.permission_to_trade:

            if self.bet_placed is True:

                if self.check_prediction_time():
                    if not self.first_run:
                        self.check_new_data()
                    else:
                        self.previous_data = self.feature_data[["open", "high", "low", "close"]].iloc[-1]
                        self.first_run = False
                    logger.info(f"Time is {self.time_now}. Going to make a prediction.")
                    self.check_trend()
                    self.check_score_and_trend()
                    logger.info(f"Long term slope is {self.long_slope}, trend is {self.long_trend}")
                    logger.info(f"Short term slope is {self.short_slope}, trend is {self.short_trend}")

                    self.feature_data = self.add_long_and_short_slopes(self.feature_data.copy())
                    self.scaled_data = scaler.transform_data(self.feature_data.copy(), self.fitted_scaler)
                    self.closer_logic()

                self.check_stop_loss_by_candle()

            if self.bet_placed is False:

                if self.check_prediction_time():
                    if self.check_time() < "23:00":
                        if not self.first_run:
                            self.check_new_data()
                        else:
                            self.previous_data = self.feature_data[["open", "high", "low", "close"]].iloc[-1]
                            self.first_run = False
                        logger.info(f"Time is {self.time_now}. Going to make a prediction.")
                        self.check_trend()
                        self.check_score_and_trend()
                        logger.info(f"Long term slope is {self.long_slope}, trend is {self.long_trend}")
                        logger.info(f"Short term slope is {self.short_slope}, trend is {self.short_trend}")

                        self.feature_data = self.add_long_and_short_slopes(self.feature_data.copy())
                        self.scaled_data = scaler.transform_data(self.feature_data.copy(), self.fitted_scaler)
                        self.make_ai_prediction = True
                        logger.info("AI prediction activated.")
                    else:
                        logger.info(f"Time is {self.time_now}. New bids will not be placed")
            
            if self.make_ai_prediction is True:

                self.sl_prediction()
                self.initialize_closer_rl = True
                self.initialize_rl = True


    def market_order(self, tp_factor, sl_factor, first_tp_limit_factor, second_tp_limit_factor, third_tp_limit_factor, fourth_tp_limit_factor, first_sl_limit_factor, deviation = 20):

        tick = self.mt5.symbol_info_tick(self.symbol)

        order_dict = {'buy': 0, 'sell': 1}
        price_dict = {'buy': tick.ask, 'sell': tick.bid}

        self.tp = float(tp_factor * price_dict[self.direction])
        self.sl = float(sl_factor * price_dict[self.direction])

        self.bet_value = price_dict[self.direction]

        self.first_tp_limit = float(first_tp_limit_factor * price_dict[self.direction])
        self.second_tp_limit = float(second_tp_limit_factor * price_dict[self.direction])
        self.third_tp_limit = float(third_tp_limit_factor * price_dict[self.direction])
        self.fourth_tp_limit = float(fourth_tp_limit_factor * price_dict[self.direction])
        self.first_sl_limit = float(first_sl_limit_factor * price_dict[self.direction])

        request = {
            "action": self.mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": self.volume_bet,
            "type": order_dict[self.direction],
            "price": price_dict[self.direction],
            "deviation": deviation,
            "magic": 100,
            "comment": "python market order",
            "type_time": self.mt5.ORDER_TIME_GTC,
            "type_filling": self.mt5.ORDER_FILLING_FOK,
        }

        order_result = self.mt5.order_send(request)

        logger.info(f"Result of the new order: {order_result}")
        
        print(order_result)

        self.order_comment = order_result.comment
        self.order_entry_price = order_result.request.price

        return order_result
        
    def close_order(self, ticket, deviation = 20):

        positions = self.mt5.positions_get()
        for pos in positions:
            tick = self.mt5.symbol_info_tick(pos.symbol)
            type_dict = {0: 1, 1: 0}  
            price_dict = {0: tick.ask, 1: tick.bid}
            if pos.ticket == ticket:
                request = {
                    "action": self.mt5.TRADE_ACTION_DEAL,
                    "position": pos.ticket,
                    "symbol": pos.symbol,
                    "volume": pos.volume,
                    "type": type_dict[pos.type],
                    "price": price_dict[pos.type],
                    "deviation": deviation,
                    "magic": 100,
                    "comment": "python close order",
                    "type_time": self.mt5.ORDER_TIME_GTC,
                    "type_filling": self.mt5.ORDER_FILLING_FOK,
                }
                
                order_result = self.mt5.order_send(request)
                print(order_result)

                self.order_comment = order_result.comment

                logger.info(f"Result of the closed order: {order_result}")   

                time.sleep(1)             
                
                return order_result

        return 'Ticket does not exist'
    
    def check_time(self):
        time_now = datetime.datetime.now()
        time_now = time_now.strftime("%H:%M")
        return time_now
    
    def check_profits(self):

        date = datetime.datetime.now()
        from_date = date.replace(hour=0, minute=0, second=0, microsecond=0)
        to_date = from_date + datetime.timedelta(days=1)
        history_orders = self.mt5.history_deals_get(from_date, to_date)

        while len(history_orders) % 2 != 0:
            logger.info(f"Waiting that the order will be registered...")
            time.sleep(5)
            history_orders = self.mt5.history_deals_get(from_date, to_date)

        logger.info(f"The order is registered")

        self.order_close_price = history_orders[-1].price
        logger.info(f"Close price is {self.order_close_price}")

        result = history_orders[-1].profit

        if result < 0:
            logger.info(f"Loss amount: {result}")
            logger.info("Order closed with loss.")
            self.previous_order = "loss"
        else:
            logger.info(f"Profit amount: {result}")
            logger.info("Order closed with profit.")
            self.previous_order = "profit"

        self.profits.append(result)

        threshold = 20000 * self.volume_bet * 0.1

        profits_total = sum(self.profits)

        if profits_total > threshold:
            logger.info(f"Total profit is {profits_total}, trading will be terminated")
            self.stop = True
        else:
            logger.info(f"Total profit is {profits_total}")
    
    def action(self):

        if self.bet_placed == False:
            if self.check_time() > self.closing_time:
                self.stop = True
                self.mt5.shutdown()
                logger.info(f"Trading closed at {self.time_now}.")

        if self.stop_los == 1 and self.direction == "sell":
            logger.info("Starting to close buy order.")
            for pos in self.mt5.positions_get():
                if pos.type == 0:  
                    self.close_order(pos.ticket)
            if self.check_existing_position():
                logger.info("Position was not closed succesfully. Trying again.")
                for pos in self.mt5.positions_get():
                    if pos.type == 0:  
                        self.close_order(pos.ticket)
            if self.closing_time is not None:
                if self.check_time() > self.closing_time:
                    self.stop = True
                    logger.info(f"Trading closed at {self.time_now}.")
            self.check_profits()
            self.direction = None
            self.bet_placed = False
            self.stop_los = 0
            self.take_profit = 0
            self.sell = False
            self.buy = False
            self.existing_position_for_console_print = None
            self.counter = 0
        
        if self.stop_los == 1 and self.direction == "buy":
            logger.info("Starting to close sell order.")
            for pos in self.mt5.positions_get():
                if pos.type == 1:  
                    self.close_order(pos.ticket)
            if self.check_existing_position():
                logger.info("Position was not closed succesfully. Trying again.")
                for pos in self.mt5.positions_get():
                    if pos.type == 1:  
                        self.close_order(pos.ticket)
            if self.closing_time is not None:
                if self.check_time() > self.closing_time:
                    self.stop = True
                    logger.info(f"Trading closed at {self.time_now}.")
            self.check_profits()
            self.direction = None
            self.bet_placed = False
            self.stop_los = 0
            self.take_profit = 0
            self.sell = False
            self.buy = False
            self.existing_position_for_console_print = None
            self.counter = 0

        if self.take_profit == 1 and self.direction == "sell":
            logger.info("Starting to close buy order.")
            for pos in self.mt5.positions_get():
                if pos.type == 0:  
                    self.close_order(pos.ticket)
            if self.check_existing_position():
                logger.info("Position was not closed succesfully. Trying again.")
                for pos in self.mt5.positions_get():
                    if pos.type == 0:  
                        self.close_order(pos.ticket)
            if self.closing_time is not None:
                if self.check_time() > self.closing_time:
                    self.stop = True
                    logger.info(f"Trading closed at {self.time_now}.")
            self.check_profits()
            self.direction = None
            self.bet_placed = False
            self.take_profit = 0
            self.stop_los = 0
            self.sell = False
            self.buy = False
            self.existing_position_for_console_print = None
            self.counter = 0
            
        if self.take_profit == 1 and self.direction == "buy":
            logger.info("Starting to close sell order.")
            for pos in self.mt5.positions_get():
                if pos.type == 1:  
                    self.close_order(pos.ticket)
            if self.check_existing_position():
                logger.info("Position was not closed succesfully. Trying again.")
                for pos in self.mt5.positions_get():
                    if pos.type == 1:  
                        self.close_order(pos.ticket)
            if self.closing_time is not None:
                if self.check_time() > self.closing_time:
                    self.stop = True
                    logger.info(f"Trading closed at {self.time_now}.")
            self.check_profits()
            self.direction = None
            self.bet_placed = False
            self.take_profit = 0
            self.stop_los = 0
            self.sell = False
            self.buy = False
            self.existing_position_for_console_print = None
            self.counter = 0
            
        if self.direction == 'buy':
            if not self.mt5.positions_total(): 
                tp_factor = 1.0016
                first_tp_limit_factor = 1.0010
                second_tp_limit_factor = 1.0013
                third_tp_limit_factor = 1.0016
                fourth_tp_limit_factor = 1.0019
                first_sl_limit_factor = 1.0006
                                      
                self.market_order(tp_factor = tp_factor, sl_factor = 0.9994, 
                                  first_tp_limit_factor = first_tp_limit_factor, second_tp_limit_factor = second_tp_limit_factor, third_tp_limit_factor = third_tp_limit_factor, fourth_tp_limit_factor = fourth_tp_limit_factor, first_sl_limit_factor = first_sl_limit_factor) 
                self.profit_and_loss_reducer_value = (self.tp - self.bet_value) / 12
                self.sell = False
                self.buy = True
                self.bet_placed = True
                self.transaction_error = self.mt5.last_error()
                self.check_existing_position()
                logger.info("Buy order placed")

        elif self.direction == 'sell':
            if not self.mt5.positions_total():
                tp_factor = 0.9984
                first_tp_limit_factor = 0.9990
                second_tp_limit_factor = 0.9987
                third_tp_limit_factor = 0.9984
                fourth_tp_limit_factor = 0.9981
                first_sl_limit_factor = 0.9994
                self.market_order(tp_factor = tp_factor, sl_factor = 1.0006, 
                                  first_tp_limit_factor = first_tp_limit_factor, second_tp_limit_factor = second_tp_limit_factor, third_tp_limit_factor = third_tp_limit_factor, fourth_tp_limit_factor = fourth_tp_limit_factor, first_sl_limit_factor = first_sl_limit_factor)
                self.profit_and_loss_reducer_value = (self.sl - self.bet_value) / 12
                self.buy = False
                self.sell = True
                self.bet_placed = True
                self.transaction_error = self.mt5.last_error()
                self.check_existing_position()
                logger.info("Sell order placed")

        if self.stop:
            self.mt5.shutdown()

        return self.stop


    def console_print(self):
        print('Time:', self.time_now)
        print("Trade allowed:", self.permission_to_trade)
        print("Long allowed:", self.permissions["long"])
        print("Short allowed:", self.permissions["short"])
        print("Bet placed:", self.bet_placed)
        print("Buy price:", self.buy_price)
        print("Sell price:", self.sell_price)
        print(f"Long term slope is {self.long_slope}, trend is {self.long_trend}")
        print(f"Short term slope is {self.short_slope}, trend is {self.short_trend}")
        print("Trend:", self.trend)
        print("Error:", self.transaction_error)
        print("Transaction comment:", self.order_comment)
        print("Penalty time started:", self.penalty_time)
        print("-----")


    


        

