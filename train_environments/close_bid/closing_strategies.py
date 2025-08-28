from closing_environment import TradingBotRL
from gymnasium import spaces
import numpy as np
import os
from stable_baselines3.common.env_checker import check_env
import pandas as pd
import random
import scalers as scalers
from sklearn.linear_model import LinearRegression
import joblib
import logging

logger = logging.getLogger()

    
class StrategyCloserLong(TradingBotRL):
    def __init__(self, render_mode = None, dataset = None, mode = "train", plot = True, still_plot = False, n_eval_episodes = 5, eval_logger = None, 
                 env_id = 0, total_timesteps = 1000000):
        super().__init__(render_mode=render_mode, dataset=dataset, mode=mode, plot=plot, still_plot=still_plot, n_eval_episodes=n_eval_episodes, 
                         eval_logger=eval_logger, env_id=env_id)
        
        self.attributes = {
            "trade_state": {
                    "relative_profit_long":{"scale": False},
                    "holding_bars": {"scale": False},
                },

                "market_dynamics": {
                    "close":{"scale": True}, 
                    "tick_volume":{"scale": True}, 
                    "spread_pct":{"scale": True}, 
                    "returns":{"scale": True},
                    "ema_5": {"scale": True},
                    "ema_10": {"scale": True},
                    "ema_15": {"scale": True},
                    "slope_ema_10": {"scale": True}
                },
                "momentum_features": {
                    "macd_diff":{"scale": True}, 
                    "momentum_composite_2":{"scale": True}, 
                    "MA_Crossover":{"scale": True}, 
                    "roc_30m":{"scale": True}
                },
                "market_condition": {
                    "atr":{"scale": True}, 
                    "bb_%b":{"scale": True}, 
                    "volatility_rolling_std":{"scale": True}, 
                    "bb_width":{"scale": True}, 
                },

                "trend_features": {
                    "adosc": {"scale": True},
                    "vwap": {"scale": True},
                    "obv": {"scale": True},
                    "long_term_slope": {"scale": True},
                    "long_slope_angle": {"scale": True},
                },
        }

        self.scale_attributes = [
            attr for group, attributes in self.attributes.items() 
            for attr, options in attributes.items()
            if options.get("scale") is True
        ]

        self.non_scale_attributes = [
            attr for group, attributes in self.attributes.items() 
            for attr, options in attributes.items() 
            if options.get("scale") is False
        ]

        self.observation_space = spaces.Dict({
                'trade_state': spaces.Box(low=-1, high=1, shape=(self.sequence_length, len(self.attributes["trade_state"].keys())), dtype=np.float32),
                'market_dynamics': spaces.Box(low=-5, high=5, shape=(self.sequence_length, len(self.attributes["market_dynamics"].keys())), dtype=np.float32),
                'momentum_features': spaces.Box(low=-5, high=5, shape=(self.sequence_length, len(self.attributes["momentum_features"].keys())), dtype=np.float32),
                'market_condition': spaces.Box(low=-5, high=5, shape=(self.sequence_length, len(self.attributes["market_condition"].keys())), dtype=np.float32),
                'trend_features': spaces.Box(low=-5, high=5, shape=(self.sequence_length, len(self.attributes["trend_features"].keys())), dtype=np.float32),
            })
        
        self.fitted_trend_scaler = joblib.load(os.getenv("TREND_SCALER_TEST"))
        
    def potential_reward(self):
        df = self.env_data.copy()
        lookahead = 12

        current_close = df["close"].iloc[self.row_number]
        future_closes = df["close"].iloc[self.row_number + 1 : self.row_number + 1 + lookahead]

        if self.current_position == "buy":
            highest_value = future_closes.max()
            potential_index = self.env_data.index.get_loc(future_closes.idxmax())
            potential_profit = highest_value - current_close
        else:
            lowest_value = future_closes.min()
            potential_index = self.env_data.index.get_loc(future_closes.idxmin())
            potential_profit = current_close - lowest_value

        self.bid_data = {
            "potential_profit": potential_profit,
            "potential_index": potential_index,
        }
        
    def reward_option_1(self):
        position_action = self.action
        self.step_dict = {
            "step": self.counter, "time": self.env_row["Time"], 
            "action": None, "bet_placed": self.bet_placed, "position_action": position_action,
            "close": self.env_row["close"], "open": self.env_row["open"],
            "realized_profit": None, "equity": self.equity, "movement": 0,
            "total_reward_step_raw": 0, "total_normalized_step_reward": 0,
        }

        movement = 0
        realized_profit = 0
        price_now = self.env_row["close"]
        price_open = self.env_row["open"]
        reward = 0

        self.position_counts[position_action] += 1

        # HOLDING LOGIC
        if position_action == 0:
            self.holding_bars += 1
            reward += self._calculate_holding_reward(price_now, price_open)

            self.position_action_list.append("hold")

        # CLOSING LOGIC
        elif position_action == 1:
            movement, realized_profit = self._calculate_realized_profit(price_now)
            reward += self._calculate_closing_reward(movement)

            self.step_dict["realized_profit"] = realized_profit
            self._register_trade(realized_profit, reward)

            self.bet_closed = True
            self.position_action_list.append("close")

        # Finalize reward
        self.reward += reward
        self.step_dict["movement"] = movement
        self.step_dict["total_normalized_step_reward"] = reward

        self.reward_dict[f"step {self.counter}"] = self.step_dict
        self.reward_list_testing.append(reward)

        self.data_collection["all_rewards"].append(reward)
        self.data_collection["all_profits"].append(movement)
        self.data_collection["total_reward_list"].append(reward)
        self.data_collection["episode_reward_list"].append(reward)

    def _calculate_holding_reward(self, price_now, price_open):
        reward = 0
        movement = price_now - self.entry_price if self.current_position == "buy" else self.entry_price - price_now
        directional_change = price_now - price_open if self.current_position == "buy" else price_open - price_now
        daily_range_scale = self.daily_price_range

        improving = False
        worsen = False

        if self.current_position == "buy":
            if price_now > self.latest_best and self.holding_bars > 1:
                improving = True
                self.latest_best = price_now
            if price_now < self.latest_worst and self.holding_bars > 1:
                worsen = True
                self.latest_worst = price_now
            drop = self.latest_worst - price_now
            improvement = price_now - self.latest_best
        elif self.current_position == "sell":
            if price_now < self.latest_best and self.holding_bars > 1:
                improving = True
                self.latest_best = price_now
            if price_now > self.latest_worst and self.holding_bars > 1:
                worsen = True
                self.latest_worst = price_now
            drop = price_now - self.latest_worst
            improvement = self.latest_best - price_now

        norm_move = movement / (daily_range_scale + 1e-6)

        if improving:
            reward += improvement / (daily_range_scale + 1e-6)
        if worsen:
            reward += drop / (daily_range_scale + 1e-6)

        if self.bid_data["potential_index"] >= self.row_number:
            reward -= self.holding_bars * 0.002
            if movement < 0 and movement > -40:
                reward += norm_move
            elif movement < -40:
                reward += norm_move * 5
            else:
                reward += (directional_change / (daily_range_scale + 1e-6)) * 0.001
                reward += norm_move * 0.01
        else:
            reward -= self.holding_bars * 0.001
            reward += (directional_change / (daily_range_scale + 1e-6)) * 0.001
            reward += norm_move * 0.01

        if self.holding_bars >= 20:
            reward -= 0.002 * np.exp(0.15 * (self.holding_bars - 20))

        return reward
    
    def _calculate_realized_profit(self, price_now):
        if self.current_position == "buy":
            movement = price_now - self.entry_price
        else:
            movement = self.entry_price - price_now

        real_profit = movement * 1.75
        return movement, real_profit
    
    def _calculate_closing_reward(self, movement):

        reward = movement / (self.daily_price_range + 1e-6)

        if movement < 0:
            self.consecutive_loss += 1
            self.previous_result = "loss"
        else:
            self.consecutive_loss = 0
            self.previous_result = "profit"

        return reward
    
    def _register_trade(self, realized_profit, reward):
        self.profit_list.append(realized_profit)
        self.profit_history.append(realized_profit)
        self.episode_profit_history.append(realized_profit)

        self.closed_trades += 1
        self.update_equity(reward, realized_profit, self.action)

        self.data_collection["total_closing_reward_list"].append(reward)
        self.data_collection["episode_closing_reward_list"].append(reward)
        self.data_collection["total_profit_list"].append(realized_profit)
        self.data_collection["episode_profit_list"].append(realized_profit)

        self.bet_placed = False
        self.relative_profit_long = 0
        self.relative_profit_short = 0
        self.positive_holding_bars = 0
        self.negative_holding_bars = 0
        self.total_holding_reward = 0
        self.total_bet_reward = 0
        self.good_entry = False
        self.holding_improvement = 0

    def step(self, action):

            print(f"[PID {os.getpid()}] Step {self.counter} in Environment {self.env_id}")

            self.counter += 1
            self.action = action
            self.reward = 0
            self.steps_counter += 1

            self.reward_option_1()
            self.track_env_reward()

            if self.counter + 2 >= self.number_of_rows:
                done = True
            elif self.bet_closed:
                done = True
            else: 
                done = False
                self.row_number += 1
            
            self.env_row = self.env_data.iloc[self.row_number]

            if self.bet_type == 1:
                self.relative_profit_long = (self.env_row["close"] - self.entry_price) / self.entry_price
                self.relative_profit_short = 0
            elif self.bet_type == 0:
                self.relative_profit_short = -((self.env_row["close"] - self.entry_price) / self.entry_price)
                self.relative_profit_long = 0

            new_row = self.env_data.iloc[self.row_number].copy().to_frame().T
            
            new_row["relative_profit_long"] = np.array(self.relative_profit_long, dtype=np.float32)
            new_row["relative_profit_short"] = np.array(self.relative_profit_short, dtype=np.float32)
            new_row["bet_type"] = np.array(self.bet_type, dtype=np.float32)
            new_row["holding_bars"] = np.array(np.log1p(self.holding_bars) / np.log1p(100), dtype=np.float32)

            scaled_row_data = scalers.transform_data(new_row, self.fitted_scaler)

            trend_featrues_for_scaling = list(self.fitted_trend_scaler.feature_names_in_)
            scaled_row_data[trend_featrues_for_scaling] = self.fitted_trend_scaler.transform(scaled_row_data[trend_featrues_for_scaling])  
        
            self.seq_buffer.append(new_row.squeeze(axis=0)) 
            self.seq_buffer_scaled.append(scaled_row_data.squeeze(axis=0))

            scaled_obs_dataframe = pd.DataFrame(self.seq_buffer_scaled)

            final_dict = {
                    "trade_state": np.array(scaled_obs_dataframe[self.attributes["trade_state"].keys()], dtype=np.float32),
                    "market_dynamics": np.array(scaled_obs_dataframe[self.attributes["market_dynamics"].keys()], dtype=np.float32), 
                    "momentum_features": np.array(scaled_obs_dataframe[self.attributes["momentum_features"].keys()], dtype=np.float32),
                    "market_condition": np.array(scaled_obs_dataframe[self.attributes["market_condition"].keys()], dtype=np.float32), 
                    "trend_features": np.array(scaled_obs_dataframe[self.attributes["trend_features"].keys()], dtype=np.float32), 
                    }
            
            info = {"Total profit": 0, "reward_dict": self.reward_dict}

            print("Timestep:", self.timesteps)
            print("Number of rows:", self.number_of_rows)
            print("Row number", self.row_number)
            print("Reward:", self.reward)

            if done:

                if self.render_mode == "human":
                    self.render()

                self.total_profit = sum(self.profit_list) 

                if len(self.eval_profits) == self.n_eval_episodes:
                    self.eval_profits = []

                self.eval_profits.append(self.total_profit)

                info = {"Total profit": round(self.total_profit, 2), "Env number": self.env_number, "real_profits": self.eval_profits, "reward_dict": self.reward_dict}

                if self.mode == "test":
                    self.env_number += 1
                    if self.env_number == self.amount_of_envs:
                        self.env_number = 0
                    if self.counter + 2 >= self.number_of_rows:
                        if self.plot:
                            self.episode_plot(self.reward_dict)


                self.episode_dict[f"{self.timesteps}"] = self.reward_dict
                self.reward_dict = {}
                self.env_number += 1
                    
            if self.steps_counter >= 2048 and self.mode == "train": 
                logger.info(f"Closed trades: {self.closed_trades}")
                if len(self.trade_history) != 0:
                    try:
                        self.step_info_logging()
                    except Exception as e:
                        logger.info(f"Error occured: {e}")
                self.episode_dict = {}
                self.closed_trades = 0
                self.data_collection["good_entry"] = 0
                self.data_collection["bad_entry"] = 0
            
            self.timesteps += 1
            terminated = done
            self.last_info = info
            
            return final_dict, self.reward, terminated, False, info  
    
    def select_bid_window(self, df):
       
        assert "long_prediction" in df.columns and "short_prediction" in df.columns, \
            "Required columns not found in DataFrame."
        
        df["original_datetime"] = pd.to_datetime(df["original_datetime"])

        # Step 1: Randomly choose bid direction
        bid_type = "long"
        #print(f"Selected bid type: {bid_type}")

        if bid_type == "long":
            self.bet_type = 1
        else:
            self.bet_type = 0

        # Step 2: Filter rows where meta prediction > 0.8
        col = f"{bid_type}_prediction"

        valid_range = 200 

        df_filtered = df[
            (df[col] == 1) &
            (df["original_datetime"].dt.hour >= 8) &
            (df["original_datetime"].dt.hour < 18) &
            (df.index >= valid_range) &
            (df.index <= len(df) - valid_range - 1)
        ]

        if df_filtered.empty:
            raise ValueError(f"No qualifying {bid_type} bids between 08:00 and 18:00 with prediction > 0.8.")

        # Step 3: Randomly pick one qualifying index
        selected_index = random.choice(df_filtered.index.tolist())
        #print(f"Selected row index: {selected_index} (time: {df.loc[selected_index, 'original_datetime']})")

        # Step 4: Get a 201-row window around the selected index
        start = max(0, selected_index - valid_range)
        end = min(len(df), selected_index + valid_range)

        window_df = df.iloc[start:end].copy()

        return window_df, selected_index

    def reset(self, seed=None, options=None):

        if seed is not None:
            np.random.seed(seed)
        self.counter = 0
        self.bet_placed = False
        self.relative_profit_long = 0
        self.relative_profit_short = 0
        self.reward = 0
        self.bet_action_list = []
        self.position_action_list = []
        self.total_profit = 0
        self.profit_list = []
        self.reward_list_testing = []
        self.trade_specs = {}
        self.environment_reward = []
        self.pending_buffer_rewards = []
        self.smoothed_reward = 0
        self.smoothed_holding_reward = 0
        self.smoothed_bonus_reward = 0
        self.consecutive_fast_close = 0
        self.bids_quality = {0: 0, 1: 0, 2: 0}
        self.position_quality = {0: 0, 1: 0}
        self.win_streak = 0
        self.consecutive_loss = 0
        self.total_holding_reward = 0
        self.closed_trades = 0
        self.good_entry = False
        self.holding_improvement = 0
        self.bet_closed = False

        self.average_equity_episode.append(self.equity)
        self.average_equity.append(self.equity)
        self.equity = 2000
        self.peak_equity = self.equity
        self.current_position = None
        self.position_size = 0
        self.holding_bars = 0
        self.positive_holding_bars = 0
        self.negative_holding_bars = 0
        self.env_wins = 0
        self.env_losses = 0
        self.relative_profit_raw = 0

        self.env_data, row_number = self.select_bid_window(self.dataset)

        self.env_data = self.add_long_and_short_slopes(self.env_data)[20:]

        self.daily_max_close = self.env_data["close"].max()
        self.daily_min_close = self.env_data["close"].min()
        self.daily_price_range = self.daily_max_close - self.daily_min_close

        self.row_number = self.env_data.index.get_loc(row_number) + 1

        self.number_of_rows = len(self.env_data.iloc[self.row_number:])

        self.env_row = self.env_data.iloc[self.row_number]

        self.price_previous = self.env_data["close"].iloc[self.row_number -1]
        self.price_current = self.env_row["close"]

        self.entry_price = self.latest_best = self.latest_worst = self.env_data["close"].iloc[self.row_number -1]

        if self.bet_type == 1:
            self.relative_profit_long = (self.env_row["close"] - self.entry_price) / self.entry_price
            self.relative_profit_short = 0
            self.current_position = "buy"
        else:
            self.relative_profit_short = -((self.env_row["close"] - self.entry_price) / self.entry_price)
            self.relative_profit_long = 0
            self.current_position = "short"

        self.potential_reward()

        raw_trade_state_features_chunk_data = pd.DataFrame({
                                                    "relative_profit_long": np.zeros((self.sequence_length,), dtype=np.float32),
                                                    "relative_profit_short": np.zeros((self.sequence_length,), dtype=np.float32),
                                                    "bet_type": np.zeros((self.sequence_length,), dtype=np.float32),
                                                    "holding_bars": np.zeros((self.sequence_length,), dtype=np.float32),
                                                })

        raw_trade_state_features_chunk_data.iloc[-1] = {
            "relative_profit_long": np.float32(self.relative_profit_long),
            "relative_profit_short": np.float32(self.relative_profit_short),
            "bet_type": np.float32(self.bet_type),
            "holding_bars": np.float32(np.array(np.log1p(self.holding_bars) / np.log1p(100))),
        }

        raw_trade_state_features_chunk = raw_trade_state_features_chunk_data[self.attributes["trade_state"].keys()]
        raw_market_dynamics_features_chunk = self.env_data[self.attributes["market_dynamics"].keys()].iloc[self.row_number + 1 - self.sequence_length : self.row_number + 1]
        raw_momentum_features_chunk = self.env_data[self.attributes["momentum_features"].keys()].iloc[self.row_number + 1 - self.sequence_length : self.row_number + 1]
        raw_market_condition_chunk = self.env_data[self.attributes["market_condition"].keys()].iloc[self.row_number + 1 - self.sequence_length : self.row_number + 1]
        raw_trend_featrues_chunk = self.env_data[self.attributes["trend_features"].keys()].iloc[self.row_number + 1 - self.sequence_length : self.row_number + 1]

        scaled_data = scalers.transform_data(self.env_data.iloc[self.row_number + 1  - self.sequence_length : self.row_number + 1], self.fitted_scaler)

        trend_featrues_for_scaling = [
            attribute for attribute, option in self.attributes["trend_features"].items()
            if option.get("scale") is True
        ]

        trend_featrues_for_scaling = list(self.fitted_trend_scaler.feature_names_in_)
        scaled_data[trend_featrues_for_scaling] = self.fitted_trend_scaler.transform(scaled_data[trend_featrues_for_scaling])  

        reset_obs = {
                   "trade_state": np.array(raw_trade_state_features_chunk, dtype=np.float32),
                   "market_dynamics": np.array(scaled_data[self.attributes["market_dynamics"].keys()], dtype=np.float32), 
                   "momentum_features": np.array(scaled_data[self.attributes["momentum_features"].keys()], dtype=np.float32),
                   "market_condition": np.array(scaled_data[self.attributes["market_condition"].keys()], dtype=np.float32), 
                   "trend_features": np.array(scaled_data[self.attributes["trend_features"].keys()], dtype=np.float32), 
                   }
        
        to_scaled_buffer = pd.DataFrame()
        
        for chunk, columns in zip(reset_obs.keys(), [list(self.attributes["trade_state"].keys()), list(self.attributes["market_dynamics"].keys()), 
                                                     list(self.attributes["momentum_features"].keys()), list(self.attributes["market_condition"].keys()), 
                                                     list(self.attributes["trend_features"].keys())]):
            to_scaled_buffer[columns] = reset_obs[chunk]

        raw_obs_to_buffer = pd.concat([raw_trade_state_features_chunk, raw_market_dynamics_features_chunk, raw_momentum_features_chunk, raw_market_condition_chunk,
                                       raw_trend_featrues_chunk], axis=1)
        
        self.seq_buffer.clear()
        self.seq_buffer_scaled.clear()

        for index, row in raw_obs_to_buffer.iterrows():
            self.seq_buffer.append(row)

        for index, row in to_scaled_buffer.iterrows():
            self.seq_buffer_scaled.append(row)

        return reset_obs, {}
    
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


class StrategyCLoserShort(TradingBotRL):
    def __init__(self, render_mode = None, dataset = None, mode = "train", plot = True, still_plot = False, n_eval_episodes = 5, eval_logger = None, 
                 env_id = 0, total_timesteps = 1000000):
        super().__init__(render_mode=render_mode, dataset=dataset, mode=mode, plot=plot, still_plot=still_plot, n_eval_episodes=n_eval_episodes, 
                         eval_logger=eval_logger, env_id=env_id)
        
        self.attributes = {
            "trade_state": {
                    "relative_profit_short":{"scale": False},
                    "holding_bars": {"scale": False},
                },

                "market_dynamics": {
                    "close":{"scale": True}, 
                    "tick_volume":{"scale": True}, 
                    "spread_pct":{"scale": True}, 
                    "returns":{"scale": True},
                    "ema_5": {"scale": True},
                    "ema_10": {"scale": True},
                    "ema_15": {"scale": True},
                    "slope_ema_10": {"scale": True}
                },
                "momentum_features": {
                    "macd_diff":{"scale": True}, 
                    "momentum_composite_2":{"scale": True}, 
                    "MA_Crossover":{"scale": True}, 
                    "roc_30m":{"scale": True}
                },
                "market_condition": {
                    "atr":{"scale": True}, 
                    "bb_%b":{"scale": True}, 
                    "volatility_rolling_std":{"scale": True}, 
                    "bb_width":{"scale": True}, 
                },

                "trend_features": {
                    "adosc": {"scale": True},
                    "vwap": {"scale": True},
                    "obv": {"scale": True},
                    "long_term_slope": {"scale": True},
                    "long_slope_angle": {"scale": True},
                },
        }

        self.scale_attributes = [
            attr for group, attributes in self.attributes.items() 
            for attr, options in attributes.items()
            if options.get("scale") is True
        ]

        self.non_scale_attributes = [
            attr for group, attributes in self.attributes.items() 
            for attr, options in attributes.items() 
            if options.get("scale") is False
        ]

        self.observation_space = spaces.Dict({
                'trade_state': spaces.Box(low=-1, high=1, shape=(self.sequence_length, len(self.attributes["trade_state"].keys())), dtype=np.float32),
                'market_dynamics': spaces.Box(low=-5, high=5, shape=(self.sequence_length, len(self.attributes["market_dynamics"].keys())), dtype=np.float32),
                'momentum_features': spaces.Box(low=-5, high=5, shape=(self.sequence_length, len(self.attributes["momentum_features"].keys())), dtype=np.float32),
                'market_condition': spaces.Box(low=-5, high=5, shape=(self.sequence_length, len(self.attributes["market_condition"].keys())), dtype=np.float32),
                'trend_features': spaces.Box(low=-5, high=5, shape=(self.sequence_length, len(self.attributes["trend_features"].keys())), dtype=np.float32),
            })
        
        self.fitted_trend_scaler = joblib.load(os.getenv("TREND_SCALER_TEST"))
        
    def potential_reward(self):
        df = self.env_data.copy()
        lookahead = 12

        current_close = df["close"].iloc[self.row_number]
        future_closes = df["close"].iloc[self.row_number + 1 : self.row_number + 1 + lookahead]

        if self.current_position == "buy":
            highest_value = future_closes.max()
            potential_index = future_closes.idxmax()
            potential_profit = highest_value - current_close
        else:
            lowest_value = future_closes.min()
            potential_index = future_closes.idxmin()
            potential_profit = current_close - lowest_value

        self.bid_data = {
            "potential_profit": potential_profit,
            "potential_index": potential_index,
        }
        
    def reward_option_1(self):
        position_action = self.action
        self.step_dict = {
            "step": self.counter, "time": self.env_row["Time"], 
            "action": None, "bet_placed": self.bet_placed, "position_action": position_action,
            "close": self.env_row["close"], "open": self.env_row["open"],
            "realized_profit": None, "equity": self.equity, "movement": 0,
            "total_reward_step_raw": 0, "total_normalized_step_reward": 0,
        }

        movement = 0
        realized_profit = 0
        price_now = self.env_row["close"]
        price_open = self.env_row["open"]
        reward = 0

        self.position_counts[position_action] += 1

        # HOLDING LOGIC
        if position_action == 0:
            self.holding_bars += 1
            reward += self._calculate_holding_reward(price_now, price_open)

            self.position_action_list.append("hold")

        # CLOSING LOGIC
        elif position_action == 1:
            movement, realized_profit = self._calculate_realized_profit(price_now)
            reward += self._calculate_closing_reward(movement)

            self.step_dict["realized_profit"] = realized_profit
            self._register_trade(realized_profit, reward)

            self.bet_closed = True
            self.position_action_list.append("close")

        # Finalize reward
        self.reward += reward
        self.step_dict["movement"] = movement
        self.step_dict["total_normalized_step_reward"] = reward

        self.reward_dict[f"step {self.counter}"] = self.step_dict
        self.reward_list_testing.append(reward)

        self.data_collection["all_rewards"].append(reward)
        self.data_collection["all_profits"].append(movement)
        self.data_collection["total_reward_list"].append(reward)
        self.data_collection["episode_reward_list"].append(reward)

    def _calculate_holding_reward(self, price_now, price_open):
        reward = 0
        movement = price_now - self.entry_price if self.current_position == "buy" else self.entry_price - price_now
        directional_change = price_now - price_open if self.current_position == "buy" else price_open - price_now
        daily_range_scale = self.daily_price_range

        improving = False
        worsen = False

        if self.current_position == "buy":
            if price_now > self.latest_best and self.holding_bars > 1:
                improving = True
                self.latest_best = price_now
            if price_now < self.latest_worst and self.holding_bars > 1:
                worsen = True
                self.latest_worst = price_now
            drop = self.latest_worst - price_now
            improvement = price_now - self.latest_best
        elif self.current_position == "sell":
            if price_now < self.latest_best and self.holding_bars > 1:
                improving = True
                self.latest_best = price_now
            if price_now > self.latest_worst and self.holding_bars > 1:
                worsen = True
                self.latest_worst = price_now
            drop = price_now - self.latest_worst
            improvement = self.latest_best - price_now

        norm_move = movement / (daily_range_scale + 1e-6)

        reward += (directional_change / (daily_range_scale + 1e-6)) * 0.01

        if movement < -60:
            reward += norm_move
        else:
            reward += norm_move * 0.1

        reward -= self.holding_bars * 0.001
            
        if self.holding_bars >= 20:
            reward -= 0.002 * np.exp(0.15 * (self.holding_bars - 20))

        return reward
    
    def _calculate_realized_profit(self, price_now):
        if self.current_position == "buy":
            movement = price_now - self.entry_price
        else:
            movement = self.entry_price - price_now

        real_profit = movement * 1.75
        return movement, real_profit
    
    def _calculate_closing_reward(self, movement):

        reward = movement / (self.daily_price_range + 1e-6)

        if movement < 0:
            self.consecutive_loss += 1
            self.previous_result = "loss"
        else:
            self.consecutive_loss = 0
            self.previous_result = "profit"

        return reward
    
    def _register_trade(self, realized_profit, reward):
        self.profit_list.append(realized_profit)
        self.profit_history.append(realized_profit)
        self.episode_profit_history.append(realized_profit)

        self.closed_trades += 1
        self.update_equity(reward, realized_profit, self.action)

        self.data_collection["total_closing_reward_list"].append(reward)
        self.data_collection["episode_closing_reward_list"].append(reward)
        self.data_collection["total_profit_list"].append(realized_profit)
        self.data_collection["episode_profit_list"].append(realized_profit)

        self.bet_placed = False
        self.relative_profit_long = 0
        self.relative_profit_short = 0
        self.positive_holding_bars = 0
        self.negative_holding_bars = 0
        self.total_holding_reward = 0
        self.total_bet_reward = 0
        self.good_entry = False
        self.holding_improvement = 0

    def step(self, action):

        print(f"[PID {os.getpid()}] Step {self.counter} in Environment {self.env_id}")

        self.counter += 1
        self.action = action
        self.reward = 0
        self.steps_counter += 1

        self.reward_option_1()
        self.track_env_reward()

        if self.counter + 2 >= self.number_of_rows:
            done = True
        elif self.bet_closed:
            done = True
        else: 
            done = False
            self.row_number += 1
        
        self.env_row = self.env_data.iloc[self.row_number]

        if self.bet_type == 1:
            self.relative_profit_long = (self.env_row["close"] - self.entry_price) / self.entry_price
            self.relative_profit_short = 0
        elif self.bet_type == 0:
            self.relative_profit_short = -((self.env_row["close"] - self.entry_price) / self.entry_price)
            self.relative_profit_long = 0

        new_row = self.env_data.iloc[self.row_number].copy().to_frame().T
        
        new_row["relative_profit_long"] = np.array(self.relative_profit_long, dtype=np.float32)
        new_row["relative_profit_short"] = np.array(self.relative_profit_short, dtype=np.float32)
        new_row["bet_type"] = np.array(self.bet_type, dtype=np.float32)
        new_row["holding_bars"] = np.array(np.log1p(self.holding_bars) / np.log1p(100), dtype=np.float32)

        scaled_row_data = scalers.transform_data(new_row, self.fitted_scaler)

        trend_featrues_for_scaling = list(self.fitted_trend_scaler.feature_names_in_)
        scaled_row_data[trend_featrues_for_scaling] = self.fitted_trend_scaler.transform(scaled_row_data[trend_featrues_for_scaling])  
    
        self.seq_buffer.append(new_row.squeeze(axis=0)) 
        self.seq_buffer_scaled.append(scaled_row_data.squeeze(axis=0))

        scaled_obs_dataframe = pd.DataFrame(self.seq_buffer_scaled)

        final_dict = {
                "trade_state": np.array(scaled_obs_dataframe[self.attributes["trade_state"].keys()], dtype=np.float32),
                "market_dynamics": np.array(scaled_obs_dataframe[self.attributes["market_dynamics"].keys()], dtype=np.float32), 
                "momentum_features": np.array(scaled_obs_dataframe[self.attributes["momentum_features"].keys()], dtype=np.float32),
                "market_condition": np.array(scaled_obs_dataframe[self.attributes["market_condition"].keys()], dtype=np.float32), 
                "trend_features": np.array(scaled_obs_dataframe[self.attributes["trend_features"].keys()], dtype=np.float32), 
                }
        
        info = {"Total profit": 0, "reward_dict": self.reward_dict}

        print("Timestep:", self.timesteps)
        print("Number of rows:", self.number_of_rows)
        print("Row number", self.row_number)
        print("Reward:", self.reward)

        if done:

            if self.render_mode == "human":
                self.render()

            self.total_profit = sum(self.profit_list) 

            if len(self.eval_profits) == self.n_eval_episodes:
                self.eval_profits = []

            self.eval_profits.append(self.total_profit)

            info = {"Total profit": round(self.total_profit, 2), "Env number": self.env_number, "real_profits": self.eval_profits, "reward_dict": self.reward_dict}

            if self.mode == "test":
                self.env_number += 1
                if self.env_number == self.amount_of_envs:
                    self.env_number = 0
                if self.counter + 2 >= self.number_of_rows:
                    if self.plot:
                        self.episode_plot(self.reward_dict)


            self.episode_dict[f"{self.timesteps}"] = self.reward_dict
            self.reward_dict = {}
            self.env_number += 1
                
        if self.steps_counter >= 2048 and self.mode == "train": 
            logger.info(f"Closed trades: {self.closed_trades}")
            if len(self.trade_history) != 0:
                try:
                    self.step_info_logging()
                except Exception as e:
                    logger.info(f"Error occured: {e}")
            self.episode_dict = {}
            self.closed_trades = 0
            self.data_collection["good_entry"] = 0
            self.data_collection["bad_entry"] = 0
        
        self.timesteps += 1
        terminated = done
        self.last_info = info
        
        return final_dict, self.reward, terminated, False, info  
    
    def select_bid_window(self, df):
       
        assert "long_prediction" in df.columns and "short_prediction" in df.columns, \
            "Required columns not found in DataFrame."
        
        df["original_datetime"] = pd.to_datetime(df["original_datetime"])

        # Step 1: Randomly choose bid direction
        bid_type = "short"
        #print(f"Selected bid type: {bid_type}")

        if bid_type == "long":
            self.bet_type = 1
        else:
            self.bet_type = 0

        # Step 2: Filter rows where meta prediction > 0.8
        col = f"{bid_type}_prediction"

        valid_range = 200 

        df_filtered = df[
            (df[col] == 1) &
            (df["original_datetime"].dt.hour >= 8) &
            (df["original_datetime"].dt.hour < 18) &
            (df.index >= valid_range) &
            (df.index <= len(df) - valid_range - 1)
        ]

        if df_filtered.empty:
            raise ValueError(f"No qualifying {bid_type} bids between 08:00 and 18:00 with prediction > 0.8.")

        # Step 3: Randomly pick one qualifying index
        selected_index = random.choice(df_filtered.index.tolist())
        #print(f"Selected row index: {selected_index} (time: {df.loc[selected_index, 'original_datetime']})")

        # Step 4: Get a 201-row window around the selected index
        start = max(0, selected_index - valid_range)
        end = min(len(df), selected_index + valid_range)

        window_df = df.iloc[start:end].copy()

        return window_df, selected_index

    def reset(self, seed=None, options=None):

        if seed is not None:
            np.random.seed(seed)
        self.counter = 0
        self.bet_placed = False
        self.relative_profit_long = 0
        self.relative_profit_short = 0
        self.reward = 0
        self.bet_action_list = []
        self.position_action_list = []
        self.total_profit = 0
        self.profit_list = []
        self.reward_list_testing = []
        self.trade_specs = {}
        self.environment_reward = []
        self.pending_buffer_rewards = []
        self.smoothed_reward = 0
        self.smoothed_holding_reward = 0
        self.smoothed_bonus_reward = 0
        self.consecutive_fast_close = 0
        self.bids_quality = {0: 0, 1: 0, 2: 0}
        self.position_quality = {0: 0, 1: 0}
        self.win_streak = 0
        self.consecutive_loss = 0
        self.total_holding_reward = 0
        self.closed_trades = 0
        self.good_entry = False
        self.holding_improvement = 0
        self.bet_closed = False

        self.average_equity_episode.append(self.equity)
        self.average_equity.append(self.equity)
        self.equity = 2000
        self.peak_equity = self.equity
        self.current_position = None
        self.position_size = 0
        self.holding_bars = 0
        self.positive_holding_bars = 0
        self.negative_holding_bars = 0
        self.env_wins = 0
        self.env_losses = 0
        self.relative_profit_raw = 0

        self.env_data, row_number = self.select_bid_window(self.dataset)

        self.env_data = self.add_long_and_short_slopes(self.env_data)[20:]

        self.daily_max_close = self.env_data["close"].max()
        self.daily_min_close = self.env_data["close"].min()
        self.daily_price_range = self.daily_max_close - self.daily_min_close

        self.row_number = self.env_data.index.get_loc(row_number) + 1
        self.potential_reward()

        self.number_of_rows = len(self.env_data.iloc[self.row_number:])

        self.env_row = self.env_data.iloc[self.row_number]

        self.price_previous = self.env_data["close"].iloc[self.row_number -1]
        self.price_current = self.env_row["close"]

        self.entry_price = self.latest_best = self.latest_worst = self.env_data["close"].iloc[self.row_number -1]

        if self.bet_type == 1:
            self.relative_profit_long = (self.env_row["close"] - self.entry_price) / self.entry_price
            self.relative_profit_short = 0
            self.current_position = "buy"
        else:
            self.relative_profit_short = -((self.env_row["close"] - self.entry_price) / self.entry_price)
            self.relative_profit_long = 0
            self.current_position = "short"

        raw_trade_state_features_chunk_data = pd.DataFrame({
                                                    "relative_profit_long": np.zeros((self.sequence_length,), dtype=np.float32),
                                                    "relative_profit_short": np.zeros((self.sequence_length,), dtype=np.float32),
                                                    "bet_type": np.zeros((self.sequence_length,), dtype=np.float32),
                                                    "holding_bars": np.zeros((self.sequence_length,), dtype=np.float32),
                                                })

        raw_trade_state_features_chunk_data.iloc[-1] = {
            "relative_profit_long": np.float32(self.relative_profit_long),
            "relative_profit_short": np.float32(self.relative_profit_short),
            "bet_type": np.float32(self.bet_type),
            "holding_bars": np.float32(np.array(np.log1p(self.holding_bars) / np.log1p(100))),
        }

        raw_trade_state_features_chunk = raw_trade_state_features_chunk_data[self.attributes["trade_state"].keys()]
        raw_market_dynamics_features_chunk = self.env_data[self.attributes["market_dynamics"].keys()].iloc[self.row_number + 1 - self.sequence_length : self.row_number + 1]
        raw_momentum_features_chunk = self.env_data[self.attributes["momentum_features"].keys()].iloc[self.row_number + 1 - self.sequence_length : self.row_number + 1]
        raw_market_condition_chunk = self.env_data[self.attributes["market_condition"].keys()].iloc[self.row_number + 1 - self.sequence_length : self.row_number + 1]
        raw_trend_featrues_chunk = self.env_data[self.attributes["trend_features"].keys()].iloc[self.row_number + 1 - self.sequence_length : self.row_number + 1]

        scaled_data = scalers.transform_data(self.env_data.iloc[self.row_number + 1  - self.sequence_length : self.row_number + 1], self.fitted_scaler)

        trend_featrues_for_scaling = [
            attribute for attribute, option in self.attributes["trend_features"].items()
            if option.get("scale") is True
        ]

        trend_featrues_for_scaling = list(self.fitted_trend_scaler.feature_names_in_)
        scaled_data[trend_featrues_for_scaling] = self.fitted_trend_scaler.transform(scaled_data[trend_featrues_for_scaling])  

        reset_obs = {
                   "trade_state": np.array(raw_trade_state_features_chunk, dtype=np.float32),
                   "market_dynamics": np.array(scaled_data[self.attributes["market_dynamics"].keys()], dtype=np.float32), 
                   "momentum_features": np.array(scaled_data[self.attributes["momentum_features"].keys()], dtype=np.float32),
                   "market_condition": np.array(scaled_data[self.attributes["market_condition"].keys()], dtype=np.float32), 
                   "trend_features": np.array(scaled_data[self.attributes["trend_features"].keys()], dtype=np.float32), 
                   }
        
        to_scaled_buffer = pd.DataFrame()
        
        for chunk, columns in zip(reset_obs.keys(), [list(self.attributes["trade_state"].keys()), list(self.attributes["market_dynamics"].keys()), 
                                                     list(self.attributes["momentum_features"].keys()), list(self.attributes["market_condition"].keys()), 
                                                     list(self.attributes["trend_features"].keys())]):
            to_scaled_buffer[columns] = reset_obs[chunk]

        raw_obs_to_buffer = pd.concat([raw_trade_state_features_chunk, raw_market_dynamics_features_chunk, raw_momentum_features_chunk, raw_market_condition_chunk,
                                       raw_trend_featrues_chunk], axis=1)
        
        self.seq_buffer.clear()
        self.seq_buffer_scaled.clear()

        for index, row in raw_obs_to_buffer.iterrows():
            self.seq_buffer.append(row)

        for index, row in to_scaled_buffer.iterrows():
            self.seq_buffer_scaled.append(row)

        return reset_obs, {}
    
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
    
if __name__ == "__main__":

    dataset = pd.read_excel("envs/combined_dataset_full_with_binary_predictions_2025_08_16_14_49_50.xlsx", 
                            nrows=10000
                            )

    env = StrategyCloserLong(dataset = dataset)

    check_env(env)
    episodes = 10
    env.reset()
    for episode in range(0,episodes):
        print(episode)
        action = env.action_space.sample()
        observation, reward, done, _, info = env.step(action)
        print(observation)
        
        if done:
            env.reset()
