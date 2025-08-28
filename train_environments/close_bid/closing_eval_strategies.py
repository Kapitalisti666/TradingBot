import scalers as scalers
from closing_environment_eval import TradingBotRLEval
import os
import numpy as np
import pandas as pd
from stable_baselines3.common.env_checker import check_env
from gymnasium import spaces
import logging
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import joblib

logger = logging.getLogger()
    
class TradingBotRLEvalCloserLong(TradingBotRLEval):
    def __init__(self, render_mode = None, dataset_list = None, mode = "train", plot = True, still_plot = False, n_eval_episodes = 5, eval_logger = None, 
                 env_id = 0, total_timesteps = 1000000):
        super().__init__(render_mode=render_mode, dataset_list=dataset_list, mode=mode, plot=plot, still_plot=still_plot, n_eval_episodes=n_eval_episodes, 
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

    def initialize_data(self):

        self.holding_bars = 0

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

        scaled_data = scalers.transform_data(self.env_data.iloc[self.row_number + 1 - self.sequence_length : self.row_number + 1], self.fitted_scaler)

        trend_features_for_scaling = list(self.fitted_trend_scaler.feature_names_in_)
        scaled_data[trend_features_for_scaling] = self.fitted_trend_scaler.transform(scaled_data[trend_features_for_scaling])

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

        return reset_obs

        
    def step(self, action):

        self._last_episode_starts = np.array([False], dtype=bool)

        print(f"[PID {os.getpid()}] Step {self.counter} in Environment {self.env_id}")

        self.counter += 1
        self.action = action
        self.reward = 0
        self.steps_counter += 1

        self.reward_option_1()
        self.track_env_reward()

        self.reset_lstm = False

        if self.counter + 2 >= self.number_of_rows:
            done = True
        elif not self.bet_closed: 
            done = False
            self.row_number += 1
            self.env_row = self.env_data.iloc[self.row_number]
        elif self.bet_closed:
            print(f"Bid result: {self.result:.2f}")
            self.bet_closed = False
            bid_prediction = 0
            self.row_number += 1
            while bid_prediction != 1: 

                self.step_dict = {
                    "step": 0, "time": None, 
                    "action": None, "bet_placed": None, "position_action": None,
                    "close": None, "equity": 0,  "movement": 0, "open": None,
                    "realized_profit": None,
                    "total_reward_step_raw": 0, "total_normalized_step_reward": 0,
                }

                bid_prediction, probs = self.supervised_model.make_prediction(self.env_data.iloc[self.row_number + 1 - self.sequence_length : self.row_number + 1])

                if bid_prediction == 1:
                    self.bet_type = 1
                    self.step_dict["action"] = 0
                elif bid_prediction == -1:
                    self.bet_type = 0
                    self.step_dict["action"] = 1
                else:
                    self.step_dict["action"] = 2

                self.step_dict["step"] = self.counter 
                self.step_dict["time"] = self.env_row["Time"] 
                self.step_dict["close"] = self.env_row["close"] 
                self.step_dict["open"] = self.env_row["open"] 

                self.price_previous = self.env_data["close"].iloc[self.row_number -1]
                self.price_current = self.env_row["close"]
                self.entry_price = self.latest_best = self.latest_worst = self.env_data["close"].iloc[self.row_number]

                self.counter += 1
                if self.counter + 2 >= self.number_of_rows:
                    bid_prediction = 1
                    done = True
                else:
                    self.row_number += 1
                    self.env_row = self.env_data.iloc[self.row_number]
                    done = False

                self.reward_dict[f"step {self.counter}"] = self.step_dict

                info = {"Total profit": 0, "reward_dict": self.reward_dict}

                terminated = done

                if bid_prediction == 1 and not done:

                    self._last_episode_starts = np.array([True], dtype=bool)

                    final_dict = self.initialize_data()

                    self.reset_lstm = True

                    return final_dict, self.reward, terminated, False, info  

        self.price_previous = self.env_data["close"].iloc[self.row_number -1]
        self.price_current = self.env_row["close"]

        if self.bet_type == 1:
            self.relative_profit_long = (self.env_row["close"] - self.entry_price) / self.entry_price
            self.relative_profit_short = 0
            self.current_position = "buy"
        else:
            self.relative_profit_short = -((self.env_row["close"] - self.entry_price) / self.entry_price)
            self.relative_profit_long = 0
            self.current_position = "short"

        new_row = self.env_data.iloc[self.row_number].copy().to_frame().T
        
        new_row["relative_profit_long"] = np.array(self.relative_profit_long, dtype=np.float32)
        new_row["relative_profit_short"] = np.array(self.relative_profit_short, dtype=np.float32)
        new_row["bet_type"] = np.array(self.bet_type, dtype=np.float32)
        new_row["holding_bars"] = np.array(np.log1p(self.holding_bars) / np.log1p(100), dtype=np.float32)

        scaled_row_data = scalers.transform_data(new_row, self.fitted_scaler)

        trend_features_for_scaling = list(self.fitted_trend_scaler.feature_names_in_)
        scaled_row_data[trend_features_for_scaling] = self.fitted_trend_scaler.transform(scaled_row_data[trend_features_for_scaling])  
    
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

            self.render()

            self.total_profit = sum(self.profit_list) 

            if len(self.eval_profits) == self.n_eval_episodes:
                self.eval_profits = []

            self.eval_profits.append(self.total_profit)

            info = {"Total profit": round(self.total_profit, 2), "Env number": self.env_number, "real_profits": self.eval_profits, "reward_dict": self.reward_dict}

            self.episode_dict[f"{self.timesteps}"] = self.reward_dict
            self.reward_dict = {}
            
            self.env_number += 1
                
        self.timesteps += 1
        terminated = done
        self.last_info = info

        return final_dict, self.reward, terminated, False, info  
    
    def episode_plot(self, reward_dict):

        reward_dataframe = pd.DataFrame.from_dict(reward_dict, orient='index')
        reward_dataframe = reward_dataframe.reset_index(drop=True)

        reward_dataframe.to_excel(f"{self.folder_path}/{self.date}_{self.timesteps}.xlsx", index = False)

        total_reward = sum(reward_dataframe["total_normalized_step_reward"])

        buy = reward_dataframe.index[reward_dataframe['action']==0].tolist() 
        sell = reward_dataframe.index[reward_dataframe['action']==1].tolist() 
        no_bet = reward_dataframe.index[reward_dataframe['action']==2].tolist() 
        close = reward_dataframe.index[reward_dataframe['position_action']==1].tolist() 
        hold = reward_dataframe.index[reward_dataframe['position_action']==0].tolist() 
        profit_list = reward_dataframe["realized_profit"][reward_dataframe['position_action']==1].tolist() 
       
        plt.plot(reward_dataframe["step"], reward_dataframe['close'], color = 'red')
        plt.plot(reward_dataframe["step"], reward_dataframe['close'], '^', markevery=buy, label='Buy', color="Green")
        plt.plot(reward_dataframe["step"], reward_dataframe["close"], 'x', markevery=close, label='Close', color = 'black')
        plt.plot(reward_dataframe["step"], reward_dataframe["close"], 'x', markevery=hold, label='Hold', color = 'yellow')

        for i, value in zip(close, profit_list):
            plt.text(reward_dataframe["step"][i], reward_dataframe["close"][i] + 2, round(value, 2), fontsize=8, color='green', ha='right')

        total_profit = round(sum(profit_list), 2)    
        self.cumulative_profit += total_profit

        self.test_profits.append(total_profit)

        date = self.env_data["original_datetime"].iloc[-1].date()
        date_str = date.strftime("%Y_%m_%d")

        logger.info(f"Environment {self.env_number}. {date_str} Total Profit: {total_profit}")
        logger.info(f"Cumulative Profit: {self.cumulative_profit}")

        if len(self.test_profits) == self.n_eval_episodes:
            self.test_profits = []
            self.cumulative_profit = 0

        plt.title(f"Environement: {self.env_number}, Total Profit: {total_profit}, Total Reward: {total_reward}")
        plt.xticks(rotation=90)
        plt.legend()

        file_name = f"{date_str}_env_number_{self.env_number}"
        file_path = f"{self.image_files}/{file_name}.png"

        plt.savefig(file_path)

        if self.still_plot:
            plt.show()
        else:
            plt.show(block = False)
            plt.pause(3)
            plt.close()
    
    def reset(self, seed=None, options=None):

        if seed is not None:
            np.random.seed(seed)

        self._last_episode_starts = np.array([True], dtype=bool)
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
        self.result = None

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

        dataset_list = self.dataset_list
        self.amount_of_envs = len(dataset_list) - 1

        if self.amount_of_envs == self.env_number:
            self.env_number = 0

        test_data_path = os.getenv("TRAIN_TEST_DATASETS")
        env = os.path.join(self.base_dir, test_data_path, f"{dataset_list[self.env_number]}")

        self.env_data = pd.read_excel(env, sheet_name = "Sheet1", header = 0)

        self.env_data = self.add_long_and_short_slopes(self.env_data)[20:]

        self.env_data.reset_index(inplace=True)

        self.daily_max_close = self.env_data["close"].max()
        self.daily_min_close = self.env_data["close"].min()
        self.daily_price_range = self.daily_max_close - self.daily_min_close

        self.row_number = self.sequence_length
        self.number_of_rows = len(self.env_data) - self.row_number

        bid_prediction = 0

        while bid_prediction != 1: 

            self.step_dict = {
                "step": 0, "time": None, 
                "action": None, "bet_placed": None, "position_action": None,
                "close": None, "equity": 0,  "movement": 0, "open": None,
                "realized_profit": None,
                "total_reward_step_raw": 0, "total_normalized_step_reward": 0,
            }

            bid_prediction, probs = self.supervised_model.make_prediction(self.env_data.iloc[self.row_number + 1 - self.sequence_length : self.row_number + 1])

            self.price_previous = self.env_data["close"].iloc[self.row_number -1]
            self.price_current = self.env_data["close"].iloc[self.row_number]

            self.entry_price = self.latest_best = self.latest_worst = self.env_data["close"].iloc[self.row_number]

            if bid_prediction == 1:
                self.bet_type = 1
                self.step_dict["action"] = 0
            elif bid_prediction == -1:
                self.bet_type = 0
                self.step_dict["action"] = 1
            else:
                self.step_dict["action"] = 2

            if self.counter + 2 >= self.number_of_rows:
                bid_prediction = 1
            else:
                self.env_row = self.env_data.iloc[self.row_number]

            self.step_dict["step"] = self.counter 
            self.step_dict["time"] = self.env_row["Time"] 
            self.step_dict["close"] = self.env_row["close"] 
            self.step_dict["open"] = self.env_row["open"] 

            self.reward_dict[f"step {self.counter}"] = self.step_dict

            self.counter += 1
            self.row_number += 1

        self.reset_lstm = True
        self.env_row = self.env_data.iloc[self.row_number]

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

        scaled_data = scalers.transform_data(self.env_data.iloc[self.row_number + 1 - self.sequence_length : self.row_number + 1], self.fitted_scaler)

        trend_features_for_scaling = list(self.fitted_trend_scaler.feature_names_in_)
        scaled_data[trend_features_for_scaling] = self.fitted_trend_scaler.transform(scaled_data[trend_features_for_scaling])

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
    
class TradingBotRLEvalCloserShort(TradingBotRLEval):
    def __init__(self, render_mode = None, dataset_list = None, mode = "train", plot = True, still_plot = False, n_eval_episodes = 5, eval_logger = None, 
                 env_id = 0, total_timesteps = 1000000):
        super().__init__(render_mode=render_mode, dataset_list=dataset_list, mode=mode, plot=plot, still_plot=still_plot, n_eval_episodes=n_eval_episodes, 
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

    def initialize_data(self):

        self.holding_bars = 0

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

        scaled_data = scalers.transform_data(self.env_data.iloc[self.row_number + 1 - self.sequence_length : self.row_number + 1], self.fitted_scaler)

        trend_features_for_scaling = list(self.fitted_trend_scaler.feature_names_in_)
        scaled_data[trend_features_for_scaling] = self.fitted_trend_scaler.transform(scaled_data[trend_features_for_scaling])

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

        return reset_obs

        
    def step(self, action):

        self._last_episode_starts = np.array([False], dtype=bool)

        print(f"[PID {os.getpid()}] Step {self.counter} in Environment {self.env_id}")

        self.counter += 1
        self.action = action
        self.reward = 0
        self.steps_counter += 1

        self.reward_option_1()
        self.track_env_reward()

        self.reset_lstm = False

        if self.counter + 2 >= self.number_of_rows:
            done = True
        elif not self.bet_closed: 
            done = False
            self.row_number += 1
            self.env_row = self.env_data.iloc[self.row_number]
        elif self.bet_closed:
            print(f"Bid result: {self.result:.2f}")
            self.bet_closed = False
            bid_prediction = 0
            self.row_number += 1
            while bid_prediction != -1: 

                self.step_dict = {
                    "step": 0, "time": None, 
                    "action": None, "bet_placed": None, "position_action": None,
                    "close": None, "equity": 0,  "movement": 0, "open": None,
                    "realized_profit": None,
                    "total_reward_step_raw": 0, "total_normalized_step_reward": 0,
                }

                bid_prediction, probs = self.supervised_model.make_prediction(self.env_data.iloc[self.row_number + 1 - self.sequence_length : self.row_number + 1])

                if bid_prediction == 1:
                    self.bet_type = 1
                    self.step_dict["action"] = 0
                elif bid_prediction == -1:
                    self.bet_type = 0
                    self.step_dict["action"] = 1
                else:
                    self.step_dict["action"] = 2

                self.step_dict["step"] = self.counter 
                self.step_dict["time"] = self.env_row["Time"] 
                self.step_dict["close"] = self.env_row["close"] 
                self.step_dict["open"] = self.env_row["open"] 

                self.price_previous = self.env_data["close"].iloc[self.row_number -1]
                self.price_current = self.env_row["close"]
                self.entry_price = self.latest_best = self.latest_worst = self.env_data["close"].iloc[self.row_number]

                self.counter += 1
                if self.counter + 2 >= self.number_of_rows:
                    bid_prediction = -1
                    done = True
                else:
                    self.row_number += 1
                    self.env_row = self.env_data.iloc[self.row_number]
                    done = False

                self.reward_dict[f"step {self.counter}"] = self.step_dict

                info = {"Total profit": 0, "reward_dict": self.reward_dict}

                terminated = done

                if bid_prediction == -1 and not done:

                    self._last_episode_starts = np.array([True], dtype=bool)

                    final_dict = self.initialize_data()

                    self.reset_lstm = True

                    return final_dict, self.reward, terminated, False, info  

        self.price_previous = self.env_data["close"].iloc[self.row_number -1]
        self.price_current = self.env_row["close"]

        if self.bet_type == 1:
            self.relative_profit_long = (self.env_row["close"] - self.entry_price) / self.entry_price
            self.relative_profit_short = 0
            self.current_position = "buy"
        else:
            self.relative_profit_short = -((self.env_row["close"] - self.entry_price) / self.entry_price)
            self.relative_profit_long = 0
            self.current_position = "short"

        new_row = self.env_data.iloc[self.row_number].copy().to_frame().T
        
        new_row["relative_profit_long"] = np.array(self.relative_profit_long, dtype=np.float32)
        new_row["relative_profit_short"] = np.array(self.relative_profit_short, dtype=np.float32)
        new_row["bet_type"] = np.array(self.bet_type, dtype=np.float32)
        new_row["holding_bars"] = np.array(np.log1p(self.holding_bars) / np.log1p(100), dtype=np.float32)

        scaled_row_data = scalers.transform_data(new_row, self.fitted_scaler)

        trend_features_for_scaling = list(self.fitted_trend_scaler.feature_names_in_)
        scaled_row_data[trend_features_for_scaling] = self.fitted_trend_scaler.transform(scaled_row_data[trend_features_for_scaling])  
    
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

            self.render()

            self.total_profit = sum(self.profit_list) 

            if len(self.eval_profits) == self.n_eval_episodes:
                self.eval_profits = []

            self.eval_profits.append(self.total_profit)

            info = {"Total profit": round(self.total_profit, 2), "Env number": self.env_number, "real_profits": self.eval_profits, "reward_dict": self.reward_dict}

            self.episode_dict[f"{self.timesteps}"] = self.reward_dict
            self.reward_dict = {}
            
            self.env_number += 1
                
        self.timesteps += 1
        terminated = done
        self.last_info = info

        return final_dict, self.reward, terminated, False, info  
    
    def episode_plot(self, reward_dict):

        reward_dataframe = pd.DataFrame.from_dict(reward_dict, orient='index')
        reward_dataframe = reward_dataframe.reset_index(drop=True)

        reward_dataframe.to_excel(f"{self.folder_path}/{self.date}_{self.timesteps}.xlsx", index = False)

        total_reward = sum(reward_dataframe["total_normalized_step_reward"])

        buy = reward_dataframe.index[reward_dataframe['action']==0].tolist() 
        sell = reward_dataframe.index[reward_dataframe['action']==1].tolist() 
        no_bet = reward_dataframe.index[reward_dataframe['action']==2].tolist() 
        close = reward_dataframe.index[reward_dataframe['position_action']==1].tolist() 
        hold = reward_dataframe.index[reward_dataframe['position_action']==0].tolist() 
        profit_list = reward_dataframe["realized_profit"][reward_dataframe['position_action']==1].tolist() 
       
        plt.plot(reward_dataframe["step"], reward_dataframe['close'], color = 'red')
        plt.plot(reward_dataframe["step"], reward_dataframe['close'], 'v', markevery=sell, label='Sell', color="blue")
        plt.plot(reward_dataframe["step"], reward_dataframe["close"], 'x', markevery=close, label='Close', color = 'black')
        plt.plot(reward_dataframe["step"], reward_dataframe["close"], 'x', markevery=hold, label='Hold', color = 'yellow')

        for i, value in zip(close, profit_list):
            plt.text(reward_dataframe["step"][i], reward_dataframe["close"][i] + 2, round(value, 2), fontsize=8, color='green', ha='right')

        total_profit = round(sum(profit_list), 2)    
        self.cumulative_profit += total_profit

        self.test_profits.append(total_profit)

        date = self.env_data["original_datetime"].iloc[-1].date()
        date_str = date.strftime("%Y_%m_%d")

        logger.info(f"Environment {self.env_number}. {date_str} Total Profit: {total_profit}")
        logger.info(f"Cumulative Profit: {self.cumulative_profit}")

        if len(self.test_profits) == self.n_eval_episodes:
            self.test_profits = []
            self.cumulative_profit = 0

        plt.title(f"Environement: {self.env_number}, Total Profit: {total_profit}, Total Reward: {total_reward}")
        plt.xticks(rotation=90)
        plt.legend()

        file_name = f"{date_str}_env_number_{self.env_number}"
        file_path = f"{self.image_files}/{file_name}.png"

        plt.savefig(file_path)

        if self.still_plot:
            plt.show()
        else:
            plt.show(block = False)
            plt.pause(3)
            plt.close()
    
    def reset(self, seed=None, options=None):

        if seed is not None:
            np.random.seed(seed)

        self._last_episode_starts = np.array([True], dtype=bool)
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
        self.result = None

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

        dataset_list = self.dataset_list
        self.amount_of_envs = len(dataset_list) - 1

        if self.amount_of_envs == self.env_number:
            self.env_number = 0

        test_data_path = os.getenv("TRAIN_TEST_DATASETS")
        env = os.path.join(self.base_dir, test_data_path, f"{dataset_list[self.env_number]}")

        self.env_data = pd.read_excel(env, sheet_name = "Sheet1", header = 0)

        self.env_data = self.add_long_and_short_slopes(self.env_data)[20:]

        self.env_data.reset_index(inplace=True)

        self.daily_max_close = self.env_data["close"].max()
        self.daily_min_close = self.env_data["close"].min()
        self.daily_price_range = self.daily_max_close - self.daily_min_close

        self.row_number = self.sequence_length
        self.number_of_rows = len(self.env_data) - self.row_number

        bid_prediction = 0

        while bid_prediction != -1: 

            self.step_dict = {
                "step": 0, "time": None, 
                "action": None, "bet_placed": None, "position_action": None,
                "close": None, "equity": 0,  "movement": 0, "open": None,
                "realized_profit": None,
                "total_reward_step_raw": 0, "total_normalized_step_reward": 0,
            }

            bid_prediction, probs = self.supervised_model.make_prediction(self.env_data.iloc[self.row_number + 1 - self.sequence_length : self.row_number + 1])

            self.price_previous = self.env_data["close"].iloc[self.row_number -1]
            self.price_current = self.env_data["close"].iloc[self.row_number]

            self.entry_price = self.latest_best = self.latest_worst = self.env_data["close"].iloc[self.row_number]

            if bid_prediction == 1:
                self.bet_type = 1
                self.step_dict["action"] = 0
            elif bid_prediction == -1:
                self.bet_type = 0
                self.step_dict["action"] = 1
            else:
                self.step_dict["action"] = 2

            if self.counter + 2 >= self.number_of_rows:
                bid_prediction = -1
            else:
                self.env_row = self.env_data.iloc[self.row_number]

            self.step_dict["step"] = self.counter 
            self.step_dict["time"] = self.env_row["Time"] 
            self.step_dict["close"] = self.env_row["close"] 
            self.step_dict["open"] = self.env_row["open"] 

            self.reward_dict[f"step {self.counter}"] = self.step_dict

            self.counter += 1
            self.row_number += 1

        self.reset_lstm = True
        self.env_row = self.env_data.iloc[self.row_number]

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

        scaled_data = scalers.transform_data(self.env_data.iloc[self.row_number + 1 - self.sequence_length : self.row_number + 1], self.fitted_scaler)

        trend_features_for_scaling = list(self.fitted_trend_scaler.feature_names_in_)
        scaled_data[trend_features_for_scaling] = self.fitted_trend_scaler.transform(scaled_data[trend_features_for_scaling])

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

    dataset_list = os.listdir("envs/test_data")

    env = TradingBotRLEvalCloserLong(dataset_list = dataset_list, render_mode="human")

    check_env(env)
    episodes = 10
    for episode in range(0,episodes):
        print(episode)
        action = env.action_space.sample()
        observation, reward, done, _, info = env.step(action)
        print(observation)
        
        if done:
            env.reset()

