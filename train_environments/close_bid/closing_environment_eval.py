import os
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3 import PPO
import pandas as pd
from stable_baselines3.common.env_checker import check_env
import random
import matplotlib.pyplot as plt
import random
import logging
from collections import deque
import scalers as scalers
import joblib
import datetime
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from test_environment.supervised_learning import SupervisedLearning

logger = logging.getLogger()

class TradingBotRLEval(gym.Env):
    def __init__(self, render_mode = None, dataset_list = None, mode = "train", plot = True, still_plot = False, n_eval_episodes = 5, eval_logger = None, 
                 env_id = 0, total_timesteps = 1000000):

        self.mode = mode
        self.eval_logger = eval_logger
        self.render_mode = render_mode
        self.env_id = env_id
        self.total_timesteps = total_timesteps

        date = datetime.datetime.now()
        self.date = date.strftime("%Y_%m_%d_%H_%M_%S")

        if self.render_mode == "human":
            self.folder_path = f"logs/eval_runs/{self.date}"
            os.makedirs(self.folder_path, exist_ok=True)

        if self.mode != "train" and self.mode != "test" and self.mode != "production":
            raise Exception(f"Invalid mode {self.mode}")
        
        logger.setLevel(logging.INFO)
        self.n_eval_episodes = n_eval_episodes

        self.plot = plot
        self.timesteps = 0
        self.steps_counter = 0
        self.env_rewards = {}             
        self.trade_history = []
        self.episode_trade_history = []
        self.profit_history = []
        self.episode_profit_history = []
        self.unrealized_reward = []
        self.realized_reward = []
        self.average_equity = []
        self.average_equity_episode = []
        self.equity = 2000
        self.position_counts = {0: 0, 1: 0, 2: 0}
        self.trade_specs = {}
        self.eval_profits = []
        self.env_success_rates = {}
        self.env_steps = {} 
        self.pending_buffer_rewards = []

        self.action = None
        self.counter = 0

        self.bet_action_list = []
        self.position_action_list = []
        self.reward_dict = {}
        self.episode_dict = {}

        self.bet_placed = False

        self.total_profit = 0
        self.relative_profit = 0
        self.profit_list = []
        self.cumulative_profit = 0
        self.test_profits = []

        self.still_plot = still_plot

        self.dataset_list = dataset_list

        self.amount_of_envs = None
        self.env_number = 0

        self.number_of_rows = None
        self.row_number = None

        self.reward = 0

        self.action_space = spaces.Discrete(2)
           
        self.sequence_length = 32

        self.attributes = {
            "trade_state": {
                    "relative_profit_long":{"scale": False}, 
                    "relative_profit_short":{"scale": False},
                    "bet_type": {"scale": False},
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
                    "rsi":{"scale": True}, 
                    "mach_hist":{"scale": True}, 
                    "roc_30m":{"scale": True}, 
                    "adx_14":{"scale": True}
                },
                "market_condition": {
                    "atr":{"scale": True}, 
                    "garman_vol":{"scale": True}, 
                    "bb_%b":{"scale": True}, 
                    "bb_width":{"scale": True}, 
                    "volatility_ratio":{"scale": True}, 
                    "obv":{"scale": True}, 
                    "volume_acceleration":{"scale": True}, 
                    "vwap":{"scale": True}, 
                    "vol_weighted_momentum":{"scale": True}
                },

                "trend_features": {
                    "long_term_slope": {"scale": True},
                    "short_term_slope": {"scale": True},
                    "long_term_slope_delta": {"scale": True},
                    "short_term_slope_delta": {"scale": True},
                    "slope_ema_diff": {"scale": True},
                    "trend_strength_std": {"scale": True},
                    "trend_agreement": {"scale": False},
                    "long_slope_angle": {"scale": True},
                    "short_slope_angle": {"scale": True},
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
        
        self.seq_buffer = deque(maxlen=self.sequence_length)
        self.seq_buffer_scaled = deque(maxlen=self.sequence_length)

        self.base_dir = os.path.dirname(__file__)
    
        self.fitted_scaler = joblib.load(os.getenv("GENERAL_SCALER_TEST"))
        self.fitted_trend_scaler = joblib.load(os.getenv("TREND_SCALER_TEST"))

        self.data_collection = {
            "total_profit_list": [], 
            "episode_profit_list": [], 
            "total_reward_list": [],
            "episode_reward_list": [],
            "total_closing_reward_list": [],
            "episode_closing_reward_list": [],
            "all_rewards": [],
            "all_profits": [],
            "good_entry": 0,
            "bad_entry": 0
            }
        
        self.supervised_model = SupervisedLearning()

        self.image_files = f"logs/{self.date}_images"

        os.makedirs(self.image_files, exist_ok=True)

        logger.info(f"Environment {env_id} initialized.")

    def reward_option_1(self):

        position_action = self.action

        self.step_dict = {
            "step": 0, "time": None, 
            "action": None, "bet_placed": None, "position_action": None,
            "close": None, "equity": 0,  "movement": 0, "open": None,
            "realized_profit": None,
            "total_reward_step_raw": 0, "total_normalized_step_reward": 0,
        }

        self.step_dict["step"] = self.counter 
        self.step_dict["time"] = self.env_row["Time"] 
        self.step_dict["close"] = self.env_row["close"] 
        self.step_dict["open"] = self.env_row["open"] 

        movement = 0
        realized_profit = 0

        logger.debug(f"[Step {self.counter}] Position Action: {position_action}")

        self.position_counts[position_action] += 1

        if position_action == 0:

            self.holding_bars += 1
            holding_reward = 0
            improving = False
            worsen = False

            if self.current_position == "buy":
                self.relative_profit_long = (self.step_dict["close"] - self.entry_price) / self.entry_price
                movement = self.step_dict["close"] - self.entry_price
                current_movement = self.step_dict["close"] - self.step_dict["open"]
                delta = self.step_dict["close"] - self.latest_best
                if self.step_dict["close"] > self.latest_best and self.holding_bars > 1:
                    improving = True
                    self.latest_best = self.step_dict["close"]
                if self.step_dict["close"] < self.latest_best and self.holding_bars > 1:
                    worsen = True
                    self.latest_worst = self.step_dict["close"]
                if self.step_dict["close"] > self.step_dict["open"]:
                    self.holding_improvement += 1
                    self.positive_holding_bars += 1
                else:
                    self.holding_improvement -= 1

            elif self.current_position == "short":
                self.relative_profit_short = -((self.step_dict["close"] - self.entry_price) / self.entry_price)
                movement = self.entry_price - self.step_dict["close"]
                current_movement = self.step_dict["open"] - self.step_dict["close"]
                delta = self.latest_best - self.step_dict["close"]
                if self.step_dict["close"] < self.latest_best and self.holding_bars > 1:
                    improving = True
                    self.latest_best = self.step_dict["close"]
                if self.step_dict["close"] > self.latest_best and self.holding_bars > 1:
                    worsen = True
                    self.latest_worst = self.step_dict["close"]
                if self.step_dict["open"] > self.step_dict["close"]:
                    self.holding_improvement += 1
                    self.positive_holding_bars += 1
                else:
                    self.holding_improvement -= 1

            self.reward += (movement / self.daily_price_range) * 0.01

            if improving:
                self.reward += 0.001

            if self.holding_bars > 12:
                self.reward -= (self.holding_bars - 12) * 0.001
            
            self.progress = (self.timesteps * 6) / self.total_timesteps
        
            self.position_action_list.append("hold")

        elif position_action == 1:

            if self.current_position == "buy":
                movement = self.step_dict["close"] - self.entry_price
                real_profit = movement * 1.75
                self.result = movement * 0.07 * 25
        
            elif self.current_position == "short":
                movement = self.entry_price - self.step_dict["close"]
                real_profit = movement * 1.75
                self.result = movement * 0.07 * 25

            self.step_dict["realized_profit"] = real_profit
            self.profit_list.append(real_profit)
            self.profit_history.append(real_profit)
            self.episode_profit_history.append(real_profit)
            realized_profit = real_profit

            if movement > 0:
                self.reward += (movement / self.daily_price_range) + (np.clip(self.holding_improvement, 1, 10) * 0.01)
            else:
                self.reward += (movement / self.daily_price_range)

            if movement < 0:
                self.consecutive_loss += 1
                self.previous_result = "loss"
            else:
                self.consecutive_loss = 0
                self.previous_result = "profit"

            self.bet_placed = False
            self.positive_holding_bars = 0
            self.negative_holding_bars = 0
            self.total_holding_reward = 0
            self.total_bet_reward = 0
            self.good_entry = False
            self.holding_improvement = 0

            self.closed_trades += 1

            self.position_action_list.append("close")

            logger.debug(f"[Step {self.counter}] Real Profit: {real_profit}")

            self.update_equity(self.reward, realized_profit, position_action)

            self.data_collection["total_closing_reward_list"].append(self.reward)
            self.data_collection["episode_closing_reward_list"].append(self.reward)

            self.data_collection["total_profit_list"].append(movement * 1.75)
            self.data_collection["episode_profit_list"].append(movement * 1.75)

            self.bet_closed = True

        self.step_dict["position_action"] = position_action

        self.step_dict["bet_placed"] = self.bet_placed
        self.step_dict["total_normalized_step_reward"] = self.reward
        self.step_dict["equity"] = self.equity
        self.step_dict["movement"] = movement

        self.reward_dict[f"step {self.counter}"] = self.step_dict

        self.data_collection["total_reward_list"].append(self.reward)
        self.data_collection["episode_reward_list"].append(self.reward)

        self.data_collection["all_rewards"].append(self.reward)
        self.data_collection["all_profits"].append(movement)

        self.reward_list_testing.append(self.reward)

    def update_equity(self, reward, realized_profit, position_action):
        if not np.isfinite(reward):
            logger.info(f"Invalid reward detected: {reward}, setting to 0.")
            reward = 0
        self.equity += realized_profit
        self.peak_equity = max(self.peak_equity, self.equity)
        self.trade_history.append(reward)
        self.episode_trade_history.append(reward)
    
    def track_env_reward(self):
        env_id = self.env_number  # Assuming each environment has a unique ID

        # Initialize if environment is new
        if env_id not in self.env_rewards:
            self.env_rewards[env_id] = 0
            self.env_success_rates[env_id] = 0
            self.env_steps[env_id] = 0

        # Update statistics
        self.env_rewards[env_id] += self.reward
        self.env_steps[env_id] += 1
        if self.reward > 0:  # Define "success" as positive reward
            self.env_success_rates[env_id] += 1

    def initialize_data(self):

        self.holding_bars = 0

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

        scaled_data = scalers.transform_data(self.env_data.iloc[self.row_number + 1 - self.sequence_length : self.row_number + 1], self.fitted_scaler)

        trend_featrues_for_scaling = [
            attribute for attribute, option in self.attributes["trend_features"].items()
            if option.get("scale") is True
        ]
        
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

        return reset_obs
    
    @property
    def _last_episode_starts(self):
        return self.__last_episode_starts

    @_last_episode_starts.setter
    def _last_episode_starts(self, value):
        self.__last_episode_starts = value


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

        if self.counter >= self.number_of_rows:
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
            while bid_prediction != 1 and bid_prediction != -1: 

                self.step_dict = {
                    "step": 0, "time": None, 
                    "action": None, "bet_placed": None, "position_action": None,
                    "close": None, "equity": 0,  "movement": 0, "open": None,
                    "realized_profit": None,
                    "total_reward_step_raw": 0, "total_normalized_step_reward": 0,
                }

                bid_prediction, probs = self.supervised_model.make_prediction(self.env_data.iloc[self.row_number  + 1 - self.sequence_length : self.row_number + 1])

                self.counter += 1
                if self.counter >= self.number_of_rows:
                    bid_prediction = 1
                    done = True
                else:
                    self.row_number += 1
                    self.env_row = self.env_data.iloc[self.row_number]
                    done = False

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

                self.reward_dict[f"step {self.counter}"] = self.step_dict

                info = {"Total profit": 0, "reward_dict": self.reward_dict}

                terminated = done

                if bid_prediction != 0 and not done:

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

        new_row = self.env_row[self.scale_attributes + ["hour_sin", "hour_cos", "trend_agreement"]].copy().to_frame().T
        
        new_row["relative_profit_long"] = np.array(self.relative_profit_long, dtype=np.float32)
        new_row["relative_profit_short"] = np.array(self.relative_profit_short, dtype=np.float32)
        new_row["bet_type"] = np.array(self.bet_type, dtype=np.float32)
        new_row["holding_bars"] = np.array(np.log1p(self.holding_bars) / np.log1p(100), dtype=np.float32)

        scaled_row_data = scalers.transform_data(new_row, self.fitted_scaler)

        trend_features_for_scaling = [
            attribute for attribute, option in self.attributes["trend_features"].items()
            if option.get("scale") is True
        ]
        
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
    
    def step_info_logging(self):

        total_closing_reward_rate = sum(1 for x in self.data_collection["total_closing_reward_list"] if x > 0) / len(self.data_collection["total_closing_reward_list"])
        epsiode_closing_reward_rate = sum(1 for x in self.data_collection["episode_closing_reward_list"] if x > 0) / len(self.data_collection["episode_closing_reward_list"])
        equity_average = np.average(self.average_equity)
        equity_episode_average = np.average(self.average_equity_episode)
        real_profits_win_rate = sum(1 for x in self.data_collection["total_profit_list"] if x > 0) / len(self.data_collection["total_profit_list"])
        real_episode_profits_win_rate = sum(1 for x in self.data_collection["episode_profit_list"] if x > 0) / len(self.data_collection["episode_profit_list"])
        real_reward_win_rate = sum(1 for x in self.data_collection["total_reward_list"] if x > 0) / len(self.data_collection["total_reward_list"])
        total_reward_mean = np.mean(self.data_collection["total_reward_list"])
        real_episode_reward_win_rate = sum(1 for x in self.data_collection["episode_reward_list"] if x > 0) / len(self.data_collection["episode_reward_list"])
        total_episode_profit = sum(self.data_collection["episode_profit_list"])
        total_episode_reward = sum(self.data_collection["episode_reward_list"])
        rewards = self.data_collection["all_rewards"]
        profits = self.data_collection["all_profits"]
        corr, _ = pearsonr(rewards, profits)
        logger.info(f"Total win rate (rewards): {real_reward_win_rate * 100:.2f}%")
        logger.info(f"Total mean reward: {total_reward_mean:.6f}")
        logger.info(f"Episode win rate (rewards): {real_episode_reward_win_rate * 100:.2f}%")
        logger.info(f"Total episode reward: {total_episode_reward:.2f}")
        logger.info(f"Total closing reward rate: {total_closing_reward_rate * 100:.2f}%")
        logger.info(f"Episode closing reward rate: {epsiode_closing_reward_rate * 100:.2f}%")
        logger.info(f"Total win rate (profit): {real_profits_win_rate * 100:.2f}%")
        logger.info(f"Episode win rate (profit): {real_episode_profits_win_rate * 100:.2f}%")
        logger.info(f"Total episode profit: {total_episode_profit:.2f}")
        logger.info(f"Reward/Profit Correlation: {corr:.5f}")
        logger.info(f"Average equity: {equity_average:.2f}")
        logger.info(f"Episode average equity: {equity_episode_average:.2f}")
        logger.info(f"Last environment: {self.env_number}")
        logger.info(f"Current time step: {self.timesteps}")
        logger.info(f"Position action distribution {self.position_counts}")
        #self.reward_plot()
        negative_environments = len(dict((k, v) for k, v in self.env_rewards.items() if v < 0).keys())
        logger.info(f"Currently there are {negative_environments} negative reward environments out of {len(self.env_rewards.keys())}")
        self.steps_counter = 0
        self.episode_trade_history = []
        self.episode_profit_history = []
        self.average_equity_episode = []
        self.pending_reward_buffer_values = []
        self.position_counts = {0: 0, 1: 0, 2: 0}
        self.data_collection["episode_closing_reward_list"] = []
        self.data_collection["episode_profit_list"] = []
        self.data_collection["episode_reward_list"] = []

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
        plt.plot(reward_dataframe["step"], reward_dataframe["close"], 'v', markevery=sell, label='Sell', color = 'blue')
        plt.plot(reward_dataframe["step"], reward_dataframe["close"], 'x', markevery=close, label='Close', color = 'black')
        plt.plot(reward_dataframe["step"], reward_dataframe["close"], 'x', markevery=hold, label='Hold', color = 'yellow')
        #plt.plot(reward_dataframe["step"], reward_dataframe["close"], 'x', markevery=no_bet, label='No bet', color = 'red')

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

        plt.title(f"Environment: {self.env_number}, Total Profit: {total_profit}, Total Reward: {total_reward}")
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

    def render(self, mode = "human"):
        self.episode_plot(self.reward_dict)

    def select_bid_window(self, df):
       
        assert "meta_model_long" in df.columns and "meta_model_short" in df.columns, \
            "Required columns not found in DataFrame."
        
        df["original_datetime"] = pd.to_datetime(df["original_datetime"])

        # Step 1: Randomly choose bid direction
        bid_type = random.choice(["long", "short"])
        #print(f"Selected bid type: {bid_type}")

        if bid_type == "long":
            self.bet_type = 1
        else:
            self.bet_type = 0

        # Step 2: Filter rows where meta prediction > 0.8
        col = f"meta_model_{bid_type}"

        valid_range = 200 

        df_filtered = df[
            (df[col] > 0.8) &
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

        env = os.path.join(self.base_dir, "envs", "test_data", f"{dataset_list[self.env_number]}")

        self.env_data = pd.read_excel(env, sheet_name = "Sheet1", header = 0)

        self.env_data = self.add_long_and_short_slopes(self.env_data)[20:]

        self.env_data.reset_index(inplace=True)

        self.daily_max_close = self.env_data["close"].max()
        self.daily_min_close = self.env_data["close"].min()
        self.daily_price_range = self.daily_max_close - self.daily_min_close

        self.row_number = self.sequence_length
        self.number_of_rows = len(self.env_data) - self.row_number

        bid_prediction = 0

        while bid_prediction != 1 and bid_prediction != -1: 

            self.step_dict = {
                "step": 0, "time": None, 
                "action": None, "bet_placed": None, "position_action": None,
                "close": None, "equity": 0,  "movement": 0, "open": None,
                "realized_profit": None,
                "total_reward_step_raw": 0, "total_normalized_step_reward": 0,
            }

            bid_prediction, probs = self.supervised_model.make_prediction(self.env_data.iloc[self.row_number + 1 - self.sequence_length : self.row_number + 1])

            self.row_number += 1

            if bid_prediction == 1:
                self.bet_type = 1
                self.step_dict["action"] = 0
            elif bid_prediction == -1:
                self.bet_type = 0
                self.step_dict["action"] = 1
            else:
                self.step_dict["action"] = 2

            self.counter += 1

            self.env_row = self.env_data.iloc[self.row_number]

            self.step_dict["step"] = self.counter 
            self.step_dict["time"] = self.env_row["Time"] 
            self.step_dict["close"] = self.env_row["close"] 
            self.step_dict["open"] = self.env_row["open"] 

            self.reward_dict[f"step {self.counter}"] = self.step_dict

        self.reset_lstm = True

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
        raw_market_dynamics_features_chunk = self.env_data[self.attributes["market_dynamics"].keys()].iloc[self.row_number + 1 -self.sequence_length : self.row_number + 1]
        raw_momentum_features_chunk = self.env_data[self.attributes["momentum_features"].keys()].iloc[self.row_number + 1 - self.sequence_length : self.row_number + 1] 
        raw_market_condition_chunk = self.env_data[self.attributes["market_condition"].keys()].iloc[self.row_number + 1 - self.sequence_length : self.row_number + 1]
        raw_trend_featrues_chunk = self.env_data[self.attributes["trend_features"].keys()].iloc[self.row_number + 1 - self.sequence_length : self.row_number + 1]

        scaled_data = scalers.transform_data(self.env_data.iloc[self.row_number + 1 - self.sequence_length : self.row_number + 1], self.fitted_scaler)

        trend_features_for_scaling = [
            attribute for attribute, option in self.attributes["trend_features"].items()
            if option.get("scale") is True
        ]
        
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
            df["ema_diff"] = df["ema_10"] - df["ema_50"]
            df["slope_ema_diff"] = df["ema_diff"].rolling(window=5, min_periods=1).apply(calc_slope, raw=False)
        else:
            df["slope_ema_diff"] = 0.0  # Fallback if EMAs missing

        # trend_strength_std: rolling std of long-term slope
        df["trend_strength_std"] = df["long_term_slope"].rolling(window=5, min_periods=1).std().fillna(0.0)

        # trend_agreement: binary flag (1 if long and short slopes agree in sign)
        df["trend_agreement"] = (np.sign(df["long_term_slope"]) == np.sign(df["short_term_slope"])).astype(int)

        # slope angles (bounded versions of slope)
        df["long_slope_angle"] = np.arctan(df["long_term_slope"])
        df["short_slope_angle"] = np.arctan(df["short_term_slope"])

        return df


if __name__ == "__main__":

    dataset_list = os.listdir("envs/test_data")

    env = TradingBotRLEval(dataset_list = dataset_list)

    check_env(env)
    episodes = 10
    for episode in range(0,episodes):
        print(episode)
        action = env.action_space.sample()
        observation, reward, done, _, info = env.step(action)
        print(observation)
        
        if done:
            env.reset()