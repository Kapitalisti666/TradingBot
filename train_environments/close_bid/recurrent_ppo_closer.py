import datetime
import pandas as pd
import os
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn as nn
import torch as th
from stable_baselines3.common.utils import get_linear_fn
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import sync_envs_normalization
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import configure
import logging
from stable_baselines3.common.vec_env import SubprocVecEnv
from collections import deque
import closing_strategies as st
import closing_eval_strategies as steval
from dotenv import load_dotenv

load_dotenv()


date = datetime.datetime.now()
date = date.strftime("%Y_%m_%d_%H_%M_%S")

logging.basicConfig(filename = f"logs/{date}.log", format = "%(asctime)s %(message)s", filemode = "w")
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

class SaveBestExplainedVarianceAndProgitAvgCallback(BaseCallback):
    def __init__(self, verbose=1):
        super(SaveBestExplainedVarianceAndProgitAvgCallback, self).__init__(verbose)
        
        self.best_explained_variance = -np.inf
        self.best_episode_total_profit = -np.inf
        self.best_total_profit = -np.inf

        self.profits = []
        self.profits_episode = deque(maxlen=20)
        self.episode_counter = 0

        now = datetime.datetime.now()
        self.current_time = now.strftime("%d_%m_%Y_%H_%M_%S")
        self.folder_path = f"models/best_model/{self.current_time}"
        os.makedirs(self.folder_path, exist_ok=True)

    def models_folder_path(self):
        return self.folder_path

    def calculate_explained_variance(self, true_values, predicted_values):
        return 1 - np.var(true_values - predicted_values) / (np.var(true_values) + 1e-8)

    def calculate_loss(self, true_values, predicted_values):
        return np.mean((true_values - predicted_values) ** 2)

    def _on_step(self) -> bool:
        for idx, done in enumerate(self.locals["dones"]):
            if done:
                profit = self.locals["infos"][idx].get("Total profit", 0)
                self.profits.append(profit)
                self.profits_episode.append(profit)

                avg_profit = np.mean(self.profits)
                total_profit = np.sum(self.profits)

                episode_avg_profit = np.mean(self.profits_episode)
                episode_total_profit = np.sum(self.profits_episode)

                self.logger.record("train/total_profit_avg", avg_profit)
                self.logger.record("train/total_profit_total", total_profit)
                self.logger.record("train/episode_profit_avg", episode_avg_profit)
                self.logger.record("train/episode_profit_total", episode_total_profit)

        return True

    def _on_rollout_end(self) -> None:
        true_values = np.array(self.model.rollout_buffer.returns)
        predicted_values = np.array(self.model.rollout_buffer.values)

        if self.episode_counter > 2:
            if true_values.size > 0 and predicted_values.size > 0:
                explained_variance = self.calculate_explained_variance(true_values, predicted_values)
                loss = self.calculate_loss(true_values, predicted_values)
                episode_total_profit = np.sum(self.profits_episode) if self.profits_episode else 0
                total_profit = np.sum(self.profits) if self.profits else 0

                print(f"[Episode {self.episode_counter}]")
                print(f"→ Explained Variance: {explained_variance:.5f}")
                print(f"→ Total Profit: {total_profit:.2f}")
                print(f"→ Episode Total Profit: {episode_total_profit:.2f}")
                print(f"→ Loss: {loss:.5f}")

                if explained_variance > self.best_explained_variance and total_profit > self.best_total_profit:
                    self.best_explained_variance = explained_variance
                    self.best_total_profit = total_profit

                    print("✅ Both explained variance and total profit improved — Saving model")

                    timestamp = datetime.datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
                    ev_str = round(self.best_explained_variance, 5)
                    total_profit_str = round(self.best_total_profit, 2)

                    self.model.save(
                        f"{self.folder_path}/best_model_ev_{ev_str}_episode_total_profit_{total_profit_str}_{timestamp}.zip"
                    )
                else:
                    print("⏩ Not saved (conditions not met)")

        self.episode_counter += 1
    
class DynamicHyperparamCallback(BaseCallback):
    def __init__(self, total_timesteps, check_freq=10_000, verbose=1):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.check_freq = check_freq

    def _on_step(self) -> bool:
        progress_remaining = 1 - (self.model.num_timesteps / self.total_timesteps)

        self.model.clip_range = lambda _: max(0.05, 0.2 * progress_remaining) 
        self.model.ent_coef = max(0.01, 0.05 * progress_remaining)

        if self.n_calls % self.check_freq == 0:
            self.logger.record("train/ent_coef", self.model.ent_coef)
            self.logger.record("train/clip_range", self.model.clip_range(0)) 
            print(f"Step {self.n_calls}: ent_coef = {self.model.ent_coef:.6f}, clip_range = {self.model.clip_range(0):.4f}")

        return True
    
class HybridFeatureExtractorLong(BaseFeaturesExtractor):
    def __init__(self, observation_space, lstm_hidden_size=128):
        super().__init__(observation_space, features_dim=128)

        self.chunks = list(observation_space.keys())
        number_of_outputs = 0

        if "pattern" in self.chunks:
            self.lstm_pattern = nn.LSTM(
                input_size=observation_space["pattern"].shape[1],
                hidden_size=lstm_hidden_size,
                batch_first=True,
                num_layers=2,
                dropout=0.2
            )
            number_of_outputs += lstm_hidden_size

        if "market_dynamics" in self.chunks:
            self.lstm_market_dynamics = nn.LSTM(
                input_size=observation_space["market_dynamics"].shape[1],
                hidden_size=lstm_hidden_size,
                batch_first=True,
                num_layers=2,
                dropout=0.2
            )
            number_of_outputs += lstm_hidden_size

        if "trade_state" in self.chunks:
            output = 16
            self.trade_state_mlp = nn.Sequential(
                nn.Linear(observation_space["trade_state"].shape[1], 32),
                nn.ReLU(),
                nn.Linear(32, output),
                nn.ReLU()
            )
            number_of_outputs += output

        if "trade_context" in self.chunks:
            self.trade_context_mlp = nn.Sequential(
                nn.Linear(observation_space["trade_context"].shape[1], 16),
                nn.ReLU(),
                nn.Linear(16, 8),
                nn.ReLU()
            )
            number_of_outputs += 8

        if "momentum_features" in self.chunks:
            output = 16
            self.momentum_mlp = nn.Sequential(
                nn.Linear(observation_space["momentum_features"].shape[1], 32),
                nn.ReLU(),
                nn.Linear(32, output),
                nn.ReLU()
            )
            number_of_outputs += output

        if "market_condition" in self.chunks:
            output = 32
            self.market_condition_mlp = nn.Sequential(
                nn.Linear(observation_space["market_condition"].shape[1], 64),
                nn.ReLU(),
                nn.Linear(64, output),
                nn.ReLU()
            )
            number_of_outputs += output

        if "trend_features" in self.chunks:
            output = 32
            self.trend_features_mlp = nn.Sequential(
                nn.Linear(observation_space["trend_features"].shape[1], 64),
                nn.ReLU(),
                nn.Linear(64, output),
                nn.ReLU()
            )
            number_of_outputs += output

        self.final_layer = nn.Sequential(
            nn.Linear(number_of_outputs + 2, 128), 
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

    def forward(self, observations):
        out = []

        if "pattern" in self.chunks:
            lstm_pattern_input = observations["pattern"]
            _, (lstm_market_pattern_out, _) = self.lstm_pattern(lstm_pattern_input)
            lstm_market_pattern_out = lstm_market_pattern_out[-1]
            out.append(lstm_market_pattern_out)

        if "market_dynamics" in self.chunks:
            lstm_market_dynamics_input = observations["market_dynamics"]
            _, (lstm_market_dynamics_out, _) = self.lstm_market_dynamics(lstm_market_dynamics_input)
            lstm_market_dynamics_out = lstm_market_dynamics_out[-1]
            out.append(lstm_market_dynamics_out)

        if "trade_state" in self.chunks:
            trade_state_last = observations["trade_state"][:, -1, :]
            trade_state_out = self.trade_state_mlp(trade_state_last)
            out.append(trade_state_out) 
       
            relative_profit_long = trade_state_last[:, 0].unsqueeze(1) 
            holding_bars = trade_state_last[:, 1].unsqueeze(1)  

            out.extend([relative_profit_long, holding_bars])

        if "trade_context" in self.chunks:
            tc_last = observations["trade_context"][:, -1, :]
            tc_out = self.trade_context_mlp(tc_last)
            out.append(tc_out)

            holding_bars = tc_last[:,0].unsqueeze(1)

            out.extend([holding_bars])

        if "momentum_features" in self.chunks:
            momentum_last = observations["momentum_features"][:, -1, :]
            momentum_out = self.momentum_mlp(momentum_last)
            out.append(momentum_out)

        if "market_condition" in self.chunks:
            market_condition_last = observations["market_condition"][:, -1, :]
            market_condition_out = self.market_condition_mlp(market_condition_last)
            out.append(market_condition_out)

        if "trend_features" in self.chunks:
            trend_features_last = observations["trend_features"][:, -1, :]
            trend_features_out = self.trend_features_mlp(trend_features_last)
            out.append(trend_features_out)

        combined = th.cat(out, dim=1)
        return self.final_layer(combined)
    
class HybridFeatureExtractorShort(BaseFeaturesExtractor):
    def __init__(self, observation_space, lstm_hidden_size=128):
        super().__init__(observation_space, features_dim=128)

        self.chunks = list(observation_space.keys())
        number_of_outputs = 0

        if "pattern" in self.chunks:
            self.lstm_pattern = nn.LSTM(
                input_size=observation_space["pattern"].shape[1],
                hidden_size=lstm_hidden_size,
                batch_first=True,
                num_layers=2,
                dropout=0.2
            )
            number_of_outputs += lstm_hidden_size

        if "market_dynamics" in self.chunks:
            self.lstm_market_dynamics = nn.LSTM(
                input_size=observation_space["market_dynamics"].shape[1],
                hidden_size=lstm_hidden_size,
                batch_first=True,
                num_layers=2,
                dropout=0.2
            )
            number_of_outputs += lstm_hidden_size

        if "trade_state" in self.chunks:
            output = 16
            self.trade_state_mlp = nn.Sequential(
                nn.Linear(observation_space["trade_state"].shape[1], 32),
                nn.ReLU(),
                nn.Linear(32, output),
                nn.ReLU()
            )
            number_of_outputs += output

        if "trade_context" in self.chunks:
            self.trade_context_mlp = nn.Sequential(
                nn.Linear(observation_space["trade_context"].shape[1], 16),
                nn.ReLU(),
                nn.Linear(16, 8),
                nn.ReLU()
            )
            number_of_outputs += 8

        if "momentum_features" in self.chunks:
            output = 16
            self.momentum_mlp = nn.Sequential(
                nn.Linear(observation_space["momentum_features"].shape[1], 32),
                nn.ReLU(),
                nn.Linear(32, output),
                nn.ReLU()
            )
            number_of_outputs += output

        if "market_condition" in self.chunks:
            output = 32
            self.market_condition_mlp = nn.Sequential(
                nn.Linear(observation_space["market_condition"].shape[1], 64),
                nn.ReLU(),
                nn.Linear(64, output),
                nn.ReLU()
            )
            number_of_outputs += output

        if "trend_features" in self.chunks:
            output = 32
            self.trend_features_mlp = nn.Sequential(
                nn.Linear(observation_space["trend_features"].shape[1], 64),
                nn.ReLU(),
                nn.Linear(64, output),
                nn.ReLU()
            )
            number_of_outputs += output

        self.final_layer = nn.Sequential(
            nn.Linear(number_of_outputs + 2, 128), 
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

    def forward(self, observations):
        out = []

        if "pattern" in self.chunks:
            lstm_pattern_input = observations["pattern"]
            _, (lstm_market_pattern_out, _) = self.lstm_pattern(lstm_pattern_input)
            lstm_market_pattern_out = lstm_market_pattern_out[-1]
            out.append(lstm_market_pattern_out)

        if "market_dynamics" in self.chunks:
            lstm_market_dynamics_input = observations["market_dynamics"]
            _, (lstm_market_dynamics_out, _) = self.lstm_market_dynamics(lstm_market_dynamics_input)
            lstm_market_dynamics_out = lstm_market_dynamics_out[-1]
            out.append(lstm_market_dynamics_out)

        if "trade_state" in self.chunks:
            # Trade State - Use only the last step (most recent data point)
            trade_state_last = observations["trade_state"][:, -1, :]
            trade_state_out = self.trade_state_mlp(trade_state_last)
            out.append(trade_state_out) 
       
            relative_profit_short = trade_state_last[:, 0].unsqueeze(1) 
            holding_bars = trade_state_last[:, 1].unsqueeze(1)  

            out.extend([relative_profit_short, holding_bars])

        if "trade_context" in self.chunks:
            tc_last = observations["trade_context"][:, -1, :]
            tc_out = self.trade_context_mlp(tc_last)
            out.append(tc_out)

            holding_bars = tc_last[:,0].unsqueeze(1)

            out.extend([holding_bars])

        if "momentum_features" in self.chunks:
            momentum_last = observations["momentum_features"][:, -1, :]
            momentum_out = self.momentum_mlp(momentum_last)
            out.append(momentum_out)

        if "market_condition" in self.chunks:
            market_condition_last = observations["market_condition"][:, -1, :]
            market_condition_out = self.market_condition_mlp(market_condition_last)
            out.append(market_condition_out)

        if "trend_features" in self.chunks:
            trend_features_last = observations["trend_features"][:, -1, :]
            trend_features_out = self.trend_features_mlp(trend_features_last)
            out.append(trend_features_out)

        combined = th.cat(out, dim=1)
        return self.final_layer(combined)
    
class TrainProfitLogger(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.profits = []

    def _on_step(self) -> bool:
        for idx, done in enumerate(self.locals["dones"]):
            if done:
                profit = self.locals["infos"][idx].get("Total profit", 0)
                self.profits.append(profit)

                avg_profit = np.mean(self.profits) if len(self.profits) > 0 else 0
                total_profit = sum(self.profits) if len(self.profits) > 0 else 0

                self.logger.record("train/real_profit_avg", avg_profit)
                self.logger.record("train/real_profit_total", total_profit)

        return True
    
class EvalProfitLogger(EvalCallback):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.episode_profits = []
        self.best_total_profit = float("-inf")

    def _on_step(self) -> bool:
        continue_training = True

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            if self.model.get_vec_normalize_env() is not None:
                try:
                    sync_envs_normalization(self.training_env, self.eval_env)
                except AttributeError as e:
                    raise AssertionError(
                        "Training and eval env are not wrapped the same way."
                    ) from e

            self._is_success_buffer = []

            episode_rewards, episode_lengths = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                render=self.render,
                deterministic=self.deterministic,
                return_episode_rewards=True,
                warn=self.warn,
                callback=self._log_success_callback,
            )

            if "real_profits" in self.eval_env.get_attr("last_info")[0]:
                profits = self.eval_env.get_attr("last_info")[0]["real_profits"]
                mean_profit = np.mean(profits)
                total_profit = sum(profits)
                self.logger.record("eval/real_profit_avg", mean_profit)
                self.logger.record("eval/real_profit_total", total_profit)

            if self.log_path is not None:
                self.evaluations_timesteps.append(self.num_timesteps)
                self.evaluations_results.append(episode_rewards)
                self.evaluations_length.append(episode_lengths)

                np.savez(
                    self.log_path,
                    timesteps=self.evaluations_timesteps,
                    results=self.evaluations_results,
                    ep_lengths=self.evaluations_length,
                )

            mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
            mean_ep_length = np.mean(episode_lengths)

            self.last_mean_reward = float(mean_reward)

            self.logger.record("eval/mean_reward", float(mean_reward))
            self.logger.record("eval/mean_ep_length", mean_ep_length)

            self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
            self.logger.dump(self.num_timesteps)

            if "real_profits" in self.eval_env.get_attr("last_info")[0]:
                profits = self.eval_env.get_attr("last_info")[0]["real_profits"]
                total_profit = sum(profits)

                if total_profit > self.best_total_profit:
                    self.best_total_profit = total_profit
                    if self.best_model_save_path is not None:
                        self.model.save(os.path.join(self.best_model_save_path, "best_model"))
                        print(f"[Eval] New best model saved with total profit: {total_profit:.2f}")

            if mean_reward > self.best_mean_reward:
                if self.best_model_save_path is not None:
                    self.model.save(os.path.join(self.best_model_save_path, "best_model"))
                self.best_mean_reward = float(mean_reward)

                if self.callback_on_new_best is not None:
                    continue_training = self.callback_on_new_best.on_step()

            if self.callback is not None:
                continue_training = continue_training and self._on_event()

        return continue_training

class ManualLSTMResetEvalCallback(BaseCallback):
    def __init__(self, eval_env, n_eval_episodes=5, eval_freq=5000, verbose=1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        now = datetime.datetime.now()
        self.current_time = now.strftime("%d_%m_%Y_%H_%M_%S")
        self.folder_path = f"models/best_model/eval_{self.current_time}"
        os.makedirs(self.folder_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq != 0:
            return True

        env = self.eval_env
        model = self.model 

        for _ in range(self.n_eval_episodes):
            obs = env.reset()
            done = False
            total_reward = 0
            lstm_state = None
            episode_start = np.ones((1,), dtype=bool)

            while not done:
                reset_lstm = env.get_attr("reset_lstm")[0]
                if reset_lstm:
                    print("🔁 LSTM reset triggered")
                    episode_start = np.ones((1,), dtype=bool)
                else:
                    episode_start = np.zeros((1,), dtype=bool)

                action, lstm_state = model.predict(
                    obs,
                    state=lstm_state,
                    episode_start=episode_start,
                    deterministic=True,
                )

                obs, reward, done, info = env.step(action)

        now = datetime.datetime.now()
        current_time = now.strftime("%d_%m_%Y_%H_%M_%S")
        self.model.save(f"{self.folder_path}/eval_{current_time}")

        return True

if __name__ == "__main__":

    now = datetime.datetime.now()
    current_time = now.strftime("%d_%m_%Y_%H_%M_%S")

    dataset =  pd.read_excel(os.getenv("TRAIN_DATASET"), nrows=10000)

    test_dataset_list = os.listdir(os.getenv("TRAIN_TEST_DATASETS"))
    n_eval_episodes = len(test_dataset_list)

    new_logger = configure(f"./logs/eval_logs/{current_time}", ["stdout", "csv", "tensorboard"])

    total_timesteps = 200000
    num_envs = 6

    def make_env(dataset, env_id):
        return lambda: st.StrategyCloserLong(dataset = dataset,  mode = "train", plot = False, still_plot = False,
                                    eval_logger = None, env_id = env_id, total_timesteps = total_timesteps)
    
    train_envs = [make_env(dataset, i) for i in range(num_envs)]
    vec_env = SubprocVecEnv(train_envs)

    def make_eval_env():
        return steval.TradingBotRLEvalCloserLong(dataset_list = test_dataset_list, mode = "train", n_eval_episodes = n_eval_episodes, eval_logger = None, render_mode = "human", 
                                           still_plot = False)
    
    eval_env = SubprocVecEnv([make_eval_env])

    policy_kwargs = dict(
        features_extractor_class=HybridFeatureExtractorLong,
        features_extractor_kwargs=dict(lstm_hidden_size=128), 
        lstm_hidden_size=128,  
        shared_lstm=False,  
        enable_critic_lstm=True,  
    ) 

    lr_scheduler = get_linear_fn(start=2.5e-4, end=5e-6, end_fraction=1)

    n_steps = 128
    batch_size = 6
    gae_lambda = 0.8
    ent_start = 0.05
    
    model = RecurrentPPO(
        policy="MultiInputLstmPolicy", 
        env=vec_env, 
        verbose=1, 
        tensorboard_log=f"./logs/tensorboard/{current_time}",
        n_steps=n_steps, 
        batch_size=batch_size, 
        gamma=0.99, 
        gae_lambda=gae_lambda, 
        learning_rate=lr_scheduler, 
        ent_coef=ent_start,
        clip_range=0.2,
        policy_kwargs=policy_kwargs, 
        vf_coef=0.5, 
        max_grad_norm=0.5,
        normalize_advantage=True
        )
    
    eval_callback = ManualLSTMResetEvalCallback(eval_env, n_eval_episodes=n_eval_episodes, eval_freq=10, verbose=1)

    save_callback = SaveBestExplainedVarianceAndProgitAvgCallback(verbose=1)
    hyperparam_callback = DynamicHyperparamCallback(total_timesteps)

    callback = CallbackList([save_callback, 
                             hyperparam_callback, 
                             eval_callback
                             ])

    model.learn(total_timesteps=total_timesteps, callback=callback)

    model.save(path=f"models/ppo_model_{current_time}.zip", exclude = ["callback"])