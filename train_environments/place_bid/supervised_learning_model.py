import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers, Model, Input
from sklearn.metrics import accuracy_score, classification_report
import datetime
from datetime import time
from keras.callbacks import EarlyStopping
import scalers as scaler
import joblib
import os
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.utils import class_weight, shuffle
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
from tensorflow.keras import regularizers
import tensorflow.keras.backend as K
from sklearn.metrics import (accuracy_score, classification_report)
import seaborn as sns
from tensorflow.keras.optimizers import Adam
from dotenv import load_dotenv

load_dotenv()

class SupervisedLearning:
    def __init__(self):

        self.fitted_scaler = joblib.load(os.getenv("GENERAL_SCALER"))
        self.fitted_slope_scaler = joblib.load(os.getenv("SLOPE_SCALER"))
        self.fitted_trend_scaler = joblib.load(os.getenv("TREND_SCALER"))

        date = datetime.datetime.now()
        self.date = date.strftime("%Y_%m_%d_%H_%M_%S")

        self.features()

    def features(self):

        self.market_dynamics = ['spread_pct', 'tick_volume', 'slope_ema_10', 'close']
        self.momentum_features = ['macd_diff', 'momentum_composite_2', "MA_Crossover", "roc_30m"]
        self.market_condition = ['atr', 'bb_%b', 'volatility_rolling_std', 'bb_width']
        self.trend_features = ['adosc', 'vwap', 'obv', "long_term_slope", "long_slope_angle"]

        self.save_path = f"models/supervised_model_{self.date}"

        return [self.market_dynamics, self.momentum_features, self.market_condition]
    
    def create_synthetic_windows(self, X_lstm, X_mf, X_mc, X_tf, y, num_augments=2, noise_level=0.001):

        synthetic_X_lstm = []
        synthetic_X_mf = []
        synthetic_X_mc = []
        synthetic_X_tf = []
        synthetic_y = []

        for i in range(len(X_lstm)):
            for _ in range(num_augments):
                jitter = np.random.normal(loc=0, scale=noise_level, size=X_lstm[i].shape)
                lstm_aug = X_lstm[i] + jitter
                mf_aug = X_mf[i] + np.random.normal(0, noise_level, size=X_mf[i].shape)
                mc_aug = X_mc[i] + np.random.normal(0, noise_level, size=X_mc[i].shape)
                tf_aug = X_tf[i] + np.random.normal(0, noise_level, size=X_tf[i].shape)

                synthetic_X_lstm.append(lstm_aug)
                synthetic_X_mf.append(mf_aug)
                synthetic_X_mc.append(mc_aug)
                synthetic_X_tf.append(tf_aug)
                synthetic_y.append(y[i])

        return shuffle(
            np.array(synthetic_X_lstm),
            np.array(synthetic_X_mf),
            np.array(synthetic_X_mc),
            np.array(synthetic_X_tf),
            np.array(synthetic_y),
            random_state=42
        )
    
    def create_new_scaler(self, df):

        trend_scaler = StandardScaler()

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
        ]] = trend_scaler.fit_transform(df[[
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

        joblib.dump(trend_scaler, f"scalers/fitted_trend_scaler_{self.date}.pkl")


    def add_long_and_short_slopes_extended(self, df):

        def calc_slope(series):
            X = np.arange(len(series)).reshape(-1, 1)
            y = series.values
            return LinearRegression().fit(X, y).coef_[0]

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
            raise("Values Missing!!")

        # trend_strength_std: rolling std of long-term slope
        df["trend_strength_std"] = df["long_term_slope"].rolling(window=5, min_periods=1).std().fillna(0.0)
        df["trend_strength_std_label"] = df["long_term_slope"].rolling(window=5, min_periods=1).std().fillna(0.0)

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

    def setup_score_labeling_binary(self, df, lookahead):

        df["future_mean"] = df["close"].shift(-1).rolling(window=lookahead, min_periods=1).mean().shift(-lookahead + 1)
        df["price_diff"] = df["future_mean"] - df["close"]
        df["rolling_volatility"] = df["close"].rolling(window=lookahead).std()
        df["slope_ema_10_label"] = df["close"].ewm(span=10).mean().diff()

        long_diff_thresh = df["price_diff"][df["price_diff"] > 0].quantile(0.3)
        short_diff_thresh = df["price_diff"][df["price_diff"] < 0].quantile(0.7)

        vol_thresh = df["rolling_volatility"].quantile(0.3)
        slope_thresh = df["slope_ema_10_label"].abs().quantile(0.3)
        trend_thresh = df["trend_strength_std_label"].quantile(0.3)

        df["score"] = (
            (df["rolling_volatility"] > vol_thresh).astype(int) +
            (df["slope_ema_10_label"].abs() > slope_thresh).astype(int) +
            (df["trend_strength_std_label"] > trend_thresh).astype(int)
        )

        df["long_label"] = 0
        df["short_label"] = 0
        df.loc[(df["score"] >= 3) & (long_diff_thresh > 0), "long_label"] = 1
        df.loc[(df["score"] >= 3) & (short_diff_thresh < 0), "short_label"] = 1

        df["label"] = df["short_label"]
        df.dropna(subset=self.market_dynamics + self.momentum_features + self.market_condition + ["label"], inplace=True)

        return df
    
    def mean_binary(self, df, lookahead):
        
        df["future_mean"] = df["close"].shift(-1).rolling(window=lookahead, min_periods=1).mean().shift(-lookahead + 1)
        df["mean_diff"] = df["future_mean"] - df["close"]
        
        long_thresh = df["mean_diff"].quantile(0.6)
        short_thresh = df["mean_diff"].quantile(0.4)

        df["long_label"] = (df["mean_diff"] > long_thresh).astype(int)
        df["short_label"] = (df["mean_diff"] < short_thresh).astype(int)
        df["label"] = (df["long_label"] == 1).astype(int)

        df.dropna(subset=self.market_dynamics + self.momentum_features + self.market_condition + ["label"], inplace=True)

        return df

    def train_with_synthetic_binary(self):

        data = pd.read_excel(os.getenv("TRAIN_DATASET_SL")) 
        df = data[data["original_datetime"].dt.year.isin([2019, 2020, 2021, 2022, 2023, 2024, 2025])].copy()

        df = self.add_long_and_short_slopes_extended(df.copy())

        lookahead = 12

        df = self.setup_score_labeling_binary(df.copy(), lookahead)

        df["time_only"] = df["original_datetime"].dt.time
        df = df[(df["time_only"] >= time(5, 0)) & (df["time_only"] <= time(23, 0))].copy()

        df_scaled = scaler.transform_data(df.copy(), self.fitted_scaler)

        seq_len = 32
        X_lstm, X_momentum, X_condition, X_trend, y = [], [], [], [], []

        md_scaled = df_scaled[self.market_dynamics]
        mf_scaled = df_scaled[self.momentum_features]
        mc_scaled = df_scaled[self.market_condition]
        tf_scaled = df_scaled[self.trend_features]

        labels = df_scaled["label"].values

        for i in range(len(df_scaled) - seq_len - lookahead):
            X_lstm.append(md_scaled.iloc[i:i+seq_len].values)
            X_momentum.append(mf_scaled.iloc[i+seq_len].values)
            X_condition.append(mc_scaled.iloc[i+seq_len].values)
            X_trend.append(tf_scaled.iloc[i+seq_len].values)
            y.append(labels[i + seq_len])

        X_lstm = np.array(X_lstm)
        X_momentum = np.array(X_momentum)
        X_condition = np.array(X_condition)
        X_trend = np.array(X_trend)
        y = np.array(y)

        pos_indices = np.where(y == 1)[0]
        neg_indices = np.where(y == 0)[0]

        np.random.seed(42)
        neg_indices_sampled = np.random.choice(neg_indices, size=len(pos_indices), replace=False)

        balanced_indices = np.concatenate([pos_indices, neg_indices_sampled])
        np.random.shuffle(balanced_indices)

        X_lstm = X_lstm[balanced_indices]
        X_momentum = X_momentum[balanced_indices]
        X_condition = X_condition[balanced_indices]
        X_trend = X_trend[balanced_indices]
        y = y[balanced_indices]

        X_lstm_train, X_lstm_test, X_mf_train, X_mf_test, X_mc_train, X_mc_test, X_tf_train, X_tf_test, y_train, y_test = train_test_split(
            X_lstm, X_momentum, X_condition, X_trend, y, test_size=0.2, random_state=42
        )

        X_lstm_aug, X_mf_aug, X_mc_aug, X_tf_aug, y_aug = self.create_synthetic_windows(
            X_lstm_train, X_mf_train, X_mc_train, X_tf_train, y_train, num_augments=1, noise_level=0.001
        )

        X_lstm_train = np.concatenate([X_lstm_train, X_lstm_aug])
        X_mf_train = np.concatenate([X_mf_train, X_mf_aug])
        X_mc_train = np.concatenate([X_mc_train, X_mc_aug])
        X_tf_train = np.concatenate([X_tf_train, X_tf_aug])
        y_train = np.concatenate([y_train, y_aug])

        class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
        class_weights = dict(enumerate(class_weights))

        input_lstm = Input(shape=(seq_len, len(self.market_dynamics)))
        x_lstm = layers.LSTM(64, return_sequences=True)(input_lstm)
        x_lstm = layers.LSTM(32)(x_lstm)

        input_mf = Input(shape=(len(self.momentum_features),))
        x_mf = layers.Dense(32, activation='relu')(input_mf)
        x_mf = layers.Dense(16, activation='relu')(x_mf)

        input_mc = Input(shape=(len(self.market_condition),))
        x_mc = layers.Dense(64, activation='relu')(input_mc)
        x_mc = layers.Dense(32, activation='relu')(x_mc)

        input_tf = Input(shape=(len(self.trend_features),))
        x_tf = layers.Dense(64, activation='relu')(input_tf)
        x_tf = layers.Dense(32, activation='relu')(x_tf)

        merged = layers.concatenate([x_lstm, x_mf, x_mc, x_tf])
        x = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(merged)
        x = layers.Dropout(0.3)(x)
        output = layers.Dense(1, activation='sigmoid')(x)

        model = Model(inputs=[input_lstm, input_mf, input_mc, input_tf], outputs=output)

        model.compile(optimizer='adam', loss="binary_crossentropy", metrics=['accuracy'])

        earlystop = EarlyStopping(monitor='val_loss', min_delta=1e-4, patience=3, verbose=1, mode='min', restore_best_weights=True)

        history = model.fit([X_lstm_train, X_mf_train, X_mc_train, X_tf_train], y_train,
                            validation_split=0.2, epochs=200, batch_size=64,
                            callbacks=[earlystop], class_weight=class_weights)

        y_probs = model.predict([X_lstm_test, X_mf_test, X_mc_test, X_tf_test])
        plt.hist(y_probs, bins=100)
        plt.title("Histogram of Prediction Probabilities")
        plt.xlabel("Predicted Probability")
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.show()

        y_pred = (y_probs > 0.5).astype(int)
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print(classification_report(y_test, y_pred))

        model.save(self.save_path + "_binary.keras")

    def setup_score_labeling_two_classes(self, df, lookahead):

        df["future_mean"] = df["close"].shift(-1).rolling(window=lookahead, min_periods=1).mean().shift(-lookahead + 1)
        df["price_diff"] = df["future_mean"] - df["close"]
        df["rolling_volatility"] = df["close"].rolling(window=lookahead).std()
        df["slope_ema_10_label"] = df["close"].ewm(span=10).mean().diff()

        # Directional thresholds
        long_diff_thresh = df["price_diff"][df["price_diff"] > 0].quantile(0.3)
        short_diff_thresh = df["price_diff"][df["price_diff"] < 0].quantile(0.7)

        vol_thresh = df["rolling_volatility"].quantile(0.3)
        slope_thresh = df["slope_ema_10_label"].abs().quantile(0.3)
        trend_thresh = df["trend_strength_std_label"].quantile(0.3)

        df["score"] = (
            (df["rolling_volatility"] > vol_thresh).astype(int) +
            (df["slope_ema_10_label"].abs() > slope_thresh).astype(int) +
            (df["trend_strength_std_label"] > trend_thresh).astype(int)
        )

        df["label"] = np.nan
        df.loc[(df["score"] >= 2) & (df["price_diff"] > long_diff_thresh), "label"] = 1  # Long
        df.loc[(df["score"] >= 2) & (df["price_diff"] < short_diff_thresh), "label"] = 0  # Short

        df.dropna(subset=self.market_dynamics + self.momentum_features + self.market_condition + ["label"], inplace=True)
        df["label"] = df["label"].astype(int)
        return df

    def train_with_synthetic_two_classes(self):
        
        data = pd.read_excel(os.getenv("TRAIN_DATASET_SL")) 
        df = data[data["original_datetime"].dt.year.isin([2019, 2020, 2021, 2022, 2023, 2024, 2025])].copy()
        df = self.add_long_and_short_slopes_extended(df.copy())

        lookahead = 12
        df = self.setup_score_labeling_two_classes(df.copy(), lookahead)
        df["time_only"] = df["original_datetime"].dt.time
        df = df[(df["time_only"] >= time(5, 0)) & (df["time_only"] <= time(23, 0))].copy()

        df_scaled = scaler.transform_data(df.copy(), self.fitted_scaler)

        seq_len = 32
        X_lstm, X_momentum, X_condition, X_trend, y = [], [], [], [], []

        md_scaled = df_scaled[self.market_dynamics]
        mf_scaled = df_scaled[self.momentum_features]
        mc_scaled = df_scaled[self.market_condition]
        tf_scaled = df_scaled[self.trend_features]
        labels = df_scaled["label"].values.astype(int)

        for i in range(len(df_scaled) - seq_len - lookahead):
            X_lstm.append(md_scaled.iloc[i:i+seq_len].values)
            X_momentum.append(mf_scaled.iloc[i+seq_len].values)
            X_condition.append(mc_scaled.iloc[i+seq_len].values)
            X_trend.append(tf_scaled.iloc[i+seq_len].values)
            y.append(labels[i + seq_len])

        X_lstm = np.array(X_lstm)
        X_momentum = np.array(X_momentum)
        X_condition = np.array(X_condition)
        X_trend = np.array(X_trend)
        y = np.array(y).astype(int)

        class_0 = np.where(y == 0)[0]
        class_1 = np.where(y == 1)[0]
        min_len = min(len(class_0), len(class_1))

        np.random.seed(42)
        balanced_indices = np.concatenate([
            np.random.choice(class_0, min_len, replace=False),
            np.random.choice(class_1, min_len, replace=False)
        ])
        np.random.shuffle(balanced_indices)

        X_lstm = X_lstm[balanced_indices]
        X_momentum = X_momentum[balanced_indices]
        X_condition = X_condition[balanced_indices]
        X_trend = X_trend[balanced_indices]
        y = y[balanced_indices]

        X_lstm_train, X_lstm_test, X_mf_train, X_mf_test, X_mc_train, X_mc_test, X_tf_train, X_tf_test, y_train, y_test = train_test_split(
            X_lstm, X_momentum, X_condition, X_trend, y, test_size=0.2, random_state=42
        )

        X_lstm_aug, X_mf_aug, X_mc_aug, X_tf_aug, y_aug = self.create_synthetic_windows(
            X_lstm_train, X_mf_train, X_mc_train, X_tf_train, y_train, num_augments=1, noise_level=0.001
        )

        X_lstm_train = np.concatenate([X_lstm_train, X_lstm_aug])
        X_mf_train = np.concatenate([X_mf_train, X_mf_aug])
        X_mc_train = np.concatenate([X_mc_train, X_mc_aug])
        X_tf_train = np.concatenate([X_tf_train, X_tf_aug])
        y_train = np.concatenate([y_train, y_aug])

        class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
        class_weights = dict(enumerate(class_weights))

        input_lstm = Input(shape=(seq_len, len(self.market_dynamics)))
        x_lstm = layers.LSTM(64, return_sequences=True)(input_lstm)
        x_lstm = layers.LSTM(32)(x_lstm)

        input_mf = Input(shape=(len(self.momentum_features),))
        x_mf = layers.Dense(32, activation='relu')(input_mf)
        x_mf = layers.Dense(16, activation='relu')(x_mf)

        input_mc = Input(shape=(len(self.market_condition),))
        x_mc = layers.Dense(64, activation='relu')(input_mc)
        x_mc = layers.Dense(32, activation='relu')(x_mc)

        input_tf = Input(shape=(len(self.trend_features),))
        x_tf = layers.Dense(64, activation='relu')(input_tf)
        x_tf = layers.Dense(32, activation='relu')(x_tf)

        merged = layers.concatenate([x_lstm, x_mf, x_mc, x_tf])
        x = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(merged)
        x = layers.Dropout(0.3)(x)
        output = layers.Dense(2, activation='softmax')(x)

        model = Model(inputs=[input_lstm, input_mf, input_mc, input_tf], outputs=output)
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        earlystop = EarlyStopping(monitor='val_loss', min_delta=1e-4, patience=3, verbose=1, mode='min', restore_best_weights=True)

        history = model.fit([X_lstm_train, X_mf_train, X_mc_train, X_tf_train], y_train,
                            validation_split=0.2, epochs=200, batch_size=64,
                            callbacks=[earlystop], class_weight=class_weights)

        y_probs = model.predict([X_lstm_test, X_mf_test, X_mc_test, X_tf_test])
        y_pred = np.argmax(y_probs, axis=1)

        plt.hist(y_probs[:, 1], bins=100)
        plt.title("Histogram of Long Class Probability")
        plt.xlabel("Predicted Probability (Long)")
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.show()

        print("Accuracy:", accuracy_score(y_test, y_pred))
        print(classification_report(y_test, y_pred))

        model.save(self.save_path + "_two_class.keras")

    def test_model_two_class(self):

        dataset_list = os.listdir(os.getenv("TRAIN_TEST_DATASETS_SL"))

        start_file = ""
        start_index = dataset_list.index(start_file)
        test_dataset_list = dataset_list[start_index:]

        model_name = ""
        model = keras.models.load_model(f"models/{model_name}")

        for trading_day in test_dataset_list:
            day_data = pd.read_excel(f"envs/test_data/{trading_day}", sheet_name = "Sheet1", header = 0)
            day_data = self.add_long_and_short_slopes_extended(day_data.copy())
            scaled_day_data = scaler.transform_data(day_data.copy(), self.fitted_scaler)
            self.bids = []
            self.predictions = []
            for index, row in scaled_day_data.iterrows():
                index += 32
                if index < len(scaled_day_data):
                    market_dynamics = np.expand_dims(scaled_day_data[self.market_dynamics].iloc[:index], axis=0)
                    momentum_features = np.expand_dims(scaled_day_data[self.momentum_features].iloc[index], axis=0)
                    market_condition = np.expand_dims(scaled_day_data[self.market_condition].iloc[index], axis=0)
                    trend_features = np.expand_dims(scaled_day_data[self.trend_features].iloc[index], axis=0)

                    self.prediction = model.predict([market_dynamics, momentum_features, market_condition, trend_features,])

                    self.two_class_prediction()

            trimmed_df = scaled_day_data.iloc[32:].copy()
            trimmed_df["bids"] = self.bids
            trimmed_df["predictions"] = self.predictions
            trimmed_df.reset_index(inplace = True)

            long = trimmed_df.index[trimmed_df['bids']=="long"].tolist() 
            short = trimmed_df.index[trimmed_df['bids']=="short"].tolist() 
            no_bet = trimmed_df.index[trimmed_df['bids']=="no bet"].tolist() 

            no_scaleling = day_data.iloc[32:].copy()
            no_scaleling.reset_index(inplace = True)

            plt.figure(figsize=(14, 6))
            plt.plot(trimmed_df["Time"], no_scaleling['close'], color = 'red')
            plt.plot(trimmed_df["Time"], no_scaleling['close'], '^', markevery=long, label='Long', color="Green")
            plt.plot(trimmed_df["Time"], no_scaleling['close'], 'v', markevery=short, label='Short', color = 'blue')
            plt.plot(trimmed_df["Time"], no_scaleling['close'], 'x', markevery=no_bet, label='No Bet', color = 'red')

            for idx in long:
                pred = trimmed_df['predictions'].iloc[idx]
                price = no_scaleling['close'].iloc[idx]
                plt.text(trimmed_df["Time"].iloc[idx], price + 10, f"{pred:.2f}", color='green', fontsize=8, ha='center')

            for idx in short:
                pred = trimmed_df['predictions'].iloc[idx]
                price = no_scaleling['close'].iloc[idx]
                plt.text(trimmed_df["Time"].iloc[idx], price + 10, f"{pred:.2f}", color='brown', fontsize=8, ha='center')

            plt.legend()
            plt.tight_layout()
            plt.show()

    def test_model_binary(self):

        dataset_list = os.listdir(os.getenv("TRAIN_TEST_DATASETS_SL"))

        start_file = ""
        start_index = dataset_list.index(start_file)
        test_dataset_list = dataset_list[start_index:]

        model_name_long = ""
        model_name_short = ""
        model_long = keras.models.load_model(f"models/{model_name_long}")
        model_short = keras.models.load_model(f"models/{model_name_short}")  

        for trading_day in test_dataset_list:
            day_data = pd.read_excel(f"envs/test_data/{trading_day}", sheet_name = "Sheet1", header = 0)
            day_data = self.add_long_and_short_slopes_extended(day_data.copy())
            scaled_day_data = scaler.transform_data(day_data.copy(), self.fitted_scaler)
            self.bids = []
            self.predictions = []
            for index, row in scaled_day_data.iterrows():
                index += 32
                if index < len(scaled_day_data):
                    market_dynamics = np.expand_dims(scaled_day_data[self.market_dynamics].iloc[:index], axis=0)
                    momentum_features = np.expand_dims(scaled_day_data[self.momentum_features].iloc[index], axis=0)
                    market_condition = np.expand_dims(scaled_day_data[self.market_condition].iloc[index], axis=0)
                    trend_features = np.expand_dims(scaled_day_data[self.trend_features].iloc[index], axis=0)

                    self.prediction_long = model_long.predict([market_dynamics,  
                                                               momentum_features, market_condition, 
                                                               trend_features])
                    self.prediction_short = model_short.predict([market_dynamics, 
                                                                 momentum_features, market_condition, 
                                                                 trend_features])

                    self.binary_prediction()

            trimmed_df = scaled_day_data.iloc[32:].copy()
            trimmed_df["bids"] = self.bids
            trimmed_df["predictions"] = self.predictions
            trimmed_df.reset_index(inplace = True)

            long = trimmed_df.index[trimmed_df['bids']=="long"].tolist() 
            short = trimmed_df.index[trimmed_df['bids']=="short"].tolist() 

            no_scaleling = day_data.iloc[32:].copy()
            no_scaleling.reset_index(inplace = True)

            plt.figure(figsize=(14, 6))
            plt.plot(trimmed_df["Time"], no_scaleling['close'], color = 'red')
            plt.plot(trimmed_df["Time"], no_scaleling['close'], '^', markevery=long, label='Long', color="Green")
            plt.plot(trimmed_df["Time"], no_scaleling['close'], 'v', markevery=short, label='Short', color = 'blue')

            for idx in long:
                pred = trimmed_df['predictions'].iloc[idx]
                price = no_scaleling['close'].iloc[idx]
                plt.text(trimmed_df["Time"].iloc[idx], price + 10, f"{pred:.2f}", color='green', fontsize=8, ha='center')

            for idx in short:
                pred = trimmed_df['predictions'].iloc[idx]
                price = no_scaleling['close'].iloc[idx]
                plt.text(trimmed_df["Time"].iloc[idx], price + 10, f"{pred:.2f}", color='brown', fontsize=8, ha='center')

            plt.legend()
            plt.tight_layout()
            plt.show()

    def binary_prediction(self):
        prediction_long = self.prediction_long.item()
        prediction_short = self.prediction_short.item()

        prediction_long = 0

        if prediction_long >= 0.7 and prediction_short < 0.5:
            bid = "long"
            prediction = prediction_long
        elif prediction_short >= 0.7 and prediction_long < 0.5:
            bid = "short"
            prediction = prediction_short
        else:
            bid = None
            prediction = None
        self.bids.append(bid)
        self.predictions.append(prediction)

    def two_class_prediction(self):

        prediction_long = self.prediction[0][1]
        prediction_short = self.prediction[0][0]

        if prediction_long >= 0.7 and prediction_short < 0.7:
            bid = "long"
            prediction = prediction_long
        elif prediction_short >= 0.7 and prediction_long < 0.7:
            bid = "short"
            prediction = prediction_short
        else:
            bid = None
            prediction = None
        self.bids.append(bid)
        self.predictions.append(prediction)

    def create_new_trading_data(self):

        data = pd.read_excel(os.getenv("TRAIN_DATASET_SL"))
        if data.isnull().values.any():
            print("Dataframe has NaN values")
        else:
            print("Dataframe does not have NaN values")

        df_with_preds = data.copy()

        df = self.add_long_and_short_slopes_extended(data)
        df = scaler.transform_data(df.copy(), self.fitted_scaler)

        print("Data loaded and scaled.")

        model_name_long = ""
        model_name_short = ""

        model_paths = [
            f"models/{model_name_long}",
            f"models/{model_name_short}",
        ]

        models = [keras.models.load_model(path) for path in model_paths]

        attributes = {
            "long_prediction_probs": self.features(),
            "short_prediction_probs": self.features()
        }

        seq_len = 32

        for model_idx, (model, attr) in enumerate(zip(models, attributes), 1):
            X_lstm, X_mf, X_mc, X_tf = [], [], [], []
            
            model_idx = attr

            for i in tqdm(range(len(df) - seq_len), desc=f"Building input for model: {model_idx}"):
                if self.market_dynamics:
                    md_seq = df[self.market_dynamics].iloc[i:i + seq_len].values.astype(np.float32)
                    X_lstm.append(md_seq)
                if self.momentum_features:
                    mf_row = df[self.momentum_features].iloc[i + seq_len].values.astype(np.float32)
                    X_mf.append(mf_row)
                if self.market_condition:
                    mc_row = df[self.market_condition].iloc[i + seq_len].values.astype(np.float32)
                    X_mc.append(mc_row)
                if self.trend_features:
                    tf_seq = df[self.trend_features].iloc[i + seq_len].values.astype(np.float32)
                    X_tf.append(tf_seq)

            X_lstm = np.array(X_lstm)
            X_mf = np.array(X_mf)
            X_mc = np.array(X_mc)
            X_tf = np.array(X_tf)

            print(f"Model {model_idx}: Inputs prepared.")

            model_inputs = []
            if X_lstm.size > 0:
                model_inputs.append(X_lstm)
            if X_mf.size > 0:
                model_inputs.append(X_mf)
            if X_mc.size > 0:
                model_inputs.append(X_mc)
            if X_tf.size > 0:
                model_inputs.append(X_tf)

            preds = model.predict(model_inputs, verbose=0)

            df_with_preds.loc[seq_len:seq_len + len(preds) - 1, model_idx] = preds[:, 0]

            print(f"Model {model_idx}: Predictions added.")

        df_with_preds["long_prediction"] = np.where(
        (df_with_preds["long_prediction_probs"] > 0.7) & (df_with_preds["short_prediction_probs"] < 0.5), 1, 0
        )

        df_with_preds["short_prediction"] = np.where(
            (df_with_preds["short_prediction_probs"] > 0.7) & (df_with_preds["long_prediction_probs"] < 0.5), 1, 0
        )

        df_with_preds.dropna(inplace=True)

        output_path = f"train_data/combined_dataset_full_with_binary_predictions_{self.date}.xlsx"
        df_with_preds.to_excel(output_path, index=True)
        print(f"Dataset with predictions saved to {output_path}")
    
if __name__ == "__main__":

    model = SupervisedLearning()
    model.train_with_synthetic_binary()
    #model.train_with_synthetic_two_classes()
    #model.create_new_trading_data()