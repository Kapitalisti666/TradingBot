import datetime
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from supervised_learning import SupervisedLearning
from rl_learning import RLlearningCloserLong, RLlearningCloserShort
from sklearn.linear_model import LinearRegression
import sys
import os
import logging
from collections import deque

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

date = datetime.datetime.now()
date = date.strftime("%Y_%m_%d_%H_%M_%S")

logging.basicConfig(filename = f"logs/{date}.log", format = "%(asctime)s %(message)s", filemode = "w")
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def stop_loss(day_data, bet_type, entry_price, trend, slope):

    current_price = day_data["close"].iloc[index]

    close = False

    price_diff_long = (current_price - entry_price) * 0.07 * 25
    price_diff_short = (entry_price - current_price) * 0.07 * 25

    if bet_type == 1 and price_diff_long < -80:
        close = True
    elif bet_type == -1 and price_diff_short < -80:
        close = True

    if close:
        print("Stop loss activated!!!")

    return close

def take_profit(day_data, bet_type, entry_price, trend, slope):

    current_price = day_data["close"].iloc[index]
    close = False

    price_diff_long = (current_price - entry_price) * 0.07 * 25
    price_diff_short = (entry_price - current_price) * 0.07 * 25

    if bet_type == 1 and price_diff_long > 100:
        close = True
    elif bet_type == -1 and price_diff_short > 100:
        close = True

    if close:
        print("Take profit activated!!!")

    return close

def min_max_scale(value, min_val, max_val):
    return np.clip((value - min_val) / (max_val - min_val), 0, 1)

def predict_trend(data, window):

    close_prices = data["close"].iloc[-window:].values

    X = np.arange(len(close_prices)).reshape(-1, 1)
    y = np.array(close_prices)
    model = LinearRegression().fit(X, y)

    return model.coef_[0]

def is_high_volatility(df: pd.DataFrame, window: int = 10) -> bool:

    if len(df) < window * 2:
        return False  # Not enough data

    df = df.copy()

    # Get last valid volatility window
    recent_vol = df["volatility"].iloc[-window:]
    current_vol = df["volatility"].iloc[-1]

    if np.isnan(current_vol):
        return False

    # Statistical threshold
    mean_vol = recent_vol.mean()
    std_vol = recent_vol.std()
    stat_thresh = mean_vol + 1.5 * std_vol

    if current_vol > stat_thresh:
        return True

    if current_vol > recent_vol.quantile(0.85):  # top 15%
        return True

    # Z-score threshold
    z_score = (current_vol - mean_vol) / std_vol
    if z_score > 1.5:
        return True

    return False

def compute_trend_score_and_label_for_rows(df, window=20, shift=12):
    trend_scores = []
    labels = []

    for i in range(len(df)):
        if i < max(window, 6):
            continue

        sub_df = df.iloc[i - window + 1:i + 1].copy()
        current_close = sub_df["close"].iloc[-1]

        # EMA diff score
        ema_5 = sub_df["close"].ewm(span=5).mean().iloc[-1]
        ema_20 = sub_df["close"].ewm(span=20).mean().iloc[-1]
        ema_diff = ema_5 - ema_20
        ema_score = min_max_scale(ema_diff, -0.015 * current_close, 0.025 * current_close)

        # VWAP score
        vwap = (sub_df["close"] * sub_df["tick_volume"]).cumsum() / sub_df["tick_volume"].cumsum()
        price_above_vwap_ratio = (sub_df["close"] > vwap).mean()
        vwap_score = min_max_scale(price_above_vwap_ratio, 0.3, 0.75)

        # Slope score
        x = np.arange(window).reshape(-1, 1)
        y = sub_df["close"].values
        slope = LinearRegression().fit(x, y).coef_[0]
        slope_score = min_max_scale(slope, -0.025, 0.06)

        # Gain score
        recent_gain = sub_df["close"].iloc[-1] - sub_df["close"].iloc[-6]
        gain_score = min_max_scale(recent_gain, -0.015 * current_close, 0.025 * current_close)

        # Combined score
        score = ema_score + vwap_score + slope_score + gain_score
        trend_scores.append(score)

        # Labeling
        if score >= 2.5:
            labels.append("Bullish")  # Bullish
        elif score <= 1.4:
            labels.append("Bearish")  # Bearish
        else:
            labels.append("Sideways")  # Sideways

    return trend_scores

def compute_trend_score_and_label_for_real_time(df_live, window=20):
    if len(df_live) < max(window, 6):
        return None, None  # Not enough data

    sub_df = df_live.iloc[-window:].copy()
    current_close = sub_df["close"].iloc[-1]

    # EMA diff score
    ema_5 = sub_df["close"].ewm(span=5).mean().iloc[-1]
    ema_20 = sub_df["close"].ewm(span=20).mean().iloc[-1]
    ema_diff = ema_5 - ema_20
    ema_score = min_max_scale(ema_diff, -0.015 * current_close, 0.025 * current_close)

    # VWAP score
    vwap = (sub_df["close"] * sub_df["tick_volume"]).cumsum() / sub_df["tick_volume"].cumsum()
    price_above_vwap_ratio = (sub_df["close"] > vwap).mean()
    vwap_score = min_max_scale(price_above_vwap_ratio, 0.3, 0.75)

    # Slope score
    y = sub_df["close"].values
    x = np.arange(len(y)).reshape(-1, 1)
    slope = LinearRegression().fit(x, y).coef_[0]
    slope_score = min_max_scale(slope, -0.025, 0.06)

    # Gain score
    recent_gain = sub_df["close"].iloc[-1] - sub_df["close"].iloc[-6]
    gain_score = min_max_scale(recent_gain, -0.015 * current_close, 0.025 * current_close)

    # Combined score
    score = ema_score + vwap_score + slope_score + gain_score

    # Labeling
    if score >= 2.5:
        label = "Bullish"
    elif score <= 1.4:
        label = "Bearish"
    else:
        label = "Sideways"

    return score, label

if __name__ == "__main__":

    still_plot = False

    dataset_list = os.listdir("test_data/")

    start_file = "combined_2025_04_28.xlsx"
    start_index = dataset_list.index(start_file)
    test_dataset_list = dataset_list[start_index:]
    auxiliary_list = dataset_list[start_index-1:-1]

    supervised_model = SupervisedLearning()
    rl_model_long = RLlearningCloserLong()
    rl_model_short = RLlearningCloserShort()

    final_results = []

    image_files = f"logs/{date}_images"

    os.makedirs(image_files, exist_ok=True)

    env_number = 0

    for trading_day, previous_day in zip(test_dataset_list, auxiliary_list):
        day_data = pd.read_excel(f"test_data/{trading_day}", sheet_name = "Sheet1", header = 0)
        previous_day = pd.read_excel(f"test_data/{previous_day}", sheet_name = "Sheet1", header = 0)
        bids = []
        predictions = []
        closing = []
        bet_placed = False
        first_round = True
        entry_price = None
        results = []
        long_probs = []
        short_probs = []
        trends = []
        ready_to_bet = False
        env_number += 1

        supervised_model.previous_meta_bid = 0

        long_bids = deque(maxlen=10)
        short_bids = deque(maxlen=10)
        bidding_zone_long = False
        bidding_zone_short = False
        average_bid_long = 0
        average_bid_short = 0
        trend_heat = 0

        bullish_count = 0
        bearish_count = 0

        bullish_zone = False
        bearish_zone = False

        bid_start_slope = 0

        number_of_bullish = [1]
        number_of_bearish = [1]

        allowed_to_bid = False
        allowed_to_long = False
        allowed_to_short = False
        
        for index, row in day_data.iterrows():
        
            index += 32

            if index < len(day_data):

                current_time = day_data["original_datetime"].iloc[index].time()
                start_time = datetime.time(9, 0) 
                end_time = datetime.time(18, 0)

                if not (start_time <= current_time <= end_time) and not bet_placed:
                    closing.append(None)
                    results.append(0)
                    trends.append(None)
                    bids.append(0)
                    continue

                if first_round:
                    trend_scores = compute_trend_score_and_label_for_rows(day_data.iloc[:index + 1])
                    short_trend_scores = compute_trend_score_and_label_for_rows(df = day_data.iloc[:index + 1], window=20)
                    long_trend_scores = compute_trend_score_and_label_for_rows(df = day_data.iloc[:index + 1], window=20)
                    first_round = False

                print(day_data.iloc[:index + 1]["original_datetime"].iloc[-1])
                score, trend = compute_trend_score_and_label_for_real_time(day_data.iloc[:index + 1])
                trends.append(trend)

                if trend == "Bullish":
                    trend_heat += 1
                    bullish_count += 1
                    bearish_count = 0
                    number_of_bullish.append(1)
                elif trend == "Bearish":
                    trend_heat -= 1
                    bearish_count += 1
                    bullish_count = 0
                    number_of_bearish.append(1)

                bull_bear_relation = len(number_of_bullish) / len(number_of_bearish)

                print(trend)
                print(f"Bull bear relation: {bull_bear_relation}")
                trend_scores.append(score)
                short_trend_scores.append(score)
                long_trend_scores.append(score)
                average_trend_score = sum(trend_scores) / len(trend_scores)
                average_trend_score_short = sum(short_trend_scores) / len(short_trend_scores)
                average_trend_score_long = sum(long_trend_scores) / len(long_trend_scores)
                slope = predict_trend(data=day_data.iloc[:index + 1], window = 5)
                print(f"Trend score: {score}")
                print(f"Trend score median: {average_trend_score}")
                print(f"Trend score short: {average_trend_score_short}")
                print(f"Trend score long: {average_trend_score_long}")
                print(f"Slope: {slope}")

                high_volatility = is_high_volatility(day_data.iloc[:index + 1])

                print(f"High volatility zone: {high_volatility}")

                if not bet_placed:

                    bid_prediction, probs = supervised_model.make_prediction(day_data.iloc[:index + 1])

                    if not allowed_to_bid:
                        if probs["long"] < 0.5:
                            allowed_to_long = True
                        if probs["short"] < 0.5:
                            allowed_to_short = True
                        if allowed_to_long and allowed_to_short:
                            allowed_to_bid = True
                        else:
                            bids.append(None)
                            closing.append(None)
                            results.append(0)
                            continue    

                    bids.append(bid_prediction)
                    entry_price = day_data["close"].iloc[index]
                    bet_type = bid_prediction

                    if bid_prediction == 1 or bid_prediction == -1:
                    
                        rsi = day_data["rsi"].iloc[index]
                        direction = "long" if bid_prediction == 1 else "short"

                        bid_start_slope = slope
                        long_bids.clear()
                        short_bids.clear()
                        ready_to_bet = False
                        bidding_zone_long = False
                        bidding_zone_short = False
                        bullish_count = 0
                        bearish_count = 0
                        bet_placed = True
                        rl_model_long.initialize_observation_space(day_data.iloc[:index + 1])
                        rl_model_short.initialize_observation_space(day_data.iloc[:index + 1])
                
                    closing.append(None)
                    results.append(0)
                    bid_time = day_data["original_datetime"].iloc[index]

                else:
                    if bet_type == 1:
                        close_prediction = rl_model_long.make_rl_prediction(day_data.iloc[:index + 1], bet_type = bet_type, entry_price = entry_price)
                    elif bet_type == -1:
                        close_prediction = rl_model_short.make_rl_prediction(day_data.iloc[:index + 1], bet_type = bet_type, entry_price = entry_price)
                
                    rl_model_long.holding_bars += 1
                    rl_model_short.holding_bars += 1
                    supervised_model.previous_prediction_long = 0
                    supervised_model.previous_prediction_short = 0

                    check_stop_loss = False
                    check_take_profit = False

                    check_stop_loss = stop_loss(day_data, bet_type, entry_price, trend, slope)
                    check_take_profit = take_profit(day_data, bet_type, entry_price, trend, slope)

                    if close_prediction == 1 or check_stop_loss:
                        bet_placed = False
                        closing.append(1)
                        if bet_type == 1:
                            result = (day_data["close"].iloc[index] - entry_price) * 0.07 * 25
                        elif bet_type == -1:
                            result = (entry_price - day_data["close"].iloc[index]) * 0.07 * 25
                            
                        results.append(result)
                        print(f"Result: {result:.2f}")
                    else:
                        closing.append(0)
                        results.append(0)

                    bids.append(None)
                    bid_time = day_data["original_datetime"].iloc[index]

        trimmed_df = day_data.iloc[32:].copy()
        trimmed_df["bids"] = bids
        trimmed_df["closing"] = closing
        trimmed_df["trends"] = trends
        trimmed_df.reset_index(inplace = True)

        trimmed_df["results"] = results

        long = trimmed_df.index[trimmed_df['bids'] == 1].tolist() 
        short = trimmed_df.index[trimmed_df['bids'] == -1].tolist() 
        no_bet = trimmed_df.index[trimmed_df['bids'] == 0].tolist() 
        hold = trimmed_df.index[trimmed_df['closing'] == 0].tolist() 
        close = trimmed_df.index[trimmed_df['closing'] == 1].tolist() 
        result_to_plot = trimmed_df['results'][trimmed_df['closing'] == 1].tolist() 
        bullish = trimmed_df.index[trimmed_df['trends'] == "Bullish"].tolist() 
        bearish = trimmed_df.index[trimmed_df['trends'] == "Bearish"].tolist() 
        sideways = trimmed_df.index[trimmed_df['trends'] == "Sideways"].tolist() 

        no_scaleling = day_data.iloc[32:].copy()
        no_scaleling.reset_index(inplace = True)

        plt.figure(figsize=(14, 6))
        plt.plot(trimmed_df["original_datetime"], no_scaleling['close'], color = 'red')
        plt.plot(trimmed_df["original_datetime"], no_scaleling['close'], '^', markevery=long, label='Long', color="Green")
        plt.plot(trimmed_df["original_datetime"], no_scaleling['close'], 'v', markevery=short, label='Short', color = 'blue')
        plt.plot(trimmed_df["original_datetime"], no_scaleling['close'], 'x', markevery=close, label='Close', color = 'black')
        plt.plot(trimmed_df["original_datetime"], no_scaleling['close'], 'x', markevery=hold, label='Hold', color = 'Yellow')

        for i, value in zip(close, result_to_plot):
            plt.text(trimmed_df["original_datetime"][i], no_scaleling["close"][i] + 2, round(value, 2), fontsize=8, color='green', ha='right')

        for i in range(len(trimmed_df) - 1):
            time_start = trimmed_df["original_datetime"].iloc[i]
            time_end = trimmed_df["original_datetime"].iloc[i + 1]
            trend = trimmed_df["trends"].iloc[i]

            if trend == "Strong Bullish":
                color = "green"
                alpha = 0.3
            elif trend == "Bullish":
                color = "green"
                alpha = 0.2
            elif trend == "Slight Bullish":
                color = "green"
                alpha = 0.1
            elif trend == "Sideways":
                color = "gray"
                alpha = 0.1
            elif trend == "Slight Bearish":
                color = "red"
                alpha = 0.1
            elif trend == "Bearish":
                color = "red"
                alpha = 0.2
            elif trend == "Strong Bearish":
                color = "red"
                alpha = 0.3
            else:
                color = "gray"
                alpha = 0.05

            plt.axvspan(time_start, time_end, facecolor=color, alpha=alpha)

        result = sum(results)

        final_results.append(result)

        cumulative_result = sum(final_results)

        file_name = trading_day.replace(".xlsx", "")

        file_path = f"{image_files}/{file_name}.png"

        day_average = cumulative_result / env_number

        logger.info(f"Env number: {env_number}")
        logger.info(f"Environment: {trading_day}")
        logger.info(f"Result: {result:.2f}")
        logger.info(f"Cumulative result: {cumulative_result:.2f}")
        logger.info(f"Day average: {day_average:.2f}")
        logger.info(f"Image saved: file://{file_path}")
        logger.info("-----")


        print(results)
        print(f"Episode Profit: {result}")
        print(f"Total Profit: {cumulative_result}")

        plt.title(f"Episode Profit: {result:.2f}")
        plt.legend()
        plt.tight_layout()

        plt.savefig(file_path)

        if still_plot:
            plt.show()
        else:
            plt.show(block = False)
            plt.pause(3)
            plt.close()

    final_result = sum(final_results)
    
    print(final_results)
    print(f"Total profit: {final_result:.2f}")


  