from trading_bot_mt5 import Trading_bot
from dotenv import load_dotenv
import time
import os
import logging
import datetime

load_dotenv()

date = datetime.datetime.now()
date = date.strftime("%Y_%m_%d_%H_%M_%S")

logging.basicConfig(filename = f"logs/{date}.log", format = "%(asctime)s %(message)s", filemode = "w")
logger = logging.getLogger()
logger.setLevel(logging.INFO)

if __name__ == '__main__':

    logger.info("Connecting to Mt5...")
    username = os.getenv("REAL_USER_NAME")
    server = os.getenv("SERVER")
    password = os.getenv("REAL_PASSWORD")
    symbol= "DE30.pro"

    trading_bot = Trading_bot(username, server, password, symbol, volume_bet = 0.01, closing_time = "18:00")
    logger.info("Mt5 connection created.")
    logger.info("Starting to trade.")

    stop = False

    while True:
        trading_bot.signals()
        stop = trading_bot.action()

        if stop == True:
            break

        trading_bot.console_print()

        time.sleep(1)
