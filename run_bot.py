import os
import subprocess
import sys
import logging
import argparse

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_bot(train_only=False, skip_train=False):
    # Проверяем, существует ли модель (для любой стратегии)
    model_exists = any(
        os.path.exists(f"models/ppo_{strat}.zip")
        for strat in ["scalping", "pipsing", "intraday"]
    )

    if not model_exists and not skip_train:
        logging.info("First run: Training model...")
        try:
            # Обучаем модель (train_model.py сам загрузит данные и обучит гибридно)
            subprocess.run([sys.executable, "scripts/train_model.py", "--auto"], check=True)
            logging.info("Training completed successfully.")
        except subprocess.CalledProcessError as e:
            logging.error(f"Training failed: {e}")
            sys.exit(1)

    if not train_only:
        logging.info("Starting bot...")
        try:
            # Запускаем main.py (он сам выберет стратегию)
            subprocess.run([sys.executable, "scripts/main.py"], check=True)
            logging.info("Bot started successfully.")
        except subprocess.CalledProcessError as e:
            logging.error(f"Bot start failed: {e}")
            sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the trading bot.")
    parser.add_argument('--train-only', action='store_true', help="Only train the model, do not start the bot.")
    parser.add_argument('--skip-train', action='store_true', help="Skip training if model exists.")
    args = parser.parse_args()
    
    run_bot(train_only=args.train_only, skip_train=args.skip_train)
