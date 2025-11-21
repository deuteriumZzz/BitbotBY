import os
import asyncio
import argparse
from scripts.train_model import main as train_main
from scripts.main import main as run_main
import logging
from dotenv import load_dotenv

load_dotenv()  # Загружаем .env
logging.basicConfig(level=logging.INFO)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--strategy', type=str, default=os.getenv('STRATEGY', 'ppo'), help='Trading strategy')
    parser.add_argument('--skip_train', action='store_true', help='Skip training if model exists')
    args = parser.parse_args()

    model_path = f"models/ppo_{args.strategy}.zip"
    if not args.skip_train or not os.path.exists(model_path):
        logging.info(f"Training model for strategy: {args.strategy}")
        train_main(strategy=args.strategy)  # Передаём стратегию
    else:
        logging.info(f"Skipping training, using existing model: {model_path}")

    logging.info("Launching live bot")
    asyncio.run(run_main(strategy=args.strategy))

if __name__ == "__main__":
    main()
