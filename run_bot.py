import os
import subprocess
import sys


def run_bot():
    # Проверяем, существует ли модель (для любой стратегии)
    model_exists = any(
        os.path.exists(f"models/ppo_{strat}.zip")
        for strat in ["scalping", "pipsing", "intraday"]
    )

    if not model_exists:
        print("First run: Training model...")
        # Обучаем модель (train_model.py сам загрузит данные и обучит гибридно)
        subprocess.run([sys.executable, "scripts/train_model.py", "--auto"], check=True)

    print("Starting bot...")
    # Запускаем main.py (он сам выберет стратегию)
    subprocess.run([sys.executable, "main.py"], check=True)


if __name__ == "__main__":
    run_bot()
