.PHONY: train train-long train-docker tune retrain backtest backtest-eth paper paper-ai live test lint fmt up down logs htpasswd

# ── Training ───────────────────────────────────────────────────────────────────
train:
	@echo ">>> Training SAC model (500k steps, ~60 min on CPU)..."
	PYTHONPATH=. python3 reinforcement_learning/train_sac.py
	@echo ">>> Done. Model saved to models/sac_model.zip"

train-long:
	@echo ">>> Training SAC model (1M steps, ~2 h on CPU)..."
	PYTHONPATH=. TOTAL_TIMESTEPS=1000000 python3 reinforcement_learning/train_sac.py

train-docker:
	@echo ">>> Starting SAC trainer in Docker (runs once, ~60 min)..."
	docker compose --profile training up trainer --build
	@echo ">>> Trainer done. Model saved to models/sac_model.zip"

tune:
	@echo ">>> Optuna hyperparameter search (30 trials × 50k steps)..."
	PYTHONPATH=. python3 reinforcement_learning/tune_sac.py
	@echo ">>> Best params saved to models/best_hyperparams.json"

retrain:
	@echo ">>> Walk-forward retraining (4-month windows, 1-month step)..."
	PYTHONPATH=. python3 -c "import asyncio, os; \
from src.data_loader import DataLoader; \
from reinforcement_learning.train_sac import train_walk_forward; \
async def r(): \
    l = DataLoader(); \
    await l.initialize(os.getenv('BYBIT_API_KEY',''), os.getenv('BYBIT_API_SECRET','')); \
    df = await l.load_ohlcv(os.getenv('TRADING_SYMBOL','BTC/USDT'), '15m', limit=17280); \
    train_walk_forward(df); \
asyncio.run(r())"

# ── Backtesting ────────────────────────────────────────────────────────────────
backtest:
	@echo ">>> Running walk-forward backtest..."
	PYTHONPATH=. python3 backtest.py

backtest-eth:
	PYTHONPATH=. BT_SYMBOL=ETH/USDT python3 backtest.py

# ── Paper trading (no real orders) ────────────────────────────────────────────
paper:
	@echo ">>> Starting bot in PAPER TRADING mode (MODE=local, no real orders)..."
	PAPER_TRADING=true MODE=local PYTHONPATH=. python3 supervisor.py

paper-ai:
	@echo ">>> Starting bot in PAPER TRADING + AI mode..."
	PAPER_TRADING=true MODE=ai PYTHONPATH=. python3 supervisor.py

# ── Live trading ───────────────────────────────────────────────────────────────
live:
	@echo ">>> Starting bot in LIVE mode. Ensure .env is configured!"
	PYTHONPATH=. python3 supervisor.py

# ── Tests & Lint ───────────────────────────────────────────────────────────────
test:
	PYTHONPATH=. python3 -m pytest tests/ -v --tb=short --cov=src --cov-report=term-missing

lint:
	python3 -m flake8 src/ tests/ reinforcement_learning/ config.py dashboard.py supervisor.py
	python3 -m mypy src/

fmt:
	python3 -m black src/ tests/ reinforcement_learning/ config.py dashboard.py supervisor.py
	python3 -m isort --profile black src/ tests/ reinforcement_learning/ config.py dashboard.py supervisor.py

# ── Docker ─────────────────────────────────────────────────────────────────────
up:
	docker compose up -d --build

down:
	docker compose down

rebuild:
	@echo ">>> Full reset: removing containers, volumes and images..."
	docker compose down -v
	docker image rm bitbotby-bot bitbotby-dashboard 2>/dev/null || true
	docker builder prune -f
	@echo ">>> Rebuilding from scratch..."
	docker compose build --no-cache
	docker compose up -d
	@echo ">>> Done. Run 'make logs' to watch startup."

logs:
	docker compose logs -f bot

htpasswd:
	@set -a && . ./.env && set +a && \
	docker run --rm httpd:alpine htpasswd -nb "$$DASHBOARD_USER" "$$DASHBOARD_PASSWORD" > nginx/htpasswd
	@echo ">>> nginx/htpasswd создан."

setup:
	@echo ">>> [1/3] Генерирую htpasswd..."
	@$(MAKE) htpasswd
	@echo ">>> [2/3] Останавливаю старые контейнеры..."
	@docker compose down
	@echo ">>> [3/3] Собираю и запускаю всё..."
	@docker compose up -d --build
	@echo ">>> Готово! Dashboard: http://localhost:8080"
