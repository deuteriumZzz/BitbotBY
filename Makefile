# Python проекта: локальное окружение .venv (3.11), иначе python3.11.
PY ?= $(shell [ -x .venv/bin/python ] && echo $(CURDIR)/.venv/bin/python || echo python3.11)
LINT_IMPORTS ?= $(shell [ -x .venv/bin/lint-imports ] && echo $(CURDIR)/.venv/bin/lint-imports || echo lint-imports)

.PHONY: train train-long train-docker tune tune-bluechip tune-altcoin retrain backtest backtest-eth alpha-report live-divergence ai-isolation paper paper-ai live test lint fmt local frontend

# ── Training ───────────────────────────────────────────────────────────────────
train:
	@echo ">>> Training SAC model (500k steps, ~60 min on CPU)..."
	$(PY) -m bitbot.ml.train_sac
	@echo ">>> Done. Model saved to models/sac_model.zip"

train-long:
	@echo ">>> Training SAC model (1M steps, ~2 h on CPU)..."
	TOTAL_TIMESTEPS=1000000 $(PY) -m bitbot.ml.train_sac

train-docker:
	@echo ">>> Starting SAC trainer in Docker (runs once, ~60 min)..."
	$(LEGACY_COMPOSE) --profile training up trainer --build
	@echo ">>> Trainer done. Model saved to models/sac_model.zip"

tune:
	@echo ">>> Optuna hyperparameter search (30 trials × 50k steps, top-5 by volume)..."
	$(PY) -m bitbot.ml.tune_sac
	@echo ">>> Best params saved to models/best_hyperparams.json"

tune-bluechip:
	@echo ">>> Tuning for BLUECHIP season (BTC/ETH/SOL/BNB/XRP)..."
	SAC_PROFILE=bluechip $(PY) -m bitbot.ml.tune_sac
	@echo ">>> Best params saved to models/best_hyperparams_bluechip.json"
	@echo ">>> Now run: SAC_PROFILE=bluechip make train"

tune-altcoin:
	@echo ">>> Tuning for ALTCOIN season (top-5 alts by volume, excl. bluechip)..."
	SAC_PROFILE=altcoin $(PY) -m bitbot.ml.tune_sac
	@echo ">>> Best params saved to models/best_hyperparams_altcoin.json"
	@echo ">>> Now run: SAC_PROFILE=altcoin make train"

retrain:
	@echo ">>> Walk-forward retraining (4-month windows, 1-month step)..."
	$(PY) -c "import asyncio, os; \
from bitbot.market.data_loader import DataLoader; \
from bitbot.ml.train_sac import train_walk_forward; \
async def r(): \
    l = DataLoader(); \
    await l.initialize(os.getenv('BYBIT_API_KEY',''), os.getenv('BYBIT_API_SECRET','')); \
    df = await l.load_ohlcv(os.getenv('TRADING_SYMBOL','BTC/USDT'), '15m', limit=17280); \
    train_walk_forward(df); \
asyncio.run(r())"

# ── Backtesting ────────────────────────────────────────────────────────────────
backtest:
	@echo ">>> Running walk-forward backtest..."
	$(PY) -m bitbot.research.backtest

backtest-eth:
	BT_SYMBOL=ETH/USDT $(PY) -m bitbot.research.backtest

alpha-report:
	@echo ">>> Checking statistical significance of real paper/live trades..."
	$(PY) -m bitbot.research.paper_alpha_report

live-divergence:
	@echo ">>> Comparing live execution price vs backtest impact model..."
	$(PY) -m bitbot.research.backtest_live_divergence

ai-isolation:
	@echo ">>> Isolating AI(LLM) signal contribution (uses real AI provider calls)..."
	$(PY) -m bitbot.research.ai_signal_isolation_backtest

# ── Paper trading (no real orders) ────────────────────────────────────────────
paper:
	@echo ">>> Starting bot in PAPER TRADING mode (MODE=local, no real orders)..."
	PAPER_TRADING=true MODE=local $(PY) -m bitbot.workers.supervisor

paper-ai:
	@echo ">>> Starting bot in PAPER TRADING + AI mode..."
	PAPER_TRADING=true MODE=ai $(PY) -m bitbot.workers.supervisor

# ── Локально без Docker: бот + дашборд + приложение (настройки из .env) ─────────
local:
	@echo ">>> Bot + dashboard on this computer. App: http://localhost:8080/app/"
	$(PY) -m bitbot.workers.supervisor --with-dashboard

frontend:
	cd frontend && npm ci && npm run build

# ── Live trading ───────────────────────────────────────────────────────────────
live:
	@echo ">>> Starting bot in LIVE mode. Ensure .env is configured!"
	$(PY) -m bitbot.workers.supervisor

# ── Tests & Lint ───────────────────────────────────────────────────────────────
test:
	$(PY) -m pytest backend/tests -v --tb=short --cov=bitbot --cov-report=term-missing

lint:
	$(PY) -m ruff check backend
	cd backend && $(LINT_IMPORTS)
	cd backend && $(PY) -m mypy bitbot

fmt:
	$(PY) -m black backend
	$(PY) -m ruff check --fix --select I backend
