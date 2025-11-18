from stable_baselines3 import PPO
from src.rl_env import TradingEnv
from src.strategies import get_strategy
import sys

strategy_name = sys.argv[1] if len(sys.argv) > 1 else "scalping"
strategy = get_strategy(strategy_name)
env = TradingEnv(strategy=strategy)
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=strategy.get("initial_train_steps", 100000))
model.save(f"models/ppo_{strategy_name}")
