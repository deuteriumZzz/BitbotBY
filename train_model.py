from stable_baselines3 import PPO
from rl_env import TradingEnv

env = TradingEnv()
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100000)  # Обучай на исторических данных
model.save("models/ppo_trading_model.zip")
