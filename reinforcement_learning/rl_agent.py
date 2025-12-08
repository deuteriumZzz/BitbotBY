import logging
import random
from collections import deque
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class DQN(nn.Module):
    def __init__(self, state_size: int, action_size: int):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, action_size)
        self.relu = nn.ReLU()

    def forward(self, state):
        x = self.relu(self.fc1(state))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        return self.fc4(x)


class RLAgent:
    def __init__(self, state_size: int, action_size: int):
        self.state_size = state_size
        self.action_size = action_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Main and target networks
        self.policy_net = DQN(state_size, action_size).to(self.device)
        self.target_net = DQN(state_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Optimizer and loss function
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()

        # Experience replay
        self.memory = deque(maxlen=10000)
        self.batch_size = 64

        # Training parameters
        self.gamma = 0.99  # discount factor
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.target_update_frequency = 1000
        self.steps_done = 0

        self.logger = logging.getLogger(__name__)
        self.logger.info(f"RL Agent initialized on {self.device}")

    def get_state(self, market_data: Dict, portfolio_state: Dict) -> np.ndarray:
        """Convert market and portfolio data into state vector"""
        state = []

        # Market features
        state.extend(
            [
                market_data.get("price", 0),
                market_data.get("volume", 0),
                market_data.get("rsi", 50),
                market_data.get("macd", 0),
                market_data.get("bollinger_upper", 0),
                market_data.get("bollinger_lower", 0),
                market_data.get("volatility", 0),
            ]
        )

        # Portfolio features
        state.extend(
            [
                portfolio_state.get("total_value", 0),
                portfolio_state.get("total_unrealized_pnl", 0),
                portfolio_state.get("concentration_risk", 0),
                portfolio_state.get("cash_balance", 0),
            ]
        )

        return np.array(state, dtype=np.float32)

    def choose_action(self, state: np.ndarray, valid_actions: List[int] = None) -> int:
        """Choose action using epsilon-greedy policy"""
        if random.random() < self.epsilon:
            # Exploration: random valid action
            if valid_actions:
                return random.choice(valid_actions)
            return random.randint(0, self.action_size - 1)

        # Exploitation: best action from policy network
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)

            if valid_actions:
                # Mask invalid actions
                mask = torch.full((self.action_size,), -float("inf")).to(self.device)
                mask[valid_actions] = 0
                q_values = q_values + mask

            return q_values.argmax().item()

    def remember(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        """Store experience in replay memory"""
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        """Train on batch from replay memory"""
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Current Q values
        current_q = self.policy_net(states).gather(1, actions)

        # Next Q values from target network
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q

        # Compute loss
        loss = self.criterion(current_q.squeeze(), target_q)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        # Update exploration rate
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        # Update target network
        self.steps_done += 1
        if self.steps_done % self.target_update_frequency == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss.item()

    def calculate_reward(self, trade_result: Dict, portfolio_state: Dict) -> float:
        """Calculate reward based on trade outcome"""
        reward = 0

        # PnL based reward
        reward += trade_result.get("pnl", 0) * 10

        # Risk penalty
        reward -= portfolio_state.get("concentration_risk", 0) * 100

        # Transaction cost penalty
        reward -= trade_result.get("transaction_cost", 0) * 5

        # Time decay for holding positions
        if trade_result.get("holding_time", 0) > 10:
            reward -= trade_result["holding_time"] * 0.1

        return reward

    def save_model(self, path: str):
        """Save model weights"""
        torch.save(
            {
                "policy_net_state_dict": self.policy_net.state_dict(),
                "target_net_state_dict": self.target_net.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
                "steps_done": self.steps_done,
            },
            path,
        )
        self.logger.info(f"Model saved to {path}")

    def load_model(self, path: str):
        """Load model weights"""
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint["policy_net_state_dict"])
        self.target_net.load_state_dict(checkpoint["target_net_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epsilon = checkpoint["epsilon"]
        self.steps_done = checkpoint["steps_done"]
        self.logger.info(f"Model loaded from {path}")

    def get_action_meanings(self) -> Dict[int, str]:
        """Get meaning of each action index"""
        return {0: "HOLD", 1: "BUY", 2: "SELL", 3: "CLOSE_POSITION"}
