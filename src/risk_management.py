import numpy as np
import logging
from typing import Dict, Any
from src.redis_client import RedisClient

class RiskManager:
    def __init__(self, initial_balance: float, risk_per_trade: float = 0.02):
        self.initial_balance = initial_balance
        self.risk_per_trade = risk_per_trade
        self.redis = RedisClient()
        self.logger = logging.getLogger(__name__)
    
    async def calculate_position_size(self, current_balance: float, entry_price: float, stop_loss: float) -> float:
        """Calculate position size based on risk management rules"""
        risk_amount = current_balance * self.risk_per_trade
        price_difference = abs(entry_price - stop_loss)
        
        if price_difference == 0:
            return 0
        
        position_size = risk_amount / price_difference
        
        # Save risk calculation to Redis
        risk_data = {
            'timestamp': np.datetime64('now').astype(str),
            'current_balance': current_balance,
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'position_size': position_size,
            'risk_amount': risk_amount
        }
        self.redis.save_trading_state('risk_calculation', risk_data)
        
        return position_size
    
    async def calculate_stop_loss(self, entry_price: float, signal: Dict[str, Any]) -> float:
        """Calculate stop loss price"""
        if signal['action'] == 'buy':
            stop_loss = entry_price * 0.95  # 5% stop loss
        elif signal['action'] == 'sell':
            stop_loss = entry_price * 1.05  # 5% stop loss
        else:
            stop_loss = entry_price
        
        return stop_loss
    
    async def calculate_take_profit(self, entry_price: float, signal: Dict[str, Any]) -> float:
        """Calculate take profit price"""
        risk_reward_ratio = 2.0  # 1:2 risk-reward ratio
        
        if signal['action'] == 'buy':
            stop_loss = await self.calculate_stop_loss(entry_price, signal)
            risk = entry_price - stop_loss
            take_profit = entry_price + (risk * risk_reward_ratio)
        elif signal['action'] == 'sell':
            stop_loss = await self.calculate_stop_loss(entry_price, signal)
            risk = stop_loss - entry_price
            take_profit = entry_price - (risk * risk_reward_ratio)
        else:
            take_profit = entry_price
        
        return take_profit
    
    async def validate_signal(self, signal: Dict[str, Any], market_data: Any) -> bool:
        """Validate trading signal based on risk rules"""
        if signal['action'] == 'hold':
            return False
        
        # Check if confidence is sufficient
        if signal.get('confidence', 0) < 0.6:
            self.logger.warning("Signal confidence too low")
            return False
        
        # Additional risk checks can be added here
        return True
