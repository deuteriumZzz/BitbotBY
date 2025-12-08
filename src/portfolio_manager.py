import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional
from datetime import datetime
from .redis_client import RedisClient
from config import Config

class PortfolioManager:
    def __init__(self):
        self.redis = RedisClient()
        self.positions = {}
        self.performance_history = []
        self.logger = logging.getLogger(__name__)
        
        # Load existing positions from Redis
        self._load_positions()
    
    def _load_positions(self):
        """Load positions from Redis"""
        positions_data = self.redis.load_trading_state('portfolio_positions')
        if positions_data:
            self.positions = positions_data
            self.logger.info(f"Loaded {len(self.positions)} positions from Redis")
    
    def _save_positions(self):
        """Save positions to Redis"""
        self.redis.save_trading_state('portfolio_positions', self.positions)
    
    async def update_portfolio(self, symbol: str, action: str, quantity: float, price: float):
        """Update portfolio with new trade"""
        try:
            trade_value = quantity * price
            current_time = datetime.now().isoformat()
            
            if action == 'buy':
                if symbol in self.positions:
                    # Average cost calculation
                    old_quantity = self.positions[symbol]['quantity']
                    old_cost = self.positions[symbol]['cost_basis']
                    new_quantity = old_quantity + quantity
                    new_cost_basis = ((old_quantity * old_cost) + trade_value) / new_quantity
                    
                    self.positions[symbol].update({
                        'quantity': new_quantity,
                        'cost_basis': new_cost_basis,
                        'last_update': current_time
                    })
                else:
                    # New position
                    self.positions[symbol] = {
                        'quantity': quantity,
                        'cost_basis': price,
                        'entry_price': price,
                        'entry_time': current_time,
                        'last_update': current_time
                    }
            
            elif action == 'sell':
                if symbol in self.positions:
                    position = self.positions[symbol]
                    if quantity >= position['quantity']:
                        # Close entire position
                        profit_loss = (price - position['cost_basis']) * position['quantity']
                        self._record_trade(symbol, 'close', position['quantity'], price, profit_loss)
                        del self.positions[symbol]
                    else:
                        # Partial close
                        profit_loss = (price - position['cost_basis']) * quantity
                        self._record_trade(symbol, 'partial_close', quantity, price, profit_loss)
                        self.positions[symbol]['quantity'] -= quantity
                        self.positions[symbol]['last_update'] = current_time
            
            # Save updated positions
            self._save_positions()
            
            # Update performance history
            await self._update_performance_history()
            
            self.logger.info(f"Portfolio updated: {action} {quantity} {symbol} @ {price}")
            
        except Exception as e:
            self.logger.error(f"Error updating portfolio: {e}")
    
    def _record_trade(self, symbol: str, action: str, quantity: float, price: float, pnl: float):
        """Record trade in performance history"""
        trade_record = {
            'symbol': symbol,
            'action': action,
            'quantity': quantity,
            'price': price,
            'pnl': pnl,
            'timestamp': datetime.now().isoformat(),
            'portfolio_value': self.calculate_portfolio_value(price)  # Requires current price
        }
        
        self.performance_history.append(trade_record)
        
        # Save to Redis
        self.redis.save_trading_state('performance_history', self.performance_history)
    
    async def _update_performance_history(self):
        """Update performance history in Redis"""
        if len(self.performance_history) > 1000:  # Keep only last 1000 trades
            self.performance_history = self.performance_history[-1000:]
        
        self.redis.save_trading_state('performance_history', self.performance_history)
    
    def calculate_portfolio_value(self, current_prices: Dict[str, float]) -> float:
        """Calculate total portfolio value"""
        total_value = 0
        
        for symbol, position in self.positions.items():
            if symbol in current_prices:
                market_value = position['quantity'] * current_prices[symbol]
                total_value += market_value
        
        return total_value
    
    def calculate_unrealized_pnl(self, current_prices: Dict[str, float]) -> Dict[str, float]:
        """Calculate unrealized PnL for each position"""
        unrealized_pnl = {}
        
        for symbol, position in self.positions.items():
            if symbol in current_prices:
                current_value = position['quantity'] * current_prices[symbol]
                cost_basis = position['quantity'] * position['cost_basis']
                unrealized_pnl[symbol] = current_value - cost_basis
        
        return unrealized_pnl
    
    async def rebalance_portfolio(self, target_allocations: Dict[str, float], current_prices: Dict[str, float]):
        """Rebalance portfolio to target allocations"""
        total_value = self.calculate_portfolio_value(current_prices)
        
        if total_value == 0:
            self.logger.warning("Cannot rebalance: portfolio value is zero")
            return
        
        current_allocations = {}
        for symbol in self.positions:
            if symbol in current_prices:
                position_value = self.positions[symbol]['quantity'] * current_prices[symbol]
                current_allocations[symbol] = position_value / total_value
        
        # Calculate required trades
        rebalance_orders = []
        
        for symbol, target_allocation in target_allocations.items():
            current_allocation = current_allocations.get(symbol, 0)
            target_value = total_value * target_allocation
            
            if symbol in current_prices and current_prices[symbol] > 0:
                if current_allocation < target_allocation:
                    # Need to buy
                    current_value = self.positions.get(symbol, {}).get('quantity', 0) * current_prices[symbol]
                    buy_value = target_value - current_value
                    quantity = buy_value / current_prices[symbol]
                    
                    if quantity > 0:
                        rebalance_orders.append({
                            'symbol': symbol,
                            'action': 'buy',
                            'quantity': quantity,
                            'reason': 'rebalance'
                        })
                
                elif current_allocation > target_allocation:
                    # Need to sell
                    current_value = self.positions[symbol]['quantity'] * current_prices[symbol]
                    sell_value = current_value - target_value
                    quantity = sell_value / current_prices[symbol]
                    
                    if quantity > 0:
                        rebalance_orders.append({
                            'symbol': symbol,
                            'action': 'sell',
                            'quantity': quantity,
                            'reason': 'rebalance'
                        })
        
        return rebalance_orders
    
    def get_portfolio_summary(self, current_prices: Dict[str, float]) -> Dict:
        """Get portfolio summary"""
        total_value = self.calculate_portfolio_value(current_prices)
        unrealized_pnl = self.calculate_unrealized_pnl(current_prices)
        total_unrealized = sum(unrealized_pnl.values())
        
        return {
            'total_value': total_value,
            'total_unrealized_pnl': total_unrealized,
            'positions_count': len(self.positions),
            'diversification': self._calculate_diversification(current_prices),
            'performance': self._calculate_performance_metrics(),
            'positions': self.positions,
            'unrealized_pnl': unrealized_pnl
        }
    
    def _calculate_diversification(self, current_prices: Dict[str, float]) -> float:
        """Calculate portfolio diversification index"""
        if not self.positions:
            return 0
        
        position_values = []
        for symbol, position in self.positions.items():
            if symbol in current_prices:
                value = position['quantity'] * current_prices[symbol]
                position_values.append(value)
        
        if not position_values:
            return 0
        
        # Herfindahl-Hirschman Index (lower is better)
        total_value = sum(position_values)
        if total_value == 0:
            return 0
        
        hhi = sum((value / total_value) ** 2 for value in position_values)
        return 1 - hhi  # Inverse so higher is better
    
    def _calculate_performance_metrics(self) -> Dict:
        """Calculate performance metrics from trade history"""
        if not self.performance_history:
            return {}
        
        winning_trades = [trade for trade in self.performance_history if trade['pnl'] > 0]
        losing_trades = [trade for trade in self.performance_history if trade['pnl'] < 0]
        
        total_pnl = sum(trade['pnl'] for trade in self.performance_history)
        avg_win = np.mean([trade['pnl'] for trade in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([trade['pnl'] for trade in losing_trades]) if losing_trades else 0
        
        return {
            'total_trades': len(self.performance_history),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': len(winning_trades) / len(self.performance_history) if self.performance_history else 0,
            'total_pnl': total_pnl,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
        }
    
    async def risk_assessment(self, current_prices: Dict[str, float]) -> Dict:
        """Perform portfolio risk assessment"""
        portfolio_value = self.calculate_portfolio_value(current_prices)
        
        if portfolio_value == 0:
            return {}
        
        # Calculate concentration risk
        position_values = []
        for symbol, position in self.positions.items():
            if symbol in current_prices:
                value = position['quantity'] * current_prices[symbol]
                position_values.append(value)
        
        concentration_risk = max(position_values) / portfolio_value if position_values else 0
        
        # Calculate volatility (simplified)
        pnl_values = [trade['pnl'] for trade in self.performance_history[-30:]]  # Last 30 trades
        volatility = np.std(pnl_values) if pnl_values else 0
        
        return {
            'concentration_risk': concentration_risk,
            'volatility': volatility,
            'value_at_risk': self._calculate_var(portfolio_value),
            'max_drawdown': self._calculate_max_drawdown(),
            'sharpe_ratio': self._calculate_sharpe_ratio()
        }
    
    def _calculate_var(self, portfolio_value: float, confidence_level: float = 0.95) -> float:
        """Calculate Value at Risk"""
        # Simplified VaR calculation
        if not self.performance_history:
            return 0
        
        pnl_values = [trade['pnl'] for trade in self.performance_history]
        return np.percentile(pnl_values, (1 - confidence_level) * 100)
    
    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown from performance history"""
        if not self.performance_history:
            return 0
        
        portfolio_values = [trade['portfolio_value'] for trade in self.performance_history]
        peak = portfolio_values[0]
        max_drawdown = 0
        
        for value in portfolio_values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        
        return max_drawdown
    
    def _calculate_sharpe_ratio(self, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        if not self.performance_history or len(self.performance_history) < 2:
            return 0
        
        returns = []
        for i in range(1, len(self.performance_history)):
            ret = (self.performance_history[i]['portfolio_value'] - 
                  self.performance_history[i-1]['portfolio_value']) / \
                  self.performance_history[i-1]['portfolio_value']
            returns.append(ret)
        
        if not returns:
            return 0
        
        excess_returns = [r - risk_free_rate/365 for r in returns]  # Daily risk-free rate
        sharpe = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(365)
        
        return sharpe
