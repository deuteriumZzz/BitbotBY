import pytest
import pandas as pd
from datetime import datetime
from src.redis_client import RedisClient

@pytest.fixture
def redis_client():
    return RedisClient()

def test_redis_connection(redis_client):
    """Test Redis connection"""
    assert redis_client.redis_client.ping()

def test_data_caching(redis_client):
    """Test market data caching"""
    test_data = pd.DataFrame({
        'close': [100, 101, 102],
        'volume': [1000, 1100, 1200]
    })
    
    # Save data
    redis_client.save_market_data('TEST', '1h', test_data)
    
    # Load data
    cached_data = redis_client.load_market_data('TEST', '1h')
    
    assert cached_data is not None
    assert cached_data['close'].iloc[0] == 100

def test_trading_state(redis_client):
    """Test trading state management"""
    test_state = {
        'last_signal': {'action': 'buy', 'confidence': 0.8},
        'timestamp': datetime.now().isoformat()
    }
    
    # Save state
    redis_client.save_trading_state('TEST_SYMBOL', test_state)
    
    # Load state
    loaded_state = redis_client.load_trading_state('TEST_SYMBOL')
    
    assert loaded_state is not None
    assert loaded_state['last_signal']['action'] == 'buy'

def test_model_storage(redis_client):
    """Test model storage"""
    test_model = {
        'parameters': {'param1': 'value1'},
        'trained_at': datetime.now().isoformat()
    }
    
    # Save model
    redis_client.save_model('TEST_STRATEGY', test_model)
    
    # Load model
    loaded_model = redis_client.load_model('TEST_STRATEGY')
    
    assert loaded_model is not None
    assert loaded_model['parameters']['param1'] == 'value1'

def test_lock_mechanism(redis_client):
    """Test distributed locking"""
    lock_acquired = redis_client.acquire_lock('test_lock')
    assert lock_acquired
    
    # Try to acquire same lock again
    lock_acquired_again = redis_client.acquire_lock('test_lock')
    assert not lock_acquired_again
    
    # Release lock
    redis_client.release_lock('test_lock')
    
    # Now should be able to acquire again
    lock_acquired_after_release = redis_client.acquire_lock('test_lock')
    assert lock_acquired_after_release
    
    redis_client.release_lock('test_lock')
