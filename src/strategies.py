def get_strategy(name):
    strategies = {
        "scalping": {
            "timeframe": "1m",
            "max_steps": 500,
            "sentiment_threshold": 0.1,
            "buy_pct": 0.02,
            "max_volume": 0.005,
            "max_loss_pct": 0.05,
            "stop_loss_pct": 0.02,
            "take_profit_pct": 0.05,
            "cycle_pause": 5,
            "train_batch": 50,
            "initial_train_steps": 50000
        },
        "pipsing": {
            "timeframe": "5m",
            "max_steps": 1000,
            "sentiment_threshold": 0.15,
            "buy_pct": 0.05,
            "max_volume": 0.01,
            "max_loss_pct": 0.1,
            "stop_loss_pct": 0.03,
            "take_profit_pct": 0.08,
            "cycle_pause": 10,
            "train_batch": 100,
            "initial_train_steps": 100000
        },
        "intraday": {
            "timeframe": "1h",
            "max_steps": 2000,
            "sentiment_threshold": 0.2,
            "buy_pct": 0.1,
            "max_volume": 0.02,
            "max_loss_pct": 0.15,
            "stop_loss_pct": 0.05,
            "take_profit_pct": 0.15,
            "cycle_pause": 60,
            "train_batch": 200,
            "initial_train_steps": 200000
        }
    }
    return strategies.get(name, strategies["scalping"])
