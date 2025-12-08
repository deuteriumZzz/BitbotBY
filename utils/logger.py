import json
import logging
import logging.handlers
import os
from datetime import datetime
from typing import Optional

from config import Config


class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging"""

    def format(self, record):
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Add extra data if present
        if hasattr(record, "extra"):
            log_data.update(record.extra)

        return json.dumps(log_data)


def setup_logger(
    name: str = __name__, level: int = logging.INFO, log_file: Optional[str] = None
) -> logging.Logger:
    """Setup and configure logger"""

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers
    if logger.handlers:
        return logger

    # Create formatters
    json_formatter = JSONFormatter()
    simple_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(simple_formatter)
    logger.addHandler(console_handler)

    # File handler if log file specified
    if log_file:
        # Create logs directory if it doesn't exist
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)

        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(json_formatter)
        logger.addHandler(file_handler)

    # Add trace level for more detailed debugging
    logging.TRACE = logging.DEBUG - 5
    logging.addLevelName(logging.TRACE, "TRACE")

    def trace(self, message, *args, **kwargs):
        if self.isEnabledFor(logging.TRACE):
            self._log(logging.TRACE, message, args, **kwargs)

    logging.Logger.trace = trace

    return logger


def get_logger(name: str = None) -> logging.Logger:
    """Get configured logger instance"""
    if name is None:
        name = Config.APP_NAME

    logger = logging.getLogger(name)

    # If logger not configured, set it up
    if not logger.handlers:
        log_file = os.path.join(Config.LOGS_DIR, f"{Config.APP_NAME}.log")
        logger = setup_logger(name, Config.LOG_LEVEL, log_file)

    return logger


class LoggerAdapter:
    """Adapter for adding contextual information to logs"""

    def __init__(self, logger: logging.Logger, extra: dict = None):
        self.logger = logger
        self.extra = extra or {}

    def add_context(self, **kwargs):
        """Add contextual information"""
        self.extra.update(kwargs)
        return self

    def remove_context(self, *keys):
        """Remove contextual information"""
        for key in keys:
            self.extra.pop(key, None)
        return self

    def log(self, level, msg, *args, **kwargs):
        """Log with contextual information"""
        if self.extra:
            kwargs["extra"] = {"extra": self.extra}
        self.logger.log(level, msg, *args, **kwargs)

    def debug(self, msg, *args, **kwargs):
        self.log(logging.DEBUG, msg, *args, **kwargs)

    def info(self, msg, *args, **kwargs):
        self.log(logging.INFO, msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        self.log(logging.WARNING, msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        self.log(logging.ERROR, msg, *args, **kwargs)

    def critical(self, msg, *args, **kwargs):
        self.log(logging.CRITICAL, msg, *args, **kwargs)

    def trace(self, msg, *args, **kwargs):
        """Trace level logging"""
        if hasattr(self.logger, "trace"):
            if self.extra:
                kwargs["extra"] = {"extra": self.extra}
            self.logger.trace(msg, *args, **kwargs)


# Global logger instance
logger = get_logger()


def log_performance(metrics: dict, strategy: str = None):
    """Log performance metrics"""
    extra = {"metrics": metrics}
    if strategy:
        extra["strategy"] = strategy

    logger.info("Performance metrics", extra={"extra": extra})


def log_trade(trade_data: dict):
    """Log trade execution"""
    logger.info("Trade executed", extra={"extra": trade_data})


def log_market_data(symbol: str, data: dict):
    """Log market data update"""
    logger.debug(
        f"Market data update for {symbol}",
        extra={"extra": {"symbol": symbol, "data": data}},
    )


def log_portfolio_update(portfolio_state: dict):
    """Log portfolio update"""
    logger.info("Portfolio updated", extra={"extra": portfolio_state})


def setup_error_handler():
    """Setup global error handler"""
    import sys

    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return

        logger.critical(
            "Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback)
        )

    sys.excepthook = handle_exception


# Setup error handler when module is imported
setup_error_handler()
