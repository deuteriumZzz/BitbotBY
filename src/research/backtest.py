"""
backtest.py — Walk-forward бэктестирование стратегий BitbotBY.

Использование:
    python3 -m src.research.backtest

Переменные окружения:
    BT_MONTHS      месяцев истории (по умолчанию 6)
    BT_SYMBOL      символ (по умолчанию Config.SYMBOL)
    BT_TIMEFRAME   таймфрейм (по умолчанию Config.TIMEFRAME, или профиля)
    BT_MIN_CONF    мин. уверенность (по умолчанию Config.MIN_SIGNAL_CONFIDENCE)
    BT_PROFILE     bluechip|altcoin|meme — подставляет таймфрейм, пул символов
                   (как в live-сканере) и SL%/TP-множитель профиля вместо
                   ATR-based SL и RR=2.0. BT_SYMBOLS/BT_SYMBOL всё ещё имеют
                   приоритет, если заданы явно.
    BT_ROLLING_WINDOWS  >=2 → anchored walk-forward по N последовательным
                   фолдам вместо одного статичного 80/20-сплита (BT_HOLDOUT_RATIO
                   игнорируется в этом режиме). Показывает разброс overfit_ratio
                   по нескольким окнам, а не одно число — см.
                   run_strategy_rolling_walk_forward(). По умолчанию 0 (выкл).
"""

import asyncio
import json
import logging
import os
import statistics
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# Разрешаем импорты из корня проекта (файл лежит в src/research/)
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)

from config import JUNK_BASES, JUNK_PREFIXES, MEME_COIN_BASES, STABLECOIN_BASES, Config
from src.market.indicators import add_indicators
from src.market.market_impact import estimate_from_df as _ac_impact
from src.research.alpha_tester import AlphaTester
from src.settings.runtime_config import _MARKET_PROFILES
from src.strategies import STRATEGY_REGISTRY, BaseStrategy

# ── Константы ──────────────────────────────────────────────────────────────

_FETCH_LIMIT = 200  # размер батча ccxt (лимит Bybit)
_CSV_CACHE_TTL = 86_400  # 24 ч
_WARMUP = 50  # минимум свечей до первого сигнала
_ATR_MULT = 1.5  # множитель ATR для стоп-лосса
_RR = 2.0  # соотношение риск/доход (1:2)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ── Датакласс ──────────────────────────────────────────────────────────────


@dataclass
class BacktestResult:
    """Метрики бэктеста для одной стратегии."""

    strategy: str
    total_trades: int
    win_rate: float  # 0.0–1.0
    avg_profit_pct: float  # средний % выигрышной сделки (положительный)
    avg_loss_pct: float  # средний % проигрышной сделки (отрицательный)
    expected_value: float  # wr*avg_profit + (1-wr)*avg_loss
    total_return_pct: float  # (итог - начало) / начало
    max_drawdown_pct: float  # наихудшая просадка от пика до минимума %
    sharpe_ratio: float  # годовой, rf=0
    num_wins: int
    num_losses: int
    trades: list = field(default_factory=list, repr=False)
    trade_returns: list = field(default_factory=list, repr=False)


# ── Вспомогательные функции CSV-кэша (без Redis, без API-ключа) ───────────


def _csv_cache_path(symbol: str, timeframe: str) -> str:
    safe = symbol.replace("/", "_").replace(":", "_")
    os.makedirs("data/cache", exist_ok=True)
    return f"data/cache/{safe}_{timeframe}.csv"


def _load_csv_cache(path: str) -> Optional[pd.DataFrame]:
    if not os.path.exists(path):
        return None
    age = time.time() - os.path.getmtime(path)
    if age > _CSV_CACHE_TTL:
        return None
    try:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
        logger.info(f"CSV-кэш загружен: {path} ({len(df)} строк)")
        return df
    except Exception as e:
        logger.warning(f"Ошибка чтения CSV-кэша: {e}")
        return None


def _save_csv_cache(df: pd.DataFrame, path: str) -> None:
    try:
        df.to_csv(path)
    except Exception as e:
        logger.warning(f"Ошибка сохранения CSV-кэша: {e}")


# ── Топ-N символов по объёму за 24ч (публичный API, без авторизации) ─────


def _passes_base_filters(base: str, is_meme_profile: bool) -> bool:
    """Общие фильтры базового тикера — зеркалит market_scanner.py:75-92."""
    if base in STABLECOIN_BASES or base in JUNK_BASES:
        return False
    if not is_meme_profile and base in MEME_COIN_BASES:
        return False
    if any(base.startswith(p) for p in JUNK_PREFIXES):
        return False
    return True


async def _fetch_profile_symbols(profile: str, n: int) -> List[str]:
    """
    Резолвит пул символов для профиля так же, как live-бот
    (src/core/trading_bot.py:1380-1396, src/market/market_scanner.py:75-117):
    bluechip — топ-N по объёму; altcoin — топ-N по объёму в диапазоне
    min/max volume профиля, ИСКЛЮЧАЯ текущий топ-20 по объёму (bluechip
    определяется динамически, а не статическим списком — BLUECHIP_BASES
    в runtime_config.py нигде в реальном коде не используется).
    """
    cfg: Dict[str, Any] = _MARKET_PROFILES[profile]
    is_meme = profile == "meme"
    try:
        import ccxt.async_support as ccxt_async

        exchange = ccxt_async.bybit(
            {"enableRateLimit": True, "options": {"defaultType": "spot"}}
        )
        try:
            tickers = await exchange.fetch_tickers()
        finally:
            await exchange.close()

        usdt: Dict[str, Any] = {}
        for sym, t in tickers.items():
            if not sym.endswith("/USDT"):
                continue
            base = sym.split("/")[0]
            if not _passes_base_filters(base, is_meme):
                continue
            qv = t.get("quoteVolume") or 0
            if qv > 0:
                usdt[sym] = qv

        if profile == "altcoin":
            top20 = sorted(usdt.items(), key=lambda x: x[1], reverse=True)[:20]
            exclude_bases = {sym.split("/")[0] for sym, _ in top20}
            usdt = {
                sym: qv
                for sym, qv in usdt.items()
                if sym.split("/")[0] not in exclude_bases
            }

        min_vol = float(cfg.get("min_volume_usdt", 0))  # type: ignore[arg-type]
        max_vol = float(cfg.get("max_volume_usdt", 0))  # type: ignore[arg-type]
        filtered = {
            sym: qv
            for sym, qv in usdt.items()
            if qv >= min_vol and (max_vol <= 0 or qv <= max_vol)
        }

        ranked = sorted(filtered.items(), key=lambda x: x[1], reverse=True)
        symbols = [sym for sym, _ in ranked[:n]]
        logger.info(
            "[%s] Top %d symbols (vol %.0f-%.0f): %s...",
            profile,
            len(symbols),
            min_vol,
            max_vol,
            ", ".join(symbols[:8]),
        )
        return symbols
    except Exception as e:
        logger.error("_fetch_profile_symbols(%s) failed: %s", profile, e)
        return Config.SYMBOLS[:n]


async def _fetch_top_symbols(n: int = 20) -> List[str]:
    """Получает топ-N символов /USDT по объёму за 24ч с Bybit (без API-ключа)."""
    try:
        import ccxt.async_support as ccxt_async

        exchange = ccxt_async.bybit(
            {
                "enableRateLimit": True,
                "options": {"defaultType": "spot"},
            }
        )
        try:
            tickers = await exchange.fetch_tickers()
        finally:
            await exchange.close()

        usdt = {
            sym: t
            for sym, t in tickers.items()
            if sym.endswith("/USDT")
            and (t.get("quoteVolume") or 0) > 0
            and sym.split("/")[0] not in STABLECOIN_BASES
        }
        ranked = sorted(
            usdt.items(), key=lambda x: x[1].get("quoteVolume", 0), reverse=True
        )
        symbols = [sym for sym, _ in ranked[:n]]
        logger.info("Top %d symbols: %s...", n, ", ".join(symbols[:5]))
        return symbols
    except Exception as e:
        logger.error(
            "fetch_tickers завершился ошибкой: %s — возврат к Config.SYMBOLS", e
        )
        return Config.SYMBOLS[:n]


# ── Загрузка OHLCV (ccxt, публичные данные, авторизация не нужна) ─────────


async def _fetch_batches(
    symbol: str,
    timeframe: str,
    since_ms: int,
    until_ms: int,
) -> pd.DataFrame:
    """Загружает OHLCV через ccxt без API-ключа (публичные данные биржи)."""
    try:
        import ccxt.async_support as ccxt_async
    except ImportError:
        logger.error("ccxt не установлен")
        return pd.DataFrame()

    exchange = ccxt_async.bybit({"enableRateLimit": True})
    all_frames: List[pd.DataFrame] = []
    current = since_ms

    try:
        while current < until_ms:
            try:
                raw = await exchange.fetch_ohlcv(
                    symbol,
                    timeframe,
                    since=current,
                    limit=_FETCH_LIMIT,
                )
            except Exception as e:
                logger.error(f"Ошибка fetch_ohlcv: {e}")
                break

            if not raw:
                break

            df_batch = pd.DataFrame(
                raw,
                columns=[
                    "timestamp",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                ],
            )
            all_frames.append(df_batch)

            last_ts = int(raw[-1][0])
            if last_ts <= current:
                break
            current = last_ts + 1

            if current >= until_ms:
                break  # достигли запрошенной конечной даты

    finally:
        await exchange.close()

    if not all_frames:
        return pd.DataFrame()

    result = (
        pd.concat(all_frames, ignore_index=True)
        .drop_duplicates(subset=["timestamp"])
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    logger.info(f"Получено {len(result)} свечей всего")
    return result


_POOL_MULTIPLIER = 3  # запас кандидатов на случай нехватки данных/делистинга


async def _resolve_symbols_with_data(
    candidates: List[str],
    target_n: int,
    timeframe: str,
    months: int,
) -> "Dict[str, pd.DataFrame]":
    """
    Идёт по ranked-списку кандидатов сверху вниз и грузит данные, пока не
    наберёт ровно target_n монет с реальной историей — а не просто берёт
    первые target_n по рейтингу и молча теряет те, что без данных.

    Пример: топ-20 запрошен, но 4 монеты из топ-20 не имеют истории на
    Bybit (молодой листинг) — тогда идём дальше по рейтингу (21-е, 22-е
    место и т.д.), пока не наберём 20 монет с реальными данными, либо не
    исчерпаем пул кандидатов.
    """
    resolved: "Dict[str, pd.DataFrame]" = {}
    skipped: List[str] = []
    for symbol in candidates:
        if len(resolved) >= target_n:
            break
        df = await load_data(symbol, timeframe, months)
        if df.empty:
            skipped.append(symbol)
            continue
        resolved[symbol] = df

    if skipped:
        logger.info(
            "Пропущено %d монет без данных (%s) — добрано с рейтинга ниже",
            len(skipped),
            ", ".join(skipped[:10]) + ("..." if len(skipped) > 10 else ""),
        )
    if len(resolved) < target_n:
        logger.warning(
            "Набрано только %d/%d монет — пул кандидатов (%d) исчерпан",
            len(resolved),
            target_n,
            len(candidates),
        )
    return resolved


async def load_data(
    symbol: str,
    timeframe: str,
    months: int,
) -> pd.DataFrame:
    """
    Загружает OHLCV-данные с CSV-кэшем, затем вычисляет индикаторы.

    Использует кэш или historical_btc.csv если API недоступен.
    """
    cache_path = _csv_cache_path(symbol, timeframe)
    cached = _load_csv_cache(cache_path)

    now_ms = int(datetime.now().timestamp() * 1000)
    since_ms = int((datetime.now() - timedelta(days=months * 30)).timestamp() * 1000)

    combined: pd.DataFrame

    if cached is not None and not cached.empty:
        first_ts = int(cached["timestamp"].min())
        last_ts = int(cached["timestamp"].max())
        frames = [cached]

        # Кэш может быть построен под меньший диапазон months (например,
        # предыдущий прогон запрашивал 3 месяца, а этот — 12): в таком
        # случае докачиваем более старую историю, а не только новые свечи.
        if first_ts > since_ms:
            logger.info(
                "Кэш не покрывает запрошенный период — докачиваем историю "
                f"с {since_ms} по {first_ts}"
            )
            backfill_df = await _fetch_batches(
                symbol,
                timeframe,
                since_ms=since_ms,
                until_ms=first_ts,
            )
            if not backfill_df.empty:
                frames.insert(0, backfill_df)

        logger.info("Кэш найден — получаем инкрементальное обновление " f"с {last_ts}")
        new_df = await _fetch_batches(
            symbol,
            timeframe,
            since_ms=last_ts + 1,
            until_ms=now_ms,
        )
        if not new_df.empty:
            frames.append(new_df)

        combined = (
            pd.concat(frames, ignore_index=True)
            .drop_duplicates(subset=["timestamp"])
            .sort_values("timestamp")
            .reset_index(drop=True)
        )
    else:
        logger.info(
            f"Кэш отсутствует — загружаем историю {months}м "
            f"для {symbol} {timeframe}..."
        )
        combined = await _fetch_batches(
            symbol,
            timeframe,
            since_ms=since_ms,
            until_ms=now_ms,
        )
        if combined.empty:
            # последний резерв: пробуем historical_btc.csv в корне проекта
            fallback = os.path.join(_ROOT, "historical_btc.csv")
            if os.path.exists(fallback):
                logger.warning(f"API недоступен — загружаем резерв " f"{fallback}")
                combined = pd.read_csv(fallback)
                combined.columns = [c.lower() for c in combined.columns]
                needed = {"open", "high", "low", "close", "volume"}
                if not needed.issubset(set(combined.columns)):
                    logger.error("В резервном CSV отсутствуют обязательные колонки")
                    return pd.DataFrame()
                if "timestamp" not in combined.columns:
                    combined["timestamp"] = range(len(combined))

    if combined.empty:
        logger.error("Нет данных — невозможно запустить бэктест")
        return pd.DataFrame()

    _save_csv_cache(combined, cache_path)

    # Обрезаем до запрошенного окна — кэш может содержать более длинную
    # историю, накопленную прошлыми прогонами с другим значением months.
    combined = combined[combined["timestamp"] >= since_ms].reset_index(drop=True)

    logger.info(f"Всего свечей до расчёта индикаторов: {len(combined)}")

    # Вычисляем технические индикаторы
    combined = add_indicators(combined)

    # Псевдонимы, необходимые стратегиям
    if "ema_12" in combined.columns:
        combined["ema_short"] = combined["ema_12"]
    if "ema_26" in combined.columns:
        combined["ema_long"] = combined["ema_26"]

    return combined


# ── Вспомогательные функции симуляции ────────────────────────────────────


def _sharpe(returns: List[float]) -> float:
    """Коэффициент Sharpe на сделку (не годовой — периоды удержания разные)."""
    if len(returns) < 2:
        return 0.0
    mean_r = statistics.mean(returns)
    std_r = statistics.stdev(returns)
    if std_r == 0.0:
        return 0.0
    # НЕ умножаем на sqrt(252): этот множитель предполагает дневные доходности.
    # Здесь P&L% на сделку с переменным временем удержания — умножение на 252
    # завышает Sharpe ~в 15 раз для стратегий на 15м.
    return mean_r / std_r


def _compute_overfit_ratio(train_sharpe: float, test_sharpe: float) -> Optional[float]:
    """
    train/test Sharpe как мера переобучения. None = неприменимо.

    Раньше `train_sharpe / max(test_sharpe, 0.01)` считалось без учёта
    знака train_sharpe: при train_sharpe<=0 (нет edge в принципе — нечему
    переобучаться) формула всё равно давала число вида train_sharpe*100,
    маскируя реальную причину ("стратегия просто плохая") под ложным
    "сильно переобучена". Возвращаем None в этом случае — вызывающий код
    обязан явно показать "нет edge", а не подсунуть обманчивое число.
    """
    if train_sharpe <= 0:
        return None
    return train_sharpe / max(test_sharpe, 0.01)


# ── Основной движок walk-forward ──────────────────────────────────────────


def run_strategy(
    name: str,
    strategy: BaseStrategy,
    df: pd.DataFrame,
    initial_balance: float,
    risk_per_trade: float,
    commission_rate: float,
    stop_loss_pct: float,
    min_confidence: float,
    maker_commission_rate: Optional[float] = None,
    tp_multiplier: Optional[float] = None,
    use_fixed_sl: bool = False,
) -> BacktestResult:
    """
    Прогоняет одну стратегию по df в режиме walk-forward.
    Без look-ahead: сигнал генерируется на df.iloc[:i+1].

    Комиссия асимметрична, как в реальном исполнении (order_executor.py,
    bybit_api.py): вход и SL — market/stop_market (тейкер, commission_rate),
    TP — limit-ордер (мейкер, обычно дешевле, maker_commission_rate).

    use_fixed_sl=True — SL всегда entry*stop_loss_pct (как в профилях
    _MARKET_PROFILES), а не динамический ATR-стоп; tp_multiplier переопределяет
    глобальный _RR (bluechip=2x, altcoin=4x, meme=6x — см. runtime_config.py).
    """
    maker_rate = (
        maker_commission_rate if maker_commission_rate is not None else commission_rate
    )
    rr = tp_multiplier if tp_multiplier is not None else _RR
    balance = initial_balance
    peak_balance = initial_balance
    max_drawdown = 0.0

    open_trade: Optional[Dict[str, Any]] = None
    closed_trades: List[Dict[str, Any]] = []
    trade_returns: List[float] = []

    # Порты из Argent считают вес позиции по всей истории один раз (он причинный),
    # а не заново на каждом шаге walk-forward.
    prepare = getattr(strategy, "prepare", None)
    if callable(prepare):
        prepare(df)

    n = len(df)
    dot_interval = max(1000, n // 10)

    for i in range(_WARMUP, n):
        if i % dot_interval == 0:
            print(".", end="", flush=True)

        window = df.iloc[: i + 1]

        if open_trade is None:
            # ── Попытка войти в позицию ───────────────────────────
            try:
                signal = strategy.generate_signal(window)
            except Exception as exc:
                logger.warning(f"[{name}] signal error at i={i}: {exc}")
                continue

            action = signal.get("action", "hold")
            confidence = float(signal.get("confidence", 0.0))

            if action in ("buy", "sell") and confidence >= min_confidence:
                # Входим на OPEN следующей свечи, а не на CLOSE сигнальной —
                # исключаем look-ahead bias (сигнал срабатывает на закрытии бара;
                # ранний реалистичный вход — открытие бара i+1).
                if i + 1 >= n:
                    continue  # нет следующего бара; пропускаем сигнал в конце данных
                next_open = float(df.iloc[i + 1]["open"])
                # Критерий Келли после 10 закрытых сделок (half-Kelly, ограничен 20%)
                n_closed = len(closed_trades)
                if n_closed >= 10:
                    wr = sum(1 for t in closed_trades if t["pnl"] > 0) / n_closed
                    b = rr  # reward/risk ratio (профильный или дефолтный _RR)
                    kelly_f = max(0.0, min((wr * b - (1 - wr)) / b * 0.5, 0.20))
                    fraction = kelly_f if kelly_f > 0 else risk_per_trade
                else:
                    fraction = risk_per_trade
                position_usdt = balance * fraction
                # Адаптивный market impact Almgren-Chriss:
                # зависит от размера ордера и волатильности
                impact = _ac_impact(window, position_usdt)
                if action == "buy":
                    entry_price = next_open * (1.0 + impact)
                else:
                    entry_price = next_open * (1.0 - impact)
                atr_val = (
                    float(window["atr"].iloc[-1]) if "atr" in window.columns else 0.0
                )

                if use_fixed_sl:
                    sl_dist = entry_price * stop_loss_pct
                elif atr_val > 0:
                    sl_dist = _ATR_MULT * atr_val
                else:
                    sl_dist = entry_price * stop_loss_pct

                if action == "buy":
                    stop_loss = entry_price - sl_dist
                    take_profit = entry_price + rr * sl_dist
                    direction = 1
                else:
                    stop_loss = entry_price + sl_dist
                    take_profit = entry_price - rr * sl_dist
                    direction = -1

                quantity = position_usdt / entry_price
                entry_comm = quantity * entry_price * commission_rate
                balance -= (
                    position_usdt + entry_comm
                )  # резервируем капитал + платим комиссию

                open_trade = {
                    "action": action,
                    "entry_price": entry_price,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                    "direction": direction,
                    "quantity": quantity,
                    "position_usdt": position_usdt,
                    "entry_comm": entry_comm,
                    "entry_i": i,
                }

        else:
            # ── Управление открытой позицией ──────────────────────
            current_price = float(window["close"].iloc[-1])
            d = open_trade["direction"]
            sl = open_trade["stop_loss"]
            tp = open_trade["take_profit"]

            sl_hit = (d == 1 and current_price <= sl) or (
                d == -1 and current_price >= sl
            )
            tp_hit = (d == 1 and current_price >= tp) or (
                d == -1 and current_price <= tp
            )

            if sl_hit or tp_hit:
                exit_impact = _ac_impact(window, open_trade["position_usdt"])
                if d == 1:  # лонг → продаём по bid
                    exit_price = current_price * (1.0 - exit_impact)
                else:  # шорт → покупаем по ask
                    exit_price = current_price * (1.0 + exit_impact)
                qty = open_trade["quantity"]
                # TP закрывается лимитным ордером (мейкер), SL — stop_market (тейкер)
                exit_rate = maker_rate if tp_hit else commission_rate
                exit_comm = qty * exit_price * exit_rate
                total_comm = open_trade["entry_comm"] + exit_comm
                pnl = (exit_price - open_trade["entry_price"]) * qty * d - total_comm
                balance += qty * exit_price - exit_comm
                balance = max(balance, 0.0)

                peak_balance = max(peak_balance, balance)
                dd = (
                    (peak_balance - balance) / peak_balance if peak_balance > 0 else 0.0
                )
                max_drawdown = max(max_drawdown, dd)

                pos_usdt = open_trade["position_usdt"]
                ret = pnl / pos_usdt if pos_usdt > 0 else 0.0
                trade_returns.append(ret)

                closed_trades.append(
                    {
                        "entry_i": open_trade["entry_i"],
                        "exit_i": i,
                        "action": open_trade["action"],
                        "entry_price": open_trade["entry_price"],
                        "exit_price": exit_price,
                        "pnl": round(pnl, 6),
                        "return_pct": round(ret, 6),
                        "closed_by": "tp" if tp_hit else "sl",
                    }
                )
                open_trade = None

    # ── Принудительное закрытие оставшейся позиции ───────────────
    if open_trade is not None and n > 0:
        d = open_trade["direction"]
        mid_close = float(df["close"].iloc[-1])
        close_impact = _ac_impact(df.iloc[-50:], open_trade["position_usdt"])
        if d == 1:
            exit_price = mid_close * (1.0 - close_impact)
        else:
            exit_price = mid_close * (1.0 + close_impact)
        qty = open_trade["quantity"]
        d = open_trade["direction"]
        exit_comm = qty * exit_price * commission_rate
        total_comm = open_trade["entry_comm"] + exit_comm
        pnl = (exit_price - open_trade["entry_price"]) * qty * d - total_comm
        balance += qty * exit_price - exit_comm
        balance = max(balance, 0.0)

        peak_balance = max(peak_balance, balance)
        dd = (peak_balance - balance) / peak_balance if peak_balance > 0 else 0.0
        max_drawdown = max(max_drawdown, dd)

        pos_usdt = open_trade["position_usdt"]
        ret = pnl / pos_usdt if pos_usdt > 0 else 0.0
        trade_returns.append(ret)

        closed_trades.append(
            {
                "entry_i": open_trade["entry_i"],
                "exit_i": n - 1,
                "action": open_trade["action"],
                "entry_price": open_trade["entry_price"],
                "exit_price": exit_price,
                "pnl": round(pnl, 6),
                "return_pct": round(ret, 6),
                "closed_by": "force",
            }
        )

    # ── Агрегация метрик ─────────────────────────────────────────
    total_trades = len(closed_trades)
    wins = [t for t in closed_trades if t["pnl"] > 0]
    losses = [t for t in closed_trades if t["pnl"] <= 0]

    win_rate = len(wins) / total_trades if total_trades > 0 else 0.0
    avg_profit = statistics.mean([t["return_pct"] for t in wins]) if wins else 0.0
    avg_loss = statistics.mean([t["return_pct"] for t in losses]) if losses else 0.0
    ev = win_rate * avg_profit + (1 - win_rate) * avg_loss

    total_return = (
        (balance - initial_balance) / initial_balance if initial_balance > 0 else 0.0
    )
    sharpe = _sharpe(trade_returns)

    return BacktestResult(
        strategy=name,
        total_trades=total_trades,
        win_rate=win_rate,
        avg_profit_pct=avg_profit,
        avg_loss_pct=avg_loss,
        expected_value=ev,
        total_return_pct=total_return,
        max_drawdown_pct=max_drawdown,
        sharpe_ratio=sharpe,
        num_wins=len(wins),
        num_losses=len(losses),
        trades=closed_trades,
        trade_returns=trade_returns,
    )


# ── Walk-forward с hold-out test set ─────────────────────────────────────


def run_strategy_with_holdout(
    name: str,
    strategy_cls: type,
    df: pd.DataFrame,
    initial_balance: float,
    risk_per_trade: float,
    commission_rate: float,
    stop_loss_pct: float,
    min_confidence: float,
    holdout_ratio: float = 0.2,
    maker_commission_rate: Optional[float] = None,
    tp_multiplier: Optional[float] = None,
    use_fixed_sl: bool = False,
) -> BacktestResult:
    """
    Прогоняет стратегию с честным hold-out test set.

    holdout_ratio=0.0 → поведение идентично run_strategy().
    holdout_ratio=0.2 → последние 20% данных выделяются как out-of-sample тест;
    walk-forward оптимизация проходит только на оставшихся 80%.
    """
    if holdout_ratio <= 0.0 or len(df) < _WARMUP * 2:
        return run_strategy(
            name,
            strategy_cls(),
            df,
            initial_balance,
            risk_per_trade,
            commission_rate,
            stop_loss_pct,
            min_confidence,
            maker_commission_rate,
            tp_multiplier,
            use_fixed_sl,
        )

    n = len(df)
    holdout_start = int(n * (1 - holdout_ratio))
    train_df = df.iloc[:holdout_start].copy()
    test_df = df.iloc[holdout_start:].copy()

    train_result = run_strategy(
        name,
        strategy_cls(),
        train_df,
        initial_balance,
        risk_per_trade,
        commission_rate,
        stop_loss_pct,
        min_confidence,
        maker_commission_rate,
        tp_multiplier,
        use_fixed_sl,
    )
    test_result = run_strategy(
        name,
        strategy_cls(),
        test_df,
        initial_balance,
        risk_per_trade,
        commission_rate,
        stop_loss_pct,
        min_confidence,
        maker_commission_rate,
        tp_multiplier,
        use_fixed_sl,
    )

    overfit_ratio = _compute_overfit_ratio(
        train_result.sharpe_ratio, test_result.sharpe_ratio
    )
    print(
        f"\n  IN-SAMPLE  (train {100-int(holdout_ratio*100)}%): "
        f"Sharpe={train_result.sharpe_ratio:.2f}  "
        f"WinRate={train_result.win_rate:.1%}  "
        f"Return={train_result.total_return_pct:.1%}"
    )
    print(
        f"  OUT-SAMPLE (test  {int(holdout_ratio*100)}%): "
        f"Sharpe={test_result.sharpe_ratio:.2f}  "
        f"WinRate={test_result.win_rate:.1%}  "
        f"Return={test_result.total_return_pct:.1%}"
    )
    if overfit_ratio is None:
        print("  Overfit ratio: n/a — no in-sample edge (train Sharpe <= 0)")
    else:
        print(f"  Overfit ratio: {overfit_ratio:.1f}x", end="")
        if overfit_ratio > 2.0:
            print("  ⚠ OVERFITTED", end="")
        print()

    return test_result


def run_strategy_rolling_walk_forward(
    name: str,
    strategy_cls: type,
    df: pd.DataFrame,
    initial_balance: float,
    risk_per_trade: float,
    commission_rate: float,
    stop_loss_pct: float,
    min_confidence: float,
    n_windows: int = 5,
    maker_commission_rate: Optional[float] = None,
    tp_multiplier: Optional[float] = None,
    use_fixed_sl: bool = False,
) -> Tuple[BacktestResult, List[Optional[float]]]:
    """
    Anchored (expanding-window) walk-forward по n_windows последовательным фолдам.

    Один статичный 80/20-сплит (run_strategy_with_holdout) даёт единственное
    число overfit_ratio, которое может оказаться случайно удачным/неудачным
    периодом. Здесь история делится на n_windows+1 последовательных фолдов;
    для i-го тестового фолда train = все данные до него (expanding, без
    заглядывания вперёд), test = сам фолд. OOS-результаты всех фолдов
    объединяются в один BacktestResult через aggregate_results, плюс
    возвращается список overfit_ratio по каждому окну (None = train Sharpe
    <= 0 в этом окне, см. _compute_overfit_ratio) — чтобы увидеть разброс,
    а не одну точку.

    :return: (объединённый OOS BacktestResult, [overfit_ratio по фолдам]).
             Если данных недостаточно — fallback на run_strategy_with_holdout
             с holdout_ratio=0.2, пустой список overfit_ratio.
    """
    n = len(df)
    if n_windows < 2 or n < _WARMUP * (n_windows + 1):
        single = run_strategy_with_holdout(
            name,
            strategy_cls,
            df,
            initial_balance,
            risk_per_trade,
            commission_rate,
            stop_loss_pct,
            min_confidence,
            holdout_ratio=0.2,
            maker_commission_rate=maker_commission_rate,
            tp_multiplier=tp_multiplier,
            use_fixed_sl=use_fixed_sl,
        )
        return single, []

    fold_size = n // (n_windows + 1)
    overfit_ratios: List[Optional[float]] = []
    test_results: List[BacktestResult] = []

    for i in range(1, n_windows + 1):
        train_end = fold_size * i
        test_end = fold_size * (i + 1) if i < n_windows else n
        train_df = df.iloc[:train_end].copy()
        test_df = df.iloc[train_end:test_end].copy()
        if len(test_df) < _WARMUP or len(train_df) < _WARMUP:
            continue

        train_result = run_strategy(
            name,
            strategy_cls(),
            train_df,
            initial_balance,
            risk_per_trade,
            commission_rate,
            stop_loss_pct,
            min_confidence,
            maker_commission_rate,
            tp_multiplier,
            use_fixed_sl,
        )
        test_result = run_strategy(
            name,
            strategy_cls(),
            test_df,
            initial_balance,
            risk_per_trade,
            commission_rate,
            stop_loss_pct,
            min_confidence,
            maker_commission_rate,
            tp_multiplier,
            use_fixed_sl,
        )
        ratio = _compute_overfit_ratio(
            train_result.sharpe_ratio, test_result.sharpe_ratio
        )
        overfit_ratios.append(ratio)
        test_results.append(test_result)
        ratio_str = "n/a" if ratio is None else f"{ratio:.1f}x"
        print(
            f"\n  Window {i}/{n_windows}: train_n={len(train_df)} test_n={len(test_df)} "
            f"train_Sharpe={train_result.sharpe_ratio:.2f} "
            f"test_Sharpe={test_result.sharpe_ratio:.2f} "
            f"overfit={ratio_str}",
            end="",
        )

    if not test_results:
        single = run_strategy_with_holdout(
            name,
            strategy_cls,
            df,
            initial_balance,
            risk_per_trade,
            commission_rate,
            stop_loss_pct,
            min_confidence,
            holdout_ratio=0.2,
            maker_commission_rate=maker_commission_rate,
            tp_multiplier=tp_multiplier,
            use_fixed_sl=use_fixed_sl,
        )
        return single, []

    combined = aggregate_results({name: test_results})[0]

    valid_ratios = [r for r in overfit_ratios if r is not None]
    no_edge_count = len(overfit_ratios) - len(valid_ratios)
    if not valid_ratios:
        print(
            f"\n  Rolling overfit ratio: n/a — no in-sample edge in any of "
            f"{len(overfit_ratios)} window(s)"
        )
    else:
        mean_of = statistics.mean(valid_ratios)
        print(f"\n  Rolling overfit ratio: mean={mean_of:.1f}x", end="")
        if len(valid_ratios) > 1:
            stdev_of = statistics.stdev(valid_ratios)
            print(
                f" stdev={stdev_of:.1f}x min={min(valid_ratios):.1f}x"
                f" max={max(valid_ratios):.1f}x",
                end="",
            )
        if no_edge_count:
            print(
                f" ({no_edge_count}/{len(overfit_ratios)} окон без in-sample edge)",
                end="",
            )
        if mean_of > 2.0:
            print("  ⚠ OVERFITTED", end="")
        print()

    return combined, overfit_ratios


# ── Агрегация по нескольким символам ─────────────────────────────────────


def aggregate_results(
    results_per_symbol: dict[str, List[BacktestResult]]
) -> List[BacktestResult]:
    """
    Объединяет BacktestResult по символам в один агрегированный результат на стратегию.

    Метрики:
      total_trades / wins / losses  → сумма по символам
      win_rate / avg_profit / avg_loss / EV → пересчёт из агрегированных счётчиков
      total_return_pct  → среднее по символам
      max_drawdown_pct  → наихудшее по символам
      sharpe_ratio      → пересчёт из объединённого списка trade_returns
    """
    strategy_buckets: dict[str, List[BacktestResult]] = {}
    for sym_results in results_per_symbol.values():
        for r in sym_results:
            strategy_buckets.setdefault(r.strategy, []).append(r)

    aggregated: List[BacktestResult] = []
    for strat, parts in strategy_buckets.items():
        total_trades = sum(p.total_trades for p in parts)
        num_wins = sum(p.num_wins for p in parts)
        num_losses = sum(p.num_losses for p in parts)
        win_rate = num_wins / total_trades if total_trades > 0 else 0.0

        all_wins = [t["return_pct"] for p in parts for t in p.trades if t["pnl"] > 0]
        all_losses = [t["return_pct"] for p in parts for t in p.trades if t["pnl"] <= 0]
        avg_profit = statistics.mean(all_wins) if all_wins else 0.0
        avg_loss = statistics.mean(all_losses) if all_losses else 0.0
        ev = win_rate * avg_profit + (1 - win_rate) * avg_loss

        avg_return = statistics.mean(p.total_return_pct for p in parts)
        max_dd = max(p.max_drawdown_pct for p in parts)

        combined_returns = [r for p in parts for r in p.trade_returns]
        sharpe = _sharpe(combined_returns)

        aggregated.append(
            BacktestResult(
                strategy=strat,
                total_trades=total_trades,
                win_rate=win_rate,
                avg_profit_pct=avg_profit,
                avg_loss_pct=avg_loss,
                expected_value=ev,
                total_return_pct=avg_return,
                max_drawdown_pct=max_dd,
                sharpe_ratio=sharpe,
                num_wins=num_wins,
                num_losses=num_losses,
                trade_returns=combined_returns,
            )
        )

    aggregated.sort(key=lambda r: r.expected_value, reverse=True)
    return aggregated


# ── Печать отчёта ────────────────────────────────────────────────────────


def print_report(
    results: List[BacktestResult],
    symbol: str,
    timeframe: str,
    months: int,
) -> None:
    sep = "=" * 68
    header = f"BACKTEST RESULTS — {symbol} {timeframe} " f"— {months} months"
    print(f"\n{sep}")
    print(header)
    print(sep)
    print(
        f"{'Rank':<5} {'Strategy':<20} {'Trades':>6}  "
        f"{'WinRate':>8}  {'EV':>7}  "
        f"{'Return':>7}  {'Drawdown':>9}  {'Sharpe':>7}"
    )
    print(
        f"{'----':<5} {'-' * 20:<20} {'------':>6}  "
        f"{'--------':>8}  {'------':>7}  "
        f"{'------':>7}  {'--------':>9}  {'------':>7}"
    )

    for rank, r in enumerate(results, 1):
        wr_str = f"{r.win_rate * 100:.1f}%"
        ev_str = f"{r.expected_value * 100:+.2f}%"
        ret_str = f"{r.total_return_pct * 100:+.1f}%"
        dd_str = f"{r.max_drawdown_pct * 100:.1f}%"
        sh_str = f"{r.sharpe_ratio:.2f}"
        print(
            f"{rank:>4}  {r.strategy:<20} {r.total_trades:>6}  "
            f"{wr_str:>8}  {ev_str:>7}  "
            f"{ret_str:>7}  {dd_str:>9}  {sh_str:>7}"
        )

    print(sep)
    if results:
        best = results[0]
        print(f"Лучшая стратегия: {best.strategy}  (наибольший EV)")
        print(f"Рекомендуемый MODE: ai с " f"DEFAULT_STRATEGY={best.strategy}")
    print(sep)


# ── Экспорт JSON ─────────────────────────────────────────────────────────


def save_json(
    results: List[BacktestResult],
    symbol: str,
    timeframe: str,
    months: int,
    output_path: str = "data/backtest_results.json",
) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    summary_list = []
    for r in results:
        summary_list.append(
            {
                "strategy": r.strategy,
                "total_trades": r.total_trades,
                "win_rate": round(r.win_rate, 4),
                "avg_profit_pct": round(r.avg_profit_pct, 6),
                "avg_loss_pct": round(r.avg_loss_pct, 6),
                "expected_value": round(r.expected_value, 6),
                "total_return_pct": round(r.total_return_pct, 6),
                "max_drawdown_pct": round(r.max_drawdown_pct, 6),
                "sharpe_ratio": round(r.sharpe_ratio, 4),
                "num_wins": r.num_wins,
                "num_losses": r.num_losses,
            }
        )

    payload = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "symbol": symbol,
        "timeframe": timeframe,
        "months": months,
        "results": summary_list,
    }

    try:
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)
        logger.info(f"Результаты сохранены в {output_path}")
    except Exception as e:
        logger.error(f"Ошибка сохранения JSON: {e}")


# ── Точка входа ──────────────────────────────────────────────────────────


async def main() -> None:
    # ── Чтение переопределений из окружения ───────────────────────
    months = int(os.getenv("BT_MONTHS", "12"))
    holdout_ratio = float(
        os.getenv("BT_HOLDOUT_RATIO", str(Config.BACKTEST_HOLDOUT_RATIO))
    )
    # BT_ROLLING_WINDOWS>=2 → anchored walk-forward по N фолдам вместо
    # единственного 80/20-сплита (см. run_strategy_rolling_walk_forward).
    # По умолчанию 0 — старое поведение (run_strategy_with_holdout).
    rolling_windows = int(os.getenv("BT_ROLLING_WINDOWS", "0"))
    min_conf = float(os.getenv("BT_MIN_CONF", str(Config.MIN_SIGNAL_CONFIDENCE)))
    # TP закрывается лимит-ордером (мейкер) в живом боте (bybit_api.py:431),
    # вход и SL — market/stop_market (тейкер, Config.COMMISSION_RATE).
    # Дефолт 0.0002 — типовая базовая мейкер-комиссия Bybit linear-фьючерсов;
    # сверьте с реальным тарифом аккаунта при необходимости.
    maker_commission_rate = float(os.getenv("BT_MAKER_COMMISSION_RATE", "0.0002"))

    # BT_PROFILE=bluechip|altcoin|meme — подставляет таймфрейм, пул символов
    # и SL%/TP-множитель ровно из _MARKET_PROFILES (runtime_config.py), как
    # в реальном боте. Без BT_PROFILE — старое поведение (ATR-based SL, RR=2.0).
    profile = os.getenv("BT_PROFILE", "").strip().lower() or None
    if profile and profile not in _MARKET_PROFILES:
        print(f"[backtest] Неизвестный BT_PROFILE={profile!r} — игнорирую")
        profile = None

    if profile:
        # _MARKET_PROFILES — гетерогенный dict (str/int/float вперемешку),
        # поэтому mypy видит значения как object — приводим явно.
        pcfg: Dict[str, Any] = _MARKET_PROFILES[profile]
        timeframe = os.getenv("BT_TIMEFRAME", str(pcfg["timeframe"]))
        stop_loss_pct = float(pcfg["sl_percent"])
        tp_multiplier: Optional[float] = float(pcfg["tp_multiplier"])
        risk_per_trade = float(pcfg["risk_per_trade"])
        use_fixed_sl = True
        default_top_n = int(pcfg["scan_top_n"])
    else:
        timeframe = os.getenv("BT_TIMEFRAME", Config.TIMEFRAME)
        stop_loss_pct = Config.STOP_LOSS_PERCENT
        tp_multiplier = None
        risk_per_trade = Config.RISK_PER_TRADE
        use_fixed_sl = False
        default_top_n = Config.SCAN_TOP_N

    # Определение символов:
    #   BT_SYMBOLS=A,B,C  → явный список (приоритет над профилем)
    #   BT_SYMBOL=A       → один явный символ
    #   BT_PROFILE=...    → пул профиля (bluechip/altcoin/meme), как в проде
    #   (ничего)          → автоматически топ BT_TOP_N по объёму за 24ч
    top_n = int(os.getenv("BT_TOP_N", str(default_top_n)))
    raw_syms = os.getenv("BT_SYMBOLS") or os.getenv("BT_SYMBOL")
    symbol_data: "Dict[str, pd.DataFrame]"
    if raw_syms:
        # Явный список — берём как есть, без добора (пользователь указал точно).
        explicit_symbols = [s.strip() for s in raw_syms.split(",") if s.strip()]
        symbol_data = await _resolve_symbols_with_data(
            explicit_symbols, len(explicit_symbols), timeframe, months
        )
    else:
        pool_n = top_n * _POOL_MULTIPLIER
        if profile:
            print(
                f"[backtest] Резолвинг пула символов профиля {profile!r} "
                f"(цель {top_n}, пул кандидатов {pool_n})..."
            )
            candidates = await _fetch_profile_symbols(profile, pool_n)
        else:
            print(
                f"[backtest] Auto-fetching top {top_n} by 24h volume "
                f"(пул кандидатов {pool_n})..."
            )
            candidates = await _fetch_top_symbols(pool_n)
        symbol_data = await _resolve_symbols_with_data(
            candidates, top_n, timeframe, months
        )

    symbols = list(symbol_data.keys())
    if not symbols:
        print("ОШИБКА: не удалось набрать ни одной монеты с данными — прерываем")
        return

    multi = len(symbols) > 1
    print(
        f"[backtest] profile={profile or '-'} | {len(symbols)} symbol(s) | "
        f"{timeframe} | {months}m | min_conf={min_conf}"
    )
    print(f"[backtest] symbols: {', '.join(symbols)}")

    import json as _json

    results_per_symbol: dict[str, List[BacktestResult]] = {}
    _total_steps = len(symbols) * len(STRATEGY_REGISTRY)
    _done_steps = 0
    print(
        "BACKTEST_PROGRESS:"
        + _json.dumps(
            {
                "pct": 0.0,
                "step": 0,
                "total": _total_steps,
                "symbol": "",
                "strategy": "",
            }
        ),
        flush=True,
    )

    for symbol in symbols:
        print(f"\n{'─' * 50}")
        print(f"  {symbol}")
        print(f"{'─' * 50}")

        df = symbol_data[symbol]  # уже загружено в _resolve_symbols_with_data
        print(f"[backtest] {len(df)} свечей загружено")

        sym_results: List[BacktestResult] = []
        for strat_name, strat_cls in STRATEGY_REGISTRY.items():
            print(f"\n[{strat_name}] ", end="", flush=True)
            if rolling_windows >= 2:
                result, _ = run_strategy_rolling_walk_forward(
                    name=strat_name,
                    strategy_cls=strat_cls,
                    df=df,
                    initial_balance=Config.INITIAL_BALANCE,
                    risk_per_trade=risk_per_trade,
                    commission_rate=Config.COMMISSION_RATE,
                    stop_loss_pct=stop_loss_pct,
                    min_confidence=min_conf,
                    n_windows=rolling_windows,
                    maker_commission_rate=maker_commission_rate,
                    tp_multiplier=tp_multiplier,
                    use_fixed_sl=use_fixed_sl,
                )
            else:
                result = run_strategy_with_holdout(
                    name=strat_name,
                    strategy_cls=strat_cls,
                    df=df,
                    initial_balance=Config.INITIAL_BALANCE,
                    risk_per_trade=risk_per_trade,
                    commission_rate=Config.COMMISSION_RATE,
                    stop_loss_pct=stop_loss_pct,
                    min_confidence=min_conf,
                    holdout_ratio=holdout_ratio,
                    maker_commission_rate=maker_commission_rate,
                    tp_multiplier=tp_multiplier,
                    use_fixed_sl=use_fixed_sl,
                )
            sym_results.append(result)
            _done_steps += 1
            _pct = round(_done_steps / _total_steps * 100, 1) if _total_steps else 100.0
            print(
                f" done — {result.total_trades} trades,"
                f" EV={result.expected_value * 100:+.2f}%"
            )
            print(
                "BACKTEST_PROGRESS:"
                + _json.dumps(
                    {
                        "pct": _pct,
                        "step": _done_steps,
                        "total": _total_steps,
                        "symbol": symbol,
                        "strategy": strat_name,
                    }
                ),
                flush=True,
            )

        sym_results.sort(key=lambda r: r.expected_value, reverse=True)
        results_per_symbol[symbol] = sym_results

        if not multi:
            print_report(sym_results, symbol, timeframe, months)

    if not results_per_symbol:
        print("ОШИБКА: нет результатов — прерываем")
        return

    # ── Сводная таблица по каждому символу (режим нескольких символов) ──────
    if multi:
        sep = "=" * 68
        for symbol, sym_results in results_per_symbol.items():
            print_report(sym_results, symbol, timeframe, months)

        # ── Агрегированный отчёт по всем символам ────────────────
        agg = aggregate_results(results_per_symbol)
        print(f"\n{sep}")
        print(f"AGGREGATED RESULTS — {len(symbols)} symbols — {timeframe} — {months}m")
        print(sep)
        print(
            f"{'Rank':<5} {'Strategy':<20} {'Trades':>6}  "
            f"{'WinRate':>8}  {'EV':>7}  {'AvgReturn':>9}  {'MaxDD':>7}  {'Sharpe':>7}"
        )
        print(
            f"{'----':<5} {'-'*20:<20} {'------':>6}  {'--------':>8}  {'------':>7}"
            f"  {'--------':>9}  {'------':>7}  {'------':>7}"
        )
        for rank, r in enumerate(agg, 1):
            print(
                f"{rank:>4}  {r.strategy:<20} {r.total_trades:>6}  "
                f"{r.win_rate*100:>7.1f}%  {r.expected_value*100:>+7.2f}%  "
                f"{r.total_return_pct*100:>+8.1f}%"
                f"  {r.max_drawdown_pct*100:>6.1f}%  {r.sharpe_ratio:>7.2f}"
            )
        print(sep)
        if agg:
            best = agg[0]
            print(
                f"Лучшая стратегия по всем символам: {best.strategy}"
                f"  (EV={best.expected_value*100:+.2f}%)"
            )
        print(sep)

        save_json(agg, ",".join(symbols), timeframe, months)
    else:
        save_json(results_per_symbol[symbols[0]], symbols[0], timeframe, months)

    # ── Тесты значимости альфы (агрегированные или по одному символу) ────────
    final_results = (
        aggregate_results(results_per_symbol)
        if multi
        else results_per_symbol[symbols[0]]
    )
    tester = AlphaTester()
    sep = "=" * 68
    print(f"\n{sep}")
    print("ЗНАЧИМОСТЬ АЛЬФЫ  (бутстрап Sharpe + знаковый ранговый тест Wilcoxon)")
    if multi:
        print("(на основе агрегированных доходностей сделок по всем символам)")
    print(sep)
    # Bonferroni-поправка на множественное тестирование: если проверяется
    # N стратегий на одних и тех же данных, часть из них пройдёт p<0.05
    # просто по теории вероятностей (data dredging), даже без реального
    # edge. Скорректированный порог = 0.05/N — жёстче, чем raw-порог у
    # каждого отдельного AlphaTester.test() (тот остаётся как есть — он
    # корректен для одиночного использования вне пакетного прогона).
    n_tests = len(final_results)
    bonferroni_alpha = 0.05 / n_tests if n_tests > 0 else 0.05
    sig_count = 0
    sig_count_corrected = 0
    for r in final_results:
        alpha = tester.test(r.trade_returns, name=r.strategy)
        print(alpha.summary())
        if alpha.is_significant:
            sig_count += 1
        if alpha.sharpe_pvalue < bonferroni_alpha:
            sig_count_corrected += 1
    print(sep)
    print(
        f"{sig_count}/{n_tests} стратегий показывают статистически "
        f"значимую альфу при p<5% (БЕЗ поправки на множественное тестирование)"
    )
    print(
        f"{sig_count_corrected}/{n_tests} проходят после Bonferroni-поправки "
        f"(p<{bonferroni_alpha:.4f}, скорректировано на {n_tests} тестов)"
    )
    if sig_count_corrected < sig_count:
        print(
            f"  ⚠ {sig_count - sig_count_corrected} стратегия(й) значимы только "
            f"без поправки — вероятно, ложные срабатывания от перебора {n_tests} "
            f"вариантов на одних данных, а не реальный edge"
        )
    print(sep)


if __name__ == "__main__":
    asyncio.run(main())
