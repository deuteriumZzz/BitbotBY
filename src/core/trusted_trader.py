"""
Сделки по постам доверенных Telegram-каналов — отдельный путь мимо конвейера.

Пост доверенного канала → разбор (монета, направление, вход/стоп/тейк) →
ордер через OrderExecutor с пометкой trusted, за секунды и прямо из события.
Сканер, стратегии, AI, пороги уверенности, market_context и фильтры скана не
участвуют и не могут отклонить вход; динамические выходы PositionMonitor такие
позиции не закрывают — только SL, TP и трейлинг.

Остаются защиты (не детекторы): стоп всегда (из поста или по умолчанию), свой
риск на сделку и свои слоты, пауза и дневной лимит убытка, спред и ликвидность,
одна сделка на монету за N минут (репосты, несколько каналов), режим счёта
как есть — Paper остаётся Paper.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    List,
    NamedTuple,
    Optional,
    Pattern,
    Tuple,
    Union,
)

from src.core.types import TRUSTED_PREFIX, is_trusted
from src.sentiment.telegram_sentiment import Post, channel_names, mentions

logger = logging.getLogger(__name__)

LAST_KEY = "bot:tg:last_trusted"  # последний разобранный пост — для приложения
UNTRUSTED_KEY = "bot:tg:untrusted"  # канал → id сделки, после которой снято доверие
_STALE_S = 120.0  # пост старше — пришёл с опозданием (обрыв связи): не входим
_MAX_SL_DIST = 0.5  # стоп дальше 50% от цены — почти наверняка ошибка разбора

# ── Разбор поста (чистые функции) ─────────────────────────────────────────────

Level = Tuple[float, bool]  # (значение, это проценты от цены); 0 — в посте нет


class Signal(NamedTuple):
    symbol: str
    side: str  # buy | sell
    entry: float  # 0 — в посте нет; входим по рынку в любом случае
    sl: Level
    tp: Level


_LONG = re.compile(r"\b(?:long|buy|buying|лонг\w*|покуп\w*)\b|🟢|📈", re.I)
_SHORT = re.compile(r"\b(?:short|sell|selling|шорт\w*|прода\w*)\b|🔴|📉", re.I)
# Отчёты о сделке («TP1 hit ✅», «закрываем лонг») — не вход.
_NOT_ENTRY = re.compile(
    r"\b(?:close[ds]?|closing|exit(?:ed)?|hit|reached|cancel\w*|"
    r"закры\w*|фиксир\w*|отмен\w*|достигнут\w*|взят\w*)\b|✅",
    re.I,
)
_NUM = r"\$?\s?(\d+(?:[.,]\d+)?)\s?(%)?"
_SL = re.compile(
    rf"(?:\bsl\b|stop[\s-]?loss|\bstop\b|стоп(?:[\s-]?лосс)?)\W{{0,3}}{_NUM}", re.I
)
_TP = re.compile(
    rf"(?:\btp\d?\b|take[\s-]?profit|targets?|цел[ьи]|тейк(?:[\s-]?профит)?)"
    rf"\W{{0,3}}{_NUM}",
    re.I,
)
_ENTRY = re.compile(
    rf"(?:entry|вход|buy[\s-]?zone|zone|по\s?цене|@)\W{{0,3}}{_NUM}", re.I
)
_TAG = re.compile(r"(?<![\w$#])[$#]([A-Za-z][A-Za-z0-9]{1,14})\b")


def _num(raw: str) -> float:
    if re.fullmatch(r"\d{1,3}(?:,\d{3})+", raw):  # 65,000 — разделитель тысяч
        raw = raw.replace(",", "")
    return float(raw.replace(",", "."))


def _level(rx: Pattern[str], text: str) -> Level:
    m = rx.search(text)
    return (_num(m.group(1)), bool(m.group(2))) if m else (0.0, False)


def tradable_map(markets: Dict[str, Any], market_type: str) -> Dict[str, str]:
    """База монеты → символ, которым торгует бот ('PEPE' → '1000PEPE/USDT')."""
    out: Dict[str, str] = {}
    for m in markets.values():
        if m.get("quote") != "USDT" or m.get("active") is False:
            continue
        if not (m.get("spot") if market_type == "spot" else m.get("linear")):
            continue
        base = str(m.get("base") or "").upper()
        if not base:
            continue
        out[base] = f"{base}/USDT"
        short = re.fullmatch(r"10+([A-Z].*)", base)
        if short:
            out.setdefault(short.group(1), out[base])
    return out


def parse_post(text: str, tradable: Dict[str, str]) -> Union[Signal, str]:
    """Сигнал из поста или причина, почему сделки по нему не будет."""
    if _NOT_ENTRY.search(text):
        return "пост о закрытии или итогах сделки, не о входе"
    long_, short = bool(_LONG.search(text)), bool(_SHORT.search(text))
    if long_ == short:
        return "в посте и лонг, и шорт" if long_ else "в посте нет направления"
    found = [b for b in tradable if mentions(text, b)]
    marked = [b for b in found if mentions(text, b, marked_only=True)]
    symbols = sorted({tradable[b] for b in (marked or found)})
    if not symbols:
        tags = [t.upper() for t in _TAG.findall(text)]
        if tags:
            return f"{tags[0]} не торгуется на Bybit"
        return "в посте нет монеты, которой торгует бот"
    if len(symbols) > 1:
        names = ", ".join(s.split("/")[0] for s in symbols)
        return f"в посте несколько монет: {names}"
    entry = _level(_ENTRY, text)[0]
    return Signal(
        symbols[0],
        "buy" if long_ else "sell",
        entry,
        _level(_SL, text),
        _level(_TP, text),
    )


def levels(sig: Signal, price: float, default_sl: float) -> Tuple[float, float, str]:
    """(стоп, тейк, причина отказа). Без стопа в посте — default_sl от цены.

    Тейк без цели в посте — в 2 раза дальше стопа.
    """
    buy = sig.side == "buy"
    sign = 1 if buy else -1

    def price_of(level: Level) -> float:
        value, pct = level
        return price * (1 - sign * value / 100) if pct else value

    sl = price_of(sig.sl) if sig.sl[0] else price * (1 - sign * default_sl)
    if (sl >= price) if buy else (sl <= price):
        return 0.0, 0.0, "цена уже за стопом из поста"
    if abs(price - sl) / price > _MAX_SL_DIST:
        return 0.0, 0.0, "стоп в посте дальше 50% от цены — похоже, не то число"
    if sig.tp[0]:
        value, pct = sig.tp
        tp = price * (1 + sign * value / 100) if pct else value
    else:
        tp = price + 2 * (price - sl)
    if (tp <= price) if buy else (tp >= price):
        return 0.0, 0.0, "цена уже прошла цель из поста"
    return sl, tp, ""


# ── Исполнение ────────────────────────────────────────────────────────────────


def _step(node: str, ok: Optional[bool], text: str) -> Dict[str, Any]:
    return {"node": node, "ok": ok, "text": text}


class TrustedTrader:
    """Слушает живые посты (TelegramSentiment.on_post) и входит по доверенным."""

    def __init__(
        self,
        *,
        api: Any,
        executor: Any,
        runtime_config: Any,
        trade_history: Any,
        telegram: Any,
        get_balance: Callable[[], Awaitable[float]],
        get_monitored: Callable[[], Dict[str, Any]],
        blocked: Callable[[], str],
        emit: Callable[[str, Dict[str, Any]], None],
        redis: Any = None,
    ) -> None:
        self._api = api
        self._executor = executor
        self._rc = runtime_config
        self._history = trade_history
        self._telegram = telegram
        self._get_balance = get_balance
        self._get_monitored = get_monitored
        self._blocked = blocked
        self._emit = emit
        self._r = redis  # сырой redis-клиент; None — без истории для приложения
        self._recent: Dict[str, Tuple[float, str]] = {}  # символ → (когда, канал)

    def _trusted(self) -> List[str]:
        return channel_names(self._rc.hot("TELEGRAM_TRUSTED_CHANNELS"))

    async def on_post(self, post: Post) -> None:
        """Живой пост любого канала; для доверенных — разбор и вход."""
        if not self._rc.hot("TELEGRAM_TRUSTED_TRADING"):
            return
        if post.channel not in self._trusted():
            return
        if time.time() - post.at > _STALE_S:
            self._done(post, None, "пост пришёл с опозданием — не входим")
            return
        markets = getattr(self._api.exchange, "markets", None) or {}
        sig = parse_post(post.text, tradable_map(markets, self._rc.get_market_type()))
        if isinstance(sig, str):
            self._done(post, None, sig)
            return
        try:
            reason = await self._open(post, sig)
        except Exception as e:
            logger.exception("Trusted post @%s: order failed", post.channel)
            reason = f"ошибка при входе: {type(e).__name__}"
        self._done(post, sig, reason)

    async def _open(self, post: Post, sig: Signal) -> str:
        rc = self._rc
        if rc.is_paused():
            return "бот на паузе"
        blocked = self._blocked()
        if blocked:
            return blocked
        if rc.get_market_type() == "spot" and sig.side == "sell":
            return "на споте шорт не открыть"
        now = time.time()
        last = self._recent.get(sig.symbol)
        dedup = float(rc.hot("TELEGRAM_TRUSTED_DEDUP_MIN") or 30) * 60
        if last and now - last[0] < dedup:
            ago = max(1, round((now - last[0]) / 60))
            return f"по монете уже входили {ago} мин назад (канал {last[1]})"
        monitored = self._get_monitored()
        slots = int(rc.hot("TELEGRAM_TRUSTED_SLOTS") or 1)
        used = sum(1 for p in monitored.values() if is_trusted(p))
        if used >= slots:
            return f"слоты для каналов заняты ({used}/{slots})"
        ticker = await self._api.exchange.fetch_ticker(sig.symbol)
        price = float(
            ticker.get("ask" if sig.side == "buy" else "bid") or ticker.get("last") or 0
        )
        if price <= 0:
            return "биржа не дала цену"
        default_sl = float(rc.hot("TELEGRAM_TRUSTED_SL") or 0.03)
        sl, tp, why = levels(sig, price, default_sl)
        if why:
            return why
        balance = await self._get_balance()
        risk = float(rc.hot("TELEGRAM_TRUSTED_RISK") or 0.005)
        # ponytail: объём не больше баланса (плечо 1×) — свой риск на сделку
        # и так мал; крупнее — когда понадобится плечо для сделок по каналам.
        qty = min(balance * risk / abs(price - sl), balance / price)
        delay = time.time() - post.at
        src = "из поста" if sig.sl[0] else f"по умолчанию {default_sl:.1%}"
        top = {
            "symbol": sig.symbol,
            "action": sig.side,
            "entry": price,
            "stop_loss": sl,
            "take_profit": tp,
            "quantity": qty,
            "strategy": f"{TRUSTED_PREFIX}{post.channel}",
            "confidence": 1.0,
            "reasoning": f"Пост @{post.channel}: {post.text[:300]}",
            "trusted": True,
            "max_slots": slots,
            "context_flags": json.dumps(
                {"source": post.channel, "post_at": post.at, "post_entry": sig.entry}
            ),
            "trace": [
                _step("scan", True, f"пост в канале @{post.channel}"),
                _step("strat", None, "доверенный канал: стратегии не участвуют"),
                _step("ai", None, "не участвует"),
                _step("sac", None, "не участвует"),
                _step(
                    "risk",
                    True,
                    f"слот для каналов {used + 1}/{slots} · риск {risk:.1%} · "
                    f"стоп {src}",
                ),
                _step(
                    "order", True, f"ордер отправлен через {delay:.1f} с после поста"
                ),
            ],
        }
        self._recent[sig.symbol] = (now, post.channel)
        await self._executor.execute(top, {}, balance)
        pos = monitored.get(sig.symbol)
        if pos and pos.get("strategy") == top["strategy"]:
            return ""
        self._recent.pop(sig.symbol, None)
        return self._executor.last_reject.pop(sig.symbol, "") or "ордер не открыт"

    def _done(self, post: Post, sig: Optional[Signal], reason: str) -> None:
        delay = time.time() - post.at
        what = f"{sig.symbol} {'LONG' if sig.side == 'buy' else 'SHORT'}" if sig else ""
        if reason:
            logger.info("Trusted post @%s %s: skipped — %s", post.channel, what, reason)
        else:
            logger.info(
                "Trusted post @%s → %s opened, post→order %.1f s",
                post.channel,
                what,
                delay,
            )
        event = {
            "channel": post.channel,
            "symbol": sig.symbol if sig else "",
            "side": sig.side if sig else "",
            "ok": not reason,
            "reason": reason,
            "latency_s": round(delay, 1),
            "at": time.time(),
            "text": post.text[:140],
        }
        self._emit("trusted.post", event)
        if self._r is not None:
            try:
                self._r.set(LAST_KEY, json.dumps(event, ensure_ascii=False))
            except Exception as e:
                logger.debug("last trusted post not stored: %s", e)

    async def review(self) -> None:
        """Каждый цикл: снять доверие после N убыточных сделок подряд."""
        n = int(self._rc.hot("TELEGRAM_TRUSTED_UNTRUST") or 0)
        trusted = self._trusted()
        if n <= 0 or not trusted or self._r is None:
            return
        for ch in trusted:
            after = int(self._r.hget(UNTRUSTED_KEY, ch) or 0)
            rows = await self._history.recent_results(f"{TRUSTED_PREFIX}{ch}", n, after)
            if len(rows) < n or any(pnl >= 0 for _, pnl in rows):
                continue
            left = [c for c in self._trusted() if c != ch]
            self._rc.set_hot("TELEGRAM_TRUSTED_CHANNELS", ",".join(left))
            self._r.hset(UNTRUSTED_KEY, ch, rows[0][0])
            logger.warning("Channel @%s untrusted after %d losses in a row", ch, n)
            self._emit("config.changed", {"keys": ["TELEGRAM_TRUSTED_CHANNELS"]})
            await self._telegram.notify(
                f"🔕 Канал @{ch} больше не доверенный: {n} убыточных сделок "
                "подряд. Вернуть можно в приложении: Настройки → Каналы Telegram.",
                parse_mode="HTML",  # в именах каналов бывает «_»
            )
