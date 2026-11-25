"""Скан рынка (узел «Сканер»): монеты по профилю, свечи, настроение, снимки и их качество.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import TYPE_CHECKING, Optional

from bitbot.workers.steps.context import CycleContext

if TYPE_CHECKING:
    from bitbot.workers.bot import TradingBot

logger = logging.getLogger("bitbot.workers.bot")


async def run(bot: "TradingBot", ctx: CycleContext) -> Optional[CycleContext]:
    _profile = bot._runtime_config.get_market_profile()
    _scan_n = bot._runtime_config.get_scan_top_n()
    _refill_kwargs: dict = {}
    if _profile == "bluechip":
        symbols = await bot.scanner.get_top_symbols(
            n=_scan_n,
        )
    elif _profile == "altcoin":
        # Динамически определяем текущий топ bluechip по объёму.
        # n=_scan_n (не хардкод 20!) — иначе при scan_top_n>20
        # монеты #21.._scan_n остаются доступны и bluechip, и
        # altcoin одновременно (реальное пересечение пулов).
        # buffer=0: этот список нужен только для исключения, не для загрузки
        # force_non_meme=True: мем-коины всё равно исключены отдельным
        # фильтром (is_meme=False вне meme-профиля), но фиксируем явно —
        # этот снапшот должен вести себя одинаково независимо от того,
        # из какого профиля его вызвали.
        _bc_syms = await bot.scanner.get_top_symbols(
            n=_scan_n, buffer=0, force_non_meme=True
        )
        _bc_bases = frozenset(s.split("/")[0] for s in _bc_syms)
        _refill_kwargs = {"altcoin_exclude_bases": _bc_bases}
        symbols = await bot.scanner.get_top_symbols(
            n=_scan_n,
            altcoin_exclude_bases=_bc_bases,
        )
    elif _profile == "meme":
        # Meme должен исключать и текущий bluechip, и текущий altcoin —
        # иначе одна и та же крупная волатильная монета (не мем-коин,
        # например AVAX) могла попасть одновременно в altcoin и meme.
        # Сами мем-коины (DOGE/SHIB/...) тут не теряются: они и так
        # исключены из _bc_syms/_ac_syms через force_non_meme=True
        # (is_meme=False для этих двух снапшотов), поэтому не попадут
        # в exclude-набор ниже.
        _bc_syms = await bot.scanner.get_top_symbols(
            n=_scan_n, buffer=0, force_non_meme=True
        )
        _bc_bases = frozenset(s.split("/")[0] for s in _bc_syms)
        _ac_syms = await bot.scanner.get_top_symbols(
            n=_scan_n,
            buffer=0,
            altcoin_exclude_bases=_bc_bases,
            force_non_meme=True,
        )
        _ac_bases = frozenset(s.split("/")[0] for s in _ac_syms)
        _exclude_bases = _bc_bases | _ac_bases
        _refill_kwargs = {"altcoin_exclude_bases": _exclude_bases}
        symbols = await bot.scanner.get_top_symbols(
            n=_scan_n,
            altcoin_exclude_bases=_exclude_bases,
        )
    else:
        symbols = await bot.scanner.get_top_symbols(n=_scan_n)
    # refill_kwargs: если после scan_all не хватило _scan_n монет с
    # валидными данными (молодой листинг, делистинг, сбой API) —
    # добираем следующими по рейтингу, а не смиряемся с недобором.
    try:
        bot.scanner.reset_quality()
    except Exception:  # статусы качества не должны ломать цикл
        pass
    market_data = await bot._scan_and_update_correlations(
        symbols, target_n=_scan_n, refill_kwargs=_refill_kwargs
    )
    bot._runtime_config.set_current_symbols(_profile, list(market_data.keys()))
    try:
        await bot._sync_sentiment_poller(_profile, list(market_data.keys()))
    except Exception as e:  # соцсети не должны срывать торговый цикл
        logger.warning("Social sentiment sync failed: %s", e)
    await bot._inject_sentiment(market_data)
    snapshots = await bot._cycle.collect_snapshots(
        list(market_data.keys()), market_data
    )
    bad_data = bot._record_data_quality(snapshots)
    bot._publish_event(
        "radar.stage",
        {
            "stage": "scan",
            "symbols": [s.get("symbol") for s in snapshots],
            "bad_data": bad_data,
        },
    )
    balance = await bot._get_balance_usdt()
    return replace(
        ctx,
        profile=_profile,
        market_data=market_data,
        snapshots=snapshots,
        balance=balance,
    )
