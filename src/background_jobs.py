"""Фоновые задачи бота, которые запускаются из приложения.

Те же скрипты, что и в Telegram-меню (бэктест, обучение SAC, тюнинг, отчёты),
отдельными процессами: бот продолжает торговать. Прогресс уходит событиями
`job.progress`, итог — событием `job.done`; короткое уведомление приходит и в Telegram.

Telegram и приложение делят замок `RuntimeConfig.try_start_job`: одну и ту же
группу задач (например, обучение SAC) нельзя запустить дважды, откуда бы
ни пришёл запуск.
"""

import asyncio
import json
import logging
import os
import shutil
import sys
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from src.bot_bus import emit, error

logger = logging.getLogger(__name__)

REPORT_LIMIT = 3500  # столько хвоста отчёта показываем (как в Telegram)


@dataclass(frozen=True)
class JobSpec:
    id: str
    group: str  # задачи одной группы не идут одновременно
    title: str
    ttl: int  # страховка замка на случай падения бота, секунды


JOBS: Dict[str, JobSpec] = {
    s.id: s
    for s in (
        JobSpec("backtest", "backtest", "Бэктест стратегий", 3 * 3600),
        JobSpec("train", "sac", "Обучение SAC", 4 * 3600),
        JobSpec("tune", "sac", "Тюнинг и обучение SAC", 8 * 3600),
        JobSpec("alpha_report", "alpha_report", "Значимость альфы", 1800),
        JobSpec("live_divergence", "live_divergence", "Live vs бэктест", 1800),
        JobSpec("ai_isolation", "ai_isolation", "AI без индикаторов", 3600),
    )
}
GROUPS = sorted({j.group for j in JOBS.values()})

_REPORT_SCRIPTS = {
    "alpha_report": "paper_alpha_report.py",
    "live_divergence": "backtest_live_divergence.py",
    "ai_isolation": "ai_signal_isolation_backtest.py",
}
_PROFILE_EXPERIENCES = {
    "altcoin": "data/experiences_altcoin.jsonl",
    "meme": "data/experiences_meme.jsonl",
}


def status(rc: Any) -> Dict[str, Any]:
    """Что идёт сейчас, прогресс и можно ли откатить модель (читает Redis)."""
    backup = rc.get_sac_backup_path()
    return {
        "running": rc.running_jobs(GROUPS),
        "progress": {
            "backtest": rc.get_backtest_progress(),
            "train": rc.get_train_progress(),
            "tune": rc.get_tune_progress(),
        },
        "rollback_available": bool(backup and os.path.exists(backup)),
    }


class JobRunner:
    def __init__(self, bot: Any) -> None:
        self._bot = bot
        self._tasks: Dict[str, "asyncio.Task[None]"] = {}

    @property
    def _r(self) -> Any:
        return self._bot.redis.redis_client

    @property
    def _rc(self) -> Any:
        return self._bot._runtime_config

    # ── Запуск ────────────────────────────────────────────────────────────────

    def start(self, job: str) -> Dict[str, Any]:
        spec = JOBS.get(job)
        if spec is None:
            return error("unknown_job", "Такой задачи нет.")
        task = self._tasks.get(spec.group)
        if (task and not task.done()) or not self._rc.try_start_job(
            spec.group, "app", spec.ttl
        ):
            return error("busy", f"«{spec.title}» уже выполняется — дождитесь конца.")
        self._tasks[spec.group] = asyncio.create_task(self._run(spec))
        return {"started": True, "job": job}

    def rollback(self) -> Dict[str, Any]:
        """Возвращает предыдущую модель SAC из резервной копии."""
        backup = self._rc.get_sac_backup_path()
        model = self._rc.get_sac_model_path()
        if not backup or not os.path.exists(backup):
            return error("no_backup", "Резервная копия модели не найдена.")
        if self._rc.running_jobs(["sac"]):
            return error("busy", "Идёт обучение — откат подождёт его конца.")
        shutil.copy2(backup, model)
        emit(self._r, "model.rolled_back", {})
        return {"rolled_back": True}

    # ── Выполнение ────────────────────────────────────────────────────────────

    async def _run(self, spec: JobSpec) -> None:
        emit(self._r, "job.progress", {"job": spec.id, "pct": 0})
        try:
            if spec.id == "backtest":
                result = await self._backtest()
            elif spec.id == "train":
                result = await self._train_stage(self._train_env())
            elif spec.id == "tune":
                result = await self._tune()
            else:
                result = await self._report(spec.id)
        except Exception as exc:
            logger.exception("job %s failed", spec.id)
            result = {"ok": False, "error": str(exc)[:200]}
        finally:
            self._rc.end_job(spec.group)
            self._clear_progress(spec.id)
        emit(self._r, "job.done", {"job": spec.id, **result})
        await self._notify(spec, result)

    def _clear_progress(self, job: str) -> None:
        rc = self._rc
        if job == "backtest":
            rc.clear_backtest_progress()
        else:
            rc.clear_train_progress()
            rc.clear_tune_progress()

    async def _notify(self, spec: JobSpec, result: Dict[str, Any]) -> None:
        try:
            if result.get("ok"):
                text = f"✅ {spec.title}: готово (запущено из приложения)."
                if result.get("worse"):
                    text += " Модель хуже рынка — рекомендуем откат."
            else:
                text = f"❌ {spec.title}: {result.get('error', 'ошибка')}"
            await self._bot.telegram.notify(text)
        except Exception:
            logger.debug("job notification failed", exc_info=True)

    async def _exec(
        self,
        script: str,
        env: Dict[str, str],
        prefix: str = "",
        on_line: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Tuple[int, str, str]:
        """Запускает скрипт; строки `PREFIX:{json}` отдаёт в on_line."""
        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            script,
            env={**os.environ, "PYTHONPATH": os.getcwd(), **env},
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        lines: List[str] = []

        async def read_out() -> None:
            assert proc.stdout
            async for raw in proc.stdout:
                line = raw.decode(errors="replace").rstrip()
                lines.append(line)
                if prefix and on_line and line.startswith(prefix):
                    try:
                        on_line(json.loads(line[len(prefix) :]))
                    except ValueError:
                        pass

        async def read_err() -> bytes:
            assert proc.stderr
            return await proc.stderr.read()

        _, err = await asyncio.gather(read_out(), read_err())
        await proc.wait()
        return (
            proc.returncode or 0,
            "\n".join(lines),
            err.decode(errors="replace")[-300:],
        )

    # ── Задачи ────────────────────────────────────────────────────────────────

    async def _backtest(self) -> Dict[str, Any]:
        def progress(info: Dict[str, Any]) -> None:
            self._rc.set_backtest_progress(info)
            emit(self._r, "job.progress", {"job": "backtest", **info})

        code, _, err = await self._exec(
            "backtest.py", {}, "BACKTEST_PROGRESS:", progress
        )
        if code != 0:
            return {"ok": False, "error": err or "бэктест упал"}
        try:
            with open(
                os.path.join("data", "backtest_results.json"), encoding="utf-8"
            ) as f:
                data = json.load(f)
        except (OSError, ValueError):
            return {"ok": True, "n": 0}
        results = data.get("results", [])
        best = max(results, key=lambda r: r.get("sharpe_ratio", -9), default=None)
        return {
            "ok": True,
            "n": len(results),
            "months": data.get("months"),
            "timeframe": data.get("timeframe"),
            "best": best
            and {
                "strategy": best["strategy"],
                "sharpe": round(best["sharpe_ratio"], 2),
                "trades": best["total_trades"],
            },
        }

    def _train_env(self) -> Dict[str, str]:
        """Окружение обучения, как в Telegram: набор монет и таймфрейм из профиля."""
        rc = self._rc
        profile = rc.get_market_profile()
        cfg = rc.get_market_profile_config(profile) if profile else {}
        return {
            "SAC_MODEL_PATH": rc.get_sac_model_path(),
            "TRAIN_TOP_N": str(cfg.get("train_top_n", 20)),
            "BT_TIMEFRAME": str(cfg.get("timeframe", "15m")),
            "SAC_PROFILE": profile or "",
            "EXPERIENCES_PATH": _PROFILE_EXPERIENCES.get(
                profile or "", "data/experiences.jsonl"
            ),
        }

    async def _train_stage(self, env: Dict[str, str]) -> Dict[str, Any]:
        # Импорт здесь: telegram_commander тянет библиотеку Telegram, нужна только тут.
        from src.telegram_commander import (  # noqa: PLC0415
            _is_worse,
            _parse_train_result,
        )

        self._rc.clear_train_progress()

        def progress(info: Dict[str, Any]) -> None:
            self._rc.set_train_progress(info)
            emit(self._r, "job.progress", {"job": "train", **info})

        code, out, err = await self._exec(
            "reinforcement_learning/train_sac.py", env, "TRAIN_PROGRESS:", progress
        )
        model = env["SAC_MODEL_PATH"]
        if code != 0 or not os.path.exists(model):
            return {"ok": False, "error": err or "обучение не завершилось"}
        res = _parse_train_result(out)
        if res and res.get("backup"):
            self._rc.set_sac_backup_path(res["backup"])
        return {
            "ok": True,
            "sac_pct": res and res.get("sac_pct"),
            "bh_pct": res and res.get("bh_pct"),
            "worse": _is_worse(res),
        }

    async def _tune(self) -> Dict[str, Any]:
        """Тюнинг (Optuna), затем обучение с найденными параметрами."""
        env = {"SAC_MODEL_PATH": self._rc.get_sac_model_path()}

        def progress(info: Dict[str, Any]) -> None:
            self._rc.set_tune_progress(info)
            emit(self._r, "job.progress", {"job": "tune", **info})

        code, _, err = await self._exec(
            "reinforcement_learning/tune_sac.py", env, "TUNE_PROGRESS:", progress
        )
        if code != 0:
            return {"ok": False, "error": err or "тюнинг не завершился"}
        emit(self._r, "job.progress", {"job": "tune", "stage": "train", "pct": 100})
        # Как в Telegram: второй этап идёт с окружением тюнинга, без профиля монет.
        return await self._train_stage(env)

    async def _report(self, job: str) -> Dict[str, Any]:
        code, out, err = await self._exec(_REPORT_SCRIPTS[job], {})
        if code != 0:
            return {"ok": False, "error": err or "отчёт не получился"}
        text = out.strip()
        if len(text) > REPORT_LIMIT:
            text = "…\n" + text[-REPORT_LIMIT:]
        return {"ok": True, "output": text}
