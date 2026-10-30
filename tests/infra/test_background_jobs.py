"""Фоновые задачи из приложения (src/infra/background_jobs.py)."""

import json
from unittest.mock import AsyncMock, MagicMock

from src.core.bot_commands import BotCommands
from src.infra import background_jobs as bj
from tests.infra.test_bot_bus import FakeRedis


class FakeProc:
    def __init__(self, out_lines, code=0, err=b""):
        self.returncode = code
        self._lines = [ln.encode() + b"\n" for ln in out_lines]
        self.stderr = MagicMock(read=AsyncMock(return_value=err))

        async def gen():
            for ln in self._lines:
                yield ln

        self.stdout = gen()

    async def wait(self):
        return self.returncode


class FakeRC:
    """RuntimeConfig с настоящим замком на словаре."""

    def __init__(self):
        self.locks, self.calls = {}, []
        self.backup = ""
        self.model = "models/sac.zip"
        self.models = {}
        self.backup_profile = ""

    def try_start_job(self, group, source, ttl):
        if group in self.locks:
            return False
        self.locks[group] = source
        return True

    def end_job(self, group):
        self.locks.pop(group, None)

    def running_jobs(self, groups):
        return {g: self.locks[g] for g in groups if g in self.locks}

    def get_backtest_progress(self):
        return None

    get_train_progress = get_tune_progress = get_backtest_progress

    def get_sac_backup_path(self):
        return self.backup

    def set_sac_backup_path(self, p, profile=""):
        self.backup, self.backup_profile = p, profile

    def get_sac_backup_profile(self):
        return self.backup_profile

    def get_sac_model_path(self):
        return self.model

    def get_sac_model_path_for_profile(self, profile):
        return self.models.get(profile, self.model)

    def profile_ids(self):
        return ["bluechip", "altcoin", "meme"]

    def get_training_profiles(self):
        return [{"id": "altcoin"}]

    def get_market_profile(self):
        return "altcoin"

    def get_market_profile_config(self, name):
        return {"train_top_n": 30, "timeframe": "1h"}

    def __getattr__(self, name):  # set_*_progress / clear_*_progress
        return lambda *a, **k: self.calls.append((name, a))


def _setup(monkeypatch, procs):
    r, rc = FakeRedis(), FakeRC()
    bot = MagicMock()
    bot.redis.redis_client = r
    bot._runtime_config = rc
    bot.telegram.notify = AsyncMock()
    seen = []

    async def fake_exec(*args, **kwargs):
        seen.append((args, kwargs))
        return procs.pop(0)

    monkeypatch.setattr("asyncio.create_subprocess_exec", fake_exec)
    return BotCommands(bot), bot, r, rc, seen


def _types(r):
    return [m["type"] for _, m in r.published]


def _done(r):
    return [m["data"] for _, m in r.published if m["type"] == "job.done"][-1]


async def test_unknown_job_and_busy(monkeypatch):
    cmds, bot, r, rc, _ = _setup(monkeypatch, [FakeProc([])])
    assert (await cmds.run_job({"job": "nope"}))["error"]["code"] == "unknown_job"
    assert await cmds.run_job({"job": "alpha_report"}) == {
        "started": True,
        "job": "alpha_report",
    }
    # пока идёт — повторный запуск и запуск «из Telegram» отклоняются
    assert (await cmds.run_job({"job": "alpha_report"}))["error"]["code"] == "busy"
    await cmds._jobs._tasks["alpha_report"]
    assert rc.locks == {}  # замок снят


async def test_backtest_reports_progress_and_result(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "backtest_results.json").write_text(
        json.dumps(
            {
                "months": 12,
                "timeframe": "15m",
                "results": [
                    {"strategy": "a", "sharpe_ratio": -0.2, "total_trades": 10},
                    {"strategy": "b", "sharpe_ratio": 0.3, "total_trades": 200},
                ],
            }
        )
    )
    proc = FakeProc(['BACKTEST_PROGRESS:{"pct": 40, "step": 8}', "шум"])
    cmds, bot, r, rc, _ = _setup(monkeypatch, [proc])
    await cmds.run_job({"job": "backtest"})
    await cmds._jobs._tasks["backtest"]
    assert _types(r)[0] == "job.progress" and _types(r)[-1] == "job.done"
    assert any(m["data"].get("pct") == 40 for _, m in r.published)
    assert _done(r)["best"] == {"strategy": "b", "sharpe": 0.3, "trades": 200}
    assert ("set_backtest_progress", ({"pct": 40, "step": 8},)) in rc.calls


async def test_train_uses_profile_env_and_keeps_backup(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    model = tmp_path / "sac.zip"
    model.write_text("m")
    result = {"sac_pct": 5.0, "bh_pct": 20.0, "backup": "models/old.zip"}
    proc = FakeProc(
        ['TRAIN_PROGRESS:{"pct": 10}', "TRAIN_RESULT:" + json.dumps(result)]
    )
    cmds, bot, r, rc, seen = _setup(monkeypatch, [proc])
    rc.model = str(model)
    await cmds.run_job({"job": "train"})
    await cmds._jobs._tasks["sac"]
    env = seen[0][1]["env"]
    assert env["TRAIN_TOP_N"] == "30" and env["BT_TIMEFRAME"] == "1h"
    assert env["SAC_PROFILE"] == "altcoin"
    assert env["EXPERIENCES_PATH"] == "data/experiences_altcoin.jsonl"
    done = _done(r)
    assert done["ok"] and done["worse"] is True  # 5% против 20%: хуже рынка
    assert rc.backup == "models/old.zip"
    bot.telegram.notify.assert_awaited()


async def test_tune_runs_two_stages(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    model = tmp_path / "sac.zip"
    model.write_text("m")
    procs = [
        FakeProc(['TUNE_PROGRESS:{"pct": 50, "trial": 15, "total": 30}']),
        FakeProc(["TRAIN_RESULT:" + json.dumps({"sac_pct": 30, "bh_pct": 10})]),
    ]
    cmds, bot, r, rc, seen = _setup(monkeypatch, procs)
    rc.model = str(model)
    await cmds.run_job({"job": "tune"})
    await cmds._jobs._tasks["sac"]
    scripts = [a[0][-1] for a in seen]
    assert scripts == [
        "reinforcement_learning/tune_sac.py",
        "reinforcement_learning/train_sac.py",
    ]
    assert _done(r)["ok"] and _done(r)["worse"] is False


async def test_failed_script_reports_error_and_frees_the_lock(monkeypatch):
    cmds, bot, r, rc, _ = _setup(monkeypatch, [FakeProc([], code=1, err=b"boom")])
    await cmds.run_job({"job": "live_divergence"})
    await cmds._jobs._tasks["live_divergence"]
    assert _done(r) == {"job": "live_divergence", "ok": False, "error": "boom"}
    assert rc.locks == {}


async def test_report_output_is_trimmed(monkeypatch):
    big = ["x" * 100] * 80
    cmds, bot, r, rc, _ = _setup(monkeypatch, [FakeProc(big)])
    await cmds.run_job({"job": "ai_isolation"})
    await cmds._jobs._tasks["ai_isolation"]
    out = _done(r)["output"]
    assert out.startswith("…\n") and len(out) <= bj.REPORT_LIMIT + 3


async def test_rollback(monkeypatch, tmp_path):
    cmds, bot, r, rc, _ = _setup(monkeypatch, [])
    assert (await cmds.rollback_model({}))["error"]["code"] == "no_backup"
    backup, model = tmp_path / "old.zip", tmp_path / "sac.zip"
    backup.write_text("old")
    model.write_text("new")
    rc.backup, rc.model = str(backup), str(model)
    rc.locks["sac"] = "telegram"  # идёт обучение — откат запрещён
    assert (await cmds.rollback_model({}))["error"]["code"] == "busy"
    rc.locks.clear()
    assert await cmds.rollback_model({}) == {"rolled_back": True}
    assert model.read_text() == "old"
    assert "model.rolled_back" in _types(r)


async def test_train_other_profile_and_rollback_restores_that_model(
    monkeypatch, tmp_path
):
    monkeypatch.chdir(tmp_path)
    meme = tmp_path / "meme.zip"
    meme.write_text("new")
    old = tmp_path / "old.zip"
    old.write_text("old")
    result = {"sac_pct": 30.0, "bh_pct": 10.0, "backup": str(old)}
    proc = FakeProc(["TRAIN_RESULT:" + json.dumps(result)])
    cmds, bot, r, rc, seen = _setup(monkeypatch, [proc])
    rc.models = {"meme": str(meme)}
    bad = await cmds.run_job({"job": "train", "profile": "nope"})
    assert bad["error"]["code"] == "unknown_profile"
    await cmds.run_job({"job": "train", "profile": "meme"})
    await cmds._jobs._tasks["sac"]
    env = seen[0][1]["env"]
    assert env["SAC_PROFILE"] == "meme" and env["SAC_MODEL_PATH"] == str(meme)
    assert rc.backup_profile == "meme"
    assert await cmds.rollback_model({}) == {"rolled_back": True}
    assert meme.read_text() == "old"  # а не модель активного профиля


def test_status_reports_rollback_and_running(tmp_path):
    rc = FakeRC()
    backup = tmp_path / "b.zip"
    backup.write_text("x")
    rc.backup = str(backup)
    rc.locks["sac"] = "telegram"
    st = bj.status(rc)
    assert st["running"] == {"sac": "telegram"} and st["rollback_available"] is True


def test_job_groups_exclude_each_other_only_where_needed():
    assert bj.JOBS["train"].group == bj.JOBS["tune"].group == "sac"
    groups = {j.group for j in bj.JOBS.values()}
    assert len(groups) == 5
