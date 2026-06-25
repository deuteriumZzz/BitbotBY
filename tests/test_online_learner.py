"""
Comprehensive pytest tests for src/online_learner.py.

Coverage targets:
  - OnlineLearner.__init__() — all five mode branches
  - on_trade_closed()        — disabled / online / periodic / hybrid
  - get_strategy_weight()    — non-hybrid, hybrid + Redis, hybrid + compute
  - _online_update()         — skips when busy, skips when no model, runs
  - _periodic_retrain_if_needed() — trigger logic, busy guard, task creation
  - _get_train_top_n()       — via RuntimeConfig and via env fallback
  - _background_retrain()    — happy path, exception propagation
  - _run_online_update()     — success, ImportError, generic Exception
  - _run_full_retrain()      — success (with/without existing model), exception
  - _record_strategy_result() / _compute_weight() / _sync_strategy_weights()
"""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_cfg(
    mode: str = "periodic",
    trigger: int = 50,
    gradient_steps: int = 30,
    model_path: str = "/tmp/test_sac_model",
) -> MagicMock:
    cfg = MagicMock()
    cfg.ONLINE_LEARNING_MODE = mode
    cfg.ONLINE_LEARNING_TRIGGER = trigger
    cfg.ONLINE_LEARNING_GRADIENT_STEPS = gradient_steps
    cfg.SAC_MODEL_PATH = model_path
    return cfg


def _make_learner(mode: str = "periodic", redis=None, **cfg_kwargs):
    """
    Instantiate OnlineLearner with a mocked Config.
    Patches src.online_learner.Config *before* import so the module-level
    attribute lookup in __init__ sees the mock.
    """
    cfg = _make_cfg(mode=mode, **cfg_kwargs)
    # Import first (uses cached module), then patch Config so __init__ sees the mock
    from src.online_learner import OnlineLearner  # ensure module is loaded

    with patch("src.online_learner.Config", cfg):
        learner = OnlineLearner(redis_client=redis)
    return learner, cfg


# ---------------------------------------------------------------------------
# __init__ — mode branch coverage
# ---------------------------------------------------------------------------


class TestOnlineLearnerInit:
    def test_init_disabled(self):
        learner, cfg = _make_learner(mode="disabled")
        assert learner._mode == "disabled"
        assert learner._closed_count == 0
        assert learner._is_training is False
        assert learner._redis is None

    def test_init_online(self):
        learner, cfg = _make_learner(mode="online")
        assert learner._mode == "online"

    def test_init_periodic(self):
        learner, cfg = _make_learner(mode="periodic", trigger=100)
        assert learner._mode == "periodic"
        assert learner._trigger == 100

    def test_init_hybrid(self):
        learner, cfg = _make_learner(mode="hybrid")
        assert learner._mode == "hybrid"

    def test_init_unknown_mode_falls_back_to_periodic(self):
        learner, cfg = _make_learner(mode="bogus_mode")
        assert learner._mode == "periodic"

    def test_init_with_redis_client(self):
        redis_mock = MagicMock()
        learner, _ = _make_learner(mode="periodic", redis=redis_mock)
        assert learner._redis is redis_mock

    def test_init_gradient_steps_stored(self):
        learner, _ = _make_learner(mode="online", gradient_steps=42)
        assert learner._gradient_steps == 42

    def test_init_strategy_results_defaultdict(self):
        learner, _ = _make_learner()
        # defaultdict should produce a deque-like container for any new key
        result = learner._strategy_results["new_strategy"]
        assert hasattr(result, "append")

    def test_init_background_tasks_is_set(self):
        learner, _ = _make_learner()
        assert isinstance(learner._background_tasks, set)


# ---------------------------------------------------------------------------
# on_trade_closed — routing logic
# ---------------------------------------------------------------------------


class TestOnTradeClosed:
    @pytest.mark.asyncio
    async def test_disabled_returns_immediately(self):
        learner, _ = _make_learner(mode="disabled")
        learner._online_update = AsyncMock()
        learner._periodic_retrain_if_needed = AsyncMock()

        await learner.on_trade_closed("BTC/USDT", "buy", 0.03)

        assert learner._closed_count == 0
        learner._online_update.assert_not_called()
        learner._periodic_retrain_if_needed.assert_not_called()

    @pytest.mark.asyncio
    async def test_online_mode_calls_online_update(self):
        learner, _ = _make_learner(mode="online")
        learner._online_update = AsyncMock()

        await learner.on_trade_closed("BTC/USDT", "buy", 0.01)

        assert learner._closed_count == 1
        learner._online_update.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_periodic_mode_calls_periodic_retrain(self):
        learner, _ = _make_learner(mode="periodic")
        learner._periodic_retrain_if_needed = AsyncMock()

        await learner.on_trade_closed("ETH/USDT", "sell", -0.02)

        assert learner._closed_count == 1
        learner._periodic_retrain_if_needed.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_hybrid_mode_records_and_syncs(self):
        learner, _ = _make_learner(mode="hybrid")
        learner._sync_strategy_weights = AsyncMock()
        learner._periodic_retrain_if_needed = AsyncMock()

        await learner.on_trade_closed("SOL/USDT", "buy", 0.05, strategy="ema_crossover")

        assert learner._closed_count == 1
        assert learner._strategy_results["ema_crossover"][0] == 1  # win
        learner._sync_strategy_weights.assert_awaited_once()
        learner._periodic_retrain_if_needed.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_hybrid_records_loss(self):
        learner, _ = _make_learner(mode="hybrid")
        learner._sync_strategy_weights = AsyncMock()
        learner._periodic_retrain_if_needed = AsyncMock()

        await learner.on_trade_closed(
            "BTC/USDT", "sell", -0.03, strategy="rsi_momentum"
        )

        assert learner._strategy_results["rsi_momentum"][0] == 0  # loss

    @pytest.mark.asyncio
    async def test_closed_count_increments_each_call(self):
        learner, _ = _make_learner(mode="periodic")
        learner._periodic_retrain_if_needed = AsyncMock()

        for _ in range(3):
            await learner.on_trade_closed("BTC/USDT", "buy", 0.01)

        assert learner._closed_count == 3

    @pytest.mark.asyncio
    async def test_pnl_zero_is_a_loss(self):
        """pnl_pct == 0 should not be counted as a win (is_win = pnl > 0)."""
        learner, _ = _make_learner(mode="hybrid")
        learner._sync_strategy_weights = AsyncMock()
        learner._periodic_retrain_if_needed = AsyncMock()

        await learner.on_trade_closed("BTC/USDT", "buy", 0.0, strategy="s")

        assert learner._strategy_results["s"][0] == 0  # 0.0 is NOT > 0


# ---------------------------------------------------------------------------
# get_strategy_weight
# ---------------------------------------------------------------------------


class TestGetStrategyWeight:
    def test_non_hybrid_always_returns_1(self):
        for mode in ("disabled", "online", "periodic"):
            learner, _ = _make_learner(mode=mode)
            assert learner.get_strategy_weight("ema_crossover") == 1.0

    def test_hybrid_no_data_returns_1(self):
        learner, _ = _make_learner(mode="hybrid")
        assert learner.get_strategy_weight("ema_crossover") == 1.0

    def test_hybrid_uses_redis_value(self):
        redis_mock = MagicMock()
        redis_mock.redis_client.get.return_value = b"1.35"
        learner, _ = _make_learner(mode="hybrid", redis=redis_mock)
        assert learner.get_strategy_weight("ema_crossover") == 1.35

    def test_hybrid_redis_none_falls_back_to_compute(self):
        redis_mock = MagicMock()
        redis_mock.redis_client.get.return_value = None
        learner, _ = _make_learner(mode="hybrid", redis=redis_mock)
        for _ in range(10):
            learner._strategy_results["ema_crossover"].append(1)
        weight = learner.get_strategy_weight("ema_crossover")
        assert weight == 1.5  # 100% WR → 0.5 + 1.0

    def test_hybrid_redis_raises_falls_back_to_compute(self):
        redis_mock = MagicMock()
        redis_mock.redis_client.get.side_effect = RuntimeError("conn error")
        learner, _ = _make_learner(mode="hybrid", redis=redis_mock)
        # < 5 results → neutral weight
        assert learner.get_strategy_weight("ema_crossover") == 1.0

    def test_hybrid_no_redis_uses_compute(self):
        learner, _ = _make_learner(mode="hybrid")
        for _ in range(10):
            learner._strategy_results["rsi_momentum"].append(0)  # all losses
        weight = learner.get_strategy_weight("rsi_momentum")
        assert weight == 0.5  # 0% WR → 0.5

    def test_hybrid_redis_returns_string_float(self):
        """Test that a string value from Redis is correctly cast to float."""
        redis_mock = MagicMock()
        redis_mock.redis_client.get.return_value = "1.25"
        learner, _ = _make_learner(mode="hybrid", redis=redis_mock)
        assert learner.get_strategy_weight("some_strategy") == 1.25


# ---------------------------------------------------------------------------
# _compute_weight — pure unit tests
# ---------------------------------------------------------------------------


class TestComputeWeight:
    def test_fewer_than_5_results_returns_1(self):
        learner, _ = _make_learner()
        for _ in range(4):
            learner._strategy_results["s"].append(1)
        assert learner._compute_weight("s") == 1.0

    def test_empty_strategy_key_returns_1(self):
        learner, _ = _make_learner()
        assert learner._compute_weight("nonexistent") == 1.0

    def test_all_wins_returns_1_5(self):
        learner, _ = _make_learner()
        for _ in range(10):
            learner._strategy_results["s"].append(1)
        assert learner._compute_weight("s") == 1.5

    def test_all_losses_returns_0_5(self):
        learner, _ = _make_learner()
        for _ in range(10):
            learner._strategy_results["s"].append(0)
        assert learner._compute_weight("s") == 0.5

    def test_50_percent_wr_returns_1_0(self):
        learner, _ = _make_learner()
        for _ in range(5):
            learner._strategy_results["s"].append(1)
        for _ in range(5):
            learner._strategy_results["s"].append(0)
        assert learner._compute_weight("s") == 1.0

    def test_result_is_rounded_to_3_places(self):
        learner, _ = _make_learner()
        # 3 wins out of 7 → WR ≈ 0.4286
        for i in range(7):
            learner._strategy_results["s"].append(1 if i < 3 else 0)
        weight = learner._compute_weight("s")
        assert weight == round(weight, 3)

    def test_lower_clamp_at_0_5(self):
        learner, _ = _make_learner()
        for _ in range(20):
            learner._strategy_results["s"].append(0)
        assert learner._compute_weight("s") >= 0.5

    def test_upper_clamp_at_1_5(self):
        learner, _ = _make_learner()
        for _ in range(20):
            learner._strategy_results["s"].append(1)
        assert learner._compute_weight("s") <= 1.5

    def test_exactly_5_results_is_computed(self):
        learner, _ = _make_learner()
        for _ in range(5):
            learner._strategy_results["s"].append(1)
        # 5 results is exactly the threshold → should compute, not return 1.0
        weight = learner._compute_weight("s")
        assert weight == 1.5


# ---------------------------------------------------------------------------
# _record_strategy_result
# ---------------------------------------------------------------------------


class TestRecordStrategyResult:
    def test_win_appends_1(self):
        learner, _ = _make_learner()
        learner._record_strategy_result("ema_crossover", True)
        assert learner._strategy_results["ema_crossover"][-1] == 1

    def test_loss_appends_0(self):
        learner, _ = _make_learner()
        learner._record_strategy_result("rsi_momentum", False)
        assert learner._strategy_results["rsi_momentum"][-1] == 0

    def test_multiple_strategies_independent(self):
        learner, _ = _make_learner()
        learner._record_strategy_result("a", True)
        learner._record_strategy_result("b", False)
        assert learner._strategy_results["a"][-1] == 1
        assert learner._strategy_results["b"][-1] == 0

    def test_window_maxlen_respected(self):
        from src.online_learner import _WEIGHT_WINDOW

        learner, _ = _make_learner()
        for _ in range(_WEIGHT_WINDOW + 10):
            learner._record_strategy_result("s", True)
        assert len(learner._strategy_results["s"]) == _WEIGHT_WINDOW


# ---------------------------------------------------------------------------
# _sync_strategy_weights
# ---------------------------------------------------------------------------


class TestSyncStrategyWeights:
    @pytest.mark.asyncio
    async def test_no_redis_noop(self):
        learner, _ = _make_learner(mode="hybrid")
        # Should complete without error
        await learner._sync_strategy_weights()

    @pytest.mark.asyncio
    async def test_writes_to_redis_when_enough_data(self):
        redis_mock = MagicMock()
        learner, _ = _make_learner(mode="hybrid", redis=redis_mock)
        for _ in range(10):
            learner._strategy_results["ema_crossover"].append(1)
        await learner._sync_strategy_weights()
        redis_mock.redis_client.setex.assert_called_once()
        args = redis_mock.redis_client.setex.call_args[0]
        assert "ema_crossover" in args[0]
        assert args[1] == 3600  # TTL

    @pytest.mark.asyncio
    async def test_skips_strategy_with_fewer_than_5_trades(self):
        redis_mock = MagicMock()
        learner, _ = _make_learner(mode="hybrid", redis=redis_mock)
        for _ in range(3):
            learner._strategy_results["ema_crossover"].append(1)
        await learner._sync_strategy_weights()
        redis_mock.redis_client.setex.assert_not_called()

    @pytest.mark.asyncio
    async def test_redis_exception_is_swallowed(self):
        redis_mock = MagicMock()
        redis_mock.redis_client.setex.side_effect = RuntimeError("redis down")
        learner, _ = _make_learner(mode="hybrid", redis=redis_mock)
        for _ in range(10):
            learner._strategy_results["ema_crossover"].append(1)
        # Must not raise
        await learner._sync_strategy_weights()

    @pytest.mark.asyncio
    async def test_multiple_strategies_all_synced(self):
        redis_mock = MagicMock()
        learner, _ = _make_learner(mode="hybrid", redis=redis_mock)
        for _ in range(6):
            learner._strategy_results["s1"].append(1)
            learner._strategy_results["s2"].append(0)
        await learner._sync_strategy_weights()
        assert redis_mock.redis_client.setex.call_count == 2


# ---------------------------------------------------------------------------
# _online_update
# ---------------------------------------------------------------------------


class TestOnlineUpdate:
    @pytest.mark.asyncio
    async def test_skips_if_already_training(self):
        learner, cfg = _make_learner(mode="online")
        learner._is_training = True
        with patch("src.online_learner.Config", cfg):
            with patch("os.path.exists", return_value=True):
                with patch.object(learner, "_run_online_update") as mock_run:
                    await learner._online_update()
        mock_run.assert_not_called()

    @pytest.mark.asyncio
    async def test_skips_if_model_not_found(self):
        learner, cfg = _make_learner(mode="online")
        with patch("src.online_learner.Config", cfg):
            with patch("os.path.exists", return_value=False):
                with patch.object(learner, "_run_online_update") as mock_run:
                    await learner._online_update()
        mock_run.assert_not_called()

    @pytest.mark.asyncio
    async def test_runs_update_in_executor(self):
        learner, cfg = _make_learner(mode="online", gradient_steps=10)
        with patch("src.online_learner.Config", cfg):
            with patch("os.path.exists", return_value=True):
                with patch.object(learner, "_run_online_update") as mock_run:
                    await learner._online_update()
        mock_run.assert_called_once_with(10)
        assert learner._is_training is False

    @pytest.mark.asyncio
    async def test_flag_reset_on_exception(self):
        learner, cfg = _make_learner(mode="online")

        def _raise(steps):
            raise RuntimeError("oops")

        with patch("src.online_learner.Config", cfg):
            with patch("os.path.exists", return_value=True):
                with patch.object(learner, "_run_online_update", side_effect=_raise):
                    with pytest.raises(RuntimeError):
                        await learner._online_update()
        assert learner._is_training is False


# ---------------------------------------------------------------------------
# _periodic_retrain_if_needed
# ---------------------------------------------------------------------------


class TestPeriodicRetrainIfNeeded:
    @pytest.mark.asyncio
    async def test_no_trigger_does_nothing(self):
        learner, _ = _make_learner(mode="periodic", trigger=50)
        learner._closed_count = 25  # not a multiple of 50

        with patch("asyncio.create_task") as mock_create:
            await learner._periodic_retrain_if_needed()
        mock_create.assert_not_called()

    @pytest.mark.asyncio
    async def test_trigger_at_exact_multiple(self):
        learner, _ = _make_learner(mode="periodic", trigger=50)
        learner._closed_count = 50

        mock_task = MagicMock()
        mock_task.add_done_callback = MagicMock()

        with patch("asyncio.create_task", return_value=mock_task) as mock_create:
            await learner._periodic_retrain_if_needed()

        mock_create.assert_called_once()
        assert learner._is_training is True

    @pytest.mark.asyncio
    async def test_skips_if_already_training(self):
        learner, _ = _make_learner(mode="periodic", trigger=50)
        learner._closed_count = 50
        learner._is_training = True

        with patch("asyncio.create_task") as mock_create:
            await learner._periodic_retrain_if_needed()

        mock_create.assert_not_called()

    @pytest.mark.asyncio
    async def test_trigger_at_second_multiple(self):
        learner, _ = _make_learner(mode="periodic", trigger=50)
        learner._closed_count = 100

        mock_task = MagicMock()
        mock_task.add_done_callback = MagicMock()

        with patch("asyncio.create_task", return_value=mock_task) as mock_create:
            await learner._periodic_retrain_if_needed()

        mock_create.assert_called_once()

    @pytest.mark.asyncio
    async def test_task_added_to_background_tasks(self):
        learner, _ = _make_learner(mode="periodic", trigger=10)
        learner._closed_count = 10

        mock_task = MagicMock()
        mock_task.add_done_callback = MagicMock()

        with patch("asyncio.create_task", return_value=mock_task):
            await learner._periodic_retrain_if_needed()

        assert mock_task in learner._background_tasks

    @pytest.mark.asyncio
    async def test_done_callback_registered(self):
        learner, _ = _make_learner(mode="periodic", trigger=10)
        learner._closed_count = 10

        mock_task = MagicMock()
        mock_task.add_done_callback = MagicMock()

        with patch("asyncio.create_task", return_value=mock_task):
            await learner._periodic_retrain_if_needed()

        mock_task.add_done_callback.assert_called_once()


# ---------------------------------------------------------------------------
# _get_train_top_n
# ---------------------------------------------------------------------------


class TestGetTrainTopN:
    def test_returns_from_env_variable(self):
        learner, _ = _make_learner()
        with patch.dict("sys.modules", {"src.runtime_config": None}):
            with patch.dict(os.environ, {"TRAIN_TOP_N": "77"}):
                n = learner._get_train_top_n()
        assert n == 77

    def test_falls_back_to_default_20_when_no_env(self):
        learner, _ = _make_learner()
        env = {k: v for k, v in os.environ.items() if k != "TRAIN_TOP_N"}
        with patch.dict("sys.modules", {"src.runtime_config": None}):
            with patch.dict(os.environ, env, clear=True):
                n = learner._get_train_top_n()
        assert n == 20

    def test_runtime_config_import_error_falls_back(self):
        """When RuntimeConfig raises any exception, fall back to env var."""
        learner, _ = _make_learner()
        mock_rc_module = MagicMock()
        mock_rc_module.RuntimeConfig.side_effect = Exception("import failed")
        with patch.dict("sys.modules", {"src.runtime_config": mock_rc_module}):
            with patch.dict(os.environ, {"TRAIN_TOP_N": "55"}):
                n = learner._get_train_top_n()
        assert n == 55

    def test_runtime_config_get_raises_falls_back(self):
        """When get_train_top_n() raises, fall back to env var."""
        learner, _ = _make_learner()
        mock_rc_instance = MagicMock()
        mock_rc_instance.get_train_top_n.side_effect = RuntimeError("redis error")
        mock_rc_module = MagicMock()
        mock_rc_module.RuntimeConfig.return_value = mock_rc_instance
        with patch.dict("sys.modules", {"src.runtime_config": mock_rc_module}):
            with patch.dict(os.environ, {"TRAIN_TOP_N": "33"}):
                n = learner._get_train_top_n()
        assert n == 33


# ---------------------------------------------------------------------------
# _background_retrain
# ---------------------------------------------------------------------------


class TestBackgroundRetrain:
    @pytest.mark.asyncio
    async def test_calls_run_full_retrain_and_resets_flag(self):
        learner, _ = _make_learner()
        learner._is_training = True

        with patch.object(learner, "_run_full_retrain") as mock_retrain:
            await learner._background_retrain()

        mock_retrain.assert_called_once()
        assert learner._is_training is False

    @pytest.mark.asyncio
    async def test_resets_flag_on_exception(self):
        learner, _ = _make_learner()
        learner._is_training = True

        def _raise():
            raise ValueError("training failed")

        with patch.object(learner, "_run_full_retrain", side_effect=_raise):
            with pytest.raises(ValueError):
                await learner._background_retrain()

        assert learner._is_training is False


# ---------------------------------------------------------------------------
# _run_online_update — synchronous internals
# ---------------------------------------------------------------------------


class TestRunOnlineUpdate:
    def test_success_path(self):
        learner, cfg = _make_learner(mode="online")
        mock_sac_module = MagicMock()
        mock_model = MagicMock()
        mock_sac_module.SAC.load.return_value = mock_model

        mock_train_module = MagicMock()
        mock_train_module._finetune_on_experiences = MagicMock()
        mock_train_module._load_norm_stats = MagicMock(return_value={})

        with patch("src.online_learner.Config", cfg):
            with patch.dict(
                "sys.modules",
                {
                    "stable_baselines3": mock_sac_module,
                    "reinforcement_learning.train_sac": mock_train_module,
                },
            ):
                with patch("os.replace") as mock_replace:
                    learner._run_online_update(gradient_steps=10)

        mock_sac_module.SAC.load.assert_called_once_with(cfg.SAC_MODEL_PATH)
        mock_model.save.assert_called_once()
        mock_replace.assert_called_once()

    def test_import_error_is_swallowed(self):
        """If stable_baselines3 is not installed, the method logs and returns."""
        learner, cfg = _make_learner(mode="online")

        # Simulate missing stable_baselines3
        with patch("src.online_learner.Config", cfg):
            with patch.dict("sys.modules", {"stable_baselines3": None}):
                # Should not raise
                learner._run_online_update(gradient_steps=10)

    def test_generic_exception_is_logged_not_raised(self):
        learner, cfg = _make_learner(mode="online")
        mock_sac_module = MagicMock()
        mock_sac_module.SAC.load.side_effect = RuntimeError("model corrupted")
        mock_train_module = MagicMock()

        with patch("src.online_learner.Config", cfg):
            with patch.dict(
                "sys.modules",
                {
                    "stable_baselines3": mock_sac_module,
                    "reinforcement_learning.train_sac": mock_train_module,
                },
            ):
                # Should not raise
                learner._run_online_update(gradient_steps=5)

    def test_model_saved_to_tmp_path(self):
        """Verify the model is saved to a .online_tmp path before os.replace."""
        learner, cfg = _make_learner(mode="online", model_path="/tmp/sac.zip")
        mock_sac_module = MagicMock()
        mock_model = MagicMock()
        mock_sac_module.SAC.load.return_value = mock_model
        mock_train_module = MagicMock()
        mock_train_module._load_norm_stats.return_value = {}

        with patch("src.online_learner.Config", cfg):
            with patch.dict(
                "sys.modules",
                {
                    "stable_baselines3": mock_sac_module,
                    "reinforcement_learning.train_sac": mock_train_module,
                },
            ):
                with patch("os.replace"):
                    learner._run_online_update(gradient_steps=5)

        saved_path = mock_model.save.call_args[0][0]
        assert saved_path.endswith(".online_tmp")


# ---------------------------------------------------------------------------
# _run_full_retrain — synchronous
# ---------------------------------------------------------------------------


class TestRunFullRetrain:
    def _run(self, learner, cfg, model_exists=True, subprocess_raises=None):
        """Helper: patch subprocess + os calls and run _run_full_retrain."""
        mock_sub_run = MagicMock()
        if subprocess_raises:
            mock_sub_run.side_effect = subprocess_raises
        else:
            mock_sub_run.return_value = MagicMock(returncode=0)

        with patch("src.online_learner.Config", cfg):
            with patch("subprocess.run", mock_sub_run):
                with patch("os.path.exists", return_value=model_exists):
                    with patch("os.replace") as mock_replace:
                        learner._run_full_retrain()
        return mock_sub_run, mock_replace

    def test_success_with_existing_model(self):
        learner, cfg = _make_learner(mode="periodic")
        mock_sub, mock_replace = self._run(learner, cfg, model_exists=True)
        mock_sub.assert_called_once()
        # Two os.replace calls: old→bak, new→current
        assert mock_replace.call_count == 2

    def test_success_without_existing_model(self):
        learner, cfg = _make_learner(mode="periodic")
        mock_sub, mock_replace = self._run(learner, cfg, model_exists=False)
        mock_sub.assert_called_once()
        # Only one os.replace call: new→current
        assert mock_replace.call_count == 1

    def test_subprocess_exception_is_swallowed(self):
        learner, cfg = _make_learner(mode="periodic")
        # Should not raise
        self._run(learner, cfg, subprocess_raises=RuntimeError("subprocess died"))

    def test_env_contains_required_keys(self):
        learner, cfg = _make_learner(mode="periodic")
        captured_env: dict = {}

        def capture_run(*args, env=None, **kwargs):
            nonlocal captured_env
            captured_env = env or {}
            return MagicMock(returncode=0)

        with patch("src.online_learner.Config", cfg):
            with patch("subprocess.run", side_effect=capture_run):
                with patch("os.path.exists", return_value=False):
                    with patch("os.replace"):
                        learner._run_full_retrain()

        for key in (
            "SAC_MODEL_PATH",
            "TRAIN_TOP_N",
            "TOTAL_TIMESTEPS",
            "EXPERIENCES_PATH",
        ):
            assert key in captured_env

    def test_sac_model_path_has_new_suffix(self):
        learner, cfg = _make_learner(mode="periodic")
        captured_env: dict = {}

        def capture_run(*args, env=None, **kwargs):
            nonlocal captured_env
            captured_env = env or {}
            return MagicMock(returncode=0)

        with patch("src.online_learner.Config", cfg):
            with patch("subprocess.run", side_effect=capture_run):
                with patch("os.path.exists", return_value=False):
                    with patch("os.replace"):
                        learner._run_full_retrain()

        assert captured_env.get("SAC_MODEL_PATH", "").endswith(".new")

    def test_train_top_n_value_in_env(self):
        learner, cfg = _make_learner(mode="periodic")
        learner._get_train_top_n = MagicMock(return_value=42)
        captured_env: dict = {}

        def capture_run(*args, env=None, **kwargs):
            nonlocal captured_env
            captured_env = env or {}
            return MagicMock(returncode=0)

        with patch("src.online_learner.Config", cfg):
            with patch("subprocess.run", side_effect=capture_run):
                with patch("os.path.exists", return_value=False):
                    with patch("os.replace"):
                        learner._run_full_retrain()

        assert captured_env.get("TRAIN_TOP_N") == "42"

    def test_timeout_passed_to_subprocess(self):
        learner, cfg = _make_learner(mode="periodic")
        captured_kwargs: dict = {}

        def capture_run(*args, **kwargs):
            nonlocal captured_kwargs
            captured_kwargs = kwargs
            return MagicMock(returncode=0)

        with patch("src.online_learner.Config", cfg):
            with patch("subprocess.run", side_effect=capture_run):
                with patch("os.path.exists", return_value=False):
                    with patch("os.replace"):
                        learner._run_full_retrain()

        assert captured_kwargs.get("timeout") == 7200


# ---------------------------------------------------------------------------
# Integration-style: on_trade_closed → _periodic_retrain_if_needed
# ---------------------------------------------------------------------------


class TestPeriodicIntegration:
    @pytest.mark.asyncio
    async def test_trigger_spawns_background_task(self):
        """Verify a background task is created exactly at trigger count."""
        learner, _ = _make_learner(mode="periodic", trigger=3)

        mock_task = MagicMock()
        mock_task.add_done_callback = MagicMock()
        learner._background_retrain = AsyncMock()

        with patch("asyncio.create_task", return_value=mock_task) as mock_create:
            await learner.on_trade_closed("BTC/USDT", "buy", 0.01)
            await learner.on_trade_closed("BTC/USDT", "buy", 0.01)
            assert mock_create.call_count == 0

            await learner.on_trade_closed("BTC/USDT", "buy", 0.01)
            assert mock_create.call_count == 1

    @pytest.mark.asyncio
    async def test_disabled_never_trains(self):
        learner, _ = _make_learner(mode="disabled")

        with patch("asyncio.create_task") as mock_create:
            for _ in range(100):
                await learner.on_trade_closed("BTC/USDT", "buy", 0.01)
        mock_create.assert_not_called()


# ---------------------------------------------------------------------------
# Hybrid mode: weight flow end-to-end
# ---------------------------------------------------------------------------


class TestHybridWeightFlow:
    @pytest.mark.asyncio
    async def test_weight_increases_after_wins(self):
        learner, _ = _make_learner(mode="hybrid")
        learner._sync_strategy_weights = AsyncMock()
        learner._periodic_retrain_if_needed = AsyncMock()

        for _ in range(10):
            await learner.on_trade_closed(
                "BTC/USDT", "buy", 0.02, strategy="ema_crossover"
            )

        weight = learner._compute_weight("ema_crossover")
        assert weight > 1.0

    @pytest.mark.asyncio
    async def test_weight_decreases_after_losses(self):
        learner, _ = _make_learner(mode="hybrid")
        learner._sync_strategy_weights = AsyncMock()
        learner._periodic_retrain_if_needed = AsyncMock()

        for _ in range(10):
            await learner.on_trade_closed(
                "SOL/USDT", "sell", -0.03, strategy="rsi_momentum"
            )

        weight = learner._compute_weight("rsi_momentum")
        assert weight < 1.0

    def test_get_strategy_weight_neutral_before_5_trades(self):
        learner, _ = _make_learner(mode="hybrid")
        for _ in range(4):
            learner._record_strategy_result("ema_crossover", True)
        assert learner.get_strategy_weight("ema_crossover") == 1.0

    def test_weight_floor_after_many_losses(self):
        learner, _ = _make_learner(mode="hybrid")
        for _ in range(20):
            learner._record_strategy_result("ema_crossover", False)
        assert learner.get_strategy_weight("ema_crossover") == 0.5

    def test_weight_ceil_after_many_wins(self):
        learner, _ = _make_learner(mode="hybrid")
        for _ in range(20):
            learner._record_strategy_result("ema_crossover", True)
        assert learner.get_strategy_weight("ema_crossover") == 1.5
