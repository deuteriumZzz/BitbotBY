// Значок стратегии по её типу: тренд, возврат к среднему, импульс, объём, пробой, осциллятор,
// свечная модель, ликвидность (ICT), настроение. Явные — для узнаваемых; остальные — по имени.

import type { ReactNode } from "react";
import {
  Anchor,
  ArrowBendUpRight,
  ArrowLineUp,
  ArrowsCounterClockwise,
  ArrowsInLineVertical,
  ChartBarHorizontal,
  ChartLineUp,
  ChatsCircle,
  Cloud,
  Compass,
  Crosshair,
  Drop,
  Equals,
  Flame,
  Gauge,
  GitFork,
  Lightning,
  Pulse,
  Rocket,
  Scales,
  ShieldStar,
  SignOut,
  Spiral,
  Target,
  Timer,
  TrendUp,
  WaveSine,
  Waves,
} from "@phosphor-icons/react";

const BY_ID: Record<string, ReactNode> = {
  sentiment: <ChatsCircle />,
  context_signal: <Compass />,
  scalping: <Timer />,
  swing_trading: <Waves />,
  fibonacci_retracement: <Spiral />,
  ichimoku: <Cloud />,
  liquidity_sweep: <Drop />,
  fair_value_gap: <ArrowsInLineVertical />,
  inverse_fair_value_gap: <ArrowsInLineVertical />,
  amd: <Target />,
  breakout: <ArrowLineUp />,
  dual_thrust: <ArrowLineUp />,
  crash_bounce: <ArrowBendUpRight />,
  spike_reversal: <ArrowBendUpRight />,
  bear_hedge: <ShieldStar />,
  pump_exit: <SignOut />,
  momentum_burst: <Rocket />,
  support_resistance: <Equals />,
  poc_reversion: <Anchor />,
  ttm_squeeze: <Crosshair />,
  dual_momentum: <Scales />,
  gtaa_ma: <ChartLineUp />,
};

/** Значок для стратегии: свой, если есть, иначе по семейству из названия. */
export function strategyIcon(id: string): ReactNode {
  if (BY_ID[id]) return BY_ID[id];
  if (/divergence|cvd/.test(id)) return <GitFork />;
  if (/volume|obv|mfi|chaikin|klinger|ease_of|elder|ad_line|vwap|money_flow/.test(id)) return <ChartBarHorizontal />;
  if (/reversion|reentry|fade|bounce/.test(id)) return <ArrowsCounterClockwise />;
  if (/momentum|burst|force/.test(id)) return <Lightning />;
  if (/trend|supertrend|halftrend|adx|sar|ssl|chandelier|coral|zlsma|vortex|aroon|ema|macd|kst|waddah/.test(id)) return <TrendUp />;
  if (/rsi|stoch|williams|cci|kdj|rvi|fisher|awesome|coppock|ifr|osc/.test(id)) return <WaveSine />;
  if (/engulfing|shooting|heikin|star/.test(id)) return <Flame />;
  if (/bollinger|keltner|band|channel/.test(id)) return <Gauge />;
  return <Pulse />;
}
