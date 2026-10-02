/**
 * Pure helpers for the sports Markets UI (Markets UI spec v1, shared with the
 * iOS and Android clients). No React here so scripts/markets.test.mjs can
 * exercise every string the UI renders.
 */
import type { SportsMarketPick, SportsMarketSummary } from '../../api/client';

// Known market types and their headings. Any other type is skipped silently.
export const MARKET_HEADINGS: Readonly<Record<string, string>> = {
  moneyline: 'Moneyline',
  spread: 'Spread',
  run_line: 'Run line',
  total: 'Total',
  team_total_home: 'Team total',
  team_total_away: 'Team total',
  asian_handicap: 'Asian handicap',
  btts: 'Both teams to score',
  draw_no_bet: 'Draw no bet',
  double_chance: 'Double chance',
  correct_score: 'Correct score',
  '1x2': 'Match result',
};

export const MARKET_DISPLAY_ORDER: readonly string[] = [
  'spread',
  'run_line',
  'asian_handicap',
  'total',
  'moneyline',
  'btts',
  'team_total_home',
  'team_total_away',
  'double_chance',
  'draw_no_bet',
  'correct_score',
  '1x2',
];

export const MARKETS_COLLAPSED_COUNT = 3;

export const MARKETS_DISCLAIMER =
  'Probabilities are model estimates for information only. SeekingBeta does not accept, place or facilitate bets.';

export const MARKET_RECORD_CAPTION = 'Live record: graded against the line shown when each pick was published.';

const MARKET_RESULT_LABELS: Readonly<Record<string, string>> = {
  win: 'Win',
  half_win: 'Half win',
  push: 'Push',
  half_loss: 'Half loss',
  loss: 'Loss',
  void: 'Void',
};

export type MarketResultTone = 'positive' | 'negative' | 'neutral';

const MARKET_RESULT_TONES: Readonly<Record<string, MarketResultTone>> = {
  win: 'positive',
  half_win: 'positive',
  push: 'neutral',
  half_loss: 'negative',
  loss: 'negative',
  void: 'neutral',
};

function isKnownMarketType(type?: string): type is string {
  return typeof type === 'string' && Object.prototype.hasOwnProperty.call(MARKET_HEADINGS, type);
}

export function marketHeading(type?: string): string | null {
  return isKnownMarketType(type) ? MARKET_HEADINGS[type] : null;
}

/** Known types only, in display order; same type keeps server order. */
export function visibleMarkets(markets?: SportsMarketPick[] | null): SportsMarketPick[] {
  if (!Array.isArray(markets)) return [];
  return markets
    .map((pick, index) => ({ pick, index }))
    .filter(({ pick }) => isKnownMarketType(pick.type))
    .sort((a, b) => {
      const byType = MARKET_DISPLAY_ORDER.indexOf(a.pick.type as string) - MARKET_DISPLAY_ORDER.indexOf(b.pick.type as string);
      return byType !== 0 ? byType : a.index - b.index;
    })
    .map(({ pick }) => pick);
}

function trimNumber(num: number, maxDecimals: number): string {
  const fixed = num.toFixed(maxDecimals);
  const trimmed = fixed.includes('.') ? fixed.replace(/0+$/, '').replace(/\.$/, '') : fixed;
  return trimmed === '-0' ? '0' : trimmed;
}

/** Lines and prices with an explicit sign: -3.5, +2.5, -0.75, -110, +150. */
export function formatSignedNumber(num: number): string {
  const text = trimNumber(num, 2);
  return num > 0 && text !== '0' ? `+${text}` : text;
}

/** Whole percent, e.g. 0.5184 -> "52%". */
export function formatMarketPercent(probability: number): string {
  if (!Number.isFinite(probability)) return '—';
  return `${Math.round(Math.min(1, Math.max(0, probability)) * 100)}%`;
}

// Totals are a number of points or goals, not a handicap, so they carry no sign.
const UNSIGNED_LINE_TYPES = new Set(['total', 'team_total_home', 'team_total_away']);

function formatLine(type: string | undefined, num: number): string {
  return type && UNSIGNED_LINE_TYPES.has(type) ? trimNumber(num, 2) : formatSignedNumber(num);
}

export function marketRowTitle(pick: SportsMarketPick): string {
  const heading = marketHeading(pick.type) ?? '';
  return pick.label ? `${heading} · ${pick.label}` : heading;
}

/** Small muted line under a market row; pieces are joined with " · ". */
export function marketSecondaryLine(pick: SportsMarketPick): string {
  const parts: string[] = [];
  if (typeof pick.modelLine === 'number') {
    const differs = typeof pick.line !== 'number' || Math.abs(pick.modelLine - pick.line) > 1e-9;
    if (differs) parts.push(`Fair ${formatLine(pick.type, pick.modelLine)}`);
  }
  if (typeof pick.pushProbability === 'number' && pick.pushProbability >= 0.01 && pick.pushProbability <= 1) {
    parts.push(`Push ${formatMarketPercent(pick.pushProbability)}`);
  }
  if (pick.market) {
    const price: string[] = [];
    if (typeof pick.market.line === 'number') price.push(formatLine(pick.type, pick.market.line));
    if (typeof pick.market.americanOdds === 'number') {
      price.push(`(${formatSignedNumber(Math.round(pick.market.americanOdds))})`);
    }
    if (price.length) parts.push(`Line ${price.join(' ')}`);
  }
  parts.push(pick.basis?.trim().toLowerCase() === 'market' ? 'Market' : 'Model view');
  return parts.join(' · ');
}

/** Distinct attributions across the rows, in first-seen order. */
export function marketAttributions(markets: SportsMarketPick[]): string[] {
  const seen: string[] = [];
  markets.forEach((pick) => {
    const attribution = pick.attribution?.trim();
    if (attribution && !seen.includes(attribution)) seen.push(attribution);
  });
  return seen;
}

export function marketResultLabel(result?: string): string | null {
  return result && Object.prototype.hasOwnProperty.call(MARKET_RESULT_LABELS, result) ? MARKET_RESULT_LABELS[result] : null;
}

export function marketResultTone(result?: string): MarketResultTone | null {
  return result && Object.prototype.hasOwnProperty.call(MARKET_RESULT_TONES, result) ? MARKET_RESULT_TONES[result] : null;
}

export type HistoryMarketBadge = {
  key: string;
  text: string;
  tone: MarketResultTone;
};

/** One badge per graded known-type market; unknown results are hidden. */
export function historyMarketBadges(markets?: SportsMarketPick[] | null): HistoryMarketBadge[] {
  const badges: HistoryMarketBadge[] = [];
  visibleMarkets(markets).forEach((pick, index) => {
    const label = marketResultLabel(pick.result);
    const tone = marketResultTone(pick.result);
    if (!label || !tone) return;
    badges.push({
      key: `${pick.marketId ?? pick.type}-${pick.side ?? ''}-${index}`,
      text: `${pick.label ?? marketHeading(pick.type)}: ${label}`,
      tone,
    });
  });
  return badges;
}

/** "Final {away}-{home}", or null unless both scores are present. */
export function finalScoreLabel(awayScore?: number | null, homeScore?: number | null): string | null {
  if (typeof awayScore !== 'number' || typeof homeScore !== 'number') return null;
  if (!Number.isFinite(awayScore) || !Number.isFinite(homeScore)) return null;
  return `Final ${trimNumber(awayScore, 1)}-${trimNumber(homeScore, 1)}`;
}

export function visibleMarketSummaries(rows?: SportsMarketSummary[] | null): SportsMarketSummary[] {
  if (!Array.isArray(rows)) return [];
  return rows.filter((row) => isKnownMarketType(row.type));
}

function formatRate(rate: number): string {
  return `${(rate * 100).toFixed(1)}%`;
}

function formatSignedUnits(units: number): string {
  const fixed = units.toFixed(1);
  if (Number(fixed) === 0) return '0.0';
  return units > 0 ? `+${fixed}` : fixed;
}

/** "{heading} {season}: {wins}-{losses}-{pushes}" plus " · {voids} void" when voids > 0. */
export function marketRecordLine(row: SportsMarketSummary): string {
  const heading = marketHeading(row.type) ?? '';
  const name = row.season ? `${heading} ${row.season}` : heading;
  const record = [row.wins, row.losses, row.pushes].map((count) => trimNumber(count ?? 0, 1)).join('-');
  const voids = row.voids ?? 0;
  return voids > 0 ? `${name}: ${record} · ${trimNumber(voids, 0)} void` : `${name}: ${record}`;
}

export function marketRecordRateLine(row: SportsMarketSummary): string {
  const graded = trimNumber(row.graded ?? 0, 0);
  if (typeof row.winRateExPush !== 'number') {
    return `Rate shown after 100 graded picks · n=${graded}`;
  }
  const parts = [`${formatRate(row.winRateExPush)} excl. pushes`];
  if (typeof row.breakEvenRate === 'number') parts.push(`break-even ${formatRate(row.breakEvenRate)}`);
  parts.push(`n=${graded}`);
  let line = parts.join(' · ');
  if (typeof row.unitsAtStatedPrice === 'number') {
    line += ` · ${formatSignedUnits(row.unitsAtStatedPrice)}u at stated prices`;
  }
  return line;
}
