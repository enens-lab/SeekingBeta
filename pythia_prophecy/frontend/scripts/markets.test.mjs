// Decode + render-helper tests for the sports Markets UI (spec v1).
// Run: npm test   (or: node --test scripts/markets.test.mjs)
//
// The repo has no unit-test runner, so this bundles the two TypeScript modules
// with esbuild (already a Vite dependency) and runs them under node:test.
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import test from 'node:test';
import { fileURLToPath } from 'node:url';
import { build } from 'esbuild';

const frontendRoot = join(dirname(fileURLToPath(import.meta.url)), '..');
const fixture = JSON.parse(readFileSync(join(frontendRoot, 'scripts', 'fixtures', 'markets_fixture.json'), 'utf8'));

const bundle = await build({
  stdin: {
    contents: [
      "export * from './src/api/client.ts';",
      "export * from './src/components/sports/marketsFormat.ts';",
    ].join('\n'),
    resolveDir: frontendRoot,
    loader: 'ts',
  },
  bundle: true,
  format: 'esm',
  platform: 'node',
  write: false,
  logLevel: 'silent',
});
const m = await import(`data:text/javascript;base64,${Buffer.from(bundle.outputFiles[0].text).toString('base64')}`);

const clone = (value) => JSON.parse(JSON.stringify(value));
const decoded = m.normalizeSportsBoardsResponse(clone(fixture));
const footballBoard = decoded.football.upcoming[0];
const soccerBoard = decoded.soccer.upcoming[0];

test('fixture: every known market decodes', () => {
  assert.deepEqual(
    footballBoard.markets.map((pick) => pick.type),
    ['spread', 'spread', 'total', 'team_total_home', 'moneyline', 'player_prop_test'],
  );
  assert.deepEqual(
    soccerBoard.markets.map((pick) => pick.type),
    ['asian_handicap', 'total', 'btts', 'double_chance', 'correct_score'],
  );
  const spread = footballBoard.markets[0];
  assert.equal(spread.label, 'BUF -2.5');
  assert.equal(spread.line, -2.5);
  assert.equal(spread.modelLine, -3);
  assert.equal(spread.modelProbability, 0.5184);
  assert.equal(spread.market.americanOdds, -110);
  assert.equal(spread.attribution, 'Lines: nflverse (CC-BY-4.0)');
  assert.equal(spread.basis, 'market');
  assert.equal(soccerBoard.markets[0].line, -0.75);
  assert.equal(soccerBoard.markets[0].outcomeProbabilities.half_win, 0.2344);
});

test('fixture: history markets, final score, market summary', () => {
  const history = decoded.football.backtests[0];
  assert.equal(history.homeScore, 27);
  assert.equal(history.awayScore, 24);
  assert.deepEqual(history.markets.map((pick) => pick.result), ['win', 'push', 'loss']);
  assert.equal(history.hitStatus, 'Top Pick');
  assert.deepEqual(decoded.football.marketSummary.map((row) => row.type), ['spread', 'total']);
  assert.equal(decoded.football.marketSummary[1].winRateExPush, null);
  assert.equal(decoded.soccer.marketSummary[0].graded, 12);
  assert.deepEqual(decoded.golf.marketSummary, []);
});

test('unknown type "player_prop_test" is skipped at render', () => {
  const rows = m.visibleMarkets(footballBoard.markets);
  assert.ok(!rows.some((pick) => pick.type === 'player_prop_test'));
  assert.deepEqual(
    rows.map((pick) => m.marketRowTitle(pick)),
    [
      'Spread · BUF -2.5',
      'Spread · KC +2.5',
      'Total · Over 47.5',
      'Moneyline · BUF moneyline',
      'Team total · BUF over 24.5',
    ],
  );
  assert.equal(m.marketHeading('player_prop_test'), null);
  const badges = m.historyMarketBadges(decoded.football.backtests[0].markets);
  assert.deepEqual(badges.map((badge) => badge.text), ['ATL -2.5: Win', 'Over 51.5: Push']);
  assert.ok(!badges.some((badge) => badge.text.includes('skip me')));
});

test('payload without markets / marketSummary still decodes', () => {
  const legacy = clone(fixture);
  for (const collection of Object.values(legacy)) {
    delete collection.marketSummary;
    delete collection.backtestLabel;
    for (const board of collection.upcoming) delete board.markets;
    for (const board of collection.backtests) {
      delete board.markets;
      delete board.homeScore;
      delete board.awayScore;
    }
  }
  const result = m.normalizeSportsBoardsResponse(legacy);
  assert.deepEqual(result.football.upcoming[0].markets, []);
  assert.deepEqual(result.football.backtests[0].markets, []);
  assert.equal(result.football.backtests[0].homeScore, undefined);
  assert.deepEqual(result.football.marketSummary, []);
  assert.equal(result.football.backtestLabel, undefined);
  // Untouched fields pass through.
  assert.equal(result.football.upcoming[0].predictions[0].winProbability, 58);
  assert.equal(result.football.backtests[0].hitStatus, 'Top Pick');
  assert.deepEqual(m.visibleMarkets(result.football.upcoming[0].markets), []);
  assert.deepEqual(m.visibleMarketSummaries(result.football.marketSummary), []);
  assert.equal(m.finalScoreLabel(undefined, undefined), null);
});

test('malformed elements are skipped, the rest survive', () => {
  const broken = clone(fixture);
  const markets = broken.football.upcoming[0].markets;
  markets.push(null, 'nope', 42, { type: 'spread', label: 'no probability' });
  markets.push({ type: 7, side: ['x'], result: { nested: true }, modelProbability: 0.4, market: 'bad' });
  markets.push({ type: 'spread', label: 'string number', modelProbability: '0.61', line: '-1.5', market: { americanOdds: 'abc' } });
  broken.football.marketSummary = [null, { type: 'spread', season: 2026, graded: '9', wins: 'x' }];
  broken.soccer.upcoming[0].markets = { not: 'an array' };
  const result = m.normalizeSportsBoardsResponse(broken);
  const types = result.football.upcoming[0].markets.map((pick) => pick.type);
  assert.equal(types.length, 8);
  assert.equal(result.football.upcoming[0].markets[6].type, undefined);
  assert.equal(result.football.upcoming[0].markets[6].side, undefined);
  assert.equal(result.football.upcoming[0].markets[6].market, undefined);
  assert.equal(result.football.upcoming[0].markets[7].modelProbability, 0.61);
  assert.equal(result.football.upcoming[0].markets[7].line, -1.5);
  assert.equal(result.football.upcoming[0].markets[7].market.americanOdds, undefined);
  assert.equal(result.football.marketSummary.length, 1);
  assert.equal(result.football.marketSummary[0].type, 'spread');
  assert.equal(result.football.marketSummary[0].season, undefined);
  assert.equal(result.football.marketSummary[0].graded, 9);
  assert.equal(result.football.marketSummary[0].wins, undefined);
  assert.deepEqual(result.soccer.upcoming[0].markets, []);
  assert.equal(m.normalizeSportsBoardsResponse(null), null);
});

test('row secondary line and footer follow the spec', () => {
  const [buf, , over, moneyline, teamTotal] = m.visibleMarkets(footballBoard.markets);
  assert.equal(m.formatMarketPercent(buf.modelProbability), '52%');
  assert.equal(m.marketSecondaryLine(buf), 'Fair -3 · Line -2.5 (-110) · Market');
  assert.equal(m.marketSecondaryLine(over), 'Fair 48.5 · Line 47.5 (-108) · Market');
  assert.equal(m.marketSecondaryLine(moneyline), 'Model view');
  assert.equal(m.marketSecondaryLine(teamTotal), 'Fair 25 · Market');
  const soccerRows = m.visibleMarkets(soccerBoard.markets);
  assert.deepEqual(soccerRows.map((pick) => pick.type), ['asian_handicap', 'total', 'btts', 'double_chance', 'correct_score']);
  assert.equal(m.marketSecondaryLine(soccerRows[0]), 'Fair -0.5 · Model view');
  assert.equal(m.marketSecondaryLine(soccerRows[1]), 'Model view', 'equal fair line is hidden');
  assert.equal(m.marketSecondaryLine({ modelProbability: 0.5, pushProbability: 0.004 }), 'Model view');
  assert.equal(m.marketSecondaryLine({ modelProbability: 0.5, pushProbability: 0.062, basis: 'market' }), 'Push 6% · Market');
  assert.deepEqual(m.marketAttributions(m.visibleMarkets(footballBoard.markets)), ['Lines: nflverse (CC-BY-4.0)']);
  assert.deepEqual(m.marketAttributions(soccerRows), []);
  assert.equal(
    m.MARKETS_DISCLAIMER,
    'Probabilities are model estimates for information only. SeekingBeta does not accept, place or facilitate bets.',
  );
  assert.equal(m.formatSignedNumber(150), '+150');
  assert.equal(m.formatSignedNumber(0), '0');
});

test('history badges, final score and market record lines', () => {
  const soccerHistory = decoded.soccer.backtests[0];
  assert.deepEqual(
    m.historyMarketBadges(soccerHistory.markets).map((badge) => [badge.text, badge.tone]),
    [
      ['Everton -0.25: Half loss', 'negative'],
      ['Both teams to score: Yes: Win', 'positive'],
    ],
  );
  assert.equal(m.finalScoreLabel(soccerHistory.awayScore, soccerHistory.homeScore), 'Final 1-1');
  assert.equal(m.finalScoreLabel(24, 27), 'Final 24-27');
  assert.equal(m.marketResultLabel('void'), 'Void');
  assert.equal(m.marketResultTone('void'), 'neutral');
  assert.equal(m.marketResultLabel('cancelled'), null);

  const [spread, total] = m.visibleMarketSummaries(decoded.football.marketSummary);
  assert.equal(m.marketRecordLine(spread), 'Spread 2026-27: 60.5-49.5-4');
  assert.equal(
    m.marketRecordRateLine(spread),
    '55.0% excl. pushes · break-even 52.4% · n=114 · +4.6u at stated prices',
  );
  assert.equal(m.marketRecordLine(total), 'Total 2026-27: 20-20-1');
  assert.equal(m.marketRecordRateLine(total), 'Rate shown after 100 graded picks · n=41');
  assert.equal(m.marketRecordLine({ ...spread, voids: 2 }), 'Spread 2026-27: 60.5-49.5-4 · 2 void');
  assert.equal(
    m.marketRecordRateLine({ ...spread, unitsAtStatedPrice: -3.25 }),
    '55.0% excl. pushes · break-even 52.4% · n=114 · -3.3u at stated prices',
  );
  const [btts] = m.visibleMarketSummaries(decoded.soccer.marketSummary);
  assert.equal(m.marketRecordLine(btts), 'Both teams to score 2026-27: 7-5-0');
  assert.equal(m.marketRecordRateLine(btts), 'Rate shown after 100 graded picks · n=12');
  assert.equal(m.MARKET_RECORD_CAPTION, 'Live record: graded against the line shown when each pick was published.');
});

test('no banned terms in any rendered market string', () => {
  const banned = /\b(lock|guaranteed|sure|edge|value|best bet)\b|\+EV/i;
  const strings = [m.MARKETS_DISCLAIMER, m.MARKET_RECORD_CAPTION, ...Object.values(m.MARKET_HEADINGS)];
  for (const board of [footballBoard, soccerBoard]) {
    for (const pick of m.visibleMarkets(board.markets)) {
      strings.push(m.marketRowTitle(pick), m.marketSecondaryLine(pick), m.formatMarketPercent(pick.modelProbability));
    }
  }
  for (const collection of [decoded.football, decoded.soccer]) {
    for (const row of m.visibleMarketSummaries(collection.marketSummary)) {
      strings.push(m.marketRecordLine(row), m.marketRecordRateLine(row));
    }
    for (const board of collection.backtests) {
      strings.push(...m.historyMarketBadges(board.markets).map((badge) => badge.text));
    }
  }
  for (const text of strings) assert.doesNotMatch(text, banned, text);
});

test('basis labels: market, market-implied, model view', () => {
  assert.equal(m.marketBasisLabel('market'), 'Market');
  assert.equal(m.marketBasisLabel(' Market_Implied '), 'Market-implied');
  assert.equal(m.marketBasisLabel('model'), 'Model view');
  assert.equal(m.marketBasisLabel(undefined), 'Model view');
});
