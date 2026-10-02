/**
 * Dev-only harness: renders the real SportsDashboard against the markets test
 * fixture by stubbing window.fetch for /api/sports/boards. Nothing in src/
 * imports this file and it is not a Vite build input, so it never ships.
 *
 *   /dev/markets-harness.html           fixture as-is
 *   /dev/markets-harness.html?legacy=1  fixture with every markets field removed
 */
import React from 'react';
import ReactDOM from 'react-dom/client';
import { MemoryRouter } from 'react-router-dom';
import { ThemeProvider } from '../src/context/ThemeContext';
import { AuthProvider } from '../src/context/AuthContext';
import { ToastProvider } from '../src/components/Toast';
import SportsDashboard from '../src/pages/sports/SportsDashboard';
import fixture from '../scripts/fixtures/markets_fixture.json';
import '../src/styles/index.css';

type AnyRecord = Record<string, any>;

function boardsPayload(): AnyRecord {
  const payload: AnyRecord = JSON.parse(JSON.stringify(fixture));
  if (new URLSearchParams(window.location.search).get('legacy') === '1') {
    Object.values(payload).forEach((collection: AnyRecord) => {
      delete collection.marketSummary;
      delete collection.backtestLabel;
      collection.upcoming.forEach((board: AnyRecord) => delete board.markets);
      collection.backtests.forEach((board: AnyRecord) => {
        delete board.markets;
        delete board.homeScore;
        delete board.awayScore;
      });
    });
  } else {
    // Mirrors _attach_sports_markets in api/service.py, which labels any
    // collection that has backtests (the fixture predates that label).
    Object.values(payload).forEach((collection: AnyRecord) => {
      if (collection.backtests.length && !collection.backtestLabel) {
        collection.backtestLabel = 'Simulated backtest: the model re-run on past games it was not trained on.';
      }
    });
  }
  return payload;
}

const realFetch = window.fetch.bind(window);
window.fetch = async (input: RequestInfo | URL, init?: RequestInit) => {
  const url = typeof input === 'string' ? input : input instanceof URL ? input.href : input.url;
  if (url.includes('/api/sports/boards')) {
    return new Response(JSON.stringify(boardsPayload()), {
      status: 200,
      headers: { 'Content-Type': 'application/json' },
    });
  }
  if (url.includes('/api/')) {
    return new Response(JSON.stringify({ detail: 'markets harness: no backend' }), { status: 503 });
  }
  return realFetch(input, init);
};

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <ThemeProvider>
      <MemoryRouter initialEntries={['/sports-dashboard']}>
        <AuthProvider>
          <ToastProvider>
            <SportsDashboard />
          </ToastProvider>
        </AuthProvider>
      </MemoryRouter>
    </ThemeProvider>
  </React.StrictMode>
);
