// Build-time prerender entry (no browser involved). Built via
// `vite build --ssr` and consumed by scripts/prerender.mjs, which renders the
// public routes to static HTML so crawlers get real content instead of an
// empty #root. The client still mounts with createRoot, which simply replaces
// the prerendered children — no hydration, so no mismatch risk.
import { renderToString } from 'react-dom/server';
import { StaticRouter } from 'react-router-dom/server';
import { ThemeProvider } from './context/ThemeContext';
import { InnerApp } from './App';

export { PRERENDER_ROUTES, SITE_ORIGIN, getRouteSeo } from './lib/seo';

export function render(url: string): string {
  return renderToString(
    <ThemeProvider>
      <StaticRouter location={url}>
        <InnerApp />
      </StaticRouter>
    </ThemeProvider>
  );
}
