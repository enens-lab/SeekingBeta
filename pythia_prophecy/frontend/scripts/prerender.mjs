// Static prerender: renders each public route with the SSR bundle and writes
// dist/<route>/index.html with route-specific title/meta/canonical baked in.
// Runs as the last step of `npm run build` (after the client + SSR builds).
//
// Serving:
//  - nginx: `try_files $uri $uri/ /index.html` picks up dist/<route>/index.html
//  - FastAPI dev fallback: serve_spa() in pythia_prophecy/api/service.py
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const frontendRoot = join(dirname(fileURLToPath(import.meta.url)), '..');
const distDir = join(frontendRoot, 'dist');

const { render, PRERENDER_ROUTES, SITE_ORIGIN, getRouteSeo } = await import(
  join(frontendRoot, 'dist-ssr', 'entry-prerender.js')
);

const template = readFileSync(join(distDir, 'index.html'), 'utf8');

const escapeAttr = (s) => s.replace(/&/g, '&amp;').replace(/"/g, '&quot;');
const escapeText = (s) => s.replace(/&/g, '&amp;').replace(/</g, '&lt;');

function replaceOne(html, pattern, replacement, label, route) {
  // Presence check rather than before/after comparison: for some routes the
  // rewrite is a no-op (e.g. the canonical for "/" already matches).
  const found = typeof pattern === 'string' ? html.includes(pattern) : pattern.test(html);
  if (!found) {
    throw new Error(`prerender: failed to rewrite ${label} for ${route} — index.html structure changed?`);
  }
  return html.replace(pattern, replacement);
}

function applySeo(html, route) {
  const seo = getRouteSeo(route);
  const canonical = new URL(seo.canonicalPath ?? route, SITE_ORIGIN).toString();
  const title = escapeText(seo.title);
  const titleAttr = escapeAttr(seo.title);
  const desc = escapeAttr(seo.description);

  html = replaceOne(html, /<title>[\s\S]*?<\/title>/, `<title>${title}</title>`, '<title>', route);
  html = replaceOne(
    html,
    /<meta\s+name="description"[\s\S]*?\/>/,
    `<meta name="description" content="${desc}" />`,
    'meta description', route,
  );
  html = replaceOne(
    html,
    /<link rel="canonical"[\s\S]*?\/>/,
    `<link rel="canonical" href="${canonical}" />`,
    'canonical', route,
  );
  html = replaceOne(
    html,
    /<meta property="og:url"[\s\S]*?\/>/,
    `<meta property="og:url" content="${canonical}" />`,
    'og:url', route,
  );
  html = replaceOne(
    html,
    /<meta property="og:title"[\s\S]*?\/>/,
    `<meta property="og:title" content="${titleAttr}" />`,
    'og:title', route,
  );
  html = replaceOne(
    html,
    /<meta\s+property="og:description"[\s\S]*?\/>/,
    `<meta property="og:description" content="${desc}" />`,
    'og:description', route,
  );
  html = replaceOne(
    html,
    /<meta name="twitter:title"[\s\S]*?\/>/,
    `<meta name="twitter:title" content="${titleAttr}" />`,
    'twitter:title', route,
  );
  html = replaceOne(
    html,
    /<meta\s+name="twitter:description"[\s\S]*?\/>/,
    `<meta name="twitter:description" content="${desc}" />`,
    'twitter:description', route,
  );
  return html;
}

for (const route of PRERENDER_ROUTES) {
  const appHtml = render(route);
  if (!appHtml || appHtml.length < 500) {
    throw new Error(`prerender: suspiciously empty render for ${route} (${appHtml.length} chars)`);
  }
  let html = replaceOne(
    template,
    '<div id="root"></div>',
    `<div id="root">${appHtml}</div>`,
    '#root', route,
  );
  html = applySeo(html, route);

  const outDir = route === '/' ? distDir : join(distDir, route.slice(1));
  mkdirSync(outDir, { recursive: true });
  writeFileSync(join(outDir, 'index.html'), html);
  console.log(`prerendered ${route} -> ${join(outDir, 'index.html').replace(frontendRoot + '/', '')} (${appHtml.length} chars of markup)`);
}
