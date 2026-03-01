type AnalyticsPrimitive = string | number | boolean;
type AnalyticsParamValue = AnalyticsPrimitive | null | undefined;

export type AnalyticsParams = Record<string, AnalyticsParamValue>;

const GA4_MEASUREMENT_ID = (import.meta.env.VITE_GA4_MEASUREMENT_ID || '').trim();
const IS_GA4_ENABLED = Boolean(GA4_MEASUREMENT_ID);
const MAX_EVENT_NAME_LENGTH = 40;

declare global {
  interface Window {
    dataLayer: unknown[];
    gtag?: (...args: unknown[]) => void;
  }
}

let initialized = false;
let scriptScheduled = false;

function normalizeEventName(eventName: string): string {
  const normalized = eventName
    .toLowerCase()
    .replace(/[^a-z0-9_]+/g, '_')
    .replace(/_+/g, '_')
    .replace(/^_+|_+$/g, '');

  const baseName = normalized || 'event_unnamed';
  const ensuredPrefix = /^[a-z]/.test(baseName) ? baseName : `event_${baseName}`;
  return ensuredPrefix.slice(0, MAX_EVENT_NAME_LENGTH);
}

function normalizeParams(params?: AnalyticsParams): Record<string, AnalyticsPrimitive> {
  if (!params) {
    return {};
  }

  const result: Record<string, AnalyticsPrimitive> = {};
  for (const [key, value] of Object.entries(params)) {
    if (value === null || value === undefined) {
      continue;
    }
    if (typeof value === 'string') {
      result[key] = value.slice(0, 100);
      continue;
    }
    if (typeof value === 'number') {
      if (!Number.isFinite(value)) {
        continue;
      }
      result[key] = value;
      continue;
    }
    if (typeof value === 'boolean') {
      result[key] = value;
    }
  }
  return result;
}

function ensureGtagStub(): void {
  window.dataLayer = window.dataLayer || [];
  if (!window.gtag) {
    window.gtag = (...args: unknown[]) => {
      window.dataLayer.push(args);
    };
  }
}

function gtagCall(...args: unknown[]): void {
  window.gtag?.(...args);
}

function scheduleScriptLoad(): void {
  if (scriptScheduled) {
    return;
  }
  scriptScheduled = true;

  const loadScript = () => {
    if (document.querySelector(`script[data-ga4-id="${GA4_MEASUREMENT_ID}"]`)) {
      return;
    }
    const script = document.createElement('script');
    script.async = true;
    script.src = `https://www.googletagmanager.com/gtag/js?id=${encodeURIComponent(GA4_MEASUREMENT_ID)}`;
    script.dataset.ga4Id = GA4_MEASUREMENT_ID;
    document.head.appendChild(script);
  };

  const idleWindow = window as Window & {
    requestIdleCallback?: (callback: IdleRequestCallback, options?: IdleRequestOptions) => number;
  };
  if (typeof idleWindow.requestIdleCallback === 'function') {
    idleWindow.requestIdleCallback(() => loadScript(), { timeout: 2000 });
    return;
  }
  window.setTimeout(loadScript, 1);
}

export function isAnalyticsEnabled(): boolean {
  return IS_GA4_ENABLED;
}

export function initAnalytics(): boolean {
  if (!IS_GA4_ENABLED || typeof window === 'undefined') {
    return false;
  }
  if (initialized) {
    return true;
  }

  ensureGtagStub();
  gtagCall('js', new Date());
  gtagCall('config', GA4_MEASUREMENT_ID, {
    send_page_view: false,
    anonymize_ip: true,
    transport_type: 'beacon',
  });
  scheduleScriptLoad();
  initialized = true;
  return true;
}

export function trackPageView(pagePath: string): void {
  if (!initAnalytics()) {
    return;
  }
  gtagCall('event', 'page_view', {
    page_path: pagePath,
    page_title: document.title,
    page_location: window.location.href,
  });
}

export function trackEvent(eventName: string, params?: AnalyticsParams): void {
  if (!initAnalytics()) {
    return;
  }
  gtagCall('event', normalizeEventName(eventName), normalizeParams(params));
}

export function setAnalyticsUser(
  userId: string | number,
  userProperties?: AnalyticsParams
): void {
  if (!initAnalytics()) {
    return;
  }

  gtagCall('set', { user_id: String(userId) });
  if (userProperties && Object.keys(userProperties).length > 0) {
    gtagCall('set', 'user_properties', normalizeParams(userProperties));
  }
}

export function clearAnalyticsUser(): void {
  if (!initAnalytics()) {
    return;
  }
  gtagCall('set', { user_id: undefined });
  gtagCall('set', 'user_properties', {});
}
