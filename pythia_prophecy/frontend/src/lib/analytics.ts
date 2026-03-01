type AnalyticsPrimitive = string | number | boolean;
type AnalyticsParamValue = AnalyticsPrimitive | null | undefined;

export type AnalyticsParams = Record<string, AnalyticsParamValue>;

const GA4_MEASUREMENT_ID = (import.meta.env.VITE_GA4_MEASUREMENT_ID || '').trim();
const IS_GA4_ENABLED = Boolean(GA4_MEASUREMENT_ID);
const MAX_EVENT_NAME_LENGTH = 40;
const CONSENT_STORAGE_KEY = 'sb_ga4_consent_v1';

type ConsentValue = 'granted' | 'denied';

export interface AnalyticsConsent {
  analytics_storage: ConsentValue;
  ad_storage: ConsentValue;
  ad_user_data: ConsentValue;
  ad_personalization: ConsentValue;
  source?: string;
  updated_at?: string;
}

const DEFAULT_DENIED_CONSENT: AnalyticsConsent = {
  analytics_storage: 'denied',
  ad_storage: 'denied',
  ad_user_data: 'denied',
  ad_personalization: 'denied',
};

declare global {
  interface Window {
    dataLayer: unknown[];
    gtag?: (...args: unknown[]) => void;
  }
}

let initialized = false;
let scriptScheduled = false;
let currentConsent: AnalyticsConsent = { ...DEFAULT_DENIED_CONSENT };

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

function normalizeConsentValue(value: unknown): ConsentValue {
  return value === 'granted' ? 'granted' : 'denied';
}

function normalizeConsent(raw?: Partial<AnalyticsConsent> | null): AnalyticsConsent {
  return {
    analytics_storage: normalizeConsentValue(raw?.analytics_storage),
    ad_storage: normalizeConsentValue(raw?.ad_storage),
    ad_user_data: normalizeConsentValue(raw?.ad_user_data),
    ad_personalization: normalizeConsentValue(raw?.ad_personalization),
    source: typeof raw?.source === 'string' ? raw.source.slice(0, 64) : undefined,
    updated_at: typeof raw?.updated_at === 'string' ? raw.updated_at : undefined,
  };
}

function readStoredConsent(): AnalyticsConsent | null {
  if (typeof window === 'undefined') {
    return null;
  }
  try {
    const raw = window.localStorage.getItem(CONSENT_STORAGE_KEY);
    if (!raw) {
      return null;
    }
    const parsed = JSON.parse(raw) as Partial<AnalyticsConsent>;
    return normalizeConsent(parsed);
  } catch {
    return null;
  }
}

function writeStoredConsent(consent: AnalyticsConsent): void {
  if (typeof window === 'undefined') {
    return;
  }
  try {
    window.localStorage.setItem(CONSENT_STORAGE_KEY, JSON.stringify(consent));
  } catch {
    // Ignore storage failures (private mode / quota).
  }
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
  const storedConsent = readStoredConsent();
  currentConsent = storedConsent ?? { ...DEFAULT_DENIED_CONSENT };
  gtagCall('consent', 'default', {
    analytics_storage: currentConsent.analytics_storage,
    ad_storage: currentConsent.ad_storage,
    ad_user_data: currentConsent.ad_user_data,
    ad_personalization: currentConsent.ad_personalization,
  });
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

export function getAnalyticsConsent(): AnalyticsConsent {
  const stored = readStoredConsent();
  return stored ?? { ...currentConsent };
}

export function hasStoredAnalyticsConsent(): boolean {
  return readStoredConsent() !== null;
}

export function updateAnalyticsConsent(
  consent: Partial<AnalyticsConsent>,
  source = 'ui'
): AnalyticsConsent {
  const normalized = normalizeConsent({
    ...currentConsent,
    ...consent,
    source,
    updated_at: new Date().toISOString(),
  });
  currentConsent = normalized;
  writeStoredConsent(normalized);
  if (initAnalytics()) {
    gtagCall('consent', 'update', {
      analytics_storage: normalized.analytics_storage,
      ad_storage: normalized.ad_storage,
      ad_user_data: normalized.ad_user_data,
      ad_personalization: normalized.ad_personalization,
    });
  }
  return normalized;
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
