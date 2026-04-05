type AnalyticsPrimitive = string | number | boolean;
type AnalyticsParamValue = AnalyticsPrimitive | null | undefined;

export type AnalyticsParams = Record<string, AnalyticsParamValue>;

const FIREBASE_APP_SCRIPT = 'https://www.gstatic.com/firebasejs/10.12.2/firebase-app-compat.js';
const FIREBASE_ANALYTICS_SCRIPT = 'https://www.gstatic.com/firebasejs/10.12.2/firebase-analytics-compat.js';
const MAX_EVENT_NAME_LENGTH = 40;
const CONSENT_STORAGE_KEY = 'sb_firebase_analytics_consent_v1';

const FIREBASE_CONFIG = {
  apiKey: (import.meta.env.VITE_FIREBASE_API_KEY || '').trim(),
  authDomain: (import.meta.env.VITE_FIREBASE_AUTH_DOMAIN || '').trim(),
  projectId: (import.meta.env.VITE_FIREBASE_PROJECT_ID || '').trim(),
  appId: (import.meta.env.VITE_FIREBASE_APP_ID || '').trim(),
  measurementId: (import.meta.env.VITE_FIREBASE_MEASUREMENT_ID || '').trim(),
  messagingSenderId: (import.meta.env.VITE_FIREBASE_MESSAGING_SENDER_ID || '').trim(),
};

const IS_FIREBASE_CONFIGURED = Object.values(FIREBASE_CONFIG).every(Boolean);

type ConsentValue = 'granted' | 'denied';

export interface AnalyticsConsent {
  analytics_storage: ConsentValue;
  ad_storage: ConsentValue;
  ad_user_data: ConsentValue;
  ad_personalization: ConsentValue;
  source?: string;
  updated_at?: string;
}

type AnalyticsStatusReason =
  | 'idle'
  | 'non_browser'
  | 'non_production'
  | 'do_not_track'
  | 'missing_config'
  | 'script_load_failed'
  | 'firebase_global_missing'
  | 'analytics_init_failed'
  | 'ready';

interface FirebaseAnalyticsCompat {
  logEvent: (name: string, params?: Record<string, unknown>) => void;
  setUserId?: (id: string | null) => void;
  setUserProperties?: (properties: Record<string, unknown>) => void;
}

declare global {
  interface Window {
    doNotTrack?: string;
    __sbAnalyticsStatus?: {
      ready: boolean;
      reason: AnalyticsStatusReason;
      details?: Record<string, unknown>;
      updatedAt: string;
    };
    firebase?: {
      apps?: Array<unknown>;
      initializeApp: (config: Record<string, string>) => unknown;
      analytics: () => FirebaseAnalyticsCompat;
    };
  }
}

const DEFAULT_DENIED_CONSENT: AnalyticsConsent = {
  analytics_storage: 'denied',
  ad_storage: 'denied',
  ad_user_data: 'denied',
  ad_personalization: 'denied',
};

let initialized = false;
let ready = false;
let bootPromise: Promise<boolean> | null = null;
let currentConsent: AnalyticsConsent = { ...DEFAULT_DENIED_CONSENT };
let pendingUserId: string | null = null;
let pendingUserProperties: Record<string, AnalyticsPrimitive> = {};
let pendingOperations: Array<(analytics: FirebaseAnalyticsCompat) => void> = [];

function debugEnabled(): boolean {
  if (typeof window === 'undefined') {
    return false;
  }
  try {
    const params = new URLSearchParams(window.location.search);
    return params.get('analytics_debug') === '1' || window.localStorage.getItem('sb_analytics_debug') === '1';
  } catch {
    return false;
  }
}

function setStatus(
  reason: AnalyticsStatusReason,
  isReady: boolean,
  details?: Record<string, unknown>,
): void {
  if (typeof window === 'undefined') {
    return;
  }
  window.__sbAnalyticsStatus = {
    ready: isReady,
    reason,
    details,
    updatedAt: new Date().toISOString(),
  };

  if (!debugEnabled()) {
    return;
  }
  const logger = isReady ? console.info : console.warn;
  logger('[firebase-analytics]', reason, details ?? {});
}

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
    return normalizeConsent(JSON.parse(raw) as Partial<AnalyticsConsent>);
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
    // Ignore storage failures.
  }
}

function hasConfig(): boolean {
  return IS_FIREBASE_CONFIGURED;
}

function missingConfigKeys(): string[] {
  return Object.entries(FIREBASE_CONFIG)
    .filter(([, value]) => !value)
    .map(([key]) => key);
}

function doNotTrackEnabled(): boolean {
  if (typeof window === 'undefined') {
    return true;
  }
  const value =
    navigator.doNotTrack
    ?? window.doNotTrack
    ?? (navigator as Navigator & { msDoNotTrack?: string }).msDoNotTrack
    ?? '0';
  return value === '1' || value.toLowerCase() === 'yes';
}

function loadScript(src: string): Promise<void> {
  return new Promise((resolve, reject) => {
    const existing = document.querySelector<HTMLScriptElement>(`script[src="${src}"]`);
    if (existing?.dataset.ready === 'true') {
      resolve();
      return;
    }
    if (existing) {
      existing.addEventListener('load', () => resolve(), { once: true });
      existing.addEventListener('error', () => reject(new Error(`Failed to load ${src}`)), { once: true });
      return;
    }

    const script = document.createElement('script');
    script.async = true;
    script.src = src;
    script.addEventListener(
      'load',
      () => {
        script.dataset.ready = 'true';
        resolve();
      },
      { once: true },
    );
    script.addEventListener('error', () => reject(new Error(`Failed to load ${src}`)), { once: true });
    document.head.appendChild(script);
  });
}

function resolvedConfig(): Record<string, string> {
  return { ...FIREBASE_CONFIG };
}

async function bootFirebaseAnalytics(): Promise<boolean> {
  if (typeof window === 'undefined') {
    setStatus('non_browser', false);
    return false;
  }

  if (!import.meta.env.PROD) {
    setStatus('non_production', false);
    return false;
  }

  if (!hasConfig()) {
    setStatus('missing_config', false, { missingKeys: missingConfigKeys() });
    return false;
  }

  if (doNotTrackEnabled()) {
    setStatus('do_not_track', false, {
      navigatorDoNotTrack: navigator.doNotTrack ?? null,
      windowDoNotTrack: window.doNotTrack ?? null,
    });
    return false;
  }

  if (bootPromise) {
    return bootPromise;
  }

  bootPromise = (async () => {
    try {
      await loadScript(FIREBASE_APP_SCRIPT);
      await loadScript(FIREBASE_ANALYTICS_SCRIPT);
    } catch (error) {
      setStatus('script_load_failed', false, {
        error: error instanceof Error ? error.message : 'Script load failed.',
      });
      return false;
    }

    if (!window.firebase) {
      setStatus('firebase_global_missing', false);
      return false;
    }

    try {
      if (!window.firebase.apps || window.firebase.apps.length === 0) {
        window.firebase.initializeApp(resolvedConfig());
      }
      window.firebase.analytics();
    } catch (error) {
      setStatus('analytics_init_failed', false, {
        error: error instanceof Error ? error.message : 'Analytics initialization failed.',
      });
      return false;
    }

    ready = true;
    setStatus('ready', true, { measurementId: FIREBASE_CONFIG.measurementId });
    flushPendingOperations();
    return true;
  })().catch((error) => {
    setStatus('analytics_init_failed', false, {
      error: error instanceof Error ? error.message : 'Analytics initialization failed.',
    });
    return false;
  });

  return bootPromise;
}

function analyticsAllowed(): boolean {
  return currentConsent.analytics_storage === 'granted';
}

function getAnalyticsInstance(): FirebaseAnalyticsCompat | null {
  if (!ready || typeof window === 'undefined' || !window.firebase) {
    return null;
  }
  try {
    return window.firebase.analytics();
  } catch {
    return null;
  }
}

function flushPendingOperations(): void {
  const analytics = getAnalyticsInstance();
  if (!analytics || !analyticsAllowed()) {
    return;
  }

  if (pendingUserId !== null || Object.keys(pendingUserProperties).length > 0) {
    analytics.setUserId?.(pendingUserId);
    if (Object.keys(pendingUserProperties).length > 0) {
      analytics.setUserProperties?.(pendingUserProperties);
    }
  }

  const operations = pendingOperations;
  pendingOperations = [];
  for (const operation of operations) {
    try {
      operation(analytics);
    } catch {
      // Keep analytics failures isolated from product flows.
    }
  }
}

function enqueueOperation(operation: (analytics: FirebaseAnalyticsCompat) => void): void {
  if (!analyticsAllowed()) {
    return;
  }

  const analytics = getAnalyticsInstance();
  if (analytics) {
    try {
      operation(analytics);
    } catch {
      // Ignore analytics failures.
    }
    return;
  }

  pendingOperations.push(operation);
  void bootFirebaseAnalytics();
}

export function isAnalyticsEnabled(): boolean {
  return hasConfig();
}

export function initAnalytics(): boolean {
  if (typeof window === 'undefined' || !hasConfig()) {
    return false;
  }

  if (initialized) {
    return true;
  }

  const storedConsent = readStoredConsent();
  currentConsent = storedConsent ?? { ...DEFAULT_DENIED_CONSENT };
  initialized = true;

  if (analyticsAllowed()) {
    void bootFirebaseAnalytics();
  }

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
  source = 'ui',
): AnalyticsConsent {
  const normalized = normalizeConsent({
    ...currentConsent,
    ...consent,
    source,
    updated_at: new Date().toISOString(),
  });

  currentConsent = normalized;
  writeStoredConsent(normalized);

  if (!initialized) {
    initAnalytics();
  }

  if (!analyticsAllowed()) {
    pendingOperations = [];
    pendingUserId = null;
    pendingUserProperties = {};
    return normalized;
  }

  void bootFirebaseAnalytics();
  return normalized;
}

export function trackPageView(pagePath: string): void {
  if (!initAnalytics()) {
    return;
  }

  enqueueOperation((analytics) => {
    analytics.logEvent('page_view', {
      page_path: pagePath,
      page_title: document.title,
      page_location: window.location.href,
    });
  });
}

export function trackEvent(eventName: string, params?: AnalyticsParams): void {
  if (!initAnalytics()) {
    return;
  }

  const normalizedName = normalizeEventName(eventName);
  const normalizedParams = normalizeParams(params);
  enqueueOperation((analytics) => {
    analytics.logEvent(normalizedName, normalizedParams);
  });
}

export function setAnalyticsUser(
  userId: string | number,
  userProperties?: AnalyticsParams,
): void {
  if (!initAnalytics()) {
    return;
  }

  pendingUserId = String(userId);
  pendingUserProperties = normalizeParams(userProperties);
  enqueueOperation((analytics) => {
    analytics.setUserId?.(pendingUserId);
    if (Object.keys(pendingUserProperties).length > 0) {
      analytics.setUserProperties?.(pendingUserProperties);
    }
  });
}

export function clearAnalyticsUser(): void {
  if (!initAnalytics()) {
    return;
  }

  pendingUserId = null;
  pendingUserProperties = {};
  enqueueOperation((analytics) => {
    analytics.setUserId?.(null);
    analytics.setUserProperties?.({});
  });
}
