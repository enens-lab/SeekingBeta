/**
 * API client for SeekingBeta backend
 */

const API_BASE = '';

// ============================================================================
// Types
// ============================================================================

export interface User {
  id: number;
  email: string;
  first_name: string;
  last_name: string;
  tier: 'free' | 'basic' | 'pro';
  email_verified: boolean;
}

export interface AuthResponse {
  access_token: string;
  token_type: string;
  user: User;
}

export interface SignupData {
  email: string;
  password: string;
  first_name: string;
  last_name: string;
  tier?: string;
  accept_terms: boolean;
  accept_privacy: boolean;
  policy_version: string;
  marketing_opt_in?: boolean;
}

export interface LoginData {
  email: string;
  password: string;
}

export interface ChangePasswordData {
  current_password: string;
  new_password: string;
}

export interface DeleteAccountData {
  password: string;
  confirm_text: string;
}

export interface Tier {
  tier?: 'free' | 'basic' | 'pro';
  name: string;
  price: number;
  stocks_limit?: number;
  timeframes: string[];
  features?: string[];
}

export interface Prediction {
  ticker: string;
  horizon: string;
  signal: 'buy' | 'hold' | 'sell' | 'strong_buy' | 'avoid';
  prob_up: number | null;
  predicted_return: number | null;
  last_close: number | null;
  timestamp: string;
}

export interface OracleData {
  watchlist: string[];
  timeframes: string[];
  available_stocks: string[];
  available_timeframes: string[];
  available_categories: Record<string, string[]>;
}

export interface Company {
  ticker: string;
  name: string;
  sector: string;
  industry: string;
  market_cap: number | null;
  description: string;
}

export interface NewsItem {
  title: string;
  link: string;
  publisher: string;
  published: string;
}

export interface AnalysisResult {
  ticker: string;
  last_close: number | null;
  prob_up: number | null;
  signal: 'buy' | 'hold' | 'sell' | 'strong_buy' | 'avoid' | null;
  predicted_return: number | null;
  error?: string | null;
}

export interface AnalysisResponse {
  results: AnalysisResult[];
  metadata: {
    model: string;
    task: string;
    period: string;
    horizon: string;
    analyzed_at: string;
    requested?: number;
    successful?: number;
    failed?: number;
  };
}

export interface UserFeatures {
  tier: string;
  features: {
    export_csv: boolean;
  };
  limits: {
    max_stocks_per_request: number;
    max_historical_days: number;
    daily_requests: number | null;
    requests_used: number;
  };
}

export interface ModelsResponse {
  models: string[];
  tasks: string[];
}

export interface UniverseResponse {
  stocks: string[];
  categories: Record<string, string[]>;
}

export interface RegimePerformance {
  trades: number;
  wins: number;
  losses: number;
  win_rate: number | null;
  avg_return_net: number | null;
}

export interface TrackRecordSummary {
  source_file: string;
  as_of: string;
  sample_size: number;
  transaction_cost_bps: number;
  hit_rate: number | null;
  sharpe_ratio: number | null;
  max_drawdown: number | null;
  total_return_gross: number | null;
  total_return_net: number | null;
  benchmark_return: number | null;
  avg_trade_return_net: number | null;
  avg_holding_days: number | null;
  regime_breakdown: Record<string, RegimePerformance>;
  notes: string[];
}

export interface TrackRecordResponse {
  available: boolean;
  summary: TrackRecordSummary | null;
  message?: string;
}

export interface TrackRecordCurvePoint {
  date: string;
  model_value: number;
  benchmark_value: number | null;
}

export interface TrackRecordCurveResponse {
  available: boolean;
  series: TrackRecordCurvePoint[];
  model_label: string;
  benchmark_label: string;
  start_value: number | null;
  end_value: number | null;
  benchmark_end_value: number | null;
  message?: string;
}

export interface BillingStatus {
  billing_enabled: boolean;
  user_tier: 'free' | 'basic' | 'pro';
  effective_tier: 'free' | 'basic' | 'pro';
  plan_tier: 'free' | 'basic' | 'pro';
  subscription_status: string;
  cancel_at_period_end: boolean;
  current_period_end: string | null;
  legacy_grace_expires_at: string | null;
  stripe_customer_id: string | null;
  stripe_subscription_id: string | null;
  price_id: string | null;
  grace_active: boolean;
}

export interface BillingCheckoutSessionResponse {
  checkout_url: string;
  session_id: string;
}

export interface BillingPortalSessionResponse {
  portal_url: string;
}

export interface BillingChangeSubscriptionResponse {
  mode: 'updated' | 'checkout' | 'no_op';
  message: string;
  checkout_url: string | null;
}

type AuthErrorCallback = (() => void) | null;

// ============================================================================
// State
// ============================================================================

let onAuthErrorCallback: AuthErrorCallback = null;

// ============================================================================
// Token Management
// ============================================================================

export function setOnAuthError(callback: AuthErrorCallback): void {
  onAuthErrorCallback = callback;
}

export function getToken(): string | null {
  return localStorage.getItem('pythia_token');
}

export function setToken(token: string | null): void {
  if (token) {
    localStorage.setItem('pythia_token', token);
  } else {
    localStorage.removeItem('pythia_token');
  }
}

// ============================================================================
// Request Helper
// ============================================================================

interface RequestOptions extends Omit<RequestInit, 'headers'> {
  headers?: Record<string, string>;
}

async function request<T>(endpoint: string, options: RequestOptions = {}): Promise<T> {
  const token = getToken();
  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
    ...options.headers,
  };

  if (token) {
    headers['Authorization'] = `Bearer ${token}`;
  }

  const response = await fetch(`${API_BASE}${endpoint}`, {
    ...options,
    headers,
  });

  if (!response.ok) {
    // Handle 401 Unauthorized - token expired or invalid
    if (response.status === 401 && token) {
      setToken(null);
      if (onAuthErrorCallback) {
        onAuthErrorCallback();
      }
    }

    const error = await response
      .json()
      .catch(() => ({ detail: 'Request failed' }));
    throw new Error(
      error.detail ||
      error.message ||
      error.error ||
      `HTTP ${response.status}`
    );
  }

  return response.json();
}

// ============================================================================
// API Modules
// ============================================================================

export const auth = {
  signup: (data: SignupData): Promise<{ message: string }> =>
    request('/api/auth/signup', {
      method: 'POST',
      body: JSON.stringify(data),
    }),

  login: (data: LoginData): Promise<AuthResponse> =>
    request('/api/auth/login', {
      method: 'POST',
      body: JSON.stringify(data),
    }),

  verifyEmail: (token: string): Promise<AuthResponse> =>
    request('/api/auth/verify-email', {
      method: 'POST',
      body: JSON.stringify({ token }),
    }),

  resendVerification: (email: string): Promise<{ message: string }> =>
    request('/api/auth/resend-verification', {
      method: 'POST',
      body: JSON.stringify({ email }),
    }),

  getMe: (): Promise<User> => request('/api/auth/me'),
  changePassword: (data: ChangePasswordData): Promise<{ message: string }> =>
    request('/api/auth/change-password', {
      method: 'POST',
      body: JSON.stringify(data),
    }),
  deleteAccount: (data: DeleteAccountData): Promise<{ message: string }> =>
    request('/api/auth/delete-account', {
      method: 'POST',
      body: JSON.stringify(data),
    }),
};

export const tiers = {
  getAll: (): Promise<Tier[]> => request('/api/tiers'),
  get: (tier: string): Promise<Tier> => request(`/api/tiers/${tier}`),
};

export const predictions = {
  get: (ticker: string, horizon = '1d'): Promise<Prediction> => {
    const normalized = horizon.toLowerCase();

    if (normalized === '5d' || normalized === '5day' || normalized === '5days') {
      return request(`/predict/lstm_5d/${ticker}`);
    }

    if (normalized === '20d' || normalized === '20day' || normalized === '20days') {
      return request(`/predict/lstm_jackpot/${ticker}`);
    }

    return request(`/predict/${ticker}?horizon=${horizon}`);
  },

  getUniverse: (): Promise<string[]> => request('/api/universe'),
};

export const oracle = {
  get: (): Promise<OracleData> => request('/api/oracle'),

  updateWatchlist: (watchlist: string[]): Promise<OracleData> =>
    request('/api/oracle/watchlist', {
      method: 'PUT',
      body: JSON.stringify({ watchlist }),
    }),

  addToWatchlist: (ticker: string): Promise<OracleData> =>
    request('/api/oracle/watchlist', {
      method: 'POST',
      body: JSON.stringify({ ticker }),
    }),

  removeFromWatchlist: (ticker: string): Promise<OracleData> =>
    request(`/api/oracle/watchlist/${ticker}`, {
      method: 'DELETE',
    }),

  updateTimeframes: (timeframes: string[]): Promise<OracleData> =>
    request('/api/oracle/timeframes', {
      method: 'PUT',
      body: JSON.stringify({ timeframes }),
    }),

  getPredictions: (): Promise<Prediction[]> => request('/api/oracle/predictions'),
};

export const health = {
  check: (): Promise<{ status: string }> => request('/healthz'),
};

export const companies = {
  get: (ticker: string): Promise<Company> => request(`/api/companies/${ticker}`),
  getNews: (ticker: string, limit = 50): Promise<NewsItem[]> =>
    request(`/api/companies/${ticker}/news?limit=${limit}`),
};

export const analysis = {
  getModels: (): Promise<ModelsResponse> => request('/api/analyze/models'),

  getUniverse: (): Promise<UniverseResponse> => request('/api/analyze/universe'),

  getUserFeatures: (): Promise<UserFeatures> => request('/api/user/features'),

  run: (data: {
    tickers: string[];
    model: string;
    task: string;
    period: string;
    horizon: string;
  }): Promise<AnalysisResponse> =>
    request('/api/analyze', {
      method: 'POST',
      body: JSON.stringify(data),
    }),
};

export const performance = {
  getTrackRecord: (model = 'lstm_5d'): Promise<TrackRecordResponse> =>
    request(`/api/performance/track-record?model=${encodeURIComponent(model)}`),
  getTrackRecordCurve: (model = 'lstm_5d'): Promise<TrackRecordCurveResponse> =>
    request(`/api/performance/curve?model=${encodeURIComponent(model)}`),
};

export const billing = {
  getStatus: (): Promise<BillingStatus> => request('/api/billing/status'),
  createCheckoutSession: (tier: 'basic' | 'pro'): Promise<BillingCheckoutSessionResponse> =>
    request('/api/billing/checkout-session', {
      method: 'POST',
      body: JSON.stringify({ tier }),
    }),
  createPortalSession: (): Promise<BillingPortalSessionResponse> =>
    request('/api/billing/portal-session', {
      method: 'POST',
    }),
  cancelSubscription: (): Promise<{ message: string }> =>
    request('/api/billing/cancel-subscription', {
      method: 'POST',
    }),
  changeSubscription: (tier: 'basic' | 'pro'): Promise<BillingChangeSubscriptionResponse> =>
    request('/api/billing/change-subscription', {
      method: 'POST',
      body: JSON.stringify({ tier }),
    }),
};
