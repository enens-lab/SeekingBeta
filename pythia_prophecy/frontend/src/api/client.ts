/**
 * API client for Pythia backend
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
}

export interface LoginData {
  email: string;
  password: string;
}

export interface Tier {
  name: string;
  price: number;
  stocks: number | 'unlimited';
  timeframes: string[];
}

export interface Prediction {
  ticker: string;
  horizon: string;
  signal: 'buy' | 'hold' | 'sell';
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
  signal: 'buy' | 'hold' | 'sell' | null;
  predicted_return: number | null;
}

export interface AnalysisResponse {
  results: AnalysisResult[];
  model: string;
  task: string;
  horizon: string;
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
  categories: Record<string, string[]>;
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

    const error = await response.json().catch(() => ({ detail: 'Request failed' }));
    throw new Error(error.detail || `HTTP ${response.status}`);
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
};

export const tiers = {
  getAll: (): Promise<Tier[]> => request('/api/tiers'),
  get: (tier: string): Promise<Tier> => request(`/api/tiers/${tier}`),
};

export const predictions = {
  get: (ticker: string, horizon = '1d'): Promise<Prediction> =>
    request(`/predict/${ticker}?horizon=${horizon}`),

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
