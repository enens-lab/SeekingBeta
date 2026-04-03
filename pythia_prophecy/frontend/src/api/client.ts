/**
 * API client for SeekingBeta.AI backend
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

export interface BetaTesterSignupData {
  full_name: string;
  email: string;
  role?: string;
  organization?: string;
  investing_experience?: string;
  testing_focus: string;
  accept_contact: boolean;
  source?: string;
}

export interface BetaTesterSignupResponse {
  message: string;
  discord_url: string;
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

export interface PasswordResetRequestData {
  email: string;
}

export interface PasswordResetConfirmData {
  token: string;
  new_password: string;
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
  attribution?: PredictionAttribution | null;
}

export interface PredictionDriver {
  feature: string;
  signed_contribution: number;
  magnitude: number;
  direction: 'positive' | 'negative' | string;
}

export interface PredictionAttribution {
  method: string;
  baseline?: string;
  steps?: number;
  sequence_length?: number;
  feature_count?: number;
  top_k?: number;
  top_drivers?: PredictionDriver[];
  top_positive_drivers?: PredictionDriver[];
  top_negative_drivers?: PredictionDriver[];
  summary?: string[];
}

export interface PredictionAttributionResponse {
  ticker: string;
  model: string;
  description: string;
  horizon: string;
  target_return: string;
  probability: number;
  signal: string;
  generated_at: string;
  data_source: string;
  attribution: PredictionAttribution;
}

export interface OracleData {
  watchlist: string[];
  timeframes: string[];
  available_stocks: string[];
  available_timeframes: string[];
  available_categories: Record<string, string[]>;
}

export interface WatchlistInsight {
  ticker: string;
  chart_url: string;
  source: 'finviz' | 'fallback';
  updated_at: string;
  price: number | null;
  change_pct: number | null;
  rsi: number | null;
  sma20: number | null;
  sma50: number | null;
  sma200: number | null;
  volume: number | null;
  rel_volume: number | null;
  atr: number | null;
  support: number | null;
  resistance: number | null;
  trend: string | null;
  summary: string | null;
}

export interface WatchlistInsightsResponse {
  watchlist: string[];
  finviz_enabled: boolean;
  insights: WatchlistInsight[];
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
  avg_win_return_net: number | null;
  avg_loss_return_net: number | null;
  profit_factor: number | null;
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

export interface SportsBoardPrediction {
  rank: number;
  playerName: string;
  winProbability: number;
  actualWinner?: boolean;
  side?: string;
  profile?: SportsPlayerProfile;
  radarMetrics?: SportsRadarMetric[];
}

export interface SportsAvailabilitySummary {
  ilAdds14?: number;
  ilActivations14?: number;
  rosterMoves14?: number;
}

export interface SportsProjectedLineupContext {
  awayCoverage?: number | null;
  homeCoverage?: number | null;
  awayContinuity?: number | null;
  homeContinuity?: number | null;
}

export interface SportsBoardDateOption {
  dateKey: string;
  label: string;
  gameCount: number;
}

export interface SportsTeamDetails {
  teamId?: number | null;
  abbreviation?: string | null;
  logoUrl?: string | null;
  wordmarkUrl?: string | null;
  primaryColor?: string | null;
  secondaryColor?: string | null;
  recordPrior?: string | null;
  recentForm?: string | null;
  bullpenSummary?: string | null;
  availabilitySummary?: string | null;
  lineupContinuity?: string | null;
  venue?: string | null;
  weather?: string | null;
}

export interface SportsRadarMetric {
  label: string;
  value: number;
}

export interface SportsPlayerStat {
  label: string;
  value: string;
}

export interface SportsPlayerProfile {
  imageUrl?: string | null;
  subtitle?: string | null;
  country?: string | null;
  stats?: SportsPlayerStat[];
}

export interface SportsLineupPlayer {
  playerId?: number | null;
  playerName: string;
  lineupSlot?: number | null;
  position?: string | null;
  batSide?: string | null;
  performanceSummary?: string | null;
  profile?: SportsPlayerProfile | null;
  radarMetrics?: SportsRadarMetric[];
}

export interface SportsUpcomingBoard {
  id: string;
  name: string;
  original_name?: string;
  tour: string;
  course: string;
  scheduledDate?: number;
  latestDate?: number;
  venue?: string;
  predictedWinner?: string;
  awayTeam?: string;
  homeTeam?: string;
  awayStarter?: string;
  homeStarter?: string;
  awayStarterProfile?: SportsPlayerProfile;
  homeStarterProfile?: SportsPlayerProfile;
  awayStarterRadar?: SportsRadarMetric[];
  homeStarterRadar?: SportsRadarMetric[];
  awayTeamDetails?: SportsTeamDetails;
  homeTeamDetails?: SportsTeamDetails;
  awayAvailability?: SportsAvailabilitySummary;
  homeAvailability?: SportsAvailabilitySummary;
  projectedLineupContext?: SportsProjectedLineupContext;
  predictionSource?: string;
  awayLineup?: SportsLineupPlayer[];
  homeLineup?: SportsLineupPlayer[];
  awayFeaturedPlayer?: SportsLineupPlayer;
  homeFeaturedPlayer?: SportsLineupPlayer;
  predictions: SportsBoardPrediction[];
}

export interface SportsHistoricalBoard {
  year: number;
  tournament: string;
  tour: string;
  hitStatus: string;
  predictedWinner?: string;
  predictedTop3?: string[];
  predictedTop5?: string[];
  actualWinner?: string;
  prob?: number;
  venue?: string;
  course?: string;
  fullField?: SportsBoardPrediction[];
  latestDate?: number;
  tournamentId?: string;
  scheduledDate?: number;
  awayTeam?: string;
  homeTeam?: string;
  awayStarter?: string;
  homeStarter?: string;
  awayStarterProfile?: SportsPlayerProfile;
  homeStarterProfile?: SportsPlayerProfile;
  awayStarterRadar?: SportsRadarMetric[];
  homeStarterRadar?: SportsRadarMetric[];
  awayTeamDetails?: SportsTeamDetails;
  homeTeamDetails?: SportsTeamDetails;
  awayLineup?: SportsLineupPlayer[];
  homeLineup?: SportsLineupPlayer[];
  awayFeaturedPlayer?: SportsLineupPlayer;
  homeFeaturedPlayer?: SportsLineupPlayer;
}

export interface SportsBoardCollection {
  upcoming: SportsUpcomingBoard[];
  backtests: SportsHistoricalBoard[];
  updated_at: string;
  source: string;
  selectedDate?: string;
  availableDates?: SportsBoardDateOption[];
}

export interface SportsBoardsResponse {
  golf: SportsBoardCollection;
  tennis: SportsBoardCollection;
  basketball: SportsBoardCollection;
  mlb: SportsBoardCollection;
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

function toFiniteNumber(value: unknown): number | null {
  if (typeof value === 'number') {
    return Number.isFinite(value) ? value : null;
  }
  if (typeof value === 'string') {
    const parsed = Number.parseFloat(value.trim());
    return Number.isFinite(parsed) ? parsed : null;
  }
  return null;
}

function normalizeHorizon(value: unknown, fallback: string): string {
  if (typeof value !== 'string') return fallback;
  const normalized = value.trim().toLowerCase();
  if (normalized === '5 days' || normalized === '5day' || normalized === '5days') return '5d';
  if (normalized === '20 days' || normalized === '20day' || normalized === '20days') return '20d';
  if (normalized === '5d' || normalized === '20d') return normalized;
  return fallback;
}

function normalizeProbability(probUpValue: unknown, probabilityValue: unknown): number | null {
  let candidate = toFiniteNumber(probUpValue);
  if (candidate === null) {
    candidate = toFiniteNumber(probabilityValue);
  }

  if (candidate === null) return null;
  if (candidate > 1 && candidate <= 100) {
    candidate = candidate / 100;
  }
  if (candidate < 0) return 0;
  if (candidate > 1) return 1;
  return candidate;
}

function normalizePredictionPayload(raw: any, ticker: string, fallbackHorizon: string): Prediction {
  return {
    ticker: (typeof raw?.ticker === 'string' && raw.ticker.trim()) || ticker.toUpperCase(),
    horizon: normalizeHorizon(raw?.horizon, fallbackHorizon),
    signal: (raw?.signal || 'hold') as Prediction['signal'],
    prob_up: normalizeProbability(raw?.prob_up, raw?.probability),
    predicted_return: toFiniteNumber(raw?.predicted_return),
    last_close: toFiniteNumber(raw?.last_close),
    timestamp:
      (typeof raw?.generated_at === 'string' && raw.generated_at) ||
      (typeof raw?.timestamp === 'string' && raw.timestamp) ||
      new Date().toISOString(),
    attribution: raw?.attribution ?? null,
  };
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
  requestPasswordReset: (data: PasswordResetRequestData): Promise<{ message: string }> =>
    request('/api/auth/password-reset', {
      method: 'POST',
      body: JSON.stringify(data),
    }),
  confirmPasswordReset: (data: PasswordResetConfirmData): Promise<{ message: string }> =>
    request('/api/auth/password-reset-confirm', {
      method: 'POST',
      body: JSON.stringify(data),
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

export const betaProgram = {
  signup: (data: BetaTesterSignupData): Promise<BetaTesterSignupResponse> =>
    request('/api/beta-testers/signup', {
      method: 'POST',
      body: JSON.stringify(data),
    }),
};

export const tiers = {
  getAll: (): Promise<Tier[]> => request('/api/tiers'),
  get: (tier: string): Promise<Tier> => request(`/api/tiers/${tier}`),
};

export const predictions = {
  get: async (ticker: string, horizon = '5d'): Promise<Prediction> => {
    const normalized = horizon.toLowerCase();

    if (normalized === '5d' || normalized === '5day' || normalized === '5days') {
      const raw = await request<any>(`/predict/lstm_5d/${ticker}`);
      return normalizePredictionPayload(raw, ticker, '5d');
    }

    if (normalized === '20d' || normalized === '20day' || normalized === '20days') {
      const raw = await request<any>(`/predict/lstm_jackpot/${ticker}`);
      return normalizePredictionPayload(raw, ticker, '20d');
    }

    throw new Error(`Unsupported horizon '${horizon}'. Allowed horizons: 5d, 20d.`);
  },

  getAttribution: (
    modelName: 'lstm_5d' | 'lstm_jackpot',
    ticker: string,
    topK = 3
  ): Promise<PredictionAttributionResponse> =>
    request(
      `/predict/lstm/${modelName}/${ticker}/attribution?method=integrated_gradients&top_k=${topK}`
    ),

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
  getWatchlistInsights: (limit = 20): Promise<WatchlistInsightsResponse> =>
    request(`/api/oracle/watchlist-insights?limit=${encodeURIComponent(limit)}`),
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

export const sports = {
  getBoards: (options?: { mlbDate?: string; basketballDate?: string }): Promise<SportsBoardsResponse> => {
    const params = new URLSearchParams();
    if (options?.mlbDate) {
      params.set('mlb_date', options.mlbDate);
    }
    if (options?.basketballDate) {
      params.set('basketball_date', options.basketballDate);
    }
    const query = params.toString();
    return request(query ? `/api/sports/boards?${query}` : '/api/sports/boards');
  },
};
