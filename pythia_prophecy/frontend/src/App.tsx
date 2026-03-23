import { ReactNode, useEffect } from 'react';
import { BrowserRouter, Routes, Route, Navigate, useLocation } from 'react-router-dom';
import { AuthProvider, useAuth } from './context/AuthContext';
import { ThemeProvider } from './context/ThemeContext';
import { ToastProvider } from './components/Toast';
import AnalyticsConsentManager from './components/AnalyticsConsentManager';
import { initAnalytics, trackPageView } from './lib/analytics';

// Pages
import Landing from './pages/Landing';
import Signup from './pages/Signup';
import Login from './pages/Login';
import VerifyEmail from './pages/VerifyEmail';
import ResetPasswordPage from './pages/ResetPasswordPage';
import Pricing from './pages/Pricing';
import Dashboard from './pages/Dashboard';
import MyOracle from './pages/MyOracle';
import Analysis from './pages/Analysis';
import Profile from './pages/Profile';
import Terms from './pages/Terms';
import Privacy from './pages/Privacy';
import RefundCancellation from './pages/RefundCancellation';
import Methodology from './pages/Methodology';
import SportsLanding from './pages/sports/SportsLanding';
import SportsDashboard from './pages/sports/SportsDashboard';

interface ProtectedRouteProps {
  children: ReactNode;
}

interface SeoMeta {
  title: string;
  description: string;
  canonicalPath?: string;
  indexable: boolean;
}

const SITE_ORIGIN = 'https://seekingbeta.ai';
const DEFAULT_SEO: SeoMeta = {
  title: 'SeekingBeta.AI | Educational Stock Model Signals',
  description:
    'SeekingBeta.AI provides educational, model-driven stock signal views with transparent probabilities and benchmarked track records.',
  canonicalPath: '/',
  indexable: true,
};

const ROUTE_SEO: Record<string, SeoMeta> = {
  '/': DEFAULT_SEO,
  '/pricing': {
    title: 'Pricing | SeekingBeta.AI',
    description:
      'Compare SeekingBeta.AI plans for educational stock model signals, horizons, and watchlist limits.',
    canonicalPath: '/pricing',
    indexable: true,
  },
  '/methodology': {
    title: 'Model Methodology | SeekingBeta.AI',
    description:
      'Learn how SeekingBeta.AI model signals are generated, evaluated, and presented for educational use.',
    canonicalPath: '/methodology',
    indexable: true,
  },
  '/terms': {
    title: 'Terms of Service | SeekingBeta.AI',
    description: 'Review the SeekingBeta.AI terms of service and platform usage terms.',
    canonicalPath: '/terms',
    indexable: true,
  },
  '/privacy': {
    title: 'Privacy Policy | SeekingBeta.AI',
    description: 'Review how SeekingBeta.AI collects, uses, and protects your data.',
    canonicalPath: '/privacy',
    indexable: true,
  },
  '/refund-cancellation': {
    title: 'Refund & Cancellation | SeekingBeta.AI',
    description: 'Read the SeekingBeta.AI refund and cancellation policy for paid subscriptions.',
    canonicalPath: '/refund-cancellation',
    indexable: true,
  },
  '/login': {
    title: 'Log In | SeekingBeta.AI',
    description: 'Log in to your SeekingBeta.AI account.',
    canonicalPath: '/login',
    indexable: false,
  },
  '/signup': {
    title: 'Sign Up | SeekingBeta.AI',
    description: 'Create your SeekingBeta.AI account.',
    canonicalPath: '/signup',
    indexable: false,
  },
  '/verify-email': {
    title: 'Verify Email | SeekingBeta.AI',
    description: 'Verify your email to activate your SeekingBeta.AI account.',
    canonicalPath: '/verify-email',
    indexable: false,
  },
  '/reset-password': {
    title: 'Reset Password | SeekingBeta.AI',
    description: 'Reset your SeekingBeta.AI account password.',
    canonicalPath: '/reset-password',
    indexable: false,
  },
  '/dashboard': {
    title: 'Dashboard | SeekingBeta.AI',
    description: 'Personalized model views and signal summaries.',
    canonicalPath: '/dashboard',
    indexable: false,
  },
  '/sports': {
    title: 'Sports Markets | SeekingBeta.AI',
    description: 'Prediction markets and win probabilities for major sports events.',
    canonicalPath: '/sports',
    indexable: false,
  },
  '/oracle': {
    title: 'Watchlist | SeekingBeta.AI',
    description: 'Manage your watchlist and preferred model horizons.',
    canonicalPath: '/oracle',
    indexable: false,
  },
  '/analysis': {
    title: 'Analysis | SeekingBeta.AI',
    description: 'Run model analysis and review signal outputs.',
    canonicalPath: '/analysis',
    indexable: false,
  },
  '/profile': {
    title: 'Profile | SeekingBeta.AI',
    description: 'Manage your account profile and billing.',
    canonicalPath: '/profile',
    indexable: false,
  },
};

function getRouteSeo(pathname: string): SeoMeta {
  return ROUTE_SEO[pathname] ?? {
    ...DEFAULT_SEO,
    canonicalPath: pathname || '/',
  };
}

function upsertMetaTag(
  queryKey: 'name' | 'property',
  queryValue: string,
  content: string,
) {
  let tag = document.head.querySelector<HTMLMetaElement>(
    `meta[${queryKey}="${queryValue}"]`,
  );
  if (!tag) {
    tag = document.createElement('meta');
    tag.setAttribute(queryKey, queryValue);
    document.head.appendChild(tag);
  }
  tag.setAttribute('content', content);
}

function upsertCanonical(url: string) {
  let canonical = document.head.querySelector<HTMLLinkElement>('link[rel="canonical"]');
  if (!canonical) {
    canonical = document.createElement('link');
    canonical.setAttribute('rel', 'canonical');
    document.head.appendChild(canonical);
  }
  canonical.setAttribute('href', url);
}

// Protected route wrapper
function ProtectedRoute({ children }: ProtectedRouteProps) {
  const { isAuthenticated, loading } = useAuth();

  if (loading) {
    return (
      <div className="loading-page">
        <div className="spinner" />
        <p>Loading...</p>
      </div>
    );
  }

  if (!isAuthenticated) {
    return <Navigate to="/login" replace />;
  }

  return <>{children}</>;
}

function AppRoutes() {
  return (
    <Routes>
      <Route path="/" element={<Landing />} />
      <Route path="/signup" element={<Signup />} />
      <Route path="/login" element={<Login />} />
      <Route path="/verify-email" element={<VerifyEmail />} />
      <Route path="/reset-password" element={<ResetPasswordPage />} />
      <Route path="/pricing" element={<Pricing />} />
      <Route path="/terms" element={<Terms />} />
      <Route path="/privacy" element={<Privacy />} />
      <Route path="/refund-cancellation" element={<RefundCancellation />} />
      <Route path="/methodology" element={<Methodology />} />
      <Route path="/sports" element={<SportsLanding />} />
      <Route
        path="/dashboard"
        element={
          <ProtectedRoute>
            <Dashboard />
          </ProtectedRoute>
        }
      />
      <Route
        path="/sports-dashboard"
        element={
          <ProtectedRoute>
            <SportsDashboard />
          </ProtectedRoute>
        }
      />
      <Route
        path="/oracle"
        element={
          <ProtectedRoute>
            <MyOracle />
          </ProtectedRoute>
        }
      />
      <Route
        path="/analysis"
        element={
          <ProtectedRoute>
            <Analysis />
          </ProtectedRoute>
        }
      />
      <Route
        path="/profile"
        element={
          <ProtectedRoute>
            <Profile />
          </ProtectedRoute>
        }
      />
      {/* Fallback to landing */}
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
}

function AnalyticsRouteTracker() {
  const location = useLocation();

  useEffect(() => {
    initAnalytics();
    const pagePath = `${location.pathname}${location.search}${location.hash}`;
    trackPageView(pagePath);

    const seo = getRouteSeo(location.pathname);
    const canonicalUrl = new URL(seo.canonicalPath ?? location.pathname, SITE_ORIGIN).toString();

    document.title = seo.title;
    upsertMetaTag('name', 'description', seo.description);
    upsertMetaTag('name', 'robots', seo.indexable ? 'index,follow' : 'noindex,nofollow');
    upsertCanonical(canonicalUrl);

    upsertMetaTag('property', 'og:url', canonicalUrl);
    upsertMetaTag('property', 'og:title', seo.title);
    upsertMetaTag('property', 'og:description', seo.description);
    upsertMetaTag('name', 'twitter:title', seo.title);
    upsertMetaTag('name', 'twitter:description', seo.description);
  }, [location.pathname, location.search, location.hash]);

  return null;
}

function App() {
  return (
    <ThemeProvider>
      <BrowserRouter>
        <AnalyticsRouteTracker />
        <AnalyticsConsentManager />
        <AuthProvider>
          <ToastProvider>
            <AppRoutes />
          </ToastProvider>
        </AuthProvider>
      </BrowserRouter>
    </ThemeProvider>
  );
}

export default App;
