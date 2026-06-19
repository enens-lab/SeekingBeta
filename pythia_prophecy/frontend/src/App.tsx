import { ReactNode, useEffect } from 'react';
import { BrowserRouter, Routes, Route, Navigate, useLocation } from 'react-router-dom';
import { AuthProvider, useAuth } from './context/AuthContext';
import { ThemeProvider } from './context/ThemeContext';
import { ToastProvider } from './components/Toast';
import AnalyticsConsentManager from './components/AnalyticsConsentManager';
import { initAnalytics, trackPageView } from './lib/analytics';
import { SITE_ORIGIN, getRouteSeo } from './lib/seo';

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
import DeleteAccount from './pages/DeleteAccount';
import Methodology from './pages/Methodology';
import TrackRecord from './pages/TrackRecord';
import Support from './pages/Support';
import SportsLanding from './pages/sports/SportsLanding';
import SportsDashboard from './pages/sports/SportsDashboard';

interface ProtectedRouteProps {
  children: ReactNode;
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
      <Route path="/delete-account" element={<DeleteAccount />} />
      <Route path="/methodology" element={<Methodology />} />
      <Route path="/track-record" element={<TrackRecord />} />
      <Route path="/support" element={<Support />} />
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

// Everything inside the router, so the static prerender (entry-prerender.tsx)
// can mount the same tree under a StaticRouter instead of BrowserRouter.
export function InnerApp() {
  return (
    <>
      <AnalyticsRouteTracker />
      <AnalyticsConsentManager />
      <AuthProvider>
        <ToastProvider>
          <AppRoutes />
        </ToastProvider>
      </AuthProvider>
    </>
  );
}

function App() {
  return (
    <ThemeProvider>
      <BrowserRouter>
        <InnerApp />
      </BrowserRouter>
    </ThemeProvider>
  );
}

export default App;
