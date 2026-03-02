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
import Pricing from './pages/Pricing';
import Dashboard from './pages/Dashboard';
import MyOracle from './pages/MyOracle';
import Analysis from './pages/Analysis';
import Profile from './pages/Profile';
import Terms from './pages/Terms';
import Privacy from './pages/Privacy';
import RefundCancellation from './pages/RefundCancellation';
import Methodology from './pages/Methodology';

interface ProtectedRouteProps {
  children: ReactNode;
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
      <Route path="/pricing" element={<Pricing />} />
      <Route path="/terms" element={<Terms />} />
      <Route path="/privacy" element={<Privacy />} />
      <Route path="/refund-cancellation" element={<RefundCancellation />} />
      <Route path="/methodology" element={<Methodology />} />
      <Route
        path="/dashboard"
        element={
          <ProtectedRoute>
            <Dashboard />
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
