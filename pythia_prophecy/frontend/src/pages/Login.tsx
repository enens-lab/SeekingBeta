import { useState, useEffect, ChangeEvent, FormEvent } from 'react';
import { Link, useNavigate, useLocation, useSearchParams } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import { useToast } from '../components/Toast';
import ThemeToggle from '../components/ThemeToggle';
import SocialAuthButtons from '../components/SocialAuthButtons';
import { trackEvent } from '../lib/analytics';

interface LocationState {
  from?: {
    pathname: string;
  };
}

function Login() {
  const navigate = useNavigate();
  const location = useLocation();
  const [searchParams, setSearchParams] = useSearchParams();
  const { login, resendVerification, isAuthenticated, error, clearError } = useAuth();
  const toast = useToast();

  const [formData, setFormData] = useState({
    email: '',
    password: '',
  });
  const [loading, setLoading] = useState(false);
  const [formError, setFormError] = useState('');
  const [showResend, setShowResend] = useState(false);
  const [resendSuccess, setResendSuccess] = useState(false);

  const state = location.state as LocationState | null;
  const from = state?.from?.pathname || '/dashboard';

  // Show session expired message if redirected due to JWT timeout
  useEffect(() => {
    if (searchParams.get('expired') === '1') {
      toast.error('Your session has expired. Please log in again.');
      // Clear the expired param from URL
      searchParams.delete('expired');
      setSearchParams(searchParams, { replace: true });
    }
  }, [searchParams, setSearchParams, toast]);

  useEffect(() => {
    if (isAuthenticated) {
      navigate(from, { replace: true });
    }
  }, [isAuthenticated, navigate, from]);

  useEffect(() => {
    clearError();
  }, [clearError]);

  const handleChange = (e: ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    setFormData((prev) => ({ ...prev, [name]: value }));
    setFormError('');
    setShowResend(false);
  };

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    setFormError('');
    setLoading(true);
    trackEvent('login_attempt', { from_path: from });

    try {
      await login(formData.email, formData.password);
      trackEvent('login_success', { redirect_to: from });
      navigate(from, { replace: true });
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Login failed';
      setFormError(message);
      trackEvent('login_error', { reason: message.slice(0, 80) });
      // Show resend option if email not verified
      if (message.toLowerCase().includes('not verified')) {
        setShowResend(true);
      }
    } finally {
      setLoading(false);
    }
  };

  const handleResend = async () => {
    trackEvent('resend_verification_attempt', { source: 'login' });
    try {
      await resendVerification(formData.email);
      setResendSuccess(true);
      trackEvent('resend_verification_success', { source: 'login' });
    } catch (err) {
      trackEvent('resend_verification_error', { source: 'login' });
      setFormError(err instanceof Error ? err.message : 'Failed to resend verification');
    }
  };

  return (
    <div className="auth-page">
      <div className="auth-container">
        <div className="auth-header">
          <div className="auth-header-row">
            <Link to="/" className="logo">
              <span className="logo-icon">β</span>
              <span className="logo-text">SeekingBeta.AI</span>
            </Link>
            <ThemeToggle />
          </div>
        </div>

        <div className="auth-card">
          <h1>Welcome back to the board</h1>
          <p className="auth-subtitle">
            Log in to open your stock and sports prediction boards
          </p>

          <SocialAuthButtons redirectTo={from} onError={setFormError} />

          <form onSubmit={handleSubmit} className="auth-form">
            {(formError || error) && (
              <div className="form-error">
                {formError || error}
                {showResend && !resendSuccess && (
                  <button
                    type="button"
                    className="resend-link"
                    onClick={handleResend}
                  >
                    Resend verification email
                  </button>
                )}
                {resendSuccess && (
                  <span className="resend-success">Verification email sent!</span>
                )}
              </div>
            )}

            <div className="form-group">
              <label htmlFor="email">Email</label>
              <input
                type="email"
                id="email"
                name="email"
                value={formData.email}
                onChange={handleChange}
                required
                autoComplete="email"
              />
            </div>

            <div className="form-group">
              <label htmlFor="password">Password</label>
              <input
                type="password"
                id="password"
                name="password"
                value={formData.password}
                onChange={handleChange}
                required
                autoComplete="current-password"
              />
              <div className="form-group-helper">
                <Link to="/reset-password" className="form-link">
                  Forgot password?
                </Link>
              </div>
            </div>

            <button type="submit" className="btn btn-primary btn-block" disabled={loading}>
              {loading ? 'Logging in...' : 'Open Dashboard'}
            </button>
          </form>

          <p className="auth-footer">
            New here? <Link to="/signup">Create your free account</Link>
          </p>
        </div>
      </div>
    </div>
  );
}

export default Login;
