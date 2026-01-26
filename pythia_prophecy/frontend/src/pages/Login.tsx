import { useState, useEffect, ChangeEvent, FormEvent } from 'react';
import { Link, useNavigate, useLocation, useSearchParams } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import { useToast } from '../components/Toast';

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

    try {
      await login(formData.email, formData.password);
      navigate(from, { replace: true });
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Login failed';
      setFormError(message);
      // Show resend option if email not verified
      if (message.toLowerCase().includes('not verified')) {
        setShowResend(true);
      }
    } finally {
      setLoading(false);
    }
  };

  const handleResend = async () => {
    try {
      await resendVerification(formData.email);
      setResendSuccess(true);
    } catch (err) {
      setFormError(err instanceof Error ? err.message : 'Failed to resend verification');
    }
  };

  return (
    <div className="auth-page">
      <div className="auth-container">
        <div className="auth-header">
          <Link to="/" className="logo">
            <span className="logo-icon">P</span>
            <span className="logo-text">Pythia</span>
          </Link>
        </div>

        <div className="auth-card">
          <h1>Welcome back</h1>
          <p className="auth-subtitle">
            Log in to access your predictions
          </p>

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
            </div>

            <button type="submit" className="btn btn-primary btn-block" disabled={loading}>
              {loading ? 'Logging in...' : 'Log in'}
            </button>
          </form>

          <p className="auth-footer">
            Don't have an account? <Link to="/signup">Sign up</Link>
          </p>
        </div>
      </div>
    </div>
  );
}

export default Login;
