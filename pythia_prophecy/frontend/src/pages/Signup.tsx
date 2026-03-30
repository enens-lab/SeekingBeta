import { useState, useEffect, ChangeEvent, FormEvent } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import ThemeToggle from '../components/ThemeToggle';
import { trackEvent } from '../lib/analytics';

const POLICY_VERSION = '2026-02-27';

function Signup() {
  const navigate = useNavigate();
  const { signup, resendVerification, isAuthenticated, error, clearError } = useAuth();

  const [formData, setFormData] = useState({
    firstName: '',
    lastName: '',
    email: '',
    password: '',
    confirmPassword: '',
    acceptTerms: false,
    acceptPrivacy: false,
    marketingOptIn: false,
  });
  const [loading, setLoading] = useState(false);
  const [success, setSuccess] = useState(false);
  const [formError, setFormError] = useState('');
  const [resendLoading, setResendLoading] = useState(false);
  const [resendMessage, setResendMessage] = useState('');

  useEffect(() => {
    if (isAuthenticated) {
      navigate('/dashboard');
    }
  }, [isAuthenticated, navigate]);

  useEffect(() => {
    clearError();
  }, [clearError]);

  const handleChange = (e: ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    setFormData((prev) => ({ ...prev, [name]: value }));
    setFormError('');
  };

  const handleCheckboxChange = (e: ChangeEvent<HTMLInputElement>) => {
    const { name, checked } = e.target;
    setFormData((prev) => ({ ...prev, [name]: checked }));
    setFormError('');
  };

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    setFormError('');

    if (formData.password !== formData.confirmPassword) {
      setFormError('Passwords do not match');
      return;
    }

    if (formData.password.length < 8) {
      setFormError('Password must be at least 8 characters');
      return;
    }
    if (!/[A-Z]/.test(formData.password)) {
      setFormError('Password must contain at least one uppercase letter');
      return;
    }
    if (!/[a-z]/.test(formData.password)) {
      setFormError('Password must contain at least one lowercase letter');
      return;
    }
    if (!/[0-9]/.test(formData.password)) {
      setFormError('Password must contain at least one number');
      return;
    }
    if (!/[!@#$%^&*()_\-+=[\]{};:'\",.<>?/\\|`~]/.test(formData.password)) {
      setFormError('Password must contain at least one special character');
      return;
    }

    if (!formData.acceptTerms) {
      setFormError('You must accept the Terms of Service');
      return;
    }

    if (!formData.acceptPrivacy) {
      setFormError('You must accept the Privacy Policy');
      return;
    }

    setLoading(true);
    trackEvent('signup_attempt', { marketing_opt_in: formData.marketingOptIn });

    try {
      await signup({
        email: formData.email,
        password: formData.password,
        first_name: formData.firstName,
        last_name: formData.lastName,
        tier: 'free',
        accept_terms: formData.acceptTerms,
        accept_privacy: formData.acceptPrivacy,
        policy_version: POLICY_VERSION,
        marketing_opt_in: formData.marketingOptIn,
      });
      setSuccess(true);
      trackEvent('sign_up', { method: 'password' });
    } catch (err) {
      trackEvent('signup_error');
      setFormError(err instanceof Error ? err.message : 'Signup failed');
    } finally {
      setLoading(false);
    }
  };

  if (success) {
    const handleResendVerification = async () => {
      setResendLoading(true);
      setResendMessage('');
      trackEvent('resend_verification_attempt', { source: 'signup_success' });
      try {
        const response = await resendVerification(formData.email);
        setResendMessage(response.message);
        trackEvent('resend_verification_success', { source: 'signup_success' });
      } catch (err) {
        trackEvent('resend_verification_error', { source: 'signup_success' });
        setResendMessage(err instanceof Error ? err.message : 'Failed to resend verification email');
      } finally {
        setResendLoading(false);
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
            <div className="success-icon">
              <svg width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14" />
                <polyline points="22 4 12 14.01 9 11.01" />
              </svg>
            </div>
            <h1>Check your inbox</h1>
            <p className="auth-subtitle">
              We&apos;ve sent a verification link to <strong>{formData.email}</strong>.
              Confirm it to unlock your boards.
            </p>
            <p className="auth-note">
              If it doesn&apos;t arrive in a minute, check spam or resend it below.
            </p>
            <div className="auth-actions">
              <button
                type="button"
                className="btn btn-outline"
                onClick={handleResendVerification}
                disabled={resendLoading}
              >
                {resendLoading ? 'Resending...' : 'Resend verification link'}
              </button>
            </div>
            {resendMessage && <p className="auth-note">{resendMessage}</p>}
            <p className="auth-note">
              You can also verify manually at{' '}
              <Link to={`/verify-email?email=${encodeURIComponent(formData.email)}`}>/verify-email</Link>.
            </p>
          </div>
        </div>
      </div>
    );
  }

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
          <h1>Create your account</h1>
          <p className="auth-subtitle">
            Start free and unlock stock and sports prediction boards in minutes
          </p>

          <form onSubmit={handleSubmit} className="auth-form">
            {(formError || error) && <div className="form-error">{formError || error}</div>}

            <div className="form-row">
              <div className="form-group">
                <label htmlFor="firstName">First name</label>
                <input
                  type="text"
                  id="firstName"
                  name="firstName"
                  value={formData.firstName}
                  onChange={handleChange}
                  required
                  autoComplete="given-name"
                />
              </div>
              <div className="form-group">
                <label htmlFor="lastName">Last name</label>
                <input
                  type="text"
                  id="lastName"
                  name="lastName"
                  value={formData.lastName}
                  onChange={handleChange}
                  required
                  autoComplete="family-name"
                />
              </div>
            </div>

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
                minLength={8}
                autoComplete="new-password"
              />
              <span className="form-hint">
                Use 8+ characters with uppercase, lowercase, a number, and a special character
              </span>
            </div>

            <div className="form-group">
              <label htmlFor="confirmPassword">Confirm password</label>
              <input
                type="password"
                id="confirmPassword"
                name="confirmPassword"
                value={formData.confirmPassword}
                onChange={handleChange}
                required
                autoComplete="new-password"
              />
            </div>

            <div className="form-group">
              <label className="consent-check">
                <input
                  type="checkbox"
                  name="acceptTerms"
                  checked={formData.acceptTerms}
                  onChange={handleCheckboxChange}
                  required
                />
                <span>
                  I agree to the <Link to="/terms">Terms of Service</Link>.
                </span>
              </label>
              <label className="consent-check">
                <input
                  type="checkbox"
                  name="acceptPrivacy"
                  checked={formData.acceptPrivacy}
                  onChange={handleCheckboxChange}
                  required
                />
                <span>
                  I acknowledge the <Link to="/privacy">Privacy Policy</Link>.
                </span>
              </label>
              <label className="consent-check optional">
                <input
                  type="checkbox"
                  name="marketingOptIn"
                  checked={formData.marketingOptIn}
                  onChange={handleCheckboxChange}
                />
                <span>
                  Send me occasional product updates (optional). You can unsubscribe anytime.
                </span>
              </label>
              <p className="form-hint">Policy version: {POLICY_VERSION}</p>
              <p className="form-hint">Every new account starts on Free. Upgrade later for more coverage, history, and export power.</p>
              <Link to="/pricing" className="form-link">Compare plans in detail</Link>
            </div>

            <button type="submit" className="btn btn-primary btn-block" disabled={loading}>
              {loading ? 'Creating account...' : 'Create Free Account'}
            </button>
          </form>

          <p className="auth-footer">
            Already have an account? <Link to="/login">Log in</Link>
          </p>
          <p className="auth-note">
            By creating an account, you agree to our <Link to="/terms">Terms</Link>,{' '}
            <Link to="/privacy">Privacy Policy</Link>, and{' '}
            <Link to="/refund-cancellation">Refund & Cancellation Policy</Link>.
          </p>
        </div>
      </div>
    </div>
  );
}

export default Signup;
