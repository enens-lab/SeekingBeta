import { useState, useEffect, ChangeEvent, FormEvent } from 'react';
import { Link, useNavigate, useSearchParams } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import { tiers as tiersApi } from '../api/client';

interface TierData {
  tier: string;
  name: string;
  price: number;
  stocks_limit: number;
  timeframes: string[];
}

function Signup() {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const { signup, isAuthenticated, error, clearError } = useAuth();

  const [formData, setFormData] = useState({
    firstName: '',
    lastName: '',
    email: '',
    password: '',
    confirmPassword: '',
    tier: searchParams.get('tier') || 'free',
  });
  const [tiers, setTiers] = useState<TierData[]>([]);
  const [loading, setLoading] = useState(false);
  const [success, setSuccess] = useState(false);
  const [formError, setFormError] = useState('');

  useEffect(() => {
    if (isAuthenticated) {
      navigate('/dashboard');
    }
  }, [isAuthenticated, navigate]);

  useEffect(() => {
    tiersApi.getAll()
      .then((data) => setTiers(data as unknown as TierData[]))
      .catch(console.error);
  }, []);

  useEffect(() => {
    clearError();
  }, [clearError]);

  const handleChange = (e: ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    setFormData((prev) => ({ ...prev, [name]: value }));
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
    if (!/[!@#$%^&*()_\-+=[\]{};:'",.<>?/\\|`~]/.test(formData.password)) {
      setFormError('Password must contain at least one special character');
      return;
    }

    setLoading(true);

    try {
      await signup({
        email: formData.email,
        password: formData.password,
        first_name: formData.firstName,
        last_name: formData.lastName,
        tier: formData.tier,
      });
      setSuccess(true);
    } catch (err) {
      setFormError(err instanceof Error ? err.message : 'Signup failed');
    } finally {
      setLoading(false);
    }
  };

  if (success) {
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
            <div className="success-icon">
              <svg width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14" />
                <polyline points="22 4 12 14.01 9 11.01" />
              </svg>
            </div>
            <h1>Check your email</h1>
            <p className="auth-subtitle">
              We've sent a verification link to <strong>{formData.email}</strong>.
              Click the link to activate your account.
            </p>
            <p className="auth-note">
              Didn't receive the email? Check your spam folder or{' '}
              <Link to="/login">try logging in</Link> to resend.
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
          <Link to="/" className="logo">
            <span className="logo-icon">P</span>
            <span className="logo-text">Pythia</span>
          </Link>
        </div>

        <div className="auth-card">
          <h1>Create your account</h1>
          <p className="auth-subtitle">
            Start getting AI-powered stock predictions
          </p>

          <form onSubmit={handleSubmit} className="auth-form">
            {(formError || error) && (
              <div className="form-error">{formError || error}</div>
            )}

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
                Min 8 chars, with uppercase, lowercase, number, and special character
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
              <label>Select your plan</label>
              <div className="tier-selector">
                {tiers.map((tier) => (
                  <label
                    key={tier.tier}
                    className={`tier-option ${formData.tier === tier.tier ? 'selected' : ''}`}
                  >
                    <input
                      type="radio"
                      name="tier"
                      value={tier.tier}
                      checked={formData.tier === tier.tier}
                      onChange={handleChange}
                    />
                    <div className="tier-option-content">
                      <div className="tier-option-header">
                        <span className="tier-name">{tier.name}</span>
                        <span className="tier-price">
                          {tier.price === 0 ? 'Free' : `$${tier.price}/mo`}
                        </span>
                      </div>
                      <ul className="tier-features-mini">
                        <li>{tier.stocks_limit === -1 ? 'All stocks' : `${tier.stocks_limit} stocks`}</li>
                        <li>{tier.timeframes.length} timeframe{tier.timeframes.length > 1 ? 's' : ''}</li>
                      </ul>
                    </div>
                  </label>
                ))}
              </div>
              <Link to="/pricing" className="form-link">Compare plans in detail</Link>
            </div>

            <button type="submit" className="btn btn-primary btn-block" disabled={loading}>
              {loading ? 'Creating account...' : 'Create account'}
            </button>
          </form>

          <p className="auth-footer">
            Already have an account? <Link to="/login">Log in</Link>
          </p>
        </div>
      </div>
    </div>
  );
}

export default Signup;
