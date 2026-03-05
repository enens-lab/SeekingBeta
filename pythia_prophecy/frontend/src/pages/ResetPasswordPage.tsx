import { FormEvent, useMemo, useState } from 'react';
import { Link, useNavigate, useSearchParams } from 'react-router-dom';
import ThemeToggle from '../components/ThemeToggle';
import { auth } from '../api/client';
import { trackEvent } from '../lib/analytics';

function validatePassword(password: string): string | null {
  if (password.length < 8) {
    return 'Password must be at least 8 characters.';
  }
  if (!/[A-Z]/.test(password)) {
    return 'Password must include at least one uppercase letter.';
  }
  if (!/[a-z]/.test(password)) {
    return 'Password must include at least one lowercase letter.';
  }
  if (!/[0-9]/.test(password)) {
    return 'Password must include at least one number.';
  }
  if (!/[!@#$%^&*()_\-+=[\]{};:'",.<>?/\\|`~]/.test(password)) {
    return 'Password must include at least one special character.';
  }
  return null;
}

function ResetPasswordPage() {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const token = (searchParams.get('token') || '').trim();
  const emailFromUrl = (searchParams.get('email') || '').trim();
  const hasToken = token.length > 0;

  const [email, setEmail] = useState(emailFromUrl);
  const [newPassword, setNewPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');
  const [passwordResetDone, setPasswordResetDone] = useState(false);

  const pageTitle = useMemo(
    () => (hasToken ? 'Set a new password' : 'Reset your password'),
    [hasToken],
  );

  const handleRequestReset = async (event: FormEvent) => {
    event.preventDefault();
    setError('');
    setSuccess('');

    const normalizedEmail = email.trim();
    if (!normalizedEmail) {
      setError('Enter the email address for your account.');
      return;
    }

    setLoading(true);
    try {
      const response = await auth.requestPasswordReset({ email: normalizedEmail });
      setSuccess(
        response.message ||
          'If the email exists, a password reset link has been sent.',
      );
      trackEvent('password_reset_request_submitted');
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to request reset.';
      setError(message);
      trackEvent('password_reset_request_failed');
    } finally {
      setLoading(false);
    }
  };

  const handleConfirmReset = async (event: FormEvent) => {
    event.preventDefault();
    setError('');
    setSuccess('');

    if (!token) {
      setError('Reset token is missing. Request a new reset link.');
      return;
    }

    if (newPassword !== confirmPassword) {
      setError('Passwords do not match.');
      return;
    }

    const validationError = validatePassword(newPassword);
    if (validationError) {
      setError(validationError);
      return;
    }

    setLoading(true);
    try {
      const response = await auth.confirmPasswordReset({
        token,
        new_password: newPassword,
      });
      setSuccess(
        response.message ||
          'Password reset successfully. You can now log in with your new password.',
      );
      setPasswordResetDone(true);
      trackEvent('password_reset_confirm_success');
    } catch (err) {
      const message =
        err instanceof Error ? err.message : 'Failed to reset password.';
      setError(message);
      trackEvent('password_reset_confirm_failed');
    } finally {
      setLoading(false);
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
          <h1>{pageTitle}</h1>
          <p className="auth-subtitle">
            {hasToken
              ? 'Choose a new password for your account.'
              : 'Enter your email and we will send a reset link.'}
          </p>

          {error && <div className="form-error">{error}</div>}
          {success && <div className="form-success">{success}</div>}

          {!hasToken && (
            <form onSubmit={handleRequestReset} className="auth-form">
              <div className="form-group">
                <label htmlFor="email">Email</label>
                <input
                  type="email"
                  id="email"
                  name="email"
                  value={email}
                  onChange={(event) => setEmail(event.target.value)}
                  autoComplete="email"
                  required
                />
              </div>

              <button type="submit" className="btn btn-primary btn-block" disabled={loading}>
                {loading ? 'Sending reset link...' : 'Send reset link'}
              </button>
            </form>
          )}

          {hasToken && !passwordResetDone && (
            <form onSubmit={handleConfirmReset} className="auth-form">
              <div className="form-group">
                <label htmlFor="newPassword">New password</label>
                <input
                  type="password"
                  id="newPassword"
                  value={newPassword}
                  onChange={(event) => setNewPassword(event.target.value)}
                  autoComplete="new-password"
                  required
                />
                <span className="form-hint">
                  Min 8 chars with uppercase, lowercase, number, and special character.
                </span>
              </div>

              <div className="form-group">
                <label htmlFor="confirmPassword">Confirm password</label>
                <input
                  type="password"
                  id="confirmPassword"
                  value={confirmPassword}
                  onChange={(event) => setConfirmPassword(event.target.value)}
                  autoComplete="new-password"
                  required
                />
              </div>

              <button type="submit" className="btn btn-primary btn-block" disabled={loading}>
                {loading ? 'Resetting password...' : 'Reset password'}
              </button>
            </form>
          )}

          <div className="auth-inline-actions">
            {passwordResetDone ? (
              <button
                type="button"
                className="btn btn-outline btn-block"
                onClick={() => navigate('/login', { replace: true })}
              >
                Go to login
              </button>
            ) : (
              <p className="auth-footer">
                <Link to="/login">Back to login</Link>
              </p>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export default ResetPasswordPage;
