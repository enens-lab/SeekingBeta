import { useState, useEffect } from 'react';
import { Link, useNavigate, useSearchParams } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import ThemeToggle from '../components/ThemeToggle';
import { trackEvent } from '../lib/analytics';

type VerificationStatus = 'form' | 'verifying' | 'success' | 'error';

function VerifyEmail() {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const { verifyEmail, resendVerification } = useAuth();

  const [status, setStatus] = useState<VerificationStatus>('form');
  const [error, setError] = useState('');
  const [code, setCode] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [resendLoading, setResendLoading] = useState(false);
  const [resendMessage, setResendMessage] = useState('');

  const token = searchParams.get('token');
  const email = searchParams.get('email') || '';

  // If token is in URL, use it automatically
  useEffect(() => {
    if (token) {
      handleVerify(token);
    }
  }, [token]);

  const handleVerify = async (verificationCode: string) => {
    setIsLoading(true);
    setStatus('verifying');
    trackEvent('email_verification_attempt', { source: token ? 'token_link' : 'manual_code' });
    try {
      await verifyEmail(verificationCode);
      setStatus('success');
      trackEvent('email_verification_success');
      setTimeout(() => {
        navigate('/dashboard');
      }, 2000);
    } catch (err) {
      setStatus('error');
      setError(err instanceof Error ? err.message : 'Verification failed');
      trackEvent('email_verification_error');
      setIsLoading(false);
    }
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (code.trim().length !== 6) {
      setError('Please enter a 6-digit code');
      return;
    }
    handleVerify(code.trim());
  };

  const handleResend = async () => {
    if (!email) {
      setError('No email provided. Return to login and request a new verification email.');
      return;
    }

    setResendLoading(true);
    setResendMessage('');
    trackEvent('resend_verification_attempt', { source: 'verify_email' });
    try {
      const response = await resendVerification(email);
      setResendMessage(response.message);
      trackEvent('resend_verification_success', { source: 'verify_email' });
    } catch (err) {
      trackEvent('resend_verification_error', { source: 'verify_email' });
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
          {status === 'form' && (
            <>
              <h1>Verify your email</h1>
              <p className="auth-subtitle">
                Enter the 6-digit code sent to your email
              </p>
              {email && <p className="auth-note">Verifying: <strong>{email}</strong></p>}

              <form onSubmit={handleSubmit} className="auth-form">
                <div className="form-group">
                  <label htmlFor="code">Verification Code</label>
                  <input
                    id="code"
                    type="text"
                    inputMode="numeric"
                    placeholder="000000"
                    value={code}
                    onChange={(e) => {
                      const value = e.target.value.replace(/\D/g, '').slice(0, 6);
                      setCode(value);
                      setError('');
                    }}
                    maxLength={6}
                    disabled={isLoading}
                    className="code-input"
                  />
                </div>

                {error && <div className="form-error">{error}</div>}

                <button
                  type="submit"
                  disabled={isLoading || code.length !== 6}
                  className="btn btn-primary btn-block"
                >
                  {isLoading ? 'Verifying...' : 'Verify Code'}
                </button>
              </form>

              <div className="auth-footer">
                <p className="auth-subtitle">Didn&apos;t receive the code?</p>
                <div className="auth-actions">
                  <button
                    type="button"
                    className="btn btn-outline"
                    onClick={handleResend}
                    disabled={resendLoading || !email}
                  >
                    {resendLoading ? 'Resending...' : 'Resend verification email'}
                  </button>
                </div>
                {resendMessage && <p className="auth-note">{resendMessage}</p>}
                <p className="auth-note">
                  <Link to="/login" className="link">Back to login</Link>
                </p>
              </div>
            </>
          )}

          {status === 'verifying' && (
            <>
              <div className="spinner" />
              <h1>Verifying your email...</h1>
              <p className="auth-subtitle">Please wait a moment</p>
            </>
          )}

          {status === 'success' && (
            <>
              <div className="success-icon">
                <svg width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14" />
                  <polyline points="22 4 12 14.01 9 11.01" />
                </svg>
              </div>
              <h1>Email verified!</h1>
              <p className="auth-subtitle">
                Your account is now active. Redirecting to dashboard...
              </p>
            </>
          )}

          {status === 'error' && (
            <>
              <div className="error-icon">
                <svg width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <circle cx="12" cy="12" r="10" />
                  <line x1="15" y1="9" x2="9" y2="15" />
                  <line x1="9" y1="9" x2="15" y2="15" />
                </svg>
              </div>
              <h1>Verification failed</h1>
              <p className="auth-subtitle">{error}</p>
              <div className="auth-actions">
                <button
                  onClick={() => {
                    setStatus('form');
                    setCode('');
                    setError('');
                  }}
                  className="btn btn-primary"
                >
                  Try again
                </button>
                {email && (
                  <button
                    type="button"
                    className="btn btn-outline"
                    onClick={handleResend}
                    disabled={resendLoading}
                  >
                    {resendLoading ? 'Resending...' : 'Resend email'}
                  </button>
                )}
              </div>
              {resendMessage && <p className="auth-note">{resendMessage}</p>}
            </>
          )}
        </div>
      </div>
    </div>
  );
}

export default VerifyEmail;
