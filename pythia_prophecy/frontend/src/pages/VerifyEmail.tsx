import { useState, useEffect } from 'react';
import { Link, useNavigate, useSearchParams } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';

type VerificationStatus = 'form' | 'verifying' | 'success' | 'error';

function VerifyEmail() {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const { verifyEmail } = useAuth();

  const [status, setStatus] = useState<VerificationStatus>('form');
  const [error, setError] = useState('');
  const [code, setCode] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const token = searchParams.get('token');

  // If token is in URL, use it automatically
  useEffect(() => {
    if (token) {
      handleVerify(token);
    }
  }, [token]);

  const handleVerify = async (verificationCode: string) => {
    setIsLoading(true);
    setStatus('verifying');
    try {
      await verifyEmail(verificationCode);
      setStatus('success');
      setTimeout(() => {
        navigate('/dashboard');
      }, 2000);
    } catch (err) {
      setStatus('error');
      setError(err instanceof Error ? err.message : 'Verification failed');
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
          {status === 'form' && (
            <>
              <h1>Verify your email</h1>
              <p className="auth-subtitle">
                Enter the 6-digit code sent to your email
              </p>

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
                <p className="auth-subtitle">
                  Didn&apos;t receive the code?{' '}
                  <Link to="/login" className="link">
                    Back to login
                  </Link>
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
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}

export default VerifyEmail;
