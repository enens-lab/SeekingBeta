import { useEffect, useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import { OAuthPayload } from '../api/client';
import {
  availableProviders,
  renderGoogleButton,
  signInWithApple,
  signInWithFacebook,
} from '../lib/socialAuth';
import { trackEvent } from '../lib/analytics';

interface SocialAuthButtonsProps {
  // Where to go after a successful social login.
  redirectTo?: string;
  // Surface errors to the host page (e.g. its form-error banner / toast).
  onError?: (message: string) => void;
}

function AppleMark() {
  return (
    <svg width="16" height="16" viewBox="0 0 16 16" aria-hidden="true" focusable="false">
      <path
        fill="currentColor"
        d="M10.94 8.5c.02 2.07 1.82 2.76 1.84 2.77-.02.05-.29 1-.95 1.97-.57.85-1.17 1.69-2.11 1.71-.92.02-1.22-.54-2.27-.54-1.05 0-1.38.52-2.25.56-.91.03-1.6-.92-2.18-1.76-1.18-1.71-2.08-4.84-.87-6.95.6-1.05 1.68-1.71 2.85-1.73.89-.02 1.73.6 2.27.6.55 0 1.57-.74 2.64-.63.45.02 1.71.18 2.52 1.37-.07.04-1.5.88-1.49 2.63zM9.3 3.38c.48-.58.8-1.39.71-2.2-.69.03-1.53.46-2.02 1.04-.44.51-.83 1.34-.72 2.13.77.06 1.55-.39 2.03-.97z"
      />
    </svg>
  );
}

function FacebookMark() {
  return (
    <svg width="16" height="16" viewBox="0 0 16 16" aria-hidden="true" focusable="false">
      <path
        fill="currentColor"
        d="M16 8a8 8 0 1 0-9.25 7.9v-5.59H4.72V8h2.03V6.24c0-2 1.2-3.11 3.02-3.11.875 0 1.79.156 1.79.156v1.97h-1.01c-.99 0-1.3.62-1.3 1.25V8h2.22l-.355 2.31H9.25v5.59A8 8 0 0 0 16 8z"
      />
    </svg>
  );
}

export default function SocialAuthButtons({
  redirectTo = '/dashboard',
  onError,
}: SocialAuthButtonsProps) {
  const { socialLogin } = useAuth();
  const navigate = useNavigate();
  const googleRef = useRef<HTMLDivElement>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const providers = availableProviders();

  const fail = (message: string) => {
    setBusy(null);
    if (onError) onError(message);
  };

  const complete = async (provider: string, payload: OAuthPayload) => {
    try {
      setBusy(provider);
      await socialLogin(provider, payload);
      trackEvent('social_login_success', { method: provider });
      navigate(redirectTo, { replace: true });
    } catch (err) {
      trackEvent('social_login_error', { method: provider });
      fail(err instanceof Error ? err.message : 'Social login failed');
    }
  };

  const handleFacebook = async () => {
    try {
      setBusy('facebook');
      const payload = await signInWithFacebook();
      await complete('facebook', payload);
    } catch (err) {
      fail(err instanceof Error ? err.message : 'Facebook sign-in failed');
    }
  };

  const handleApple = async () => {
    try {
      setBusy('apple');
      const payload = await signInWithApple();
      await complete('apple', payload);
    } catch (err) {
      fail(err instanceof Error ? err.message : 'Apple sign-in failed');
    }
  };

  // Google uses the official Identity Services button (it returns the ID token).
  useEffect(() => {
    if (!providers.includes('google') || !googleRef.current) return;
    let active = true;
    renderGoogleButton(
      googleRef.current,
      (payload) => {
        if (active) complete('google', payload);
      },
      (msg) => {
        if (active) fail(msg);
      },
    ).catch((e) => fail(e instanceof Error ? e.message : 'Google sign-in unavailable'));
    return () => {
      active = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  if (providers.length === 0) return null;

  return (
    <div className="social-auth">
      <div className="social-buttons">
        {providers.includes('google') && <div ref={googleRef} className="social-google" />}

        {providers.includes('apple') && (
          <button
            type="button"
            className="social-btn social-btn-apple"
            onClick={handleApple}
            disabled={busy !== null}
          >
            <AppleMark />
            <span>Continue with Apple</span>
          </button>
        )}

        {providers.includes('facebook') && (
          <button
            type="button"
            className="social-btn social-btn-facebook"
            onClick={handleFacebook}
            disabled={busy !== null}
          >
            <FacebookMark />
            <span>Continue with Facebook</span>
          </button>
        )}
      </div>

      <p className="social-consent">
        By continuing, you agree to our <a href="/terms">Terms</a> and{' '}
        <a href="/privacy">Privacy Policy</a>.
      </p>

      <div className="social-divider">
        <span>or</span>
      </div>
    </div>
  );
}
