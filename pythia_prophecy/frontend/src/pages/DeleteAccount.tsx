import { FormEvent, useState } from 'react';
import Header from '../components/Header';
import Footer from '../components/Footer';
import { auth, setToken } from '../api/client';

type Status = 'idle' | 'working' | 'done';

function DeleteAccount() {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirmText, setConfirmText] = useState('');
  const [status, setStatus] = useState<Status>('idle');
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (event: FormEvent) => {
    event.preventDefault();
    setError(null);

    if (!email || !password) {
      setError('Enter the email and password for the account you want to delete.');
      return;
    }
    if (confirmText.trim().toUpperCase() !== 'DELETE') {
      setError('Type DELETE in the confirmation box to proceed.');
      return;
    }

    setStatus('working');
    try {
      // Authenticate, then permanently delete — same server flow as the in-app
      // "Delete account" action (POST /api/auth/delete-account).
      const res = await auth.login({ email: email.trim(), password });
      setToken(res.access_token);
      await auth.deleteAccount({ password, confirm_text: 'DELETE' });
      setToken(null); // account is gone; never leave a session behind
      setStatus('done');
    } catch (err) {
      setToken(null);
      setStatus('idle');
      setError(err instanceof Error ? err.message : 'Could not delete the account. Please try again or email support@seekingbeta.ai.');
    }
  };

  return (
    <>
      <Header />
      <main className="legal-page">
        <article className="legal-content">
          <h1>Delete your account</h1>
          <p className="legal-updated">Last updated: June 18, 2026</p>
          <p>
            You can permanently delete your SeekingBeta.AI account and the personal data associated
            with it at any time — from inside the app or here on the web. Deletion is permanent and
            cannot be undone.
          </p>

          <section>
            <h2>Delete your account here</h2>
            <p>
              Enter your account credentials and type <strong>DELETE</strong> to confirm. This
              permanently removes your account without needing to install the app.
            </p>

            {status === 'done' ? (
              <div className="form-success" role="status">
                <p>
                  Your account and associated data have been deleted. Any active subscription has
                  been canceled. You can close this page.
                </p>
              </div>
            ) : (
              <form onSubmit={handleSubmit} className="auth-form" style={{ maxWidth: 420 }}>
                {error && (
                  <div className="form-error" role="alert">
                    {error}
                  </div>
                )}
                <div className="form-group">
                  <label htmlFor="delete-email">Email</label>
                  <input
                    id="delete-email"
                    type="email"
                    autoComplete="email"
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    disabled={status === 'working'}
                  />
                </div>
                <div className="form-group">
                  <label htmlFor="delete-password">Password</label>
                  <input
                    id="delete-password"
                    type="password"
                    autoComplete="current-password"
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    disabled={status === 'working'}
                  />
                </div>
                <div className="form-group">
                  <label htmlFor="delete-confirm">Type DELETE to confirm</label>
                  <input
                    id="delete-confirm"
                    type="text"
                    autoComplete="off"
                    placeholder="DELETE"
                    value={confirmText}
                    onChange={(e) => setConfirmText(e.target.value)}
                    disabled={status === 'working'}
                  />
                </div>
                <button type="submit" className="btn btn-primary btn-block" disabled={status === 'working'}>
                  {status === 'working' ? 'Deleting…' : 'Permanently delete my account'}
                </button>
              </form>
            )}
          </section>

          <section>
            <h2>Other ways to delete</h2>
            <ul>
              <li>
                <strong>In the app:</strong> open <em>Profile → Delete account</em>, confirm with
                your password, and your account is removed immediately.
              </li>
              <li>
                <strong>By email:</strong> write to{' '}
                <a href="mailto:support@seekingbeta.ai?subject=Account%20deletion%20request">
                  support@seekingbeta.ai
                </a>{' '}
                from your account email address and we will delete your account and data within 30
                days.
              </li>
            </ul>
          </section>

          <section>
            <h2>What gets deleted</h2>
            <p>When your account is deleted, we permanently remove:</p>
            <ul>
              <li>your account profile (name and email address);</li>
              <li>your login credentials;</li>
              <li>your watchlist and saved preferences;</li>
              <li>any active subscription, which is canceled as part of deletion.</li>
            </ul>
          </section>

          <section>
            <h2>What we may retain</h2>
            <p>
              We may retain a limited set of records where required by law or for legitimate
              business purposes — for example, billing and transaction records needed for tax,
              accounting, or fraud-prevention obligations. Any retained records are kept only for as
              long as legally required and are then deleted. Retained records are not used to
              re-create your account or to contact you for marketing.
            </p>
          </section>
        </article>
      </main>
      <Footer />
    </>
  );
}

export default DeleteAccount;
