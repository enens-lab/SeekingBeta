import Header from '../components/Header';
import Footer from '../components/Footer';

function Support() {
  return (
    <>
      <Header />
      <main className="legal-page">
        <article className="legal-content">
          <h1>Support</h1>
          <p className="legal-updated">We typically reply within 1–2 business days.</p>
          <p>
            Need help with your account, billing, or the SeekingBeta.AI app? Email us at{' '}
            <a href="mailto:support@seekingbeta.ai">support@seekingbeta.ai</a> and include the
            email address on your account plus a short description of the issue.
          </p>

          <section>
            <h2>Account &amp; sign-in</h2>
            <ul>
              <li>
                Forgot your password? Use the reset link on the login screen, or in the iOS app go
                to the Account tab and choose the password reset option.
              </li>
              <li>
                Didn&apos;t receive your verification email? Check spam, then request a new one from
                the verification screen. Verification links expire after a short period.
              </li>
              <li>
                You can change your password anytime from Profile once you are signed in.
              </li>
            </ul>
          </section>

          <section>
            <h2>Subscriptions &amp; billing</h2>
            <ul>
              <li>
                <strong>Purchased in the iOS app (App Store):</strong> subscriptions are billed to
                your Apple Account and renew automatically unless cancelled at least 24 hours
                before the end of the current period. Manage or cancel in Settings → your name →
                Subscriptions on your device, or via the &quot;Manage in App Store&quot; button in
                the app&apos;s Profile tab. Use &quot;Restore Purchases&quot; in Profile if your
                plan does not appear after reinstalling or switching devices.
              </li>
              <li>
                <strong>Purchased on the website:</strong> manage your plan from your website
                account. See our <a href="/refund-cancellation">Refund &amp; Cancellation Policy</a>{' '}
                for details.
              </li>
              <li>
                Refunds for App Store purchases are handled by Apple at{' '}
                <a href="https://reportaproblem.apple.com" target="_blank" rel="noreferrer">
                  reportaproblem.apple.com
                </a>
                .
              </li>
            </ul>
          </section>

          <section>
            <h2>Deleting your account</h2>
            <p>
              You can permanently delete your account in the iOS app under Profile → Delete
              Account, or by emailing{' '}
              <a href="mailto:support@seekingbeta.ai">support@seekingbeta.ai</a> from the address on
              your account. Deleting your account does not cancel an active App Store subscription —
              cancel the renewal in your App Store subscriptions first.
            </p>
          </section>

          <section>
            <h2>Predictions &amp; data</h2>
            <ul>
              <li>
                Signals and boards are informational and educational only — not investment advice.
                Predictions are probabilistic and may be wrong; past performance does not guarantee
                future results.
              </li>
              <li>
                Stock signals refresh on market days; sports boards update as new slates and
                results come in. If something looks stale, pull to refresh in the app.
              </li>
              <li>
                The iOS app can run signals offline using an on-device model — the source pill
                shows &quot;Offline&quot; instead of &quot;Live&quot; when it does.
              </li>
            </ul>
          </section>

          <section>
            <h2>Policies</h2>
            <p>
              <a href="/terms">Terms of Service</a> · <a href="/privacy">Privacy Policy</a> ·{' '}
              <a href="/refund-cancellation">Refund &amp; Cancellation</a> ·{' '}
              <a href="/methodology">Methodology</a>
            </p>
          </section>
        </article>
      </main>
      <Footer />
    </>
  );
}

export default Support;
