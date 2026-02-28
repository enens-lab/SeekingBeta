import Header from '../components/Header';
import Footer from '../components/Footer';

function Privacy() {
  return (
    <>
      <Header />
      <main className="legal-page">
        <article className="legal-content">
          <h1>Privacy Policy</h1>
          <p className="legal-updated">Last updated: February 27, 2026</p>

          <section>
            <h2>1. Information We Collect</h2>
            <p>We collect the following categories of information:</p>
            <ul>
              <li>Account data (name, email, password hash, plan tier, verification status).</li>
              <li>Usage data (app activity, watchlist usage, request logs, diagnostics).</li>
              <li>Email event data (delivery, bounce, complaint, suppression status).</li>
              <li>Security data (IP address, user agent, consent and authentication events).</li>
            </ul>
          </section>

          <section>
            <h2>2. How We Use Information</h2>
            <p>We use personal information to:</p>
            <ul>
              <li>Provide and secure the service.</li>
              <li>Authenticate users and manage subscriptions.</li>
              <li>Send transactional communications (verification and account notices).</li>
              <li>Maintain deliverability and abuse prevention systems.</li>
              <li>Comply with legal obligations.</li>
            </ul>
          </section>

          <section>
            <h2>3. Legal Basis</h2>
            <p>
              We process information based on your consent, contractual necessity, legitimate
              interests in operating a secure service, and legal compliance obligations.
            </p>
          </section>

          <section>
            <h2>4. Email Communications</h2>
            <p>
              Transactional emails are sent for account operations. Marketing emails are optional and
              can be unsubscribed from at any time. We automatically suppress addresses for bounce and
              complaint events to reduce abuse risk.
            </p>
          </section>

          <section>
            <h2>5. Data Sharing</h2>
            <p>We share data only with service providers required to run SeekingBeta, including:</p>
            <ul>
              <li>AWS infrastructure and SES messaging services.</li>
              <li>SMTP/email delivery providers configured for transactional messaging.</li>
              <li>Hosting and operational tooling used to deliver and secure the service.</li>
            </ul>
          </section>

          <section>
            <h2>6. Data Retention</h2>
            <p>
              We retain data for as long as needed to provide the service, enforce our agreements,
              resolve disputes, and satisfy legal obligations.
            </p>
          </section>

          <section>
            <h2>7. Security</h2>
            <p>
              We apply reasonable administrative, technical, and organizational safeguards. No system
              can guarantee absolute security.
            </p>
          </section>

          <section>
            <h2>8. Your Rights and Choices</h2>
            <p>
              You may request account or privacy support by contacting{' '}
              <a href="mailto:support@seekingbeta.ai">support@seekingbeta.ai</a>. We will respond
              according to applicable law.
            </p>
          </section>

          <section>
            <h2>9. Policy Updates</h2>
            <p>
              We may update this Policy periodically. Material updates will be reflected by an
              updated effective date.
            </p>
          </section>
        </article>
      </main>
      <Footer />
    </>
  );
}

export default Privacy;
