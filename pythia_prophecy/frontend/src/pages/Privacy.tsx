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
          <p>
            This Privacy Policy explains how EnEns LLC ("EnEns," "we," "us," or "our")
            collects, uses, shares, and protects information when you access or use SeekingBeta
            at seekingbeta.ai (the "Service").
          </p>

          <section>
            <h2>1. Information We Collect</h2>
            <p>We collect the following categories of information:</p>
            <p>A. Account data</p>
            <p>
              Name, email address, password hash, plan tier, verification status, and account
              preferences.
            </p>
            <p>B. Usage and diagnostics data</p>
            <p>
              App activity (including watchlist usage), request logs, feature usage, performance
              and diagnostics data, and error reports.
            </p>
            <p>C. Email event and deliverability data</p>
            <p>
              Delivery, bounce, complaint, and suppression status for transactional and optional
              marketing emails.
            </p>
            <p>D. Security and access data</p>
            <p>
              IP address, device/browser user agent, consent and authentication events, and
              security-related logs.
            </p>
            <p>E. Payment information (processed by Stripe)</p>
            <p>
              If you purchase a subscription (for example, a paid plan currently offered at
              $9.99/month or $19.99/month), payments are processed by Stripe, Inc. ("Stripe"), our
              payment processor. We do not store your full payment card number. Stripe may collect
              and process your payment information in accordance with its own privacy practices. We
              receive limited payment-related details such as subscription status, billing period,
              payment method type, the last four digits of a card, and transaction identifiers.
            </p>
          </section>

          <section>
            <h2>2. How We Use Information</h2>
            <p>We use information to:</p>
            <ul>
              <li>Provide, operate, maintain, and secure the Service (including free and paid plans).</li>
              <li>Authenticate users and manage subscriptions and billing status.</li>
              <li>Send transactional communications (verification, security, billing, and account notices).</li>
              <li>Maintain deliverability and abuse prevention systems (including bounce/complaint suppression).</li>
              <li>Monitor reliability, troubleshoot issues, and improve Service performance.</li>
              <li>Comply with legal obligations and enforce our agreements.</li>
            </ul>
          </section>

          <section>
            <h2>3. Email Communications</h2>
            <p>
              Transactional emails are sent for account operations (such as verification, security
              notices, and billing/account notices). Marketing emails (if offered) are optional and
              can be unsubscribed from at any time using the link in the email. We automatically
              suppress addresses associated with repeated bounce and complaint events to reduce abuse
              risk and protect deliverability.
            </p>
          </section>

          <section>
            <h2>4. Cookies and Similar Technologies</h2>
            <p>We may use cookies and similar technologies to:</p>
            <ul>
              <li>Keep you signed in and maintain session integrity,</li>
              <li>Remember preferences,</li>
              <li>Measure reliability and usage (analytics),</li>
              <li>Prevent fraud and abuse.</li>
            </ul>
            <p>
              You can control cookies through your browser settings. If you disable cookies, some
              features of the Service may not function properly.
            </p>
          </section>

          <section>
            <h2>5. Data Sharing</h2>
            <p>We share data only with service providers required to run SeekingBeta, including:</p>
            <ul>
              <li>AWS infrastructure and hosting services.</li>
              <li>
                AWS SES and/or other SMTP/email delivery providers configured for transactional
                messaging and deliverability management.
              </li>
              <li>Stripe for payment processing and subscription management.</li>
              <li>
                Hosting, monitoring, logging, analytics, and security tooling used to deliver and
                protect the Service.
              </li>
            </ul>
            <p>
              Service providers are authorized to process information only on our instructions and
              for the purposes described in this Policy.
            </p>
            <p>We may also share information:</p>
            <ul>
              <li>
                For legal, safety, and compliance reasons if we reasonably believe disclosure is
                necessary to comply with law, respond to lawful requests, protect rights and safety,
                prevent fraud or abuse, or enforce our Terms.
              </li>
              <li>
                As part of a business transfer (e.g., merger, acquisition, financing,
                reorganization, bankruptcy, or sale of assets), where information may be transferred
                as part of that transaction.
              </li>
              <li>With your direction (for example, if you request a data export).</li>
            </ul>
            <p>No sale of personal information. We do not sell personal information in exchange for money.</p>
          </section>

          <section>
            <h2>6. Data Retention</h2>
            <p>
              We retain information for as long as needed to provide the Service, enforce our
              agreements, resolve disputes, maintain security and abuse-prevention logs, and satisfy
              legal obligations. Retention depends on data type and context. We may retain certain
              records after account closure where required or reasonably necessary for the purposes
              above.
            </p>
          </section>

          <section>
            <h2>7. Security</h2>
            <p>
              We apply reasonable administrative, technical, and organizational safeguards designed
              to protect information. However, no system can guarantee absolute security.
            </p>
          </section>

          <section>
            <h2>8. Your Rights and Choices (US)</h2>
            <p>
              You may request account or privacy support by contacting{' '}
              <a href="mailto:support@seekingbeta.ai">support@seekingbeta.ai</a>. We will respond
              consistent with applicable law and may need to verify your request.
            </p>
            <p>
              If you want to update your payment method or billing details, you can do so through
              your account billing settings (powered by Stripe) or by contacting{' '}
              <a href="mailto:support@seekingbeta.ai">support@seekingbeta.ai</a>.
            </p>
          </section>

          <section>
            <h2>9. Policy Updates</h2>
            <p>
              We may update this Policy periodically. Material updates will be reflected by an
              updated effective date and, where appropriate, additional notice.
            </p>
          </section>

          <section>
            <h2>10. Contact</h2>
            <p>EnEns LLC</p>
            <p>
              Email: <a href="mailto:support@seekingbeta.ai">support@seekingbeta.ai</a>
            </p>
            <p>Address: [Your business mailing address]</p>
          </section>
        </article>
      </main>
      <Footer />
    </>
  );
}

export default Privacy;
