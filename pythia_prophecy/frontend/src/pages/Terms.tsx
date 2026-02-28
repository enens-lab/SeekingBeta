import Header from '../components/Header';
import Footer from '../components/Footer';

function Terms() {
  return (
    <>
      <Header />
      <main className="legal-page">
        <article className="legal-content">
          <h1>Terms of Service</h1>
          <p className="legal-updated">Last updated: February 27, 2026</p>
          <p>
            These Terms of Service ("Terms") govern your access to and use of SeekingBeta at
            seekingbeta.ai (the "Service"), operated by EnEns LLC ("EnEns," "we," "us," or
            "our"). By accessing or using the Service, you agree to these Terms and our Privacy
            Policy.
          </p>

          <section>
            <h2>1. Agreement</h2>
            <p>
              By using SeekingBeta, you agree to these Terms. If you do not agree, do not use the
              Service.
            </p>
          </section>

          <section>
            <h2>2. Service Description</h2>
            <p>
              SeekingBeta provides stock analysis, model outputs, and decision-support content for
              educational and research purposes. The Service may include automated or model-driven
              outputs.
            </p>
          </section>

          <section>
            <h2>3. No Financial Advice</h2>
            <p>
              SeekingBeta is not a broker-dealer, investment adviser, or financial planner. Content
              provided by SeekingBeta is not personalized investment advice and should not be the
              sole basis for any investment decision.
            </p>
            <p>
              Risk disclosure: Investing involves risk, including the possible loss of principal.
              Past performance is not indicative of future results. You are solely responsible for
              your investment decisions.
            </p>
          </section>

          <section>
            <h2>4. Accounts</h2>
            <p>
              You are responsible for maintaining account security and for all activity under your
              account. You must provide accurate registration information and keep it up to date.
            </p>
          </section>

          <section>
            <h2>5. Plans, Subscriptions, Billing, and Cancellation (Monthly; Stripe)</h2>
            <p>
              SeekingBeta may offer a free plan and paid monthly subscription plans (for example,
              plans currently offered at $9.99/month and $19.99/month). Plan features and pricing
              are described on the Service and may change from time to time.
            </p>
            <p>
              Payments are processed by Stripe. By subscribing, you authorize EnEns and Stripe to
              charge your selected payment method on a recurring monthly basis until you cancel.
            </p>
            <p>
              You may cancel at any time; cancellation takes effect at the end of your current paid
              monthly billing period. You will retain access to paid features through the end of
              that billing period.
            </p>
            <p>
              If your payment method fails or your subscription becomes past due, we may suspend or
              downgrade paid features until payment is successfully processed.
            </p>
          </section>

          <section>
            <h2>6. Refunds</h2>
            <p>
              Except where required by law, payments are non-refundable after a billing cycle
              begins. No prorated refunds are provided for partial billing periods.
            </p>
          </section>

          <section>
            <h2>7. Acceptable Use</h2>
            <p>
              You may not use SeekingBeta for unlawful activity, abuse, unauthorized access,
              scraping in violation of applicable limits, or activity that degrades service
              reliability. You may not attempt to bypass rate limits or security controls.
            </p>
            <p>
              We may suspend or terminate access to the Service if we reasonably believe you
              violated these Terms or if necessary to protect the Service and users.
            </p>
          </section>

          <section>
            <h2>8. Market Data, Timing, and Third-Party Information Disclaimer</h2>
            <p>
              The Service may display or reference market prices, quotes, charts, corporate actions,
              news, indicators, or other third-party information ("Market Data"). Market Data may
              be delayed, incomplete, inaccurate, not real-time, or subject to outage. EnEns does
              not guarantee the accuracy, completeness, timeliness, or availability of Market Data
              or any outputs derived from it.
            </p>
            <p>
              The Service is not intended for time-sensitive trading decisions. You are responsible
              for verifying information independently (including through official sources) before
              making any investment or trading decision.
            </p>
          </section>

          <section>
            <h2>9. Intellectual Property</h2>
            <p>
              The Service, including software, design, text, graphics, logos, and other content, is
              owned by EnEns or its licensors and is protected by intellectual property laws. Except
              as expressly permitted, no rights are granted to you.
            </p>
            <p>
              You may not copy, modify, distribute, sell, or lease any part of the Service or
              included content unless expressly authorized.
            </p>
          </section>

          <section>
            <h2>10. Disclaimers and Limitation of Liability</h2>
            <p>
              SeekingBeta is provided on an "as is" and "as available" basis. We do not guarantee
              uninterrupted availability, data accuracy, model performance, or any financial
              outcome.
            </p>
            <p>
              To the fullest extent permitted by law, EnEns disclaims liability for indirect,
              incidental, special, consequential, or punitive damages, and any loss of profits,
              revenue, data, or goodwill.
            </p>
            <p>
              To the fullest extent permitted by law, EnEns's total liability for any claim arising
              out of or relating to the Service will not exceed the amount you paid to EnEns for
              the Service in the 12 months before the event giving rise to the claim.
            </p>
            <p>
              Some jurisdictions do not allow certain limitations; in that case, liability is
              limited to the greatest extent permitted by law.
            </p>
          </section>

          <section>
            <h2>11. Governing Law</h2>
            <p>
              These Terms are governed by the laws of California, USA, without regard to conflict of
              law principles.
            </p>
          </section>

          <section>
            <h2>12. Contact and Legal Notices</h2>
            <p>
              Questions and legal notices should be sent to{' '}
              <a href="mailto:support@seekingbeta.ai">support@seekingbeta.ai</a>.
            </p>
            <p>EnEns LLC mailing address: [Your business mailing address]</p>
          </section>
        </article>
      </main>
      <Footer />
    </>
  );
}

export default Terms;
