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

          <section>
            <h2>1. Agreement</h2>
            <p>
              These Terms of Service govern your access to and use of SeekingBeta. By using
              SeekingBeta, you agree to these Terms.
            </p>
          </section>

          <section>
            <h2>2. Service Description</h2>
            <p>
              SeekingBeta provides stock analysis, model outputs, and decision-support content for
              educational and research purposes.
            </p>
          </section>

          <section>
            <h2>3. No Financial Advice</h2>
            <p>
              SeekingBeta is not a broker-dealer, investment adviser, or financial planner. Content
              provided by SeekingBeta is not personalized investment advice and should not be the
              sole basis for any investment decision.
            </p>
          </section>

          <section>
            <h2>4. Accounts</h2>
            <p>
              You are responsible for maintaining account security and for all activity under your
              account. You must provide accurate registration information.
            </p>
          </section>

          <section>
            <h2>5. Subscriptions, Billing, and Cancellation</h2>
            <p>
              Paid features are provided under recurring subscriptions when enabled. You may cancel
              at any time; cancellation takes effect at the end of your current paid billing period.
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
              reliability.
            </p>
          </section>

          <section>
            <h2>8. Disclaimers and Limitation of Liability</h2>
            <p>
              SeekingBeta is provided on an “as is” and “as available” basis. We do not guarantee
              uninterrupted availability, data accuracy, model performance, or any financial outcome.
              To the fullest extent permitted by law, SeekingBeta disclaims liability for indirect,
              incidental, special, consequential, or punitive damages.
            </p>
          </section>

          <section>
            <h2>9. Governing Law</h2>
            <p>
              These Terms are governed by the laws of California, USA, without regard to conflict of
              law principles.
            </p>
          </section>

          <section>
            <h2>10. Contact and Legal Notices</h2>
            <p>
              Questions and legal notices should be sent to{' '}
              <a href="mailto:support@seekingbeta.ai">support@seekingbeta.ai</a>.
            </p>
          </section>
        </article>
      </main>
      <Footer />
    </>
  );
}

export default Terms;
