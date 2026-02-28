import Header from '../components/Header';
import Footer from '../components/Footer';

function RefundCancellation() {
  return (
    <>
      <Header />
      <main className="legal-page">
        <article className="legal-content">
          <h1>Refund &amp; Cancellation Policy</h1>
          <p className="legal-updated">Last updated: February 27, 2026</p>

          <section>
            <h2>1. Cancellation</h2>
            <p>
              You may cancel your subscription at any time. Cancellation prevents future renewal
              charges and takes effect at the end of your current paid billing period.
            </p>
          </section>

          <section>
            <h2>2. Access After Cancellation</h2>
            <p>
              After cancellation, you retain access to paid features through the remainder of the
              billing period that has already been paid.
            </p>
          </section>

          <section>
            <h2>3. Refunds and Proration</h2>
            <p>
              Except where required by law, we do not provide refunds or prorated credits after a
              billing cycle starts.
            </p>
          </section>

          <section>
            <h2>4. Billing Disputes</h2>
            <p>
              If you believe there is a billing error, contact{' '}
              <a href="mailto:support@seekingbeta.ai">support@seekingbeta.ai</a> promptly with your
              account email and transaction details.
            </p>
          </section>

          <section>
            <h2>5. Contact</h2>
            <p>
              For cancellation, refund, or billing support, email{' '}
              <a href="mailto:support@seekingbeta.ai">support@seekingbeta.ai</a>.
            </p>
          </section>
        </article>
      </main>
      <Footer />
    </>
  );
}

export default RefundCancellation;
