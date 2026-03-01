import { useEffect, useState } from 'react';
import {
  AnalyticsConsent,
  getAnalyticsConsent,
  hasStoredAnalyticsConsent,
  updateAnalyticsConsent,
} from '../lib/analytics';

function buildDeniedConsent(): AnalyticsConsent {
  return {
    analytics_storage: 'denied',
    ad_storage: 'denied',
    ad_user_data: 'denied',
    ad_personalization: 'denied',
  };
}

function buildGrantedConsent(): AnalyticsConsent {
  return {
    analytics_storage: 'granted',
    ad_storage: 'granted',
    ad_user_data: 'granted',
    ad_personalization: 'granted',
  };
}

function AnalyticsConsentManager() {
  const [showBanner, setShowBanner] = useState(false);
  const [showModal, setShowModal] = useState(false);
  const [formConsent, setFormConsent] = useState<AnalyticsConsent>(buildDeniedConsent());

  useEffect(() => {
    const stored = hasStoredAnalyticsConsent();
    const current = getAnalyticsConsent();
    setFormConsent(current);
    setShowBanner(!stored);
  }, []);

  const saveConsent = (source: string) => {
    updateAnalyticsConsent(formConsent, source);
    setShowBanner(false);
    setShowModal(false);
  };

  const acceptAll = () => {
    setFormConsent(buildGrantedConsent());
    updateAnalyticsConsent(buildGrantedConsent(), 'consent_accept_all');
    setShowBanner(false);
    setShowModal(false);
  };

  const rejectAll = () => {
    setFormConsent(buildDeniedConsent());
    updateAnalyticsConsent(buildDeniedConsent(), 'consent_reject_all');
    setShowBanner(false);
    setShowModal(false);
  };

  return (
    <>
      {showBanner && (
        <div className="consent-banner" role="dialog" aria-live="polite" aria-label="Cookie consent">
          <div className="consent-banner-content">
            <p>
              We use analytics and ads consent signals for measurement and modeling. You can accept,
              reject, or customize your consent preferences.
            </p>
            <div className="consent-banner-actions">
              <button type="button" className="btn btn-outline" onClick={rejectAll}>
                Reject Non-Essential
              </button>
              <button
                type="button"
                className="btn btn-ghost"
                onClick={() => {
                  setShowModal(true);
                  setShowBanner(false);
                }}
              >
                Customize
              </button>
              <button type="button" className="btn btn-primary" onClick={acceptAll}>
                Accept All
              </button>
            </div>
          </div>
        </div>
      )}

      {!showBanner && (
        <button
          type="button"
          className="consent-settings-btn"
          onClick={() => setShowModal(true)}
          aria-label="Open analytics consent settings"
        >
          Privacy Settings
        </button>
      )}

      {showModal && (
        <div className="consent-modal-overlay" onClick={() => setShowModal(false)}>
          <div className="consent-modal" onClick={(event) => event.stopPropagation()}>
            <h3>Analytics & Advertising Consent</h3>
            <p>
              Control which consent signals are sent to Google Analytics. Required functionality
              remains enabled.
            </p>

            <label className="consent-row">
              <input
                type="checkbox"
                checked={formConsent.analytics_storage === 'granted'}
                onChange={(event) =>
                  setFormConsent((prev) => ({
                    ...prev,
                    analytics_storage: event.target.checked ? 'granted' : 'denied',
                  }))
                }
              />
              <span>Analytics cookies (`analytics_storage`)</span>
            </label>

            <label className="consent-row">
              <input
                type="checkbox"
                checked={formConsent.ad_storage === 'granted'}
                onChange={(event) =>
                  setFormConsent((prev) => ({
                    ...prev,
                    ad_storage: event.target.checked ? 'granted' : 'denied',
                  }))
                }
              />
              <span>Ads cookies (`ad_storage`)</span>
            </label>

            <label className="consent-row">
              <input
                type="checkbox"
                checked={formConsent.ad_user_data === 'granted'}
                onChange={(event) =>
                  setFormConsent((prev) => ({
                    ...prev,
                    ad_user_data: event.target.checked ? 'granted' : 'denied',
                  }))
                }
              />
              <span>Ads measurement data (`ad_user_data`)</span>
            </label>

            <label className="consent-row">
              <input
                type="checkbox"
                checked={formConsent.ad_personalization === 'granted'}
                onChange={(event) =>
                  setFormConsent((prev) => ({
                    ...prev,
                    ad_personalization: event.target.checked ? 'granted' : 'denied',
                  }))
                }
              />
              <span>Ads personalization (`ad_personalization`)</span>
            </label>

            <div className="consent-modal-actions">
              <button type="button" className="btn btn-outline" onClick={rejectAll}>
                Reject All
              </button>
              <button type="button" className="btn btn-primary" onClick={() => saveConsent('consent_custom_save')}>
                Save Preferences
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  );
}

export default AnalyticsConsentManager;
