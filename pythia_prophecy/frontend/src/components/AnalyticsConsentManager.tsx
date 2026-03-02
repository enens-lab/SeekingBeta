import { useEffect, useState } from 'react';
import {
  AnalyticsConsent,
  getAnalyticsConsent,
  hasStoredAnalyticsConsent,
  updateAnalyticsConsent,
} from '../lib/analytics';

type ConsentToggleKey = Exclude<keyof AnalyticsConsent, 'source' | 'updated_at'>;

type ConsentOption = {
  key: ConsentToggleKey;
  label: string;
  description: string;
  signal: string;
};

const CONSENT_OPTIONS: ConsentOption[] = [
  {
    key: 'analytics_storage',
    label: 'Usage analytics',
    description: 'Helps us understand site traffic and product usage.',
    signal: 'analytics_storage',
  },
  {
    key: 'ad_storage',
    label: 'Ads cookies',
    description: 'Allows ads-related cookies for campaign measurement.',
    signal: 'ad_storage',
  },
  {
    key: 'ad_user_data',
    label: 'Ads measurement data',
    description: 'Allows sending data used for ad performance reporting.',
    signal: 'ad_user_data',
  },
  {
    key: 'ad_personalization',
    label: 'Ads personalization',
    description: 'Allows use of data for personalized ads and remarketing.',
    signal: 'ad_personalization',
  },
];

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

  const toggleConsentField = (key: ConsentToggleKey, granted: boolean) => {
    setFormConsent((prev) => ({
      ...prev,
      [key]: granted ? 'granted' : 'denied',
    }));
  };

  return (
    <>
      {showBanner && (
        <div className="consent-banner" role="dialog" aria-live="polite" aria-label="Cookie consent">
          <div className="consent-banner-content">
            <p>
              We use optional analytics and advertising signals to improve measurement. Required site
              features always stay on.
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
              Choose which optional analytics and advertising signals are enabled. Core site
              functionality remains on.
            </p>

            <div className="consent-required-row">
              <span className="consent-required-dot" aria-hidden="true" />
              <div>
                <strong>Required functionality (always on)</strong>
                <small>Authentication, security, and core site operations.</small>
              </div>
            </div>

            {CONSENT_OPTIONS.map((option) => (
              <label className="consent-row" key={option.key}>
                <input
                  type="checkbox"
                  checked={formConsent[option.key] === 'granted'}
                  onChange={(event) => toggleConsentField(option.key, event.target.checked)}
                />
                <span className="consent-row-content">
                  <strong>{option.label}</strong>
                  <small>{option.description}</small>
                  <small className="consent-signal">Signal: {option.signal}</small>
                </span>
              </label>
            ))}

            <div className="consent-modal-actions">
              <button type="button" className="btn btn-outline" onClick={rejectAll}>
                Reject All
              </button>
              <button type="button" className="btn btn-ghost" onClick={acceptAll}>
                Accept All
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
