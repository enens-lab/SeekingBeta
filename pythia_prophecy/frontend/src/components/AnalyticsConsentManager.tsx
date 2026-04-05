import { useEffect, useState } from 'react';
import {
  AnalyticsConsent,
  getAnalyticsConsent,
  hasStoredAnalyticsConsent,
  updateAnalyticsConsent,
} from '../lib/analytics';

type ConsentToggleKey = 'analytics_storage';

type ConsentOption = {
  key: ConsentToggleKey;
  label: string;
  description: string;
};

const CONSENT_OPTIONS: ConsentOption[] = [
  {
    key: 'analytics_storage',
    label: 'Usage analytics',
    description: 'Helps us understand which pages and features people actually use.',
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
    ad_storage: 'denied',
    ad_user_data: 'denied',
    ad_personalization: 'denied',
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
              We use optional analytics to understand traffic and product usage. Core site features always stay on.
            </p>
            <div className="consent-banner-actions">
              <button type="button" className="btn btn-outline" onClick={rejectAll}>
                Reject Analytics
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
            <h3>Privacy Controls</h3>
            <p>
              Choose whether optional analytics are on. Core site functionality is always on.
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
