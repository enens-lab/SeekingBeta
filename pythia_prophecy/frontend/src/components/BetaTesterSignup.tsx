import { FormEvent, useState } from 'react';
import { betaProgram, BetaTesterSignupData } from '../api/client';
import { trackEvent } from '../lib/analytics';

const DEFAULT_DISCORD_URL = 'https://discord.gg/ckTC8JhWU9';

const EXPERIENCE_OPTIONS = [
  { value: '', label: 'Select experience level' },
  { value: 'beginner', label: 'Beginner (0-1 years)' },
  { value: 'intermediate', label: 'Intermediate (2-5 years)' },
  { value: 'advanced', label: 'Advanced (5+ years)' },
  { value: 'professional', label: 'Professional / Institutional' },
];

function BetaTesterSignup() {
  const [formData, setFormData] = useState<BetaTesterSignupData>({
    full_name: '',
    email: '',
    role: '',
    organization: '',
    investing_experience: '',
    testing_focus: '',
    accept_contact: false,
    source: 'landing_page',
  });
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [successMessage, setSuccessMessage] = useState<string | null>(null);
  const [discordUrl, setDiscordUrl] = useState(DEFAULT_DISCORD_URL);

  const onInputChange = (
    event: React.ChangeEvent<HTMLInputElement | HTMLSelectElement | HTMLTextAreaElement>
  ) => {
    const { name, value } = event.target;
    setFormData((prev) => ({ ...prev, [name]: value }));
    setError(null);
  };

  const onCheckboxChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const { name, checked } = event.target;
    setFormData((prev) => ({ ...prev, [name]: checked }));
    setError(null);
  };

  const resetForm = () => {
    setFormData({
      full_name: '',
      email: '',
      role: '',
      organization: '',
      investing_experience: '',
      testing_focus: '',
      accept_contact: false,
      source: 'landing_page',
    });
  };

  const onSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setError(null);
    setSuccessMessage(null);

    if (!formData.accept_contact) {
      setError('Please confirm that we can contact you about beta access.');
      return;
    }

    setSubmitting(true);
    trackEvent('beta_signup_attempt', { source: 'landing_page' });
    try {
      const response = await betaProgram.signup(formData);
      setSuccessMessage(response.message);
      setDiscordUrl(response.discord_url || DEFAULT_DISCORD_URL);
      trackEvent('beta_signup_success', { source: 'landing_page' });
      resetForm();
    } catch (e) {
      const message = e instanceof Error ? e.message : 'Could not submit your application.';
      setError(message);
      trackEvent('beta_signup_error', { source: 'landing_page' });
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <section className="beta-program-section" id="beta-testers">
      <div className="beta-program-container">
        <div className="beta-program-copy">
          <h2 className="section-title">Join early access</h2>
          <p className="section-subtitle">
            Help us improve SeekingBeta.AI before a wider launch. We want people who will really use it, notice what feels off, and tell us clearly.
          </p>
          <ul className="beta-program-list">
            <li>Try new features first</li>
            <li>Talk directly with the product team</li>
            <li>Get help getting started</li>
          </ul>
          <a
            className="beta-program-discord-link"
            href={DEFAULT_DISCORD_URL}
            target="_blank"
            rel="noopener noreferrer"
            onClick={() => trackEvent('discord_click', { source: 'beta_program_copy' })}
          >
            Join the community on Discord
          </a>
        </div>

        <form className="beta-program-form" onSubmit={onSubmit}>
          <div className="beta-form-grid">
            <label className="form-group">
              <span>Full Name *</span>
              <input
                type="text"
                name="full_name"
                value={formData.full_name}
                onChange={onInputChange}
                required
                maxLength={120}
                placeholder="Your full name"
              />
            </label>

            <label className="form-group">
              <span>Email *</span>
              <input
                type="email"
                name="email"
                value={formData.email}
                onChange={onInputChange}
                required
                placeholder="you@company.com"
              />
            </label>

            <label className="form-group">
              <span>Role</span>
              <input
                type="text"
                name="role"
                value={formData.role}
                onChange={onInputChange}
                maxLength={120}
                placeholder="Researcher, trader, sports analyst, PM, etc."
              />
            </label>

            <label className="form-group">
              <span>Organization</span>
              <input
                type="text"
                name="organization"
                value={formData.organization}
                onChange={onInputChange}
                maxLength={160}
                placeholder="Company or independent"
              />
            </label>

            <label className="form-group">
              <span>Market / Research Experience</span>
              <select
                name="investing_experience"
                value={formData.investing_experience}
                onChange={onInputChange}
              >
                {EXPERIENCE_OPTIONS.map((option) => (
                  <option key={option.value} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </select>
            </label>
          </div>

          <label className="form-group form-group-full">
            <span>What do you want to pressure-test most? *</span>
            <textarea
              name="testing_focus"
              value={formData.testing_focus}
              onChange={onInputChange}
              required
              minLength={12}
              maxLength={1200}
              placeholder="Tell us what you want to test most (for example: board quality, watchlist flow, analysis results, or billing)."
              rows={4}
            />
          </label>

          <label className="consent-check beta-consent">
            <input
              type="checkbox"
              name="accept_contact"
              checked={formData.accept_contact}
              onChange={onCheckboxChange}
              required
            />
            <span>I agree to be contacted for beta program updates and testing coordination.</span>
          </label>

          <div className="beta-form-actions">
            <button type="submit" className="btn btn-primary btn-lg" disabled={submitting}>
              {submitting ? 'Submitting...' : 'Request Early Access'}
            </button>
            <a
              href={DEFAULT_DISCORD_URL}
              target="_blank"
              rel="noopener noreferrer"
              className="btn btn-outline btn-lg"
              onClick={() => trackEvent('discord_click', { source: 'beta_program_form' })}
            >
              Discord
            </a>
          </div>

          {error && <p className="form-error">{error}</p>}

          {successMessage && (
            <div className="beta-form-success">
              <p>{successMessage}</p>
              <a
                href={discordUrl}
                target="_blank"
                rel="noopener noreferrer"
                onClick={() => trackEvent('discord_click', { source: 'beta_program_success' })}
              >
                Join Discord for updates
              </a>
            </div>
          )}
        </form>
      </div>
    </section>
  );
}

export default BetaTesterSignup;
