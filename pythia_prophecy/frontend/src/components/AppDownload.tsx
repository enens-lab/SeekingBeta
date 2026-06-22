import { trackEvent } from '../lib/analytics';

// Live App Store listing (app id 6761283945, EnEns LLC — verified via iTunes lookup 2026-06-22).
const APP_STORE_URL = 'https://apps.apple.com/us/app/seekingbeta-ai/id6761283945';

function AppleLogo() {
  return (
    <svg className="store-badge-logo" width="22" height="26" viewBox="0 0 384 512" fill="currentColor" aria-hidden="true">
      <path d="M318.7 268.7c-.2-36.7 16.4-64.4 50-84.8-18.8-26.9-47.2-41.7-84.7-44.6-35.5-2.8-74.3 20.7-88.5 20.7-15 0-49.4-19.7-76.4-19.7C32.3 141.2-8.4 184.8-8.4 273.7c0 26.3 4.8 53.5 14.4 81.5 12.8 36.8 59 127 107.2 125.5 25.2-.6 43-17.9 75.8-17.9 31.8 0 48.3 17.9 76.4 17.9 48.6-.7 90.4-82.5 102.6-119.4-65.2-30.7-61.7-90-61.7-92.1zm-56.6-164.2c27.3-32.4 24.8-61.9 24-72.5-24.1 1.4-52 16.4-67.9 34.9-17.5 19.8-27.8 44.3-25.6 71.9 26.1 2 49.9-11.4 69.5-34.3z" />
    </svg>
  );
}

function GooglePlayLogo() {
  return (
    <svg className="store-badge-logo" width="24" height="24" viewBox="0 0 512 512" fill="currentColor" aria-hidden="true">
      <path d="M325.3 234.3L104.6 13l280.8 161.2-60.1 60.1zM47 0C34 6.8 25.3 19.2 25.3 35.3v441.3c0 16.1 8.7 28.5 21.7 35.3l256.6-256L47 0zm425.2 225.6l-58.9-34.1-65.7 64.5 65.7 64.5 60.1-34.1c18-14.3 18-46.5-1.2-60.8zM104.6 499l280.8-161.2-60.1-60.1L104.6 499z" />
    </svg>
  );
}

function AppDownload() {
  return (
    <section className="app-download" id="get-the-app">
      <div className="app-download-inner">
        <div className="app-download-copy">
          <span className="app-download-eyebrow">
            <span className="app-download-eyebrow-dot" />
            NOW ON THE APP STORE
          </span>
          <h2 className="app-download-title">
            Your picks,
            <br />
            now in your pocket.
          </h2>
          <p className="app-download-sub">
            Check live stock and sports picks, build your watchlist, and review the full track record,
            wherever you are, right from your iPhone.
          </p>

          <div className="app-download-badges">
            <a
              className="store-badge store-badge--live"
              href={APP_STORE_URL}
              target="_blank"
              rel="noopener noreferrer"
              onClick={() => trackEvent('app_download_click', { store: 'app_store' })}
              aria-label="Download SeekingBeta.AI on the App Store"
            >
              <AppleLogo />
              <span className="store-badge-text">
                <span className="store-badge-top">Download on the</span>
                <span className="store-badge-name">App Store</span>
              </span>
            </a>

            <div className="store-badge store-badge--soon" role="img" aria-label="SeekingBeta.AI is coming soon to Google Play">
              <GooglePlayLogo />
              <span className="store-badge-text">
                <span className="store-badge-top store-badge-top--accent">COMING SOON</span>
                <span className="store-badge-name">Google Play</span>
              </span>
            </div>
          </div>

          <p className="app-download-trust">
            Free to download · Research only. We don’t place trades or bets.
          </p>
        </div>

        <div className="app-download-device" aria-hidden="true">
          <div className="device-phone">
            <img
              className="device-shot"
              src="/app-preview-iphone.jpg"
              alt="SeekingBeta.AI iOS app showing the dashboard and track record"
              width={640}
              height={1391}
              loading="lazy"
              decoding="async"
            />
          </div>
        </div>
      </div>
    </section>
  );
}

export default AppDownload;
