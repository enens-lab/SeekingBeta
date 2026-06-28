// Social login client helpers: provider SDK loading + sign-in flows.
//
// Provider-agnostic and config-gated — a provider's button only appears when its
// client-side config (client ID / app ID) is present, so Facebook can go live
// before Google/Apple console setup is done. Each flow returns an OAuthPayload
// that POSTs to /api/auth/oauth/{provider}; the server is the only authority.
import type { OAuthPayload } from '../api/client';

const env = import.meta.env as Record<string, string | undefined>;

export const socialConfig = {
  enabled: (env.VITE_SOCIAL_LOGIN_ENABLED || '').toLowerCase() === 'true',
  googleClientId: (env.VITE_GOOGLE_CLIENT_ID || '').trim(),
  facebookAppId: (env.VITE_FACEBOOK_APP_ID || '').trim(),
  facebookGraphVersion: (env.VITE_FACEBOOK_GRAPH_VERSION || 'v19.0').trim(),
  appleServicesId: (env.VITE_APPLE_SERVICES_ID || '').trim(),
  appleRedirectUri: (env.VITE_APPLE_REDIRECT_URI || '').trim(),
};

export type SocialProviderId = 'google' | 'apple' | 'facebook';

// Order here = display order. Only providers with config are returned.
export function availableProviders(): SocialProviderId[] {
  if (!socialConfig.enabled) return [];
  const out: SocialProviderId[] = [];
  if (socialConfig.googleClientId) out.push('google');
  if (socialConfig.appleServicesId) out.push('apple');
  if (socialConfig.facebookAppId) out.push('facebook');
  return out;
}

// --- one-time async <script> loader (deduped by src) ---------------------------
const scriptPromises: Record<string, Promise<void> | undefined> = {};
function loadScript(src: string): Promise<void> {
  const existing = scriptPromises[src];
  if (existing) return existing;
  const promise = new Promise<void>((resolve, reject) => {
    if (document.querySelector(`script[src="${src}"]`)) {
      resolve();
      return;
    }
    const s = document.createElement('script');
    s.src = src;
    s.async = true;
    s.defer = true;
    s.onload = () => resolve();
    s.onerror = () => reject(new Error(`Failed to load ${src}`));
    document.head.appendChild(s);
  });
  scriptPromises[src] = promise;
  return promise;
}

/* eslint-disable @typescript-eslint/no-explicit-any */

// --- Facebook ------------------------------------------------------------------
let fbInitialized = false;
async function ensureFacebook(): Promise<any> {
  if (!socialConfig.facebookAppId) throw new Error('Facebook sign-in is not configured');
  await loadScript('https://connect.facebook.net/en_US/sdk.js');
  const w = window as any;
  if (!fbInitialized) {
    w.FB.init({
      appId: socialConfig.facebookAppId,
      cookie: false,
      xfbml: false,
      version: socialConfig.facebookGraphVersion,
    });
    fbInitialized = true;
  }
  return w.FB;
}

export async function signInWithFacebook(): Promise<OAuthPayload> {
  const FB = await ensureFacebook();
  return new Promise<OAuthPayload>((resolve, reject) => {
    FB.login(
      (response: any) => {
        const token = response?.authResponse?.accessToken;
        if (response?.status === 'connected' && token) {
          resolve({ credential: token });
        } else {
          reject(new Error('Facebook sign-in was cancelled'));
        }
      },
      { scope: 'public_profile,email' },
    );
  });
}

// --- Apple (web, via Services ID popup) ----------------------------------------
let appleInitialized = false;
async function ensureApple(): Promise<any> {
  if (!socialConfig.appleServicesId) throw new Error('Apple sign-in is not configured');
  await loadScript(
    'https://appleid.cdn-apple.com/appleauth/static/jsapi/appleid/1/en_US/appleid.auth.js',
  );
  const w = window as any;
  if (!appleInitialized) {
    w.AppleID.auth.init({
      clientId: socialConfig.appleServicesId,
      scope: 'name email',
      redirectURI: socialConfig.appleRedirectUri || `${window.location.origin}/login`,
      usePopup: true,
    });
    appleInitialized = true;
  }
  return w.AppleID;
}

export async function signInWithApple(): Promise<OAuthPayload> {
  const AppleID = await ensureApple();
  const res = await AppleID.auth.signIn();
  const idToken = res?.authorization?.id_token;
  if (!idToken) throw new Error('Apple sign-in failed');
  const payload: OAuthPayload = { credential: idToken };
  if (res?.authorization?.code) payload.authorization_code = res.authorization.code;
  // Apple sends the name ONLY on the very first authorization.
  if (res?.user?.name) {
    payload.name = { first: res.user.name.firstName, last: res.user.name.lastName };
  }
  return payload;
}

// --- Google (renders the official Identity Services button) --------------------
let googleInitialized = false;
let googleOnCredential: ((p: OAuthPayload) => void) | null = null;
let googleOnError: ((msg: string) => void) | null = null;

export async function renderGoogleButton(
  el: HTMLElement,
  onCredential: (payload: OAuthPayload) => void,
  onError: (msg: string) => void,
): Promise<void> {
  if (!socialConfig.googleClientId) throw new Error('Google sign-in is not configured');
  await loadScript('https://accounts.google.com/gsi/client');
  const w = window as any;
  if (!w.google?.accounts?.id) {
    onError('Google SDK unavailable');
    return;
  }
  // Keep the live handlers current across re-mounts (the GIS callback is set once).
  googleOnCredential = onCredential;
  googleOnError = onError;
  if (!googleInitialized) {
    w.google.accounts.id.initialize({
      client_id: socialConfig.googleClientId,
      callback: (resp: any) => {
        if (resp?.credential) googleOnCredential?.({ credential: resp.credential });
        else googleOnError?.('Google sign-in failed');
      },
    });
    googleInitialized = true;
  }
  el.innerHTML = '';
  const width = Math.min(400, Math.floor(el.getBoundingClientRect().width) || 320);
  w.google.accounts.id.renderButton(el, {
    theme: 'outline',
    size: 'large',
    type: 'standard',
    text: 'continue_with',
    shape: 'rectangular',
    logo_alignment: 'center',
    width,
  });
}
/* eslint-enable @typescript-eslint/no-explicit-any */
