# Social Login Implementation Plan — SeekingBeta.AI

Prepared: June 28, 2026
Status: Plan (no code yet). Build-ready.
Scope: Add social login — **Apple + Google + Facebook for v1** — across Web, iOS, and Android, behind a **provider-agnostic backend contract** so Microsoft / X / etc. slot in later without endpoint or schema churn.

This doc mirrors the structure and rigor of the team's billing handoff/plan work (`APPLE_BILLING_HANDOFF.md`) and, critically, the **existing billing token-verify pattern** in `pythia_prophecy/api/service.py` (`POST /api/billing/{apple,google}/verify`): the server is the only authority, a forged token raises and **never grants**, and provider differences are isolated behind one dispatch.

> **Legend.** 🔎 = value/string/endpoint to re-confirm against **live** provider docs at build time (the *mechanics* are stable; exact URLs, claim names, scope strings, and API versions drift). 👤 = **USER-ONLY** action in a provider console that Claude/CI cannot do. 🐢 = slow or review-gated (can block a release; start early).

---

## 0. Decisions & the provider-agnostic contract

### 0.1 Decisions already locked
- **Providers (v1):** Apple, Google, Facebook. Backend is **provider-agnostic** — adding a provider is a new verifier + an `aud`/app-id allow-list entry, not a new endpoint or table.
- **Account linking:** when a social login's **VERIFIED** email matches an existing account (email/password **or** another provider), **LINK** to that existing account. No duplicate accounts. (Security rules in §D — verified-email-only, `subject` is the durable join key.)
- **Session model is unchanged:** after the backend verifies a provider token it issues **the same access + refresh JWT** the email/password flow already issues via `create_access_token(user_id, email, token_type)` in `api/auth.py`. Clients store and refresh them exactly as today. The provider token is used **once**, at verify time, and is never persisted as a session credential.

### 0.2 The contract — what the client sends

One endpoint, provider-dispatched (mirrors `_billing_provider_from_tiers` at `service.py:2407`):

```
POST /api/auth/oauth/{provider}        provider ∈ {google, apple, facebook}  (allow-list)
Content-Type: application/json

{
  "credential": "<id_token | identity_token | access_token>",  // the provider token
  "nonce": "<client-generated nonce>",      // optional; Apple/Google replay binding
  "name": { "first": "...", "last": "..." } // optional; Apple FIRST sign-in ONLY (§B.3)
}
```

- **Google** → `credential` = OpenID **ID token** (RS256 JWT).
- **Apple** → `credential` = **identityToken** (RS256 JWT). `name` is present **only on the first authorization** (Apple never resends it — see §B.3).
- **Facebook** → `credential` = OAuth **access token** (opaque string, verified server-to-server).

The client **never** asserts identity itself (no client-supplied email is trusted) and never sends a password.

### 0.3 The contract — what the backend returns

Exactly the existing `TokenResponse` shape (so every client decodes it with zero new model work):

```json
{
  "access_token": "<HS256 JWT, 24h>",
  "refresh_token": "<HS256 JWT, 7d>",
  "token_type": "bearer",
  "user": { "id", "email", "first_name", "last_name", "tier", "email_verified" }
}
```

A social login whose provider asserts a verified email returns `email_verified: true` and the user is immediately past the `email_verified` route guard — **no verification email is sent** for that path.

### 0.4 Internal verifier interface (one impl per provider)

```
verify(credential, nonce?) -> {
  provider: str,
  subject: str,          # provider stable id: Google/Apple 'sub', Facebook 'id' — THE JOIN KEY
  email: str | None,     # Facebook may omit
  email_verified: bool,  # provider's assertion
  name: {first, last} | None
}
```

Each `verify()` **raises** on any failure (bad signature / wrong `aud` / expired / revoked / app mismatch) → endpoint returns `401`. It **never** falls back to trusting the client. Same posture as the Google billing verify ("authoritative fetch, never grants on a forged token", `service.py` ~line 4322).

---

## A. USER-ONLY provider-console setup 👤

These are ordered. **Start the 🐢 items first** — they gate the release.

### A.1 Google (Google Cloud Console) 👤
One Google Cloud project + one OAuth consent screen, then **separate OAuth 2.0 client IDs per platform**:

1. 👤 Create / select a Google Cloud project; configure the **OAuth consent screen** (app name, support email, logo, privacy-policy + terms URLs). 🐢 If you request anything beyond `email profile openid`, this needs Google **verification review** — but we request only least-privilege scopes (§F), so review is minimal.
2. 👤 Create **OAuth Client ID — iOS**: bind to the iOS **Bundle ID**.
3. 👤 Create **OAuth Client ID — Android**: bind to the Android **package name + SHA-1 signing-cert fingerprint**. 🔎🐢 Register **all** certs you ship — debug cert, release cert, **and Play App Signing's cert** (Play re-signs; its SHA-1 is in Play Console → App integrity). Missing the Play cert = Google Sign-In fails only in production.
4. 👤 Create **OAuth Client ID — Web**: set Authorized JavaScript origins (`https://seekingbeta.ai`, dev origins) and redirect URIs. The Web client ID typically doubles as the `serverClientId`/requested audience.
5. 👤 Collect all client IDs → backend `aud` allow-list (§B.5). 🔎 Confirm which client ID each SDK stamps into `aud` (platform vs web/server client ID) and allow-list accordingly.
6. Scopes: **`openid email profile`** only.

### A.2 Apple — "Sign in with Apple" (Apple Developer portal) 👤🐢
Apple has the most setup and is **mandatory for the iOS build** (§F):

1. 👤 In **Certificates, Identifiers & Profiles**, enable the **Sign in with Apple** capability on the app's **App ID** (Bundle ID). For iOS native this is the only identifier needed; `aud` = Bundle ID.
2. 👤 Add the **Sign in with Apple** entitlement to the Xcode project (capability + provisioning profile regeneration). 🐢 Provisioning-profile changes propagate slowly.
3. 👤 For **Web + Android** (no native Apple SDK): create a **Services ID** (distinct identifier from the App ID), enable Sign in with Apple on it, and register **Return URLs / redirect URIs** (`https://seekingbeta.ai/...`). Here `aud` = the **Services ID**.
4. 👤🐢 Create a **Sign in with Apple key (`.p8`)** under the Team; record its **Key ID** + **Team ID**. The `.p8` is downloadable **once** — store it in the secrets path (§B.6). It is needed to (a) build the `client_secret` JWT for the web token exchange and (b) call the **revoke endpoint** required for account deletion (§F).
5. 👤 Backend `aud` allow-list must include **both** the Bundle ID (iOS) and the Services ID (web/Android).
6. 🔎 Re-confirm at build: JWKS at `https://appleid.apple.com/auth/keys`; `email_verified` may arrive as the **string** `"true"` (coerce); nonce handling; the `auth/revoke` contract.

### A.3 Facebook (Meta for Developers) 👤🐢
1. 👤 Create a **Facebook App**; add the **Facebook Login** product.
2. 👤 Configure **Valid OAuth Redirect URIs** (web) and register the iOS **Bundle ID** / Android **package + key hash** for the native SDKs.
3. 👤 Request permissions **`email` + `public_profile`** only.
4. 👤🐢 **Data Deletion Callback (MANDATORY, review gate):** provide either a **Data Deletion Request Callback URL** (a signed server endpoint Facebook calls — see §B.7) **or** a Data Deletion Instructions URL. Without it the app can be restricted. 🔎 Confirm current callback spec.
5. 👤🐢 App Review / Business Verification: moving the app out of Dev Mode so non-test users can log in requires Meta **App Review** for `email`/`public_profile` and possibly business verification. **Start this early** — it is the slowest Facebook gate.
6. 👤 Record **App ID** + **App Secret** → backend secrets (§B.6).

### A.4 Slow/gated summary (start these first) 🐢
- Apple `.p8` key + Services ID + provisioning-profile propagation.
- Apple revoke wiring (needed before you can ship account deletion that's App Store-compliant).
- Facebook App Review + Data Deletion Callback.
- Google Play App Signing SHA-1 registration (prod-only failure if missed).

---

## B. Backend changes (file-cited insertion points)

All paths under `/Users/huyngo/Downloads/pythia/pythia_prophecy/`.

### B.1 New module — `api/social_auth.py` (CREATE)
Isolates per-provider verification, exactly as billing isolates per-provider verify logic. Exposes the §0.4 interface:

```
verify_social_credential(provider, credential, nonce=None) -> SocialIdentity   # dispatch
_verify_google(credential, nonce)   -> SocialIdentity   # RS256/JWKS
_verify_apple(credential, nonce)    -> SocialIdentity   # RS256/JWKS
_verify_facebook(credential)         -> SocialIdentity   # Graph debug_token + /me
```

**Why a new module and not `auth.py`:** `auth.py` issues/validates **HS256** JWTs with a shared secret (`_encode_jwt`/`_decode_jwt`). Provider tokens (Google, Apple) are **RS256 signed by the provider's rotating JWKS** — the shared-secret path **cannot** validate them. Do **not** extend `_decode_jwt`. Add a JWKS/RS256 dependency: `google-auth` (Google) and `PyJWT[crypto]` + `PyJWKClient` (Apple). Facebook needs no JWT lib — it's HTTP to the Graph API.

Per-provider verification (build-time 🔎 on every string):
- **Google:** verify signature against JWKS (`https://www.googleapis.com/oauth2/v3/certs` 🔎, cache per `Cache-Control`); `iss ∈ {https://accounts.google.com, accounts.google.com}`; `aud ∈ ALLOWED_GOOGLE_CLIENT_IDS`; `exp` valid. Extract `sub`, `email`, `email_verified`. Prefer `google.oauth2.id_token.verify_oauth2_token`.
- **Apple:** verify against `https://appleid.apple.com/auth/keys` 🔎 (cache; refetch on unknown `kid`); `iss == https://appleid.apple.com`; `aud ∈ {BUNDLE_ID, SERVICES_ID}`; `exp` valid; optional `nonce` match. Extract `sub`, `email`, `email_verified` (coerce string→bool 🔎), `is_private_email`.
- **Facebook:** `GET /debug_token?input_token=<user>&access_token=<APP_ID>|<APP_SECRET>` 🔎 — require `data.is_valid == true` and `data.app_id == FACEBOOK_APP_ID`, capture `data.user_id`. Then `GET /me?fields=id,name,email` 🔎 (pin Graph version, e.g. `/v19.0/`). Recommend `appsecret_proof` (HMAC-SHA256 of user token keyed by app secret). **Email may be absent** — handle `None` (see §D).

### B.2 New endpoint — `POST /api/auth/oauth/{provider}` in `api/service.py`
Insert after the `/api/auth/resend-verification` endpoint (≈ line 3402), alongside the other auth routes. Mirror the billing verify endpoints (`POST /api/billing/apple/verify` ~4283, `POST /api/billing/google/verify` ~4322) for shape, rate-limiting, and the never-grant-on-forgery posture.

```python
@app.post("/api/auth/oauth/{provider}", response_model=TokenResponse, tags=["Authentication"])
async def oauth_verify(provider: str, request: Request, data: OAuthVerifyRequest):
    # 0. if not social_login_enabled(): 404   (feature flag, §E)
    # 1. provider in {"google","apple","facebook"} else 400
    # 2. rate-limit (per-IP; mirror AUTH_LOGIN limits) — service.py rate-limit helper
    # 3. identity = verify_social_credential(provider, data.credential, data.nonce)  # raises -> 401
    # 4. user = resolve_or_create_user(identity, data.name)   # §B.4 link-by-verified-email
    # 5. access  = create_access_token(user.id, user.email, "access")
    #    refresh = create_access_token(user.id, user.email, "refresh")
    # 6. initialize billing state if new user (mirror signup's upsert_customer_state, service.py ~3215)
    # 7. return TokenResponse(access_token, refresh_token, "bearer", UserResponse(user))
```

Add a new flag helper `social_login_enabled()` next to the existing billing flag helpers, gated on `SOCIAL_LOGIN_ENABLED` (§E). Wire `ensure_auth_identities_table()` into startup validation (≈ line 645, after the billing-store check). Add the import near the other store imports (≈ line 144).

### B.3 ⚠️ Apple-only: persist the name on FIRST sign-in
Apple returns the user's **name** (and the only chance at a shared real email) **only in the very first authorization response**, *outside* the JWT (iOS `user` field / web form-post `user` JSON). Every later sign-in yields **only** the `identityToken` (so `sub` + email). The client forwards it as the optional `name` in the contract (§0.2); the backend **MUST persist it the first time it appears** (on create, or backfill onto an existing row if name was empty). If not stored then, it's gone forever.

### B.4 New store + table — `api/auth_identities_store.py` (CREATE)
Mirrors `billing_store.py` (idempotent upserts, per-provider lookup keys, source tracking). One user can link multiple providers, so use a **join table**, not per-provider columns on `users`:

```sql
CREATE TABLE auth_identities (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id       TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    provider      TEXT NOT NULL,            -- 'google'|'apple'|'facebook'|'microsoft'|...
    subject       TEXT NOT NULL,            -- provider 'sub'/'id' — THE JOIN KEY
    email_at_link TEXT,                     -- email seen when linked (audit; emails change)
    raw_profile   JSONB,                    -- name/picture/private-relay flags
    linked_at     TIMESTAMPTZ DEFAULT NOW(),
    updated_at    TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (provider, subject)              -- one provider identity -> at most one user
);
CREATE INDEX idx_auth_identities_user_id ON auth_identities (user_id);
CREATE INDEX idx_auth_identities_email   ON auth_identities (email_at_link);
```

Functions:
- `ensure_auth_identities_table()`
- `upsert_auth_identity(user_id, provider, subject, email_at_link, raw_profile)` — idempotent
- `get_identity_by_provider_subject(provider, subject) -> dict | None`
- `get_identities_by_user_id(user_id) -> list[dict]`
- `delete_identities_for_user(user_id)` — for account deletion (§B.7)

**`users.hashed_password` becomes nullable.** Social-only users have no password (`api/database.py` users table, ≈ line 44). Keep the existing `email_verified` column — a verified-email social link can set it to `1`.

### B.5 `resolve_or_create_user(identity, name)` — find-or-create + LINK
Implement inside the endpoint module (or a small helper). **Run inside a DB transaction** (concurrent first-time logins must not fork into two accounts). Resolution order — full security rationale in §D:

```
1. (provider, subject) already in auth_identities?  -> that user. Log in. (email irrelevant)
2. else, IF identity.email_verified AND identity.email is not None:
      existing = find_user_by_verified_email(identity.email)   # users.email, lowercased
      if existing: upsert_auth_identity(existing.id, ...); set users.email_verified=1; log in
3. else (no verified email, or no match): create user (hashed_password NULL,
      email_verified = identity.email_verified, name from Apple-first-signin if present)
      + upsert_auth_identity(new.id, ...)
```

`find_user_by_verified_email(email)` → add to `api/database.py` after `delete_user_account` (≈ line 281): `SELECT * FROM users WHERE email = ? (lowercased)`. **Never** link on an unverified provider email (step 2 guard).

### B.6 New models — `api/models.py`
Insert after `BillingStatusResponse` (≈ line 414):

```python
class OAuthVerifyRequest(BaseModel):
    credential: str
    nonce: Optional[str] = None
    name: Optional[OAuthName] = None     # Apple first-sign-in only

class OAuthName(BaseModel):
    first: Optional[str] = None
    last: Optional[str] = None
```

No new response model — reuse `TokenResponse`. `UserResponse` already carries `email_verified`.

### B.7 Env / secrets — EC2 deploy env + compose (mirror Google-billing wiring)
The billing service-account pattern is the template: flags/IDs come from the box `.env` via `docker-compose.yml` (`prophecy-api.environment`, lines 135–216), and secret files are **mounted read-only** via `./pythia_prophecy/secrets:/app/secrets:ro` (line 223). Add, immediately after the Google Play block (after line 198):

```yaml
      # Social login (Apple + Google + Facebook). Provider-agnostic verify.
      SOCIAL_LOGIN_ENABLED: ${SOCIAL_LOGIN_ENABLED:-false}
      # Google: allow-list ALL platform client IDs as valid token audiences.
      GOOGLE_OAUTH_CLIENT_IDS: ${GOOGLE_OAUTH_CLIENT_IDS:-}     # comma-separated: ios,android,web
      # Apple: allow-list Bundle ID (iOS) + Services ID (web/Android).
      APPLE_OAUTH_BUNDLE_ID: ${APPLE_OAUTH_BUNDLE_ID:-}
      APPLE_OAUTH_SERVICES_ID: ${APPLE_OAUTH_SERVICES_ID:-}
      APPLE_OAUTH_TEAM_ID: ${APPLE_OAUTH_TEAM_ID:-}
      APPLE_OAUTH_KEY_ID: ${APPLE_OAUTH_KEY_ID:-}
      # .p8 mounted read-only via the existing secrets volume (like the Play SA JSON).
      APPLE_OAUTH_PRIVATE_KEY_FILE: ${APPLE_OAUTH_PRIVATE_KEY_FILE:-/app/secrets/apple-signin-key.p8}
      # Facebook: server-to-server debug_token + Graph /me.
      FACEBOOK_APP_ID: ${FACEBOOK_APP_ID:-}
      FACEBOOK_APP_SECRET: ${FACEBOOK_APP_SECRET:-}
      FACEBOOK_GRAPH_VERSION: ${FACEBOOK_GRAPH_VERSION:-v19.0}   # 🔎 pin + bump on deprecation
```

- The Apple `.p8` rides the **existing** `./pythia_prophecy/secrets:/app/secrets:ro` mount (gitignored, host-only) — no new volume.
- `deploy_ec2.sh` already rebuilds + recreates `prophecy-api`; the only deploy delta is populating the box `.env` and dropping `apple-signin-key.p8` into the host `secrets/` dir. Add an env-sanity line to the deploy script (analogous to the existing divination env check) to assert `SOCIAL_LOGIN_ENABLED` and that the `.p8` is present when Apple is on.
- Add `google-auth` and `PyJWT[crypto]` to `pythia_prophecy/requirements.txt`.

### B.8 Account deletion clears identities
`DELETE /api/auth/delete-account` (existing) must additionally:
1. `delete_identities_for_user(user_id)` (FK `ON DELETE CASCADE` covers the rows, but call explicitly so provider-side revocation runs first).
2. **Apple:** call `https://appleid.apple.com/auth/revoke` 🔎 (Services ID + `.p8` client-secret JWT) for each Apple identity. **App Store review requirement.**
3. **Facebook:** honor the Data Deletion Callback (§B.7 endpoint) and optionally `DELETE /{user-id}/permissions` to de-authorize.
4. **Google:** revoke any stored grant/refresh token if held; delete the row.
5. **Facebook Data Deletion Callback endpoint** (new, public, signed-request verified): `POST /api/auth/facebook/data-deletion` → verify Meta's signed_request against `FACEBOOK_APP_SECRET`, delete the user's data, return `{ url, confirmation_code }`. Idempotent; must not leave a dangling `(provider, subject)` row.

---

## C. Client changes per surface

### C.1 Web — React (`pythia_prophecy/frontend/`)
1. **`src/api/client.ts`** — add to the `auth` module: `auth.oauth(provider, credential, nonce?, name?) → request('POST', '/api/auth/oauth/' + provider, ...)`. Response is the standard `TokenResponse`; store `access_token` under the existing `pythia_token` localStorage key and inject `Bearer` as today.
2. **`src/context/AuthContext.tsx`** — add `socialLogin(provider)` and `oauthError`/`socialLoading` state; on success call the same post-login path as `login()`/`verifyEmail()`.
3. **`src/pages/Login.tsx`** (≈ lines 114–167) and **`src/pages/Signup.tsx`** (≈ lines 201–318) — add a social-button row **above** the password form, with an "Or continue with" divider. Buttons gated on a build flag so they're hidden until backend is live (§E).
4. **`src/pages/OAuthCallback.tsx`** (NEW, web redirect flows) — read `code`/`state` (or token) from the URL, validate `state` against the value stashed pre-redirect (CSRF), call `auth.oauth(...)`, store JWT, redirect to dashboard.
5. Provider JS: Google Identity Services (returns an ID token to the browser), Apple JS (Services ID redirect → identityToken), Facebook JS SDK (access token). Each yields the `credential` the contract expects.

### C.2 iOS — Swift (`/Users/huyngo/Downloads/SeekingBeta.AI/`)
Apple is **mandatory** here (§F); ship Apple + Google for v1.
1. **`SeekingBeta.AI/Features/AuthView.swift`** — in `actionCard`, add **"Sign in with Apple"** (`SignInWithAppleButton` / `ASAuthorizationAppleIDProvider`) and **"Continue with Google"** buttons.
2. **`SeekingBeta.AI/Features/AuthViewModel.swift`** — add `loginWithApple()` and `loginWithGoogle()` after `login()`. Apple: run `ASAuthorizationController`, capture `identityToken` **and**, on first sign-in, the `fullName` (forward as `name`); generate a `nonce` and pass it. Google: Google Sign-In SDK → ID token. Both call the new API method, then the existing `applyAuthIfPresent` → `setTokens` → `setCurrentUser` path; refresh billing.
3. **`SeekingBeta.AI/Networking/APIClient.swift`** — add `loginWithSocialProvider(provider, credential, nonce?, name?) async throws -> AuthUser?` after `login()`; reuse `applyAuthIfPresent` (lines 294–303). `requiresAuth: false`.
4. **`SeekingBeta.AI/Networking/Endpoint.swift`** — add `Endpoint.oauth(provider:)` → `POST /api/auth/oauth/{provider}`, `requiresAuth: false`.
5. **`SeekingBeta.AI/Auth/AuthModels.swift`** — add `OAuthVerifyRequest { credential, nonce?, name? }`. `AuthResponse`/`AuthUser` already decode the standard response; no session/keychain changes (provider-agnostic).
6. Xcode: add the **Sign in with Apple** capability (§A.2) and the Google Sign-In SDK + URL scheme.

### C.3 Android — Kotlin (`/Users/huyngo/Downloads/SeekingBeta.AI Android/`)
Ship Google + Facebook for v1 (Apple-on-Android is a later add via the Services ID web flow).
1. **`data/model/Auth.kt`** — add `OAuthVerifyRequest(credential, nonce?, name?)`. Reuse `TokenResponse`.
2. **`core/net/SeekingBetaApi.kt`** — add `suspend fun oauthVerify(provider: String, credential: String, nonce: String? = null): TokenResponse` after `resendVerification` (≈ line 48), via `client.requestElement("POST", "/api/auth/oauth/$provider", ..., requiresAuth=false)`.
3. **`features/auth/AuthViewModel.kt`** — add `initiateGoogleLogin()` (Google **Credential Manager** → ID token) and `initiateFacebookLogin()` (Facebook Login SDK → access token); both call `oauthVerify`, then the **existing** `session.applyAuth(token)` → `refreshCurrentUser` / `refreshBilling` path. No `SessionStore`/`TokenStore` changes — they're already provider-agnostic.
4. **`features/auth/AuthScreen.kt`** — in `CredentialsCard` (after the primary button), add an "Or continue with" label + a row of Google / Facebook `SBSecondaryButton`s.
5. **`core/social/` (NEW)** — `GoogleCredentialManager.kt`, `FacebookLoginManager.kt`; init Facebook SDK in `MainActivity`. Gradle: `androidx.credentials` + `play-services-auth`, `com.facebook.android:facebook-android-sdk`.

> Clients never see provider differences in the response: every surface stores the same access+refresh JWT and continues through its existing post-login path.

---

## D. Account linking + SECURITY

**Goal:** social login with a **verified** email matching an existing account → LINK (no duplicate). This is also the #1 account-takeover vector, so the rules are strict and non-negotiable.

**Resolution order (every social sign-in), transactional:**
1. **Match by `(provider, subject)` FIRST.** If `auth_identities` already has this `(provider, sub)`, that's the user. Email is irrelevant (emails change; `sub` doesn't). `UNIQUE(provider, subject)` guarantees one identity → one user.
2. **No identity match → link by email ONLY if the provider asserts the email is verified.**
   - Google `email_verified == true`; Apple verified (coerce string→bool); **Facebook: do NOT auto-link** unless independently verified (FB gives no clean per-user verified boolean, and may omit email entirely).
   - If a `users` row exists with that normalized (lowercased) email **and** the provider says verified → link the identity to that user; optionally set `email_verified = 1`.
   - **If the provider says NOT verified → never link.** An attacker who can mint an unverified-email token for `victim@example.com` must **not** be handed the victim's account. Treat as new signup or block pending verification.
3. **No match at all → create a new password-less user** (`hashed_password` NULL) + the identity row.

**Hard rules:**
- **`subject`, not email, is the durable join key.** Email is a one-time bridge under the verified-email guard.
- **Collisions / races:** enforce `UNIQUE(provider, subject)` and `users.email UNIQUE` at the DB; do the link in a transaction so concurrent first-time logins can't fork.
- **Normalize emails** before compare (lowercase). Do **not** "smartly" canonicalize Gmail dot/plus aliases — match the exact normalized string.
- **No silent identity moves / re-link takeover:** an inbound social token must never *reassign* an existing `(provider, subject)` to a different user — `UNIQUE` blocks it; surface an error, never reassign. Linking a **new** provider to an account requires the user to already be authenticated (link from inside the account) or to pass the verified-email rule.
- **Apple private-relay & Facebook-no-email identities** often can't link by email at all — correct and safe; they become standalone identities unless linked while authenticated.
- **Forged-token posture:** `verify()` raises → `401`. Never trust a client-supplied email/identity. Same as billing's "never grants on forged token."

---

## E. Phased sequence (backend-first, behind a flag)

Everything ships dark behind `SOCIAL_LOGIN_ENABLED` (backend, default `false`) and per-client build flags. The endpoint 404s while the flag is off, so clients can be merged before the providers are reviewed.

- **Phase 0 — Console setup (parallel, START NOW 🐢):** §A. Apple `.p8`/Services ID, Facebook App Review + Data Deletion, Google client IDs incl. Play App Signing SHA-1.
- **Phase 1 — Backend (framework-agnostic, fully testable):** `social_auth.py`, `auth_identities_store.py` + table, `OAuthVerifyRequest`, `find_user_by_verified_email`, `POST /api/auth/oauth/{provider}`, deletion clears identities + Apple revoke + FB data-deletion endpoint, env/compose/secrets, `requirements.txt`. Unit-test each verifier with mocked JWKS / Graph responses; test the link/create/forged-token matrix. Flag stays `false` in prod; enable in staging.
- **Phase 2 — Web:** buttons + `OAuthCallback` + client method. Web is the cheapest end-to-end loop (no app-store review) → validate the contract here first.
- **Phase 3 — iOS + Android (parallel):** native SDK integration; clients merged behind build flags. iOS **must** include Sign in with Apple before submission (§F).
- **Phase 4 — Enable + monitor:** flip `SOCIAL_LOGIN_ENABLED=true` (staging → prod), reveal client buttons, watch link/create rates and `401` verification failures.

Testability is preserved: Phases 1–2 need no device; Phase 3's logic (ViewModel/API) is separable from the platform SDK glue.

---

## F. Compliance

- **App Store Guideline 4.8 — Sign in with Apple is REQUIRED on iOS** whenever you offer Google/Facebook (third-party social login) that collects name/email. 🔎 4.8 has carve-outs, but practically: **the iOS build MUST ship Sign in with Apple** alongside Google/Facebook or risk rejection. Confirm current 4.8 wording at submission.
- **Least-privilege scopes:** Google `openid email profile`; Apple name + email; Facebook `email` + `public_profile`. No friends/posting/calendar — over-scoping triggers extra review and erodes trust.
- **Facebook Data Deletion (MANDATORY):** callback or instructions URL, wired to the deletion path (§B.7, §B.8).
- **Account deletion must clear identities + revoke at provider:** delete all `auth_identities` rows; **Apple revoke** (`auth/revoke`, App Store requirement 🔎); Facebook honor data-deletion callback + de-authorize; Google revoke stored grant if held. Idempotent; no dangling `(provider, subject)` rows (else the user can never re-link).

---

## G. Key files / insertion-point index

| File | Action | Insertion point |
|---|---|---|
| `pythia_prophecy/api/social_auth.py` | **CREATE** | per-provider RS256/JWKS + Graph verifiers; `verify_social_credential` dispatch |
| `pythia_prophecy/api/auth_identities_store.py` | **CREATE** | `auth_identities` table + upsert/lookup/delete (mirror `billing_store.py`) |
| `pythia_prophecy/api/service.py` | MODIFY | import store (~144); startup `ensure_auth_identities_table()` (~645); `POST /api/auth/oauth/{provider}` after resend-verification (~3402); FB data-deletion endpoint; deletion clears identities + Apple revoke; `social_login_enabled()` flag helper |
| `pythia_prophecy/api/models.py` | MODIFY | `OAuthVerifyRequest` + `OAuthName` after `BillingStatusResponse` (~414) |
| `pythia_prophecy/api/database.py` | MODIFY | `find_user_by_verified_email()` after `delete_user_account` (~281); make `users.hashed_password` nullable (~44) |
| `pythia_prophecy/api/auth.py` | NO CHANGE to JWT issuance | reuse `create_access_token(user_id, email, token_type)`; do **not** route provider tokens through HS256 `_decode_jwt` |
| `pythia_prophecy/requirements.txt` | MODIFY | add `google-auth`, `PyJWT[crypto]` |
| `docker-compose.yml` | MODIFY | social env block after Google Play block (after line 198); `.p8` rides existing `secrets:/app/secrets:ro` mount (line 223) |
| `deploy_ec2.sh` | MODIFY | env-sanity assert for `SOCIAL_LOGIN_ENABLED` + `.p8` presence |
| **Web** `frontend/src/api/client.ts` | MODIFY | `auth.oauth(provider, credential, nonce?, name?)` |
| **Web** `frontend/src/context/AuthContext.tsx` | MODIFY | `socialLogin(provider)` + state |
| **Web** `frontend/src/pages/Login.tsx` / `Signup.tsx` | MODIFY | social button row above password form |
| **Web** `frontend/src/pages/OAuthCallback.tsx` | **CREATE** | redirect handler + `state` CSRF check |
| **iOS** `Features/AuthView.swift` | MODIFY | Apple + Google buttons in `actionCard` |
| **iOS** `Features/AuthViewModel.swift` | MODIFY | `loginWithApple()` (nonce + first-signin name), `loginWithGoogle()` |
| **iOS** `Networking/APIClient.swift` | MODIFY | `loginWithSocialProvider(...)` after `login()` |
| **iOS** `Networking/Endpoint.swift` | MODIFY | `Endpoint.oauth(provider:)` |
| **iOS** `Auth/AuthModels.swift` | MODIFY | `OAuthVerifyRequest` |
| **Android** `data/model/Auth.kt` | MODIFY | `OAuthVerifyRequest` |
| **Android** `core/net/SeekingBetaApi.kt` | MODIFY | `oauthVerify(provider, credential, nonce?)` (~line 48) |
| **Android** `features/auth/AuthViewModel.kt` | MODIFY | `initiateGoogleLogin()`, `initiateFacebookLogin()` |
| **Android** `features/auth/AuthScreen.kt` | MODIFY | social buttons in `CredentialsCard` |
| **Android** `core/social/{GoogleCredentialManager,FacebookLoginManager}.kt` | **CREATE** | SDK glue; init FB in `MainActivity` |

**Existing-code anchors for the implementer:**
- `api/auth.py` — `create_access_token(user_id, email, token_type)` issues the access+refresh JWT post-verify. HS256 `_encode_jwt`/`_decode_jwt` are for **our** sessions only — **not** for RS256 provider tokens.
- `api/service.py` — billing verify endpoints to mirror: `POST /api/billing/apple/verify` (~4283), `POST /api/billing/google/verify` (~4322; note rate-limiting + authoritative-fetch + never-grant-on-forgery); provider dispatch `_billing_provider_from_tiers` (~2407).
- `docker-compose.yml` — Google Play billing env (lines 187–198) + read-only `secrets` mount (line 223) are the exact template for social secrets.
- `deploy_ec2.sh` — already rebuilds/recreates `prophecy-api`; the only deploy delta is box `.env` + dropping `apple-signin-key.p8` into the host `secrets/`.
