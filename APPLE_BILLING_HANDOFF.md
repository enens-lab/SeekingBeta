# Apple Billing Handoff

Prepared: April 5, 2026

## Problem

The iOS app completed the StoreKit purchase flow, but the backend returned:

- `404 Not Found` on `POST /api/billing/apple/verify`

This was confirmed against production at:

- `https://seekingbeta.ai/api/billing/apple/verify`

That means the failure was not caused by a user's pre-existing Stripe-backed web account. The mobile app was successfully reaching the backend, but the backend route did not exist.

## What Was Added

The following backend files were updated:

- `pythia_prophecy/api/service.py`
- `pythia_prophecy/api/models.py`
- `pythia_prophecy/api/billing_store.py`

Implemented changes:

- added `POST /api/billing/apple/verify`
- added a backend `AppleVerifyRequest` model
- extended billing status responses so iOS can decode Apple and Stripe entitlement state side-by-side
- extended the billing store schema to persist Apple subscription metadata
- mapped Apple product IDs to `basic` and `pro`
- returned a unified billing status payload after Apple verification

## Important Payload Detail

The current iOS app field name is `signed_transaction_info`, but the app is actually sending:

- `transaction.jsonRepresentation`

That is JSON, not a signed JWS from the App Store Server API.

To avoid blocking QA, the backend route was intentionally made tolerant of the current payload shape:

- it first tries to parse JSON directly
- if parsing fails, it then tries to decode a JWS payload body without signature validation

This lets the mobile flow work now, but it is not full Apple server-side receipt verification yet.

## Current Intent

This patch is meant to:

- unblock end-to-end mobile subscription testing
- persist Apple entitlements in the billing store
- let mixed Apple + Stripe account states be represented more clearly

This patch is not yet a full production-grade App Store Server integration.

## Recommended Follow-Up

1. Deploy `pythia_prophecy` with these billing changes.
2. Verify the billing store migrations run successfully in the live environment.
3. Test mobile purchase on a fresh in-app account and a fresh Apple Sandbox tester.
4. Confirm `GET /api/billing/status` shows Apple entitlement fields after purchase.
5. Later, replace the current tolerant parser with real App Store Server verification if stronger trust guarantees are needed.

## Clean Test Setup

To rule out account crossover issues after deploy:

1. Create a brand-new app account in mobile with a fresh email alias.
2. Verify that email inside the app flow.
3. Do not use that same email on the website.
4. Sign into a separate Apple Sandbox tester account on the device/simulator.
5. Purchase the subscription from that new mobile-only app account.

This isolates:

- backend account identity
- prior website/Stripe identity
- Apple purchase identity

## Validation Run

The updated backend files compiled successfully with:

```bash
python3 -m py_compile \
  pythia_prophecy/api/models.py \
  pythia_prophecy/api/billing_store.py \
  pythia_prophecy/api/service.py
```
