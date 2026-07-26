# SeekingBeta.AI — Market-Driven Improvement Roadmap

*Compiled 2026-07-07 from a 6-lens market research sweep (~150 live sources: competitor sites/pricing, app-store review mining, Reddit communities, retention/growth literature, compliance guidance) + a full codebase feature inventory + adversarial critique pass. Scope: balanced acquisition/retention/conversion, quick wins + 1–2 big bets, stocks & sports equally, near-zero marginal cost (existing EC2/RunPod/SMTP, free APNs/FCM; no paid feeds or LLM APIs), solo dev.*

---

## 1. Market findings

### AI stock-signal market
- Incumbents (Danelfin, TipRanks, Zacks, AltIndex, Prospero) cluster at **$19–60/mo** — SeekingBeta's $9.99/$19.99 is in-band, on the value side.
- **Every** competitor at this band pairs pricing with a free trial and/or money-back guarantee; SeekingBeta has neither → conversion leak.
- The universal paywall pattern: **gate the ranked "top picks" list, keep per-ticker lookup free** (Zacks #1 Rank, AltIndex Top 25, Danelfin ideas).
- Named market-wide weaknesses SeekingBeta can exploit: **thin explainability** (still unsolved even by Danelfin's factor scores), **weak or absent mobile apps** (Danelfin has none; Zacks UX dated), **unverifiable backtests**.
- Recommendation-change alerts on watchlisted tickers are the standard retention mechanic.

### Sports-picks market
- Documented **"transparency crisis"**: reviewers estimate ~9 of 10 handicappers inflate records. Timestamped, complete, graded pick ledgers (DataGolf, DeepBetting, Juice Reel) are the market's trust currency and its main conversion lever.
- Table stakes: free daily pick, per-pick reasoning, personal pick tracking, push alerts.
- **Golf + tennis coverage is rare** (only Dimers offers both, at ~3x SeekingBeta's price) — a real wedge, especially for international SEO.
- **NFL is SeekingBeta's most conspicuous gap** — every mainstream rival leads with it; missing-league coverage is the most-cited switching trigger.
- Personal bet/pick tracking (Pikkit) built a multi-million-user business on the tracking loop alone.

### User pain (review + community mining)
- Rage triggers: accuracy overclaiming, billing dark patterns (TipRanks' 4.8★ is dragged by billing 1★s), notification spam (~93% of sports-app pushes are promotional), paywalls too tight to let users verify claims before paying.
- Delight factors: transparent records **including losses**, clear reasoning per pick, fair free tiers, clean mobile UX.
- Users are told by reviewers to verify a service ~30 days before paying — the free tier should make auditing possible.

### Retention mechanics (evidence-backed)
- Finance has the **highest push opt-in of any category (~72%)**; contextual watchlist pushes open 3–14x more than generic blasts; 95% of silent installs churn in 90 days.
- Fixed-format, same-time daily briefs (Morning Brew pattern: 40–47% opens) build the habit loop.
- Trials convert ~5x better than pure freemium; ~31% of Android churn is involuntary (billing failures) — dunning recovers 15–20%.
- Share cards outperform referral programs at small scale (Duolingo: 5–10x more sharing than its referral scheme).
- Streaks work when forgiving and tied to *check-ins*, not transactions (regulator-safe for a gambling-adjacent product).

### Trust / compliance
- SEC "AI-washing" actions + FTC Operation AI Comply target unverifiable AI accuracy claims; Apple 2.3.1 is the practical store-rejection vector.
- The "scamdicapper" red-flag list (claims >58–70% winners, "locks," cherry-picked screenshots) is a free positioning playbook to invert: **sell modest numbers hard, show losses, publish methodology**.
- Tier-differentiated pick quality is a known fraud marker → "every tier sees the same models — paid sees more, sooner" is a differentiating trust claim SeekingBeta can make truthfully today.
- Responsible-gaming signals (1-800-GAMBLER, "analytics only, we never take bets", 18+/21+) are becoming expected of prediction apps in 2026; voluntary adoption reads as credibility.

### Positioning conclusion
The deepest buyer objection in both markets is **record trustworthiness, not pick quality**. Reframe the product promise from convenience ("picks in one place") to verifiability:

> **"Every prediction published, timestamped, and graded — stocks and sports."**

No competitor found operates in both verticals; none can copy the cross-market ledger claim.

---

## 2. Current-state audit (verified in code)

**Real tiers** (`pythia_prophecy/api/models.py` `TIER_CONFIG`): Free $0 / Basic $9.99 / Pro $19.99 monthly — gated by watchlist size (5/15/∞), analyses/day (10/50/∞), stocks per request, lookback, CSV. All tiers get the same models (`lstm_5d`, `lstm_jackpot`). Universe 6,664 tickers; sports boards fully public.

**Assets already in place:** track record w/ Sharpe/drawdown/regime breakdown; sports backtest browsers w/ per-sport accuracy; Oracle watchlist + Finviz insights; batch Analysis + CSV; ~60 analytics events; Discord funnel; **on-device TFLite inference in both mobile apps** (unmarketed differentiator).

**Dormant assets (near-free to activate):**
| Asset | Location | State |
|---|---|---|
| `email_alerts_enabled`, `daily_digest_enabled` prefs | `api/compliance_store.py` | Stored, never read — no sender exists |
| Feature-attribution endpoint | divination engine | Live, never proxied to users; web "top drivers" is canned (`Analysis.tsx:65`) |
| `lstm_quant` third model | `service.py:5594,6695` | Endpoint live, in no tier's model list |
| NHL engine | `pythia_divination/sports/hockey` + disabled tab (`SportsDashboard.tsx:1698`) | Half-built, UI stub exists |
| Paper trading / portfolio | engine `/paper/*`, `/portfolio/allocate` | Unexposed to users |
| Push notifications | — | Zero APNs/FCM code; clean slate |

---

## 2b. BLOCKER found while building Wave 1.3: grading is stale

Verified against production on 2026-07-26. The platform publishes picks daily but
does not write graded outcomes back promptly for most sports:

| Sport | Live boards today | Graded history ends | Season sample |
|---|---|---|---|
| MLB | 15/day | **2025-09-28** (a full season behind) | 0 |
| Golf | yes | **no dates at all** on 102 backtests | 0 |
| Tennis | 4/day | 2026-07-19 (current) | 95 |
| Soccer | seasonal | 2026-07-12 (current) | 90 |
| Basketball | off-season | 2026-04-02 (season end, expected) | 475 |
| Football | off-season | 2026-01-04 (season end, expected) | 16 |
| Stocks | daily | backtest curve ends **2026-05-11** | 1,100 |

Consequences, in priority order:

1. **The positioning shipped in Wave 1.1 is not yet fully true.** "Every pick
   published, timestamped, and graded" holds for tennis and soccer, but MLB
   picks are published and never graded. This is the single highest-priority
   fix on the board.
2. **Wave 2.1 (the verifiable ledger, the flagship trust asset) cannot ship on
   this data.** A ledger is only worth building on top of a grading pipeline
   that closes the loop daily.
3. The Daily Brief's "yesterday went W-L" line will read 0-0 essentially every
   day until grading catches up, and the weekly Receipts email has almost
   nothing to report.

Also unexplained: basketball's season summary was observed at 649 samples / 471
hits (72.6%) in one reading and 475 / 204 (43%) in later readings, which were
themselves stable across three consecutive calls. Root-cause this before those
figures are published as trust claims, since a number that moves is worse than
no number.

**Recommended next action:** a "close the loop" work item ahead of Wave 2.1,
writing graded outcomes back for every sport within 24h of an event finishing,
starting with MLB (highest volume, worst gap) and dating the golf backtests.

## 3. Roadmap

### Wave 1 — Quick wins (~2–7 days each)

| # | Item | Goal | Summary |
|---|---|---|---|
| 1.1 | **Positioning + honest-numbers copy sweep** | trust/conversion | Verifiable-record reframe; kill guarantee-ish phrasing; "What we are / aren't" box; "same models every tier" line; responsible-gaming footer on sports; store-listing hygiene |
| 1.2 | **Daily Brief** | retention | Fixed-format morning email (existing SMTP + dead `daily_digest_enabled` pref) + same content as app landing view; push variant later |
| 1.3 | **Weekly "Receipts" email** | retention/trust | Monday graded-results digest to `newsletter_enabled` opt-ins only: wins AND losses w/ sample sizes, one teaser pick, locked-pick count |
| 1.4 | **ASO overhaul** | acquisition | Keyword titles/subtitles, indexed screenshot captions, Play long-description keywords, 2–3 Custom Product Pages by intent |
| 1.5 | **Additive paywall** | conversion | NEW gated surfaces only (no clawbacks): ranked "Today's strongest signals" list for Basic/Pro w/ blurred locked rows; `lstm_quant` as Pro-exclusive third model |
| 1.6 | **Trials + fair billing** | conversion | 7-day Basic trial (StoreKit/Play intro offers + Stripe); pre-renewal reminder email; honest refund page (platform flows, no self-stated guarantee); Play dunning handling |

### Wave 2 — Medium builds (~1–3 weeks each)

| # | Item | Goal | Summary |
|---|---|---|---|
| 2.1 | **Verifiable pick ledger + methodology page** *(flagship)* | trust | Append-only public ledger: price-at-publish (stocks) / model probability + event time (sports; **no betting odds** — no license-clean free source), permalinks, auto-graded outcomes, live-vs-backtest split, CSV export; daily hash **committed to a public GitHub repo** for third-party timestamping; calibration table + retrain changelog; becomes homepage + Show HN artifact |
| 2.2 | **"Why this pick" cards** | trust/retention | Proxy the engine's attribution endpoint; 3–4 templated glanceable factors + confidence-bucket hit rate; no LLM |
| 2.3 | **Watchlist signal-flip push alerts** | retention | APNs/FCM both apps; push only on stance-flips + graded outcomes; daily cap, quiet hours; free = 3 slots; "no promo pushes, ever" pledge page |
| 2.4 | **"My Record" tails + paper portfolio** | retention | Sports: Tail button → auto-graded personal record. Stocks: "Follow the model" virtual $10k via engine `/paper/*`. Check-in streaks w/ forgiveness; no urgency pushes |
| 2.5 | **Share cards** | acquisition | Server-rendered graded-pick cards (timestamp + running record + ledger deep link); losses shareable too; double as OG images |
| 2.6 | **Home-screen widgets** | retention | WidgetKit/Glance: today's board + yesterday's record — the brief without push permission |
| 2.7 | **Surface NHL** | acquisition | Finish half-built hockey pipeline + enable existing tab before October |

### Wave 3 — Big bets

| # | Item | Goal | Summary |
|---|---|---|---|
| 3.1 | **NFL board for 2026 season** *(recommended)* | acquisition | nflverse (free) + RunPod training + existing board UI; September = annual acquisition spike; deadline soft (NHL proves pattern first); publish honest backtest, modest numbers |
| 3.2 | **Focused SEO + free tools** | acquisition | Dozens (not thousands) of high-quality golf-tournament/tennis-draw prediction pages from board data; 2–3 client-side calculators; expand only if pages rank. Avoids Google scaled-content-abuse exposure |

### Rejected / deferred (deliberately)
24h-delayed free stock picks (leaks 5-day-horizon edge) · self-stated 30-day IAP refund guarantee (platform-impossible) · win-only share prompts (cherry-picking) · "streak at risk" urgency pushes (contradicts no-promo pledge) · mass programmatic SEO (policy risk) · marketing email to lapsed users w/o consent (GDPR/CAN-SPAM) · displaying betting odds (licensing + compliance surface).

**Legal note:** keep personalization (alerts, tail records) as *delivery filtering of identical universal content* to preserve the impersonal-publisher posture; a one-time legal sanity check is advisable. Seasonality churn (off-season cancel points): mitigate later via annual plans + cross-vertical bridging in the Daily Brief.

---

## 4. Sequencing rationale

Wave 1 is copy, cron jobs, gating logic, and store metadata — all shippable solo in days each, and items 1.2/1.3 light up preference toggles users can already see. Wave 2's ledger (2.1) is the flagship trust asset everything else links back to (share cards, SEO pages, Show HN launch). Push (2.3) requires new mobile builds, so it batches naturally with widgets (2.6) into one store submission. NHL (2.7) before October derisks the NFL bet by proving the "new sport on existing pipeline" pattern cheaply.
