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

**Recommended next action:** a "close the loop" work item ahead of Wave 2.1,
writing graded outcomes back for every sport within 24h of an event finishing,
starting with MLB (highest volume, worst gap) and dating the golf backtests.

### RESOLVED 2026-07-26: football and basketball graded every game as an away win

A second, worse defect found while investigating the football accuracy figure.
Published backtests contained **0 home wins across 323 football and 533
basketball games**, against a realistic 54% for MLB. Both sports' accuracy
numbers were not measuring the model at all; they were measuring how often it
happened to pick the away team.

Root cause: `actual_winner` was derived with
`int(getattr(row, "home_win", 0) or 0)`, so an **absent** label silently became
"the away team won" instead of raising. Two independent conditions hit that
default: a `predictions.merge(dataset, ...)` where both frames carry `home_win`
(pandas suffixes both to `home_win_x`/`home_win_y`, leaving no plain column), and
source tables carrying only `home_score`/`away_score`.

Fixed in `export_football_frontend_data.py` and
`export_basketball_frontend_data.py` (commit 9c6091d) via a `_resolve_home_win()`
resolver that returns `None` when the label is genuinely unknowable, so callers
skip ungraded games rather than inventing results, plus explicit merge suffixes.
Datasets regenerated and deployed (commit 207a771).

| Metric | Published before | Actual |
|---|---|---|
| Football 2026 season | 2/16 (12.5%) | **10/16 (62.5%)** |
| Football all graded | 12.4% | **65.3%** (vs 55.1% always-pick-home) |
| Basketball 2026 season | 204/475 (43%) | **471/649 (72.6%)** |
| Home-win rate, both | 0% | **55%** |

The NFL model is competitive and well calibrated: Brier 0.223, log loss 0.638
(beating the 0.693 coin-flip line), and accuracy rising with confidence (58.6%
below 60% confidence, 82.9% at 60-70%, 90% above 70%). The HGB baseline is
notably worse calibrated (log loss 0.757, worse than a coin flip), so the torch
model should remain the served one. **No modelling work was needed; the metric
was broken, not the model.**

This also explains the basketball "instability" noted during Wave 1.3: the first
reading (649/471) was correctly graded data, and later readings came from the
corrupted export. The figure was never unstable.

**Lesson worth institutionalizing:** a defaulted `getattr` on a label column
converts missing data into confident, wrong, publishable numbers. Grading code
should fail loudly or mark a result ungraded; it must never guess.

**Tripwire added (commit 6702473).** `ops/validate_sports_boards.py` runs inside
`refresh_sports_runpod.sh` after the S3 boards sync and before prophecy-api is
rebuilt. It snapshots the serving boards, and if any sport's home-win rate falls
outside 30-80% over 30+ gradeable games it rolls back and refuses to rebuild.
Verified both ways. It lives in the EC2 repo rather than the worker image
specifically so it protects against a stale worker.

### Grading loop, remaining state (as of 2026-07-26)

**MLB is an INGEST gap, not a grading gap.** `mlb_training_dataset_latest` on the
box contains **zero 2026 rows** (range 2020-07-23 to 2025-09-28), so the exporter
is correct and simply has nothing 2026 to grade. Live boards work because they
come from a runtime feed, not this dataset. The RunPod worker runs MLB
**export-only**; only tennis has an ingest + build step in its command list
(`runpod/sports/handler.py`).

> **DANGER before wiring this up:** `sports/mlb/build_training_dataset.py`
> **overwrites** `mlb_training_dataset_latest.{csv,parquet}` with only the seasons
> resolved from its arguments. Running `--season 2026` would **destroy the
> 12,755-row 2020-2025 history the models train on.** The safe invocation is a
> full-range rebuild:
> ```
> python -m sports.mlb.ingest_history --season 2026
> python -m sports.mlb.build_training_dataset --season-start 2020 --season-end 2026
> ```
> This is multi-season and memory-heavy (MLB export alone already needs >3 GB and
> was moved to RunPod after an OOM outage), and it must be validated end to end
> before being added to the nightly path. **Left unwired deliberately, pending a
> decision, rather than risking the training dataset.**

#### ATTEMPTED 2026-07-26 (authorized) and STOPPED: the rebuild would lose the pitchers

The 2026 ingest itself worked: **2,456 scheduled games, 1,574 completed with
details**. The rebuild then failed twice, both times upstream of the write, so the
live dataset was never modified (verified still 12,755 rows against backup).

1. `merge_starter_features` raised *"trying to merge on object and float64 columns
   for key away_probable_pitcher_id"*. A season whose probable pitchers were never
   announced contributes an all-null **object** column, so the concatenated frame
   became object while `starter_logs` stayed float64. Fixed in commit a92b1ca by
   coercing shared join keys to float64 on both sides.
2. Then `enrich_dataset_with_statcast` raised *"No objects to concatenate"*, because
   `groupby` drops NaN keys and the pitcher-ID column was entirely NaN, yielding zero
   groups.

**Root cause, and the reason this is stopped rather than patched through:** the
normalized schedules on disk no longer carry probable-pitcher IDs at all. 2024 and
2025 **lack the column entirely**; the 2026 pull returned **2,456/2,456 nulls**. The
committed dataset, by contrast, has them populated at **12,719/12,755**. So a rebuild
from current files would join starter features on an all-null key and silently empty
all **90** `away_starter_*` / `home_starter_*` columns. Starting pitcher is among the
most predictive inputs in baseball, so that is a material model regression wearing
the costume of a successful build.

`build_training_dataset` now **raises before writing** if every starter feature is
empty (commit a92b1ca), on the same principle as the away-win tripwire: a function
that overwrites the canonical dataset must fail loudly rather than degrade quietly.

**Do this first, before retrying the rebuild:** recover starting pitchers for
completed games from the **game details** feed rather than the schedule's
`probable_pitcher` field. Probables are only published shortly before first pitch, so
any historical schedule pull returns null and always will. Note that MLB **grading**
does not depend on this: completed 2026 games already carry scores and results, so
the backtest path needs the results, not the pitcher features.

**Golf: half fixed.** `event_start_date` is now threaded into golf backtests
(commit 6702473), which dates 53 of 172 boards locally (2024-10 to 2025-12). The
remaining boards lack the column upstream, and nothing covers 2026 yet. Dates
reach production on the next worker image rebuild.

**Stocks:** the backtest curve still ends 2026-05-11 and needs a fresh backtest
run to advance. Untouched.

### NHL: not close to shippable (assessed 2026-07-26)

The roadmap listed this as a cheap coverage win. It is not. What exists:
`sports/hockey/` has the full module set (client, ingest_history,
build_training_dataset, feature_engineering, torch_model, train_baseline,
train_torch, branding, constants). What is missing:

- **Data**: only **60 games ingested, 2025-10-07 to 2025-10-15** (nine days). No
  training dataset was ever built. Nine days cannot train a credible model.
- **No trained artifacts** (`artifacts/` has no hockey/nhl directory).
- **No exporter** (`scripts/export_hockey_frontend_data.py` does not exist; the
  football equivalent is ~900 lines).
- **No BFF plumbing**: `SportsBoardsResponse` in `pythia_prophecy/api/models.py`
  has no `hockey` field, and there is no collection assembly for it.
- **No RunPod worker entry**, and the UI tab is deliberately disabled.

Realistic order of work: full multi-season ingest -> build dataset -> train and
validate -> exporter -> BFF field + assembly -> worker entry -> enable the tab.
That is the "medium build" the roadmap estimated, not a finishing touch, and the
honest-numbers positioning means a model trained on thin data should not ship at
all. October season start is the natural deadline; the ingest is the thing to
start now because everything else waits on it.

#### UPDATE 2026-07-26: data pipeline fixed and working; model does NOT earn a launch

The NHL data problem is solved. Three separate defects were blocking it, all fixed:

| Defect | Effect | Commit |
|---|---|---|
| Ingest looped the current 32 teams per season and `raise_for_status()`'d | Backfill died instantly on `roster/SEA/20202021` (Seattle joined 2021-22) | 57925ad |
| `_LEAKY_COLUMNS` listed `home_win` | Stripped the training TARGET; 261 columns, no label | d3f9d17 |
| `build_goalie_game_logs` looped unguarded metric list | Bare `KeyError: won` (feed carries `team_won`) | d3f9d17 |
| Merge produced `home_score_detail` / `away_score_detail` | Final score leaked; baseline hit **ROC AUC 1.000 / 100% accuracy** | 7fe6c14 |

Resulting dataset is sound: **7,428 games, 6 seasons, 257 columns**, home-win rate
53.7% (real NHL is ~54-55%), per-season counts exactly right (868 for the
COVID-shortened 2020-21, then 1,312 = 32x82/2 for each full season), and an
empirical correlation audit shows max |corr| with the label of 0.192 and zero
features above 0.5.

**But both trained models fail to beat a trivial baseline**, measured on the same
1,486-game held-out split (2025-03-26 to 2026-04-16):

| Model | Accuracy | vs always-pick-home (53.1%) | ROC AUC | Log loss |
|---|---|---|---|---|
| HistGradientBoosting | 53.8% | **+0.7 pts** | 0.546 | 0.742 (worse than coin flip) |
| Torch | 52.8% | **-0.3 pts** | 0.552 | 0.686 |
| *NFL, for contrast* | *65.3%* | *+10.2 pts* | — | *0.638* |

An AUC of ~0.55 is barely distinguishable from noise, and the HGB variant is worse
calibrated than a coin flip. This is consistent with NHL being the hardest major
North American sport to model (low scoring, high variance, goalie-dominated).

#### FEATURE ATTEMPT 2026-07-26: the ceiling is data, not features. STOP HERE.

Added the best-documented NHL predictors that were missing entirely (only *goalie*
rest existed, never the team's): rest days, back-to-backs, third-in-four-nights,
7-day schedule density, road-trip length, plus power-play goals conceded and PIM
drawn as special-teams proxies. Correctly implemented and sanity-checked against
reality (back-to-back rate 14-17%, ~2.9 games/7 days, ~4 days mean rest), leakage
re-audited (max |corr| 0.192, none above 0.5), 216 -> 249 model-visible features.

**They did not work.** Same held-out split (n=1,486), baseline 53.1%:

| Model | Before | After |
|---|---|---|
| HistGradientBoosting | 53.8%, AUC 0.546 | 53.4%, AUC 0.553 |
| Torch | 52.8%, AUC 0.552 | 53.6%, AUC 0.554 |
| RandomForest | — | **54.5%, AUC 0.554** (best) |

Best case is **+1.4 points** over always-picking-home against a 57-60% bar.
Univariate correlations explain it: rest differential **+0.009**, back-to-back
**±0.014**, strongest new feature **0.049**. The effects the hockey literature
treats as real carry almost no signal in this data.

**What would actually move it is not obtainable from the free NHL feed:**
confirmed pregame goalie starts (the dominant factor in hockey outcomes — we hold
only the *previous* starter's history, and the NHL doesn't confirm until ~1h before
puck drop), and shot-quality metrics (xG, Corsi/Fenwick, high-danger chances) which
require a paid provider or play-by-play scraping. Both violate the near-zero-cost
constraint.

**Recommendation: stop investing in NHL.** This is a data ceiling, not an effort
problem, and further feature work is very likely to return the same result. The
features are kept because they cost nothing and make the dataset complete; the
negative result is recorded so it isn't rediscovered later.

**Recommendation: do not ship NHL boards.** Publishing a board whose model has no
measurable edge would contradict the exact positioning Wave 1.1 shipped, and the
honest presentation the ledger demands would show a coin flip. The expensive part
(clean multi-season data) is now done and committed, so the remaining work is
model quality, not plumbing: richer features (rest/travel, back-to-backs, goalie
starts confirmed pregame, special-teams rates), calibration, and a target of
roughly 57-60% before it earns a launch. The exporter, BFF field, worker entry and
UI tab should stay unbuilt until a model clears that bar.

## 2c. Merge-suffix audit (2026-07-26)

One pandas behaviour caused **five** distinct failures in a single day, three of them
silent and publishing wrong numbers for months:

| # | Failure | Silent? |
|---|---|---|
| 1 | Football + basketball graded every game as an away win | **yes** — months of wrong accuracy |
| 2 | Hockey final score leaked into training (fake ROC AUC 1.000) | **yes** |
| 3 | MLB starter features silently blanked | **yes** |
| 4 | MLB rebuild died on object-vs-float64 pitcher key | no, crashed |
| 5 | MLB export died on object-vs-int64 on the same key | no, crashed |

Root shape every time: a merge renames or shadows a column, then code reads the empty
twin (silent) or hits a dtype mismatch (loud). **The silent variety is the dangerous
one** — it yields confident, publishable, wrong numbers.

**Static scan** of `scripts/export_*_frontend_data.py` and
`sports/*/feature_engineering.py`: **24 merge sites with no explicit `suffixes=`**,
concentrated in `sports/mlb/feature_engineering.py` (11), `sports/football` (3),
`sports/hockey` (3), `sports/pga` (3), `sports/basketball` (2).

**Empirical scan** of every produced dataset and model artifact for collision
fingerprints found **no active collisions**:

- The `_detail` columns in the hockey (16) and MLB (12) datasets come from the
  *correct* pattern, `suffixes=("", "_detail")`, where the left frame keeps the clean
  name. Benign by construction.
- `profile_has_x` in the golf dataset is a **false positive** — a real field meaning
  "player has an X/Twitter account", sibling to `profile_has_instagram`.
- All 21 `validation_predictions.csv` artifacts are clean.

**Conclusion: current outputs are sound; the 24 sites are latent risk, not active
harm.** They would bite when a source schema changes — which is exactly how all five
of today's failures arose (the 2026 season arriving with different columns).

**Recommended convention, cheap to adopt going forward:** pass explicit non-empty
`suffixes=("", "_<source>")` on every merge so the left frame always keeps clean
names, and where a column is expected to carry data, assert it is not entirely null
rather than trusting it survived. The highest-value single guard is the one that
catches the silent case: after a build, flag any clean column that is fully null while
its suffixed twin holds data. That is the precise signature of the MLB starter bug.

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
