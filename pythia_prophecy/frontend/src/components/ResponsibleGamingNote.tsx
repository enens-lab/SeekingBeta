/**
 * Voluntary responsible-gaming note for sports surfaces. We are analytics-only
 * (no bets taken or placed), but prediction apps are expected to carry this in
 * 2026 — adopting it voluntarily reads as credibility, not obligation.
 */
function ResponsibleGamingNote() {
  return (
    <div className="responsible-gaming-note" role="note">
      <span className="responsible-gaming-strong">Analytics only — we never take or place bets.</span>{' '}
      If you choose to bet elsewhere, keep it fun and legal: 18+ (21+ in some regions). Problem
      gambling help (US): call or text <a href="tel:1-800-426-2537">1-800-GAMBLER</a>.
    </div>
  );
}

export default ResponsibleGamingNote;
