import { useMemo, useState, type CSSProperties } from 'react';
import './SportsVisuals.css';

type TeamLogoProps = {
  logoUrl?: string | null;
  label: string;
  abbreviation?: string | null;
  primaryColor?: string | null;
  size?: 'sm' | 'md' | 'lg';
};

function TeamLogo({
  logoUrl,
  label,
  abbreviation,
  primaryColor,
  size = 'md',
}: TeamLogoProps) {
  const [broken, setBroken] = useState(false);

  const fallbackText = useMemo(() => {
    if (abbreviation && abbreviation.trim()) return abbreviation.trim().slice(0, 3).toUpperCase();
    const initials = label
      .split(/\s+/)
      .filter(Boolean)
      .slice(0, 3)
      .map((token) => token[0]?.toUpperCase() || '')
      .join('');
    return initials || 'TM';
  }, [abbreviation, label]);

  const style = useMemo(
    () =>
      ({
        '--team-accent': primaryColor || 'rgba(255, 255, 255, 0.18)',
      }) as CSSProperties,
    [primaryColor]
  );

  return (
    <div className={`team-logo team-logo-${size}`} style={style} aria-hidden="true">
      {logoUrl && !broken ? (
        <img src={logoUrl} alt={`${label} logo`} onError={() => setBroken(true)} loading="lazy" />
      ) : (
        <span className="team-logo-fallback">{fallbackText}</span>
      )}
    </div>
  );
}

export default TeamLogo;
