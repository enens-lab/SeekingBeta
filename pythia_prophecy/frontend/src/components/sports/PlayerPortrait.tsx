import { useMemo, useState } from 'react';
import './SportsVisuals.css';

type PlayerPortraitProps = {
  imageUrl?: string | null;
  label: string;
  size?: 'sm' | 'md' | 'lg';
};

function PlayerPortrait({ imageUrl, label, size = 'md' }: PlayerPortraitProps) {
  const [broken, setBroken] = useState(false);

  const fallbackText = useMemo(() => {
    const initials = label
      .split(/\s+/)
      .filter(Boolean)
      .slice(0, 2)
      .map((token) => token[0]?.toUpperCase() || '')
      .join('');
    return initials || 'SB';
  }, [label]);

  return (
    <div className={`player-portrait player-portrait-${size}`} aria-hidden="true">
      {imageUrl && !broken ? (
        <img src={imageUrl} alt={`${label} portrait`} onError={() => setBroken(true)} loading="lazy" />
      ) : (
        <span className="player-portrait-fallback">{fallbackText}</span>
      )}
    </div>
  );
}

export default PlayerPortrait;
