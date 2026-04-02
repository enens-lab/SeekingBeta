import type { SportsPlayerProfile } from '../../api/client';
import PlayerPortrait from './PlayerPortrait';
import './SportsVisuals.css';

type PlayerProfileCardProps = {
  name: string;
  profile?: SportsPlayerProfile | null;
  compact?: boolean;
  align?: 'left' | 'right';
};

function PlayerProfileCard({
  name,
  profile,
  compact = false,
  align = 'left',
}: PlayerProfileCardProps) {
  const stats = (profile?.stats || []).slice(0, compact ? 3 : 4);

  return (
    <div className={`player-profile-card ${compact ? 'compact' : ''} ${align === 'right' ? 'align-right' : ''}`}>
      <PlayerPortrait imageUrl={profile?.imageUrl} label={name} size={compact ? 'sm' : 'md'} />
      <div className="player-profile-copy">
        <div className="player-profile-header">
          <strong className="player-profile-name">{name}</strong>
          {profile?.country ? <span className="player-profile-country">{profile.country}</span> : null}
        </div>
        {profile?.subtitle ? <div className="player-profile-subtitle">{profile.subtitle}</div> : null}
        {stats.length > 0 ? (
          <div className="player-stat-chip-row">
            {stats.map((stat) => (
              <span key={`${name}-${stat.label}`} className="player-stat-chip">
                <span>{stat.label}</span>
                <strong>{stat.value}</strong>
              </span>
            ))}
          </div>
        ) : null}
      </div>
    </div>
  );
}

export default PlayerProfileCard;
