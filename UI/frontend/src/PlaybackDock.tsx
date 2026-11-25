import { ChevronDown, Pause, Play, RotateCcw, SkipBack, SkipForward } from 'lucide-react';
import { detectionKey, timeLabel, type Detection } from './types';

export type PlaybackCommand = (command: string, args?: Record<string, number>) => void;

export default function PlaybackDock({
  isLive,
  cursor,
  duration,
  start = 0,
  playback,
  command,
  available,
  events = [],
  onEvent,
}: {
  isLive: boolean;
  cursor: number;
  duration: number;
  start?: number;
  playback: { playing: boolean; following: boolean; speed: number };
  command: PlaybackCommand;
  available: boolean;
  events?: Detection[];
  onEvent?: (event: Detection) => void;
}) {
  return (
    <footer className="playback-dock">
      <div className="playback-top">
        <div className="playback-title">
          <span className={`status-dot ${playback.playing ? 'teal' : 'gray'}`} />
          {isLive
            ? playback.following
              ? '실시간 관측'
              : '화면 일시정지 · 수집 중'
            : playback.playing
              ? '재생 중'
              : '일시정지'}
          <span className="keyboard-hint">SPACE</span>
        </div>
        <div className="playback-time">
          <strong>{timeLabel(cursor, true)}</strong>
          <span>/ {timeLabel(duration, true)}</span>
        </div>
      </div>
      <div className="playback-controls">
        <button
          className="icon-button"
          aria-label="처음으로"
          onClick={() => {
            command('pause');
            command('seek', { t_ms: 0 });
          }}
          disabled={!available}
        >
          <RotateCcw size={17} />
        </button>
        <button
          className="icon-button"
          aria-label="5초 뒤로"
          onClick={() => command('seek', { t_ms: cursor - 5000 })}
          disabled={!available}
        >
          <SkipBack size={17} />
        </button>
        <button
          className="play-button"
          aria-label={(isLive ? playback.following : playback.playing) ? '일시정지' : '재생'}
          onClick={() =>
            command((isLive ? playback.following : playback.playing) ? 'pause' : 'play')
          }
          disabled={!available}
        >
          {(isLive ? playback.following : playback.playing) ? (
            <Pause size={18} fill="currentColor" />
          ) : (
            <Play size={18} fill="currentColor" />
          )}
        </button>
        <button
          className="icon-button"
          aria-label="5초 앞으로"
          onClick={() => command('seek', { t_ms: cursor + 5000 })}
          disabled={!available}
        >
          <SkipForward size={17} />
        </button>
        <div className="timeline-track">
          <input
            type="range"
            aria-label="재생 시점"
            min={isLive ? start : 0}
            max={duration}
            step="100"
            value={cursor}
            onChange={(e) => command('seek', { t_ms: Number(e.target.value) })}
            disabled={!available}
            style={
              {
                '--progress': `${(cursor / Math.max(1, duration)) * 100}%`,
              } as React.CSSProperties
            }
          />
          <div className="timeline-markers">
            {events.map((e) => (
              <button
                key={detectionKey(e)}
                title={`${timeLabel(e.decided_at_ms, true)} ${e.title}`}
                aria-label={e.title}
                style={{ left: `${(e.decided_at_ms / duration) * 100}%` }}
                onClick={() => onEvent?.(e)}
              />
            ))}
          </div>
        </div>
        {isLive ? (
          <button className="button small live-return" onClick={() => command('go_live')}>
            실시간으로 복귀
          </button>
        ) : (
          <label className="speed-select">
            <select
              aria-label="재생 속도"
              value={playback.speed}
              onChange={(e) => command('speed', { speed: Number(e.target.value) })}
            >
              {[0.25, 0.5, 1, 2, 4].map((s) => (
                <option value={s} key={s}>
                  {s}×
                </option>
              ))}
            </select>
            <ChevronDown size={12} />
          </label>
        )}
      </div>
    </footer>
  );
}
