import { useEffect, useRef, useState } from 'react';
import type { Frame } from './types';

export function useReplay(runId: string, isLive = false) {
  const [frame, setFrame] = useState<Frame | null>(null);
  const [playback, setPlayback] = useState({ playing: false, speed: 1, following: isLive, live: isLive });
  const [connection, setConnection] = useState('connecting');
  const [packet, setPacket] = useState(-1);
  const socket = useRef<WebSocket | null>(null);
  const last = useRef({ t_ms: 0, speed: 1 });
  useEffect(() => {
    if (!runId) return;
    let disposed = false, retry: ReturnType<typeof setTimeout>, retries = 0;
    last.current = { t_ms: 0, speed: 1 }; setFrame(null); setPlayback({ playing: false, speed: 1, following: isLive, live: isLive }); setPacket(-1);
    function connect() {
      setConnection(retries ? 'reconnecting' : 'connecting');
      const ws = new WebSocket(`${location.protocol === 'https:' ? 'wss:' : 'ws:'}//${location.host}/api/ws/runs/${runId}`);
      socket.current = ws;
      ws.onopen = () => {
        if (disposed) { ws.close(); return; }
        const resumed = retries > 0; setConnection('connected'); retries = 0;
        if (!isLive || resumed) ws.send(JSON.stringify({ action: 'seek', t_ms: last.current.t_ms }));
        ws.send(JSON.stringify({ action: 'speed', speed: last.current.speed }));
      };
      ws.onmessage = event => {
        if (disposed) return;
        const data = JSON.parse(event.data);
        if (data.type === 'frame') {
          last.current = { t_ms: data.frame.t_ms, speed: data.playback.speed };
          setFrame(data.frame); setPlayback(data.playback); setPacket(data.sequence);
        }
      };
      ws.onclose = () => {
        if (disposed) return;
        setConnection('reconnecting'); setPlayback(p => ({ ...p, playing: false }));
        retry = setTimeout(connect, Math.min(10000, 1000 * 2 ** retries++));
      };
      ws.onerror = () => ws.close();
    }
    connect();
    return () => { disposed = true; clearTimeout(retry); socket.current?.close(); };
  }, [runId, isLive]);
  useEffect(() => {
    if (packet < 0) return;
    const id = requestAnimationFrame(() => {
      if (socket.current?.readyState === WebSocket.OPEN) socket.current.send(JSON.stringify({ action: 'ack', sequence: packet }));
    });
    return () => cancelAnimationFrame(id);
  }, [packet]);
  const command = (action: string, payload: Record<string, number> = {}) => {
    if (socket.current?.readyState === WebSocket.OPEN) socket.current.send(JSON.stringify({ action, ...payload }));
  };
  return { frame: frame?.run_id === runId ? frame : null, playback, connection, command };
}
