import { useCallback, useEffect, useRef, useState } from 'react';
import { api } from './api';

export function useServerHealth() {
  const [status, setStatus] = useState<'checking' | 'online' | 'offline'>('checking');
  const [wholeRunOnly, setWholeRunOnly] = useState<boolean>();
  const [edition, setEdition] = useState<'v0.2' | 'legacy'>();
  const mounted = useRef(false);
  const pending = useRef<Promise<boolean> | null>(null);
  const check = useCallback(() => {
    if (pending.current) return pending.current;
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 2500);
    pending.current = api<{
      status: string;
      whole_run_only?: boolean;
      edition?: 'v0.2' | 'legacy';
    }>('/health', {
      signal: controller.signal,
    })
      .then((result) => {
        const ok = result.status === 'ok';
        if (mounted.current) {
          setStatus(ok ? 'online' : 'offline');
          if (ok) {
            setWholeRunOnly(result.whole_run_only ?? false);
            setEdition(result.edition ?? 'legacy');
          }
        }
        return ok;
      })
      .catch(() => {
        if (mounted.current) setStatus('offline');
        return false;
      })
      .finally(() => {
        clearTimeout(timeout);
        pending.current = null;
      });
    return pending.current;
  }, []);
  useEffect(() => {
    mounted.current = true;
    check();
    const interval = setInterval(check, 5000);
    const onFocus = () => {
      void check();
    };
    window.addEventListener('focus', onFocus);
    return () => {
      mounted.current = false;
      clearInterval(interval);
      window.removeEventListener('focus', onFocus);
    };
  }, [check]);
  return { status, wholeRunOnly, edition, check };
}
