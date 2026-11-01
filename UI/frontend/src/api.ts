export class ApiError extends Error {
  constructor(message: string, public kind: 'network' | 'http' | 'response', public status?: number) {
    super(message); this.name = 'ApiError';
  }
}

export async function api<T>(path: string, options?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`/api${path}`, options);
  } catch {
    throw new ApiError('로컬 대시보드 서버에 연결할 수 없습니다. 서버를 실행한 뒤 다시 시도하세요.', 'network');
  }
  if (!response.ok) {
    const body = await response.json().catch(() => null);
    const detail = body?.detail;
    throw new ApiError(typeof detail === 'string' ? detail : detail ? JSON.stringify(detail) : `서버가 요청을 처리하지 못했습니다. (HTTP ${response.status})`, 'http', response.status);
  }
  try { return await response.json(); }
  catch { throw new ApiError('대시보드 서버의 응답 형식이 올바르지 않습니다. 접속 주소와 서버 상태를 확인하세요.', 'response'); }
}
export function post<T>(path: string, body: unknown = {}) {
  return api<T>(path, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
}
export function upload<T>(path: string, file: File) {
  const data = new FormData(); data.append('file', file);
  return api<T>(path, { method: 'POST', body: data });
}
