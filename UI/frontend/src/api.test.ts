import { afterEach, expect, it, vi } from 'vitest';
import { ApiError, api, upload } from './api';
afterEach(()=>vi.unstubAllGlobals());
it('turns a browser connection failure into an actionable Korean error',async()=>{
  vi.stubGlobal('fetch',vi.fn().mockRejectedValue(new TypeError('Failed to fetch')));
  await expect(api('/runs')).rejects.toMatchObject({name:'ApiError',kind:'network',message:expect.stringContaining('서버를 실행')});
});
it('keeps file validation errors separate from connection failures',async()=>{
  vi.stubGlobal('fetch',vi.fn().mockResolvedValue(new Response(JSON.stringify({detail:'재생할 수 있는 유효한 레코드가 없습니다.'}),{status:422})));
  await expect(api('/runs/import')).rejects.toMatchObject({kind:'http',status:422,message:'재생할 수 있는 유효한 레코드가 없습니다.'});
});
it('reports a non-JSON server error without losing its HTTP status',async()=>{
  vi.stubGlobal('fetch',vi.fn().mockResolvedValue(new Response('Bad gateway',{status:502})));
  await expect(api('/runs')).rejects.toMatchObject({kind:'http',status:502,message:expect.stringContaining('502')});
});
it('does not misreport invalid JSON as a network failure',async()=>{
  vi.stubGlobal('fetch',vi.fn().mockResolvedValue(new Response('<html>wrong server</html>',{status:200})));
  await expect(api('/runs')).rejects.toMatchObject({kind:'response'});
});
it('sends a Korean JSONL filename as multipart without overriding its boundary',async()=>{
  const fetch=vi.fn().mockResolvedValue(new Response(JSON.stringify({accepted_records:4281,excluded_records:1}),{status:200}));
  vi.stubGlobal('fetch',fetch);
  const file=new File(['{}\n'],'비행수집_2026-10-05_185146_mavlink_raw.jsonl',{type:'application/x-ndjson'});
  expect(await upload('/runs/import',file)).toEqual({accepted_records:4281,excluded_records:1});
  const [url,options]=fetch.mock.calls[0];
  expect(url).toBe('/api/runs/import');expect(options.headers).toBeUndefined();
  expect((options.body as FormData).get('file')).toMatchObject({name:file.name});
});
it('exposes a stable error kind for a manual retry without auto-reposting uploads',async()=>{
  const fetch=vi.fn().mockRejectedValue(new TypeError('network'));vi.stubGlobal('fetch',fetch);
  await expect(upload('/runs/import',new File(['{}'],'flight.jsonl'))).rejects.toBeInstanceOf(ApiError);
  expect(fetch).toHaveBeenCalledTimes(1);
});
