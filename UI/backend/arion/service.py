"""Explicit local server lifecycle; no login item, scheduler, or public binding."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid

ROOT=Path(__file__).resolve().parents[2]
DATA=Path(os.environ.get('ARION_DATA_DIR',ROOT/'.data'))
PORT=int(os.environ.get('ARION_PORT','8765'))
STATE=DATA/f'server-{PORT}.json'
URL=f'http://127.0.0.1:{PORT}'


def health():
    try:
        with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(URL+'/api/health',timeout=1) as response:
            return json.load(response)
    except (OSError,ValueError,urllib.error.URLError):return None


def is_arion(result):
    return result and result.get('status')=='ok' and (result.get('application')=='arion-dashboard' or result.get('scope')=='local-replay-and-live')


def start():
    running=health()
    if is_arion(running):
        print(f'이미 실행 중입니다: {URL}');return 0
    if running:
        print(f'{PORT} 포트에 다른 서버가 응답하고 있습니다. 종료하지 않았습니다.',file=sys.stderr);return 1
    if not (ROOT/'frontend/dist/index.html').exists():
        print('먼저 ./setup.sh를 실행하세요.',file=sys.stderr);return 1
    DATA.mkdir(parents=True,exist_ok=True)
    token=uuid.uuid4().hex;env=dict(os.environ,ARION_SERVER_INSTANCE=token)
    log_path=DATA/f'server-{PORT}.log'
    with log_path.open('ab') as log:
        proc=subprocess.Popen([sys.executable,'-m','uvicorn','arion.main:app','--host','127.0.0.1','--port',str(PORT),'--no-access-log'],
                              cwd=ROOT/'backend',env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
    for _ in range(80):
        if proc.poll() is not None:break
        result=health()
        if result and result.get('instance_id')==token:
            STATE.write_text(json.dumps({'pid':proc.pid,'instance_id':token,'url':URL}))
            print(f'대시보드 실행 완료: {URL}\n로그: {log_path}\n종료: ./server.sh stop')
            return 0
        time.sleep(.1)
    if proc.poll() is None:proc.terminate()
    print(f'실행하지 못했습니다. 로그를 확인하세요: {log_path}',file=sys.stderr);return 1


def stop():
    if not STATE.exists():
        print('이 실행 도구가 시작한 서버 기록이 없습니다. 다른 서버를 종료하지 않았습니다.');return 0
    saved=json.loads(STATE.read_text());result=health()
    if not result:
        print('응답하는 서버가 없습니다. PID만으로 프로세스를 종료하지 않았습니다.');return 0
    if result.get('instance_id')!=saved['instance_id']:
        print('다른 서버가 실행 중입니다. 종료하지 않았습니다.',file=sys.stderr);return 1
    try:os.kill(saved['pid'],signal.SIGTERM)
    except ProcessLookupError:pass
    for _ in range(80):
        if not health():
            STATE.unlink(missing_ok=True);print('대시보드를 종료했습니다.');return 0
        time.sleep(.1)
    print('종료 대기 중입니다. 잠시 뒤 ./server.sh status로 확인하세요.');return 1


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['start','stop','status']);args=parser.parse_args()
    if args.action=='start':return start()
    if args.action=='stop':return stop()
    result=health();print(f'실행 중: {URL}' if is_arion(result) else '대시보드 서버가 실행 중이 아닙니다.')
    return 0 if is_arion(result) else 1

if __name__=='__main__':raise SystemExit(main())
