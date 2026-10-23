# X500 GPU smoke 인수인계 v1

이 패키지는 **실제 정상 SITL 일부로 로컬에서 통과한 LSTM 학습 → validation → 저장 → 복원 → 재개 시험**을 서버에서 그대로 실행하기 위한 것이다. GPU 실행은 아직 미실행이다. 접속 정보·CUDA build·드라이버·스케줄러는 미확인이다.

## 1. 전송과 hash 검증

`x500-gpu-smoke-v1.tar.gz`와 함께 받은 `.tar.gz.sha256`을 같은 디렉터리에 두고 다음을 실행한다. SHA256은 전달받은 archive 식별값과도 대조한다.

```bash
sha256sum -c x500-gpu-smoke-v1.tar.gz.sha256
tar -xzf x500-gpu-smoke-v1.tar.gz
cd x500-gpu-smoke-v1
python3 run_smoke.py --verify-only
```

verify-only는 Python 표준 라이브러리만 사용한다. `SHA256SUMS`와 `MANIFEST.json`이 코드·입력·config·환경 명세·reference의 모든 파일을 검사한다. 원본 개발 경로가 provenance/reference에 기록되어 있지만 실행에 사용하지 않는다. 출력은 반드시 패키지 **밖의 새 디렉터리**로 지정한다.

## 2. AI 격리 환경

기준: Python3.10(실제 로컬3.10.22),PyTorch2.1.2(실제 로컬2.1.2+cpu). 서버는 Python3.10/PyTorch2.1 계열·A6000 48GB로 예정되어 있으나 실제 환경은 미확인이다. 이 고정 시험은 torch2.1.2 release를 검사하며 CUDA local build suffix는 서버 확인 후 기록한다.

- `environment/requirements-common.lock`: 로컬에서 확인한 비-Torch 실행 의존성의 정확한 버전.
- `environment/policy.json`: 실행 조건, 미확인 CUDA/server 항목.
- `environment/cpu-reference.json`: 로컬에서 실제 사용한 패치 버전과 CPU 환경.

서버에 Python3.10 AI 전용 환경이 있으면 해당 Python을 사용한다. 새로 만들 경우 서버에서 확인한 Python3.10 실행 파일로 패키지 밖에 venv를 만든다.

```bash
python3.10 -m venv ../x500-ai-env
../x500-ai-env/bin/python -m pip install -r environment/requirements-common.lock
```

그다음 **서버 드라이버와 사용 가능한 패키지 저장소를 확인하여** CUDA용 torch2.1.2를 해당 격리 환경에 설치한다. 이 단계의 CUDA wheel URL/build는 접속 정보와 서버 inventory를 받은 뒤 확정한다. CPU wheel을 서버 GPU용으로 재사용하지 않는다. 시스템/시뮬레이션 환경은 변경하지 않는다. 이 패키지는 wheelhouse를 포함하지 않으므로 새 환경 설치에는 해당 버전의 패키지 저장소 또는 별도 wheelhouse가 필요하다.

현재 환경이 준비되어 있으면 추가 설치 없이 바로 다음 단계로 진행한다. package source는 wrapper가 경로를 고정하므로 `pip install -e`가 필요 없다. 전체 배포판 버전·실제 import 경로·CUDA/cuDNN·GPU 정보는 실행 결과에 저장된다.

## 3. 같은 작은 시험 실행

```bash
../x500-ai-env/bin/python run_smoke.py --device cuda --out ../gpu-attempt-01
```

실제 격리 환경의 Python 경로로 대체한다. `cuda:0`처럼 장치를 명시할 수도 있다. CUDA 요청 후 사용 불가이면 실패(exit2)하며 CPU로 바꾸지 않는다. `python -O`/`PYTHONOPTIMIZE`도 거절한다. 학습/검증의 assert 검사를 무력화할 수 있기 때문이다. wrapper는 패키지 코드의 import 경로를 검사하고, 하위 프로세스의 `PYTHONPATH`를 고정한다. 실행하는 현재 디렉터리와 무관하게 사용할 수 있다.

재현 설정은 이전과 동일하다: LSTM hidden16, float32,seed17,batch16,lr0.001,history100/common19/context6,future10,train64창/validation64창. 정상hover/horizontal의독립4개run에서32창씩 선택했다. scaler는기존정상train9개run으로fit했으며검증자료로fit하지않았다.

연속4epoch와2epoch저장후**새프로세스**에서2epoch재개를 비교한다. last/best 가중치·optimizer·RNG·epoch·data order·history 및 export가 정확히 같아야 한다(atol=rtol=0). 저장·복원 출력은 같은 장치에서 최대절대오차1e-6 이하. GPU 결과가 CPU 수치와 정확히 같아야 한다고 요구하지는 않는다. GPU에서의 연속/재개 일치를 확인한다. 기준을 완화해서 통과시키지 않는다.

CPU 재현은 같은 명령의 `--device cpu`와 새 출력 경로를 사용한다. CPU에서는 원래v5의 history와 model.safetensors hash 일치까지 추가 검사한다. 다른CPU/라이브러리에서 불일치하면 실패를 보존하고 원인을 분석한다.

## 4. 결과 판정과 다음 단계

- 프로세스 exit0 **그리고** `OUT/result.json`의 `passed:true`, `device:cuda…`, `gpu_status:verified_small_lifecycle`를 확인해야 GPU 시험 통과다.
- `OUT/preflight.json`: 패키지 hash·실제 import 경로·환경/장치.
- `OUT/command.json`, `lifecycle.log`: 실행 argv/cwd/exit code와 로그.
- `OUT/lifecycle/report.json`: 세 프로세스·loss·복원오차·재개 동일성.
- `OUT/lifecycle/{full,resumed}` 및 `interrupted.pt`: 모델·상태·RNG checkpoint.
- `OUT/result.json`: 최종 상태·생성 파일 hash. 실패는 error와 로그를 그대로 보존하고 새 경로에서 재시험한다.

GPU 통과 결과를 확인한 **뒤** 예비 학습 → 본 학습으로 진행한다. 접속 전까지 로컬 개발을 계속할 수 있다. 전체LSTM/Transformer 완성·최종경보보정·독립이상평가를 이 작은 GPU 시험의 선행 조건으로 추가하지 않는다.

## 포함 범위와 기존 증거

`ai/src`와 lifecycle tool은 v5 통과 당시와 byte-identical하다. `input/`은 hash로 고정된 npz/config/provenance이며 미래Y는loss용label이고forward입력은X/C뿐이다. `reference/`에 CPU 준비 완료 판정, CPU 결과·모델, C9/v4 보고서와 계보를 포함했다. 역사적 증거의 원본 절대경로는 그대로 유지했다.

SITL 원시 전체·PX4/시뮬레이터·기존 venv·서버 자격증명은 포함하지 않는다. 서버에서 작은 학습 시험을 재현하는 데 원시 비행 폴더는 필요하지 않다. 원시→전처리 인과성·C9는 이미 검증한 별도 증거이며, 이 패키지의 실행을 새 C9 live 비행으로 보고하지 않는다. GPU 통과도 전체 AI 완료나 탐지 성능/실기체 성능 검증은 아니다.
