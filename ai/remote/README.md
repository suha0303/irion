# 원격 환경·자원 검증

서버 주소·사용자·포트·클론 경로·기존 Python은 `./ai/drone-ai configure`로 직접 입력한다. 소스의 고정 SSH 별칭이나 계정 경로를 선택하지 않는다. SSH 정보를 설정한 로컬 프로필의 `./ai/drone-ai connect --profile default`는 호스트 키 검사를 유지한 SSH와 서버 tmux를 연다. 비밀번호는 터미널에 입력하고 프로필에 저장하지 않는다.

현재 실제 GPU 승인은 프로젝트 `AGENTS.md`의 p02 / DIS01 / 단일 노드 / GPU1개로 제한한다. 서버 입력 기능이 다른 자원 사용을 승인하지는 않는다. 로그인 노드에서 CUDA 계산을 하지 않으며 Slurm의 `CUDA_VISIBLE_DEVICES`를 변경하지 않는다.

`gpu_probe.py`는 실제 할당 소유자·job ID·노드·GPU1개와 선택한 Python/PyTorch를 검사한다. 지원 런타임은 Python3.10의 `torch21` 또는 명시 승인된 `mamba25`다. `environment-mamba25.json`의 과거 관측값은 보존 증거이며 새 서버 경로의 기본값이 아니다.

`setup`·`doctor`는 기존 환경을 검사하며 설치·업그레이드하지 않는다. `build_probe_package.py`와 `allocate_probe.sh`는 기존 패키지 회귀검증용이다. 새 사용자는 [drone-ai 사용 안내](../../docs/ai/PORTABLE_SETUP.md)를 따른다.
