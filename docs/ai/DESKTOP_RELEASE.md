# ARION V0.2 오프라인 데스크톱 릴리즈

## Ubuntu 실행

Ubuntu x64에서는 `ARION-0.2.0-rc.3-ubuntu-amd64.deb`와 같은 이름의 `.sha256` 파일을 사용한다. Ubuntu 데스크톱의 시스템 그래픽 라이브러리를 전제로 하며 패키지에 화면, Chromium, Python, CPU 모델 8종과 해당 판정 기준을 담는다.

```bash
sha256sum -c ARION-0.2.0-rc.3-ubuntu-amd64.deb.sha256
sudo apt install ./ARION-0.2.0-rc.3-ubuntu-amd64.deb
```

앱 메뉴에서 **ARION**을 열면 전용 창이 열린다. 화면에 쓰는 서버는 루프백 주소에서만 열리며 창을 닫으면 앱이 시작한 서버와 수신 작업을 종료한다. Ubuntu의 로그·프로파일·파인튜닝 결과는 `${XDG_DATA_HOME:-$HOME/.local/share}/arion`에 저장한다. 별도 저장 위치를 쓰려면 `ARION_DATA_DIR`를 지정한다.

2026-11-25 Ubuntu 24.04 검증에서 설치 의존성과 패키지 해시를 확인했다. 패키지를 푼 뒤 프로젝트 경로와 외부 네트워크를 가린 환경에서 앱 창, 31번 BIN 검사·프로파일 등록·8종 분석(8,299프레임), 그래프 응답과 MAVLink UDP 원문 기록을 확인했다. 같은 패키지의 CPU 파인튜닝·보정 회귀 시험 5개를 통과했다. Ubuntu 22.04는 휴대용 런타임과 시스템 라이브러리 의존 조건을 맞췄고, 해당 OS에서 설치·실행 검증을 진행한다.

Windows·macOS 검증과 동일한 소스 `a27890a`로 Ubuntu 후보를 다시 빌드했다. 최신 `.deb`와 SHA-256은 `ai/releases/arion-v0.2.0-rc.3/ubuntu-a27890a/`에 있다. 이 노트북의 Ubuntu 24.04.5 x64에서 APT 의존성과 SHA-256을 확인했고, 임시 루트에 `dpkg`로 설치했다. 루프백만 있는 네트워크 공간에서 GNOME 데스크톱 창, 모델 목록 8종, 앱과 백엔드의 정상 종료를 확인했다. 임시 루트의 Electron 창 시험에서는 `--disable-setuid-sandbox`를 사용했다. 합성 PX4 ULog 350프레임과 기존 ArduPilot 31번 BIN 8,299프레임을 각각 8종으로 분석하고 그래프 응답과 MAVLink UDP 원문 저장·종료를 확인했다.
[Ubuntu 설치·오프라인 검증 기록](../../ai/results/new19-20261125/ubuntu-validation/ubuntu-offline.json)

같은 Ubuntu 앱에 PX4 SITL(SIH 쿼드콥터)을 `127.0.0.1:14690`의 MAVLink UDP로 연결했다. 앱의 연결 상태와 PX4 시스템 ID 201·컴포넌트 1을 확인했다. 하트비트, 자세, IMU, 위치, 배터리 등 UDP 데이터그램 130개를 원문으로 저장했고 기록 마감 후 SHA-256이 일치했다. 이 시험은 시동을 해제한 상태에서 실시한 수신 검증이다.

같은 설치본에서 PX4·Gazebo X500 정상 호버 시나리오를 1배속으로 한번 실행했다. 이륙·약 3m 호버·착륙·시동 해제를 확인했다. 앱의 10초 버퍼 비행 중 모델 4종은 각각 717프레임을 판정했고 경보는 0건이었다. 기록 마감 후 전체 비행 모델 4종은 동일한 UDP 원문에서 유효 817프레임을 채점했다. LSTM 2종은 정상 범위, Transformer 2종은 이상 의심으로 표시됐다. 원본 모델과 기준으로 수행한 정상 시뮬레이션 1건의 결과다. [PX4 실행 단계](../../ai/results/new19-20261125/ubuntu-validation/px4-cycle.json)

저장된 프레임으로 전체 비행 판정 점수를 재현했다. Transformer 64×2는 착륙 말미 팩 전압 회복을 최대 113 mV 낮게 예측했다. 마지막 17프레임이 이 모델 점수 제곱의 57.4%를 차지했다. Transformer 128×4는 여러 호버 구간의 전압 과대예측과 착륙 말미 오차가 합쳐져 기준을 넘었다. [모델별 점수와 구간 기여분](../../ai/results/new19-20261125/ubuntu-validation/px4-score-breakdown.json)

## Windows와 macOS 빌드

Windows x64와 macOS Apple Silicon·Intel 각각에서 네이티브 빌드를 실행한다. Windows는 Inno Setup 6을 사용하며 현 Windows 후보의 실시간 수신은 MAVLink UDP를 기준으로 한다. macOS 빌드는 `.app`과 Applications 바로가기를 담은 `.dmg`를 만든다. V0.2.0-rc.3의 Mac DMG는 ad-hoc 서명으로 만든 공증 전 테스트용이다. 일반 배포본에는 Developer ID 서명, 공증 티켓 첨부와 Gatekeeper 시험이 남아 있다. [Apple 공증 안내](https://developer.apple.com/documentation/security/notarizing-macos-software-before-distribution)

공개 저장소의 소스 구조에서는 `UI/frontend`에서 `npm ci --ignore-scripts` 및 `npm run build`를 실행한 후 루트에서 네이티브 빌더를 실행한다. `PYTHONPATH=ai`를 설정하고 다음 인자를 사용한다.

```text
python -B -m localflight.desktop_native --model-source PATH_TO_VERIFIED_MODELS --output build/installers
```

빌더는 호스트용 Python과 Electron을 다운로드하고, 고정한 버전과 적용할 바이너리의 SHA-256을 검사한다. 패키지 사용자가 앱을 실행할 때 이 빌드 단계가 실행되지는 않는다. 모델 입력은 기존 V0.2의 `manifest.json`, `SHA256SUMS`, 8종 각각의 `fit/bundle.json`, `fit/model.safetensors`, `calibration/calibration.json`이다. 패키지에 필요한 모델 파일만 복사하며 학습·검증 원자료와 사용자 폴더는 복사하지 않는다.

GitHub Actions의 **ARION desktop release candidates**는 비공개 검증 저장소에서 수동으로 실행한다. 검증 커밋에 V0.2 소스와 해시가 고정된 `release-inputs/models`를 준비한다. 공개 소스 저장소는 모델 입력 경로를 Git에서 제외한다. Windows x64, macOS Apple Silicon, macOS Intel 환경에서 따로 빌드한다. 설치와 실행, 외부 접속을 차단한 상태의 합성 ULog 8종 분석, 루프백 MAVLink UDP 원문 저장 시험을 통과한 설치본을 SHA-256 및 검증 결과와 함께 Actions 아티팩트에 올린다. [릴리즈 전 체크리스트](ARION_RELEASE_CHECKLIST.md)
