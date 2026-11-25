# ARION V0.2 릴리즈 전 체크리스트

## Windows에서 바로 시험하기

1. `suha0303` GitHub 계정으로 로그인하고 `arion-windows-x64` ZIP을 받는다. Git이나 Python 설치 없이 시험할 수 있다. [Windows x64 검증본 다운로드](https://github.com/suha0303/irion-release-validation/actions/runs/45615492341/artifacts/15022692378)
2. ZIP을 푼 폴더에서 PowerShell을 열고 `Get-FileHash .\installers\ARION-*-windows-x64.exe -Algorithm SHA256`을 실행한다. 같은 폴더의 `.sha256`과 값이 같은지 확인한다. 인터넷 연결을 끊고 EXE로 설치한 뒤 시작 메뉴에서 **ARION**을 연다.
3. **로그 모니터링 → 비행 로그 가져오기**에서 `verification\release-smoke.ulg`를 연다. **로컬 AI · 기체 관리 → 로그 준비**에서 이 로그로 프로파일을 만든다. PX4, 4로터, 4S, 12 Ah, 모터 출력 범위 1000–2000, IMU·배터리·액추에이터 인스턴스 0을 사용한다. 모터 채널은 앞오른쪽 1, 앞왼쪽 2, 뒤오른쪽 3, 뒤왼쪽 4다. 같은 폴더의 `release-smoke-profile.json`에 입력값이 들어 있다.
4. **로그 준비**에서 해당 프로파일을 선택하고 시작 3초, 종료 38초를 넣어 **사용 가능 구간 찾기**를 누른다. 사용 가능 구간을 선택하고 근거에 `합성 시험 로그`를 적는다. 용도를 **독립 평가·예측**, 건강 상태를 **모름**으로 두고 **선택 구간 저장**을 누른다. **로그 분석**에서 저장한 로그와 **전체 8종 일괄**을 선택해 실행한다. **로그 모니터링**에서 그래프와 비행 후 판정이 뜨는지 확인한다.
5. **실시간 비행 모니터링 → 실시간 연결**에서 **MAVLink UDP**, 주소 `127.0.0.1`, 포트 `14580`, 시스템 ID `1`로 **수집 시작**을 누른다. 아래 PowerShell 명령으로 합성 MAVLink 하트비트를 보낸다. 수집 건수가 늘어나는지 확인하고 **비행 종료 · 결과 분석**으로 기록을 마감한다. 이 절차는 로그 분석과 UDP 수신을 확인하는 기능 시험이다.

```powershell
$packet = [Convert]::FromBase64String('/QkAAAABAQAAAAAAAAACDAADA7Mx')
$udp = [Net.Sockets.UdpClient]::new()
try {
  1..20 | ForEach-Object {
    [void]$udp.Send($packet, $packet.Length, '127.0.0.1', 14580)
    Start-Sleep -Milliseconds 200
  }
} finally {
  $udp.Dispose()
}
```

## Actions 검증 기록

세 후보는 최신 V0.2 소스 `a27890a`와 해시로 확인한 기본 모델 8종을 사용했다. 각 ZIP에 설치본, 설치본 SHA-256, 합성 시험 로그, 프로파일 입력값, `release-check.json`을 담았다. [빌드·검증 실행](https://github.com/suha0303/irion-release-validation/actions/runs/45615492341)

| 대상 | 확인 항목 | 결과 |
| --- | --- | --- |
| Windows Server 2025 x64 | EXE 설치·앱 실행, 오프라인 ULog 8종·350프레임, MAVLink UDP 원문 저장 | [통과 · EXE 아티팩트](https://github.com/suha0303/irion-release-validation/actions/runs/45615492341/artifacts/15022692378) |
| macOS 15 Apple Silicon | DMG 마운트·앱 복사·실행, 오프라인 ULog 8종·350프레임, MAVLink UDP 원문 저장 | [통과 · DMG 아티팩트](https://github.com/suha0303/irion-release-validation/actions/runs/45615492341/artifacts/15021449575) |
| macOS 15 Intel | DMG 마운트·앱 복사·실행, 오프라인 ULog 8종·350프레임, MAVLink UDP 원문 저장 | [통과 · DMG 아티팩트](https://github.com/suha0303/irion-release-validation/actions/runs/45615492341/artifacts/15022431264) |

Windows 후보는 서명 전 EXE이고 macOS 후보는 내부 검증용 ad-hoc 서명을 사용한다. 오프라인 시험은 외부 접속을 차단한 내장 백엔드에서 합성 ULog와 루프백 MAVLink로 수행했다. 사용자 PC의 첫 실행, 외부 장치 연결, 실제 비행 탐지 성능은 별도로 확인한다.

## Ubuntu 노트북 검증 기록

- 같은 검증 커밋의 Ubuntu 24.04.5 x64 후보에서 임시 루트 설치와 실제 앱 창을 확인했다. 외부 접속 없이 합성 PX4 ULog 350프레임과 실기체 ArduPilot 31번 BIN 8,299프레임을 8종으로 분석했다. [설치·오프라인 검증](../../ai/results/new19-20261125/ubuntu-validation/ubuntu-offline.json)
- PX4 SITL X500 정상 호버의 이륙·착륙과 MAVLink 원문 마감을 확인했다. 비행 중 예측용 4종은 각각 717프레임에 경보 0건이었다. 비행 후 4종은 817프레임을 사용했다. LSTM 2종이 정상 범위, Transformer 2종이 이상 의심을 표시했다. [비행 결과](../../ai/results/new19-20261125/ubuntu-validation/px4-cycle.json)
- 같은 CUAVv5 제어기의 로그를 날짜별로 분리해 LSTM 128×4의 두 후보를 만들었다. 31번 평가 로그에서 전용 기준까지 적용한 비행 중 모델의 오경보는 0%였다. 비행 후 모델은 정상 로그에 이상 의심이 남아 두 후보를 검증 대기로 보존했다. [후보 재분석](../../ai/results/new19-20261125/ubuntu-validation/cuavv5-comparison.json)

## 공개 전 확인 항목

- [x] **출처 고정.** 후보 커밋과 기본 모델 8종의 가중치·판정 기준 해시를 기록했다. 설치본마다 SHA-256을 포함했다.
- [ ] **OS별 실행.** Ubuntu 22.04·24.04 x64, Windows x64, macOS Apple Silicon·Intel에서 설치 또는 이미지 마운트, 첫 실행, 종료를 확인한다.
- [ ] **오프라인 분석.** 대상 OS에서 외부 네트워크를 차단한 상태로 로그 가져오기, 구간 저장, CPU 분석 8종, 그래프와 종료 후 판정을 확인한다.
- [ ] **MAVLink 수신.** 로컬 UDP 수신, 원문 저장, 기록 마감을 확인한다. 실제 기체 연결과 비행 중 성능은 별도 시험으로 기록한다.
- [ ] **기체별 파인튜닝.** 학습·보정·평가 로그를 분리하고, 후보와 전용 보정 기준을 같은 조합으로 선택한다. 기본 모델과 기존 결과가 유지되는지 확인한다.
- [ ] **업데이트와 파일 범위.** 사용자 로그·프로파일·후보가 앱 교체 후 다시 열리는지 확인한다. 배포 파일을 검사하고 모델 가중치·포함 의존성의 재배포 조건을 확인한다.
- [ ] **일반 배포용 서명.** Windows 설치본의 서명 상태를 기록한다. 일반 배포용 macOS 파일은 Developer ID 서명, 공증과 티켓 첨부, Gatekeeper 검증을 완료한다. V0.2.0-rc.3에 첨부한 macOS 파일은 공증 전 테스트용이다.
- [ ] **공개 릴리즈.** 검증 결과의 OS·아키텍처·제약을 릴리즈 노트에 적고 같은 버전의 설치 파일과 체크섬을 첨부한다.
