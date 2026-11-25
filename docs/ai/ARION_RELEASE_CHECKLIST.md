# ARION V0.2 릴리즈 전 체크리스트

## Windows에서 바로 시험하기

1. 브라우저에서 이 문서를 열고, 빌드에 사용한 GitHub 계정으로 [비공개 검증 저장소의 Actions](https://github.com/suha0303/irion-release-validation/actions)에 로그인한다. 완료한 최신 **ARION desktop release candidates** 실행에서 `arion-windows-x64` 아티팩트를 받는다. 같은 아티팩트에 설치 파일, SHA-256과 합성 시험 로그가 들어간다.
2. 다운로드한 ZIP을 풀고 PowerShell에서 `Get-FileHash .\ARION-*-windows-x64.exe -Algorithm SHA256`을 실행한다. 동봉한 `.sha256`과 값이 같은지 확인한다. 인터넷 연결을 끊고 설치한 뒤 시작 메뉴에서 **ARION**을 연다.
3. **로컬 AI · 기체 관리**에서 기본 V0.2 모델 8종을 확인한다. **로그 모니터링 → 비행 로그 가져오기**에서 동봉한 `release-smoke.ulg`를 연다. 입력 설정은 PX4, 4로터, 4S, 12 Ah, 모터 출력 범위 1000–2000, IMU·배터리·액추에이터 인스턴스 0이다. 모터 채널은 앞오른쪽 1, 앞왼쪽 2, 뒤오른쪽 3, 뒤왼쪽 4를 사용한다. 구간 준비에서 부팅 기준 3–38초를 검사해 저장한 뒤 모델 8종으로 분석한다. 그래프와 비행 후 판정이 뜨는지 확인한다. 합성 로그이므로 표시된 판정으로 탐지 성능을 평가하지 않는다.
4. **실시간 비행 모니터링 → 실시간 연결**에서 MAVLink UDP 주소 `127.0.0.1`, 포트 `14580`, 시스템 ID `1`로 수집을 시작한다. 다음 PowerShell 명령으로 합성 MAVLink 하트비트를 보낸다. 앱에서 수집 건수가 늘어나는지 확인하고 **비행 종료 · 결과 분석**으로 기록을 마감한다. 한 종류의 메시지만 보내므로 이 단계는 수신 확인에 해당한다.

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

## 공개 전 확인 항목

- [ ] **출처 고정.** 빌드에 사용한 Git 커밋과 기본 모델 8종의 가중치·판정 기준 해시를 기록한다. 설치본의 SHA-256을 생성한다.
- [ ] **OS별 실행.** Ubuntu 22.04·24.04 x64, Windows x64, macOS Apple Silicon·Intel에서 설치 또는 이미지 마운트, 첫 실행, 종료를 확인한다.
- [ ] **오프라인 분석.** 대상 OS에서 외부 네트워크를 차단한 상태로 로그 가져오기, 구간 저장, CPU 분석 8종, 그래프와 종료 후 판정을 확인한다.
- [ ] **MAVLink 수신.** 로컬 UDP 수신, 원문 저장, 기록 마감을 확인한다. 실제 기체 연결과 비행 중 성능은 별도 시험으로 기록한다.
- [ ] **기체별 파인튜닝.** 학습·보정·평가 로그를 분리하고, 후보와 전용 보정 기준을 같은 조합으로 선택한다. 기본 모델과 기존 결과가 유지되는지 확인한다.
- [ ] **업데이트와 파일 범위.** 사용자 로그·프로파일·후보가 앱 교체 후 다시 열리는지 확인한다. 배포 파일을 검사하고 모델 가중치·포함 의존성의 재배포 조건을 확인한다.
- [ ] **서명.** Windows 설치본의 서명 상태를 기록한다. 공개 macOS 파일은 Developer ID 서명, 공증과 티켓 첨부, Gatekeeper 검증을 완료한다.
- [ ] **공개 릴리즈.** 검증 결과의 OS·아키텍처·제약을 릴리즈 노트에 적고 같은 버전의 설치 파일과 체크섬을 첨부한다.
