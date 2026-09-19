# 출처·버전·라이선스

모든 외부 코드는 공식 PX4/Gazebo 및 공개 라이브러리에서 신규 취득했다.
이전 사용자 프로젝트/저장소를 참조·복사하지 않았다.

| 자료 | 채택 버전/내용 | 출처 / 라이선스 |
|---|---|---|
| PX4 | v1.16.0, 6ea3539157ca358c70a515878b77077af7d4611d | https://github.com/PX4/PX4-Autopilot — BSD-3-Clause |
| Gazebo | Harmonic sim8.15.0, transport13.6.0 | https://gazebosim.org/ — Apache-2.0 |
| X500 geometry | gz models e05f4312d3f28aa621157610584a4870406cb6d3 | pinned PX4 submodule; upstream LICENSE retained |
| CMU eVTOL battery | article14226830 version3, VAH07 | https://doi.org/10.1184/R1/14226830.v3 — CC-BY-4.0, full attribution/metadata in artifacts/sources/cmu-metadata.json |
| CMU paper | Scientific Data dataset description | https://www.nature.com/articles/s41597-023-02180-5 |
| Holybro | AIR2216II KV920/T1045II performance table | https://holybro.com/products/px4-development-kit-x500-v2 — factual table transcribed; no blanket redistribution license asserted |
| Holybro table image | X500MotorSpec.png | https://cdn.shopify.com/s/files/1/0604/5905/7341/files/X500MotorSpec.png?v=1678791632 |
| UIUC | APC SF10x4.7 static kt0835 | https://m-selig.ae.illinois.edu/props/volume-1/data/apcsf_10x4.7_static_kt0835.txt — copyright UIUC, no explicit reuse license identified; different-prop context only |
| Numerical comparator | SciPy solve_ivp DOP853 | .venv version pinned in requirements.lock, BSD license |

CMU original size20,868,474bytes; MD5 `7ba1092696df05426bee98a40ca5fe22`.
Full SHA-256/metadata/download URL is in battery-fit.json. Current sign conversion is −I_mA/1000
(discharge positive); time duplicate71 and largest gap30.058s are retained and reported.
Original CMU CSV/README and UIUC data remain in data/public; derived profiles are distinct.
Manufacturer grams-force use9.80665m/s², RPM uses2π/60. No manufacturer throttle-percent to duty identity is assumed.

실제 파일별 hash는 artifacts/provenance.json, 패치 재적용 도구는 sim/tools/patch_px4.py,
적용된 텍스트 diff는 artifacts/px4-local.patch이다. RawCapture.hpp는 새 파일이므로 diff와 별도로
sim/adapters 원본 및 provenance hash를 함께 사용한다. vendor의 원래 LICENSE 파일을 유지한다.
