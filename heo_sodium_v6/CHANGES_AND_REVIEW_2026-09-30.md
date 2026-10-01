# 변경점 + 총 리뷰 — 2026-09-30 (uMLIP 12종 벤치마크: torch-sim 배치 엔진 · 설치 안정화)

> 대상: 이 벤치마크를 런팟에서 돌리는 사람, 그리고 다음 세션.
> 이 문서 하나로 (1) 오늘 무엇이 왜 바뀌었는지, (2) 무엇이 검증됐고 무엇이 아직 아닌지,
> (3) 팟에서 어떤 순서로 돌리고 무엇을 봐야 하는지를 다룬다.
> 배경 문서: `SESSION_NOTES_2026-09-17_bench.md`, `untitled folder/BENCH_PIPELINE_REVIEW.md`,
> `untitled folder/MODEL_BENCHMARK_SPEC.md`.

---

## 0. 한 문단 요약

목표는 **12개 모델이 런팟에서 전부 설치되고 실행되는 것**이다. 오늘 한 일은 두 가지다.
첫째, 이완 엔진을 ASE(한 번에 1구조)에서 **프로덕션 ORB와 같은 torch-sim 배치 엔진**으로 바꿀 수
있게 12개 모델 전부에 배치 경로를 만들었다. 둘째, 설치가 실패할 수 있던 인프라 문제(디스크,
옛 venv 재사용)를 고쳤다. 배치 경로는 스모크 테스트가 모델별로 검증하고, **검증에 실패한 모델은
자동으로 ASE 엔진으로 돌아가므로 12개 모두 실행은 된다.** 로컬(CPU)에서 실제 모델로 확인한 것은
CHGNet·MACE·Prophet 셋이고, 나머지 9개의 배치 경로는 팟 스모크가 처음 판정한다.

## 1. 왜 바꿨나

| 사실 | 의미 |
|---|---|
| 프로덕션 ORB: 7,073조성(약 8.5만 이완)을 3–4시간 | torch-sim 배치 — 수백 구조를 GPU에서 동시에 |
| 09-17 벤치 CHGNet: top-30 약 240구조에 10시간 가까이 | ASE — 한 번에 1구조, GPU가 대부분 놀고 있음 |
| 벤치 1,089회 = 앵커 128 + top-30 × 32 + Na 1 | ASE로는 느린 모델이 36시간 데드라인을 넘길 수 있음 |
| 이후 계획: 모델별 전수 스크리닝(약 9.5만 이완) | ASE로는 사실상 불가능, torch-sim이면 가능 |

결론: 계산 시간의 관건은 모델이 아니라 **엔진**이다. 벤치와 전수 스크리닝 모두 torch-sim 배치로 간다.

## 2. 파일별 변경점

### `bench_models.py`
- **`build_ts(key, calc)`** 추가: ASE 빌더가 만든 calculator 위에 torch-sim 모델을 얹는다.
- **`TS_PIP`**: 어댑터별 torch-sim 설치 스펙. `torch-sim-atomistic==0.5.2` 고정
  (MatterSim은 자체 의존성으로 ≥0.6이 들어오므로 비움, PET는 `metatomic-torchsim` 추가).
- **직접 작성한 배치 래퍼 3종** (모두 ASE calculator의 가중치·전처리를 그대로 재사용):
  - `OCPBatchModel` — M05 eSEN, M06 EquiformerV3, M09 EquFlashV2. 세 모델 모두 fairchem v1
    `OCPCalculator` 기반이라 `a2g.convert → data_list_collater → trainer.predict`를 여러 구조에 대해 한 번에 호출.
  - `CHGNetBatchModel` — `graph_converter` → `predict_graph`. 에너지는 원자당 값 × 원자 수,
    응력은 GPa × `stress_weight`(eV/Å³) — calculator와 같은 변환.
  - `ProphetBatchModel` — Prophet의 최상위 forward가 원래 배치형(`n_graph`, `cell [G,3,3]`)이라
    구조별 그래프를 엣지 오프셋을 더해 이어붙임.
- **네이티브 6종**은 각 패키지의 torch-sim 클래스 사용: ORB(`torch_sim.models.orb`),
  MACE(`torch_sim.models.mace`), SevenNet(`sevenn.torchsim`), MatterSim(`mattersim.torchsim`),
  TECE(`tace.interface.torchsim`), PET(`metatomic_torchsim` + `upet.get_upet`).
- **M06 venv: python 3.11 → 3.12** (torch-sim이 py≥3.12 요구. 벤더링된 fairchem은 <3.13 허용,
  torch 2.7.1 cu126과 PyG pt27 cp312 휠 존재 확인).

### `bench_anchor_runner.py`
- **`HEO_BENCH_ENGINE = torchsim | ase`** (기본 torchsim).
- **torch-sim 엔진** (`ts_relax`, `ts_relax_items`): 프로덕션 `heo_worker.relax_batch`와 같은 설정
  — FIRE + Frechet 셀 필터, fmax 0.05, 300스텝, InFlightAutoBatcher.
  - 청크(`HEO_TS_CHUNK`, 기본 256구조) 단위로 체크포인트 → 언제 죽어도 이어서 계산.
  - 배치가 실패하면(OOM 등) 예외를 놓고 GPU 메모리를 비운 뒤 **구조 단위로 재시도** (프로덕션 09-04 수정과 같은 방식).
  - **수렴 여부를 실제로 판정**해 `conv`에 기록 (프로덕션은 항상 True로 적었음).
- **체크포인트 태그를 엔진별로 분리**: torch-sim은 `<tag>-ts`, ASE는 `<tag>`. 재개 판정도 태그별.
  시간 파일도 `bench_times_ts.csv`로 분리. → 09-17 ASE 결과와 절대 섞이지 않는다.
- torch-sim 모델 생성에 실패하면 ASE로 내려가고 그 사실을 `anchor_bench.json`의 `warnings`와 `engine`에 남긴다.

### `bench_install.py`
- **캐시를 볼륨으로**: `UV_CACHE_DIR`, `PIP_CACHE_DIR` → `/workspace/heo_bench/.cache`.
  torch 4개 버전의 휠(15–20 GB)이 컨테이너 디스크(기본 20 GB)를 채우는 것을 막는다.
- **옛 venv 자동 재구축**: venv 생성 시 `.heo_venv`(파이썬 버전·격리 여부)를 기록하고, 없거나 다르면
  삭제 후 재생성. 기존 로직은 스모크를 통과한 venv만 스펙 변경을 감지해서, 09-17에 실패했던 venv
  (py3.13 M11, 공유 env M01/M02)가 새 스펙에서도 그대로 재사용될 수 있었다.
- **`ts_pip` 단계**: torch-sim을 ASE 스택 설치 뒤에, 같은 버전 고정(constraints) 아래에서 설치.
  **실패해도 설치 전체는 실패하지 않는다** (해당 모델은 ASE 엔진으로 실행).
- **스모크 테스트 추가 검사**:

| 검사 | 내용 | 실패 시 |
|---|---|---|
| ⑦ 유한차분 | 3% 압축 셀에서 모델이 보고한 응력·힘 vs 자기 에너지의 dE/dε, −dE/dx. 10% 이내 OK, 50% 초과 GROSS | GROSS면 그 모델 불합격 (단위·부호 오류) |
| ⑧ 배치 == ASE | 크기가 다른 셀 3개를 한 배치로 계산, 구조별로 ASE calculator와 비교 (허용: 2e-3 eV/atom, 2e-2 eV/Å, 2e-3 eV/Å³) | `ts_ok=False` → ASE 엔진 |
| ⑨ torch-sim 이완 | 러너의 `ts_relax`를 그대로 호출, 에너지가 내려가는지 | 〃 |
| ⑩ 배치 타이밍 | 8 × 128원자 셀 한 스텝 시간 (`t_ts_128_batch8_s`) | 〃 |

  ⑦이 필요한 이유: 기존 스모크는 응력이 "유한값"인지만 봐서 GPa↔eV/Å³(160배) 오류를 통과시킬 수 있었다.
  셀 이완과 dV/dA/dh가 전부 응력에 달려 있다.

### `heo_bench_runpod_v1.ipynb`
| 셀 | 변경 |
|---|---|
| 0b | `HEO_BENCH_ENGINE = "torchsim"` 추가 |
| 1b | 워치독 스냅샷이 `venvs/`, `.cache/`, `model_ckpts/`를 훑지 않음 (수십만 파일을 매분 stat하던 것 제거) |
| 2 | `ENGINE_OF` — `install_report.json`의 `ts_ok`가 True인 모델만 torchsim, 나머지는 ase. 엔진 표 출력 |
| 4 | `top30_v3.csv`(DFT `build_manifest.py`가 읽는 파일)가 있으면 그 30조성을 우선 사용, 기존 정렬 결과와의 겹침 출력 |
| 5 | 모델별 엔진을 러너에 전달 |
| 7 | `bench_summary.csv`에 `engine` 컬럼 |
| 8, 9 | 문구만 엔진 변경에 맞게 수정 |

원본 노트북 백업: 세션 스크래치패드(`heo_bench_runpod_v1.ipynb.bak`) — 필요하면 프로젝트 폴더로 옮겨 둘 것.

### `models_registry.yml`, `SESSION_NOTES_2026-09-17_bench.md`
- 레지스트리: 엔진 주석 블록 추가, M06 py3.12 반영.
- 세션 노트: §9로 오늘 변경 요약 추가.

## 3. 설계 결정과 근거

1. **torch-sim 0.5.2로 통일.** 0.6.x는 `nvalchemi-toolkit-ops[torch]>=0.4` → torch ≥2.8을 요구해
   eSEN(torch 2.4)·EquiformerV3(2.7.1)에 설치할 수 없다. M09도 공식 requirements가 nvalchemi 0.3.0 고정이라 0.6과 충돌.
   0.5.2에는 ORB 0.5.5용 래퍼와 SevenNet·TACE가 쓰는 `torchsim_nl`이 모두 있다.
2. **MatterSim만 ≥0.6** (패키지가 강제). 0.5.2와 0.6.2의 `optimizers/fire.py`, `cell_filters.py`를 비교한 결과
   알고리즘은 같고, 셀이 발산할 때만 작동하는 클램프(|log F| > 2)가 추가된 정도.
3. **torch-sim 0.3.0(옛 fairchem v1 래퍼 포함)은 쓰지 않는다.** FIRE가 다른 구현이라 옵티마이저 차이가
   모델 차이에 섞인다. 대신 `OCPCalculator` 경로를 직접 배치화했다.
4. **래퍼는 ASE calculator를 재사용한다.** 가중치를 따로 로드하지 않으므로 "ASE로 검증한 그 모델"과 배치 모델이 같다는 것이 구조적으로 보장되고, 스모크 ⑧이 수치로 확인한다.
5. **torch-sim 실패는 치명적이지 않게.** 설치 단계도, 스모크 ⑧–⑩도, 러너의 모델 생성도 실패하면 ASE로 내려간다. 최우선 목표가 "12개 모두 실행"이기 때문.
6. **수렴 기준은 프로덕션과 동일** — 원자 힘만(`include_cell_forces=False`). §7의 미결 사항 참조.
7. **공유 env 모델(M03/M04/M07/M08)에는 torch 고정을 추가하지 않았다.** 09-17에 그 구성으로 성공했으므로 건드리지 않는다.

## 4. 검증 결과

### 4-1. 로컬에서 실제 모델로 확인 (macOS, CPU, torch 2.8 + torch-sim 0.5.2)

| 검사 | CHGNet | MACE-MPA-0 | Prophet-OAME |
|---|---|---|---|
| ⑧ 배치 vs ASE: 에너지 (eV/atom) | 4.8e-7 | 0 | ≤ 4.8e-7 |
| ⑧ 힘 (eV/Å) | 2.3e-5 | 1e-14 | ≤ 1.7e-5 |
| ⑧ 응력 (eV/Å³) | 1.1e-6 | 1e-15 | ≤ 1.5e-6 |
| ⑦ 유한차분 응력 3성분·힘 2성분 | 전부 OK (1% 이내) | 전부 OK (0.2% 이내) | 전부 OK (1% 이내) |
| ⑨ torch-sim FIRE 이완 | 통과, 2/2 수렴 | 통과, 2/2 수렴 | 미실행 (아래) |
| 스모크 ①–⑩ 전체 | 통과 | 통과 | ⑧·⑦만 개별 실행 |

- Prophet은 62M 모델이라 128원자 셀 단계에서 이 Mac(RAM 8.6 GB)이 스와핑으로 멈췄다. 래퍼 정확성의 핵심인 ⑧과 ⑦만 따로 실행해 통과.
- **러너 `main()` 전체** (CHGNet, torch-sim, 앵커 4종 × 8구조 + Na = 33구조, 5스텝 제한, 임시 무작위 SQS):
  이완 33/33·실패 0 → 체크포인트(`bench-chgnet-ts`) → 앵커 집계 → 관문 5개 → XRD(d₀₀₃ 잔차 1.6e-7 Å) → JSON `status=ok`.
  - 재개: 다시 실행 시 "33 cached, 0 to relax".
  - 태그 분리: ASE 엔진으로 실행 시 "0 cached, 33 to relax", 두 태그 행 공존, 집계는 자기 태그만 사용.
  - 이 테스트의 관문·dE 값은 5스텝·무작위 SQS라 물리적 의미 없음 (배선 검증용).
- **설치 흐름** (`bench_install.py --models mock`): `.heo_venv` 없는 옛 venv를 감지해 재구축, 캐시가 `.cache`로 감.
  (mock 스모크 자체는 이 Mac의 scipy 휠 로드 오류로 실패 — 로컬 OS 문제, 리눅스 팟과 무관.)

### 4-2. 소스·메타데이터 대조로 확인 (실행은 안 함)
- 12개 패키지의 버전 호환, 휠 존재(torch cu121/cu126, PyG pt24/27/29, dm-tree cp312), HF·figshare 체크포인트 경로.
- 12개 ASE 로더의 시그니처 (ORB, SevenNet `7net-omni`+`modal='mpa'`, fairchem 1.10 `OCPCalculator`,
  EquiformerV3 `test_discovery.py`와 같은 호출, GGNN `UCalculator`, Prophet, TACE, upet).
- 네이티브 torch-sim 클래스 6종의 생성자 시그니처, torch-sim 0.5.2에 필요한 심볼 존재.

### 4-3. 팟에서만 확인되는 것 (아직 미검증)
- 9개 모델(ORB×2, SevenNet, eSEN, EquiformerV3, MatterSim, EquFlashV2, TECE, PET)의 스모크 ⑧–⑩.
- CUDA에서의 autobatcher 메모리 추정 (CPU에서는 지원되지 않아 건너뜀).
- Prophet의 128원자 단계와 torch-sim 이완.
- torch 2.4(eSEN)에서 torch-sim 0.5.2가 동작하는지 (의존성 선언은 torch≥2).
- 실제 처리량 (모델별 `t_ts_128_batch8_s`).

## 5. 팟 실행 순서와 확인할 것

1. **업로드**: 노트북, `heo_worker.py`, `bench_models.py`, `bench_anchor_runner.py`, `bench_install.py`,
   `xrd_tools.py`, `models_registry.yml` (목록은 이전과 같음).
2. **터미널**: `pkill -f pod_watchdog.py; pkill -f bench_anchor_runner` → `export HF_TOKEN=...`, `export RUNPOD_API_KEY=...`.
3. **셀 0 – 2만 먼저 실행.** 스펙 해시가 전부 바뀌어 venv 12개가 새로 만들어진다 (1–2시간 예상).
4. **셀 2 출력 확인**:
   - `검증 통과 12/12`인가. 실패 모델은 `logs/install_<모델>.log`.
   - 엔진 표: 어떤 모델이 `ase`로 떨어졌고 이유(`ts_error`)는 무엇인가.
   - `ts_diff`(배치 vs ASE 차이)와 `8x128at step` 시간.
   - `results/install_report.json`의 `fd`(유한차분)에 WARN이 있는 모델.
5. **소요 시간 가늠**: torch-sim 모델은 대략 (구조 수 ÷ 8) × 평균 스텝 수 × `t_ts_128_batch8_s`.
   ASE로 떨어진 모델은 1,089 × 평균 스텝 수 × `t_128_s`. 36시간을 넘길 것 같으면 `HEO_MAX_HOURS`를 늘린다.
6. **나머지 셀 실행.** 진행은 `logs/<모델>.log`의 `N/M relaxed (k/n converged, x s/structure amortised)`.
7. **결과**: `bench_summary.csv`의 `status` → `engine` → `gate_*` → `rho_ddE_topN`·verdict 일치율 순으로 읽는다.

### 환경 변수 (오늘 추가된 것)
| 변수 | 기본 | 뜻 |
|---|---|---|
| `HEO_BENCH_ENGINE` | `torchsim` | `ase`로 두면 전 모델 09-17 경로 |
| `HEO_TS_CHUNK` | 256 | optimize 한 번에 넣는 구조 수 = 체크포인트 간격 |
| `HEO_TS_CELL_CONV` | 0 | 1이면 수렴 판정에 셀 힘 포함 |
| `HEO_TS_MAX_ATOMS` | 500000 | autobatcher 메모리 추정이 시도하는 최대 원자 수 |
| `HEO_TS_MEMORY_SCALER` | (자동) | 메모리 추정을 건너뛰고 값을 직접 지정 |

## 6. 총 리뷰 — 이 세션에서 확인한 것 전체

### 6-1. 워크플로우
- 벤치는 **stage 0** (DFT 없이): 관문 0(레지스트리)·관문 1(앵커 방향) + top-30 4-point를 12모델이 재계산해 ORB 기준선과 비교.
- 핵심 숫자는 `rho_ddE_topN`(채점 성분 ddE의 순위 재현), verdict 일치율, MAE_dE.
- 지금 표가 말하는 것은 "ORB와 얼마나 같은가"이고, "누가 더 정확한가"는 DFT 도착 후.
- **기준선은 v4 프로덕션 값이 아니라 벤치 안에서 다시 돌린 orb_mpa**다. 이제 벤치도 torch-sim이라 셀 8은 "같은 엔진, 벤치 vs v4" 일관성 검사가 된다.
- **벤치의 δ = max(σ_vac, 5)** 로, 프로덕션의 `|dE_ORB − dE_MACE|` 항이 없다. 벤치 verdict를 `results_v3.csv`의 verdict와 직접 비교하면 안 된다 (모델 간 비교는 같은 정의라 문제없음).

### 6-2. 설치 리뷰 (모델별)
| 모델 | 판정 | 남은 조건 |
|---|---|---|
| M01/M02 ORB 0.5.5 | 스펙·API 일치 | — |
| M03 MACE, M04 SevenNet-Omni, M07 MatterSim, M08 CHGNet | 09-17과 같은 팟 템플릿이면 재현 (공유 env) | 템플릿을 바꾸면 보장 없음 |
| M05 eSEN | 스펙 일치 | `facebook/OMAT24` 승인 + `HF_TOKEN` |
| M06 EquiformerV3 | 스펙 일치, py3.12로 변경 | 저장소 `main`을 커밋 고정 없이 clone |
| M09 EquFlashV2 | 스펙 일치, figshare API 다운로드 확인 | — |
| M10 Prophet, M11 TECE, M12 PET-OAM | 스펙·API 일치 | — |

### 6-3. GPU
- **Blackwell(RTX 5090, RTX PRO 6000 Blackwell, B200)은 피할 것.** cu121/cu126 빌드에 sm_100/120 커널이 없다.
  `torch.cuda.is_available()`은 True라 스모크 ②는 통과하고 ④에서 실패한다. M05는 torch 2.4 고정이라 구조적으로 불가.
- A100·H100·L40S·RTX 4090 계열은 문제없음. V100·T4는 M09의 cuEquivariance 지원이 불확실.
- GPU 수는 성공 여부가 아니라 소요 시간에만 영향 (모델 단위 라운드로빈).

### 6-4. DFT 비교 준비 상태
- DFT는 (조성, 상, x, **ORB의 k_best**) 한 구조만 계산하고, 시작 구조는 ORB가 이완한 CIF다.
- 벤치는 빈자리 5샘플을 전부 `energies_bench.csv`에 남기므로 DFT의 k로 조인 가능.
- `targets_x4.csv`는 각 모델 자신의 k_best 기준이다. DFT와 부호 대조는 **구조 단위 조인**으로 해야 한다 — 이 분석 스크립트는 아직 없다.
- DFT 앵커는 `HOST_v1`, `HEO_v1`만 (벤치는 v1·v2 모두 계산).

### 6-5. 전수 스크리닝 (이후 단계)
- 7,073조성 × 12구조 + top-500 × 20구조 ≈ **9.5만 이완** = 벤치의 약 87배.
- torch-sim이면 가능 (ORB 실적 3–4시간, 다른 모델은 모델 비용에 비례). 실제 시간은 이번 벤치의 처리량으로 계산한다.
- 미결: 모델별 채점에서 δ의 모델 쌍 항을 어떻게 둘지 (채점 정의 — 승인 필요).

## 7. 미결 사항 (결정 필요)

1. **수렴 기준에 셀 힘을 넣을지.** 지금은 프로덕션과 같게 원자 힘만 본다. 셀이 덜 이완된 채 끝날 수 있어 dV 비교에는
   `HEO_TS_CELL_CONV=1`이 더 엄밀하지만, v4 결과와 조건이 달라진다.
2. **`HEO_MAX_HOURS`**: 기본 36시간 유지 중. 셀 2의 타이밍을 보고 결정.
3. **Ehull**: `HEO_BENCH_EHULL=0` 유지 중 (경쟁상 수백 구조 추가 계산).

## 8. 알려진 한계·리스크

- 로컬 검증은 CPU·소형 셀·3개 모델 한정. **팟 스모크가 최종 판정**이다.
- 직접 작성한 래퍼(OCP·CHGNet·Prophet)는 매 스텝 CPU에서 그래프를 만든다. CHGNet 기준 128원자 구조당 약 23 ms.
  배치가 크면 CPU가 병목이 될 수 있다 (필요하면 `HEO_TS_MAX_ATOMS`로 배치 크기를 제한).
- torch-sim 엔진은 구조별 스텝 수를 내주지 않는다. `bench_times_ts.csv`의 시간은 배치 벽시계를 구조 수로 나눈 값이고 `nsteps = -1`.
  ASE 엔진의 `t_relax_median_s`와 직접 비교하면 안 된다.
- 엔진이 섞이면(일부 모델만 ASE) 모델 간 비교에 경로 차이가 낀다. `engine` 컬럼을 확인하고, 섞였다면 리포트에 명시할 것.
- torch-sim으로 가는 모델은 09-17의 ASE 결과를 재사용하지 않고 처음부터 다시 계산한다 (의도된 동작).
- 문서 `BENCH_PIPELINE_REVIEW.md`의 일부 서술(전 모델 ASE, M09 수동 다운로드, 워치독 terminate=none)은 현재 코드와 다르다. 기준은 이 문서와 `models_registry.yml`.
