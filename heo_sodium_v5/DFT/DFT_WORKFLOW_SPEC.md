# DFT_WORKFLOW_SPEC — top-30 4-point DFT 검증 (atomate2 + jobflow + FireWorks)

> 실행자: Claude Code. 저장소 루트 CLAUDE.md, CHANGE_SPEC_v4.md, HANDOFF_next_session.md를 먼저 읽을 것.
> 목표: MLIP 스크리닝 결과(top-30 + 앵커 2종)를 DFT(VASP, MP 화폐)로 검증하고,
> 같은 계산에서 자화 감사·연화 진단·파인튜닝 라벨을 동시에 확보한다.

---

## 0. 이 계산이 답하는 질문 4개 (설계의 근거)

1. **랭킹 검증** — MLIP의 O3/P3 부호·순서가 DFT 화폐에서도 성립하는가
2. **전하 회계 검증** — 1단계 예측(n_Ni3, n_Mn3 …)이 사이트별 자화로 확인되는가
3. **연화 진단 (F0)** — ORB 연화계수 c, 오차 유형(기울기형/산포형)
4. **파인튜닝 라벨 (F1 대비)** — 이완 궤적의 모든 ionic step

→ ISPIN=2 / LORBIT=11 없는 계산은 질문 2를 못 하므로 **무효**로 취급.

## 1. 범위

| 집합 | 조성 수 | 구조 수 | 비고 |
|---|---|---|---|
| top-30 (v4 results csv, score 오름차순) | 30 | 30 × 4x × 2상 = 240 | x = 1/0.81/0.67/0.52 |
| 앵커 | 2 (HOST, HEO) | 2 × 8 = 16 | **Ti0.3 제외** (사용자 결정) |
| E_Na (bcc Na) | — | 1 | 같은 설정의 DFT 참조 (MLIP E_Na 혼용 금지) |
| 파일럿 | top-1 | 4 (std) + 4 (gam) | §4 |
| **합계** | | **256 + 1 + 8** | |

빈자리가 있는 x(0.81/0.67/0.52)는 MLIP이 고른 대표 샘플 1개(`k_best`, `k_best_x081`, `k_best_x067`)만 계산.
→ 한계 문단: "대표 빈자리 배열 선택은 MLIP 화폐에서 이뤄짐".

## 2. 환경 (한 번만)

```bash
pip install atomate2 jobflow fireworks custodian pymatgen
```

설정 파일 (`~/atomate2_config/`) — 자리표시자는 파일럿 때 연구실 서버 값으로 채움:

**my_launchpad.yaml** — MongoDB 접속. Mongo 불가 시 jobflow `JobStore` 로컬 파일 모드로 대체(§8 미검증 #1).

**my_qadapter.yaml** (핵심 — "노드 상주 일꾼" 방식)
```yaml
_fw_name: CommonAdapter
_fw_q_type: SLURM
queue: 56core                     # 기본 파티션 (36core용 qadapter를 두 번째로 둘 수 있음)
nodes: 1
ntasks_per_node: 56
walltime: <PARTITION_MAX_WALLTIME>  # 예: 3-00:00:00
job_name: heo_dft
rocket_launch: rlaunch -c ~/atomate2_config rapidfire --nlaunches 0 --timeout <WALLTIME_SEC_MINUS_10PCT> --sleep 30
pre_rocket: |
  module load <VASP_MODULE>
  source <CONDA_PATH>/bin/activate atomate2
```
- `rapidfire --nlaunches 0`: 한 sbatch 작업이 노드를 잡은 채 계산을 순차로 계속 뽑아 실행
- `--timeout`: walltime보다 짧게 → 끝낼 수 없는 계산은 시작하지 않고 종료

**my_fworker.yaml**
```yaml
name: labserver
category: ''
query: '{}'
env:
  vasp_cmd: srun -n 56 <VASP_STD_PATH>      # 또는 mpirun -np 56
  vasp_gamma_cmd: srun -n 56 <VASP_GAM_PATH>
```

**atomate2.yaml** / **FW_config.yaml**: 위 파일 경로 지정, `VASP_CMD`/`VASP_GAMMA_CMD` 매핑, POTCAR 디렉토리(`PMG_VASP_PSP_DIR`, MP 기본 POTCAR 세트).

## 3. STEP 1 — 구조 추출: `build_manifest.py`

**원칙: 새로 생성하지 않는다. MLIP 캐시에서만 꺼낸다.** (새로 만들면 SQS 배열·빈자리 패턴이 달라져 MLIP–DFT 비교가 무의미)

입력: v4 results csv(top-30 선정용), MLIP 캐시(HEO_WORKDIR), 앵커 캐시(CELL 9 산출물).
캐시 키: `(comp_id, phase, na_count, sample_idx, MODEL_TAG)` — 불변량 10 그대로.

절차:
1. results csv에서 score 오름차순 top-30의 comp_id, 각 x의 k_best 읽기
2. 앵커 2종(HOST, HEO)의 8구조를 앵커 캐시에서 읽기 (워커 CELL 9의 정의 그대로 — 조성을 재입력하지 말 것)
3. 각 구조 검산 (실패 시 abort, 목록 출력):
   - Na 개수 ∈ {27, 22, 18, 14}이고 x_tag와 일치
   - 글자 분류기(stacking_report)가 phase 태그(O3/P3)와 일치
   - 구조 해시 기록 (CONTCAR 대조용)
4. `dft/{comp_id}/{phase}/{x_tag}/POSCAR` 저장 + `manifest.csv` 기록

**manifest.csv 컬럼** (추적의 단일 원천 — 이 파일 외에 상태를 기록하지 않는다):
`row_id, set(top30|anchor|pilot|E_Na), comp_id, phase, x_tag, na_count, k_best, struct_hash, mlip_E, mlip_dE, priority, status(pending|queued|running|done|fizzled|manual), fw_id, task_id, note`

## 4. 파일럿 (256 던지기 전 필수 — 통과 없이 본 실행 금지)

top-1 조성의 O3/P3 × x100/x052 = 4 계산을 `vasp_std`와 `vasp_gam`으로 각각 → 8 계산.
확정할 것:
- [ ] MAGMOM 초기값이 회계대로 수렴하는가 (사이트별 자화 판독 → n_Ni3 대조)
- [ ] 1계산당 벽시계 시간 T (→ 총 소요 ≈ 256·T/5)
- [ ] gam vs std: |ΔdE| < 5 meV/f.u. **이고** 힘 MAE < 0.02 eV/Å 이면 Γ-only 채택 (F1 라벨 품질 때문에 힘까지 확인)
- [ ] custodian 복구 동작 확인 (의도적 NELM 축소 케이스 1개)
- [ ] TaskDoc에 `output.magnetization`(사이트별), `calcs_reversed[].output.ionic_steps`가 실제로 채워지는지

## 5. STEP 2 — 워크플로: `build_workflows.py`

atomate2 `MPGGARelaxMaker → MPGGAStaticMaker` 2단 (MP 화폐: MPRelaxSet/MPStaticSet 기본값 유지).

**MP 기본값에서 손대는 것만** (`user_incar_settings`):
```python
{
  "ISPIN": 2, "LORBIT": 11,          # 사이트별 자화 — 필수
  "ISIF": 3,                          # 셀 이완 (dV 비교)
  "EDIFF": 1e-5, "EDIFFG": -0.02, "NSW": 99, "IBRION": 2,
  "LREAL": "Auto", "NCORE": 4, "LWAVE": False, "LCHARG": False,
  # LDAU / U값 / ENCUT / POTCAR: MP 기본값 그대로 — 화폐 통일. 절대 override 금지
}
```
**MAGMOM 초기값** — 회계(n_Ni3, n_Mn3, n_Co2, n_V3, n_V5)와 정합하게 사이트별 부여.
출발값(파일럿에서 조정): Ni²⁺ 2.0 / Ni³⁺ 1.0 / Ni⁴⁺ 0.0 / Mn⁴⁺ 3.0 / Mn³⁺ 4.0 /
Co³⁺ 0.0(저스핀 가정) / Co²⁺ 3.0 / V³⁺ 2.0 / V⁴⁺ 1.0 / V⁵⁺ 0.0 / 그 외(Li, Mg, Al, Ti⁴⁺, Sb⁵⁺, Sn⁴⁺, Cu²⁺→1.0, Zn 0, Fe³⁺ 5.0 …) 0 또는 표기값.
어느 Ni가 Ni³⁺인지는 회계가 정하지 않으므로: 초기값은 평균값(예: Ni 사이트 전부 (2·n_Ni2 + 1·n_Ni3)/n_Ni)으로 주고 SCF가 결정하게 한다 — 감사는 결과 자화로.

k-point: MP 밀도 기본. 파일럿 통과 시 `vasp_gam` + Γ KPOINTS로 전환.

**메타데이터** (STEP 4 질의 열쇠): `metadata={"row_id", "set", "comp_id", "phase", "x_tag", "na_count", "k_best", "struct_hash"}`
**우선순위**: `_priority`: x100/x052 = 10, x081/x067 = 5, 앵커 = 20 (앵커·랭킹 검증 먼저).

E_Na: bcc Na 2원자 셀, 같은 Maker·같은 설정(ISPIN=2 유지, 자화 0 수렴 확인). 이 값만 V_seg/V_avg에 사용.

## 6. STEP 3 — 제출·실행

```bash
python build_workflows.py --manifest manifest.csv --set anchor,top30   # lpad add
nohup qlaunch rapidfire -m 5 --nlaunches infinite --sleep 300 > qlaunch.log 2>&1 &
```
- `-m 5`: 내 작업이 큐에 최대 5개 (실행+대기) — 노드 5개 상주 일꾼. 상황 따라 1~5 조정(실행 중 변경 가능)
- 모니터: `lpad get_wflows -s FIZZLED -d more`, `lpad get_fws -s RUNNING`
- 회수: 크론 또는 수동으로 `lpad detect_lostruns --rerun` (walltime에 잘린 계산 재제출)
- FIZZLED 재제출: `lpad rerun_fws -s FIZZLED` (custodian 복구 실패분). 2회 실패 시 manifest status=manual 표기 후 사람 확인
- 완료마다 manifest status 갱신 스크립트(`sync_manifest.py`): LaunchPad 상태 → manifest

## 7. STEP 4/5 — 수집·감사: `collect.py`, `audit.py`

**collect.py**: 메타데이터로 TaskDoc 257건 질의 → `raw/{row_id}.json` + `dft_raw.csv`
(energy, structure.lattice.matrix, forces, stress, magnetization per site, converged flags, incar 요약, ionic_steps 수)

**audit.py** (파생량 + 판정 — 우리 물리):

| 감사 | 계산 | 출력·판정 |
|---|---|---|
| 수렴/상 | converged 플래그, CONTCAR에 글자 분류기 재판독(n_phase_flip) | 미수렴·상 뒤집힘 → 재작업 큐 |
| 부호 일치 | dE_DFT(x) = (E_P3 − E_O3)/27 vs mlip_dE 부호 | 조성별·x별 일치 표, 일치율 |
| 전압 | V_seg (Δn = 5,4,4; DFT E_Na), V_avg; Σ(Δn·V_seg)/13 ≈ V_avg 검산 | 전부 > 0; HOST V_avg ≈ 3.1 V (창 2.5–3.4) |
| 부피 | `structure.lattice.matrix`로 det/외적 → dV, dA, dh_perp, vm_strain (워커의 dV_report 재사용) | 앵커 dA<0/dh>0 문법, MLIP 대비 상관 |
| 자화 | 사이트별 μB → 산화수 (Ni²⁺≈1.7 / Ni³⁺≈0.9 / Ni⁴⁺≈0; Mn⁴⁺≈3 / Mn³⁺≈3.8; 문턱은 파일럿에서 확정) → n_Ni3/n_Mn3 집계 | 1단계 예측과 대조 (반증 시험) |
| 연화 (F0) | ionic_steps 프레임 → `frames.extxyz` 내보내기 → **GPU 머신에서 ORB static** (고정 기하, 이완 금지) → 힘 산점 기울기 c, cMAE | 기울기형(균일) vs 산포형; 부호 불일치와 교차표 |
| 앵커 GATE 2 (DFT판) | HOST vs HEO의 dE_x067, x* 비교 | 차이가 σ 이상인가 → Zhao 서사 정합 판정 |

**산출물**: `dft_verification.csv` (한 행 = 한 조성; 컬럼은 `{양}_mlip, {양}_dft, sign_ok_{x}` 쌍),
`report_dft.md` (게이트 pass/fail, 불일치 목록, 파인튜닝 발동 판정), 그림 **.pptx** (부호 패리티, 자화 히스토그램, 연화 산점, 앵커 V(x) 계단), `frames.extxyz` (F0/F1용).

**판정 규칙**: top-30 중 x052 부호 일치 ≥ 27 **및** c 기울기 균일 → 랭킹 확정, c 기록, 파인튜닝 없음.
그 외 → F1 파인튜닝 발동 (STEP 1 목록만 학습용으로 교체, STEP 2–4 재사용. 앵커는 학습 영구 제외).

## 8. 금지 조항

1. 구조 신규 생성 금지 — 캐시에서만
2. U / ENCUT / POTCAR 변경 금지 — 화폐
3. ISPIN=2 / LORBIT=11 누락 계산 무효
4. 연화 측정에 이완 혼입 금지 — 고정 기하 static끼리만
5. MLIP E_Na 혼용 금지 — DFT E_Na만
6. manifest 외 상태 기록 금지
7. 채점 정의(SCORE_TERMS/관문/δ) 수정 금지 — 발견은 보고만

## 9. 인수 기준

- [ ] 파일럿 8계산 §4 체크리스트 전부 통과, T 기록, gam/std 결정 기록
- [ ] manifest 257행 status 전부 done (manual 행은 사유 기록)
- [ ] dft_verification.csv + report_dft.md + pptx 생성
- [ ] 부호 일치율·자화 감사·c 값이 report에 명시, 파인튜닝 발동 여부 판정 문장 포함
- [ ] frames.extxyz 생성 (프레임 수, 조성 커버리지 기록)

## 10. 미검증 지점 (작업 중 확인 후 체크)

- [ ] #1 연구실 서버 MongoDB 가용 여부 → 불가 시 jobflow 로컬 JobStore 전환
- [ ] #2 atomate2 Maker 클래스명(`MPGGARelaxMaker`/`MPGGAStaticMaker`) 및 `user_incar_settings` 전달 경로 — 설치 버전 문서 대조
- [ ] #3 TaskDoc에서 사이트별 자화·ionic_steps 필드 경로 (파일럿에서 실물 확인)
- [ ] #4 앵커 캐시의 키 형식이 스크리닝 캐시와 동일한지
- [ ] #5 `rlaunch rapidfire --timeout` 동작 (walltime 직전 신규 계산 미시작 확인)
- [ ] #6 파티션별 최대 walltime, 56core 노드 메모리 (108원자 ISPIN=2 GGA+U 수용 여부)
