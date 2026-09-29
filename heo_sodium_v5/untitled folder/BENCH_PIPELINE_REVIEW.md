# BENCH_PIPELINE_REVIEW — uMLIP 12종 벤치마크 + XRD 파이프라인 전체 리뷰 (2026-09-17)

> 대상 독자: HEO Na 층상 양극재 스크리닝 프로젝트 참여자.
> 이 문서 하나로 (1) 무엇이 새로 생겼는지, (2) 어떻게 돌리는지, (3) 결과를 어떻게 읽는지를 커버한다.
> 근거 스펙: MODEL_BENCHMARK_SPEC.md, XRD_SPEC.md. 기반 코드: heo_worker.py(v4), heo_na_runpod_v4.ipynb.

---

## 1. 왜 이 파이프라인인가 (한 문단)

주모델 ORB-v3 mpa 선정 근거를 "리더보드 F1"이 아니라 **우리 관측량(dE 부호·x*·전압)에 대한 직접
증거**로 바꾸는 것이 목표다. 정답지인 DFT(CORE 256 이완 + FRAMES)가 아직 없으므로, 지금 가능한
것부터 돌린다: **실험 사실 2개**(HOST는 O3→P3 전이 + V≈3.1 V — Komaba 2012 / HEO는 전이 지연 —
Zhao 2020)를 12개 uMLIP가 재현하는지(관문 1), 그리고 12모델이 서로 얼마나 다른 그림을 그리는지.
Ti03 앵커는 영구 제외(Wang 2017과 모순, 2026-09-03 결정). 절대 오차(MAE)·최종 우열 판정은 DFT
도착 후 같은 산출물로 일괄 수행한다.

## 2. 새로 생긴 파일 (Sodium_mlip_v1/)

| 파일 | 역할 |
|---|---|
| `bench_models.py` | 12모델 + mock의 ASE calculator 어댑터. 설치 메타(pip, 격리 여부)와 빌더가 1:1. 실패 시 AdapterError로 "무엇을 시도했는지"를 남긴다 |
| `bench_anchor_runner.py` | **모델 1개분 벤치마크 전 과정** (CLI `--adapter <key>`). v4 노트북 셀 5(앵커 게이트)+5d(x4 파일럿)의 계산을 그대로 수행하되, 관문 assert를 **기록**으로 바꿈 — 모델이 관문에 떨어지는 것 자체가 벤치마크 데이터 |
| `xrd_tools.py` | XRD_SPEC 구현: 점유율 평균 → CuKa 패턴 → (003)/(110)/(104) 추적 → d₀₀₃=h_perp/3 검산 → O3/P3 지문 판독 → HOST 방향 시험. 순수 후처리 (원자·격자 수정 0, 이완 재실행 0) |
| `models_registry.yml` | 관문 0 레지스트리: 체크포인트·화폐·라이선스·격리 필요 여부, verified/unverified/blocked 정직 표기 |
| `heo_bench_runpod_v1.ipynb` | 런팟 실행 노트북 (v4 관례 그대로: CONFIG→워치독→INSTALL→SQS→런처→집계→XRD→리포트) |
| `heo_worker.py` | heo_worker-3.py의 이름 변경 사본 (러너·노트북이 import) |

**기존 파일은 하나도 수정하지 않았다.** 스크리닝 workdir(heo_v2/heo_v4)에는 아무것도 쓰지 않는다.

## 3. 데이터 흐름

```
/workspace/heo_v2/structures/sqs (v3 캐시, 읽기전용)
        │ 복사 (셀 4; 없으면 icet로 생성)
        ▼
/workspace/heo_bench/shared_base/structures/sqs   ← 12모델 공통 시작구조 (스펙 §10-1)
        │
        ▼  런처(셀 5): 모델별 서브프로세스, GPU 라운드로빈
/workspace/heo_bench/runs/<adapter>/              ← 모델별 격리 workdir (invariant 6)
   checkpoints/energies_bench.csv                 ← CKPT_FIELDS 계약 + (comp_id,tag) 재개
   checkpoints/bench_times.csv                    ← 구조당 시간·스텝
   structures/relaxed/*.cif                       ← XRD 입력
   results/anchor_bench.json                      ← 관문·진단·XRD·속도 요약 (모델 1개)
   results/anchors_x4{,_raw,_dV,_xrd}.csv
        │
        ▼  집계(셀 7)
/workspace/heo_bench/results/bench_summary.csv    ← 모델 × 전 지표 (최종 표)
/workspace/heo_bench/results/benchmark_report_stage0.md
/workspace/heo_bench/xrd_v4/results_xrd_v4.csv    ← 기존 v4 결과에 XRD 컬럼 (셀 8b)
/workspace/heo_bench/results/xrd_waterfall.pptx   ← 앵커 2종 + top-5 (셀 8c)
```

계산량: 모델당 129 이완(앵커 4변형 × 32구조 + bcc Na) + **Ehull 확장**(기본 켜짐): top-30
프리스틴 쌍 60구조 + 경쟁상 재이완(개수는 hull 캐시·원소 풀에 따라 수백). 전 모델 동일 조건 —
ASE FIRE + FrechetCellFilter, fmax 0.05 eV/Å, max_steps 300, 모델 고유 cutoff는 기본값(모델의 일부).

**top-N 4-point 비교 (2026-09-17 확정, 벤치마크의 핵심 질문)**: results_v3.csv의 top-N(기본 30,
`HEO_BENCH_TOPN`) 조성을 **각 모델이 프로덕션과 동일한 4-point 전체(32구조/조성)로 계산** →
모델별 `targets_x4.csv`(조성 × dE(x)·verdict·ddE·x*·V_seg·Ef·dV 분해). 교차 집계 컬럼:
`rho_ddE_topN`(**프로덕션 채점 성분인 ddE의 순위 재현** — 이 표의 가장 중요한 숫자),
`agree_verdict_desod/x067`(상 판정 일치율), `MAE_dE_desod/x067_vs_M01`.
`HEO_BENCH_TARGET_WHICH=pris`로 프리스틴 쌍만으로 축소 가능(Ehull·dE_pris만 나옴).

**Ehull 확장 (2026-09-17 추가, 사용자 결정)**: v4 셀 8의 정의를 그대로 재사용 — 핵심은 hull이
**양쪽 다 in-model**이라는 것. 각 모델이 (a) 타깃 화합물과 (b) MP 경쟁상 구조(공유 캐시,
`shared_base/hull_cache/competitors.json`)를 **자기 에너지로** 이완해 자기 hull을 짓는다. 서로 다른
화폐의 에너지가 한 hull에서 만나는 일이 없다. LP hull(v4에서 pymatgen 대비 1e-6 검증)로
Ehull → Ehull_eff = Ehull − kT·S_conf (T=1173 K), T* 계산. 켜고 끄기: `HEO_BENCH_TOPN`(기본 30),
`HEO_BENCH_EHULL=0`으로 비활성. 덤: top-30 프리스틴 쌍 덕에 **dE_pris 모델 비교가 n=8 → n=30**으로 커진다.

## 4. 실행 방법 (RunPod)

1. PyTorch CUDA 이미지 팟 + Network Volume(`/workspace/heo_v2`, `/workspace/heo_v4`) 마운트.
2. 위 표의 6개 파일을 한 폴더에 업로드.
3. 터미널에서 필요한 env export (노트북에 키를 쓰지 말 것):
   - `HF_TOKEN` — eSEN이 gated면 필요
   - `HEO_EQUFLASH_CKPT` — M09는 Figshare 수동 다운로드 후 경로 지정
   - `HEO_BENCH_MODELS` — 스모크는 `orb_mpa`, 전체는 빈 값
   - `HEO_XRD_SCOPE` — `pilot`(HOST만) / `top30`(기본) / `top500`
4. **스모크 먼저**: `HEO_BENCH_MODELS=orb_mpa`로 Run All → 셀 8에서 |ΔdE| < 5 meV/f.u. 확인.
5. 전체 실행은 연결 끊김에 안전한 터미널 경로 권장:
   `nohup jupyter nbconvert --to notebook --execute --inplace heo_bench_runpod_v1.ipynb > nbrun.log 2>&1 &`
6. 종료 후 `results/` 회수, 팟 수동 종료(워치독은 terminate=none — 로그만 남김).

로컬 검증 완료(2026-09-17): mock 어댑터로 129구조 이완→집계→XRD→JSON 전 과정 통과,
재개(129 cached) 확인, d₀₀₃ 검산 잔차 1.5e-7 Å(기계 정밀도), 관문·방향시험 전부 산출.

## 5. 결과 해석 가이드라인

### 5-1. 부호·방향 규약 (오독 최다 지점 — HANDOFF §1과 동일)

- **dE = [E(P3) − E(O3)]/27 [meV/f.u.]. dE > 0 = O3 승리(좋음), dE < 0 = P3(gliding).**
- gap_* = HEO − HOST. **클수록(양수일수록) "HEO가 O3를 더 오래 버틴다" = 지연 방향.**
- 판정: dE > +δ → O3, dE < −δ → P3, |dE| ≤ δ → AMB. δ = max(σ_vac, 5.0), x별 독립.
- x 태그: pris/x081/x067/desod = Na 27/22/18/14 = x 1.0/0.815/0.667/0.519.

### 5-2. `bench_summary.csv` 읽는 순서

**1단계 — status부터.** `ok`가 아닌 행(adapter_error/no_output 등)은 그 모델의 관문 0 실패
기록이다. 지우지 말 것(스펙 §10-6) — "M06은 ASE 인터페이스가 없어 실행 불가"도 논문 SI의 문장이다.

**2단계 — 관문 1 (gate_* 컬럼 5개).** 실험 사실 재현 여부:

| 컬럼 | 의미 | False면 |
|---|---|---|
| `gate_host_glide` | HOST가 x=0.52에서 P3 (전이 재현) | 이 모델은 우리 계의 핵심 물리를 못 봄 — **탈락** |
| `gate_host_V_window` | HOST V_avg ∈ [2.5, 3.4] (실험 ~3.1, GGA 계열은 0.3–0.5 V 과소) | 화폐/에너지 스케일 문제 |
| `gate_pris_direction` | x=1에서 dE_pris(HEO) − dE_pris(HOST) > margin | 0 K 화폐가 해상하는 유일한 앵커 방향(v4 게이트)마저 못 맞춤 |
| `gate_host_xstar_in_window` | HOST 전이가 그리드 창 안에서 발생 | 전이 위치가 비정상 |
| `gate_vseg_positive` | 모든 구간 전압 > 0 (invariant 9) | 열역학 부호 오류 |

주의: **HEO 지연(gap_x067 등)은 관문이 아니라 진단**이다. v4에서 확정했듯 0 K PBE-화폐
열역학에는 지연 신호가 없다(ORB −4.9 / MACE −5.2 일치). 어떤 모델이 gap을 크게 양수로 낸다면
그건 "지연 재현"이 아니라 **동료들과 다른 소리를 내는 것**이므로 오히려 검증 대상.

**3단계 — 모델 간 상호 비교 (참고치).**
- `HOST_dE_*`, `HEO_dE_*` 8개 컬럼: 모델별 전이 풍경. 부호가 갈리는 지점(주로 x081/x067)이
  모델 간 실질 차이다.
- `rho_dE_vs_M01_n8`, `sign_agree_vs_M01`: 기준선(orb_mpa) 대비. **n=8이므로 순위 매기지 말 것** —
  "M0X가 기준선과 크게 다르다/비슷하다"의 스크리닝용. 우열 선언은 DFT 후 짝지은 부트스트랩
  CI가 0을 제외할 때만(스펙 §5).
- `E_Na`: 모델 자기 화폐의 bcc Na. PBE ~−1.31 eV/atom에서 크게 벗어나면 화폐가 다르다는 뜻
  (orb_omat이 그래야 정상 — 화폐 대조군의 존재 이유).
- `x_star_mid`: 전이 구간 중점. 모델 간 이 값의 산포가 "그리드 해상도 대비 모델 불확실성"의 크기.
- **기하 컬럼** `HOST/HEO_dV_pct_desod, dA_pct_desod, dh_perp_pct_desod, vm_strain_desod` (+x067판):
  변경 A의 det 분해 그룹 평균. dE 부호가 같아도 기하가 다른 모델을 가려낸다 — 정상 방향은
  면내 수축(dA<0) + 층간 팽창(dh_perp>0). 원본은 모델별 `anchors_x4_dV.csv`.
- **Ehull 확장 컬럼**: `rho_ehull_eff`(top-N Ehull_eff 순위 상관 vs M01), `n_surv_topN`(그 모델
  기준 관문 통과자 수), `surv_overlap_M01`(M01 생존자 중 그 모델도 살린 비율 — **"모델을 바꿨으면
  top-30이 얼마나 달라졌을까"의 직접 답**), `rho_dE_pris_topN`/`sign_dE_pris_topN`(n=30 프리스틴
  dE 비교 — n=8 앵커 비교보다 통계력이 훨씬 큼). 주의: Ehull 절대값의 모델 간 비교는 MP 화폐
  모델끼리만 정식이고, orb_omat 등 대조군의 값은 화폐 진단으로 읽는다.

**4단계 — 실무 지표.** `t_relax_median_s`(구조당 이완 시간), `conv_rate`(수렴률).
conv_rate < 0.9면 그 모델은 이완 불안정 — S/R 교차 진단표(스펙 §6)의 "이완 불안정" 행.

### 5-3. XRD 컬럼 해석 (세 번째 판독기)

XRD는 에너지(dE)·기하(P_order 글자 분류기)에 이은 **회절 지문 판독기**다. 세 판독기가 서로
독립적 방법으로 같은 답을 내는지가 요점.

| 컬럼 | 의미 | 해석 |
|---|---|---|
| `tt_003` ↓ (저각 이동) | 층간 팽창 (d₀₀₃ = h_perp/3 ↑) | 탈소듐 시 정상 방향 — O 층간 반발 |
| `tt_110` ↑ (고각 이동) | 면내 수축 (a_eff ↓) | 탈소듐 시 정상 방향 — TM 산화로 결합 단축 |
| `xrd_dir_ok_003/110` | HOST에서 위 두 방향이 x에 대해 단조인가 | False = 모델 기하가 실험 서열과 모순 |
| `chk_d003_resid` | \|d₀₀₃(피크) − h_perp/3\| | > 0.005 Å = 캐시 키/셀 판독 오류 (물리 아님 — 버그 탐지기) |
| `xrd_sim_O3/P3`, `xrd_phase` | HOST 기준 지문과의 코사인 유사도 | 이 구조가 "어느 상처럼 회절하는가" |
| `xrd_agree_letter` | XRD 판정 == P_order 판정 | False 다발 = 상 뒤집힘(n_phase_flip) 의심 구간 |
| `I_104_rel` | (104)류 상대 세기 | 상 지문 보조. **세기는 상대 비교만** (Debye-Waller 없음) |
| `xrd_split_110` | (110) 분열 | 단사정 왜곡 신호 — 5샘플 평균으로만 실체 판단 |

**XRD로 하지 말 것** (XRD_SPEC §8): 실험 2θ 절대값 비교(0 K + 화폐 오프셋), 피크 폭 비교
(FWHM 0.15°는 장식), 고각 세기 신뢰, SRO 논의(점유율 평균은 완전 무작위 가정).

### 5-4. `orb_path_consistency.csv` (셀 8)

같은 orb_mpa를 v4는 torch-sim으로, 벤치는 ASE로 이완했다. |ΔdE| < 5 meV/f.u.(δ 바닥)이면
"경로 차이는 판정에 무영향" 각주가 성립. 넘으면 모든 모델 간 비교 옆에 이 수치를 병기할 것.

### 5-5. 기존 v4 결과의 XRD (`xrd_v4/results_xrd_v4.csv`)

스크리닝 top-30(또는 top-500)의 이완 구조를 실험 언어로 번역한 표. 사용처:
- **후보별 (003)/(110) 이동량** = dh_perp/dA의 실험 관측 가능 형태 → 실험자 커뮤니케이션
- `xrd_agree_letter`가 False인 행 = 글자 분류기와 회절이 불일치 → n_phase_flip과 교차 검토
- HOST 파일럿 실패 시 top-N 확장은 자동 중단됨(§5) — 실패 행 목록부터 볼 것

## 6. 흔한 함정

1. **화폐 대조군을 탈락자로 읽지 말 것.** orb_omat(OMat)·PET-MAD(PBEsol)는 "화폐 불일치가
   dE에 얼마나 영향을 주는가"를 답하기 위해 참가한다. E_Na·V_avg가 다르게 나오는 것이 정보다.
2. **n=8 ρ로 순위표 만들지 말 것.** 상위권 압축 + 표본 8개. 랭킹 ρ는 DFT CORE(+EXT60) 후.
3. **mock 행이 결과 표에 섞이면 안 됨.** mock은 배선 검사 전용(레지스트리 명시).
4. **AMB는 실패가 아니라 판정 유보.** |dE| ≤ δ 구간을 억지로 O3/P3로 읽지 말 것.
5. **HEO 지연을 관문으로 승격하지 말 것.** v4에서 은퇴한 게이트다(§5-2 참조).
6. **벤치 결과로 채점 정의(SCORE_TERMS/관문/δ)를 바꾸지 말 것** — 사용자 승인 필수(협업 규약).

## 7. DFT 도착 후 다음 단계 (이 파이프라인이 그대로 확장됨)

1. **Tier S**: FRAMES(DFT 이완 궤적)에 각 모델 static forward → c_force(연화계수)·cMAE_force·
   MAE_E_atom. 러너의 relax 루프를 `static`으로 바꾼 변형이면 충분 — 구조·CSV 계약 동일.
2. **관문 2**: sign_R(x052) ≥ 기준선 − CI (짝지은 부트스트랩).
3. **지표 3**: rho_ddE on CORE 30 (+EXT60 권장 — 범위 제한 해소).
4. **최종 결정문**: 주모델 1 + 계보 다른 대조모델 1 → δ의 |dE_A − dE_B| 항 재계산 영향은
   보고만(채점 정의 불변).

## 8. 알려진 한계·리스크 (정직 고지)

- M06(EquiformerV3+DeNS-OAM): 공식 ASE calculator 미확인 → `blocked`. 실행 불가로 끝날 수 있음.
- M09/M10/M11: 체크포인트는 실재 확인했으나 calculator 클래스명은 후보 시도 방식 —
  팟 첫 실행에서 AdapterError 메시지를 보고 빌더 한 곳만 고치면 됨.
- M11(TECE): python 3.13 + torch 2.13 전용 venv — 설치가 가장 무겁다(체크포인트 891MB).
- 벤치의 ASE 경로는 프로덕션(torch-sim)과 다름 — 셀 8이 정량화하며, 리포트에 자동 병기.
- 앵커 2종 × 4x는 방향 시험이지 정확도 시험이 아니다. "이 모델이 더 정확하다"는 문장은
  DFT 정답지 없이는 쓸 수 없다.
