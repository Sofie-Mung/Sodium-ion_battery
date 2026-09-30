# SSE_PIPELINE_SPEC.md — Li₆PS₅Cl 듀얼 도핑 MLIP 스크리닝 파이프라인

> **대상 독자:** Claude Code (구현 담당)
> **작성 목적:** 설계 결정, 물리적 근거, 판정 기준, 불변식을 빠짐없이 전달해서, 구현 과정에서 설계 의도가 바뀌지 않게 하는 것
> **실행 환경:** RunPod GPU pod (4× RTX PRO 4500 또는 A100)
> **범위:** MLIP 계산(구조 생성 → relax → 안정성 → MD → 전도도 분석)까지. **DFT 계산은 이 파이프라인에서 실행하지 않는다.** DFT용 구조만 export 한다.

---

## 0. 먼저 읽을 것 — 절대 규칙

아래 규칙은 구현 중 어떤 이유로도 바꾸지 않는다. 바꿔야 할 이유가 생기면 **구현을 멈추고 사용자에게 먼저 묻는다.**

1. **연구 목적 고정:** 후보가 너무 많아 DFT로 다 계산할 수 없으므로, MLIP로 전체를 계산해 **전체 비교 결과 + 상위 후보**를 산출한다. DFT 검증은 사용자가 별도로 수행한다.
2. **모델은 ORB v3와 SevenNet 두 개.** 두 모델의 에너지를 절대로 하나의 hull이나 하나의 비교에 섞지 않는다 (currency 일관성). 모든 참조상도 같은 모델로 계산한다.
3. **도펀트는 반드시 슈퍼셀 안에서 치환한다.** 단위셀에서 치환한 뒤 복제(tiling)하는 방식은 금지한다 (인공적인 도펀트 초격자가 생긴다).
4. **S/Cl 무질서도는 모든 조성에서 같은 값으로 고정한다.** 조성마다 다르면 도펀트 효과가 무질서도 효과에 묻힌다.
5. **판정 기준(Section 9)은 결과를 보기 전에 `PILOT_CRITERIA.md`로 저장하고, 결과를 본 뒤 수정하지 않는다.**
6. **불변식(Section 5.6) assert는 제거하거나 완화하지 않는다.** 실패하면 즉시 중단한다.
7. **기존 기능이나 코드를 지우고 싶으면 먼저 사용자에게 허락을 받는다.**
8. **모든 전도도 결과에는 신뢰구간을 붙인다.** 신뢰구간 없는 σ 값은 출력하지 않는다.
9. **체크포인트 이름, 패키지 API, 좌표 등 확실하지 않은 것은 추측하지 않는다.** 공식 문서나 실제 설치된 패키지로 확인하고, 확인한 내용을 `run_manifest.json`에 기록한다.

### 0.1 사용자 결정에 의한 변경 (2026-09-28, Stage I 결과가 나오기 전)

throughput 측정 결과(416원자, 1 fs): SevenNet 7net-omni 0.23 ns/day/system(배치 4에서 OOM), ORB v3 약 3 ns/day/GPU(배치 4).
원래 설계(162 MD × 2 ns)는 약 750 GPU-일이 걸려서, 사용자가 GPU 4장 기준 4~8시간에 끝나도록 아래와 같이 바꿨다.
이 절의 내용이 아래 본문(규칙 2, 6.x, 7.5, 9의 G-model)보다 우선한다.

| 항목 | 원래 | 변경 |
|---|---|---|
| 모델 | ORB v3 + SevenNet | **ORB v3만**. Stage I-M은 모델 선정이 아니라 "ORB 검증"(G-a 통과 필수)이다. G-model은 평가하지 않는다. SevenNet은 나중에 상위 후보 재채점용으로만 선택적으로 쓴다 |
| timestep | 1 fs | 2 fs |
| 온도 | 500/600/700 K | 600/750/900 K (게이트 온도는 계속 600 K) |
| NPT / NVT | 20 ps / 2000 ps | 10 ps / 100 ps |
| MD config 수 | 3 | 2 |
| 300 K 외삽 σ | 보고 | 판정에 쓰지 않고 출력도 하지 않는다 (CSV 컬럼은 남아 있다) |
| 예산 | 72 h | 8 h (`budget_hours`) |

알려진 대가: 비율 σ_X/σ_A의 95% CI가 대략 ±35%로 넓어져서, 약 1.5배 이상의 차이만 확실히 구분할 수 있다. 600 K에서 확산이 느린 조성은 non-diffusive로 표시될 수 있다.

---

## 1. 연구 배경 (구현자가 알아야 할 물리)

### 1.1 호스트
- **Li₆PS₅Cl (argyrodite), 공간군 F-43m.** 관용 단위셀 = 4 f.u. = Li₂₄P₄S₂₀Cl₄ (52원자).
- **Wyckoff 자리:**
  - P: 4b (PS₄ 사면체 중심)
  - S: 16e (PS₄에 결합된 S)
  - "free" S와 Cl: 4a / 4c (이 두 자리에서 S/Cl 교환 = 음이온 무질서)
  - Li: 48h 자리가 **쌍(pair)**을 이루며, 한 쌍에는 Li가 하나만 들어간다 (실질 점유 50%). 단위셀당 Li 24개.
- 2×2×2 슈퍼셀 = **416원자** = Li₁₉₂ P₃₂ S₁₆₀ Cl₃₂.

### 1.2 전하 장부 (가장 중요)
Li⁺ 자리에 Mⁿ⁺가 들어가면 양전하가 (n−1)만큼 남으므로 Li⁺를 추가로 (n−1)개 빼야 한다. 즉 **Mⁿ⁺ 1개당 Li n개 감소.**
P⁵⁺ 자리에 Ge⁴⁺가 들어가면 양전하가 1 부족하므로 **Li 1개 추가.** 추가된 Li는 기존 48h 쌍이 사실상 만석이라 **격자간 자리(16e/T4 등)**에 들어간다.

$$N_{\text{Li}} = 192 - \sum_i n_i\,k_i + \sum_j (5 - v_j)\,g_j \quad (\text{슈퍼셀 기준})$$

- kᵢ: Li 자리 금속 i의 개수, nᵢ: 그 가수
- gⱼ: P 자리 도펀트 j의 개수, vⱼ: 그 가수 (Ge = 4 → +1)

### 1.3 설계 원칙
- **P 자리 Ge는 단위셀당 최대 2개 (슈퍼셀 최대 16개, P의 50%).** 그 이상이면 P가 사라져 LPSCl이 아닌 다른 화합물이 된다.
- **전도도는 장벽에 지수적으로 민감하다.** 장벽 0.03 eV 오차 → 300 K에서 σ 약 3배 차이. 그래서 (a) 모든 σ는 순수 호스트 대비 **비율 σ/σ_host**로 비교하고, (b) 신뢰구간을 필수로 붙이고, (c) 두 모델의 순위 부호 일치를 확인한다.
- **같은 Li 개수끼리만 비교한다.** 가수 합이 Li 개수를 결정하고, Li 개수가 σ를 크게 좌우하므로, 전체를 한 줄로 세우면 "1가 금속이 많을수록 좋다"만 재발견하게 된다.

### 1.4 파이프라인 전체 구조
```
Stage I-M : 모델 선정 (ORB v3 vs SevenNet → 주 모델 / 대조 모델 결정)
Stage I   : 단일종 파일럿 (9 조성) → 게이트 판정
Stage II  : 다성분(3종 1:1:1, 2종 2:1) 시너지 스크리닝 → 전체 비교 + 상위 후보
Export    : 상위 후보 구조를 DFT 인수인계 폴더로 출력
```

---

## 2. 환경과 파일 구성

### 2.1 패키지
| 패키지 | 용도 | 비고 |
|---|---|---|
| `orb-models` | ORB v3 | conservative force 모드, checkpoint는 사용자의 기존 HEO 파이프라인과 같은 계열(`orb-v3-conservative-*-mpa`)을 우선. 실제 이름은 설치 후 확인하고 기록 |
| `sevenn` | SevenNet | 최신 공식 universal checkpoint 중 **MP 호환 PBE 계열 modal**을 선택. 멀티 fidelity 모델이면 modal 이름을 명시적으로 지정하고 기록 |
| `torch-sim` | relax, MD 엔진 | 사용자 기본 엔진 (0.6.0). **SevenNet 지원 여부를 먼저 확인.** 미지원이면 SevenNet만 ASE calculator + ASE MD로 fallback 하고 그 사실을 기록 |
| `pymatgen` | 구조, 대칭, PhaseDiagram, MP API | MP API key는 환경변수 `MP_API_KEY` |
| `ase` | 구조 I/O, fallback MD | |
| `kinisi` | 확산계수 + 불확실성 | https://github.com/bjmorgan/kinisi |
| `pandas`, `numpy`, `scipy` | 분석 | |
| `python-pptx` | 편집 가능한 figure 출력 | **PNG 금지.** 모든 그림은 pptx 네이티브 차트로 |

### 2.2 파일 구성 (래퍼 셸 스크립트 금지)
```
sse_screening/
├── sse_screening.ipynb      # 전체 파이프라인. Run All 가능. Stage 플래그로 구간 선택
├── sse_worker.py            # 모든 함수 (구조 생성, relax, hull, MD, kinisi, pptx)
├── PILOT_CRITERIA.md        # Section 9 판정 기준 (Stage I 실행 전에 작성/커밋)
├── inputs/
│   ├── Li6PS5Cl.cif         # 사용자 제공 호스트 CIF (필수)
│   └── exp_reference.csv    # 사용자 제공 실험 기준값 (Section 6.3)
├── outputs/
│   ├── run_manifest.json    # 패키지 버전, checkpoint, seed, 설정 전체
│   ├── structures/          # 생성/relax 구조 (extxyz)
│   ├── trajectories/        # MD 궤적 (unwrapped 좌표)
│   ├── results_model_selection.csv
│   ├── results_single.csv
│   ├── results_multi.csv
│   ├── top_candidates.csv
│   ├── figures.pptx
│   └── dft_handoff/
└── logs/
```

### 2.3 공통 규약
- **노트북 구성:** 첫 셀에 `CONFIG` dict 하나로 모든 파라미터를 모은다 (별도 config 파일 없음). 셀 순서대로 실행하면 끝까지 돈다.
- **Stage 플래그:** `RUN_STAGE_IM`, `RUN_STAGE_I`, `RUN_STAGE_II`, `RUN_EXPORT`.
- **재개(resume):** 모든 단계는 결과 CSV/파일이 이미 있으면 건너뛴다. key = (model, composition_id, config_id, temperature).
- **seed:** 모든 무작위 과정은 `CONFIG["seed"]`에서 파생된 seed를 쓰고, 파생 seed를 결과에 기록한다.
- **GPU 분배:** GPU당 worker 프로세스 1개, torch-sim autobatcher로 GPU 내 배치 처리 (HEO v2 `heo_worker.py` 방식).
- **종료:** Run All 마지막 셀에서 결과 저장을 확인한 뒤 pod 자동 종료 (`runpodctl stop pod $RUNPOD_POD_ID`). `CONFIG["auto_terminate"]`로 끌 수 있게 한다.
- **언어:** 코드, 주석, docstring, 변수명, 에러 메시지는 모두 영어.

---

## 3. CONFIG 기본값

```python
CONFIG = {
    "seed": 20260927,
    "host_cif": "inputs/Li6PS5Cl.cif",
    "supercell": (2, 2, 2),

    # anion disorder: fraction of 4c sites occupied by Cl (same for ALL compositions)
    "scl_disorder_4c_cl_fraction": 0.375,      # 12 of 32 swaps in 2x2x2

    # Li 48h pair detection
    "li48h_pair_cutoff": 2.0,                  # Angstrom; verify with distance histogram, see 5.2

    # P-site cap
    "max_ge_per_supercell": 16,                # 50% of P

    # structure generation
    "n_configs_pilot": 8,                      # 4 random + 2 clustered + 2 dispersed
    "n_configs_stage2": 5,                     # 3 random + 1 clustered + 1 dispersed
    "min_interatomic_dist": 1.6,               # Angstrom, hard assert
    "interstitial_min_cation_dist": 2.0,       # Angstrom, candidate site filter
    "vacancy_near_radius": 4.0,                # Angstrom, "near dopant" definition

    # relax
    "relax_fmax": 0.02,                        # eV/A
    "relax_max_steps": 1000,
    "relax_cell": True,                        # FIRE + Frechet cell filter

    # config selection for MD
    "t_syn": 800.0,                            # K, for Boltzmann weights (reported, see 7.3)
    "n_configs_md_pilot": 3,
    "n_configs_md_stage2_short": 2,

    # MD
    "md_timestep_fs": 1.0,
    "md_temperatures_pilot": [500, 600, 700],
    "md_npt_ps": 20.0,
    "md_nvt_ps_pilot": 2000.0,                 # to be finalized after throughput test
    "md_nvt_ps_stage2_short": 500.0,
    "md_stage2_short_T": 600,
    "md_save_every_fs": 100.0,
    "md_thermostat": "nose_hoover",            # fallback: langevin with friction <= 0.002 /fs, record it

    # analysis
    "kinisi_start_dt_ps": 2.0,                 # skip ballistic regime
    "ci_level": 0.95,
    "t_extrapolate": 300.0,

    # models
    "models": ["orb_v3", "sevenn"],
    "auto_terminate": True,
}
```

---

## 4. 조성 정의

### 4.1 조성 스키마
각 조성은 다음 dict로 정의한다 (슈퍼셀 기준 개수).
```python
{
  "id": "M2",
  "li_site": {"Mg": 24},        # element -> count on Li sites
  "p_site": {"Ge": 16},         # element -> count on P sites
  "stage": "I",
  "role": "divalent, compensated",
}
```
가수는 고정 표 `VALENCE = {"Na":1,"Ag":1,"Cu":1,"Mg":2,"Zn":2,"Ca":2,"Al":3,"Ga":3,"In":3,"Sc":3,"Y":3,"Ge":4,"P":5}`를 사용한다.

### 4.2 Stage I 파일럿 조성 (9개)
단위셀 기준 표기는 이해용이며, 실제 입력은 슈퍼셀 개수(×8)다.

| ID | Li 자리 (슈퍼셀) | P 자리 (슈퍼셀) | N_Li | N_Li / 24 per uc | 역할 |
|---|---|---|---|---|---|
| A  | — | — | 192 | 24 | 호스트 앵커 |
| F  | — | Ge 16 | 208 | 26 | Ge 경향 확인 (게이트 G-a) |
| N0 | Na 24 | — | 168 | 21 | 1가, 보상 없음 |
| N2 | Na 24 | Ge 16 | 184 | 23 | 1가, 보상 |
| M0 | Mg 24 | — | 144 | 18 | 2가, 보상 없음 |
| M2 | Mg 24 | Ge 16 | 160 | 20 | 2가, 보상 |
| L0 | Al 24 | — | 120 | 15 | 3가, 보상 없음 |
| L2 | Al 24 | Ge 16 | 136 | 17 | 3가, 보상 |
| D  | Mg 4 | Ge 8 | 192 | 24 | A와 등-Li (게이트 G-c) |

**Li 조정량 계산 예시** (구현 검증용 단위 테스트에 그대로 넣을 것):
- 금속을 Li 자리에 넣은 직후 정규 자리의 Li = 192 − Σkᵢ
- 조정량 `delta = N_Li − (192 − Σkᵢ)`
  - delta < 0 → |delta|개의 Li를 정규 자리에서 제거 (공공)
  - delta > 0 → delta개의 Li를 격자간 자리에 삽입

| ID | 192 − Σk | N_Li | delta | 동작 |
|---|---|---|---|---|
| A  | 192 | 192 | 0 | — |
| F  | 192 | 208 | +16 | 격자간 삽입 16 |
| N0 | 168 | 168 | 0 | — |
| N2 | 168 | 184 | +16 | 격자간 삽입 16 |
| M0 | 168 | 144 | −24 | 공공 24 |
| M2 | 168 | 160 | −8 | 공공 8 |
| L0 | 168 | 120 | −48 | 공공 48 |
| L2 | 168 | 136 | −32 | 공공 32 |
| D  | 188 | 192 | +4 | 격자간 삽입 4 |

### 4.3 Stage II 조성
- **원소 풀 (11개):**
  - 1가: Na, Ag, Cu (**Cu는 `flag_redox_risk=True`로 표시**, 제외하지는 않음)
  - 2가: Mg, Zn, Ca
  - 3가: Al, Ga, In, Sc, Y
- **단일종 기준선:** 각 원소 24개(단위셀당 3개) × Ge ∈ {0, 8, 16} → 33 조성
- **3종 (1:1:1):** 서로 다른 3종 각 8개 (단위셀당 1개씩) → C(11,3) = 165 조합
- **2종 (2:1):** 1가 원소 A 16개 + 다른 원소 B 8개 → 3 × 10 = 30 조합. (1가만으로는 서로 다른 3종 조합이 거의 나오지 않으므로 이 그룹을 2종으로 보완한다.)
- **P 보상:** 각 조합 × Ge ∈ {0, 8, 16}
- **합계:** (165 + 30) × 3 = **585 조성** (+ 단일종 33)
- 각 조성에 `li_group = N_Li` 열을 붙인다. 가수 조합별 N_Li (Ge 16 기준, 단위셀 환산):

| 가수 조합 | Li 손실/uc | N_Li/uc (Ge 2/uc) |
|---|---|---|
| (1,1,1) | 3 | 23 |
| (1,1,2) | 4 | 22 |
| (1,2,2), (1,1,3) | 5 | 21 |
| (2,2,2), (1,2,3) | 6 | 20 |
| (2,2,3), (1,3,3) | 7 | 19 |
| (2,3,3) | 8 | 18 |
| (3,3,3) | 9 | 17 |

---

## 5. 구조 생성

### 5.1 호스트 로딩
1. `inputs/Li6PS5Cl.cif`를 읽는다. **좌표를 코드에 하드코딩하거나 추측하지 않는다.** 파일이 없으면 중단하고 사용자에게 요청한다.
2. 대칭 확인: `SpacegroupAnalyzer`로 F-43m (No. 216)인지 assert.
3. Wyckoff 라벨로 4a, 4b, 4c, 16e, 48h 자리를 식별한다. CIF에 부분 점유(48h 0.5, 4a/4c 혼합)가 있으면 이 단계에서 정렬(ordering)한다.

### 5.2 Li 48h 쌍 처리
1. 단위셀의 48h 자리 간 거리 히스토그램을 그려 `logs/li48h_pair_hist.txt`에 저장하고, `li48h_pair_cutoff`가 쌍 내부 거리와 쌍 간 거리 사이에 있는지 확인한다. 아니면 중단하고 보고한다.
2. 쌍으로 묶고, 각 쌍에서 무작위로 하나만 Li로 채운다 (config마다 다른 seed).
3. assert: 단위셀당 Li = 24, 슈퍼셀 Li = 192.

### 5.3 S/Cl 무질서
- 정렬 상태 기준: Cl은 4a, free S는 4c.
- 슈퍼셀에서 `round(fraction × 32) = 12`개의 4a Cl과 12개의 4c S를 무작위로 맞교환한다.
- **이 무질서 배치는 조성마다 새로 뽑되, 교환 개수는 항상 12로 고정한다.**
- 근거: LPSCl 전도도는 S/Cl 무질서도에 크게 의존하며, 최적 범위는 연구마다 25% (ACS AMI 2024, https://pubs.acs.org/doi/10.1021/acsami.4c08865) 또는 37.5–50% (Chem. Mater. 2025, https://pubs.acs.org/doi/10.1021/acs.chemmater.4c01152)로 보고되었다. 이 파이프라인은 그중 하나로 고정해 숨은 변수를 통제하는 것이 목적이다.

### 5.4 도펀트 배치
**P 자리:** 32개 P 중 gⱼ개를 선택해 Ge로 교체.
**Li 자리:** 192개 Li 중 Σkᵢ개를 선택해 금속으로 교체.

config 모드별 선택 규칙:
| 모드 | Li 자리 도펀트 선택 | 공공(Li 제거) 선택 |
|---|---|---|
| `random` | 무작위 | 무작위 |
| `clustered` | Ge 원자에 가까운 Li 자리 우선 | 도펀트 반경 `vacancy_near_radius` 이내 우선 |
| `dispersed` | 도펀트 간 최소거리를 최대화 (greedy) | 도펀트에서 먼 자리 우선 |

- Ge가 없는 조성의 `clustered`는 "Li 자리 도펀트끼리 가깝게"로 정의한다.
- 다성분 조성에서는 원소 종류를 무작위로 섞어 배치한다 (종류별 개수는 고정).

### 5.5 격자간 Li 삽입 (delta > 0일 때)
1. **후보 자리 열거:** 음이온(S, Cl) 위치로 Delaunay 사면체를 만들고, 각 사면체의 중심을 후보로 한다.
2. **필터:** 모든 양이온(Li, P, Ge, 금속)과의 거리 ≥ `interstitial_min_cation_dist`, 모든 음이온과의 거리 ≥ 1.9 Å.
3. **라벨링:** 가능하면 호스트 대칭 기준 Wyckoff 라벨(특히 16e)을 붙인다. 문헌상 16e/T4 격자간 자리가 cage 사이 이동에 핵심이므로 **16e 라벨 자리를 우선 선택**하고, 부족하면 나머지 후보에서 고른다.
4. 삽입한 Li끼리도 최소거리 조건을 만족해야 한다.
5. 선택된 자리의 라벨 분포를 결과에 기록한다 (`n_interstitial_16e`, `n_interstitial_other`).
6. 후보가 delta보다 적으면 중단하고 보고한다.

근거:
- Si⁴⁺ 치환이 T4 자리 Li 점유를 활성화해 inter-cage 확산을 높인다 (Mater. Adv. 2024, https://pmc.ncbi.nlm.nih.gov/articles/PMC10911230/)
- 격자간 16e 자리 점유가 Li-rich/Li-poor argyrodite 공통의 고전도 원인이라는 보고 (https://bin-ouyang.com/assets/pdf/chien.pdf)

### 5.6 불변식 (모든 생성 구조에 hard assert)
```
INV-1  sum of formal charges == 0
INV-2  count(Li) == 192 - sum(n_i * k_i) + sum((5 - v_j) * g_j)
INV-3  count(P) + count(P-site dopants) == 32
INV-4  count(S) == 160 and count(Cl) == 32
INV-5  dopant counts == composition spec exactly
INV-6  count(Cl on 4c-derived sites) == 12   (disorder fixed)
INV-7  min interatomic distance >= CONFIG["min_interatomic_dist"]
INV-8  sum of P-site dopants <= CONFIG["max_ge_per_supercell"]
INV-9  total atoms == 416 - (Li vacancies) + (Li interstitials)
```
불변식 실패 시 해당 구조를 버리지 말고 **파이프라인 전체를 중단**한다.

---

## 6. Stage I-M — 모델 선정

### 6.1 목적
ORB v3와 SevenNet 중 **주 모델**(전체 스크리닝용)과 **대조 모델**(상위 후보 재채점용)을 결정한다. 두 모델 모두 끝까지 사용하며, 역할만 정한다.

### 6.2 대상
- 조성 A, F (각 config 3개)
- 두 모델 각각 독립적으로: relax → NPT → NVT (500/600/700 K) → kinisi

### 6.3 기준값 (사용자 제공)
`inputs/exp_reference.csv` 형식:
```
quantity,value_low,value_high,source
LPSCl_sigma_RT_mScm,,,
LPSCl_Ea_eV,,,
LPSCl_lattice_a_A,,,
```
**Claude Code는 이 값을 채우지 않는다.** 비어 있으면 절대값 비교는 건너뛰고 경고만 남긴다.

### 6.4 측정 항목
| 항목 | 내용 |
|---|---|
| `a_relaxed` | relax된 A의 격자상수 |
| `sigma_A_300K`, CI | A의 300 K 외삽 σ |
| `Ea_A`, CI | A의 활성화 에너지 |
| `ratio_F_A`, CI | σ_F / σ_A (600 K 및 300 K 외삽) |
| `throughput_ns_per_day` | 416원자, 1 fs, 단일 GPU 기준 |

### 6.5 선정 규칙 (미리 고정)
1. 두 모델 모두 **σ_F/σ_A > 1 (CI 하한 > 1)**을 만족하면 → 처리 속도가 빠른 모델을 주 모델로.
2. 한 모델만 만족하면 → 그 모델이 주 모델. 다른 모델은 대조로 유지하되 `results_model_selection.csv`에 불일치를 기록.
3. 둘 다 만족하지 않으면 → **중단하고 사용자에게 보고.**
4. `exp_reference.csv`가 채워져 있으면 A의 σ와 Eₐ가 실험 범위 안인지 추가로 기록한다 (선정에는 참고용).

---

## 7. Stage I — 단일종 파일럿

### 7.1 목적
- (a) 주 모델이 Ge 치환 경향을 재현하는가
- (b) 가수별로 P 보상이 전도도를 얼마나 회복시키는가
- (c) Li 개수가 같을 때 도펀트 자체가 σ를 바꾸는가
- (d) ln σ가 N_Li에 대해 근사적으로 선형인가 (Stage II의 S_syn 정의가 이 가정에 의존)

### 7.2 relax
- torch-sim FIRE + Frechet cell filter, `fmax = 0.02 eV/Å`, 최대 1000 step.
- 모든 config(8개)를 **두 모델 각각**으로 relax한다.
- 저장: 에너지(eV/atom), 수렴 여부, 격자상수, 부피, 도펀트–Ge 최근접 거리, 도펀트–공공 최근접 거리.
- 수렴하지 않은 구조는 `converged=False`로 표시하고 MD 대상에서 제외한다.

### 7.3 MD용 config 선택
- **에너지가 가장 낮은 3개 config**를 MD로 보낸다.
- Boltzmann 가중치 `w_j = exp(-E_j / kT_syn) / Σ` (T_syn = 800 K)도 계산해 기록한다.
- 416원자 셀에서는 config 간 에너지 차가 커서 가중치가 한 config에 몰릴 수 있다. 그래서 **주 결과는 3개 config의 동일 가중 평균**으로 하고, Boltzmann 가중 결과는 민감도 분석으로 함께 보고한다.

### 7.4 안정성 (Ehull)
1. **참조상 수집:** 각 조성의 화학계(Li–P–S–Cl + 도펀트 원소 + Ge)에 대해 MP API로 **모든 안정상과 hull 근처(E_hull ≤ 0.05 eV/atom) 상**을 가져온다. 최소한 다음이 포함되는지 assert:
   Li₂S, LiCl, Li₃PS₄, P₂S₅, Li₄GeS₄, GeS₂, 해당 도펀트의 황화물과 염화물 (예: Na₂S, NaCl, MgS, MgCl₂, Al₂S₃, AlCl₃).
2. **모든 참조상을 같은 모델, 같은 relax 설정으로 relax한다.** MP 에너지를 그대로 쓰지 않는다.
3. pymatgen `PhaseDiagram`에 MLIP 에너지로 `PDEntry`를 만들어 각 config의 `e_above_hull`을 구한다.
4. 호스트 A 자체의 `e_above_hull`도 기록하고, 도핑 조성은 **`de_hull_vs_host = e_above_hull(doped) − e_above_hull(A)`**로도 보고한다.
5. 파일럿에서는 **탈락시키지 않고 표시만** 한다.

### 7.5 MD 프로토콜
1. **NPT 20 ps** (0 GPa, 해당 온도): 마지막 10 ps의 평균 cell을 구한다.
2. **NVT 생산**: 평균 cell로 고정, `md_nvt_ps_pilot` 길이, thermostat는 Nosé–Hoover 우선. Langevin만 가능하면 friction ≤ 0.002 fs⁻¹로 하고 기록한다 (강한 friction은 확산을 왜곡).
3. timestep 1 fs, 100 fs마다 저장. **unwrapped 좌표**로 저장한다 (또는 unwrap 가능한 형태로 cell과 image flag를 함께 저장).
4. 온도: 500, 600, 700 K. 400 K 이상에서 비-Arrhenius 거동이 보고되어 있으므로 더 높은 온도는 쓰지 않는다.
5. **처리 속도 측정을 먼저 한다:** 조성 A, 600 K, 20 ps를 돌려 ns/day를 측정하고, 전체 파일럿 예상 GPU 시간을 `logs/throughput.txt`에 적는다. 예상 시간이 GPU 4장 기준 72시간을 넘으면 `md_nvt_ps_pilot`를 줄이기 전에 **사용자에게 보고하고 확인받는다.**
6. **확산 영역 확인 플래그:**
   - `flag_nondiffusive`: 마지막 시점 Li MSD < 20 Å² 이면 True
   - `msd_loglog_slope`: kinisi 피팅 구간에서 log MSD vs log t 기울기 (0.9–1.1이 정상)
   - 플래그가 켜진 결과도 버리지 않고 표시만 한다.

규모: 9 조성 × 3 config × 3 온도 × 2 모델 = **162 MD**. (Stage I-M의 A, F 결과는 재사용한다.)

### 7.6 전도도 분석
1. **kinisi**로 Li tracer 확산계수 D*와 그 사후분포(posterior samples)를 구한다. `start_dt = 2 ps`로 ballistic 구간을 제외한다.
2. **Nernst–Einstein:**
$$\sigma = \frac{N_{\text{Li}}\, e^2}{V\, k_B T}\, D^*$$
   V는 NVT cell 부피. Haven ratio = 1 가정을 결과 파일 메타데이터에 명시한다.
3. **config 합산:** ln σ의 config 평균을 구하고, 총 분산 = (config 내 불확실성 평균) + (config 간 분산)으로 합친다.
4. **Arrhenius:** 세 온도의 ln D vs 1/T를 가중 선형회귀로 피팅. kinisi posterior에서 샘플링해 Eₐ와 300 K 외삽값의 CI를 구한다 (bootstrap 또는 posterior propagation).
5. **호스트 비율:** 같은 모델, 같은 온도의 σ_A posterior와 결합해 `ratio_vs_A`와 CI를 구한다.
6. 참고: AIMD 기반 확산계수의 통계 오차 처리 원칙은 He et al., npj Comput. Mater. 2018 (https://doi.org/10.1038/s41524-018-0074-y)을 따른다.

### 7.7 Stage I 결과 표 `results_single.csv` (Stage II 단일종 기준선도 같은 파일에 append)
```
model, composition_id, stage, li_site, p_site, n_li, n_li_per_uc,
config_id, config_mode, seed, converged, e_per_atom, e_above_hull, de_hull_vs_host,
boltzmann_weight, a, b, c, volume,
T, D_star, D_star_ci_low, D_star_ci_high, sigma_Scm, sigma_ci_low, sigma_ci_high,
flag_nondiffusive, msd_loglog_slope,
n_interstitial_16e, n_interstitial_other, n_vacancy_near, n_vacancy_far
```
조성 단위 요약 표 `results_single_summary.csv`:
```
model, composition_id, n_li_per_uc, sigma_600K, ci, Ea, Ea_ci, sigma_300K, ci,
ratio_vs_A_600K, ci, ratio_vs_A_300K, ci, de_hull_vs_host_mean, notes
```

---

## 8. Stage II — 다성분 시너지 스크리닝

### 8.1 목적
Li 자리에 여러 금속을 섞었을 때 **단일종 평균보다 전도도가 높은 조합**을 찾는다. 스크리닝은 **주 모델**로 수행하고, 최종 상위 후보만 대조 모델로 재채점한다.

### 8.2 단일종 기준선
- 원소 풀 11개 × Ge {0, 8, 16} = 33 조성.
- 생성 config 5개 → relax → 저에너지 2개로 짧은 MD (600 K, 500 ps).
- 이 값이 S_syn의 분모가 된다.

### 8.3 구조 생성과 relax
- 585 조성 × config 5개 = 2925 relax (주 모델).
- Section 5의 규칙과 불변식을 그대로 적용한다.

### 8.4 엔트로피 보정 안정성
0 K MLIP hull에는 배위 엔트로피가 빠져 있어 다성분 조성이 부당하게 불리해진다. 다음으로 보정한다.

$$\Delta S_{\text{conf}} = -k_B \sum_{s} N_s \ln x_s, \qquad x_s = N_s / 192$$

- s ∈ {Li(정규 자리), 각 금속, 공공} — 정규 Li 자리 192개 위에서 계산한다. 격자간 Li는 포함하지 않는다.
- 호스트(모두 Li)는 0이므로 호스트 대비 값이 된다.

$$E_{\text{hull,corr}} = e_{\text{above\_hull}} - \frac{T_{\text{syn}}\,\Delta S_{\text{conf}}}{N_{\text{atoms}}}$$

- 이상 혼합 가정이므로 **엔트로피 효과의 상한**이다. 결과 파일에 이 가정을 명시한다.
- **필터 문턱값:** `de_hull_corr_vs_host ≤ threshold`. threshold는 Stage I 종료 후 파일럿 결과(특히 D, N2 같은 온건한 조성의 값)를 보고 **Stage II 시작 전에** 정해서 `PILOT_CRITERIA.md`에 기록한다. 기본 제안값은 0.03 eV/atom. Claude Code는 이 값을 임의로 정하지 않고, Stage I 요약과 함께 사용자에게 제안한 뒤 확인받는다.

### 8.5 짧은 MD와 S_syn
- 필터 통과 조성: 저에너지 config 2개, 600 K, 500 ps.
- **시너지 점수 (같은 Ge 수준의 단일종과 비교):**

$$S_{\text{syn}} = \frac{\sigma_{\text{mix}}}{\left(\prod_i \sigma_{M_i}^{\,k_i/K}\right)}, \qquad K = \sum_i k_i$$

  - 3종 1:1:1 → 세 단일종 σ의 기하평균
  - 2종 2:1 → (σ_A² · σ_B)^{1/3}
  - 단일종 기준선은 같은 Ge 수준(0/8/16)의 값을 사용한다.
- 왜 기하평균인가: 혼합 조성의 Li 손실은 구성 단일종 Li 손실의 평균과 같다. ln σ가 N_Li에 선형이면(게이트 G-lin), 기하평균 분모가 Li 개수 효과를 자동으로 상쇄한다. G-lin이 실패하면 분모를 "N_Li에 대한 단일종 ln σ 보간값"으로 교체한다 (Section 9 참조).
- S_syn의 CI는 분자와 분모 posterior를 결합해 구한다.

### 8.6 그룹별 선별
- `li_group`(N_Li)별로 S_syn 상위 5개를 선별한다. **전체 통합 순위는 만들지 않는다** (만들더라도 참고용 열로만).
- Cu 포함 조합은 `flag_redox_risk=True`로 표시하고 순위에서 빼지 않는다.

### 8.7 긴 MD (확정)
- 선별된 조성을 **새로운 seed**로 다시 생성·relax·MD 한다. 선별에 쓴 궤적을 연장하지 않는다 (winner's curse 방지: 선별 과정의 운 좋은 통계가 확정 결과에 섞이지 않게).
- Stage I과 같은 프로토콜: config 3개 × 500/600/700 K × `md_nvt_ps_pilot`.
- 같은 Ge 수준의 해당 단일종들도 같은 프로토콜로 긴 MD를 돌려 S_syn 분모를 갱신한다.

### 8.8 대조 모델 재채점
- 8.7의 최종 후보와 그 단일종 분모를 **대조 모델로** 같은 프로토콜로 계산한다 (relax부터, 참조상 hull도 대조 모델로 다시).
- 기록: 두 모델의 S_syn, `sign_agree = (S_syn_main − 1) * (S_syn_contrast − 1) > 0`, 그룹 내 Spearman 순위 상관.

### 8.9 결과 표
`results_multi.csv` (조성 단위):
```
composition_id, li_site, p_site, valence_combo, li_group, n_li_per_uc, flag_redox_risk,
e_above_hull, de_hull_vs_host, dS_conf, de_hull_corr_vs_host, pass_stability,
sigma_600K_short, ci, S_syn_short, ci,
selected_for_long, sigma_600K_long, ci, Ea, Ea_ci, sigma_300K, ci, S_syn_long, ci,
S_syn_contrast, ci, sign_agree
```
`top_candidates.csv`: 그룹별 최종 후보 (S_syn_long CI 하한 > 1 AND sign_agree == True를 `confirmed=True`로 표시, 나머지도 순위와 함께 수록).

---

## 9. 판정 기준 (PILOT_CRITERIA.md에 그대로 복사, 결과 보기 전 고정)

| ID | 기준 | 판정 규칙 | 실패 시 |
|---|---|---|---|
| G-a | Ge 경향 재현 | 주 모델에서 σ_F/σ_A의 95% CI 하한 > 1 (600 K) | 중단, 사용자 보고 |
| G-b | P 보상 효과 | 각 가수 X ∈ {N, M, L}에서 σ_X2/σ_X0 > 1 (CI 하한 기준) | 기록만, Stage II 해석에 반영 |
| G-c | 도펀트 자체 효과 | σ_D/σ_A의 95% CI가 1을 포함하지 않음 | 기록만 (포함하면 "등-Li에서 효과 없음"으로 보고) |
| G-lin | ln σ의 N_Li 선형성 | A, F, N0, N2, M0, M2, L0, L2의 ln σ(600 K) vs N_Li 선형 회귀에서 R² ≥ 0.8 이고 잔차에 체계적 곡률 없음 (2차항 계수의 CI가 0 포함) | S_syn 분모를 보간 방식으로 교체 |
| G-model | 두 모델 일치 | 파일럿 9 조성의 ratio_vs_A 순위 Spearman ρ ≥ 0.6 | 기록, 사용자 보고 (중단은 아님) |
| G-stab | Stage II 안정성 문턱 | Stage I 종료 후 제안 → 사용자 확인 후 기록 | — |

---

## 10. Figure (figures.pptx, 편집 가능한 네이티브 차트)

PNG를 만들지 않는다. `python-pptx`의 chart 객체(XL_CHART_TYPE)로 데이터를 넣어 사용자가 PowerPoint에서 직접 수정할 수 있게 한다. 슬라이드마다 원본 데이터 CSV 경로를 노트에 적는다.

1. **모델 선정:** A, F의 σ(600 K)와 σ_F/σ_A, 두 모델 비교 (막대, 오차 막대)
2. **보상 회복 지도:** x = N_Li/uc, y = σ/σ_A (log), 가수별 계열 (N0→N2, M0→M2, L0→L2 화살표 대신 두 점 연결 선), A, F, D 표시
3. **Arrhenius:** 파일럿 9 조성 ln σ vs 1000/T
4. **G-lin 검정:** ln σ(600 K) vs N_Li, 회귀선
5. **안정성:** de_hull_corr_vs_host 분포 (Stage II), 문턱선
6. **S_syn 분포:** Li 그룹별 (산점도, x = 그룹, y = S_syn, CI)
7. **그룹별 상위 후보:** 그룹당 상위 5개 막대, 대조 모델 값 병기

---

## 11. DFT 인수인계 export (`dft_handoff/`)

DFT 계산은 하지 않는다. 사용자가 별도로 수행할 수 있게 구조만 정리한다.
```
dft_handoff/
├── README.md                      # 각 폴더 설명, 모델, seed, 조성 정의
├── pilot/<composition_id>/
│   ├── relaxed_config_<k>.vasp    # 주 모델 relax 구조 (POSCAR 형식)
│   └── md_frames_<T>K.extxyz      # NVT 궤적에서 균등 간격 20 프레임
├── top_candidates/<composition_id>/
│   ├── relaxed_config_<k>.vasp
│   └── md_frames_600K.extxyz
└── references/<model>/            # hull 참조상 relax 구조
```
- 파일럿에서 D, M2, L2는 반드시 포함한다 (도펀트 주변 MLIP 신뢰도를 DFT로 확인하기 위한 프레임).
- 메타데이터 JSON에 모델, checkpoint, 에너지, 원래 config_id를 기록한다.

---

## 12. 구현 순서와 수용 테스트

각 단계의 테스트가 통과해야 다음 단계로 넘어간다.

1. **환경 확인:** 두 모델 로딩, 416원자 구조 1개로 에너지/힘 계산, torch-sim의 SevenNet 지원 여부 확인 → `run_manifest.json` 기록.
2. **전하 장부 단위 테스트:** Section 4.2 표의 9개 조성에 대해 `N_Li`와 `delta`가 표와 정확히 일치.
3. **구조 생성 테스트:** 9개 조성 × 8 config 생성, 불변식 INV-1~9 전부 통과. 48h 쌍 히스토그램 저장.
4. **kinisi 검증 테스트:** 알려진 D를 가진 합성 Brownian 궤적을 만들어 kinisi가 D를 CI 안에서 복원하는지 확인.
5. **스모크 테스트:** 조성 A, 두 모델, 600 K, NPT 2 ps + NVT 5 ps가 끝까지 돌고 unwrapped 궤적이 저장되는지 확인.
6. **처리 속도 측정** (7.5-5) → 사용자 보고.
7. Stage I-M → Stage I → 게이트 판정 → **사용자 확인 (G-stab 문턱값 포함)** → Stage II → Export.

**Stage I이 끝나면 반드시 멈추고** 요약(게이트 결과표, figures.pptx 슬라이드 1–4, 처리 시간)을 보고한 뒤 사용자 확인을 받고 Stage II를 시작한다.

---

## 13. 하지 말 것

- 단위셀 치환 후 tiling
- 두 모델의 에너지를 하나의 hull/비교에 섞기
- MP 에너지를 MLIP 에너지와 섞어 hull 계산
- 조성마다 다른 S/Cl 무질서도
- 신뢰구간 없는 σ 출력
- 판정 기준을 결과를 본 뒤 수정
- 불변식 제거·완화
- 선별에 쓴 궤적을 연장해 확정 결과로 사용
- Li 그룹을 무시한 통합 순위를 주 결과로 제시
- PNG figure 생성
- 사용자 허락 없이 기존 코드/기능 삭제
- 확인하지 않은 checkpoint 이름, API, 좌표 사용

---

## 14. 사용자에게 확인이 필요한 항목 (구현 시작 시 체크리스트)

- [ ] `inputs/Li6PS5Cl.cif` 제공 (부분 점유 포함 CIF 권장)
- [ ] `inputs/exp_reference.csv` 값과 출처 (비어 있어도 진행 가능)
- [ ] ORB v3 / SevenNet checkpoint 최종 확인
- [ ] 처리 속도 측정 후 MD 길이 확정
- [ ] Stage I 종료 후 G-stab 문턱값 확정

---

## 15. 참고 문헌

- Disorder-dependent Li diffusion in Li₆PS₅Cl (MLP, 25 ns, 6500 atoms): https://pubs.acs.org/doi/10.1021/acsami.4c08865
- Atomic mechanism of Li diffusion in Li₆PS₅Cl via MLP (S/Cl disorder 37.5–50%): https://pubs.acs.org/doi/10.1021/acs.chemmater.4c01152
- Si⁴⁺ substitution activates T4 interstitial occupancy (Li₆₊ₓP₁₋ₓSiₓS₅Br): https://pmc.ncbi.nlm.nih.gov/articles/PMC10911230/
- Ge substitution and Li site disorder in Li₆₊ₓP₁₋ₓGeₓS₅I: https://pmc.ncbi.nlm.nih.gov/articles/PMC8815078/
- Ge substitution in Li₆₊ₓGeₓP₁₋ₓS₅Br (up to 5× σ): https://pubs.rsc.org/en/content/articlehtml/2025/ta/d5ta01651g
- Aliovalent Li-site doping of Li₆PS₅Cl (Mg, Ba, Zn, Al, Y): https://pmc.ncbi.nlm.nih.gov/articles/PMC11106650/
- High-entropy (multi-cation P-site) argyrodite, entropy engineering: https://doi.org/10.1002/anie.202404874
- uMLIP benchmark for solid ion conductors (MatterSim, MACE, SevenNet, CHGNet, M3GNet, ORB): https://arxiv.org/abs/2502.09970
- Statistical variance of diffusional properties from MD: https://doi.org/10.1038/s41524-018-0074-y
- kinisi: https://github.com/bjmorgan/kinisi
- Shannon ionic radii: https://doi.org/10.1107/S0567739476001551
