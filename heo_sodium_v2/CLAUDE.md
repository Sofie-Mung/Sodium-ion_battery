# CLAUDE.md — HEO Na 층상 양극재 MLIP 스크리닝 v2: 코드 검토 인수인계

이 문서는 `heo_screening_v2.ipynb`를 셀 단위로 재검증하기 위한 맥락 전부입니다. 코드는 mock 모드로 로직이 검증됐고,
**실제 외부 API 4곳은 미검증**입니다(§6). 너의 첫 임무는 그 4곳을 실제 패키지로 확인하고, 그다음 §7의 검토 과제를 수행하는 것.

---

## 0. 작업 규칙 (먼저 읽을 것)

- 대화는 **한국어**, 코드 주석·변수명·커밋 메시지는 **영어**.
- 셀을 고칠 때는 패치가 아니라 **그 셀 전체를 완성본으로** 제시 (사용자는 수동 편집 실수를 피하고 싶어함).
- **불필요해 보이는 코드를 지우려면 먼저 물어볼 것.** 특히 `assert`(불변량)는 절대 임의로 제거·완화하지 말 것.
- 부호 규약·단위 규약(§3)을 바꾸지 말 것. 바꿔야 한다고 생각되면 근거를 제시하고 승인받은 뒤에.
- `results/compositions.csv`는 **입력 전용**. 어떤 단계도 덮어쓰지 않는다 (md5 불변량).
- 방법론적 판단이 필요하면 "왜"를 물리로 설명한 뒤 선택지를 제시. 논문(링크 포함)을 근거로 댈 것.
- 사용자는 원리를 이해하고 코드를 쓰기를 원한다. 설명은 쉽고 자세하게, 비유를 먼저, 그다음 수식.

---

## 1. 프로젝트 한 줄 요약

호스트 O3-NaNi₀.₅Mn₀.₅O₂의 TM층에 도펀트 16종 중 5종을 넣은 고엔트로피 조성 수천 개를 MLIP(ORB-v3)로 스크리닝해,
x=1→0.52 탈소듐 시 **O3→P3 gliding 전이가 억제되는 조성**을 찾고 top-30을 DFT(VASP, GGA+U)로 검증한다.
v1 파이프라인(7,279조성, 완료)은 "코드가 이해를 앞섰다"는 이유로 원리부터 재학습한 뒤 **v2로 처음부터 다시 짠 것**이다.

---

## 2. 파이프라인이 답하는 질문 두 개 (Bartel 2022의 두 축)

| 축 | 질문 | 지표 | 원칙 |
|---|---|---|---|
| ① 분해 안정성 | 단일상으로 **만들 수 있나?** (phase separation) | `Ehull`, `Ehull_eff`, `T_star` | 엔트로피는 입장권 |
| ② 폴리모프 안정성 | 만든 뒤 충전 시 **O3로 버티나?** (phase transition) | `dE_pris`, `dE_desod`, `ddE` | 화학이 경기를 이김 |

관통 원칙: **"입력하지 말고 읽어내라"** — 산화수도, 상(phase)도, 오차도 우리가 정하지 않고 계산이 정한 것을 판독한다.

---

## 3. 수식과 규약 (검토 시 대조 기준)

셀: 108원자 = Na₂₇ TM₂₇ O₅₄, TM 총전하 +81 (Na⁺ 27개와 O²⁻ 54개를 상쇄: −(27·(+1) + 54·(−2)) = +81). 탈소듐 = Na 13개 제거, x = 14/27 = 0.5185, n = 13.

| 양 | 식 | 단위 | 주의 |
|---|---|---|---|
| `V_avg` | `[E_desod_O3 + 13·E_Na − E_pris_O3] / 13` | V | 방전 반응 부호 규약. **분자에 탈소듐 에너지가 먼저 → 항상 양수.** 음수면 부호 버그(v1에서 실제로 있었음) |
| `dE_state` | `(E_P3,state − E_O3,state) × 1000 / 27` | meV/f.u. | 셀 값을 27로 나눔. \|dE\| > 500이면 단위 누락 |
| `ddE` | `dE_desod − dE_pris` | meV/f.u. | 이중차분. 클수록(덜 음수) 좋음 |
| 억제 판정 | `phase_pris == O3 and phase_desod == O3` | — | `judge_phase(dE, δ)`: > +δ → O3, < −δ → P3, 그 외 AMB |
| `delta` | `max(sigma_vac, |dE_desod − dE_desod_B|, 5)` | meV/f.u. | 회색지대. `sigma_vac` = 빈자리 5샘플 표준편차/27 |
| `dV_O3` | `100·|Vol_desod_O3 − Vol_pris_O3| / Vol_pris_O3` | % | O3 유지 경로 |
| `dV_trans` | `100·|Vol_desod_P3 − Vol_pris_O3| / Vol_pris_O3` | % | 상전이 포함 |
| `dV_nat` | phase_desod==O3 → dV_O3, 아니면 dV_trans | % | **이긴 상의 부피** (v1은 O3↔O3 일괄 → 과소평가 의심) |
| `Sconf` | `−Σ x_i ln x_i` (TM 자리당, k_B) | k_B | Scheme A ≈ 1.89, B ≈ 1.94 |
| `sconf_per_atom` | `Sconf × 27/108` | k_B | 무질서는 TM 부격자에만 |
| `Ehull_eff` | `Ehull − k_B·T·sconf_per_atom`, T=1173 K | eV/atom | 보정항 ≈ **46–48 meV/atom** (불변량 3) |
| `T_star` | `Ehull / (k_B·sconf_per_atom)` | K | 판이 뒤집히는 온도. ≤1173이면 열역학적 합성 가능 |
| 전하중성 | `Σ n_i·v_i = 81` (정수) | e | v1은 평균 3.0±0.10(±2.7e) 허용 → 7,279; v2 정확식 → **7,073** (A 3,773, B 2,982, C 318) |
| 채택 배분 | 부족 → Ni²⁺→Ni³⁺, V⁴⁺→V⁵⁺ / 과잉 → Mn⁴⁺→Mn³⁺, Co³⁺→Co²⁺, V⁴⁺→V³⁺ | 개수 | `n_Ni3, n_Mn3, n_Co2, n_V3, n_V5` — DFT 자화 감사가 검증할 예측 |

**화학양론 격자 3종** (27 TM 자리). 전하중성은 *어떤 산화수 배분이 존재하는가*만 보므로, 같은 원소 집합이라도
비율이 달라지면 통과/탈락이 뒤집힙니다. 그래서 격자를 몇 개 까느냐가 곧 탐색 범위입니다.

| 스킴 | 패턴 | 행 수 | 언제 쓰나 |
|---|---|---|---|
| A | Ni6 Mn6 + 도펀트 3-3-3-3-3 | 3,773 | 기본 |
| B | Ni4 Mn4 + 도펀트 4-4-4-4-3 | 2,982 | 기본 |
| C | Ni6 Mn6 + 도펀트 4-3-3-3-2 | 318 | **A도 B도 정수해가 없는 조합에만** (구제 격자) |

A+B로 탈락하는 도펀트 조합 595개 중 **593개는 비율만 바꾸면 해가 생기고**, 진짜 화학적으로 불가능한 건 2개뿐입니다
(Mg-Cu-Zn-Li-Ca 전부 1~2가 / Si-Ti-Zr-Sn-Sb 전부 4~5가). C는 A에서 가장 가까운 이웃 격자라 엔트로피 최대점에서
멀어지지 않으면서 282개 조합(그중 228개가 sconf ≥ 1.85)을 되찾습니다. C를 모든 조합에 적용하면 +4,055행(+60% 계산량)이지만
새로 구제되는 조합은 똑같이 282개뿐이라, **구제 조합에만** 적용합니다(+318행, +4.7%). mock top-30에 C가 4개 들어옵니다.

**S1-1 — 변형 선택 규칙.** B/C는 "어느 도펀트가 자리를 덜/더 받는가"에 따라 여러 변형이 생깁니다. 규칙은
`sconf 최대` → (동점이면) `itertools.combinations 순서 = DOPANTS 리스트 순서`. Scheme B는 2,982행 중 **2,087행이 이 동점**이라
사실상 리스트 순서가 정합니다(B 23%, Mg 14%, Al 12%가 short를 흡수, Co/Ca/V는 0%). 버려진 변형은 원소 수가 달라
`n_Ni3` 등 예측도 달라집니다. 동점 변형을 전부 열거하면 +101% 계산량이라 현행 유지하되, **논문 한계에 명시**할 것.
Scheme C는 구제 조합의 가능한 변형이 전부 동점이라 **전부 채택**했고, 따라서 C에는 임의 선택이 없습니다.

적층 규칙: Na 갭 양쪽 산소층 글자가 **다르면 O(팔면체), 같으면 P(프리즘)**. O3 = CABCAB(갭 전부 O) → glide 벡터 (2/9,1/9)로 P3 = CAABBC(갭 전부 P). 단위 k(슬랩 k + 바로 위 Na층)를 k·v 이동.

앵커(자, 2026-09-03 재설계): HOST NaNi₀.₅Mn₀.₅O₂ = {Ni14Mn13, Ni13Mn14} 평균 → 실험: gliding(P3), V≈3.1 V.
양성 앵커 = **HEO** NaNi₀.₁₂Cu₀.₁₂Mg₀.₁₂Fe₀.₁₅Co₀.₁₅Mn₀.₁Ti₀.₁Sn₀.₁Sb₀.₀₄O₂ (Zhao Angew. 2020, 전이 지연 — 용량 60%+가 O3 영역)
= 27자리 {Ni3 Cu3 Mg3 Fe4 Co4 Mn3 Ti3 Sn3 Sb1}, 같은 조성·다른 난수 2변형 평균. 게이트 v4(2026-09-03):
**pristine 방향** `dE_pris(HEO) − dE_pris(HOST) > margin_pris`(변형 표준편차, floor 5) + **셀 5c 단일 TM 캘리브레이션**.
탈소듐 순서(gap_074, gap_052)는 게이트가 아니라 **진단 기록**: x=0.52는 Zhao 만충전점(110 mAh/g ≈ Δx 0.48)이라 실험도
HEO가 P3이고, x=0.74 순서는 ORB(−4.9)·MACE(−5.2)가 일치 판독 → 0 K PBE 열역학에 없는 신호(전이 지연 = 엔트로피/
운동학 기원, npj Comput. Mater. 2025 doi:10.1038/s41524-025-01954-2)로 판정. 절대 상 판정은 어느 x에서도 비요구.
마일드 지점: Na 7개 제거(x=20/27≈0.74, `N_NA_REMOVE_MILD`, 태그 `{ph}_mild_{k}`, salt 200+k).
(옛 Ti₀.₃ 앵커는 삭제: "x=0.52에서 O3 유지"는 무근거였고 Wang Adv. Mater. 2017의 가역 O3−P3ㆍ실제 ORB 실행 결과와 모순.)
**앵커는 항상 검증 전용, 학습(파인튜닝)에 넣지 않는다.**

---

## 4. 노트북 구조 (35셀, 위→아래 Run All)

`CELL N`은 안정 식별자이고, `## Stage N` 헤더는 **실행 순서**입니다(설계 스펙의 단계 번호와 다름 — 스펙 8은
실행에서 Stage 5 앵커와 Stage 11 불변량으로 갈라짐). 앵커가 스크리닝 앞에 있는 것은 의도(문지기)이며 뒤로 옮기면 안 됩니다.

```
CELL 1  CONFIG (단일 진실 원천, 환경변수 오버라이드, LIMIT)   CELL 2  imports
CELL 3  Stage 1 조성 열거 → results/compositions.csv (+md5)   [옛 필터 재현 7,279 / 정확식 7,073]
CELL 4  Stage 2  O3 템플릿, 글자 분류기, 적층 검증기, P3 glide 자동 선택, Na–O 거리 assert
CELL 5  Stage 2  TM 배치(icet SQS / random 폴백), JSON 캐시, 탈소듐(O3·P3 같은 빈자리 패턴), build_set
CELL 6  Stage 3  엔진: torchsim(기본, v1 검증 패턴) | ase | mock; 샤드별 ckpt jsonl; 스트리밍 run_many
CELL 7  Stage 3  Na(bcc) 참조, 같은 모델, −1.6~−1.0 eV assert
CELL 8  Stage 4  metrics_from(), judge_phase()
CELL 9  Stage 5  앵커 4셀 → 2변형 평균 → 불변량 5 assert → RULER   ← 실패 시 스크리닝 중단
CELL 9b Stage 5b HEO_PARALLEL=1이면 GPU당 subprocess로 Pass1+2 실행·대기 → ckpt 재로드 (이후 셀은 캐시 히트)
CELL 10 Stage 6  Pass 1 (pristine O3/P3)
CELL 11 Stage 7  hull: MP 단일 쿼리 → 같은 모델 재이완 → Ehull_eff, T*
CELL 11b Stage 8 (선택) 대조 모델 dE_desod 로드 → DE_B      ← 반드시 상 판정 **앞**
CELL 12 Stage 9  Pass 2 (생존자 탈소듐) → δ=max(σ_vac,|dE_A−dE_B|,floor) → phase_desod → dV_nat
CELL 13 Stage 10 용량, ruler_pos, rank-sum 점수
CELL 14 Stage 11 강건성(δ×2, 대조, legacy), 불변량 7개, audit.json, results_v2/top30_v2.csv
CELL 16 Stage 12 top-N 재이완·CIF·confidence
CELL 17 Stage 13 tar → rclone → webhook → runpodctl/REST 종료
```

> **옛 CELL 15 폐지(2026-09-01).** 대조 모델 병합이 CELL 14의 출력 기록 *뒤에* 있어서 δ를 갱신해도
> `phase_desod`·`suppressed`·채점·저장 표·내보낸 CIF에 전혀 반영되지 않았습니다(가짜 대조 200 meV/f.u.를 물려도
> `top30_v2.csv`가 바이트 동일). 로드를 **CELL 11b**(판정 전)로 옮기고, 불일치 통계와 `contrast_top30_overlap`은
> CELL 12/14로 이전했습니다. 삭제가 아니라 이전입니다.

주요 환경변수: `HEO_MODE`(mock|orb|mace) `HEO_WORKDIR` `HEO_LIMIT`(0=전체) `HEO_PARALLEL` `HEO_NGPU` `HEO_SHARD/HEO_NSHARDS`
`HEO_ENGINE`(torchsim|ase) `HEO_BATCH`(256) `HEO_FMAX`(0.05) `HEO_MAXSTEPS`(300) `HEO_STOP_AFTER=pass2`(샤드 프로세스용)
`HEO_TERMINATE_MODE`(auto|stop|delete|none) `HEO_RCLONE_REMOTE` `HEO_WEBHOOK` `HEO_CONTRAST_RESULTS` `HEO_LEGACY_TOP30` `MP_API_KEY`

---

## 5. 불변량 (`assert`, 위반 시 즉시 중단 — 제거 금지)

1. `V_avg > 0` 전 조성
2. `|dE| < 500 meV/f.u.` (단위 누락 탐지)
3. Scheme A의 `k_B·T·sconf_per_atom` ∈ [42, 52] meV/atom
4. phase_desod==P3 조성의 다수에서 `dV_trans ≥ dV_O3`
5. 앵커 게이트 v4 (2026-09-03): (a) HOST V∈[2.5,3.4] (GGA 0.3–0.5 V 과소평가, Aydinol PRB 1997; 실측 ORB 2.69 V),
   (b) HOST phase_desod==P3 (x=0.52), (c) pristine 방향 `dE_pris(HEO) − dE_pris(HOST) > margin_pris`
   (실측 ORB +20.0, MACE +23.4). gap_074·gap_052는 진단 기록만(실측 ORB −4.9/−12.1, MACE −5.2/−12.6 — 두 모델 일치
   → 0 K 열역학 밖 신호, 논문 내용).
5b. 셀 5c 캘리브레이션: NaCoO₂/NaNiO₂/NaMnO₂ 전부 `dE_pris>0`, `dE_pris>dE_mild>dE_desod` 단조,
   NaCoO₂ `dE_desod<0`(x=0.52 P3), |dE|<500. 실패 시 스크리닝 중단.
6. 체크포인트의 모델 태그가 단일 (hull 화폐 == 스크리닝 화폐)
7. compositions.csv md5 불변

mock 실행 기대값: 7,073 (A 3,773 / B 2,982 / C 318) / 7,279(옛 필터) / 엔트로피 A 46.0, B 47.5, C 46.4 /
앵커(v3 워커, 2026-09-03 검증): HOST V 2.743 dE_desod −42.50 P3, HEO V 2.956 dE_desod +9.11 O3, gap_052 +51.6 /
gap_074 +33.5 (HOST dE_mild −17.17, HEO +16.37) / **게이트 v4**: gap_pris +9.9 > margin_pris 5.0 /
캘리브레이션(셀 5c): NaCoO₂ dE_pris +34.7 dE_mild +0.8 dE_desod −20.5 x_cross 0.733, NaNiO₂ +27.9/−4.4/−17.8/0.776,
NaMnO₂ +27.4/−14.7/−44.2/0.831, V_avg 2.65–2.71 (결정론, 비트 동일해야 함; mild·캘리브레이션 추가 후에도
기존 0.52 값 비트 동일 확인됨). [v2 노트북 mock의 옛 기대값: HOST −37.1, Ti03 +75.9 — Ti03 앵커 폐지로 무효]
`compositions.csv` md5 = `0fc86c95929b0c775d7ed0505cafc76a` (7,073행 기준).

---

## 6. 미검증 지점 4곳 — 실제 패키지로 먼저 확인할 것

| # | 위치 | 가정 | 확인 방법 |
|---|---|---|---|
| 1 | CELL 6 ORB 로더 | `pretrained.orb_v3_conservative_inf_mpa(device=)`가 (model, adapter) 튜플 또는 단일 객체; `ORBCalculator(model, [atoms_adapter=], device=)` | `pip install orb-models`, 반환형과 시그니처 출력 |
| 2 | CELL 6 torch-sim | `ts.optimize(..., optimizer=ts.optimizers.Optimizer.fire, init_kwargs={"cell_filter": CellFilter.frechet}, autobatcher=True)`; `final.energy`, `final.forces`, `final.system_idx`/`batch`, `ts.io.state_to_atoms` | `pip install torch-sim-atomistic==0.6.*`; Lennard-Jones 모델로 2구조 배치 이완 후 속성명 확인. `_fmax_per_structure`가 None을 돌려주면 conv 판정이 무력화됨 |
| 3 | CELL 5 icet | `generate_sqs_from_supercells(cs, supercells=[sc], target_concentrations=conc, n_steps=, random_seed=)`; 반환 구조의 **자리 순서 보존** | `pip install icet`; 27자리 3종으로 실행, 반환 Atoms의 Na 인덱스가 `NA_IDX`와 같은지 |
| 4 | CELL 11 mp-api | ~~exclude_elements 단일 쿼리~~ **검증 결과 가정 틀림(2026-09-03)**: `exclude_elements`는 서버 측에서 **문자열 60자 제한**(HTTP 422). 2단계로 교체: ① `search(is_stable=True, num_elements=(1,7), fields=["material_id","elements"])` 경량 쿼리(~40k행) ② 로컬에서 원소집합 ⊆ 풀 필터 ③ 해당 id만 `material_ids=` 200개 청크로 `structure` 수신 | 수정 반영됨(v3 셀 8). 실행 시 풀 내 항목 수(~2,000 예상)와 캐시 생성 확인 |

각 확인 후: 실제 시그니처와 다르면 해당 셀을 완성본으로 다시 제시하고, mock 실행이 여전히 §5 기대값을 내는지 재확인.

---

## 7. 검토 과제 (순서대로)

1. **mock 전체 실행**: `HEO_MODE=mock HEO_TERMINATE_MODE=none jupyter nbconvert --to notebook --execute heo_screening_v2.ipynb` → §5 기대값 대조.
2. **병렬 경로**: `HEO_PARALLEL=1 HEO_NGPU=2 HEO_LIMIT=300 python heo_screening_v2.py` (nbconvert --to script 후) → 두 샤드 ckpt 생성, 병합 시 "0 structures to relax" 확인.
3. §6의 4곳 실제 패키지 검증.
4. **셀별 독해 검토** — 각 셀에 대해 (a) §3 수식과 일치하는가, (b) 단위(eV/cell vs meV/f.u.)가 경계에서 바뀌는 지점이 명시적인가, (c) 데이터 누수/입력 덮어쓰기 없는가, (d) 샤드 간 경쟁 조건(파일 쓰기)이 없는가.
5. 특히 의심해 볼 곳:
   - CELL 3 `assign_charge` 탐욕 배분이 `feasible_exact`와 동치인지 (허용 산화수가 연속 정수라는 가정에 의존)
   - CELL 4 `stacking_report`의 gap 판정 허용오차(0.06)와 `layers_by_z` tol(0.02)이 이완 후 구조에도 안전한지 (현재는 템플릿에만 적용)
   - CELL 5 `build_set`에서 P3 탈소듐이 O3와 같은 `na_pattern`을 쓰는지 (짝지은 비교의 핵심)
   - CELL 8 `sigma_vac`가 O3·P3 중 큰 쪽인지, δ 단위가 meV/f.u.인지
   - CELL 11 `ehull_for`가 조성 원소의 부분집합 경쟁상만 쓰는지; `PDEntry` 에너지가 셀 총에너지(원자당 아님)인지 — pymatgen은 내부에서 원자당으로 정규화함
   - CELL 13 `capacity_mAh_g`: redox 저장고에 Cu/Co/V 기여를 넣을지는 미결 — 현재는 Ni(2e⁻)−n_Ni3+n_Mn3+n_Co2
   - CELL 14 강건성 `top_ids`가 CELL 13의 점수 정의와 동일한 방향인지 (`ddE` 내림차순)
   - CELL 16 top-N 재이완이 캐시 에너지와 일치하는지(결정론) — 실전에서 첫 5개로 확인
6. 결과 보고 형식: 문제 → 근거(어느 원칙/수식 위반) → 수정 셀 완성본 → mock 재실행 결과.

---

## 8. 확정된 결정 (되돌리려면 사용자와 논의)

- x 격자: 스크리닝 2점(1.0, 0.52), top 후보만 4점(1.0/0.85/0.67/0.52) — 후자 미구현
- 빈자리: 무작위 5샘플 + σ 기록, top 후보만 Kaufman(PRM 2019) 질서 패턴 추가 — 후자 미구현
- δ: 조성별 `max(σ_vac, 모델 불일치, 5 meV/f.u.)`
- 모델: ORB-v3 conservative **mpa**(DFT 앵커 화폐 일치). v1은 **omat**이었음 — v1 top30과 직접 비교하려면 omat 대조 실행 (WORKDIR 분리 필수)
- 대조 모델: MACE-MPA-0 (다른 아키텍처 + 같은 화폐)
- 앵커 셀: 2변형 평균
- Ehull_eff 컷: 0.10 (분포 본 뒤 재조정 가능)
- 파인튜닝: base 실행 → DFT 앵커/Tier1 → 연화 계수·부호 일치 측정 → 필요 시 (앵커는 학습 제외, 조성 단위 홀드아웃)
- 정확 전하중성으로 Scheme B 524개 제외: 사용자 최종 확인 대기 (`USE_EXACT_NEUTRALITY`)
- **앵커 재설계 (2026-09-03, 사용자 승인)**: Ti03 앵커 삭제(출처 부재 + Wang Adv. Mater. 2017의 가역 O3−P3 보고와 모순;
  실제 ORB 실행도 Ti03→P3 −31 meV/f.u.로 문헌과 일치). 양성 앵커 = Zhao Angew. 2020 HEO(§3), 게이트 = 방향형(A안).
  전압 창 [2.7,3.4]→[2.5,3.4]. `OXI`에 Fe(3,) 추가(앵커 전용, 열거 md5 불변 확인). ruler_pos = 0=HOST, 1=HEO.
- **게이트 지점 x=0.52→0.74 이동 (2026-09-03, 사용자 승인)**: 첫 실측 ORB에서 gap_052 = −10.5 (HEO −31.1 / HOST −20.6,
  둘 다 P3). 이는 Zhao 실험과 모순이 아님: 110 mAh/g ÷ 229 mAh/g(Na 1개당) ⇒ 만충전 x≈0.52, 용량 60%가 O3 ⇒ 전이 x≈0.71,
  즉 실험도 x=0.52에서 HEO는 P3이고 그 지점의 dE 순서는 비구속. 방향 게이트를 실험이 두 앵커를 구분하는
  x = 20/27 ≈ 0.74(Na 7 제거)로 이동: worker에 `N_NA_REMOVE_MILD=7`(`HEO_NNAREMOVE_MILD`), `build_items/tags_for("mild")`,
  태그 `{ph}_mild_{k}`(salt 200+k; "desod" 부분문자열 금지 — `metrics_from` 오염 방지), mock은 n 선형 보간
  (n=13 분기는 기존 값 비트 동일 유지). 절대 상 판정은 진단 출력만(방향형이 계통 오차를 상쇄하므로 문턱 완화 불필요).
  ruler.json에 dE_mild_HOST/HEO·x_mild·gap·margin·gap_052 저장; ruler_pos는 여전히 x=0.52 앵커 기반이라
  gap_052≤0이면 분모 부호 반전 — 좌표로만 읽을 것(셀 5가 경고 출력). 스크리닝 x 격자는 불변(§8 첫 항목).
- **게이트 v4 + 0 K 캘리브레이션 (2026-09-03, 사용자 승인)**: x=0.74 실측에서 ORB gap −4.9 / MACE gap −5.2 (margin 9~13,
  둘 다 HOST를 x=0.74에서 O3로 판독 — Komaba와 불일치). 서로 다른 아키텍처·같은 화폐의 두 모델이 1 meV 안쪽으로 일치
  → 원인은 모델이 아니라 공유물: 0 K PBE 열역학에 전이-지연 신호가 없음(엔트로피/운동학 기원, npj Comput. Mater. 2025).
  결정: (i) 탈소듐 방향 게이트를 **진단으로 강등**(gap_074·gap_052 기록, 논문의 음성 결과·300 K 후속 동기), (ii) 게이트를
  pristine 방향으로 이동(두 모델 +20.0/+23.4로 견고), (iii) **셀 5c 신설** — NaCoO₂/NaNiO₂/NaMnO₂(27×3가=+81 정확,
  SQS 불필요)를 같은 기계로 x=1/0.74/0.52 계산, 발표된 0 K DFT 사실(부호·단조·Co P3@0.52; Toumar PR Applied 2015,
  Kaufman PRM 2019, Delmas 1981)과 대조해 "MLIP→화폐" 충실성만 검증 — 스크리닝은 같은 화폐 내 랭킹이므로 이것으로 충분.
  정량 기울기(연화)는 표로 출력해 논문 그림과 수동 대조. 스크리닝은 "PBE-화폐 0 K 열역학 랭킹"으로 재규정, 화폐→실험
  연결은 top-30 DFT·유한온도 후속이 담당(§9 한계에 추가할 것). DFT 앵커 계산은 여전히 §8 파인튜닝 계획의 일부(연기).
- **Run All 안전장치 (2026-09-03)**: 런처는 백그라운드 subprocess라 즉시 반환 → 비대기 전체 실행 시 부분 집계 위험.
  셀 7b/10b 배리어 신설(`monitor(n, wait=True)`, 워커 전원 정상 종료까지 블로킹) + 셀 8에 pass-1 커버리지 ≥98% assert,
  셀 11의 완결성 assert를 생존자 대비 ≥98%로 강화. 모니터 셀(7/10)은 수동 새로고침용으로 그대로 유지.
- **자동 종료 비활성 (2026-09-03, 사용자 요청)**: v3 노트북 기본 `HEO_TERMINATE_MODE=none`. 워치독은 계속 돌되
  **로그 전용** — `watchdog.log`에 "LOG-ONLY - would terminate now: <사유>"만 남기고 팟은 절대 내리지 않는다.
  팟은 사용자가 직접 stop (과금 주의). `stop`/`delete`를 export 하면 종전 동작 복원.
- **파인튜닝 A/B 대조 (2026-09-03)**: base(스톡 ORB) 실행 후 파인튜닝 실행은 별도 `HEO_WORKDIR` +
  `HEO_ORB_CKPT=<가중치>`(MODEL_TAG에 `+ft-<파일명>` 접미 → 불변량 6이 캐시 혼용 차단) +
  `HEO_BASE_RESULTS=<base results_v3.csv>` → **CELL 12b**가 부호 일치율·연화 기울기(slope_vs_base)·
  Spearman·phase 혼동 행렬·suppressed/top-30 겹침·앵커 gap 변화를 `results/compare_base.{csv,json}`으로 저장.
  구조가 comp_rng 결정론이라 두 실행은 같은 입력 구조를 이완 — 차이는 순수 모델 차이.
  audit.json에 `run_label`(HEO_RUN_LABEL)·`orb_ckpt`·`worker_md5` 기록. ORB 파인튜닝 체크포인트 로더는
  state_dict/lightning 래퍼를 시도하는 일반형 — **실제 체크포인트로 §6식 검증 필요(미검증)**.

---

## 9. 알려진 한계 (논문 한계 문단 후보, 코드 결함 아님)

- hull 경쟁상은 MP **안정상만** (준안정 경쟁상 누락, v1 동일)
- P3 템플릿은 이상 적층 (O'3/P'3 왜곡 변형체 없음)
- torch-sim 경로에서 confidence head 미출력 (top-N만 사후 single-point)
- 2점 x 격자, 무작위 빈자리 — 위 §8 후속 계획
- uMLIP 연화(Deng et al. npj Comput. Mater. 2025): 에너지 차이 절대값 과소 가능 → 랭킹은 보존, 문턱은 앵커로 보정

---

## 10. 핵심 참고문헌 (설명 근거로 쓸 것)

- 전압: Aydinol et al. PRB 56, 1354 (1997); Urban, Seo, Ceder npj Comput. Mater. 2, 16002 (2016)
- 산화수: Walsh et al. Nat. Mater. 17, 958 (2018); Raebiger, Lany, Zunger Nature 453, 763 (2008); Sit et al. Inorg. Chem. 50, 10259 (2011)
- 전하중성 열거: Davies et al. Chem 1, 617 (2016) (SMACT)
- 안정성 두 축: Bartel J. Mater. Sci. 57, 10475 (2022)
- 엔트로피 안정화: Rost et al. Nat. Commun. 6, 8485 (2015); HE 층상 Na 양극(양성 앵커): Zhao et al. Angew. 59, 264 (2020), doi:10.1002/anie.201912171
- Ti 치환 O3−P3 가역 전이(Ti03 앵커 폐지 근거): Wang et al. Adv. Mater. 29, 1700210 (2017)
- Na 빈자리 질서: Kaufman & Van der Ven PRM 3, 015402 (2019)
- uMLIP 연화·파인튜닝: Deng et al. npj Comput. Mater. 11, 9 (2025); Hänseroth et al. JPCL 17, 3152 (2026); Liu et al. J. Appl. Phys. 139, 041101 (2026)
- 배터리 uMLIP 벤치: arXiv 2601.10938 (2026); ORB-v3: arXiv 2504.06231

---

## 11. 파일 배치

```
repo/
  CLAUDE.md                      ← 이 문서
  heo_screening_v2.ipynb         ← 파이프라인 (nbconvert --to script 로 .py 생성해 diff/grep에 사용)
  heo_v2/ (HEO_WORKDIR)
    results/  compositions.csv(.md5)  anchors.csv  results_v2.csv  top30_v2.csv  audit.json  E_Na.json
    ckpt/     energies_shard{k}.jsonl  hull_entries.json
    structures/ sqs/{comp_id}.json  relaxed/{comp_id}__{tag}.cif
```
