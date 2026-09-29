# CHANGE_SPEC_v4 — dV 정밀화 + Na 조성 4-point 확장

> 대상: heo_na_runpod_v3.ipynb + heo_worker.py (v3 파이프라인)
> 실행자: Claude Code. 작업 전 저장소 루트의 CLAUDE.md를 먼저 읽고 규약(불변량 7개, MODEL_TAG, 캐시 구조, 짝지은 빈자리 원칙)을 확인할 것.
> 원칙: 이 변경은 **딱 두 가지**만 추가/교체한다. 채점(SCORE_TERMS, rank_sum, 관문 3개)의 정의는 절대 건드리지 않는다.

---

## 0. 범위 선언

| ID | 변경 | 종류 |
|----|------|------|
| A | dV 계산을 a²c 로그미분 근사 → 격자벡터 삼중곱(det) 정확식 + 분해(dA, dh, von Mises)로 교체 | 버그 수리급 교체 |
| B | Na 조성 2점(x=1, 0.52) → 4점(x=1, 0.81, 0.67, 0.52) 확장 + O3→P3 전이 위치(x*) 판정 | 기능 추가 |

**하지 않는 것 (명시적 금지):**
- SCORE_TERMS 변경 금지 (ddE / Ehull_eff / dV_nat 3항 유지, 방향 유지)
- 관문(Ehull_eff<0.10, dE_pris>+5, suppressed) 정의 변경 금지
- 기존 불변량 1–7 제거/완화 금지 (추가만 허용)
- δ 정의 변경 금지 (단, x별로 각각 산출하는 것은 허용 — B-4)
- 기존 결과 캐시를 덮어쓰는 실행 금지 (신규 WORKDIR 사용 — §5)

---

## 1. 변경 A — dV 계산 교체

### A-1. 현행 문제 (교체 사유, 코드 주석에 요약해 넣을 것)

현행 dV/V ≈ 2·(Δa/a) + (Δc/c)는 두 가정 위의 근사다:
(i) 셀 각도 고정(γ=120°, α=β=90° 유지), (ii) 소변형 선형화.
Frechet cell filter가 셀 6자유도를 모두 풀기 때문에 탈소듐 셀에서 (i)이 깨질 수 있고
(각도 1–2° 드리프트만으로 % 스케일 오차), 13/27 탈소듐의 큰 변형에서 (ii)도 흔들린다.
또한 스칼라 dV는 면내 수축(−)과 층간 팽창(+)이 상쇄되어 구조 요동의 실체를 가린다.

### A-2. 새 정의

정확 부피: V = |det L| (L: 3×3 격자 행렬, 행 = a, b, c)
면내 넓이: A_par = |a × b|
수직 층고: h_perp = V / A_par  ← c 벡터 길이가 아님. 기울기 면역.
층간 간격: d_inter = h_perp / 3 (3-슬랩 셀 규격)
변형 텐서: F = L_desod^T (L_pris^T)^{-1},  E = (F^T F − I)/2
  - tr(E) ≈ 부피부, 편차부 von Mises 크기 = 모양 변형(붕괴 감지)
  - F^T F 구성이므로 이완 중 셀 회전은 자동 소거됨

구현 (신규 유틸, heo_worker.py에 추가 — 완전 자족 함수):

```python
import numpy as np

def cell_metrics(cell):
    """cell: 3x3 array-like, rows = a, b, c (Angstrom).
    Returns exact volume, in-plane area, tilt-immune perpendicular height."""
    cell = np.asarray(cell, dtype=float)
    a, b, _c = cell
    V = abs(np.linalg.det(cell))
    A_par = np.linalg.norm(np.cross(a, b))
    h_perp = V / A_par
    return V, A_par, h_perp

def dV_report(cell_pris, cell_desod, n_slabs=3):
    """Volume change decomposition between paired cells.
    Requires lattice-vector correspondence (guaranteed: P3 built from O3 by glide,
    desod built from pris by Na removal in the same cell)."""
    Vp, Ap, hp = cell_metrics(cell_pris)
    Vd, Ad, hd = cell_metrics(cell_desod)
    Lp = np.asarray(cell_pris, dtype=float).T
    Ld = np.asarray(cell_desod, dtype=float).T
    F = Ld @ np.linalg.inv(Lp)
    E = 0.5 * (F.T @ F - np.eye(3))
    vol_part = np.trace(E)
    dev = E - np.eye(3) * vol_part / 3.0
    vm = float(np.sqrt(2.0 / 3.0 * np.sum(dev * dev)))
    return {
        "dV_pct":      100.0 * (Vd - Vp) / Vp,
        "dA_pct":      100.0 * (Ad - Ap) / Ap,
        "dh_perp_pct": 100.0 * (hd - hp) / hp,
        "d_inter_pris": hp / n_slabs,
        "vm_strain":   vm,
    }
```

### A-3. 통합 규칙

1. dV_nat의 **분기 규칙(이긴 상의 desod 부피 사용)과 절대값 처리, 점수 항 지위는 그대로**.
   내부의 부피 산출만 dV_report 경유로 교체한다.
2. 기존 a²c 경로는 첫 검증 실행 1회에 한해 `dV_legacy` 컬럼으로 병기하여 회귀 비교에 쓰고,
   비교 리포트 확인 후 후속 커밋에서 제거한다 (사용자 삭제 승인 완료됨 — 이 명세가 승인 문서).
3. 신규 컬럼 (모든 상·상태 조합 및 §2의 모든 x에 대해):
   `dV_pct, dA_pct, dh_perp_pct, vm_strain, d_inter_pris` (+ 과도기 `dV_legacy`)

### A-4. 검산 (실행 후 자동 리포트에 포함)

- 항등식: (1 + dV/100) ≈ (1 + dA/100)(1 + dh/100), 잔차 > 0.1%p 인 행 = 계산 버그 후보로 목록화
- 각도 보존 셀(이완 후 γ≈120°±0.5°, α,β≈90°±0.5°)에서 |dV_new − dV_legacy| < 0.05%p 확인
- top-30 겹침률(legacy dV 기준 vs new dV 기준): 리포트만. 크게 다르면 그 자체가 발견(a²c 왜곡의 증거)
- d_inter_pris ∈ [4.8, 6.0] Å 창 밖이면 WARNING (셀 판독/슬랩 수 착오 탐지)

---

## 2. 변경 B — Na 조성 4-point + x* 판정

### B-1. x 그리드 (정수 정합 — 협상 불가)

108원자 셀, Na 자리 27개 기준:

| 태그 | Na 개수 | x = Na/27 | 제거 수 |
|------|---------|-----------|---------|
| x100 | 27 | 1.0000 | 0 (기존 pristine) |
| x081 | 22 | 0.8148 | 5 |
| x067 | 18 | 0.6667 | 9 |
| x052 | 14 | 0.5185 | 13 (기존 desod) |

- assert: Na 개수 ∈ {27, 22, 18, 14} 외 값 금지 (신규 불변량 8, §3)
- 각 x는 독립 평형 상태다. 빈자리 패턴을 x 간에 중첩(nested)시킬 의무 없음.
  단, **같은 x 안에서 O3/P3 동일 패턴(짝지은 빈자리) 원칙은 x마다 반드시 유지** (기존 원칙의 x별 적용).
- x별 무작위 5샘플, 시드 규약: 기존 시드 함수에 na_count를 섞어 재현 가능하게
  (예: seed = hash((comp_id, phase, na_count, sample_idx)) 방식 — 기존 규약에 맞춰 구현).

### B-2. 적용 범위 제어 (예산)

환경변수 `HEO_X4_SCOPE`:
- `"top500"` (기본값): 기존 v3 결과(HEO_BASE_RESULTS)의 score 상위 500 조성에만 중간 x 2점 추가
- `"survivors"`: 관문 통과 후보 전체
- `"anchors"`: 앵커 3종만 (파일럿 모드)
- `"all"`: 전 조성 (예산 경고 로그 출력)

추가 계산량 추정 (top500 기준): 500 × (x 2개) × (상 2개) × (샘플 5개) = 10,000 이완.
기존 Pass 2와 같은 배치 경로(torch-sim FIRE + Frechet cell filter, 동일 수렴 기준)를 재사용한다.

### B-3. x별 산출량

각 조성 × 각 x ∈ {x081, x067} (x100, x052는 기존 값 재사용)에 대해:

- dE(x) = [E(P3, x) − E(O3, x)] / 27  [meV/f.u.] — 샘플별 짝지은 차의 평균, σ_vac(x) = 짝지은 차의 표준편차
- δ(x) = max(σ_vac(x), 5.0) — 대조 모델 dE가 해당 x에 존재하면 |dE_A − dE_B| 항도 포함 (기존 δ 정의의 x별 적용, 정의 자체 변경 아님)
- verdict(x) = judge_phase(dE(x), δ(x)) ∈ {O3, AMB, P3} — 기존 judge_phase 함수 재사용 (신규 판정 함수 작성 금지)
- 부피: 변경 A의 dV_report(cell_pris_O3, cell_x)를 자연상 기준으로 — dV_nat(x) 정의는 x052와 동일 분기

### B-4. x* 국소화 알고리즘

입력: verdicts = [verdict(x100), verdict(x081), verdict(x067), verdict(x052)] (x 내림차순).
verdict(x100)은 기존 dE_pris 판정 재사용 (관문 2 통과 조성은 O3 보장).

```text
find i = smallest index with verdicts[i] != "O3"   # first departure from O3
if i does not exist:
    x_star_class = "suppressed_full"        # O3 survives the whole window
    x_star_lo, x_star_hi = None, 0.5185     # x* < 0.52 (window 밖)
elif verdicts[i] == "P3":
    if all(v == "P3" or v == "AMB" for v in verdicts[i:]):
        x_star_class = "transition"
        x_star_lo, x_star_hi = x[i], x[i-1] # x* is bracketed in (x[i], x[i-1])
        x_star_mid = (x[i] + x[i-1]) / 2
    else:
        x_star_class = "nonmonotonic"       # e.g., O3, P3, O3, ... — review flag
        x_star_lo, x_star_hi = None, None
elif verdicts[i] == "AMB":
    if all(v != "O3" for v in verdicts[i:]):
        x_star_class = "amb_boundary"       # departure is real but verdict is gray
        x_star_lo, x_star_hi = x[i], x[i-1]
    else:
        x_star_class = "nonmonotonic"
```

- "nonmonotonic"은 오류가 아니라 검토 플래그다 (빈자리 샘플링 잡음 또는 진짜 재진입 거동).
  개수를 요약 리포트에 집계하고, 5샘플 min-tag 대신 평균 dE로 재판정한 결과를 병기할 것.
- suppressed(기존 관문 3)의 의미는 불변: verdict(x052) == "O3". x*는 **추가 정보**이지 관문 대체가 아니다.

### B-5. V(x) 계단 곡선과 x-방향 형성에너지

자연상 에너지 정의: E_nat(x) = 그 x에서 verdict가 가리키는 상의 에너지
(AMB이면 min(E_O3, E_P3) 사용, 컬럼에 사용 상 기록).

구간 전압 (Δn = 5, 4, 4):
  V_seg(x_i → x_{i+1}) = [E_nat(x_{i+1}) + Δn·E_Na − E_nat(x_i)] / Δn
- assert V_seg > 0 (불변량 1의 구간별 확장, 신규 불변량 9)
- 기존 V_avg(1→0.52)와의 검산: Σ(Δn_i · V_seg_i) / 13 ≈ V_avg, 잔차 > 1 mV면 WARNING

x-방향 형성에너지 (양끝 기준 지렛대):
  Ef(x) = E_nat(x) − [ E_nat(x100)·(x−0.5185) + E_nat(x052)·(1−x) ] / (1 − 0.5185)
- Ef(x081), Ef(x067) 기록. Ef < 0 = 안정 중간상(경사 전압 구간), Ef ≥ 0 = 2상 공존 경향(평탄 구간).
- 볼록성 플래그: Ef 두 점의 부호 조합으로 {convex, concave, mixed} 분류만 기록 (해석은 사람 몫).

### B-6. 스키마 추가 (results CSV)

```
dE_x081, sigma_vac_x081, delta_x081, verdict_x081,
dE_x067, sigma_vac_x067, delta_x067, verdict_x067,
x_star_class, x_star_lo, x_star_hi, x_star_mid,
V_seg_100_081, V_seg_081_067, V_seg_067_052,
Ef_x081, Ef_x067, convexity_flag,
dV_pct_x081, dA_pct_x081, dh_perp_pct_x081, vm_strain_x081,
dV_pct_x067, dA_pct_x067, dh_perp_pct_x067, vm_strain_x067
```
(변경 A 컬럼은 x052/pristine 쌍에도 동일 적용. 기존 컬럼명은 하나도 변경하지 않는다 — 하위 호환.)

### B-7. 캐시/태그

- 캐시 키에 na_count 포함 필수 (x081/x067 결과가 x052 캐시와 충돌 금지) — 신규 불변량 10
- MODEL_TAG 규약 불변. 신규 실행은 `HEO_WORKDIR=heo_v4` 로 분리, 기존 heo_v3 산출물은 읽기 전용 입력(HEO_BASE_RESULTS)으로만 사용

---

## 3. 불변량 추가 (기존 1–7 뒤에 append, 번호는 저장소의 실제 최종 번호에 이어 붙일 것)

- **불변량 8**: 모든 구조의 Na 개수 ∈ {27, 22, 18, 14}. 위반 시 즉시 abort.
- **불변량 9**: 모든 V_seg > 0. (기존 V_avg>0의 구간별 강화)
- **불변량 10**: 캐시 키는 (comp_id, phase, na_count, sample_idx, MODEL_TAG)를 모두 포함.
- 짝지은 빈자리 assert(동일 패턴 O3/P3)를 x별로 적용 — 기존 불변량의 적용 범위 확장이므로 새 번호 불필요, 해당 assert에 na_count 루프만 추가.

---

## 4. 채점 불변 선언 (Claude Code가 지킬 것)

- score = rank(ddE, desc) + rank(Ehull_eff, asc) + rank(dV_nat, asc) — **정의·항·방향 모두 그대로**.
- dV_nat에 들어가는 수치만 A의 정확식으로 바뀐다 (분기 규칙 동일).
- x*, V_seg, Ef, vm_strain 등 신규 량은 전부 **보고 컬럼**이다. 점수·관문에 넣지 않는다.
  (점수 반영 여부는 top-30 이동량 리포트를 보고 사용자가 별도 결정.)

---

## 5. 실행 계획 (순서 고정)

1. **mock 모드 전체 체인** — 신규 코드 경로(dV_report, x 루프, x* 분류)가 mock에서 끝까지 도는지. 불변량 8–10 발화 테스트 포함(고의 위반 케이스 1개씩).
2. **HEO_X4_SCOPE=anchors 파일럿** — HOST / Ti0.3 / HEO 3조성 실계산.
   기대 결과(방향 시험의 4-point 확장):
   - HOST: 창 안에서 transition 등급, x_star가 실험 상 지도(Komaba 2012: O3→O′3→P3 순차 전이)와 정합하는 위치
   - Ti0.3: HOST보다 낮은 x_star 또는 suppressed_full
   - HEO: 지연된 전이 (Zhao 2020 서사)
   여기서 어긋나면 중단하고 사용자에게 보고 (스크리닝 확장 진행 금지).
3. **HEO_X4_SCOPE=top500 본 실행** — 완료 후 자동 리포트:
   - A 회귀 리포트 (§A-4 전체)
   - x_star_class 분포 (suppressed_full / transition / amb_boundary / nonmonotonic 개수)
   - V_seg 검산 잔차 분포
   - top-30 이동: legacy dV vs new dV 기준 겹침률
4. RunPod 자동 종료 훅 유지 (기존 규약).

---

## 6. 인수 기준 (전부 충족 시 완료)

- [ ] mock 전 체인 녹색 + 불변량 8–10 발화 테스트 통과
- [ ] 앵커 파일럿 3조성 방향 시험 통과 (§5-2 기대 결과)
- [ ] 항등식 잔차 > 0.1%p 행 비율 < 1%
- [ ] 각도 보존 셀에서 dV_new ≈ dV_legacy (< 0.05%p)
- [ ] 기존 컬럼명/기존 캐시 무손상 (heo_v3 디렉토리 해시 불변)
- [ ] 요약 리포트 md 자동 생성 (results/report_v4.md)

## 7. 주의/미검증 지점 (작업 중 확인 후 이 문서에 체크 표시로 갱신할 것)

- [ ] 탈소듐 함수가 na_count 파라미터화에 이미 대응하는지 (13 하드코딩 여부 grep: "13", "desod")
- [ ] judge_phase가 x별 δ를 인자로 받는 시그니처인지 (전역 δ 참조면 리팩토링 필요 — 정의 변경 없이 인자화만)
- [ ] 대조 모델(MACE) dE가 x081/x067에 없을 때 δ(x)가 max(σ_vac, 5)로 자연 강등되는지
- [ ] torch-sim 배치가 서로 다른 na_count(95/99/103원자) 혼합 배치를 허용하는지 (autobatcher 동작 확인)
- [ ] dV_legacy 병기 시 컬럼 수 증가로 인한 다운스트림(CELL 12b compare_base) 스키마 가정 파손 여부
