# heo_v3.md — 2026-09-03 세션 정리: 앵커 게이트 재설계와 0 K 검증 체계

이 문서는 v3 노트북 첫 실전(RunPod, RTX 5090 ×6) 과정에서 내린 **방법론적 결정과 그 근거**를 정리한 것이다.
운영성 트러블슈팅(MP API 파라미터 제한 등)은 CLAUDE.md §6에 반영했으므로 여기서는 다루지 않는다.

---

## 1. 한 줄 요약

실전 ORB에서 앵커 게이트가 두 번(x=0.52, x=0.74) 실패했고, MACE 대조로 그 원인이 **모델이 아니라 관측량**임을
확인했다 — Zhao HEO의 O3→P3 전이 지연은 0 K PBE 열역학에 들어 있지 않은 신호(엔트로피/운동학 기원)다.
게이트를 0 K가 지지할 수 있는 형태(pristine 방향 + 문헌 DFT 캘리브레이션)로 재정의했고(v4),
스크리닝은 **"PBE-화폐 0 K 열역학 랭킹"**으로 재규정했다. 실험 연결은 top-30 DFT와 유한온도 후속 연구가 담당한다.

핵심 사슬: `MLIP ──①──> PBE 계열 DFT(화폐) ──②──> 실험(300 K + kinetics)`
스크리닝(같은 화폐 안에서 7,073개 랭킹)이 의존하는 것은 ①뿐이다. 이번 세션의 결론은
"②의 탈소듐 구간이 끊어져 있으므로, 게이트는 ①만 검증해야 한다"는 것.

---

## 2. 게이트 변천사 (전부 사용자 승인)

| 버전 | 게이트 | 결과 | 폐기/강등 근거 |
|---|---|---|---|
| v1 | Ti03 절대 앵커 | (세션 전) 폐기 | 출처 부재, Wang Adv. Mater. 2017과 모순 |
| v2 | 방향형 @ x=0.52: `dE_desod(HEO) − dE_desod(HOST) > margin` | **실패** gap −10.5 (margin 7.5) | Zhao 만충전 = x≈0.52 (110 mAh/g ÷ 229 mAh/g per Na ⇒ Δx 0.48) → 실험도 그 지점에선 **둘 다 P3**. 순서는 실험 비구속 |
| v3 | 방향형 @ x=0.74 (`N_NA_REMOVE_MILD=7`, x=20/27) | **실패** ORB gap −4.9 (margin 12.9), MACE gap −5.2 (margin 9.0) | 두 아키텍처가 1 meV 안쪽 일치 → 모델 노이즈 아님. 신호 자체가 0 K 열역학에 없음 |
| **v4 (현행)** | (a) HOST V∈[2.5,3.4] (b) HOST P3@0.52 (c) pristine 방향 `dE_pris(HEO) − dE_pris(HOST) > margin_pris` (d) 셀 5c 캘리브레이션 | **통과** gap_pris +20.4 (margin 5.0) | 화폐가 실제로 분해하는 방향 (ORB +20.0/+20.4, MACE +23.4로 재현) |

x=0.74를 골랐던 근거(v3): Zhao의 "용량 60%+가 O3" ⇒ HEO 전이 x≈0.71, HOST는 훨씬 이른 x에서 gliding(Komaba 2012)
→ x≈0.74가 실험이 두 앵커를 구분하는 유일한 창. 거기서도 방향이 안 나온 것이 결정적 진단이었다.

---

## 3. 실측 수치 기록 (108원자 셀, meV/f.u.)

### ORB-v3 conservative mpa (2회 실행, GPU 비결정성 ±수 meV 재현)

| x | HOST dE | HEO dE | gap (HEO−HOST) |
|---|---|---|---|
| 1.00 (pris) | +118.9~121.3 | +139.0~144.6 | **+20.0~+20.4** ✓ |
| 0.74 (mild) | +51.6~+63.2 | +51.7~+58.0 | −4.9 ~ +1.0 (≈0) |
| 0.52 (desod) | −11.9~−20.7 | −24.6~−34.8 | −10.5 ~ −20.4 |

### MACE-MPA-0 대조 (다른 아키텍처, 같은 화폐)

gap_pris +23.4 / gap_074 **−5.2** / gap_052 −12.6 — ORB와 1 meV 안쪽 일치.

### 해석

- 모델 내부 전이점(선형 보간): HOST x*≈0.58, HEO x*≈0.61. 실험(≈0.77 / ≈0.71) 대비 **둘 다 지연**(uMLIP 연화로
  설명되는 계통 오차) + **순서 미세 역전**(신호 ~5 meV/f.u. < 노이즈 8~13).
- `n_phase_flip = 0` 확인 → mild의 +50대 값은 P3 셀이 이완 중 O3로 되돌아간 인공물이 아니라 진짜 짝지은 비교.
- HOST 전압 2.67~2.74 V (실험 3.1 V, GGA 계열 0.3–0.5 V 과소평가와 일치, Aydinol PRB 1997). E_Na −1.30~−1.31 eV.
- 문헌 근거: npj Comput. Mater. 2025 (doi:10.1038/s41524-025-01954-2) — 고엔트로피 층상 산화물에서 적층상은
  "열역학보다 운동학이 지배"; 우리 결과는 이를 독립 재현한 셈. **논문의 음성 결과 + 300 K 후속 연구 동기 문단 재료.**

---

## 4. 셀 5c — 자 캘리브레이션 (신설)

새 DFT 없이 ①(MLIP→화폐 충실성)을 검증하는 장치. NaCoO₂/NaNiO₂/NaMnO₂는

- 27 × M³⁺ = +81로 전하중성이 **정확히** 맞고 (우리 기계 그대로 사용),
- 단일 원소라 SQS 불필요 (place_tm_random, 몇 분 계산),
- O3/P3 x-의존성의 0 K GGA(+U) 정답이 발표되어 있다
  (Toumar et al. PR Applied 4, 064002 (2015); Kaufman & Van der Ven PRM 3, 015402 (2019); Delmas 1981 O3→O′3→P3).

각 화합물을 x = 1 / 0.74 / 0.52에서 O3/P3 계산. **assert는 확실한 문헌 사실만**:
① 셋 다 dE_pris > 0, ② dE_pris > dE_mild > dE_desod 단조, ③ NaCoO₂ dE_desod < 0 (x=0.52 P3), ④ |dE| < 500.
정량 표(`results/calibration.csv`: dE 3점, V_avg, x_cross)는 Toumar/Kaufman 그림과 **수동 대조** — 기울기가
곧 연화 계수 실측값(파인튜닝 평가 기준선).

---

## 5. 코드 변경 목록

### heo_worker.py
- `N_NA_REMOVE_MILD = 7` (`HEO_NNAREMOVE_MILD`), x = 20/27 ≈ 0.741
- `vacancy_pattern(struct, rng, n_remove=N_NA_REMOVE)` 파라미터화
- `build_items/tags_for`에 `which="mild"` 경로. 태그 `{ph}_mild_{k}`, salt 200+k.
  **태그에 "desod" 금지** — `metrics_from`이 부분문자열로 desod를 수집하므로 n=7/13 오염 방지
- mock: dE를 제거 Na 수에 선형 보간, 단 n=13 분기는 기존 식 유지 → **기존 mock 기대값 비트 동일**

### 노트북 (heo_na_runpod_v3.ipynb, 39셀)
- **셀 5 (게이트 v4)**: pristine 방향 assert, gap_074/gap_052는 진단 출력·ruler.json 기록만.
  셀 주석에 v1→v4 변천사 전부 보존. `margin_pris` = 변형 2개 std(ddof=0, = 반차) clip ≥5
- **셀 5c 신설**: §4의 캘리브레이션 (66구조)
- **셀 7b/10b 배리어 신설**: 런처가 백그라운드 subprocess라 Run All 시 부분 집계 위험 →
  `monitor(n, wait=True)`로 워커 전원 정상 종료까지 블로킹 (비정상 종료 시 assert)
- **셀 8**: pass-1 커버리지 ≥98% assert (순서 실수 이중 방어; 98%인 이유는 relax_batch가 개별 실패 구조만 드랍)
- **셀 11**: 완결성 assert를 생존자 대비 ≥98%로 강화
- **셀 12b (파인튜닝 A/B 대조) 확장**: `mean_shift`(부호 있는 계통 이동), `phase_flips`(부호 vs δ 원인 분해),
  `dE_desod_by_scheme`(A/B/C별 일치율·기울기), `top30_score_overlap`(실제 rank-sum top30_v3.csv 겹침 —
  DFT행 목록이 몇 개 바뀌는지), `anchor_gap`에 HOST/HEO dE·E_Na 상세

### ruler.json 스키마 (v4)
`dE_pris/desod/mild_HOST/HEO`, `x_mild`, `gap`(=gap_pris)·`margin`, `gap_074`, `gap_052`, `E_Na`, `model`

---

## 6. mock 기대값 (결정론, 비트 동일해야 함 — CLAUDE.md §5와 동일)

- 앵커: HOST V 2.743 dE_desod −42.50 P3 / HEO V 2.956 dE_desod +9.11 O3 / gap_052 +51.6 / gap_074 +33.5 /
  **gap_pris +9.9 > margin_pris 5.0**
- 캘리브레이션: NaCoO₂ +34.7/+0.8/−20.5 (x_cross 0.733), NaNiO₂ +27.9/−4.4/−17.8 (0.776),
  NaMnO₂ +27.4/−14.7/−44.2 (0.831), V_avg 2.65–2.71
- mild·캘리브레이션 추가 후에도 기존 0.52 값 비트 동일 확인됨

---

## 7. 유의사항·미결

- **ruler_pos 주의**: 셀 11의 ruler_pos는 여전히 x=0.52 앵커 기반인데 gap_052 < 0이라 분모 부호가 반전 —
  "1에 가까울수록 좋음"이 아니라 좌표로만 읽을 것 (셀 5가 경고 출력)
- **본 실행 후 확인할 것**: 후보들의 ddE 분포 폭. 앵커의 교훈이 "~10 meV/f.u. 차이는 순서 판별 불가"이므로
  top-30이 그 폭 안에 몰려 있으면 상위권은 통계적 동률로 읽어야 함 (δ×2 강건성 지표로 정량화)
- **캘리브레이션 정량 대조**: `calibration.csv`를 Toumar 2015 / Kaufman 2019 그림과 눈으로 대조 (미완)
- **DFT 앵커** (연기, §8 파인튜닝 계획의 일부): GGA+U로 앵커 24구조 계산 시 "MLIP 오차 vs 화폐 한계"가 최종 분리됨.
  화폐 자체가 gap≈0이면 GGA+U 파인튜닝으로는 탈소듐 방향 게이트를 영원히 통과 못 함 — 그래서 v4에서 뺀 것
- **논문 한계 문단에 추가할 것**: "운동학적/유한온도 전이 지연은 이 스크리닝의 범위 밖" + §3의 두 모델 일치 데이터
- **자동 종료**: 기본 `HEO_TERMINATE_MODE=none`(로그 전용 워치독). 복원은 셀 1 실행 전
  `os.environ["HEO_TERMINATE_MODE"]="stop"`(권장) 또는 `"delete"` — 워치독은 셀 1b 시점 값으로 고정되므로
  이미 떠 있으면 커널 재시작 후 처음부터 (체크포인트로 전부 캐시 히트)
- **파인튜닝 A/B 실행법** (CLAUDE.md §8): 별도 `HEO_WORKDIR` + `HEO_ORB_CKPT` + `HEO_BASE_RESULTS` → 셀 12b가
  `results/compare_base.{csv,json}` 생성. 앵커는 학습 제외(검증 전용) 원칙 유지 — 게이트가 파인튜닝의 검증 세트가 됨
