# XRD_SPEC — MLIP 이완 구조의 시뮬레이션 XRD (기존 결과에 후처리 컬럼 추가)

> 실행자: Claude Code. 저장소 루트 CLAUDE.md, CHANGE_SPEC_v4.md, HANDOFF_next_session.md를 먼저 읽을 것.
> 성격: **순수 후처리**. 이완 재실행 0, 채점 정의 변경 0. 캐시의 이완 구조를 읽어 XRD 패턴을 계산하고
> 결과 csv에 컬럼을 추가한다. 계산 비용은 CPU 분 단위.

---

## 0. 목적 (세 가지)

1. **실험 언어로 번역**: dA/dh 숫자를 (110)/(003) 피크 이동으로 — 실험자가 읽는 형태의 산출물
2. **검산**: d₀₀₃ = h_perp/3 이어야 하므로 XRD 피크 위치가 변경 A(det 분해)의 독립 검산기
3. **세 번째 상 판독기**: 글자 분류기(기하)·dE(에너지)에 이어 XRD 지문(회절)으로 O3/P3 판정 → 세 판독기 합의표

## 1. 함정과 처방 (반드시 적용)

**함정**: SQS 108원자 셀은 도펀트가 특정 배열로 앉아 있어, 그대로 계산하면 실제 무질서 고용체에는 없는
초격자(superlattice) 피크가 생긴다.
**처방 — 점유율 평균(virtual crystal)**: 이완 기하는 그대로 두고 종만 평균으로 교체.
- 모든 TM 자리 → 그 조성의 TM 분수 조성 (예: {Ni:6/27, Mn:6/27, Li:3/27, …})
- 모든 Na 자리 → {Na: x} (x = na_count/27; 빈자리는 점유율로 표현)
- O 자리는 그대로
→ 기하(glide, 층간 거리, 셀 왜곡)는 보존, 배열 초격자 피크만 소멸. pymatgen은 분수 점유 자리의
산란인자를 가중평균으로 처리한다.

```python
from pymatgen.core import Composition
def occupancy_average(struct, tm_counts: dict, na_count: int, TM_SET: set):
    s = struct.copy()
    tm_frac = Composition(tm_counts).fractional_composition.as_dict()
    x = na_count / 27.0
    for i, site in enumerate(s):
        el = site.specie.symbol
        if el in TM_SET:
            s.replace(i, tm_frac)
        elif el == "Na":
            s.replace(i, {"Na": x})
    return s
```
(TM_SET = 호스트 + 16종 도펀트 전체. 원자 위치는 변경 금지.)

## 2. 패턴 계산

```python
from pymatgen.analysis.diffraction.xrd import XRDCalculator
calc = XRDCalculator(wavelength="CuKa")          # 1.5406 Å — 실험(Komaba 2012)과 동일
pat  = calc.get_pattern(s_avg, two_theta_range=(10, 80), scaled=True)
```
- 브로드닝: 각 델타 피크에 pseudo-Voigt(η=0.5) 적용, FWHM = 0.15° (설정값, 물리 아님 — 리포트에 명시)
- 2θ 격자 0.01°로 연속 곡선 I(2θ) 생성, 최대 100 정규화
- 빈자리 샘플 5개가 있는 x에서는 5개 패턴을 평균 (`xrd_avg5`), k_best 단독 패턴도 별도 저장 (`xrd_kbest`)

## 3. 피크 추적 — 변경 A 분해와의 1:1 대응 (검산의 핵심)

supercell의 hkl 인덱스는 실험 육방정 셀과 다르므로 **면간거리 d로 대응**시킨다.
변경 A의 분해에서 유효 육방정 셀 상수를 계산:
- `a_eff = sqrt(2·A_par / (9·√3))`  (면내 3×3 초격자: A_par = 9·(√3/2)·a²)
- `c_eff = h_perp`  (3-슬랩 셀 높이 = 육방정 c)
예상 위치: `d_003 = c_eff/3`, `d_110 = a_eff/2` → `2θ = 2·asin(λ/2d)`
절차: 예상 2θ ±0.3° 창에서 계산 패턴의 최대 피크를 찾아 `tt_003`, `tt_110` 기록.
**검산**: |d_003(피크) − h_perp/3| < 0.005 Å 이어야 함. 위반 행 = 캐시 키 오류 또는 셀 판독 오류 → 목록화.

추가 추적: (104)류 — 예상 d에서 탐색, 세기 기록 (상 지문 보조).
단사정 왜곡 신호: gamma_pris/ab_dev가 임계 초과인 행은 (110) 분열 여부 플래그 (`xrd_split_110`).

## 4. 상 지문 — 세 번째 판독기

기준 지문: HOST(NaNi0.5Mn0.5O2) x=1의 이완 O3 구조와 P3 구조의 점유율 평균 패턴 (`ref_O3`, `ref_P3`).
각 구조 패턴과의 코사인 유사도(2θ 15–75°, 브로드닝 후 곡선)로:
- `xrd_sim_O3`, `xrd_sim_P3`, `xrd_phase = argmax` (마진 |sim_O3 − sim_P3| < 0.05 → "AMB")
- `xrd_agree_letter`: 글자 분류기 판정과 일치 여부
- `xrd_agree_energy`: dE 부호 판정(verdict)과 일치 여부 — 단 dE 판정은 "어느 상이 낮은가"이고
  XRD/글자는 "이 구조가 어느 상인가"이므로, 비교 대상은 **각 상 템플릿 구조가 이완 후에도 그 상으로 남았는가**
  (n_phase_flip과 같은 질문). 세 판독기 합의표는 (글자, XRD) 2판독기 + n_phase_flip 교차로 작성.
기준 지문의 격자는 조성마다 다르므로 유사도 계산 전 2θ 축을 d-공간 축으로 바꾸고 (003) 위치로 정렬(shift-normalize)한 뒤 비교.

## 5. 앵커 실험 대조 (HOST 파일럿 = 첫 실행)

- HOST의 x 4점 워터폴(O3·P3 각각) 생성
- Komaba 2012 ex situ XRD(Inorg. Chem. 51, 6211)의 (003) 이동 방향·상 전이 x 구간과 정성 비교
  (절대 2θ 일치는 기대하지 않음 — 0 K, 화폐 오프셋. 이동 방향·비율·전이 구간만)
- 통과 기준: (003)이 x 감소에 따라 저각 이동(층간 팽창), (110) 고각 이동(a 수축), 전이 구간이 실험 서열과 모순 없음
- 실패 시 중단·보고 (top-500 확장 금지)

## 6. 적용 범위와 순서

1. HOST 파일럿 (§5) → 검산 통과 확인
2. HEO 앵커 + top-30 (x 4점 × O3/P3 × k_best + 5샘플 평균)
3. top-500 (4-point 있는 행 전체)
4. 워터폴 pptx: 앵커 2종 + top-5

## 7. 결과 csv 추가 컬럼 (기존 컬럼명 변경 없음)

x별·상별로 (`{x_tag}_{phase}` 접미):
`tt_003, tt_110, d_003, d_110, a_eff, c_eff, chk_d003_resid, I_104_rel, xrd_split_110,
xrd_sim_O3, xrd_sim_P3, xrd_phase, xrd_agree_letter`
패턴 원본: `xrd/{comp_id}/{x_tag}_{phase}_{kbest|avg5}.npz` (2θ, I, hkl-d 목록)

## 8. 한계 (리포트 명시)

- 0 K 구조: Debye-Waller 없음 → 고각 세기 과대. 세기는 상대 비교만, 위치가 주 정보
- 크기·변형 브로드닝·선호 배향 없음: FWHM은 장식. 실험 피크 폭과 비교 금지
- 점유율 평균은 SRO를 지움(완전 무작위 가정) — NSR 2022 리뷰가 지적한 PDF 영역은 다루지 않음
- 단사정 왜곡이 SQS 표본 잡음인지 실제인지는 5샘플 평균으로만 판단

## 9. 금지

1. 원자 위치·격자 수정 금지 (종 교체만)
2. 이완 재실행 금지
3. 채점 정의 변경 금지 — XRD 판정은 보고 컬럼
4. 기존 컬럼 덮어쓰기 금지

## 10. 참고

- pymatgen XRDCalculator: https://pymatgen.org/pymatgen.analysis.diffraction.html
- Komaba et al., Inorg. Chem. 51, 6211 (2012) — HOST ex situ XRD 대조
- Natl. Sci. Rev. 9, nwab146 (2022) — XRD(평균 구조) vs PDF(국소 구조) 구분의 근거
- Zhong et al., Phys. Rev. Materials 9, 105404 (2025) — MLIP → 구조 변환 → 특성화 대조의 표준 형식
