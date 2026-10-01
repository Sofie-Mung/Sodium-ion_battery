# MODEL_BENCHMARK_SPEC — 우리 계(系) 기준 uMLIP 벤치마크 (Tier S / Tier R)

> 실행자: Claude Code. 저장소 루트 CLAUDE.md, DFT_WORKFLOW_SPEC.md, HANDOFF_next_session.md를 먼저 읽을 것.
> 목적: ORB-v3 mpa를 주모델로 쓴 근거를 **우리 관측량(dE 부호·랭킹·x*)에 대한 직접 증거**로 바꾸고,
> 주모델 1 + 계보 다른 대조모델 1을 데이터로 결정한다. 채점 정의는 건드리지 않는다.

---

## 0. 설계 원칙

1. **정답지 = 우리 계의 DFT** (MP 화폐). Matbench F1은 참고일 뿐 — 상위권이 F1 0.90~0.91에 몰려 있어 F1로는 갈리지 않는다.
2. **Tier S(고정 기하)와 Tier R(이완)을 분리** — 이완을 섞으면 모델 오차와 재최적화 오차가 뒤엉킨다(Deng 2025).
3. **판정은 가중합이 아니라 관문 사다리** — 용도가 "랭킹 + 부호"이므로 MAE보다 부호·방향이 먼저.
4. **2단 벤치마크** — Tier S는 싸므로 전원, Tier R은 비싸므로 예선 통과 5개만.

## 1. 시험지 (DFT 정답, DFT_WORKFLOW_SPEC 산출물 재사용)

| 집합 | 내용 | 용도 |
|---|---|---|
| CORE | top-30 + 앵커 2(HOST, HEO) × 4x × 2상 = 256 최종 이완값 | Tier R 정답 |
| FRAMES | 위 256 이완의 모든 ionic step (수천 프레임, E·F·응력 라벨) | Tier S 정답 |
| E_Na | DFT bcc Na | Tier R 전압 검산용 (모델은 자기 E_Na 사용) |
| EXT60 (권장, 사용자 결정) | 중위 20 + 하위 10 조성 × x100/x052 × 2상 = 120 계산 | Tier R 랭킹 ρ의 범위 제한 해소 |

EXT60 없이 top-30만으로 잰 ρ는 상위권 압축으로 과소평가됨 — 리포트에 그 한계를 명시할 것.
선정 규칙(EXT60): v4 results의 score 백분위 40–60에서 20, 80–100에서 10, 스킴 A/B 비율 유지, 앵커 제외.

## 2. 참가 모델 (관문 0에서 각 항목 실행 전 확인 — §7)

| ID | 모델/체크포인트 | 계보 | 화폐(확인 대상) | 역할 |
|---|---|---|---|---|
| M01 | ORB-v3 conservative **mpa** | 비등변 | MP | 현행 주모델 (기준선) |
| M02 | ORB-v3 conservative **omat** | 비등변 | **OMat** | **화폐 대조군** (같은 아키텍처, 다른 화폐) |
| M03 | MACE-MPA-0 (또는 MP-0b3) | 등변 | MP | 현행 대조 |
| M04 | SevenNet-Omni (MPA 헤드) | 등변 | MP | 다중태스크 계보 (공개 데이터셋 15개) |
| M05 | eSEN-30M-OAM | 등변, 대형 | MP 마무리 | 리더보드 상위 |
| M06 | EquiformerV3+DeNS-OAM | 등변, 대형 | MP 마무리 | 리더보드 F1 1위 |
| M07 | MatterSim-v1 | 등변 | MP 호환 주장 | 별도 데이터 계보 |
| M08 | CHGNet | 자화 인식 | MP | 자화 예측 가능한 유일 참가자 |
| M09 | EquFlashV2-45M-OAM (또는 EquFlash-29M-OAM) | 등변, FlashTP 가속 | MP 마무리 | 등변+속도 — 주모델 후보 |
| M10 | Prophet-OAME-MBD | 등변, 62.3M | OAME — 마무리 화폐 확인 | 신규 상위 |
| M11 | TECE-OAM-RRA-1.0 | TACE 계열 | -oam 접미 — 확인 | 신규 상위 |
| M12 | PET (OAM 변형이 있으면 그것; PET-MAD는 PBEsol이므로 대조군으로만) | PET | 확인 | κ_SRME 최저 — 곡률 재현 후보 |

OMat/PBEsol 화폐 모델은 탈락시키지 않고 **대조군으로 참가**시킨다: "화폐 불일치가 dE(같은 조성 차분)에 얼마나 영향을 주는가"는 벤치마크가 답할 질문이다.

## 3. Tier S — 고정 기하 (전원 참가)

**입력**: FRAMES 전부 + CORE 최종 구조 256.
**절차**: 각 모델로 각 프레임에 static forward (E, F, 응력). 원자·셀 이동 없음. 배치 처리, 같은 GPU.
**산출 (모델별)**:
- `c_force`: 힘 성분 산점(x=DFT, y=모델)의 회귀 기울기 — 연화계수. 전체 + 원소별 + x별(x100/x081/x067/x052)
- `cMAE_force`: 기울기를 1로 되돌린 후 잔차 MAE — 비체계적 오차 크기 (**핵심 지표**)
- `MAE_E_atom`: 프레임 E/atom 오차 (화폐 오프셋은 조성별 평균 제거 후 — 화폐 대조군 해석용으로 제거 전/후 둘 다 기록)
- `dE_S(x)`: CORE의 DFT-이완 O3/P3 구조에 모델 에너지 → (E_P3 − E_O3)/27, DFT dE와 부호·MAE
- `sign_S(x)`: dE_S 부호 일치율 (x052, x067)
- `anchor_S`: HOST/HEO의 dE_S 부호와 순서
- `t_static`: 프레임당 시간 (같은 GPU, 같은 배치 크기)
**통계**: 모든 지표에 부트스트랩 95% CI (프레임 단위 재표집; dE는 조성 단위).

## 4. Tier R — 이완 (예선 통과 5개)

**예선 규칙**: 관문 0 통과 모델 중 (a) anchor_S 방향 통과, (b) cMAE_force 하위 5 (낮을수록 좋음). 동률 시 sign_S.
M01(현행)은 기준선으로 항상 포함(예선 결과와 무관, 5개 외 +1 가능).

**입력**: CORE(+EXT60)의 **이완 전 시작 구조** — MLIP 캐시의 SQS 템플릿·k_best 빈자리, 모든 모델 동일.
**동일 조건**: torch-sim FIRE, Frechet cell filter, fmax = 0.05 eV/Å, max_steps 동일, 같은 GPU.
torch-sim 미지원 모델은 ASE FIRE + FrechetCellFilter로 같은 fmax·스텝 — 속도 표에 경로 표기.
모델 고유 cutoff·이웃 수는 기본값(모델의 일부).
**E_Na**: 각 모델이 bcc Na를 자기 화폐로 계산 (혼용 금지).
**산출 (모델별)**: 파이프라인 관측량 전부 — dE_R(x), ddE, V_avg, V_seg, dV/dA/dh_perp/vm_strain(det 기반), x_star_class/mid, n_phase_flip, converged 비율, t_relax(구조당 시간).
**대조**:
- `sign_R(x)`: DFT 이완 dE와 부호 일치율 (x052, x067) + CI
- `rho_ddE`: ddE Spearman (CORE 30 / CORE+EXT60 60 — 둘 다 보고)
- `anchor_R`: HOST 전이 + V_avg ∈ [2.5, 3.4] (≈3.1) + HEO가 HOST보다 지연(dE_x067 차이 > σ)
- `MAE_dE_R`, `MAE_V`, `MAE_dV`
- `reopt = |dE_R − dE_S|` 조성별 — 재최적화 오차 (S/R 교차 진단표)

## 5. 판정 — 관문 사다리 (순서 고정)

| 단계 | 기준 | 탈락 시 |
|---|---|---|
| 관문 0 화폐 | MP 화폐 체크포인트 확인 | 대조군으로 강등 (결과는 보고) |
| 관문 1 앵커 방향 | Tier R: HOST 전이·V≈3.1, HEO 지연 | 탈락 |
| 관문 2 부호 일치 | sign_R(x052) ≥ 기준선(M01) − CI 이내, 짝지은 부트스트랩 차이 | 탈락 |
| 지표 3 랭킹 | rho_ddE (EXT60 포함값 우선) | 순위 |
| 지표 4 오차 | cMAE_force, MAE_dE_R, MAE_dV | 동률 해소 |
| 지표 5 실무 | t_relax, torch-sim 배치, conservative, 라이선스 | 동률 해소 |

**모델 간 비교는 짝지은 차이로**: 같은 구조에서의 오차 차이를 부트스트랩 → "A > B"의 95% CI가 0을 제외할 때만 우열 선언. 그렇지 않으면 "동률".
**최종 산출**: 주모델 1 + **계보가 다른** 대조모델 1 (비등변↔등변 등). 계보 독립성은 δ의 |dE_A − dE_B| 항의 가치.

## 6. S/R 교차 진단표 (리포트 필수 포함)

| Tier S | Tier R | 진단 | 처방 |
|---|---|---|---|
| 좋음 | 좋음 | 합격 | 주모델 후보 |
| 좋음 | 나쁨 | 이완 불안정(옵티마이저 궁합, 비보존 힘, 상 뒤집힘) | 설정 재시험 / static 전용 |
| 나쁨(c<1, cMAE 작음) | 괜찮음 | 연화가 이중차분에서 상쇄 | 사용 가능, c 기록 |
| 나쁨(cMAE 큼) | 나쁨 | 비체계적 오차 | 탈락 |

## 7. 관문 0 확인 항목 (모델별, 실행 전 — 미검증)

각 모델에 대해 저장소/모델 카드에서 확인하고 `models_registry.yml`에 기록:
- [ ] 정확한 체크포인트명·버전·다운로드 경로
- [ ] 학습 데이터 계보: 사전학습 / 마무리 학습(MPtrj·sAlex 포함 여부 → MP 화폐 판정)
- [ ] conservative 힘 제공 여부 (direct-only면 Tier R에서 명시)
- [ ] torch-sim 지원 여부 → 없으면 ASE 경로
- [ ] 라이선스 (연구·공개 가능 여부)
- [ ] 기본 cutoff / 최대 이웃 수
- [ ] M12 PET: OAM 변형 존재 여부 (없으면 PET-MAD를 PBEsol 대조군으로)
- [ ] M10 Prophet: OAME의 마무리 화폐 / ELEMENTA 데이터 성격
- [ ] M11 TECE: 아키텍처·마무리 화폐

## 8. 실행 순서

1. 관문 0 레지스트리 작성 → 사용자 검토
2. DFT CORE·FRAMES 도착 즉시 Tier S 전원 (GPU 머신) → 예선표
3. 예선 통과 5(+M01) Tier R → CORE, EXT60 순
4. 판정표 + 교차 진단표 + 결정문
5. 결정된 대조모델로 δ 재계산 영향 확인 (기존 MACE 대비 AMB 개수 변화) — 보고만, 채점 정의 변경 없음

## 9. 산출물

- `models_registry.yml` (관문 0)
- `tier_s_results.csv` (모델 × 지표 × CI), 힘 산점 pptx (모델별 c 표기)
- `tier_r_results.csv`, 부호 일치 히트맵(모델 × 조성 × x), 앵커 V(x) 계단 비교 pptx
- `benchmark_report.md`: 관문 통과표, 짝지은 비교 CI, S/R 교차 진단, **결정문**(주모델·대조모델·근거)
- 논문 SI "모델 선정" 절 초안 (결정문 확장)

## 10. 금지 사항

1. 모델별로 시작 구조·fmax·스텝·GPU를 다르게 두지 않는다
2. Tier S에서 원자·셀을 움직이지 않는다
3. 다른 모델의 E_Na를 섞지 않는다
4. 채점 정의(SCORE_TERMS/관문/δ) 수정 금지 — 대조모델 교체 영향은 보고만
5. 앵커는 어떤 모델의 미세조정에도 사용 금지
6. 화폐 불일치 모델을 조용히 제외하지 않는다 — 대조군으로 결과를 남긴다
