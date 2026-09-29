# 세션 노트 2026-09-17 — uMLIP 12종 벤치마크 파이프라인 구축·실행

> 이 세션에서 논의·결정·구현한 것 전부. 해석 가이드는 `untitled folder/BENCH_PIPELINE_REVIEW.md` 참조.

---

## 1. 목표와 최종 스코프 (결정 순서대로)

1. 시작: HANDOFF + MODEL_BENCHMARK_SPEC 기반으로 MLIP 모델별 비교를 런팟에서 실행 가능하게.
2. DFT 정답지(CORE/FRAMES)는 **아직 없음** → DFT 불필요한 부분부터 (Tier S는 나중에).
3. 참가 모델 **12종 전부** (M01~M12). 웹 조사로 12종 모두 공개 실물 확인.
4. **Ti03 앵커 영구 제외** (워커에도 2026-09-03부로 이미 반영돼 있었음). 앵커 = HOST/HEO 2종.
5. 기존 heo_worker.py + v4 노트북을 **베이스로 재사용** (새 프레임워크 금지) — 셀 5(앵커 게이트)
   + 5d(x4 파일럿)가 곧 "모델 1개분 벤치마크"임을 확인하고 그 로직을 러너로 추출.
6. **XRD_SPEC 추가** → 시뮬레이션 XRD를 벤치와 기존 v4 결과 양쪽에 후처리로 적용.
7. dV 분해를 비교 표로 승격 (스펙 §4에 있었는데 누락했던 것).
8. **Ehull 추가** (사용자 결정) — v4 셀 8 정의 그대로: 경쟁상도 각 모델이 자기 에너지로
   재이완하는 in-model hull + LP. 화폐 혼합 없음.
9. **최종 스코프 확정**: 전체 스크리닝 재실행이 아니라, **results_v3.csv의 top-N(기본 30)을
   12모델이 4-point 전체(32구조/조성)로 재계산**해서 ORB 기준선과 비교하는 것이 핵심.
   → `rho_ddE_topN`(채점 성분 순위 재현), verdict 일치율, MAE_dE가 최종 표의 중심.

## 2. 만든 파일 (팟 업로드 대상 6개)

| 파일 | 역할 |
|---|---|
| `heo_bench_runpod_v1.ipynb` | 실행 노트북 (v4 관례: CONFIG→워치독→INSTALL→SQS→런처→집계→XRD→리포트) |
| `heo_worker.py` | heo_worker-3.py 이름 변경 사본 (구조 생성·메트릭 전부 여기서 import) |
| `bench_models.py` | 12모델 + mock ASE 어댑터. **orb는 `orb-models<0.6` 핀** |
| `bench_anchor_runner.py` | 모델 1개분: 앵커 128 + top-N×32 이완 → 집계 → Ehull → XRD → JSON |
| `xrd_tools.py` | XRD_SPEC 구현 (점유율 평균, 피크 추적, d₀₀₃ 검산, 상 지문) |
| `models_registry.yml` | 관문 0 레지스트리 (verified/unverified/blocked 정직 표기) |

문서(업로드 불필요, `untitled folder/`): BENCH_PIPELINE_REVIEW.md(해석 가이드), 스펙 md 4종.

## 3. 설계 핵심 (왜 이렇게 됐나)

- **전 모델 ASE FIRE + FrechetCellFilter 통일** (fmax 0.05, 300스텝) — 스펙 §10-1.
  프로덕션(torch-sim)과의 경로 차이는 셀 8이 orb로 정량화 (|ΔdE| < 5 meV/f.u.면 합격 각주).
- **모델별 workdir 격리** (`runs/<모델>`, invariant 6) + **shared_base에는 SQS만** 공유
  → 12모델 동일 시작구조, 에너지·기하는 절대 안 섞임.
- **관문 assert → 기록**: 모델이 관문에 떨어지는 것도 벤치마크 데이터 (스펙 §10-6).
- **체크포인트 재개**: (comp_id, tag) 단위 CSV append — 언제 죽여도/다시 돌려도 이어짐.
- 모델당 계산량: 앵커 128 + top-30×32=960 + Na 1 = **1,089 이완** (EHULL=0 기준).
  Ehull 켜면 + 경쟁상 수백 (원소 풀에 좌우, top-N 줄여도 거의 안 줄어듦).
- E_Na는 모델별 자기 계산 (§10-3). 비 PBE 화폐는 경고+기록 (대조군 허용, §10-6).

## 4. 실행 절차 (확정본)

```bash
# 터미널 (jupyter 띄우는 셸에서)
pkill -f pod_watchdog.py; pkill -f bench_anchor_runner        # 이전 프로세스 정리
pip install pandas numpy scipy pymatgen ase pyyaml icet python-pptx matplotlib huggingface_hub papermill
export RUNPOD_API_KEY=...                                     # stop 폴백
export HF_TOKEN=...                                           # eSEN gated면
mkdir -p /workspace/ckpts && wget -O /workspace/ckpts/equflashv2-45m-oam.pt \
  "https://figshare.com/ndownloader/files/65435007"           # M09 체크포인트 (Figshare)
```

```python
# 셀 1 (직접 대입 — setdefault 아님!)
os.environ["HEO_TERMINATE_MODE"] = "stop"
os.environ["HEO_BENCH_MODELS"]   = ""        # 빈 값 = 12종
os.environ["HEO_BENCH_EHULL"]    = "0"       # 경쟁상 생략 (top-30 자체는 돎)
os.environ["HEO_EQUFLASH_CKPT"]  = "/workspace/ckpts/equflashv2-45m-oam.pt"
```

- MP_API_KEY **불필요** (Ehull은 v2 hull_cache 복사; 캐시 없을 때 새로 만들 때만 v4 셀 8 + 키).
- 실행: Run All 또는 `nohup jupyter nbconvert --to notebook --execute --inplace <노트북> &`
- 진행 확인: `tail -f /workspace/heo_bench/logs/<모델>.log`, 전체는 `results/notebook_stdout.log`
- 끝나면 DONE 작성 5분 뒤 워치독이 팟 stop. 유휴 180분+GPU 유휴도 stop. 최대 36h.

## 5. 이 세션에서 잡은 함정들 (재발 방지용)

| 함정 | 내용 | 처방 |
|---|---|---|
| **setdefault 고착** | 노트북 셀의 `os.environ.setdefault`는 커널에 이미 박힌 값을 못 바꿈 — terminate/models 수정이 안 먹혔던 원인 | 직접 대입 `os.environ[...] = ...` 또는 커널 재시작 |
| **워치독 detached** | 모드 바꿔도 이미 뜬 워치독은 옛 모드로 계속 돎 | `pkill -f pod_watchdog.py` 후 셀 1b 재실행 |
| **orb-models 신버전 API 드리프트** | 최신 휠에서 `forcefield.calculator.ORBCalculator` 소멸 → adapter_error 2회 | `orb-models<0.6` 핀 (bench_models.py에 반영) + 다중 경로 스캔 |
| **EHULL=0이 top-N까지 끄던 버그** | 구버전 러너는 EHULL=0에 타깃 계산도 스킵 | 러너 수정 완료 — 스위치 분리 (EHULL=0은 경쟁상만 끔) |
| **로컬 디스크 100%** | macOS가 Desktop 파일을 iCloud로 이빅션(dataless) → 읽기 타임아웃 | 공간 확보가 선행 조건 (세션 초반 실제 발생) |
| **d₀₀₃ 검산 위양성** | 브로드닝 곡선 꼭짓점은 이웃 피크 겹침으로 ~0.01° 이동 | 원시 델타 피크의 정확한 d로 검산 (잔차 1.5e-7 Å 확인) |
| **스모크 vs 전체** | `HEO_BENCH_MODELS="orb_mpa"`면 1모델만 도는 게 정상 | 전체는 `""` |

## 6. 결과 파일과 읽는 순서 (요약 — 상세는 REVIEW.md §5)

1. `results/bench_summary.csv` — ① status(실패 모델도 행으로 남음) → ② gate_* 5개(관문 1)
   → ③ **rho_ddE_topN**·verdict 일치율(핵심 질문) → ④ dV·XRD·속도
2. `runs/<모델>/results/targets_x4.csv` — 조성 10~30개 × dE(x)·x*·V_seg·Ef·dV (모델별 원본)
3. `results/orb_path_consistency.csv` — ASE↔torch-sim 경로 검증 (|ΔdE|<5 이어야 각주 성립)
4. `xrd_v4/results_xrd_v4.csv` + `results/xrd_waterfall.pptx` — 기존 v4 결과의 XRD 번역
5. 부호 규약 (오독 주의): **dE>0 = O3 승리**, gap=HEO−HOST 클수록 지연 방향, AMB는 판정 유보.
6. 우열 선언 금지: 지금 표는 "ORB와 얼마나 같나"이지 "누가 더 정확한가"가 아님 — 후자는 DFT 후.

## 7. 남은 리스크 (정직)

- **orb<0.6 핀**: 문서화된 API가 있는 버전대라는 근거의 최선 추측 — 팟에서
  `[orb] ORBCalculator found in ...` 로그로 최종 확인 필요.
- **M09/M10/M11/M12 어댑터**: 클래스명이 후보 시도 방식 — adapter_error 나오면
  `runs/<모델>/results/anchor_bench.json`의 error/traceback 보고 bench_models.py 한 곳 수정 사이클.
- **M06 (EquiformerV3)**: 공식 ASE 인터페이스 부재 → '실행 불가'로 끝날 가능성 높음 (예정된 결과).
- **M05 (eSEN)**: HF gated면 HF_TOKEN 필요.
- **M11 (TECE)**: py3.13+torch2.13 전용 venv — 설치 자체가 가장 무거움.
- 무거운 모델들이 벽시계를 끔 — 급하면 HEO_BENCH_MODELS에서 빼고 나중에 추가 (재개라 무손실).

## 8. 다음 단계

1. 12종 실행 완료 → bench_summary 검토 (orb status=ok 확인이 1순위 — 기준선 없으면 비교 컬럼 공란)
2. 실패 모델 어댑터 수정 → Run All 재실행 (성공분은 전부 cached)
3. Ehull 원하면 `HEO_BENCH_EHULL` 지우고 재실행 (경쟁상만 추가 계산)
4. 앵커 x059(Na 16) 추가 검토 (HANDOFF §4-2 — 전이 구간 해상)
5. DFT CORE/FRAMES 도착 → Tier S(관문 2·cMAE_force) — 같은 구조·CSV 계약으로 확장 예정
