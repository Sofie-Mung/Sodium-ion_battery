# 세션 노트 2026-10-01 — 12모델 벤치 실제 실행 (새 팟, XRD 제거, Ehull 켬)

> 대상: 이 벤치를 이어서 돌리는 사람과 다음 세션.
> 기준 문서는 여전히 `CHANGES_AND_REVIEW_2026-09-30.md`이고, 이 문서는 그 이후 바뀐 것만 다룬다.
> 아래 코드 수정은 전부 로컬에서 문법 검사만 했다. 동작 확인은 팟 실행 결과로 한 것만 "확인됨"으로 적었다.

---

## 1. 지금 상태 (한 문단)

L40S 팟에서 11모델 벤치가 실행 중이다 (사용자 보고 기준). eSEN만 Hugging Face 승인 대기로 빠져 있다.
대상은 **top-30 조성 + 앵커 2종(HOST, HEO)** 이고 전수 7,073 조성이 아니다. Ehull은 켰다.
DFT는 같은 30조성에 대해 별도로 계산 중이고, 결과가 나오면 비교한다.

## 2. 오늘 결정한 것

| 결정 | 내용 |
|---|---|
| 순서 | 12모델 벤치를 먼저 돌리고, DFT 결과가 나오면 비교 |
| 저장소 | 새 볼륨에서 시작. 기존 팟에서 필요한 파일만 묶어 옮김 (§3) |
| XRD | 전부 제거. 더 이상 필요 없음 |
| Ehull | 켬 (`HEO_BENCH_EHULL=1`). 모델마다 자기 에너지로 자기 hull을 지음 |
| GPU | L40S. Blackwell 계열은 쓰지 않음 |
| eSEN | 승인 전까지 제외하고 11종으로 진행 (`HEO_REQUIRE_ALL_MODELS=0`) |
| 전수 스크리닝 | 이번 벤치의 모델별 속도를 본 뒤 결정 |

## 3. 새 볼륨으로 옮긴 것

기존 팟에서 `/workspace/heo_bench_seed.tgz` (6.7 MB, 파일 1,179개)를 만들어 새 팟 `/workspace`에 풀었다.

- `heo_v2/results/results_v3.csv`, `top30_v3.csv`, `compositions_master.csv`
- SQS 셀 34개 (top-30 + 앵커 4변형) — 12모델이 DFT와 같은 원자 배열에서 출발하기 위해 필수
- `heo_v4/results/anchors_x4.csv` (셀 8의 orb 일관성 검사용)
- `heo_v4/hull_cache/competitors.json` (Ehull 경쟁상 구조, 5.5 MB) — 이게 있어서 `MP_API_KEY`가 필요 없다
- 에너지 체크포인트 12개, 이완 CIF 1,128개 (XRD용이었고 지금은 안 쓰임)

셀 4 출력으로 확인됨: 앵커 4개와 top-30 SQS 복사, `top30_v3.csv`와 `results_v3.csv` 정렬 상위 30이 30/30 일치, 경쟁상 캐시 복사.

완전히 빈 볼륨으로 돌리면 top-N이 조용히 꺼져 앵커 129개만 계산된다.

## 4. 코드 변경

| 파일 | 변경 |
|---|---|
| `heo_bench_runpod_v1.ipynb` | 셀 8b·8c(XRD) 삭제, 셀 7·9와 안내문의 XRD 항목 제거. 셀 0b에 `HEO_BENCH_LOCAL`, `HEO_BENCH_EHULL=1`. 셀 1이 venv·가중치·캐시 경로를 로컬 디스크로 지정. 셀 4의 복사를 `shutil.copyfile`로. 셀 2·5가 `MPLBACKEND=Agg` 전달 |
| `bench_anchor_runner.py` | `run_xrd`, `--skip-xrd`, `import xrd_tools` 제거. 시작 시 `MPLBACKEND=Agg`, venv 실행 폴더를 PATH에 추가 |
| `bench_install.py` | `HEO_BENCH_LOCAL` 지원 (venv와 uv/pip 캐시 위치). 스모크 ①에서 `xrd_tools` 제거. `MPLBACKEND=Agg`, PATH 추가 |
| `bench_models.py` | TECE 설치 목록에 `ninja` 추가 |
| 삭제 | `xrd_tools.py`, `untitled folder/XRD_SPEC.md` |

팟 업로드 대상은 이제 6개다: 노트북, `heo_worker.py`, `bench_models.py`, `bench_anchor_runner.py`, `bench_install.py`, `models_registry.yml`.

## 5. 오늘 잡은 함정 (재발 방지용)

| 증상 | 원인 | 처방 |
|---|---|---|
| 12개 전부 0–3초 만에 설치 실패 (`Operation not permitted`) | `/workspace` 네트워크 볼륨이 chmod를 지원하지 않음. venv 생성, 휠 압축 해제, git clone이 모두 실행 권한 설정에서 실패 | `HEO_BENCH_LOCAL=/root/heo_local` — venv·캐시·가중치를 컨테이너 디스크에. 결과와 체크포인트는 볼륨에 그대로. 설치 통과 확인됨 |
| `No module named 'bench_anchor_runner'` | 팟에 파일을 안 올림 | 6개 파일이 `/workspace`에 정확한 이름으로 있는지 `ls`로 확인 |
| `CUDA error: no kernel image is available` (ORB 2종, Prophet, PET) | GPU가 RTX PRO 6000 Blackwell(MIG). cu126 빌드에 해당 커널 없음. 팟 이미지의 torch를 쓰는 MACE·MatterSim·CHGNet만 통과 | L40S로 교체 |
| `'module://matplotlib_inline.backend_inline' is not a valid value` (EquiformerV3, EquFlashV2, TECE) | Jupyter의 그래프 백엔드 설정이 모델 venv로 넘어감 | 하위 프로세스에 `MPLBACKEND=Agg` 강제 |
| `Ninja is required to load C++ extensions` (TECE) | tace가 첫 계산 때 C++ 확장을 컴파일. venv를 활성화 없이 호출해서 venv 안의 ninja가 PATH에 없음 | venv 실행 폴더를 PATH에 추가. 급하면 `pip install ninja`를 시스템에 |
| eSEN 로드 실패 | `hf.co/facebook/OMAT24`가 승인제 | 모델 페이지에서 접근 신청 → 토큰 발급 → `export HF_TOKEN=...` |
| pip 캐시 비활성 경고 | 볼륨 파일 소유자가 `nobody` | 무시해도 됨 |

부수 효과로 알아둘 것: 컨테이너 디스크는 팟이 멈추면 지워진다. 계산 결과는 볼륨에 남지만 **팟을 다시 띄우면 설치(셀 2)를 처음부터 다시 해야 한다.** 컨테이너 디스크는 150 GB 이상을 권했는데 이 숫자는 추정이다.

## 6. 앞으로 개선·확인해야 할 것

### 6-1. 이번 실행에서 바로 확인할 것

- [ ] `bench_summary.csv`의 `status`와 `engine` 컬럼 — 몇 개가 `ok`이고 어느 모델이 ASE로 떨어졌는지
- [ ] SevenNet — 이전 팟에서 torch-sim 이완 검사가 PyTorch 내부 오류(NVML)로 실패해 ASE로 내려갔다. L40S에서도 그런지 미확인
- [ ] MACE — 이전 팟(MIG)에서 배치 한 스텝이 5.7초로 MatterSim·CHGNet(0.36초)의 15배였다. L40S 수치 미확인
- [ ] TECE — ninja 수정 후 실제 컴파일까지 통과했는지
- [ ] 36시간 한도(`HEO_MAX_HOURS`) 안에 끝나는지. Ehull을 켜서 모델당 경쟁상 수백 구조가 추가됨
- [ ] 엔진이 섞였으면(일부만 ASE) 모델 간 비교에 경로 차이가 낀다는 점을 리포트에 명시

### 6-2. 아직 없는 코드

- **DFT와의 구조 단위 조인 스크립트.** DFT는 ORB가 고른 빈자리 샘플(k_best) 한 구조만 계산한다. 각 모델의 `targets_x4.csv`는 자기 k_best 기준이라 그대로 비교하면 안 되고, `energies_bench.csv`의 5샘플에서 DFT와 같은 k를 골라 맞대야 한다.
- **Tier S (고정 기하).** DFT 이완 궤적의 각 구조에 모델을 적용해 힘 오차와 연화계수를 재는 단계.
- **12모델 전수 스크리닝.** 약 9.5만 이완으로 이번 벤치의 약 87배. 벤치 소요 시간 × 87로 모델별 시간을 추정한 뒤 범위를 정한다.

### 6-3. 설계상 미결

- 수렴 판정에 셀 힘을 넣을지 (`HEO_TS_CELL_CONV`). 지금은 프로덕션과 같게 원자 힘만 본다.
- 벤치의 판정 불확실성 δ에는 프로덕션의 모델 쌍 항이 없다. 벤치 verdict를 `results_v3.csv`의 verdict와 직접 비교하면 안 된다.
- 직접 만든 배치 래퍼(eSEN, EquiformerV3, EquFlashV2, CHGNet, Prophet)는 매 스텝 CPU에서 그래프를 만든다. 전수로 가면 병목이 될 수 있다.
- Blackwell GPU를 쓰려면 torch 빌드를 cu128로 올려야 하고, eSEN은 torch 2.4 고정이라 불가능하다.

### 6-4. 문서 정리

- 09-17 세션 노트, 09-30 변경 문서, `BENCH_PIPELINE_REVIEW.md`에 XRD 서술과 옛 업로드 목록(`xrd_tools.py` 포함)이 남아 있다.
- `DFT/RUNBOOK.md`의 scp 경로가 옛 폴더(`~/Desktop/AIM/Mlip_ORB_#1/Sodium_mlip_v1/DFT`)다.
- `DFT_WORKFLOW_SPEC.md`는 FireWorks + MongoDB로 적혀 있지만 실제 실행은 RUNBOOK의 sbatch 일꾼 방식이다.

## 7. 랩미팅에서 말할 것

### 한 일
- uMLIP 12종 벤치마크 파이프라인을 팟에서 실제로 돌렸고, 11종이 실행 중이다. eSEN은 Meta 승인 대기.
- 이완 엔진을 ASE(한 번에 1구조)에서 torch-sim 배치로 바꿨다. 프로덕션 ORB와 같은 엔진이다.
- DFT가 계산 중인 것과 같은 30조성, 같은 SQS 시작 구조를 쓴다.
- 모델마다 자기 에너지로 hull을 지어 합성 가능성(Ehull)도 비교한다.

### 이 벤치가 답하는 것
- ORB가 고른 top-30을 다른 모델도 같은 순위, 같은 상 판정(O3/P3)으로 보는가.
- 실험 사실 두 가지(HOST는 O3→P3 전이와 약 3.1 V, HEO는 전이 지연)를 각 모델이 재현하는가.
- 핵심 숫자: ddE 순위 상관(`rho_ddE_topN`), 판정 일치율, Ehull 순위 상관과 생존자 겹침.

### 답하지 못하는 것 (먼저 밝힐 것)
- **어느 모델이 더 정확한가는 아직 말할 수 없다.** 지금 표는 "ORB와 얼마나 같은가"이고, 정확도는 DFT가 나와야 한다.
- **전수 스크리닝이 아니다.** ORB가 떨어뜨린 조성 중 다른 모델이 상위로 올렸을 조성이 있는지는 모른다.
- **Ehull은 DFT로도 검증되지 않는다.** DFT가 경쟁상을 계산하지 않으므로 모델 간 일치 여부까지만 본다.
- top-30은 이미 관문을 통과한 조성이라 값의 범위가 좁고, 순위 상관이 실제보다 낮게 나올 수 있다.

### 예상 질문
- *안정성을 얻느라 용량은 얼마나 희생했나?* — 용량 회계가 아직 점수 밖에 있다. top-30의 활성 원소 수를 세는 것은 계산 없이 가능하다.
- *왜 12개나 돌리나?* — 리더보드 F1은 상위권이 0.90–0.91에 몰려 변별력이 없다. 우리 관측량(dE 부호, 순위)으로 직접 고르기 위해서다.
- *전수는 얼마나 걸리나?* — 실측은 ORB뿐이다(약 8.5만 이완에 3–4시간). 나머지는 이번 벤치 시간 × 87로 추정할 예정이다. 대형 등변 모델은 torch-sim이어도 수일 단위일 수 있다.

### 결정받으면 좋은 것
- 전수 스크리닝을 12모델 전부로 할지, 일부 모델만 할지, top-500 재채점으로 줄일지.
- 중하위 조성 30개를 DFT에 추가할지 (스펙의 EXT60).
- eSEN 승인이 늦어지면 11종으로 확정할지.

## 8. 다음 단계

1. 벤치 완주 → `bench_summary.csv`와 셀 2 엔진 표 검토 (§6-1)
2. 모델별 소요 시간으로 전수 스크리닝 일정 추정
3. eSEN 승인되면 `HF_TOKEN` 넣고 재실행 (끝난 모델은 체크포인트에서 건너뜀)
4. DFT 도착 → 구조 단위 조인 스크립트 작성 → 부호 일치, Tier S
5. 새 아이디어 탐색은 `IDEATION_START.md`에서 별도 세션으로
