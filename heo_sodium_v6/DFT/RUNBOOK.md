# RUNBOOK — DFT 검증 실행 매뉴얼 (쉬운 버전)

스펙 문서: `DFT_WORKFLOW_SPEC.md` (왜 이렇게 하는지는 거기에, **어떻게 하는지**는 여기에)

## 큰 그림 — 딱 이것만 이해하면 됨

- 계산은 총 **265개**: top30 240 + 앵커 16 + Na금속 1 + 파일럿 8
- **계산 240개 ≠ sbatch 잡 240개.** `./submit.sh 10` 한 줄이 sbatch를 10번 쳐서
  일꾼(worker) 잡 10개를 올리고, 잡 10개가 **같은 manifest.csv를 공유 큐로 보면서**
  끝나는 대로 다음 계산을 하나씩 가져감 → 노드당 평균 ~24개가 자동 배분됨
  (미리 24개씩 나누는 것보다 부하 균형·실패 복구에 유리)
- 진행 상태는 **manifest.csv 한 파일에만** 기록됨. 이 파일을 쓰는 프로그램은
  `sync_manifest.py` 하나뿐 — 내가 직접 고칠 일은 "manual 행 되돌리기"뿐 (아래 문제해결)

```
[RunPod]      build_manifest.py → dft_bundle.tgz (POSCAR + manifest + 스크립트 전부)
                  │ scp
[연구실 서버]  tar xzf → config.yaml 채우기 → ./submit.sh --pilot 1 (파일럿 8개)
              → 통과하면 ./submit.sh 10 (본 실행 257개) → sync_manifest.py로 관리
              → 다 되면 collect.py → frames.extxyz
[GPU 머신]    orb_static.py (frames → orb_on_dft.extxyz)   ← 왕복 파일은 이 둘뿐
[연구실 서버]  audit.py → dft_verification.csv + report_dft.md + figs_dft.pptx
```

### 자주 헷갈리는 것

| 오해 | 실제 |
|---|---|
| 서버에 스크립트만 넣으면 됨 | **RunPod에서 만든 번들**이 필요 (POSCAR·manifest가 MLIP 캐시에서 나옴) |
| `bash worker.sbatch` 직접 실행 | **`./submit.sh N`** 사용 (config 주입 + 파일럿 게이트 확인 + sbatch N번을 대신 해줌) |
| 잡마다 계산 목록을 지정해야 함 | 지정 불필요 — 일꾼들이 알아서 하나씩 가져감 (`.claim` 디렉토리로 중복 방지) |
| 잡이 죽으면 처음부터 다시 | `./submit.sh` 재제출만 하면 됨 — 끝난 계산(DONE)은 건너뜀 |

---

## 1단계 — RunPod에서 번들 만들기 (여기가 시작, 건너뛸 수 없음)

구조 240개는 RunPod의 MLIP 캐시에서만 꺼낼 수 있다 (스펙 §8-1: 신규 생성 금지).

```bash
# Mac에서: 스크립트를 RunPod으로
scp -r ~/Desktop/AIM/Mlip_ORB_#1/Sodium_mlip_v1/DFT <runpod>:/workspace/

# RunPod에서 (HEO_WORKDIR=/workspace/heo_v4 환경 확인)
cd /workspace/DFT
python build_manifest.py --out dft_bundle
```

- [ ] 출력에서 **"manifest: 265 rows {top30: 240, anchor: 16, E_Na: 1, pilot: 8}"** 확인
- [ ] 검산 실패 시 ABORT + 실패 목록이 출력됨 → 여기서 멈추고 원인 확인 (진행 금지)
- [ ] POSCAR 아무거나 3개 열어서 육안 확인

```bash
scp dft_bundle.tgz <서버계정>@<서버주소>:~/
```

## 2단계 — 서버 1회 설정

```bash
tar xzf dft_bundle.tgz && cd dft_bundle     # 이 디렉토리가 작업 루트 = $RUN_ROOT
export RUN_ROOT=$PWD

# 환경 만들기 (1회)
conda create -n atomate2 python=3.11 -y && conda activate atomate2
pip install atomate2 jobflow custodian pymatgen emmet-core monty ase pandas pyyaml python-pptx

# atomate2 클래스가 설치 버전에 있는지 확인 (스펙 미검증 #2)
python -c "from atomate2.vasp.jobs.mp import MPGGARelaxMaker, MPGGAStaticMaker; \
           import inspect; print(inspect.signature(MPGGARelaxMaker))"
#   ↳ run_vasp_kwargs가 없거나 vasp_cmd 키를 거부하면: worker.sbatch에
#     export ATOMATE2_VASP_CMD="srun -n 56 <vasp_std>" 추가하고 runner.py의
#     run_vasp_kwargs 인자 제거 (이 경우 파일럿은 std/gam 두 번 나눠 실행)

# POTCAR 등록 (1회)
pmg config -p /path/to/POT_GGA_PAW_PBE ~/psp_resources
pmg config --add PMG_VASP_PSP_DIR ~/psp_resources
```

- [ ] `vi config.yaml` — `<...>` 자리표시자 채우기 (이 파일만 손으로 편집):
  - `vasp_module` (module avail로 확인), `vasp_cmd_std`/`vasp_cmd_gam`의 VASP 경로
  - `pmg_vasp_psp_dir`
  - `partition`/`walltime`이 서버 실제값과 맞는지 `sinfo`로 확인 (미검증 #6)

## 3단계 — 파일럿 8계산 (통과 전엔 본 실행이 자동 거부됨)

```bash
./submit.sh --pilot 1        # 잡 1개, 파일럿 8행만 처리
squeue -u $USER              # heo_dft 잡 확인
```

- [ ] 첫 계산 시작되면 INCAR 실물 검사:
  ```bash
  grep -E "ISPIN|LORBIT|ISIF|LDAU|ENCUT" calcs/*/run-*/*/INCAR
  ```
  ISPIN=2, LORBIT=11, ISIF=3 있어야 하고 / LDAU·ENCUT은 MP 기본값 그대로여야 함 (§8-2)
- [ ] 8계산 완료 후 (`ls calcs/*/DONE`이 8개):
  ```bash
  python sync_manifest.py --root $RUN_ROOT
  python collect.py --root $RUN_ROOT --sets pilot --no-frames
  python audit.py --root $RUN_ROOT --pilot     # 체크리스트 PASS/FAIL 출력
  ```
  - missing_fields 경고가 나오면 collect.py 상단 `PATHS` 표 한 곳만 수정 (미검증 #3)
- [ ] 전부 PASS면 audit 출력의 권고대로 `config.yaml` 두 값 갱신:
  - `kmode`: gam이 기준(|ΔdE|<5 meV/f.u. **및** 힘 MAE<0.02 eV/Å) 통과하면 `gam`, 아니면 `std` 유지
  - `t_est_hours`: 실측 최대 T × 1.3
- [ ] 일꾼 로그(`logs/worker_*.log`)에서 "deadline; exiting cleanly" 확인 (미검증 #5)

## 4단계 — 본 실행 (257 = 240 + 16 + 1)

```bash
./submit.sh 10               # sbatch × 10. 노드당 ~24계산 자동 배분
```

이후 내가 할 일은 이 두 줄을 주기적으로 (크론 30분 권장):

```bash
python sync_manifest.py --root $RUN_ROOT   # 상태 반영 + set×status 집계표 출력
squeue -u $USER                            # 일꾼이 다 빠졌는데 pending 남았으면 → ./submit.sh 10
```

크론 등록 예:
```bash
crontab -e
# */30 * * * * cd $RUN_ROOT && conda run -n atomate2 python sync_manifest.py --root $RUN_ROOT >> logs/sync.log 2>&1
```

- 소요 예상: walltime 3일에 일꾼당 한 라운드 ~12계산 → **재제출 1회 포함 총 ~6일** (T≈6h 기준)
- [ ] 257행 전부 status=done 될 때까지 반복

### 문제 해결 (sync_manifest 출력에 뜨는 것들)

| 상태 | 뜻 | 내가 할 일 |
|---|---|---|
| fizzled | 계산 실패 1회 (custodian 복구도 실패) | 없음 — 다음 일꾼이 자동 재시도 |
| manual | 2회 실패 or 잡이 3번 유실됨 | `calcs/{row_id}/FIZZLED.*`와 `logs/` 확인 → 원인 해결 → manifest.csv에서 그 행 `status=pending, attempts=0`으로 고쳐서 재투입 |
| running인데 squeue에 잡이 없음 | walltime 잘림/노드 사망 | 없음 — sync가 자동으로 클레임 풀고 pending 복귀 |

## 5단계 — 수집 → GPU 왕복 → 감사

```bash
# 257행 전부 done 확인 후
python collect.py --root $RUN_ROOT           # dft_raw.csv + frames.extxyz

# GPU 머신 왕복 (ORB 연화 측정 — 고정 기하 static만, 이완 금지 §8-4)
scp frames.extxyz <gpu>:~/
# GPU에서: python orb_static.py --frames frames.extxyz --out orb_on_dft.extxyz
scp <gpu>:~/orb_on_dft.extxyz $RUN_ROOT/

python audit.py --root $RUN_ROOT             # 최종 산출물 3종 생성
```

- [ ] 최종 확인 (인수 기준 §9): 파일럿 기록 / manifest 257행 done (manual은 사유 기록) /
  `dft_verification.csv` + `report_dft.md` + `figs_dft.pptx` / 부호 일치율·자화 감사·c값·
  파인튜닝 발동 판정 문장 / frames.extxyz 프레임 수·조성 커버리지

---

## 설계 결정 기록

- **Mongo 불사용** (미검증 #1 해소): FireWorks 대신 sbatch 상주 일꾼 + `os.mkdir` 원자 클레임(NFS 안전) + manifest 단일 상태 저장. jobflow는 로컬(run_locally)로만 사용
- **일꾼 10개 기본** (2026-09-22 사용자 결정): 노드당 ~24계산 동적 배분. `./submit.sh N`으로 증감 자유
- **앵커 = HOST_v1, HEO_v1 고정** (2026-09-18 사용자 결정, Ti0.3 제외 유지). v1-only라 variant σ가 없어 GATE-2 유의성 문턱은 δ=5 meV/f.u. (report에 한계로 기록)
- **상 판독 = prismatic_order** (§3-3의 "글자 분류기" 문구와의 편차): stacking_report는 이완 셀의 buckling에서 assert로 중단됨 — heo_worker 자체가 문서화한 이완-강건 판독기를 사용, report에 기록
- **E_Na**: bcc Na 2원자 셀을 runner가 생성(스펙 §5가 명시한 유일한 비캐시 구조), kmode와 무관하게 항상 std + MP k-밀도(금속). V_seg/V_avg에는 DFT E_Na만(§8-5)
- **파일럿 행은 별도 row_id** (`...__std`/`...__gam`): §9의 "257행 done"은 pilot 제외 집합에 적용
