# 알려진 문제와 운영 메모

사용법은 [README.md](../README.md), MLE-bench 운영 기록은 [mle-bench.md](mle-bench.md). kaggle-master 때 겪은 것 중 이 저장소 코드에 해당하는 것만 옮김.

## 알려진 문제와 해결

### 모델

| 문제 | 원인 | 해결 |
|---|---|---|
| exaone `CUDA out of memory` (Allstate, H100 80GB) | 기본 메모리 계획이 남은 메모리 전부를 목표로 잡는데, 캐시 단편화(예약됐지만 안 쓰는 16GiB)로 실제 할당이 실패 | `max_vram_bytes` 를 GPU 메모리의 70% 로 지정 → 계획 전에 캐시를 비우고 그 안에서 청크를 나눔 (`fab/models/tfm.py`) |
| limix 계열 import 실패 (`Numba needs NumPy 2.2 or less`) | 다른 패키지가 numpy 를 2.5 로 올림, Kaggle 기본 numba 0.61 은 numpy<2.3 | `pip install -U numba` (0.68 은 numpy<2.6 지원) 추가 |
| limix2m `KeyError: 'tf_mlp_layer_type'` | `stable-ai/LimiX-2M` 체크포인트가 현재 LimiX 코드와 설정 키 불일치 | README 가 가리키는 `stable-ai/LimiX-1_2M` 사용 |
| mitra 회귀 `different number of output (1!=1000)` | mitra-regressor-2 는 분위수 출력, AutoGluon 1.6 sklearn 인터페이스와 불일치 | 회귀만 `autogluon/mitra-regressor`(v1) 사용 |
| mitra `NotFittedError` (Pipeline) | sklearn 1.9 Pipeline 의 fitted 검사를 Mitra 가 통과 못 함 | Pipeline 대신 `_Imputed` 래퍼로 직접 결측 처리 |
| exaone `No available kernel` | flash attention 고정 선택, T4(sm75)에는 해당 커널 없음 | sm80 미만 GPU 에서 SDPA 를 MATH 백엔드로 패치 |
| exaone `CUDA out of memory (445 GiB)` (S4E2) | T4 대체용 math 어텐션이 어텐션 행렬을 통째로 생성 | mem-efficient 커널(sm75 지원) 우선, 안 되는 head 폭만 math |
| exaone `CUDA out of memory (157 GiB)` (Spaceship) | mem-efficient 패치 후에도 발생, math 대체 경로로 빠지는 것으로 추정 | 미해결, 보류 (kNN 컨텍스트 `@knn1000` 으로 우회) |
| kumo `CUDA out of memory (83 GiB)`, tabicl 작업자 무응답 종료 (S6E10) | 44만 행을 한 번에 예측 | TFM 예측을 2만 행씩 나눠 처리 (`PREDICT_CHUNK`), LimiX/Kumo 는 모델을 한 번만 로드 |
| kumo `CUDA out of memory (13.7 GiB)` (S4E2) | 컨텍스트 1.6만 행 + 질의 2만 행 | Kumo 만 예측 청크 2,048행 |
| tabpfn_v2 `Number of samples 16,606 ... greater than 10,000` (S4E2) | TabPFN v2 공식 컨텍스트 한도 1만 행 | `ignore_pretraining_limits=True` |
| CatBoost `cat_features must be integer or string` | pandas 3 에서 category→str 변환 시 NaN 유지 | 결측을 `"__NA__"` 문자열로 명시 변환 |
| sap_rpt Titanic accuracy 0.62 (다수 클래스 수준) | SAP-RPT 가 X 와 y(새 0..n 인덱스)를 인덱스 기준 concat → fold 로 자른 X 의 원래 인덱스와 라벨이 어긋남 | 넘기기 전에 `reset_index(drop=True)` |

### 실행 · 데이터

| 문제 | 원인 | 해결 |
|---|---|---|
| Allstate `Expected 132 fields in line 188321, saw 133` | 데이터 묶음 안에 `train.csv` 와 깨진 옛 버전 `train.csv.zip` 이 같이 있어 zip 이 정상 파일을 덮어씀 | `fab/core/download.py` 의 `extract()` 를 `fab download`·Kaggle·Colab 이 같이 씀: zip 안의 zip 까지 풀되, 이미 있는 파일과 같은 zip 안에 압축 안 된 쌍이 있는 x.zip 은 건너뜀 |
| results.csv 에 test=True 인 Kaggle 실행 37개의 test.npy 가 없어 제출 못 함 (kaggle-master#22) | 예측 위치가 코드 위치 기준 (`ROOT/results`) → 에이전트 worktree 에서 수집한 예측이 git 제외라 worktree 와 함께 사라짐 (results.csv 행만 PR 로 메인에 들어옴) | git 제외 실행 출력은 worktree 에서도 메인 체크아웃의 `results/` 에 씀 (`OUTPUTS`, `KM_OUTPUTS` 로 바꿈). `fab lb` 가 아직 안 낸 마지막 실행의 예측이 없으면 알림 |
| GBDT 가 Databricks GPU 노드에서 극도로 느림 (Restaurant 110행 xgb fold 하나 593s, 로컬 0s) | 모델별 프로세스 6개가 한 노드에서 동시 실행, lgbm·xgb 가 각자 전체 코어만큼 스레드를 띄워 CPU 경합 | GBDT 는 CPU serverless 작업으로 따로 제출 (`fab databricks` 가 자동 분리), GPU 는 기본 H100 1장. 병렬 작업자마다 CPU 스레드를 CPU 수 / 작업자 수로 제한 (`OMP_NUM_THREADS`·`KM_THREADS`, Kaggle·로컬에도 적용) |
| macOS 로컬에서 lgbm·xgb import 실패 `Library not loaded: @rpath/libomp.dylib` (kaggle-master#43, catboost·TabPFN·TabICL 은 됨) | lightgbm·xgboost 의 macOS 패키지는 OpenMP 런타임을 넣지 않고 Homebrew 경로(Apple Silicon `/opt/homebrew/opt/libomp/lib`, Intel `/usr/local/opt/libomp/lib`)에서 찾음. lightgbm 은 MacPorts(`/opt/local/lib/libomp`)도 찾지만 xgboost 는 Homebrew 경로만. catboost 는 OpenMP 를 안 쓰고, torch 는 `libomp.dylib` 를 자체 포함 | `brew install libomp` (README 준비 단계). 없으면 `make_lgbm`·`make_xgb` 가 이 해결법을 에러에 붙임. 확인 (arm64, libomp 23.1.5): Python 3.12 lightgbm 4.7.0·xgboost 3.4.1, Python 3.11 xgboost 3.2.0 |
| macOS 에서 lightgbm·xgboost 와 torch 를 한 프로세스에서 쓰면 segfault(139) 또는 멈춤 (kaggle-master#43, `brew install libomp` 후) | 둘 다 OpenMP 를 실제로 쓰면 순서와 상관없이 발생 (lgbm 학습 → torch 행렬곱, torch → lgbm 둘 다). libomp 를 하나로 맞춰도 (`DYLD_LIBRARY_PATH` 로 torch 것이나 Homebrew 것만 로드) 같음, `OMP_NUM_THREADS=1` 로도 일부 순서는 그대로. 한 프로세스에서 한쪽만 쓰면 문제없음 (torch 2.14.1, 2026-11-10). Linux 는 확인 안 함: GPU 경로는 원래 모델별 프로세스, 원격 CPU 작업은 GBDT 만 받아서 섞인 적이 없음 | `fab run` 이 macOS 에서는 GPU 가 없어도 모델마다 새 프로세스로 실행 (GPU 경로의 `run_parallel` 을 작업자 1개로). 모델 하나 안에서 둘을 섞는 것 (`llm_lgbm`·`llm8b_lgbm`, 컨텍스트 래퍼의 `_imp`·`_leaf`) 은 macOS 로컬에서 못 돌림 → 원격 (`fab databricks`·`fab kaggle`) |
| serverless CPU 작업이 정상 종료인데 FAILED | IPython 위에서 실행돼 `sys.exit(0)` 도 예외로 처리 | 종료 코드가 0 이 아닐 때만 `sys.exit` |

## 운영 메모

### Databricks

- 워크스페이스에 따라 Artifact Registry 업로드가 403 (관리자 권한 없음, Preview 미활성으로 추정) → 권한 있는 워크스페이스로 옮김.
- 워크스페이스 Previews 의 "AI Runtime Beta Features" 를 켜야 UC 이미지 실행 가능 (켠 뒤 반영까지 수 분).
- 베이스 이미지(dcs-base-aws-runtime)에 없어 추가한 것: libgomp1 (lightgbm), gcc·libc6-dev·python3.12-dev (kumo 의 triton 컴파일).
- 이미지 빌드는 attestation 을 꺼야 함 (`--provenance=false --sbom=false`). 켜면 OCI index 가 되어 업로드 마지막(manifest)에서 500.

### 로컬 (uv)

- 사내 프록시 환경에서는 `uv run`·`uv sync` 가 `uv.lock` 을 다시 씀 (kaggle-master#41): 사용자 uv 설정(`~/.config/uv/uv.toml`)의 기본 인덱스가 사내 PyPI 프록시라 패키지 `source`·`url` 이 pypi.org 에서 프록시 주소로 바뀌고 `revision` 도 달라짐 (버전은 그대로, 약 2,300줄 변경). 사내망에서는 pypi.org 직접 접속이 막혀 있어 `pyproject.toml` 에 pypi.org 인덱스를 고정할 수도 없음 → 저장소에서는 막지 않음.
- 대신 잠금 파일을 건드리지 않게 실행: `uv run --frozen ...` 또는 셸에 `export UV_FROZEN=1`. 이미 바뀌었으면 `git checkout -- uv.lock`. 의존성을 실제로 바꿔 잠금 파일을 커밋할 때는 프록시 주소가 섞이지 않았는지 diff 로 확인.
- 프로젝트 디렉터리를 옮기면 `.venv` 가 깨짐 (kaggle-master#42): `.venv/bin/` 콘솔 스크립트(`kaggle`, `hf`, `torchrun`, `jupyter`, `tqdm` 등) 첫 줄 shebang 과 `activate` 의 `VIRTUAL_ENV` 가 설치 당시의 절대 경로라 예전 경로를 가리킴. 실행하면 `No such file or directory` 만 나오고 없는 쪽이 python 이라는 게 안 보임 (`fab download` 가 `kaggle` 을 못 찾아 실패해서 발견). `uv sync` 는 패키지가 이미 설치돼 있으면 스크립트를 다시 쓰지 않음.
- 해결: `rm -rf .venv && uv sync --frozen` (uv 캐시에서 복사라 금방 끝남). 패키지 하나만이면 `uv sync --frozen --reinstall-package <이름>` 이지만 `activate` 는 그대로. 확인: `head -1 .venv/bin/kaggle` 이 현재 경로의 python 인지.

### 셸

- zsh 에서 모델 목록을 변수로 넘길 때는 배열(`"${M[@]}"`) 사용. 따옴표 없는 `$M` 은 단어 분리가 안 돼 인자 하나로 전달됨.
