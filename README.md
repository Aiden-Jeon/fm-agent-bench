# fm-agent-bench

표 형식 ML 문제를 세 가지 방식으로 풀고, 같은 채점 기준으로 비교한다.

| 방식 | 내용 | 비용 |
|---|---|---|
| **FM** | 튜닝 없는 Tabular Foundation Model 한 번 추론 (TabPFN, TabICL, Kumo 등). GBDT(LightGBM·XGBoost·CatBoost)는 기준선 | 초~분, GPU |
| **Agent** | LLM 코딩 에이전트 (하네스 × LLM: AIDE·Claude Code·Codex CLI × Sonnet 5.5·GPT-6.1 Terra), 1시간 · A10 1장 | 1시간 GPU + 토큰 |
| **FM + Agent** | C1: 에이전트가 만든 피처를 FM 에 넣음 / C2: FM 을 도구로 받은 에이전트 | |

질문

1. FM 한 번 추론과 에이전트 1시간의 차이는 얼마인가, 어떤 데이터에서 어느 쪽이 이기는가 (비용 대비 포함)
2. (C1) 에이전트가 만든 전처리를 FM 에 넣으면 FM 이 좋아지는가, GBDT 보다 더 좋아지는가
3. (C2) FM 을 쓸 수 있는 에이전트는 더 잘하는가, 실제로 FM 을 쓰는가

벤치마크

- **MLE-bench** (`openai/mle-bench`): Kaggle 대회를 다시 나눈 test 로 채점, 메달 기준선은 원래 리더보드.
  표 형식 대회 7개. 공개 리더보드 에이전트(24시간) 결과와도 비교 가능
- **TabArena**: 연동 예정 (프로토콜 확인 중, [docs/plans/roadmap.md](docs/plans/roadmap.md))

계획과 단계는 [docs/plans/roadmap.md](docs/plans/roadmap.md), MLE-bench 실행 기록·결정은 [docs/mle-bench.md](docs/mle-bench.md).

## 지금까지 (P0, 2026-11-09~10, 설정마다 1회)

MLE-bench test 점수, 괄호는 원래 Kaggle 리더보드에서의 위치. FM 은 원본 열만 쓰고 튜닝 없음.

nomad2018-predict-transparent-conductors (열별 RMSLE 평균 ↓, 879팀, 금 0.05589)

| 방식 | 점수 | 시간 |
|---|---|---|
| Agent: Codex × GPT | **0.04847 (1위)** | 59분 |
| Agent: Claude Code × Sonnet +cont | 0.05267 (4위) | 5분 |
| Agent: AIDE × Sonnet | 0.05494 (6위) | 60분 |
| FM: kumo | 0.05734 (17위) | 22초 |
| Agent: Claude Code × Sonnet | 0.06134 (42위) | 20초 (스스로 끝냄) |
| Agent: AIDE × GPT | 제출 없음 | 60분 |

spaceship-titanic (accuracy ↑, 2288팀, test 870행)

| 방식 | 점수 | 시간 |
|---|---|---|
| GBDT: catboost / FM: kumo / FM: tabpfn_v2 | **0.8241 (2위)** / 0.8218 (6위) / 0.8207 (11위) | 26초 / 47초 / 100초 |
| Agent: Codex × GPT, AIDE × GPT | 0.8184 (41위) | 58~60분 |
| Agent: Claude Code × Sonnet | 0.8161 (58위) | 18초 |
| Agent: AIDE × Sonnet, Claude Code × Sonnet +cont | 0.8115 (199위) | 60분 / 3분 |

- 표 밖의 정보가 있는 nomad 는 에이전트가 앞섬: Codex 가 결정 구조 파일(`geometry.xyz`)로 대칭·SOAP·그래프 피처를 만듦.
  FM 은 그 파일을 안 봄 → C1 의 첫 대상
- 순수 표 형식인 spaceship 은 FM·GBDT 한 번이 에이전트 1시간보다 앞섬
- 어느 에이전트도 FM 을 쓰지 않음 (환경에 미리 깔지 않음, C2 에서 바꿈)
- 1회씩이라 시드 분산은 모름. 1시간 예산이라 공개 리더보드(24시간)와 섞어 순위 매기지 않음

결과 표: `results/mlebench.csv` (FM 채점), `results/mlebench_runs.csv` (에이전트 실행), `results/mlebench_agents.csv`
(공개 리더보드 에이전트), `results/results.csv` (FM 의 CV). `mlebench_agents.csv` 를 뺀 셋은 원본이 Volume 의 `tables/` 이고
로컬 파일은 `fab sync` 로 받은 사본 (아래 사용).

## 구조

```
competitions/<slug>/config.yaml   # task, target, metric, CV, mlebench: <MLE-bench id>
competitions/<slug>/features.py   # build(train, test) -> (X, X_test, cat_cols)
fab/
  cli.py                          # `fab <명령>` 진입점
  core/                           # 대회 설정·데이터·fold, 공통 피처, 지표, 결과표, 데이터 준비 (fab download)
  models/                         # FM·GBDT 레지스트리 (fab/models/__init__.py), @spec 컨텍스트 선택 (context.py)
  runner/                         # FM 실행: fab databricks 가 제출 (TFM: AI Runtime GPU, GBDT: serverless CPU), 원격에서 fab run
  submission.py                   # test.npy → 제출 파일 (여러 모델이면 평균, 원격 정리 작업 안에서)
  bench/mlebench/                 # MLE-bench 분할 준비 (로컬 mle-bench 가상환경), 채점 재료 업로드·원격 채점 (remote.py), 공개 에이전트 점수
  agent/                          # 에이전트 실행: 로컬 런처 (__init__.py) + 원격 실행기 (runner.py: LLM 프록시·형식 검사 서버)
  remote/                         # 원격 상태(Volume 의 state/)·정리 작업 (reconcile.py: 결과 합치기·채점, `fab remote kick` 으로 실행)·fab sync
docker/air/                       # Databricks 실행 이미지 (FM 패키지 미리 설치)
results/                          # git 추적 표 (위, fab sync 로 받은 사본). 실행 출력은 Volume 의 outputs/ (fab sync --pull 로 골라 받기)
docs/
```

## 준비

1. `uv sync --frozen` (사내 PyPI 프록시 환경에서는 `--frozen` 으로 `uv.lock` 이 바뀌지 않게, [docs/known-issues.md](docs/known-issues.md))
2. mle-bench 를 옆에 따로 설치 (`../mle-bench`, 자체 가상환경): `fab/bench/mlebench/__init__.py` 위쪽 설명. 위치는 `FAB_MLEBENCH`.
   로컬에서는 분할 준비와 정답 위치 확인에만 씀 (채점은 원격)
3. Kaggle API 토큰 (`~/.kaggle/access_token`) + 대회 Rules 동의: MLE-bench 도 원본 zip 은 Kaggle 에서 받음
4. Databricks (FM 실행·에이전트 실행·채점): databricks CLI 로그인 후 저장소 루트에 `databricks.local.yaml` (git 제외).
   그다음 `uv run --frozen fab remote deploy` 로 정리 작업을 만듦 (한 번. 스케줄 없음, `fab remote kick` 으로만 돎)

```yaml
profile: <~/.databrickscfg 프로필>
volume: /Volumes/<catalog>/<schema>/<volume>   # 데이터·실행 결과·원격 상태·결과표·채점용 정답. 에이전트 SP 는 못 읽어야 함
accelerator: GPU_1xH100                         # 선택. FM 실행 기본 GPU
image: <catalog>.<schema>.<image>:<tag>         # 선택. docker/air 로 만든 이미지. 없으면 실행마다 pip 설치
mleb_agent:
  run_as: <서비스 프린시펄 application id>      # 에이전트는 전용 SP 로 실행
  volume: /Volumes/<catalog>/<schema>/<volume>  # SP 가 읽고 쓰는 scratch Volume
  experiment: <MLflow experiment 이름>          # 선택. 기본 fab-mleb-agent
  traces_table: <catalog>.<schema>.<prefix>_otel_spans  # 선택. 에이전트 trace (아래 "에이전트 trace")
  gateway: true                                 # 선택. LLM 요청을 FM API 대신 Unity Gateway 로
```

### 에이전트 trace (선택)

`traces_table` 이 있으면 실행기 프록시가 LLM 호출마다 MLflow 형식 span(요청·응답·토큰·429 대기)을 그 UC 표에 쓰고,
실행 하나를 trace 하나로 묶는다 (`meta.json` 의 `trace_id`, Session = 실행 키). Traces 목록의 Input 은 에이전트가 받은 지시문,
Output 은 한 줄 요약 (제출·종료 방식·LLM 호출 수·토큰). Claude Code 의 OTLP trace(interaction·llm_request·tool 실행)는 프록시가
SP 토큰을 붙여 같은 표로 넘기면서 실행 trace 안으로 옮긴다: root > interaction > llm_request > 프록시 LLM span, interaction > tool
(resource 의 `fab.run` = 실행 키). Codex 의 OTLP 는 끔: 내부 Rust span 만 6분에 약 2.6만 개 나오고 도구·명령 내용이
없음 (2026-11-11 시험) → 명령은 `codex.jsonl`, 요청·응답은 프록시 span. AIDE 는 OTLP 가 없음.
전송 기록은 `otel.jsonl`, 실행 중에는 `fab mleb agent peek` 의 "trace 전달". 보는 곳은 표에 묶인 MLflow experiment 의 Traces 탭.

표 만들기 (한 번, 표 이름·experiment 는 `databricks.local.yaml` 에만 적음):

1. `CREATE SCHEMA <catalog>.<schema>` (default storage catalog 는 안 됨)
2. `mlflow>=3.14` 로 `mlflow.set_experiment(name, trace_location=UnityCatalog(catalog, schema, table_prefix))` → `<prefix>_otel_spans` 등 생김
3. 에이전트 SP 에 `USE SCHEMA` + 표에 `MODIFY, SELECT` (ALL_PRIVILEGES 로는 안 됨, 문서)

Unity Gateway 의 unified trace 표는 쓰지 않음: 설정이 메타스토어 단위라 여러 워크스페이스가 함께 쓰는 메타스토어에서는
다른 워크스페이스의 Gateway 요청까지 그 표에 쌓이고, Beta 동안은 한 번 정한 위치를 옮기기 어려움.

`gateway: true` 는 따로: LLM 요청을 Unity Gateway(`/ai-gateway/anthropic|mlflow|openai/...`, 모델 `system.ai.*`)로 보내
`system.ai_gateway.usage` 에 남긴다 (요청마다 `Databricks-Ai-Gateway-Request-Tags` 에 실행 키). FM API 와 분당 한도·거부 조건이
다를 수 있어 설정 이름에 `+gw` 를 붙이고 (`claude-code-sonnet+gw`) 결과 표의 `llm_route` (`gateway` / `fm_api`)로도
구분한다. 섞어 비교하지 않는다.

## 사용

로컬은 제출과 결과표 동기화·분석만 한다. 학습·에이전트·결과 합치기·채점은 모두 Databricks 에서 돈다 (로컬 `fab run` 은 막혀 있음).
학습·에이전트는 **제출한 뒤 맥북을 꺼도 계속 돌고**, 끝난 뒤 `fab remote kick` 으로 정리 작업을 한 번 돌리면 결과를 합치고 채점한다.

- 로컬 명령은 제출만 하고 Volume 의 `state/` 에 제출 기록을 남김
- 정리 작업 `fab-reconcile` (내 신원, serverless CPU, 스케줄 없음, `fab/remote/reconcile.py`)이 돌 때 끝난 실행을 처리: 예측·OOF 를
  Volume 의 `outputs/` 로 옮기고, `--split-folds` 묶음을 합치고, 채점 요청은 그 모델의 학습이 끝나기를 기다렸다가 채점,
  에이전트 실행은 회수·채점·사후 검사 → Volume 의 `tables/` 결과표에 씀
- 아직 끝나지 않은 실행과 그 모델의 채점 요청은 그대로 남아 다음 kick 때 처리 (몇 번 돌려도 같은 결과)
- 예측 배열은 로컬로 내려오지 않음 (합치기·제출 파일 만들기도 원격, 메모리는 fold·모델 수에 비례하지 않게 하나씩 읽음)
- 정리 작업이 쓰는 fab 코드는 로컬 명령을 낼 때마다 지금 코드로 다시 올라감

```bash
uv run --frozen fab                                   # 명령 목록, 명령별 인자는 fab <명령> -h

# FM: MLE-bench 분할 준비 (로컬) → CV + test 예측 (Databricks) → MLE-bench test 로 채점 (정리 작업)
uv run --frozen fab download mleb-spaceship-titanic    # Kaggle zip → MLE-bench 분할 → public 만 data/
uv run --frozen fab databricks mleb-spaceship-titanic lgbm catboost tabpfn_v2 kumo  # TFM: AI Runtime GPU, GBDT: serverless CPU
uv run --frozen fab mleb grade spaceship-titanic catboost kumo --each      # 채점 요청 (학습이 아직 돌고 있어도 바로 내도 됨)

# 진행 확인과 결과 받기 (언제든)
uv run --frozen fab remote status                     # 처리 대기 항목·최근 정리 작업 실행
uv run --frozen fab remote kick                       # 정리 작업 한 번 실행 (끝난 실행만 처리)
uv run --frozen fab sync                              # Volume 의 tables/ → results/*.csv
uv run --frozen fab sync --pull outputs/mleb-spaceship-titanic/catboost     # 필요한 예측만 골라 받기

# 공개 리더보드 에이전트 점수 (표 형식 대회만) → results/mlebench_agents.csv
uv run --frozen fab mleb agents

# Agent: AI Runtime GPU(A10)에서 전용 SP 로 1시간 실행 → 회수 → 원격 채점 → results/mlebench_runs.csv
uv run --frozen fab mleb agent run nomad2018-predict-transparent-conductors codex-gpt claude-code-sonnet
uv run --frozen fab mleb agent peek                    # 실행 중 진행 (LLM 호출·429)
uv run --frozen fab mleb agent list                    # 처리 상태 (회수·채점은 정리 작업, 결과는 fab sync)
```

타깃이 둘인 대회(nomad2018)는 타깃마다 대회 하나(`mleb-nomad2018-formation`, `-bandgap`)로 돌리고, `fab mleb grade` 가
열을 합쳐 제출 하나로 채점한다.

## 모델

`fab/models/__init__.py` 의 `REGISTRY`·`VARIANTS` 에 등록된 이름.

| 이름 | 모델 | 비고 |
|---|---|---|
| `lgbm` / `xgb` / `catboost` | GBDT 기준선 | 고정 파라미터, early stopping 없음 |
| `tabpfn_v2` | TabPFN v2 | 라이선스 동의 없이 사용 가능 |
| `tabpfn` / `tabpfn35` | TabPFN 최신 / 3.5 | 브라우저 라이선스 동의 필요 |
| `tabicl` | TabICL v2 | |
| `kumo` / `kumo_e4` | NVIDIA Kumo-Tabular (large) / estimator 4개 | 회귀는 999 분위수 평균 (지표가 MAE 면 중앙값) |
| `exaone` | EXAONE-Tabular (LG) | 가중치 비상업 라이선스 |
| `limix2m` | LimiX-2M | 비상업 라이선스 |
| `mitra` / `mitra_ft` | Mitra-v2 zero-shot / +50 step fine-tune | 회귀는 v1 체크포인트 |
| `causilo` | Nums AI Causilo | 가중치 비상업 라이선스 |
| `sap_rpt` / `sap_rpt8k` | SAP-RPT-1-OSS, 컨텍스트 2048 / 8192 | HF gated + 연구용 라이선스, `fab databricks` 전용 |

이름 뒤 `@<spec>` 은 컨텍스트 선택 래퍼 (예: `tabpfn_v2@knn1000`, `fab/models/context.py`).
맨 뒤 `~<variant>` 는 피처 변형: 그 실행만 `competitions/<대회>/features/<variant>.py` 의 `build` 를 씀 (예: `catboost~f2`,
`tabpfn_v2@knn1000~f2`). 실행 이름(결과 폴더, `results.csv`·`mlebench.csv` 의 `model`, `fab mleb grade` 인자)은 `~` 까지
포함한 전체 이름이고, `results.csv` 의 `features` (변형 이름, 기본은 `base`)·`n_features` (넣은 열 수) 에도 남음.
`build(train, test, data_dir)` 처럼 `data_dir` 를 받으면 대회 데이터 폴더를 넘김 (표 밖 파일을 읽는 변형용).
변형 파일의 `TFM_MAX_FEATURES` 는 그 변형의 TFM 피처 상한 (None = 상한 없음, 없으면 대회 설정 `tfm_max_features`),
`PIP` 는 원격에서 더 설치할 패키지. nomad 의 `~f2s`·`~f2` 는 [docs/plans/roadmap.md](docs/plans/roadmap.md) 의 C1.

## 출처

[kaggle-master](https://github.com/Aiden-Jeon/kaggle-master) (비공개로 전환)에서 FM 실행·MLE-bench 부분만 새 히스토리로 옮김.
Kaggle 대회 13개의 FM 벤치마크(Kaggle LB 기준)와 Kaggle·Colab 실행기, Kaggle 제출·순위 집계는 그쪽에 남김.
코드·문서의 `kaggle-master#N` 은 그 저장소의 이슈.
