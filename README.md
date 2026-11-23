# fm-agent-bench

표 형식 ML 문제를 네 가지 방식으로 풀고, 같은 채점 기준으로 비교한다.

| 방식 | 내용 | 비용 |
|---|---|---|
| **FM** | 튜닝 없는 Tabular Foundation Model 한 번 추론 (TabPFN, TabICL, Kumo 등, train 전체로 한 번 fit = refit). GBDT(LightGBM·XGBoost·CatBoost)는 기준선 (같은 refit) | 초~분, GPU |
| **Agent** | LLM 코딩 에이전트 (하네스 × LLM: AIDE·Claude Code·Codex CLI × Sonnet 5.5·GPT-6.1 Terra), 1시간 · A10 1장 | 1시간 GPU + 토큰 |
| **FM + Agent 피처** (C1) | 에이전트가 만든 피처를 FM 에 넣어 한 번 추론 | Agent 1회 + FM 한 번 |
| **FM 을 깐 Agent** (C2) | TFM 을 미리 설치한 환경에서 도는 에이전트 (설정 이름 뒤 `+fm`) | Agent 와 같음 |

질문

1. FM 한 번 추론과 에이전트 1시간의 차이는 얼마인가, 어떤 데이터에서 어느 쪽이 이기는가 (비용 대비 포함)
2. (C1) 에이전트가 만든 전처리를 FM 에 넣으면 FM 이 좋아지는가, GBDT 보다 더 좋아지는가
3. (C2) FM 을 쓸 수 있는 에이전트는 더 잘하는가, 실제로 FM 을 쓰는가

벤치마크

- **MLE-bench** (`openai/mle-bench`): Kaggle 대회의 원래 train 을 다시 나눈 test 로 채점하고, 원래 리더보드로 순위·메달을 매기는 벤치마크.
  이 중 표 형식 대회 4개 (spaceship·nomad·tps-may-2022·leaf) 를 씀
- **TabArena** (`autogluon/tabarena` v0.1, 51개): OpenML 고정 분할로 채점하고 공개된 TFM·GBDT 의 split 별 결과와 비교.
  FM (tabicl·kumo 51개, tabpfn_v2 33개) 은 모든 split, Agent 는 1회 1시간이라 15개 (파일럿 2개 × 2회 + #26 10개 × 1회 + 행 추가 kddcup09 · NATICUSdroid · Bioresponse 1회씩) 의 r0f0 × 5설정만 ([docs/tabarena.md](docs/tabarena.md))

계획과 단계 (P0~P6, C1·C2 규칙) 는 [docs/plans/roadmap.md](docs/plans/roadmap.md).

## 성능 대시보드

대회마다 각 방식의 최고 점수 (MLE-bench test), 괄호는 원래 Kaggle 리더보드에서의 위치. 대회별 전체 표·해석은 [docs/mle-bench.md](docs/mle-bench.md).

| 방식 | nomad2018 (RMSLE ↓) | spaceship-titanic (accuracy ↑) | tps-may-2022 (AUC ↑) | leaf-classification (logloss ↓) |
|---|---|---|---|---|
| FM | 0.05723 (16위) | 0.8218 (6위) | 0.99113 (444위)¹ | 0.02059 (229위) |
| GBDT | 0.06036 (34위) | **0.8276 (2위)** | 0.97185 (585위)¹ | 0.35967 (1001위) |
| Agent | 0.04567 (1위) | 0.8241 (2위) | **0.99734 (240위)** | **0.00000 (1위)**² |
| FM + Agent 피처 (C1) | **0.04488 (1위)** | – | 0.99381 (392위)⁴ | 0.00114 (29위)⁴ |
| FM 을 깐 Agent (C2) | 0.04545 (1위)³ | – | 0.99681 (296위)³ | **0.00000 (1위)**²·³ |

**읽는 법**

- FM·GBDT 는 튜닝 없음. test 예측은 refit (CV 없이 train 전체로 한 번 학습, 2026-11-15). 옛 5-fold 값은 [docs/mle-bench.md](docs/mle-bench.md) 의 "refit 으로 다시".
- Agent·C2 는 1~2회 실행 중 최고. – 는 아직 안 돌림.

**각주**

- ¹ 사람 피처 (`~f1`). FM 은 컨텍스트 20만 행. refit 은 컨텍스트를 한 번만 뽑아서 5-fold 의 0.99232 보다 조금 낮음.
- ² 분할 구조로 test 행을 배정한 결과 (외부 데이터 아님).
- ³ C2 최고 실행
  - nomad: Codex × GPT +fm 2회째. TabPFN·TabICL 을 제출에 섞음, Codex Agent 2회째와 같은 수준.
  - leaf: Codex × GPT +fm 2회째와 AIDE × GPT +fm 2회째. 둘 다 분할 구조 배정이고, AIDE 는 FM 을 안 씀.
  - tps: Codex × GPT +fm. 아무도 FM 을 안 씀.
  - FM 을 제출에 넣어 같은 설정의 Agent 보다 두 번 다 좋아진 건 leaf × Claude Code +cont 와 nomad × AIDE × Sonnet ([C2 2회째](docs/mle-bench.md#c2-2회째-leaf--nomad-2026-11-13)).
- ⁴ C1 에 쓴 피처
  - nomad·tps: Codex × GPT. tps 의 FM 은 kumo 컨텍스트 20만 행.
  - leaf: AIDE × Sonnet 1회째 (그 에이전트 1·2회째 점수 0.00385·0.00195), FM 은 tabicl. Codex 피처는 ² 라서 안 씀.

**C1 해석**: leaf 의 FM 향상은 대부분 test 99행 중 한 행이고, CV (5-fold) 로 보면 작음.
leaf·tps 에서는 같은 피처를 넣은 GBDT 가 FM 보다 더 좋아짐. 그래서 질문 2 의 답이 "예" 인 곳은 아직 nomad × kumo 하나.

**결과 파일**

| 파일 | 내용 | 출처 |
|---|---|---|
| `results/mlebench.csv` | FM·GBDT 채점 | Volume 의 `tables/` 원본을 `fab sync` 로 받은 사본 |
| `results/mlebench_runs.csv` | Agent 실행 | 〃 |
| `results/results.csv` | CV | 〃 |
| `results/mlebench_agents.csv` | 공개 리더보드 에이전트 | 로컬 `fab mleb agents` 로 만듦 |

### TabArena

데이터셋마다 각 방식의 최고 점수 (TabArena r0f0 split 의 test). 대회별 전체 표·해석은 [docs/tabarena.md 의 "Agent 확장 1회째"](docs/tabarena.md#agent-확장-1회째-26-2026-11-16).

| 데이터셋 (지표) | FM | GBDT | Agent | FM + Agent 피처 (C1) | FM 을 깐 Agent (C2) |
|---|---|---|---|---|---|
| QSAR_fish (RMSE ↓) | **0.8855** | 0.9123 | 0.9003 | 0.8949 | 0.8885 |
| MIC (logloss ↓) | 0.4412 | 0.4712 | 0.4445 | **0.4280** | 0.4313 |
| Is-this-a-good-customer (AUC ↑) | 0.7435 | 0.6803 | 0.7383 | **0.7451** | 0.7208 |
| splice (logloss ↓) | 0.0669 | 0.0940 | 0.0618 | **0.0601** | 0.0659 |
| students_dropout (logloss ↓) | **0.5251** | 0.5702 | 0.5515 | 0.5271 | 0.5256 |
| polish_bankruptcy (AUC ↑) | 0.9966 | 0.9656 | 0.9956 | 0.9987 | **0.9992** |
| QSAR-TID-11 (RMSE ↓) | 0.7867 | 0.8459 | 0.7976 | 0.7718 | **0.7646** |
| APSFailure (AUC ↑) | 0.9973 | 0.9951 | 0.9952 | **0.9975** | 0.9960 |
| Amazon (AUC ↑) | 0.8723 | 0.8808 | 0.8932 | 0.8701 | **0.8975** |
| physiochemical (RMSE ↓) | 2.852 | 3.961 | 2.977 | 2.868 | **2.840** |
| hazelnut (AUC ↑) | 0.9961 | 0.9738 | 0.9936 | 0.9944 | **0.9962** |
| in_vehicle (AUC ↑) | 0.8629 | 0.8410 | **0.8685** | 0.8673 | 0.8679 |
| kddcup09 (AUC ↑) | 0.8406 | 0.8398 | 0.8441 | 0.8265 | **0.8496** |
| NATICUSdroid (AUC ↑) | 0.9879 | 0.9851 | 0.9915 | 0.9885 | **0.9919** |
| Bioresponse (AUC ↑) | 0.8901 | 0.8887 | 0.8921 | 0.8950 | **0.8973** |

**읽는 법**

- Agent 가 1회 1시간이라 51개 중 15개 (파일럿 2 + #26 10 + 행 추가 3) 의 r0f0 하나에서만 돌림. FM·GBDT 는 51개 모든 split 을 돌렸지만 여기서는 같은 r0f0 값
- FM 은 refit 셋 (kumo·tabicl·tabpfn_v2) 중 최고 (15개 모두 kumo), GBDT 는 lgbm·catboost refit 중 나은 쪽, 튜닝 없음. Agent 는 5설정 × 1~2회 (파일럿 2회, #26 · 행 추가 1회) 중 최고
- 15개 중 1등: FM 9 · Agent 6 (splice · Amazon · in_vehicle · kddcup09 · NATICUSdroid · Bioresponse) · GBDT 0. 앞 넷은 범주 열이 많고 오염 위험도 큼, NATICUSdroid 는 데이터 구조, Bioresponse 는 범주 없음 · 1σ 안 (아래)
- kddcup09 (고카디널리티 범주 11열, [docs/tabarena.md 의 "행 추가: kddcup09_appetency"](docs/tabarena.md#행-추가-kddcup09_appetency-r0f0-2026-11-19--11-20)): Agent 최고가 FM 보다 조금 나음 (−0.38σ, 1σ 안). 1·2등 Agent (AIDE × Sonnet · Codex) 는 데이터 모양으로 데이터셋을 알아봄 → 범주와 오염이 여기서도 갈리지 않음. FM 칸은 우리 kumo: TabArena Kumo (r0f0 AUC 0.8526) 보다 뒤지는 유일한 데이터 (TabArena 의 AutoGluon 전처리 차이), TabArena Kumo 로 견주면 Agent 다섯 모두 FM 보다 나쁨
- NATICUSdroid (대조 행: 범주 86열 모두 0/1 · 덜 알려짐, [docs/tabarena.md 의 "행 추가: NATICUSdroid"](docs/tabarena.md#행-추가-naticusdroid-r0f0-2026-11-20--11-21)): Agent 최고 (Codex) 가 FM 보다 −2.8σ 낫지만 **그 몫은 데이터 구조**. 이 데이터는 꼴 (86열) 이 같은 행이 모두 반대 레이블 짝이라, test 70행 (train 에 짝이 있는 42행 + test 안 짝 28행) 을 Codex 가 후처리로 배정함 (MLE-bench leaf ² 와 같은 꼴). 같은 배정을 kumo 에 하면 0.9916 으로 Codex 보다 낫고, 70행을 빼면 kumo 가 Agent · GBDT · C1 보다 나음 (C2 × Codex +fm 만 조금 나음, −0.2σ). 데이터셋 · 논문을 댄 실행은 없음 → 알려지지 않은 범주 데이터에서 Agent 의 피처 · 모델은 FM 을 넘지 못함. C1 (0.9885, 개수 · 꼴 빈도 · 이웃 수 7열) · C2 (Codex +fm, Kumo 80% + 같은 배정 + test 의사 레이블) 도 같은 70행의 몫이 섞임
- Bioresponse (대조 행: 범주 0 · 1,776열 모두 수치 · 유명한 Kaggle 대회, [docs/tabarena.md 의 "행 추가: Bioresponse"](docs/tabarena.md#행-추가-bioresponse-r0f0-2026-11-21)): Agent 최고 (Codex) 가 FM 보다 조금 나음 (−0.32σ, 1σ 안, FM 보다 나은 Agent 3 / 5). 알아본 실행은 AIDE × Sonnet (첫 계획 글에 "Bioresponse-style dataset") · Codex 짐작 ("Merck on Kaggle") 이지만 각각 3등 · 1등이고, 2등 (알아본 흔적 없음) 과 1등의 차는 0.00004. 최종 코드에 대회에 특정한 기법은 없고 다섯 모두 트리 + 커널 SVM 섞기 → 범주가 없으면 알려진 데이터라도 Agent 의 몫이 작고 알아본 덕도 안 보임. 완전 중복 0 이라 배정 몫은 없음. 이 행은 FM 쪽이 따라잡음: C1 (kumo + PCA 30 · 행 요약 3, 0.8950) · C2 둘 (FM 에 고른 열) 이 Agent 최고보다 나음. kddcup09 · NATICUSdroid 와 나란히 놓은 표는 그 절
- C2 = codex-gpt+fm · aide-sonnet+fm 1회씩 중 최고 (15개). 대부분 FM 근처로 모이고 (30회 중 21회가 1σ 안), FM 보다 뚜렷이 나은 건 polish 와 Agent 가 이미 이기던 Amazon · in_vehicle · NATICUSdroid (Codex +fm, 위 배정 포함) · Bioresponse (Codex +fm, −1.14σ). QSAR-TID · kddcup09 도 C2 가 모든 방식 중 최고 (kddcup09 는 Codex +fm 이 Kumo 에 타깃 인코딩한 범주를 넣음, kumo 보다 −0.99σ) ([docs/tabarena.md 의 "C2 (+fm)"](docs/tabarena.md#c2-fm-fm-을-받은-agent-12개-r0f0-2026-11-18--11-19)). C2 를 더하면 (C1 빼고) 15개 중 1등은 C2 9 (MIC · polish · hazelnut · QSAR-TID · Amazon · physiochemical · kddcup09 · NATICUSdroid · Bioresponse, hazelnut 은 FM 과 0.0001 차) · FM 4 · Agent 2 (splice · in_vehicle)
- C1 = 데이터셋마다 Agent 실행 ("C1 준비" 의 카드로 고른 것) 의 전처리 (`competitions/ta-*/features/f2.py`, train 으로만 fit) 를 넣은 FM (kumo·tabicl refit `~f2`) 중 최고. C1 을 넣으면 (C2 빼고) 1등은 C1 7 · FM 4 · Agent 4, C2 까지 넣으면 C2 8 · C1 4 · FM 2 · Agent 1 (in_vehicle). FM 의 향상이 GBDT 의 향상보다 큰 곳은 15개 중 7개 (QSAR-TID · MIC 등, Amazon 은 둘 다 나빠짐. kddcup09 는 빈도 38열만 옮겨 kumo 가 크게 나빠짐. NATICUSdroid 는 kumo 가 좋아졌지만 구조 70행 밖에서는 나빠짐. Bioresponse 는 PCA 30 · 행 요약 3열로 FM 만 좋아짐: kumo −4.5% · tabicl −1.3%, GBDT 는 그대로), Elo 에는 안 넣음 ([docs/tabarena.md 의 "C1"](docs/tabarena.md#c1-에이전트-피처--fm-r0f0-2026-11-18))
- 피처가 FM 에 준 영향 (C1 · C2, [docs/tabarena.md 의 "C1 · C2 의 피처"](docs/tabarena.md#c1--c2-의-피처-얼마나-바뀌었고-fm-에-영향을-줬나-15개-r0f0-2026-11-23)): 열 수가 늘어난 정도는 FM 변화와 관계없고 (C1 의 늘어난 비율과 kumo 변화의 Spearman 0.09) FE 몫은 대개 GBDT 가 가져감. FM 을 도운 것은 고카디널리티 범주 · 응답자 키의 타깃 의존 통계 (TE · 키별 타깃 집계, C1 규칙 3 밖이라 C2 에만 있음) 와 넓은 데이터의 요약 · 열 고르기 (조금, QSAR-TID 의 C2 × AIDE 는 반대), 좁은 수치 데이터의 비율 · 백색화는 FM 을 그대로 두거나 나쁘게 함. C2 에서 FM 에 원본을 넣은 실행은 대개 FM 한 번 근처 (10회 중 9회 ±0.4σ 안)
- 결과 표: `results/tabarena.csv` (FM·GBDT 채점), `results/tabarena_runs.csv` (Agent 실행)

**Elo** (같은 15개 r0f0, 설정마다)

| 방식 | 방법 | Elo | 95% 구간 |
|---|---|---|---|
| FM | kumo (ours) | **1876** | +223 / −105 |
| FM | tabicl (ours) | 1538 | +133 / −124 |
| FM | tabpfn_v2 (ours) | 1301 | +112 / −160 |
| Agent | Codex × GPT | **1721** | +195 / −138 |
| Agent | AIDE × Sonnet | 1616 | +143 / −107 |
| Agent | AIDE × GPT | 1547 | +118 / −98 |
| Agent | Claude Code × Sonnet +cont | 1478 | +88 / −95 |
| Agent | Claude Code × Sonnet | 1300 | +76 / −127 |
| GBDT (기준점) | catboost (ours) | 1272 | +92 / −126 |
| GBDT (기준점) | lgbm (ours) | 1176 | +100 / −170 |

- TabArena 와 같은 식: RandomForest 기본 = 1000, Bradley-Terry, 데이터셋 부트스트랩 200번 95% 구간, TabArena 공개 결과 (기본 설정) 와 한 풀.
  같은 풀의 공개 결과: Kumo-Tabular 2006 · TabICLv2 1506 · CatBoost 1379 · TabICL v1 1326 · TabPFNv2 1226 · LightGBM 1211 · RandomForest 1000
- Agent 는 설정마다 회차 평균 (위 점수표는 회차 중 최고). 위 표의 칸별 최고를 묶어 Elo 를 매기지 않는 것은 고른 결과라 부풀고, Agent 가 GBDT 를 15개 모두 이겨 값이 안 정해져서
- Agent 넷이 기본 GBDT 보다 위. Codex 는 kumo 보다 156 아래 (기대 승률 29%), AIDE 둘과 함께 tabicl 보다 위. 데이터셋이 15개뿐이라 구간이 넓고, Agent 끼리·tabicl 과의 순서는 단단하지 않음.
  Bioresponse 를 더해 (14개 → 15개) kumo 1926 → 1876, tabicl 1612 → 1538 (Bioresponse 에서 kumo 는 Agent 셋에, tabicl 은 Agent 다섯 · GBDT · RandomForest 에 짐), Codex 1737 → 1721, AIDE × Sonnet 1636 → 1616 (모든 값이 조금씩 바뀜, 같은 풀 정의)
- 모든 방법·평균 순위·승률은 [docs/tabarena.md 의 "Elo"](docs/tabarena.md#elo-2026-11-16). `uv run --frozen fab ta elo` → `results/tabarena_elo.csv`

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
                                  #   C2 (--fm): fm_notes.txt (지시문의 TFM 안내), fm_warmup.py (사전 점검), fm_audit.py (TFM import 기록)
  remote/                         # 원격 상태(Volume 의 state/)·정리 작업 (reconcile.py: 결과 합치기·채점, `fab remote kick` 으로 실행)·fab sync
docker/air/                       # Databricks 실행 이미지 (FM 패키지 미리 설치)
results/                          # git 추적 결과 표 (성능 대시보드 아래 설명, TabArena 는 tabarena*.csv). 실행 출력은 Volume 의 outputs/ (fab sync --pull 로 골라 받기)
docs/                             # mle-bench.md (결과·결정), remote.md (원격 실행 흐름), known-issues.md, plans/
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

학습·에이전트·채점은 Databricks 에서 돌고, 끝난 뒤 `fab remote kick` 으로 결과를 합치고 채점한다 (흐름은 [docs/remote.md](docs/remote.md)).

```bash
uv run --frozen fab                                   # 명령 목록, 명령별 인자는 fab <명령> -h

# FM: MLE-bench 분할 준비 (로컬) → test 예측 (Databricks, `_refit` = train 전체로 한 번. 접미사가 없으면 5-fold CV + test) → MLE-bench test 로 채점 (정리 작업)
uv run --frozen fab download mleb-spaceship-titanic    # Kaggle zip → MLE-bench 분할 → public 만 data/
uv run --frozen fab databricks mleb-spaceship-titanic lgbm_refit catboost_refit tabpfn_v2_refit kumo_cat_refit  # TFM: AI Runtime GPU, GBDT: serverless CPU
uv run --frozen fab mleb grade spaceship-titanic catboost_refit kumo_cat_refit --each      # 채점 요청 (학습이 아직 돌고 있어도 바로 내도 됨)
uv run --frozen fab mleb fm spaceship-titanic=tabpfn_v2_refit,tabicl_kv_refit,kumo_cat_refit leaf-classification=kumo_cat_refit \
    --accelerator GPU_8xH100                          # 대회 여럿을 GPU 작업 하나로 (여러 장이면 대회·모델마다 GPU 하나) + 모델별 채점 요청

# 진행 확인과 결과 받기 (언제든)
uv run --frozen fab remote status                     # 처리 대기 항목·최근 정리 작업 실행
uv run --frozen fab remote kick                       # 정리 작업 한 번 실행 (끝난 실행만 처리)
uv run --frozen fab sync                              # Volume 의 tables/ → results/*.csv
uv run --frozen fab sync --pull outputs/mleb-spaceship-titanic/catboost     # 필요한 예측만 골라 받기

# 공개 리더보드 에이전트 점수 (표 형식 대회만) → results/mlebench_agents.csv
uv run --frozen fab mleb agents

# Agent: AI Runtime GPU(A10)에서 전용 SP 로 1시간 실행 → 회수 → 원격 채점 → results/mlebench_runs.csv
uv run --frozen fab mleb agent run nomad2018-predict-transparent-conductors codex-gpt claude-code-sonnet
uv run --frozen fab mleb agent run nomad2018-predict-transparent-conductors codex-gpt --fm   # C2: TFM 을 깐 환경 (+fm)
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
| `tabpfn_v2` | TabPFN v2 | 라이선스 동의 없이 사용 가능, 분류는 10 클래스까지 (leaf-classification 은 못 씀) |
| `tabpfn` / `tabpfn35` | TabPFN 최신 / 3.5 | 브라우저 라이선스 동의 필요 |
| `tabicl` | TabICL v2 | 10 클래스를 넘으면 계층 분류. KV 캐시 기본 (2026-11-15, OOM 이거나 TabICL 이 거부하면 (10 클래스 넘음) 끔, 옛 동작 `tabicl_nokv`) |
| `kumo` / `kumo_e4` | NVIDIA Kumo-Tabular (large) / estimator 4개 | 회귀는 999 분위수 평균 (지표가 MAE 면 중앙값). 2026-11-15 부터 질의 청크 8192 · 범주 열을 범주로 (옛 동작 `kumo_c2k`·`kumo_numcat`, 새 결과 이름은 `kumo_cat`) |
| `exaone` | EXAONE-Tabular (LG) | 가중치 비상업 라이선스 |
| `limix2m` | LimiX-2M | 비상업 라이선스 |
| `mitra` / `mitra_ft` | Mitra-v2 zero-shot / +50 step fine-tune | 회귀는 v1 체크포인트 |
| `causilo` | Nums AI Causilo | 가중치 비상업 라이선스 |
| `sap_rpt` / `sap_rpt8k` | SAP-RPT-1-OSS, 컨텍스트 2048 / 8192 | HF gated + 연구용 라이선스, `fab databricks` 전용 |

모델 이름 (`@` 앞) 끝의 `_refit` 은 CV 없이 train 전체로 한 번 fit → test 만 (예: `kumo_cat_refit`, `catboost_refit~f2`). 2026-11-15 부터 FM·GBDT 결과는 refit.
결과표의 `opts` 열은 실행 때의 모델 설정 (빈칸 = 2026-11-15 전 실행, [docs/tabarena.md](docs/tabarena.md) 의 "FM 프로토콜과 이름").
이름 뒤 `@<spec>` 은 컨텍스트 선택 래퍼 (예: `tabpfn_v2@knn1000`, `fab/models/context.py`).
맨 뒤 `~<variant>` 는 피처 변형: 그 실행만 `competitions/<대회>/features/<variant>.py` 의 `build` 를 씀 (예: `catboost~f2`,
`tabpfn_v2@knn1000~f2`). 실행 이름(결과 폴더, `results.csv`·`mlebench.csv` 의 `model`, `fab mleb grade` 인자)은 `~` 까지
포함한 전체 이름이고, `results.csv` 의 `features` (변형 이름, 기본은 `base`)·`n_features` (넣은 열 수) 에도 남음.
`build(train, test, data_dir)` 처럼 `data_dir` 를 받으면 대회 데이터 폴더를 넘김 (표 밖 파일을 읽는 변형용).
변형 파일의 `TFM_MAX_FEATURES` 는 그 변형의 TFM 피처 상한 (None = 상한 없음, 없으면 대회 설정 `tfm_max_features`),
`PIP` 는 원격에서 더 설치할 패키지. nomad 의 `~f2s`·`~f2`, leaf·tps-may 의 `~f2` 는 [docs/plans/roadmap.md](docs/plans/roadmap.md) 의 C1.

## 출처

[kaggle-master](https://github.com/Aiden-Jeon/kaggle-master) (비공개로 전환)에서 FM 실행·MLE-bench 부분만 새 히스토리로 옮김.
Kaggle 대회 13개의 FM 벤치마크(Kaggle LB 기준)와 Kaggle·Colab 실행기, Kaggle 제출·순위 집계는 그쪽에 남김.
코드·문서의 `kaggle-master#N` 은 그 저장소의 이슈.
