# MLE-bench: FM vs 하네스 × LLM 에이전트

MLE-bench 의 표 형식 대회에서 FM·GBDT·Agent·C1·C2 를 비교한 결과와, 그 결과를 만든 실행 환경·규칙을 모은 문서.

- 대회별 최고 점수 한 줄 요약: [README](../README.md) 의 "성능 대시보드"
- 앞으로의 단계, C1·C2 의 세부 규칙: [plans/roadmap.md](plans/roadmap.md)
- 처음 내용은 kaggle-master 의 `docs/plans/mle-bench.md` 에서 옮김 (명령·경로는 이 저장소 이름으로 바꿈)

목차

- [용어](#용어)
- [대회별 결과](#대회별-결과) — 결론만 보려면 여기까지
- [실험 설계](#실험-설계)
- [단계별 결과 상세](#단계별-결과-상세): P0 · +cont · P1 · C1 · P3 · C2 · C1 확장
- [에이전트 실행 환경 (트랙 B)](#에이전트-실행-환경-트랙-b)
- [명령](#명령)

## 용어

| 용어 | 뜻 |
|---|---|
| FM | 튜닝 없는 Tabular Foundation Model 한 번 추론 (`tabpfn_v2`·`tabicl`·`kumo`) |
| GBDT | `lgbm`·`xgb`·`catboost`. 고정 파라미터 기준선 |
| Agent | 하네스 × LLM 이 1시간 · A10 1장으로 직접 푼 것 (`codex-gpt`, `claude-code-sonnet`, `aide-sonnet`, `aide-gpt`) |
| +cont | Claude Code 가 일찍 끝나면 같은 세션을 이어 가게 한 별도 설정 ([+cont 규칙](#일찍-끝날-때-cont-규칙)) |
| C1 | 에이전트가 만든 피처를 FM·GBDT 에 넣어 한 번 추론 |
| C2 (+fm) | TFM 3개를 미리 깔아 둔 환경의 Agent |
| F0 / F1 / F2 | 원본 열 / 사람 피처 / 에이전트 피처. 모델 이름 뒤에 `~f1`·`~f2` 처럼 붙임 |
| `@rand200000` | TFM 컨텍스트를 무작위 20만 행으로 (기본은 1만 행) |
| 트랙 A / B | 공개 리더보드 에이전트 점수와 비교 / 직접 돌린 Agent 와 비교 |
| 점수 (N위) | MLE-bench test 점수, 괄호는 원래 Kaggle 리더보드에서 그 점수의 위치 |

## 대회별 결과

2026-11-13 기준 (P0~P3·C1·C2·C1 확장). FM·GBDT 는 튜닝 없음. Agent·C2 는 "1회째 / 2회째" (tps 와 leaf 의 Claude Code 공식 설정은 1회째만).
단계별 자세한 표·해석은 [단계별 결과 상세](#단계별-결과-상세).

### nomad2018-predict-transparent-conductors

열별 RMSLE 평균 ↓ · 879팀 · 금 0.05589

| 방식 | 점수 | 시간 |
|---|---|---|
| **FM + Agent 피처 (C1)**: kumo~f2 / kumo~f2s | **0.04523 (1위)** / 0.04803 (1위) | 15분 / 1분 (피처 계산 포함) |
| FM 을 깐 Agent (C2): Codex × GPT +fm | 0.04589 (1위) / 0.04545 (1위) | 57분 / 57분 (FM 을 제출에 넣음: kumo / TabPFN·TabICL) |
| FM 을 깐 Agent (C2): AIDE × Sonnet +fm | 0.04554 (1위) / 0.04729 (1위) | 60분 (kumo / 세 FM 모두를 제출에 넣음) |
| Agent: Codex × GPT | 0.04847 (1위) / 0.04567 (1위) | 59분 / 57분 |
| GBDT + Agent 피처 (C1): catboost~f2s | 0.04976 (1위) | 23분 |
| FM 을 깐 Agent (C2): Claude Code × Sonnet +cont +fm | 0.13875 (833위) / 0.05006 (1위) | 6분 / 32분 (1회째 에이전트 버그, 2회째 Kumo 약 85% 를 섞어 제출) |
| Agent: Claude Code × Sonnet +cont | 0.05267 (4위) / 0.05157 (2위) | 5분 / 19분 |
| Agent: AIDE × GPT | 제출 없음 / 0.05308 (5위) | 60분 |
| FM 을 깐 Agent (C2): AIDE × GPT +fm | 0.05306 (5위) | 60분 (FM 안 씀) |
| Agent: AIDE × Sonnet | 0.05494 (6위) / 0.05568 (9위) | 60분 |
| FM: kumo (원본 열) | 0.05734 (17위) | 22초 |
| Agent: Claude Code × Sonnet | 0.06134 (42위) / 0.06253 (53위) | 20초 / 40초 (스스로 끝냄) |

- 원본 열만 쓴 FM 보다 Agent 가 앞섬. 이 대회에는 표 밖의 정보(결정 구조 파일 `geometry.xyz`)가 있고, Codex 는 이걸로 피처를 만듦
- 그 피처를 FM 에 넣으면 (C1) FM 한 번이 에이전트 1시간을 넘음 (kumo~f2 0.04523)
- 다만 GBDT 도 비슷하게 좋아짐. "FM 에 특히 도움" 이라 할 만한 건 kumo × 전체 피처 하나뿐
- Agent 2회째에도 설정 순위는 같음
- C2 의 Claude Code 1회째 0.13875 는 FM 탓이 아님: 학습·test 의 열 순서가 어긋난 에이전트 버그 ([C2](#c2-fm-을-깐-agent-nomad-1회째-2026-11-12)).
  2회째는 같은 류 버그 없이 0.05006 (Agent 두 번 0.05267 / 0.05157 보다 좋음)
- C2 의 AIDE × Sonnet +fm 은 두 번 다 FM 을 제출에 넣어 같은 설정의 Agent 두 번과 FM 한 번 (kumo 0.05734) 보다 좋음 (2회째에도 재현)
- C2 최고는 Codex 2회째 0.04545 지만 Codex Agent 2회째 (0.04567) 와 같은 수준. C1 의 kumo~f2 (0.04523) 를 넘은 C2 실행은 없음
  ([C2 2회째](#c2-2회째-leaf--nomad-2026-11-13))

### spaceship-titanic

accuracy ↑ · 2288팀 · test 870행

| 방식 | 점수 | 시간 |
|---|---|---|
| GBDT: catboost / FM: kumo / FM: tabpfn_v2 | **0.8241 (2위)** / 0.8218 (6위) / 0.8207 (11위) | 26초 / 47초 / 100초 |
| Agent: Codex × GPT | 0.8184 (41위) / 0.8241 (2위) | 58분 |
| Agent: AIDE × GPT | 0.8184 (41위) / 0.8149 (77위) | 60분 |
| Agent: Claude Code × Sonnet | 0.8161 (58위) / 0.8058 (375위) | 18초 / 16초 |
| Agent: AIDE × Sonnet | 0.8115 (199위) / 0.8115 (199위) | 60분 |
| Agent: Claude Code × Sonnet +cont | 0.8115 (199위) / 0.8103 (217위) | 3분 / 2분 |

- 순수 표 형식 대회. 방식 사이 차이가 대부분 test 크기에서 오는 흔들림 (정확도 표준오차 약 0.013) 안 → 뚜렷한 차이 없음
- 예외는 Claude Code × Sonnet 2회째 하나 (0.8058, 최고 0.8241 과 0.018 차이, 16초 만에 스스로 끝낸 실행)

### tabular-playground-series-may-2022

AUC ↑ · 1152팀 · 금 0.99823 · test 10만 행

| 방식 | 점수 | 시간 |
|---|---|---|
| Agent: Codex × GPT | **0.99734 (240위)** | 59분 |
| FM 을 깐 Agent (C2): Codex × GPT +fm | 0.99681 (296위) | 60분 (FM 안 씀) |
| Agent: Claude Code × Sonnet +cont | 0.99658 (311위) | 38분 |
| FM 을 깐 Agent (C2): AIDE × Sonnet +fm | 0.99582 (351위) | 60분 (FM 안 씀) |
| FM + Agent 피처 (C1): kumo@rand200000~f2 (Codex 피처, 컨텍스트 20만) | 0.99453 (382위) | 32분 |
| FM 을 깐 Agent (C2): Claude Code × Sonnet +cont +fm | 0.99255 (410위) | 53분 (FM 안 씀) |
| FM: kumo@rand200000~f1 (사람 피처, 컨텍스트 20만) | 0.99232 (417위) | 25분 |
| FM 을 깐 Agent (C2): AIDE × GPT +fm | 0.99174 (428위) | 60분 (FM 안 씀) |
| Agent: AIDE × GPT / Claude Code × Sonnet / AIDE × Sonnet | 0.99156 (434위) / 0.99145 (436위) / 0.99137 (437위) | 60분 / 1분 / 60분 |
| GBDT + Agent 피처 (C1): catboost~f2 | 0.98638 (504위) | 30분 |
| GBDT: catboost~f1 | 0.97223 (581위) | 13분 |
| FM: kumo (원본 열, 컨텍스트 1만) | 0.93168 (708위) | 10분 |

- 컨텍스트를 20만 행으로 늘린 kumo 가 GBDT 최고보다 AUC 0.020 높고, Sonnet·AIDE 에이전트보다 위
- Codex·Claude Code +cont 는 교차 피처·신경망으로 그보다 위. 메달은 아무도 없음
- FM 을 깔아 줘도 (C2) 아무도 FM 을 안 씀 ([C2 마무리](#c2-마무리-leaf--tps-may--nomad-aide-2026-11-12))
- Codex 의 피처를 넣으면 (C1) kumo 0.99232 → 0.99453 이지만 Codex 1시간에는 못 미치고, GBDT 가 더 좋아짐 (오차 −51% 대 kumo −29%,
  [C1 확장](#c1-확장-leaf--aide--sonnet--tps-may--codex-2026-11-13-12))

### leaf-classification

99 클래스 logloss ↓ · 1596팀 · 금 0.0 · test 99행

| 방식 | 점수 | 시간 |
|---|---|---|
| Agent: Codex × GPT | **0.00000 (1위)** / **0.00000 (1위)** (분할 구조를 씀, 아래) | 23분 / 56분 |
| FM 을 깐 Agent (C2): Codex × GPT +fm | 0.00002 (25위) / **0.00000 (1위)** | 29분 / 58분 (분할 구조. 1회째 FM 시험만, 2회째 TabICL 확률로 배정) |
| FM 을 깐 Agent (C2): AIDE × GPT +fm | 0.03090 (399위) / **0.00000 (1위)** | 60분 (1회째 tabicl 시험만, 2회째 FM 안 쓰고 분할 구조) |
| FM + Agent 피처 (C1): tabicl~f2 / kumo~f2 (AIDE × Sonnet 1회째 피처) | 0.00232 (29위) / 0.00603 (58위) | 29초 / 3분 |
| Agent: AIDE × Sonnet | 0.00385 (33위) / 0.00195 (29위) | 60분 |
| FM 을 깐 Agent (C2): AIDE × Sonnet +fm | 0.01392 (142위) / 0.01748 (179위) | 60분 (두 번 다 kumo 를 제출에 넣음) |
| Agent: AIDE × GPT | 0.01821 (193위) / 0.01436 (148위) | 60분 |
| FM 을 깐 Agent (C2): Claude Code × Sonnet +cont +fm | 0.02348 (268위) / 0.01989 (219위)¹ | 37분 / 14분 (kumo·tabicl / kumo 를 제출에 넣음) |
| FM: kumo | 0.01998 (219위) | 3분 |
| Agent: Claude Code × Sonnet +cont | 0.04197 (569위) / 0.05077 (619위) | 10분 / 5분 |
| FM: tabicl | 0.05069 (619위) | 1분 |
| Agent: Claude Code × Sonnet | 0.06297 (682위) | 2분 |
| GBDT + Agent 피처 (C1): catboost~f2 | 0.34624 (991위) | 90분 |
| GBDT: lgbm (최고) | 0.40629 (1015위) | 6분 |

- 클래스당 약 7행이라 GBDT 는 거의 못 배우고 FM 이 크게 앞섬
- Codex 의 0.0 은 데이터 수집 방식을 쓴 결과: "원래 종마다 10장" 에서 test 의 종별 개수를 계산해 test 행을 1:1 로 배정.
  외부 데이터는 아님. nomad 의 정답 섞기와 같은 종류라 따로 읽음
- FM 을 깔아 주면 (C2) 1회째는 네 설정 모두 FM 을 씀, 2회째는 AIDE × GPT 만 안 씀. 두 번 다 제출에 넣은 건 Claude Code +cont
  (Agent 0.042 / 0.051 → 0.023 / 0.020, FM 한 번 kumo 0.020 수준) 와 AIDE × Sonnet (Agent 0.0039 / 0.0020 → 0.0139 / 0.0175, 두 번 다
  Agent 보다 나쁘지만 FM 한 번보다는 좋음)
- 0.0 네 번 (Codex Agent 두 번, Codex +fm 2회째, AIDE × GPT +fm 2회째) 과 Codex +fm 1회째 0.00002 는 모두 분할 구조 배정.
  이 설정들의 C2 − Agent 차이는 FM 이 아니라 이 배정을 찾았는지로 갈림
- ¹ 2회째 4번째 시도. 하네스 14분째 AI Runtime 플랫폼 오류로 노드가 죽어 그때까지 저장한 제출로 채점 (`run_crashed`, 앞 3번은 제출 없이 죽어 뺌,
  [C2 2회째](#c2-2회째-leaf--nomad-2026-11-13))
- AIDE × Sonnet 1회째의 이미지 9열을 넣으면 (C1) tabicl 한 번 0.00232 로 그 에이전트 1회째 (0.00385) 와 2회째 (0.00195) 사이.
  다만 FM 의 test 향상 대부분은 99행 중 한 행이고, CV 로는 FM 이 거의 그대로·GBDT 가 크게 좋아짐 ([C1 확장](#c1-확장-leaf--aide--sonnet--tps-may--codex-2026-11-13-12))

### 모든 대회 공통 주의

- Agent 는 1시간, 공개 리더보드 에이전트는 24시간 → 섞어 순위 매기지 않음
- 조건이 다름: 에이전트는 시간·GPU 를 마음대로 씀 → "FM 한 번 추론 vs 에이전트 N시간" 으로 읽음
- 메달 기준선은 원래 Kaggle 리더보드, 채점은 다시 나눈 test → 근사. tabular playground 는 아주 작은 차이로 메달이 갈림
- LLM 이 공개 Kaggle 풀이를 학습했을 수 있음 (FM 은 아님)
- P0·P1·P3 의 Agent 는 아무도 FM 을 쓰지 않음 (환경에 미리 깔지 않음). C2 에서 바꿈
- C2 의 AIDE × Sonnet +fm 은 처음 3개가 프록시 문제 (#10) 로 실패해 결과에서 빼고 (`fab mleb agent exclude`), 프록시를 고친 뒤 다시 돌린 결과 (2026-11-12)

### 결과 파일

| 파일 | 내용 |
|---|---|
| `results/mlebench.csv` | FM·GBDT 의 MLE-bench test 채점 |
| `results/mlebench_runs.csv` | Agent 실행 (점수·시간·토큰·429·사후 검사) |
| `results/results.csv` | FM·GBDT 의 CV |
| `results/mlebench_agents.csv` | 공개 리더보드 에이전트 점수 (`fab mleb agents`) |

`mlebench_agents.csv` 를 뺀 셋은 원본이 Volume 의 `tables/` 이고, 로컬 파일은 `fab sync` 로 받은 사본.

## 실험 설계

### 질문

1. ML 에이전트(하네스 × LLM)가 몇 시간 들여 만든 풀이와, 튜닝 없는 FM 한 번 추론의 성능이 얼마나 차이 나는가
2. (C1) 에이전트가 만든 전처리(피처)를 FM 에 넣으면 FM 이 좋아지는가, GBDT 보다 더 좋아지는가
3. (C2) FM 을 쓸 수 있는 에이전트는 더 잘하는가, 실제로 FM 을 쓰는가

### MLE-bench 를 쓰는 방식

- **채점**: MLE-bench(`openai/mle-bench`)는 원래 train 을 다시 나눠 자체 test(정답 private)를 만들고 `mlebench grade-sample` 로 채점.
  메달·중앙값 기준선은 원래 Kaggle 리더보드에서 가져옴. → FM 도 Kaggle 리더보드가 아니라 **이 test 로 채점**해야 에이전트와 비교됨
- **공개 리더보드 에이전트**: 약 25개 조합, 24시간 · A10 GPU. `runs/` 에 대회별 채점 결과(점수·기준선·메달)만 있고 풀이 코드는 없음.
  Git LFS 라 `media.githubusercontent.com/media/openai/mle-bench/main/runs/...` 로 받음
- `runs/run_group_experiments.csv` 는 LFS 포인터가 병합 충돌 난 채로 올라가 있어 두 버전을 모두 받아 합침
  (한쪽에만 MARS·MARS+·AIBuildAI, 다른 쪽에만 Famou-Agent-2.0 gemini-3-pro)

### 대회 범위

순수 표 형식부터. 시계열·분자는 나중에. 목록은 `fab/bench/mlebench/__init__.py` 의 `TABULAR`.

| 대회 | 세트 | 특징 | 상태 |
|---|---|---|---|
| spaceship-titanic | dev | 저장소에 `features.py` 있음 | P0 |
| nomad2018-predict-transparent-conductors | low | 작음(약 2.4천 행), 타깃 2개 (RMSLE 평균) | P0 |
| tabular-playground-series-may-2022 | low | train 80만 행 (MLE-bench 분할), 금~동 폭 AUC 0.00005 | P3 |
| leaf-classification | low | 99 클래스 → TabPFN v2 는 못 씀 (10 클래스까지), TabICL·Kumo 는 됨 | P3 |
| playground-series-s3e18 | dev | 다중 타깃(2개) | 안 돌림 |
| tabular-playground-series-dec-2021 | low | 큼(수백만 행), 리더보드가 촘촘해 메달 판정이 민감 | 안 돌림 |
| new-york-city-taxi-fare-prediction | low | 아주 큼 → 컨텍스트 샘플링 필수 | 안 돌림 |

dev 세트 대회는 공개 리더보드 채점 결과에 없음 → 트랙 B 로만 비교.

### 비교 트랙

- **트랙 A (공개 결과)**: 공개 에이전트 점수 vs FM, 같은 MLE-bench test. 거의 무료
- **트랙 B (직접 실행)**: 1회 **1시간** 예산. 풀이 코드가 남아 C1 의 입력이 됨

트랙 B 의 설정 4개:

| 하네스 | LLM | 비교 축 |
|---|---|---|
| AIDE | Sonnet 5.5 | LLM 축 (하네스 고정) |
| AIDE | GPT-6.1 Terra | LLM 축 |
| Claude Code | Sonnet 5.5 | 하네스 축 (LLM 고정) |
| Codex CLI | GPT-6.1 Terra | 하네스 축 |

- 1시간은 공개 리더보드(24시간)보다 훨씬 짧음 → 트랙 A·B 점수를 섞어 순위 매기지 않음
- 실행마다 에이전트가 쓴 모델(특히 TFM 사용 여부)을 기록

### C1: 에이전트 피처 → FM

- 에이전트 최종 풀이에서 전처리만 떼어 `competitions/<slug>/features/<variant>.py` 의 `build(train, test)` 로 옮김 (LLM 변환 + 사람 검토)
- 조건: F0 / F1 / F2 × {tabpfn_v2, tabicl, kumo} + 대조군 {lgbm, catboost}
- 지표: (FM 의 F0 대비 향상) − (GBDT 의 F0 대비 향상). **양수일 때만 "FM 에 특히 도움"**. 점수는 같은 fold 의 CV + MLE-bench test 둘 다
- 추출·동일성 확인·누수 규칙은 roadmap 의 "C1 규칙"

### C2: FM 을 깐 Agent

`--fm` 이면 tabpfn_v2·tabicl·kumo 를 깔고 가중치를 받아 둔 뒤 (하네스 1시간 밖), 지시문에 이름과 사용 예(`fab/agent/fm_notes.txt`)를 더함.
나머지는 공식 설정과 같음. 세부 규칙과 결과 표의 열은 roadmap 의 "C2 규칙".

## 단계별 결과 상세

시간순. 단계 이름은 roadmap 과 같음.

### P0: FM 채점 (2026-11-09)

튜닝 없음, 원본 피처.

nomad2018 (열별 RMSLE 평균 ↓, 금 0.05589 · 중앙값 0.06988, 879팀)

| 모델 | MLE-bench test | LB 순위 | 공개 에이전트 33개 실험 (시드 중앙값) 중 |
|---|---|---|---|
| kumo | 0.05734 | 17 | 8위 (앞: MARS+ 0.0499, MARS, LoongFlow, Disarray, PiEvolve 24h, AIRA-dojo, Famou-2.0 gemini-3-pro 0.0568) |
| tabicl | 0.05838 | 22 | |
| tabpfn_v2 | 0.05874 | 25 | |
| catboost | 0.06001 | 31 | |
| lgbm | 0.06107 | 40 | |
| xgb | 0.06434 | 76 | |

- FM 은 결정 구조 파일(`geometry.xyz`) 없이 원본 열만, 몇 초~수십 초. 공개 에이전트는 12~24시간
- R&D-Agent(gpt-5) 0.0607, AIBuildAI(Opus 4.6) 0.0626, AIDE(o1-preview) 0.0600 보다 kumo·tabicl·tabpfn_v2 가 좋음
- FM 은 1회 실행, 공개 에이전트는 시드 중앙값 → 같은 기준이 아님 (FM 의 시드 분산은 안 잼, roadmap 의 "P1 범위")

spaceship-titanic (accuracy ↑, test 870행 → 1행 = 0.11%p, 공개 에이전트 결과 없음)

| 모델 | catboost | kumo | tabpfn_v2 | tabicl | xgb | lgbm |
|---|---|---|---|---|---|---|
| test | 0.8241 | 0.8218 | 0.8207 | 0.8184 | 0.8184 | 0.8172 |

CV 1위(kumo 0.8254)와 test 1위가 다름.

### P0: Agent 1회째 (2026-11-09)

1시간 · A10 · 1회씩. 원본 기록은 `results/mlebench_runs.csv`.

nomad2018 (RMSLE ↓, 금 0.05589, 879팀)

| 설정 | 점수 | LB | 하네스 시간 | LLM 호출 | 비고 |
|---|---|---|---|---|---|
| Codex × GPT-6.1 Terra | **0.04847** | **1** | 59분 (스스로 끝냄) | 350 | 결정 구조 파일로 대칭·국소 환경(SOAP, `dscribe`)·그래프 피처. `ase`·`dscribe`·`torch-geometric` 직접 설치. 429 대기 4분 |
| AIDE × Sonnet 5.5 | 0.05494 | 6 | 60분 (시간 종료) | 7 | |
| (FM: kumo, 원본 열만) | 0.05734 | 17 | 22초 | – | 트랙 A |
| Claude Code × Sonnet 5.5 | 0.06134 | 42 | 20초 (스스로 끝냄) | 4 | 원본 열 + 부피, LightGBM 하나 |
| AIDE × GPT-6.1 Terra | 제출 없음 | – | 60분 (시간 종료) | 66 | 아래 |

- 공개 리더보드 최고 (MARS+, 24시간 · H100 2장) 의 시드 중앙값 0.0499 보다 Codex × GPT 1시간이 좋음 (1회 결과라 시드 분산은 모름)
- AIDE × GPT 의 제출 없음: GPT 가 코드를 `if __name__ == "__main__"` 로 감쌌는데, AIDE 가 빈 전역 범위로 exec 해서 `main()` 이 안 돎.
  28개 중 27개가 버그

spaceship-titanic (accuracy ↑, 금 0.82066, 2288팀)

| 설정 | 점수 | LB | 하네스 시간 | LLM 호출 |
|---|---|---|---|---|
| (FM·GBDT: catboost / kumo / tabpfn_v2) | 0.8241 / 0.8218 / 0.8207 | 2 / 6 / 11 | 26초 / 47초 / 100초 | – |
| Codex × GPT | 0.8184 | 41 | 58분 (스스로 끝냄) | 338 (429 대기 7분) |
| AIDE × GPT | 0.8184 | 41 | 60분 (시간 종료) | 51 |
| Claude Code × Sonnet | 0.8161 | 58 | 18초 (스스로 끝냄) | 4 |
| AIDE × Sonnet | 0.8115 | 199 | 60분 (시간 종료) | 17 |

정리

- **하네스 성격이 뚜렷함**: Codex 는 시간을 거의 다 쓰며 수백 번 호출, Claude Code 는 20초 · 4턴 만에 끝냄,
  AIDE 는 코드 실행 시간이 길어 호출이 적음 → Claude Code 는 남은 시간을 이어 쓰는 별도 설정(+cont)으로 한 번 더 돌림 (바로 아래)
- **FM**: 어느 실행도 FM 을 쓰지 않음 (Codex 가 spaceship 에서 TabPFN·AutoGluon 설치 여부만 확인)
- **비용**: 채점에 넣은 8회의 하네스 시간 합 A10 약 6시간 (+ 실행마다 설치 약 2분). 입력 토큰 8,800만 (대부분 Codex, 그중 97% 캐시), 출력 84만.
  제외한 실행 8회는 대부분 시작 직후 실패 (AIDE 3회·Codex 1회는 각 1시간 가까이, 목록은 [운영 기록](#운영-기록))
- C1 의 첫 대상: nomad × Codex 의 결정 구조 피처

### +cont: 일찍 끝나는 Claude Code 이어 가기 (2026-11-10)

규칙은 [+cont 규칙](#일찍-끝날-때-cont-규칙). 1회씩, 이때 상한은 "이어 가기 20번".

이어 가기는 원격에서 동작했지만 두 대회 모두 **20번 상한**에 걸려 1시간 중 3~5분만 씀.
Sonnet 이 이어 가라는 글에 "제출은 확정, 더 안 바꿈" 이라며 한 턴(약 2초)에 끝내는 일이 많음.

| 대회 | 점수 (공식 설정 → +cont) | 하네스 시간 | LLM 호출 | 첫 종료 | 실제로 작업한 이어 가기 | 거절(도구 0개)한 이어 가기 | 마지막 작업 |
|---|---|---|---|---|---|---|---|
| spaceship | 0.8161 → 0.8115 (199위) | 3분 14초 (20번 상한) | 30 | 24초 | 4번 (CV·CatBoost 설정·피처·LightGBM 혼합) | 16번 (마지막 11번 연속) | 2분 56초 |
| nomad | 0.06134 (42위) → **0.05267 (4위, 금)** | 5분 20초 (20번 상한) | 37 | 21초 | 6번 (LightGBM+CatBoost 혼합, `geometry.xyz` 이웃 거리·동경 분포 피처) | 14번 (마지막 7번 연속) | 5분 4초 |

- nomad: 결정 구조 피처를 이어 가기에서 처음 씀. 공식 설정의 Claude Code 는 "약해서" 가 아니라 **"일찍 끝내서"** 낮았음.
  그래도 Codex 1시간 (0.04847) 보다는 못함
- spaceship: test 870행에서 4행 차이, 잡음 범위. CV 0.812 에서 더 못 올림
- 토큰: spaceship 입력 109만 / 출력 7천, nomad 144만 / 1.2만
- 해석: 어느 상한이든 Claude Code +cont 는 사실상 3~5분짜리 설정. 남은 시간을 알려 줘도 모델이 할 일을 더 찾지 못함

**상한을 바꿈 (2026-11-11)**: 도구 0개로 끝난 구간이 **연속 3번이면 멈춤** (`idle`), 안전장치로 전체 40번 (`max`). 근거는 위 두 실행:

- 연속 1번은 짧음: 거절 뒤 다시 작업하는 경우가 있음 (nomad 는 거절 5번 연속 뒤 13번째에 다시 작업, spaceship 은 4번 뒤 9번째).
  연속 1번이면 nomad 의 제출 개선 3번을 놓침
- 연속 2·3·5번은 두 실행 모두 최종 제출이 같음 (제출이 바뀐 마지막 이어 가기는 nomad 7번째, spaceship 1번째).
  다시 작업한 구간도 점수가 안 올라 제출을 안 바꿈 → 여유를 하나 둔 3. 5 는 거절로 호출만 늘림
- "무작업" = 그 구간의 도구 호출 0개 (Claude Code 의 `tool_use`, Codex 의 명령 실행 등 item).
  "제출이 안 바뀜" 으로 세면 실험한 뒤 일부러 안 바꾼 구간까지 무작업이 되어 쓰지 않음
- 상한 없이 남은 시간 내내 이어 가면: 거절 한 번(약 2초)마다 대화 전체를 다시 보내 1,500번 넘게 부름 (입력 토큰 수천만 이상, 429)
- 이어 가기 글은 그대로 둠 (더 구체적으로 시키면 코칭이 되어 다른 설정이 됨)

이 2회는 다시 돌리지 않음. 기록을 구간별로 다시 재생하면 새 규칙에서는 nomad 11구간 (이어 가기 10번, 약 4분 20초),
spaceship 8구간 (7번, 약 2분 30초) 에서 멈추고 최종 제출은 그대로 → 점수 같음.
단 `harness_seconds`·`continues`·토큰 열은 옛 규칙(20번) 값이라 P1 의 2회째와 섞어 보지 않음.

### P1: Agent 2회째 (2026-11-11)

같은 설정을 한 번 더 (1시간 · A10). +cont 는 새 상한 (도구 0개 구간 연속 3번이면 멈춤).

nomad (RMSLE ↓)

| 설정 | 1회째 | 2회째 | 2회째 시간 |
|---|---|---|---|
| Codex × GPT | 0.04847 (1위) | **0.04567 (1위)** | 57분, 429 34번 |
| Claude Code × Sonnet +cont | 0.05267 (4위) | 0.05157 (2위) | 18분 48초 (이어 가기 9번) |
| AIDE × GPT | 제출 없음 | 0.05308 (5위) | 60분 |
| AIDE × Sonnet | 0.05494 (6위) | 0.05568 (9위) | 60분 |
| Claude Code × Sonnet | 0.06134 (42위) | 0.06253 (53위) | 40초 (스스로 끝냄) |

spaceship (accuracy ↑)

| 설정 | 1회째 | 2회째 | 2회째 시간 |
|---|---|---|---|
| Codex × GPT | 0.81839 (41위) | **0.82414 (2위)** | 58분, 429 56번 |
| AIDE × GPT | 0.81839 (41위) | 0.81494 (77위) | 60분 |
| AIDE × Sonnet | 0.81149 (199위) | 0.81149 (199위) | 60분 |
| Claude Code × Sonnet +cont | 0.81149 (199위) | 0.81034 (217위) | 2분 21초 (이어 가기 7번) |
| Claude Code × Sonnet | 0.81609 (58위) | 0.80575 (375위) | 16초 (스스로 끝냄) |

- **nomad 는 설정 순위가 두 번 다 같음** (Codex > +cont > AIDE > Claude Code). 점수 흔들림은 0.001~0.003.
  1회째에 제출이 없던 AIDE × GPT 는 이번엔 제출함 → 실패는 설정의 성질이 아니라 실행마다의 일
- **spaceship 은 1·2회째 차이가 0.000~0.010** 으로 정확도 표준오차(약 0.013) 안 → "방식 사이에 뚜렷한 차이 없음" 그대로.
  Codex 2회째 0.82414 는 FM·GBDT 최고 (catboost 0.82414) 와 같음
- **+cont 새 상한**: 두 실행 모두 `idle` 로 멈춤. nomad 는 무작업 1번 뒤 7번째 구간에서 14분을 더 작업함 (연속 1번 상한이었으면 놓쳤을 작업).
  구간별 도구 수: spaceship [2, 1, 3, 0, 1, 0, 0, 0], nomad [2, 4, 2, 2, 1, 0, 3, 0, 0, 0]
- 429 는 여전히 Codex 만 (34~56번). 어느 실행도 FM 을 쓰지 않음

### C1: 에이전트 피처 → FM (2026-11-11, nomad × Codex)

F2 의 정의·추출은 roadmap 의 "nomad × Codex 의 F2". 요약하면 `~f2s` 는 사람이 읽을 수 있는 물리 피처 (원본·격자·대칭·RDF·이웃 통계, 1,109열),
`~f2` 는 여기에 SOAP·위상·푸리에·스펙트럼을 더한 타깃 비의존 피처 전부 (38,832열).

MLE-bench test (열별 RMSLE 평균 ↓). 괄호는 원래 리더보드 순위와 F0 대비 변화. 튜닝 없음, 실행 1회.

| 모델 | F0 (원본 12열) | F2s (1,109열) | F2 (38,832열, FM 은 상위 500열) |
|---|---|---|---|
| kumo | 0.05734 (17위) | 0.04803 (1위, −16.2%) | **0.04523 (1위, −21.1%)** |
| tabpfn_v2 | 0.05874 (25위) | 0.04972 (1위, −15.4%) | 0.04929 (1위, −16.1%) |
| tabicl | 0.05838 (22위) | 0.04972 (1위, −14.8%) | 0.04931 (1위, −15.5%) |
| catboost | 0.06001 (31위) | 0.04976 (1위, −17.1%) | 0.04992 (1위, −16.8%) |
| lgbm | 0.06107 (40위) | 0.04998 (1위, −18.2%) | 0.05102 (2위, −16.5%) |
| (참고) Codex × GPT 1시간 | 0.04847 / 0.04567 (1·2회째) | | |

CV (같은 fold 5개): F0 → F2s → F2

| 모델 | formation | bandgap | 학습 시간 (두 타깃 합, 초) |
|---|---|---|---|
| kumo | 0.0316 → 0.0194 → 0.0195 | 0.0841 → 0.0741 → 0.0730 | 45 / 71 / 898 |
| tabpfn_v2 | 0.0321 → 0.0212 → 0.0203 | 0.0856 → 0.0753 → 0.0749 | 71 / 207 / 1,008 |
| tabicl | 0.0317 → 0.0212 → 0.0205 | 0.0845 → 0.0762 → 0.0753 | 13 / 125 / 916 |
| catboost | 0.0353 → 0.0235 → 0.0235 | 0.0912 → 0.0801 → 0.0805 | 25 / 1,398 / 40,442 |
| lgbm | 0.0359 → 0.0229 → 0.0227 | 0.0942 → 0.0806 → 0.0800 | 6 / 200 / 6,031 |
| (참고) Codex 최종 풀이 | 0.0179 | 0.0707 | |

- **에이전트 피처 + FM 한 번이 에이전트 1시간을 넘음**: `kumo~f2` 0.04523 < Codex 0.04567·0.04847.
  같은 피처의 GBDT 도 모두 금메달 선 (0.05589) 안. 에이전트 점수의 상당 부분이 피처에서 나왔다는 뜻
  (Codex 의 CNN·GNN 같은 모델 쪽은 F2 로 옮기지 않음)
- **"FM 에 특히 도움" 은 갈림** (C1 지표 = FM 향상 − GBDT 향상):
  - F2s: GBDT 가 더 좋아짐 (−17~18% 대 FM −15~16%)
  - F2: kumo 만 GBDT 보다 더 좋아짐 (−21% 대 −17%). tabpfn·tabicl 은 GBDT 와 비슷
  - → 에이전트 피처는 FM 과 GBDT 를 비슷하게 돕고, FM 은 F0 때의 앞섬을 유지한 채 같이 올라감.
    FM 에 특히 이득이라는 근거는 kumo × F2 하나 (실행 1회. 짝지은 비교라 노이즈는 작지만 확정은 아님)
- **F2 (전부) 는 F2s 보다 거의 안 나음** (kumo 만 test −0.003). 비용은 FM 약 5~13배, GBDT 약 30배 (catboost 는 대회당 약 5.6시간)
  → 사람이 읽을 수 있는 물리 피처 (RDF·이웃 통계·대칭) 만으로 대부분이 나옴
- 해석 주의: test 에 train 과 같은 구조가 있음 (Codex 가 정답을 섞던 행). SOAP 같은 기술자를 받은 FM 은 비슷한 행을 보고 예측하므로
  그 효과를 피처로 정당하게 얻을 수 있음 → kumo × F2 의 추가 이득이 여기서 왔을 수 있음 (확인 안 함)

### P3: tps-may-2022 · leaf-classification (2026-11-12)

#### 준비

fab 대회 2개를 더함. 분할은 MLE-bench 그대로 (`fab download`, 체크섬 일치).

| fab 대회 | train / test | 지표 | 메달 기준선 (원래 LB) | 공개 에이전트 (실험별 시드 중앙값) |
|---|---|---|---|---|
| `mleb-tps-may-2022` | 800,000 / 100,000, 31열 | AUC ↑ | 금 0.99823 · 은 0.99822 · 동 0.99818 · 중앙값 0.97268 (1,152팀) | 31개 실험, 금은 Disarray 0.99825 하나. 다음 MLEvolve 0.99771 · MARS 0.99760 |
| `mleb-leaf-classification` | 891 / 99, 192열, 99 클래스 | logloss ↓ | 금 0.0 · 은 0.00791 · 동 0.01526 · 중앙값 0.10835 (1,596팀) | 33개 실험 중 17개가 메달 선. MARS·MARS+·Famou-2.0 gemini-2.5-pro 는 0.0 (금) |

피처

- leaf F0: margin·shape·texture 192개 (대회가 이미지에서 미리 뽑은 수치). 잎 이미지(`images/`, 990장)는 안 씀 — nomad 의 `geometry.xyz` 처럼 에이전트만 씀
- tps F0: 원본 31열. `f_27` (10글자 문자열, 80만 행에 고유값 67만 개, test 값의 28% 만 train 에 있음) 은 범주형 하나로 둠
  → FM 에는 사전순 정수 코드라 거의 정보가 없음
- tps `~f1` (사람 피처): `f_27` 을 글자 위치별 10열 + 서로 다른 글자 수로 바꿈 (41열). 대회 당시 공개 노트북에서 널리 쓴 처리.
  F0 만 두면 "FM 이 이 한 가지 처리를 못 해서 지는" 비교가 되므로 같이 둠. C1 의 F2 (에이전트 피처) 와는 다름

모델 제약

- 클래스 수 (leaf): TabPFN v2 는 10 클래스까지라 fit 에서 거부 (`TabPFNValidationError`, tabpfn 9.2) → 표에서 뺌.
  TabICL 은 10 을 넘으면 계층 분류로 자동 전환 (`support_many_classes` 기본값). Kumo 도 됨 (로컬에는 패키지가 없어 실행 전엔 몰랐음).
  클래스당 train 이 최소 6행이라 층화 5-fold 의 학습 쪽에 99 클래스가 모두 들어감
- xgb 는 tps F0 에서 뺌: `f_27` 범주 67만 개 때문에 로컬 8코어에서 64만 행 · 20트리 28초 → 800트리 5-fold 면 CPU 4코어에서 몇 시간.
  lgbm 은 50트리 1.3초 (범주 분할 상한이 있음). `~f1` 은 범주가 없어 xgb 도 됨
- FM 컨텍스트 (tps): 대회 설정 `tfm_max_rows: 10000` (kaggle-master 관례), 큰 컨텍스트는 `@rand200000`.
  kaggle-master 의 비슷한 크기 (S6E1·S6E10, 60~70만 행, H100) 에서 kumo@rand200000 약 33분, tabicl@rand200000 약 42분, catboost 약 35분 (5-fold 합)

분류 확률의 행 합 맞추기

- 예측할 때 행 합을 1 로 맞춤 (`fab/core/metrics.py` 의 `normalize_proba`, `fab run` 의 predict 와 제출 둘 다). NaN·합 0·뚜렷한 음수는 에러
- 까닭: FM 은 float32·float16 확률이라 99 클래스면 합이 1 에서 벗어남 (float16 균등 확률이면 3e-5).
  leaf 채점기는 1 ± 1e-6 이 아니면 무효로 보고, sklearn 1.9 log_loss 는 경고만 하고 그대로 계산 → CV logloss 와 채점 점수가 다른 확률을 보게 됨
- 이 변경 전 실행은 다시 돌리지 않음: spaceship 의 지표 accuracy 는 argmax 라 그대로 (보조 지표 auc·logloss 는 조금 달라질 수 있음)

확인

- Agent 노드 (A10G, 12코어, 메모리 51GB, `/tmp` 500GB, P0·P1 실행 기록) 에서 tps (CSV 283MB) 는 문제없음
- 로컬: 피처 모양 (leaf 891×192, tps F0 800,000×31 · `~f1` ×41), 합성 예측으로 만든 제출 파일을 로컬 mle-bench 로 채점해 두 대회 모두
  `valid_submission=True` (leaf 는 두 모델 평균까지)

#### 결과: leaf-classification

1회씩. "공개 실험 중" = 공개 리더보드 에이전트 (24시간) 의 실험별 시드 중앙값 사이 위치 (`results/mlebench_agents.csv`, 참고만).

logloss ↓ · 1,596팀 · 금 0.0 · 은 0.00791 · 동 0.01526 · 중앙값 0.10835

| 방식 | test | 공개 실험 중 | 시간 | CV (5-fold) |
|---|---|---|---|---|
| Agent: Codex × GPT | **0.00000 (1위, 금)** | 공동 1위 | 23분 (스스로 끝냄) | |
| Agent: AIDE × Sonnet | 0.00385 (33위, 은) | 10위 / 33 | 60분 | |
| Agent: AIDE × GPT | 0.01821 (193위) | | 60분, 429 12번 | |
| FM: kumo | 0.01998 (219위) | 18위 | 2분 44초 (A10) | 0.0096 ± 0.0084 |
| Agent: Claude Code × Sonnet +cont | 0.04197 (569위) | | 10분 (이어 가기 10번) | |
| FM: tabicl | 0.05069 (619위) | 21위 | 54초 (A10) | 0.0148 ± 0.0092 |
| Agent: Claude Code × Sonnet | 0.06297 (682위) | | 2분 19초 (스스로 끝냄) | |
| GBDT: lgbm / catboost / xgb | 0.40629 (1,015위) / 0.45896 (1,037위) / 0.88510 (1,293위) | 24위~ | 6분 / 84분 / 3분 | 0.279 / 0.327 / 0.758 |

- **GBDT 는 이 대회에서 안 됨**: fold 학습에서 클래스당 약 7행이라 트리가 거의 못 배움. FM 은 같은 192열로 GBDT 최고 (lgbm 0.406) 의
  1/8 (tabicl) ~ 1/20 (kumo). FM 대 GBDT 차이가 지금까지 대회 중 가장 큼 (nomad 는 비슷, spaceship 은 GBDT 가 1위)
- **Kumo 는 99 클래스를 그대로 받음**, TabICL 은 계층 분류. TabPFN v2 는 10 클래스까지라 안 돌림
- **Codex 의 0.0 은 분할 구조를 쓴 결과**
  - 원래 데이터가 종마다 정확히 10장 → test 의 종별 개수 = 10 − train 개수를 계산
  - test 99행을 종별 자리에 1:1 로 배정 (`scipy.optimize.linear_sum_assignment`, 점수는 192열 + 이미지 거리 모델) → 98/99행에 확률 1.0
  - 외부 데이터·Kaggle 풀이는 안 봄 (`kaggle_urls` 0). 공개 리더보드 1위권 (MARS 등 0.0) 도 비슷한 방법일 수 있음 (확인 안 함)
  - nomad 의 "정답 섞기" 와 같은 종류 (데이터 수집 방식에서 오는 사전 정보) → 에이전트 대 FM 비교에서 따로 읽음.
    같은 배정을 FM 확률에 얹으면 FM 도 0.0 근처가 될 수 있음 (안 해 봄. C1 처럼 "모델 몫이 아닌 몫")
- AIDE 둘은 배정을 안 씀: 잎 이미지 거리 + 정규화 판별 분석(RDA) 등 고전 모델 (88~89/99행 확률 > 0.999). Claude Code 는 표만 씀
- CV 와 test 순서는 FM·GBDT 에서 같음 (test 99행이라 test 흔들림은 큼)

#### 결과: tabular-playground-series-may-2022

AUC ↑ · 1,152팀 · 금 0.99823 · 동 0.99818 · 중앙값 0.97268

| 방식 | test | 공개 실험 중 | 시간 |
|---|---|---|---|
| Agent: Codex × GPT | **0.99734 (240위)** | 5위 / 30 | 59분 (스스로 끝냄), 429 135번 (대기 9분 24초) |
| Agent: Claude Code × Sonnet +cont | 0.99658 (311위) | 7위 | 38분 (이어 가기 9번) |
| **FM: kumo@rand200000~f1** | **0.99232 (417위)** | 10위 | 25분 (H100) |
| Agent: AIDE × GPT | 0.99156 (434위) | | 60분 |
| Agent: Claude Code × Sonnet | 0.99145 (436위) | | 1분 23초 (스스로 끝냄) |
| Agent: AIDE × Sonnet | 0.99137 (437위) | | 60분 (LLM 호출 3번) |
| FM: tabicl@rand200000~f1 | 0.98189 (534위) | | 32분 (H100) |
| FM: kumo~f1 (컨텍스트 1만) | 0.97442 (572위) | | 11분 (A10) |
| GBDT: catboost~f1 / xgb~f1 / lgbm~f1 | 0.97223 (581위) / 0.96626 (600위) / 0.96585 (601위) | 18위~ | 13분 / 3분 / 5분 |
| FM: tabicl~f1 / tabpfn_v2~f1 (컨텍스트 1만) | 0.95646 (628위) / 0.93095 (710위) | | 5분 / 26분 (A10) |
| F0 (`f_27` 그대로): kumo / catboost / lgbm / tabicl / tabpfn_v2 | 0.93168 / 0.91743 / 0.90474 / 0.89663 / 0.87639 | | |

- **`~f1` (f_27 글자 분해) 이 모든 모델을 AUC 0.04~0.06 올림**. F0 의 순위는 그대로 (kumo > catboost > lgbm > tabicl > tabpfn_v2)
- **컨텍스트가 FM 순위를 정함**: 1만 행이면 kumo~f1 0.9744 ≈ catboost~f1 0.9722, 20만 행이면 0.9923 으로 GBDT 최고보다 0.020 높음
  (kaggle-master 의 S6E1·S6E10 과 같은 방향). tabicl 도 1만 → 20만에서 0.956 → 0.982
- **FM 한 번 (25분) 이 Sonnet 에이전트 셋과 AIDE × GPT 보다 위, Codex·Claude Code +cont 보다 아래**.
  위쪽 두 에이전트는 `f_27` 분해에 더해 교차 피처·신경망을 씀 (Codex: PyTorch MLP + GBDT 앙상블)
  → 메달권 (0.998) 은 피처 간 상호작용을 찾아야 나옴. 어느 쪽도 메달은 없음
- Claude Code +cont 가 공식 설정 (1분 23초, 0.99145) 보다 크게 오름: 이어 가기 9번에 38분을 씀 (지금까지 +cont 중 가장 김, nomad P1 은 19분)
- CV 와 test 순서가 거의 같음 (test 10만 행)
- 어느 에이전트도 FM 을 쓰지 않음. Codex 의 `kaggle_urls` 1 은 대회 설명(description.md)의 시작 노트북 링크를 기록에 다시 출력한 것 (사후 검사 오탐)

#### 실행 메모

- FM GPU 작업의 H100 이 약 1시간 동안 안 잡힘 (PENDING, 워크스페이스에 다른 H100 작업 없음).
  컨텍스트 1만 행 FM 과 leaf 는 A10 으로 다시 냄 (`results.csv` 의 note `p3 (A10, H100 대기)`), `@rand200000` 둘만 H100.
  A10 에서 tabpfn_v2 는 1만 행 tps 에 21~26분
- 비용: Agent 10회 A10 하네스 시간 합 6.2시간 (+ 실행마다 설치), 입력 토큰 6,580만 (Codex 둘이 6,290만), 출력 53만.
  FM 학습 시간 합 (준비·대기 제외) GPU 2.3시간 (A10 1.4 + H100 0.95), GBDT CPU 2.9시간 (leaf catboost 99 클래스 84분, tps catboost F0 53분)

### C2: FM 을 깐 Agent (nomad 1회째, 2026-11-12)

#### 준비

- 규칙은 roadmap 의 "C2 규칙". `--fm` 이면 FM 3개를 깔고 가중치를 받아 둔 뒤 지시문에 이름과 사용 예를 더함
- 실행 중 TFM import 기록 (`fab/agent/fm_audit.py`) 은 공식 설정에도 깖. 에이전트에게 알리지 않고, 동작도 바꾸지 않음
- 로컬에서 확인한 것: import 기록 (실제 import 만 남고 `importlib.util.find_spec` 은 안 남음, `importlib.resources` 정상),
  사후 검사 열 (합성 기록), 안내문 조각 나누기·코드 블록 파싱, CLI
- 사용 예 실행 (사전 점검) 은 로컬에서 못 돌림 → 첫 원격 실행의 `meta.json` 의 `fm_warmup`·`fm_build_tools` 로 확인.
  실패하면 하네스를 시작하지 않아 LLM 비용은 안 들고, 그 행은 `exclude`
- 순서: P0·P1 처럼 Sonnet 줄과 GPT 줄을 하나씩 동시에. 먼저 nomad (C1 에서 "에이전트 피처 + FM" 이 강했던 대회), 그다음 spaceship.
  비교 대상은 같은 설정의 Agent 1·2회째

#### 결과

열별 RMSLE 평균 ↓. FM 사용 열 (`tfm_runtime`·`tfm_runtime_procs`·`tfm_at_submit`) 은 `results/mlebench_runs.csv` 의 값 (사후 검사 규칙을 고치면 `fab mleb agent recheck` 로 다시 계산).

| 설정 | C2 (+fm) | Agent 1회째 / 2회째 | 시간 | 실제로 쓴 FM (`tfm_runtime`, 프로세스 수) | 제출에 든 FM (`tfm_at_submit`) |
|---|---|---|---|---|---|
| Codex × GPT | **0.04589 (1위)** | 0.04847 / 0.04567 | 57분, 429 39번 | kumo·tabicl·tabpfn (55, kumo 41) | kumo |
| Claude Code × Sonnet +cont | 0.13875 (833위) | 0.05267 / 0.05157 | 6분 14초 (이어 가기 9번) | tabicl·tabpfn (6) | tabicl·tabpfn |

- **두 에이전트 모두 FM 을 실제로 씀** (P0·P1 에서는 아무도 안 씀). 사전 점검은 두 실행 모두 세 모델 통과 (모델당 10~30초),
  Kumo 컴파일 도구는 노드에 이미 있음 (`fm_build_tools: present`)
- **Codex**: 16분부터 Kumo 를 결정 구조 피처와 함께 계속 씀 (`cv_kumo_*.py` 등. 공간군·상별 Kumo, 커널·GNN 과 앙상블).
  점수는 2회째 (0.04567) 와 같은 수준 → 이 1회로는 FM 덕에 나아졌다고 못 함. C1 의 `kumo~f2` (0.04523) 와도 비슷
- **Claude Code: 에이전트 버그로 test 만 깨짐**
  - 교차 검증(OOF) 은 RMSLE 약 0.0495 (Agent 때보다 나음) 인데 test 는 0.139
  - `c.py` 부터 학습 행렬은 [기본 열, `sg`, 결정 구조 피처] 순서, test 행렬은 [기본 열, 결정 구조 피처, `sg`] 순서
    (`X['sg']` 는 결정 구조 피처를 붙이기 전에, `T['sg']` 는 뒤에 만듦) → TabPFN·TabICL 에 `.values` 로 넘겨 열이 위치로 어긋남
  - CV 는 학습 행렬만 써서 못 잡음. 제출은 `c3.py` 가 만든 `sub6.csv` 를 `cp` 한 것 (LightGBM 0.15 + TabPFN 0.45 + TabICL 0.40)
  - FM 환경 탓은 아님: 열 이름으로 맞추는 GBDT 만 썼다면 안 깨졌을 버그 → FM 의 numpy 인터페이스가 드러낸 것에 가까움

#### 이 실행에서 고친 것

- 사전 점검: Codex 실행의 Kumo 분류 사전 점검 정확도가 0.0 (Claude 실행은 1.0) → 확률 열 순서가 실행마다 다름.
  사용 예에 열을 클래스 순서로 정렬하는 줄을 넣고 (fab 의 Kumo 와 같은 방식), 사전 점검이 정확도·RMSE 기준도 보게 함
  (nomad 는 회귀라 이 두 실행의 점수와는 상관없음)
- `tfm_at_submit`: 처음엔 제출 파일 시각 (`cp` 한 때) 으로 봐서 Claude Code 가 빈칸이었음 → 작업 디렉터리에서 제출과
  내용이 같은 파일 중 가장 이른 시각으로 봄

### C2 마무리: leaf · tps-may · nomad AIDE (2026-11-12)

nomad 는 FM 과 GBDT 차이가 작아 "FM 을 줘서 나아졌나" 가 안 갈림 → FM 이 크게 앞서는 대회에서 돌림 (leaf: FM 이 GBDT 의 1/8~1/20 오차,
tps-may: kumo 20만 행 컨텍스트가 GBDT 보다 AUC 0.02 높음). 1시간 · A10 · rep 1. 비교는 같은 설정의 Agent (P3·P0·P1).
뺀 것: spaceship (차이가 test 흔들림 안), claude-code-sonnet 공식 설정 (몇 초 만에 스스로 끝나 FM 을 쓸 틈이 없음 → +cont 로 봄).

**aide-sonnet+fm 은 처음 3개가 실패라 결과에서 빼고 다시 돌림 (#10)**: Sonnet 이 긴 프롬프트(+fm 안내문)에 Chat Completions
content 를 파트 목록으로 답했고 프록시가 그대로 넘겨 AIDE 의 코드 추출이 `TypeError` 를 냄.

- tps: 첫 응답에서 → AIDE 가 종료 코드 1 로 끝남 (하네스 24초, 제출 없음)
- nomad (3번째 응답)·leaf (15번째): 예외 뒤 AIDE 기록이 끊겼는데 (AIDE 첫 기록부터 nomad 는 25초, leaf 는 13분 뒤) 프로세스가 끝나지 않아
  1시간 제한에서 강제 종료됨 (`timed_out`, exit 없음). 그 전에 저장된 제출은 채점됐음 (nomad 0.06189, leaf 0.02965) → 표에 안 넣음
- 프록시를 고치고 (2026-11-12, "FM API 호환 처리") 세 대회를 다시 돌림 (`…__aide-sonnet-fm__r1__20261112-2259`·`-2311`·`-2312`).
  셋 다 1시간 끝까지 돎, 파트 목록 응답을 문자열로 바꾼 호출이 leaf 3 · tps 11 · nomad 18 번 (`proxy.jsonl` 의 `flattened`)
- 다시 돌린 3개는 **동시에** 돌림 ("같은 엔드포인트는 한 번에 하나" 의 예외, 아래 "동시 실행 부하 확인")

"안 씀" 은 끝까지 돈 실행에서 FM import 가 없었다는 뜻.

#### 결과: leaf-classification

logloss ↓, 금 0.0 · 동 0.01526

| 설정 | Agent | C2 (+fm) | 쓴 FM (`tfm_runtime`) | 제출에 든 FM (`tfm_at_submit`) |
|---|---|---|---|---|
| Claude Code × Sonnet +cont | 0.04197 (569위) | **0.02348 (268위)** | kumo · tabicl | kumo · tabicl |
| Codex × GPT | 0.00000 (1위, 금) | 0.00002 (25위, 은) | kumo · tabicl | 없음 |
| AIDE × GPT | 0.01821 (193위) | 0.03090 (399위) | tabicl | 없음 |
| AIDE × Sonnet | 0.00385 (33위, 은) | 0.01392 (142위, 동) | kumo · tabicl · tabpfn | kumo |
| (참고) FM 한 번: kumo / tabicl | 0.01998 / 0.05069 | | | |

#### 결과: tabular-playground-series-may-2022

AUC ↑, 동 0.99818

| 설정 | Agent | C2 (+fm) | 쓴 FM |
|---|---|---|---|
| Codex × GPT | 0.99734 (240위) | 0.99681 (296위) | 안 씀 |
| Claude Code × Sonnet +cont | 0.99658 (311위) | 0.99255 (410위) | 안 씀 |
| AIDE × GPT | 0.99156 (434위) | 0.99174 (428위) | 안 씀 |
| AIDE × Sonnet | 0.99137 (437위) | 0.99582 (351위) | 안 씀 |
| (참고) FM 한 번: kumo@rand200000~f1 | 0.99232 (417위) | | |

#### 결과: nomad2018 (AIDE 두 설정)

RMSLE ↓, 금 0.05589

| 설정 | Agent 1회째 / 2회째 | C2 (+fm) | 쓴 FM |
|---|---|---|---|
| AIDE × GPT | 제출 없음 / 0.05308 (5위) | 0.05306 (5위) | 안 씀 |
| AIDE × Sonnet | 0.05494 (6위) / 0.05568 (9위) | **0.04554 (1위, 금)** | kumo · tabicl · tabpfn (kumo 를 제출에 넣음) |
| (참고) FM 한 번: kumo | 0.05734 (17위) | | |

#### 해석

1회째 기준. leaf · nomad 를 2회째까지 보고 다시 읽은 것은 [C2 2회째](#c2-2회째-leaf--nomad-2026-11-13) 의 해석.

- **FM 을 쓰는지는 대회에 따라 갈림**: leaf 는 네 설정 모두 FM 을 import 함 (P3 의 Agent 는 아무도 안 씀). tps 는 아무도 안 씀 —
  코드 import 0, 메시지에서도 언급 없음 (Codex·Claude Code 확인). 이유는 확인 안 함: train 80만 행이고, 안내문에 TabPFN 1만 행 제약을
  적었으며, FM 이 앞서려면 20만 행 컨텍스트를 줘야 함 (P3). nomad 는 Codex·Claude Code (위 1회째)·AIDE × Sonnet 이 씀, AIDE × GPT 는 안 씀
- **FM 을 제출에 넣어 같은 설정의 Agent 보다 좋아진 건 둘**: leaf × Claude Code +cont (0.0420 → 0.0235, 작업 10분 → 37분, FM 한 번
  kumo 0.01998 에는 못 미침) 와 nomad × AIDE × Sonnet (0.0549 / 0.0557 → 0.0455, FM 한 번 kumo 0.0573 보다도 좋음).
  반대로 leaf × AIDE × Sonnet 은 kumo 를 제출에 넣고 Agent 보다 나빠짐 (0.0039 → 0.0139, 그래도 FM 한 번보다는 좋음).
  Codex 는 leaf 에서 FM 을 시험만 하고 분할 구조 배정 (P3) 으로 다시 거의 0 점, nomad 에서는 Kumo 를 제출에 넣었지만 2회째와 같은 수준
- 나머지 차이는 FM 탓으로 볼 수 없음: FM 을 제출에 안 넣은 실행끼리의 차이 (leaf × AIDE × GPT 0.018 → 0.031, tps × Claude Code +cont
  0.9966 → 0.9926, tps × AIDE × Sonnet 0.9914 → 0.9958) 는 실행마다의 흔들림이거나, 안내문을 읽고 FM 을 시험하는 데 1시간 일부를 쓴 몫 (1회씩이라 못 가름)
- 질문 3 의 지금 답: FM 을 깔아 주면 FM 이 앞서는 작은 데이터(leaf)와 nomad 에서는 에이전트가 쓰고, 두 실행은 그 덕에 같은 설정의
  Agent 보다 좋아짐 (한 실행은 나빠짐, 1회씩). FM 이 큰 컨텍스트를 줘야 앞서는 큰 데이터(tps)에서는 아무도 안 씀
- FM 을 쓴 C2 실행 중 FM 한 번보다 좋은 건 Codex 둘 (nomad 0.04589 < kumo 0.05734, leaf 0.00002 < kumo 0.01998) 과 AIDE × Sonnet 둘
  (nomad 0.04554, leaf 0.01392). Codex 는 nomad 에서 Agent 2회째 (0.04567) 와 같은 수준이고 leaf 는 FM 을 제출에 안 넣은 분할 구조
  배정이라 FM 덕이라고 할 근거는 없음. AIDE × Sonnet 은 두 대회 다 kumo 를 제출에 넣음
- 같은 대회의 다른 Agent 최고 (nomad Codex 2회째 0.04567, leaf Codex 0.0, tps Codex 0.99734) 를 뚜렷이 넘은 C2 실행은 없음.
  nomad × AIDE × Sonnet +fm (0.04554) 이 0.0001 앞서지만 1회씩이라 같은 수준으로 봄

#### 동시 실행 부하 확인 (2026-11-12)

aide-sonnet+fm 3개 (leaf · tps · nomad) 를 같은 워크스페이스·Sonnet 엔드포인트에서 동시에 돌리고, 혼자 돈 aide-sonnet 실행 7개의
`proxy.jsonl` 과 비교. 혼자 7개 = P1 2개 (nomad·spaceship `r2`) + P3 2개 (leaf·tps) + C2 첫 시도 3개 (#10 으로 실패한 것, 실패 전까지의
호출: leaf 15 · nomad 3 · tps 1번). P0 의 aide-sonnet 2개는 실행별 진행 기록 (`state/agent`) 이전이라 `proxy.jsonl` 을 못 받아 뺌.
AIDE 는 1시간 입력이 1.7만~61만 토큰 (`results/mlebench_runs.csv` 의 AIDE 18개) 이라 분당 한도 (약 150만) 를 거의 안 씀.
동시 3개의 입력 합도 1시간 약 102만 (분당 평균 약 1.7만)

| | 혼자 (7개) | 동시 (3개) |
|---|---|---|
| 429 | 0 | 0 |
| 호출당 지연 중앙값 | 6.8~12.4초 | 2.9~9.3초 |
| 출력 속도 중앙값 (출력 토큰/초) | 115~166 | 114~165 |
| 출력 속도 하위 10% | 71~133 | 67~91 |

- 부하 없음: 429 없고 출력 속도 중앙값이 혼자일 때 범위 안. nomad 의 지연 p90 (32초, 혼자 16초) 은 답이 길어서 (출력 p90 6,469 토큰,
  혼자 3,110)
- 하위 10% 출력 속도는 동시 쪽이 조금 낮지만 혼자 쪽 범위 (71~) 와 겹침. 3개 · 1회라 작은 차이는 못 가름
- → AIDE 실행끼리는 같은 엔드포인트에서 동시에 돌려도 조건이 바뀌지 않는다고 봄. Codex 는 1시간 입력이 4,200만~4,500만 토큰 (P0 에서 429)
  이라 그대로 한 번에 하나

### C2 2회째: leaf · nomad (2026-11-13)

C2 마무리의 결과가 모두 1회라 "FM 을 제출에 넣어 같은 설정의 Agent 보다 좋아진 두 실행" (leaf × Claude Code +cont, nomad × AIDE × Sonnet) 과
반대 방향 (leaf × AIDE × Sonnet) 이 반복해도 나오는지 봄 (#13). 1시간 · A10 · rep 2. leaf 는 공식 설정 Agent 도 1회뿐이라 Agent 2회째도 같이 돌림.
뺀 것: tps-may (1회째 아무도 FM 을 안 씀), spaceship (차이가 test 흔들림 안), nomad × AIDE × GPT +fm (1회째 FM 을 안 씀).

- 순서: GPT 줄은 Codex 셋 (leaf +fm → leaf → nomad +fm) 을 하나씩, 그 뒤 leaf × AIDE × GPT 둘 (+fm·Agent) 을 동시에.
  Sonnet 줄은 AIDE 셋 (leaf +fm·leaf·nomad +fm) 을 동시에, Claude Code +cont 셋을 그 사이에 하나씩 ("동시 실행 부하 확인")
- 429 는 nomad × Codex +fm 만 84번 (대기 6분). 다른 실행은 0
- leaf × Claude Code +cont +fm 은 4번 돌림: 플랫폼 오류 (아래 "실행 메모") 로 앞 3번은 제출 없이 죽어 `exclude`, 4번째는 실행기에
  "진행 중 제출 사본" 을 넣은 뒤라 죽기 전 마지막 제출로 채점 (`run_crashed`)

"쓴 FM" 은 `tfm_runtime` (실행 중 import). "제출에 든 FM" 의 2회째는 최종 풀이 코드·AIDE 기록으로 확인한 것. `tfm_at_submit` (근사) 과
다른 곳: leaf × AIDE × Sonnet (`tabicl`, 실제는 Kumo 단독: 블렌드 탐색이 Kumo 가중치 1.0 을 고름, Kumo 는 하위 프로세스라 안 잡힘),
leaf × Claude Code (`kumo · tabicl`, 실제는 Kumo 단독: Kumo 를 돌린 프로세스가 tabicl 도 import), nomad × Codex·Claude Code (빈칸,
실제는 FM 예측을 파일로 저장해 다른 프로세스에서 섞음).

#### 결과: leaf-classification

logloss ↓, 금 0.0 · 동 0.01526. "1회째 / 2회째"

| 설정 | Agent | C2 (+fm) | 쓴 FM (2회째) | 제출에 든 FM (1회째 / 2회째) |
|---|---|---|---|---|
| Claude Code × Sonnet +cont | 0.04197 / 0.05077 | **0.02348 / 0.01989**¹ | kumo · tabicl | kumo · tabicl / kumo |
| Codex × GPT | 0.00000 / 0.00000 | 0.00002 / 0.00000 | kumo · tabicl · tabpfn | 없음 / tabicl (분할 구조 배정의 확률) |
| AIDE × GPT | 0.01821 / 0.01436 | 0.03090 / 0.00000 | 안 씀 | 없음 / 없음 (2회째 분할 구조 배정) |
| AIDE × Sonnet | 0.00385 / 0.00195 | 0.01392 / 0.01748 | kumo · tabicl · tabpfn | kumo / kumo |
| (참고) FM 한 번: kumo / tabicl | 0.01998 / 0.05069 | | | |

¹ 2회째는 하네스 14분째 (이어 가기 7번째 구간) 에 노드가 죽음. 그때 제출은 Kumo 단독 (에이전트 CV 0.0179). 남은 46분은 못 씀.
  `results/mlebench_runs.csv` 의 `continues` 6 은 고치기 전 회수 규칙 (돌던 구간을 안 셈) 의 값, 실행기 기록으로는 7

#### 결과: nomad2018

RMSLE ↓, 금 0.05589. "1회째 / 2회째"

| 설정 | Agent | C2 (+fm) | 쓴 FM (2회째) | 제출에 든 FM (1회째 / 2회째) |
|---|---|---|---|---|
| Codex × GPT | 0.04847 / 0.04567 | 0.04589 / **0.04545** | kumo · tabicl · tabpfn (88 프로세스) | kumo / TabPFN·TabICL 저장 예측을 섞음 |
| AIDE × Sonnet | 0.05494 / 0.05568 | **0.04554 / 0.04729** | kumo · tabicl · tabpfn | kumo / kumo · tabicl · tabpfn |
| Claude Code × Sonnet +cont | 0.05267 / 0.05157 | 0.13875 / **0.05006** | kumo · tabicl · tabpfn | tabicl · tabpfn (열 순서 버그) / Kumo 약 85% |
| (참고) FM 한 번: kumo / C1: kumo~f2 | 0.05734 / 0.04523 | | | |

- nomad × Claude Code +cont +fm 2회째 (32분, 이어 가기 13번): 1회째 같은 열 순서 버그 없음. 최종 제출은 Kumo 예측 셋 (결정 구조 피처 포함,
  `kf.py`·`k.py`) 0.35·0.35·0.15 + 앞 블렌드 (`sub6.csv`) 0.15

#### 해석

- **세 방향 모두 2회째에도 같음**:
  - leaf × Claude Code +cont: 두 번 다 FM 을 제출에 넣고 (kumo·tabicl / kumo) Agent 두 번보다 좋음 (0.042 / 0.051 → 0.023 / 0.020).
    점수는 FM 한 번 (kumo 0.01998) 수준이고, 2회째는 14분에 끝난 실행
  - nomad × AIDE × Sonnet: 두 번 다 kumo 를 제출에 넣고 Agent 두 번보다 좋음 (0.0549 / 0.0557 → 0.0455 / 0.0473), FM 한 번 (0.0573) 보다도 좋음
  - leaf × AIDE × Sonnet: 두 번 다 kumo 를 제출에 넣고 Agent 두 번보다 나쁨 (0.0039 / 0.0020 → 0.0139 / 0.0175). FM 한 번보다는 좋음.
    이 설정의 Agent 는 FM 없이 FM 한 번보다 10배 좋아서, FM 으로 바꾸면 오히려 손해
- 새로 본 것: nomad × Claude Code +cont 도 2회째는 Kumo 를 제출에 넣고 Agent 두 번보다 좋음 (0.0527 / 0.0516 → 0.0501). 1회째는 버그라 비교 못 함
- FM 을 쓰고도 같은 설정의 Agent 와 같은 수준: nomad × Codex (Agent 0.0485 / 0.0457, C2 0.0459 / 0.0455). 두 번 다 FM 을 제출에 넣었지만
  Agent 2회째와 차이 0.0002 안
- leaf 의 0.0 은 FM 과 무관: Codex 는 Agent·+fm 모두, AIDE × GPT 는 +fm 2회째에 분할 구조 배정을 찾음 (AIDE × GPT +fm 2회째는 FM 을 안 씀).
  AIDE × GPT 의 C2 − Agent 가 1회째 +0.013, 2회째 −0.014 로 부호가 바뀐 것도 FM 이 아니라 이 배정을 찾았는지 차이
- **쓰는지 (질문 3)**: 1회째에 FM 을 쓴 7 설정 중 6 개가 2회째에도 씀 (leaf 넷 중 AIDE × GPT 만 안 씀, nomad 셋 모두 씀).
  제출에 넣는 FM 은 실행마다 바뀌기도 함 (nomad × Codex kumo → TabPFN·TabICL, nomad × Claude Code TabICL·TabPFN → Kumo,
  nomad × AIDE × Sonnet kumo → 세 FM)
- **FM 한 번 · C1 대비**: nomad 에서 FM 을 제출에 넣은 C2 는 모두 FM 한 번 (kumo 0.05734) 보다 좋지만 C1 의 kumo~f2 (0.04523) 를 넘은 건 없음
  (최고 Codex 2회째 0.04545). leaf 에서 FM 을 제출에 넣은 실행 중 (분할 구조 배정인 Codex 2회째는 빼고) FM 한 번 (0.01998) 보다 뚜렷이 좋은 건 AIDE × Sonnet 둘,
  Claude Code 는 1회째가 나쁘고 2회째가 같은 수준
- 지금 답: FM 을 깔아 주면 에이전트는 대부분 다시 씀. Agent 가 FM 한 번보다 약하거나 비슷한 설정 (leaf × Claude Code +cont: 0.042~0.051
  대 0.020 / nomad × AIDE × Sonnet: 0.055 대 0.057) 은 FM 을 제출에 넣어 두 번 다 좋아짐 (nomad × Claude Code +cont: 0.052 대 0.057 은
  버그 없는 2회째만). Agent 가 FM 한 번보다 훨씬 강한
  설정 (leaf × AIDE × Sonnet: 0.002~0.004 대 0.020) 은 FM 을 넣어 반복해서 나빠짐. 2회씩이라 규칙이라기보다 지금까지의 모양

#### 실행 메모

- **leaf × Claude Code +cont +fm 이 4번 모두 AI Runtime 플랫폼 오류로 죽음**: 2분째 SIGTERM (exit 143) 1번, `Your compute failed because of a
  Databricks platform issue` 3번 (12·15·14분째). 뒤 3번은 모두 에이전트가 leaf (99 클래스) 에 Kumo 분류를 `num_estimators` 를 8 에서 16·24 로
  올려 돌리던 중 (GPU 메모리를 잡은 채 사용률 0%, 시스템 메모리 약 6GB 로 OOM 아님). 같은 시간대의 다른 실행 10개 (Kumo 를 쓴 실행 포함) 는 정상
  → 이 조합에서 노드가 죽는 것으로 보임 (원인은 확인 안 함, 같은 조건 재현 시험 안 함)
- 실행기가 진행 중 제출 사본을 남기게 고침 ("구현" 의 "진행 중 제출 사본"). 4번째 시도는 2분마다 제출 사본이 올라간 것을 확인했고, 노드가 죽은 뒤
  정리 작업이 그 사본으로 채점함

### C1 확장: leaf × AIDE × Sonnet · tps-may × Codex (2026-11-13, #12)

nomad 는 F0 에서 FM 과 GBDT 차이가 작아, FM 이 GBDT 를 크게 앞서는 두 대회에서 질문 2 를 다시 봄. F2 의 정의·추출·동일성 확인은
roadmap 의 "leaf × AIDE × Sonnet 의 F2"·"tps-may-2022 × Codex × GPT 의 F2". 튜닝 없음, 실행 1회. leaf 의 TFM 은 A10 (F0 와 같음),
tps 의 TFM 은 H100 `@rand200000` (`~f1` 과 같음), GBDT 는 CPU. 결과 표의 note `c1 (A10)` (leaf) · `c1` (tps).

#### 결과: leaf-classification

logloss ↓, test 99행, 금 0.0 · 은 0.00791 · 동 0.01526. F2 = 원본 192 + 에이전트 이미지 9열 (201열)

| 모델 | F0 test | F2 test | CV F0 → F2 (같은 fold 5개) | F2 가 나은 fold | 학습 시간 F2 |
|---|---|---|---|---|---|
| tabicl | 0.05069 (619위) | **0.00232 (29위, 은)** | 0.01476 → 0.01350 (−8.6%, −0.0013) | 4/5 | 29초 |
| kumo | 0.01998 (219위) | 0.00603 (58위, 은) | 0.00960 → 0.00969 (+1.0%, +0.0001) | 3/5 | 3분 |
| catboost | 0.45896 (1037위) | 0.34624 (991위) | 0.3269 → 0.2366 (−27.6%, −0.090) | 5/5 | 90분 |
| lgbm | 0.40629 (1015위) | 0.36952 (1004위) | 0.2794 → 0.2382 (−14.8%, −0.041) | 4/5 | 5분 |
| (참고) AIDE × Sonnet 1시간 1회째 / 2회째 | 0.00385 (33위, 은) / 0.00195 (29위, 은) | | 1회째 0.01669 (에이전트 자기 fold) | | |

- **test 로는 FM 이 크게 좋아짐** (tabicl −95%, kumo −70% 대 GBDT −9%·−25%). tabicl~f2 한 번 (0.00232) 이 피처를 만든 AIDE × Sonnet 1회째 (0.00385) 보다 좋고 2회째 (0.00195) 보다는 나쁨
- **다만 F0 test 는 한 행이 좌우**: kumo·tabicl 둘 다 같은 행 (id 451, 정답 Quercus_Pontica 를 Quercus_Agrifolia 로) 에서 손실 1.35 · 4.17 →
  test 손실 합의 68% · 83%. F2 에서는 이 행이 맞음 (F2 의 최악 행 손실 0.21 · 0.11). 가장 나쁜 3행을 빼면 kumo 0.0043 → 0.0028,
  tabicl 0.0058 → 0.0007 (tabicl 은 99행 중 88행이 나아짐, kumo 는 69행)
- **CV 로는 반대**: FM 은 거의 그대로 (kumo 는 평균이 같고 3/5 fold 만 나음, tabicl 은 4/5 fold 에서 −0.0013), GBDT 는 크게 좋아짐
  (catboost 는 5 fold 모두). C1 지표 (FM 향상 − GBDT 향상) 는 % 로도 절대 차이로도 음수 → "FM 에 특히 도움" 아님.
  FM 은 F0 에서 이미 CV 0.01 근처라 9열이 더할 몫이 작고, GBDT 는 바닥 (0.28~0.33) 에서 출발
- 읽는 법: test 의 큰 FM 향상은 99행 중 한 행의 몫이라 CV 를 따름 → 에이전트 이미지 9열은 GBDT 를 더 돕고, FM 은 tabicl 만 조금.
  그래도 FM + 에이전트 피처 한 번이 에이전트 1시간과 같은 수준 (test 0.0023 대 1·2회째 0.0039·0.0020, CV 0.0135 대 1회째 0.0167 은 fold 가 달라 참고만).
  에이전트의 이득 대부분은 모델 쪽 (LR · PCA-LR · MLP 섞기) 이었고, 그 몫을 FM 이 대신함
- 이미지 9열은 실제로는 배경 화소의 통계 + 이미지 크기 (roadmap). 원본 192열에 없는 잎 크기 정보가 들어간 것으로 봄 (열별 기여는 확인 안 함)

#### 결과: tabular-playground-series-may-2022

AUC ↑, test 10만 행, 동 0.99818. F1 = 사람 피처 41열, F2 = 에이전트 피처 244열. 괄호의 % 는 F1 대비 오차 (1 − AUC) 변화

| 모델 | F0 | F1 | F2 | CV F1 → F2 | 학습 시간 F1 → F2 |
|---|---|---|---|---|---|
| kumo@rand200000 | – | 0.99232 (417위) | **0.99453 (382위, −29%)** | 0.99125 → 0.99379 (−29%) | 25분 → 32분 |
| tabicl@rand200000 | – | 0.98189 (534위) | 0.97019 (593위, +65%) | 0.98082 → 0.96939 (+60%) | 32분 → 140분 |
| catboost | 0.91743 (751위) | 0.97223 (581위) | 0.98638 (504위, −51%) | 0.97218 → 0.98645 (−51%) | 13분 → 30분 |
| lgbm | 0.90474 (794위) | 0.96585 (601위) | 0.98367 (529위, −52%) | 0.96553 → 0.98378 (−53%) | 5분 → 17분 |
| (참고) Codex × GPT 1시간 | 0.99734 (240위) | | | | |

- **GBDT 가 더 좋아짐**: 오차 −51~53% 대 kumo −29%, tabicl 은 나빠짐 → C1 지표 음수. CV 와 test 가 같은 방향이고 차이가 `cv_std`
  (0.0001~0.0005) 보다 훨씬 큼
- 순위는 그대로 kumo 가 위 (0.99453 대 catboost 0.98638) 지만 앞서는 폭이 줄어듦: test 오차 kumo 대 catboost 0.0077 대 0.0278 (F1, 3.6배) →
  0.0055 대 0.0136 (F2, 2.5배)
- tabicl 은 244열에서 나빠지고 4배 느림 (이유 확인 안 함. GMM 확률과 그 log 처럼 서로 겹치는 열이 많음)
- 에이전트 피처 + FM 한 번 (0.99453) 도 Codex 1시간 (0.99734) 에 못 미침 → tps 에서 에이전트 이득은 피처만이 아님
  (NN 10개 섞기 · meta 보정 = 모델 쪽, F2 로 옮기지 않음)

#### 해석 (nomad 포함 세 대회)

- **"FM 에 특히 도움" (C1 지표 양수) 은 여전히 nomad × kumo × F2 하나**. FM 이 GBDT 를 크게 앞서는 leaf·tps 에서는 에이전트 피처가
  GBDT 를 더 많이 도움 (leaf CV, tps CV·test) → FM 이 이미 앞서는 곳에서는 GBDT 쪽 격차를 줄이는 방향
- 그래도 **"에이전트 피처 + FM 한 번" 은 세 대회 모두 FM·GBDT 최고** (nomad kumo~f2, leaf tabicl~f2, tps kumo~f2)
- 에이전트 1시간과 비교: nomad 는 조금 넘고 (kumo~f2 0.04523 < Codex 2회째 0.04567), leaf 는 피처를 만든 에이전트의 두 실행 사이
  (AIDE × Sonnet 0.00385 · 0.00195 대 tabicl~f2 0.00232), tps 는 못 미침 (0.99453 < Codex 0.99734). C1 은 1회씩이고 차이가 작거나 (nomad 0.0004)
  test 가 작아 (leaf 99행) 확정은 아님 → "에이전트 1시간과 같은 수준까지는 감, tps 처럼 모델 쪽 이득이 큰 대회는 못 따라감"

## 에이전트 실행 환경 (트랙 B)

2026-11-09 에 정함. 구현은 `fab/agent/__init__.py` (로컬 런처) + `fab/agent/runner.py` (원격 실행기), 회수·채점은 정리 작업 `fab/remote/reconcile.py`.

### 기본 설정

| 항목 | 정한 것 |
|---|---|
| 위치 | Databricks 서버리스 GPU 작업, **A10 1장** (MLE-bench 공식 설정과 같은 GPU) |
| LLM | 워크스페이스 FM API: `databricks-claude-sonnet-5-5`, `databricks-gpt-6-1-terra` |
| 하네스 × LLM | AIDE × {Sonnet, GPT} (Chat Completions), Claude Code × Sonnet (Anthropic Messages), Codex CLI × GPT (Responses) |
| 하네스 버전 | AIDE `thesofakillers/aideml@v6.3.3` (MLE-bench 가 쓰는 포크), Claude Code 2.1.319, Codex 0.167.0 |
| 패키지 | 일반 ML 패키지만 (MLE-bench 공식 이미지 수준). TFM 은 미리 안 깖 → 에이전트가 직접 설치해 썼는지 기록. C2 (`+fm`) 만 TFM 3개를 깖 |
| 인터넷 | 허용. 기록(대화·명령)을 남기고 사후 검사 (kaggle.com 풀이·토론 접근, 답 찾기) |
| 신원·권한 | 전용 서비스 프린시펄(SP). 권한은 FM 엔드포인트 2개 조회 + 전용 scratch Volume 읽기·쓰기만 |
| 지시문 | MLE-bench `environment/instructions.txt` 를 경로만 바꿔 씀 (아래 "지시문") |
| 형식 검사 | 검증 서버 대신 `localhost:5000/validate`: sample_submission 과 열·행 수·id·결측만 비교, 메시지 영어 |
| 시간 | 벽시계 1시간 (LLM 대기 + 학습 포함), 넘으면 강제 종료. 그때 `submission.csv` 가 없으면 실패 |
| 회수 | 제출 파일, 작업 디렉터리의 코드 전부, 대화 기록, 토큰 사용량 |
| 범위 | 큰 대회 (taxi·tps-dec-2021 등) 는 노드 메모리 확인 전까지 뺌. tps-may-2022 는 P3 에서 확인 후 돌림 |

### 하네스 공통 규칙

세 하네스에 똑같이 적용하고, 하네스마다 조건을 따로 맞추지 않음 (kaggle-master#35·#37).

| 규칙 | 내용 | 기록 (`results/mlebench_runs.csv`) |
|---|---|---|
| 시간 예산 | 벽시계 1시간 하나. 설치 후 하네스 시작부터 셈 (LLM 대기·429 대기·학습 포함). 늘리지 않음 | `harness_seconds` |
| 스스로 끝냄 | 시간이 남아도 이어 가게 하지 않음 (`--resume`·"계속하라" 재지시 없음). 하네스 그대로의 동작으로 봄. 이어 가는 건 별도 설정 `+cont` 뿐이고 섞지 않음 | `timed_out` (False = 스스로 끝냄), `continues` |
| 429 대기 | 프록시가 기다렸다 다시 보내고, 기다린 시간도 1시간에 들어감. 대기만큼 예산을 늘리지 않음 (AIDE 는 자기 `time_limit` 을 벽시계로 세서, 실행기가 시간을 늘려도 AIDE 에는 적용이 안 됨 → 늘리면 하네스마다 조건이 달라짐) | `retries_429`·`waited_429` |
| 동시 실행 | 같은 FM 엔드포인트를 쓰는 실행은 한 번에 하나 (분당 토큰 한도를 나눠 쓰지 않게). 예외: C2 의 aide-sonnet+fm 다시 돌리기 3개는 동시에 (부하 없음 확인, "동시 실행 부하 확인") | `retries_429` |

결과를 읽을 때

- 점수 옆에 하네스 시간과 종료 방식 (스스로 끝냄 / 시간 종료) 을 같이 적음. Claude Code 는 "1시간 예산" 에서 실제로는 20초 안팎만 씀
  → 하네스 비교는 "같은 예산을 받았을 때 하네스가 어떻게 쓰는가" 까지 포함한 결과로 읽음
- 429 대기도 같이 적음. Codex 는 호출마다 컨텍스트 전체 (약 10만 토큰) 를 보내서 대기가 가장 김 (P0: 4~7분) → 그만큼 불리
- 한도 상향은 워크스페이스 쪽 일 (kaggle-master#37). 한도가 바뀌면 그 전후 실행을 따로 표시

### 일찍 끝날 때 (+cont 규칙)

2026-11-10 결정 (kaggle-master#35).

- **공식 설정은 그대로**: 이어 가지 않음. P0 의 Claude Code (18~20초 · 4턴) 는 이 설정의 결과
- **별도 설정 `<config>+cont`** (`fab mleb agent run ... --continue`)
  - 하네스가 정상 종료(exit 0) 했는데 2분 이상 남았으면 같은 세션을 이어 감 (Claude Code `-p --resume <session_id>`, Codex `exec resume <thread_id>`)
  - 보내는 글은 세 하네스 같음: "시간 한도에 아직 안 닿았고 N분 M초 남았다, 남은 시간으로 점수를 올려라, 유효한 제출을 늘 저장해 두라"
    (`fab/agent/runner.py` 의 `CONTINUE_MESSAGE`)
  - 멈추는 조건: 도구를 하나도 안 쓰고 끝난 구간이 연속 3번, 또는 전체 40번 (2026-11-11. P0 은 전체 20번만, 근거는 [+cont 결과](#cont-일찍-끝나는-claude-code-이어-가기-2026-11-10))
  - 벽시계 1시간은 처음부터 셈 (이어 간 시간도 포함)
- 하네스별
  - AIDE: 해당 없음. 자기 `time_limit` 까지 돌아서 일찍 끝나지 않고, 이어 갈 세션도 없음 → `--continue` 를 거부
  - Codex: 58~59분을 써서 규칙이 거의 안 걸림. 실제로 걸리는 건 Claude Code
  - 세션 id 를 출력에서 못 찾으면: Claude Code 는 `--continue` (cwd 의 최근 대화), Codex 는 이어 가지 않음
    (`--last` 와 글을 같이 주면 글이 SESSION_ID 자리로 읽힐 수 있음). Codex 의 resume 은 원격에서 아직 안 돌아 봄 (로컬에서 인자 해석·JSON 출력만 확인)
- 기록
  - `continues` (이어 간 횟수), `first_exit_seconds` (공식 설정이었다면 끝났을 시각)
  - meta.json 의 `segments`: 구간마다 시작·길이·종료 코드·`tools` (도구 호출 수, AIDE 는 null)
  - meta.json 의 `continue_stop`: 멈춘 까닭 (`idle`·`max`·`min_left`·`timed_out`·`exit N`·`no_session`)
  - 이 열이 생기기 전의 공식 설정 행은 `continues` 0, `first_exit_seconds` = `harness_seconds`
- 왜 둘 다 두나: 공식 설정만 두면 Claude Code 는 "1시간 예산" 비교에 사실상 1분짜리로 들어가고, +cont 만 두면 하네스 그대로의 동작이 안 보임.
  둘을 나란히 보면 "Claude Code 가 약한지, 일찍 끝내서인지" 가 갈림. 두 설정은 결과 표에서 섞지 않음

### 정답 격리와 권한

- 정답(private) 은 로컬 mlebench 캐시와, 채점용으로 올린 내 Volume (`databricks.local.yaml` 의 `volume` 아래 `mlebench/data/`) 에만 있음.
  에이전트 SP 의 scratch Volume 에는 public 만 올림
- 2026-11-10 채점을 원격 정리 작업 (`fab/remote/reconcile.py`, 내 신원) 으로 옮기면서 정답이 워크스페이스에 올라감
  → 격리는 SP 권한에 기댐. SP 는 정답 Volume 스키마에 `USE_SCHEMA`·`READ_VOLUME` 이 없음
  (`databricks grants get-effective` 로 확인, 그룹에서 받은 `READ_METADATA` 만). **SP 권한을 넓힐 때 이 스키마는 빼야 함**
- SP 권한: USE CATALOG·USE SCHEMA·READ/WRITE VOLUME 만. TFM 실행용 Volume 은 메타데이터만 보임 (위치는 `databricks.local.yaml` 의 `mleb_agent`)
- 작업 제출은 `jobs submit` + `run_as` + `ai_runtime_task`

### 구현

- **LLM 프록시**: FM API 시스템 엔드포인트에는 ACL 을 못 검 → 실행기 안 로컬 프록시가 모델을 설정값으로 고정하고 요청마다 토큰을 기록 (`proxy.jsonl`)
  - SP 토큰은 요청마다 SDK 에서 받아 1시간을 넘겨도 만료 안 됨
  - AIDE 는 모델 이름이 `gpt-` 로 시작해야 OpenAI 백엔드를 써서 별칭 (`gpt-km-*`) 을 씀
- **지시문**: 세 하네스가 같은 글을 받음
  - MLE-bench `environment/instructions.txt` 그대로
  - \+ `fab/agent/additional_notes.txt`: AIDE 의 추가 안내에서 단계 수 줄을 빼고 "시간이 되면 종료되니 유효한 제출을 늘 저장" 한 문장을 더함
  - \+ 대회 description
- **사후 검사** (정리 작업, 예전 `fetch`): 코드·기록의 TFM 이름, `pip` 로 새로 설치한 패키지, Kaggle 풀이·토론 URL 수, 토큰 합계, 다른 모델 요청 여부
- **노드 사양** (kaggle-master#32): 실행기가 시작할 때 `meta.json` 의 `node` 에 적고, 정리 작업이 `node_cpus`·`node_mem_gb`·`node_tmp_free_gb`·`node_gpu` 열로 옮김
  - CPU 수 (`os.cpu_count`·affinity·cgroup 한도), 메모리 (물리·`/proc/meminfo`·cgroup 한도), `/tmp`·HOME 디스크 여유, GPU (이름·메모리)
  - CPU·메모리는 호스트·affinity·cgroup 한도 중 가장 작은 값. 항목마다 따로 재서 하나가 실패해도 나머지는 남음, `nvidia-smi` 는 30초 제한
  - 큰 대회 (taxi·tps-dec-2021·tps-may-2022) 판단에 씀
- **진행 중 제출 사본** (2026-11-13, `sync_progress`): 2분마다 진행 기록 (`progress/`) 을 올릴 때 같이 올림
  - `meta.json`: 회차마다 그때까지의 meta (노드 사양·사전 점검·구간 수 `segments_started`·`harness_seconds`)
  - `submission.csv` + `submission.json`: 제출이 바뀌면 (읽는 사이 바뀌었으면 다음 회차에). json (경로·mtime·크기·sha256·형식 검사) 은
    csv 를 다 쓴 뒤에 쓰고, 새 csv 를 쓰기 전에 지움 → sha256 이 맞는 쌍만 씀 (쓰다 죽은 csv 는 안 씀)
  - `workspace.tgz`: 10분마다 (`WORKSPACE_EVERY`, gzip 이 에이전트와 CPU 를 나눠 써서 회차마다는 안 함). 실패해도 제출 사본과 무관
  - `pip_before.txt`·`full_instructions.txt`: 한 번
  - 컨테이너가 중간에 죽어 마지막 회수 (`meta.json`) 가 없는 실행은 정리 작업이 이 사본으로 회수·채점하고 `run_crashed` 열에 True
    (`recover_from_progress`). 시간 초과 실행이 그때까지 저장한 제출로 채점되는 것과 같은 규칙. 죽을 때 돌던 구간까지 이어 가기로 셈,
    `timed_out` 은 빈칸, `pip_added` 는 빈칸 (`pip_after.txt` 없음). 잘린 meta·tgz 는 안 씀
  - 못 건지는 것: 마지막 2분 안에 바뀐 제출, 마지막 10분 안의 코드. 이 열 이전 실행은 False (실행기가 죽은 실행은 exclude 로 뺐음)
- **로컬 확인 (2026-11-09)**: 프록시의 Anthropic Messages·Responses 스트리밍, Chat Completions, 모델 고정, 사용량 기록, 형식 검사.
  이 노트북의 Claude Code·Codex 는 회사 관리 설정이 공급자를 강제해서 로컬 하네스 시험은 대표성이 없음 → 원격에서 확인

### FM API 호환 처리

2026-11-09, 프록시·하네스 설정. 두 LLM 에 똑같이 적용.

| 문제 | 처리 |
|---|---|
| 샘플링 파라미터 | `temperature`·`top_p`·`top_k` 를 뺌. Sonnet 5.5 는 둘 다 거부, GPT-6.1 Terra 는 temperature 1 만 허용. AIDE 는 temperature 를 늘 보냄 (MLE-bench 의 o1-preview 설정도 기본값 1) |
| AIDE 의 시스템 메시지 | `agent.convert_system_to_user=True`. Claude 가 시스템 메시지만 있는 요청을 거부함 (MLE-bench 도 o1-preview 에서 씀) |
| AIDE 피드백 단계의 함수 호출 | Sonnet 은 강제 tool_choice·json_schema 를, GPT 는 Chat Completions 함수 도구를 거부 → 프록시가 도구를 "이 스키마의 JSON 만 답하라" 지시로 바꾸고 답을 함수 호출 응답으로 감쌈 (최대 3번). 원래 AIDE 는 네이티브 함수 호출 |
| Claude Code 의 도구 검색 | `ENABLE_TOOL_SEARCH=false` (kaggle-master#40). 켜지면 첫 요청에 `anthropic-beta: advanced-tool-use-2025-11-20` 이 붙고 FM API 가 400 으로 거부 (P0 두 실행 모두 첫 요청 1회, 0.2초 뒤 다시 보내 결과엔 영향 없었음). 프록시에서 헤더만 빼면 지연 로드 도구가 검색 없이 남을 수 있어 기능 자체를 끔 → 처음부터 모든 도구를 넘김. 프록시가 요청마다 `beta`·`n_tools` 를 `proxy.jsonl` 에 기록 |
| 429 (분당 입력 토큰 한도) | 프록시가 기다렸다 다시 보냄 (아래 "429") |
| Chat Completions content 가 파트 목록 | 2026-11-12 (#10). 긴 프롬프트에서 Sonnet 이 `choices[].message.content` 를 문자열이 아닌 `[{"type": "text", "text": ...}]` 로 답함 (+fm 안내문이 붙은 AIDE 에서 일반 응답에도 나와 코드 추출이 `TypeError` 로 죽음). 프록시가 비스트리밍 Chat Completions 응답 (200) 을 다 받아 글 파트만 이어 붙인 문자열로 바꿔 넘기고, `proxy.jsonl` 에 본 파트 type 들을 `flattened` 로 기록. 글이 아닌 파트 (`text` 가 문자열이 아닌 것 포함) 는 버리고 앞 500자를 `flattened_dropped` 로, 글 파트가 없어 빈 문자열이 되면 `flattened_empty`. trace 에는 FM API 가 준 원래 본문. 상태 줄은 다 읽고 바꾼 뒤에 보냄: FM API 연결·읽기가 실패하면 (끊김·시간 초과, 스트리밍도 보내기 전이면) 502 + `proxy.jsonl` 의 `proxy_error` → 하네스가 다시 요청. GPT 는 문자열로 답해 바뀌지 않음. 스트리밍 Chat Completions 는 쓰는 하네스가 없어 그대로 넘김 |

### 429

재시도 규칙

- 대기 1→2→4→8→15→30초 (상한 30초, 최대 15번, 합 약 330초) 에 ±25% 흔들기
- 429 에 `Retry-After` 가 있으면 그보다 일찍 보내지 않음 (60초에서 자르고 0~25% 늦춰 흔듦). 짧은 값으로 기본 간격보다 줄이지는 않음 → 합 약 330초 유지
- P0 은 5→60초 (8번, 합 315초) 였음 → 2026-11-10 에 아래 측정을 보고 줄임

측정 (2026-11-10, kaggle-master#37)

- P0 에서 429 를 받은 건 **Codex × GPT 만** (1시간 입력 4,200만~4,500만 토큰, 다른 설정은 5만~55만). AIDE × GPT 도 0번
  → 모델보다 하네스의 사용량 문제. 한도는 엔드포인트별 (오류에 `rate limit for databricks-gpt-6-1-terra`)
- Codex 의 첫 429 는 28~31분 (호출당 입력이 약 19만 토큰을 넘고 5분 평균 분당 100만~120만 토큰일 때). 처음 15분은 0번
- 429 받은 요청 대부분이 첫 재시도 (5초 뒤) 에 통과, 재시도당 평균 대기 8~11초 → 대기 간격을 줄인 근거
- 직접 부하 시험 (GPT-6.1 Terra, 요청당 14.5만 토큰 · 캐시 100%, 분당 0.5→2M 단계, 워크스페이스 2곳 동시):
  두 워크스페이스 결과가 같음 — 1.5M/분까지 429 없음, 2M/분 단계에서 429, 60초 창 최대 처리 1.89M
  → 두 워크스페이스가 같은 티어, 한도는 워크스페이스마다 따로 (동시에 둘 다 1.75M/분 처리)
- 설명 안 되는 점: Codex 는 직전 60초 입력 0.43M 에서도 429 를 받았고 2.4M 에서도 통과함, 동시 요청도 대부분 1개.
  한도가 실제 입력이 아닌 추정치 (예: 입력 + 최대 출력) 로 세거나 1분 창이 아닌 방식일 수 있음 (확인 안 함)
- 근본 해결은 GPT 엔드포인트 티어 상향. 모델은 바꾸지 않음 (다른 GPT 엔드포인트도 같은 티어일 가능성이 크고, LLM 축이 깨짐)

### 운영 기록

첫 원격 실행에서 고친 것

- Volume 의 command_path 옆에 `env_vars.json`·`secret_env_vars.json` (빈 배열) 이 있어야 런처가 시작함
- 코드 tarball 은 최상위 폴더 하나로 묶어야 함
- `run_as` 에는 SP 의 Manager 와 별도로 **Use** 역할이 필요
- 프록시가 `Accept-Encoding` 을 넘기면 gzip 에러 본문이 깨져서 뺌

제외한 실행 (`fab mleb agent exclude`, 사유는 Volume 의 `state/agent/<key>.done.json`, 2026-11-10 이전은 `runs.json`)

- 런처 시작 실패 4회, 프록시 Accept-Encoding 버그 1회, AIDE temperature 거부·빈 messages 2회, Codex 429 로 27분에 중단 1회 (점수 0.8092)
- 2026-11-13 AI Runtime 플랫폼 오류 4회 (모두 leaf × Claude Code +cont +fm 2회째, 같은 시간대의 다른 실행 10개는 정상):
  하네스 2분째 SIGTERM (exit 143) 1회, `Your compute failed because of a Databricks platform issue` 3회 (12·15·14분째, 모두 Kumo
  분류를 돌리던 중 GPU 메모리를 잡은 채 사용률 0%, 시스템 메모리 6GB 로 OOM 아님). 앞 3회는 에이전트가 저장해 둔 제출도 컨테이너와
  같이 사라져 뺌 → 위 "진행 중 제출 사본" 을 넣고 다시 돌림, 4회째는 그 사본으로 채점 ([C2 2회째](#c2-2회째-leaf--nomad-2026-11-13))

### 참고: 실행 위치를 정하기 전의 비교

결정 전 (2026-11-09) 의 비교라 지금 상태와 다른 칸이 있음 (정답 격리).

| | 로컬 Docker (MLE-bench 공식 환경) | Databricks 서버리스 GPU 작업 (고름) |
|---|---|---|
| 같은 조건 | 공식 컨테이너·지시문·검증 서버 그대로 | 지시문 (`environment/instructions.txt`)·검증 스크립트를 옮겨 씀 |
| GPU | 없음 (Mac) | A10 (공식 설정과 같은 GPU) |
| 정답 격리 | 컨테이너 마운트로 | 당시 계획: private 정답은 로컬에만 두고 원격에 안 올림. **2026-11-10 부터는 정답이 워크스페이스에 있고 SP 권한으로만 격리** (위 "정답 격리와 권한") |
| LLM 인증 | API 키를 컨테이너에 넘겨야 함 | 워크스페이스 FM API 를 쓰면 작업 안 인증으로 해결 |
| 하네스 | AIDE·OpenHands·MLAB 은 공식 `agents/` 에 있음, Claude Code·Codex 는 추가 | 셋 다 작업 안에서 설치 |

## 명령

전체 사용법은 [README](../README.md) 의 "사용". mle-bench 는 `../mle-bench` 에 따로 설치 (`fab/bench/mlebench/__init__.py` 위쪽 설명, 위치는 `FAB_MLEBENCH`).
학습은 Databricks 에서만 (로컬 `fab run` 은 막힘, 2026-11-10).

| 명령 | 하는 일 |
|---|---|
| `fab download mleb-<대회>` | Kaggle zip → MLE-bench 분할 → 체크섬 확인 → public 만 `data/` (`config.yaml` 의 `source: mlebench`) |
| `fab databricks mleb-<대회> <모델>...` | FM·GBDT 학습 (CV + test 예측) |
| `fab mleb grade <MLE-bench id> <모델>... --each` | 채점 요청. 정리 작업이 학습이 끝나기를 기다려 `mlebench grade-sample` → `results/mlebench.csv` |
| `fab mleb agents` | 공개 리더보드 에이전트 점수 (표 형식 대회만) → `results/mlebench_agents.csv` |
| `fab mleb agent run <MLE-bench id> <설정>... [--continue] [--fm]` | Agent 실행 제출 → 회수·채점 → `results/mlebench_runs.csv` |
| `fab mleb agent peek` / `list` | 실행 중 진행 (LLM 호출·429) / 처리 상태 |
| `fab mleb agent recheck` | 사후 검사 열 다시 계산 (검사 규칙을 고친 뒤) |
| `fab mleb agent exclude` | 실행을 결과에서 뺌 (실행 환경 문제 등, 사유를 남김) |
| `fab remote kick` → `fab sync` | 결과 합치기·채점 → `results/*.csv` 로 받기 |

- 타깃이 둘인 nomad2018 은 fab 대회 2개 (`mleb-nomad2018-formation`, `-bandgap`) → `fab mleb grade` 가 열을 합쳐 한 제출로 채점
- 피처 변형은 모델 이름 뒤 `~<variant>` (예: `catboost~f2`) → `competitions/<slug>/features/<variant>.py`, 결과에 `features`·`n_features` 기록

예: P0 (spaceship)

```bash
uv run --frozen fab download mleb-spaceship-titanic
uv run --frozen fab databricks mleb-spaceship-titanic catboost tabpfn_v2
uv run --frozen fab mleb grade spaceship-titanic catboost tabpfn_v2 --each
uv run --frozen fab remote kick     # 학습이 끝난 뒤
uv run --frozen fab sync
```

예: P3 (FM 은 H100 1장 + serverless CPU)

```bash
uv run --frozen fab databricks mleb-leaf-classification lgbm xgb catboost tabicl kumo
uv run --frozen fab mleb grade leaf-classification lgbm xgb catboost tabicl kumo --each
T=(lgbm catboost tabpfn_v2 tabicl kumo lgbm~f1 xgb~f1 catboost~f1 tabpfn_v2~f1 tabicl~f1 kumo~f1 kumo@rand200000~f1 tabicl@rand200000~f1)
uv run --frozen fab databricks mleb-tps-may-2022 "${T[@]}"
uv run --frozen fab mleb grade tabular-playground-series-may-2022 "${T[@]}" --each
```

예: Agent 실행 순서 (같은 엔드포인트끼리는 한 번에 하나 → Sonnet 줄과 GPT 줄을 하나씩 동시에)

```bash
# P3: 대회마다
uv run --frozen fab mleb agent run leaf-classification aide-sonnet      # → claude-code-sonnet → claude-code-sonnet --continue
uv run --frozen fab mleb agent run leaf-classification codex-gpt        # → aide-gpt

# C2: GPT 줄 / Sonnet 줄
uv run --frozen fab mleb agent run nomad2018-predict-transparent-conductors codex-gpt --fm   # → aide-gpt --fm
uv run --frozen fab mleb agent run nomad2018-predict-transparent-conductors claude-code-sonnet --continue --fm  # → aide-sonnet --fm → claude-code-sonnet --fm
```
