# 로드맵: FM vs Agent vs FM + Agent

2026-11-10 방향 전환 (kaggle-master 에서 분리). 질문은 [README](../../README.md) 첫머리.

## 비교 단위

같은 문제·같은 채점 기준에서 방식만 바꿈.

| 방식 | 정의 | 기록할 것 |
|---|---|---|
| FM | 원본 열(F0) 또는 사람 피처(F1)로 TFM 한 번 추론, 튜닝 없음. GBDT 는 같은 피처의 기준선 | 점수, 시간, GPU |
| Agent | 하네스 × LLM, 벽시계 1시간 · A10 1장, 환경에 TFM 없음 | 점수, 시간, 토큰, 429 대기, 쓴 모델, 사후 검사 |
| C1 (Agent → FM) | 에이전트 최종 풀이에서 전처리만 떼어(F2) FM·GBDT 에 넣음 | FM 의 F0 대비 향상 − GBDT 의 F0 대비 향상 |
| C2 (FM 을 도구로 받은 Agent) | 환경에 TFM 을 깔고 지시문에 알림, 나머지는 Agent 와 같음 | Agent 와 같음 + FM 을 실제로 썼는지 (`tfm_in_code`) |

순서: **C1 → C2**. C1 은 이미 돌린 에이전트 풀이로 할 수 있고, C2 의 결과를 읽으려면 "에이전트 피처 + FM" 이 얼마나 되는지
(C1) 를 먼저 알아야 함 (C2 의 에이전트가 FM 을 써서 좋아졌는지, 피처 덕인지 가르려면).

## 벤치마크

| | MLE-bench | TabArena |
|---|---|---|
| 상태 | 연동됨 (P0) | 연동 전 |
| 채점 | 다시 나눈 test (정답은 로컬 mlebench 캐시 + 채점용 내 Volume, 에이전트 SP 는 못 읽음. 채점은 serverless CPU 작업) | 확인 필요 |
| 비교 대상 | 공개 리더보드 에이전트 (24시간, 실행별 점수만) | 확인 필요 (튜닝·앙상블한 모델 리더보드로 알고 있음) |
| 범위 | 표 형식 대회 7개 (`fab/bench/mlebench/__init__.py` 의 `TABULAR`) | 확인 필요 |

TabArena 연동 전에 확인할 것

- 데이터셋 목록·크기, 분할 방식(반복 CV 의 fold 수·반복 수), 지표 → `fab run` 의 고정 fold 와 맞출 수 있는지
- 공개 리더보드의 기준선(모델·설정)을 그대로 비교 대상으로 쓸 수 있는지, FM 이 이미 리더보드에 있는지
- 에이전트에게 줄 형식: MLE-bench 처럼 train/test csv + 설명(description.md) + sample_submission 을 만들어야 함.
  정답(test fold 의 타깃)은 로컬에만 두고 Volume 에는 올리지 않음 (MLE-bench 와 같은 격리)
- 비용: 데이터셋 × 설정 × 시드 × 1시간 → 데이터셋을 고를지 (먼저 MLE-bench 처럼 몇 개로 파일럿)

## 단계

| 단계 | 내용 | 완료 기준 |
|---|---|---|
| P0 ✓ | MLE-bench spaceship·nomad: FM 채점, Agent 4설정 × 1회 (+cont 2회) | [docs/mle-bench.md](../mle-bench.md) |
| P1 | 신뢰도: Agent 2회째, FM 시드 반복 (같은 기준으로 비교) | 대회별 설정마다 시드 2, FM 시드 분산 |
| P2 | **C1**: nomad × Codex 풀이에서 피처 추출 → F0/F2s/F2 × {tabpfn_v2, tabicl, kumo} + {lgbm, catboost} | 향상폭 차이 표, 추출 피처 값이 에이전트 배열과 같음 |
| P3 | MLE-bench 표 형식 대회 확장 (tps-may-2022, leaf-classification 먼저, 큰 대회는 노드 사양 확인 후) | 대회별 FM × Agent 표, 메달 |
| P4 | TabArena 연동 (위 확인 → `fab/bench/tabarena/`) | FM 채점이 TabArena 기준과 맞음, Agent 파일럿 1~2개 |
| P5 | **C2**: Agent 설정에 FM 추가 (예: `codex-gpt+fm`) | Agent 와 C2 를 같은 대회·시드로 |
| P6 | 정리 | 방식 × 벤치마크 결과 문서 |

kaggle-master 에 열려 있던 이슈 (그쪽에 남김, 진행할 때 이 저장소에 새로 엶):
P1 = kaggle-master#34 (Agent 2회째)·#38 (FM 시드), P2 = #36 (C1), P3 = #39 (트랙 A 확장).

## C1 규칙 (kaggle-master 계획에서 가져옴)

피처 세트: F0 = 원본 열, F1 = 사람 피처, F2 = 에이전트 피처 (실행 이름 `~f2s`·`~f2` 처럼 붙임, 4 참고)

1. 추출: 에이전트 최종 풀이에서 전처리만 떼어 `build(train, test) -> (X, X_test, cat_cols)` 로 (LLM 변환 + 사람 검토)
2. 동일성 확인 (2026-11-11 바꿈): 옮긴 피처 값이 에이전트 작업 디렉터리의 피처 배열과 같은지 (값이 남아 있으면 값, 없으면
   에이전트 기록의 모양). 그리고 해석의 최소 조건으로 GBDT 의 F2 가 F0 보다 나은지.
   원래 규칙 ("추출 피처로 에이전트 모델을 다시 돌려 에이전트 점수가 나오는지") 은 nomad × Codex 에 못 씀: 점수 대부분이
   공간군별 CNN·GNN·커널 모델에서 나오고 피처를 받는 표형 모델은 sg 227 의 25% 뿐 → 피처를 옮겨도 그 점수가 나올 이유가 없음.
   해석: C1 은 에이전트 이득 중 피처 몫만 잼 (모델 몫은 아님)
3. 누수: 타깃을 쓰는 피처(target encoding, OOF 스태킹)는 지금 형식(train+test 전체 계산)에선 CV 누수 →
   1차는 타깃 비의존 피처만, 2차에 fold 별 fit 형식(`build(train, test, y)`)
4. 피처 변형: 모델 이름 뒤 `~<variant>` (예: `fab databricks <대회> catboost tabpfn_v2~f2`) →
   `competitions/<slug>/features/<variant>.py`, 결과에 `features`·`n_features` 기록 (README "모델").
   `--features` 인자 대신 이름에 붙인 까닭: 실행 이름이 제출·상태·결과 폴더·채점까지 그대로 다녀서 F0 와 F2 를 한 번에 내고
   같은 표에서 나란히 봄. `tfm_max_features` 를 넘는지는 `n_features` 와 대회 설정으로 봄
5. 점수: 같은 fold 의 CV + 벤치마크 test 둘 다
6. TFM 피처 상한: 변형 파일의 `TFM_MAX_FEATURES` (없으면 대회 설정 `tfm_max_features`). 상한을 넘으면 fold 학습 데이터의
   LightGBM 중요도 상위만 TFM 에 감 (누수 없음) → TFM 이 GBDT 가 고른 열을 받으니 "TFM 에 특히 도움" 에 GBDT 몫이 섞일 수 있음.
   그래서 작은 변형은 상한 없이 다 넣고, 큰 변형만 상한

### nomad × Codex 의 F2 (2026-11-11)

에이전트: `nomad2018-...__codex-gpt__r1__20261109-200654` (test 0.04847, CV 0.0443). 코드는
`competitions/mleb-nomad2018-formation/features/_geo.py` (bandgap 은 같은 파일을 씀).

| 변형 | 블록 | 열 | TFM 상한 |
|---|---|---|---|
| `~f2s` | 원본 12 + 격자 파생 10 + 대칭 16 + RDF·이웃 통계 1,072 (사람이 읽을 수 있는 물리 피처) | 1,109 | 없음 |
| `~f2` | `~f2s` + SOAP 평균·표준편차 12,000 + 위상 14,071 + 푸리에 8,580 + 스펙트럼 3,072 (타깃 비의존 전부) | 38,832 | 500 |

- 뺀 것: 정답 섞기 (train 과 같은 구조인 test 행에 train 정답을 90~95% 섞음, `finalize_submission.py:19-28`),
  fold 마다 타깃 상관으로 열 고르기, 복셀·그래프·원자별 SOAP (CNN·GNN·커널 모델의 입력), 매개변수만 다른 두 번째 SOAP
- 동일성 확인 (로컬, 같은 MLE-bench 분할 2,160 + 240행): 블록 모양이 에이전트 기록과 모두 같음, 대칭 블록은 에이전트 배열과
  값이 같음 (최대 차이 1.2e-7, float32). 나머지 큰 배열은 20MB 를 넘어 작업 디렉터리 회수에서 빠져 값 비교는 못 함
  (계산식·상수·순서는 그대로 옮김). 만드는 시간: 로컬 8코어 약 20초
- 해석 주의: test 에 train 과 같은 구조가 있음. SOAP 같은 기술자를 받은 TFM 은 비슷한 행을 보고 예측하므로 정답 섞기가 노린
  효과를 피처로 정당하게 얻을 수 있음 → `~f2` 의 TFM 향상이 특히 클 수 있음
- 에이전트 CV (참고, 손으로 정한 섞기 비율이라 낙관적): formation 0.0179, bandgap 0.0707. F0 의 CV: kumo 0.0316 / 0.0841,
  catboost 0.0353 / 0.0912

## 정해야 할 것

- +cont 이어 가기 상한 (전체 20번 vs 연속 거절 N번): P1 의 Agent 2회째 전에 ([docs/mle-bench.md](../mle-bench.md) 의 +cont 결과)
- C2 의 FM 목록과 안내 문구: 어떤 TFM 을 깔지 (라이선스·설치 시간), 지시문에 얼마나 알릴지 (이름만 / 사용 예까지)
- 해석 주의: Agent 는 LLM 이 공개 Kaggle 풀이를 학습했을 수 있음 (FM 은 아님). TabArena 데이터셋도 같은 문제가 있는지
