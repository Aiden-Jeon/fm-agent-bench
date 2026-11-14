# 로드맵: FM vs Agent vs FM + Agent

2026-11-10 방향 전환 (kaggle-master 에서 분리). 질문은 [README](../../README.md) 첫머리.

## 비교 단위

같은 문제·같은 채점 기준에서 방식만 바꿈.

| 방식 | 정의 | 기록할 것 |
|---|---|---|
| FM | 원본 열(F0) 또는 사람 피처(F1)로 TFM 한 번 추론, 튜닝 없음. GBDT 는 같은 피처의 기준선 | 점수, 시간, GPU |
| Agent | 하네스 × LLM, 벽시계 1시간 · A10 1장, 환경에 TFM 없음 | 점수, 시간, 토큰, 429 대기, 쓴 모델, 사후 검사 |
| C1 (Agent → FM) | 에이전트 최종 풀이에서 전처리만 떼어(F2) FM·GBDT 에 넣음 | FM 의 F0 대비 향상 − GBDT 의 F0 대비 향상 |
| C2 (FM 을 도구로 받은 Agent) | 환경에 TFM 을 깔고 지시문에 알림, 나머지는 Agent 와 같음 | Agent 와 같음 + FM 을 실제로 썼는지 (`tfm_runtime`·`tfm_at_submit`, 아래 "C2 규칙") |

순서: **C1 → C2**. C1 은 이미 돌린 에이전트 풀이로 할 수 있고, C2 의 결과를 읽으려면 "에이전트 피처 + FM" 이 얼마나 되는지
(C1) 를 먼저 알아야 함 (C2 의 에이전트가 FM 을 써서 좋아졌는지, 피처 덕인지 가르려면).

## 벤치마크

| | MLE-bench | TabArena |
|---|---|---|
| 상태 | 연동됨 (P0) | 연동됨 (P4 파일럿, 2026-11-14, `fab/bench/tabarena/`, [docs/tabarena.md](../tabarena.md)) |
| 채점 | 다시 나눈 test (정답은 로컬 mlebench 캐시 + 채점용 내 Volume, 에이전트 SP 는 못 읽음. 채점은 serverless CPU 작업) | OpenML task 의 고정 외부 분할 (3-fold 층화 × 반복 10번 (2,500 행 미만) 또는 3번 → 30·9 split, 합 816). 지표 이진 ROC AUC · 다중 log loss · 회귀 RMSE. 합치기는 Elo (기본 RandomForest = 1000) · 개선 여지 · 순위. 데이터가 타깃까지 공개라 정답 격리는 우리 쪽에서 (+ 사후 검사) |
| 비교 대상 | 공개 리더보드 에이전트 (24시간, 실행별 점수만) | 모델 58개 (기본 / 튜닝 200개 / 튜닝 + 앙상블, 8-fold bag) + 시스템 8개 (AutoGluon 4시간 등, LLM 에이전트 시스템은 아직 없음). TFM 이미 있음 (TabPFNv2·TabICLv2·Kumo-Tabular 등, 기본 설정, test 는 train 전체로 refit). split 별 결과는 공개 parquet 로 받음 |
| 범위 | 표 형식 대회 7개 (`fab/bench/mlebench/__init__.py` 의 `TABULAR`) | TabArena-v0.1 51개 (이진 30 · 다중 8 · 회귀 13), 748~150,000 행, 피처 5~1,777, 클래스 최대 8. 우리 FM 은 tabpfn_v2 33개 (공식 범위), tabicl·kumo 51개 |

TabArena 연동 전에 확인할 것 (2026-11-13 조사, 답은 [docs/tabarena.md](../tabarena.md))

- (답함) 데이터셋 목록·크기, 분할 방식(반복 CV 의 fold 수·반복 수), 지표 → `fab run` 의 고정 fold 와 맞출 수 있는지:
  외부 split 하나를 대회 하나로, `fab run` 의 5-fold 는 그 train 안의 내부 CV 로 둠 ("2. 분할·평가 프로토콜")
- (답함) 공개 리더보드의 기준선(모델·설정)을 그대로 비교 대상으로 쓸 수 있는지, FM 이 이미 리더보드에 있는지:
  있음, split 별 결과를 받아 나란히 놓음. 전처리·앙상블 수가 달라 똑같이 나오진 않음 ("3. 리더보드·기준선")
- (답함) 에이전트에게 줄 형식: MLE-bench 처럼 train/test csv + 설명(description.md) + sample_submission 을 만들어야 함.
  정답(test fold 의 타깃)은 로컬에만 두고 Volume 에는 올리지 않음 (MLE-bench 와 같은 격리).
  설명은 OpenML·원래 출처에서 우리가 씀. 데이터가 공개라 인터넷을 열면 사후 검사로만 막음.
  데이터셋 이름·출처는 안 알림 ("모델링 실력" 비교, 2026-11-13 정함, "5" 의 "익명화")
- (답함) 비용: 데이터셋 × 설정 × 시드 × 1시간 → 데이터셋을 고를지 (먼저 MLE-bench 처럼 몇 개로 파일럿):
  Agent 는 Lite split (r0f0) 만, 파일럿 4개 (hazelnut · in_vehicle_coupon · Amazon_employee_access · physiochemical_protein), Agent 는 앞 둘부터 ("7")

## 단계

| 단계 | 내용 | 완료 기준 |
|---|---|---|
| P0 ✓ | MLE-bench spaceship·nomad: FM 채점, Agent 4설정 × 1회 (+cont 2회) | [docs/mle-bench.md](../mle-bench.md) |
| P1 ✓ | 신뢰도: Agent 2회째 (FM 시드 반복·부트스트랩은 안 함, 아래 "P1 범위") | [docs/mle-bench.md](../mle-bench.md) 의 "P1" |
| P2 ✓ | **C1**: nomad × Codex 풀이에서 피처 추출 → F0/F2s/F2 × {tabpfn_v2, tabicl, kumo} + {lgbm, catboost}. 확장 (#12, 2026-11-13): leaf × AIDE × Sonnet · tps-may × Codex 의 `~f2` × {tabicl, kumo} + {lgbm, catboost} | [docs/mle-bench.md](../mle-bench.md) 의 "C1"·"C1 확장" |
| P3 ✓ | MLE-bench 표 형식 대회 확장: tps-may-2022, leaf-classification (FM × Agent 1회씩, 2026-11-12). 남은 대회 (s3e18·dec-2021·taxi) 는 나중에 | [docs/mle-bench.md](../mle-bench.md) 의 "P3" |
| P4 | TabArena 연동 (`fab/bench/tabarena/`, 2026-11-14). FM 맞춤 (2026-11-15, 51개 모두 refit = TabArena TFM 과 같은 test 예측 방식): tabicl 통과 (표준편차 안 98.0%, 비김을 뺀 승패 모두 같음), kumo 는 표준편차 안 94.1% 이나 kddcup09 하나에서 승패가 달라 못 넘음 (범주 처리 차이로 봄). 5-fold (39개, 비교용) 의 기울기는 refit 에서 사라짐. tabpfn_v2 는 5-fold 33개에서 통과. Agent 파일럿: hazelnut · in_vehicle r0f0 × 5설정 × 1회 ([docs/tabarena.md](../tabarena.md) 의 "파일럿 결과"). 남음: Agent 2회째 (#17). 끝: 큰 12개 (#16), 사후 검사·제출·채점 손보기 (#18~#20), 39개 refit | FM 채점이 TabArena 기준과 맞음, Agent 파일럿 1~2개 |
| P5 ✓ | **C2**: Agent 설정에 FM 추가 (`--fm` → `codex-gpt+fm`, 아래 "C2 규칙"). nomad 1회째 (Codex·Claude Code +cont, 2026-11-12, [docs/mle-bench.md](../mle-bench.md) 의 "C2"). 마무리 (leaf·tps-may 4설정, nomad AIDE 2설정, 2026-11-12, "C2 마무리"). aide-sonnet+fm 3개는 #10 (프록시) 을 고친 뒤 다시 돌림 (2026-11-12, 동시 실행). 2회째: leaf 4설정 (+ Agent 2회째)·nomad 3설정 (#13, 2026-11-13, "C2 2회째") | Agent 와 C2 를 같은 대회·시드로 |
| P6 | 정리 | 방식 × 벤치마크 결과 문서 |

kaggle-master 에 열려 있던 이슈 (그쪽에 남김, 진행할 때 이 저장소에 새로 엶):
P1 = kaggle-master#34 (Agent 2회째)·#38 (FM 시드), P2 = #36 (C1), P3 = #39 (트랙 A 확장).

## P1 범위 (2026-11-11)

- FM 시드 반복은 안 함: TFM 의 시드는 앙상블 전처리 순서 정도만 바꿔 test 크기에서 오는 흔들림보다 작을 것으로 봄 (재 보지는 않음).
  C1 (F0 대 F2) 은 같은 모델·같은 fold 에서 피처만 바꾼 짝지은 비교라 fold 별 일관성과 `cv_std` 로 봄
- test 점수 부트스트랩도 안 함: 지금 결론은 간단한 기준으로 갈림. spaceship 은 정확도 표준오차 √(0.82×0.18/870) ≈ 0.013 이
  방식 사이 차이(P0 0.006~0.013)와 비슷 → "뚜렷한 차이 없음" (P1 에서 밖인 건 16초 만에 끝낸 Claude Code 2회째 0.018). nomad 는 Codex 0.048 대 FM 0.057 (원래 리더보드 1위 대 17위)로 큼.
  근소한 차이로 승패를 말해야 할 때만 그 비교에 붙임
- 반복이 필요한 건 Agent: 1회 결과가 극단적으로 갈림 (nomad 에서 AIDE × GPT 제출 없음, Codex × GPT 1위)
- 실행: 같은 LLM 엔드포인트끼리는 동시에 안 돌림 (분당 토큰 한도를 나눠 씀) → Sonnet 줄 (aide-sonnet, claude-code-sonnet,
  claude-code-sonnet+cont) 과 GPT 줄 (aide-gpt, codex-gpt) 을 하나씩 동시에, 두 대회. +cont 는 새 상한 (연속 무작업 3번)

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

### leaf × AIDE × Sonnet 의 F2 (2026-11-13, #12)

에이전트: `leaf-classification__aide-sonnet__r1__20261112-083418` (test 0.00385, 에이전트 CV 0.01669). 코드는
`competitions/mleb-leaf-classification/features/f2.py`. Codex (test 0.00000) 는 안 씀: 점수가 분할 구조로 test 행을 종별
자리에 배정한 데서 나옴 (P3) → 피처로 옮길 몫이 아님.

| 변형 | 블록 | 열 | TFM 상한 |
|---|---|---|---|
| `~f2` | 원본 192 + 이미지 9 (흑백 128 미만 = 배경 화소의 비율, 이미지 종횡비·log 넓이, 배경 화소 모멘트의 이심률·log 분산 2·상관, log 경계 길이, 경계 상자 안 비율) | 201 | 없음 |

- 최종 풀이 (`best_solution.py`, 노드 34920bd2) 는 원본 열 + `imgfeat` 9열에 LR · PCA-LR · MLP 를 로그 공간에서 섞은 것.
  피처 쪽은 9열이 전부 → `~f2s` 를 따로 두지 않음 (모두 사람이 읽을 수 있음). 이미지 거리·RDA 같은 건 최종 풀이에 없음
- 뺀 것: 모델 쪽 (StandardScaler·PCA whiten 은 fold 안 fit, 섞기 가중치·온도는 OOF 로 고름). 타깃 피처·배정은 없음
- 이미지는 검은 배경에 흰 잎 → 에이전트의 `< 128` (잎이라고 여김) 은 실제로는 배경. 값을 맞추려고 그대로 둠:
  `img_area` = 1 − 잎 비율, 모멘트·이심률·상관은 배경 화소의 것. 배경의 경계 상자는 늘 이미지 전체라 `img_extent` = `img_area`
  (중복 열). 이미지는 잎 경계 상자 + 여백 약 30px 이라 종횡비·넓이 ≈ 잎 크기 (원본 192열은 크기 정보가 없음)
- 동일성 확인 (로컬): 옮긴 피처로 에이전트 LR 성분 (같은 fold `random_state=0`, StandardScaler + LR(C=10)) 을 다시 돌리면 OOF
  logloss 0.052495360956868, 에이전트 기록 0.05249536095684915 (차이 1.8e-14). 피처 배열은 작업 디렉터리에 안 남음. 만드는 시간 약 4초
- 해석 주의: 더한 게 9열뿐이라 F2 효과가 작을 수 있음 (에이전트 점수 대부분은 모델 쪽). GBDT 는 F0 에서 거의 못 배움 (클래스당 약 7행)
  → "FM 향상 − GBDT 향상" 을 % 로 보면 GBDT 가 바닥에서 출발해 기울 수 있어 절대 차이도 같이 적음. test 99행 → CV 를 같이 봄

### tps-may-2022 × Codex × GPT 의 F2 (2026-11-13, #12)

에이전트: `tabular-playground-series-may-2022__codex-gpt__r1__20261112-102453` (test 0.99734). 코드는
`competitions/mleb-tps-may-2022/features/f2.py`. 사람 피처 `~f1` (f_27 글자 위치 10 + 서로 다른 글자 수, 41열) 이 이미 있어 F0 · F1 · F2.

| 변형 | 블록 | 열 | TFM 상한 |
|---|---|---|---|
| `~f2` | 원본 30 + f_27 (위치 10 · 글자별 개수 20 · 고유 수 · 합 · 이웃 반복) 33 + f_28 선형 나머지 1 + PCA 8 + 쌍 (f_00·01·02·05 곱·합·차 18, 글자 위치 같음 45) + f_30 별 GMM (성분 2~16 확률 + 로그 가능도) 58 + GMM 확률 log 51 | 244 | 없음 |

- 최종 제출 (`build_submission2.py`) = NN 10개 + LightGBM 2개 (lgb2·lgb4, lgb4 는 전체 train 재학습 lgbfull4 와 반반) 를 온도로 섞고 `meta_correction` 을 더함. 피처는 그 성분들이 쓴 것의
  합집합: `lgb4.py` 의 193열 + `nn_gmmlog.py` 만 쓴 GMM log 51열
- 뺀 것: `stacked.npy` (OOF 예측 = 타깃, 최종 성분은 안 씀), `meta_correction` (검증 예측 위에서 학습한 LightGBM 보정 = 모델 쪽),
  NN 의 원-핫·표준화 (입력 인코딩), lgb 의 글자 열 범주형 지정 (`~f1` 과 같게 수치형)
- GMM·PCA·선형 회귀는 train+test 전체로 fit (타깃 비의존, 에이전트 그대로). GMM 은 `data/_features/` 에 캐시
- 동일성 확인 (로컬, 같은 MLE-bench 분할 80만 + 10만 행): 열 수가 에이전트 기록과 같음 (lgb 193, NN 입력 448 → 479 → 506 → 557 이
  GMM 31·58·log 51 을 더한 것과 맞음). 작업 디렉터리에 남은 `gmm2·3·4.npy` 와 값이 같음 (최대 차이 6e-8, float32). 선형 회귀 계수가
  `lgb4.log` 출력과 같음. `gmm8·12·16`·`gmm_log` 는 20MB 를 넘어 회수에서 빠져 값 비교는 못 함. 만드는 시간: 로컬 8코어 약 30초 (캐시 뒤 4초)
- FM 은 컨텍스트가 순위를 정함 → `~f1` 의 0.99232 와 같은 `@rand200000` 컨텍스트. 244열이라 `~f1` (41열) 보다 메모리가 큼

## C2 규칙 (2026-11-12)

`fab mleb agent run <대회> <설정> --fm` → 설정 이름 `<config>+fm` (`+cont` 와 같이 쓰면 `claude-code-sonnet+cont+fm`).
나머지(지시문 본문·1시간·A10·프록시·이어 가기 규칙)는 공식 설정과 같음. 코드는 `fab/agent/runner.py` 의 "C2".

| 항목 | 정한 것 | 까닭 |
|---|---|---|
| 깔 TFM | `tabpfn_v2` (tabpfn 9.2.0), `tabicl` (2.2.0), `kumo` (structured-data-models 0.2.1, `nvidia/Kumo-Tabular` v1.0.1) | C1 의 TFM 셋. 셋 다 라이선스 동의 없이 받음 (TabPFN 은 v2 만, Kumo 는 Apache-2.0). 버전은 FM 실행 이미지와 같게 |
| 가중치 | 하네스 시작 전에 받아 둠 (HOME 캐시) | 받는 시간을 1시간에 넣지 않음. 설치·사전 점검도 1시간 밖 (공식 설정의 패키지 설치와 같은 자리) |
| 지시문 | 공식 지시문 + 추가 안내 뒤, 대회 설명 앞에 `fab/agent/fm_notes.txt` (영어 약 3,300자): 이름·제약 (TabPFN 1만 행·10 클래스 등) + 그대로 도는 사용 예. "optional" 이라고 적고 쓰라고 권하지 않음 | 이름만 주면 TabPFN 기본값(최신 버전)이 브라우저 라이선스 동의에서 실패하고 Kumo 는 `TableTensor` API 를 찾아야 해서, "FM 이 도움이 되나" 대신 "API 를 찾아내나" 를 잼 |
| 사전 점검 | `fab/agent/fm_warmup.py` 가 안내문의 사용 예를 **그대로** 합성 데이터(3 클래스 + 회귀)로 실행. 하나라도 실패하면 하네스를 시작하지 않음 (meta.json 의 `fm_warmup`) | 에이전트가 받은 예가 실제로 돎을 보장, 실행마다 받은 TFM 이 달라지지 않게 |
| Kumo 컴파일 도구 | gcc·Python.h 가 없으면 apt 로 설치 시도 (meta.json 의 `fm_build_tools`) | FM 실행 이미지에서 베이스에 없어 넣었던 것 (docs/known-issues.md). 에이전트 노드는 이미지가 아닌 기본 환경 |
| 썼는지 | 실행 중 import 기록 (`fab/agent/fm_audit.py`, 모든 실행에 깖, 에이전트에게는 안 알림) + 코드 검사 | 아래 열 |

결과 표(`results/mlebench_runs.csv`)에 더한 열

| 열 | 뜻 |
|---|---|
| `fm_offered` | +fm 이 깔아 준 TFM (사전 점검 통과). 공식 설정은 비어 있음 |
| `tfm_imported` | 작업 디렉터리 코드의 import 문 (Kumo 는 `sdm`). `tfm_in_code` 는 이름만 (주석 포함, `KumoTabular` 도 이제 셈) |
| `tfm_runtime`·`tfm_runtime_procs` | 실행 중 하네스와 그 자식 프로세스가 실제로 import 한 TFM, 그런 프로세스 수 (모듈 실행이 끝나야 셈: 설치 여부만 보는 `find_spec` 은 안 셈) |
| `tfm_at_submit` | 마지막 `submission.csv` 를 쓴 시각에 살아 있던, TFM 을 import 한 프로세스의 TFM. 끝난 시각은 exit 줄, 없으면 (강제 종료·`os._exit`: AIDE 의 코드 실행 프로세스, multiprocessing 자식) 30초마다 남기는 alive 줄로 어림. 근사: TFM 예측을 파일로 남기고 다른 프로세스에서 섞으면 빠지고, TFM 을 import 만 한 프로세스가 다른 모델로 제출하면 잡힘 → 최종 풀이 코드로 확인 |

- 이 기록 이전 실행(P0·P1)은 `tfm_runtime`·`tfm_at_submit` 가 비어 있음 (None). `fab mleb agent recheck` 로 다시 계산해도 같음.
  CSV 에서는 None 과 빈 목록(기록은 있는데 안 씀)이 둘 다 빈칸 → 가르는 건 `tfm_runtime_procs` (빈칸 = 기록 없음, 0 = 안 씀)
- import 기록은 loader 클래스를 바꾸지 않고 그 모듈 loader 인스턴스의 `exec_module` 만 감쌈 (pkg_resources 처럼 loader 종류로 고르는 코드가 그대로 동작)
- +fm 실행은 지시문에 TFM 이름이 있어 `tfm_mentions` 가 늘 참 → 보지 않음
- 환경 차이: TFM 설치가 공유 의존성(torch·numpy 등)의 버전을 바꿀 수 있음 → 같은 대회의 공식 설정 실행과 `pip_before.txt` 를 비교해 적음
- 읽는 법: C2 점수 − 같은 설정 Agent 점수 (1·2회째). FM 을 썼으면 C1 (같은 대회의 "에이전트 피처 + FM") 과도 비교:
  nomad 에서 Codex 가 자기 피처로 kumo 를 돌리면 C1 의 `kumo~f2s` 0.04803·`kumo~f2` 0.04523 근처가 기대값

## 정해야 할 것

(정함) +cont 이어 가기 상한: 도구 0개 구간 연속 3번 + 전체 40번 (2026-11-11, [docs/mle-bench.md](../mle-bench.md) 의 "+cont")

(정함) C2 의 FM 목록과 안내 문구: tabpfn_v2·tabicl·kumo, 이름 + 사용 예 (2026-11-12, 위 "C2 규칙")

(정함) TabArena 연동 (2026-11-13, [docs/tabarena.md](../tabarena.md) 의 "결정"): v0.1 51개, Agent 는 r0f0 split 하나,
데이터셋 이름·출처는 안 알림 ("모델링 실력" 비교), FM 은 지금 설정 그대로 두고 TabArena 와의 차이를 기준 (데이터셋 80% 가
split 표준편차 안 + 승패 방향 같음) 으로 봄, (데이터셋, split) 마다 대회, `openml`·`pyarrow` 는 옆 venv, GBDT 는 우리 것을 돌림

- 해석 주의: Agent 는 LLM 이 공개 Kaggle 풀이를 학습했을 수 있음 (FM 은 아님). TabArena 데이터셋도 같은 문제가 있는지
  → 있음: 51개 중 Kaggle 15 (대회 3), 교과서급 UCI 다수. 공개 풀이 수는 세지 않음 ([docs/tabarena.md](../tabarena.md) 의 "6. 오염").
  TabArena 는 데이터셋 이름을 안 알려 정답 찾기를 줄이지만, 피처 이름으로 알아볼 수 있는 데이터는 남음
