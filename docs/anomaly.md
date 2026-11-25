# 표 이상탐지 (ADBench): tabular FM 대 Agent, 기존 방법 = Elo 기준점

질문: tabular FM 의 비지도 확장 (공식 tabpfn-extensions `TabPFNUnsupervisedModel.outliers()`) 이 표 이상탐지에서 실제로 먹히는지.
같은 데이터에서 LLM 코딩 에이전트 (기존 5설정 × 1시간) · 기존 이상탐지 방법과 한 풀에 넣고 Elo 로 답한다.
배경은 minipfn issue #117 (tabular FM 을 회귀·분류 밖 문제에 쓰는 방법과 벤치 목록). 여기서 하는 것은 minipfn 구현이 아니라
**공개된 tabular FM 이 이 문제에서 먹히는지를 이 저장소의 FM 대 Agent 틀로 검증**하는 것 (minipfn 코드·가중치는 안 씀).

- 2026-11-25 설계 (사용자 확인: 결정 표), 같은 날 연동 구현 · 로컬 확인 ("로컬 확인"), 원격 실행 ("실행 기록"). 순서·규칙은 TabArena 트랙 ([tabarena.md](tabarena.md), P4) 을 따름
- 확인 못 한 것은 "확인 못 함" 으로 적음. 추측으로 채우지 않음
- 계획·단계: [plans/roadmap.md](plans/roadmap.md)

목차

- [요약](#요약)
- [출처](#출처)
- [1. 데이터셋](#1-데이터셋)
- [2. 분할·평가 프로토콜](#2-분할평가-프로토콜)
- [3. 기존 방법 풀과 기준점](#3-기존-방법-풀과-기준점)
- [4. FM: TabPFN 비지도 확장](#4-fm-tabpfn-비지도-확장)
- [5. Elo](#5-elo)
- [6. Agent 형식 · 정답 격리 · 익명화](#6-agent-형식--정답-격리--익명화)
- [7. 오염과 사후 검사](#7-오염과-사후-검사)
- [8. 연동 설계](#8-연동-설계)
- [9. 비용 어림](#9-비용-어림)
- [로컬 확인 (2026-11-25)](#로컬-확인-2026-11-25)
- [실행 기록](#실행-기록)
- [결정 (2026-11-25)](#결정-2026-11-25)
- [확인 못 한 것](#확인-못-한-것)

## 요약

| 항목 | 정한 것 (초안) |
|---|---|
| 벤치 | ADBench (`Minqi824/ADBench`, BSD-2) 의 Classical 47개 중 규칙으로 고른 **12개** (1) |
| 설정 | 비지도: 라벨 없는 train (이상점이 원래 비율로 섞임) 으로 fit → test 행마다 이상 점수. ADBench 의 분할 그대로 (seed 1·2·3 = split `s1`·`s2`·`s3`, 1만 행 넘으면 1만으로 표본, 층화 70/30, train 으로 fit 한 MinMax) (2) |
| 지표 | AUROC (주), AUPRC (보조). Elo 오차 = 1 − AUROC |
| 기준점 | **IsolationForest (pyod 기본) = 1000** (3). 같은 풀: pyod 기본 설정 10개 (CPU, 우리가 돌림) |
| 맞춤 확인 | 우리 분할 + pyod 로 ADBench 논문 Table D4 의 IForest·KNN AUROC 가 12개 × 3 seed 평균에서 **소수 둘째 자리까지 같음** (2, 로컬) |
| FM | `TabPFNUnsupervisedModel.outliers(n_permutations=10)` 기본, 점수 = −log 밀도. 가중치는 TabPFN v2 (Prior Labs License 1.1, 동의 없이 받음, 결정 2) (4) |
| Agent | 기존 5설정 × 1시간 · A10, 라벨 없는 train · test + 설명 → `id,score`. train 으로만 fit (결정 4). 파일럿 cardio · fault (`s1`) (6) |
| 격리 | 정답 (test 라벨) 은 로컬 캐시와 채점용 내 Volume 에만. 데이터셋 이름·출처는 안 알림 (익명 id `anomaly-p01`…). 공개 데이터라 알아본 흔적은 사후 검사 (7) |
| 범위 밖 | C1·C2 (Agent 결과를 보고 정함), 결측값·생성·밀도, kumo·tabicl 로 chain rule 래퍼 새로 만들기 (후속) |

## 출처

| 출처 | 쓴 곳 |
|---|---|
| ADBench 저장소 [`Minqi824/ADBench` @ 3dac822](https://github.com/Minqi824/ADBench/tree/3dac8221081e190f157d78e93bfa8867f90d0965) (2026-01-08 커밋, BSD-2) | `adbench/datasets/data_generator.py` (분할), `adbench/run.py` (seed·비지도 설정), `adbench/baseline/PyOD.py` (기본 설정), `adbench/myutils.py` (지표), `adbench/datasets/Classical/*.npz` (데이터) |
| ADBench 논문: Han et al., "ADBench: Anomaly Detection Benchmark", NeurIPS 2022 D&B, [arXiv:2206.09426](https://arxiv.org/abs/2206.09426) | Table B1 (데이터셋·분야), Table D4 (비지도 14개 AUROC, seed 3개 평균) |
| tabpfn-extensions [`PriorLabs/tabpfn-extensions` @ a049fd7](https://github.com/PriorLabs/tabpfn-extensions/tree/a049fd76afa0593a616156e1008b5482e9c905da) (v0.6.3, Apache-2.0) | `src/tabpfn_extensions/unsupervised/unsupervised.py` (`outliers`), `utils.py` (`infer_categorical_features`) |
| TabPFN 가중치 라이선스 (HF 모델 카드) | v2 (`Prior-Labs/TabPFN-v2-clf`·`-reg`): Prior Labs License 1.1 (Apache-2.0 + 표시 조항), 동의 없이 받음. 3.5 (`Prior-Labs/tabpfn_3_5`): TABPFN-3.5 Non-Commercial License, 받을 때 동의 (브라우저 또는 `TABPFN_TOKEN`) |
| TACTIC, [arXiv:2603.14171](https://arxiv.org/abs/2603.14171) | 학습 context 에 이상점이 섞이면 TabPFN 식 이상탐지가 "불안정" 하다는 것까지만 원문 확인 (그 밖의 내용은 확인 못 함) |

## 1. 데이터셋

### 고르는 규칙 (ADBench Classical 47개에 차례로)

| # | 규칙 | 까닭 | 남음 |
|---|---|---|---|
| a | ADBench Table B1 의 분야가 Image 인 것 뺌 (ALOI · celeba · InternetAds · letter · mnist · optdigits · pendigits · skin). CV·NLP embedding 10개는 처음부터 안 봄 | 고전 tabular 위주 (이미지 화소·임베딩에서 뽑은 피처는 따로 판단) | 39 |
| b | 행 1,000 미만 뺌 (12개) | ADBench 는 1,000 행 미만을 복원 추출로 1,000 행까지 불림 (`generate_duplicates`) → 같은 행이 train·test 양쪽에 들어가 점수가 부풂 | 27 |
| c | 피처 5~40 개만 (넓은 6개 · 3열 2개 뺌) | FM 비용이 순열 × 피처 수 (4). 3열 (http · smtp) 은 너무 쉬움 · 이상 0.03% | 19 |
| d | ADBench 분할의 test 이상 행이 20개 미만이면 뺌 (fraud 5 · vowels 15) | AUROC 가 이상 몇 개로 갈림 | 17 |
| e | 완전히 같은 행이 절반 넘으면 뺌 (donors 94%) | 같은 행이 train·test 에 걸침 | 16 |
| f | 같은 원본에서 이상 정의만 다른 묶음은 ADBench 번호가 앞선 하나: {annthyroid, thyroid}, {cardio, Cardiotocography}, {landsat, satellite, satimage-2} | 같은 표가 여러 번 들어가 Elo 를 기울이지 않게 | **12** |

### 12개

행·피처·이상 비율은 원본 (ADBench npz). train·test 는 ADBench 분할 (1만 행 넘으면 1만으로 표본 → 7,000 / 3,000). 범주 열 = tabpfn-extensions 가 classifier 로 보는 열 (4).
IForest·최고 = ADBench Table D4 (비지도 14개, seed 3개 평균 AUROC ×100).

| ADBench | 분야 | 행 | 피처 | 이상 % | train / test | test 이상 | 범주 열 | IForest | 최고 (방법) |
|---|---|---|---|---|---|---|---|---|---|
| 2_annthyroid | Healthcare | 7,200 | 6 | 7.42 | 5,040 / 2,160 | 160 | 0 | 82.01 | 82.01 (IForest) |
| 6_cardio | Healthcare | 1,831 | 21 | 9.61 | 1,281 / 550 | 53 | 4 | 93.19 | 95.55 (PCA) |
| 10_cover | Botany | 286,048 | 10 | 0.96 | 7,000 / 3,000 | 28 | 0 | 86.74 | 93.73 (PCA) |
| 12_fault | Physical | 1,941 | 27 | 34.67 | 1,358 / 583 | 202 | 4 | 57.02 | 72.98 (KNN) |
| 19_landsat | Astronautics | 6,435 | 36 | 20.71 | 4,504 / 1,931 | 400 | 0 | 47.64 | 63.61 (DeepSVDD) |
| 22_magic.gamma | Physical | 19,020 | 10 | 35.16 | 7,000 / 3,000 | 1,050 | 0 | 73.25 | 82.38 (KNN) |
| 23_mammography | Healthcare | 11,183 | 6 | 2.32 | 7,000 / 3,000 | 68 | 0 | 86.39 | 90.75 (ECOD) |
| 27_PageBlocks | Document | 5,393 | 10 | 9.46 | 3,775 / 1,618 | 153 | 0 | 89.57 | 90.92 (ECOD) |
| 32_shuttle | Astronautics | 49,097 | 9 | 7.15 | 7,000 / 3,000 | 214 | 0 | 99.56 | 99.56 (IForest) |
| 41_Waveform | Physics | 3,443 | 21 | 2.90 | 2,410 / 1,033 | 30 | 0 | 71.47 | 75.03 (COPOD) |
| 44_Wilt | Botany | 4,819 | 5 | 5.33 | 3,373 / 1,446 | 77 | 0 | 41.94 | 53.25 (SOD) |
| 47_yeast | Biology | 1,484 | 8 | 34.16 | 1,038 / 446 | 152 | 2 | 37.76 | 47.92 (DeepSVDD) |

- landsat · Wilt · yeast 는 ADBench 의 비지도 14개가 모두 AUROC 0.5 근처 이하: 라벨의 "이상" 이 밀도상 드문 행이 아님 (분류 데이터의 소수 클래스를 이상으로 둔 것).
  여기서 0.5 아래로 크게 내려간 방법은 "드문 행일수록 정상" 을 잡은 것 → 라벨 없이 점수 방향을 뒤집을 근거는 없음. Agent 가 데이터셋을 알아보고 방향을 뒤집는지는 사후 검사로 봄 (7)
- cover 는 test 이상이 28개라 split 사이 흔들림이 큼 (IForest 표준편차 2.5점, 2)
- 데이터셋 이름은 ADBench 번호를 뗀 것 (`annthyroid` 등), 대회 폴더는 `ad-<이름>` (8)

## 2. 분할·평가 프로토콜

ADBench 원문 (`data_generator.py` 의 `DataGenerator.generator`, `run.py` 의 `RunPipeline`, `parallel='unsupervise'`)

| 단계 | ADBench | 우리 |
|---|---|---|
| seed | `seed_list = [1, 2, 3]`, 데이터마다 `Utils.set_seed(seed)` (numpy · random · torch · tf 전역) | 같음. split 이름 `s1`·`s2`·`s3` |
| 작은 데이터 | 1,000 행 미만이면 복원 추출로 1,000 행 | 해당 데이터셋을 뺌 (1 의 b) |
| 큰 데이터 | 1만 행 넘으면 `set_seed(seed)` 다시 → `np.random.choice(n, 10000, replace=False)` | 같음 |
| 나누기 | `train_test_split(X, y, test_size=0.3, shuffle=True, stratify=y)` (random_state 없음 → numpy 전역 상태) | 같은 호출 |
| 척도 | `MinMaxScaler().fit(X_train)` → train·test 변환 | 같음. **모든 방법 (기존 · FM · Agent) 이 같은 척도의 표를 받음** (Agent 의 train.csv · test.csv 가 이 값) |
| 비지도 라벨 | 라벨 비율 `la = 0` → `y_train` 은 모두 0 (이상점은 라벨 없이 train 에 원래 비율로 섞임 = 오염) | 같음: train 에 라벨을 안 줌 |
| 학습 · 점수 | `clf.fit(X_train)` → `predict_score(X_test)` (pyod `decision_function`, 클수록 이상) | 같음 |
| 지표 | `roc_auc_score`, `average_precision_score` (test) | 같음. Elo 오차 = 1 − AUROC |
| 합치기 | seed 3개 평균 | split 마다 따로 채점, Elo 는 데이터셋마다 가중치 합 1 (5) |

재현 확인 (2026-11-25, 로컬, pyod 3.6.8 · scikit-learn 1.9.1 · numpy 2.5.5). 위 분할로 seed 마다 전역 seed 를 다시 놓고 pyod 기본 설정을 fit:

| 방법 | 12개 중 Table D4 와 차이 0.01 이하 | 가장 큰 차이 |
|---|---|---|
| IForest | 12 | 0.00 |
| KNN | 12 | 0.00 |
| ECOD | 1 (mammography) | landsat −20.5, Waveform −10.9 (나머지 ±4.2 안) |

- IForest·KNN 이 같다는 것은 분할 (표본 · 층화 70/30 · 척도) 이 ADBench 와 행 단위로 같다는 뜻 (seed 마다 같은 행을 써야 평균이 소수 둘째 자리까지 같음)
- ECOD 차이는 pyod 버전 (ADBench `requirements.txt` 는 `pyod==1.0.0`) 의 ECOD 구현 변화로 봄 (어디가 바뀌었는지는 확인 못 함). 우리는 지금 pyod 를 쓰고 차이를 적음.
  맞춤 판정은 기준점 IForest 로 함
- pyod 의 `random_state` 기본값은 None 이라 ADBench 는 전역 seed 로 정해짐 → 우리도 fit 직전에 전역 seed 를 같은 값으로 놓음
- 파일의 행 순서는 ADBench 분할이 내놓은 순서 그대로 (`train_test_split` 이 섞은 무작위 순서라 라벨 정보 없음). TabArena 처럼 id 순으로 다시 놓으면
  행 순서로 표본을 뽑는 IForest 가 ADBench 와 달라지고 (cardio 0.9315 대 0.9319), pyod PCA 가 cardio s2 · s3 에서 NaN 을 냄 (성분 분산 0 으로 나눔) →
  ADBench 순서로 두니 둘 다 Table D4 와 같음 ("로컬 확인")

## 3. 기존 방법 풀과 기준점

pyod 기본 설정 (인자 없이), CPU. ADBench 의 비지도 14개 중:

| 넣음 (10) | 뺌 |
|---|---|
| **IForest (기준점)**, PCA, OCSVM, LOF, CBLOF, HBOS, KNN, COPOD, ECOD, LODA | COF · SOD (행 수의 제곱 · 느림, ADBench 에서 최고인 곳이 적음), DeepSVDD · DAGMM (torch · tensorflow 학습, 기본 설정이 버전마다 다름) |

- 기준점 IsolationForest = 1000: ADBench 에서 비지도 평균 순위가 높은 편이고 설정이 단순 (TabArena 의 RandomForest 처럼 "잘 알려진 기본 기준선")
- ADBench 공개값 (Table D4) 은 맞춤 확인에만 씀 (2). 풀의 점수는 모두 우리 실행 (같은 split 의 test)
- 결과 표 `results/anomaly_reference.csv` 에 Table D4 의 12개 행 (14개 방법) 을 옮겨 둠 (구현 때)

## 4. FM: TabPFN 비지도 확장

`TabPFNUnsupervisedModel(tabpfn_clf=TabPFNClassifier(...), tabpfn_reg=TabPFNRegressor(...)).fit(X_train).outliers(X_test, n_permutations=10)`

| 항목 | 원문 (tabpfn-extensions a049fd7) | 우리 |
|---|---|---|
| 계산 | 피처 순서를 무작위로 10개 (`efficient_random_permutation`, Python `random`). 순열마다 chain rule: 피처 j 를 앞 피처들로 TabPFN 이 예측한 조건부 밀도 (회귀: bar 분포의 −NLL, 범주: 클래스 확률) 의 log 를 더함. 첫 피처는 정규 잡음 한 열을 입력으로. 10개를 log-sum-exp 로 산술 평균 | 그대로. 점수 = −`outliers()` (낮은 밀도 = 이상) |
| context | `fit(X_train)` 의 행 전부 (라벨 없음 → **이상점이 섞인 context**) | train 전부 (≤ 7,000 행, TabPFN v2 의 1만 행 한도 안) |
| 범주 열 | `infer_categorical_features`: 값 40개 이하 · 값마다 평균 10행 넘는 열 → classifier. 그 열의 값은 `astype(int)` 로 클래스 번호가 됨 | 그대로. 주의: 입력이 MinMax 된 [0, 1] 이라 값이 셋 이상인 범주 열은 `astype(int)` 에서 0 과 1 로 뭉개짐 (cardio 3열 · fault 2열 · yeast 1열). 기본 동작의 한계로 적고 고치지 않음 |
| seed | 순열은 전역 `random` | split seed 로 `random`·numpy·torch 를 놓음. TabPFN 의 `random_state` 는 기본 |
| 가중치 | 패키지 기본 = 최신 (tabpfn 9.2.0 은 v3.5, 비상업 · 받을 때 동의) | **v2** (결정 2, 사용자): `create_default_for_version(ModelVersion.V2)` 를 분류 · 회귀 둘 다 (`tabpfn-v2-classifier.ckpt` · `tabpfn-v2-regressor.ckpt`, 우리 `tabpfn_v2` 와 같은 가중치). Prior Labs License 1.1, 받을 때 동의 없음 → 토큰 · 가중치 올리기가 필요 없음. 패키지 기본 v3.5 는 쓰지 않음. train ≤ 7,000 행 · 피처 ≤ 36 이라 v2 의 공식 한도 (1만 행 · 500 피처 · 10 클래스) 안 |
| 원격 | | GPU (H100 1장), 실행마다 `pip install tabpfn-extensions==0.6.3` (이미지에 tabpfn 9.2.0 있음), telemetry 끔 (`TABPFN_DISABLE_TELEMETRY=1`, `tabpfn_common_utils` 의 `telemetry_enabled`) |

이름: `tabpfn_unsup` (v2, 순열 10). 결과표의 `opts` 에 가중치 버전 · 순열 수를 남김 (`n_permutations=10 version=v2`).

학습 context 오염의 영향 (질문의 일부). TACTIC 은 "불안정" 까지만 원문 확인 → 결과에서 직접 봄:

1. 데이터셋 사이: 이상 비율 (1% ~ 35%) 과 `tabpfn_unsup − IForest` (AUROC) 의 관계 (12개라 판단은 약함)
2. (결정 6) 깨끗한 context 진단: train 의 정상 행만 context 로 쓴 `tabpfn_unsup+clean` 과 `iforest+clean` (train 라벨을 씀 → 비지도가 아님, **Elo 에 안 넣는 진단**). 오염된 것과의 차이 = 오염 몫.
   train 라벨은 FM 작업만 읽는 내 Volume 자리에 두고 Agent 데이터 묶음에는 넣지 않음

## 5. Elo

`fab/bench/tabarena/elo.py` 의 `elo()` 를 그대로 씀. 기준점만 인자로: `elo(long, ..., anchor="RandomForest")` (`fill` 도 같은 인자). TabArena 쪽은 기본값 그대로라
`results/tabarena_elo.csv` 가 비트 단위로 같아야 함 (구현 뒤 `fab ta elo` 로 확인). forecasting 트랙과 같은 시그니처.

| 풀 | 단위 | 참가자 |
|---|---|---|
| `fm` | 12개 × split 3개 (36), 짝 가중치 1/3 → 데이터셋마다 합 1 | 기존 10개 + `tabpfn_unsup` |
| `agent` | Agent 를 돌린 데이터셋의 `s1` (파일럿 2개) | 위 + Agent 5설정 (회차 평균, 실패 회차는 IForest 값으로 채움 = TabArena 와 같은 처리) |

- `agent` 풀은 데이터셋이 10개 미만이라 95% 구간을 비움 (`MIN_CI_DATASETS`, TabArena 와 같음). `fm` 풀 12개는 부트스트랩 200번
- 오차가 같으면 비김 반 (AUROC 가 같은 값은 드묾)

## 6. Agent 형식 · 정답 격리 · 익명화

대회 하나 = (데이터셋, split). Agent 는 `s1` 만 (TabArena 의 r0f0 와 같은 자리).

| 파일 | 내용 |
|---|---|
| `train.csv` | `id`, `f0` … `f{d−1}` (MinMax 된 값, 2). 라벨 없음. 이상 행이 섞여 있다는 것만 알림 (비율은 안 알림) |
| `test.csv` | 같은 열. 점수를 낼 행 |
| `sample_submission.csv` | `id,score` (모두 0). score 는 클수록 이상, 순위만 봄 (AUROC) |
| `description.md` | 과제 · 파일 · 제출 형식 · 지표. 열 뜻 · 분야 · 출처는 없음 (ADBench npz 에 열 이름이 없고, 분야를 주면 알아보기 쉬워짐, 결정 5) |
| 정답 | test 행의 라벨 (0/1). 로컬 `FAB_ANOMALY/data/answers/` 와 채점용 내 Volume 의 `anomaly/answers/` 에만 |

- 행 id 는 split 마다 고정 seed 의 무작위 순열 (TabArena 와 같음), 파일은 id 순
- 검증 서버 (`fab/agent/runner.py` 의 `check_submission`) 는 그대로 씀: 열 · 행 수 · id · 빈칸 (점수 값 범위는 안 봄)
- 지시문: 공식 지시문 + 추가 안내 + 대회 설명 (TabArena 와 같음). 설명의 "Rules" 에 **train 행으로만 fit, test 행으로 fit · 조정 · 변환하지 말 것** (FM · 기존 방법과 같은 inductive 조건, 결정 4). 지켰는지는 사후 검사 (`test_fit_hints` + 코드 읽기)
- 익명 id `anomaly-p01` … (원격 실행 이름 · scratch Volume 의 데이터 묶음 · run.json). 대회 이름은 내 Volume 의 제출 기록에만. 제출 전 누출 검사:
  데이터셋 이름 (원래 · 소문자 · 구분자 바꾼 것), ADBench 번호 이름 (`6_cardio`), `ADBench`·`ODDS` 가 설명 · 파일 이름 · 코드 묶음에 없는지 (TabArena 의 `leak_check` 를 같이 씀)
- 채점: 원격 정리 작업이 내 신원으로 (TabArena 의 Grader 와 같은 꼴) → `results/anomaly_runs.csv`

파일럿 2개 (결정 3): **cardio** (1,831 행 · 21 피처 · 밀도 방법이 잘 됨: IForest 93) 와 **fault** (1,941 행 · 27 피처 · 밀도 방법이 약함: IForest 57, KNN 73).
둘 다 작아 1시간 안에 여러 번 돌려 볼 수 있고, "FM 이 이기는 곳 / Agent 가 방향을 찾을 여지가 있는 곳" 을 하나씩. 나머지는 파일럿 결과를 보고.

## 7. 오염과 사후 검사

ADBench 데이터 다수는 유명한 공개 데이터 (UCI · ODDS: thyroid · cardiotocography · forest cover · MAGIC gamma telescope · mammography · page blocks · shuttle · yeast 등).
행 수 · 피처 수 · 이상 비율만으로도 알아볼 수 있음. 원래 라벨을 다른 데서 가져오면 test 라벨을 맞출 수 있으므로 사후 검사로 기록 (TabArena 와 같음).

| 검사 | 찾는 것 |
|---|---|
| 데이터 출처 접근 (기존 `DATA_SOURCE_PATTERN` + 더할 것) | `github.com/Minqi824/ADBench`, `adbench` 패키지, `odds.cs.stonybrook.edu`, `pyod.utils.data`, `sklearn.datasets.fetch_*` (`fetch_covtype`·`fetch_kddcup99`), UCI · OpenML · Kaggle (기존) |
| 알아본 흔적 | LLM 글 · 코드에 12개 데이터셋 이름이나 원래 이름 (예: "MAGIC gamma", "Statlog", "cardiotocography") |
| 점수 방향 | 0.5 아래 데이터 (landsat · Wilt · yeast) 에서 점수를 뒤집었는지, 그 근거 (데이터셋 지식 대 데이터에서 본 것) |
| test 사용 | 규칙 (train 만) 을 어기고 test 행의 피처로 fit 했는지 (`test_fit_hints`: train · test 이어 붙이기, test 로 fit 하는 코드 줄 수 → 코드 읽기) |
| TFM | 기존 열 (`tfm_in_code` · `tfm_runtime` 등). 공식 설정 Agent 가 스스로 tabpfn 을 깔아 쓰면 Agent 결과로 셈 (TabArena 결정 11) |

## 8. 연동 설계

TabArena 와 같은 틀. 새 코드는 `fab/bench/anomaly/` 에 (`__init__.py` 준비 · 채점 · 비교 · Elo · 명령, `methods.py` 방법, `run.py` 원격 실행, `remote.py` Volume · 채점기, `posthoc.py` 사후 검사). 2026-11-25 구현.

| 자리 | 내용 |
|---|---|
| 데이터 위치 | `FAB_ANOMALY` (절대경로, 기본 `<저장소>/../anomaly`): `src/adbench/` (ADBench 저장소, Classical npz), `data/answers/<대회>.{csv,json}` (정답), `.venv/` (로컬 확인용 pyod · tabpfn-extensions, uv.lock 은 안 바꿈) |
| 대회 | `competitions/ad-<이름>/` (s1, git 추적: `config.yaml` 의 `anomaly: {dataset, adbench, split, seed, anon_id}` + `description.md`), 다른 split 은 `competitions/ads-<이름>-s<k>/` (생성, git 제외) |
| 명령 | `fab ad prepare|fm|grade|compare|elo|agent` (`fab/cli.py` 에 `ad` 한 줄) |
| 실행 | 기존 `fab databricks` 실행기 (`RUNNER`) 를 그대로: CPU 작업 (pyod) · GPU 작업 (TabPFN). 실행기가 부르는 `fab run` 이 `anomaly` 설정이 있는 대회는 `fab.bench.anomaly.run` 으로 넘김 (`fab/runner/run.py` 에 몇 줄), 설치는 `pip_args` 가 이 대회의 방법을 보고 (pyod · tabpfn-extensions). 출력은 FM 과 같은 자리 (`<대회>/<방법>/test.npy` + `test.json`, `results.csv` 행) → 정리 작업의 `fm` 단계가 그대로 합침 |
| 채점 | 정리 작업의 `grade` 에 `bench=anomaly` (TabArena 의 `grade_tabarena` 꼴), Agent 는 `agent` 단계에서 `bench=anomaly` → `results/anomaly.csv` · `results/anomaly_runs.csv` (`fab sync` 표에 추가) |
| Agent | `fab/agent` 의 `main(bench=...)` 에 `anomaly` (대회 · 익명 id · 누출 검사 · 정답 올리기). 실행기 (`runner.py`) 는 안 바꿈 (지시문 · 검증 서버가 이미 일반적) |
| Elo | `elo.py` 에 `anchor` 인자 (5) |

운영 주의: 정리 작업 코드는 로컬 명령을 낸 체크아웃의 것으로 다시 올라감 (docs/known-issues.md). 이 브랜치가 머지되기 전에 다른 worktree 에서
`fab remote kick` 하면 옛 코드가 `bench=anomaly` 항목 (채점 요청 · Agent 실행) 을 MLE-bench 쪽으로 처리하다 3번 실패해 오류로 닫음 →
이 트랙의 항목이 남아 있는 동안 kick 은 이 worktree 에서. 채점 요청은 작업이 끝난 뒤 내도 됨 (`fab ad fm --no-grade` → `fab ad grade`).

같이 도는 forecasting 트랙과 겹치는 곳: `fab/cli.py`, `fab/core/competition.py` (설정 키 하나), `fab/runner/run.py`·`common.py` (넘기는 몇 줄), `fab/agent/__init__.py`, `fab/remote/{__init__,reconcile,sync}.py`, `elo.py`, README, roadmap. 나중에 머지되는 쪽이 rebase.

## 9. 비용 어림

| 실행 | 어림 (설계 때) | 잰 값 (2026-11-25) |
|---|---|---|
| 기존 10개 + `iforest+clean` | serverless CPU 작업 하나, 수 분 ~ 30분 (12개 × 3 split × 10개) | 작업 18분 (환경 · pyod 설치 포함, 대회 36개를 차례로. 방법 하나는 대회마다 0~3초, CBLOF · HBOS 가 가장 김) |
| `tabpfn_unsup` | TabPFN fit + 예측이 split 마다 순열 10 × 피처 수 번 → 12개 합 1,690번 / split, 3 split 5,070번. H100 1장에서 번당 1~3초로 어림하면 **1.5 ~ 4 GPU 시간** (재 보지 않음) | **s1 (12개) 에 `+clean` 까지 H100 47분** (`tabpfn_unsup` 만 약 24분, 번당 약 0.85초). 데이터셋마다 25초 (Wilt, 5 피처) ~ 552초 (landsat, 36 피처) |
| `+clean` 진단 (결정 6) | `tabpfn_unsup` 의 약 0.9배 (context 가 정상 행만) + IForest 몇 초 | `tabpfn_unsup` 의 0.5 ~ 1.0배 (이상이 많은 데이터는 context 가 작아짐) |
| Agent | 2개 × 5설정 × 1회 = **10회 × 1시간** (A10, 노드 대기 포함 상한) + 토큰 | 파일럿 2개 10회 + Wilt 5회 (+ 플랫폼 오류 다시 내기 1회). AIDE 는 늘 60분 (시간 종료), Codex 49 ~ 57분, Claude Code 10초 ~ 1분 |

## 로컬 확인 (2026-11-25)

원격에 내기 전 형식 확인. 저장소를 임시 폴더에 복사해 (`results/` 를 건드리지 않게) `FAB_REMOTE=1 fab run <대회> <방법...>` → `grade` → `compare` 를 옆 가상환경
(pyod 3.6.8 · scikit-learn 1.9.1 · numpy 2.5.5 · tabpfn 9.2.0 · tabpfn-extensions 0.6.3) 에서. 점수는 결과표에 안 넣음 (결과는 원격 실행으로).

| 확인 | 결과 |
|---|---|
| 분할 재현 (12개 × 3 seed, 분할 함수만) | IForest · KNN 의 seed 평균이 Table D4 와 12/12 같음 (2) |
| 실행 경로 (cardio · fault · yeast × s1~s3 × 기존 10개) | 99행 채점. Table D4 와 차이 0.005 이하 21/30: IForest · KNN · LOF · OCSVM · HBOS · COPOD 는 3/3 같음, PCA 2/3 (yeast −0.020), CBLOF 1/3, ECOD · LODA 0/3 (pyod 버전 · 무작위 투영) |
| 깨끗한 context (`iforest+clean`, 같은 9개) | seed 평균 AUROC: cardio 0.932 → 0.951, fault 0.570 → 0.643, yeast 0.378 → 0.403 (train 라벨이 FM 작업 경로로 제대로 들어감) |
| TabPFN 경로 (`tabpfn_unsup`, 합성 150 × 3, 순열 2, **v2 가중치** 로 · CPU) | 점수 모양 · 유한 · 같은 seed 같은 점수, 평균 4 옮긴 가짜 이상 AUROC 1.0. (원격과 같은 v2 가중치, GPU 시간은 못 잼) |
| Elo (`fab ad elo`, 위 9개 + 가짜 Agent 표) | fm · agent 풀 계산, `+fm` 설정 · `+clean` 은 빠짐, 실패 회차는 IForest 값 (imputed 0.5) |
| TabArena Elo 그대로인지 | `elo()`·`fill()` 에 `anchor` 를 더한 뒤 `fab ta elo` → `results/tabarena_elo.csv` 차이 없음 (바이트 단위) |
| 누출 검사 (12개) | 에이전트에게 가는 설명 · 파일 이름 · 익명 id · 코드 묶음 (runner.py 등 + MLE-bench 지시문) 에 데이터셋 · ADBench 이름 없음 |
| 설치 · 묶음 | `pip_args` 가 CPU 작업에 `pyod==3.6.8`, GPU 작업에 `tabpfn==9.2.0` · `tabpfn-extensions==0.6.3` 을 넣음. 기존 테스트 29개 통과 · ruff 통과 |

## 실행 기록

| 때 (KST) | 무엇 | 대상 | 상태 |
|---|---|---|---|
| 2026-11-25 07:59 | A: 기존 10개 + `iforest+clean` (`--note ad-base`) | 12개 × s1~s3 (36), serverless CPU 작업 하나 (`ad-12datasets-36splits-cpu-20261125-075904`) + 채점 요청 `anomaly__20261125-080029-017778` | 08:18 끝. 작업 상태는 FAILED (종료 코드 1) 지만 36개 대회를 끝까지 돌았고 결과는 남음: 실패한 방법이 있는 대회가 있어서. 실패: pyod PCA 가 cardio s1 · s3, fault s2 · s3 에서 점수 NaN (로컬에서는 같은 데이터로 통과 → 플랫폼에 따라 성분 분산 0 으로 나눔), CBLOF 가 waveform s1 에서 "Could not form valid cluster separation" (pyod 기본 n_clusters=8). 빠진 칸은 Elo 에서 IForest 값 (imputed). 작업 실행이 두 번 (task 재시도) 이고 두 번째에 PCA 실패가 fault 둘에 더 나옴 → 정리 작업이 합친 쪽 확인 필요 |
| 2026-11-25 08:14 | Agent 파일럿 1: cardio (`anomaly-p02`, s1) × 5설정 (aide-sonnet · claude-code-sonnet · aide-gpt · codex-gpt · claude-code-sonnet+cont), 한 번에 (TabArena #26 과 같은 방식). 내기 전 `fab mleb agent list` 에 다른 실행 없음 | A10 × 1시간 × 5 | 다섯 모두 SUCCESS (AIDE 63분 · Codex 51분 · Claude Code 3분 · +cont 5분) |
| 2026-11-25 10:16 | 정리 작업 kick (이 worktree, 대기 항목은 모두 이 트랙 것) | fm 1 · agent 5 · grade 1 | SUCCESS. `fab sync` → `anomaly.csv` 393행 (36 × 11 − 실패 3: PCA cardio s1 · s3, CBLOF waveform s1. 정리 작업이 합친 것은 첫 번째 task 실행의 출력), `anomaly_runs.csv` 5행 |
| 2026-11-25 10:32 | Agent 파일럿 2: fault (`anomaly-p04`, s1) × 5설정, 한 번에. 내기 전 `fab mleb agent list` 에 다른 실행 없음 | A10 × 1시간 × 5 | 제출됨 |

### 기존 방법 맞춤 (원격, 12개 × 3 seed)

`fab ad compare` (우리 seed 평균 − ADBench Table D4, |차| ≤ 0.005 인 데이터셋 수): IForest · KNN · LOF · OCSVM · HBOS · COPOD **12/12**, PCA 8/11
(cardio 는 seed 1개만 · Wilt +0.198 · yeast −0.020 · mammography +0.009), CBLOF 4/11, ECOD 3/12 (landsat −0.205 · Waveform −0.109), LODA 0/12 (shuttle +0.376 등, 대부분 위로).
기준점 IForest 가 12개 모두 공개값과 같음 → 분할 · 척도 · 채점이 ADBench 와 같음. 다른 값은 pyod 1.0.0 → 3.6.8 의 구현 차이로 보고 (어디가 바뀌었는지는 확인 못 함) 우리 실행 값을 씀.

### cardio Agent (s1, 1회째)

| 설정 | AUROC | AUPRC | 시간 | 메모 |
|---|---|---|---|---|
| Codex × GPT | **0.9538** | 0.6746 | 49분 | pyod 를 스스로 깔았지만 최종은 sklearn IForest 여러 개 · 회전 IForest · OCSVM · 노름 · 꼬리 점수를 train 순위로 맞춰 손으로 정한 가중치로 섞음 (`detector.py`, train 만 fit). 첫 추론 글에서 모양 (1,831 × 21) 으로 "cardiotocography" 로 알아봄, 끝 무렵 "ADBench performance shows IForest is strong" (기억). 원본 받기 · 주소 접근 없음 |
| AIDE × GPT | 0.9260 | 0.5792 | 60분 (시간 종료) | |
| Claude Code × Sonnet +cont | 0.8930 | 0.4827 | 29초 | |
| Claude Code × Sonnet | 0.8782 | 0.4440 | 10초 | |
| AIDE × Sonnet | 0.7991 | 0.3497 | 60분 (시간 종료) | |

같은 split 의 기존 방법: LODA 0.9591 · `iforest+clean` 0.9541 · OCSVM 0.9517 · ECOD 0.9463 · COPOD 0.9380 · **IForest 0.9324** · CBLOF 0.8880 · HBOS 0.8785 · KNN 0.7815 · LOF 0.6702.
Agent 최고 (Codex) 는 IForest 보다 +0.021, 기존 최고 (LODA) 보다 −0.005. 나머지 넷은 IForest 보다 낮음.

### fault Agent (s1, 1회째)

| 설정 | AUROC | AUPRC | 시간 | 메모 |
|---|---|---|---|---|
| Codex × GPT | **0.7224** | 0.5267 | 57분 | 최종 = 12개 성분 (조건부 밀도 · MAF · GMM · 오토인코더 · 숲 근접도 등, 파일 50여 개) 을 손으로 정한 가중치로 섞음 (`final_blend.py`). 첫 추론 글에서 "1941 rows and 27 features, likely from the UCI arrhythmia collection" 이라 짐작하고 비교로 ODDS 이름 (cardiotocography · thyroid · pima · vertebral) 을 늘어놓음 → `ad_names_seen` 의 annthyroid · cardio 는 이 거짓 양성, fault 는 못 알아봄. `test_fit_hints` 0 (계보는 다 읽지 않음) |
| AIDE × GPT | 0.6489 | 0.4842 | 60분 (시간 종료) | |
| AIDE × Sonnet | 0.6246 | 0.4490 | 60분 (시간 종료) | |
| Claude Code × Sonnet +cont | 0.6076 | 0.4395 | 62초 | |
| Claude Code × Sonnet | 0.5866 | 0.4151 | 11초 | |

같은 split 의 기존 방법: **KNN 0.7413** · CBLOF 0.6743 · `iforest+clean` 0.6206 · LOF 0.5774 · **IForest 0.5647** · LODA 0.5515 · OCSVM 0.4414 · HBOS 0.4373 · ECOD 0.4322 · PCA 0.4322 · COPOD 0.4137.
Agent 다섯 모두 IForest 보다 높지만 최고 (Codex) 도 KNN 보다 −0.019.

### Wilt Agent (s1, 1회째, 2026-11-25 14:21 ~ 15:40)

TabPFN 이 가장 크게 이긴 데이터 (FM 쪽에 유리한 대조, s1 의 TabPFN 결과를 보고 정함). 이상 5.33%, 기존 10개가 모두 0.51 이하.

| 설정 | AUROC | AUPRC | 시간 | 메모 |
|---|---|---|---|---|
| Codex × GPT | **0.8006** | 0.1131 | 53분 | **알아봄**: 첫 추론에 "Analyzing Wilt Dataset … 4,819 samples with 5 continuous features and 261 anomalies", 원래 열 (`class, G, R, NIR, Shd, B`) 과 "the AUC for Wilt's models" 를 기억에서 꺼냄. 받기 · 주소 접근 없음. 최종 (`make_candidate_submission.py` + `candidate_v2.py`) = 열 몇 개에 log 변환 → IForest · KNN · MCD · GMM · 밀집도 (GMM 성분 무게) 점수를 train 순위로 맞춰 섞고 작은 "clump" 점수를 더함, train 만 fit. 열 고른 log 변환이 지식에서 왔는지는 가르지 못함 |
| Claude Code × Sonnet +cont | 0.7380 | 0.0902 | 32초 | 첫 실행은 플랫폼 오류 (노드 시작 때 `env_vars.json` 읽기 실패, 52초, 하네스 시작 전) → 제외 (`fab ad agent exclude`) 후 같은 설정으로 다시 냄 |
| AIDE × Sonnet | 0.7367 | 0.0883 | 60분 (시간 종료) | |
| Claude Code × Sonnet | 0.6560 | 0.0690 | 12초 | |
| AIDE × GPT | 0.6344 | 0.0654 | 60분 (시간 종료) | |

같은 split: **`tabpfn_unsup` 0.7450** · LOF 0.5121 · KNN 0.5027 · IForest 0.4087 … (기존 10개 모두 0.51 이하), `tabpfn+clean` 0.9494.
Agent 다섯 모두 기존 10개보다 높음 (밀도로 드문 행을 찾는 기본 방법이 라벨과 반대인 데이터). Codex (알아봄) 만 TabPFN 보다 높음 (+0.056), 나머지 넷은 TabPFN 보다 낮음.

### FM: TabPFN v2 비지도 (s1, 2026-11-25 13:25 ~ 14:12, H100 1장 47분)

AUROC (s1). 기존 최고 = 기존 10개 중 최고, 순위 = 기존 10개 + `tabpfn_unsup` 11개 중. clean = train 의 정상 행만 context (train 라벨을 씀, Elo 밖). 이상 비율 순.

| 데이터셋 | 이상 % | `tabpfn_unsup` | IForest | 기존 최고 | 순위 | `tabpfn+clean` | `iforest+clean` |
|---|---|---|---|---|---|---|---|
| cover | 0.96 | **0.400** | 0.890 | 0.954 (LODA) | 11 | 0.889 | 0.862 |
| mammography | 2.32 | 0.880 | 0.878 | 0.908 (COPOD) | 5 | 0.897 | 0.893 |
| Waveform | 2.90 | 0.760 | 0.730 | 0.817 (LOF) | 3 | 0.843 | 0.763 |
| Wilt | 5.33 | **0.745** | 0.409 | 0.512 (LOF) | 1 | 0.949 | 0.459 |
| shuttle | 7.15 | 0.979 | 0.998 | 0.998 (IForest) | 9 | 1.000 | 0.997 |
| annthyroid | 7.42 | **0.883** | 0.818 | 0.818 (IForest) | 1 | 0.949 | 0.928 |
| PageBlocks | 9.46 | 0.732 | 0.894 | 0.921 (ECOD) | 11 | 0.898 | 0.924 |
| cardio | 9.61 | **0.531** | 0.932 | 0.959 (LODA) | 10 | 0.860 | 0.954 |
| landsat | 20.71 | **0.607** | 0.469 | 0.580 (KNN) | 1 | 0.782 | 0.632 |
| yeast | 34.16 | 0.433 | 0.375 | 0.446 (CBLOF) | 3 | 0.475 | 0.386 |
| fault | 34.67 | 0.642 | 0.565 | 0.741 (KNN) | 3 | 0.796 | 0.621 |
| magic.gamma | 35.16 | **0.849** | 0.741 | 0.828 (KNN) | 1 | 0.893 | 0.776 |

읽은 것 (seed 하나, 판단은 약함)

- TabPFN 은 IForest 보다 8 / 12 에서 낫고, 4 개 (Wilt · annthyroid · landsat · magic.gamma) 에서 11개 중 1등. Wilt 는 기존 10개가 모두 0.51 이하인데 0.745
- 대신 크게 무너지는 곳이 셋: cover 0.400 (0.5 아래), cardio 0.531, PageBlocks 0.732 (각각 11 · 10 · 11 등)
- **그 셋은 context 의 이상점이 원인**: 정상 행만 context 로 주면 cover 0.400 → 0.889 (+0.49), cardio 0.531 → 0.860 (+0.33), PageBlocks 0.732 → 0.898 (+0.17). 같은 진단에서 IForest 는 −0.03 ~ +0.03. 
  cover 는 이상이 1% (train 67행) 뿐인데도 가장 크게 다침 → 해치는 것은 오염 "비율" 이 아니라 이상 행끼리 모여 있어 context 안에 비슷한 이웃이 있는지로 보임 (TabPFN 이 context 의 이웃으로 밀도를 매기면 모인 이상점은 서로를 "정상" 으로 만듦. 이 해석은 확인 안 함).
  이상 비율과 clean 효과의 순위 상관 −0.11, 이상 비율과 `tabpfn − IForest` 의 순위 상관 +0.44 (12개)
- clean context 의 TabPFN 은 12개 중 11개에서 기존 최고 이상 또는 근처 (Wilt 0.949 · annthyroid 0.949 · shuttle 1.000) → 밀도 모형 자체는 좋고, 비지도 설정의 약점은 오염된 context. TACTIC 의 "불안정" 과 같은 방향
- Agent 파일럿 두 곳은 TabPFN 에 불리한 데이터: cardio (무너지는 셋 중 하나) 0.531 대 Codex 0.954, fault 0.642 대 Codex 0.722

Elo (IForest = 1000, s1 만 · 이 절의 계산. `fab ad elo` 의 fm 풀은 3 split 이라 TabPFN 의 s2 · s3 이 IForest 값으로 채워짐 → seed 2 · 3 을 돌린 뒤 그것을 씀)

| 풀 | 결과 |
|---|---|
| fm (12개 s1, 부트스트랩 200번) | IForest 1000 · **tabpfn_unsup 966 (+207 / −198)** · ECOD 949 · KNN 922 · LODA 922 · CBLOF 914 · COPOD 900 · LOF 873 · PCA 854 · HBOS 811 · OCSVM 788. TabPFN 은 IForest 와 구간이 겹침 (이길 때 크게, 질 때 크게) |
| agent (cardio · fault s1, 구간 없음) | **Codex 1448** · AIDE × GPT 1117 · LODA 1117 · CBLOF 1089 · KNN · OCSVM 1038 · Claude Code +cont 1012 · IForest 1000 · ECOD 963 · AIDE × Sonnet 938 · Claude Code 913 · COPOD · **tabpfn_unsup 888** · PCA 875 · LOF · HBOS 806 |

- 실행 메모: 13:23 에 다른 체크아웃 (이상탐지 코드 없음) 의 kick 이 이 5개와 B 1회째 채점 요청을 옛 코드로 처리해 시도 1/3 실패 (agent: `mlebench/code.tgz` 없음, grade: `KeyError: 'cid'`). 바로 이 worktree 에서 다시 kick 해 정상 처리. forecast 세션에 kick 전 상대 트랙 대기 항목 확인을 요청함
- B 1회째 (13:02, run_id 680939975486579) 는 3분 만에 12개 모두 실패: CUDA 에서 TabPFN 회귀의 raw 공간 bar 분포는 float64 (`_place_raw_space_bardist`, MPS 만 float32) 인데 tabpfn-extensions 가 y 를 logits (float32) 에 맞춰 `ignore_init` 의 `y[mask] = borders[0]` 에서 dtype 오류. 로컬 MPS 확인에서는 안 드러남 → `methods.py` 의 `Regressor` 가 full 출력의 logits 를 borders dtype 으로 맞춤. 13:25 다시 냄 (run_id 773094351672969)
| 2026-11-25 13:25 | B: `tabpfn_unsup` · `tabpfn_unsup+clean` (s1, v2 가중치) | H100 1장 | 47분 SUCCESS (실패 0, landsat 이 가장 김 552초). 첫 제출 13:02 (`ad-12datasets-12splits-20261125-130236`, run_id 680939975486579) + 채점 요청 `anomaly__20261125-130244-497071` |
| – | Agent 파일럿 2: fault (`anomaly-p04`) × 5설정 | | cardio 를 kick · sync 한 뒤 |

## 결정 (2026-11-25)

7·8 의 틀 (벤치 · 설정 · 지표 · 기준점) 은 작업 지시에서, 1~6 은 2026-11-25 설계 초안을 보고 사용자가 정함.

| # | 정할 것 | 정한 것 (추천) | 까닭 |
|---|---|---|---|
| 1 | 데이터셋 | **ADBench Classical 중 규칙 (1) 로 고른 12개** (사용자) | 고전 tabular · FM 비용 · 분할이 깨끗한 것. 규칙이 정해져 있어 바꾸려면 규칙을 고침 |
| 2 | TabPFN 가중치 | **v2** (사용자, 2026-11-25. 처음에는 패키지 기본 v3.5 로 정했다가 같은 날 "3.5 는 쓰지 말자") | Prior Labs License 1.1 (상업 이용 가능, 표시 조항), 동의 · 토큰이 필요 없음. 우리 `tabpfn_v2` (TabArena 트랙의 FM) 와 같은 가중치. v3.5 (비상업 · 받을 때 동의) 는 안 돌림 → "기본 설정" 중 가중치만 패키지 기본과 다름 (순열 10 · `n_estimators` 등은 기본) |
| 3 | Agent 파일럿 | **cardio · fault 의 s1** × 5설정 × 1회 (사용자) | 6 |
| 4 | Agent 의 test 피처 사용 | **train 으로만 fit (설명의 Rules), 지켰는지 사후 검사** (사용자. 추천안은 "막지 않고 기록") | FM · 기존 방법 (train 만 fit) 과 같은 조건 |
| 5 | 설명에 분야 | **안 줌** (사용자) | ADBench 에 열 뜻이 없고, 분야를 주면 알아보기 쉬워짐 ("모델링 실력" 비교, TabArena 결정 3 과 같은 뜻) |
| 6 | 깨끗한 context 진단 | **넣음 (Elo 밖)** (사용자): `tabpfn_unsup+clean` · `iforest+clean` | 질문 "context 의 이상점이 얼마나 해치나" 에 직접 답함. train 라벨은 FM 작업만 읽는 자리에 |
| 7 | 기준점 · 지표 | IsolationForest (pyod 기본) = 1000, AUROC (주) · AUPRC (보조), 오차 1 − AUROC | 작업 지시 |
| 8 | split | ADBench seed 1·2·3 그대로 (FM · 기존 방법 모두), Agent 는 s1 | ADBench 규약. 재현 확인 (2) |

## 확인 못 한 것

- pyod 1.0.0 → 3.6.8 사이 ECOD 가 어떻게 바뀌었는지 (landsat 에서 20점 차)
- TabPFN 비지도 확장의 GPU 시간 (9 의 어림은 재 보지 않은 값)
- ADBench 데이터의 원래 출처별 라이선스 (ADBench 저장소는 BSD-2, 데이터는 각 출처를 따름)
- TACTIC 의 실험 설정 · 결과 (원문에서 "불안정" 까지만 확인)
