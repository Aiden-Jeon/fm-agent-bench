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
- [결과 (12개 × 3 seed, Agent 3개 · s1, 2026-11-25)](#결과-12개--3-seed-agent-3개--s1-2026-11-25)
- [TFM 이 이득을 보는 조건 (2026-11-25 설계, 사용자 확인)](#tfm-이-이득을-보는-조건-2026-11-25-설계-사용자-확인)
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
| 범위 밖 | C1·C2 (Agent 결과를 보고 정함), 결측값·생성·밀도, tabicl 로 chain rule 래퍼 새로 만들기 (후속). Kumo 래퍼는 조건 조사에서 만듦 (2026-11-25, "TFM 이 이득을 보는 조건") |
| **답 (2026-11-25)** | **반쯤 먹힘.** TabPFN 비지도는 IForest 와 Elo 가 같은 수준 (966 대 1000, 구간이 겹침). 4개에서는 11개 중 1등 (Wilt 는 기존 방법이 모두 0.51 이하인데 0.730), 3개 (cover · cardio · PageBlocks) 에서는 꼴찌로 무너짐. 무너지는 까닭은 **context 에 섞인 이상점**: 정상 행만 context 로 주면 12개 모두 오르고 (+0.02 ~ +0.41) 8개에서 기존 최고와 같거나 나음. Agent 는 Codex × GPT 만 뚜렷이 강함 (3개 Agent 풀 Elo 1531, TabPFN 1041) 이지만 3개 중 2개 (cardio · Wilt) 에서 데이터셋을 알아봄 → "결과" |
| **조건 (2026-11-26)** | 통제 실험 ("TFM 이 이득을 보는 조건"): **열 사이 관계가 깨진 이상에서 TFM 이 이기고 (합성 dependency 에서 10 / 10 기존 방법보다 높음), 서로 닮은 이상이 train 에 섞이면 무너짐** (비율보다 test 이상과 닮은 이상이 context 에 있는지, cover 는 0.5% 로도). 이상의 성질을 모르면 IForest 가 기본, 정상 행만 모을 수 있으면 TFM 이 첫 후보 (KNN · LOF 와 구간 겹침). 라벨 없는 context 정리로는 못 고침. Kumo (우리 래퍼) 도 같은 모양 → TFM 일반 쪽 |

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

## 결과 (12개 × 3 seed, Agent 3개 · s1, 2026-11-25)

`uv run --frozen fab ad elo` → `results/anomaly_elo.csv`. 채점은 `results/anomaly.csv` (기존 · FM), `results/anomaly_runs.csv` (Agent).

### FM 대 기존 방법 (seed 3개 평균 AUROC)

순위 = 기존 10개 + `tabpfn_unsup` 11개 중 (평균으로). clean 은 train 의 정상 행만 context (train 라벨을 씀, Elo 밖).

| 데이터셋 | 이상 % | `tabpfn_unsup` | 순위 | IForest | 기존 최고 | `tabpfn+clean` | clean 효과 |
|---|---|---|---|---|---|---|---|
| Wilt | 5.33 | **0.730** | 1 | 0.419 | 0.507 (LOF) | 0.937 | +0.207 |
| annthyroid | 7.42 | **0.883** | 1 | 0.820 | 0.820 (IForest) | 0.949 | +0.066 |
| landsat | 20.71 | **0.603** | 1 | 0.476 | 0.580 (KNN) | 0.779 | +0.176 |
| magic.gamma | 35.16 | **0.839** | 1 | 0.732 | 0.824 (KNN) | 0.884 | +0.044 |
| fault | 34.67 | 0.660 | 3 | 0.570 | 0.730 (KNN) | 0.805 | +0.144 |
| mammography | 2.32 | 0.869 | 4 | 0.864 | 0.908 (ECOD) | 0.888 | +0.019 |
| yeast | 34.16 | 0.418 | 4 | 0.378 | 0.453 (LOF) | 0.457 | +0.040 |
| Waveform | 2.90 | 0.701 | 5 | 0.715 | 0.750 (COPOD) | 0.772 | +0.072 |
| shuttle | 7.15 | 0.968 | 8 | 0.996 | 0.996 (IForest) | 1.000 | +0.032 |
| PageBlocks | 9.46 | 0.732 | 11 | 0.896 | 0.915 (ECOD) | 0.901 | +0.169 |
| cardio | 9.61 | 0.522 | 11 | 0.932 | 0.954 (PCA) | 0.869 | +0.346 |
| cover | 0.96 | 0.521 | 11 | 0.867 | 0.942 (LODA) | 0.927 | +0.406 |

- TabPFN 의 split 사이 표준편차는 대부분 0.01 ~ 0.05, cover 만 0.116 (test 이상 28 ~ 35개). s1 하나로 본 것 ("FM: TabPFN v2 비지도 (s1)") 과 같은 그림
- IForest 보다 7 / 12 에서 나음, 1등 4개, 꼴찌 3개. clean context 는 12개 모두에서 오름 (IForest 의 clean 효과는 −0.01 ~ +0.14). clean TabPFN 은 8 / 12 에서 기존 최고 (오염된 train 으로 fit) 와 같거나 나음 (−0.01 안)
- 가장 크게 다치는 cover 는 이상이 1% 뿐 → 해치는 것은 오염 비율이 아니라 이상 행이 모여 서로를 "정상" 으로 보이게 하는지로 보임 (확인 안 함). 이상이 35% 인 magic.gamma · fault 는 오히려 TabPFN 이 1 · 3등

### Elo (IForest = 1000)

| 풀 | 방법 (Elo, 95% 구간) |
|---|---|
| fm: 12개 × 3 split (36 단위), 부트스트랩 200번 | IForest **1000** (+96 / −88) · ECOD 969 (+182 / −125) · **tabpfn_unsup 966** (+200 / −169) · LODA 946 · KNN 939 · CBLOF 927 · COPOD 919 · PCA 912 · LOF 876 · HBOS 850 · OCSVM 824 |
| agent: cardio · fault · Wilt 의 s1 (3 단위, 구간 없음) | **Codex × GPT 1531** · AIDE × GPT 1129 · Claude Code × Sonnet +cont 1110 · AIDE × Sonnet 1041 · KNN 1041 · **tabpfn_unsup 1041** · Claude Code × Sonnet 1008 · CBLOF 1008 · IForest 1000 · LODA 992 · ECOD 943 · LOF 909 · OCSVM 909 · PCA 901 · COPOD 857 · HBOS 756 |

- fm 풀: TabPFN 은 IForest · ECOD 와 구간이 겹침 → 기존 방법보다 낫다고 말할 수 없음. 이길 때 크게 이기고 질 때 크게 져서 구간이 가장 넓음
- CBLOF (waveform s1) · PCA (cardio s1 · s3) 의 실패 칸은 IForest 값 (imputed 0.028 · 0.056)
- agent 풀은 데이터셋 3개라 순서는 단단하지 않음. Codex 는 세 곳 모두 IForest · TabPFN 보다 높고 (0.954 · 0.722 · 0.801), 나머지 Agent 넷은 섞임

### FM 대 Agent (s1)

| 데이터셋 | TabPFN | Agent 최고 | 기존 최고 | 알아봄 |
|---|---|---|---|---|
| cardio | 0.531 | **0.954** (Codex) | 0.959 (LODA) | Codex: 모양으로 "cardiotocography", ADBench 결과 기억 |
| fault | 0.642 | 0.722 (Codex) | **0.741** (KNN) | 없음 (Codex 는 "UCI arrhythmia" 로 잘못 짐작) |
| Wilt | 0.745 | **0.801** (Codex) | 0.512 (LOF) | Codex: "UCI Wilt dataset", 원래 열 · 벤치 AUC 기억 |

- 질문 (FM 대 Agent) 의 답: 이 세 곳에서는 Codex × GPT 가 TabPFN 보다 모두 나음. 그러나 Codex 가 이긴 곳 중 둘은 데이터셋을 알아본 곳이고, 알아보지 못한 fault 에서는 기존 KNN 보다 낮음 → "Agent 가 모델링으로 이겼다" 로 읽기 어려움. 다른 Agent 넷은 Wilt 에서만 TabPFN 근처 (0.63 ~ 0.74)
- Wilt 에서 Agent 다섯이 모두 기존 방법보다 높은 것은 "밀도로 드문 행" 이 라벨과 반대인 데이터에서, 데이터를 보고 점수 방향 · 변환을 고른 몫 (Codex 는 알아본 몫이 섞임)
- 파일럿 두 곳 (cardio · fault) 은 TabPFN 에 불리한 데이터 (cardio 는 무너지는 셋 중 하나) 였고, Wilt 는 TabPFN 결과를 보고 고른 유리한 데이터 → 세 곳 모두 고른 데이터라 일반화하지 않음

### 후속 (이번 범위 밖)

- context 의 이상점을 줄이는 비지도 방법 (예: 1차 점수로 상위 몇 % 를 context 에서 빼고 다시 점수) — clean 진단이 가리키는 개선 여지. 라벨 없이 되는지 확인할 것 → **"TFM 이 이득을 보는 조건" 의 E3 에서 확인: 거의 못 메움**
- tabicl 로 같은 chain rule 래퍼 (작업 지시의 후속). Kumo 래퍼는 "TFM 이 이득을 보는 조건" 에서 만듦
- Agent 를 알아보기 어려운 데이터 (열을 섞고 값을 바꾼 것, TabArena 의 obfuscate 처럼) 로 다시

## TFM 이 이득을 보는 조건 (2026-11-25 설계, 사용자 확인)

질문: 어떤 조건에서 TabPFN 비지도 (`tabpfn_unsup`, v2 · 위 4 그대로) 가 기존 이상탐지 방법보다 이득을 보나 → 실무 판단 기준 (언제 쓰고 언제 쓰지 말지).
위 "결과" 의 답 (IForest 와 같은 수준, 4개 1등 · 3개 꼴찌, 무너지는 까닭은 context 의 이상점) 을 통제 실험으로 가른다.

- Agent 는 안 돌림. kumo · tabicl 래퍼는 범위 밖 (후속)
- **공식 풀 (`fm` · `agent`, IForest = 1000) 과 기존 결과표 행은 안 바꿈.** 변형 (합성 이상 · 오염 조절 · 정리 · clean) 은 `results/anomaly_conditions*.csv` 와 따로 낸 풀에만. 끝에 `fab ad elo` 로 `results/anomaly_elo.csv` 가 바이트 단위로 같은지 확인
- 읽는 법: E0 ~ E4 = 실험 번호, H = 가설. "1차" = 이번에 돌릴 것, "2차" = 1차 결과를 보고

### 0. 지금 결과로 본 데이터 특성 (조건 5, 로컬, 2026-11-25)

`uv run --frozen fab ad cond traits` → `results/anomaly_conditions_traits.csv` (12개 × 3 split). 새로 돌린 실험 없이, 이미 있는 채점 (`results/anomaly.csv`) 과
분할에서 잰 특성만. 모임 = test 이상 행마다 context (train) 의 가까운 이웃 10개 (MinMax 공간, 유클리드) 중 이상 행 비율의 평균, 괄호는 그것 ÷ 이상 비율 (1 = 무작위로 흩어짐).
**모임은 train 라벨로 잰 진단값** (방법의 입력이 아님). 범주 열 = tabpfn-extensions 규칙 (4), 괄호 = 그중 값이 셋 이상이라 `astype(int)` 에서 뭉개지는 열.
기존 최고 = split 마다 기존 10개 중 최고의 평균 (위 표의 "평균의 최고" 와 조금 다름). split 3개 평균, `TabPFN − 최고` 순.

| 데이터셋 | 이상 % | 피처 | 범주 열 | 같은 행 % | 모임 (배) | 기존 중앙값 | 기존 최고 | TabPFN | TabPFN − 최고 | 순위 (11개) | clean 효과 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Wilt | 5.3 | 5 | 0 | 0.0 | 0.20 (3.8) | 0.365 | 0.507 | 0.730 | +0.223 | 1.0 | +0.207 |
| annthyroid | 7.4 | 6 | 0 | 2.4 | 0.31 (4.2) | 0.684 | 0.820 | 0.883 | +0.062 | 1.0 | +0.066 |
| magic.gamma | 35.3 | 10 | 0 | 0.5 | 0.62 (1.7) | 0.697 | 0.824 | 0.839 | +0.015 | 1.0 | +0.044 |
| landsat | 20.7 | 36 | 0 | 0.0 | 0.74 (3.6) | 0.451 | 0.598 | 0.603 | +0.005 | 1.3 | +0.176 |
| shuttle | 6.9 | 9 | 0 | 0.0 | 0.94 (13.6) | 0.986 | 0.996 | 0.968 | −0.028 | 8.7 | +0.032 |
| mammography | 2.3 | 6 | 0 | 30.0 | 0.47 (20.3) | 0.861 | 0.910 | 0.870 | −0.040 | 5.3 | +0.019 |
| yeast | 34.2 | 8 | 2 (1) | 3.0 | 0.49 (1.4) | 0.396 | 0.461 | 0.418 | −0.043 | 4.3 | +0.040 |
| fault | 34.7 | 27 | 4 (2) | 0.0 | 0.51 (1.5) | 0.540 | 0.730 | 0.660 | −0.070 | 2.7 | +0.144 |
| Waveform | 2.9 | 21 | 0 | 0.0 | 0.24 (8.3) | 0.682 | 0.780 | 0.701 | −0.080 | 5.7 | +0.072 |
| PageBlocks | 9.5 | 10 | 0 | 0.0 | 0.61 (6.5) | 0.880 | 0.915 | 0.732 | −0.183 | 10.7 | +0.169 |
| cardio | 9.6 | 21 | 4 (4) | 0.8 | 0.73 (7.7) | 0.917 | 0.945 | 0.522 | −0.423 | 10.3 | +0.347 |
| cover | 1.0 | 10 | 0 | 0.0 | 0.73 (71.9) | 0.895 | 0.946 | 0.521 | −0.424 | 11.0 | +0.406 |

순위 상관 (Spearman, 데이터셋 12개 · split 평균. p 는 참고만: 12개 · 여러 번 봄)

| 특성 | `TabPFN − IForest` | `TabPFN − 최고` | clean 효과 |
|---|---|---|---|
| 기존 중앙값 (기존 방법이 잘 되는 데이터인지) | **−0.80** (p 0.002) | −0.48 | −0.02 |
| 모임 (배) | **−0.67** (p 0.017) | −0.40 | +0.08 |
| 모임 (비율) | −0.34 | — | +0.15 |
| 이상 % | +0.39 | +0.19 | −0.11 |
| 피처 수 | −0.13 | −0.48 | +0.39 |
| 행 수 (train) | −0.01 | +0.26 | −0.26 |
| 범주 열 수 | −0.25 | −0.32 | −0.04 |
| 같은 행 % | −0.04 | +0.15 | −0.51 |

읽은 것 (12개라 판단은 약함, 아래 실험의 우선순위를 정하는 데만 씀)

- **TabPFN 은 기존 방법이 안 되는 데이터에서 상대적으로 이김**: 기존 중앙값이 0.5 근처 이하 (Wilt · landsat · yeast) 이거나 중간 (annthyroid · magic.gamma) 이면 1 ~ 4등,
  기존 방법이 0.88 넘게 잘 되는 데이터 (cover · cardio · PageBlocks · shuttle) 에서 8 ~ 11등. 라벨의 이상이 "밀도로 드문 행" 이 아닐 때 열 사이 조건부 관계로 보는 TabPFN 이 덜 틀림 (H1a 의 단서)
- **이상이 모인 데이터에서 IForest 에 짐** (모임 배수와 −0.67): cover 는 test 이상의 context 이웃 중 73% 가 이상 (무작위의 72배), cardio 7.7배. 그러나 같은 모임 비율 (0.73 ~ 0.74) 인 landsat 에서는 이김,
  shuttle (0.94) 은 −0.03 뿐 → 모임 하나로는 설명이 안 됨. 이상 비율은 상관이 약하고 방향도 반대 (+0.39) → "비율보다 모임" 쪽 단서지만, 모임 · 기존 방법이 잘 되는지 · 이상 종류가 서로 얽혀 있어 가를 수 없음 → **E1 · E2 로 통제**
- **clean 효과는 이 특성들로 거의 예측이 안 됨**: Wilt 는 모임이 0.20 으로 가장 흩어졌는데 +0.21. 오염이 해치는 정도는 데이터마다 다르고 특성으로는 안 보임 → 라벨 없이 쓸 수 있는 정리 (E3) 가 실무에 더 직접적
- 피처 수는 `TabPFN − 최고` 와 −0.48 (넓을수록 불리 쪽) 이지만 cardio · Waveform · fault 가 21 ~ 27 피처에 겹쳐 얽힘. 범주 열 뭉개짐 (4) 은 cardio · fault · yeast 셋뿐이라 따로 볼 수 없음 → 이번에는 안 다룸 (남은 것)
- cardio 의 뭉개지는 범주 열은 다시 세면 4개 (값 5 · 18 · 6 · 3개, s1). 위 4 의 "3열" 은 고치지 않고 여기 적음

우선순위: E2 (오염 비율 × 모임) 와 E3 (라벨 없는 정리) 가 위 단서를 직접 가름 → 1차에 넣음. E1 (이상 종류) 은 H1a 를 통제된 데이터로 보는 것 → 1차.
E4 (정상 데이터가 있는 설정) 는 CPU 만 들고 공정 비교의 빈칸 → 1차. 데이터셋 늘리기 (조건 5) 는 1차 결과를 보고 2차.

### 실험 설계 (2026-11-25, 사용자 확인)

공통: 지표 AUROC (주) · AUPRC (보조). 같은 test 행에서 방법끼리 비교. 기존 방법 = pyod 기본 10개 (3). TabPFN = `tabpfn_unsup` (v2, 순열 10 — E0 를 통과하면 순열 3).
결과는 `results/anomaly_conditions.csv` (행 = 실험 · 데이터셋 · split · 조건 · 방법), 조건별 Elo 는 `results/anomaly_conditions_elo.csv` (풀마다 IForest 또는 IForest+clean = 1000).

| # | 무엇 | 데이터 · 조건 격자 | 방법 | seed | 가설 |
|---|---|---|---|---|---|
| E0 | 순열 수 줄여도 되는지 | 12개 × s1, 순열 3 (`tabpfn_unsup_p3`) 대 이미 있는 순열 10 | TabPFN | s1 | 통과 기준: 12개 중 10개 이상에서 \|차\| ≤ 0.01, 나머지도 ≤ 0.02, 11개 중 순위가 1 넘게 안 바뀜. 통과하면 E1 ~ E3 의 TabPFN 은 순열 3 (비용 ×0.3), 아니면 10 |
| E1 | 이상 종류 | ADBench `generate_realistic_synthetic` 그대로 (아래), 4종 (local · global · dependency · cluster) × 12개 | 기존 10 + TabPFN + `tabpfn+clean` · `iforest+clean` | 생성 · 분할 seed 1 (1차), 2차에 2 · 3 | H1a: dependency 에서 TabPFN 이 기존 최고 이상 (열 사이 관계를 봄). H1b: cluster 에서 TabPFN 이 가장 약하고 clean 으로 가장 크게 오름 (context 의 이상 무리가 서로를 정상으로 만듦). 참고: ADBench 그림 5 는 local 에 LOF, global 에 KNN 이 최고, cluster 에서 대부분 약함 |
| E2 | context 오염 비율 × 모임 | 실제 데이터 6개 (cover · cardio · PageBlocks = 무너짐, Wilt · annthyroid · magic.gamma = 이김) 의 s1. test 는 그대로, context = train 정상 행 전부 + train 이상 n_a 행. 비율 r = n_a / context ∈ {0, 0.5, 1, 2, 5, 10, 20}% (train 이상의 80% 안에서 되는 칸만), 배치 2: **흩어짐** (train 이상에서 무작위) · **모임** (무작위 이상 하나를 중심으로 train 이상 중 가까운 n_a 개). 뽑기 seed 2개 | 기존 10 + TabPFN | 분할 s1, 뽑기 2 | H2: 같은 r 에서 모임 < 흩어짐 (TabPFN), 그 차이가 r 의 효과보다 큼. 기존 방법 (특히 IForest) 은 덜 다침 |
| E3 | 라벨 없는 context 정리 | 12개 × s1 | `tabpfn_unsup+trim{k}`: train 을 2겹으로 나눠 서로를 context 로 TabPFN 점수 → 상위 k% 를 빼고 남은 train 으로 test 점수. `+iftrim{k}`: 빼는 기준을 IForest (train fit, train 점수). `+trim10x2`: 2번 반복. k ∈ {5, 10, 20}. 대조: `iforest+iftrim10` (같은 정리를 기존 방법에) | s1 | H3: 라벨 없는 정리가 clean 과의 차이를 절반 이상 메움 (특히 cover · cardio · PageBlocks). **주 지표는 미리 정한 k = 10** (결과를 보고 k 를 고르면 라벨을 쓴 셈). 나머지 k 는 민감도. 비지도 조건이라 기존 10개와 한 풀 (`trim` 풀) |
| E4 | 정상 데이터가 있는 설정 (semi-supervised) 공정 비교 | 공식 12개 × s1 ~ s3 | 기존 10개 모두 `+clean` (지금은 `iforest+clean` 만) 대 `tabpfn_unsup+clean` (있음) | s1 ~ s3 | H4: clean 끼리 비교하면 TabPFN 이 기존 최고 근처 이상 (지금 단서: 8 / 12). `clean` 풀 Elo (IForest+clean = 1000) |

E1 의 생성 (ADBench `data_generator.py` 그대로, 출처 표의 커밋): `set_seed(seed)` → 1만 행 넘으면 다시 `set_seed` · 1만 행 표본 → 원래 이상은 버리고 **정상 행으로 생성 모형을 적합해 정상 · 이상 둘 다 새로 뽑음** (작업 지시의 "정상 행에 심음" 과 달리 ADBench 는 정상도 생성 모형에서 뽑음 → 코드를 따름). 정상 · 이상 행 수는 원래와 같음.
local = GMM (성분 수 1 ~ 9 중 BIC 최소, `random_state=seed`) 에서 정상, 공분산 × α (5) 로 이상 · cluster = 평균 × α (5) · global = 각 열 [min × 1.1, max × 1.1] 균등 (`percentage=0.1`) ·
dependency = `copulas.VineCopula('center')` 로 정상, 열마다 `GaussianKDE` 로 독립 이상 (피처 50 넘으면 50개 표본, 우리 12개는 모두 36 이하). 그다음 층화 70/30 · train 으로 MinMax (2 와 같음).
dependency 는 `copulas` 패키지가 필요 (옆 가상환경 · 생성은 로컬). 뺀 칸 2개 (2026-11-25 로컬):

- **landsat**: VineCopula 표본이 느림. 잰 값: annthyroid (6피처 · 정상 6,666행) 표본 366초, Wilt (5 · 4,562) 150초, yeast (8 · 977) 14초 → 피처 쌍 수 × 행 수에 비례로 보면 36피처 landsat 은 3시간 넘음
- **PageBlocks**: copulas 표본에서 `ValueError: f(a) and f(b) must have different signs` (scipy `brentq`, KDE 역함수). ADBench 원문 주석도 "we found that copula function may occur error in some datasets"

나머지 10개는 6 프로세스로 약 20분. copulas 0.15.0 은 옆 가상환경 (`FAB_ANOMALY/.venv`) 에 설치했고 그때 pandas 가 3.1.1 → 2.3.3 으로 바뀜 (tabpfn · pyod · copulas import 는 확인. 이 저장소의 uv.lock 은 그대로).
local · global · cluster 36개는 copulas 없이 13초. 형식 확인 (로컬, 결과표 밖): ADBench 그림 5 와 같은 방향 (local 은 LOF, global 은 KNN 이 셋 중 최고, cluster 에서 LOF 0.52).

E2 의 r = 0 은 clean (`tabpfn_unsup+clean`) 과 같은 context. 원래 비율 칸은 이미 있는 `tabpfn_unsup` (s1). cover 는 train 이상이 68개라 r 0.5% 만 (모임 · 흩어짐 대조 1칸) → 대신 원래 비율의 무작위 · 모임은 이미 있음.
MinMax 는 s1 그대로 (칸마다 다시 맞추지 않음 → 척도가 칸 사이에 같음). 칸은 94개 (cover 5 · cardio 17 · PageBlocks 17 · Wilt 13 · annthyroid 17 · magic.gamma 25).

조작 확인 (`results/anomaly_conditions_e2_cells.csv`, train 라벨로 잰 진단값. 칸 평균, r > 0):

| 데이터셋 | context 이상끼리 평균 거리 (모임 / 흩어짐) | test 이상의 가림 (모임 / 흩어짐) |
|---|---|---|
| cover | 0.368 / 0.552 | 0.531 / 0.560 |
| cardio | 0.867 / 1.356 | 0.286 / 0.337 |
| PageBlocks | 0.355 / 0.774 | 0.147 / 0.245 |
| Wilt | 0.067 / 0.131 | 0.050 / 0.047 |
| annthyroid | 0.173 / 0.334 | 0.145 / 0.145 |
| magic.gamma | 0.340 / 0.728 | 0.152 / 0.231 |

가림 = test 이상 행마다 context 의 가까운 이웃 10개 중 이상 비율. **모임 배치는 context 이상끼리의 거리를 절반쯤으로 줄이지만, 한 곳에만 있어 test 이상을 가리는 정도는 같거나 작음.**
→ H2 는 배치 (이름) 만이 아니라 두 측정값 (모인 정도 · 가림) 에 대한 TabPFN AUROC 로도 봄. "모임이 해친다" 가 "context 안에 test 이상과 닮은 이상이 있으면 해친다 (가림)" 인지 "이상끼리 뭉쳐 있으면 해친다" 인지를 가름.

### 비용 어림 (TabPFN, H100 1장, 잰 값 기준)

잰 값: s1 12개 `tabpfn_unsup` 23분 (데이터셋마다 25초 Wilt ~ 552초 landsat), `+clean` 20분 (results/results.csv 의 seconds). 순열 3 은 그 0.3배로 어림 (E0 에서 잼). 노드 대기 · 설치는 따로 (지금까지 작업마다 10 ~ 60분).

| 실험 | 단위 | 순열 10 | 순열 3 |
|---|---|---|---|
| E0 | s1 12개 1번 | — | 약 7분 |
| E1 | 4종 × (`tabpfn_unsup` + `+clean`) × 12개 × seed 1 | 약 2.9시간 | 약 0.9시간 |
| E2 | 6개 · 칸 98개 (cover 5 · cardio 17 · PageBlocks 17 · Wilt 13 · annthyroid 17 · magic.gamma 25) | 약 1.9시간 | 약 0.6시간 |
| E3 | 12개 × (2겹 train 점수 약 2배 + 다시 점수 k 3개 + iftrim 3개 + 반복 1) | 약 3.8시간 | 약 1.2시간 |
| E4 | CPU 작업 하나 (기존 10개 `+clean` × 36) | — | 약 20분 (CPU) |
| **1차 합** | | **약 8.6 GPU 시간** | **약 2.8 GPU 시간 + E0** |

- 기존 방법 (pyod) 은 모든 실험에서 CPU 작업 하나씩 (작업마다 20 ~ 40분 어림)
- 격자가 큰 2차 (E1 의 seed 2 · 3, 데이터셋 늘리기) 는 1차 결과로 정함

### 연동

- 코드: `fab/bench/anomaly/conditions.py` (`fab ad cond …`): 생성 (E1) · context 칸 (E2) 은 **로컬에서 대회 폴더로** 만듦 (`competitions/adc-<실험>-<…>/`, git 제외, 설정에 `anomaly.condition`). 정답 · train 라벨은 지금처럼 `FAB_ANOMALY` 와 내 Volume 에만
- 정리 (E3) · 순열 (E0) 은 방법 이름 (`tabpfn_unsup+trim10`, `tabpfn_unsup_p3` 등) 으로 `methods.py` · `run.py` 에 더함. 실행 · 채점 경로는 지금 것 그대로 (`fab databricks` 실행기 → 정리 작업 `grade bench=anomaly`)
- **조건 실험의 kick 은 머지 전 이 worktree (`Aiden-Jeon/anomaly-tfm-conditions`) 에서만.** 정리 작업 코드 (`fab/code.tgz`) 는 로컬 명령을 낸 체크아웃의 것으로 다시 올라가고, 이 브랜치 전 코드는 조건 대회의 채점 행을 공식 `anomaly.csv` 에 넣음 (`anomaly_conditions.csv` 를 모름)
- **채점 행은 설정에 `condition` 이 있는 대회이거나 변형 방법이면 `results/anomaly_conditions.csv` 로** (공식 `anomaly.csv` 에 안 섞임). E4 의 `<기존>+clean` 은 공식 대회의 기존 `+clean` 과 같은 종류라 지금처럼 `anomaly.csv` 에 행을 더함 (기존 행은 그대로, fm 풀은 `+clean` 을 빼므로 Elo 그대로) — 결정 9
- 정리 작업의 채점 경로가 바뀌므로 **머지 전 kick 은 이 worktree 에서만** (운영 주의, 8)

### E0 결과: 순열 3 대 10 (2026-11-25, s1)

`results/anomaly_conditions.csv` 의 `tabpfn_unsup_p3` 대 공식 `tabpfn_unsup` (같은 split · 같은 seed, 순열만 3 대 10). 순위 = 기존 10개 + TabPFN 11개 중.

| 데이터셋 | 순열 10 | 순열 3 | 차 | 순위 (10 → 3) |
|---|---|---|---|---|
| annthyroid | 0.8829 | 0.8858 | +0.003 | 1 → 1 |
| cardio | 0.5310 | 0.5225 | −0.009 | 10 → 10 |
| cover | 0.3999 | 0.4330 | **+0.033** | 11 → 11 |
| fault | 0.6424 | 0.6427 | +0.000 | 3 → 3 |
| landsat | 0.6073 | 0.6075 | +0.000 | 1 → 1 |
| magic.gamma | 0.8490 | 0.8498 | +0.001 | 1 → 1 |
| mammography | 0.8797 | 0.8769 | −0.003 | 5 → 6 |
| PageBlocks | 0.7320 | 0.7566 | **+0.025** | 11 → 11 |
| shuttle | 0.9794 | 0.9771 | −0.002 | 9 → 9 |
| Waveform | 0.7602 | 0.7613 | +0.001 | 3 → 3 |
| Wilt | 0.7450 | 0.7444 | −0.001 | 1 → 1 |
| yeast | 0.4332 | 0.4343 | +0.001 | 3 → 3 |

- 미리 정한 기준 (10 / 12 이상 \|차\| ≤ 0.01, 나머지 ≤ 0.02, 순위 1 넘게 안 바뀜) 중 **"나머지 ≤ 0.02" 를 못 넘음** (cover +0.033, PageBlocks +0.025) → 기준대로면 통과 못 함
- 크게 다른 둘은 모두 context 오염으로 무너지는 데이터 (11등은 그대로). 순열 3 이 조금 덜 무너짐. 나머지 10개는 ±0.01 안, 순위는 mammography 만 5 → 6
- GPU 시간: 순열 3 은 s1 12개에 8.3분 (순열 10 의 23분의 0.36배, 노드 대기 · 설치 제외)

### E4 결과: 정상 데이터가 있는 설정의 공정 비교 (12개 × 3 split, 2026-11-25)

기존 10개 모두 `+clean` (train 의 정상 행만으로 fit) 대 `tabpfn_unsup+clean`. 채점 행은 `results/anomaly.csv` (결정 9, 기존 행 · `anomaly_elo.csv` 그대로 확인).
`clean` 풀 Elo (IForest+clean = 1000, 부트스트랩 200번, `results/anomaly_conditions_elo.csv`):

| 방법 | Elo (95% 구간) |
|---|---|
| **tabpfn_unsup+clean** | **1151** (+217 / −138) |
| knn+clean | 1115 (+94 / −85) |
| lof+clean | 1097 (+88 / −78) |
| iforest+clean · pca+clean | 1000 |
| ecod · cblof · loda · copod · hbos · ocsvm (+clean) | 957 · 904 · 883 · 871 · 795 · 731 |

split 평균 AUROC, 기존 최고 = 기존 10개 `+clean` 중

| 데이터셋 | `tabpfn+clean` | 기존 최고 (+clean) | 차 | 순위 (11개) |
|---|---|---|---|---|
| Wilt | 0.937 | 0.537 (LOF) | +0.400 | 1 |
| annthyroid | 0.949 | 0.922 (IForest) | +0.028 | 1 |
| magic.gamma | 0.884 | 0.860 (KNN) | +0.024 | 1 |
| fault | 0.805 | 0.798 (KNN) | +0.007 | 1 |
| landsat | 0.779 | 0.773 (KNN) | +0.007 | 1 |
| shuttle | 1.000 | 0.996 (IForest) | +0.004 | 1 |
| Waveform | 0.772 | 0.768 (LOF) | +0.004 | 1 |
| yeast | 0.457 | 0.471 (CBLOF) | −0.014 | 3 |
| mammography | 0.888 | 0.911 (ECOD) | −0.023 | 4 |
| PageBlocks | 0.901 | 0.934 (PCA) | −0.033 | 6 |
| cover | 0.927 | 0.993 (LOF) | −0.066 | 7 |
| cardio | 0.869 | 0.971 (PCA) | −0.103 | 10 |

- **H4 는 반쯤 맞음**: 정상 데이터가 있으면 TabPFN 이 Elo 1등이고 12개 중 7개에서 1등이지만, KNN · LOF 와 구간이 겹침. 위 "결과" 의 "clean TabPFN 은 8 / 12 에서 기존 최고 (오염된 train 으로 fit) 와 같거나 나음" 은 기존 방법에는 clean 을 안 준 비교였음 → 기존 방법도 clean 을 받으면 이웃 기반 방법이 크게 오름. 36 단위 평균 AUROC (오염 → clean): TabPFN 0.704 → 0.847 (+0.143) · LOF 0.649 → 0.791 (+0.142) · KNN 0.704 → 0.803 (+0.099) · IForest 0.722 → 0.767 (+0.045) · PCA 0.675 → 0.722 · ECOD 0.695 → 0.714.
  **오염에 가장 약한 것은 TabPFN 만이 아니라 context (이웃) 로 밀도를 매기는 방법 모두** (LOF 도 TabPFN 만큼 다침), IForest · ECOD 같은 분할 · 주변분포 방법은 덜 다침
- clean 끼리도 cover · cardio · PageBlocks 는 짐 → 그 셋에서 TabPFN 이 무너지는 것은 오염만이 아님 (cardio 는 범주 열 4개가 뭉개지는 데이터, 4)
- 빠진 칸: `pca+clean` 이 cardio s2 에서 NaN (기존 PCA 와 같은 까닭) → Elo 에서 IForest+clean 값

### E1 결과: 이상 종류 (ADBench 합성, seed 1, 2026-11-26)

데이터셋 평균 AUROC (local · global · cluster 12개, dependency 10개). 기존 최고 = 오염된 train 으로 fit 한 기존 10개 중 평균이 가장 높은 것.

| 종류 | `tabpfn_unsup` | `tabpfn+clean` | `kumo_unsup` | `kumo+clean` | IForest | KNN | LOF | 기존 최고 | Elo 위 (12개 풀: 기존 10 + 두 FM, IForest = 1000) |
|---|---|---|---|---|---|---|---|---|---|
| local (공분산 × 5) | 0.896 | 0.911 | 0.886 | 0.910 | 0.847 | 0.879 | **0.921** | LOF 0.921 | LOF 1404 · TabPFN 1242 · KNN 1199 · Kumo 1166 |
| global (균등, 범위 × 1.1) | 0.999 | 1.000 | 0.999 | 0.999 | 0.998 | **1.000** | 0.854 | KNN 1.000 | KNN 1230 · TabPFN 1202 · Kumo 1193 |
| **dependency** (열 독립, KDE) | **0.986** | 0.992 | **0.992** | 0.995 | 0.782 | 0.922 | 0.872 | KNN 0.922 | **TabPFN 4194 · Kumo 4123** · KNN 1505 (TabPFN 은 기존 10개보다 10 / 10 에서 높음) |
| **cluster** (평균 × 5) | **0.776** | **1.000** | **0.739** | **1.000** | 0.989 | 0.794 | 0.439 | OCSVM 0.995 | OCSVM 1158 · IForest 1000 … **TabPFN 375 (평균 순위 9.2) · Kumo 39 (10.7)** · LOF 맨 아래 |

- **H1a 확인**: 열 사이 관계만 깬 dependency 이상에서 TabPFN 이 10개 모두 기존 10개보다 높음 (0.986, 기존 최고 KNN 0.922). 주변분포 · 분할 방법 (ECOD · COPOD · HBOS · PCA 0.62 ~ 0.66, IForest 0.78) 은 거의 못 잡음
- **H1b 확인**: 멀리 떨어진 무리 (cluster) 는 IForest · OCSVM 이 0.99 로 거의 다 잡는데 TabPFN 은 0.776. clean context 면 1.000 → 무너지는 까닭은 **context 에 같은 무리가 있어 서로를 "정상" 으로 만듦**. 같은 일이 이웃 기반 KNN (0.794 → 0.998) · LOF (0.439 → 0.998) 에도 남 (ADBench 그림 5d 와 같은 방향)
- **Kumo (우리 래퍼) 도 같음**: dependency 0.992 (TabPFN 과 함께 Elo 1 · 2위), cluster 0.739 (평균 순위 10.7 / 12) → clean 1.000, local 0.886 · global 0.999
- local 은 LOF 다음 2등, global 은 모두 거의 1.0. Elo 는 거의 완벽히 갈리는 단위가 많아 크기가 부풂 (dependency · cluster) → 순위 · AUROC 로 읽음
- 뺀 칸: dependency 의 landsat · PageBlocks (위 생성 절). 실패: CBLOF 가 cluster 의 cardio · Waveform 에서 클러스터 실패 (Elo 에서 IForest 값)

### E2 결과: context 오염 비율 × 배치 (실제 데이터 6개 · s1, 뽑기 2, 2026-11-26)

`tabpfn_unsup` 의 뽑기 평균 AUROC (r = 0 은 정상만 = clean context). 괄호는 같은 칸의 IForest.

| 데이터셋 | r = 0 | 0.5% 모임 / 흩어짐 | 1% | 2% | 5% | 10% · 20% |
|---|---|---|---|---|---|---|
| cover | 0.889 (0.862) | **0.524 / 0.479** (0.842 / 0.875) | — | — | — | — |
| cardio | 0.860 (0.954) | 0.853 / 0.819 | 0.846 / 0.802 | 0.794 / 0.747 | **0.694 / 0.648** (0.953 / 0.956) | — |
| PageBlocks | 0.898 (0.924) | 0.890 / 0.891 | 0.883 / 0.876 | 0.873 / 0.850 | 0.823 / 0.793 (0.918 / 0.920) | — |
| Wilt | 0.949 (0.459) | 0.910 / 0.910 | 0.881 / 0.878 | 0.841 / 0.833 (0.391 / 0.412) | — | — |
| annthyroid | 0.949 (0.928) | 0.944 / 0.939 | 0.938 / 0.938 | 0.928 / 0.923 | 0.899 / 0.905 (0.856 / 0.830) | — |
| magic.gamma | 0.893 (0.776) | 0.891 / 0.891 | 0.890 / 0.889 | 0.888 / 0.888 | 0.882 / 0.880 | 0.875 / 0.873 · 0.860 / 0.865 (0.725 / 0.742) |

칸 88개 (r > 0) 에서 r = 0 대비 떨어진 AUROC

| 방법 | 평균 하락 (모임 / 흩어짐) | 하락 ~ 가림 (Spearman) | 하락 ~ 비율 |
|---|---|---|---|
| **tabpfn_unsup** | **0.051 / 0.064** | 0.52 | 0.27 |
| LOF | 0.027 / 0.024 | 0.87 | 0.69 |
| IForest | 0.021 / 0.014 | 0.11 | 0.39 |
| KNN | 0.016 / 0.015 | 0.73 | 0.59 |
| ECOD · LODA | 0.006 · 0.006 | 0.67 · 0.32 | 0.55 · 0.53 |

- **TabPFN 이 오염에 가장 민감**: 같은 칸에서 평균 하락이 IForest 의 3 ~ 4배. cover 는 이상 35행 (context 의 0.5%) 만으로 0.889 → 0.5 근처 (IForest 는 그대로)
- **H2 ("같은 비율이면 모임이 더 해침") 는 반증**: 모임 배치 (context 이상끼리 거리 절반) 보다 흩어짐 배치가 더 해침 (cardio · PageBlocks · cover 에서 0.03 ~ 0.05 더). 하락은 비율 (0.27) 보다 **가림** (test 이상의 context 이웃 중 이상 비율, 0.52) 과 더 맞음
- 고친 해석: 해치는 것은 "이상이 뭉쳐 있음" 이 아니라 **test 의 이상과 닮은 이상이 context 에 있음**. 원래 cover 는 이상이 한 무리 (위 0 의 모임 배수 72) 라 test 이상을 어떤 표본이든 가림 → 비율 1% 로도 무너짐. 이상이 서로 닮지 않은 데이터 (magic.gamma) 는 20% 에서도 −0.03
- 비율은 여전히 효과가 있음 (같은 배치에서 r 이 늘면 단조로 내려감). 비율 · 가림이 같이 움직여 둘을 완전히 가르지는 못함

### E3 결과: 라벨 없는 context 정리 (12개 × s1, 2026-11-26)

메운 몫 = (정리 − 오염) / (clean − 오염). 주 지표는 미리 정한 `+trim10` (결정 12).

| 데이터셋 | 오염 | clean | `+trim10` | `+trim10x2` | `+iftrim10` | `+iftrim20` |
|---|---|---|---|---|---|---|
| cover | 0.400 | 0.889 | 0.431 (0.06) | 0.426 (0.05) | 0.508 (0.22) | 0.603 (0.42) |
| cardio | 0.531 | 0.860 | 0.578 (0.14) | 실패 | 실패 | 실패 |
| PageBlocks | 0.732 | 0.898 | 0.744 (0.07) | 0.765 (0.20) | 0.781 (0.30) | 0.823 (0.55) |
| Wilt | 0.745 | 0.949 | 0.735 (−0.05) | 0.723 (−0.11) | 0.723 (−0.11) | 0.682 (−0.31) |
| fault | 0.642 | 0.796 | 0.614 (−0.19) | 0.599 (−0.29) | 0.603 (−0.25) | 0.594 (−0.32) |
| landsat | 0.607 | 0.782 | 0.603 (−0.03) | 0.600 (−0.04) | 0.586 (−0.12) | 0.583 (−0.14) |
| 12개 중앙값 | | | **0.07** | 0.20 | −0.01 | 0.06 |

- **H3 반증**: 주 지표 `+trim10` 은 clean 과의 차이를 중앙값 7% 만 메움 (가장 나은 `+trim10x2` 도 20%). `trim` 풀 Elo 도 `tabpfn_unsup` 1004 → `+trim10` 1031 (구간이 겹침)
- IForest 로 고른 정리 (`+iftrim20`) 는 무너지는 데이터 (cover · PageBlocks) 에서 차이의 42 ~ 55% 를 메우지만, **TabPFN 이 이기는 데이터 (Wilt · fault · landsat) 에서는 내려감**: 그곳의 이상은 밀도로 드물지 않아 IForest 가 정상을 빼 버림 → 어느 데이터인지 라벨 없이 모르면 쓸 수 없음
- 실패 칸: 정리 뒤 context 에서 조건 열이 상수가 되어 TabPFN 이 거부 (`All features are constant`, cardio 4 · yeast 2). Elo 에서는 정리 안 한 같은 방법의 값으로 채움 (`fill_trim`, IForest 값으로 채우면 cardio 에서 정리 변형이 부풂)
- 기존 방법에 같은 정리 (`<기존>+trim10`) 는 대부분 ±0.02 (IForest 1000 → 989)

### Kumo (2026-11-25 범위 추가, 사용자)

질문의 몫: 위 조건들이 TabPFN 에만 해당하는지, TFM 일반인지 (Kumo 에서도 같은 조건에서 이득 · 손해가 나는지). tabicl 은 그대로 범위 밖.

| 항목 | 정한 것 | 까닭 · 확인 |
|---|---|---|
| 공식 API | **없음**: `structured-data-models` 0.2.1 (import `sdm`, docker/air 와 같은 버전) 의 휠 소스에 이상탐지 · 비지도 · 밀도 · 생성 API 가 없음. tabpfn-extensions 에서 가져온 것은 ECOC (클래스가 많은 분류, `sdm/models/ecoc.py`) 뿐. `KumoTabular` 회귀는 999 분위수 (`q001` … `q999`) 를 냄 (`sdm/models/kumo/tabular/model.py`) | 2026-11-25 휠을 받아 소스만 읽음 |
| 이름 | `kumo_unsup` (clean 진단 `kumo_unsup+clean`) — **우리 래퍼** (공식 구현 아님), 결과 표 옆에 적음 | `methods.py` 의 `kumo_scores` |
| chain rule | tabpfn-extensions 0.6.3 의 `outliers()` 그대로: 순열 10 (원문 `efficient_random_permutation`, 같은 seed → TabPFN 과 같은 순열), 첫 열은 정규 잡음 한 열을 입력, 열마다 log 밀도를 더하고 순열마다의 밀도를 산술 평균 (log-sum-exp) | 같은 분해를 두 FM 이 쓰게 |
| 범주 열 | 판정 규칙도 원문 그대로 (값 ≤ 40 · 값마다 평균 10행 넘음, 값 수 ≤ 10 이면 분류). 클래스는 원문처럼 `astype(int)` (MinMax 된 값이라 0 · 1 로 뭉개지는 것까지 같게, 위 4). 분류 = 실제 클래스의 `predict` 확률, 모르는 값은 0.1 (원문과 같음) | 원문의 한계도 같게 둬야 FM 끼리 비교가 됨 |
| 연속 열 밀도 | 999 분위수 (정렬) → CDF: 분위수 사이 선형, q001 아래 · q999 위는 질량 0.001 의 지수 꼬리 (척도 = 가장자리 10칸 폭 / 10, 경계에서 밀도가 이어짐, 최소 h). 밀도 = (F(y + h) − F(y − h)) / 2h, **h = 0.005** (MinMax 단위). 같은 값이 많아 분위수가 겹쳐도 유한 | 분위수 → 밀도 근사 (결정 13) |
| 회귀 타깃 | context 의 평균 · 표준편차로 표준화해 넘기고 분위수를 되돌림 (`fab/models/tfm.py` 의 `_Transductive` 와 같음). 앙상블 8 (`n_estimators`, 우리 `kumo` 기본), CUDA 에서 float16 autocast, 질의 2,048행씩 | |

근사 확인 (로컬 CPU, 2026-11-25)

- 분위수 → 밀도 (참 분위수를 넣어): 안쪽 (0.1% ~ 99.9%) 의 \|log 밀도 오차\| 중앙값 N(0.5, 0.1) 0.000 · N(0.5, 0.02) 0.009 · Beta(2, 5) 0.000 (h = 0.005). h = 0.02 면 좁은 분포에서 0.128 로 커짐 → 0.005 로. 꼬리 밖은 지수 꼬리라 값은 참과 다르지만 순위는 같음 (Spearman 1.000). 같은 값 (점 질량 50%) 도 유한
- 래퍼 전체 (상관 0.9 인 2차원 정규 + 0/1 범주 열, context 400 · test 200, 이상 20 = 상관을 깬 행, 순열 2): `kumo_unsup` 의 −log 밀도와 참 −log 밀도 (연속 두 열) 의 Spearman 0.901, AUROC 0.946. 같은 데이터의 `tabpfn_unsup_p3` 0.901 · 0.946 (점수끼리 상관 0.85: 다른 점수인데 순위 지표가 비슷). CPU 36초 (TabPFN 4초)

K0 결과 (공식 12개 × s1, 2026-11-26 00:07 끝, H100 1장 77분: `kumo_unsup` 38분 · `+clean` 36분 = TabPFN 의 약 1.7배)

| 데이터셋 | `tabpfn_unsup` | `kumo_unsup` | `tabpfn+clean` | `kumo+clean` | IForest | 기존 최고 |
|---|---|---|---|---|---|---|
| Wilt | 0.745 | 0.738 | 0.949 | 0.925 | 0.409 | 0.512 |
| annthyroid | 0.883 | 0.869 | 0.949 | 0.930 | 0.818 | 0.818 |
| magic.gamma | 0.849 | 0.837 | 0.893 | 0.885 | 0.741 | 0.828 |
| landsat | 0.607 | **0.646** | 0.782 | 0.805 | 0.469 | 0.580 |
| fault | 0.642 | 0.663 | 0.796 | 0.829 | 0.565 | 0.741 |
| Waveform | 0.760 | 0.758 | 0.843 | 0.865 | 0.730 | 0.817 |
| mammography | 0.880 | 0.829 | 0.897 | 0.877 | 0.878 | 0.908 |
| yeast | 0.433 | 0.411 | 0.475 | 0.475 | 0.375 | 0.446 |
| shuttle | 0.979 | 0.855 | 1.000 | 0.993 | 0.998 | 0.998 |
| PageBlocks | 0.732 | 0.663 | 0.898 | 0.858 | 0.894 | 0.921 |
| cardio | 0.531 | 0.447 | 0.860 | 0.885 | 0.932 | 0.959 |
| cover | 0.400 | 0.495 | 0.889 | **0.956** | 0.890 | 0.954 |

- **Kumo 도 TabPFN 과 같은 모양** (우리 래퍼, 분위수 근사): 이기는 곳 (Wilt · annthyroid · magic.gamma · landsat 에서 IForest 보다 +0.05 ~ +0.33) 과 무너지는 곳 (cover · cardio · PageBlocks) 이 같고, clean context 로 12개 모두 오름 (+0.06 ~ +0.46). 두 FM 의 데이터셋별 AUROC 차는 대부분 0.05 안 (shuttle −0.12 · cardio −0.08 · PageBlocks −0.07 은 Kumo 가 낮음)
- `kumo` 풀 Elo (s1, IForest = 1000): tabpfn_unsup 990 (+207 / −174) · kumo_unsup 929 (+174 / −196) → 둘 다 IForest 와 구간이 겹침. s1 하나라 판단은 약함
- 첫 단서: 위 조건 (오염 context 가 해침 · 라벨의 이상이 밀도로 드물지 않은 데이터에서 이김) 은 TabPFN 만의 것이 아니라 **"context 로 조건부 밀도를 매기는 TFM" 일반** 쪽

K0 + K1 (공식 12개 × 3 split, seed 평균, 2026-11-26)

| 데이터셋 | `tabpfn_unsup` | `kumo_unsup` | `tabpfn+clean` | `kumo+clean` | IForest |
|---|---|---|---|---|---|
| Wilt | 0.730 | 0.722 | 0.937 | 0.914 | 0.419 |
| annthyroid | 0.883 | 0.874 | 0.949 | 0.933 | 0.820 |
| magic.gamma | 0.839 | 0.828 | 0.884 | 0.876 | 0.732 |
| landsat | 0.603 | 0.644 | 0.779 | 0.803 | 0.476 |
| fault | 0.660 | 0.681 | 0.805 | 0.843 | 0.570 |
| Waveform | 0.701 | 0.696 | 0.772 | 0.789 | 0.715 |
| mammography | 0.869 | 0.824 | 0.888 | 0.869 | 0.864 |
| yeast | 0.418 | 0.380 | 0.457 | 0.464 | 0.378 |
| shuttle | 0.968 | 0.846 | 1.000 | 0.991 | 0.996 |
| PageBlocks | 0.732 | 0.725 | 0.901 | 0.876 | 0.896 |
| cardio | 0.522 | 0.463 | 0.869 | 0.884 | 0.932 |
| cover | 0.521 | 0.576 | 0.927 | 0.974 | 0.867 |

- `kumo` 풀 Elo (12개 × 3 split, IForest = 1000, 부트스트랩 200번): tabpfn_unsup 979 (+176 / −159) · **kumo_unsup 906** (+158 / −178) · ECOD 972 · KNN 947 … OCSVM 836. 둘 다 IForest 와 구간이 겹침 (공식 fm 풀과 같은 결론, 풀이 달라 TabPFN 값은 966 과 조금 다름)
- 두 FM 의 데이터셋별 차는 shuttle (−0.12) · cardio (−0.06) · mammography (−0.05) 말고는 ±0.05 안. 이기는 넷 · 무너지는 셋이 같음, clean 효과도 12개 모두 + (Kumo +0.06 ~ +0.42)

Kumo E3 (라벨 없는 정리, s1): `kumo_unsup+trim10` 이 clean 과의 차이를 메운 몫 중앙값 **0.05**, `+iftrim10` −0.01 (TabPFN 0.07 · −0.01). `+iftrim10` 은 무너지는 데이터에서 메움 (cardio 0.43 · cover 0.33 · PageBlocks 0.15 · shuttle 0.99) · 이기는 데이터에서 내려감 (Wilt −0.09 · annthyroid −0.09 · landsat −0.08) → TabPFN 과 같은 모양. Kumo 는 정리 뒤 상수 열에서도 실패하지 않음 (cardio 도 값 있음)

Kumo E2 (오염 비율 × 배치, 94칸): `kumo_unsup` 의 r = 0 대비 평균 하락 **모임 0.053 / 흩어짐 0.072** (TabPFN 0.051 / 0.064), 하락 ~ 가림 Spearman 0.57 · ~ 비율 0.23 (TabPFN 0.52 · 0.27).
cover 0.5% 에서 0.956 → 0.687 / 0.627 (모임 / 흩어짐), cardio 5% 에서 0.885 → 0.657 / 0.592, magic.gamma 20% 에서 0.885 → 0.853 / 0.857 → **TabPFN 과 같은 모양** (오염에 가장 민감, 흩어짐이 더 해침, 이상이 서로 닮은 데이터에서 적은 비율로도 무너짐)

실험 (2026-11-26 모두 끝, Kumo E1 결과는 위 "E1 결과" 표의 Kumo 열)

| # | 무엇 | 대상 | 방법 |
|---|---|---|---|
| K0 | 기본 비교 (위 "결과" 의 Kumo 판) + 시간 재기 | 공식 12개 × s1 | `kumo_unsup` · `kumo_unsup+clean` |
| K1 | 기본 비교 나머지 | 공식 12개 × s2 · s3 | 같음 |
| K2 | 조건 E1 · E2 · E3 의 Kumo 판 | E1 46 · E2 94 · E3 12 (정리는 `+trim10` · `+iftrim10` 만) | `kumo_unsup` (+ E1 은 `+clean`) |

- 결과는 `results/anomaly_conditions.csv` (공식 대회의 Kumo 행도 변형이라 여기로), Elo 는 공식 풀과 따로 (`kumo` 풀: 기존 10 + `tabpfn_unsup` + `kumo_unsup`, IForest = 1000)
- 비용 (K0 로 잰 값: TabPFN 의 1.7배): K1 (s2 · s3, `+clean` 까지) 약 2.5시간 · K2 의 E1 (46개, `+clean` 까지) 약 4.7시간 · E2 (94칸) 약 3.2시간 · E3 (`+trim10` · `+iftrim10`) 약 2.2시간. 모두 약 12.6 GPU 시간 (노드 대기 · 설치 제외)

### 판단 기준: 언제 TFM 을 쓰고 언제 기존 방법을 쓸지 (2026-11-26)

위 E0 ~ E4 · Kumo 에서. TFM = TabPFN v2 비지도 (`tabpfn_unsup`) 와 Kumo 로 같은 chain rule 을 만든 우리 래퍼 (`kumo_unsup`), 기존 방법 = pyod 기본 10개.

| 상황 | 쓸 것 | 근거 |
|---|---|---|
| **정상 행만 모을 수 있음** (정상 데이터로 fit 하는 semi-supervised) | TFM 을 첫 후보로. 이상이 멀리 떨어진 무리 · 범위 밖 값이면 IForest · KNN 으로도 충분 | E4: `tabpfn+clean` Elo 1위 · 12개 중 7개 1등 (KNN · LOF +clean 과 구간 겹침). E1 의 global · cluster 는 clean 이면 모두 0.99 이상 |
| 라벨 없는 train 뿐 + **이상이 "열 사이 관계가 깨진 행"** (열마다 값은 정상 범위) 으로 보임 | **TFM** | E1 dependency: TabPFN 0.986 · Kumo 0.992 (기존 최고 KNN 0.922 · IForest 0.782). 실제 데이터에서 기존 방법이 0.5 근처인 Wilt · landsat 에서 1등 |
| 라벨 없는 train 뿐 + **같은 종류의 이상이 되풀이됨** (서로 닮은 이상, 한 고장 유형 · 한 무리) | **TFM 을 쓰지 말 것** → IForest (또는 OCSVM · ECOD) | E1 cluster: TabPFN 0.776 대 IForest 0.989. E2: test 이상과 닮은 이상이 context 에 있으면 0.5% (35행) 만으로 cover 0.89 → 0.5. **이상 비율이 낮다는 것은 안전 신호가 아님** |
| 라벨 없는 train 뿐 + 이상의 성질을 모름 | **IForest 를 기본**으로. TFM 은 이길 때 크게 이기고 질 때 크게 짐 | 공식 fm 풀 Elo: IForest 1000 · TabPFN 966 · (kumo 풀) Kumo 906, 구간 겹침. 꼴찌 셋에서 −0.18 ~ −0.42 |
| 오염된 context 를 라벨 없이 고쳐 TFM 을 쓰려 함 (상위 k% 빼고 다시) | **기대하지 말 것** | E3: 미리 정한 `+trim10` 은 clean 과의 차이를 중앙값 7% (Kumo 5%) 만 메움. IForest 로 고르면 무너지는 데이터는 고치지만 TFM 이 이기는 데이터를 망침 |
| 이웃 기반 기존 방법 (KNN · LOF) | 오염에 약한 것은 TFM 과 같음 → 위 "되풀이되는 이상" 에서는 같이 피할 것 | E4: 오염 → clean 평균 AUROC LOF +0.14 · TabPFN +0.14 · KNN +0.10 · IForest +0.05. E1 cluster 에서 KNN 0.794 · LOF 0.439 |

- **TFM 일반**: 위 모양은 TabPFN 과 Kumo 에서 같음 (K0 ~ K2: 이기는 · 무너지는 데이터, clean 효과, 오염 민감도 · 흩어짐이 더 해침, 정리가 안 됨) → 특정 모델이 아니라 "context 의 이웃으로 조건부 밀도를 매기는 방식" 의 성질로 봄. Kumo 쪽은 우리 래퍼 · 분위수 근사라는 단서가 붙음
- 실무 확인: 이상 예가 몇 개라도 있으면 그것끼리 닮았는지 (같은 유형인지) 를 보면 위 표의 어느 칸인지 가를 수 있음 (라벨 없는 진단은 이번에 검증하지 않음 → 남은 것)

#### 가설별

| 가설 | 판정 | 한 줄 |
|---|---|---|
| H1a dependency 이상에서 TFM 이 강함 | **확인** | TabPFN 10 / 10 에서 기존 10개보다 높음, Kumo 0.992 (Elo 1 · 2위) |
| H1b cluster 이상에서 TFM 이 약함 (context 의 무리) | **확인** | TabPFN 0.776 · Kumo 0.739 → clean 1.000. KNN · LOF 도 같음 |
| H2 같은 비율이면 "모임" 이 더 해침 | **반증 → 고친 해석** | 흩어짐이 더 해침. 해치는 것은 뭉침이 아니라 **test 이상과 닮은 이상이 context 에 있음 (가림)**: 하락 ~ 가림 0.52 · ~ 비율 0.27 |
| H3 라벨 없는 정리가 clean 과의 차이를 절반 이상 메움 | **반증** | 7% (주 지표), 가장 나은 변형도 20% |
| H4 clean 끼리 비교해도 TFM 이 기존 최고 근처 이상 | **반쯤** | Elo 1위지만 KNN · LOF 와 구간 겹침, cover · cardio · PageBlocks 는 clean 끼리도 짐 |
| (Kumo) 조건이 TFM 일반 | **대체로 확인** | 위 "TFM 일반" |

#### 남은 것

- E1 ~ E3 · K2 는 seed 1 (split s1) 하나. 효과가 큰 것 (dependency · cluster · cover 의 오염) 은 seed 하나로도 뚜렷하지만, E2 의 작은 차 (모임 / 흩어짐 0.01 ~ 0.05) 는 뽑기 2개뿐
- 데이터셋 늘리기 (조건 5 의 2차, ADBench Classical 의 다른 데이터를 1 의 규칙으로) 는 안 함: 12개 상관은 단서로만 씀
- 라벨 없이 "되풀이되는 이상" 인지 가르는 진단 (예: TFM · IForest 점수의 순위 상관, 상위 점수 행끼리의 거리), TFM 과 IForest 섞기, 작은 검증 라벨로 고르기 — 검증 안 함
- cardio 의 범주 열 뭉개짐 (tabpfn-extensions 의 `astype(int)`, 4) 이 clean 끼리도 지는 까닭인지 확인 안 함
- E2 에서 가림과 비율을 완전히 가르지 못함 (같이 움직임). dependency 의 landsat · PageBlocks 는 생성 못 함
- Kumo 래퍼의 밀도 근사 (h, 꼬리) 민감도는 합성 확인만. tabicl 래퍼는 범위 밖

### 정한 것 (2026-11-25 사용자 확인: 추천안 그대로)

| # | 정할 것 | 정한 것 | 까닭 |
|---|---|---|---|
| 9 | E4 의 `<기존>+clean` 행 자리 | `results/anomaly.csv` (기존 `+clean` 과 같은 자리, 기존 행 · fm Elo 그대로) | 공식 대회 · 같은 종류의 진단. 나머지 변형은 `anomaly_conditions.csv` |
| 10 | 순열 수 | E0 를 먼저 돌려 통과하면 순열 3 → **E0 기준 미달로 순열 10** (2026-11-25 사용자) | 비용 ×0.3. 통과 못 하면 10 (1차 8.6 GPU 시간) |
| 11 | 1차 범위 | E0 → E4 · E2 · E3 → E1 (위 격자, seed 1) | 단서가 가장 직접적인 것부터. E1 의 dependency 는 생성 시간을 보고 |
| 12 | E3 의 주 지표 | k = 10 고정 | 라벨 없이 고를 수 있는 값이어야 실무 답이 됨 |
| 13 | Kumo 연속 열의 밀도 | 999 분위수 → 선형 CDF + 지수 꼬리, 폭 2h 구간 확률 (h = 0.005) | 위 "Kumo". 분위수만 나오므로 근사가 필요. 알려진 밀도로 확인 |

## 실행 기록

| 때 (KST) | 무엇 | 대상 | 상태 |
|---|---|---|---|
| 2026-11-25 07:59 | A: 기존 10개 + `iforest+clean` (`--note ad-base`) | 12개 × s1~s3 (36), serverless CPU 작업 하나 (`ad-12datasets-36splits-cpu-20261125-075904`) + 채점 요청 `anomaly__20261125-080029-017778` | 08:18 끝. 작업 상태는 FAILED (종료 코드 1) 지만 36개 대회를 끝까지 돌았고 결과는 남음: 실패한 방법이 있는 대회가 있어서. 실패: pyod PCA 가 cardio s1 · s3, fault s2 · s3 에서 점수 NaN (로컬에서는 같은 데이터로 통과 → 플랫폼에 따라 성분 분산 0 으로 나눔), CBLOF 가 waveform s1 에서 "Could not form valid cluster separation" (pyod 기본 n_clusters=8). 빠진 칸은 Elo 에서 IForest 값 (imputed). 작업 실행이 두 번 (task 재시도) 이고 두 번째에 PCA 실패가 fault 둘에 더 나옴 → 정리 작업이 합친 쪽 확인 필요 |
| 2026-11-25 08:14 | Agent 파일럿 1: cardio (`anomaly-p02`, s1) × 5설정 (aide-sonnet · claude-code-sonnet · aide-gpt · codex-gpt · claude-code-sonnet+cont), 한 번에 (TabArena #26 과 같은 방식). 내기 전 `fab mleb agent list` 에 다른 실행 없음 | A10 × 1시간 × 5 | 다섯 모두 SUCCESS (AIDE 63분 · Codex 51분 · Claude Code 3분 · +cont 5분) |
| 2026-11-25 10:16 | 정리 작업 kick (이 worktree, 대기 항목은 모두 이 트랙 것) | fm 1 · agent 5 · grade 1 | SUCCESS. `fab sync` → `anomaly.csv` 393행 (36 × 11 − 실패 3: PCA cardio s1 · s3, CBLOF waveform s1. 정리 작업이 합친 것은 첫 번째 task 실행의 출력), `anomaly_runs.csv` 5행 |
| 2026-11-25 10:32 | Agent 파일럿 2: fault (`anomaly-p04`, s1) × 5설정, 한 번에. 내기 전 `fab mleb agent list` 에 다른 실행 없음 | A10 × 1시간 × 5 | 다섯 모두 SUCCESS, 13:3x kick 으로 채점 |
| 2026-11-25 14:20 | TabPFN s2 · s3 (`tabpfn_unsup` · `+clean`, 24 split) | H100 1장 | SUCCESS, 작업 156분 (실행기 안 88분, 나머지는 노드 대기 · 설치), 실패 0 |
| 2026-11-25 14:21 | Agent 3: Wilt (`anomaly-p11`, s1) × 5설정 (TabPFN 의 s1 결과를 보고 FM 에 유리한 대조로 고름) | A10 × 1시간 × 5 | 넷 SUCCESS, +cont 는 플랫폼 오류 (`ENV_VARS_FILE_READ_FAILED`) → 제외 · 14:24 다시 냄 → SUCCESS |
| 2026-11-25 22:11 | 조건 E0: `tabpfn_unsup_p3` (순열 3) | 공식 12개 × s1, H100 1장 (`adc-e0-20261125-221145`, run_id 180585864336424) + 채점 요청 `anomaly__20261125-221154-428825`. 내기 전 `fab remote status` 대기 0 | 제출 |
| 2026-11-25 22:12 | 조건 E4: 기존 9개 `+clean` (`iforest+clean` 은 있음) | 공식 12개 × s1 ~ s3 (36), CPU (`adc-e4-cpu-20261125-221208`, run_id 756418314324206) + `anomaly__20261125-221440-697410` | 제출 |
| 2026-11-25 22:15 | 조건 E2: 기존 10개 | E2 94칸, CPU (`adc-e2-cpu-cpu-20261125-221522`, run_id 480167934317774) + `anomaly__20261125-221811-966950` | 제출 |
| 2026-11-25 22:18 | 조건 E3: 기존 10개 `+trim10` | 공식 12개 × s1, CPU (`adc-e3-cpu-cpu-20261125-221816`, run_id 655063116172112) + `anomaly__20261125-222043-479773` | 제출 |
| 2026-11-25 22:22 | 조건 E1: 기존 10개 + `+clean` 10개 | E1 46개 (local · global · cluster 12개씩, dependency 10개), CPU (`adc-e1-cpu-cpu-20261125-222212`, run_id 610156669795673) + `anomaly__20261125-222451-705786` | 제출 |
| 2026-11-25 22:28 | kick (이 worktree, 대기 항목은 모두 이 트랙 것) → `fab sync` | E0 · E3 (CPU) 채점 | SUCCESS. `anomaly_conditions.csv` 129행 (E0 12 + E3 117: pca+trim10 2 · cblof+trim10 1 빠짐), `anomaly.csv` 그대로 489행. 같이 받은 `mlebench_runs.csv` 변경은 되돌림 |
| 2026-11-25 22:3x | 결정 10 갱신 (사용자): E0 가 기준 미달 → 조건 실험의 TabPFN 은 **순열 10** (`tabpfn_unsup`, 공식과 같은 설정). Kumo 를 FM 으로 범위에 넣음 (사용자, 아래 "Kumo") | | |
| 2026-11-25 22:37 | 조건 E1 GPU: `tabpfn_unsup` · `+clean` | E1 46개, H100 1장 (`adc-e1-gpu-20261125-223703`, run_id 1035955298232745, 시간 제한 480분) + `anomaly__20261125-223710-935635` | 제출 |
| 2026-11-25 22:37 | 조건 E2 GPU: `tabpfn_unsup` | E2 94칸, H100 1장 (`adc-e2-gpu-20261125-223715`, run_id 517457707250671, 360분) + `anomaly__20261125-223722-328814` | 제출 |
| 2026-11-25 22:37 | 조건 E3 GPU: `tabpfn_unsup+trim{5,10,20}` · `+trim10x2` · `+iftrim{5,10,20}` | 공식 12개 × s1, H100 1장 (`adc-e3-gpu-20261125-223726`, run_id 1125493504987868, 600분) + `anomaly__20261125-223733-918295` | 제출 |
| 2026-11-25 22:50 | Kumo K0: `kumo_unsup` · `+clean` (우리 래퍼) | 공식 12개 × s1, H100 1장 (`ad-12datasets-12splits-20261125-224959`, run_id 306904000437459, 360분) + `anomaly__20261125-225007-084920`. 내기 전 대기 6개는 모두 이 트랙 것 | 제출 |
| 2026-11-25 22:58 · 23:0x | E1 · E2 · E4 CPU 작업 셋 끝 (`INTERNAL_ERROR/FAILED` 로 찍힘: 대회는 모두 끝까지 돌고 몇 칸의 방법 실패 (pyod PCA 가 cardio 에서 NaN, CBLOF 클러스터 실패) 로 종료 코드 1, 공식 첫 CPU 작업과 같음) → kick → `fab sync` | | `anomaly.csv` +323행 (E4, 기존 행 그대로 · `fab ad elo` 결과 바이트 단위로 같음), `anomaly_conditions.csv` +1,851행 (E1 · E2 기존 방법) |
| 2026-11-26 00:07 · 00:1x | K0 SUCCESS (77분) → kick (이 worktree) → `fab sync` | | `anomaly_conditions.csv` +24행 (12 × 2, 실패 0) |
| 2026-11-26 07:22 | 결정 (사용자): Kumo 나머지 모두 (약 12.6 GPU 시간) | | |
| 2026-11-26 07:22 | Kumo K1: `kumo_unsup` · `+clean` | 공식 12개 × s2 · s3, H100 1장 (`ad-12datasets-24splits-20261126-072250`, run_id 212764738957296, 480분) + `anomaly__20261126-072258-568863` | 제출 |
| 2026-11-26 07:23 | Kumo K2-E1: `kumo_unsup` · `+clean` | E1 46개, H100 1장 (`adc-e1-gpu-kumo-20261126-072303`, run_id 829635953276105, 720분) + `anomaly__20261126-072310-900953` | 제출 |
| 2026-11-26 07:23 | Kumo K2-E2: `kumo_unsup` | E2 94칸, H100 1장 (`adc-e2-gpu-kumo-20261126-072315`, run_id 991509643797888, 540분) + `anomaly__20261126-072322-521058` | 제출 |
| 2026-11-26 07:23 | Kumo K2-E3: `kumo_unsup+trim10` · `+iftrim10` | 공식 12개 × s1, H100 1장 (`adc-e3-gpu-kumo-20261126-072327`, run_id 93601287978257, 420분) + `anomaly__20261126-072333-785026`. 내기 전 대기 3개 = TabPFN E1 ~ E3 의 채점 (작업은 끝남) | 제출 |
| 2026-11-26 07:3x | TabPFN E1 (01:13 SUCCESS) · E2 (00:18 SUCCESS) · E3 (02:00 `INTERNAL_ERROR`: 12개 끝까지, yeast · cardio 의 정리 변형 6칸 실패로 종료 코드 1) → kick (이 worktree) → `fab sync` | | `anomaly_conditions.csv` +264행 |
| 2026-11-26 09:56 · 10:2x | Kumo K1 (2시간 34분) · K2-E3 (2시간 33분) SUCCESS → kick (이 worktree, 대기 항목은 모두 이 트랙 것) → `fab sync` | | `anomaly_conditions.csv` +72행 (K1 48 · E3 24, 실패 0) |
| 2026-11-26 11:20 · 11:3x | Kumo K2-E2 SUCCESS (3시간 57분) → kick (이 worktree) → `fab sync` | | `anomaly_conditions.csv` +94행 (실패 0) |
| 2026-11-26 12:10 · 12:2x | Kumo K2-E1 SUCCESS (4시간 47분) → kick (이 worktree) → `fab sync` | | `anomaly_conditions.csv` +92행 (실패 0), 대기 항목 0 |
| 2026-11-25 15:4x · 18:20 | kick 두 번 (Wilt Agent · 제외, 그다음 TabPFN s2 · s3 채점). forecast 세션과 "상대 트랙 grade 대기 없음 + 상대 agent 대기 항목이 모두 아직 RUNNING" 일 때만 kick 하기로 합의. 18:20 kick 때는 forecast agent 둘이 끝난 채 대기 중이라 그쪽 제안대로 이쪽이 먼저 (그 둘은 시도 1/3, forecast 브랜치 kick 으로 처리) | | SUCCESS, 이상탐지 대기 항목 0 |

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
