# TabArena (P4): 조사 · 연동 · 파일럿

TabArena 를 이 저장소의 두 번째 벤치마크로 붙임. 1~8 은 연동 전 조사 (2026-11-13), 9 는 연동 구현, "파일럿 결과" 는 2026-11-14 실행.

- 조사일 2026-11-13. TabArena 저장소는 `autogluon/tabarena` 의 `c9f57b2` (2026-11-11 커밋) 기준
- 확인 못 한 것은 "확인 못 함" 으로 적음. 추측으로 채우지 않음
- 계획·단계: [plans/roadmap.md](plans/roadmap.md) 의 "벤치마크"·"단계" P4. MLE-bench 연동: [mle-bench.md](mle-bench.md)

목차

- [요약](#요약)
- [출처](#출처)
- [1. 데이터셋](#1-데이터셋)
- [2. 분할·평가 프로토콜](#2-분할평가-프로토콜)
- [3. 리더보드·기준선](#3-리더보드기준선)
- [4. TFM 제약과 우리 FM 이 돌 수 있는 데이터셋](#4-tfm-제약과-우리-fm-이-돌-수-있는-데이터셋)
- [5. 에이전트에게 줄 형식과 정답 격리](#5-에이전트에게-줄-형식과-정답-격리)
- [6. 오염](#6-오염)
- [7. 비용 어림과 파일럿](#7-비용-어림과-파일럿)
- [8. 연동 설계 초안](#8-연동-설계-초안)
- [9. 연동 (2026-11-14 구현)](#9-연동-2026-11-14-구현)
- [파일럿 결과 (2026-11-14)](#파일럿-결과-2026-11-14)
- [결정 (2026-11-13)](#결정-2026-11-13)
- [부록: 데이터셋 51개](#부록-데이터셋-51개)

## 요약

| 항목 | 찾은 것 |
|---|---|
| 데이터셋 | TabArena-v0.1: 51개 (이진 30 · 다중 8 · 회귀 13), 748~150,000 행, 피처 5~1,777, 클래스 최대 8. OpenML 스위트 457 (`tabarena-v0.1`) |
| 분할 | 외부 3-fold 층화 CV. 행이 2,500 미만이면 10번, 아니면 3번 반복 → 데이터셋마다 30 또는 9 split, 합 816. OpenML task 의 고정 분할 그대로 받음 |
| 지표 | 이진 ROC AUC, 다중 log loss, 회귀 RMSE. 결과 파일은 `metric_error` (AUC 는 1 − AUC) |
| 합치기 | Elo (기본 RandomForest = 1000, 부트스트랩 200번), 개선 여지 (improvability), 평균 순위, 승률 |
| TFM | 이미 많음: TabPFNv2, TabICL·TabICLv2, Kumo-Tabular (large·medium·small), TabPFN-2.5·2.6·3·3.5, Mitra, LimiX, TabDPT 등. 대부분 기본 설정 한 개만 (TabPFNv2 는 튜닝 결과도 있음) |
| TFM 설정 | 8-fold 내부 CV 로 검증 예측만 만들고, test 는 **외부 train 전체로 다시 fit 한 모델** 이 예측 (`refit_folds=True`) → 우리 FM 한 번 추론과 같은 꼴 |
| fold 별 결과 | 공개 parquet 로 받음 (방법당 수십 KB~수 MB). split 마다 `metric_error`·학습·추론 시간 |
| 우리 FM | tabpfn_v2 는 공식 범위 안 33개, tabicl·kumo 는 51개 전부 (TabArena 의 TabICLv2·Kumo-Tabular 가 51개 다 돎) |
| 격리 | 데이터 전체 (타깃 포함) 가 OpenML 에 공개 → MLE-bench 와 같은 정답 격리는 하되, 인터넷을 여는 한 "에이전트가 원본을 받아 답을 찾는 것" 은 사후 검사로만 막음 |

## 출처

| 출처 | 쓴 곳 |
|---|---|
| 논문: Erickson et al., "TabArena: A Living Benchmark for Machine Learning on Tabular Data", NeurIPS 2025 D&B, [arXiv:2506.16791](https://arxiv.org/abs/2506.16791) (v4, 2025-11-03). [HTML](https://arxiv.org/html/2506.16791v4) | 2.1 (모델·튜닝), 2.2 (선별), 2.3 (평가), A.1 (대체), A.2 (시간 제한), B.1 (선별 기준), C.1 (FM refit), D.1 (Elo), E.6 (오염) |
| 저장소: [autogluon/tabarena @ c9f57b2](https://github.com/autogluon/tabarena/tree/c9f57b2dced40dbef3daa95fffeeba7d15d926db) | README, `AGENTS.md`, 아래 경로들 |
| 데이터셋 메타데이터 [`curated_tabarena_dataset_metadata.csv`](https://github.com/autogluon/tabarena/blob/c9f57b2dced40dbef3daa95fffeeba7d15d926db/packages/tabarena/src/tabarena/benchmark/task/metadata/data/curated_tabarena_dataset_metadata.csv) | 1, 4, 6, 부록 |
| 모델 설정 [`models/<key>/info.py`·`model.py`](https://github.com/autogluon/tabarena/tree/c9f57b2dced40dbef3daa95fffeeba7d15d926db/packages/tabarena/src/tabarena/models) | 3, 4 |
| 리더보드 방법 목록 [`contexts/tabarena/methods.py`](https://github.com/autogluon/tabarena/blob/c9f57b2dced40dbef3daa95fffeeba7d15d926db/packages/tabarena/src/tabarena/contexts/tabarena/methods.py) | 3 |
| OpenML 에서 직접 받는 예 [`run_get_tabarena_datasets_from_openml.py`](https://github.com/autogluon/tabarena/blob/c9f57b2dced40dbef3daa95fffeeba7d15d926db/examples/advanced/run_get_tabarena_datasets_from_openml.py) | 2, 8 |
| fold 별 결과 내보내기 [`run_export_results_per_split.py`](https://github.com/autogluon/tabarena/blob/c9f57b2dced40dbef3daa95fffeeba7d15d926db/examples/plots/run_export_results_per_split.py) | 3 |
| OpenML API: [스위트](https://www.openml.org/api/v1/json/study/tabarena-v0.1), [task 363621](https://www.openml.org/api/v1/json/task/363621), [dataset 46913](https://www.openml.org/api/v1/json/data/46913), [분할 파일](https://openml.org/api_splits/get/363621/Task_363621_splits.arff) | 2, 5 |
| 리더보드: [tabarena.ai](https://tabarena.ai) → [HF Space TabArena/leaderboard](https://huggingface.co/spaces/TabArena/leaderboard) | 동적 페이지라 내용을 못 읽음 (확인 못 함) |
| Kumo-Tabular 소개 [HF 블로그](https://huggingface.co/blog/nvidia/kumo-tabular) | 6 |

공개 결과 parquet 는 `https://data.tabarena.ai/cache/artifacts/<suite>/methods/<method>/results/model_results.parquet`
(2025-06-12 묶음은 `https://tabarena.s3.amazonaws.com/cache/...`) 에서 받음. 이 문서의 FM·GBDT 비교 숫자는 이 파일들의 기본 설정 행
(`*_c1_*`) 에서 계산 (받은 묶음: TabPFNv2_GPU·TabICL_GPU·LightGBM·RandomForest @2025-06-12, CatBoost @2026-07-13,
TabICLv2 @2026-09-16, Kumo-Tabular @2026-10-06).

## 1. 데이터셋

TabArena-v0.1 = 51개. 1,053개 (기존 벤치마크 14개) 에서 손으로 골라냄 (논문 2.2). 전체 목록은 [부록](#부록-데이터셋-51개).

| | 값 |
|---|---|
| 과제 | 이진 30 · 다중 8 · 회귀 13 |
| 행 | 748 ~ 150,000 (중앙값 6,497) |
| 피처 (메타데이터 `num_features`) | 5 ~ 1,777 (중앙값 18). 500 넘는 것 3개 (Bioresponse 1,777 · hiva_agnostic 1,618 · QSAR-TID-11 1,025) |
| 클래스 | 2 ~ 8 (10 넘는 것 없음) |
| 출처 | UCI 26 · Kaggle 15 · OpenML 6 · 기타 4 |
| 라이선스 | CC BY 4.0 26, Public 8, Public Domain 6, CC0 5, MIT 2, DbCL v1.0 2, CC BY-SA 1, CC BY-NC-SA 1 (miami_housing) |
| 받는 곳 | OpenML 스위트 [457 `tabarena-v0.1`](https://www.openml.org/api/v1/json/study/tabarena-v0.1) (task 51개). task id 363612~363712, dataset id 46904~46980 |

- 선별 기준 (논문 B.1): train 500~250,000 행, IID, 실제 데이터, 학술 벤치마크에 쓸 수 있는 라이선스, 타깃·test 분포가 새는 전처리를 거치지 않은 것
- OpenML 의 데이터셋은 TabArena 팀이 다시 올린 판 (피처 이름을 알아보기 쉽게 바꿈, 중복 같은 이상을 설명에 적음. 예: [46913](https://www.openml.org/api/v1/json/data/46913) "duplicates (29%)")
- 파일은 OpenML 의 parquet (`parquet_url`, 예 `https://data.openml.org/datasets/0004/46913/dataset_46913.pq`) 또는 ARFF
- 다음 판: 같은 저장소의 BeyondArena (142개, IID·시간·그룹 분할, [arXiv:2606.30410](https://arxiv.org/abs/2606.30410)). README 는 "BeyondArena will supersede TabArena-v0.1 in the future" (언제인지는 확인 못 함). 이 문서는 v0.1 만 봄

## 2. 분할·평가 프로토콜

| 항목 | 값 | 출처 |
|---|---|---|
| 외부 분할 | 3-fold CV, 분류는 클래스 층화 | 논문 2.3, OpenML task 의 `estimation_procedure` (`number_folds` 3, `stratified_sampling` true) |
| 반복 | 행 < 2,500 → 10번, 그 외 3번 (250,000 넘으면 1번이지만 v0.1 에는 없음) | 논문 2.3, `run_get_tabarena_datasets_from_openml.py` |
| split 수 | 30 split 17개 + 9 split 34개 = 816 | 메타데이터 `tabarena_num_repeats` |
| split 번호 | 결과 파일의 `fold` 열 = 3 × repeat + fold (0~29) | `benchmark/task/utils.py` |
| 분할 받기 | OpenML task 의 `Task_<id>_splits.arff` (열 `type`·`rowid`·`repeat`·`fold`). 반복마다 모든 행이 test 에 한 번 | 직접 받아 확인 (task 363621: TEST 7,480줄 = 748 × 10) |
| OpenML 쪽 반복 수 | 메타데이터 `openml_num_repeats` 는 모두 10. TabArena 는 9 split 데이터셋에서 앞 3번만 씀 | 메타데이터 |
| 지표 | 이진 ROC AUC · 다중 log loss · 회귀 RMSE (결과 파일의 `metric` = `roc_auc`·`log_loss`·`rmse`) | 논문 2.3, 결과 parquet |
| 내부 검증 | 8-fold CV × 1 (`8x1`) bagging. 튜닝·앙상블 선택은 이 검증 점수로 | 논문 2.1, `AGENTS.md` "Validation protocol" |
| TabArena-Lite | 모든 데이터셋의 첫 split (r0f0) 하나 | 논문 3.2, README |
| 합치기 | Elo: 짝 비교 승·무·패, 기본 RandomForest = 1000, 부트스트랩 200번 95% 구간. 개선 여지 = 1 − (그 풀의 최고 오차 / 오차). 평균 순위·승률도 냄. 참가자 묶음 (`EntrantPool`) 마다 다시 계산 | 논문 D.1, `AGENTS.md` "Entrant pools" |
| 못 돈 결과 | 기본 RandomForest 결과로 대체 (`imputed`) | 논문 A.1 |

`fab run` 의 고정 fold 와 맞추기

- TabArena 의 외부 split 하나 = 우리 "대회" 하나 (train/test). OpenML 분할을 그대로 쓰면 TabArena 와 같은 행으로 학습·채점함
- `fab run` 의 5-fold (`folds.csv`) 는 그 split 의 train 안에서 도는 내부 CV 로 남김. TabArena 의 내부 8-fold 와 수는 다르지만 test 점수에는 영향 없음 (FM 은 train 전체로 fit, 아래 3)
- 비교는 test 점수 (`metric_error`) 끼리만. 우리 CV 점수와 TabArena `metric_error_val` 은 내부 분할이 달라 나란히 놓지 않음
- 지표는 `fab/core/metrics.py` 에 이미 있음 (`auc`·`logloss`·`rmse`)

## 3. 리더보드·기준선

현재 리더보드 목록 (`tabarena_method_metadata_collection`, 2026-11-11 커밋) 은 모델 58개 + 시스템 8개.

| 구분 | 내용 |
|---|---|
| 모델 (model) | TabArena 가 같은 규칙으로 돌림: 기본 설정 1개 + 무작위 설정 최대 200개, 결과는 기본 / 튜닝 / 튜닝 + 앙상블 세 줄. 설정 하나 · split 하나에 1시간 (논문 2.1, A.2) |
| 시스템 (system) | 전처리·검증·튜닝을 스스로 하는 파이프라인: AutoGluon 1.4~1.6 (4시간), TabFM+, TabLDM+, Chakra-Tab (호스팅 API). README 는 "LLM agents" 도 시스템으로 받는다고 적음. 지금 목록에 `with-llm` 태그를 단 시스템은 없음 |
| GBDT | LightGBM·XGBoost·CatBoost (기본 + 튜닝 200개). 기본 설정도 8-fold bag (8개 평균) 이고 refit 하지 않음 (논문 2.1) |
| 하드웨어 | 권장 8 CPU · 32GB RAM · 100GB 디스크, GPU 종류는 확인 못 함 (논문 3) |

TFM (이미 있음. TabICL·TabICLv2·Kumo 는 기본 설정만 (`can_hpo=False`), TabPFNv2 는 튜닝 결과도 있음. 이 문서의 비교는 모두 기본 설정)

| TabArena 이름 | 묶음 (suite) | 설정 | 우리 대응 |
|---|---|---|---|
| TabPFNv2 (`TabPFNv2_GPU`) | tabarena-2025-06-12 | 기본 + HPO 200개 (결과 파일에 201개). 패키지 버전·`n_estimators` 는 확인 못 함 (AutoGluon 쪽 구현) | `tabpfn_v2` (tabpfn 9.2.0, v2 가중치, `n_estimators` 라이브러리 기본) |
| TabICL (`TabICL_GPU`) | tabarena-2025-06-12 | v1.1 체크포인트, `n_estimators` 32, 분류만 | – |
| TabICLv2 | tabarena-2026-09-16 | `tabicl>=2.0.0`, 체크포인트 `tabicl-classifier-v2-20260212` / `tabicl-regressor-v2-20260212`, 분류·회귀 | `tabicl` (tabicl 2.2.0, 라이브러리 기본 체크포인트. v2-20260212 인지는 확인 못 함) |
| Kumo-Tabular (large) | tabarena-2026-10-06 | `structured-data-models` git `98f6128`, Hub `nvidia/Kumo-Tabular` 태그 v1.0.1 커밋에 고정, `num_estimators` 16, 컨텍스트 상한 20만 행 | `kumo` (structured-data-models 0.2.1, `nvidia/Kumo-Tabular` v1.0.1, `n_estimators` 8) |
| 그 밖 | | RealTabPFN-2.5, TabPFN-2.6·3·3.5(-Fast), Mitra·Mitra-v2, LimiX·LimiX-2, TabDPT, EXAONE-Tabular, Causilo, SAP-RPT-OSS, TabFM 등 | `fab/models` 에 일부 있음 (`tabpfn35`·`mitra`·`limix2m`·`exaone`·`causilo`·`sap_rpt`) |

- TFM 은 `refit_folds=True`: 내부 8-fold 는 검증 예측만 내고 버림, test 는 외부 train 전체로 다시 fit 한 모델이 예측
  (`models/kumo_tabular/model.py`·`tabpfn_3_5/model.py` 등, 논문 C.1 "For all foundation models, we refit on training and validation data").
  → 우리 FM 한 번 추론과 같은 꼴이라 split 별로 나란히 놓을 수 있음
- 다른 점 (점수가 똑같이 나올 거라 기대하지 않을 까닭)
  - 전처리: TabArena 는 AutoGluon 의 모델 무관 전처리를 거친 뒤 모델에 넘김 (`bundle.py` 의 `model_agnostic_preprocessing=True`). 우리는 `fab/models/tfm.py` 의 범주 → 정수 코드
  - 설정: Kumo 는 앙상블 16 대 8, 패키지 git `98f6128` 대 0.2.1 (가중치는 둘 다 Hub v1.0.1). TabPFNv2 는 TabArena 쪽 버전을 확인 못 함
  - LimiX-2 는 TabArena 가 누수 경고를 붙임 (blood-transfusion·diabetes, [tabarena#648](https://github.com/autogluon/tabarena/issues/648)). 우리 FM 셋에는 해당 없음
- fold 별 결과: 공개 결과 parquet (방법당 816행: `dataset`·`fold`·`metric_error`·`metric_error_val`·`time_train_s`·`time_infer_s`) 로 받음.
  리더보드에 쓰인 숫자는 `TabArenaContext().compare(output_dir=..., plot=False)` 의 `results_per_split.csv` (README "Per-split results", 호스팅은 안 함)
- 주의: 2025-06-12 의 TabPFNv2·TabICL 결과 파일에는 못 도는 데이터셋 행도 있고, 값이 기본 RandomForest 와 똑같음 (TabPFNv2 18개, TabICL 15개 = 메타데이터 `can_run_*` 가 False 인 것과 일치) → 비교할 때 빼야 함
- 리더보드의 Elo 값 자체는 HF Space 가 동적이라 못 읽음 (확인 못 함). 필요하면 `tabarena[plot]` 을 따로 깔아 위 `compare` 로 계산

기본 설정끼리의 데이터셋별 비교 (공개 결과에서 계산, split 평균 `metric_error`, 기본 GBDT = CatBoost·LightGBM 중 나은 쪽)

| 비교 | FM 이 나은 데이터셋 | 비고 |
|---|---|---|
| Kumo-Tabular vs 기본 GBDT | 49 / 51 | 진 곳: Amazon_employee_access (+3%), E-CommereShippingData (+0.04%) |
| TabICLv2 vs 기본 GBDT | 42 / 51 | 크게 진 곳: Amazon (+24%), hiva_agnostic (+14%), kddcup09_appetency (+10%), Bioresponse (+8%) |
| TabPFNv2 vs 기본 GBDT (돌 수 있는 33개) | 17 / 33 | 크게 진 곳: in_vehicle_coupon_recommendation (+31%), NATICUSdroid (+21%), wine_quality (+11%) |
| 셋 중 최고 FM vs 기본 GBDT | 50 / 51 | 진 곳은 Amazon_employee_access 하나 (CatBoost 0.1182 대 Kumo 0.1223, 1 − AUC) |

- GBDT 는 튜닝하지 않은 기본값끼리. 튜닝·앙상블한 GBDT 와는 비교하지 않음
- 데이터셋별 숫자는 [부록](#부록-데이터셋-51개)

## 4. TFM 제약과 우리 FM 이 돌 수 있는 데이터셋

TabArena 가 거는 제약 (논문 2.1)

| 모델 | 제약 | 돌 수 있는 데이터셋 | 빠지는 것 |
|---|---|---|---|
| TabPFNv2 | train ≤ 10,000 행 · 피처 ≤ 500 · 클래스 ≤ 10 | 33 (이진 20 · 다중 6 · 회귀 7) | train 1만 행 초과 15개 + 피처 500 초과 3개. 클래스로 빠지는 것은 없음 (최대 8) |
| TabICL (v1) | train ≤ 100,000 행 · 피처 ≤ 500 · 분류만 | 36 (이진 29 · 다중 7) | 회귀 13개 + Bioresponse·hiva_agnostic |
| TabICLv2 · Kumo-Tabular | 결과가 51개 모두 있음 (RandomForest 와 같은 값 없음) | 51 | 없음 |

- train 행 = 전체의 2/3 (외부 3-fold). 메타데이터 `can_run_tabpfnv2` 가 이 기준과 맞음 (예: in_vehicle_coupon 12,684 행 → train 8,456 → 돎, HR_Analytics 19,158 행 → train 12,772 → 안 돎)
- 가장 큰 train 은 GiveMeSomeCredit 의 10만 행

우리 FM (`fab/models/tfm.py`)

| 우리 모델 | TabArena 와 같은 범위 | 범위 밖에서 | 비교 상대 |
|---|---|---|---|
| `tabpfn_v2` | 33개 | `ignore_pretraining_limits=True` 라 돌기는 함. 큰 데이터는 대회 설정 `tfm_max_rows`, 넓은 데이터는 `tfm_max_features` (LightGBM 중요도 상위) 로. 이 18개는 TabArena 에 비교할 TabPFNv2 결과가 없음 (RF 대체값) | TabPFNv2 (2025-06-12) |
| `tabicl` | 51개 (2.x 는 회귀도 됨) | – | TabICLv2 (2026-09-16) |
| `kumo` | 51개 | 컨텍스트 상한 없이 돌면 큰 데이터에서 GPU 메모리를 확인해야 함 (TabArena 는 20만 행 상한, v0.1 에는 그보다 큰 train 없음) | Kumo-Tabular (2026-10-06) |

## 5. 에이전트에게 줄 형식과 정답 격리

MLE-bench 처럼 split 하나를 대회 하나로 만듦.

| 파일 | 만드는 법 |
|---|---|
| `train.csv` | OpenML 데이터 중 split 의 TRAIN 행 + 타깃. 행 id 는 새로 매김 (OpenML `rowid` 를 그대로 주지 않음) |
| `test.csv` | TEST 행, 타깃 뺌 |
| `sample_submission.csv` | id + 예측 열 (이진: 양성 확률 1열, 다중: 클래스별 확률, 회귀: 값). 지표가 AUC·log loss 라 확률 제출 (`submit_proba`) |
| `description.md` | 직접 써야 함 (아래) |
| 정답 | split 의 TEST 행 타깃. 로컬 캐시 + 채점용 내 Volume 에만 (MLE-bench 의 `private` 와 같은 자리, 에이전트 SP 는 못 읽음) |

설명 (`description.md`) 을 가져올 곳

- OpenML 데이터셋 설명: TabArena 팀이 쓴 짧은 틀 (라이선스·원래 출처·인용·"see the reference and the original data source for details"·선별 메모). 과제 설명으로는 부족함
- 피처 이름: TabArena 가 알아보기 쉽게 바꿔 둠 (예: `DonatedBloodInMarch2007`). 그대로 씀
- 원래 출처 페이지 (메타데이터 `original_data_url`: UCI DOI 26개, Kaggle 데이터셋·대회 15개 등): 열 뜻·수집 배경. 페이지 글을 옮겨 쓸 수 있는지는 출처마다 라이선스를 봐야 함 (확인 못 함) → 우리 말로 짧게 다시 씀
- 지표·제출 형식·평가 방식은 MLE-bench 의 description 형식을 따라 우리가 붙임

격리

- MLE-bench 와 같은 격리는 그대로 됨: 정답은 로컬 + 채점용 Volume, 에이전트 SP 에는 public 만 (`docs/mle-bench.md` 의 "정답 격리와 권한")
- 하지만 TabArena 데이터는 타깃까지 전부 OpenML·UCI·Kaggle 에 공개 → 인터넷을 여는 한 에이전트가 원본을 받아 test 행의 타깃을 찾을 수 있음
  (MLE-bench 도 test 가 Kaggle train 에서 나와 같은 구조). 막는 방법은 사후 검사뿐 → `openml.org`·`archive.ics.uci.edu`·Kaggle 데이터셋 URL 접근을 사후 검사에 더함
- 데이터셋 이름·출처는 알리지 않음 (2026-11-13 정함, [결정](#결정-2026-11-13) 3). 아래 "익명화"
- TabArena 는 "inductive" 예측만 받음: test 행끼리 서로 보거나 test 로 전처리를 fit 하면 안 됨 (README "Inductive predictions only"). MLE-bench 식 Agent 는 train+test 로 전처리하는 일이 흔함 (C1 의 tps-may GMM·PCA 도 train+test fit) → 우리 Agent 점수를 TabArena 리더보드 숫자와 나란히 놓을 때 이 차이를 적음

익명화 (결정 3: "모델링 실력" 비교)

| 줌 | 안 줌 |
|---|---|
| 과제 (한 행이 무엇이고 타깃이 무슨 뜻인지), 열 뜻, 지표, 제출 형식 | 데이터셋 이름, 원래 출처·URL·수집 기관·연도·인용 |
| TabArena 가 바꿔 둔 피처 이름 그대로 | OpenML id·`rowid` (행 id 는 새로 매김) |

- 까닭: 타깃이 공개라 이름·출처가 있으면 정답 찾기가 쉬움. 열 뜻은 피처 공학에 필요하고 피처 이름에 이미 드러나 있어 줌
- 에이전트에게 보이는 곳 (원격 대회 id, scratch Volume 의 데이터 묶음 이름, 지시문·경로) 에는 익명 id (예: `tabarena-p01`) 를 씀.
  익명 id ↔ 데이터셋 대응표는 에이전트 노드에 올라가는 코드 묶음에 들어가지 않게 둠 (구현할 때 확인)
- 그래도 피처 이름만으로 알아볼 수 있는 유명 데이터가 있음 → 사후 검사 (위) 로 원본 접근을 봄
- MLE-bench 는 대회 이름·설명을 그대로 줌 → 두 벤치마크의 Agent 는 조건이 다름. 섞어 읽을 때 적음

## 6. 오염

LLM (Agent) 이 데이터셋과 풀이를 학습했을 수 있는지. 공개 풀이 수를 세지는 않았음 (확인 못 함). 아래는 출처로 나눈 것.

| 묶음 | 데이터셋 | 볼 점 |
|---|---|---|
| Kaggle 대회 | Amazon_employee_access, Bioresponse, GiveMeSomeCredit | 대회 풀이·토론이 공개. 가장 위험 |
| Kaggle 데이터셋 | 12개 (Bank_Customer_Churn, HR_Analytics, customer_satisfaction_in_airline, Food_Delivery_Time 등) | 노트북이 많은 편 |
| 교과서급 UCI·기타 | credit-g, diabetes (Pima), wine_quality, bank-marketing, houses, diamonds, blood-transfusion, concrete, airfoil, splice, anneal 등 | 튜토리얼·논문에 자주 나옴 |
| 덜 알려진 것 | hazelnut-spread-contaminant-detection (OpenML 2020), MIC, seismic-bumps, NATICUSdroid, polish_companies_bankruptcy, QSAR-TID-11, jm1, coil2000 등 | 상대적으로 덜 위험 (정도는 확인 못 함) |

- 논문 E.6: TFM 의 오염은 "we currently have no way of detecting contamination", 벤치마크 데이터로 직접 사전학습한 것은 TabDPT 뿐이라고 적음
- Kumo-Tabular 는 합성 데이터만으로 사전학습 ([HF 블로그](https://huggingface.co/blog/nvidia/kumo-tabular): "pretrained entirely on artificial tables"). 벤치마크와의 겹침 점검은 블로그에 없음
- roadmap "정해야 할 것" 의 해석 주의와 같음: Agent 는 LLM 기억에 기댈 수 있고 FM 은 (합성 사전학습이면) 아님. 파일럿은 덜 알려진 것 위주로 고르고, Kaggle 대회 하나를 오염 대조군으로 둠

## 7. 비용 어림과 파일럿

### FM·GBDT

- TabArena 기록 (기본 설정, 8-fold bag + refit 포함한 split 하나의 학습 시간 중앙값): Kumo 3.3초, TabICLv2 1.7초, TabPFNv2 7.9초, CatBoost 12초, LightGBM 4.7초. 큰 데이터셋은 수십 초~수 분 (예: physiochemical_protein Kumo 59초, CatBoost 85초). 하드웨어는 확인 못 함
- 우리 `fab run` 은 split 마다 내부 5-fold CV + test 예측 = fit 6번
- 전부 돌리면: 816 split × (FM 3 + GBDT 2) = 4,080 실행 (fit 약 2.4만 번). 파일럿 4개면 9~30 split × 4 = 57 split × 5 모델
- MLE-bench 처럼 Databricks GPU 작업으로 묶어 제출하면 됨. 데이터셋·split 이 많아 작업 수가 늘어나는 게 문제 (8 의 설계 참고)

### Agent

- 1회 = 설정 하나 × split 하나 × 1시간 · A10 1장. 지금 설정 4개 (+cont 1개) → 데이터셋 하나 × 1회 = 5시간 (+fm 까지 하면 10시간)
- 같은 엔드포인트는 한 번에 하나 (분당 토큰 한도): Sonnet 줄 3개 (aide-sonnet, claude-code-sonnet, +cont), GPT 줄 2개 (aide-gpt, codex-gpt) → 데이터셋 하나에 벽시계 약 3시간
- 모든 split 에 Agent 는 불가능 (hazelnut 하나만 30 split × 5 = 150시간). **Agent 는 Lite split (r0f0) 하나만** (TabArena-Lite 와 같은 split, [결정](#결정-2026-11-13) 2). 반복은 split 을 늘리지 않고 같은 split 에서 2회째
- 51개 전부 Lite × 5 설정 × 1회 = 255시간 (벽시계 약 150시간). 파일럿 2개 × 1회 = 10시간 (벽시계 약 6시간)
- 토큰: MLE-bench 기록으로 Codex 는 1시간 입력 4,200만~4,500만 토큰, 나머지 설정 5만~55만 ([mle-bench.md](mle-bench.md) 의 "429"). 금액은 어림하지 않음

### 파일럿 추천

기준: FM 이 이기는 곳 / GBDT 가 이기는 곳 / 크기 다양 / 오염 적음. 숫자는 공개 결과의 split 평균 (기본 GBDT = CatBoost·LightGBM 중 나은 쪽 대비 %).

| 데이터셋 | 과제 · 행 · 피처 | split | 출처 | TabPFNv2 | TabICLv2 | Kumo | 고른 까닭 |
|---|---|---|---|---|---|---|---|
| **hazelnut-spread-contaminant-detection** | 이진 · 2,400 · 31 | 30 | OpenML 2020, CC BY-SA | −53% | −80% | −90% | FM 이 크게 이기는 곳, 작음, 덜 알려짐. 1 − AUC 가 Kumo 0.0026 대 CatBoost 0.0258 → 천장 근처라 Agent 차이가 작게 나올 수 있음 |
| **in_vehicle_coupon_recommendation** | 이진 · 12,684 · 25 (범주 88%) | 9 | UCI 2017, CC BY 4.0 | **+31%** | −3% | −10% | "FM 이 이기나" 가 어느 FM 이냐에 달린 곳. TabPFNv2 범위 안 (train 8.5천) 인데 GBDT 보다 크게 나쁨 |
| **Amazon_employee_access** | 이진 · 32,769 · 10 (범주 100%, 고카디널리티) | 9 | Kaggle 대회 2013 | 범위 밖 | +24% | +3% | 기본 GBDT (CatBoost) 가 모든 FM 보다 나은 유일한 곳. Kaggle 대회라 오염 대조군 |
| **physiochemical_protein** | 회귀 · 45,730 · 10 | 9 | UCI 2013, CC BY 4.0 | 범위 밖 | −15% | −20% | 회귀, 큼 (TabPFNv2 범위 밖 → 우리 `tabpfn_v2` 는 `tfm_max_rows` 로). 교과서급은 아님 |

- Agent 파일럿 (P4 완료 기준 "1~2개"): **in_vehicle_coupon_recommendation + hazelnut** 을 먼저. 둘 다 오염이 덜하고 FM 결과가 갈림
- FM 은 네 개 모두 + TabArena 비교용으로 51개 전체도 싸게 가능 (위 FM 비용)
- 대안: GBDT 쪽을 하나 더 원하면 E-CommereShippingData (모두 비김, Kaggle 데이터셋), 다중 분류를 원하면 MIC (8 클래스, 1,699 행, Kumo −7%, TabPFNv2 +3%)

## 8. 연동 설계 초안

### MLE-bench 연동과 비교

| | MLE-bench (`fab/bench/mlebench/`) | TabArena (`fab/bench/tabarena/`, 제안) |
|---|---|---|
| 데이터 받기 | Kaggle zip → mlebench `prepare` 로 다시 나눔 | OpenML parquet + 분할 ARFF. 다시 나누지 않음 (분할이 고정) |
| 대회 단위 | 대회 하나 = test 하나 | 데이터셋 하나에 split 9~30개 (아래 "대회 단위") |
| 정답 | mlebench 캐시의 `private` + 내 Volume | 같은 자리: 로컬 캐시 + 내 Volume. 에이전트 SP 에는 public 만 |
| 채점 | `mlebench grade-sample` (원격 정리 작업) | 우리 지표 (`fab/core/metrics.py` 의 auc·logloss·rmse) 로 직접. mlebench 같은 외부 채점기 없음 |
| 비교 대상 | 원래 Kaggle 리더보드 순위·메달, 공개 에이전트 점수 | TabArena 의 같은 split 결과 (FM·GBDT·AutoGluon). 메달 없음 → 같은 split 의 순위·개선 여지, 원하면 Elo |
| 설명 | 대회 description 그대로 | 우리가 씀 (5) |
| 지시문·하네스·프록시·사후 검사 | `fab/agent` | 그대로 씀. 사후 검사에 OpenML·UCI URL 추가 |
| 외부 패키지 | mle-bench 저장소 + 자체 venv (`FAB_MLEBENCH`) | `openml` (+ parquet 읽기용 `pyarrow`) 를 자체 venv 에 (`FAB_TABARENA`, 아래) |

### 필요한 것

- `fab/bench/tabarena/__init__.py`
  - `SUITE = 457`, 쓸 task id 목록 (TabArena 메타데이터 CSV 의 필요한 열만. TabArena 저장소는 Apache-2.0)
  - `prepare(comp)`: driver 로 데이터·분할을 받아 split 의 public (`train.csv`·`test.csv`·`sample_submission.csv`·`description.md`) 을 `data/` 에, 정답은 캐시에
  - `reference()`: 공개 결과 parquet 를 받아 `results/tabarena_reference.csv` (데이터셋 · split · 방법 · `metric_error` · 시간) 로. 2025-06-12 의 TabPFNv2·TabICL 은 `can_run_*` False 행을 뺌
  - `grade()`: 우리 test 예측 → split 점수 → `results/tabarena.csv`. 같은 split 의 TabArena 결과와 나란히
- `fab/bench/tabarena/driver.py`: 자체 venv 에서 `-I` 로 실행 (mlebench driver 와 같은 방식). OpenML 받기·분할 풀기만 하고 결과는 JSON·CSV 로 넘김
- `competitions/ta-<dataset>/config.yaml`: `tabarena: {task_id: 363674}` + 지금 키 (`task`·`target`·`metric`·`submit_proba`·`cv`·`tfm_max_rows`)
  + 에이전트용 익명 id (5 의 "익명화")
- 대회 설명: `competitions/ta-<dataset>/description.md` (손으로 씀, 저장소에 둠. 이름·출처 없이)
- 사후 검사 (`fab/remote/reconcile.py` 쪽): OpenML·UCI·Kaggle 데이터셋 URL 접근 수

### 대회 단위 (A 로 정함)

| 안 | 내용 | 장점 | 단점 |
|---|---|---|---|
| A | (데이터셋, split) 마다 대회 폴더 (`ta-hazelnut-r0f0`) | 지금 코드 (`load_competition`·`data_dir`·결과 폴더·채점) 를 그대로 씀 | FM 전 split 이면 폴더가 816개. 만들어 두지 않고 생성하면 됨 |
| B | 데이터셋마다 대회 하나 + `--split r0f0` 인자 | 폴더가 51개 | `data_dir`·결과 경로·`folds.csv`·채점 요청에 split 을 다 끼워야 함 |

- A 로 함: 저장소에는 r0f0 (`ta-<dataset>`) 만 두고, 다른 split 은 FM 만 돌리는 생성 폴더 (커밋 안 함)

### 의존성

- TabArena 패키지 (`tabarena`) 는 AutoGluon 사전 배포판·Python 3.11~3.13·모델별 extra 를 요구 → 저장소 uv 환경에 넣지 않음
- 데이터·분할만이면 `openml` + `pyarrow` 면 충분 (지금 `uv.lock` 에 둘 다 없음). 이것도 mle-bench 처럼 옆 venv 에 두면 `uv.lock` 을 안 바꿈
- 결과 parquet 받기는 HTTP 만으로 됨 (위 URL). 읽기에 `pyarrow` 가 필요해 driver 쪽에서 CSV 로 바꿔 넘김
- Elo 까지 TabArena 와 같은 식으로 매기려면 `tabarena[plot]` (또는 `bencheval`) 을 그 옆 venv 에. 파일럿에는 필요 없음

### P4 완료 기준 "FM 채점이 TabArena 기준과 맞음" 의 확인 방법 (2026-11-13 정함, 우선 이 기준으로)

- 우리 FM 은 지금 설정 그대로 (Kumo 앙상블 8, 우리 전처리). TabArena 설정에 맞추지 않고 차이를 기록함 ([결정](#결정-2026-11-13) 4)
- 같은 split · 같은 지표로 우리 `tabpfn_v2`·`tabicl`·`kumo` 의 `metric_error` 를 TabArena 의 TabPFNv2·TabICLv2·Kumo-Tabular 와 비교. 범위는 모든 split (r0f0 만이 아님), TabPFNv2 는 돌 수 있는 33개
- 맞다고 보는 기준 (FM 마다 따로)
  - 데이터셋의 80% 이상에서 |우리 split 평균 − TabArena split 평균| ≤ TabArena 의 split 표준편차
  - TabArena 기본 GBDT 와 비교한 승패 (FM 이 나은지) 가 우리 FM 과 TabArena FM 에서 같음.
    비김은 세지 않음: |TabArena FM − 기본 GBDT| ≤ TabArena split 표준편차인 데이터셋은 뺌 (2026-11-14 정함, [결정](#결정-2026-11-13) 9)
- 기준을 못 넘으면 Kumo 의 `n_estimators` 16 으로 한 번 더 (`VARIANTS` 에 `kumo_e16`) 돌려 설정 차이인지 가림. 80%·표준편차 1배는 근거가 약한 첫 기준이라 결과를 보고 다시 정할 수 있음

## 9. 연동 (2026-11-14 구현)

8 의 설계대로 `fab/bench/tabarena/` 를 만듦. 바뀐 곳과 까닭.

| 부분 | 한 것 |
|---|---|
| 데이터 (`driver.py`) | 옆 venv (`FAB_TABARENA/.venv`: openml 0.15.1 · pyarrow · pandas) 에서 `-I` 로. `openml.tasks.get_task` → 데이터 + 고정 분할 (`get_train_test_split_indices`). 반복마다 TEST 가 전체를 한 번씩 나누는지 확인하고, TEST 행 위치만 `splits.csv` 로 넘김. 받은 것은 `FAB_TABARENA/data/cache/<task_id>/` |
| split 대회 (`__init__.py`) | 저장소: `competitions/ta-<데이터셋>/` (r0f0, `config.yaml`·`description.md`·`features.py`). 다른 split 은 `tas-<데이터셋>-r<r>f<f>/` 로 생성 (git 제외, 설정은 r0f0 를 본뜸). r0f0 폴더가 없는 47개는 r0f0 도 `tas-` |
| public | `train.csv`·`test.csv`·`sample_submission.csv`·`description.md`. 행 id 는 데이터셋·split 마다 고정 seed 의 무작위 순열 (train·test 가 한 순열을 나눔 → id 에 정보 없음), 파일은 id 순 (OpenML 순서도 안 줌). CSV 를 pandas 기본값으로 읽으면 범주 값 `nan`·`NA` 같은 글자는 결측이 됨 (TabArena 데이터의 범주 `nan` 은 원래 결측, in_vehicle·MIC·HR_Analytics 등) → FM·에이전트·채점 모두 같은 결측으로 봄. 클래스 이름이 그렇게 바뀌면 `prepare` 가 멈춤 (51개에는 없음) |
| 정답 | `FAB_TABARENA/data/answers/<대회>.{csv,json}` (클래스·양성 클래스·지표) → 채점할 때 내 Volume 의 `tabarena/answers/` 로 (`remote.py`). 에이전트 SP 의 scratch Volume 에는 public 만 |
| 피처 (F0) | 원본 열 그대로. 범주형 = OpenML 범주 표시 + 문자열 열 (in_vehicle 의 `coupon` 은 OpenML 에서 string 이라 표시가 없음). Amazon 의 숫자 id 9열도 범주 (TabArena 와 같은 표시). MLE-bench 의 `auto_types` (자료형으로만 고름) 와 다름 |
| 채점 (`score_submission`) | fab 지표 (`auc`·`logloss`·`rmse`) 로 직접. 이진은 양성 = train 타깃 정렬의 두 번째 (`fab run` 의 LabelEncoder 와 같음), 다중은 확률을 1e-15 아래로 자르고 행 합 1. `metric_error` (AUC 는 1 − AUC) 를 같이 남김 |
| FM·GBDT 제출 (`fab ta fm`) | 데이터셋 하나 = GPU 작업 하나 + CPU 작업 하나. 작업 안에서 split 대회마다 `fab run` 을 차례로 (`fab/runner/databricks.py` 의 `JOBS`). 816 split 을 작업 816개로 내지 않으려고. 대회가 끝날 때마다 결과를 Volume 에 복사 (작업이 중간에 죽어도 끝난 split 은 남음) |
| 원격 채점 | 채점 요청 `bench: tabarena` → 정리 작업이 학습이 끝나기를 기다려 split × 모델마다 채점 → `results/tabarena.csv`. 예측이 없는 (학습 실패) 것은 `missing`, 채점에 실패한 것 (무효 제출, test 예측의 NaN 등) 은 `failed` 로 그 (대회, 모델) 만 빼고 끝냄 (같은 입력이면 다시 해도 같아서 재시도 안 함). 그 밖의 예외는 대회째 되돌리고 다시 시도. 숫자가 아닌 예측 (확률 대신 레이블) 은 무효 제출 |
| 비교 (`fab ta reference`·`compare`) | 공개 parquet 7개 (2 절) 의 기본 설정 행 → `results/tabarena_reference.csv` (5,310행). TabPFNv2 18개·TabICL 15개 데이터셋을 `can_run_*` 로 뺌. 남은 행에 대체값 (`imputed`) 없음. `compare` 의 평균·기본 GBDT 는 고른 split 에서, 표준편차는 늘 데이터셋의 모든 split 에서 (`--splits r0f0` 처럼 하나만 고르면 그 split 의 표준편차가 0 이라서) |
| Agent (`fab ta agent run ta-<데이터셋> <설정>`) | `fab mleb agent` 와 같은 실행기·지시문 (MLE-bench `instructions.txt` 그대로)·프록시. 결과는 `results/tabarena_runs.csv` (열은 `mlebench_runs.csv` 와 같고 대회 열이 `comp`·`anon_id`) |

명령 (설치는 `fab/bench/tabarena/__init__.py` 위쪽 설명, 위치는 `FAB_TABARENA`. Agent 는 `FAB_MLEBENCH` 도 필요: 지시문)

| 명령 | 하는 일 |
|---|---|
| `fab ta prepare <데이터셋...\|all> --splits all` | OpenML 받기 → split 대회 (public 은 `data/`, 정답은 `FAB_TABARENA`). `fab download ta-<데이터셋>` 은 그 split 하나 |
| `fab ta fm <데이터셋...> --models tabpfn_v2 tabicl kumo lgbm catboost --splits all [--accelerator GPU_1xA10] [--max-concurrent 5] [--retries 2]` | 데이터셋마다 GPU·CPU 작업 하나씩 + 채점 요청. 동시 GPU 작업은 `--max-concurrent` 개까지 (아래) |
| `fab ta grade <데이터셋...> --models ...` | 채점 요청만 (원격 정리 작업) |
| `fab ta timing <데이터셋...> --models ...` | 원격 채점 시간 재기: 예전 경로 대 지금 경로 (결과표는 안 바뀜, 아래 "원격 채점 속도") |
| `fab ta reference` | TabArena 공개 결과 → `results/tabarena_reference.csv` |
| `fab ta compare [--models ...] [--splits r0f0] [--detail]` | 데이터셋별 비교 + FM 맞춤 기준 판정 |
| `fab ta agent run ta-<데이터셋> <설정>... [--continue] [--fm]` | Agent (r0f0, 익명 id). `list`·`peek`·`recheck`·`exclude` 는 `fab mleb agent` 와 같음 |
| `fab remote kick` → `fab sync` | 결과 합치기·채점 → `results/tabarena*.csv` |

```bash
export FAB_TABARENA=../tabarena FAB_MLEBENCH=../mle-bench
uv run --frozen fab ta prepare hazelnut-spread-contaminant-detection --splits all
uv run --frozen fab ta fm hazelnut-spread-contaminant-detection --models tabpfn_v2 tabicl kumo lgbm catboost
uv run --frozen fab ta agent run ta-hazelnut-spread-contaminant-detection aide-sonnet aide-gpt   # Sonnet 줄·GPT 줄 하나씩
uv run --frozen fab remote kick && uv run --frozen fab sync
uv run --frozen fab ta compare --detail
```

동시 GPU 작업 상한 (`--max-concurrent`, 2026-11-14, #19)

- 워크스페이스의 GPU 쿼터 (H100 은 8노드, 다른 사람·다른 명령의 작업 포함) 를 넘기면 작업이 2초 만에 `System error ... Workspace has exceeded its GPU quota for GPU_1xH100` 으로 끝남 (아래 "운영 기록")
- 기본 (`--max-concurrent 5`): 이 명령이 낸 GPU 작업의 상태를 `--poll` 초 (기본 60) 마다 보고, 5개 아래일 때만 다음 데이터셋을 냄. GBDT (CPU 작업) 는 세지 않음
- 10분 안에 System error (쿼터 초과 포함) 로 끝난 GPU 작업은 GPU 쪽만 `--retries` 번 (기본 2) 다시 냄. 5분 × 다시 낸 횟수만큼 기다렸다가 (쿼터를 다른 작업이 쓰고 있으면 바로 내도 또 실패).
  노드가 안 떠 45분 뒤 실패한 것 (`An error occurred during execution`) 은 다시 내지 않음 (같은 일이 되풀이될 수 있어 사람이 봄)
- 채점 요청은 데이터셋의 GPU 작업이 (다시 내기까지) 끝난 뒤에 냄. 먼저 내면 정리 작업이 실패한 작업만 보고 "예측 없음" 으로 채점을 끝낼 수 있어서
- **로컬 명령이 끝까지 떠 있어야 함** (데이터셋 12개면 몇 시간). 끊거나 제출이 실패하면 낸 작업은 원격에서 계속 돌고, 아직 안 낸 데이터셋·다시 낼 데이터셋·채점 요청을 아직 안 한 데이터셋을 출력 → `fab ta fm`·`fab ta grade` 로 이어서
- 세는 것은 이 명령이 낸 GPU 작업뿐. Agent (A10) 나 다른 명령이 같은 GPU 종류를 쓰고 있으면 상한을 그만큼 낮춤. `--max-concurrent 0` = 예전처럼 한꺼번에 내고 바로 끝남

익명화 (5 의 "익명화" 를 구현)

- 에이전트에게 보이는 이름은 설정의 `anon_id` (`tabarena-p01` hazelnut, `p02` in_vehicle, `p03`·`p04` 는 Amazon·physiochemical 에 붙여 두었지만 Agent 는 안 돌림): 원격 실행 이름·MLflow run·Volume 의 데이터 묶음 (`data/<anon_id>-<내용 해시>.tgz`: 설명·데이터를 고치면 새 묶음, 옛 묶음을 다시 쓰지 않음)·`run.json`·실행 키
- 대회 이름은 내 Volume 의 제출 기록 (`state/agent/<key>.json` 의 `comp`) 에만. 정리 작업이 이걸로 정답을 찾음
- 제출 전 누출 검사 (`fab/agent/__init__.py` 의 `leak_check`): 에이전트 노드에 가는 코드 묶음 (실행기·지시문·`run.json`)·실행 이름·데이터 파일 이름·`description.md` 에 데이터셋 이름 (구분자 바꾼 꼴 포함)·대회 이름·OpenML task·dataset id 가 있으면 제출 안 함.
  이름은 단어 경계로 봄 (짧은 이름 `mic` 가 `dynamic` 에 걸리지 않게), 숫자는 앞뒤가 숫자가 아닐 때. 검사 전에 `data/description.md` 를 대회 폴더의 것으로 맞춤
- 로컬 확인: 코드 묶음 8개 파일에 TabArena 데이터셋 이름 51개가 하나도 없음 (대응표가 섞이지 않음). 심은 누출 (run.json 에 이름, 설명에 task id) 은 잡음
- `description.md` 는 영어, 이름·출처·연도·수집 기관 없이. 열 뜻은 OpenML 설명·원래 출처를 보고 우리 말로 씀. hazelnut 은 "식품 용기·마이크로파 센서" 까지만 (무슨 식품인지 안 씀). 레이블 부호화가 문서화되지 않은 것 (hazelnut, TabArena 메모) 은 그렇게 적음
- 지시문은 MLE-bench 그대로라 "MLE-bench, an offline version of Kaggle competitions" 라는 글이 들어감 (데이터셋 이름은 아님). 두 벤치마크의 Agent 가 같은 지시문을 받게 둠

사후 검사

- `data_source_urls` (모든 에이전트 실행에 더함): `openml.org`·`import openml`·`from openml... import`·`fetch_openml`·`archive.ics.uci.edu`·`doi.org/10.24432` (UCI DOI)·`ucimlrepo`·`kaggle.com/datasets`·`kaggle.com/c/`·`kaggle.com/competitions/<x>/data`·`kagglehub`·`kaggle datasets|competitions download` 의 수 (코드 + 하네스 기록). MLE-bench 실행은 설명에 Kaggle 주소가 있어 뜻이 없음 → TabArena 실행에서만 봄
- 처음 규칙의 `from\s+openml` 은 Codex 의 생각 글 ("trained on real datasets from OpenML") 에 걸려 import 문만으로 고침 (2026-11-14)
- `pip install --target <폴더>` 로 깐 패키지는 `pip_added` (pip freeze 전·후 비교) 에 안 나옴 → `pip_target` 열을 더함 (2026-11-14, #18).
  실행한 명령의 `pip install` 중 `--target`·`-t`·`--prefix`·`--root` 또는 실행 중 만든 가상환경의 pip 로 깐 패키지 이름. 규칙은 [mle-bench.md](mle-bench.md) 의 "구현" 사후 검사.
  hazelnut × Codex 1회째는 `recheck` 로 tabpfn·tabicl·transformers·tokenizers·safetensors·scikit-learn·einops·huggingface-hub 이 잡힘. TFM 은 실행 중 import 기록 (`tfm_runtime`) 도 잡음

원격 채점 속도 (2026-11-14, #20)

- 먼저 잼: 지난 정리 작업 기록 (단계마다 시각을 찍음). 57분 걸린 것은 FM 합치기 (`fm`, 실행 29개) 28.5분 + 채점 약 27분 (1,809 (대회, 모델), 대회 약 600개).
  43분 걸린 것 (kumo_e16, 모델 하나 × 708 대회) 도 채점 27분 → 모델 수보다 **대회 수에 비례** (대회마다 약 2.3초)
- 대회마다 하던 것: 결과표 5개 읽기 → 모델마다 test.csv 읽기·제출 csv 를 Volume 에 쓰고 앞부분 출력·정답 csv·json 읽기·제출 csv 다시 읽기 → `tabarena.csv` 읽고 쓰기 →
  결과표 5개를 Volume 에 쓰기 (`tables.flush`) → 진행 기록 쓰기
- 고친 것
  - 제출 파일을 만들지 않고 `test.npy` 로 바로 채점 (`submission_frame`·`score_frame`: 예전 `make_submission`·`score_submission` 과 같은 계산), 출력 없음
  - 정답·test id 는 대회마다 한 번 (예전은 모델마다), `test_info` 는 (대회, 모델) 마다 한 번 (예전 세 번)
  - 표는 채점 요청마다 한 번 씀. 되돌리기는 요청 단위 (실패하면 요청째 다시, 예전 진행 기록이 있으면 그 대회는 건너뜀)
  - FM 합치기의 복사를 `copy2` → `copyfile` (권한·시각을 안 복사). 차이는 작음 (아래)
- 재는 법: `fab ta timing <데이터셋> --models ...` → 정리 작업 안에서 같은 (대회, 모델) 을 지금 경로 → 예전 경로 → 지금 경로로 채점 (Volume 읽기 캐시 때문에 뒤에 도는 쪽이 빠를 수 있어 앞뒤로).
  표는 Volume 의 임시 폴더에 쓰고 진짜 표는 되돌림. 두 경로의 점수를 비교

같은 양 (hazelnut 30 split × tabpfn_v2·tabicl·kumo = 90) 을 정리 작업 안에서 두 번 잼 (초)

| | 1번째 | 2번째 | 나눔 (2번째) |
|---|---|---|---|
| 예전 경로 | 65.4 | 62.9 | 표 쓰기 44.0 · 제출 파일·채점 18.8 |
| 지금 경로 (예전보다 먼저) | 19.9 | 20.8 | `test_info` 7.2 · 예측 읽기 4.4 · test id 3.8 · 정답 3.5 · 표 1.5 · 계산 0.4 |
| 지금 경로 (예전 다음, 캐시) | 10.4 | 9.4 | `test_info` 4.8 · 표 1.4 · 나머지 3.2 |
| 두 경로 점수 | 90 / 90 같음 (차이 0) | 90 / 90 같음 | |

- 채점은 약 3배 빠름 (대회당 2.1초 → 0.7초). 대회 600개면 약 21분 → 약 7분으로 어림 (대회 수에 비례한다고 보고)
- 결과값: 로컬에서 예전 코드와 지금 코드로 같은 예측을 채점해 90 / 90 같음. 원격 표의 점수와는 1e-16 정도 차이 (원격·로컬 환경 차이, 예전 코드도 같은 차이). 측정을 돌린 뒤 `results/tabarena.csv` 그대로
- FM 합치기 (실행 하나 = 파일 362개, 30 split × 3 모델 × 4 + results.csv): `copyfile` 83초 → `copy2` 88초 → `copyfile` 66초 (1번째 측정은 `copy2` 115초 → `copyfile` 66초).
  순서 영향이 커서 `copyfile` 이 조금 빠른 정도로 봄. 파일마다 Volume 왕복 (약 0.2초) 이 대부분 → 남은 큰 몫은 합치기 (실행 30개면 30~60분). 줄이려면 복사하지 않고 실행 폴더에서 바로 읽거나 묶어 옮겨야 함 (안 함)
- 정리 작업은 채점 요청마다 걸린 시간 (`seconds`·`timer`) 과 합치기 시간을 `state/{grade,fm}/<key>.done.json` 과 기록에 남김

로컬 확인 (학습 없이, `uv run --frozen` + 저장소 밖 스크립트)

- 파일럿 4개 × 모든 split (57개): 크기, train·test id 가 0..n−1 순열을 나눔, test 의 값·정답이 OpenML TEST 행과 같음, 같은 repeat 의 fold 3개 test 합 = 전체, id 순위와 타깃 상관 |r| < 0.05
- 채점: 상수 제출 AUC 0.5, 정답 제출 AUC 1 (행 순서를 섞어도), 뒤집은 제출 AUC 0 (양성 클래스 확인), 회귀 상수 제출의 RMSE = 식, 행이 빠지거나 열 이름이 틀리면 무효. 가짜 `test.npy` → `make_submission` → 채점 = 직접 계산 (FM 경로)
- 피처: 범주 열이 category (정수 값은 `1` 꼴, train·test 를 따로 읽어 생기는 `1.0` 섞임 없음), 생성 split 의 `features.py` 가 r0f0 와 같은 열
- 합 516개 통과
- 코드 리뷰 반영 뒤 (2026-11-14): 레이블을 낸 제출 → 무효, test 예측에 NaN 인 모델만 빠지고 같은 대회의 다른 모델은 채점, 설명을 고치면 묶음 해시가 바뀜,
  `--splits r0f0` 의 표준편차·기본 GBDT, 모델 이름 오타를 제출 전에 잡음 (`fab ta fm`·`grade`), 누출 검사의 단어 경계 등 19개. 모든 split 판정 결과는 그대로

운영 기록

| 일 | 내용 | 처리 |
|---|---|---|
| H100 동시 7개쯤 | 작업을 한꺼번에 35개 냈더니 8번째부터 `System error occurred during execution` 으로 0분에 실패 (27개). 먼저 낸 7개는 돎, 하나가 끝난 뒤 낸 것은 돎. 실패 메시지 끝에 까닭이 있었음: `Workspace has exceeded its GPU quota for GPU_1xH100. The quota limit for this workspace is 8 node(s)` (2초 만에 끝남, 나중에 확인) | 동시 GPU 작업 5개까지만 내는 순서 제출로 다시 (저장소 밖 스크립트) → `fab ta fm --max-concurrent` 로 넣음 (#19) |
| H100 이 안 뜸 | `An error occurred during execution of task AI Runtime Task` 로 약 45분 뒤 실패, 로그가 하나도 없음 (physiochemical 2번, anneal 1번). 같은 anneal r0f0 를 다시 내도 H100 은 20분 동안 로그 0줄 (취소), A10 은 바로 떠서 split 하나 104초 | 우리 코드가 아니라 노드가 뜨지 않은 것으로 봄. 2만 행 이하 데이터셋은 A10 으로 (점수는 GPU 와 무관, 시간은 다름) |
| zsh 인자 | `--models $m` (`m="tabicl kumo"`) 이 zsh 에서 안 나뉘어 모델 이름 "tabicl kumo" 로 제출됨 | 취소, 다시 냄. 그 채점 요청은 예측 없음으로 끝남 |
| 로컬 확인 행 | 로컬 확인 스크립트가 가짜 모델 행 4개를 `results/tabarena.csv` 에 씀 (표 위치가 `FAB_OUTPUTS` 밖) | 그 로컬 표를 지우고 `fab sync` 로 원격 표를 받음. 스크립트는 임시 위치로 |

## 파일럿 결과 (2026-11-14)

숫자는 `metric_error` (낮을수록 좋음: 이진 1 − AUC, 다중 log loss, 회귀 RMSE). "TabArena" 는 공개 결과의 기본 설정 (3).
FM·GBDT 는 튜닝 없음, 우리 `fab run` 의 test 예측 = train 의 5-fold 모델 5개 평균 (TabArena TFM 은 train 전체로 refit, 아래 "차이").
결과 파일: `results/tabarena.csv` (우리 split 채점), `results/tabarena_reference.csv`, `results/tabarena_runs.csv` (Agent).

### Agent vs FM (r0f0)

hazelnut (`tabarena-p01`): 이진 · train 1,600 / test 800 · 피처 30 (모두 수치)

| 방식 | metric_error | AUC | 시간 |
|---|---|---|---|
| **FM: kumo** | **0.00449** | 0.99551 | 10초 |
| TabArena Kumo-Tabular (참고) | 0.00381 | | |
| Agent: Codex × GPT | 0.00640 | 0.99360 | 58분 (TabICL·TabPFN 을 스스로 깔아 씀, 아래) |
| FM: tabicl | 0.00729 | 0.99271 | 6초 |
| FM: tabpfn_v2 | 0.01491 | 0.98509 | 6초 |
| Agent: AIDE × GPT | 0.01624 | 0.98376 | 60분 |
| Agent: Claude Code × Sonnet +cont | 0.01885 | 0.98115 | 6분 (이어 가기 7번) |
| Agent: AIDE × Sonnet | 0.02108 | 0.97892 | 60분 |
| Agent: Claude Code × Sonnet | 0.02219 | 0.97781 | 37초 (스스로 끝냄) |
| GBDT: lgbm / catboost | 0.02947 / 0.03181 | 0.97053 / 0.96819 | 5초 / 28초 |

in_vehicle (`tabarena-p02`): 이진 · train 8,456 / test 4,228 · 피처 24 (범주 22)

| 방식 | metric_error | AUC | 시간 |
|---|---|---|---|
| **Agent: Codex × GPT** | **0.13150** | 0.86850 | 57분 (FM 안 씀) |
| TabArena Kumo-Tabular (참고) | 0.13716 | | |
| Agent: AIDE × Sonnet | 0.13813 | 0.86187 | 60분 |
| Agent: AIDE × GPT | 0.13883 | 0.86118 | 60분 |
| FM: kumo | 0.14524 | 0.85477 | 15초 |
| Agent: Claude Code × Sonnet +cont | 0.14911 | 0.85089 | 10분 (이어 가기 7번) |
| FM: tabicl | 0.15074 | 0.84926 | 8초 |
| GBDT: catboost | 0.16031 | 0.83969 | 2분 |
| Agent: Claude Code × Sonnet | 0.16192 | 0.83808 | 27초 (스스로 끝냄) |
| GBDT: lgbm | 0.16331 | 0.83669 | 6초 |
| FM: tabpfn_v2 | 0.19410 | 0.80590 | 11초 |

- 1회씩 (반복 없음). r0f0 한 split 의 test 는 800 / 4,228 행 → 차이 0.002 정도는 split 사이 흔들림 (아래 표의 split 표준편차: hazelnut kumo 0.0009, in_vehicle kumo 0.0032) 과 비슷
- hazelnut: FM 한 번 (kumo) 이 모든 Agent 보다 나음. FM 없는 Agent 중 최고는 AIDE × GPT 0.0162 (kumo 의 3.6배 오차). 모든 Agent 가 우리 GBDT 보다는 나음
- in_vehicle: Agent 셋 (Codex·AIDE 둘) 이 FM 최고 (kumo 0.1452) 보다 나음. Codex 는 TabArena 의 Kumo (0.1372) 보다도 나음.
  Codex 는 인구통계 열로 응답자 id 를 다시 만들고 (`person`, 같은 응답자가 여러 시나리오에 답함) 쿠폰 × 상황 교차 피처 + CatBoost·LightGBM·XGBoost·선형 모델 섞기
  (`assemble.py`). 응답자 id 피처는 train+test 를 합쳐 만듦 → TabArena 의 "inductive" 규칙 (5) 으로는 안 되는 풀이
- Codex × GPT 의 hazelnut 풀이는 공식 설정 (TFM 없는 환경) 인데 `pip install --target /tmp/...` 로 `tabpfn==2.0.9`·`tabicl==2.2.0` 을 깔고 HF 에서
  가중치 (TabPFN v2, TabICL v2-20260212) 를 받아 CV·미세 조정·의사 레이블에 씀. 제출 = TabICL 예측 + 의사 레이블 섞기 (`finalize_submission.py`).
  → 이 실행은 "Agent" 가 아니라 사실상 "FM 을 스스로 찾은 Agent". `tfm_runtime` 이 잡음 (TFM import 프로세스 90개), `pip_added` 는 못 잡음 (9 의 "빈틈").
  MLE-bench P0 의 Codex 는 TabPFN 이 깔렸는지만 보고 안 썼음
- Claude Code 공식 설정은 MLE-bench 처럼 1분 안에 스스로 끝냄 (27~37초). +cont 는 이어 가기 상한 (도구 0개 구간 연속 3번) 에 걸려 6~10분
- 사후 검사: 10개 실행 모두 `data_source_urls` 0, `kaggle_urls` 0 (원본 데이터 접근 흔적 없음). 429 재시도는 Codex 만 (100번·19번)
- 토큰 (입력): Codex 4,200만~4,700만, 나머지 9만~64만 (MLE-bench 와 같은 꼴)
- GPU (A10) 를 쓴 실행은 Codex 둘뿐: hazelnut 은 TabICL·TabPFN 을 GPU 로, in_vehicle 은 CatBoost `task_type='GPU'`·torch 신경망 일부 (최종 제출은 GBDT·선형 섞기).
  AIDE × Sonnet hazelnut 은 탐색 중 torch·cuda 를 한 번 시도했고 최종은 SVC. 나머지 7개는 LightGBM·CatBoost·로지스틱·SVC·sklearn MLP 를 CPU 로만
  (작업 디렉터리 코드·하네스 기록에서 `device='cuda'`·`task_type='GPU'`·`gpu_hist`·nvidia-smi 를 셈, 저장소 밖 스크립트)
- Agent 의 A10 은 유지 (2026-11-14 사용자가 정함): MLE-bench 의 기본 설정 (A10 1장 · 1시간) 과 C2 (+fm, TFM 에 GPU 필요) 와 같은 환경으로 두려고.
  GPU 와 인터넷이 있으면 공식 설정에서도 FM 을 스스로 깔 수 있음 (hazelnut 의 Codex) → 그런 실행은 `tfm_runtime` 으로 가려 따로 읽음

### FM 이 TabArena 와 맞는지

범위: 파일럿 4개 + 2만 행 이하 35개 = 39개 데이터셋의 모든 split (690개). TabPFNv2 는 TabArena 가 돈 33개.
2만 행 넘는 12개 (houses·superconductivity·credit_card·bank-marketing·Food_Delivery·kddcup09·diamonds·Diabetes130US·APSFailure·SDSS17·customer_satisfaction·GiveMeSomeCredit) 는
아직 안 돌림 (사용자가 작은 것부터로 정함, 아래 "남은 것").

| 우리 FM | TabArena | 데이터셋 | split 표준편차 안 | 승패 같음 (비김 뺌) | 승패 같음 (비김까지, 참고) | 우리 − TabArena (표준편차 단위, 중앙값) | 우리가 나은 데이터셋 | 판정 (8 의 기준) |
|---|---|---|---|---|---|---|---|---|
| `tabicl` | TabICLv2 | 39 | 37 (94.9%) | 19 / 19 | 39 / 39 | +0.08 | 13 | **통과** |
| `kumo` | Kumo-Tabular | 39 | 37 (94.9%) | 25 / 25 | 38 / 39 | +0.14 | 10 | **통과** |
| `kumo_e16` | Kumo-Tabular | 39 | 37 (94.9%) | 25 / 25 | 38 / 39 | +0.14 | 10 | **통과** |
| `tabpfn_v2` | TabPFNv2 | 33 | 30 (90.9%) | 9 / 9 | 27 / 33 | −0.15 | 23 | **통과** |

- 판정은 비김을 뺀 기준 (2026-11-14 정함, 결정 9). 비김까지 세던 처음 기준으로는 kumo (1개)·tabpfn_v2 (6개) 가 못 넘었음 (아래)
- 비김으로 빠지는 데이터셋이 많음: tabicl 20 / 39, kumo 14 / 39, tabpfn_v2 24 / 33. TabArena 안에서도 FM 과 기본 GBDT 가 split 표준편차 안인 데이터셋이 그만큼
  → 승패 기준은 차이가 뚜렷한 데이터셋 (tabpfn_v2 는 9개뿐) 만 봄. 맞춤의 주 근거는 표준편차 기준

- 표준편차 기준 (80%) 은 셋 다 넘음 (90.9~94.9%)
- 승패가 다른 7개는 모두 우리 FM 과 기본 GBDT 의 차이가 3% 안 (비김에 가까움): tabpfn_v2 는 E-CommereShippingData·Is-this-a-good-customer·MIC·online_shoppers·polish_companies·seismic-bumps
  (우리는 GBDT 보다 0.5~2.8% 나음, TabArena TabPFNv2 는 0.04~6.2% 나쁨. 6% 는 polish 하나), kumo 는 E-CommereShippingData (−0.16% 대 +0.04%)
- 표준편차 밖: tabicl 은 airfoil (+1.05) · hiva_agnostic (+1.35), kumo 는 Amazon (+1.26) · in_vehicle (+1.48), tabpfn_v2 는 airfoil (+1.21) · NATICUSdroid (−1.03) · jm1 (−1.09)
  (+ = 우리가 나쁨). airfoil 은 세 FM 모두 우리가 나쁨 (회귀, 1,503 행)
- 방향: kumo·tabicl 은 대체로 조금 나쁨 (kumo 39개 중 29개, tabicl 26개). tabpfn_v2 는 대체로 조금 나음 (33개 중 23개)
- 처음 기준 (승패가 "모두" 같음) 으로는 kumo·tabpfn_v2 가 못 넘어 결정 4 대로 `kumo_e16` 을 같은 39개에 돌림 (아래, 결과 같음).
  그 뒤 비김을 빼기로 함 (결정 9). 승패가 달랐던 7개는 모두 비김으로 빠짐

차이 (점수가 똑같지 않을 까닭, 3 의 "다른 점" 에 더함)

- test 예측: 우리는 train 의 80% 를 컨텍스트로 한 fold 모델 5개의 평균, TabArena 는 train 전체로 refit 한 모델 하나 → 컨텍스트가 20% 적음. kumo·tabicl 이 조금 나쁜 쪽으로 기우는 것과 맞는 방향
- Kumo 앙상블 8 대 16 (아래 `kumo_e16` 로 봄)
- TabPFNv2: TabArena 쪽은 2025-06-12 묶음 (패키지 버전 확인 못 함), 우리는 tabpfn 9.2.0 의 v2 가중치

파일럿 4개 (모든 split)

| 데이터셋 | 모델 | split | 우리 (평균 ± 표준편차) | TabArena (평균 ± 표준편차) | 표준편차 안 |
|---|---|---|---|---|---|
| hazelnut | kumo | 30 | 0.0032 ± 0.0009 | 0.0026 ± 0.0009 | O |
| | tabicl | 30 | 0.0055 ± 0.0015 | 0.0051 ± 0.0014 | O |
| | tabpfn_v2 | 30 | 0.0115 ± 0.0028 | 0.0121 ± 0.0029 | O |
| | lgbm / catboost | 30 | 0.0234 / 0.0258 | 0.0273 / 0.0258 | O / O |
| in_vehicle | kumo | 9 | 0.1502 ± 0.0030 | 0.1445 ± 0.0039 | X |
| | tabicl | 9 | 0.1583 ± 0.0042 | 0.1562 ± 0.0041 | O |
| | tabpfn_v2 | 9 | 0.2095 ± 0.0109 | 0.2111 ± 0.0071 | O |
| | lgbm / catboost | 9 | 0.1695 / 0.1695 | 0.1642 / 0.1610 | X / X |
| Amazon | kumo | 9 | 0.1304 ± 0.0070 | 0.1223 ± 0.0064 | X |
| | tabicl | 9 | 0.1492 ± 0.0056 | 0.1469 ± 0.0044 | O |
| | tabpfn_v2 | 9 | 0.1730 ± 0.0062 | – (범위 밖) | |
| | lgbm / catboost | 9 | 0.1495 / **0.1132** | 0.1573 / 0.1182 | O / O |
| physiochemical | kumo | 9 | 2.7828 ± 0.0335 | 2.7708 ± 0.0342 | O |
| | tabicl | 9 | 2.9532 ± 0.0323 | 2.9428 ± 0.0362 | O |
| | tabpfn_v2 | – | 못 돌림 (H100 노드가 두 번 안 뜸, 9 의 "운영 기록"). TabArena 범위 밖이라 비교 대상 없음 | | |
| | lgbm / catboost | 9 | 3.9391 / 3.9525 | 3.4769 / 3.5213 | X / X |

- Amazon 은 TabArena 처럼 기본 CatBoost 가 모든 FM 보다 나음 (우리 CatBoost 0.1132 가 TabArena 의 0.1182 보다도 나음)
- GBDT 는 참고선 (결정 7): 우리 GBDT 는 고정 반복 (lgbm 800 × lr 0.02, catboost 1500 × lr 0.03, 조기 종료 없음) 이라 큰 회귀 (physiochemical, train 3만) 에서 TabArena 보다 13% 나쁨

### kumo_e16

결정 4 대로 Kumo 의 앙상블을 TabArena 와 같은 16 으로 (`fab/models` 의 `VARIANTS` 에 `kumo_e16`) 같은 39개 · 690 split 에 다시 돌림 (A10).

| | split 표준편차 안 | 승패 같음 | 우리 − TabArena (표준편차 단위, 중앙값) | 우리가 나은 데이터셋 | 모델 시간 합 |
|---|---|---|---|---|---|
| `kumo` (앙상블 8) | 37 / 39 | 38 / 39 | +0.14 | 10 | 3.3시간 |
| `kumo_e16` (앙상블 16) | 37 / 39 | 38 / 39 | +0.14 | 10 | 6.6시간 |

- 결과가 거의 같음: 데이터셋별 split 평균의 변화 중앙값 −0.01%, 16 이 나은 데이터셋 25 / 39. 못 넘는 데이터셋도 같음
  (Amazon 0.1304 → 0.1305 · in_vehicle 0.1502 → 0.1502 은 표준편차 밖, E-CommereShippingData 는 비김으로 승패가 다름)
- → 앙상블 수는 TabArena 와의 차이를 설명하지 못함. 남은 후보는 test 예측의 컨텍스트 (우리는 train 의 80% 로 fit 한 fold 모델 5개 평균, TabArena 는 train 전체 refit) 와
  전처리·패키지 판 (3 의 "다른 점"). 컨텍스트 쪽은 train 전체로 fit 한 test 예측을 따로 만들어야 가림 (지금 `fab run` 에 없음)
- 우리 `kumo` 는 앙상블 8 그대로 둠 (결정 4: MLE-bench·C1·C2 와 같은 FM). 비용만 2배

### 비용·시간

| 작업 | GPU | 벽시계 | 비고 |
|---|---|---|---|
| FM 파일럿 (4개 × 57 split × TFM 3) | H100 × 4 작업 | 9~19분 / 작업 | physiochemical 은 실패 2번 (노드), tabicl·kumo 를 따로 다시 |
| GBDT 파일럿 (4개 × 57 split × 2) | serverless CPU × 4 | 6~29분 | |
| FM 맞춤 35개 (651 split) | H100 7개 (동시 한도까지) + A10 28개 | 8~63분 / 작업 | 넓은 데이터 (Bioresponse·hiva·QSAR-TID-11) 가 약 1시간 |
| kumo_e16 (39개, 690 split) | A10 39 작업 (동시 5개) | 벽시계 약 2시간 (14:35~16:39) | 모델 시간 6.6시간 |
| Agent 10회 | A10 × 10 | 실행당 1시간 + 준비 | Sonnet 줄·GPT 줄 하나씩 동시에, 벽시계 약 4시간 30분 (09:00~13:30) |

- FM 모델 시간 합 (`results.csv` 의 seconds): kumo 3.3시간 (709 split-실행). 51개 전부의 어림 (H100 12~26시간) 은 큰 12개가 대부분

### 데이터셋별 GPU

채점에 들어간 FM 실행이 어느 GPU 에서 돌았는지. 기본은 H100 1장 (`fab/runner/databricks.py` 의 `DEFAULT_ACCELERATOR`, `databricks.local.yaml` 에 `accelerator` 없음).
11:40 부터는 H100 이 잘 뜨지 않아 (9 의 "운영 기록") `--accelerator GPU_1xA10` 으로 냄. GBDT 는 모두 serverless CPU, Agent 는 모두 A10 (MLE-bench 와 같은 조건).

| GPU | 모델 | 데이터셋 | 수 |
|---|---|---|---|
| H100 | tabpfn_v2 · tabicl · kumo | hazelnut-spread-contaminant-detection · in_vehicle_coupon_recommendation · Amazon_employee_access (파일럿, 08:55 제출) | 3 |
| H100 | tabicl · kumo | physiochemical_protein (09:52 재실행. tabpfn_v2 는 H100 에서 두 번 실패해 결과 없음) | 1 |
| H100 | tabpfn_v2 · tabicl · kumo | airfoil_self_noise · Another-Dataset-on-used-Fiat-500 · Bank_Customer_Churn · blood-transfusion-service-center · churn · coil2000_insurance_policies · QSAR_fish_toxicity (35개를 한꺼번에 낸 것 중 뜬 7개, 10:19~10:35 제출) | 7 |
| A10 | tabpfn_v2 · tabicl · kumo | anneal · concrete_compressive_strength · credit-g · diabetes · E-CommereShippingData · Fitness_Club · healthcare_insurance_expenses · heloc · Is-this-a-good-customer · Marketing_Campaign · maternal_health_risk · miami_housing · NATICUSdroid · online_shoppers_intention · polish_companies_bankruptcy · qsar-biodeg · seismic-bumps · splice · students_dropout_and_academic_success · taiwanese_bankruptcy_prediction · website_phishing · wine_quality · MIC · jm1 (11:44~13:42) | 24 |
| A10 | tabicl · kumo (TabPFNv2 범위 밖) | Bioresponse · hiva_agnostic · HR_Analytics_Job_Change_of_Data_Scientists · QSAR-TID-11 | 4 |
| A10 | kumo_e16 | 위 39개 모두 (14:35~16:39) | 39 |

- 점수는 GPU 와 무관하다고 보고 같은 표에 섞음 (같은 데이터셋을 두 GPU 로 돌려 비교해 보지는 않음). 시간 (`results.csv` 의 `seconds`) 은 GPU 에 따라 다르므로 데이터셋끼리 시간을 비교할 때는 이 표를 봄
- `kumo` (H100 11개 · A10 28개) 와 `kumo_e16` (모두 A10) 은 H100 11개에서는 GPU 도 다름. 그래도 결과가 거의 같음 (위 "kumo_e16")

### 남은 것

| 항목 | 이슈 | 상태 |
|---|---|---|
| 2만 행 넘는 12개 데이터셋 (FM 맞춤 확인용, tabicl·kumo). H100 이 필요할 수 있음 (컨텍스트 최대 8만 행), 어림 H100 10~20시간 | [#16](https://github.com/Aiden-Jeon/fm-agent-bench/issues/16) | 보류 |
| Agent 2회째 (결정 2: 같은 r0f0 에서, 10회) | [#17](https://github.com/Aiden-Jeon/fm-agent-bench/issues/17) | |
| 사후 검사가 `pip install --target` 을 못 잡음 (9 의 "빈틈") | [#18](https://github.com/Aiden-Jeon/fm-agent-bench/issues/18) | |
| `fab ta fm` 의 동시 GPU 작업 상한 (지금은 저장소 밖 스크립트) | [#19](https://github.com/Aiden-Jeon/fm-agent-bench/issues/19) | |
| 원격 채점 속도 (정리 작업 한 번 약 1시간) | [#20](https://github.com/Aiden-Jeon/fm-agent-bench/issues/20) | |
| physiochemical 의 `tabpfn_v2` (TabArena 범위 밖이라 비교에는 필요 없음) | – | 안 함 |
| Kumo·TabICL 이 TabArena 보다 조금 나쁜 쪽으로 기우는 까닭 (컨텍스트 80% 대 refit) | – | 필요하면 |

## 결정 (2026-11-13)

3·4·9 는 사용자가 정함. 나머지는 추천안을 그대로 받음 (바꾸려면 여기를 고침).

| # | 정할 것 | 정한 것 | 까닭 |
|---|---|---|---|
| 1 | v0.1 대 BeyondArena | **TabArena-v0.1 (51개)** | 우리 FM 셋의 split 별 결과가 모두 공개돼 있어 "TabArena 와 맞나" 를 바로 봄. BeyondArena 는 시간·그룹 분할이라 지금 `fab run` 의 KFold 로는 샘 (그쪽 TFM 결과 유무는 확인 안 함). 데이터 받는 부분은 따로 떼어 나중에 옮길 수 있게 |
| 2 | Agent split | **r0f0 하나, 반복은 같은 split 에서 2회째** | MLE-bench P1 에서 Agent 는 실행끼리 흔들림이 split 보다 큼. FM 은 모든 split, Agent 와의 비교는 r0f0 점수끼리 |
| 3 | 데이터셋 이름 공개 | **안 알림 — "모델링 실력" 비교** | 타깃이 공개라 이름·출처가 있으면 정답 찾기가 쉬움. 과제·열 뜻·지표만 줌. 인터넷은 열고 사후 검사 (5 의 "익명화") |
| 4 | FM 설정 | **지금 설정 그대로, 차이만 기록. 기준은 8 의 "확인 방법"** | MLE-bench 결과·C1·C2 와 같은 FM 이어야 이어 읽음. 기준 (80%·표준편차 1배) 은 우선 이대로 진행 |
| 5 | 대회 단위 | **A: (데이터셋, split) 마다 대회. 저장소에는 r0f0 만** | 지금 코드 경로를 그대로 씀 |
| 6 | 의존성 | **옆 venv (`FAB_TABARENA`) 에 `openml`·`pyarrow`** | `uv.lock` 안 바뀜 (사내 프록시), mlebench driver 와 같은 방식 |
| 7 | GBDT 기준선 | **우리 `lgbm`·`catboost` 를 돌림, TabArena 기본 GBDT 는 참고선** | Agent·C1 과 같은 조건의 대조군은 우리 것뿐. 비용 거의 없음 |
| 8 | 파일럿 | **FM 은 4개 (위 표) + 맞춤 확인용 51개, Agent 는 hazelnut · in_vehicle_coupon 부터** | 7 의 "파일럿 추천" |
| 9 | 맞춤 기준의 비김 (2026-11-14 추가) | **\|TabArena FM − 기본 GBDT\| ≤ TabArena split 표준편차인 데이터셋은 승패를 세지 않음** | 처음 기준에서 승패가 달랐던 7개가 모두 FM 과 GBDT 의 차이 3% 안. split 흔들림 안의 승패는 맞춤의 근거가 못 됨 |

확인 못 한 것

- 리더보드 화면의 Elo·순위 (HF Space 가 동적)
- TabArena TabPFNv2 (2025-06-12) 의 tabpfn 버전·`n_estimators`, 우리 `tabicl` 2.2.0 의 기본 체크포인트가 TabICLv2 (20260212) 와 같은지
- TabArena 실행 GPU 종류
- 데이터셋별 공개 풀이 수 (오염 정도)
- 원래 출처 페이지 설명을 옮겨 써도 되는 라이선스인지 (출처별)
- BeyondArena 가 v0.1 을 언제 대체하는지

## 부록: 데이터셋 51개

행 순. 피처는 메타데이터 `num_features`. split = 외부 split 수. TabPFNv2 = 메타데이터 `can_run_tabpfnv2`.
마지막 세 열은 공개 결과의 split 평균 `metric_error` 를 기본 GBDT (CatBoost·LightGBM 중 나은 쪽) 와 비교한 % (음수 = FM 이 나음, – = 결과 없음 또는 RF 대체값):
pfn2 = TabPFNv2 (2025-06-12), icl2 = TabICLv2 (2026-09-16), kumo = Kumo-Tabular (2026-10-06).

| 데이터셋 | task | 과제 | 행 | 피처 | 클래스 | split | 출처 | 라이선스 | TabPFNv2 | pfn2 | icl2 | kumo |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| blood-transfusion-service-center | 363621 | 이진 | 748 | 5 | 2 | 30 | UCI | CC BY 4.0 | O | -6 | -7 | -5 |
| diabetes | 363629 | 이진 | 768 | 9 | 2 | 30 | OpenML | CC0: Public Domain | O | -6 | -3 | -6 |
| anneal | 363614 | 다중 | 898 | 39 | 5 | 30 | UCI | CC BY 4.0 | O | -60 | -48 | -71 |
| QSAR_fish_toxicity | 363698 | 회귀 | 907 | 7 |  | 30 | UCI | CC BY 4.0 | O | -1 | -2 | -3 |
| credit-g | 363626 | 이진 | 1,000 | 21 | 2 | 30 | UCI | CC BY 4.0 | O | +6 | -4 | -9 |
| maternal_health_risk | 363685 | 다중 | 1,014 | 7 | 3 | 30 | UCI | CC BY 4.0 | O | -5 | -17 | -21 |
| concrete_compressive_strength | 363625 | 회귀 | 1,030 | 9 |  | 30 | UCI | CC BY 4.0 | O | +1 | -6 | -10 |
| qsar-biodeg | 363696 | 이진 | 1,054 | 42 | 2 | 30 | UCI | CC BY 4.0 | O | -8 | -15 | -15 |
| healthcare_insurance_expenses | 363675 | 회귀 | 1,338 | 7 |  | 30 | Kaggle | Database Contents License (DbCL) v1.0 | O | +4 | -2 | -3 |
| website_phishing | 363707 | 다중 | 1,353 | 10 | 3 | 30 | UCI | CC BY 4.0 | O | -9 | -12 | -17 |
| Fitness_Club | 363671 | 이진 | 1,500 | 7 | 2 | 30 | Kaggle | Public Domain | O | -4 | -4 | -4 |
| airfoil_self_noise | 363612 | 회귀 | 1,503 | 6 |  | 30 | UCI | CC BY 4.0 | O | -28 | -31 | -42 |
| Another-Dataset-on-used-Fiat-500 | 363615 | 회귀 | 1,538 | 8 |  | 30 | Kaggle | CC0: Public Domain | O | -1 | -3 | -5 |
| MIC | 363711 | 다중 | 1,699 | 112 | 8 | 30 | UCI | CC BY 4.0 | O | +3 | -2 | -7 |
| Is-this-a-good-customer | 363682 | 이진 | 1,723 | 14 | 2 | 30 | Kaggle | CC0: Public Domain | O | +1 | +0 | -3 |
| Marketing_Campaign | 363684 | 이진 | 2,240 | 26 | 2 | 30 | Kaggle | Public | O | -9 | -20 | -22 |
| hazelnut-spread-contaminant-detection | 363674 | 이진 | 2,400 | 31 | 2 | 30 | OpenML | CC BY-SA | O | -53 | -80 | -90 |
| seismic-bumps | 363700 | 이진 | 2,584 | 16 | 2 | 9 | UCI | CC BY 4.0 | O | +0 | -6 | -7 |
| splice | 363702 | 다중 | 3,190 | 61 | 3 | 9 | UCI | CC BY 4.0 | O | -4 | -10 | -21 |
| Bioresponse | 363620 | 이진 | 3,751 | 1777 | 2 | 9 | Kaggle | Public Domain | – | – | +8 | -11 |
| hiva_agnostic | 363677 | 다중 | 3,845 | 1618 | 3 | 9 | Other | Public | – | – | +14 | -0 |
| students_dropout_and_academic_success | 363704 | 다중 | 4,424 | 37 | 3 | 9 | UCI | CC BY 4.0 | O | -3 | -3 | -8 |
| churn | 363623 | 이진 | 5,000 | 20 | 2 | 9 | OpenML | MIT License | O | -5 | -15 | -7 |
| QSAR-TID-11 | 363697 | 회귀 | 5,742 | 1025 |  | 9 | OpenML | Public | – | – | +5 | -7 |
| polish_companies_bankruptcy | 363694 | 이진 | 5,910 | 65 | 2 | 9 | UCI | CC BY 4.0 | O | +6 | -60 | -89 |
| wine_quality | 363708 | 회귀 | 6,497 | 13 |  | 9 | UCI | CC BY 4.0 | O | +11 | -4 | -6 |
| taiwanese_bankruptcy_prediction | 363706 | 이진 | 6,819 | 95 | 2 | 9 | UCI | CC BY 4.0 | O | +4 | -4 | -21 |
| NATICUSdroid | 363689 | 이진 | 7,491 | 87 | 2 | 9 | UCI | CC BY 4.0 | O | +21 | -13 | -16 |
| coil2000_insurance_policies | 363624 | 이진 | 9,822 | 86 | 2 | 9 | UCI | CC BY 4.0 | O | +2 | -6 | -11 |
| Bank_Customer_Churn | 363619 | 이진 | 10,000 | 11 | 2 | 9 | Kaggle | Public | O | -1 | -1 | -5 |
| heloc | 363676 | 이진 | 10,459 | 24 | 2 | 9 | Kaggle | Public | O | -1 | -2 | -2 |
| jm1 | 363712 | 이진 | 10,885 | 22 | 2 | 9 | OpenML | Public Domain | O | +6 | -10 | -15 |
| E-CommereShippingData | 363632 | 이진 | 10,999 | 11 | 2 | 9 | Kaggle | Public Domain | O | +0 | -0 | +0 |
| online_shoppers_intention | 363691 | 이진 | 12,330 | 18 | 2 | 9 | UCI | CC BY 4.0 | O | +0 | -5 | -8 |
| in_vehicle_coupon_recommendation | 363681 | 이진 | 12,684 | 25 | 2 | 9 | UCI | CC BY 4.0 | O | +31 | -3 | -10 |
| miami_housing | 363686 | 회귀 | 13,776 | 16 |  | 9 | OpenML | CC BY-NC-SA | O | +8 | -3 | -11 |
| HR_Analytics_Job_Change_of_Data_Scientists | 363679 | 이진 | 19,158 | 13 | 2 | 9 | Kaggle | CC0: Public Domain | – | – | -1 | -2 |
| houses | 363678 | 회귀 | 20,640 | 9 |  | 9 | Other | Public | – | – | -8 | -15 |
| superconductivity | 363705 | 회귀 | 21,263 | 82 |  | 9 | UCI | CC BY 4.0 | – | – | -5 | -9 |
| credit_card_clients_default | 363627 | 이진 | 30,000 | 24 | 2 | 9 | UCI | CC BY 4.0 | – | – | -3 | -4 |
| Amazon_employee_access | 363613 | 이진 | 32,769 | 10 | 2 | 9 | Kaggle | Public Domain | – | – | +24 | +3 |
| bank-marketing | 363618 | 이진 | 45,211 | 14 | 2 | 9 | UCI | CC BY 4.0 | – | – | +3 | -2 |
| Food_Delivery_Time | 363672 | 회귀 | 45,451 | 10 |  | 9 | Kaggle | Database Contents License (DbCL) v1.0 | – | – | +3 | -2 |
| physiochemical_protein | 363693 | 회귀 | 45,730 | 10 |  | 9 | UCI | CC BY 4.0 | – | – | -15 | -20 |
| kddcup09_appetency | 363683 | 이진 | 50,000 | 213 | 2 | 9 | Other | Public | – | – | +10 | -7 |
| diamonds | 363631 | 회귀 | 53,940 | 10 |  | 9 | Other | MIT License | – | – | -4 | -7 |
| Diabetes130US | 363630 | 이진 | 71,518 | 48 | 2 | 9 | UCI | CC BY 4.0 | – | – | +2 | -3 |
| APSFailure | 363616 | 이진 | 76,000 | 171 | 2 | 9 | UCI | CC BY 4.0 | – | – | -23 | -39 |
| SDSS17 | 363699 | 다중 | 78,053 | 12 | 3 | 9 | Kaggle | Public Domain | – | – | -5 | -20 |
| customer_satisfaction_in_airline | 363628 | 이진 | 129,880 | 22 | 2 | 9 | Kaggle | CC0: Public Domain | – | – | -9 | -24 |
| GiveMeSomeCredit | 363673 | 이진 | 150,000 | 11 | 2 | 9 | Kaggle | Public | – | – | -1 | -2 |
