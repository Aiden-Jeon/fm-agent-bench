# TabArena (P4): 조사 · 연동 · 파일럿

TabArena 를 이 저장소의 두 번째 벤치마크로 붙임. 1~8 은 연동 전 조사 (2026-11-13), 9 는 연동 구현, "파일럿 결과" 는 2026-11-14 실행 (Agent 2회째는 2026-11-14~15), "Agent 확장 1회째" 는 2026-11-16~17 (#26), "행 추가: kddcup09_appetency" 는 2026-11-19~20.

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
- [FM 프로토콜과 이름 (2026-11-15)](#fm-프로토콜과-이름-2026-11-15)
- [파일럿 결과 (2026-11-14~15)](#파일럿-결과-2026-11-1415)
- [Agent 확장 1회째 (#26, 2026-11-16)](#agent-확장-1회째-26-2026-11-16)
- [범주 비율과 FM − GBDT (51개, 2026-11-17)](#범주-비율과-fm--gbdt-51개-2026-11-17)
- [범주 대 오염: 알아볼 수 없게 바꾼 3개 (2026-11-17)](#범주-대-오염-알아볼-수-없게-바꾼-3개-2026-11-17)
- [C2 (+fm): FM 을 받은 Agent (12개 r0f0, 2026-11-18 · 11-19)](#c2-fm-fm-을-받은-agent-12개-r0f0-2026-11-18--11-19)
- [C1 준비: 쓸 Agent 실행과 최종 제출의 계보 (2026-11-18)](#c1-준비-쓸-agent-실행과-최종-제출의-계보-2026-11-18)
- [C1: 에이전트 피처 → FM (r0f0, 2026-11-18)](#c1-에이전트-피처--fm-r0f0-2026-11-18)
- [행 추가: kddcup09_appetency (r0f0, 2026-11-19 · 11-20)](#행-추가-kddcup09_appetency-r0f0-2026-11-19--11-20)
- [Elo (2026-11-16)](#elo-2026-11-16)
- [빠르게 돌리기 (2026-11-14)](#빠르게-돌리기-2026-11-14)
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
| 우리 FM | tabpfn_v2 는 공식 범위 안 33개, tabicl·kumo 는 51개 전부 (TabArena 의 TabICLv2·Kumo-Tabular 가 51개 다 돎). 2026-11-15 부터 FM·GBDT 모두 refit, Kumo 는 범주 열을 범주로 (결정 11) |
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
- `fab run` 의 5-fold (`folds.csv`) 는 그 split 의 train 안에서 도는 내부 CV 로 남김. test 예측도 이 fold 모델 5개 (train 의 80% 씩) 의 평균이라
  TabArena TFM (train 전체로 refit, 아래 3) 과 다름 → 처음엔 39개는 이대로, 2만 행 넘는 12개는 refit (모델 이름 `_refit`, 결정 10·"빠르게 돌리기").
  2026-11-15 부터 FM·GBDT 모두 refit (CV 없음, 결정 11). 5-fold 는 옛 결과와 CV 를 볼 때만
  (처음 설계 때 "FM 은 train 전체로 fit 이라 test 점수에 영향 없음" 이라고 적었으나 틀림, 2026-11-14 고침)
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
- test 예측 방식 (2026-11-14 추가, [결정](#결정-2026-11-13) 10·11): 판정은 refit (train 전체로 한 번, `tabicl_kv_refit`·`kumo_cat_refit`·`tabpfn_v2_refit`).
  test 예측을 내는 방식이 TabArena TFM 과 같음. 2026-11-15 부터 Agent vs FM · MLE-bench · C1 에 쓰는 FM 도 refit 이라 (결정 11) 판정이 실험의 FM 을 그대로 덮음.
  Kumo 는 범주 열을 범주로 넘기도록 고친 뒤 판정 (결정 11, "파일럿 결과" 의 "kddcup09"). 2026-11-14 의 5-fold 판정 (39개) 은 옛 프로토콜의 기록
- 기준을 못 넘으면 Kumo 의 `n_estimators` 16 으로 한 번 더 (`VARIANTS` 에 `kumo_e16`) 돌려 설정 차이인지 가림. 80%·표준편차 1배는 근거가 약한 첫 기준이라 결과를 보고 다시 정할 수 있음 (2026-11-15: 앙상블·청크·refit 은 kddcup09 를 설명하지 못하고 범주 처리가 대부분을 설명함, "kddcup09")

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
| `fab ta fm <데이터셋...> --models tabpfn_v2_refit tabicl_kv_refit kumo_cat_refit lgbm_refit catboost_refit --splits all [--accelerator GPU_1xA10] [--max-concurrent 5] [--retries 2]` | 데이터셋마다 GPU·CPU 작업 하나씩 + 채점 요청. 동시 GPU 작업은 `--max-concurrent` 개까지 (아래) |
| `fab ta grade <데이터셋...> --models ...` | 채점 요청만 (원격 정리 작업) |
| `fab ta timing <데이터셋...> --models ...` | 원격 채점 시간 재기: 예전 경로 대 지금 경로 (결과표는 안 바뀜, 아래 "원격 채점 속도") |
| `fab ta reference` | TabArena 공개 결과 → `results/tabarena_reference.csv` |
| `fab ta compare [--models ...] [--splits r0f0] [--detail]` | 데이터셋별 비교 + FM 맞춤 기준 판정 |
| `fab ta agent run ta-<데이터셋> <설정>... [--continue] [--fm]` | Agent (r0f0, 익명 id). `list`·`peek`·`recheck`·`exclude` 는 `fab mleb agent` 와 같음 |
| `fab remote kick` → `fab sync` | 결과 합치기·채점 → `results/tabarena*.csv` |

```bash
export FAB_TABARENA=../tabarena FAB_MLEBENCH=../mle-bench
uv run --frozen fab ta prepare hazelnut-spread-contaminant-detection --splits all
uv run --frozen fab ta fm hazelnut-spread-contaminant-detection --models tabpfn_v2_refit tabicl_kv_refit kumo_cat_refit lgbm_refit catboost_refit
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

- 에이전트에게 보이는 이름은 설정의 `anon_id` (`tabarena-p01` hazelnut, `p02` in_vehicle, `p03`·`p04` 는 Amazon·physiochemical 에 붙여 둠. 이때는 Agent 를 안 돌렸고 #26 부터 돌림): 원격 실행 이름·MLflow run·Volume 의 데이터 묶음 (`data/<anon_id>-<내용 해시>.tgz`: 설명·데이터를 고치면 새 묶음, 옛 묶음을 다시 쓰지 않음)·`run.json`·실행 키
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
- 작업 디렉터리 코드를 볼 때 에이전트가 `pip install --target` 으로 작업 디렉터리 안에 깐 패키지의 코드는 뺌 (2026-11-16, #26).
  polish × Codex 가 `deps_ignore/`·`tabpfn_pkg/` 에 sklearn 을 깔아 그 `datasets` 코드 (`fetch_openml`·`openml.org`·`archive.ics.uci.edu` 주소) 가 `data_source_urls` 에 271번 걸렸음 (에이전트가 쓴 코드·하네스 기록·LLM 요청에는 0).
  규칙: 작업 디렉터리 안의 `*.dist-info/RECORD` (pip 가 깐 파일 목록) 에 적힌 파일만 안 봄 (`fab/agent` 의 `installed_files`). 폴더 깊이로 짐작하지 않음 →
  `--target .` 으로 작업 디렉터리 맨 위에 깔아도 에이전트 코드 (하위 폴더 포함) 는 보고, `--target` 폴더 맨 위에 깔린 단일 모듈 (`six.py` 등) 은 뺌. `*.egg-info` (`pip install -e .` 로 깐 에이전트 자기 패키지) 는 빼지 않음.
  처음 규칙 (`*.dist-info`·`*.egg-info` 가 있는 폴더의 하위 폴더 안 파일은 안 봄) 은 `--target .` 이면 하위 폴더의 에이전트 코드까지 빠져 PR #29 리뷰에서 바꿈 (2026-11-17)
  `recheck` (처음 규칙, 2026-11-16) 뒤 바뀐 것은 그 실행 하나 (`data_source_urls` 271 → 0, `tfm_in_code` 의 autogluon (깐 패키지 안의 이름) 빠짐). TabArena·MLE-bench 의 다른 실행은 값이 같음.
  바뀐 규칙으로 `recheck` 를 다시 함 (2026-11-17, main `ccffa44`): `tabarena_runs.csv` 70행 · `mlebench_runs.csv` 53행 모두 처음 규칙의 값과 같음 (polish × Codex 의 `data_source_urls` 0 그대로).
  → 지금까지 작업 디렉터리 맨 위에 패키지를 깐 실행 (`--target .`) 은 없어서 처음 규칙에서도 빠진 에이전트 코드는 없었음. 바뀐 규칙은 앞으로를 위한 것

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

## FM 프로토콜과 이름 (2026-11-15)

결정 11 로 FM·GBDT 의 test 예측을 refit 으로, tabicl·kumo 의 기본값을 바꿈. 옛 5-fold 결과와 새 결과가 결과표에서 섞이지 않게 이렇게 가름.

| 가르는 것 | 어디에 | 규칙 |
|---|---|---|
| test 예측 방식 | 모델 이름 | 끝이 `_refit` (예: `kumo_cat_refit`, `catboost_refit~f2`, `kumo_cat_refit@rand200000~f1`) = refit. 없으면 5-fold (fold 모델 5개 평균). `results.csv` 의 `n_folds` 0 · `cv` 빈칸도 refit |
| 모델 설정 | `opts` 열 (`results.csv`·`tabarena.csv`·`mlebench.csv`, `test.json`) | 실행할 때의 설정 글 (`fab.models.model_opts`). 예: `cats=categorical chunk=8192 n_estimators=8` (kumo), `kv_cache=True` (tabicl), `default` (GBDT·tabpfn). **빈칸 = 2026-11-15 전 실행** (tabicl 캐시 없음, kumo 청크 2048 · 범주를 수치로. 이름이 `_kv`·`_c8k` 면 그 설정) |
| 바뀐 기본값 | `fab/models` | `tabicl`: 캐시 켬 (옛 동작 `tabicl_nokv`). `kumo`: 청크 8192 · 범주를 범주로 (옛 동작 `kumo_c2k` 의 청크, `kumo_numcat` 의 범주). 옛 진단 이름 `kumo_e4`·`kumo_e16`·`kumo_c8k`·`kumo_c2k`·`kumo_c32k` 는 범주를 수치로 고정 (그 이름의 옛 결과와 같은 뜻) |
| 섞임 경고 | `fab ta compare` | 한 모델 이름의 행이 실제로 다른 설정으로 섞이면 경고. `opts` 글을 그대로 비교하지 않고 빈칸·글에 없는 키를 "그 인자가 생기기 전 값" (`fab.models` 의 `OPTS_BEFORE`) 으로 채워 비교 (`effective_opts`) → 글 모양만 다른 옛 행 (빈칸·`cats=` 없음) 은 안 셈. 팩토리에 기본값 있는 인자를 더하거나 기본값을 바꾸면 `OPTS_BEFORE` 에 그 전 값을 적음 |
| `opts` 읽을 때 | | `cats=` 가 없는 kumo 의 `opts` (2026-11-15 오전 kddcup09 의 `kumo_e16_refit`·`kumo_c2k_refit`·`kumo_c8k`) = 범주를 수치로. tabicl 의 `kv_cache=True` 는 설정 값이라 TabICL 이 캐시를 거부하거나 (클래스가 10 을 넘는 계층 분류) OOM 이면 실제로는 캐시 없이 돎 (점수는 같음). MLE-bench 묶음 (평균·nomad 두 타깃) 의 `opts` 는 구성원마다 `|` 로, 옛 구성원은 `(빈칸)` |

새 결과 이름 (두 벤치마크 같음): `tabpfn_v2_refit` · `tabicl_kv_refit` · `kumo_cat_refit` · `lgbm_refit` · `xgb_refit` · `catboost_refit` (+ `~f1`·`~f2s`·`~f2`, `@rand200000`).
`tabicl_kv_refit` 은 51개 판정에 쓴 이름 그대로 (기본 `tabicl_refit` 과 같은 설정). `kumo_cat_refit` 은 기본 `kumo_refit` 과 같은 설정이지만 `kumo`·`kumo_refit` 은 옛 행이 있어 새 이름을 씀.
범주 열이 없는 데이터셋에서는 `kumo_cat_refit` 과 `kumo_c8k_refit` 이 같은 설정 (범주 처리만 다름) → 그 17개는 `kumo_c8k_refit` 값을 그대로 씀.
접미사 없는 이름 (`kumo`·`tabicl`·`lgbm` 등) 의 옛 행은 5-fold 이고 지우지 않음 (옛 표·해석의 근거). 새로 5-fold 를 돌릴 일이 있으면 `opts` 로 가름

## 파일럿 결과 (2026-11-14~15)

숫자는 `metric_error` (낮을수록 좋음: 이진 1 − AUC, 다중 log loss, 회귀 RMSE). "TabArena" 는 공개 결과의 기본 설정 (3).
FM·GBDT 는 튜닝 없음. 2026-11-15 부터 FM·GBDT 는 모두 refit (CV 없이 train 전체로 한 번 fit → test 만, TabArena TFM 의 `refit_folds=True` 와 같은 꼴, 결정 11).
"Agent vs FM" 의 FM·GBDT 도 refit 값이고 (옛 5-fold 값은 표 오른쪽 열), "FM 이 TabArena 와 맞는지" 의 판정도 refit (tabicl·kumo 51개, tabpfn_v2 33개).
Kumo 는 범주 열을 범주로 넘기도록 고친 뒤 값 (`kumo_cat_refit`, 아래 "kddcup09"). 5-fold 표 (39개) 는 옛 프로토콜의 기록으로 남김.
결과 파일: `results/tabarena.csv` (우리 split 채점), `results/tabarena_reference.csv`, `results/tabarena_runs.csv` (Agent), `results/tabarena_elo.csv` (아래 "Elo").

### Agent vs FM (r0f0)

hazelnut (`tabarena-p01`): 이진 · train 1,600 / test 800 · 피처 30 (모두 수치)

FM·GBDT 는 한 번 (결정적), Agent 는 같은 r0f0 에서 2번 (#17, 결정 2). 줄 순서는 1회째 점수 순 (TabArena 참고 줄은 kumo 옆), 시간은 1회째 / 2회째.
굵게 = 그 회차의 전체 최고 (FM·GBDT 의 한 번은 두 회차 모두와 비교). FM·GBDT 는 refit (결정 11), 오른쪽 열은 2026-11-14 까지 쓰던 5-fold 값

| 방식 | 1회째 | 2회째 | 시간 | 옛 5-fold |
|---|---|---|---|---|
| **FM: kumo** | **0.00390** | (2회째도 최고) | 10초 | 0.00449 |
| TabArena Kumo-Tabular (참고) | 0.00381 | | | |
| Agent: Codex × GPT | 0.00640 ¹ | 0.01336 | 58분 / 56분 | |
| FM: tabicl | 0.00669 | | 6초 | 0.00729 |
| FM: tabpfn_v2 | 0.01340 | | 6초 | 0.01491 |
| Agent: AIDE × GPT | 0.01624 | 0.01505 | 60분 / 60분 | |
| Agent: Claude Code × Sonnet +cont | 0.01885 | 0.01806 | 6분 / 2분 (이어 가기 7번 / 7번) | |
| Agent: AIDE × Sonnet | 0.02108 | 0.01406 | 60분 / 60분 | |
| Agent: Claude Code × Sonnet | 0.02219 | 0.03446 | 37초 / 19초 (스스로 끝냄) | |
| GBDT: lgbm / catboost | 0.02623 / 0.02879 | | 2초 / 7초 | 0.02947 / 0.03181 |

- ¹ TabICL·TabPFN 을 스스로 깔아 씀 (아래). 2회째는 깔지 않음. 공식 설정의 실행이라 Agent 결과로 셈 (결정 11)
- 모델 이름: `kumo_c8k_refit` (hazelnut 은 범주 열이 없어 범주를 고친 `kumo_cat_refit` 과 같은 설정) · `tabicl_kv_refit` · `tabpfn_v2_refit` · `lgbm_refit` · `catboost_refit`

in_vehicle (`tabarena-p02`): 이진 · train 8,456 / test 4,228 · 피처 24 (범주 22)

| 방식 | 1회째 | 2회째 | 시간 | 옛 5-fold |
|---|---|---|---|---|
| **Agent: Codex × GPT** | **0.13150** | **0.13295** | 57분 / 58분 (FM 안 씀) | |
| FM: kumo | 0.13707 | | 9초 | 0.14524 |
| TabArena Kumo-Tabular (참고) | 0.13716 | | | |
| Agent: AIDE × Sonnet | 0.13813 | 0.13904 | 60분 / 60분 | |
| Agent: AIDE × GPT | 0.13882 | 0.14057 | 60분 / 60분 | |
| FM: tabicl | 0.14891 | | 5초 | 0.15074 |
| Agent: Claude Code × Sonnet +cont | 0.14911 | 0.14919 | 10분 / 19분 (이어 가기 7번 / 8번) | |
| GBDT: catboost | 0.15899 | | 29초 | 0.16031 |
| Agent: Claude Code × Sonnet | 0.16192 | 0.16192 ² | 27초 / 27초 (스스로 끝냄) | |
| GBDT: lgbm | 0.16221 | | 1초 | 0.16331 |
| FM: tabpfn_v2 | 0.18466 | | 7초 | 0.19410 |

- ² 두 번 모두 같은 CatBoost 스크립트 하나 (`a.py`: 하이퍼파라미터·기본 시드 같음, 출력·import 만 다름) → 제출 파일이 바이트까지 같음 (채점 실수 아님)

- 모델 이름: `kumo_cat_refit` (범주 열을 범주로) · `tabicl_kv_refit` · `tabpfn_v2_refit` · `lgbm_refit` · `catboost_refit`
- 5-fold → refit 으로 FM·GBDT 모두 r0f0 가 조금 나아짐 (오차 −4~−13%). in_vehicle 의 kumo 는 범주 고침이 더 큼: 범주를 수치로 넘긴 refit 0.14401 → 범주로 0.13707
  (TabArena Kumo 0.13716 과 거의 같음). hazelnut 은 순위가 그대로, in_vehicle 은 바뀜 (아래)
- 아래 줄들은 1회째 기준 (2회째는 "2회째 (#17)"). r0f0 한 split 의 test 는 800 / 4,228 행 → 차이 0.002 정도는 split 사이 흔들림 (아래 표의 split 표준편차: hazelnut kumo 0.0009, in_vehicle kumo 0.0032) 과 비슷
- hazelnut: FM 한 번 (kumo) 이 모든 Agent 보다 나음. TFM 을 깔지 않은 Agent 중 최고는 AIDE × GPT 0.0162 (kumo 의 4.2배 오차). 모든 Agent 가 우리 GBDT 보다는 나음
- in_vehicle: **Codex 만 FM 최고 (kumo 0.1371) 보다 나음**. AIDE 둘 (0.1381·0.1388) 은 kumo 와 0.001~0.002 차이로 바로 아래 (split 흔들림 정도).
  5-fold · 범주를 수치로 넘기던 kumo (0.1452) 로는 Agent 셋이 FM 보다 나았음 → "in_vehicle 은 Agent 셋 > FM" 은 Kumo 의 범주 처리 탓이 컸음.
  Codex 는 TabArena 의 Kumo (0.1372) 보다도 나음.
  Codex 는 인구통계 열로 응답자 id 를 다시 만들고 (`person`, 같은 응답자가 여러 시나리오에 답함) 쿠폰 × 상황 교차 피처 + CatBoost·LightGBM·XGBoost·선형 모델 섞기
  (`assemble.py`). 응답자 id 피처는 train+test 를 합쳐 만듦 → TabArena 의 "inductive" 규칙 (5) 으로는 안 되는 풀이
- Codex × GPT 의 hazelnut 풀이는 공식 설정 (TFM 없는 환경) 인데 `pip install --target /tmp/...` 로 `tabpfn==2.0.9`·`tabicl==2.2.0` 을 깔고 HF 에서
  가중치 (TabPFN v2, TabICL v2-20260212) 를 받아 CV·미세 조정·의사 레이블에 씀. 제출 = TabICL 예측 + 의사 레이블 섞기 (`finalize_submission.py`).
  → 공식 설정에서 스스로 TFM 을 찾아 깐 것도 Agent 의 풀이로 보고 **Agent 결과로 셈** (2026-11-15 사용자가 정함, 결정 11). 표시는 남김: `tfm_runtime` 이 잡음 (TFM import 프로세스 90개),
  `pip_target` 이 잡음 (`recheck`, 9 의 "사후 검사"), `pip_added` 는 못 잡음. 이 실행을 빼도 hazelnut 의 "FM > 모든 Agent" 는 같음. MLE-bench P0 의 Codex 는 TabPFN 이 깔렸는지만 보고 안 썼음
- Claude Code 공식 설정은 MLE-bench 처럼 1분 안에 스스로 끝냄 (27~37초). +cont 는 이어 가기 상한 (도구 0개 구간 연속 3번) 에 걸려 6~10분
- 사후 검사: 10개 실행 모두 `data_source_urls` 0, `kaggle_urls` 0 (원본 데이터 접근 흔적 없음). 429 재시도는 Codex 만 (100번·19번)
- 토큰 (입력): Codex 4,200만~4,700만, 나머지 9만~64만 (MLE-bench 와 같은 꼴)
- GPU (A10) 를 쓴 실행은 Codex 둘뿐: hazelnut 은 TabICL·TabPFN 을 GPU 로, in_vehicle 은 CatBoost `task_type='GPU'`·torch 신경망 일부 (최종 제출은 GBDT·선형 섞기).
  AIDE × Sonnet hazelnut 은 탐색 중 torch·cuda 를 한 번 시도했고 최종은 SVC. 나머지 7개는 LightGBM·CatBoost·로지스틱·SVC·sklearn MLP 를 CPU 로만
  (작업 디렉터리 코드·하네스 기록에서 `device='cuda'`·`task_type='GPU'`·`gpu_hist`·nvidia-smi 를 셈, 저장소 밖 스크립트)
- Agent 의 A10 은 유지 (2026-11-14 사용자가 정함): MLE-bench 의 기본 설정 (A10 1장 · 1시간) 과 C2 (+fm, TFM 에 GPU 필요) 와 같은 환경으로 두려고.
  GPU 와 인터넷이 있으면 공식 설정에서도 FM 을 스스로 깔 수 있음 (hazelnut 의 Codex) → 그런 실행도 Agent 결과로 셈 (결정 11). `tfm_runtime`·`pip_target` 으로 어느 실행인지는 보임

### 2회째 (#17, 2026-11-14~15)

같은 r0f0 · 같은 설정. 데이터셋마다 1회째와 같은 순서 (Sonnet 줄: AIDE → Claude Code → +cont, GPT 줄: AIDE → Codex. 같은 엔드포인트는 한 번에 하나), hazelnut → in_vehicle.
다만 hazelnut × Codex 는 첫 시도가 취소돼 (아래 "뺀 실행") in_vehicle × Codex 다음, 맨 끝에 다시 돌림 (hazelnut 묶음 시작 약 6시간 뒤, 그동안 이 저장소의 다른 GPT 실행 없음). 10회 모두 정상 종료·채점.

- 같은 순위인가: **그렇다.** hazelnut 은 2회째에도 FM 한 번 (refit kumo 0.0039, tabicl 0.0067) 이 모든 Agent 보다 나음. 2회째 Agent 최고는 Codex 0.0134 (kumo 의 3.4배 오차).
  in_vehicle 은 2회째에도 Codex (0.1330) 만 FM 최고 (refit kumo 0.1371) 보다 나음, AIDE × Sonnet (0.1390) · AIDE × GPT (0.1406) 는 그 아래. Codex 는 또 TabArena Kumo (0.1372) 보다도 나음.
  (2026-11-14 의 5-fold · 범주를 수치로 넘긴 kumo 0.1452 로는 AIDE 둘도 FM 보다 나았음. FM 을 refit 으로 바꾼 뒤 다시 씀, 결정 11)
  1회째와 2회째 차이: in_vehicle 은 모두 0.002 안 (split 흔들림 안), hazelnut 은 AIDE × Sonnet 0.021 → 0.014, Claude Code 공식 0.022 → 0.034 (우리 GBDT 보다 나쁨, 19초에 스스로 끝냄),
  Codex 0.006 → 0.013 (아래). hazelnut 의 순위 안쪽 (Agent 끼리) 은 흔들리지만 "FM > 모든 Agent" 는 그대로
- Codex 가 공식 설정에서 TFM 을 스스로 까는가: **2회째는 안 깔았음** (두 데이터셋 모두 `tfm_runtime`·`tfm_in_code`·`pip_target` 빈칸).
  hazelnut 2회째는 처음에 `pip show tabpfn tabicl autogluon ...` 으로 깔렸는지만 보고 (없음), SVC·커널 릿지·NCA·torch MLP 섞기 (`generate_final.py`) 로 감 → 0.0134.
  1회째의 0.0064 는 TFM 을 깐 덕으로 보임 (2회째 Codex 는 AIDE 들과 비슷한 자리). 두 실행 모두 Agent 결과로 셈 (결정 11). 1회째는 TFM 을 스스로 깔아 쓴 실행 (`tfm_runtime`·`pip_target`)
- in_vehicle 의 Codex 2회째도 인구통계 열로 응답자 id (`respondent`) 를 다시 만들고 train+test 를 합쳐 만든 응답자 id 로 응답자별 타깃 집계 피처 (다른 fold 에서) → GBDT·선형 모델 등 섞기 (`combine.py`).
  1회째와 같은 "inductive 규칙 (5) 으로는 안 되는" 풀이가 두 번 다 나옴 → in_vehicle 의 "Codex > FM" 은 이 풀이 몫으로 읽음
- 사후 검사: 10회 모두 `data_source_urls` 0, `kaggle_urls` 0 (원본 데이터 접근 흔적 없음). 429 재시도는 Codex 만 (hazelnut 4번, in_vehicle 57번). 입력 토큰 Codex 4,054만·4,234만, 나머지 9만~57만
- 뺀 실행: hazelnut × Codex 2회째 첫 시도 (`tabarena-p01__codex-gpt__r2__20261114-205933`) 는 하네스 14분째 Databricks 에서 "Run cancelled by user" 로 취소 (누가 했는지 기록 없음).
  진행 사본으로 0.01233 으로 채점됐지만 (`run_crashed`) 1시간 조건이 아니라 `fab ta agent exclude` 로 빼고 같은 rep 2 로 다시 돌림 (위 0.01336)

### FM 이 TabArena 와 맞는지

범위: 51개 데이터셋의 모든 split (816개). 우리 FM 은 모두 refit (CV 없이 train 전체로 한 번 fit → test 예측, TabArena TFM 의 `refit_folds=True` 와 같은 꼴, 결정 10·11).
판정 (2026-11-15)

| 우리 FM | TabArena | 데이터셋 | split 표준편차 안 | 승패 같음 (비김 뺌) | 승패 같음 (비김까지, 참고) | 우리 − TabArena (표준편차 단위, 중앙값) | 우리가 나은 데이터셋 | 판정 (8 의 기준) |
|---|---|---|---|---|---|---|---|---|
| `tabicl_kv_refit` | TabICLv2 | 51 | 50 (98.0%) | 29 / 29 | 50 / 51 | +0.01 | 23 | **통과** |
| `kumo_cat_refit` (범주 열이 있는 34개) + `kumo_c8k_refit` (나머지 17개, 같은 설정) | Kumo-Tabular | 51 | 50 (98.0%) | 35 / 36 | 50 / 51 | +0.01 | 17 | 못 넘음 (kddcup09 하나) → **완료로 봄** (아래, 사용자) |
| `tabpfn_v2_refit` | TabPFNv2 | 33 | 31 (93.9%) | 9 / 9 | 28 / 33 | −0.13 | 25 | **통과** |

- TabICL·TabPFNv2 는 통과. TabPFNv2 는 refit 으로 처음 판정 (2026-11-15, 33개 654 split): 5-fold (90.9%) 보다 조금 나음, 표준편차 밖은 NATICUSdroid (−1.07) · in_vehicle (−1.55) 둘 다 우리가 나은 쪽.
  우리가 나은 데이터셋이 33개 중 25개 (중앙값 −0.13): 5-fold 의 −0.15 와 거의 같아 "TabArena TabPFNv2 보다 조금 나음" 은 test 예측 방식 탓이 아님 (패키지·가중치 판 차이로 짐작, 3 의 "다른 점")
- Kumo 는 범주 열을 범주로 넘기도록 고친 뒤 (`_Kumo` 의 `cats`, 결정 11) 표준편차 안 94.1% → 98.0%. kddcup09 하나가 아직 밖 (+1.67) 이고 승패가 다름:
  우리 0.1569 · 기본 GBDT 0.1543 · TabArena 0.1434. 이 하나 때문에 8 의 기준은 못 넘지만, 남은 차이는 TabArena 가 모델 앞에 두는 AutoGluon 전처리로 다 설명됨 (아래 "kddcup09").
  우리 FM 은 TabArena 전처리를 맞추지 않으므로 (결정 4) 이 데이터셋을 적어 두고 **P4 완료로 봄** (2026-11-15, 사용자)
- 비김으로 빠지는 데이터셋: tabicl 22 / 51, kumo 15 / 51, tabpfn_v2 24 / 33. 맞춤의 주 근거는 표준편차 기준
- 판정은 비김을 뺀 기준 (2026-11-14 정함, 결정 9)

kddcup09 (2026-11-15): 무엇 탓인지

kddcup09_appetency × 9 split. 기준은 2026-11-14 의 `kumo_c8k_refit` (앙상블 8 · 청크 8192 · refit · 범주를 수치로). 줄마다 하나만 바꿈.
TabArena Kumo-Tabular 0.1434 ± 0.0081, 기본 GBDT (CatBoost) 0.1543

| 우리 모델 | 바꾼 것 | 9 split 평균 | 우리 − TabArena (표준편차 단위) | 기본 GBDT 보다 |
|---|---|---|---|---|
| `kumo_c8k_refit` | (기준) | 0.1618 | +2.27 | 나쁨 |
| `kumo_e16_refit` | 앙상블 16 (TabArena 와 같음) | 0.1611 | +2.19 | 나쁨 |
| `kumo_c2k_refit` | 질의 청크 2048 | 0.1619 | +2.29 | 나쁨 |
| `kumo_c8k` | 5-fold (fold 모델 5개 평균) | 0.1618 | +2.27 | 나쁨 |
| `kumo_cat_refit` | 범주 열을 범주로 (Kumo 기본 전처리: 값 순 맞춤 · 50개 이상이면 빈도 열 · 범주 표시) | 0.1569 | +1.67 | 나쁨 |
| `kumo_cat_agp_refit` | 위 + AutoGluon 전처리 흉내 (진단용): 한 번만 나온 범주 값 → 결측, 상수·중복 열 빼기 | **0.1434** | **+0.01** | 나음 |

- 앙상블·청크·refit 은 평균이 거의 그대로 (서로 0.0008 안, split 마다는 0.005 안) → 원인 아님. 넷 다 9 split 모두에서 TabArena 보다 나쁨
- 범주 처리가 차이의 약 1/4 (0.1618 → 0.1569, 차이 0.0184 중 0.0049). 우리 `_Kumo` 는 범주를 정수 코드로 바꿔 수치로 넘겼음 → Kumo 가 범주 열인지 모르고 빈도 열도 없음.
  kddcup09 의 범주 38열 중 9열은 값이 1,667~11,732 가지, 한 번만 나온 값이 열마다 745~8,472 행 (r0f0 train 33,333 행 기준)
- 나머지는 AutoGluon 전처리 흉내로 다 없어짐 (TabArena 와 같은 값). kddcup09 에는 똑같은 열 (Var200 = Var214, Var198 = Var220 = Var222) 과 한 번만 나온 범주 값이 많음.
  흉내는 AutoGluon 코드를 옮긴 것이 아니라 짐작 (중복 열 빼기 · 드문 범주 → 결측) → TabArena 의 실제 전처리와 같다고는 못 함. 어느 쪽이 몫이 큰지는 나눠 보지 않음
- `kumo_cat_agp_refit` 은 진단용으로 남김 (기본 아님). 기본으로 하면 FM 이 바뀌어 51개·MLE-bench 를 다시 돌려야 하고, 우리 FM 은 전처리를 맞추지 않기로 함 (결정 4)

범주 고침이 바꾼 것 (범주 열이 있는 34개, `kumo_c8k_refit` → `kumo_cat_refit`, 537 split)

| | 표준편차 안 | 승패 같음 (비김 뺌) | 우리 − TabArena (중앙값 · 평균, 표준편차 단위) | TabArena 에 더 가까워진 데이터셋 | 범주로 넘긴 쪽이 나은 데이터셋 |
|---|---|---|---|---|---|
| 범주를 수치로 (`kumo_c8k_refit`) | 31 / 34 | 22 / 23 | +0.02 · +0.17 | | |
| 범주를 범주로 (`kumo_cat_refit`) | 33 / 34 | 22 / 23 | +0.01 · +0.09 | 25 / 34 | 18 / 34 |

- 크게 가까워진 것: in_vehicle +1.34 → +0.15, Amazon +1.19 → +0.20, MIC +0.40 → −0.08, splice +0.32 → −0.01, kddcup09 +2.27 → +1.67
- 멀어진 것: Marketing_Campaign +0.11 → +0.79 (아직 안) 하나가 큼. 나머지 8개는 0.07 안 (hiva_agnostic·customer_satisfaction 등)
- 범주 열이 없는 17개는 범주 처리가 상관없어 (`cats` 는 범주 열에만) `kumo_c8k_refit` 값이 그대로 고친 Kumo 의 값

2026-11-14 의 판정 (기록: Kumo 는 범주를 수치로, tabpfn_v2 는 5-fold)


범위: 51개 데이터셋의 모든 split (816개). 우리 FM 은 refit (`tabicl_kv_refit`·`kumo_c8k_refit`, "빠르게 돌리기"): CV 없이 train 전체로 한 번 fit → test 예측.
TabArena TFM 의 `refit_folds=True` 와 같은 꼴 (결정 10). 2만 행 넘는 12개 (houses·superconductivity·credit_card·bank-marketing·Food_Delivery·kddcup09·diamonds·Diabetes130US·APSFailure·SDSS17·customer_satisfaction·GiveMeSomeCredit, 108 split) 는 #16 (2026-11-14),
나머지 39개 (파일럿 4개 + 2만 행 이하 35개, 708 split) 는 2026-11-14 23:38 KST 에 낸 작업 하나로 (아래 "39개 refit", 표의 시각은 UTC 2026-11-14T14:41~15:41). KV 캐시는 점수가 같고, Kumo 질의 청크 8192 는 TabArena 표준편차의 0.1배 안쪽으로 조금 다름 ("빠르게 돌리기" 의 "시험: 점수")

| 우리 FM | TabArena | 범위 | 데이터셋 | split 표준편차 안 | 승패 같음 (비김 뺌) | 승패 같음 (비김까지, 참고) | 우리 − TabArena (표준편차 단위, 중앙값) | 우리가 나은 데이터셋 | 판정 (8 의 기준) |
|---|---|---|---|---|---|---|---|---|---|
| `tabicl_kv_refit` | TabICLv2 | **51개** | 51 | 50 (98.0%) | 29 / 29 | 50 / 51 | +0.01 | 23 | **통과** |
| `kumo_c8k_refit` | Kumo-Tabular | **51개** | 51 | 48 (94.1%) | 35 / 36 | 49 / 51 | +0.01 | 19 | **못 넘음** (kddcup09 하나) |
| `tabicl_kv_refit` | TabICLv2 | 39개 | 39 | 38 (97.4%) | 19 / 19 | 38 / 39 | +0.01 | 17 | 통과 |
| `kumo_c8k_refit` | Kumo-Tabular | 39개 | 39 | 37 (94.9%) | 25 / 25 | 38 / 39 | +0.01 | 13 | 통과 |
| `tabicl_kv_refit` | TabICLv2 | 12개 | 12 | 12 (100%) | 10 / 10 | 12 / 12 | +0.01 | 6 | 통과 |
| `kumo_c8k_refit` | Kumo-Tabular | 12개 | 12 | 11 (91.7%) | 10 / 11 | 11 / 12 | +0.01 | 6 | 못 넘음 (kddcup09) |

- 표준편차 기준 (80%) 은 둘 다 넘음 (98.0% · 94.1%). TabICL 은 승패도 모두 같아 통과
- Kumo 는 kddcup09_appetency 하나에서 승패가 다름: 우리 0.1618 대 TabArena 0.1434 (표준편차 2.27배 나쁨), 기본 GBDT 0.1543 보다 우리는 나쁘고 TabArena 는 나음 → 8 의 기준으로는 못 넘음
- kddcup09 는 피처 212개 중 범주 38개, 값이 수천 가지인 열도 있음. 우리 `kumo` 는 범주를 정수 코드로 바꿔 수치로 넘김 (`fab/models/tfm.py` 의 `_Kumo`).
  표준편차 밖인 Amazon (+1.19, 모두 범주) · in_vehicle (+1.34, 범주 22 / 24) 과 같은 꼴 → 범주 처리 차이로 **짐작** (확인 안 함).
  Amazon · in_vehicle 은 5-fold 에서도 밖 (+1.26 · +1.48) 이라 refit 탓이 아님. kddcup09 는 refit (청크 8192) 결과뿐이라 refit·청크 탓인지 가리지 못함 (아래 "남은 것")
- 표준편차 밖: tabicl 은 hiva_agnostic (+1.40) 하나, kumo 는 Amazon (+1.19) · in_vehicle (+1.34) · kddcup09 (+2.27). + = 우리가 나쁨.
  5-fold 에서 밖이던 airfoil (tabicl +1.05) 은 refit 으로 −0.02
- 승패가 다른 것 (비김까지 세면): 둘 다 E-CommereShippingData (비김으로 빠짐) + kumo 의 kddcup09
- 비김으로 빠지는 데이터셋: tabicl 22 / 51, kumo 15 / 51. 맞춤의 주 근거는 표준편차 기준 (아래 5-fold 와 같은 까닭)
- 판정은 비김을 뺀 기준 (2026-11-14 정함, 결정 9)

39개 refit: 5-fold 에서 보던 기울기가 사라짐

| 우리 FM (39개) | 5-fold: 우리 − TabArena (중앙값 · 평균, 표준편차 단위) | refit | 우리가 나은 데이터셋 (5-fold → refit) | 5-fold → refit 변화 (중앙값, 표준편차 단위) | refit 이 나은 데이터셋 | TabArena 에 더 가까워진 데이터셋 |
|---|---|---|---|---|---|---|
| TabICL | +0.08 · +0.12 | **+0.01** · +0.07 | 13 → 17 | −0.01 (상대 −0.13%) | 24 / 39 | 27 / 39 |
| Kumo | +0.14 · +0.21 | **+0.01** · +0.10 | 10 → 13 | −0.07 (상대 −0.34%) | 28 / 39 | 31 / 39 |

- 5-fold 의 기울기 (+0.08 · +0.14, 9 의 "차이" 에서 본 "컨텍스트 80% 대 refit") 가 refit 에서 +0.01 · +0.01 로 사라짐 (12개 refit 과 같은 값). → 기울기는 test 예측 방식 (train 의 80% 로 fit 한 fold 모델 5개 평균) 탓으로 봄 (refit 은 Kumo 청크도 바뀌었지만 "시험: 점수" 에서 청크 차이는 표준편차의 0.09배 안)
- 많이 바뀐 것: airfoil (1,503 행) TabICL +1.05 → −0.02 · Kumo +0.64 → −0.08, hazelnut (2,400 행) Kumo +0.73 → +0.06, maternal_health_risk (1,014 행) Kumo +0.51 → −0.02.
  다만 중간 크기에서도 크게 바뀜: TabICL jm1 (10,885 행) −0.48 → +0.54 (|변화| 1.02, 둘째로 큼, 나빠지는 쪽), in_vehicle (12,684 행) −0.46, Bioresponse (3,751 행) +0.44.
  변화의 크기와 행 수는 관계가 뚜렷하지 않음 (스피어만 TabICL 0.26 · Kumo 0.01, 행 2,500 미만 17개와 나머지 22개의 변화 중앙값도 비슷) → "작은 데이터셋일수록 컨텍스트 20% 가 큼" 은 이 결과로 보이지 않음.
  보인 것은 전체 중앙값의 기울기가 사라진 것까지
- Kumo 는 polish_companies −0.18 → +0.15 가 나빠지는 쪽으로 가장 큼
- 평균 (+0.07 · +0.10) 은 0 보다 조금 큼: 표준편차 밖 데이터셋 (hiva · Amazon · in_vehicle) 이 끌어올림. 그래도 우리가 나은 데이터셋은 39개 중 17 · 13 개 → 중앙값 근처의 작은 차이는 우리가 조금 나쁜 쪽이 많음 (특히 Kumo)

5-fold (`fab run` 기본 = Agent vs FM · MLE-bench · C1 · C2 에 쓰는 FM, 2026-11-14): 39개. TabPFNv2 는 refit 으로 안 돌림 (TabArena 가 돈 33개, 5-fold 만).
tabpfn_v2 의 판정은 이 5-fold 표뿐이라 refit 인 tabicl·kumo 판정과 같은 프로토콜이 아님 (중앙값 −0.15 에 5-fold 의 기울기가 섞였을 수 있음)

| 우리 FM | TabArena | 데이터셋 | split 표준편차 안 | 승패 같음 (비김 뺌) | 승패 같음 (비김까지, 참고) | 우리 − TabArena (표준편차 단위, 중앙값) | 우리가 나은 데이터셋 | 판정 (8 의 기준) |
|---|---|---|---|---|---|---|---|---|
| `tabicl` | TabICLv2 | 39 | 37 (94.9%) | 19 / 19 | 39 / 39 | +0.08 | 13 | 통과 |
| `kumo` | Kumo-Tabular | 39 | 37 (94.9%) | 25 / 25 | 38 / 39 | +0.14 | 10 | 통과 |
| `kumo_e16` | Kumo-Tabular | 39 | 37 (94.9%) | 25 / 25 | 38 / 39 | +0.14 | 10 | 통과 |
| `tabpfn_v2` | TabPFNv2 | 33 | 30 (90.9%) | 9 / 9 | 27 / 33 | −0.15 | 23 | **통과** |

- 판정은 비김을 뺀 기준 (결정 9). 비김까지 세던 처음 기준으로는 kumo (1개)·tabpfn_v2 (6개) 가 못 넘었음 (아래)
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

- test 예측: 5-fold 는 train 의 80% 를 컨텍스트로 한 fold 모델 5개의 평균, TabArena 는 train 전체로 refit 한 모델 하나 → 컨텍스트가 20% 적음.
  51개 맞춤은 refit 으로 바꿔 이 차이를 없앰. 5-fold 의 기울기 (+0.08 · +0.14) 는 refit 에서 +0.01 (위 "39개 refit")
- Kumo 앙상블 8 대 16 (아래 `kumo_e16` 로 봄, 차이 없음)
- Kumo 범주: 우리는 정수 코드로 넘김 (위 kddcup09 · Amazon · in_vehicle)
- TabPFNv2: TabArena 쪽은 2025-06-12 묶음 (패키지 버전 확인 못 함), 우리는 tabpfn 9.2.0 의 v2 가중치

파일럿 4개 (모든 split, FM·GBDT 는 5-fold. refit 은 표 아래)

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

- refit (평균 ± 표준편차, 표준편차 안): hazelnut kumo 0.0027 ± 0.0008 O · tabicl 0.0051 ± 0.0014 O, in_vehicle kumo 0.1497 ± 0.0034 X · tabicl 0.1564 ± 0.0041 O,
  Amazon kumo 0.1300 ± 0.0071 X · tabicl 0.1501 ± 0.0055 O, physiochemical kumo 2.7730 ± 0.0346 O · tabicl 2.9405 ± 0.0333 O. Amazon 의 tabicl (+0.53 → +0.73) 말고는 모두 5-fold 보다 TabArena 에 가까움 (Kumo 의 Amazon·in_vehicle 은 그래도 밖)
- Amazon 은 TabArena 처럼 기본 CatBoost 가 모든 FM 보다 나음 (우리 CatBoost 0.1132 가 TabArena 의 0.1182 보다도 나음)
- GBDT 는 참고선 (결정 7): 우리 GBDT 는 고정 반복 (lgbm 800 × lr 0.02, catboost 1500 × lr 0.03, 조기 종료 없음) 이라 큰 회귀 (physiochemical, train 3만) 에서 TabArena 보다 13% 나쁨
- 2026-11-15 refit (결정 11, 평균 ± 표준편차, 표준편차 안): kumo (범주 고침, `kumo_cat_refit`) in_vehicle 0.1451 ± 0.0038 O · Amazon 0.1236 ± 0.0060 O (둘 다 처음으로 안),
  tabpfn_v2 (`tabpfn_v2_refit`) hazelnut 0.0105 ± 0.0027 O · in_vehicle 0.2002 ± 0.0080 X (우리가 나은 쪽으로 밖, −1.55).
  GBDT (`lgbm_refit` / `catboost_refit`): hazelnut 0.0214 / 0.0242, in_vehicle 0.1679 / 0.1687, Amazon 0.1521 / 0.1149, physiochemical 3.9464 / 3.9523.
  5-fold 와 비교하면 hazelnut·in_vehicle 은 조금 나음 (오차 −0.5~−9%), Amazon 은 조금 나쁨 (+1.5~+1.7%), physiochemical 은 같음 (TabArena 보다 13% 나쁜 것도 그대로)

### kumo_e16

결정 4 대로 Kumo 의 앙상블을 TabArena 와 같은 16 으로 (`fab/models` 의 `VARIANTS` 에 `kumo_e16`) 같은 39개 · 708 split 에 다시 돌림 (A10).

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
| kumo_e16 (39개, 708 split) | A10 39 작업 (동시 5개) | 벽시계 약 2시간 (14:35~16:39) | 모델 시간 6.6시간 |
| #16 refit (12개, 108 split × 2) | 8×H100 작업 하나 (+ 시험 작업 하나) | 10분 43초 | "빠르게 돌리기" 의 "#16 에 쓴 것" |
| 39개 refit (708 split × 2) | 8×H100 작업 하나 | 37분 38초 (KST 11-14 23:38 ~ 11-15 00:16, 노드 3분 안) | 모델 시간 3.2시간 (tabicl 1.4 · kumo 1.8). 약 5 H100시간. 원격 채점 24.6분 (같은 정리 작업에서 Agent 4개 대기도 처리) |
| Agent 10회 | A10 × 10 | 실행당 1시간 + 준비 | Sonnet 줄·GPT 줄 하나씩 동시에, 벽시계 약 4시간 30분 (09:00~13:30) |
| kddcup09 원인 (9 split × `kumo_e16_refit`·`kumo_c2k_refit`·`kumo_c8k`, 2026-11-15) | 8×H100 작업 하나 | 8.1분 | 모델 시간 0.5시간 (5-fold `kumo_c8k` split 하나 96초, refit 34~83초) |
| `tabpfn_v2_refit` (33개, 654 split, 2026-11-15) | 8×H100 작업 하나 | 17.8분 | 모델 시간 1.2시간 (split 하나 중앙값 6초). 5-fold 는 2.2시간 |
| `kumo_cat_refit` (34개, 537 split, 2026-11-15) | 8×H100 작업 하나 | 18.8분 | 모델 시간 1.6시간 (split 하나 중앙값 8초) |
| GBDT refit (파일럿 4개 × 57 split × `lgbm_refit`·`catboost_refit`, 2026-11-15) | serverless CPU 하나 | 20.0분 | 모델 시간 0.3시간 |
| kddcup09 AutoGluon 흉내 (9 split × `kumo_cat_agp_refit`, 2026-11-15) | 8×H100 작업 하나 | 3.0분 | 노드가 바로 뜸 |

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
| 8×H100 (작업 하나) | tabicl_kv_refit · kumo_c8k_refit | 2만 행 넘는 12개 (#16, 2026-11-14) | 12 |
| 8×H100 (작업 하나) | tabicl_kv_refit · kumo_c8k_refit | 위 39개 모두 (KST 2026-11-14 23:38 ~ 11-15 00:16) | 39 |
| 8×H100 (작업 하나씩) | kumo_e16_refit · kumo_c2k_refit · kumo_c8k, 그다음 kumo_cat_agp_refit | kddcup09_appetency (2026-11-15) | 1 |
| 8×H100 (작업 하나) | tabpfn_v2_refit | TabPFNv2 범위 33개 (2026-11-15) | 33 |
| 8×H100 (작업 하나) | kumo_cat_refit | 범주 열이 있는 34개 (2026-11-15) | 34 |
| serverless CPU (작업 하나) | lgbm_refit · catboost_refit | 파일럿 4개 (2026-11-15) | 4 |

- 51개 맞춤 표 (refit) 는 모두 H100. 5-fold (비교용) 는 위처럼 H100·A10 이 섞임
- 점수는 GPU 와 무관하다고 보고 같은 표에 섞음 (같은 데이터셋을 두 GPU 로 돌려 비교해 보지는 않음). 시간 (`results.csv` 의 `seconds`) 은 GPU 에 따라 다르므로 데이터셋끼리 시간을 비교할 때는 이 표를 봄
- `kumo` (H100 11개 · A10 28개) 와 `kumo_e16` (모두 A10) 은 H100 11개에서는 GPU 도 다름. 그래도 결과가 거의 같음 (위 "kumo_e16")

### 남은 것

| 항목 | 이슈 | 상태 |
|---|---|---|
| 2만 행 넘는 12개 데이터셋 (FM 맞춤 확인용, tabicl·kumo) | [#16](https://github.com/Aiden-Jeon/fm-agent-bench/issues/16) | 끝 (refit, 8×H100 작업 하나 11분. 위 "FM 이 TabArena 와 맞는지") |
| Kumo 가 refit 51개에서 기준을 못 넘음 (kddcup09 하나). 가릴 것: 앙상블·청크·refit, 셋 다 아니면 범주 처리 | – | 끝 (2026-11-15): 앙상블·청크·refit 은 아님. 범주 처리를 고쳐 (`kumo_cat_refit`, 범주 열이 있는 34개 다시) 표준편차 안 98.0%. kddcup09 의 나머지는 TabArena 의 AutoGluon 전처리 (흉내로 0.1434 = TabArena). 우리 FM 은 전처리를 맞추지 않음 (결정 4) → 적어 두고 P4 완료 (사용자). "FM 이 TabArena 와 맞는지" 의 "kddcup09" |
| Agent 2회째 (결정 2: 같은 r0f0 에서, 10회) | [#17](https://github.com/Aiden-Jeon/fm-agent-bench/issues/17) | 끝 (2026-11-15, 위 "2회째 (#17)") |
| Agent 확장: 회귀·다중을 넣은 후보 10개 × 5설정 (r0f0, 1회째) | [#26](https://github.com/Aiden-Jeon/fm-agent-bench/issues/26) | 끝 (2026-11-17, 아래 "Agent 확장 1회째 (#26)"). C2 (+fm)·2회째는 따로 정함 |
| 사후 검사가 `pip install --target` 을 못 잡음 → `pip_target` 열 (9 의 "사후 검사") | [#18](https://github.com/Aiden-Jeon/fm-agent-bench/issues/18) | 완료 (2026-11-14) |
| `fab ta fm` 의 동시 GPU 작업 상한 → `--max-concurrent` (9 의 "명령") | [#19](https://github.com/Aiden-Jeon/fm-agent-bench/issues/19) | 완료 (2026-11-14) |
| 원격 채점 속도 (정리 작업 한 번 약 1시간) → 채점 약 3배 빠름, 남은 몫은 FM 합치기 (9 의 "원격 채점 속도") | [#20](https://github.com/Aiden-Jeon/fm-agent-bench/issues/20) | 완료 (2026-11-14) |
| physiochemical 의 `tabpfn_v2` (TabArena 범위 밖이라 비교에는 필요 없음) | – | 안 함 |
| tabpfn_v2 를 refit 으로 (33개, tabicl·kumo 와 같은 프로토콜로) | – | 끝 (2026-11-15): 통과 (표준편차 안 93.9%, 승패 9 / 9). "FM 이 TabArena 와 맞는지" |
| Kumo·TabICL 이 TabArena 보다 조금 나쁜 쪽으로 기우는 까닭 (컨텍스트 80% 대 refit) | – | 끝 (2026-11-15): 39개도 refit 으로 돌려 tabicl·kumo 의 51개를 한 프로토콜로 (결정 10). 기울기 +0.08 · +0.14 → +0.01 · +0.01 ("FM 이 TabArena 와 맞는지" 의 "39개 refit") |

## Agent 확장 1회째 (#26, 2026-11-16)

파일럿 둘은 모두 이진이라 "FM 한 번이 1시간 Agent 를 이기는 데이터는 어떤 종류인가" 를 말하기 어려웠음.
회귀·다중을 넣고 FM 이 크게 이기는 곳·근소한 곳·GBDT 가 이기는 곳을 섞은 10개를 r0f0 에서 5설정 × 1회 돌림 ([#26](https://github.com/Aiden-Jeon/fm-agent-bench/issues/26)).
C2 (+fm) 와 2회째 (결정 2) 는 이 결과를 보고 따로 정함.

### 후보 (#26 의 표를 Kumo 범주 고침 뒤로 다시)

#26 의 FM 값은 #27 전 Kumo (`kumo_c8k_refit`, 범주를 수치로). 범주 열이 있는 데이터셋은 `kumo_cat_refit` 으로 다시 계산함.
`metric_error` 의 모든 split 평균, FM 최고 = refit 셋 (`kumo`·`tabicl_kv`·`tabpfn_v2`) 중 나은 쪽 (10개 모두 kumo), GBDT = TabArena 공개 기본 CatBoost·LightGBM 중 나은 쪽.
σ = 그 FM 의 split 표준편차.

| 데이터셋 | 과제 | train / test (r0f0) | 피처 (범주) | FM 최고 (#26 → 지금) | GBDT | FM − GBDT (#26 → 지금) | ÷ σ (#26 → 지금) | TabArena Kumo − GBDT (참고) | 묶음 |
|---|---|---|---|---|---|---|---|---|---|
| QSAR_fish_toxicity | 회귀 | 604 / 303 | 6 (0) | 0.851 (같음) | CatBoost 0.877 | −2.9% | 0.5 | −3.0% | 근소 |
| MIC | 다중 8 | 1,132 / 567 | 111 (94) | 0.428 → **0.420** | CatBoost 0.455 | −5.9% → **−7.7%** | 1.3 → 1.9 | −7.4% | 중간 |
| Is-this-a-good-customer | 이진 | 1,148 / 575 | 13 (8) | 0.245 → 0.244 | CatBoost 0.251 | −2.5% → −2.9% | 0.3 | −3.1% | 근소 |
| splice | 다중 3 | 2,126 / 1,064 | 60 (60) | 0.092 → **0.087** | CatBoost 0.111 | −17.2% → **−21.3%** | 1.2 → 1.6 | −21.2% | 크게 |
| students_dropout_and_academic_success | 다중 3 | 2,949 / 1,475 | 36 (17) | 0.507 (같음) | CatBoost 0.552 | −8.1% | 3.2 | −8.3% | 중간 |
| polish_companies_bankruptcy | 이진 | 3,940 / 1,970 | 64 (0) | 0.00435 (같음) | CatBoost 0.0390 | −88.8% ¹ | 20.4 | −89.4% | 크게 |
| QSAR-TID-11 | 회귀 | 3,828 / 1,914 | 1,024 (0) | 0.721 (같음) | LightGBM 0.772 | −6.6% | 1.1 | −7.2% | 중간 |
| APSFailure | 이진 | 50,666 / 25,334 | 170 (0) | 0.00461 (같음) | CatBoost 0.00756 | −39.0% | 2.0 | −39.4% | 크게 |
| physiochemical_protein | 회귀 | 30,486 / 15,244 | 9 (0) | 2.773 (같음) | LightGBM 3.477 | −20.2% | 19.2 | −20.3% | 크게 |
| Amazon_employee_access (대조군) | 이진 | 21,846 / 10,923 | 9 (9) | 0.1300 → **0.1236** | CatBoost 0.1182 | **+9.9% → +4.5%** | 1.6 → **0.8** | +3.4% | GBDT 이김 |

¹ 1 − AUC 라 값이 작아 % 가 큼 (절대 차 0.035)

- 바뀐 것은 범주 열이 있는 넷 (MIC · Is-this-a-good-customer · splice · Amazon). 나머지 여섯 중 다섯은 범주 열이 없어 `kumo_c8k_refit` = `kumo_cat_refit` (같은 값). students_dropout 는 범주 열 (17개) 이 있지만 값이 거의 같음 (0.5068 → 0.5069)
- 묶음 (크게 −15% 이하 · 근소 3% 안 · 중간 · GBDT 이김) 은 하나도 안 바뀜
- **Amazon 대조군은 그대로 둠**: 기본 GBDT 가 FM 보다 나은 방향은 같지만 차이가 +9.9% → +4.5% (1σ 안) 로 줄었음. TabArena Kumo (+3.4%) 와 가까워짐 (#26 의 "선행 PR 의 범주 처리 결과와 같이 봄").
  tabicl 은 +27% 로 GBDT 에 크게 짐. 대안 kddcup09_appetency 는 +1.7% (0.3σ) 로 더 약해서 바꿀 까닭이 없음. 다른 대안 (website_phishing −17.0%, hiva_agnostic −0.4%) 도 그대로 대안
- train / test 는 r0f0 split 파일로 센 값 (#26 은 2/3 · 1/3 어림)

### 대회 준비

- 새 r0f0 대회 8개 (`competitions/ta-<데이터셋>/`, git 추적). 익명 id 는 배치 순서로 `tabarena-p05` (QSAR_fish) ~ `p12` (APSFailure). Amazon `p03`·physiochemical `p04` 는 파일럿 때 만들어 둔 것 그대로
  - `config.yaml` 은 `default_config` (FM 이 돈 `tas-<데이터셋>-r0f0` 과 같은 설정: 범주 열·타깃·지표) 에 `anon_id` 만 더함 → 같은 r0f0 데이터·같은 피처 (행 id 는 데이터셋·split 의 고정 seed 라 `tas-…-r0f0` 과 바이트까지 같음)
  - 지표: 이진 `auc` · 다중 `logloss` · 회귀 `rmse`, 분류는 `submit_proba`. 다중 제출은 클래스별 확률 열 (열 이름 = 클래스 이름, sample_submission 의 순서)
  - r0f0 폴더가 생기면 그 split 의 대회 이름이 `tas-<데이터셋>-r0f0` → `ta-<데이터셋>` 으로 바뀜. FM 의 r0f0 행은 옛 이름 그대로 결과표에 있고 (다시 안 돌림), GBDT refit 은 새 이름. 비교는 (데이터셋, split) 로 함
- `description.md` (영어): 과제·열 뜻·지표·제출 형식. 이름·출처·연도·수집 기관 없이, 열 뜻은 OpenML 설명·원래 출처를 보고 우리 말로.
  이름이 그대로 걸리는 말은 피함: splice 는 "exon/intron boundary" 로만 (단어 `splice` 를 안 씀), QSAR_fish 는 "acute aquatic toxicity towards a small freshwater fish species", APSFailure 는 "air pressure system", students 는 "drops out, is still enrolled, or graduates".
  MIC 는 열 이름이 약어라 무리별로 뜻을 적음 (병력·입원 때·입원 뒤 사흘). 클래스 이름 (`cardiogenic_shock` 등) 이 과제를 드러내므로 "heart attack (myocardial infarction)" 은 씀.
  APSFailure 는 원래 열 뜻이 공개되지 않음 (익명 센서) → 히스토그램 무리 (`ag_000`–`ag_009` 등 7개) 만 적음
- `fab ta prepare <10개> --splits all`: 대회 153개 (r0f0 10 + 생성 split 143). 정답은 `FAB_TABARENA/data/answers/`

로컬 확인 (학습 없이, 저장소 밖 스크립트, 파일럿 때 "로컬 확인" 과 같은 검사 + 회귀·다중 채점)

- 10개 × 모든 split (153개): 크기, train·test id 가 0..n−1 순열을 나눔, 열 순서, test 의 값·정답이 OpenML TEST 행과 같음, 같은 repeat 의 fold 3개 test 합 = 전체, 피처 (`features.py`) 의 범주 열이 설정과 같고 `1`·`1.0` 섞임 없음
- 채점 (모든 split): 이진 = 상수 AUC 0.5 · 정답 AUC 1, 다중 = 균등 확률 logloss = ln K · 정답 원-핫 ≈ 0, 회귀 = 상수 (train 평균) RMSE = 식 · 정답 0.
  r0f0 만 더: 이진 뒤집기 AUC 0 · 행 섞기 · 레이블 제출 무효, 다중 = 사전 확률 logloss = 직접 계산 · 행 합이 1 이 아니어도 (×7) 정규화 · 행 섞기 · 열 순서가 다르면 무효 · 행 빠지면 무효, 회귀 = 정답 +2 → RMSE 2 · 행 섞기 · 행 빠지면·열 이름 틀리면 무효
- `leak_check` (`fab/agent`): 10개 모두 설명·데이터 파일 이름·익명 id·실행 이름·코드 묶음 (실행기·지시문 등 7개 파일) 에 이름·대회 이름·OpenML id 없음. 새 설명 8개에 다른 데이터셋 이름도 없음 (MIC 설명의 "diabetes" (당뇨 병력 열) 가 데이터셋 `diabetes` 이름에 걸리지만 MIC 를 알려 주는 말은 아님)
- 합 5,670개 통과. 걸린 9개는 id–타깃 상관 |r| < 0.05 (작은 데이터 셋: QSAR_fish train 604, MIC 1,132, Is-this 1,148). id 는 데이터와 무관한 seed 순열이고 z (= r√n) 가 최대 2.1, split 사이 표준편차 0.8~1.0 이라 우연 범위로 봄

### GBDT refit

- 8개 (Amazon·physiochemical 은 파일럿 때 있음) × 모든 split (135) × `lgbm_refit`·`catboost_refit` 을 serverless CPU 작업 하나로 (`fab ta fm ... --one-job`, 2026-11-16 13:28 KST 제출)
- 1.5시간에 33 split (QSAR_fish 30 + MIC 3) 만 끝남. **MIC 의 CatBoost 가 split 하나에 약 30분** (1,742~1,828초, lgbm 3초). TabArena 의 CatBoost 도 MIC 는 split 중앙값 2,070초 (splice 578초, 나머지 7~203초) → 범주 94열 × 8클래스 탓으로 봄.
  차례로 돌면 MIC 만 약 15시간이라 작업 시간 제한 (12시간) 에 걸려 뒤 데이터셋은 못 돎
- → 15:05 에 취소 (끝난 33 split 은 Volume 에 남음) 하고 남은 것을 CPU 작업 9개로 동시에 냄 (사용자가 정함): 데이터셋마다 하나 (Is-this · splice · students · polish · QSAR-TID-11 · APSFailure, 모든 split) + MIC 남은 27 split 을 9개씩 3개 (`--splits r1f0,...`).
  새 작업마다 채점 요청을 다시 냄. 처음 작업의 채점 요청 (135 대회) 은 MIC 의 새 작업과 같은 (대회, 모델) 을 기다려서 MIC 가 끝날 때까지 남음 →
  QSAR_fish 30 split · MIC r0f0~r0f2 는 채점 요청을 따로 냄 (`fab ta grade ... --splits`). 그 결과 이번 GBDT 270행 (8개 × 모든 split × 2모델) 이 모두 두 번 채점돼 `results/tabarena.csv` 에 두 줄씩 있음 (값은 같음). `compare`·`elo` 는 `latest_grades` 로 마지막 행만 씀 → 표를 직접 셀 때는 (대회, 모델) 로 중복을 뺄 것
- 걸린 시간: CPU 작업 6개는 4~21분 (splice 21분), MIC 3개는 각 4시간 30분쯤 (15:13 → 19:45 무렵). 모델 시간 합 15.8시간 중 MIC 의 CatBoost 가 14.9시간 (split 중앙값 1,791초).
  다음 중앙값: splice CatBoost 131초, APSFailure 54초, QSAR-TID-11 31초, students 26초, 나머지 15초 아래. lgbm 은 모두 18초 아래
- 결과: 10개 모두 모든 split 에 `lgbm_refit`·`catboost_refit` (빠짐 없음)
- TabArena 기본 GBDT 와 견주면 (`fab ta compare`, 파일럿 4개 포함 12개): split 평균이 TabArena split 표준편차 안인 데이터셋 CatBoost 9 / 12, LightGBM 7 / 12.
  우리 GBDT 는 기본 설정 한 번 fit (TabArena 는 8-fold bagging) 이라 맞춤 기준은 두지 않음 (결정 7: TabArena GBDT 는 참고선). 요약 표에는 두 GBDT 값을 같이 적음

### 실행 (2026-11-16 13:31 ~ 11-17 01:27 KST)

| 배치 | 데이터셋 (익명 id) | 제출 (KST) | 목적 |
|---|---|---|---|
| 0 | QSAR_fish (`p05`) → MIC (`p06`) | 13:31 · 14:56 | 회귀 값·다중 확률 제출이 채점되는지 먼저 확인 |
| 1 | Is-this-a-good-customer (`p07`) → splice (`p08`) → students_dropout (`p09`) → polish (`p10`) | 16:11 · 17:20 · 18:29 · 19:36 | 중간 크기, 다중·근소 차 |
| 2 | QSAR-TID-11 (`p11`) → Amazon (`p03`) → physiochemical (`p04`) → APSFailure (`p12`) | 20:48 · 21:59 · 23:08 · 00:20 | 넓이, 큰 데이터, GBDT 이김 |

- 데이터셋마다 5설정 (`codex-gpt`·`aide-sonnet`·`claude-code-sonnet`·`aide-gpt`·`claude-code-sonnet --continue`) 을 한 번에 내고, 다 끝나면 `fab remote kick` → `fab sync` → 검사 (제출 유효·중단 없음·429·원본 URL) → 다음 데이터셋 (저장소 밖 스크립트). 배치마다 내기 전에 사용자 확인
- 같은 설정은 한 번에 하나 (데이터셋을 차례로). 데이터셋 하나에 약 70분 (AIDE 60분 + 설치·회수·채점), 50회 약 12시간
- **조건이 파일럿과 다른 점**: 파일럿 (#17) 은 Sonnet 줄·GPT 줄을 하나씩 돌렸고, 이번에는 "빠르게 돌리기" 의 #17 안대로 Codex × GPT 와 AIDE × GPT 가 GPT 엔드포인트를 같이 씀 (AIDE 둘·Claude Code 둘도 동시에).
  429 재시도 (아래 표) 는 Codex 0~97번 (파일럿 4~100번), AIDE × GPT 0~11번 (파일럿 0번) → 늘었지만 대기 합이 몇 분 안이라 줄마다 하나씩으로 돌아가지 않음 (기준: Codex 200번·AIDE × GPT 50번을 넘으면 멈춤)
- 50회 모두 정상 종료·채점 (`valid_submission` 모두 참, `run_crashed` 없음, 다시 돌린 것·뺀 것 없음). 회귀 (QSAR_fish RMSE) 와 다중 (MIC logloss) 채점은 배치 0 의 첫 실행에서 확인 (sample 과 같은 클래스 열, 로컬 확인과 같은 계산)
- 큰 데이터 (APSFailure train 5만 · physiochemical 3만) 도 1시간 안에 모두 제출 (AIDE 는 늘 그렇듯 시간 종료, Codex 는 56·58분에 스스로 끝냄)

| 429 재시도 | QSAR_fish | MIC | Is-this | splice | students | polish | QSAR-TID | Amazon | physio | APS |
|---|---|---|---|---|---|---|---|---|---|---|
| Codex × GPT | 37 | 2 | 10 | 0 | 52 | 97 | 14 | 0 | 21 | 0 |
| AIDE × GPT | 9 | 0 | 1 | 0 | 6 | 11 | 4 | 0 | 1 | 0 |

- Sonnet 설정은 모두 0. 입력 토큰: Codex 2,670만~4,963만, AIDE × Sonnet 13만~806만 (Is-this 가 가장 많음), AIDE × GPT 12만~60만, Claude Code 6만~13만, +cont 31만~216만
- 걸린 시간: Claude Code 공식 설정 10~52초 (스스로 끝냄), +cont 1~44분 (이어 가기 5~11번, 모두 "도구 0개 구간 연속 3번" 으로 멈춤), Codex 36~59분, AIDE 60분 (시간 종료)

### 데이터셋별 Agent vs FM vs GBDT (r0f0, 1회째)

숫자는 `metric_error` (이진 1 − AUC, 다중 logloss, 회귀 RMSE, 낮을수록 좋음). FM·GBDT 는 refit 한 번 (결정 11), Agent 는 1회째 하나. 줄 순서는 점수 순, **굵게** = 그 데이터셋 최고.
TabArena 참고 줄은 공개 결과의 같은 split (r0f0). `kumo` 는 범주 열이 있으면 `kumo_cat_refit`, 없으면 `kumo_c8k_refit` (같은 설정). `tabpfn_v2` 는 TabArena 범위 밖 (QSAR-TID-11·APSFailure·Amazon·physiochemical) 은 없음.
비고의 `tfm_runtime`·`pip_target` = 공식 설정에서 Agent 가 스스로 TFM 을 깔아 씀 (Agent 결과로 셈, 결정 11). 시간은 Agent 는 하네스 시간, FM·GBDT 는 모델 시간.

#### QSAR_fish_toxicity (`tabarena-p05`): 회귀 · 피처 6 · 오염 낮음

| 방식 | 1회째 (r0f0) | 시간 | 종료·이어 가기 | 비고 |
|---|---|---|---|---|
| **FM: kumo** | **0.8855** | 8초 | | |
| TabArena Kumo-Tabular (참고) | 0.8852 | | | |
| FM: tabicl | 0.8938 | 5초 | | |
| Agent: Codex × GPT | 0.9003 | 59분 | 스스로 끝냄 | `tfm_runtime` tabicl tabpfn · 429 37번 |
| FM: tabpfn_v2 | 0.9080 | 6초 | | |
| Agent: AIDE × GPT | 0.9084 | 60분 | 시간 종료 | 429 9번 |
| Agent: Claude Code × Sonnet | 0.9121 | 20초 | 스스로 끝냄 |  |
| GBDT: catboost | 0.9123 | 2초 | | |
| Agent: AIDE × Sonnet | 0.9136 | 60분 | 시간 종료 |  |
| Agent: Claude Code × Sonnet +cont | 0.9218 | 63초 | 스스로 끝냄 (이어 가기 5번) |  |
| GBDT: lgbm | 0.9308 | 0초 | | |

#### MIC (`tabarena-p06`): 다중 · 피처 111 · 오염 낮음

| 방식 | 1회째 (r0f0) | 시간 | 종료·이어 가기 | 비고 |
|---|---|---|---|---|
| **FM: kumo** | **0.4412** | 9초 | | |
| TabArena Kumo-Tabular (참고) | 0.4418 | | | |
| Agent: AIDE × GPT | 0.4445 | 60분 | 시간 종료 |  |
| Agent: Codex × GPT | 0.4554 | 55분 | 스스로 끝냄 | 429 2번 |
| FM: tabicl | 0.4587 | 7초 | | |
| Agent: AIDE × Sonnet | 0.4591 | 60분 | 시간 종료 |  |
| Agent: Claude Code × Sonnet +cont | 0.4598 | 3분 | 스스로 끝냄 (이어 가기 9번) |  |
| GBDT: catboost | 0.4712 | 1742초 | | |
| Agent: Claude Code × Sonnet | 0.4767 | 20초 | 스스로 끝냄 |  |
| FM: tabpfn_v2 | 0.4838 | 8초 | | |
| GBDT: lgbm | 1.1010 | 3초 | | |

#### Is-this-a-good-customer (`tabarena-p07`): 이진 · 피처 13 · 오염 중간

| 방식 | 1회째 (r0f0) | 시간 | 종료·이어 가기 | 비고 |
|---|---|---|---|---|
| **FM: kumo** | **0.2565** | 8초 | | |
| TabArena Kumo-Tabular (참고) | 0.2571 | | | |
| FM: tabpfn_v2 | 0.2577 | 5초 | | |
| FM: tabicl | 0.2600 | 5초 | | |
| Agent: Claude Code × Sonnet +cont | 0.2617 | 86초 | 스스로 끝냄 (이어 가기 5번) |  |
| Agent: Claude Code × Sonnet | 0.2660 | 23초 | 스스로 끝냄 |  |
| Agent: AIDE × GPT | 0.2717 | 60분 | 시간 종료 | 429 1번 |
| Agent: AIDE × Sonnet | 0.2752 | 60분 | 시간 종료 |  |
| Agent: Codex × GPT | 0.2792 | 36분 | 스스로 끝냄 | `tfm_runtime` tabicl tabpfn · `pip_target` einops huggingface-hub interpret-core posthog pydantic-settings python-dotenv safetensors scikit-learn skrub tabicl tabpfn · 429 10번 |
| GBDT: catboost | 0.3197 | 7초 | | |
| GBDT: lgbm | 0.3363 | 1초 | | |

#### splice (`tabarena-p08`): 다중 · 피처 60 · 오염 높음 (교과서급)

| 방식 | 1회째 (r0f0) | 시간 | 종료·이어 가기 | 비고 |
|---|---|---|---|---|
| **Agent: Codex × GPT** | **0.06179** | 59분 | 스스로 끝냄 |  |
| FM: kumo | 0.06693 | 9초 | | |
| TabArena Kumo-Tabular (참고) | 0.06792 | | | |
| Agent: AIDE × GPT | 0.06987 | 60분 | 시간 종료 |  |
| FM: tabicl | 0.07219 | 7초 | | |
| Agent: AIDE × Sonnet | 0.07781 | 60분 | 시간 종료 |  |
| FM: tabpfn_v2 | 0.08207 | 8초 | | |
| Agent: Claude Code × Sonnet +cont | 0.08445 | 8분 | 스스로 끝냄 (이어 가기 9번) |  |
| GBDT: lgbm | 0.09400 | 1초 | | |
| GBDT: catboost | 0.09583 | 135초 | | |
| Agent: Claude Code × Sonnet | 0.1388 | 13초 | 스스로 끝냄 |  |

#### students_dropout_and_academic_success (`tabarena-p09`): 다중 · 피처 36 · 오염 낮음

| 방식 | 1회째 (r0f0) | 시간 | 종료·이어 가기 | 비고 |
|---|---|---|---|---|
| **FM: kumo** | **0.5251** | 9초 | | |
| TabArena Kumo-Tabular (참고) | 0.5238 | | | |
| FM: tabpfn_v2 | 0.5480 | 8초 | | |
| FM: tabicl | 0.5515 | 6초 | | |
| Agent: AIDE × Sonnet | 0.5515 | 60분 | 시간 종료 |  |
| Agent: AIDE × GPT | 0.5533 | 60분 | 시간 종료 | 429 6번 |
| Agent: Codex × GPT | 0.5536 | 55분 | 스스로 끝냄 | 429 52번 |
| Agent: Claude Code × Sonnet +cont | 0.5608 | 3분 | 스스로 끝냄 (이어 가기 8번) |  |
| Agent: Claude Code × Sonnet | 0.5630 | 25초 | 스스로 끝냄 |  |
| GBDT: catboost | 0.5702 | 27초 | | |
| GBDT: lgbm | 0.6419 | 2초 | | |

#### polish_companies_bankruptcy (`tabarena-p10`): 이진 · 피처 64 · 오염 낮음~중간

| 방식 | 1회째 (r0f0) | 시간 | 종료·이어 가기 | 비고 |
|---|---|---|---|---|
| **FM: kumo** | **0.00342** | 10초 | | |
| TabArena Kumo-Tabular (참고) | 0.00330 | | | |
| Agent: Codex × GPT | 0.00436 | 58분 | 스스로 끝냄 | `tfm_runtime` tabpfn · `pip_target` einops eval-type-backport huggingface-hub pydantic-settings scikit-learn tabpfn tabpfn-common-utils · 429 97번 |
| FM: tabicl | 0.01594 | 6초 | | |
| Agent: AIDE × GPT | 0.03090 | 60분 | 시간 종료 | 429 11번 |
| Agent: AIDE × Sonnet | 0.03166 | 60분 | 시간 종료 |  |
| Agent: Claude Code × Sonnet +cont | 0.03289 | 3분 | 스스로 끝냄 (이어 가기 8번) |  |
| GBDT: catboost | 0.03437 | 14초 | | |
| FM: tabpfn_v2 | 0.03541 | 8초 | | |
| GBDT: lgbm | 0.03743 | 4초 | | |
| Agent: Claude Code × Sonnet | 0.03816 | 10초 | 스스로 끝냄 |  |

#### QSAR-TID-11 (`tabarena-p11`): 회귀 · 피처 1,024 · 오염 낮음

| 방식 | 1회째 (r0f0) | 시간 | 종료·이어 가기 | 비고 |
|---|---|---|---|---|
| **FM: kumo** | **0.7867** | 15초 | | |
| TabArena Kumo-Tabular (참고) | 0.7840 | | | |
| Agent: Codex × GPT | 0.7976 | 57분 | 스스로 끝냄 | 429 14번 |
| Agent: AIDE × Sonnet | 0.8145 | 60분 | 시간 종료 |  |
| Agent: AIDE × GPT | 0.8221 | 60분 | 시간 종료 | 429 4번 |
| Agent: Claude Code × Sonnet +cont | 0.8251 | 29분 | 스스로 끝냄 (이어 가기 11번) |  |
| Agent: Claude Code × Sonnet | 0.8302 | 36초 | 스스로 끝냄 |  |
| GBDT: lgbm | 0.8459 | 2초 | | |
| FM: tabicl | 0.8643 | 29초 | | |
| GBDT: catboost | 0.8729 | 32초 | | |
| FM: tabpfn_v2 | – |  | | |

#### APSFailure (`tabarena-p12`): 이진 · 피처 170 · 오염 중간

| 방식 | 1회째 (r0f0) | 시간 | 종료·이어 가기 | 비고 |
|---|---|---|---|---|
| **FM: kumo** | **0.00265** | 38초 | | |
| TabArena Kumo-Tabular (참고) | 0.00250 | | | |
| FM: tabicl | 0.00454 | 29초 | | |
| Agent: AIDE × Sonnet | 0.00478 | 60분 | 시간 종료 |  |
| Agent: Claude Code × Sonnet +cont | 0.00487 | 4분 | 스스로 끝냄 (이어 가기 9번) |  |
| GBDT: lgbm | 0.00488 | 19초 | | |
| GBDT: catboost | 0.00496 | 57초 | | |
| Agent: Claude Code × Sonnet | 0.00521 | 19초 | 스스로 끝냄 |  |
| Agent: Codex × GPT | 0.00525 | 56분 | 스스로 끝냄 |  |
| Agent: AIDE × GPT | 0.00572 | 60분 | 시간 종료 |  |
| FM: tabpfn_v2 | – |  | | |

#### Amazon_employee_access (`tabarena-p03`): 이진 · 피처 9 · 오염 높음 (대회)

| 방식 | 1회째 (r0f0) | 시간 | 종료·이어 가기 | 비고 |
|---|---|---|---|---|
| **Agent: AIDE × Sonnet** | **0.1068** | 60분 | 시간 종료 |  |
| Agent: Claude Code × Sonnet +cont | 0.1093 | 14분 | 스스로 끝냄 (이어 가기 11번) |  |
| Agent: Codex × GPT | 0.1102 | 55분 | 스스로 끝냄 |  |
| Agent: AIDE × GPT | 0.1106 | 60분 | 시간 종료 |  |
| GBDT: catboost | 0.1192 | 42초 | | |
| Agent: Claude Code × Sonnet | 0.1201 | 52초 | 스스로 끝냄 |  |
| FM: kumo | 0.1277 | 12초 | | |
| TabArena Kumo-Tabular (참고) | 0.1247 | | | |
| FM: tabicl | 0.1511 | 8초 | | |
| GBDT: lgbm | 0.1576 | 3초 | | |
| FM: tabpfn_v2 | – |  | | |

#### physiochemical_protein (`tabarena-p04`): 회귀 · 피처 9 · 오염 중간

| 방식 | 1회째 (r0f0) | 시간 | 종료·이어 가기 | 비고 |
|---|---|---|---|---|
| **FM: kumo** | **2.8520** | 13초 | | |
| TabArena Kumo-Tabular (참고) | 2.8485 | | | |
| Agent: Codex × GPT | 2.9772 | 58분 | 스스로 끝냄 | 429 21번 |
| FM: tabicl | 3.0168 | 6초 | | |
| Agent: AIDE × Sonnet | 3.2058 | 60분 | 시간 종료 |  |
| Agent: AIDE × GPT | 3.2756 | 60분 | 시간 종료 | 429 1번 |
| Agent: Claude Code × Sonnet +cont | 3.2828 | 44분 | 스스로 끝냄 (이어 가기 10번) |  |
| Agent: Claude Code × Sonnet | 3.6471 | 27초 | 스스로 끝냄 |  |
| GBDT: lgbm | 3.9614 | 2초 | | |
| GBDT: catboost | 3.9767 | 5초 | | |
| FM: tabpfn_v2 | – |  | | |

데이터셋별 메모

- QSAR_fish: FM 근소 (FM − GBDT −2.9%) 인 곳에서도 Agent 는 FM 을 못 넘음. Codex 는 TabICL·TabPFN 을 보통 `pip install` 로 깔아 씀 (`pip_added`, `tfm_runtime`) → 0.9003 (kumo 0.8855, tabicl 0.8938 보다 나쁨)
- MIC: AIDE × GPT 0.4445 가 kumo 0.4412 바로 아래 (0.2σ). 우리 lgbm 1.10 은 기본 설정이 8클래스·작은 데이터에서 과적합 (TabArena LightGBM 도 기본 GBDT 중 나쁜 쪽)
- Is-this-a-good-customer: Agent 다섯이 모두 FM 셋보다 나쁨. 최고는 23초·86초에 끝낸 Claude Code (0.2660·0.2617). Codex 는 TabICL·TabPFN 을 `--target` 으로 깔아 썼지만 0.2792 로 Agent 중 꼴찌
- splice: **Codex 만 FM 보다 나음** (0.0618 대 kumo 0.0669, −7.7%, 0.3σ = split 흔들림 안). 풀이는 코돈·k-mer 단어 피처 + 로지스틱·CatBoost·SVM·신경망 섞기 + train 의 거의 같은 서열 찾기 (60자리 중 일치 수) + 경계 바로 옆 AG/GT 두 염기.
  하네스 기록에서 과제를 "splice junction dataset" 으로 부름 (설명의 exon/intron boundary 에서 나온 말로 봄). 원본 URL 접근 없음
- students_dropout: FM 이 뚜렷이 나음 (Agent 최고 AIDE × Sonnet 0.5515, kumo +5.0%, 1.9σ)
- polish: kumo 0.00342 에 Codex 0.00436 이 가까움, 나머지 Agent 는 0.031~0.038 (GBDT 0.034 근처, FM 의 약 10배 오차). Codex 는 비율 열 사이의 산술 관계 (예: `sales_to_total_assets` = `total_sales_to_total_assets` 인 행 무리) 를 찾아 피처로 만들고,
  train 의 같은·거의 같은 행 찾기 + 스스로 깐 TabPFN (`--target`) 을 섞음. FM 이 GBDT 보다 크게 나은 까닭 (열 사이 구조) 을 Agent 중 Codex 만 찾은 것으로 봄
- QSAR-TID-11 (피처 1,024, 모두 0/1 지문): Codex 0.7976 이 kumo 0.7867 다음 (0.2σ). Codex 는 지문에 맞는 Tanimoto 유사도로 SVR·가우스 과정·kNN + 잡음 제거 자동 부호기·PCA·torch MLP·GBDT 섞기. 다른 Agent 는 0.81~0.83 (우리 GBDT 0.85·0.87 보다는 나음)
- APSFailure (train 5만): kumo 0.00265 가 가장 좋고, Agent 최고 AIDE × Sonnet 0.00478 은 우리 lgbm 0.00488 과 비슷 (FM 의 1.8배 오차). Codex 0.00525 는 GBDT 보다 나쁨
- **Amazon (대조군)**: Agent 다섯이 모두 FM (kumo 0.1277) 보다 낫고, 넷이 우리 CatBoost (0.1192) 보다도 나음 (최고 AIDE × Sonnet 0.1068, CatBoost 대비 −10%). 이 데이터의 알려진 풀이 그대로 (AIDE × Sonnet: 열 두 개씩 묶은 범주 + 빈도 + 원-핫 로지스틱 + CatBoost·LightGBM 섞기).
  Codex 는 하네스 기록에서 "the original Amazon dataset", "top models with one-hot encoding might achieve around .89" 처럼 데이터셋을 알아봄 (이름은 알리지 않았음, 결정 3). 원본·Kaggle URL 접근은 0. AIDE × Sonnet·Claude Code 기록에는 데이터셋 이름이 없음
- physiochemical (train 3만): kumo 2.852, Codex 2.977 (+4.4%, 3.4σ), 나머지 Agent 3.21~3.65 (GBDT 3.96 보다는 나음)

### 요약: 어떤 데이터에서 FM 이 이기나 (#26 10개 + 파일럿 2개, r0f0 1회째)

| 데이터셋 | 과제 | 행 | 피처 (범주) | FM − GBDT (TabArena 기본, split 평균, ÷σ) | FM − 우리 GBDT refit (split 평균) | 오염 (판단) | r0f0 FM 최고 (kumo) | Agent 최고 (설정) | Agent − FM (÷σ) | FM 보다 나은 Agent | 승자 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| QSAR_fish_toxicity | 회귀 | 907 | 6 (0) | -2.9% (-0.5) | -3.9% | 낮음 | 0.8855 | 0.9003 (Codex × GPT) | +1.7% (+0.3) | 0 / 5 | FM |
| MIC | 다중 | 1,699 | 111 (94) | -7.7% (-1.9) | -9.0% | 낮음 | 0.4412 | 0.4445 (AIDE × GPT) | +0.7% (+0.2) | 0 / 5 | FM |
| Is-this-a-good-customer | 이진 | 1,723 | 13 (8) | -2.9% (-0.3) | -12.8% | 중간 | 0.2565 | 0.2617 (Claude Code × Sonnet +cont) | +2.0% (+0.3) | 0 / 5 | FM |
| splice | 다중 | 3,190 | 60 (60) | -21.3% (-1.6) | -22.3% | 높음 (교과서급) | 0.06693 | 0.06179 (Codex × GPT) | -7.7% (-0.3) | 1 / 5 | **Agent** |
| students_dropout_and_academic_success | 다중 | 4,424 | 36 (17) | -8.1% (-3.2) | -8.3% | 낮음 | 0.5251 | 0.5515 (AIDE × Sonnet) | +5.0% (+1.9) | 0 / 5 | FM |
| polish_companies_bankruptcy | 이진 | 5,910 | 64 (0) | -88.8% (-20.4) | -89.4% | 낮음~중간 | 0.00342 | 0.00436 (Codex × GPT) | +27.6% (+0.6) | 0 / 5 | FM |
| QSAR-TID-11 | 회귀 | 5,742 | 1,024 (0) | -6.6% (-1.1) | -8.1% | 낮음 | 0.7867 | 0.7976 (Codex × GPT) | +1.4% (+0.2) | 0 / 5 | FM |
| APSFailure | 이진 | 76,000 | 170 (0) | -39.0% (-2.0) | -40.4% | 중간 | 0.00265 | 0.00478 (AIDE × Sonnet) | +80.4% (+1.5) | 0 / 5 | FM |
| Amazon_employee_access | 이진 | 32,769 | 9 (9) | +4.5% (+0.8) | +7.5% | 높음 (대회) | 0.1277 | 0.1068 (AIDE × Sonnet) | -16.4% (-3.3) | 5 / 5 | **Agent** |
| physiochemical_protein | 회귀 | 45,730 | 9 (0) | -20.2% (-19.2) | -29.7% | 중간 | 2.8520 | 2.9772 (Codex × GPT) | +4.4% (+3.4) | 0 / 5 | FM |
| hazelnut-spread-contaminant-detection | 이진 | 2,400 | 30 (0) | -89.7% (-27.5) | -87.6% | 낮음 | 0.00390 | 0.00640 (Codex × GPT) | +64.1% (+3.0) | 0 / 5 | FM |
| in_vehicle_coupon_recommendation | 이진 | 12,684 | 24 (22) | -9.9% (-3.9) | -13.6% | 중간 | 0.1371 | 0.1315 (Codex × GPT) | -4.1% (-1.4) | 1 / 5 | **Agent** |

- σ = FM (kumo) 의 split 표준편차 (모든 split). "Agent − FM" 은 r0f0 한 split 끼리의 차이를 그 σ 로 나눈 것 (1σ 안 = split 흔들림 정도)
- 승자 = r0f0 에서 FM·GBDT·Agent 다섯 중 최고. 12개 중 **FM 9 · Agent 3 (in_vehicle · splice · Amazon) · GBDT 0**. Amazon 에서 우리 CatBoost 는 FM 보다 낫지만 Agent 넷보다 나쁨
- 13번째 행 kddcup09 (2026-11-20) 는 이 표에 넣지 않고 [행 추가: kddcup09_appetency](#행-추가-kddcup09_appetency-r0f0-2026-11-19--11-20) 에 둠 (Agent 최고 0.1559 · kumo 0.1594, 승자 Agent → 13개 중 FM 9 · Agent 4)

경향 (12개, Agent 1회째. 데이터셋이 적어 판단은 약함)

- **FM 이 지는 셋은 모두 범주 열이 대부분인 데이터**: splice (범주 100%), Amazon (100%, 고카디널리티 id), in_vehicle (92%). 나머지 9개 (범주 열 없음 6 · 범주 47~85% 3) 는 모두 FM 이 이김 (MIC 는 범주 85% 이지만 0/1 표시·순서 있는 단계 위주, FM 0.2σ 차로 이김)
- **같은 셋이 오염 위험도 큼**: splice (교과서급), Amazon (Kaggle 대회, Codex 가 알아봄), in_vehicle (Codex 가 응답자 id 를 train+test 로 다시 만드는 "inductive" 밖 풀이, 파일럿). 오염 "낮음"·"낮음~중간" 6개 (hazelnut · QSAR_fish · MIC · students · polish · QSAR-TID) 는 모두 FM 이 이김.
  → "범주 많은 데이터에서 Agent 가 강함" 과 "알려진 데이터에서 Agent 가 강함" 이 이 12개에서는 갈리지 않음 (둘이 겹침)
- **크기**: 작음 (< 2.5k) 4개·중간 4개·큼 4개 모두에서 FM 이 대부분 이김. 크기만으로 갈리는 경향은 없음. 큰 데이터 (APSFailure·physiochemical) 도 Agent 가 1시간 안에 끝냄
- **FM − GBDT 차이가 큰 곳** (polish · hazelnut · APS · physio · splice, −20% 이하): Agent 대부분은 GBDT 근처에 머묾. 따라오는 것은 Codex 뿐 (polish 0.6σ, physio 3.4σ, hazelnut 3.0σ 뒤, splice 는 넘음)
- **근소한 곳** (QSAR_fish · Is-this, 3% 안): Agent 10회 중 FM 을 넘은 것 없음. 데이터셋마다 최고 Agent 만 0.3σ 안 뒤이고, 나머지는 더 뒤 (가장 먼 것: Is-this 의 Codex 1.1σ, QSAR_fish 의 +cont 0.7σ)
- **Agent − FM 이 split 흔들림 (1σ) 밖**인 것은 6개: FM 이 나음 4 (physio 3.4σ · hazelnut 3.0σ · students 1.9σ · APS 1.5σ), Agent 가 나음 2 (Amazon 3.3σ · in_vehicle 1.4σ). 나머지 6개 (QSAR_fish · MIC · Is-this · splice · polish · QSAR-TID) 는 1회째 한 번으로는 승패를 말하기 어려움 → 2회째 (결정 2) 가 필요한 곳
- 설정별: 데이터셋 최고 Agent 는 12개 중 Codex × GPT 7, AIDE × Sonnet 3 (students · APS · Amazon), AIDE × GPT 1 (MIC), Claude Code × Sonnet +cont 1 (Is-this). Codex 가 공식 설정에서 TFM 을 스스로 깐 것은 #26 10개 중 3 (QSAR_fish · Is-this · polish) 이고 셋 모두 FM 을 못 넘음

### 사후 검사 (#26 50회)

- `data_source_urls` 0 · `kaggle_urls` 0 (50회 모두). polish × Codex 는 처음에 271 로 나왔지만 Agent 가 작업 디렉터리에 깐 sklearn 패키지 코드 탓이라 검사를 고침 (9 의 "사후 검사", `recheck` 뒤 0)
- 원본 데이터 접근 흔적은 없음. 다만 Amazon 은 Codex 가 기록에서 데이터셋을 알아봄 (위). 이름을 알리지 않아도 피처 이름 (`MGR_ID`·`ROLE_ROLLUP_1` 등) 으로 알아볼 수 있는 대회 데이터 (5 의 "익명화" 가 적은 남는 위험)
- TFM 을 스스로 깐 실행 (`tfm_runtime`): Codex × QSAR_fish (tabicl·tabpfn, 보통 `pip install`), Codex × Is-this (tabicl·tabpfn, `--target`), Codex × polish (tabpfn, `--target`). Agent 결과로 셈 (결정 11)
- "inductive" 규칙 (5): splice·polish 의 Codex 는 test 행마다 train 의 같은·가까운 행을 찾음 (train 레이블만 씀, test 끼리 안 봄 → 규칙 안). train+test 로 전처리를 fit 했는지는 50회 모두는 보지 않음 (확인 못 함)

### 실행 메모

| 일 | 내용 | 처리 |
|---|---|---|
| GBDT refit 느림 | MIC 의 CatBoost 가 split 하나에 약 30분 → 작업 하나로는 12시간 제한에 걸림 (위 "GBDT refit") | 취소하고 9개 작업으로 나눠 동시에 (사용자). 끝난 33 split 은 그대로 씀 |
| 로그인 만료 (14:28) | 로컬 Databricks CLI 의 토큰 갱신이 실패해 상태 조회가 안 됨. 원격 작업은 계속 돎 | 사용자가 다시 로그인 (약 25분 멈춤). 그동안 끝난 QSAR_fish 4회는 다음 kick 에 채점 |
| 진행 스크립트 버그 | 마지막 데이터셋 (polish) 을 내고 기다리지 않고 끝남 | 고쳐서 배치 2 를 polish 대기부터 다시 시작 |
| 로컬 앱 재시작 (22시쯤) | 배치 2 진행 스크립트 (로컬) 가 같이 죽음. 이미 낸 Amazon 5회는 원격에서 계속 돎 | Amazon 대기부터 다시 시작 (새 세션으로 띄움). 빠진 실행 없음 |
| 사후 검사 오탐 | polish × Codex 의 `data_source_urls` 271 (깐 패키지 코드) | 깐 패키지 코드를 빼고 `recheck` (9 의 "사후 검사"). 빼는 규칙은 리뷰 뒤 `*.dist-info/RECORD` 목록으로 바꾸고 `recheck` 를 다시 함 (값 같음, 2026-11-17) |
| 같은 값 두 번 채점 | 처음 GBDT 작업의 채점 요청 (135 대회) 과 새 작업마다 낸 채점 요청이 둘 다 처리돼 이번 GBDT 270행이 모두 두 번 | 값이 같아 그대로 둠 (`compare`·`elo` 는 마지막 행만 씀) |

비용: Agent A10 약 35~40시간 (하네스 시간 합 31.0시간 + 실행마다 설치·회수 5~10분, 작업별 시간은 재지 않음), GBDT serverless CPU 모델 시간 15.8시간 (취소한 작업 1.6시간 포함). 어림 (50 A10 시간) 안

### 다음에 정할 것

- ~~**C2 (+fm)**~~: 7개 × 2설정 (14회, 2026-11-18) 뒤 배치 2 의 5개 × 2설정 (10회, 2026-11-19) 로 12개 모두 (아래 "C2 (+fm)")
- **2회째 (결정 2)**: Agent − FM 이 1σ 안인 6개 (QSAR_fish · MIC · Is-this · splice · polish · QSAR-TID) 는 1회째로 승패를 말하기 어려움. 10개 모두 (약 50 A10 시간) 또는 1σ 안인 6개만 (약 30)
- 범주 많음 대 오염을 가르려면 **범주가 많고 덜 알려진 데이터** 가 필요 (예: 대안의 hiva_agnostic 은 수치, website_phishing 은 범주 9열 · UCI). 후보를 더 찾을지

## 범주 비율과 FM − GBDT (51개, 2026-11-17)

"Agent 확장 1회째" 의 요약에서 Agent 가 이긴 셋 (splice · Amazon · in_vehicle) 은 모두 범주 열이 대부분이고 오염 위험도 높음. 그 단서 중 "범주" 쪽을 Agent 없이 FM 결과 (51개 × 모든 split, 816) 로 먼저 봄:
**범주 열 비율이 높을수록 kumo (ours) 와 TabArena 기본 GBDT 의 차이가 줄어드는가.**
FM − GBDT 는 LLM 이 끼지 않는 비교라 오염과는 무관함. 그래서 이 절은 범주와 오염을 가르지 않음 (데이터를 알아볼 수 없게 바꿔 직접 가르는 일은 따로 함). 새로 돌린 것 없음 (Agent·GPU·원격 작업 없음).

`uv run --frozen fab ta cats` → `results/tabarena_cats.csv` (`fab/bench/tabarena/cats.py`). 로컬 캐시 (`FAB_TABARENA/data/cache`) 의 51개와 `results/tabarena.csv` · `results/tabarena_reference.csv` 만 읽음.

### 어떻게 셌나

| 항목 | 기준 |
|---|---|
| 범주 열 | F0 이 쓰는 `categorical(meta)` 그대로 (OpenML 범주 표시 + 문자열 열, 9 의 "피처 (F0)") |
| 피처 | `meta` 의 `features` (타깃 뺌). 부록의 "피처" 는 메타데이터 `num_features` (타깃 포함) 라 1 큼 |
| 카디널리티 | 범주 열마다 결측을 뺀 고유값 수, 데이터 전체 (train·test 를 가르지 않음). 결측이 글 `nan` 으로 든 열이 있음 (MIC 93열 · kddcup09 22열 · HR 7열 · in_vehicle 6열 · Diabetes130US 5열) → 셀 때만 결측으로 봄 (F0 은 `nan` 을 범주 값 하나로 씀, 고치지 않음) |
| 이진 | 고유값이 2개인 범주 열 (결측이 더 있어도). "이진 아닌 범주" = 고유값 3개 이상 |
| 고카디널리티 | 고유값이 100 을 넘는 범주 열 수 (`> 100`), 가장 큰 범주 열의 고유값 ÷ 행 수 (`최대 / 행`) |
| 차이 (%) | (split 평균 `metric_error` − 기본 GBDT) ÷ 기본 GBDT. 기본 GBDT = TabArena CatBoost·LightGBM 중 모든 split 평균이 나은 쪽 (`default_gbdt`, 부록과 같음). 음수 = FM 이 나음 |
| ÷σ | (split 평균 − 기본 GBDT) ÷ 그 방법의 split 표준편차 (모든 split, ddof 1. "Agent 확장 1회째" 요약 표와 같은 값) |
| kumo (ours) | `kumo_cat_refit`, 없으면 `kumo_c8k_refit` (Elo 의 FM 풀과 같음). 참고로 TabArena Kumo-Tabular · TabICLv2 를 같은 식으로 (σ 는 각자의 split 표준편차) |
| 상관 | 스피어만 순위 상관 (scipy, p 는 양측). 양수 = 범주가 많을수록 차이가 0 쪽 (FM 의 우위가 작음) |
| 오염 묶음 | 6 의 표에 이름이 적힌 것 + Kaggle 데이터셋 (부록 출처 Kaggle 15개 중 대회 3개를 뺀 12개). 이름이 없는 17개는 "미분류" (판단하지 않음) |

### 51개 (범주 비율 순)

| 데이터셋 | 과제 | 행 | 피처 | 범주 (%) | 이진 | 고유값 중앙 · 최대 | 최대 / 행 | > 100 | kumo (ours) (÷σ) | Kumo-Tabular (÷σ) | TabICLv2 (÷σ) | 오염 묶음 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| website_phishing | 다중 | 1,353 | 9 | 9 (100%) | 2 | 3 · 3 | 0.2% | 0 | -17.0% (-1.9) | -17.1% (-1.9) | -11.9% (-1.1) | 미분류 |
| splice | 다중 | 3,190 | 60 | 60 (100%) | 0 | 5 · 6 | 0.2% | 0 | -21.3% (-1.6) | -21.2% (-1.6) | -9.9% (-0.5) | 교과서급 |
| hiva_agnostic | 다중 | 3,845 | 1,617 | 1,617 (100%) | 1,617 | 2 · 2 | 0.1% | 0 | -0.4% (-4.4) | -0.4% (-4.2) | +13.6% (+2.8) | 미분류 |
| NATICUSdroid | 이진 | 7,491 | 86 | 86 (100%) | 86 | 2 · 2 | 0.0% | 0 | -16.5% (-1.8) | -16.4% (-1.8) | -13.1% (-1.6) | 덜 알려진 |
| Amazon_employee_access | 이진 | 32,769 | 9 | 9 (100%) | 0 | 343 · 7,518 | 22.9% | 8 | +4.5% (+0.8) | +3.4% (+0.6) | +24.3% (+6.2) | 대회 |
| in_vehicle_coupon_recommendation | 이진 | 12,684 | 24 | 22 (92%) | 7 | 5 · 25 | 0.2% | 0 | -9.9% (-3.9) | -10.2% (-4.0) | -3.0% (-1.1) | 미분류 |
| MIC | 다중 | 1,699 | 111 | 94 (85%) | 78 | 2 · 9 | 0.5% | 0 | -7.7% (-1.9) | -7.4% (-1.9) | -2.1% (-0.4) | 덜 알려진 |
| anneal | 다중 | 898 | 38 | 32 (84%) | 14 | 2 · 8 | 0.9% | 0 | -70.9% (-2.1) | -70.9% (-2.1) | -48.4% (-1.3) | 교과서급 |
| HR_Analytics_Job_Change_of_Data_Scientists | 이진 | 19,158 | 12 | 10 (83%) | 1 | 6 · 123 | 0.6% | 1 | -1.5% (-0.5) | -1.5% (-0.5) | -0.8% (-0.2) | Kaggle |
| Diabetes130US | 이진 | 71,518 | 47 | 39 (83%) | 9 | 4 · 758 | 1.1% | 3 | -3.5% (-1.5) | -3.5% (-1.5) | +2.4% (+1.0) | 미분류 |
| customer_satisfaction_in_airline | 이진 | 129,880 | 21 | 16 (76%) | 2 | 6 · 6 | 0.0% | 0 | -24.0% (-7.8) | -24.2% (-8.2) | -8.8% (-2.7) | Kaggle |
| credit-g | 이진 | 1,000 | 20 | 13 (65%) | 2 | 4 · 10 | 1.0% | 0 | -9.6% (-0.9) | -9.4% (-0.9) | -3.6% (-0.4) | 교과서급 |
| Is-this-a-good-customer | 이진 | 1,723 | 13 | 8 (62%) | 3 | 3 · 22 | 1.3% | 0 | -2.9% (-0.3) | -3.1% (-0.4) | +0.3% (+0.0) | Kaggle |
| bank-marketing | 이진 | 45,211 | 13 | 8 (62%) | 3 | 3 · 12 | 0.0% | 0 | -1.5% (-0.7) | -1.6% (-0.7) | +2.8% (+0.7) | 교과서급 |
| healthcare_insurance_expenses | 회귀 | 1,338 | 6 | 3 (50%) | 2 | 2 · 4 | 0.3% | 0 | -3.5% (-0.5) | -3.5% (-0.5) | -1.8% (-0.3) | Kaggle |
| Fitness_Club | 이진 | 1,500 | 6 | 3 (50%) | 1 | 6 · 7 | 0.5% | 0 | -4.3% (-0.6) | -4.3% (-0.6) | -3.9% (-0.6) | Kaggle |
| students_dropout_and_academic_success | 다중 | 4,424 | 36 | 17 (47%) | 8 | 6 · 46 | 1.0% | 0 | -8.1% (-3.2) | -8.3% (-3.4) | -3.3% (-1.2) | 미분류 |
| online_shoppers_intention | 이진 | 12,330 | 17 | 7 (41%) | 1 | 9 · 20 | 0.2% | 0 | -8.4% (-2.3) | -8.3% (-2.3) | -4.8% (-1.1) | 미분류 |
| Bank_Customer_Churn | 이진 | 10,000 | 10 | 4 (40%) | 3 | 2 · 3 | 0.0% | 0 | -5.4% (-0.7) | -5.4% (-0.7) | -1.3% (-0.2) | Kaggle |
| E-CommereShippingData | 이진 | 10,999 | 10 | 4 (40%) | 1 | 3 · 5 | 0.0% | 0 | +0.2% (+0.1) | +0.0% (+0.0) | -0.2% (-0.1) | Kaggle |
| Marketing_Campaign | 이진 | 2,240 | 25 | 9 (36%) | 6 | 2 · 663 | 29.6% | 1 | -11.4% (-0.8) | -22.2% (-1.6) | -19.6% (-1.5) | Kaggle |
| Food_Delivery_Time | 회귀 | 45,451 | 9 | 3 (33%) | 0 | 4 · 1,320 | 2.9% | 1 | -2.4% (-4.7) | -2.4% (-4.9) | +3.0% (+3.8) | Kaggle |
| diamonds | 회귀 | 53,940 | 9 | 3 (33%) | 0 | 7 · 8 | 0.0% | 0 | -7.0% (-4.8) | -7.0% (-4.9) | -4.1% (-2.4) | 교과서급 |
| seismic-bumps | 이진 | 2,584 | 15 | 4 (27%) | 2 | 2.5 · 3 | 0.1% | 0 | -6.8% (-0.8) | -7.0% (-0.8) | -6.3% (-0.6) | 덜 알려진 |
| churn | 이진 | 5,000 | 19 | 4 (21%) | 2 | 2.5 · 51 | 1.0% | 0 | -7.2% (-0.5) | -7.3% (-0.5) | -14.7% (-1.2) | 미분류 |
| airfoil_self_noise | 회귀 | 1,503 | 5 | 1 (20%) | 0 | 27 · 27 | 1.8% | 0 | -42.5% (-8.0) | -42.5% (-8.1) | -30.6% (-5.7) | 교과서급 |
| SDSS17 | 다중 | 78,053 | 11 | 2 (18%) | 0 | 3,632 · 6,264 | 8.0% | 2 | -19.7% (-7.8) | -19.8% (-8.0) | -5.1% (-1.7) | Kaggle |
| kddcup09_appetency | 이진 | 50,000 | 212 | 38 (18%) | 4 | 17.5 · 15,415 | 30.8% | 11 | +1.7% (+0.3) | -7.1% (-1.3) | +10.0% (+1.5) | 미분류 |
| Another-Dataset-on-used-Fiat-500 | 회귀 | 1,538 | 7 | 1 (14%) | 0 | 3 · 3 | 0.2% | 0 | -5.2% (-1.6) | -5.2% (-1.6) | -3.0% (-0.9) | Kaggle |
| credit_card_clients_default | 이진 | 30,000 | 23 | 3 (13%) | 1 | 4 · 7 | 0.0% | 0 | -3.8% (-2.5) | -3.8% (-2.5) | -3.0% (-1.7) | 미분류 |
| qsar-biodeg | 이진 | 1,054 | 41 | 5 (12%) | 3 | 2 · 11 | 1.0% | 0 | -15.2% (-1.0) | -15.2% (-1.0) | -15.1% (-0.9) | 미분류 |
| wine_quality | 회귀 | 6,497 | 12 | 1 (8%) | 1 | 2 · 2 | 0.0% | 0 | -5.9% (-2.3) | -5.9% (-2.3) | -4.4% (-1.5) | 교과서급 |
| miami_housing | 회귀 | 13,776 | 15 | 1 (7%) | 1 | 2 · 2 | 0.0% | 0 | -11.0% (-2.2) | -11.2% (-2.3) | -2.6% (-0.5) | 미분류 |
| coil2000_insurance_policies | 이진 | 9,822 | 85 | 5 (6%) | 0 | 10 · 39 | 0.4% | 0 | -11.3% (-2.2) | -11.2% (-2.3) | -6.4% (-1.2) | 덜 알려진 |
| blood-transfusion-service-center | 이진 | 748 | 4 | 0 (0%) | 0 | – | – | 0 | -5.4% (-0.5) | -5.5% (-0.5) | -6.6% (-0.6) | 교과서급 |
| diabetes | 이진 | 768 | 8 | 0 (0%) | 0 | – | – | 0 | -6.1% (-0.5) | -6.2% (-0.5) | -3.3% (-0.2) | 교과서급 |
| QSAR_fish_toxicity | 회귀 | 907 | 6 | 0 (0%) | 0 | – | – | 0 | -2.9% (-0.5) | -3.0% (-0.5) | -2.2% (-0.4) | 미분류 |
| maternal_health_risk | 다중 | 1,014 | 6 | 0 (0%) | 0 | – | – | 0 | -21.2% (-1.9) | -21.0% (-1.8) | -16.7% (-1.4) | 미분류 |
| concrete_compressive_strength | 회귀 | 1,030 | 8 | 0 (0%) | 0 | – | – | 0 | -10.2% (-0.9) | -10.2% (-0.9) | -6.4% (-0.6) | 교과서급 |
| hazelnut-spread-contaminant-detection | 이진 | 2,400 | 30 | 0 (0%) | 0 | – | – | 0 | -89.7% (-27.5) | -89.9% (-26.7) | -80.3% (-14.8) | 덜 알려진 |
| Bioresponse | 이진 | 3,751 | 1,776 | 0 (0%) | 0 | – | – | 0 | -9.8% (-2.0) | -10.7% (-1.9) | +8.2% (+1.3) | 대회 |
| QSAR-TID-11 | 회귀 | 5,742 | 1,024 | 0 (0%) | 0 | – | – | 0 | -6.6% (-1.1) | -7.2% (-1.1) | +4.7% (+0.7) | 덜 알려진 |
| polish_companies_bankruptcy | 이진 | 5,910 | 64 | 0 (0%) | 0 | – | – | 0 | -88.8% (-20.4) | -89.4% (-22.1) | -60.0% (-6.2) | 덜 알려진 |
| taiwanese_bankruptcy_prediction | 이진 | 6,819 | 94 | 0 (0%) | 0 | – | – | 0 | -22.2% (-2.4) | -21.5% (-2.4) | -4.1% (-0.4) | 미분류 |
| heloc | 이진 | 10,459 | 23 | 0 (0%) | 0 | – | – | 0 | -2.4% (-1.3) | -2.4% (-1.3) | -1.6% (-0.8) | Kaggle |
| jm1 | 이진 | 10,885 | 21 | 0 (0%) | 0 | – | – | 0 | -15.2% (-7.7) | -15.1% (-7.9) | -10.3% (-4.3) | 덜 알려진 |
| houses | 회귀 | 20,640 | 8 | 0 (0%) | 0 | – | – | 0 | -14.9% (-12.2) | -15.0% (-13.6) | -7.8% (-5.5) | 교과서급 |
| superconductivity | 회귀 | 21,263 | 81 | 0 (0%) | 0 | – | – | 0 | -8.8% (-4.1) | -8.7% (-4.2) | -4.6% (-2.1) | 미분류 |
| physiochemical_protein | 회귀 | 45,730 | 9 | 0 (0%) | 0 | – | – | 0 | -20.2% (-19.2) | -20.3% (-19.5) | -15.4% (-13.9) | 미분류 |
| APSFailure | 이진 | 76,000 | 170 | 0 (0%) | 0 | – | – | 0 | -39.0% (-2.0) | -39.4% (-2.0) | -23.5% (-1.0) | 미분류 |
| GiveMeSomeCredit | 이진 | 150,000 | 10 | 0 (0%) | 0 | – | – | 0 | -2.2% (-1.3) | -2.2% (-1.3) | -0.5% (-0.3) | 대회 |

- 범주 열이 있는 것 34개, 없는 것 17개 ("FM 프로토콜과 이름" 의 34 · 17 과 같음)
- hiva_agnostic 은 F0 기준으로 1,617열 모두 범주 (0/1). "다음에 정할 것" 의 "hiva_agnostic 은 수치" 와 다름 (그 문장은 값이 숫자라는 뜻으로 보임). NATICUSdroid 도 86열 모두 0/1 범주. MIC 는 94열 중 78열이 0/1 (+ 결측)
- kumo (ours) 가 기본 GBDT 보다 나쁜 것은 3개: E-CommereShippingData (+0.2%, +0.1σ) · kddcup09_appetency (+1.7%, +0.3σ) · Amazon (+4.5%, +0.8σ). 모두 1σ 안. kddcup09 는 TabArena Kumo 가 −7.1% 라 우리 kumo 만 뒤짐 ("FM 이 TabArena 와 맞는지" 의 "kddcup09": 남은 차이는 TabArena 의 AutoGluon 전처리)

### 상관 (51개)

| 무엇과 | kumo (ours) % | ÷σ | Kumo-Tabular % | ÷σ | TabICLv2 % | ÷σ |
|---|---|---|---|---|---|---|
| 범주 비율 | +0.20 (p 0.16) | +0.19 (0.18) | +0.20 (0.16) | +0.19 (0.19) | +0.21 (0.13) | +0.27 (0.053) |
| 이진 아닌 범주 비율 | +0.20 (0.17) | +0.21 (0.15) | +0.19 (0.19) | +0.20 (0.16) | +0.20 (0.16) | +0.23 (0.10) |
| 고유값 > 100 인 열 수 | **+0.30 (0.030)** | +0.19 (0.18) | +0.20 (0.16) | +0.12 (0.42) | **+0.31 (0.029)** | **+0.31 (0.027)** |
| 최대 / 행 (범주 있는 34개) | +0.05 (0.76) | +0.24 (0.18) | −0.06 (0.74) | +0.19 (0.27) | +0.02 (0.92) | +0.22 (0.22) |
| 행 수 | +0.15 (0.29) | −0.31 (0.026) | +0.13 (0.38) | −0.34 (0.014) | +0.26 (0.07) | −0.03 (0.83) |

- 범주 열이 있는 34개만: 범주 비율과 kumo (ours) % 는 +0.01 (0.94), ÷σ 는 +0.13 (0.45). 이진 범주의 몫 (이진 ÷ 범주) 과는 +0.02 (0.92)
  → 51개의 +0.20 은 대부분 "범주 0 대 있음" 의 차이에서 옴. 범주가 있는 데이터 안에서는 비율이 높아도 차이가 줄지 않음
- 행 수와 ÷σ 의 음의 상관은 큰 데이터일수록 split 흔들림 (σ) 이 작아 ÷σ 가 커지는 것 (% 로는 +0.15)

### 묶음별 중앙값

범주 비율 (0~50% 는 50% 포함, 50~90% 는 90% 포함. % 와 ÷σ 는 각각의 중앙값. "나은 수" = 기본 GBDT 보다 나은 데이터셋 수)

| 범주 비율 | 개수 | kumo (ours) | ÷σ | 나은 수 | Kumo-Tabular | ÷σ | 나은 수 | TabICLv2 | ÷σ | 나은 수 |
|---|---|---|---|---|---|---|---|---|---|---|
| 0% | 17 | −10.2% | −2.0 | 17/17 | −10.7% | −1.9 | 17/17 | −6.4% | −0.8 | 15/17 |
| 0~50% | 20 | −6.9% | −1.9 | 18/20 | −7.1% | −1.9 | 19/20 | −4.0% | −1.0 | 18/20 |
| 50~90% | 8 | −5.6% | −1.2 | 8/8 | −5.5% | −1.2 | 8/8 | −1.4% | −0.3 | 5/8 |
| 90% 이상 | 6 | −13.2% | −1.9 | 5/6 | −13.3% | −1.9 | 5/6 | −6.4% | −0.8 | 4/6 |

이진 아닌 범주 비율 (0/1 표시 위주인 hiva · NATICUSdroid · MIC 를 내려 보냄)

| 이진 아닌 범주 비율 | 개수 | kumo (ours) | ÷σ | 나은 수 | Kumo-Tabular | ÷σ | 나은 수 | TabICLv2 | ÷σ | 나은 수 |
|---|---|---|---|---|---|---|---|---|---|---|
| 0% | 21 | −10.2% | −2.0 | 21/21 | −10.7% | −2.0 | 21/21 | −4.6% | −0.8 | 18/21 |
| 0~50% | 22 | −6.9% | −1.3 | 20/22 | −7.1% | −1.6 | 21/22 | −3.6% | −0.9 | 18/22 |
| 50~90% | 6 | −9.8% | −1.7 | 6/6 | −9.8% | −1.7 | 6/6 | −3.3% | −0.7 | 5/6 |
| 90% 이상 | 2 (splice · Amazon) | −8.4% | −0.4 | 1/2 | −8.9% | −0.5 | 1/2 | +7.2% | +2.8 | 1/2 |

고카디널리티

| 묶음 | 개수 | kumo (ours) | ÷σ | 나은 수 | Kumo-Tabular | ÷σ | 나은 수 | TabICLv2 | ÷σ | 나은 수 |
|---|---|---|---|---|---|---|---|---|---|---|
| 범주 없음 | 17 | −10.2% | −2.0 | 17/17 | −10.7% | −1.9 | 17/17 | −6.4% | −0.8 | 15/17 |
| 범주 있음, 고유값 > 100 열 없음 | 27 | −7.7% | −1.9 | 26/27 | −7.5% | −1.9 | 26/27 | −3.9% | −1.0 | 24/27 |
| 고유값 > 100 열 있음 | 7 | **−2.4%** | **−0.8** | 5/7 | −3.5% | −1.5 | 6/7 | **+2.4%** | **+1.0** | 3/7 |

- 범주 비율로 묶으면 0% → 50~90% 로 갈수록 중앙값이 0 쪽으로 가다가 (−10.2 → −6.9 → −5.6%) 90% 이상에서 다시 커짐 (−13.2%). 단조가 아님. 90% 이상 6개 중 FM 이 지는 것은 Amazon 하나
- 고카디널리티 열이 있는 7개 (Marketing_Campaign · HR · Amazon · Food_Delivery · kddcup09 · Diabetes130US · SDSS17) 가 가장 차이가 작음. kumo (ours) 가 지는 3개 중 2개 (Amazon · kddcup09) 가 여기. 다만 SDSS17 (−19.7%) · Marketing (−11.4%) 처럼 크게 이기는 것도 있음

### 과제·크기와 섞임

- 과제: 과제마다 따로 봐도 방향은 같음 (범주 비율과 kumo % 의 상관: 이진 30개 +0.31 (p 0.09) · 다중 8개 +0.27 (0.52) · 회귀 13개 +0.36 (0.23)). 다중은 범주가 많아도 차이가 큼 (묶음 중앙값 −14~−39%). 회귀에는 범주 50% 넘는 데이터가 없음 → 50% 이상 묶음 14개는 이진 9 · 다중 5
- 크기: 범주 비율과 행 수는 상관 없음 (+0.01). 하지만 고유값 > 100 열 수와 행 수는 +0.38 (p 0.006): 고카디널리티 7개 중 5개가 2만 행 넘음. 2만 행 넘는 14개 안에서는 범주 비율과 kumo % 가 +0.41 (0.14), 2만 행 이하 37개 안에서는 +0.12 (0.49) → "크고 범주가 있는 데이터" 에서 차이가 작은 쪽으로 보이지만 개수가 적어 크기 탓인지 고카디널리티 탓인지 가르지 못함

### 오염 묶음과 겹침

| 오염 묶음 (6) | 개수 | 범주 비율 중앙값 | 고유값 > 100 열 있음 | kumo (ours) 중앙값 | ÷σ |
|---|---|---|---|---|---|
| Kaggle 대회 | 3 | 0% | 1 (Amazon) | −2.2% | −1.3 |
| Kaggle 데이터셋 | 12 | 40% | 4 | −3.9% | −0.8 |
| 교과서급 | 11 | 20% | 0 | −9.7% | −1.6 |
| 덜 알려진 | 8 | 3% | 0 | −13.3% | −2.0 |
| 미분류 | 17 | 13% | 2 (kddcup09 · Diabetes130US) | −8.8% | −2.2 |

- 51개에서도 범주·고카디널리티와 출처가 겹침: Kaggle 출처 (대회 포함 15개) 가 범주 비율이 높고 고카디널리티 7개 중 5개를 차지함. 덜 알려진 8개는 범주가 적고 (중앙값 3%) 고카디널리티가 없음
- Kaggle 출처에서 FM − GBDT 가 작은 것은 LLM 기억과는 무관한 비교 (FM·GBDT 둘 다 LLM 아님) → 데이터 성격 (범주·크기 등) 의 차이로 보임 (나눠 재지 않음). 그래도 Agent 쪽 질문 ("범주 덕인가, 알려진 데이터라서인가") 을 이 51개에서 고르는 데이터셋으로 가르기 어렵다는 점은 12개와 같음
- 덜 알려진 데이터 중 범주가 많은 것은 MIC (85%, 0/1 위주) · NATICUSdroid (100%, 모두 0/1) 뿐이고, 둘 다 이진 아닌 범주 비율로는 낮음. 이진 아닌 범주가 많고 오염 판단이 없는 것 (미분류): website_phishing (78%) · in_vehicle (63%) · Diabetes130US (57%, 고카디널리티 3열)

### 읽은 것 (51개, 판단은 약함)

- **범주 비율만으로는 차이가 줄지 않음**: 51개 상관 +0.20 (p 0.16) 은 약하고, 범주가 있는 34개 안에서는 0 (+0.01). 묶음 중앙값도 단조가 아님 (90% 이상이 −13.2%). FM 은 범주 비율 묶음 모두에서 대부분 이김 (kumo (ours) 48/51)
- **고카디널리티 쪽이 더 가까운 단서**: 고유값 > 100 열이 있는 7개는 중앙값 −2.4% (−0.8σ), 상관 +0.30 (p 0.03, TabICLv2 도 +0.31). 다만 7개뿐이고, 크기 (5개가 2만 행 넘음) · 출처 (5개가 Kaggle) 와 겹쳐 가르지 못함. p 는 여러 지표를 본 것을 보정하지 않은 값. TabArena Kumo-Tabular 로는 약함 (+0.20, p 0.16, 중앙값 −3.5%): kddcup09 · Marketing 에서 TabArena 쪽이 더 이김 → 고카디널리티 신호의 일부는 전처리 (드문 값 처리 등) 에 달려 있을 수 있음 (나눠 재지 않음)
- Agent 가 이긴 셋과 견주면: Amazon 은 이 단서와 맞음 (범주 100% · 고카디널리티 8열 · FM 이 GBDT 에 짐). splice (−21.3%, 고유값 최대 6) 와 in_vehicle (−9.9%, −3.9σ, 최대 25) 은 FM 이 GBDT 보다 꽤 나은 데이터 → 이 둘에서 Agent 가 이긴 것은 "범주라서 FM 이 GBDT 만큼 약해짐" 으로는 설명되지 않음 (Agent 가 FM 보다 나은 까닭은 이 절로는 확인 못 함)
- 데이터셋 51개 (범주 있는 34, 고카디널리티 7) 로는 묶음마다 개수가 적음. 범주 열 기준은 OpenML 표시를 따름 (숫자 id 를 범주로 둔 것 · 0/1 을 범주로 둔 것이 섞임)

## 범주 대 오염: 알아볼 수 없게 바꾼 3개 (2026-11-17)

"Agent 확장 1회째" 의 요약에서 Agent 가 FM 을 이긴 셋 (splice · Amazon_employee_access · in_vehicle_coupon_recommendation) 은 범주 열이 대부분이고 오염 위험도 큼 → 둘이 겹쳐 가르지 못했음.
새 데이터셋을 찾지 않고 **같은 3개를 알아볼 수 없게 바꾼 대회** 로 FM·GBDT·Agent 5설정을 r0f0 에서 다시 돌림. 바꾼 데이터에서도 Agent 가 이기면 범주 열 덕, 지면 알아본 덕으로 읽기로 함 (설계는 사용자가 정함).
51개로 범주 비율을 본 것은 바로 위 "범주 비율과 FM − GBDT".

### 바꾼 방법 (`fab/bench/tabarena/obfuscate.py`)

| 항목 | 한 것 |
|---|---|
| 대회 | `competitions/ta-<데이터셋>-obf/` (r0f0). 익명 id `tabarena-p13` (splice) · `p14` (Amazon) · `p15` (in_vehicle). 설정의 `tabarena.obfuscate` (`seed` 20261117, 설명에 있으면 멈출 말 `hide`) |
| 행 | 원래와 같은 OpenML split · 같은 행. 행 id 는 원래처럼 데이터셋·split 의 고정 seed 순열이지만 seed 에 `obfuscate.seed` 를 더함 → 원래 대회와 다른 id (r0f0 에서 같은 id 인 행은 우연뿐) |
| 열 이름 | 피처 `f0, f1, …` (원래 순서 그대로, splice 의 이웃 위치 관계는 데이터로 찾을 수 있게 둠), 타깃 `target` |
| 범주 값 | 열마다 따로 무작위 토큰 (영소문자로 시작하는 5글자 영소문자·숫자). 대응은 (task id, seed, 열 위치) 의 rng 로 원래 값 → 토큰을 무작위로 배정 (원래 값의 순서와 무관). 데이터 전체 값으로 만들어 train·test·모든 split 에서 같은 대응. 결측은 결측 (in_vehicle 의 글 `nan` 도 결측, 원래 public 을 pandas 로 읽은 것과 같은 자리). splice 의 A/C/G/T·드문 기호, Amazon 의 id 9열, in_vehicle 의 0/1 표시 열도 범주라 토큰 |
| 수치 열 | 이름만 바꿈 (in_vehicle 의 `f3` = temperature, `f4` = time) |
| 클래스 | `c0, c1, …` = 원래 클래스 이름의 정렬 순서 (splice EI/IE/N → c0/c1/c2, Amazon·in_vehicle No/Yes → c0/c1) → 이진의 양성 클래스 (c1 = 원래 Yes) · 다중의 열 순서가 원래와 같음 |
| 대응표 | `FAB_TABARENA/data/obfuscate/<대회>.json` 에만 (정답과 같은 자리, 에이전트에게 안 보임). seed 와 코드로 다시 만들 수 있음 (저장소를 알면) |
| 설명 | 과제 종류, "열 이름·범주 값이 코드로 바뀌어 무엇을 뜻하는지 정보 없음", 범주 열 (splice 60·Amazon 9 모두, in_vehicle 은 `f3`·`f4` 빼고 22), 토큰은 열마다 따로라 다른 열의 같은 토큰은 무관, 지표·제출 형식 (원래 설명의 문단 틀). 도메인·원래 이름·클래스 비율은 안 적음. 그 밖의 데이터 성질은 in_vehicle 의 "범주 결측은 빈칸" 하나만 |
| 결과 행 | 결과표의 `dataset` 키는 `<데이터셋>~obf` (예: `splice~obf`), 대회 이름 `ta-…-obf`, note `obf r0f0`. `latest_grades` 가 기본으로 `~obf` 행을 빼서 `compare` · `elo` (FM 풀) · `cats` 에 안 들어감. Agent 풀은 `committed()` 가 바꾼 대회를 빼서 (대회 → 데이터셋 대응에 없음) 안 들어감 |
| 비교 | `uv run --frozen fab ta obf` → `results/tabarena_obf.csv` (원래 대 바꾼 것, FM·GBDT·Agent 설정별, σ = 원래 kumo 의 모든 split 표준편차, ddof 1) |
| 제출 전 검사 | `fab ta agent run` 이 바꾼 대회면 `obf_leaks` 도 봄: 설명·파일 열 머리에 원래 열 이름 (3글자 이상)·타깃·클래스 이름·`hide` 말, 범주 열 값이 모두 대응표의 토큰인지 |

로컬 확인 (학습 없이, 저장소 밖 스크립트, 1,278개 통과)

- 원래 public 과 1:1: OpenML 행 위치로 두 id 를 맞춰 train·test 의 행 집합, 범주 값 (원래 값 → 대응 = 바꾼 값), 결측 자리, 수치 값, train 타깃·정답이 모두 같음. 열 머리 `id, f0…, target`, 원래 열 순서
- 열마다 대응이 단사 (토큰 수 = 원래 값 수), public 에서 본 짝도 양쪽 모두 1:1. 토큰이 그 데이터셋의 어느 원래 값과도 같지 않음. 피처 (`features.py`) 의 범주 열·열마다 범주 수가 원래와 같음
- 이름 누출: 설명·열 머리·모든 셀에 원래 열 이름·클래스 이름·원래 범주 값 없음, 설명에 도메인 말 (position · nucleotide · DNA · exon · coupon · employee · resource 등) 없음, `leak_check` · `obf_leaks` 통과
- 채점: 원래 대회의 기존 Agent 제출 10개 (in_vehicle 5 · Amazon 3 · splice 1 + 무작위) 를 id 와 클래스 열 이름만 바꿔 (행도 섞어) 채점 → 점수가 원래와 같음 (차 < 1e-12). 이진 뒤집기 = 1 − AUC, 정답 제출 AUC 1, 원래 클래스 열 이름으로 낸 제출은 무효
- 풀 격리: 가짜 `~obf` 행 (오차 1e-6, 가장 새 시각) 을 결과표에 넣어도 `compare` · `elo` 두 풀의 값이 넣기 전과 같음

### FM·GBDT 원래 대 바꾼 것 (r0f0)

`fab ta fm ta-splice-obf ta-in-vehicle-coupon-recommendation-obf --models tabpfn_v2_refit tabicl_kv_refit kumo_cat_refit lgbm_refit catboost_refit --splits r0f0 --one-job --accelerator GPU_1xA10` 과
`fab ta fm ta-amazon-employee-access-obf --models tabicl_kv_refit kumo_cat_refit lgbm_refit catboost_refit --splits r0f0 --one-job --accelerator GPU_1xH100` (train 2.2만 행, tabpfn 은 범위 밖). `metric_error`, 차 ÷σ = (바꾼 것 − 원래) ÷ 원래 kumo 의 split 표준편차.

| 모델 | splice 원래 | 바꾼 것 | ÷σ | Amazon 원래 | 바꾼 것 | ÷σ | in_vehicle 원래 | 바꾼 것 | ÷σ |
|---|---|---|---|---|---|---|---|---|---|
| kumo (`kumo_cat_refit`) | 0.06693 | 0.06924 | +0.15 | 0.12772 | 0.12752 | −0.03 | 0.13707 | 0.13650 | −0.14 |
| tabicl (`tabicl_kv_refit`) | 0.07219 | 0.07337 | +0.08 | 0.15110 | 0.15320 | +0.33 | 0.14891 | 0.14865 | −0.06 |
| tabpfn_v2 | 0.08207 | 0.08335 | +0.09 | – | – | | 0.18466 | 0.18068 | −0.99 |
| lgbm | 0.09400 | 0.09275 | −0.08 | 0.15757 | 0.15753 | −0.01 | 0.16221 | 0.16220 | −0.00 |
| catboost | 0.09583 | 0.09685 | +0.07 | 0.11920 | 0.12198 | +0.43 | 0.15899 | 0.15824 | −0.19 |
| σ (원래 kumo) | 0.0150 | | | 0.0064 | | | 0.0040 | | |

- 14개 모두 차가 1σ 안 (가장 큰 것: in_vehicle tabpfn_v2 −0.99σ (−2.2%), Amazon catboost +0.4σ (+2.3%), splice kumo +0.15σ (+3.4%)). 이름·값 표기에 거의 영향을 안 받음 → Agent 비교의 FM 기준이 바꾼 뒤에도 같음
- 조금 다른 까닭은 나눠 재지 않음 (확인 못 함). 범주 값의 글자·정렬 순서가 바뀌면 ordinal 부호 (CatBoost·LightGBM 의 범주 번호, TabPFN·TabICL 의 전처리) 가 달라짐. Kumo 가 글자를 쓰는지도 확인 못 함
- FM 이 GBDT 보다 나은지 (FM − GBDT) 는 세 데이터셋 모두 그대로 (splice·in_vehicle 은 FM, Amazon 은 CatBoost 가 kumo 보다 나음)
- 모든 split 으로 넓히면 (splice 9 · Amazon 9 · in_vehicle 9 = 27 split × 5모델): 모델 시간은 r0f0 의 약 9배 (GPU 수 분, CatBoost splice 약 20분) 라 비용은 작음. r0f0 차가 모두 1σ 안이라 이번에는 안 함 (제안만)

### Agent 원래 대 바꾼 것 (r0f0, 1회)

같은 설정 (5설정 · 공식 지시문 · 1시간 · A10). 같은 엔드포인트는 한 번에 하나 (Sonnet 줄: AIDE → Claude Code → +cont, GPT 줄: AIDE → Codex, 데이터셋 splice → Amazon → in_vehicle).
15회 모두 정상 종료·유효 제출 (`run_crashed` 없음). 원래 값은 #26 1회째 (splice · Amazon), 파일럿 1회째 / 2회째 (in_vehicle, ÷σ 는 두 회차 평균과).
"− kumo" = 같은 대회의 kumo 와의 차 ÷σ (음수 = Agent 가 FM 보다 나음). **굵게** = 그 대회의 최고. "알아봄" = 하네스 기록·작업 디렉터리에서 데이터셋 이름을 댄 실행 (아래 "알아본 흔적").

splice (`tabarena-p13`), σ 0.0150, kumo 0.06693 → 0.06924

| 설정 | 원래 | 바꾼 것 | 차 ÷σ | 원래 − kumo | 바꾼 것 − kumo | 알아봄 (바꾼 것) |
|---|---|---|---|---|---|---|
| Codex × GPT | **0.06179** | **0.06012** | −0.11 | −0.34 | **−0.61** | 예 ("the UCI splice dataset", 429 단계 중 9번째, 시작 몇 분 안) |
| AIDE × Sonnet | 0.07781 | 0.07801 | +0.01 | +0.73 | +0.58 | 예 (첫 답: "This looks like the Splice dataset") |
| AIDE × GPT | 0.06987 | 0.08136 | +0.77 | +0.20 | +0.81 | 아니오 |
| Claude Code × Sonnet +cont | 0.08445 | 0.09917 | +0.98 | +1.17 | +2.00 | 예 ("looks like the splice dataset") |
| Claude Code × Sonnet | 0.1388 | 0.1148 | −1.60 | +4.79 | +3.04 | 아니오 |

Amazon (`tabarena-p14`), σ 0.0064, kumo 0.12772 → 0.12752, CatBoost 0.11920 → 0.12198

| 설정 | 원래 | 바꾼 것 | 차 ÷σ | 원래 − kumo | 바꾼 것 − kumo | 알아봄 (바꾼 것) |
|---|---|---|---|---|---|---|
| AIDE × Sonnet | **0.1068** | **0.1093** | +0.38 | −3.27 | **−2.86** | 예 (첫 답: "This looks like the Amazon Employee Access data") |
| Codex × GPT | 0.1102 | 0.1094 | −0.12 | −2.75 | −2.83 | 예 (늦게: 461 단계 중 385번째 "Amazon employee access") |
| Claude Code × Sonnet +cont | 0.1093 | 0.1193 | +1.56 | −2.88 | −1.29 | 아니오 |
| Claude Code × Sonnet | 0.1201 | 0.1202 | +0.01 | −1.19 | −1.14 | 아니오 |
| AIDE × GPT | 0.1106 | 0.1209 | +1.61 | −2.67 | −1.03 | 아니오 |

in_vehicle (`tabarena-p15`), σ 0.0040, kumo 0.13707 → 0.13650

| 설정 | 원래 (1회째 / 2회째) | 바꾼 것 | 차 ÷σ | 원래 − kumo | 바꾼 것 − kumo | 알아봄 (바꾼 것) |
|---|---|---|---|---|---|---|
| Codex × GPT | **0.13150** / 0.13295 | 0.13709 | +1.20 | −1.20 | +0.14 | 아니오 |
| AIDE × Sonnet | 0.13813 / 0.13904 | 0.14986 | +2.79 | +0.37 | +3.30 | 아니오 |
| Claude Code × Sonnet +cont | 0.14911 / 0.14919 | 0.15128 | +0.53 | +2.99 | +3.65 | 아니오 |
| AIDE × GPT | 0.13882 / 0.14057 | 0.15313 | +3.32 | +0.65 | +4.11 | 아니오 |
| Claude Code × Sonnet | 0.16192 / 0.16192 | 0.16627 | +1.08 | +6.14 | +7.36 | 아니오 |

바꾼 대회의 최고는 in_vehicle 이 **FM: kumo 0.13650** (Codex 0.13709 가 +0.14σ 로 다음).

요약

| 데이터셋 | FM 보다 나은 Agent (원래 → 바꾼 것) | 최고 Agent − kumo ÷σ (원래 → 바꾼 것) | 승자 (원래 → 바꾼 것) | 알아본 실행 (바꾼 것) |
|---|---|---|---|---|
| splice | 1 / 5 → 1 / 5 (Codex) | −0.34 → −0.61 | Agent → Agent (둘 다 1σ 안) | 3 / 5 |
| Amazon | 5 / 5 → 5 / 5 | −3.27 → −2.86 | Agent → Agent | 2 / 5 |
| in_vehicle | 1 / 5 → 0 / 5 | −1.20 (1회째 −1.38) → +0.14 | Agent → **FM** | 0 / 5 |

- 바꾼 뒤 1σ 넘게 나빠진 것 15회 중 6회: in_vehicle 4회 (AIDE × GPT +3.3σ · AIDE × Sonnet +2.8σ · Codex +1.2σ · Claude Code +1.1σ), Amazon 2회 (AIDE × GPT · +cont +1.6σ). 거의 그대로인 것: splice·Amazon 의 Codex 와 AIDE × Sonnet (−0.1~+0.4σ). 나아진 것은 splice × Claude Code 공식 (−1.6σ, 원래도 Agent 꼴찌)
- 원래 쪽 회차 흔들림 (in_vehicle 1회째 ↔ 2회째) 은 0.43σ 안 (AIDE × GPT 가 가장 큼) → in_vehicle 의 +2.8~+3.3σ (AIDE 둘) 는 그보다 큼. 바꾼 쪽은 1회뿐이라 splice·Amazon 의 ±1σ 안 차는 실행 흔들림과 가르지 못함

### 알아본 흔적

세는 것 (저장소 밖 스크립트): 하네스 기록 (`codex.jsonl`·`aide.jsonl`·`claude-code.jsonl`·`harness_stderr.txt`) 과 작업 디렉터리의 글 파일 (`.py`·`.md`·`.txt`·`.log`·`.json`·`.ipynb`·`.sh` 등, pip 가 깐 파일 (`*.dist-info/RECORD`) 과 `site-packages` 는 뺌) 에서
splice · splicing · DNA · RNA · nucleotide · exon · intron · codon · donor/acceptor · gene/genome · `ACGT` · Amazon · employee · "access request" · `MGR_ID`·`ROLE_*` · coupon · vehicle/driver · UCI · Kaggle · OpenML · "famous/well-known/classic dataset" 의 수. 걸린 것은 앞뒤 글을 모두 읽음.
`Kaggle`·`driver` 는 모든 실행에서 지시문 ("MLE-bench, an offline version of Kaggle competitions" · "appropriate drivers installed") · AIDE 의 시스템 글 ("You are a Kaggle grandmaster") · `nvidia-smi` 출력에서만 나옴 → 아래 표에서 뺌.

| 실행 (바꾼 것) | 하네스 기록 | 작업 디렉터리 | 무엇을 알아봤나 |
|---|---|---|---|
| splice × Codex | splice 6 · DNA 6 · nucleotide 5 · exon 2 · intron 1 · donor/acceptor 4 · gene 7 · UCI 3 | 0 | 생각 글에서 "UCI splice dataset", "splice junctions involving a 60 base pair DNA sequence … 3190 instances", "UCI Splice … known bases c0 GT, c1 AG". 9번째 단계 (시작 몇 분 안) 부터 |
| splice × AIDE × Sonnet | – | splice 86 · DNA 64 · nucleotide 4 | 첫 답 (설계 글) "This looks like the Splice dataset (60 categorical DNA positions, 3 classes, ~3190 rows)" 가 AIDE 기록 (`logs/`) 의 매 단계 메모에 되풀이 |
| splice × Claude Code +cont | splice 1 | 0 | "Small data (looks like the splice dataset)" |
| Amazon × AIDE × Sonnet | – | Amazon 16 · employee 16 | 첫 답 "This looks like the Amazon Employee Access data (9 categorical features, with f5 and f8 having the same cardinality)" |
| Amazon × Codex | Amazon 1 · employee 2 · UCI 1 | 0 | 처음엔 "likely anonymized UCI dataset" (3번째), 끝 무렵 "the dataset related to Amazon employee access" (385번째) |
| 나머지 10회 | 0 | 0 | splice × AIDE × GPT · Claude Code, Amazon × AIDE × GPT · Claude Code 둘, in_vehicle 5회 모두 |

- 모두 **데이터 모양** 으로 알아봄 (이름·출처·URL 은 안 줌, 원본 접근 0): splice 는 열 60개 × 값 4~8개 · 3,190행 · 3클래스, Amazon 은 범주 9열 · 두 열의 카디널리티가 같음 (원래 `ROLE_TITLE` · `ROLE_CODE`, 343개씩 1:1)
- in_vehicle 은 5회 모두 알아본 흔적 없음 (Codex 는 처음에 "2018 Indonesia law" 같은 엉뚱한 짐작을 한 번 함)
- 견줌 (원래 대회, 로컬에 받아 둔 9회만): 원래 splice × Codex 도 splice·DNA·codon 450번 등, 원래 Amazon × Codex 는 "the original Amazon dataset", in_vehicle 은 설명에 쿠폰이 있어 쿠폰 말이 많음 (원래 대회는 과제 설명이 도메인을 주므로 이 수는 뜻이 다름)

### 풀이 읽기

splice × Codex (원래 풀이: 코돈 · k-mer · 경계 옆 AG/GT · 거의 같은 서열 찾기)

- **열마다 다른 토큰을 한 알파벳으로 다시 맞춤** (`align.py`): 열마다 빈도 상위 4개 토큰 → 0~3, 이웃 열 쌍 (59쌍) 의 4×4 빈도를 한 공통 전이 행렬로 설명하는 위치별 4기호 순열을 찾음 (24 × 24 동적 계획 + 재시작 100번. `align.py` 의 시험은 c2 (원래 N) 클래스 train 행으로). 바꾼 데이터의 "열마다 따로" 를 데이터 구조로 되돌림 (원래 A/C/G/T 이름은 모름: 공통 알파벳의 이름 없는 4기호)
- 그 위에서 k-mer 스펙트럼 · 위치별 n-gram · 클래스별 Markov 혼합 · GMM + 로지스틱 · GBDT · 선형 SVM 섞기, 마지막에 **train 의 거의 같은 서열** (Hamming ≤ 22/60) 이 한 클래스뿐이면 그 클래스를 0.95 로 덮음 (`compose_submission.py`)
- 남은 것: k-mer · 거의 같은 서열 찾기. 사라진 것: 코돈 (코드 0) · 글자 그대로의 AG/GT (코드 0, 생각 글에서 "c0 GT, c1 AG" 라고 한 번 말함). torch 신경망도 안 씀 (원래는 씀)
- 원래 0.06179 → 0.06012 (−0.11σ). 알아본 뒤에도 쓸 수 있는 기억은 "DNA 서열이다 (이웃 위치가 Markov 처럼 이어짐)" 정도이고, 토큰 → 염기 대응은 기억으로 알 수 없음 (무작위)

in_vehicle × Codex (원래: 인구통계 열로 응답자 id 를 다시 만듦, train+test 를 합쳐)

- **응답자 id 풀이가 남음**: `person = f7..f19` (= 원래 gender·age·maritalStatus·has_children·education·occupation·income·car·Bar·CoffeeHouse·CarryAway·RestaurantLessThan20·Restaurant20To50, 응답자 열 13개 그대로) 와 `item = f0..f6 + f20..f22` (상황·쿠폰 열) 로 나눔. 생각 글에 "user ID" · "individual"
- 최종 섞기 (`build_submission.py`) 에 들어간 것: `person` 키 (13열을 이어 붙인 글) 를 범주 피처로 쓴 LightGBM·CatBoost (`lgb_grp63` · `cat_group6` · `cat_groupall8`) + 다른 GBDT · 열마다 무게를 준 일치 커널 SVM (`cv_svc`) · 범주 값 PCA 중심 피처 · 두 탑 신경망 섞기.
  `person` 이 같은 행끼리의 커널 SVM (`cv3_svc_profile.py`) 도 만들었지만 최종 섞기에는 없음. 키·PCA 는 **train+test 를 합친 표** (`pd.concat([tr, te])`) 에서 만듦 → 원래와 같은 "inductive 밖" 꼴 (5). 응답자별 타깃 집계 피처 (원래 2회째) 는 최종 섞기에서 못 찾음
- 그래도 0.13150 / 0.13295 → 0.13709 (kumo 0.13650 보다 +0.14σ 나쁨). → **응답자 id 풀이만으로는 FM 을 못 넘음**. 원래 "Codex > FM" 은 응답자 id 풀이 + 값의 뜻 (쿠폰 종류 × 상황 교차, 순서 있는 구간) 을 같이 쓴 몫으로 읽음 (둘을 나눠 재지 않음)

Amazon (알아본 둘 · 못 알아본 셋)

- 고카디널리티 범주의 흔한 풀이 그대로: 열 두 개씩 묶은 범주 · 빈도 피처 · 원-핫 로지스틱 · CatBoost (AIDE 둘·+cont·Codex, Claude Code 공식은 CatBoost 하나). 원래 대회의 AIDE × Sonnet · Codex · +cont 풀이와 같은 꼴 (짝 묶음·빈도 코드가 양쪽 모두에 있음)
- 못 알아본 셋 (AIDE × GPT · Claude Code 둘) 도 모두 FM 보다 나음 (−1.0~−1.3σ). 다섯 모두 우리 CatBoost (바꾼 것 0.1220) 보다도 나음

### 해석 (r0f0 1회, 판단은 약함)

| 데이터셋 | 읽기 |
|---|---|
| Amazon | **범주 열 덕** 으로 읽음. 바꾼 뒤에도 5 / 5 가 FM 보다 나음 (최고 −2.9σ), 알아보지 못한 3회도 FM 보다 나음. 고카디널리티 범주의 흔한 풀이 (짝 묶음·빈도·원-핫) 가 Kumo 보다 나은 데이터 (위 "범주 비율과 FM − GBDT" 의 고카디널리티 단서와 맞음: 이 데이터는 CatBoost 도 Kumo 보다 나음). 알아본 둘 (AIDE × Sonnet · Codex) 이 최고 둘이고 못 알아본 것 중 최고 (+cont 0.1193) 보다 약 1.6σ 나음 → 알아본 몫이 조금 있을 수 있음 (1회라 가르지 못함) |
| splice | **가르지 못함**. 바꾼 뒤에도 Codex 만 FM 보다 나음 (−0.61σ, 원래 −0.34σ, 둘 다 1σ 안) 이지만, 바꾼 데이터를 Codex · AIDE × Sonnet · +cont 가 모양으로 splice 라고 알아봄 → "알아볼 수 없게" 가 안 됨. 다만 알아봐도 토큰 → 염기 대응은 알 수 없고, Codex 의 풀이는 데이터 구조 (이웃 열 쌍으로 알파벳 맞추기 · k-mer · 거의 같은 서열) 위주로 바뀜. 알아보지 못한 AIDE × GPT 는 원래 +0.20σ → +0.81σ 로 FM 과 멀어짐 |
| in_vehicle | **알아본 (뜻을 안) 덕** 으로 읽음. 바꾼 뒤 0 / 5, 최고 Codex 가 kumo 보다 +0.14σ, AIDE 둘은 +2.8~+3.3σ 나빠짐 (회차 흔들림 0.4σ 의 7배 넘음). 알아본 흔적은 없음. 응답자 id 풀이 (데이터 구조) 는 남았지만 그것만으로는 FM 을 못 넘음. 바꾸기는 이름만이 아니라 **값의 뜻** (나이·소득·방문 빈도의 순서, 쿠폰 종류, 0/1 표시) 도 지움 → "알려진 데이터라서" 와 "값에 순서·뜻이 있어서" 를 이 실험으로는 나누지 못함 |

- 셋을 합치면: "범주 열이 많으면 Agent 가 FM 을 이긴다" 는 Amazon (고카디널리티 id) 에서만 바꾼 뒤에도 남음. splice 는 1σ 안이고 오염 통제가 실패함 (모양으로 알아봄), in_vehicle 은 뜻을 지우면 FM 이 이김
- 모양으로 알아보는 것은 이 방법의 한계: 유명하고 모양이 독특한 데이터 (splice 의 60 × ACGT, Amazon 의 9열·같은 카디널리티 쌍) 는 이름·값을 지워도 LLM 이 알아봄. 더 지우려면 행 수·열 수를 바꾸거나 (부분 추출·가짜 열) 열 순서를 섞어야 하는데, 그러면 같은 데이터의 비교가 아니게 됨 (안 함)
- 1회뿐 (결정 2 의 2회째 없음). splice·Amazon 의 ±1σ 안 차는 실행 흔들림과 가르지 못함

### 사후 검사 (15회)

- `data_source_urls` 0 · `kaggle_urls` 0 (15회 모두). 원본 데이터 접근 흔적 없음 (알아본 5회도 원본을 받으려 하지 않음)
- `tfm_runtime` · `tfm_in_code` · `pip_target` · `pip_added` 모두 빈칸 (공식 설정에서 TFM 을 스스로 깐 실행 없음)
- 429 재시도: Codex splice 16번 · Amazon 13번 · in_vehicle 0번, 나머지 0. 같은 엔드포인트를 한 번에 하나만 써서 #26 (Codex 0~97번) 보다 적음
- 걸린 시간: AIDE 60분 (모두 시간 종료), Codex 56~57분 (스스로 끝냄), Claude Code 공식 29~60초, +cont 5~11분 (이어 가기 6·8·12번). 입력 토큰: Codex 3,316만~3,502만, 나머지 8.5만~78만
- "inductive" (5): in_vehicle × Codex 는 train+test 를 합친 표에서 응답자 키·PCA 를 만듦 (위). splice × Codex 의 토큰 순위 부호는 train+test 를 합친 빈도로 만듦 (`align.py`, 타깃은 안 씀). 다른 실행은 보지 않음 (확인 못 함)

### 비용

| 항목 | 시간 |
|---|---|
| FM (GPU) | A10 작업 하나 4.6분 (splice·in_vehicle, 모델 시간 2.1분) + H100 작업 하나 2.2분 (Amazon, 1.0분) |
| GBDT (serverless CPU) | 작업 두 개 4.0분 · 1.9분 (모델 시간 합 3.6분, splice CatBoost 138초) |
| Agent (A10) | 15회 작업 시간 (설치·회수 포함) 합 **10.1 A10 시간** (하네스 시간 합 9.3시간). 어림 (15) 안 |
| 걸린 벽시계 | 12:12 → 20:16 (8시간). GPT 줄 6회가 길고, 로컬 대기 스크립트가 16:24~18:08 에 멈춰 (아래) 약 1.7시간 늦어짐 |

### 실행 메모

| 일 | 내용 | 처리 |
|---|---|---|
| 클래스 이름 안 바뀜 | 처음 `prepare` 에서 타깃 클래스가 원래 이름 그대로 (`EI`·`Yes`). pandas 3 은 문자열 열이 `object` 가 아니라서 `dtype == object` 판정이 거짓 | 과제 종류 (`is_classification`) 로 판정하게 고치고 다시 `prepare`. 원격에 올리기 전이라 영향 없음 |
| 로컬 대기 멈춤 | 줄 스크립트 (저장소 밖, Databricks CLI 로 2분마다 상태 조회) 가 16:24~18:08 에 다음 실행을 안 냄. Amazon × Codex 는 16:24 에 끝났고 in_vehicle +cont 도 그 사이 끝남 (작업 시간은 59분 · 13분) | 18:08 에 이어서 냄. 점수·조건에는 영향 없음 (같은 엔드포인트를 둘 쓰지는 않음). 맥 잠자기 또는 CLI 조회가 멈춘 것으로 봄 (확인 못 함) |
| 도구의 대기 상한 | 로컬 줄을 처음에 도구의 백그라운드 명령으로 띄웠는데 4시간 상한이 있어 첫 제출 직후 `nohup` 으로 다시 띄움 (첫 실행은 run_id 를 기다리는 것으로 이어 받음) | 빠진·두 번 낸 실행 없음 |

## C2 (+fm): FM 을 받은 Agent (12개 r0f0, 2026-11-18 · 11-19)

Agent 를 돌린 12개 모두의 r0f0 에서, 환경에 TFM 을 깐 Agent (`codex-gpt+fm`·`aide-sonnet+fm`) 를 1회씩 돌림 (24회: 배치 0·1 의 7개 14회는 2026-11-18, 배치 2 의 5개 10회는 2026-11-19).
규칙은 roadmap 의 "C2 규칙" 그대로: tabpfn_v2·tabicl·kumo 를 깔고 가중치를 받아 둔 뒤 (하네스 1시간 밖), 지시문에 `fab/agent/fm_notes.txt` (이름·사용 예, "optional").
`uv run --frozen fab ta agent run ta-<데이터셋> <설정> --fm` → 설정 이름 `<설정>+fm`, 결과는 `results/tabarena_runs.csv`.

- **배치 2 를 돌림 (2026-11-19, 사용자가 정함)**: 11-18 에 멈췄던 QSAR-TID-11 · in_vehicle · Amazon · physiochemical · APSFailure (10회) 를 같은 조건 (설정 2개 · 1회 · r0f0 · `fm_notes.txt` 그대로) 으로. Agent 가 FM 을 이긴 3개 (splice · Amazon · in_vehicle) 모두 C2 가 생김
- 설정은 2개만 (2026-11-17 사용자가 정함): MLE-bench C2 에서 효과가 갈린 Codex × GPT · AIDE × Sonnet. 두 설정은 엔드포인트가 달라 동시에, 같은 설정은 데이터셋을 차례로
- **조건 차이**: `fm_notes.txt` 는 MLE-bench C2 와 같게 그대로 둠 (2026-11-17 사용자가 정함). Kumo 예는 "수치만, 범주는 정수로" → 범주 열을 범주로 넘기는 우리 FM (`kumo_cat_refit`, 결정 11) 보다 약한 사용 예일 수 있음. 또 Agent 안의 FM 은 Agent 가 짠 CV·refit 으로 돎 (우리 FM 은 refit 한 번)
- Elo 의 Agent 풀과 obf 표에는 +fm 을 넣지 않음 (`without_fm`, 넣으면 풀이 바뀌어 기존 Elo 가 모두 달라짐. 배치 2 뒤에도 `fab ta elo` 값 같음)

### 결과 (r0f0, 1회)

이 표는 12개 (2026-11-19). kddcup09 의 C2 2회 (2026-11-20) 는 [행 추가: kddcup09_appetency](#행-추가-kddcup09_appetency-r0f0-2026-11-19--11-20) 의 "C2 (+fm)".

`metric_error` (낮을수록 좋음). Agent = 같은 설정의 공식 설정 (hazelnut · in_vehicle 은 2회 평균, 나머지 1회). σ = kumo (ours) 의 split 표준편차 (요약 표와 같음). 굵게 = kumo 보다 나음

| 데이터셋 | kumo (ours) | σ | 설정 | Agent | C2 (+fm) | C2 − Agent (÷σ) | C2 − kumo (÷σ) | 제출에 든 FM (최종 코드) |
|---|---|---|---|---|---|---|---|---|
| hazelnut (`p01`) | 0.00390 | 0.00084 | AIDE × Sonnet | 0.01757 | **0.00384** | −16.3 | −0.07 | Kumo 100% (test 의사 레이블 ¹) |
| | | | Codex × GPT | 0.00988 | 0.00398 | −7.0 | +0.10 | Kumo 100% (test 의사 레이블 ¹) |
| in_vehicle (`p02`) | 0.1371 | 0.0040 | AIDE × Sonnet | 0.1386 | 0.1383 | −0.07 | +0.30 | TabICL 70% (순위 섞기, 나머지 CatBoost). TabPFN 은 CV 만, Kumo 안 씀 ³ |
| | | | Codex × GPT | 0.1322 | **0.1321** | −0.03 | −1.23 | 직접 약 5% (Kumo 2.4 · TabPFN 1.9 · TabICL 1.2%) + stacker 입력. 주 성분은 응답자 키 위 CatBoost · LGBM · logistic. Kumo 에 범주는 정수 코드 |
| Amazon (`p03`) | 0.1277 | 0.0064 | AIDE × Sonnet | 0.1068 | **0.1091** | +0.36 | −2.91 | TabICL 50% · TabPFN 30% (범주는 정수 코드 + 개수 + TE), Kumo 안 씀 ³ |
| | | | Codex × GPT | 0.1102 | **0.1025** | −1.20 | −3.94 | TabICL 24.5% · Kumo 16% · TabPFN 7.8%. Kumo 에는 범주 열을 빼고 빈도 · 쌍 개수 · 이웃 통계 80열 ³ |
| physiochemical (`p04`) | 2.852 | 0.037 | AIDE × Sonnet | 3.206 | **2.849** | −9.7 | −0.09 | Kumo 100% |
| | | | Codex × GPT | 2.977 | **2.840** | −3.7 | −0.32 | Kumo 중심 약 93% (Kumo 셋 + Kumo 위 CatBoost 잔차) · TabICL 6% |
| QSAR_fish (`p05`) | 0.8855 | 0.050 | AIDE × Sonnet | 0.9136 | 0.8891 | −0.49 | +0.07 | Kumo 주 성분 |
| | | | Codex × GPT | 0.9003 | 0.8885 | −0.24 | +0.06 | Kumo 주 성분 (정수 열 둘에 `categorical`, 가중 0.025) |
| MIC (`p06`) | 0.4412 | 0.019 | AIDE × Sonnet | 0.4591 | 0.4520 | −0.37 | +0.57 | Kumo · TabPFN |
| | | | Codex × GPT | 0.4554 | **0.4313** | −1.28 | −0.52 | Kumo 주 성분 |
| Is-this (`p07`) | 0.2565 | 0.021 | AIDE × Sonnet | 0.2752 | 0.2793 | +0.19 | +1.09 | TabPFN 27% |
| | | | Codex × GPT | 0.2792 | 0.2798 | +0.03 | +1.12 | 없음 (0%) |
| splice (`p08`) | 0.06693 | 0.015 | AIDE × Sonnet | 0.07781 | 0.07274 | −0.34 | +0.39 | Kumo 주 성분 (원-핫으로 넘김) |
| | | | Codex × GPT | 0.06179 | **0.06588** ² | +0.27 | −0.07 | Kumo 주 성분 + train 근접 행 kNN 보정 |
| students (`p09`) | 0.5251 | 0.014 | AIDE × Sonnet | 0.5515 | 0.5275 | −1.70 | +0.18 | Kumo 100% |
| | | | Codex × GPT | 0.5537 | 0.5256 | −1.99 | +0.04 | Kumo 100% (object 열에 `categorical`, 가중 0.95) |
| polish (`p10`) | 0.00342 | 0.0017 | AIDE × Sonnet | 0.03166 | 0.00374 | −16.5 | +0.19 | Kumo 100% (test 의사 레이블 ¹) |
| | | | Codex × GPT | 0.00436 | **0.00081** | −2.1 | −1.54 | Kumo · GBDT + 비율 열 쌍 피처 |
| QSAR-TID-11 (`p11`) | 0.7867 | 0.047 | AIDE × Sonnet | 0.8145 | 0.8082 | −0.14 | +0.46 | TabPFN 32% (PCA 100, 나머지 Tanimoto 커널 SVR · MLP · LGBM) ³ |
| | | | Codex × GPT | 0.7976 | **0.7646** | −0.70 | −0.47 | Kumo 약 86% (비트 고르기 · PCA 변형 여럿) · TabICL 5% |
| APSFailure (`p12`) | 0.00265 | 0.0015 | AIDE × Sonnet | 0.00478 | 0.00474 ⁴ | −0.03 | +1.42 | TabICL 25% (LGBM · XGB · CatBoost 와 같은 가중 순위 평균), Kumo 안 씀 |
| | | | Codex × GPT | 0.00525 | 0.00396 | −0.87 | +0.90 | TabICL 약 35% (LGBM · XGB · CatBoost · MLP 와 순위 섞기), Kumo 안 씀 ³ |

- ¹ Kumo 예측으로 test 행에 의사 레이블을 붙여 컨텍스트에 넣음 → TabArena 의 "inductive" 규칙 (5) 밖. 셋 다 점수는 FM 수준이거나 그 근처
- ² 하네스가 끝난 뒤 노드가 플랫폼 오류로 죽어 진행 사본으로 채점 (`run_crashed`). 하네스 58분, 제출은 끝까지 씀
- ³ 배치 2 에서 따로 본 것: 제출에 train+test 피처로 fit 한 블록이 있음 (라벨 없음, transductive). in_vehicle × AIDE 응답자 개수, Amazon 둘 빈도 · 개수 (Codex 는 공출현 SVD 도), QSAR-TID × AIDE PCA, APS × Codex 결측 채우기 · 상관 (그 LGBM 성분 4.6%). test 라벨 · 의사 레이블은 배치 2 에 없음. 배치 0·1 은 의사 레이블만 봄
- ⁴ 3번째 실행 (1·2차는 하네스 2~4분째 플랫폼 오류로 `exclude`, 아래 사후 검사)
- "제출에 든 FM" 은 최종 코드로 확인한 것 (가중치 사슬을 끝까지 따라 곱함, stacker 안의 몫은 못 셈). `tfm_at_submit` (근사) 은 Codex 가 FM 예측을 파일로 남겨 다른 프로세스에서 섞어 대부분 빈칸, AIDE × MIC 는 tabicl 이 더 잡힘 (실제는 Kumo · TabPFN)

### 읽은 것 (12개 · 24회, 1회씩이라 판단은 약함)

- **FM 을 받으면 같은 설정의 Agent 보다 나아짐: 24회 중 20회.** 아닌 것은 Is-this 둘 (FM 을 거의 안 넣음) · splice × Codex (공식 설정 0.0618 이 이미 FM 보다 나았음) · Amazon × AIDE (0.1068 → 0.1091). in_vehicle 둘 · APS × AIDE 는 거의 그대로 (−0.07σ · −0.03σ · −0.03σ)
- **점수는 FM 한 번 (kumo) 근처로 모임**: 24회 중 17회가 kumo 와 1σ 안. 밖은 Is-this 둘 · APS × AIDE (나쁨) 과 Agent 가 이미 FM 을 넘던 곳 · 피처를 찾은 곳 (Amazon 둘 · in_vehicle × Codex · polish × Codex, 나음)
- **Agent 가 이긴 in_vehicle · Amazon 에서 FM 을 쥐여 주면**: Amazon × Codex 만 뚜렷이 나아짐 (0.1102 → 0.1025, −1.2σ, AUC 0.8975 로 12개 표의 Amazon 최고). 제출의 약 48% 가 FM (TabICL · Kumo · TabPFN) 이고 나머지는 Agent 의 공출현 SVD · 이웃 통계 위 GBDT. Amazon × AIDE 는 TabICL · TabPFN 80% 로 바꿨다가 조금 나빠짐 (+0.36σ). in_vehicle 은 둘 다 그대로: Codex 는 FM 을 5% 쯤만 넣고 응답자 키 위 GBDT 그대로, AIDE 는 TabICL 70% 인데 점수가 같음
- **Kumo 에 범주를 넘긴 방식 (특히 in_vehicle · Amazon)**: 배치 2 에서 `categorical` stype 을 쓴 실행은 없음. in_vehicle × Codex 는 안내문대로 정수 코드 (Kumo 2.4%), Amazon × Codex 는 범주 열을 아예 빼고 빈도 · 이웃 통계 같은 수치 80열만 (Kumo 16%), AIDE 는 두 곳 모두 Kumo 를 안 쓰고 TabICL · TabPFN 에 정수 코드. 그래서 "안내문이 범주를 정수로 넘기라고 해서 Kumo 가 약해졌나" 는 이 두 곳에서도 가르지 못함 (Kumo 가 주 성분인 실행이 없음). 범주를 범주로 넘기는 우리 kumo (0.1277 · 0.1371) 는 두 곳 모두 Agent 보다 나쁨
- **FM 보다 뚜렷이 나은 곳**: polish × Codex (−1.5σ, 비율 열 관계 피처 + Kumo · GBDT) 와 Agent 가 이미 이기던 Amazon 둘 · in_vehicle × Codex. QSAR-TID × Codex 는 1σ 안이지만 (−0.47σ) 0.7646 으로 이 데이터셋의 모든 방식 중 최고 (C1 0.7718): Agent 가 Kumo 에 넣을 비트를 고른 변형 (ExtraTrees 상위 비트 · PCA) 을 여럿 섞음. C1 (에이전트 피처 → FM) 과 같은 방향
- **FM 보다 1σ 넘게 나쁜 것은 Is-this 둘 · APS × AIDE**: Is-this 는 FM 을 제출에 거의 안 넣음 (Codex 0%, AIDE TabPFN 27%). FM 이 근소하게 이기는 데이터 (FM − GBDT −2.9%) 라 Agent 가 CV 에서 FM 을 고르지 않은 것으로 봄. APS × AIDE 는 TabICL (컨텍스트 음성 9,000 행 표본) 을 GBDT 셋과 같은 가중으로 섞어 공식 설정과 같은 점수 (+1.42σ, 공식 +1.45σ). 1시간에 LLM 호출 15번 (노드마다 TabICL 5-fold 가 길었음)
- **어느 FM 을 썼나**: 배치 0·1 은 거의 Kumo 였고, 배치 2 는 갈림. physiochemical 둘 · QSAR-TID × Codex 는 Kumo 중심, in_vehicle × AIDE · Amazon 둘 · APS × Codex 는 TabICL 이 가장 큼. 배치 2 의 AIDE 는 physiochemical 말고 넷 모두, APS × Codex 도 Kumo 를 안 씀
- **공식 설정에서 TFM 을 스스로 깐 실행과 비교**: Codex × QSAR_fish (공식 0.9003, TFM 깖) → +fm 0.8885, Codex × hazelnut 1회째 (공식 0.0064, TFM 깖) → +fm 0.0040. 깔아 주면 더 나아짐 (가중치·사용 예를 받은 덕으로 봄, 나눠 재지 않음)
- MLE-bench C2 와 같은 그림: FM 을 쥐여 주면 대개 FM 수준으로 올라오고, 넘는 것은 Agent 가 FM 이 못 보는 구조를 찾은 곳 (nomad × AIDE, 여기서는 polish × Codex · Amazon 둘 · in_vehicle × Codex. QSAR-TID × Codex 는 1σ 안이지만 이 데이터셋 최고)

### 사후 검사 · 비용

- 24회 모두 `valid_submission` 참, `data_source_urls` 0 · `kaggle_urls` 0. 429 는 Codex 0~154번 (physiochemical 154), AIDE × Sonnet 0. 기준 (Codex 200번) 안
- 환경 밖에 더 깐 것 (`pip_added`): Is-this × Codex (interpret-core), physiochemical × Codex (transformers · tokenizers · huggingface_hub, 최종 제출 사슬의 스크립트는 안 씀). `llm_errors` 는 QSAR-TID × Codex 1번 외 0
- 배치 2 의 데이터셋 알아봄 (에이전트 글에 원래 이름): in_vehicle × AIDE ("in-vehicle coupon recommendation dataset"), Amazon 둘 ("Amazon Employee Access", "original Amazon challenge"), physiochemical 둘 ("CASP"), APS × Codex ("UCI APS"). 원본 URL 은 0 (내려받지 않음). #32 의 "알아봄" 과 같은 꼴
- 실행: 배치 0·1 은 2026-11-18 07:22 → 17:40 KST (배치 0: QSAR_fish · MIC, 배치 1: Is-this · hazelnut · splice · students · polish). 배치 2 는 2026-11-19 07:47 → 16:38 KST (in_vehicle → Amazon → QSAR-TID → physiochemical → APS, 두 설정 동시). 배치마다 내기 전 사용자 확인
- **배치 2 의 플랫폼 오류 2회 (APS × AIDE)**: 1차 (`…-120834`) · 2차 (`…-140031`) 모두 하네스 약 2~4분째 AI Runtime `INTERNAL_ERROR` (작업 10분 · 8분, 일반 메시지). 둘 다 둘째 노드의 TabICL (5-fold, 컨텍스트 약 9.7k 행) 이 도는 중이었고, 2차의 MLflow 시스템 지표로는 GPU 메모리 0.5 → 18GB (23GB 중) 뒤 실패, 호스트 메모리는 8GB 아래. 노드 로그는 SP 의 MLflow 실행에 있어 내 로그인으로는 못 읽음. 진행 사본 채점 (0.00483 · 0.0047, `run_crashed`) 은 1시간 조건이 아니라 둘 다 `exclude` 하고 같은 설정 rep 1 로 다시 돌림 (사용자가 정함). 3차 (`…-153007`) 는 같은 꼴의 TabICL 노드를 돌리고도 1시간을 다 씀 → 이 설정의 고정 실패는 아님 (원인 확인 못 함)
- 비용: 하네스 시간 합 배치 0·1 13.6시간 + 배치 2 9.9시간 (죽은 두 번 약 4분 포함). TFM 설치·사전 점검·회수를 더한 A10 시간은 배치 0·1 약 16~17시간, 배치 2 약 11시간으로 어림 (작업마다 62~65분, 작업별로 재지 않음)
- 11-18 에 TabArena C1 (`~f2`) 의 FM · GBDT 행이 `fab sync` 로 같이 받아졌지만 그쪽 작업의 것이라 넣지 않음. 배치 2 의 sync 에는 다른 행이 섞이지 않음

## C1 준비: 쓸 Agent 실행과 최종 제출의 계보 (2026-11-18)

C1 (에이전트 피처 → FM, [plans/roadmap.md](plans/roadmap.md) 의 "C1 규칙") 을 TabArena 에서 하려고 두 가지를 정함.
데이터셋마다 어느 Agent 실행의 풀이를 쓸지 (고르는 규칙), 그 실행의 최종 제출 (`submission.csv`) 에 실제로 들어간 성분·피처가 무엇인지 (계보 도구 + 카드).
F2 로 옮겨 FM·GBDT 를 돌리는 것 (`features/f2.py`) 은 다음 단계. 이번에는 카드의 "F2 후보" 까지.

- "최종 풀이" = 그 실행의 최종 제출에 실제로 들어간 것. 작업 디렉터리에 있지만 최종 섞기에 안 들어간 실험은 F2 가 아님
- 도구는 정해진 방식으로 찾을 수 있는 것 (계보·import·파일 입출력) 만 함. 피처의 뜻·타깃 의존·inductive 여부는 카드에서 사람·LLM 이 읽고 채움 (C1 규칙 1 의 "LLM 변환 + 사람 검토" 와 같음)
- 에이전트 지시문 (`fab/agent/additional_notes.txt` 등) 은 안 바꿈 (바꾸면 지금까지 결과와 이어 읽을 수 없음)

### 고르는 규칙 (2026-11-18 사용자가 정함)

| 규칙 | 내용 |
|---|---|
| R1 후보 | 원래 대회 (`~obf` 는 뺌) 의 r0f0 1회째 5설정. 12개 × 5 = 60회 ("Agent 확장 1회째" 의 요약 표와 같은 기준. 파일럿의 2회째는 후보 아님) |
| R2 고르기 | 데이터셋마다 Agent 최고 (`metric_error` 최소) 하나 = 요약 표의 "Agent 최고 (설정)" |
| R3 빼기 | 다음이면 다음 순위로 내려감 (카드에 둘 다 남김): (a) 점수가 피처가 아니라 배정·외부 정보에서 나옴 (MLE-bench leaf × Codex 를 뺀 선례), (b) 최종 제출에 원본 열 말고 피처 블록이 없음 (F2 = F0, 예: 스스로 깐 TFM 에 원본 열만), (c) 1차 규칙 (타깃 비의존 · inductive 안) 을 지키는 블록이 하나도 안 남음. 판단은 카드에서 |
| R3 해석 (2026-11-18 사용자가 정함) | (c) 의 "inductive 안" 은 train 으로만 다시 fit 할 수 있으면 됨: train+test 로 fit 한 블록 (빈도 수 · PCA · 레벨 등) 은 같은 계산을 train 행으로만 fit 해 F2 후보로 둠. (b) 의 "피처 블록" 에는 모델 바로 앞의 전처리도 넣음 (타깃 비의존 · inductive 안이면: fold 안 백색화, 고정 seed 랜덤 투영 등) |
| R4 하한 | 우리 GBDT refit (lgbm·catboost 중 나은 쪽, r0f0) 보다 나쁜 Agent 는 안 고름 (피처+모델로도 기본 GBDT 를 못 넘으면 피처 몫을 해석하기 어려움). 걸리면 그 데이터셋은 C1 빈칸. C1 규칙 2 의 "GBDT 의 F2 > F0" 은 C1 실행 뒤 따로 봄 |
| R5 하네스 | 데이터셋마다 하나 (최고) 만. 다른 하네스는 C1 결과를 보고 정함 ("정할 것") |

### 12개에 적용 (r0f0 1회째)

σ = FM (kumo) 의 모든 split 표준편차 (요약 표와 같음). 점수는 `metric_error`.

| 데이터셋 | 실행 id | 설정 | 점수 | Agent − FM ÷σ | 고른 까닭 | 뺀 까닭 · 주의 |
|---|---|---|---|---|---|---|
| QSAR_fish_toxicity | `tabarena-p05__codex-gpt__r1__20261116-133105` | Codex × GPT | 0.9003 | +0.29 | Agent 최고. 최종 ExtraTrees 성분 (최종의 0.8) 이 원본 6열 + 랜덤 투영 32열을 받음 → R3(b) 아님 | 투영도 블록으로 봄 (R3 해석). 다른 기준이었다면 AIDE × GPT (0.9084) 도 원본 6열만이라 Claude Code 공식 (0.9121) 까지 내려갔음. 스스로 깐 TabICL 은 모델 쪽. test 303행 중 55행이 train 과 6열 일치, 그 행에 train 정답 평균을 0.10~0.15 섞음 (뺄 것) |
| MIC | `tabarena-p06__aide-gpt__r1__20261116-145653` | AIDE × GPT | 0.4445 | +0.17 | Agent 최고 | 없음 |
| Is-this-a-good-customer | `tabarena-p07__claude-code-sonnet-cont__r1__20261116-161213` | Claude Code × Sonnet +cont | 0.2617 | +0.25 | Agent 최고 | 비율 3열뿐, 에이전트 CV 로도 효과 작음 (0.7445 대 0.7436) |
| splice | `tabarena-p08__codex-gpt__r1__20261116-172003` | Codex × GPT | 0.06179 | −0.34 | Agent 최고, FM 보다 나음 | 피처 블록 성분은 최종의 약 0.36. 생성 모델·OOF 스택·train 정답 이웃 치환은 뺌 (치환은 피처와 섞여 있어 R3(a) 로 실행째 빼지는 않음). 어휘·레벨이 train+test |
| students_dropout_and_academic_success | `tabarena-p09__aide-sonnet__r1__20261116-182911` | AIDE × Sonnet | 0.5515 | +1.88 | Agent 최고 | 범주 빈도 17열은 train+test |
| polish_companies_bankruptcy | `tabarena-p10__codex-gpt__r1__20261116-193556` | Codex × GPT | 0.00436 | +0.56 | Agent 최고 | 정확 일치·근접 이웃 정답 (test 28 · 40행), 스스로 깐 TabPFN 은 모델 쪽. 회계 항등식·차이 블록은 fit 없음 |
| QSAR-TID-11 | `tabarena-p11__codex-gpt__r1__20261116-204754` | Codex × GPT | 0.7976 | +0.23 | Agent 최고 | 1차 규칙 안 블록은 행 합 1열뿐이라 빈도 합·PCA 128 (train+test fit) 을 train 으로만 다시 fit 해 씀 (R3 해석, 아니면 R3(c)). 이웃 라벨 특징·쌍 랭킹은 뺌 |
| APSFailure | `tabarena-p12__aide-sonnet__r1__20261117-002041` | AIDE × Sonnet | 0.00478 | +1.46 | Agent 최고 | GBDT (lgbm 0.00488) 보다 0.07σ 만 나음 → 해석 약함 |
| Amazon_employee_access | `tabarena-p03__aide-sonnet__r1__20261116-215907` | AIDE × Sonnet | 0.1068 | −3.27 | Agent 최고, FM 보다 나음 | 최종의 타깃 비의존 블록 (빈도 45 · nunique 72) 이 모두 train+test → train 으로만 다시 fit 해 씀 (R3 해석, 아니면 R3(c) 로 Claude Code +cont 0.1093). 이득의 대부분은 타깃 인코딩·스택 (뺄 것) 으로 보임 |
| physiochemical_protein | `tabarena-p04__codex-gpt__r1__20261116-230759` | Codex × GPT | 2.9772 | +3.41 | Agent 최고 | 최종의 0.90 이 스태킹 메타 (타깃 의존). 피처만 받는 성분은 0.10 (nn · cat10). 정확 일치 조회 753행은 점수 원인 아님 (조회 무게를 바꿔도 OOF 같음) |
| hazelnut-spread-contaminant-detection | `tabarena-p01__aide-gpt__r1__20261114-090001` | AIDE × GPT | 0.01624 | +14.67 | 다음 순위 (GBDT 보다 −11.9σ) | 최고 Codex (0.00640) 는 R3(b): 최종 `finalize_submission.py` 가 TabICL 변형 (원본 열 · 잡음 늘리기 · 의사 레이블) 만 섞음. AIDE 의 블록은 fold 안에서 fit 한 백색화 30열뿐 (커널 SVM 의 입력) → 블록으로 봄 (R3 해석, 아니면 hazelnut 은 C1 빈칸) |
| in_vehicle_coupon_recommendation | `tabarena-p02__codex-gpt__r1__20261114-122945` | Codex × GPT | 0.13150 | −1.38 | Agent 최고, FM 보다 나음 | 응답자 템플릿 묶음 (6열) 이 train+test. `person` 키 자체는 행 단위 (fit 없음, 앞 절의 "train+test 로 응답자 id" 는 정확히는 이 묶음). 응답자별 타깃 집계는 최종에 없음 |

- 12개 모두 R4 (GBDT 보다 나음) 통과. R3 로 뺀 것은 hazelnut × Codex 하나. R3 해석 (위) 에 걸린 넷: train 으로만 다시 fit (Amazon · QSAR-TID), 모델 앞 전처리를 블록으로 (QSAR_fish · hazelnut)
- 설정별: Codex × GPT 6 · AIDE × Sonnet 3 · AIDE × GPT 2 · Claude Code × Sonnet +cont 1

### 계보 도구 (`fab ta agent lineage`, `fab/agent/lineage.py`)

```
uv run --frozen fab ta agent lineage <key 또는 실행 폴더>                      # 사람이 읽는 나무
uv run --frozen fab ta agent lineage <key> --json --out lineage.json           # 기계가 읽는 JSON
```

- 회수한 실행 폴더 (`meta.json` · 하네스 기록 · `workspace.tgz` · `submission.csv`) 만 읽음 (원격 없음). key 는 `AGENT_RUNS` (메인 체크아웃의 `results/_mlebench/agents/`, `FAB_OUTPUTS` 로 바꿈) 아래에서 찾음. 없으면 `fab sync --pull outputs/_mlebench/agents/<key>` (worktree 에서는 `FAB_OUTPUTS=<worktree>/results` 를 붙이면 메인 체크아웃에 안 씀. 결과표도 같이 받으니 바뀐 `results/*.csv` 는 되돌릴 것)
- `fab mleb agent lineage` 도 같음 (MLE-bench 실행)

하는 일

| 단계 | 규칙 |
|---|---|
| 실행 목록 | Codex: `command_execution` 의 명령 (`bash -lc` 안쪽). Claude Code: Bash 명령 + Write·Edit 도구 (Bash 의 cwd 는 다음 명령으로 이어짐). AIDE: `journal.json` 의 노드마다 코드 하나 = 실행 하나 (step 순), 최종 = best 노드 (`node_id.txt`) |
| 명령 풀기 | `python x.py` · `python - <<'PY' … PY` (인라인) · `python -c` · `bash x.sh` (내용을 풀어 봄) · `cat > x.py <<EOF` (내용이 남음) · `cp`/`mv` · `sed -i` (내용 모름) · `> 파일`. 셸 `for x in a b; do …; done` 은 값이 글자 목록이면 펼침. `nohup`·`timeout`·`nice -n`·환경 변수 대입은 뗌 |
| 성공 여부 | exit 0 = 성공, 124·137·143 (시간 제한·신호로 끊김) = 모름 (쓰기 후보로 셈: physio × Codex 의 `metric_local.py` 는 143 으로 끊겼지만 결과 npz 를 최종 섞기가 읽음), 그 밖 = 실패 (쓰기로 안 셈). Claude Code 는 `is_error` |
| 코드 안 경로 | AST: 글자 · f-string (`{…}` = `*`) · Path `/` · `os.path.join` · `+` · `%` · 변수 대입 · 글자 목록을 도는 `for` (값마다 펼침). 부른 함수로 역할: `np.load`·`read_csv`·`open(r)` 읽기, `np.save`·`to_csv`·`open(w)`·`joblib.dump` 쓰기, `shutil.copy`·`os.replace`·`Path.replace` 복사. 함수 인자로 넘어가 역할을 모르는 경로는 "언급" (읽기로 봄, 쓰기로는 안 셈). `exec(open('x.py').read())`·`runpy.run_path` 는 코드 의존으로 따라감. `import` 한 모듈 중 작업 디렉터리에 `<모듈>.py` 가 있으면 로컬 모듈 (그 경로도 같이) |
| 생산자 | 파일을 읽은 실행보다 앞선 실행 중: 정확한 경로 (`*` 없음) 로 쓴 실행이 있으면 그 마지막 (덮어쓰기). 없으면 `*` 밖 글자가 가장 많은 쓰기의 마지막. 그 쓰기가 서로 다른 코드 둘 이상이면 "여럿" (고르지 않음), 쓰기가 없으면 "못 찾음", 파일 이름이 통째로 변수 (`f'{name}.npz'`) 면 "이름이 변수" (따라가지 않음). 정확한 쓰기 뒤에 같은 파일과 맞는 `*` 쓰기가 있으면 `later_wildcard_writers` 로 표시. 같은 코드가 쓰는 경로를 그 코드 안 첫 쓰기보다 앞 (줄·열) 에서 읽으면 "읽고 고쳐 다시 쓰기" (`rewrites`, 예: 제출을 읽어 보정하고 덮어씀) → 앞선 다른 코드의 쓰기로 이어 감. 쓴 뒤 읽으면 (검사 서버에 올리기 등) "자기가 씀" |
| 끌어온 코드 | 코드에 `exec(`·`compile(`·`runpy` 가 있고 `.py` 를 읽으면 그 파일의 쓰기도 이 실행의 쓰기로 셈 (`s = Path('x.py').read_text().replace(...); exec(compile(s, ...))` 꼴). 읽은 글에 `.replace(` 가 있으면 `exec_text_modified` (글자를 고쳐 돌렸을 수 있음: 도구는 치환을 풀지 않음, 카드에서 확인). 읽은 글을 잘라 (`.split(…)[0]` 등) 앞부분만 돌리면 쓰기로 안 셈. 명령 앞의 환경 변수 (`VERSION=6`, 스레드 수는 뺌) 는 실행마다 `env` 로 남김 |
| 코드 판 | 실행 시점의 코드: 기록에 전체 내용 (heredoc · Write, Edit 은 적용) 이 남은 판이 있으면 그것 (`판=log`), 아니면 작업 디렉터리 최종본 (`판=workspace`). 최종본인데 실행 뒤 그 파일을 고친 명령 (`sed -i` · 인라인 쓰기 등) 이 있으면 `changed_after_run` (실행한 판과 다를 수 있음) |
| 빼는 파일 | 작업 디렉터리의 pip 가 깐 파일 (`*.dist-info/RECORD` 목록, `fab/agent` 의 `installed_files`) 은 코드로 안 봄 (9 의 "사후 검사" 와 같은 규칙) |
| 내는 것 | 제출 경로와 그것을 마지막으로 쓴 실행, 제출과 바이트가 같은 작업 디렉터리 파일, 노드 (코드) 마다 실행 명령·순서·판·import·로컬 모듈·읽는 파일 (상태: 찾음 / 여럿 / 못 찾음 / 데이터 / 코드 / 자기가 씀 / 이름이 변수, 생산자, 파일 크기·npy 모양·csv 행열), AIDE 는 best 노드의 조상 (step·metric), 못 찾은 연결 목록 |

하지 않는 것: 피처의 뜻·타깃 의존·train+test fit·섞기 무게의 해석 (카드), 셸 변수·argparse·명령 출력으로만 정해지는 경로, 같은 스크립트를 인자만 바꿔 여러 번 돌린 경우 어느 실행이 어느 파일을 썼는지 (스크립트 단위로 모으고 실행 목록을 남김).

로컬 확인 (2026-11-18)

- 단위 테스트 15개: `uv run --frozen python -m unittest discover -s tests -v` (가짜 실행 폴더로 Codex · Claude Code · AIDE, 경로 식, 셸 풀기, 끊긴 실행, 여럿·못 찾음을 고르지 않는지, pip 가 깐 파일 빼기, 읽고 고쳐 다시 쓰기, 고쳐 exec · 잘라 exec)
- 로컬에 있는 회수물 46개 (TabArena 38 · MLE-bench 8, 하네스 셋 모두) 에 돌림: 46개 모두 제출을 쓴 실행을 찾음, 죽은 것 없음, 46개 모두 회수한 `submission.csv` 와 작업 디렉터리의 제출 사본이 바이트까지 같음. 걸린 시간 대부분 1~5초, 큰 Codex 실행 15~60초 (splice 56초)
- 손으로 따라간 기록과 견줌: tps-may × Codex `build_submission2.py` · leaf × AIDE × Sonnet 노드 `34920bd2` (roadmap 의 C1 확장) · hazelnut × Codex `finalize_submission.py` (파일럿) 는 같음.
  카드를 쓰며 코드 읽기와 어긋난 셋은 도구를 고쳐 맞춤: QSAR_fish × Codex (제출을 세 번 읽고 고쳐 다시 씀 → 처음엔 "자기가 씀" 으로 멈춤), physio × Codex (최종 #272 가 `compose_submission_nn.py` 의 무게 식을 고쳐 exec → 처음엔 그 앞 #261 을 냄), Amazon-obf × Codex (같은 꼴, 최종 #259).
  "범주 대 오염" 의 "풀이 읽기" 와는 둘이 다름 (아래 "정할 것"): splice-obf × Codex 의 최종 제출은 `stack_both_test.py` 의 `stackboth_avg_0.1.csv` 를 복사한 것 (그 절은 `compose_submission.py`), in_vehicle-obf × Codex 는 `build_weighted.py` 의 `candidate.csv` 를 복사한 것 (그 절은 `build_submission.py`). 둘 다 작업 디렉터리에서 제출과 바이트가 같은 파일로 다시 확인함

### 카드

같은 틀 (머리 표 · 최종 제출을 만든 성분 · 피처 블록 (타깃 의존 · train+test fit · fold 안 fit · 몫) · F2 후보와 뺄 것 · 동일성 확인에 쓸 값 · 못 찾은 연결). 도구 출력 + 코드 읽기 (Is-this 는 직접, 나머지 11장은 하위 에이전트가 쓰고 검토함).

| 카드 | F2 후보 (열) | 조건부 (train 으로만 다시 fit 하면) | 동일성 확인 |
|---|---|---|---|
| [QSAR_fish](../competitions/ta-qsar-fish-toxicity/c1.md) | `~f2` 38 (변환 6 + 랜덤 투영 32) | – | `power_proj.log` 의 CV RMSE 0.83228 |
| [MIC](../competitions/ta-mic/c1.md) | `~f2m` 128 (+ 결측 지시 17) | – | 트리 CV 0.413678 → 선형 더한 0.409726 |
| [Is-this](../competitions/ta-is-this-a-good-customer/c1.md) | `~f2` 16 (비율 3) | – | `r.py` CV AUC 0.76056 |
| [splice](../competitions/ta-splice/c1.md) | (행 단위 원-핫뿐, 어휘가 train+test) | `~f2s` 수천 (k-mer · 쌍 원-핫 · 구간 개수), `~f2` 5만+ (TFM 상한) | 배열 모양 (3190, 1310) · (3190, 48727), 성분 OOF logloss |
| [students](../competitions/ta-students-dropout-and-academic-success/c1.md) | `~f2s` 43 (파생 7) | `~f2` 60 (+ 범주 빈도 17) | LR CV logloss 0.55776 |
| [polish](../competitions/ta-polish-companies-bankruptcy/c1.md) | `~f2s` 138 (항등식 74), `~f2` 300 (+ 차이 236) | `~f2e` 362 (+ 같은 값 쌍 224) | 블록별 CatBoost CV AUC (0.99394 · 0.99244) |
| [QSAR-TID-11](../competitions/ta-qsar-tid-11/c1.md) | (행 합 1열뿐) | 1,024 + 빈도 합 3~4 + PCA 128 | `pca_features.npy` (5742, 128) |
| [APSFailure](../competitions/ta-apsfailure/c1.md) | `~f2` 248 (결측 수 · 무리 합 · 무리 안 비율 78) | – | LGB OOF AUC 0.99130 |
| [Amazon](../competitions/ta-amazon-employee-access/c1.md) | – | `~f2` 117 (빈도 45 · nunique 72) | CB OOF AUC 0.87854 |
| [physiochemical](../competitions/ta-physiochemical-protein/c1.md) | `~f2` 45 (비율 36), `~f2s` 54 (log · log 차이) | – | `full_cat10_preds.npz` 값, holdout RMSE 3.30665 |
| [hazelnut](../competitions/ta-hazelnut-spread-contaminant-detection/c1.md) | `~f2s` 30 (fold 안 백색화) | – | 노드 CV 0.985474 · holdout 0.988203 |
| [in_vehicle](../competitions/ta-in-vehicle-coupon-recommendation/c1.md) | `~f2s` 약 69 (행 단위 교차 · 구간) | `~f2` 약 75 (+ 응답자 템플릿 6) | xgb 3-fold CV 0.817503 (GPU) |

### 정할 것

- **C1 실행 (다음 단계)**: 카드의 F2 후보를 `competitions/ta-<데이터셋>/features/f2.py` 로 옮기고 동일성 확인 → F0/F2 × {tabicl, kumo, tabpfn_v2} + {lgbm, catboost} (r0f0 먼저, 결과를 보고 모든 split). 데이터셋마다 FM·GBDT 작업 하나씩
  - 따로 진행 중인 C1 실행 (브랜치 `Aiden-Jeon/tabarena-c1`, 카드 전에 시작) 의 `f2.py` 는 이 카드·R3 해석과 다른 곳이 있음.
    그 PR 이 올라온 뒤 카드 기준으로 고쳐 다시 돌림 (2026-11-18 사용자가 정함: 이 절의 규칙이 기준). 2026-11-18 에 본 차이:

    | 데이터셋 | 그 `f2.py` | 카드 기준으로 고칠 것 |
    |---|---|---|
    | train+test fit 블록 (Amazon 빈도·nunique, students 범주 빈도, splice n-gram 어휘, QSAR-TID 비트 빈도, polish equals 쌍 고르기) | 에이전트 그대로 train+test 로 계산 (inductive 밖 표시만) | train 행으로만 fit (R3 해석) |
    | in_vehicle | 응답자 id 계열 (`person` 키 · `person_demo` · 설문 묶음) 을 모두 뺌 (55열) | `person` 키는 행 단위 (fit 없음) → 넣음. 설문 묶음 (6열) 은 train 으로만 다시 fit 해 넣음 (약 75열) |
    | QSAR-TID-11 | 지문 요약 3열만 (1,027열), PCA 128 은 커널 SVR 입력이라 뺌 | PCA 128 도 넣음 (모델 바로 앞의 전처리, R3 해석), train 으로만 fit |
    | Amazon | 쌍 36 · 삼중 28 을 범주로 넣음 (190열) | 최종 제출에 타깃 비의존 꼴로 안 들어감 → 뺌 (117열) |
    | 같음 | APSFailure 248 · Is-this 16 · physio 45 · QSAR_fish (원본 6 + 투영 32) · students 60 · polish · splice 의 열 범위 (트리 성분 입력 1,165, 카드보다 좁고 정확 → 그쪽을 따름) | – |
    | 없음 | MIC · hazelnut | 카드대로 새로 씀 |

    그쪽 r0f0 결과 (in_vehicle · splice · Amazon) 는 이 차이가 섞인 값이라 고친 뒤 다시 돌린 값으로 바꿈
    (해결: PR #35, 2026-11-18)
- (정함 2026-11-18, 사용자) inductive 밖 블록은 train 으로만 다시 fit 해 옮김 · 모델 바로 앞의 전처리도 블록으로 봄 (위 "R3 해석"). 카드의 "조건부" 열이 C1 1차 후보.
  다시 fit 하면 값이 에이전트 배열과 달라지므로 동일성 확인은 train+test 판으로 먼저 맞춘 뒤 train 판으로 바꿈
- splice 카드의 메모: 최종 명령 (#293) 이 "애매한 염기" 보정 6행 중 1행 (id 2376) 만 제출에 옮김 (나머지 5행을 안 옮긴 까닭은 코드에 없음). 점수·F2 에는 영향 없음 (예측 쪽 보정)
- **하네스**: 12개 중 Codex 가 다수 → C1 결과가 "Codex 피처" 에 기울 수 있음. 다른 하네스 최고를 더할지 (예: Codex 가 최고인 곳 중 다른 하네스가 1σ 안인 QSAR_fish · splice · QSAR-TID) 는 C1 결과를 보고
- **"풀이 읽기" (범주 대 오염) 를 다시 볼 것**: 도구로 보니 splice-obf · in_vehicle-obf × Codex 의 최종 제출 스크립트가 그 절에 적은 것과 다름 (위 "로컬 확인"). in_vehicle-obf 의 `build_weighted.py` 는 `profile_svc` 성분 (`cv_svc_profile.py`·`cv3_svc_profile.py` 의 출력, 어느 쪽인지는 도구로 "여럿") 을 읽음 → 그 절의 "`person` 이 같은 행끼리의 커널 SVM 은 최종 섞기에 없음" 과 맞지 않음. 그 절의 결론 (응답자 id 풀이가 남음 · FM 을 못 넘음) 은 점수로 정해져 바뀌지 않지만, 성분 목록은 이 도구로 다시 쓸 것
- **앞으로의 대안 (안만, 구현 안 함)**: 1시간이 끝난 뒤 같은 세션에 "최종 제출을 만든 파이프라인 (스크립트·입력 파일·섞기 무게) 을 설명하라" 고 따로 물어 기록. 점수에는 영향이 없지만 (제출 뒤) 다시 실행해야 하고, 답은 도구 결과와 대조하는 참고로만 씀 (에이전트의 설명이 틀릴 수 있음). 지시문은 그대로 둠

## C1: 에이전트 피처 → FM (r0f0, 2026-11-18)

질문 2 ("에이전트가 만든 전처리를 FM 에 넣으면 FM 이 좋아지는가, GBDT 보다 더 좋아지는가") 를 Agent 를 돌린 12개의 r0f0 에서 봄.
MLE-bench 의 C1 ([mle-bench.md](mle-bench.md) 의 "C1"·"C1 확장") 과 같은 방법: 데이터셋마다 **Agent 최고 실행** 의 최종 풀이에서 전처리만 떼어
`competitions/ta-<데이터셋>/features/f2.py` 로 옮기고, F0 (원본) 대 F2 를 FM 둘 (`kumo_cat_refit` · `tabicl_kv_refit`) 과 GBDT 둘
(`lgbm_refit` · `catboost_refit`) 로 돌림 (튜닝 없음, refit 한 번). 실행 이름은 `<모델>~f2`, 결과표의 note `c1`. tabpfn_v2 는 뺌 (C1 확장 #12 와 같은 조합).

- 규칙은 [plans/roadmap.md](plans/roadmap.md) 의 "C1 규칙" 그대로 (2026-11-18 사용자): 모델·섞기·후처리는 모델 몫이라 뺌, 타깃을 쓰는 피처 (타깃 인코딩·OOF 스태킹) 는 뺌,
  점수가 피처가 아닌 데서 나온 부분 (train 정답을 쓰는 보정) 은 뺌. 원-핫·번호 매김 같은 모델 입력 인코딩은 빼고 그 열은 원래 꼴 (범주) 로 둠
- **기준은 "C1 준비" 절의 카드 · R3 해석** (2026-11-18 사용자가 정함). 이 절의 첫 판 (아래 "첫 판에서 바꾼 것") 은 카드 전에 데이터셋마다 판단했고, 카드 기준으로 8개를 고쳐 다시 돌림
  - **train+test 로 fit 한 비지도 변환** (빈도·어휘·비트 빈도·PCA·쌍 고르기·설문 묶음) 은 같은 계산을 **train 행으로만 fit** 해 옮김. test 에만 있는 값은 빈도 0 · 어휘 밖 0 · −1 처럼 정해 둠
  - **모델 바로 앞의 전처리** (fold 안 백색화, PCA 커널 입력 등) 도 타깃 비의존 · inductive 안이면 피처 블록으로 봄
  - 동일성 확인은 train+test 판으로 먼저 에이전트 값과 맞춘 뒤 train 판으로 바꾸고, 두 단계를 모두 표에 적음
- 모델 이름: F0 의 kumo 는 범주 열이 없는 데이터셋이면 `kumo_c8k_refit` 값 ("FM 프로토콜과 이름"). F2 는 모두 `kumo_cat_refit~f2` (범주가 없으면 같은 설정)
- `~f2` 는 Elo 풀에 넣지 않음 (`fab ta elo` 는 모델 이름을 정확히 맞춰 고름 → `results/tabarena_elo.csv` 그대로, 2026-11-18 확인)

### 옮긴 피처

출처 Agent 는 README 점수표의 Agent 최고 (설정 · r0f0 `metric_error`). 열은 F0 → F2. 동일성 확인은 로컬 (Mac, 2026-11-18): 옮긴 피처로 에이전트의 표형 성분을 같은 설정·fold·seed 로 다시 돌리거나 남은 배열과 비교 (hazelnut · MIC 는 에이전트 풀이를 r0f0 로 다시 돌림). "옛" 은 이 절의 첫 판 (PR #35 의 처음 커밋).

| 데이터셋 | 출처 Agent | 옮긴 블록 | 열 | 뺀 것 (이유) | train+test fit | 동일성 확인 |
|---|---|---|---|---|---|---|
| splice | Codex × GPT 0.06179 | 트리 성분 (cat·xgb·ext) 입력의 합집합: 창별 염기 묶음 개수·최장 연속·2–3-gram 개수 987, 위치별 2·3-mer 와 중앙 위치 쌍 73 (범주), motif 그룹 1 (범주) | 60 → 1,121 (옛 1,165) | 생성 모델 9종·SVM 커널·신경망 (모델, 신경망 입력은 OOF 우도), 최근접 train 투표 (train 정답), 섞기·제곱·평탄화, LR 성분의 희소 k-mer one-hot (가중치 2%, xgb 의 k-mer 와 겹침, 사용자) | n-gram 어휘 → **train 으로 fit** (test 에만 있는 44개는 열 없음) | train+test 판 1,165열이 옛 판과 배열이 같음 (옛 판으로 ext-XGB 0.1041 대 기록 0.1037). train 판: 남은 1,121열은 값이 같고, 빠진 44열은 train 행에서 모두 0 (test 6행에만 값) |
| Amazon | AIDE × Sonnet 0.1068 | 원본·쌍 36 의 값 빈도 45, a 별 b 의 고유 수 72 | 9 → 126 (옛 190) | **쌍 36 · 삼중 28 의 교차 열 자체** (최종에서 타깃 인코딩 키·one-hot LR 입력으로만 쓰여 타깃 비의존 꼴로 안 들어감, 카드), 교차 45열의 타깃 인코딩 (규칙 3), one-hot LR 의 OOF logit (스태킹), 모델·순위 섞기 | 빈도·고유 수 → **train 으로 fit** (test 에만 있는 값은 0) | train+test 판 126열이 옛 판의 같은 열과 같음 (옛 판으로 CatBoost 성분 CV 0.87735 대 기록 0.87854). train 판: 117열 모두 바뀜 (셀의 66% · test 75%) |
| in_vehicle | Codex × GPT 0.1315 | `person` 키 (인구통계·방문 빈도 13열을 이은 값, 행 단위) 와 순서 숫자 등 10, 교차 20 (person·person_demo × 상황, 쿠폰 × 상황, 범주), 순서·방문 빈도 통계·시나리오 15, 설문 묶음 6 (`survey_block`·`survey_template` 과 person 교차) | 24 → 75 (옛 55) | 응답자별 타깃 집계 (`add_stats`, 규칙 3), 응답자 × 장소 과거 정답 합 (최종 체인 밖), one-hot·2차 다항, 모델 | 설문 묶음 6열 → **train 으로 fit**. 묶음이 6개 → 2개, `survey_block` 이 −1 인 행 60% (train 5,095 / 8,456, test 61%), `survey_template` 'amb' 60% | train+test 판의 새 51열이 에이전트 출력과 같음 (에이전트 코드를 pandas 2.3 으로). 그 열로 xgb_more (seed 11, fold 0) CPU 0.817321 = 에이전트 피처 그대로 (예측 같음, 로그 GPU 0.817503). train 판: 설문 6열만 바뀜 (행의 99%), 나머지 69열 같음 |
| Is-this | Claude Code × Sonnet +cont 0.2617 | 금액/소득, 금액/기간, (금액/기간)/소득 | 13 → 16 | CatBoost seed 평균 | 없음 | CatBoost CV 0.7531 대 기록 0.7606. 원본만 넣은 `r3.py` 기록도 0.7428 대 0.7436 으로 다름 → 1,148행의 플랫폼 차이 |
| students | AIDE × Sonnet 0.5515 | 학업 비율 7 (학기별 승인/등록·승인/평가, 차, 합, 성적 평균), 범주 17열의 값 빈도 17 | 36 → 60 | 범주 번호 매김 (모델 입력), 8개 모델·스태킹·온도 | 빈도 → **train 으로 fit** (test 에만 있는 값은 0) | train+test 판 60열이 옛 판과 같음 (옛 판으로 CatBoost CV 0.53378 대 기록 0.53332). train 판: 빈도 17열만 바뀜 (개수라 셀의 99.8%), 나머지 43열 같음 |
| APS | AIDE × Sonnet 0.00478 | 결측 수 1, 히스토그램 무리 7개의 합 7, 무리 안 비율 70 | 170 → 248 | 세 모델 순위 평균 | 없음 | **LightGBM 성분 OOF AUC 가 16자리까지 같음** (0.9912980404204428) |
| QSAR_fish | Codex × GPT 0.9003 | 무작위 사영 32 (열 3·4 sqrt → Yeo-Johnson → 표준화 → 정규 난수 6 × 32, seed 42, 변환은 train 만으로 fit) | 6 → 38 | 단조 변환한 6열 자체, log1p 변형·다른 seed 의 사영, SVR·kNN·TabICL, train 과 같은 행에 정답 섞기·kNN 잔차 보정 (train 정답) | 없음 (train fit) | **forest 예측이 남은 배열과 같음** (최대 차 3.6e-15) |
| polish | Codex × GPT 0.00436 | CatBoost 성분 셋 입력의 합집합: 회계 항등식 22 × 3 + 비율 8 = 74, 같은 접미사 열끼리의 차 + 재무 = 231 (identity 와 정의가 같은 5 는 한 번), 자주 같은 열 쌍 100 × (같음 표시·차, diff_ 와 겹치는 차는 한 번) = 143 | 64 → 512 (옛 533) | fillna(−1e10) (모델 입력), 모델·fold 안 변환·TabPFN, 그룹 사전 확률 1.25배, 최근접·같은 train 행 정답 덮어쓰기 (train 정답) | 쌍 고르기 → **train 으로** (112쌍 → 100쌍) | train+test 판의 열 수가 로그와 같음 (138 · 300 · 362 / 112쌍), 열마다 에이전트 함수 출력과 같음 (항등식 CatBoost seed 123 AUC 0.99491 대 남은 oof 0.99458). train 판: 512열 모두 train+test 판의 같은 열과 같음, 쌍 12개 빠짐 |
| QSAR-TID-11 | Codex × GPT 0.7976 | 켜진 비트 수, X · freq, X · (−log freq) (freq = 비트 빈도), **PCA 128** (PCA 커널 SVR 의 입력 `pca_features.npy`, 모델 바로 앞의 전처리) | 1,024 → 1,155 (옛 1,027) | Tanimoto 커널·kNN·SVR, 임계값 증강·순서형 NN, 최대 유사도 이동·최근접 Ridge·spline 보정 (train 정답), `lgb_cv.py` 의 X · √freq (최종 LGB 는 extra 없음) | 비트 빈도·PCA 평균·고유벡터 → **train 으로 fit** | train+test 판 PCA 가 `pca_features.npy` 와 열마다 r ≥ 0.999996 (부호 맞춘 뒤). 지문 요약 3열은 옛 판 (xgbC 0.7538 대 기록 0.7550). train 판: PCA 는 128열 중 5열만 \|r\| > 0.99 (중앙값 0.52), 요약 2열은 상대 차 최대 0.4% |
| physiochemical | Codex × GPT 2.977 | 쌍 비율 a / (b + 0.0001) 36 | 9 → 45 | meta 모델의 OOF 예측 입력 (스태킹), 학습한 거리 피처, GP·local·NN, 같은 행 정답 섞기·clip | 없음 | **lgb63_42 의 test 예측이 남은 배열과 같음** (최대 차 0) |
| hazelnut | **AIDE × GPT 0.01624** (옛 Codex × GPT 0.00640) | fold 안 백색화 30 (StandardScaler → 공분산 eigh → 고유벡터 / √(λ + 0.1), 커널 SVC 의 거리 공간, 모델 바로 앞의 전처리) | 30 → 60 (옛 피처 없음) | Codex 는 R3(b) 로 뺌 (모든 성분이 스스로 깐 TabICL 에 원본 열). Fisher 방향·공분산 대비 metric·Ledoit–Wolf (y 로 fit), 커널 SVC·설정 고르기·fold 평균, 점수 z-score (단조) | 없음 (에이전트도 학습 행으로 fit. 여기서는 train 전체로) | train fit Z 가 에이전트 `SensorMetric().fit_transform` 과 같음. 이 Z 로 바꾼 `best_solution.py` 를 r0f0 로 돌려 카드 값 dev 5-fold AUC 0.985474 · holdout 0.988203 재현 |
| MIC | AIDE × GPT 0.4445 | 수치 17열의 결측 지시 (`MissingIndicator(features="all")`, 선형 사망 성분 0.30 의 입력, 카드의 `~f2m`) | 111 → 128 (옛 피처 없음) | 중앙값 대치·표준화·OneHot (선형 모델 입력), CatBoost 입력의 −999, 사망·원인 두 단계 (dead 라벨 = 타깃), 보정·섞기 | 없음 (행마다) | 지시 열이 `MissingIndicator` 출력과 train·test 모두 같음 (r0f0 train 에서 17열 모두 결측이 있음). 에이전트 `solution.py` 를 r0f0 로 돌린 값이 지시 열을 바꿔 넣어도 같음 (트리 CV 0.413857 → 선형 더해 0.409785, 제출 파일도 같음. 카드 값 0.413678 → 0.409726 과의 차는 플랫폼) |

- 옮긴 계산식·출처 파일은 각 `f2.py` 의 설명. 작업 디렉터리는 Volume 의 `outputs/_mlebench/agents/<key>/` (`fab sync --pull`)
- 피처 상한 (C1 규칙 6) 은 모두 두지 않음: 가장 넓은 QSAR-TID 1,155열도 TabArena FM 이 돈 너비 (1,777열) 안

### 결과 (r0f0 test `metric_error`, 낮을수록 좋음)

이 표는 12개 (2026-11-18). kddcup09 의 `~f2` (2026-11-20) 는 [행 추가: kddcup09_appetency](#행-추가-kddcup09_appetency-r0f0-2026-11-19--11-20) 의 "C1".

이진 1 − AUC, 다중 logloss, 회귀 RMSE. 괄호는 F0 대비 변화. **굵게** = 그 데이터셋에서 FM·GBDT·Agent 중 최고. * = 2026-11-18 에 `f2.py` 를 카드 기준으로 고쳐 다시 돌린 8개 (아래 "첫 판에서 바꾼 것").
Agent 최고는 README 점수표의 값 (hazelnut 은 Codex × GPT. C1 의 출처는 AIDE × GPT).

| 데이터셋 | kumo F0 → F2 | tabicl F0 → F2 | lgbm F0 → F2 | catboost F0 → F2 | Agent 최고 |
|---|---|---|---|---|---|
| QSAR_fish | **0.8855** → 0.8949 (+1.1%) | 0.8938 → 0.9112 (+2.0%) | 0.9308 → 0.9159 (−1.6%) | 0.9123 → 0.9010 (−1.2%) | 0.9003 |
| Is-this | 0.2565 → **0.2549** (−0.6%) | 0.2600 → 0.2615 (+0.6%) | 0.3363 → 0.3345 (−0.5%) | 0.3197 → 0.3188 (−0.3%) | 0.2617 |
| splice * | 0.0669 → **0.0601** (−10.2%) | 0.0722 → 0.0626 (−13.2%) | 0.0940 → 0.0973 (+3.5%) | 0.0958 → 0.0756 (−21.1%) | 0.0618 |
| students * | **0.5251** → 0.5271 (+0.4%) | 0.5515 → 0.5528 (+0.2%) | 0.6419 → 0.6589 (+2.6%) | 0.5702 → 0.5565 (−2.4%) | 0.5515 |
| polish * | 0.00342 → **0.00132** (−61.4%) | 0.01594 → 0.00469 (−70.6%) | 0.03743 → 0.00520 (−86.1%) | 0.03437 → 0.00626 (−81.8%) | 0.00436 |
| QSAR-TID-11 * | 0.7867 → **0.7718** (−1.9%) | 0.8643 → 0.8372 (−3.1%) | 0.8459 → 0.8765 (+3.6%) | 0.8729 → 0.8873 (+1.6%) | 0.7976 |
| APS | 0.00265 → **0.00254** (−4.3%) | 0.00454 → 0.00480 (+5.7%) | 0.00488 → 0.00487 (−0.2%) | 0.00496 → 0.00556 (+12.1%) | 0.00478 |
| Amazon * | 0.1277 → 0.1299 (+1.7%) | 0.1511 → 0.1799 (+19.0%) | 0.1576 → 0.1374 (−12.8%) | 0.1192 → 0.1268 (+6.4%) | **0.1068** |
| physiochemical | **2.852** → 2.868 (+0.6%) | 3.017 → 3.035 (+0.6%) | 3.961 → 3.744 (−5.5%) | 3.977 → 3.765 (−5.3%) | 2.977 |
| in_vehicle * | 0.1371 → 0.1327 (−3.2%) | 0.1489 → 0.1574 (+5.7%) | 0.1622 → 0.1540 (−5.0%) | 0.1590 → 0.1370 (−13.8%) | **0.1315** |
| MIC * | 0.4412 → **0.4280** (−3.0%) | 0.4587 → 0.4518 (−1.5%) | 1.101 → 1.106 (+0.4%) | 0.4712 → 0.4652 (−1.3%) | 0.4445 |
| hazelnut * | **0.00390** → 0.00561 (+43.8%) | 0.00669 → 0.00732 (+9.4%) | 0.02623 → 0.02159 (−17.7%) | 0.02879 → 0.02362 (−18.0%) | 0.00640 |

C1 지표: 각 쪽 최고끼리 (FM = kumo·tabicl 중 최고, GBDT = lgbm·catboost 중 최고, F0·F2 각각). C1 = FM 의 향상 − GBDT 의 향상 (향상 = 오차가 준 비율), 양수면 "FM 에 특히 도움".
1 − AUC 처럼 작은 값은 % 가 기울 수 있어 절대 차도 같이 적음. FM 최고는 12개 모두 kumo (F0·F2 둘 다). MIC 의 lgbm 은 F0 부터 1.10 (다중 8클래스, 기본 설정) 이라 GBDT 최고는 catboost.

| 데이터셋 | FM 최고 F0 → F2 | GBDT 최고 F0 → F2 | C1 (%p) | C1 (절대) | FM F2 − Agent | 첫 판 C1 (%p) |
|---|---|---|---|---|---|---|
| QSAR-TID-11 | −1.9% (−0.0149) | +3.6% (+0.0306) | **+5.5** | +0.0455 | −3.2% | +2.4 |
| Amazon | +1.7% (+0.0022) | +6.4% (+0.0076) | +4.6 (둘 다 나빠짐) | +0.0054 | +21.7% (못 미침) | +5.4 |
| APS | −4.3% (−0.00011) | −0.2% (−0.00001) | +4.1 | +0.0001 | −46.9% | +4.1 (그대로) |
| MIC | −3.0% (−0.0132) | −1.3% (−0.0060) | +1.7 | +0.0072 | **−3.7% (FM F0 −0.7%)** | 피처 없음 |
| Is-this | −0.6% (−0.0016) | −0.3% (−0.0010) | +0.3 | +0.0006 | −2.6% | +0.3 (그대로) |
| QSAR_fish | +1.1% (+0.0095) | −1.2% (−0.0113) | −2.3 | −0.0208 | −0.6% | −2.3 (그대로) |
| students | +0.4% (+0.0020) | −2.4% (−0.0137) | −2.8 | −0.0157 | −4.4% | −2.9 |
| physiochemical | +0.6% (+0.016) | −5.5% (−0.217) | −6.1 | −0.234 | −3.7% | −6.1 (그대로) |
| splice | −10.2% (−0.0069) | −19.6% (−0.0184) | −9.3 | −0.0116 | **−2.8% (넘음)** | −8.7 |
| in_vehicle | −3.2% (−0.0044) | −13.8% (−0.0220) | **−10.6** | −0.0176 | +0.9% (못 미침) | −0.1 |
| polish | −61.4% (−0.0021) | −84.9% (−0.0292) | −23.6 | −0.0271 | −69.7% | −29.6 |
| hazelnut | +43.8% (+0.0017) | −17.7% (−0.0046) | **−61.6** | −0.0064 | −12.3% | 피처 없음 |

### 해석 (r0f0 한 split, 실행 1회. 판단은 약함)

- **"FM 에 특히 도움" (C1 양수) 은 12개 중 5개**지만, 둘 다 좋아진 위에서 FM 이 더 좋아진 곳은 QSAR-TID (+5.5%p, 절대 0.046) · MIC (+1.7%p) 둘이고, APS (+4.1%p 지만 절대 0.0001) · Is-this (+0.3%p) 는 아주 작음.
  Amazon (+4.6%p) 은 FM 도 GBDT 도 F2 가 F0 보다 나쁜데 GBDT 가 더 나빠진 것. 같은 split 에서 피처만 바꾼 짝지은 비교라 split 사이 흔들림 (kumo 의 split 표준편차: Amazon 0.0064 · QSAR-TID 0.047 · APS 0.0015 · Is-this 0.021) 보다는 노이즈가 작겠지만, 1회라 크기는 재지 못함.
  나머지 7개는 GBDT 가 더 좋아짐. MLE-bench 의 C1 (3대회 중 nomad × kumo 하나만 양수) 과 같은 그림: **에이전트 피처는 대개 GBDT 를 더 돕는다** (첫 판과 같은 결론)
- GBDT 쪽이 더 좋아진 곳의 꼴: polish (회계 항등식 위반) · physiochemical (비율) · students (비율) 처럼 **열 사이의 산술 관계** 를 만든 피처. 트리는 열 하나씩 나누므로 이런 관계를 스스로 못 만들고, FM 은 F0 에서 이미 어느 정도 잡음 (polish 의 FM F0 가 GBDT 의 1/10).
  여기에 in_vehicle (응답자 키 `person` · 설문 묶음과 그 교차 범주: catboost 0.1590 → 0.1370, kumo 0.1371 → 0.1327) · hazelnut (백색화: GBDT −18%, kumo +44%) 가 더해짐.
  FM 이 더 좋아진 곳은 QSAR-TID (지문 비트 빈도 요약 + PCA 128: kumo · tabicl 둘 다 좋아지고 lgbm · catboost 는 둘 다 나빠짐) · MIC (결측 지시)
- **FM 의 F2 가 F0 보다 나쁜 곳이 5개** (QSAR_fish · students · physiochemical · Amazon · hazelnut): Amazon 말고는 F0 에서 FM 이 Agent 를 이긴 곳이고, 열을 더하면 컨텍스트 안의 거리·주의가 흐려지는 것으로 봄 (나눠 재지 않음).
  hazelnut 은 원본 30열을 선형 변환한 30열을 더한 것 (정보는 같음) 인데 kumo 가 크게 나빠짐 (0.0039 → 0.0056). tabicl 은 12개 중 8개에서 나빠짐 (좋아진 것은 splice · polish · QSAR-TID · MIC)
- 해석의 최소 조건 "GBDT 의 F2 가 F0 보다 나은가": 최고끼리 12개 중 10개 예 (QSAR-TID · Amazon 은 아니오). 모델별로는 catboost 9 / 12 (QSAR-TID · APS · Amazon 에서 나빠짐), lgbm 8 / 12 (splice · students · QSAR-TID · MIC 에서 나빠짐)
- **Agent 와 비교**: FM + 에이전트 피처 한 번이 Agent 1시간보다 나은 곳은 FM 이 F0 로 이미 이긴 9개 + **splice** (kumo~f2 0.0601 < Codex 0.0618, 옮긴 것은 트리 성분 입력뿐인데도). MIC 는 F0 로도 이겼고 F2 로 더 벌어짐 (0.4280 대 0.4445).
  Amazon 은 더 못 미침 (0.1299 대 0.1068): 카드대로 교차 열을 빼고 빈도·고유 수를 train 으로만 fit 하니 F0 보다도 나쁨 (첫 판 0.1147. 두 변경을 나눠 재지 않음).
  in_vehicle 은 응답자 키·설문 묶음을 넣어도 kumo 가 거의 그대로 (0.1318 → 0.1327, Codex 0.1315 와의 차 0.0012 = kumo 의 split 표준편차 0.0040 의 0.3배). 늘어난 몫은 GBDT 가 가져감
- 12개의 r0f0 최고 (README 점수표, C1 을 넣어 다시 따짐): C2 를 빼면 FM + 에이전트 피처 6 (MIC · Is-this · splice · polish · QSAR-TID · APS), FM 4 (QSAR_fish · students · physiochemical · hazelnut), Agent 2 (Amazon · in_vehicle), GBDT 0 (첫 판: C1 5 · FM 5 · Agent 2. MIC 가 FM → C1).
  C2 (12개, 2026-11-19) 까지 넣으면 C2 5 (polish · hazelnut · QSAR-TID · Amazon · physiochemical) · C1 4 (MIC · Is-this · splice · APS) · FM 2 (QSAR_fish · students) · Agent 1 (in_vehicle) (C2 가 7개였을 때: C1 5 · FM 3 · C2 2 · Agent 2)
- 주의: r0f0 한 split · 1회. 차이가 1σ 안인 곳 (Is-this · APS · in_vehicle · QSAR_fish · students) 은 방향만 봄. 이번 판의 F2 는 모두 train 행으로만 fit (inductive 안). 비교하는 Agent 점수는 에이전트가 train+test 로 fit 한 그대로

### 첫 판에서 바꾼 것 (2026-11-18)

첫 판 (PR #35 의 처음 커밋, 위 표의 "첫 판 C1") 은 "C1 준비" 절의 카드 전에 데이터셋마다 판단해 옮겼음. 사용자가 카드 · R3 해석을 기준으로 정해 (2026-11-18) 8개의 `f2.py` 를 고쳐 다시 돌림. 4개 (APS · Is-this · physiochemical · QSAR_fish) 는 그대로라 다시 안 돌림.

| 데이터셋 | 열 옛 → 새 | 바꾼 것 | FM 최고 F2 옛 → 새 | GBDT 최고 F2 옛 → 새 | C1 (%p) 옛 → 새 |
|---|---|---|---|---|---|
| splice | 1,165 → 1,121 | n-gram 어휘를 train 으로만 fit (test 에만 있는 44개 빠짐) | 0.0594 → 0.0601 | 0.0752 → 0.0756 | −8.7 → −9.3 |
| students | 60 → 60 | 범주 빈도 17열을 train 으로만 fit | 0.5264 → 0.5271 | 0.5550 → 0.5565 | −2.9 → −2.8 |
| polish | 533 → 512 | equals 쌍 고르기를 train 행으로만 (112쌍 → 100쌍) | 0.00151 → 0.00132 | 0.00503 → 0.00520 | −29.6 → −23.6 |
| QSAR-TID-11 | 1,027 → 1,155 | 비트 빈도를 train 으로만 fit, PCA 128 (train 으로만 fit) 을 더함 | 0.7752 → 0.7718 | 0.8535 → 0.8765 | +2.4 → +5.5 |
| Amazon | 190 → 126 | 쌍 36 · 삼중 28 의 교차 열을 뺌, 빈도 45 · 고유 수 72 를 train 으로만 fit | 0.1147 → 0.1299 | 0.1135 → 0.1268 | +5.4 → +4.6 |
| in_vehicle | 55 → 75 | `person` 키 · `person_demo` 와 그 교차를 다시 넣음, 설문 묶음 6열을 train 으로만 fit 해 넣음 | 0.1318 → 0.1327 | 0.1527 → 0.1370 | −0.1 → −10.6 |
| MIC | – → 128 | 결측 지시 17열 (첫 판은 "옮길 피처 없음") | – → 0.4280 | – → 0.4652 | – → +1.7 |
| hazelnut | – → 60 | 출처를 Codex → AIDE × GPT (R3(b)), 백색화 30열 (첫 판은 "옮길 피처 없음") | – → 0.00561 | – → 0.02159 | – → −61.6 |

- 결론 ("에이전트 피처는 대개 GBDT 를 더 돕는다") 은 그대로. 크게 바뀐 곳은 Amazon (FM·GBDT 모두 F2 가 F0 보다 나빠짐. 교차 열 빼기와 train fit 중 어느 쪽 몫인지는 안 나눔) 과 in_vehicle (응답자 키·설문 묶음을 넣으니 GBDT 가 크게 좋아짐, C1 −0.1 → −10.6)
- **in_vehicle 의 설문 묶음은 train 으로만 fit 하면 거의 빈 블록**: train 에서 22개 시나리오가 다 있는 응답자가 2명뿐이라 묶음이 6개 → 2개, `survey_block` 이 −1 인 행이 train 60% (5,095 / 8,456) · test 61%, `survey_template` 이 'amb' 인 행도 60%. 그래도 규칙대로 지금 `f2.py` 그대로 씀 (2026-11-18 사용자)
- train+test 판의 동일성 확인은 "옮긴 피처" 표의 "동일성 확인" 열: 8개 모두 에이전트 값 (배열·성분 CV·열 수) 과 맞은 뒤 train 판으로 바꿈. `f2.py` 의 계산은 고칠 데가 없었음
- 카드와 코드가 어긋난 곳 (hazelnut 원본 30열 · MIC `features="all"` · QSAR-TID 의 LGB 는 extra 없음 · polish 의 합친 열 231 · 143) 은 #34 머지 전에 카드를 고침
- 결과표: `results/*.csv` 는 `fab sync` 사본이라 같은 (대회, 모델) 의 첫 판 `~f2` 행 (2026-11-18T05:50~07:00 UTC) 도 남아 있음. `latest_grades` 가 (대회, 모델) 마다 마지막 행을 쓰므로 위 표는 새 행 (14:16~14:40 UTC)

### 실행

| 항목 | 내용 |
|---|---|
| 명령 | `uv run --frozen fab ta fm <데이터셋…> --models kumo_cat_refit~f2 tabicl_kv_refit~f2 lgbm_refit~f2 catboost_refit~f2 --splits r0f0 --note c1`. 첫 판은 3개 → 7개 두 번, 다시 돌린 것은 8개 한 번 (동시 GPU 상한 기본 5) |
| GPU | 1×H100 작업 (첫 판 10개 + 다시 8개), 작업마다 2~4분 (MIC 6분. 모델 시간 kumo 46~78초 · tabicl 6~37초) |
| CPU | serverless 작업 (첫 판 10개 + 다시 8개). catboost 가 가장 긺 (MIC 29분 · splice 약 6분, 나머지 2분 아래), lgbm 26초 아래 |
| 코드 | `fab ta fm`·`fab ta grade` 의 `check_models` 가 `~` 를 모두 거부하던 것을, 고른 대회 모두에 `features/<변형>.py` 가 있으면 받게 고침 (`fab/bench/tabarena/__init__.py`. 생성 split 대회 `tas-*` 는 그대로 거부) |
| 결과 | `results/tabarena.csv` 의 `~f2` 72행 (note `c1`, 첫 판 40 + 다시 32), `results/results.csv` 의 실행 72행 (`n_features`) |

## 행 추가: kddcup09_appetency (r0f0, 2026-11-19 · 11-20)

README 점수표에 데이터셋을 하나씩 더하기로 하고 (Agent 를 돌린 12개 → 13개), 첫 행으로 `kddcup09_appetency` 를 돌림.
고른 까닭: 12개에서 Agent 가 FM 을 이긴 까닭으로 가장 단단히 남은 단서가 고카디널리티 범주 ("범주 대 오염" 에서 바꿔도 남은 것은 Amazon 뿐).
kddcup09 는 51개 중 고카디널리티 범주 열이 가장 많고 (범주 38열 중 고유값 > 100 인 열 11개, 최대 15,415), 우리 kumo 가 TabArena 기본 GBDT 보다 나쁜 셋 중 하나 (+1.7%, 0.3σ, `results/tabarena_cats.csv`).
변수 이름이 원래 `Var1`.. 로 익명이라 열 뜻으로는 알아보기 어려움 → "고카디널리티 범주에서 Agent 가 FM 을 이기나" 를 이 한 행으로 봄.

순서는 지금까지 12개와 같음: 대회 준비·GBDT refit·Agent 5설정 × 1회 = "Agent 확장 1회째", C1 = "C1 준비"·"C1" 의 규칙, C2 = "C2 (+fm)" 의 2설정·안내문 그대로. FM 은 51개 모든 split 결과가 이미 있어 새로 안 돌림.

### 대회 준비

- `competitions/ta-kddcup09-appetency/` (`config.yaml`·`features.py`·`description.md`). `config.yaml` 은 `default_config` (FM 이 돈 `tas-kddcup09-appetency-r0f0` 과 같은 범주 38열·지표 `auc`·`submit_proba`) + `anon_id`
- **익명 id 는 `tabarena-p16`**: `p13` 은 `ta-splice-obf` (#32), `p14`·`p15` 는 다른 바꾼 대회가 씀. 실행 키가 익명 id 로 시작해서 겹치면 실행이 섞임
- **타깃 열 이름만 바꿈 (`appetency` → `BoughtNewProduct`, 2026-11-19 사용자가 정함)**: 원래 이름이 데이터셋을 바로 알려 주는 말 (description 에 안 써도 CSV 머리에 보임). 값·행·id 는 그대로.
  설정에 `tabarena.target_from: appetency` (OpenML 타깃 이름) 를 적으면 `prepare` 가 열 이름만 `target` 으로 바꿈 (`fab/bench/tabarena/__init__.py`). `target_from` 이 OpenML 타깃과 다르거나, 없는데 `target` 이 데이터의 타깃과 다르면 (오타 등) 멈춤. 새 이름이 피처·`id` 와 겹쳐도 멈춤.
  생성 split (`tas-*`) 은 `split_comp` 가 `target_from` 을 떼고 원래 이름으로 둠: Volume 에는 없는 대회만 올리므로 (`upload_data`) FM 때 올린 데이터를 그대로 씀. `ensure_prepared` 는 `data/train.csv` 의 마지막 열이 설정의 `target` 과 다르면 멈춤 (이름을 바꾼 뒤 옛 data/ 가 남은 경우)
- `description.md` (영어): "a large company 의 고객이 새 상품·서비스를 샀는지". 대회 이름 · 연도 · 통신사 · "appetency" 를 안 씀. 열은 "anonymized numeric 174 (`Var1`..`Var190`, 번호가 빔) · categorical 38 (`Var191`..`Var229`, 섞은 글자, 몇 열은 값이 수천 가지)" 와 결측 · 드문 값 · test 에만 있는 값만 적음
- `fab ta prepare kddcup09_appetency --splits all`: 9 split (r0f0 train 33,333 / test 16,667, 양성 1.8%)

로컬 확인 (학습 없이, 저장소 밖 스크립트, "Agent 확장 1회째" 의 "대회 준비" 와 같은 검사): 9 split 모두 크기, train·test id 가 0..n−1 을 나눔, 열 순서, train·test·정답이 OpenML 행과 같음, repeat 마다 fold 3개 test 합 = 전체, 범주 열이 설정과 같고 `1`·`1.0` 섞임 없음, 채점 (상수 0.5 · 정답 1, r0f0 은 뒤집기 0 · 행 섞기 · 레이블 제출 무효 · 빠진 행 무효) → **107개 모두 통과**.
r0f0 정답은 FM 이 쓴 `tas-…-r0f0` 정답과 머리 한 줄 말고 같음. id–타깃 상관 r = −0.004 (z −0.75).
`leak_check`: 설명 · 데이터 파일 이름 · 익명 id · 실행 이름 · 코드 묶음 7개 통과. 더 찾아본 말 (appetency · kdd · cup · orange · telecom · churn · upsell · 2009 · competition · kaggle · openml · uci, 다른 데이터셋 50개 이름) 은 설명 · CSV 머리 · 코드 묶음에 없음

### GBDT refit

- 9 split × `lgbm_refit`·`catboost_refit` 을 serverless CPU 작업 하나로 (`fab ta fm kddcup09_appetency --models lgbm_refit catboost_refit --splits all --note "kddcup09 GBDT refit"`, 2026-11-19 21:52 → 22:35 KST). 나눌 까닭 없음: TabArena CatBoost 가 split 중앙값 169초, 우리 CatBoost 238초 (r0f0), lgbm 10초
- 9 split 평균 (1 − AUC): catboost 0.1582 ± 0.0093, lgbm 0.2275 ± 0.0106. TabArena 기본 CatBoost 0.1543 · LightGBM 0.2024 → catboost 는 표준편차 안, lgbm 은 기본 설정이 이 데이터에 약함 (TabArena LightGBM 보다도 나쁨)

### Agent (2026-11-19 21:54 ~ 11-20 00:05 KST)

- 5설정 × 1회 (r0f0). 같은 엔드포인트는 한 번에 하나: Sonnet 줄 `aide-sonnet` → `claude-code-sonnet` → `claude-code-sonnet --continue`, GPT 줄 `codex-gpt` → `aide-gpt` 를 줄마다 차례로, 두 줄은 동시에 (저장소 밖 스크립트가 앞 작업이 끝나면 다음을 냄, 끝나면 `fab remote kick`)
- 5회 모두 정상 종료·채점 (`valid_submission` 참, `run_crashed` 없음, 다시 돌린 것·뺀 것 없음)

숫자는 `metric_error` (1 − AUC, 낮을수록 좋음). FM·GBDT 는 refit 한 번, 줄 순서는 점수 순, **굵게** = 최고. TabArena 참고 줄은 공개 결과의 같은 split (r0f0). `tabpfn_v2` 는 TabArena 범위 밖 (`can_run_tabpfnv2` 거짓).

#### kddcup09_appetency (`tabarena-p16`): 이진 · 피처 212 (범주 38) · 오염 높음 (대회, 알아봄)

| 방식 | 1회째 (r0f0) | 시간 | 종료·이어 가기 | 비고 |
|---|---|---|---|---|
| TabArena Kumo-Tabular (참고) | 0.1474 | | | AutoGluon 전처리 뒤 ("FM 이 TabArena 와 맞는지" 의 "kddcup09") |
| **Agent: AIDE × Sonnet** | **0.1559** | 60분 | 시간 종료 | 첫 계획 글에 "KDD Cup 2009 upselling task" |
| Agent: Codex × GPT | 0.1569 | 58분 | 스스로 끝냄 | 429 83번 · 생각 글에 "orange appetency" |
| TabArena CatBoost (참고) | 0.1572 | | | |
| Agent: Claude Code × Sonnet +cont | 0.1587 | 11분 | 스스로 끝냄 (이어 가기 9번) | |
| FM: kumo | 0.1594 | 65초 | | |
| Agent: AIDE × GPT | 0.1599 | 60분 | 시간 종료 | |
| GBDT: catboost | 0.1602 | 238초 | | |
| Agent: Claude Code × Sonnet | 0.1603 | 32초 | 스스로 끝냄 | |
| FM: tabicl | 0.1846 | 18초 | | |
| GBDT: lgbm | 0.2288 | 10초 | | |
| FM: tabpfn_v2 | – | | | 범위 밖 |

- **Agent 최고 (AIDE × Sonnet) 가 FM (kumo) 보다 조금 나음**: 0.1559 대 0.1594, Agent − FM −2.2% (−0.38σ, σ = kumo 의 split 표준편차 0.0090) → 1σ 안. FM 보다 나은 Agent 3 / 5 (AIDE × Sonnet · Codex · Claude Code +cont), 나머지 둘도 kumo 와 0.1σ 안
- 우리 kumo · catboost · Agent 다섯이 0.156~0.160 안에 몰림 (폭 0.0044 = σ 의 0.5배). tabicl (0.1846) · lgbm (0.2288) 만 뚜렷이 뒤짐
- 우리 kumo 는 TabArena Kumo (r0f0 0.1474) 보다 0.012 나쁨 (51개 중 우리 kumo 가 TabArena Kumo 와 표준편차 밖으로 다른 하나뿐인 데이터, AutoGluon 전처리 차이). TabArena Kumo 로 견주면 Agent 다섯 모두 FM 보다 나쁨
- 풀이 (최고만 읽음, 아래 C1 카드): AIDE × Sonnet 은 LGB · XGB · CatBoost 4개의 OOF 순위 평균 (0.25 · 0.25 · 0.5). 범주는 LGB · XGB 에 값 빈도 + OOF 타깃 인코딩, CatBoost 에 원본 범주 + 빈도

사후 검사 (5회)

- `data_source_urls` 0 · `kaggle_urls` 0. 429 는 Codex 83번 (기준 200번 안), 나머지 0. `pip_added` · `tfm_runtime` 없음 (공식 설정에서 TFM 을 깐 실행 없음), `llm_errors` 0
- 입력 토큰: Codex 4,596만, Claude Code +cont 88만, AIDE × GPT 33만, AIDE × Sonnet 28만, Claude Code 9만

알아본 흔적 ("범주 대 오염" 의 "알아본 흔적" 과 같은 방식: 하네스 기록과 작업 디렉터리 글 파일, pip 가 깐 파일 · `site-packages` 뺌. 센 말: appetency · KDD · Orange · telecom · churn · upsell · Cup · 2009 · CRM · UCI · Kaggle · OpenML · famous/well-known/classic dataset)

| 실행 | 하네스 기록 | 작업 디렉터리 | 무엇을 알아봤나 |
|---|---|---|---|
| Codex × GPT | appetency 1 · Orange 3 · churn 2 | 0 | `codex.jsonl` 8번째 줄 (시작 몇 분 안) 생각 글 "potential orange churn", 11번째 "orange appetency of 1.8% and an expected AUC of around 0.8", 94번째 "AUC close to 0.85 in the Orange dataset" |
| AIDE × Sonnet | – | KDD 44 · upsell 24 · Cup 24 · 2009 39 (AIDE 기록 `logs/`) | 첫 노드 (step 0) 계획 글 "This looks like the KDD Cup 2009 upselling task" (부과제는 틀림), step 7 "the KDD 2009 data", step 17 "KDD-style data" |
| 나머지 3회 | 0 | 0 | AIDE × GPT · Claude Code 둘 (`Kaggle`·`competition` 은 지시문 · AIDE 시스템 글에서만 나옴) |

- 둘 다 **데이터 모양으로** 알아봄 (이름·출처는 안 줌, 타깃 이름도 바꿈, 원본 접근 0): `Var1`..`Var230` 꼴 열 이름 · 수치 174 / 범주 38 · 범주 값의 섞은 글자 · 양성 1.8%
- 알아본 둘이 Agent 1·2등. 읽은 AIDE × Sonnet 의 풀이 (아래 C1 카드) 는 흔한 꼴 (빈도 · 타깃 인코딩 + GBDT) 이고 외부 정보를 쓴 흔적은 없음 (Codex 의 최종 풀이는 안 읽음)

### C1

"C1 준비" 의 고르는 규칙: R2 Agent 최고 = AIDE × Sonnet (`tabarena-p16__aide-sonnet__r1__20261119-215413`, 0.1559, Agent − FM −0.38σ), R4 우리 GBDT (catboost 0.1602) 보다 나음, R3 해당 없음.
카드 [competitions/ta-kddcup09-appetency/c1.md](../competitions/ta-kddcup09-appetency/c1.md) (계보 도구 + 코드 읽기, 2026-11-20).

| 데이터셋 | 출처 Agent | 옮긴 블록 | 열 | 뺀 것 (이유) | train+test fit | 동일성 확인 |
|---|---|---|---|---|---|---|
| kddcup09 | AIDE × Sonnet 0.1559 | 범주 38열의 값 빈도 (결측 → `"NA"` 를 한 값으로) | 212 → 250 | 범주 38열의 OOF 타깃 인코딩 (규칙 3, LGB · XGB 의 주 범주 입력), 그 키인 범주 코드 (모델 입력 아님), 세 모델 · seed 평균 · 순위 섞기 무게 | 빈도 → **train 으로 fit** (test 에만 있는 값 0, test 칸의 2.6%) | train+test 판 빈도 38열이 에이전트 식 (`best_solution.py:18-20`) 과 배열이 같음. 그 열로 에이전트 CatBoost 성분 (4개 · 같은 fold · seed, 우리 코드) 을 다시 돌려 OOF AUC 0.84421 대 기록 0.84406 (fold 차 최대 0.0017, Mac 대 Linux 노드, catboost 1.2.10 같음). train 판: train 칸의 96% · test 칸 모두 바뀜, 같은 CatBoost OOF 0.84319 |

결과 (r0f0 test `metric_error`, `uv run --frozen fab ta fm kddcup09_appetency --models kumo_cat_refit~f2 tabicl_kv_refit~f2 lgbm_refit~f2 catboost_refit~f2 --splits r0f0 --note c1`, 2026-11-20 09:23 KST, GPU 작업 6분 · CPU 작업 catboost 238초)

| 데이터셋 | kumo F0 → F2 | tabicl F0 → F2 | lgbm F0 → F2 | catboost F0 → F2 | Agent 최고 |
|---|---|---|---|---|---|
| kddcup09 | 0.1594 → 0.1847 (+15.9%) | 0.1846 → 0.1735 (−6.0%) | 0.2288 → 0.2134 (−6.7%) | 0.1602 → 0.1624 (+1.4%) | **0.1559** |

| 데이터셋 | FM 최고 F0 → F2 | GBDT 최고 F0 → F2 | C1 (%p) | C1 (절대) | FM F2 − Agent |
|---|---|---|---|---|---|
| kddcup09 | +8.9% (+0.0141, kumo → tabicl) | +1.4% (+0.0022, catboost) | −7.5 (둘 다 나빠짐) | −0.0119 | +11.3% (못 미침) |

- **에이전트 피처는 이 데이터에서 FM 을 돕지 않음**: kumo 가 크게 나빠짐 (+0.0253 = kumo split 표준편차의 2.8배, 같은 설정 `cats=categorical chunk=8192 n_estimators=8`). 빈도 38열은 범주 38열을 수치로 다시 쓴 것이고, kumo 는 값이 50개 이상인 범주 열에 빈도 열을 스스로 만듦 → 겹친 정보를 더한 것. hazelnut (같은 정보의 선형 변환 30열을 더하자 kumo +44%) 과 같은 꼴로 봄 (나눠 재지 않음)
- tabicl 은 좋아짐 (−6.0%) 이지만 kumo F0 보다는 여전히 나쁨. GBDT 쪽은 lgbm 만 좋아지고 catboost 는 조금 나빠짐: 에이전트 풀이에서 빈도와 짝이던 타깃 인코딩이 빠져 (규칙 3) GBDT 가 받는 몫은 빈도뿐
- C1 칸 (README) = F2 의 FM 최고 = tabicl~f2 0.1735 (AUC 0.8265): F0 kumo · Agent · GBDT 모두보다 나쁨. 13개 중 C1 이 FM F0 보다 나쁜 여섯째 (QSAR_fish · students · physiochemical · Amazon · hazelnut 에 더해), FM F2 가 Agent 에 못 미친 셋째 (Amazon · in_vehicle 에 더해)

### C2 (+fm)

"C2 (+fm)" 와 같은 2설정 · 1회 · r0f0 · `fm_notes.txt` 그대로 (`uv run --frozen fab ta agent run ta-kddcup09-appetency <설정> --fm`, 2026-11-20 14:12 → 15:17 KST, 두 설정 동시).
σ = kumo 의 split 표준편차 0.0090. 굵게 = kumo 보다 나음.

| 데이터셋 | kumo (ours) | σ | 설정 | Agent | C2 (+fm) | C2 − Agent (÷σ) | C2 − kumo (÷σ) | 제출에 든 FM (최종 코드) |
|---|---|---|---|---|---|---|---|---|
| kddcup09 (`p16`) | 0.1594 | 0.0090 | AIDE × Sonnet | 0.1559 | **0.1528** | −0.35 | −0.73 | TabICL 50% (상위 30 · 60 · 120열 셋) · Kumo 17% · LGB · XGB 33% 의 같은 무게 순위 평균 |
| | | | Codex × GPT | 0.1569 | **0.1504** | −0.72 | −0.99 | **Kumo 60%** (z 점수 logit 섞기, Kumo 변형 여럿, `assemble_stage10.py`) · `cat` 묶음 24% · `te` 묶음 16% (이름으로 CatBoost · 타깃 인코딩 GBDT 로 봄, 그 스크립트는 안 읽음) |

- 두 실행 모두 같은 설정의 Agent · 우리 kumo · Agent 최고보다 나음 → 이 행의 r0f0 최고는 C2 (Codex × GPT +fm, AUC 0.8496). 둘 다 1σ 안 (Codex −0.99σ)
- **Kumo 에 범주를 넘긴 방식: 둘 다 `categorical` stype 을 안 씀** (안내문의 "수치만" 예 그대로).
  Codex 는 범주 열을 그대로 넣지 않고 **수치 열 + 범주 값 빈도 (train+test) + 값이 30개 아래인 범주의 정수 코드 + 값이 10개 이상인 범주 · 그 앞 3·4글자 묶음의 OOF 타깃 인코딩** 중 LGB 중요도 상위 65열을 모두 `numerical` 로 (`kumofull*.py`·`jointkumofull1.py`), 컨텍스트는 양성 전부 + 음성 1천~1만 행 표본.
  AIDE 는 수치 + 빈도 (train+test) + 행별 결측 수 + OOF 타깃 인코딩 중 LGB 중요도 상위 60열을 `numerical` 로 (TabICL 도 같은 표에서 30 · 60 · 120열), 컨텍스트는 train 전체
- 그래서 C2 의 FM 은 **타깃 인코딩한 범주를 받은 FM**: 범주를 범주로 넘긴 우리 kumo (0.1594) · 빈도만 더한 C1 (`~f2`, kumo 0.1847 · tabicl 0.1735) 보다 나음. 타깃 인코딩은 C1 규칙 3 으로 뺀 블록이라 C1 과 C2 가 갈린 몫이 여기에 있을 수 있음 (나눠 재지 않음)
- 사후 검사: 2회 모두 `valid_submission` 참, `run_crashed` 없음, `data_source_urls` 0 · `kaggle_urls` 0, `pip_added` 없음, `llm_errors` 0. 429 는 Codex 15번. `tfm_runtime` Codex kumo · tabicl · tabpfn, AIDE kumo · tabicl
- test 정답 · 의사 레이블은 없음. 빈도는 둘 다 train+test 로 셈 (라벨 없음, transductive), 타깃 인코딩은 train 정답으로만
- 알아봄: 둘 다 다시 알아봄. Codex 생각 글 "likely from a KDD dataset containing 50,000 entries" (`codex.jsonl` 7번째 줄), "Orange's upselling … typical AUC around 0.85" (50번째), AIDE 노드 계획 글 "This is the KDD Cup 2009 upselling task". 원본 URL 접근 0

### 해석 (r0f0 한 split, 실행 1회. 판단은 약함)

- **고카디널리티 범주에서 Agent 가 FM 을 이기나**: 방향은 예, 크기는 작음. Agent 최고 0.1559 대 kumo 0.1594 (−0.38σ, 1σ 안), FM 보다 나은 Agent 3 / 5. Amazon (−3.3σ, 5 / 5) 만큼 뚜렷하지 않음. 비교하는 kumo 는 TabArena Kumo 보다 0.012 나쁜 우리 kumo 이고, TabArena Kumo (0.1474) 로 견주면 Agent 다섯이 모두 짐
- **범주와 오염은 여기서도 갈리지 않음**: Agent 1·2등 (AIDE × Sonnet · Codex) 과 C2 둘이 모두 데이터 모양으로 데이터셋을 알아봄 (이름 · 출처 · 타깃 이름을 숨겨도). 13개 중 Agent 가 FM 을 이긴 넷 (splice · Amazon · in_vehicle · kddcup09) 이 모두 범주가 많고 오염 위험도 큼 ("Agent 확장 1회째" 의 경향과 같음)
- **이 데이터에서 Agent 의 몫은 범주의 타깃 인코딩으로 보임**: 읽은 풀이 셋 (Agent 최고 AIDE × Sonnet, C2 둘. 나머지 Agent 넷은 안 읽음) 이 모두 빈도 + OOF 타깃 인코딩 + GBDT 꼴. 타깃 인코딩을 뺀 C1 (빈도만) 은 FM 을 돕지 못하고 (kumo +15.9%), 타깃 인코딩을 FM 에 넣은 C2 는 FM 을 넘음. 우리 kumo 가 범주를 범주로 받아도 (Kumo 기본 전처리) 이 몫을 못 따라옴 → "FM 이 고카디널리티 범주에서 약한" 꼴이 kddcup09 에도 있음 (Amazon 과 같은 방향, 크기는 작음)
- 다만 r0f0 한 split · 1회씩이고, 1등 (C2 Codex) 과 FM 의 차도 1σ 안. 2회째나 다른 split 은 안 돌림

### 비용

- Agent: A10 약 3.6시간 (작업 시간 합 216분 이하: 하네스 합 3.2시간 + 실행마다 설치·회수. 줄마다 끝을 5분 간격으로 봐서 위로 어림). 어림 (약 6시간) 안
- GBDT: serverless CPU 작업 하나 43분 (9 split × lgbm · catboost). C1 `~f2`: 1×H100 작업 6분 + CPU 작업 하나 (catboost 238초)
- C2: A10 약 2.2시간 (작업 둘이 각각 65분 이하, 하네스 59분 · 60분 + TFM 설치 · 사전 점검 · 회수)

### 실행 메모

| 일 | 내용 | 처리 |
|---|---|---|
| 익명 id 겹침 | 지시문의 `tabarena-p13` 이 이미 `ta-splice-obf` 의 id | `p16` 으로 (위 "대회 준비") |
| 타깃 이름 | `appetency` 가 CSV 머리에 그대로 보임 | r0f0 에서만 `BoughtNewProduct` 로 (사용자). 생성 split 은 Volume 의 FM 데이터와 같게 원래 이름 |
| 결과표 사본 | `fab sync` 가 받은 `results/mlebench_runs.csv` 는 MLE-bench 행의 줄만 바뀜 (값 같음, #29 와 같은 꼴) | 이 PR 과 상관없어 되돌림 |

## Elo (2026-11-16)

점수를 TabArena 처럼 한 숫자로: 우리 FM·Agent 를 TabArena 공개 결과 (기본 설정 7개, 3) 와 한 풀에 넣고 **RandomForest (기본) = 1000** 인 Elo.
`uv run --frozen fab ta elo` → `results/tabarena_elo.csv` (`fab/bench/tabarena/elo.py`). 결과표가 바뀌면 (`fab sync`, `fab ta reference`) 다시 돌림.

계산은 TabArena (`autogluon/tabarena` c9f57b2, bencheval 의 `EloHelper`·`BenchmarkEvaluator.compute_elo`, 논문 D.1) 와 같음

| 항목 | 값 |
|---|---|
| 모형 | Bradley-Terry 최대우도 (로지스틱 회귀, C = 1e6). Elo 400 차이 = 기대 승률 91% (1 / (1 + 10^(−Δ/400))) |
| 짝 비교 | 같은 (데이터셋, split) 안에서 `metric_error` 가 낮으면 이김, 같으면 반씩. 가중치 1 / (그 데이터셋의 split 수) → 30 split 데이터셋과 9 split 데이터셋이 같은 무게 |
| 빈칸 | 못 돈 (방법, 데이터셋, split) 은 RandomForest 기본 결과로 채움 (TabArena 의 fillna). Agent 의 실패한 회차 (무효 제출·점수 없음) 도 빼지 않고 회차 평균 전에 RandomForest 값으로. `imputed` = 채운 비율 (Agent 는 실패 회차 비율 포함) |
| 95% 구간 | 데이터셋을 복원 추출하는 부트스트랩 200번 (seed 0), 보정 전 값의 2.5%·97.5% 분위수 − 점추정. 점추정은 전체 데이터의 최대우도 (부트스트랩 중앙값 아님). 마지막에 RandomForest 를 1000 으로 옮김 (`post_calibrate`, 구간 폭은 RandomForest 와 무관) |
| `rank`·`winrate` | 단위 (데이터셋, split) 마다 순위·승률 (비김 반) 을 같은 가중치로 평균 |
| 확인 | 같은 입력 (두 풀) 을 TabArena 저장소의 `elo_utils.py` 로 계산 → 점추정 같음 (소수 첫째 자리까지), FM 풀의 구간도 같음 (2026-11-16) |

- 풀 (참가자 묶음) 이 다르면 Elo 도 다름. TabArena 리더보드는 방법이 훨씬 많아 (모델 58개 + 시스템, 튜닝·앙상블 줄 포함) 같은 방법도 숫자가 다를 수 있음 → 이 표끼리만 비교
- 리더보드 화면의 Elo 는 여전히 못 읽음 ("확인 못 한 것")

### FM 풀: 51개 × 모든 split (816)

우리 FM 은 refit (결정 11). `kumo (ours)` = 범주 열이 있는 34개는 `kumo_cat_refit`, 없는 17개는 같은 설정인 `kumo_c8k_refit` ("FM 프로토콜과 이름").
우리 GBDT (`lgbm_refit`·`catboost_refit`) 는 12개 데이터셋뿐이라 이 풀에 안 넣음 (넣으면 39개가 RandomForest 값).

| 방법 | Elo | 95% 구간 | 평균 순위 | 승률 | 채운 비율 |
|---|---|---|---|---|---|
| TabArena Kumo-Tabular | 1871 | +138 / −78 | 1.85 | 0.906 | 0 |
| **kumo (ours)** | **1829** | +134 / −77 | 2.07 | 0.881 | 0 |
| TabArena TabICLv2 | 1538 | +53 / −44 | 4.10 | 0.656 | 0 |
| **tabicl (ours)** | **1535** | +49 / −48 | 4.12 | 0.653 | 0 |
| TabArena CatBoost | 1333 | +44 / −48 | 6.02 | 0.443 | 0 |
| TabArena TabICL (v1) | 1276 | +47 / −61 | 6.59 | 0.379 | 0.294 (범위 밖 15개) |
| **tabpfn_v2 (ours)** | **1269** | +64 / −96 | 6.65 | 0.372 | 0.353 (범위 밖 18개) |
| TabArena TabPFNv2 | 1223 | +62 / −84 | 7.11 | 0.321 | 0.353 (범위 밖 18개) |
| TabArena LightGBM | 1175 | +44 / −54 | 7.56 | 0.271 | 0 |
| TabArena RandomForest | 1000 | +48 / −66 | 8.94 | 0.118 | 0 |

- 우리 FM 셋이 모두 TabArena 의 같은 모델 바로 옆 (구간 안): kumo −43, tabicl −3, tabpfn_v2 +47 (반올림 전 값의 차). "FM 이 TabArena 와 맞는지" 의 판정과 같은 그림
- kumo (ours) 는 TabArena CatBoost 기본보다 +495 (기대 승률 95%), tabicl (ours) 보다 +294 (84%)
- tabpfn_v2 의 18개는 TabArena TabPFNv2 와 같은 데이터셋을 RandomForest 로 채움 (TabArena 와 같은 처리) → 두 줄은 같은 조건. 돌 수 있는 33개만의 Elo 는 따로 안 냄

### Agent 풀: Agent 를 돌린 데이터셋의 r0f0 (13개, 2026-11-20 다시)

파일럿 2개 (hazelnut · in_vehicle) + #26 10개 + 행 추가 1개 (kddcup09, 2026-11-20) 의 r0f0. 위 풀의 방법 + 우리 GBDT (refit) + Agent 5설정. Agent 는 설정마다 회차 평균
(파일럿은 2회, #26 · 행 추가는 1회 → 결과표의 `runs` 평균 1.15. TabArena 가 seed 를 평균하는 것과 같게). 실패한 회차 없음 (`imputed` 0). tabpfn_v2 는 범위 밖 5개 (QSAR-TID-11 · APSFailure · Amazon · physiochemical · kddcup09) 를 RandomForest 로 채움.
+fm (C2) · `~f2` (C1) 는 넣지 않음 (전과 같음).

| 방법 | Elo | 95% 구간 | 평균 순위 | 승률 |
|---|---|---|---|---|
| TabArena Kumo-Tabular | 2026 | +2349 / −136 | 1.92 | 0.942 |
| kumo (ours) | 1923 | +516 / −109 | 2.62 | 0.899 |
| **Agent: Codex × GPT** | **1697** | +210 / −247 | 4.92 | 0.755 |
| **Agent: AIDE × Sonnet** | **1598** | +143 / −159 | 6.31 | 0.668 |
| tabicl (ours) | 1588 | +143 / −160 | 6.46 | 0.659 |
| TabArena TabICLv2 | 1554 | +141 / −173 | 7.00 | 0.625 |
| **Agent: AIDE × GPT** | **1549** | +120 / −264 | 7.08 | 0.620 |
| **Agent: Claude Code × Sonnet +cont** | **1502** | +94 / −172 | 7.85 | 0.572 |
| TabArena CatBoost | 1381 | +109 / −229 | 9.92 | 0.442 |
| TabArena TabICL (v1) | 1325 | +100 / −204 | 10.88 | 0.382 |
| tabpfn_v2 (ours) | 1323 | +151 / −206 | 10.92 | 0.380 |
| **Agent: Claude Code × Sonnet** | **1314** | +77 / −223 | 11.08 | 0.370 |
| catboost (ours) | 1268 | +111 / −237 | 11.85 | 0.322 |
| TabArena TabPFNv2 | 1253 | +101 / −188 | 12.08 | 0.308 |
| TabArena LightGBM | 1194 | +64 / −189 | 13.00 | 0.250 |
| lgbm (ours) | 1139 | +100 / −246 | 13.77 | 0.202 |
| TabArena RandomForest | 1000 | +59 / −247 | 15.35 | 0.103 |

- 데이터셋 13개라 95% 구간을 냄 (10개부터). 단위가 (데이터셋, r0f0) 13개뿐이라 구간이 넓음. TabArena Kumo 의 위쪽 폭이 2,000을 넘는 것은 부트스트랩 묶음에서 Kumo 가 거의 다 이기면 최대우도가 위로 벌어지는 것으로 보임 (나눠 재지 않음, 점추정은 그대로 읽음). kumo (ours) 는 kddcup09 에서 Agent 셋에 져서 위쪽 폭이 2,204 → 516 으로 줄었음
- 순서: kumo (ours) 1923 > **Codex 1697** > **AIDE × Sonnet 1598** > tabicl (ours) 1588 > TabICLv2 1554 ≈ AIDE × GPT 1549 > Claude Code +cont 1502 > CatBoost 기본 1381 > Claude Code 공식 1314 > 우리 GBDT (1268·1139).
  Agent 넷이 기본 GBDT 보다 위, Codex 는 우리 FM 최고 (kumo) 보다 226 아래 (기대 승률 21%) · tabicl 보다 위. 구간이 겹쳐 Agent 끼리·tabicl 과의 순서는 단단하지 않음
- 12개 (2026-11-17) 에서 바뀐 것: kddcup09 에서 Agent 셋이 kumo 를 이겨 kumo 1964 → 1923, Agent 는 모두 올라감 (Codex 1680 → 1697, AIDE × Sonnet 1569 → 1598 으로 tabicl 위, AIDE × GPT 1542 → 1549, Claude Code +cont 1481 → 1502, Claude Code 1297 → 1314). FM 풀 (51개) 은 그대로
- 파일럿 2개로 낸 값 (2026-11-16, kumo 2540 · Codex 2372 · AIDE 둘 1944 · Claude Code +cont 1628 · Claude Code 1314) 보다 Agent 가 FM 에 가까움: 13개 중 Agent 가 FM 을 이긴 데이터셋이 넷 ("Agent 확장 1회째" 의 요약 + kddcup09)
- RandomForest 가 거의 다 져서 (승률 0.10) 숫자 폭이 FM 풀보다 넓음. FM 풀과 숫자를 나란히 놓지 말 것 (같은 방법도 kumo 1829 대 1923)

## 빠르게 돌리기 (2026-11-14)

#16 (2만 행 넘는 12개 × 9 split × tabicl·kumo) 과 #17 (Agent 2회째) 을 빨리 돌릴 방법을 찾음.
minipfn (TabArena-Lite 하네스) 에서 가져올 만한 것을 보고, 우리 쪽 병목을 잰 뒤, 작은 시험 (데이터셋 3개 × r0f0) 으로 전후를 잼.

### 병목 (#16 첫 실행의 로그, H100)

| 데이터셋 r0f0 | train × 피처 | test | split 하나 | tabicl | kumo | 나머지 (프로세스 2개) |
|---|---|---|---|---|---|---|
| GiveMeSomeCredit | 100,000 × 10 | 50,000 | 306초 | 107초 (fold 마다 fit 1 · 예측 20) | 188초 (fit 0 · 예측 36) | 약 11초 |
| customer_satisfaction | 86,586 × 21 | 43,294 | 277초 | 105초 | 159초 | 약 13초 |
| APSFailure | 50,667 × 170 | 25,333 | 265초 | 125초 (fit 9 · 예측 15) | 125초 (예측 23) | 약 15초 |

- 시간 대부분이 **예측**. `fab run` 은 fold 5개마다 (검증 2만 + test 5만 행) 을 컨텍스트 8만 행으로 예측 → test 를 5번 예측하고 검증 행도 예측함
- TabICL: 래퍼가 예측을 2만 행씩 나누는데 (`PREDICT_CHUNK`) KV 캐시가 꺼져 있어 **청크마다 컨텍스트 8만 행을 다시 계산** (7만 행 = 4번)
- Kumo: 컨텍스트는 이미 한 번만 인코딩 (KV 캐시). 질의 청크 2048 은 T4 에 맞춘 값
- 데이터셋 하나 (9 split) 는 H100 에서 약 40~46분. 12개 모두면 H100 5~7시간 어림 (작은 데이터셋은 더 짧음. 이슈의 10~20시간보다 짧음)
- 노드 뜨기: 시험 작업 (아래) 은 로그가 나오기까지 약 15분 (작업 32분 중 split 셋 17분)

### minipfn 에서 본 것

| 아이디어 (minipfn) | 우리에게 | 까닭 |
|---|---|---|
| CV 없이 train 전체로 한 번 fit → test 예측 (`benchmarks/arena.py`, 기본 프로토콜) | **씀** (`_refit`) | 예측이 5번 → 1번, 검증 행 예측도 없음. TabArena 의 TFM test 예측 (`refit_folds=True`) 과 같은 꼴 |
| KV 캐시 (`fit_with_cache`, `inference/rotation_ensemble.py`) · 큰 표 청크 추론 (#70) | **씀** (TabICL 패키지의 `kv_cache`, Kumo 질의 청크) | minipfn 구현은 자기 모델 전용. 같은 생각을 각 패키지의 옵션으로 |
| 데이터셋을 GPU 여럿에 LPT 로 나눔 (`--shards`, #76: 8×H100 에서 35분 → 7분) | **씀** (split 대회를 GPU 마다, 아래 "8×H100") | 실행기가 split 을 차례로 돌려 여러 장 작업에서도 GPU 를 다 못 씀 (`fab run` 은 모델마다만 GPU 를 나눔 → tabicl·kumo 면 2장) |
| bf16 autocast | 안 씀 (이미 있음) | Kumo 는 float16 autocast, TabICL 은 `use_amp="auto"` (1만 행 넘으면 켜짐, H100 은 FA3 도) |
| TF32 · `torch.compile` · fused Adam (#50) | 안 씀 | 학습 쪽 |
| 넓은 표에서 멤버마다 피처 부분집합 (#71) | 안 씀 | 모델이 바뀜 (FM 설정 그대로, 결정 4) |
| 추론 시간 측정 스크립트 (#80) | 참고만 | 저장소에 결과 숫자가 없음 |

### 넣은 것

- 모델 이름 끝 `_refit` (`fab/models` 의 `REFIT_SUFFIX`): `fab run` 이 CV 없이 train 전체로 fit 한 번 → test 만 예측. `oof.npy`·CV 점수 없음 (`results.csv` 의 `cv` 는 NaN, `n_folds` 0).
  `--fold`·`--max-folds`·`--no-test` 와는 같이 못 씀. 채점·`compare` 는 그대로 (`kumo_refit` 의 비교 대상 = Kumo-Tabular). 원격 정리 작업은 고치지 않음 (모델 이름으로만 봄)
- `tabicl_kv`: TabICL `kv_cache=True`. 캐시를 만들다 OOM 이면 캐시 없이 다시 fit
- `kumo_c8k`·`kumo_c32k`: Kumo 질의 청크 8192·32768 (기본 2048)
- 이름이 달라 지금 `tabicl`·`kumo` 결과를 덮어쓰지 않음. 예: `fab ta fm <데이터셋> --models tabicl_kv_refit kumo_c8k_refit --splits all`
- 실행기 (`fab/runner/databricks.py` 의 `RUNNER`): GPU 가 여럿이고 대회가 여럿이면 GPU 마다 작업자 하나, 남은 대회를 train 파일이 큰 것부터 하나씩
  (`CUDA_VISIBLE_DEVICES` 하나 · CPU 스레드를 작업자 수로 나눔). 대회가 끝날 때마다 Volume 에 복사 (`results.csv` 는 fab 의 표 잠금을 잡고).
  `fab run` 은 받은 `CUDA_VISIBLE_DEVICES` 를 그대로 씀 (예전엔 `nvidia-smi -L` 로 전부 세고 0번부터 다시 붙여, 나눠 줘도 GPU 0 으로 몰렸을 것)
- `fab ta fm --one-job`: 데이터셋 여럿의 split 을 GPU 작업 하나 (+ CPU 작업 하나) 로. 상한·다시 내기 없음. 예:
  `fab ta fm <데이터셋...> --models tabicl_kv_refit kumo_c8k_refit --splits all --accelerator GPU_8xH100 --one-job`

### 시험: 모델 시간 (초, split 하나)

GiveMeSomeCredit · APSFailure · diamonds 의 r0f0 를 GPU 작업 하나에 (H100 하나 · A10 하나). "지금" 은 #16 첫 실행의 같은 데이터셋 로그.

| 모델 | GMSC H100 | APS H100 | diamonds H100 | GMSC A10 | APS A10 | diamonds A10 |
|---|---|---|---|---|---|---|
| tabicl (지금, 5-fold) | 107~133 | 125 | | | | |
| `tabicl_kv` (5-fold) | 47 | 99 | 14 | | | |
| `tabicl_refit` | 57 | 31 | 5 | 160 | 149 | 13 |
| `tabicl_kv_refit` | **14** | **25** | **5** | OOM | 111 | 13 |
| kumo (지금, 5-fold) | 188~202 | 125 | | | | |
| `kumo_c8k` (5-fold) | 146 | 116 | 33 | | | |
| `kumo_refit` | 66 | 33 | 13 | 253 | 144 | 41 |
| `kumo_c8k_refit` | **45** | **31** | **11** | 221 | 136 | 38 |
| `kumo_c32k_refit` | 43 | 30 | 10 | | | |
| **합: 지금 → `tabicl_kv_refit` + `kumo_c8k_refit`** | **295~335 → 59** | **250 → 56** | → 16 | | | |

- refit + KV + 청크 8k 로 split 하나의 모델 시간이 약 1/5. 여기에 프로세스 시작 등이 모델 하나에 4~8초 (split 마다 10~15초) 더해짐 → split 하나 약 70초, 데이터셋 하나 (9 split) 약 11분
- TabICL KV 캐시는 5-fold 에서도 2.3배 (GMSC). 컨텍스트가 크고 test 가 클수록 이득이 큼 (APS 는 청크 2개라 1.3배)
- Kumo 청크 8192 는 2048 보다 6~32% 빠름 (refit). 32768 은 8192 와 거의 같음
- A10 은 H100 보다 3~5배 느리고, TabICL KV 캐시는 컨텍스트 10만 행 (GMSC) 에서 fit 중 OOM (5만 × 170 은 들어감). → 큰 데이터는 H100

### 시험: 점수 (test `metric_error`, r0f0, H100)

| 데이터셋 | 모델 | 지금 (5-fold) | refit | TabArena r0f0 | TabArena split 표준편차 |
|---|---|---|---|---|---|
| GiveMeSomeCredit | tabicl | 0.12950 | 0.12948 (`kv`: 0.12948) | 0.12928 | 0.00251 |
| | kumo | 0.12698 | 0.12691 (`c8k`: 0.12685) | 0.12693 | 0.00230 |
| APSFailure | tabicl | 0.004665 | 0.004539 (`kv`: 0.004539) | 0.004578 | 0.00184 |
| | kumo | 0.002674 | 0.002522 (`c8k`: 0.002652) | 0.002503 | 0.00146 |
| diamonds | tabicl | 506.19 ¹ | 504.52 (`kv`: 504.51) | 502.96 | 8.83 |
| | kumo | 483.10 ¹ | 481.45 (`c8k`: 481.42) | 481.24 | 7.51 |

¹ 지금 방식의 diamonds 결과가 아직 없어 `tabicl_kv`·`kumo_c8k` 의 5-fold 로 (아래처럼 같거나 거의 같음)

- KV 캐시는 점수가 같음: `tabicl_kv` (5-fold) 의 test 점수가 지금 `tabicl` 과 6자리까지 같고 (GMSC 0.129503 · APS 0.004665), fold 점수도 5자리까지 같음
- Kumo 청크는 아주 조금 다름 (float16 계산 순서): 5-fold GMSC 0.126977 → 0.126991, refit APS 0.002522 → 0.002652 (TabArena 표준편차의 0.09배).
  같은 `kumo_refit` 도 H100 과 A10 이 이만큼 다름 (APS 0.002522 대 0.002456)
- refit 은 6개 모두 지금보다 TabArena 쪽으로 옮겨 감 (차이는 TabArena 표준편차의 0.01~0.22배). 9 의 "차이" 에서 본 "컨텍스트 80% 대 refit" 과 맞는 방향. 1 split 이라 크기는 못 가름
- 시험 결과는 Volume 의 결과표·`tabarena.csv` 에도 들어감 (모델 이름 `*_refit`·`tabicl_kv`·`kumo_c8k` 등, note "speedup 시험"). H100·A10 이 같은 이름이라 표에는 나중에 합친 쪽만 남음 → 위 점수는 실행마다 예측을 받아 로컬에서 채점 (원격 채점과 같은 함수)

### 시험: 8×H100 (split 을 GPU 마다)

GiveMeSomeCredit · diamonds 의 모든 split (18개) × `tabicl_kv_refit`·`kumo_c8k_refit` 을 `GPU_8xH100` 작업 하나에 (`--one-job`).

| | 8×H100 작업 하나 | 1×H100 차례로 (위 r0f0 값으로 어림) |
|---|---|---|
| 작업 전체 (노드 뜨기 포함) | 4분 33초 (이번엔 노드가 바로 뜸) | |
| 18 split 계산 | 221초 (GMSC 8개 동시 → diamonds 9개 → 남은 GMSC 1개) | 약 850초 (GMSC 70초 × 9 + diamonds 24초 × 9) |
| GMSC split 하나 (동시에 돌 때) | 150~161초 (tabicl 50 · kumo 66) | 약 70초 (tabicl 14 · kumo 45) |
| diamonds split 하나 | 25~40초 | 약 24초 |

- 계산이 약 3.8배 빠름 (8배가 아님). 8개가 동시에 돌면 모델 하나가 느려짐: GMSC 의 tabicl fit 10초 → 46초, kumo 예측 45초 → 66초.
  diamonds 차례 (끝나는 시각이 흩어짐) 는 덜 느려짐. 까닭 (CPU·메모리 대역·프로세스가 한꺼번에 뜸) 은 안 가림
- GPU 시간으로는 더 듦: 8장 × 4.5분 = 36 GPU분, 1장으로 차례로면 약 15 GPU분. 끝부분은 GPU 1장만 돎 (GMSC 9번째)
- 점수: TabICL 은 1×H100 과 같음 (r0f0: 0.129477 · 504.514 둘 다 같음). Kumo 는 실행마다 조금 다름 (GMSC 0.126854 → 0.126843, diamonds 481.42 → 480.79)

refit 의 9 split 평균 (지금 방식의 5-fold 와 비교할 수 있는 것은 GMSC 5 split, #16 첫 실행에서)

| 데이터셋 | 모델 | refit (9 split) | TabArena (9 split) | (refit − TabArena) / 표준편차 | GMSC 5 split: 5-fold → refit (refit 이 나은 split) |
|---|---|---|---|---|---|
| GiveMeSomeCredit | `kumo_c8k_refit` | 0.13072 | 0.13073 | −0.01 | 0.13013 → 0.13007 (3 / 5) |
| | `tabicl_kv_refit` | 0.13301 | 0.13296 | +0.02 | 0.13237 → 0.13238 (3 / 5) |
| diamonds | `kumo_c8k_refit` | 484.20 | 483.84 | +0.05 | |
| | `tabicl_kv_refit` | 499.52 | 499.02 | +0.06 | |

- 넷 모두 TabArena 와 표준편차의 0.06배 안 (8 의 기준 "표준편차 안" 을 넉넉히 넘음)

#16 에 쓰면 (어림, refit 기준 1×H100 계산 1~2시간)

| 방법 | 벽시계 | 비고 |
|---|---|---|
| 1×H100 작업 5개 동시 (`--max-concurrent 5`) | 약 15~30분 + 노드 뜨기 | 데이터셋 하나 (9 split) 가 약 11분. 노드가 안 뜨는 일이 작업마다 생길 수 있음 |
| 8×H100 작업 하나 (`--one-job`) | 약 15~30분 + 노드 뜨기 한 번 | 노드 하나라 쿼터·"동시 7개" 문제를 피함. GPU 시간은 2배쯤 |
| 지금 방식 (5-fold) 을 8×H100 으로 | 약 1.3~1.8시간 + 노드 뜨기 (H100 5~7시간 ÷ 3.8) | 5-fold 는 계산이 길어 8×H100 의 이득이 큼 |

- refit 이면 두 방법의 벽시계가 비슷함 → 8×H100 은 "노드 하나로 끝냄" 이 이득. 5-fold 를 그대로 두면 8×H100 이 크게 줄임
- 8×H100 이 GPU_1xH100 의 8노드 쿼터와 같이 세지는지는 확인 안 함 (이번에는 다른 H100 작업이 없을 때 냄)

### #17 (Agent)

minipfn 에서 가져올 것 없음 (Agent 실행은 1시간 고정, GPU 계산이 병목이 아님). 줄일 곳은 동시 실행 수.

| 1회째 (`results/tabarena_runs.csv`) | 1시간 입력 토큰 | 429 |
|---|---|---|
| Codex × GPT | 4,164만~4,691만 | 19~100번 |
| AIDE × GPT · AIDE × Sonnet | 19만~48만 | 0 |
| Claude Code × Sonnet (+cont) | 9만~64만 | 0 |

- 분당 한도 (약 150만, [mle-bench.md](mle-bench.md) 의 "429") 를 쓰는 것은 Codex 뿐. AIDE 끼리 동시에 돌려도 부하가 없음은 확인함 ("동시 실행 부하 확인")
- 안: Codex 둘 (hazelnut → in_vehicle) 만 GPT 엔드포인트에서 하나씩, 나머지 8개는 그와 동시에. 벽시계 약 4시간 30분 → 약 2시간 + 준비 (Codex 2번).
  다만 AIDE × GPT 가 Codex 와 같은 엔드포인트를 동시에 씀 (1회째와 조건이 다름, 입력은 Codex 의 1% 쯤)
- 2회째는 이 안을 쓰지 않고 1회째처럼 Sonnet 줄 · GPT 줄을 하나씩 돌림 (hazelnut × Codex 만 취소 뒤 맨 끝에 다시, 2026-11-15 끝). 결과는 "파일럿 결과" 의 "2회째 (#17)"

### #16 에 쓴 것 (2026-11-14, 사용자가 정함)

- refit (`tabicl_kv_refit`·`kumo_c8k_refit`) · `GPU_8xH100` 작업 하나 (`--one-job`). 결정 10
- 남은 10개 × 9 split = 90 split: 작업 전체 10분 43초 (계산 약 9분 30초, 노드는 바로 뜸). 모델 실행 180개 모두 성공, OOM 없음.
  GiveMeSomeCredit·diamonds 18 split 은 위 8×H100 시험의 것
- 원격 채점: 90 대회 180행 59.5초, 빠짐·실패 0. 결과는 "파일럿 결과" 의 "FM 이 TabArena 와 맞는지"

### 39개에 쓴 것 (결정 10 의 (b), 2026-11-14 23:38 KST 제출)

- 2만 행 이하 39개 × 모든 split (708) × `tabicl_kv_refit`·`kumo_c8k_refit` 을 `GPU_8xH100` 작업 하나로 (`--one-job --timeout 180`). 데이터·정답은 5-fold 때 Volume 에 올린 것 그대로
- 시각은 KST, `results.csv`·`tabarena.csv` 의 시각은 UTC (모델 실행 2026-11-14T14:41~15:15, 채점 15:35~15:41)
- 작업 전체 37분 38초 (노드 3분 안에 뜸), 모델 실행 1,416개 모두 성공, OOM 없음. 모델 시간 합 3.2시간 (split 하나 중앙값 tabicl 5.7초 · kumo 8.4초, 데이터셋으로는 Bioresponse 833초가 가장 김)
- 어림 (8×H100 40~60분 + 노드) 보다 조금 짧음. 계산 약 34분 중 모델 시간은 3.2시간 ÷ 8장 = 약 24분, 나머지는 프로세스 시작 등 (나눠 재지 않음)
- 원격 채점: 708 대회 1,416행, 빠짐·실패 0

### 정할 것 (적용할 때 본 것, 기록)

- refit 을 쓰면 39개 (5-fold) 와 프로토콜이 다름 → (a) 12개만 refit 으로 하고 표에 적음, (b) 39개도 refit 으로 다시 (H100 약 1시간 어림, "컨텍스트 80% 대 refit" 도 가려짐), (c) 지금대로 5-fold (H100 5~7시간 어림).
  → **(b) 로 정함** (2026-11-14 밤, 사용자, 결정 10). 위 "39개에 쓴 것"
- `tabicl` 의 KV 캐시를 기본으로 할지 (점수 같음, OOM 이면 캐시 없이). Kumo 청크 8192 를 기본으로 할지 (점수가 아주 조금 바뀜 → MLE-bench·C1·C2 와 같은 FM 이 아님, 결정 4).
  → **둘 다 기본으로** (2026-11-15, 사용자, 결정 11). MLE-bench·C1 의 FM 도 refit 으로 다시 돌려 같은 FM 이 됨 ("FM 프로토콜과 이름")
- GPU: H100 (A10 은 3~5배 느림). 1×H100 여럿 대 8×H100 하나 (위 "8×H100")

## 결정 (2026-11-13)

3·4·9·11 은 사용자가 정함. 나머지는 추천안을 그대로 받음 (바꾸려면 여기를 고침).

| # | 정할 것 | 정한 것 | 까닭 |
|---|---|---|---|
| 1 | v0.1 대 BeyondArena | **TabArena-v0.1 (51개)** | 우리 FM 셋의 split 별 결과가 모두 공개돼 있어 "TabArena 와 맞나" 를 바로 봄. BeyondArena 는 시간·그룹 분할이라 지금 `fab run` 의 KFold 로는 샘 (그쪽 TFM 결과 유무는 확인 안 함). 데이터 받는 부분은 따로 떼어 나중에 옮길 수 있게 |
| 2 | Agent split | **r0f0 하나, 반복은 같은 split 에서 2회째** | MLE-bench P1 에서 Agent 는 실행끼리 흔들림이 split 보다 큼. FM 은 모든 split, Agent 와의 비교는 r0f0 점수끼리 |
| 3 | 데이터셋 이름 공개 | **안 알림 — "모델링 실력" 비교** | 타깃이 공개라 이름·출처가 있으면 정답 찾기가 쉬움. 과제·열 뜻·지표만 줌. 인터넷은 열고 사후 검사 (5 의 "익명화") |
| 4 | FM 설정 | **지금 설정 그대로, 차이만 기록. 기준은 8 의 "확인 방법"** | MLE-bench 결과·C1·C2 와 같은 FM 이어야 이어 읽음. 기준 (80%·표준편차 1배) 은 우선 이대로 진행. 결정 10 뒤에도 모델 설정은 그대로이고 test 예측 방식만 refit 으로 판정. 실험에 쓰는 5-fold FM 은 5-fold 표 (39개) 가 근거. 2026-11-15 결정 11 로 tabicl 캐시·Kumo 청크·Kumo 범주 처리와 test 예측 방식 (refit) 을 바꿈 |
| 5 | 대회 단위 | **A: (데이터셋, split) 마다 대회. 저장소에는 r0f0 만** | 지금 코드 경로를 그대로 씀 |
| 6 | 의존성 | **옆 venv (`FAB_TABARENA`) 에 `openml`·`pyarrow`** | `uv.lock` 안 바뀜 (사내 프록시), mlebench driver 와 같은 방식 |
| 7 | GBDT 기준선 | **우리 `lgbm`·`catboost` 를 돌림, TabArena 기본 GBDT 는 참고선** | Agent·C1 과 같은 조건의 대조군은 우리 것뿐. 비용 거의 없음 |
| 8 | 파일럿 | **FM 은 4개 (위 표) + 맞춤 확인용 51개, Agent 는 hazelnut · in_vehicle_coupon 부터** | 7 의 "파일럿 추천" |
| 9 | 맞춤 기준의 비김 (2026-11-14 추가) | **\|TabArena FM − 기본 GBDT\| ≤ TabArena split 표준편차인 데이터셋은 승패를 세지 않음** | 처음 기준에서 승패가 달랐던 7개가 모두 FM 과 GBDT 의 차이 3% 안. split 흔들림 안의 승패는 맞춤의 근거가 못 됨 |
| 10 | FM 맞춤 확인의 test 예측 (2026-11-14 추가, 같은 날 밤 고침) | **tabicl·kumo 의 51개 모두 refit (CV 없이 train 전체로 한 번, `tabicl_kv_refit`·`kumo_c8k_refit`), `GPU_8xH100` 작업 하나.** 처음에는 #16 의 12개만 refit, 그 뒤 "정할 것" 의 (b) 로 39개도 refit | TabArena 의 TFM test 예측 (`refit_folds=True`) 과 같은 방식이고 계산이 약 1/5 ("빠르게 돌리기"). tabicl·kumo 의 51개가 한 프로토콜이 되고 5-fold 의 기울기 (컨텍스트 80%) 가 사라짐. 5-fold 결과는 실험에 쓰는 FM 의 판정으로 남김. tabpfn_v2 는 5-fold 만 (refit 안 돌림). minipfn 도 기본이 refit. MLE-bench·C1·C2 의 FM (`fab run` 기본 5-fold) 은 그대로. 2026-11-15 결정 11 로 실험의 FM·GBDT 도 refit, tabpfn_v2 도 refit 33개 |
| 11 | FM 기준 프로토콜·기본값 (2026-11-15, 사용자가 정함) | **Agent 와 비교하는 FM 은 모두 refit (CV 없이 train 전체로 한 번). GBDT 도 refit. `tabicl` 은 KV 캐시 켬 (OOM 이면 캐시 없이), `kumo` 는 질의 청크 8192 · 범주 열을 범주로. 공식 설정에서 Agent 가 스스로 TFM 을 깔아 쓴 실행도 Agent 결과로 셈 (`tfm_runtime`·`pip_target` 표시는 남김). 이름·`opts` 열로 옛 결과와 가름 ("FM 프로토콜과 이름")** | refit 은 TabArena TFM 의 test 예측과 같고 5-fold 의 기울기 (컨텍스트 80%) 가 사라지며 계산은 약 1/5. GBDT 는 CPU 만이라 비용이 작아 같은 프로토콜로 맞춤 (사용자). KV 캐시는 점수가 같고 청크 8192 는 6~32% 빠름. 범주를 수치로 넘기던 Kumo 는 kddcup09 에서 앙상블·청크·refit 과 상관없이 TabArena 보다 2.2σ 나빴고, 범주로 바꾸니 범주 열이 있는 34개 중 25개가 TabArena 에 가까워짐. 스스로 깐 TFM 도 Agent 가 고른 풀이 (hazelnut × Codex 1회째) |

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
범주 = F0 의 범주 열 수 (타깃을 뺀 피처 중 %, "범주 비율과 FM − GBDT" 의 기준. `fab ta cats`).

| 데이터셋 | task | 과제 | 행 | 피처 | 범주 | 클래스 | split | 출처 | 라이선스 | TabPFNv2 | pfn2 | icl2 | kumo |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| blood-transfusion-service-center | 363621 | 이진 | 748 | 5 | 0 | 2 | 30 | UCI | CC BY 4.0 | O | -6 | -7 | -5 |
| diabetes | 363629 | 이진 | 768 | 9 | 0 | 2 | 30 | OpenML | CC0: Public Domain | O | -6 | -3 | -6 |
| anneal | 363614 | 다중 | 898 | 39 | 32 (84%) | 5 | 30 | UCI | CC BY 4.0 | O | -60 | -48 | -71 |
| QSAR_fish_toxicity | 363698 | 회귀 | 907 | 7 | 0 |  | 30 | UCI | CC BY 4.0 | O | -1 | -2 | -3 |
| credit-g | 363626 | 이진 | 1,000 | 21 | 13 (65%) | 2 | 30 | UCI | CC BY 4.0 | O | +6 | -4 | -9 |
| maternal_health_risk | 363685 | 다중 | 1,014 | 7 | 0 | 3 | 30 | UCI | CC BY 4.0 | O | -5 | -17 | -21 |
| concrete_compressive_strength | 363625 | 회귀 | 1,030 | 9 | 0 |  | 30 | UCI | CC BY 4.0 | O | +1 | -6 | -10 |
| qsar-biodeg | 363696 | 이진 | 1,054 | 42 | 5 (12%) | 2 | 30 | UCI | CC BY 4.0 | O | -8 | -15 | -15 |
| healthcare_insurance_expenses | 363675 | 회귀 | 1,338 | 7 | 3 (50%) |  | 30 | Kaggle | Database Contents License (DbCL) v1.0 | O | +4 | -2 | -3 |
| website_phishing | 363707 | 다중 | 1,353 | 10 | 9 (100%) | 3 | 30 | UCI | CC BY 4.0 | O | -9 | -12 | -17 |
| Fitness_Club | 363671 | 이진 | 1,500 | 7 | 3 (50%) | 2 | 30 | Kaggle | Public Domain | O | -4 | -4 | -4 |
| airfoil_self_noise | 363612 | 회귀 | 1,503 | 6 | 1 (20%) |  | 30 | UCI | CC BY 4.0 | O | -28 | -31 | -42 |
| Another-Dataset-on-used-Fiat-500 | 363615 | 회귀 | 1,538 | 8 | 1 (14%) |  | 30 | Kaggle | CC0: Public Domain | O | -1 | -3 | -5 |
| MIC | 363711 | 다중 | 1,699 | 112 | 94 (85%) | 8 | 30 | UCI | CC BY 4.0 | O | +3 | -2 | -7 |
| Is-this-a-good-customer | 363682 | 이진 | 1,723 | 14 | 8 (62%) | 2 | 30 | Kaggle | CC0: Public Domain | O | +1 | +0 | -3 |
| Marketing_Campaign | 363684 | 이진 | 2,240 | 26 | 9 (36%) | 2 | 30 | Kaggle | Public | O | -9 | -20 | -22 |
| hazelnut-spread-contaminant-detection | 363674 | 이진 | 2,400 | 31 | 0 | 2 | 30 | OpenML | CC BY-SA | O | -53 | -80 | -90 |
| seismic-bumps | 363700 | 이진 | 2,584 | 16 | 4 (27%) | 2 | 9 | UCI | CC BY 4.0 | O | +0 | -6 | -7 |
| splice | 363702 | 다중 | 3,190 | 61 | 60 (100%) | 3 | 9 | UCI | CC BY 4.0 | O | -4 | -10 | -21 |
| Bioresponse | 363620 | 이진 | 3,751 | 1777 | 0 | 2 | 9 | Kaggle | Public Domain | – | – | +8 | -11 |
| hiva_agnostic | 363677 | 다중 | 3,845 | 1618 | 1,617 (100%) | 3 | 9 | Other | Public | – | – | +14 | -0 |
| students_dropout_and_academic_success | 363704 | 다중 | 4,424 | 37 | 17 (47%) | 3 | 9 | UCI | CC BY 4.0 | O | -3 | -3 | -8 |
| churn | 363623 | 이진 | 5,000 | 20 | 4 (21%) | 2 | 9 | OpenML | MIT License | O | -5 | -15 | -7 |
| QSAR-TID-11 | 363697 | 회귀 | 5,742 | 1025 | 0 |  | 9 | OpenML | Public | – | – | +5 | -7 |
| polish_companies_bankruptcy | 363694 | 이진 | 5,910 | 65 | 0 | 2 | 9 | UCI | CC BY 4.0 | O | +6 | -60 | -89 |
| wine_quality | 363708 | 회귀 | 6,497 | 13 | 1 (8%) |  | 9 | UCI | CC BY 4.0 | O | +11 | -4 | -6 |
| taiwanese_bankruptcy_prediction | 363706 | 이진 | 6,819 | 95 | 0 | 2 | 9 | UCI | CC BY 4.0 | O | +4 | -4 | -21 |
| NATICUSdroid | 363689 | 이진 | 7,491 | 87 | 86 (100%) | 2 | 9 | UCI | CC BY 4.0 | O | +21 | -13 | -16 |
| coil2000_insurance_policies | 363624 | 이진 | 9,822 | 86 | 5 (6%) | 2 | 9 | UCI | CC BY 4.0 | O | +2 | -6 | -11 |
| Bank_Customer_Churn | 363619 | 이진 | 10,000 | 11 | 4 (40%) | 2 | 9 | Kaggle | Public | O | -1 | -1 | -5 |
| heloc | 363676 | 이진 | 10,459 | 24 | 0 | 2 | 9 | Kaggle | Public | O | -1 | -2 | -2 |
| jm1 | 363712 | 이진 | 10,885 | 22 | 0 | 2 | 9 | OpenML | Public Domain | O | +6 | -10 | -15 |
| E-CommereShippingData | 363632 | 이진 | 10,999 | 11 | 4 (40%) | 2 | 9 | Kaggle | Public Domain | O | +0 | -0 | +0 |
| online_shoppers_intention | 363691 | 이진 | 12,330 | 18 | 7 (41%) | 2 | 9 | UCI | CC BY 4.0 | O | +0 | -5 | -8 |
| in_vehicle_coupon_recommendation | 363681 | 이진 | 12,684 | 25 | 22 (92%) | 2 | 9 | UCI | CC BY 4.0 | O | +31 | -3 | -10 |
| miami_housing | 363686 | 회귀 | 13,776 | 16 | 1 (7%) |  | 9 | OpenML | CC BY-NC-SA | O | +8 | -3 | -11 |
| HR_Analytics_Job_Change_of_Data_Scientists | 363679 | 이진 | 19,158 | 13 | 10 (83%) | 2 | 9 | Kaggle | CC0: Public Domain | – | – | -1 | -2 |
| houses | 363678 | 회귀 | 20,640 | 9 | 0 |  | 9 | Other | Public | – | – | -8 | -15 |
| superconductivity | 363705 | 회귀 | 21,263 | 82 | 0 |  | 9 | UCI | CC BY 4.0 | – | – | -5 | -9 |
| credit_card_clients_default | 363627 | 이진 | 30,000 | 24 | 3 (13%) | 2 | 9 | UCI | CC BY 4.0 | – | – | -3 | -4 |
| Amazon_employee_access | 363613 | 이진 | 32,769 | 10 | 9 (100%) | 2 | 9 | Kaggle | Public Domain | – | – | +24 | +3 |
| bank-marketing | 363618 | 이진 | 45,211 | 14 | 8 (62%) | 2 | 9 | UCI | CC BY 4.0 | – | – | +3 | -2 |
| Food_Delivery_Time | 363672 | 회귀 | 45,451 | 10 | 3 (33%) |  | 9 | Kaggle | Database Contents License (DbCL) v1.0 | – | – | +3 | -2 |
| physiochemical_protein | 363693 | 회귀 | 45,730 | 10 | 0 |  | 9 | UCI | CC BY 4.0 | – | – | -15 | -20 |
| kddcup09_appetency | 363683 | 이진 | 50,000 | 213 | 38 (18%) | 2 | 9 | Other | Public | – | – | +10 | -7 |
| diamonds | 363631 | 회귀 | 53,940 | 10 | 3 (33%) |  | 9 | Other | MIT License | – | – | -4 | -7 |
| Diabetes130US | 363630 | 이진 | 71,518 | 48 | 39 (83%) | 2 | 9 | UCI | CC BY 4.0 | – | – | +2 | -3 |
| APSFailure | 363616 | 이진 | 76,000 | 171 | 0 | 2 | 9 | UCI | CC BY 4.0 | – | – | -23 | -39 |
| SDSS17 | 363699 | 다중 | 78,053 | 12 | 2 (18%) | 3 | 9 | Kaggle | Public Domain | – | – | -5 | -20 |
| customer_satisfaction_in_airline | 363628 | 이진 | 129,880 | 22 | 16 (76%) | 2 | 9 | Kaggle | CC0: Public Domain | – | – | -9 | -24 |
| GiveMeSomeCredit | 363673 | 이진 | 150,000 | 11 | 0 | 2 | 9 | Kaggle | Public | – | – | -1 | -2 |
