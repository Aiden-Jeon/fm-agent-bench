# Forecast (P7): 시계열 예측에서 tabular FM 대 Agent

tabular FM 을 시계열 예측에 쓰는 공개 구현 (TabPFN-TS, TabICL forecaster) 이 실제로 먹히는지, 같은 task 에서 LLM 코딩 에이전트
(기존 5설정 × 1시간) · 기존 예측 방법과 비교해 Elo 로 답한다. 배경은 minipfn issue #117 (tabular FM 을 회귀·분류 밖 문제에 쓰는
방법 목록). 여기서 하는 것은 minipfn 구현이 아니라 공개된 tabular FM 의 검증이고 minipfn 코드·가중치는 쓰지 않는다.

- **상태: 설계 확정 (2026-11-25, 사용자).** 아래 "결정" 1~9. 구현 · 원격 실행 · Agent 는 그다음 (원격은 실행 전마다 확인)
- 조사일 2026-11-25. fev 저장소는 `autogluon/fev` 의 `ecce671` (2026-11-23 커밋), 패키지 `fev` 0.10.0, 데이터 `autogluon/fev_datasets` 의 `d3e7e72` (HF, 2026-10-21 수정)
- TabArena 트랙 ([tabarena.md](tabarena.md)) 의 순서·규칙을 그대로 따름. 다른 점만 적음
- 확인 못 한 것은 "확인 못 함" 으로 적음. 추측으로 채우지 않음

목차

- [요약](#요약)
- [출처](#출처)
- [1. 벤치: fev-bench](#1-벤치-fev-bench)
- [2. task 고르기](#2-task-고르기)
- [3. 분할: window](#3-분할-window)
- [4. 지표](#4-지표)
- [5. 기준점과 Elo 풀](#5-기준점과-elo-풀)
- [6. 기존 방법 풀](#6-기존-방법-풀)
- [7. FM](#7-fm)
- [8. Agent: 제출 형식·정답 격리·익명화](#8-agent-제출-형식정답-격리익명화)
- [9. 오염과 사후 검사](#9-오염과-사후-검사)
- [10. 비용 어림](#10-비용-어림)
- [11. 구현 계획](#11-구현-계획)
- [12. 구현 (2026-11-25)](#12-구현-2026-11-25)
- [결정 (2026-11-25)](#결정-2026-11-25)
- [확인 못 한 것](#확인-못-한-것)

## 요약

| 항목 | 정한 것 |
|---|---|
| 벤치 | fev-bench 100 task 중 10개 후보 (규칙은 2), Agent 파일럿 2개: `redset_1H` · `solar_with_weather_1H` |
| 분할 | fev task 의 rolling window 그대로. FM·기존 방법은 모든 window, **Agent 는 마지막 window 하나** (TabArena 의 r0f0 자리) |
| 지표 | **주 지표 SQL** (scaled quantile loss, fev-bench 의 `eval_metric`), 보조 MASE (점 예측). fev 의 정의를 numpy 로 옮기고 fev 와 같은 값인지 로컬에서 확인 |
| 기준점 | **Seasonal Naive = 1000** (fev-bench skill score 의 기준) |
| Elo | `fab/bench/tabarena/elo.py` 의 `elo()` 를 기준점 인자 (`anchor="RandomForest"` 기본) 로 일반화해 그대로. 풀 둘: FM 풀 (task 평균, 모든 window) · Agent 풀 (마지막 window) |
| 기존 방법 | Seasonal Naive · AutoETS (statsforecast) + 시계열 FM Chronos-2 · Chronos-Bolt · TimesFM-2.5 (Apache-2.0 가중치): 공개 결과 (task 평균) 를 받고, window 별 값이 없어서 우리가 다시 돌림. **AutoARIMA 는 공개 결과로 FM 풀에만** (CPU 약 4시간을 아낌, Agent 풀에서는 뺌) |
| FM (이 트랙) | TabPFN-TS (fev 의 `tabpfn-ts` 래퍼: `tabpfn-time-series` 1.0.10 · TabPFN **v2** 회귀, 공개 결과 "TabPFN-TS (v2)" 와 같은 설정) · TabICL forecaster (`tabicl[forecast]` 2.2.0). 기본 설정 한 번. TabPFN v3.5 는 뺌 (결정 6) |
| Agent | 기존 5설정 × 1시간 · A10. 제출 = 마지막 window 의 점 예측 + 분위수 9개 (0.1~0.9). 정답 격리·익명 id (`forecast-p01`…) · 사후 검사 |
| 범위 밖 | C1 · C2 (Agent 결과를 보고 사용자가 정함), 시계열 이상탐지 (TSB-AD, 이 트랙 다음) |

## 출처

| 출처 | 쓴 곳 |
|---|---|
| 논문: Shchur et al., "fev-bench: A Realistic Benchmark for Time Series Forecasting", [arXiv:2509.26468](https://arxiv.org/abs/2509.26468) | 1, 4, 5 |
| 저장소 [autogluon/fev @ ecce671](https://github.com/autogluon/fev/tree/ecce67120aae4827e9702b978a9e9327ecde81d0) (Apache-2.0): `benchmarks/fev_bench/tasks.yaml` (100 task 정의) · `results/*.csv` (모델 54개의 task 별 결과) · `models.yaml` · `models/<이름>/model.py` (모델 래퍼) | 1, 2, 6, 7 |
| 패키지 `fev` 0.10.0 (`Task`·`EvaluationWindow`·`metrics.py`) | 3, 4 |
| 데이터 [HF autogluon/fev_datasets @ d3e7e72](https://huggingface.co/datasets/autogluon/fev_datasets) (config 100개, 라이선스 표시 `other` = 데이터셋마다 다름) | 2, 8 |
| 리더보드 [HF Space autogluon/fev-leaderboard](https://huggingface.co/spaces/autogluon/fev-leaderboard) | 동적 페이지라 안 읽음 (확인 못 함). 숫자는 저장소의 `results/*.csv` 에서 |
| TabPFN-TS [PriorLabs/tabpfn-time-series](https://github.com/PriorLabs/tabpfn-time-series): 1.0.10 (fev 래퍼 `models/tabpfn-ts`, v2 체크포인트 `tabpfn-v2-regressor-2noar4o2.ckpt`). 최신 1.4.0 의 기본은 v3.5 (`defaults.py`) | 7 |
| TabICL forecaster [튜토리얼](https://tabicl.readthedocs.io/en/latest/tutorials/time_series_forecasting.html), `tabicl` 2.2.0 의 `tabicl/forecast/_forecaster.py` | 7 |

공개 결과 스냅샷: fev 저장소 `ecce671` 의 `benchmarks/fev_bench/results/` (2026-11-23). 받은 것은 `results/forecast_reference.csv` 에 원래 열 (`model_name`·`task_name`·`SQL`·`MASE`·`WQL`·`WAPE`·`fev_version`·`fev_commit`·`model_kwargs`·`trained_on_this_dataset`) 그대로 + 스냅샷 commit.

## 1. 벤치: fev-bench

| 항목 | 찾은 것 |
|---|---|
| task | 100개. 데이터셋 config 하나 + horizon · window 수 · seasonality · 공변량 열 이름. 단변량 65 · 다변량 35, 공변량 있는 것 46 (known dynamic · past dynamic · static) |
| 주기 | 5분 ~ 연 단위. horizon 5 ~ 288 |
| window | task 마다 1~20개 (`num_windows`). window 마다 cutoff 가 다르고 cutoff 앞 전부가 과거 (max_context_length 없음). 기본 `initial_cutoff = −horizon − (W−1)·step`, step = horizon → 마지막 window 는 시리즈 끝 horizon 개를 맞힘 |
| 지표 | 100개 모두 `eval_metric: SQL`, 보조 MASE · WAPE (ε=1) · WQL (ε=1). 분위수 0.1~0.9 (9개) |
| 합치기 (fev-bench) | skill score (Seasonal Naive 대비 오차 비의 기하평균) · 승률. Elo 는 없음 → 우리가 TabArena 식 Elo 를 매김 |
| 공개 결과 | task 마다 한 줄 (window 평균). **window 별 값은 없음** (3 의 결정에 영향) |
| 우리 풀의 공개 결과 | Seasonal Naive 100 · AutoETS 97 · AutoARIMA 96 · Chronos-2 100 · Chronos-Bolt 100 · TimesFM-2.5 100 · TabPFN-TS (v2-reg, `tabpfn-time-series` 1.0.10) 99 · TabPFN-TS-3 (v3 `20260506_timeseries` 체크포인트) 100. TabICL forecaster 는 없음 (`ts-icl` 은 다른 모델, EDF) |
| 사전학습 겹침 | 결과 표의 `trained_on_this_dataset`: TimesFM-2.5 10 task (favorita_transactions · fred_md · kdd_cup_2022_10T · m5 · proenfo_gfc12/14/17), Chronos-2 · Chronos-Bolt · TabPFN-TS 0 |

## 2. task 고르기

규칙 (순서대로 적용, 10개를 남김)

1. **단변량** (`target` 하나): 제출 표가 (id, timestamp) 마다 한 줄로 단순. 다변량 35개는 이번에 뺌 (fev 의 `as_univariate` 로 펴면 같은 형식이 되므로 나중에 넣을 수 있음)
2. **Agent 가 1시간에 풀 크기**: 마지막 window 의 과거 점 수 (시리즈 수 × 길이) ≤ 30만, 시리즈 ≤ 1,000. A10 에서 GBDT·통계 모형을 여러 번 돌려 볼 수 있고 AutoARIMA 기준선도 몇 분 안
3. **기준 FM 의 사전학습과 안 겹침**: 우리 풀의 기존 FM (Chronos-2 · Bolt · TimesFM-2.5) 이 `trained_on_this_dataset` 인 task 는 뺌 (proenfo_gfc14 · gfc17 · favorita_transactions_1D 가 걸림)
4. **섞기**: 주기 (시간 · 일 · 주 · 월), 공변량 (없음 · known · past), 영역 (에너지 · 수요 · 보건 · 경제 · IT 부하), 오염 위험 (덜 알려진 것 위주 + Kaggle 대회 하나를 대조군, TabArena 와 같음)
5. **방법 사이 차가 있는 것**: 공개 결과의 SQL skill (Seasonal Naive 대비) 이 방법마다 비슷하면 (예: hospital_admissions_1D 는 0.34~0.35) 비교가 안 갈림 → 하나만 남김

후보 10개 (마지막 window 기준. skill = 1 − SQL / Seasonal Naive 의 SQL, 공개 결과의 task 평균)

| # | task | 주기 · H · W | 시리즈 × 길이 (중앙값) | 마지막 window 과거 점 | 공변량 | 영역 · 오염 | skill: AutoETS / Chronos-2 / TimesFM-2.5 / TabPFN-TS / TabPFN-TS-3 |
|---|---|---|---|---|---|---|---|
| 1 | **redset_1H** (파일럿) | 시 · 24 · 10 | 138 × 2,160 | 279,758 | static 1 (`subset`) | 클라우드 DW 부하 (2024 공개) · 낮음 | −0.22 / 0.30 / 0.30 / 0.32 / 0.31 |
| 2 | **solar_with_weather_1H** (파일럿) | 시 · 24 · 20 | 1 × 49,648 | 49,624 | known 7 (날씨 예보) · past 2 | 태양광 발전 · 낮음 | −0.80 / 0.37 / 0.33 / 0.43 / 0.46 |
| 3 | ercot_1D | 일 · 28 · 20 | 8 × 6,452 | 51,392 | 없음 | 전력 부하 · 중간 (공개 계통 데이터) | −0.03 / 0.35 / 0.38 / 0.27 / 0.37 |
| 4 | epf_be | 시 · 24 · 20 | 1 × 52,416 | 52,392 | known 2 (발전·부하 예측) | 전력 가격 · 높음 (EPF 교과서 벤치) | −0.33 / 0.56 / 0.57 / 0.53 / 0.54 |
| 5 | rohlik_orders_1D | 일 · 61 · 5 | 7 × 1,197 | 7,186 | known 4 · past 9 | 주문 수 · **높음 (Kaggle 대회 2024, 대조군)** | 0.07 / 0.38 / 0.35 / 0.24 / 0.32 |
| 6 | solar_1D | 일 · 28 · 10 | 137 × 365 | 46,169 | 없음 | 태양광 · 중간 | 0.27 / 0.34 / 0.32 / 0.32 / 0.33 |
| 7 | ecdc_ili | 주 · 13 · 10 | 25 × 201 | 4,472 | 없음 | 독감 유사 질환 · 낮음 | −0.08 / 0.40 / 0.41 / 0.38 / 0.35 |
| 8 | hospital | 월 · 12 · 4 | 767 × 84 | 55,224 | 없음 | 병원 수요 · 중간 (Monash 묶음) | 0.10 / 0.15 / 0.16 / 0.14 / 0.17 |
| 9 | us_consumption_1M | 월 · 12 · 10 | 31 × 792 | 24,180 | 없음 | 소비 지출 · 중간 (공개 경제 통계) | 0.46 / 0.46 / 0.40 / 0.41 / 0.37 |
| 10 | hospital_admissions_1D | 일 · 28 · 20 | 8 × 1,731 | 13,622 | 없음 | 입원 수 · 낮음 | 0.35 / 0.35 / 0.35 / 0.34 / 0.35 |

- 파일럿 2개를 고른 까닭: 둘 다 덜 알려짐. redset_1H 는 시리즈가 많고 공변량이 거의 없으며 방법 사이 차가 큼 (AutoETS −0.22 ~ TabPFN-TS 0.32, Chronos-Bolt −0.17). solar_with_weather_1H 는 known 공변량 (날씨 예보) 이 있고 공개 결과에서 **tabular FM (TabPFN-TS) 이 시계열 FM 보다 나은** task 라 이 트랙의 질문에 바로 닿음
- 뺀 것 중 알아 둘 것: M_DENSE · LOOP_SEATTLE · SZ_TAXI · hierarchical_sales 는 크기는 맞지만 Moirai-2.0 사전학습과 겹침 (우리 풀에 Moirai 는 없어 규칙 3 에는 안 걸림, 후보를 더 늘릴 때 씀). restaurant · rossmann · favorita · m5 · walmart · kdd_cup_2022 는 Kaggle 대회 (오염 높음) 또는 크기 초과
- 10 의 hospital_admissions_1D 는 규칙 5 에 걸리지만 시리즈가 적은 일 단위 보건 데이터의 대표로 하나 남김 (Agent 를 돌릴지는 파일럿을 보고)

## 3. 분할: window

| 방식 | 받는 것 | 까닭 |
|---|---|---|
| FM · 기존 방법 | 모든 window (fev 의 `iter_windows()`) | fev-bench 공개 결과 (window 평균) 와 같은 조건 → "우리 실행이 공개 결과와 맞나" 확인 (TabArena 의 "FM 이 TabArena 와 맞는지") |
| Agent | **마지막 window 하나** (`get_window(W−1)`) | window 를 여럿 주면 뒤 window 의 과거에 앞 window 의 정답이 들어 있어 격리가 안 됨 (window 마다 따로 1시간은 비용이 W 배). 마지막 window = 주어진 데이터 끝 다음 horizon 개 → "미래 예측" 그대로. TabArena 의 r0f0 자리 |

- window 별 공개 결과가 없어서 Agent 풀의 기존 방법·FM 은 우리가 돌린 실행의 마지막 window 값을 씀 (6). 공개 결과는 FM 풀과 맞춤 확인에만
- 과거는 cutoff 앞 전부 (fev-bench 에 `max_context_length` 없음). 모델마다의 컨텍스트 상한 (TabPFN-TS 등) 은 모델 기본값 그대로
- **cutoff 가 정수인 task 는 시리즈마다 자기 끝의 horizon 개** (fev 의 정의, 2026-11-25 확인). 시리즈 길이가 다른 redset_1H (예측 시작 33가지, 2024-04-04 ~ 05-30) · ecdc_ili (10가지) · hospital_admissions_1D (2가지) 는 한 시리즈의 과거가 다른 시리즈의 예측 구간과 달력으로 겹침. FM 도 같은 입력을 받으므로 (Chronos-2 의 cross-learning 등) 그대로 두고 설명에 "시리즈마다 끝이 다르다" 고 적음. 나머지 7개는 예측 시작이 같음
- fev 가 데이터를 읽을 때 빼는 시리즈 (cutoff 앞이 `min_context_length` 보다 짧거나 뒤가 horizon 보다 짧은 것) 는 우리도 뺌 (fev 의 `get_input_data()` 를 그대로 써서 같은 행)

## 4. 지표

fev 0.10.0 `metrics.py` 의 정의 그대로. y 는 정답, ŷ 는 점 예측, q̂_τ 는 분위수 τ 예측, 시리즈 i 의 계절 오차 s_i = 과거 전체에서 mean |y_t − y_{t−m}| (m = task 의 seasonality).

| 지표 | 식 | 쓰는 곳 |
|---|---|---|
| **SQL** (주) | mean_{i,h,τ} 2·\|(y − q̂_τ)(1[y ≤ q̂_τ] − τ)\| / s_i | Elo (주), 순위 |
| MASE | mean_{i,h} \|y − ŷ\| / s_i | Elo (보조, 점 예측만 보는 Agent 실력) |
| WQL (ε=1) · WAPE (ε=1) | 척도 대신 \|y\| 평균으로 나눔 | 기록만 |

- s_i 가 정의되지 않는 시리즈 (과거가 m 보다 짧음, 전부 NaN, s_i = 0) 는 평균에서 빠짐 (fev 와 같음). 정답의 NaN 도 nanmean 으로 빠짐
- task 의 점수 = window 별 점수의 평균 (fev 의 `evaluation_summary`). Agent 는 window 하나라 그 값
- 원격 채점은 fev 없이: 정리 작업 환경 (uv.lock) 에 fev 가 없고 lock 을 안 바꾸려고 위 식을 `fab/bench/forecast/metrics.py` 에 numpy 로 옮김 (`fab/core/metrics.py` 는 표 형식 (y, pred) 꼴이라 손대지 않음 — 이상탐지 트랙과 겹치지 않게). 로컬에서 같은 예측을 fev `compute_metrics` 와 우리 식으로 채점해 비교 (`fab fc check all`, 2026-11-25): 10개 모두 상대 차 최대 6.8e−8 (CSV 가 float32 값을 짧은 글자로 적어서. 7개는 1e−9 아래)
- `metric_error` = SQL (낮을수록 좋음, TabArena 와 같은 열 이름) → `elo()` 에 그대로

## 5. 기준점과 Elo 풀

`elo()` 를 그대로 쓰고 기준점만 인자로: `elo(long, rounds=200, seed=0, min_ci=10, anchor="RandomForest")`, `fill(long, methods, anchor="RandomForest")`. 기본값이 지금과 같아 TabArena 결과는 비트 단위로 같아야 함 → 고친 뒤 `uv run --frozen fab ta elo` 로 `results/tabarena_elo.csv` 가 안 바뀌는지 확인. 이상탐지 트랙 (`bench-anomaly`) 도 같은 시그니처로 고침 (나중에 머지되는 쪽이 rebase).

| 풀 | 단위 | 참가자 | 기준점 |
|---|---|---|---|
| **FM 풀** | (task, `all`) — window 평균 하나 | 공개 결과 (Seasonal Naive · AutoETS · AutoARIMA · Chronos-2 · Chronos-Bolt · TimesFM-2.5 · TabPFN-TS · TabPFN-TS-3) + 우리 실행 (AutoARIMA · TabPFN-TS-3 을 뺀 같은 방법 + TabICL) | 공개 Seasonal Naive = 1000 |
| **Agent 풀** | (task, `w<W−1>`) — 마지막 window | 우리 실행 (기존 방법 5 + FM 2) + Agent 5설정 (설정마다 회차 평균). AutoARIMA 없음 (window 별 값이 없고 다시 안 돌림) | 우리 Seasonal Naive = 1000 |

- 빈칸 (공개 결과의 AutoETS 3 · AutoARIMA 4 · TabPFN-TS 1 task, 실패한 실행·Agent 회차) 은 TabArena 처럼 Seasonal Naive 값으로 채우고 `imputed` 에 남김
- task 가 10개라 95% 구간은 냄 (`MIN_CI_DATASETS = 10`). 파일럿 2개 단계에서는 점추정만 (TabArena 와 같음)
- 풀이 다르면 Elo 도 다름 (TabArena 와 같음): 두 풀 · 다른 트랙 숫자를 나란히 놓지 않음
- SQL Elo 가 주, MASE Elo 는 같은 풀로 하나 더 (`metric` 열로 가름)

## 6. 기존 방법 풀

| 방법 | 구현 | 설정 | 우리가 돌리는 까닭 |
|---|---|---|---|
| Seasonal Naive (기준점) | statsforecast (fev `models/statsforecast`) | `{"model": "seasonal_naive"}` | window 별 값 (Agent 풀). 공개 결과와 같아야 함 → 채점·분할 맞춤의 첫 확인 |
| AutoETS | 〃 | `max_season_length: 200` (공개와 같게) | 〃 |
| AutoARIMA | (안 돌림) | 공개 `max_season_length: 100` | **공개 결과로 FM 풀에만** (결정 5, 2026-11-25 사용자). CPU 가 오래 걸려서 (공개 결과 후보 10개에 약 4시간) |
| Chronos-2 | fev `models/chronos-2` | `batch_size: 100` | 시계열 전용 FM (Apache-2.0) |
| Chronos-Bolt (base) | fev `models/chronos` | `batch_size: 256` | 〃 |
| TimesFM-2.5 (200M) | fev `models/timesfm-2.5` | 기본 | 〃 (Apache-2.0). 후보 10개는 사전학습 겹침 없음 |

- 래퍼는 fev 저장소 `ecce671` 의 `models/<이름>/model.py` 와 `requirements.txt` 를 원격 작업에서 그 commit 으로 받아 씀 (Apache-2.0, 출처 표시). 우리 저장소에는 안 넣음 (래퍼마다 패키지 버전이 달라 실행마다 pip 설치, `docker/air` 이미지는 안 바꿈)
- 맞춤 기준 (TabArena 8 을 본뜸, 초안): 방법마다 후보 10개 중 8개 이상에서 |우리 task 평균 SQL − 공개 SQL| / 공개 SQL ≤ 2%. Seasonal Naive 는 결정적이라 같아야 함 (다르면 분할·채점을 고침). 패키지 버전 차 (공개는 대부분 fev 0.8.0 · commit `f323f6c`) 는 기록
- 튜닝·앙상블 방법 (catboost · lightgbm 의 mlforecast, Stat. Ensemble 등) 은 넣지 않음: "튜닝 없는 FM 한 번" 대 "Agent 1시간" 의 틀에서 기존 방법은 기준점 역할

## 7. FM

이 트랙의 "FM" = 표 형식 FM 을 시계열에 쓴 공개 구현. 기본 설정 한 번, 튜닝 없음.

| 이름 | 패키지 | 모델 | feature | 라이선스 |
|---|---|---|---|---|
| `tabpfn_ts` | fev `models/tabpfn-ts` 래퍼 (`tabpfn` 7.1.1 · `tabpfn-time-series` 1.0.10, `TabPFNTimeSeriesPredictor` 로컬 모드, 컨텍스트 5,000) | TabPFN **v2** 회귀 (`tabpfn-v2-regressor-2noar4o2.ckpt`) | running index, 달력 sin/cos, 자동 계절 (FFT) + known 공변량 그대로. 타깃 lag 없음 | 패키지 Apache-2.0, v2 가중치는 수락 없이 받음 (표 형식 트랙의 `tabpfn_v2` 와 같음) |
| `tabicl_ts` | `tabicl[forecast]` 2.2.0 (`TabICLForecaster`) | TabICL v2 회귀 (기본 체크포인트) | index · 달력 · 자동 주기 (FFT). 공변량은 context/future 의 추가 열로 | BSD-3 (패키지·가중치, 확인 필요) |

- TabPFN-TS 는 fev 래퍼 (`models/tabpfn-ts-3`) 와 같은 `predict_fev` 경로 (패키지가 fev 를 직접 받음). 공개 TabPFN-TS-3 (v3 체크포인트) · TabPFN-TS (v2) 결과는 참고선 (FM 풀)
- TabICL 은 fev 래퍼가 없어 우리가 씀 (`fev.convert_input_data(window, "pandas", as_univariate=True)` → `predict_df(context, future_df)` → 분위수 열 이름만 바꿈). past dynamic 공변량은 미래 값이 없어서 뺌 (TabPFN-TS 와 같음), static 은 안 씀 (TabPFN-TS 도 안 씀)
- 변형 후보 (구현은 공식 둘 다음, 문서에만): 기존 FM (kumo · tabicl 회귀) 에 TabPFN-TS 식 feature (running index, 달력 sin/cos, 계절, lag 없음) 를 만들어 회귀로 넣는 `kumo~tsfeat` · `tabicl~tsfeat`. 같은 feature 로 GBDT (`catboost~tsfeat`) 도 대조군
- GPU: H100 1장 (`fab databricks` 기본). 예측 시간은 공개 결과의 `inference_time_s` 와 비교해 남김

## 8. Agent: 제출 형식·정답 격리·익명화

대회 폴더: `competitions/fc-<task>/` (접두사 `fc-`, TabArena 의 `ta-` 자리). `config.yaml` 의 `forecast:` 블록 (fev task 이름 · window · `anon_id`), `description.md` (영어, 우리가 씀).

| 파일 (에이전트가 받음) | 내용 |
|---|---|
| `train.csv` | 마지막 window 의 과거: `id, timestamp, target` + past dynamic · known dynamic · static 공변량 열 (static 은 시리즈마다 같은 값을 반복). id 순 · 시각 순 |
| `test.csv` | 맞힐 칸: `id, timestamp` (시리즈마다 horizon 개) + known dynamic 공변량의 미래 값 |
| `sample_submission.csv` | `id, timestamp, predictions, 0.1, 0.2, …, 0.9` (fev 의 열 이름 그대로). 예시 값 = 시리즈의 마지막 값 |
| `description.md` | 과제 (무엇의 시계열인지 일반 말로), 주기 · horizon · 시리즈 수, 열 뜻, 지표 (SQL 주 · MASE 보조, 식과 seasonality), 제출 형식. 데이터셋 이름 · 출처 · 수집 기관 · 지역 이름 없음 |
| 정답 (안 줌) | `id, timestamp, target` (test 칸). `FAB_FORECAST/data/answers/` (로컬) + 채점용 내 Volume 의 `forecast/answers/` 에만. 채점에 필요한 과거 (계절 오차 s_i) 는 public `train.csv` 에서 |

- **형식 검사 서버** (`fab/agent/runner.py` 의 `check_submission`): 지금은 열 · 행 수 · id 집합만 봄 → sample 에 `timestamp` 열이 있으면 (id, timestamp) 짝 집합이 같은지, 예측 열이 숫자인지까지 보게 넓힘 (다른 벤치는 그대로). 분위수가 단조인지는 fev 처럼 강제하지 않음
- 지시문은 TabArena 처럼 MLE-bench `instructions.txt` + `additional_notes.txt` 그대로, 대회 설명만 다름
- **익명화** (TabArena 결정 3 과 같은 뜻)

| 줌 | 안 줌 |
|---|---|
| 실제 timestamp (달력 · 휴일 효과는 모든 방법이 씀, FM 도 실제 시각을 받음) | 데이터셋 · fev task 이름, 원래 출처 · URL · 기관 · 지역 이름 |
| 공변량 열 이름 (`temp`, `humidity` 처럼 뜻이 있는 것, TabArena 의 피처 이름과 같음) | 원래 시리즈 id (ERCOT 지역 · 나라 · 병원 이름 등) → `s001` 꼴로 새로 매김 (대응표는 로컬 정답 옆) |
| 범주형 static 값 중 뜻만 있는 것 (`subset`: provisioned · serverless) | 타깃 원래 이름 (`target` 으로) |

- 공변량 이름이 출처를 짐작하게 하는 곳은 그대로 둠 (TabArena 의 피처 이름과 같은 규칙): rohlik_orders_1D 의 `frankfurt_shutdown` (오염 대조군이라 상관없음), redset_1H 의 static 값 `provisioned` · `serverless`
- 누출 검사 (`fab/agent/__init__.py` 의 `leak_check` 를 그대로): 코드 묶음 · run.json · 데이터 파일 이름 · description.md · **파일의 id 값과 열 머리** 에 task 이름 (구분자 바꾼 꼴 포함) · 원래 id 가 있으면 제출 안 함
- Agent 회차: 파일럿 2 task × 5설정 × 1회 (10 Agent-시간), 결과를 보고 나머지 8개 (40 Agent-시간). LLM 엔드포인트 줄 규칙 (Sonnet 줄 · GPT 줄은 동시에 하나씩, 내기 전 `fab mleb agent list`) 을 지킴

## 9. 오염과 사후 검사

- fev-bench 데이터는 모두 공개 (HF · 원래 출처). 시계열은 **미래 값 자체가 공개** (예: 계통 운영자 부하, 보건 통계, 경제 통계) → 인터넷을 여는 한 에이전트가 원본을 받아 test 칸을 채울 수 있음. 막는 방법은 사후 검사뿐 (TabArena · MLE-bench 와 같은 구조)
- `data_source_urls` 에 더함: `huggingface.co/datasets/autogluon` · `autogluon/fev_datasets` · `chronos_datasets` · `import fev` · `datasets.load_dataset` 의 HF 데이터 · 영역별 원본 (계통 운영자 · ECDC · FRED · ENTSO-E · Kaggle 대회 데이터 주소, 구현 때 후보 10개의 원래 출처로 목록을 만듦)
- 알아본 흔적: 생각 글 · 코드에 데이터셋 이름이나 원래 출처를 쓴 것을 TabArena 처럼 기록 (`fab ta agent lineage` 같은 읽기). 점수는 고치지 않고 적음
- TFM 사용: `TFM_PATTERN` 에 시계열 FM (chronos · timesfm · moirai · toto · tirex · tabpfn_time_series · `TabICLForecaster`) 을 더해 `tfm_in_code` · `tfm_runtime` 에 남김. Agent 가 스스로 깐 FM 도 Agent 결과로 셈 (TabArena 결정 11 과 같음)
- 시계열 FM 의 사전학습 겹침은 `trained_on_this_dataset` 로만 앎 (공개 표시. 확인 방법 없음, 후보 10개는 우리 풀 FM 과 겹침 없음). TabPFN · TabICL 은 합성 사전학습 (각 논문)

## 10. 비용 어림

공개 결과의 `training_time_s + inference_time_s` (GPU 종류 모름) 로 어림. 후보 10개 · 모든 window.

| 실행 | 어림 | 비고 |
|---|---|---|
| Seasonal Naive · AutoETS | CPU 5분 | serverless CPU |
| AutoARIMA | 0 (안 돌림) | 공개 결과만 (결정 5) |
| Chronos-2 · Chronos-Bolt · TimesFM-2.5 | GPU 5분 미만 | H100 1장 |
| TabPFN-TS (v2) | GPU 약 1.5시간 (공개 TabPFN-TS 90분) | H100 1장 |
| TabICL forecaster | GPU 1~3시간 (공개 결과 없음, TabPFN-TS 와 비슷하다고 봄. 로컬 작은 task 로 먼저 잼) | 〃 |
| **합** | **GPU 약 4~8 H100-시간 + CPU 수 분** | 실행 전 확인받음 |
| Agent 파일럿 | 2 task × 5설정 × 1시간 = A10 10시간 + 토큰 | 실행 전 확인받음 |
| Agent 나머지 | 8 task × 5설정 = A10 40시간 + 토큰 | 파일럿을 보고 |

## 11. 구현 계획

| 부분 | 파일 | 한 것 / 할 것 |
|---|---|---|
| 옆 환경 | `FAB_FORECAST` (절대경로, 예: `…/projects/fm-agent-bench-project/forecast`): `.venv` (fev 0.10.0 · datasets), `data/cache` (HF), `data/answers`, `fev-repo` (ecce671) | 만듦 (조사용). uv.lock 은 안 바꿈 |
| 데이터 준비 | `fab/bench/forecast/driver.py` (옆 venv 에서 `-I`, fab 을 import 안 함) | fev `Task` → 마지막 window 의 past · future · 정답 → CSV, 모든 window 의 cutoff 기록 |
| 대회 · 채점 | `fab/bench/forecast/__init__.py` · `remote.py` | `fab fc prepare · check · describe · fm · reference · elo · agent`. 채점 = 4 의 numpy 식, 원격 정리 작업의 `bench: forecast` 요청 (TabArena 의 `grade_tabarena` 옆에 같은 꼴) |
| 원격 FM · 기존 방법 | `fab/bench/forecast/run_fev.py` (원격 실행 스크립트) + 제출은 `fab/runner/databricks.py` 의 `submit` · `submit_cpu` 를 재사용 | task × 방법마다 모든 window 예측 → window 별 점수 + 마지막 window 예측 (Agent 와 같은 채점기로 다시 채점) → Volume 의 `outputs/forecast/` |
| Elo | `fab/bench/tabarena/elo.py` | `anchor` 인자. TabArena 비트 단위 확인 |
| Agent | `fab/agent/__init__.py` (bench=forecast: `fc_target`, 표 `results/forecast_runs.csv`), `runner.py` (`check_submission` 의 (id, timestamp)), `TFM_PATTERN` · `DATA_SOURCE_PATTERN` | 지시문 · 실행기는 그대로 |
| CLI | `fab/cli.py` 에 `fc` 한 줄 | 이상탐지 트랙과 겹치는 곳 (rebase 로 맞춤) |
| 결과 | `results/forecast.csv` (우리 실행 window 별 · task 평균), `forecast_reference.csv` (공개 결과), `forecast_runs.csv` (Agent), `forecast_elo.csv` | |
| 로컬 확인 | 원격 전에 | 작은 task (ecdc_ili) 로 준비 → 정답 = fev `get_ground_truth`, Seasonal Naive 로컬 예측의 우리 채점 = fev 채점 = 공개 값, sample_submission 이 형식 검사를 통과, 누출 검사 |

## 12. 구현 (2026-11-25)

11 의 계획대로 만듦. 계획과 다른 것 · 구현하며 정한 것.

| 부분 | 한 것 |
|---|---|
| 파일 | `fab/bench/forecast/`: `__init__.py` (대회 · 채점 · CLI `fab fc`), `driver.py` (옆 venv, fev 로 window 꺼내기 · fev 로 채점 확인), `metrics.py` (fev 식 numpy), `remote.py` (정답 업로드 · 채점기), `run_fev.py` (원격 실행기), `describe.py` (설명 만들기), `tasks.yaml` (fev task 정의 10개, fev @ ecce671 그대로). 대회 `competitions/fc-<task>/` (config.yaml · description.md) 10개. 테스트 `tests/test_forecast.py` |
| 옆 환경 | `FAB_FORECAST` = `…/projects/fm-agent-bench-project/forecast` (절대경로): `.venv` (fev 0.10.0), `.venv-fm` (TabPFN-TS 1.4.0 · TabICL 2.2.0, 로컬 시험용), `hf/` (HF 캐시), `data/cache/<task>/w<window>/` (원래 id), `data/answers/` (정답 · 계절 오차 · id 대응표) |
| 대회 | 행 = (id, timestamp). 시리즈 id 는 `s001…` (task 이름에서 정한 seed 의 무작위 순서, 대응표는 정답 json 에만). static 공변량은 train.csv 에 시리즈마다 같은 값으로. 계절 오차 s_i 는 준비할 때 train 에서 계산해 정답 csv 의 `scale` 열로 (채점 때 train 을 안 읽음) |
| 채점 | `score_frame`: 열 이름 · 행 수 · (id, timestamp) 짝 (시각으로 비교: `2024-01-01` = `2024-01-01 00:00:00` = ISO `T`) · 숫자 · NaN 을 보고 4 의 식. `metric_error` = `score` = SQL, 표에 MASE · WQL · WAPE 도 |
| 형식 검사 서버 | `fab/agent/runner.py` 의 `check_submission`: sample 에 `timestamp` 열이 있으면 (id, timestamp) 짝 집합 (시각으로) 과 예측 열이 숫자인지까지. 없으면 예전과 같음 (로컬 확인 11가지 경우) |
| 원격 FM · 기존 방법 | `fab fc fm <task...>`: 방법 묶음마다 `databricks air run` (H100 1장, 내 신원, 환경 6) 하나. 실행기가 방법마다 venv (`uv venv` + `uv pip install fev==0.10.0 <패키지>`) 를 만들고 task 마다 worker 를 부름. 기본 묶음 셋: `tabpfn_ts` / `tabicl_ts` / `snaive autoets chronos2 chronos_bolt timesfm25`. 제출 기록은 `state/forecast/`, 정리 작업의 새 단계 `forecast` 가 회수 → `results/forecast.csv` (window 별 fev 점수 `scorer=fev`, task 평균 `window=all`, 마지막 window 를 Agent 와 같은 채점기로 `scorer=ours, is_last`) |
| fev 래퍼 | 공개 결과가 나온 commit (`f323f6c` · `aa66862`) 이 아니라 **스냅샷 `ecce671` 의 래퍼 하나로** (`aa66862` 에는 timesfm-2.5 래퍼가 없음). 래퍼의 `requirements.txt` 를 그대로 깔고, 패키지 저장소에 고정 버전이 없으면 모델 패키지 (fev · statsforecast · chronos-forecasting · timesfm · tabpfn-time-series · tabicl · torch) 의 고정은 두고 나머지만 풀어 다시 (`relaxed_pins` 열). 로컬 사내 프록시에 `autogluon.timeseries==1.5.0` 이 없어 statsforecast 래퍼에서 이렇게 됨 (래퍼가 쓰는 것은 주기 → seasonality 표 하나) |
| TabPFN v3.5 → v2 (2026-11-25, 사용자) | 처음엔 `tabpfn-time-series` 1.4.0 의 기본 (v3.5) 을 계획했지만 v3.5 가중치는 `TABPFN_TOKEN` (API 키) 이나 브라우저 로그인으로 라이선스를 수락해야 받음 → **v3.5 는 빼고** 공개 결과 "TabPFN-TS (v2)" 의 fev 래퍼 그대로 (수락 없음, 공개 결과와 맞춰 볼 수 있음). TabPFN-TS-3 (v3 시계열 체크포인트) 은 공개 결과로 FM 풀에만 |
| TabICL | fev 래퍼가 없어 `run_fev.py` 의 `tabicl_ts`: window 마다 `convert_input_data(window, "pandas")` → `TabICLForecaster().predict_df(context, future_df)` (known 공변량만, 기본 체크포인트 `tabicl-regressor-v2-20260212.ckpt` · HF `jingang/TabICL`, 점 예측은 기본 `mean`) |
| 사후 검사 | `TFM_PATTERN` 에 chronos · timesfm · moirai · toto · tirex, `fm_audit.WATCH` 에 시계열 FM 모듈 (tabpfn_time_series → `tabpfn_ts`), `DATA_SOURCE_PATTERN` 에 fev 데이터 묶음 · `import fev` · 후보 10개의 원래 출처 주소 (GiftEval · redset · EPF zenodo · neuralprophet-data · ECDC · BEA). 기존 TabArena · MLE-bench 행은 `recheck` 를 다시 돌리지 않음 (회수 기록이 Volume 에만 있음) — 새 이름이 옛 실행에 나올 일은 드묾 |
| Elo | `elo(…, anchor=)` · `fill(…, anchor=)`. 기본값 RandomForest 그대로 → `uv run --frozen fab ta elo` 로 `results/tabarena_elo.csv` 가 바이트 단위로 같음 (2026-11-25 확인) |

**정리 작업 주의 (머지 전)**: 정리 작업 코드는 `fab remote kick` 을 낸 체크아웃의 것. 이 브랜치가 머지되기 전 다른 worktree (예: `tabarena-diabetes130us`) 에서 kick 하면 그 코드에는 `forecast` 단계가 없어 FM 작업은 다음 kick 까지 남고 (해는 없음), **forecast 에이전트 실행은 MLE-bench 채점기로 가 3번 실패하면 오류로 끝남** → Agent 파일럿을 낸 뒤에는 이 브랜치에서만 kick (아니면 머지 뒤에)

로컬 확인 (원격 없이, 2026-11-25)

- `fab fc check all`: 10개 모두 우리 채점 = fev 채점 (위), 정답 = fev `get_ground_truth`, 엉터리 제출 4가지 무효, 누출 0. 누출 검사는 설명 · 시리즈 id · 열 머리 · 코드 묶음 7개 파일 · run.json · 실행 이름. 심은 누출 3가지 (run.json 의 task 이름, cid 의 대회 이름, 원래 시리즈 id) 는 잡음
- `run_fev.py` 를 로컬 CPU 로 ecdc_ili 에: Seasonal Naive task 평균 SQL 3.7763740396 (공개 3.7763740412) · MASE 4.1680850511 (공개 같음), AutoETS SQL 4.0788004372 (공개 4.0788004375). TabICL-TS 는 SQL 2.290 (공개 결과 없음, CPU 83초). 마지막 window 를 정리 작업의 채점기 (`collect`) 로 다시 채점하면 fev 점수와 같음
- TabPFN-TS (v2) 를 로컬 CPU 로 ecdc_ili 에: task 평균 SQL 2.3379 (공개 2.3365, 상대 0.06%) · MASE 2.7485 (공개 2.7528), 140초. 가중치는 수락 없이 받음
- 공개 결과만의 FM 풀 Elo (SQL, 10 task): Chronos-2 1752 · TabPFN-TS-3 1678 · TimesFM-2.5 1638 · TabPFN-TS (v2) 1535 · Chronos-Bolt 1410 · AutoARIMA 1291 · AutoETS 1187 · Seasonal Naive 1000 (우리 실행을 더하기 전, `results/forecast_elo.csv`)

## 결정 (2026-11-25)

1~5 는 사용자가 선택지로 정함, 6~9 는 추천안을 그대로 받음. 바꾸려면 여기를 고침.

| # | 정할 것 | 정한 것 | 까닭 |
|---|---|---|---|
| 1 | 벤치 · task | **fev-bench 의 단변량 10개 (2 의 표), Agent 파일럿 redset_1H · solar_with_weather_1H** | 2 의 규칙 |
| 2 | Agent 분할 | **마지막 window 하나, 1회** | 3. window 여럿은 격리가 안 됨 |
| 3 | 주 지표 | **SQL (보조 MASE)** | fev-bench 의 `eval_metric`, 분위수 예측까지 봄. 점 예측만 잘하는 Agent 는 MASE 로 따로 보임 |
| 4 | 기준점 | **Seasonal Naive = 1000** | fev-bench skill score 의 기준 (사용자 허락) |
| 5 | 기존 방법 | **SN · AutoETS · Chronos-2 · Chronos-Bolt · TimesFM-2.5 는 우리가 다시 돌림 (공개 결과는 맞춤 확인 · FM 풀). AutoARIMA 는 공개 결과로 FM 풀에만** | window 별 공개 값이 없음. ARIMA 는 CPU 약 4시간이라 Agent 풀에서 뺌 (사용자) |
| 6 | FM | **TabPFN-TS (fev 래퍼, TabPFN v2) · TabICL 2.2.0 forecaster, 기본 설정.** TabPFN v3.5 는 뺌 (2026-11-25, 사용자: 라이선스 수락이 필요해서) | 공식 둘. 변형 (`~tsfeat`) 은 그다음 |
| 7 | 채점 | **fev 식을 numpy 로 옮겨 원격 정리 작업에서, fev 와 같은 값인지 로컬 확인** | uv.lock 을 안 바꿈 |
| 8 | 익명화 | **이름 · 출처 · 원래 id 숨김, 실제 timestamp · 공변량 이름은 줌** | 8 의 표 |
| 9 | 대회 접두사 · 명령 | **`competitions/fc-<task>/`, `fab fc …`, 익명 id `forecast-p01`…** | TabArena 의 `ta-`·`fab ta`·`tabarena-p01` 자리 |

## 확인 못 한 것

- fev-bench 리더보드 화면의 집계 (HF Space 가 동적)
- 공개 결과를 낸 GPU 종류 (시간 어림의 근거가 약함)
- 후보 10개 각 데이터의 원래 라이선스 (HF 표시는 `other`), 설명을 쓸 때 원래 출처 글을 옮겨 써도 되는지 → 우리 말로 짧게 씀
- TabICL forecaster 의 기본 체크포인트가 표 형식 트랙의 `tabicl` (TabICLv2) 과 같은지, 가중치 라이선스
- fev 0.8.0 (공개 결과) 과 0.10.0 (우리) 의 지표 계산이 같은지 → 로컬에서 Seasonal Naive 로 확인할 것
