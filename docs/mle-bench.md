# MLE-bench: 하네스 × LLM 에이전트 vs TFM

kaggle-master 의 `docs/plans/mle-bench.md` 를 옮긴 것 (2026-11-10 까지의 계획·결정·P0 결과). 명령·경로는 이 저장소 이름으로 바꿈.
앞으로의 단계는 [plans/roadmap.md](plans/roadmap.md).

두 가지 질문:

1. ML 에이전트(하네스 × LLM)가 몇 시간 들여 만든 풀이와, 튜닝 없는 TFM 한 번 추론의 성능이 얼마나 차이 나는가
2. 에이전트가 만든 전처리(피처)를 TFM 에 넣으면 TFM 성능이 오르는가 — 오른다면 GBDT 보다 더 오르는가

## 전제

- MLE-bench(`openai/mle-bench`)는 원래 train 을 다시 나눠 자체 test(private 정답)를 만들고 `mlebench grade-sample` 로 채점.
  메달·중앙값 기준선은 원래 Kaggle 리더보드에서 가져옴. → TFM 도 Kaggle LB 가 아니라 **이 test 로 채점**해야 에이전트와 비교됨.
- 공개 리더보드 에이전트(약 25개 조합, 24시간·A10 GPU)는 `runs/` 에 대회별 채점 결과(점수·기준선·메달)가 있음 (Git LFS,
  `media.githubusercontent.com/media/openai/mle-bench/main/runs/...` 로 받음). 풀이 코드는 없음.
- 표 형식 대회 범위 (순수 표 형식 우선, 시계열·분자는 나중에)

| 대회 | 세트 | 비고 |
|---|---|---|
| spaceship-titanic | dev | 저장소에 `features.py` 있음 → 파일럿 |
| playground-series-s3e18 | dev | 다중 타깃(2개) |
| nomad2018-predict-transparent-conductors | low | 작음(약 2.4천 행), 타깃 2개 (RMSLE 평균) → 파일럿 |
| tabular-playground-series-dec-2021 | low | 큼(수백만 행), 리더보드가 촘촘해 메달 판정이 민감 |
| tabular-playground-series-may-2022 | low | train 80만 행 (MLE-bench 분할), 금~동 폭 AUC 0.00005 → P3 (아래 "P3 준비") |
| new-york-city-taxi-fare-prediction | low | 아주 큼 → 컨텍스트 샘플링 필수 |
| leaf-classification | low | 99 클래스 → TabPFN v2 는 못 씀 (10 클래스), TabICL·Kumo 는 됨 → P3 (아래 "P3 준비"·"P3 결과") |

dev 세트 대회는 공개 리더보드 채점 결과에 없음 → 트랙 B 로만 비교.

## 질문 1: 비교 트랙

- **트랙 A (공개 결과)**: 공개 에이전트 점수 vs TFM (같은 MLE-bench test). 거의 무료.
- **트랙 B (직접 실행)**: 1회 **1시간** 예산. 풀이 코드가 남아 질문 2 의 입력이 됨.

| 하네스 | LLM | 비교 축 |
|---|---|---|
| AIDE | Sonnet 5.5 | LLM 축 (하네스 고정) |
| AIDE | GPT (Sonnet 급, 모델명 미정) | LLM 축 |
| Claude Code | Sonnet 5.5 | 하네스 축 (LLM 고정) |
| Codex CLI | GPT (위와 같은 모델) | 하네스 축 |

- 시드 2 → 대회 7 × 설정 4 × 시드 2 = 56회 × 1시간 ≈ GPU 56시간 + API 비용 (P0 에서 실측 후 확정)
- 1시간은 공개 리더보드(24시간)보다 훨씬 짧음 → 트랙 A·B 점수를 섞어 순위 매기지 않음. 큰 대회(dec-2021·may-2022·taxi)는
  1시간이면 데이터 읽기 + 모델 한두 개가 한계일 수 있음 → P0 에서 확인.
- 실행마다 에이전트가 쓴 모델(특히 TabPFN 등 TFM 사용 여부)을 기록.

## 질문 2: 에이전트 피처 → TFM

1. **추출**: 에이전트 최종 풀이에서 전처리만 떼어 `build(train, test) -> (X, X_test, cat_cols)` 로 옮김 (LLM 변환 + 사람 검토)
2. **동일성 확인**: 옮긴 피처 값이 에이전트의 피처 배열과 같은지 + GBDT 의 F2 가 F0 보다 나은지 (2026-11-11 바꿈.
   원래 "에이전트 모델을 다시 돌려 점수 재현" 은 nomad × Codex 처럼 점수가 구조 모델에서 나오면 못 씀, [roadmap](plans/roadmap.md) 의 C1)
3. **누수**: 타깃을 쓰는 피처(target encoding, OOF 스태킹)는 지금 형식(train+test 전체 계산)에선 CV 누수 → 1차는 타깃 비의존 피처만,
   2차에 fold 별 fit 형식(`build(train, test, y)`)으로 확장
4. **조건**: F0 원본 최소 피처 / F1 사람 피처 / F2… 에이전트 피처 × {tabpfn_v2, tabicl, kumo} + 대조군 {lgbm, catboost}
5. **지표**: (TFM 의 F0 대비 향상) − (GBDT 의 F0 대비 향상). 양수일 때만 "TFM 에 특히 도움". 피처 수와 `tfm_max_features` 적용 여부도 기록.
   점수는 같은 fold 의 CV + MLE-bench test 채점 둘 다.

## 저장소 변경

- `config.yaml` 에 `source: mlebench` → `fab download` 가 `mlebench prepare` 를 부르고 public 데이터만 `data/` 에 둠 (private 정답은 에이전트 작업 공간 밖)
- 모델 이름 뒤 `~<variant>` (예: `catboost~f2`) → `competitions/<slug>/features/<variant>.py`, 결과에 `features`·`n_features` 기록
- `fab mleb grade` → test.npy 로 제출 파일 → `mlebench grade-sample` → `results/mlebench.csv`
- `fab mleb agents` → 공개 채점 결과 수집 (표 형식 대회만)
- `fab mleb agent <harness> <llm> <slug> --hours 1` → 원격 실행, 코드·로그 보관
- `fab mleb extract <run>` → 피처 추출 + 동일성 확인
- 다중 타깃(nomad, s3e18), 많은 클래스(leaf), board 의 MLE-bench 탭

## 단계

| 단계 | 내용 | 완료 기준 |
|---|---|---|
| P0 | spaceship + nomad2018 로 준비 → TFM → 채점, 에이전트 1회(1시간) + 추출 | 끝까지 돌고 실제 비용 측정 |
| P1 | 표 형식 대회 전체 TFM vs 트랙 A | 대회별 TFM × 에이전트 표, 메달 여부 |
| P2 | 트랙 B 행렬 | 설정 4 × 시드 2 |
| P3 | 에이전트 피처로 TFM 재실행 | F0/F1/F2 × 모델 표, 향상폭 차이 |
| P4 | EXPERIMENTS.md 정리 | |

## 사용법 (P0 에서 만든 것)

```bash
# mle-bench 는 ../mle-bench 에 따로 설치 (fab/bench/mlebench/__init__.py 위쪽 설명)
uv run fab download mleb-spaceship-titanic            # Kaggle zip → MLE-bench 분할 → 체크섬 확인 → public 만 data/
uv run fab databricks mleb-spaceship-titanic catboost tabpfn_v2   # 학습은 Databricks 에서만 (로컬 fab run 은 막힘, 2026-11-10)
uv run fab mleb grade spaceship-titanic catboost tabpfn_v2 --each   # 채점 요청 (Volume 에 기록만)
uv run fab remote kick                                              # 학습이 끝난 뒤: 결과 합치기·채점
uv run fab sync                                                     # → results/mlebench.csv (점수·메달·원래 LB 순위)
uv run fab mleb agents                                 # 공개 에이전트 점수 → results/mlebench_agents.csv
```

- 타깃이 둘인 nomad2018 은 fab 대회 2개(`mleb-nomad2018-formation`, `-bandgap`) → `fab mleb grade` 가 열을 합쳐 한 제출로 채점
- `runs/run_group_experiments.csv` 는 LFS 포인터가 병합 충돌 난 채로 올라가 있어 두 버전을 모두 받아 합침
  (한쪽에만 MARS·MARS+·AIBuildAI, 다른 쪽에만 Famou-Agent-2.0 gemini-3-pro)

## P0 결과 (2026-11-09, 튜닝 없음, 원본 피처)

nomad2018 (열별 RMSLE 평균 ↓, 금 0.05589 · 중앙값 0.06988, 원래 LB 879팀)

| 모델 | MLE-bench test | LB 순위 | 공개 에이전트 33개 실험(시드 중앙값) 중 |
|---|---|---|---|
| kumo | 0.05734 | 17 | 8위 (앞: MARS+ 0.0499, MARS, LoongFlow, Disarray, PiEvolve 24h, AIRA-dojo, Famou-2.0 gemini-3-pro 0.0568) |
| tabicl | 0.05838 | 22 | |
| tabpfn_v2 | 0.05874 | 25 | |
| catboost | 0.06001 | 31 | |
| lgbm | 0.06107 | 40 | |
| xgb | 0.06434 | 76 | |

- TFM 은 결정 구조 파일(geometry.xyz) 없이 원본 열만, 몇 초~수십 초. 에이전트는 12~24시간.
- R&D-Agent(gpt-5) 0.0607, AIBuildAI(Opus 4.6) 0.0626, AIDE(o1-preview) 0.0600 보다 kumo·tabicl·tabpfn_v2 가 좋음
- TFM 은 1회 실행, 에이전트는 시드 중앙값 → 같은 기준이 아님 (TFM 의 시드 분산은 따로 재야 함)

spaceship-titanic (accuracy ↑, test 870행 → 1행 = 0.11%p, 공개 에이전트 결과 없음): catboost 0.8241 · kumo 0.8218 ·
tabpfn_v2 0.8207 · tabicl 0.8184 · xgb 0.8184 · lgbm 0.8172. CV 1위(kumo 0.8254)와 test 1위가 다름.

## P0 트랙 B 결과 (2026-11-09, 1시간 · A10 · 1회씩)

`results/mlebench_runs.csv`. 점수는 MLE-bench test, LB 순위는 원래 Kaggle 리더보드에서의 위치.

nomad2018 (RMSLE ↓, 금 0.05589, 879팀)

| 설정 | 점수 | LB | 하네스 시간 | LLM 호출 | 비고 |
|---|---|---|---|---|---|
| Codex × GPT-6.1 Terra | **0.04847** | **1** | 59분 (스스로 끝냄) | 350 | 결정 구조 파일로 대칭·국소 환경(SOAP, `dscribe`)·그래프 피처, `ase`·`dscribe`·`torch-geometric` 직접 설치. 429 대기 4분 |
| AIDE × Sonnet 5.5 | 0.05494 | 6 | 60분 (시간 종료) | 7 | |
| (kumo, 원본 열만) | 0.05734 | 17 | 22초 | – | 트랙 A |
| Claude Code × Sonnet 5.5 | 0.06134 | 42 | 20초 (스스로 끝냄) | 4 | 원본 열 + 부피, LightGBM 하나 |
| (별도) Claude Code × Sonnet **+cont** | 0.05267 | 4 (금) | 5분 20초 (이어 가기 20번 상한) | 37 | 아래 |
| AIDE × GPT-6.1 Terra | 제출 없음 | – | 60분 (시간 종료) | 66 | GPT 가 `if __name__ == "__main__"` 로 감싼 코드를 AIDE 가 빈 전역 범위로 exec → main() 이 안 돎, 28개 중 27개 버그 |

공개 리더보드 최고(MARS+, 24시간·H100 2장) 시드 중앙값 0.0499 → Codex × GPT 1시간이 그보다 좋음 (1회 결과라 시드 분산은 모름)

spaceship-titanic (accuracy ↑, 금 0.82066, 2288팀, test 870행 → 1행 = 0.11%p)

| 설정 | 점수 | LB | 하네스 시간 | LLM 호출 |
|---|---|---|---|---|
| (catboost / kumo / tabpfn_v2) | 0.8241 / 0.8218 / 0.8207 | 2 / 6 / 11 | 26초 / 47초 / 100초 | – |
| Codex × GPT | 0.8184 | 41 | 58분 (스스로 끝냄) | 338 (429 대기 7분) |
| AIDE × GPT | 0.8184 | 41 | 60분 (시간 종료) | 51 |
| Claude Code × Sonnet | 0.8161 | 58 | 18초 (스스로 끝냄) | 4 |
| (별도) Claude Code × Sonnet **+cont** | 0.8115 | 199 | 3분 14초 (이어 가기 20번 상한) | 30 |
| AIDE × Sonnet | 0.8115 | 199 | 60분 (시간 종료) | 17 |

- 비용: 채점에 넣은 8회의 하네스 시간 합 A10 약 6시간 (+ 실행마다 설치 약 2분), 입력 토큰 8,800만 (대부분 Codex, 그중 97% 캐시), 출력 84만.
  제외한 실행 8회는 대부분 시작 직후 실패 (AIDE 3·Codex 1 은 각 1시간 가까이)
- 하네스 성격: Codex 는 시간을 거의 다 쓰며 수백 번 호출, Claude Code 는 20초·4턴 만에 끝냄, AIDE 는 코드 실행 시간이 길어 호출이 적음
  → Claude Code 는 남은 시간을 이어 쓰는 별도 설정(+cont)으로 한 번 더 돌림 (아래 실행 환경의 "스스로 끝낼 때", kaggle-master#35)
- TFM: 어느 실행도 TFM 을 쓰지 않음 (Codex 가 spaceship 에서 TabPFN·AutoGluon 설치 여부만 확인)
- 질문 2 의 첫 대상: nomad × Codex 의 결정 구조 피처 → TFM

+cont (2026-11-10, kaggle-master#35, 1회씩): 이어 가기는 원격에서 동작했지만 두 대회 모두 **20번 상한**에 걸려 1시간 중 3~5분만 씀.
Sonnet 이 이어 가라는 글에 "제출은 확정, 더 안 바꿈" 으로 한 턴(약 2초)에 끝내는 일이 많음

| 대회 | 첫 종료 | 실제로 작업한 이어 가기 | 거절(도구 0개)한 이어 가기 | 마지막 작업 |
|---|---|---|---|---|
| spaceship | 24초 | 4번 (CV·CatBoost 설정·피처·LightGBM 혼합) | 16번 (마지막 11번 연속) | 2분 56초 |
| nomad | 21초 | 6번 (LightGBM+CatBoost 혼합, `geometry.xyz` 이웃 거리·동경 분포 피처) | 14번 (마지막 7번 연속) | 5분 4초 |

- nomad 는 0.06134(42위) → 0.05267(4위, 금): 결정 구조 피처를 이어 가기에서 처음 씀. 공식 설정의 Claude Code 는
  "약해서" 가 아니라 "일찍 끝내서" 낮았음. Codex 1시간(0.04847)보다는 못함
- spaceship 은 0.8161 → 0.8115 (test 870행에서 4행 차이, 잡음 범위). CV 0.812 에서 더 못 올림
- 토큰: spaceship 109만/7천, nomad 144만/1.2만
- 상한을 바꿈 (2026-11-11): **도구 0개로 끝난 구간이 연속 3번이면 멈춤** (`idle`), 안전장치로 전체 40번 (`max`). 근거는 위 두 실행:
  - 거절 뒤 다시 작업하는 경우가 있음 (nomad 는 거절 5번 연속 뒤 13번째에 다시 작업, spaceship 은 4번 뒤 9번째) → 연속 1번은 짧음
    (nomad 의 제출 개선 3번을 놓침)
  - 연속 2·3·5번은 두 실행 모두 최종 제출이 같음 (제출이 바뀐 건 nomad 7번째, spaceship 1번째 이어 가기가 마지막). 다시 작업한
    구간도 점수가 안 올라 제출을 안 바꿈 → 여유를 하나 둔 3. 5 는 거절로 호출만 늘림
  - 무작업 = 그 구간의 도구 호출 0개 (Claude Code 의 `tool_use`, Codex 의 명령 실행 등 item). "제출이 안 바뀜" 으로 세면 실험 뒤
    일부러 안 바꾼 구간까지 무작업이 되어 쓰지 않음
  - 상한 없이 남은 시간 내내 이어 가면 거절 한 번(약 2초)마다 대화 전체를 다시 보내 1,500번 넘게 부름 (입력 토큰 수천만 이상, 429)
  - 이어 가기 글은 그대로 (더 구체적으로 시키면 코칭이 되어 별도 설정)
- 이 2회는 다시 돌리지 않음: 기록을 구간별로 다시 재생하면 새 규칙에서는 nomad 11구간(이어 가기 10번, 약 4분 20초),
  spaceship 8구간(7번, 약 2분 30초)에서 멈추고 최종 제출은 그대로 → 점수 같음. `harness_seconds`·`continues`·토큰은 옛 규칙(20번)
  값이라 P1 의 2회째와 이 열들은 섞어 보지 않음
- 해석: 어느 상한이든 Claude Code +cont 는 사실상 3~5분 설정. 남은 시간을 알려 줘도 모델이 할 일을 더 찾지 못함

## P1 결과: Agent 2회째 (2026-11-11, 1시간 · A10)

같은 설정을 한 번 더 (rep 2). +cont 는 새 상한 (도구 0개 구간 연속 3번이면 멈춤, 위 P0 의 +cont). 순위는 원래 Kaggle 리더보드 위치.

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

- nomad 는 설정 순위가 두 번 다 같음 (Codex > +cont > AIDE > Claude Code). 점수 흔들림은 0.001~0.003. 1회째에 제출이 없던 AIDE × GPT
  만 이번엔 제출함 → 실패가 설정의 성질이 아니라 실행마다의 일
- spaceship 은 1·2회째 차이가 0.000~0.010 으로 정확도 표준오차(약 0.013) 안 → "방식 사이에 뚜렷한 차이 없음" 그대로.
  Codex 2회째 0.82414 는 FM·GBDT 최고(catboost 0.82414)와 같음
- +cont 의 새 상한: 두 실행 모두 `idle` 로 멈춤. nomad 는 무작업 1번 뒤 7번째 구간에서 14분을 더 작업함 (연속 1번 상한이었으면
  놓쳤을 작업). 구간별 도구 수: spaceship [2, 1, 3, 0, 1, 0, 0, 0], nomad [2, 4, 2, 2, 1, 0, 3, 0, 0, 0]
- 429 는 여전히 Codex 만 (34~56번). 어느 실행도 TFM 을 쓰지 않음

## C1 결과: 에이전트 피처 → FM (2026-11-11, nomad × Codex)

F2 의 정의·추출은 [roadmap](plans/roadmap.md) 의 "nomad × Codex 의 F2". MLE-bench test (열별 RMSLE 평균 ↓), 괄호는 원래 리더보드 순위와
F0 대비 변화. 튜닝 없음, 실행 1회.

| 모델 | F0 (원본 12열) | F2s (1,109열) | F2 (38,832열, TFM 은 상위 500) |
|---|---|---|---|
| kumo | 0.05734 (17위) | 0.04803 (1위, −16.2%) | **0.04523 (1위, −21.1%)** |
| tabpfn_v2 | 0.05874 (25위) | 0.04972 (1위, −15.4%) | 0.04929 (1위, −16.1%) |
| tabicl | 0.05838 (22위) | 0.04972 (1위, −14.8%) | 0.04931 (1위, −15.5%) |
| catboost | 0.06001 (31위) | 0.04976 (1위, −17.1%) | 0.04992 (1위, −16.8%) |
| lgbm | 0.06107 (40위) | 0.04998 (1위, −18.2%) | 0.05102 (2위, −16.5%) |
| (참고) Codex × GPT 1시간 | 0.04847 / 0.04567 (1·2회째) | | |

CV (같은 fold 5개, formation / bandgap): F0 → F2s → F2

| 모델 | formation | bandgap | 학습 시간 (두 타깃 합, 초) |
|---|---|---|---|
| kumo | 0.0316 → 0.0194 → 0.0195 | 0.0841 → 0.0741 → 0.0730 | 45 / 71 / 898 |
| tabpfn_v2 | 0.0321 → 0.0212 → 0.0203 | 0.0856 → 0.0753 → 0.0749 | 71 / 207 / 1,008 |
| tabicl | 0.0317 → 0.0212 → 0.0205 | 0.0845 → 0.0762 → 0.0753 | 13 / 125 / 916 |
| catboost | 0.0353 → 0.0235 → 0.0235 | 0.0912 → 0.0801 → 0.0805 | 25 / 1,398 / 40,442 |
| lgbm | 0.0359 → 0.0229 → 0.0227 | 0.0942 → 0.0806 → 0.0800 | 6 / 200 / 6,031 |
| (참고) Codex 최종 풀이 | 0.0179 | 0.0707 | |

- **에이전트 피처 + FM 한 번이 에이전트 1시간을 넘음**: `kumo~f2` 0.04523 < Codex 0.04567·0.04847. 같은 피처의 GBDT 도 모두 금메달 선
  (0.05589) 안. 에이전트가 만든 것 중 점수에 든 몫의 상당 부분이 피처였음 (모델 몫: Codex 의 CNN·GNN 은 F2 로 옮기지 않음)
- **"FM 에 특히 도움" (C1 지표 = TFM 향상 − GBDT 향상) 은 갈림**:
  F2s 에서는 GBDT 가 더 좋아짐 (−17~18% 대 TFM −15~16%). F2 에서는 kumo 만 GBDT 보다 더 좋아짐 (−21% 대 −17%), tabpfn·tabicl 은 GBDT 와
  비슷 → 에이전트 피처는 FM 과 GBDT 를 비슷하게 돕고, FM 은 F0 때의 앞섬을 유지한 채 같이 올라감. FM 에 특히 이득이라는 근거는
  kumo × F2 하나 (실행 1회, 짝지은 비교라 노이즈는 작지만 확정은 아님)
- **F2 (전부) 는 F2s 보다 거의 안 나음** (kumo 만 test −0.003). 비용은 TFM 약 5~13배, GBDT 약 30배 (catboost 는 대회당 약 5.6시간) →
  사람이 읽을 수 있는 물리 피처(RDF·이웃 통계·대칭)만으로 대부분이 나옴
- 해석 주의: test 에 train 과 같은 구조가 있음 (Codex 가 정답을 섞던 행). SOAP 같은 기술자를 받은 TFM 은 비슷한 행을 보고 예측하므로
  그 효과를 피처로 정당하게 얻을 수 있음 → kumo × F2 의 추가 이득이 여기서 왔을 수 있음 (확인 안 함)

## P3 준비: tps-may-2022 · leaf-classification (2026-11-12)

fab 대회 2개를 더함. 분할은 MLE-bench 그대로 (`fab download`, 체크섬 일치), 결과는 아직 없음.

| fab 대회 | MLE-bench | train / test | 지표 | 메달 기준선 (원래 LB) | 공개 에이전트 (실험별 시드 중앙값) |
|---|---|---|---|---|---|
| `mleb-tps-may-2022` | tabular-playground-series-may-2022 | 800,000 / 100,000, 31열 | AUC ↑ | 금 0.99823 · 은 0.99822 · 동 0.99818 · 중앙값 0.97268 (1,152팀) | 31개 실험, 금은 Disarray 0.99825 하나. 다음 MLEvolve 0.99771 · MARS 0.99760 |
| `mleb-leaf-classification` | leaf-classification | 891 / 99, 192열, 99 클래스 | logloss ↓ | 금 0.0 · 은 0.00791 · 동 0.01526 · 중앙값 0.10835 (1,596팀) | 33개 실험 중 17개가 메달 선 (중앙값 기준), MARS·MARS+·Famou-2.0 gemini-2.5-pro 는 0.0 (금) |

- 피처
  - leaf F0: margin·shape·texture 192개 (대회가 이미지에서 미리 뽑은 수치). 잎 이미지(`images/`, 990장)는 안 씀 — nomad 의
    `geometry.xyz` 처럼 에이전트만 씀
  - tps F0: 원본 31열. `f_27` (10글자 문자열, 80만 행에 고유값 67만 개, test 값의 28% 만 train 에 있음)은 범주형 하나로 둠
    → TFM 에는 정수 코드(사전순)라 거의 정보가 없음
  - tps `~f1` (사람 피처): `f_27` 을 글자 위치별 10열 + 서로 다른 글자 수로 바꿈 (41열). 대회 당시 공개 노트북에서 널리 쓴 처리.
    F0 만으로는 FM 이 이 한 가지 처리를 못 해서 지는 비교가 되므로 같이 둠. C1 의 F2 (에이전트 피처) 와는 다름
- 클래스 수 (leaf): TabPFN v2 는 10 클래스까지라 fit 에서 거부 (`TabPFNValidationError`, tabpfn 9.2) → 표에서 뺌.
  TabICL 은 10 을 넘으면 계층 분류로 자동 전환 (`support_many_classes` 기본값). Kumo 도 됨 (P3 결과, 로컬에는 패키지가 없어 실행 전엔 몰랐음).
  클래스당 train 이 최소 6행이라 층화 5-fold 의 학습 쪽에 99 클래스가 모두 들어감
- 분류 확률은 예측할 때 행 합을 1 로 맞춤 (`fab/core/metrics.py` 의 `normalize_proba`, `fab run` 의 predict 와 제출 둘 다).
  TFM 은 float32·float16 확률이라 99 클래스면 합이 1 에서 벗어남 (float16 균등 확률이면 3e-5). leaf 채점기는 1 ± 1e-6 이 아니면 무효이고,
  sklearn 1.9 log_loss 는 경고만 하고 그대로 계산해 CV logloss 와 채점 점수가 다른 확률을 보게 됨. NaN·합 0·뚜렷한 음수는 에러.
  이 변경 전 실행은 다시 돌리지 않음: spaceship 의 지표 accuracy 는 argmax 라 행 합을 맞춰도 같음 (보조 지표 auc·logloss 는 조금 달라질 수 있음)
- TFM 컨텍스트 (tps): 대회 설정 `tfm_max_rows: 10000` (kaggle-master 관례), 큰 컨텍스트는 `@rand200000`. kaggle-master 의 비슷한 크기
  (S6E1·S6E10, 60~70만 행, H100) 에서 kumo@rand200000 약 33분, tabicl@rand200000 약 42분, catboost 약 35분 (5-fold 합)
- xgb 는 tps F0 에서 뺌: `f_27` 범주 67만 개 때문에 로컬 8코어 64만 행 20트리 28초 → 800트리 5-fold 면 CPU 4코어에서 몇 시간.
  lgbm 은 50트리 1.3초 (범주 분할 상한이 있음). `~f1` 은 범주가 없어 xgb 도 됨
- Agent 노드 (A10G, 12코어, 메모리 51GB, `/tmp` 500GB, P0·P1 실행 기록) 에서 tps (CSV 283MB) 는 문제없음
- 로컬 확인: 피처 모양 (leaf 891×192, tps F0 800,000×31 · `~f1` ×41), 합성 예측으로 제출 파일 → 로컬 mle-bench 채점이 두 대회 모두
  `valid_submission=True` (leaf 는 두 모델 평균까지)

실행 (FM: H100 1장 + serverless CPU, Agent: 같은 엔드포인트끼리는 한 번에 하나)

```bash
# FM
uv run --frozen fab databricks mleb-leaf-classification lgbm xgb catboost tabicl kumo
uv run --frozen fab mleb grade leaf-classification lgbm xgb catboost tabicl kumo --each
T=(lgbm catboost tabpfn_v2 tabicl kumo lgbm~f1 xgb~f1 catboost~f1 tabpfn_v2~f1 tabicl~f1 kumo~f1 kumo@rand200000~f1 tabicl@rand200000~f1)
uv run --frozen fab databricks mleb-tps-may-2022 "${T[@]}"
uv run --frozen fab mleb grade tabular-playground-series-may-2022 "${T[@]}" --each

# Agent (Sonnet 줄과 GPT 줄을 하나씩 동시에, 대회마다)
uv run --frozen fab mleb agent run leaf-classification aide-sonnet      # → claude-code-sonnet → claude-code-sonnet --continue
uv run --frozen fab mleb agent run leaf-classification codex-gpt        # → aide-gpt
```

## P3 결과: tps-may-2022 · leaf-classification (2026-11-12, 1회씩)

MLE-bench test, 괄호는 원래 Kaggle 리더보드 순위. FM 은 튜닝 없음. Agent 는 1시간 · A10 · 1회 (P1 처럼 2회째는 아직 없음).
"공개 실험 중" = 공개 리더보드 에이전트(24시간) 실험별 시드 중앙값 사이 위치 (`results/mlebench_agents.csv`, 섞어 순위 매기지 않고 참고만).

leaf-classification (logloss ↓, 1,596팀, 금 0.0 · 은 0.00791 · 동 0.01526 · 중앙값 0.10835)

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

- **GBDT 는 이 대회에서 안 됨**: 클래스당 학습 약 7행 (fold 학습) 이라 트리가 거의 못 배움. FM 은 같은 192열로 GBDT 최고 (lgbm 0.406) 의 1/8 (tabicl)~1/20 (kumo).
  FM 대 GBDT 차이가 지금까지 대회 중 가장 큼 (nomad 는 GBDT 와 비슷, spaceship 은 GBDT 가 1위)
- **Kumo 는 99 클래스를 받음**, TabICL 은 계층 분류. TabPFN v2 는 10 클래스까지라 안 돌림
- **Codex 의 0.0 은 분할 구조를 쓴 결과**: 원래 데이터가 종마다 정확히 10장이라는 점에서 test 의 종별 개수 = 10 − train 개수를 계산하고,
  test 99행을 종별 자리에 1:1 로 배정 (`scipy.optimize.linear_sum_assignment`, 점수는 192열 + 이미지 거리 모델) → 98/99행에 확률 1.0.
  외부 데이터·Kaggle 풀이는 안 봄 (`kaggle_urls` 0). 공개 리더보드 1위권 (MARS 등 0.0) 도 비슷한 방법일 수 있음 (확인 안 함).
  nomad 의 "정답 섞기" 와 같은 종류 (데이터 수집 방식에서 오는 사전 정보) → 에이전트 대 FM 비교에서 따로 읽음.
  같은 배정을 FM 확률에 얹으면 FM 도 0.0 근처가 될 수 있음 (안 해 봄, C1 처럼 "모델 몫이 아닌 몫")
- AIDE 둘은 배정을 안 씀: 잎 이미지 거리 + 정규화 판별 분석(RDA) 등 고전 모델 (88~89/99행 확률 > 0.999). Claude Code 는 표만 씀
- CV 와 test 순서는 FM·GBDT 에서 같음 (test 99행이라 test 흔들림은 큼)

tabular-playground-series-may-2022 (AUC ↑, 1,152팀, 금 0.99823 · 동 0.99818 · 중앙값 0.97268)

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
- **FM 한 번 (25분) 이 Sonnet 에이전트 셋과 AIDE × GPT 보다 위, Codex·Claude Code +cont 보다 아래**. 에이전트 위쪽 둘은 `f_27` 분해에 더해
  교차 피처·신경망 (Codex: PyTorch MLP + GBDT 앙상블) 을 씀 → 메달권 (0.998) 은 피처 간 상호작용을 찾아야 나옴. 어느 쪽도 메달은 없음
- Claude Code +cont 가 공식 설정 (1분 23초, 0.99145) 보다 크게 오름: 이어 가기 9번에 38분을 씀 (지금까지 +cont 중 가장 김, nomad P1 은 19분)
- CV 와 test 순서가 거의 같음 (test 10만 행)
- 어느 에이전트도 TFM 을 쓰지 않음. Codex 의 `kaggle_urls` 1 은 대회 설명(description.md)의 시작 노트북 링크를 기록에 다시 출력한 것 (사후 검사 오탐)

실행 메모

- FM GPU 작업의 H100 이 약 1시간 동안 안 잡힘 (PENDING, 워크스페이스에 다른 H100 작업 없음). 1만 행 컨텍스트 TFM 과 leaf 는 A10 으로 다시 냄
  (`results.csv` 의 note `p3 (A10, H100 대기)`), `@rand200000` 둘만 H100. A10 에서 tabpfn_v2 는 1만 행 tps 에 21~26분
- 비용: Agent 10회 A10 하네스 시간 합 6.2시간 (+ 실행마다 설치), 입력 토큰 6,580만 (Codex 둘이 6,290만), 출력 53만.
  FM 학습 시간 합 (준비·대기 제외) GPU 2.3시간 (A10 1.4 + H100 0.95), GBDT CPU 2.9시간 (leaf catboost 99 클래스 84분, tps catboost F0 53분)

## C2 준비: FM 을 깐 환경의 Agent (2026-11-12)

규칙·결과 표 열은 [roadmap](plans/roadmap.md) 의 "C2 규칙". 요약: `--fm` 이면 tabpfn_v2·tabicl·kumo 를 깔고 가중치를 받아 둔 뒤
(하네스 1시간 밖) 지시문에 이름과 사용 예(`fab/agent/fm_notes.txt`)를 더함. 나머지는 공식 설정과 같음.
실행 중 TFM import 기록(`fab/agent/fm_audit.py`)은 공식 설정에도 깖 (에이전트에게 안 알림, 동작은 바꾸지 않음).

확인한 것 (로컬): import 기록 (실제 import 만 남고 `importlib.util.find_spec` 은 안 남음, `importlib.resources` 정상),
사후 검사 열 (합성 기록), 안내문 조각 나누기·코드 블록 파싱, CLI. **사용 예 실행(사전 점검)은 로컬에서 못 돌림** →
첫 원격 실행의 `meta.json` 의 `fm_warmup`·`fm_build_tools` 로 확인 (실패하면 하네스를 시작하지 않아 LLM 비용은 안 듦, 행은 `exclude`).

실행 (P0·P1 과 같은 순서: Sonnet 줄과 GPT 줄을 하나씩 동시에). 먼저 nomad (C1 에서 "에이전트 피처 + FM" 이 강했던 대회),
그다음 spaceship. 비교 대상은 같은 설정의 1·2회째

```bash
# GPT 줄
uv run --frozen fab mleb agent run nomad2018-predict-transparent-conductors codex-gpt --fm   # → aide-gpt --fm
# Sonnet 줄
uv run --frozen fab mleb agent run nomad2018-predict-transparent-conductors claude-code-sonnet --continue --fm  # → aide-sonnet --fm → claude-code-sonnet --fm
```

## C2 결과 1회째: nomad (2026-11-12, 1시간 · A10)

열별 RMSLE 평균 ↓, 괄호는 원래 리더보드 순위. 비교는 같은 설정의 Agent 1·2회째 (위 P0·P1). FM 사용 열은 회수 기록으로 이 브랜치의
사후 검사를 돌린 값 (아래 "운영 메모": 결과 표의 이 열들은 `fab mleb agent recheck` 뒤에 채워짐).

| 설정 | C2 (+fm) | Agent 1회째 / 2회째 | 시간 | 실제로 쓴 FM (`tfm_runtime`, 프로세스 수) | 제출에 든 FM (`tfm_at_submit`) |
|---|---|---|---|---|---|
| Codex × GPT | **0.04589 (1위)** | 0.04847 / 0.04567 | 57분, 429 39번 | kumo·tabicl·tabpfn (55, kumo 41) | kumo |
| Claude Code × Sonnet +cont | 0.13875 (833위) | 0.05267 / 0.05157 | 6분 14초 (이어 가기 9번) | tabicl·tabpfn (6) | tabicl·tabpfn |

- **두 에이전트 모두 FM 을 실제로 씀** (P0·P1 에서는 아무도 안 씀). 사전 점검은 두 실행 모두 세 모델 통과 (모델당 10~30초),
  Kumo 컴파일 도구는 노드에 이미 있음 (`fm_build_tools: present`)
- **Codex**: 16분부터 Kumo 를 결정 구조 피처와 함께 계속 씀 (`cv_kumo_*.py` 등, 공간군·상별 Kumo, 커널·GNN 과 앙상블).
  점수는 2회째 (0.04567) 와 같은 수준 → 이 1회로는 FM 덕에 나아졌다고 못 함. C1 의 `kumo~f2` (0.04523) 와도 비슷
- **Claude Code: 에이전트 버그로 test 만 깨짐**. 교차 검증(OOF) 은 RMSLE 약 0.0495 (Agent 때보다 나음) 인데 test 0.139.
  `c.py` 부터 학습 행렬은 [기본 열, `sg`, 결정 구조 피처] 인데 test 행렬은 [기본 열, 결정 구조 피처, `sg`] 순서
  (`X['sg']` 를 결정 구조 피처를 붙이기 전에, `T['sg']` 는 뒤에 만듦) → TabPFN·TabICL 에 `.values` 로 넘겨 열이 위치로 어긋남.
  CV 는 학습 행렬만 써서 못 잡음. 제출은 `c3.py` 가 만든 `sub6.csv` 를 `cp` 한 것 (LightGBM 0.15 + TabPFN 0.45 + TabICL 0.40).
  FM 환경 탓은 아님: 열 이름으로 맞추는 GBDT 만 썼다면 안 깨졌을 버그라 FM 의 numpy 인터페이스가 드러낸 것에 가까움
- 사전 점검에서 고친 것: Codex 실행의 Kumo 분류 사전 점검 정확도가 0.0 (Claude 실행은 1.0) → 확률 열 순서가 실행마다 다름.
  사용 예에 열을 클래스 순서로 정렬하는 줄을 넣고 (fab 의 Kumo 와 같은 방식), 사전 점검이 정확도·RMSE 기준도 보게 함
  (nomad 는 회귀라 이 두 실행의 점수와는 상관없음)
- `tfm_at_submit` 고친 것: 처음엔 제출 파일 시각(`cp` 한 때)으로 봐서 Claude Code 가 빈칸이었음 → 작업 디렉터리에서 제출과
  내용이 같은 파일의 가장 이른 시각으로 봄

## C2 마무리: leaf · tps-may · nomad (2026-11-12, 진행 중)

C2 1회째 (nomad) 는 FM 과 GBDT 차이가 작은 대회라 "FM 을 줘서 나아졌나" 가 안 갈림 → FM 이 크게 앞서는 대회에서 돌림
(leaf: FM 이 GBDT 의 1/8~1/20 오차, tps-may: kumo 20만 행 컨텍스트가 GBDT 보다 AUC 0.02 높음). 같은 설정의 Agent (P3·P0·P1) 와 비교.

| 대회 | 설정 (`+fm`, rep 1) |
|---|---|
| leaf-classification | codex-gpt, aide-gpt, aide-sonnet, claude-code-sonnet+cont |
| tabular-playground-series-may-2022 | codex-gpt, aide-gpt, aide-sonnet, claude-code-sonnet+cont |
| nomad2018 (1회째에 없던 것) | aide-gpt, aide-sonnet |

- 같은 LLM 엔드포인트끼리는 차례로 (GPT 줄 · Sonnet 줄을 하나씩 동시에), 2026-11-12 14:47 시작
- 뺀 것: spaceship (FM·GBDT·Agent 차이가 test 흔들림 안), claude-code-sonnet 공식 설정 (몇 초 만에 스스로 끝나 FM 을 쓸 틈이 없음 → +cont 로 봄)
- 볼 것: 점수 (같은 설정의 Agent 대비), FM 을 실제로 썼는지 (`tfm_runtime`), 제출에 들었는지 (`tfm_at_submit`)

## 트랙 B 실행 환경 (2026-11-09 결정)

| 항목 | 결정 |
|---|---|
| 위치 | Databricks 서버리스 GPU 작업, **A10 1장** (MLE-bench 공식 설정과 같은 GPU) |
| LLM | 워크스페이스 FM API: `databricks-claude-sonnet-5-5`, `databricks-gpt-6-1-terra` |
| 하네스 × LLM | AIDE × {Sonnet, GPT} (Chat Completions), Claude Code × Sonnet (Anthropic Messages), Codex CLI × GPT (Responses) |
| 패키지 | 일반 ML 패키지만 (MLE-bench 공식 이미지 수준). TFM 은 미리 안 깖 → 에이전트가 직접 설치해 썼는지 기록. C2 (`+fm`) 만 TFM 3개를 깖 (위 "C2 준비") |
| 인터넷 | 허용. 기록(대화·명령)을 남기고 사후 검사 (kaggle.com 풀이·토론 접근, 답 찾기) |
| 신원·권한 | 전용 서비스 프린시펄로 실행. 권한은 FM 엔드포인트 2개 조회 + 전용 scratch Volume 읽기·쓰기만 |
| 지시문 | MLE-bench `environment/instructions.txt` 를 경로만 바꿔 씀. 검증 서버 대신 형식만 보는 스크립트 (열·행 수·id) |
| 시간 | 벽시계 1시간 (LLM 대기 + 학습 포함), 넘으면 강제 종료. 그때 `submission.csv` 가 없으면 실패 |
| 일찍 끝날 때 | 본 결과는 이어 가지 않음 (스스로 끝냄·하네스 시간 표시). 별도 설정 `+cont` 만 같은 세션을 이어 감 (아래) |
| 회수 | 제출 파일, 작업 디렉터리의 코드 전부, 대화 기록, 토큰 사용량 |
| 범위 | 큰 대회(taxi·tps-dec-2021 등)는 노드 메모리 확인 전까지 트랙 B 에서 뺌 (GitHub issue) |
| P0 | spaceship·nomad × 설정 4 × 1회 = 8회 → 실제 비용 보고 나머지 결정 |

정답(private)은 로컬 mlebench 캐시와, 채점용으로 올린 내 Volume(`databricks.local.yaml` 의 `volume` 아래 `mlebench/data/`)에만 있음.
에이전트 SP 의 scratch Volume 에는 public 만 올림. 2026-11-10 채점을 원격 정리 작업(`fab/remote/reconcile.py`, 내 신원)으로 옮기면서
정답이 워크스페이스에 올라감 → 격리는 SP 권한에 기댐: SP 는 정답 Volume 의 스키마에 `USE_SCHEMA`·`READ_VOLUME` 이 없음
(`databricks grants get-effective` 로 확인, 그룹에서 받은 `READ_METADATA` 만). SP 권한을 넓힐 때 이 스키마는 빼야 함.

### 하네스 공통 규칙 (2026-11-09, kaggle-master#35·kaggle-master#37)

세 하네스에 똑같이 적용. 하네스마다 조건을 따로 맞추지 않음.

| 규칙 | 내용 | 기록 (`results/mlebench_runs.csv`) |
|---|---|---|
| 시간 예산 | 벽시계 1시간 하나. LLM 대기·429 대기·학습·설치 후 하네스 시작부터 셈. 늘리지 않음 | `harness_seconds` |
| 스스로 끝냄 (kaggle-master#35) | 시간이 남아도 이어 가게 하지 않음 (`--resume`·"계속하라" 재지시 없음). 하네스 그대로의 동작으로 봄. 이어 가는 건 별도 설정 `+cont` 뿐이고 본 결과와 섞지 않음 (아래 "스스로 끝낼 때") | `timed_out` (False = 스스로 끝냄), `continues` |
| 429 대기 (kaggle-master#37) | 프록시가 기다렸다 다시 보내고, 기다린 시간도 1시간에 들어감. 대기만큼 예산을 늘리지 않음: AIDE 는 자기 `time_limit` 을 벽시계로 세서 실행기가 시간을 늘려도 AIDE 에는 적용이 안 됨 → 늘리면 하네스마다 조건이 달라짐 | `retries_429`·`waited_429` |
| 동시 실행 | 같은 FM 엔드포인트를 쓰는 실행은 한 번에 하나 (분당 토큰 한도를 나눠 쓰지 않게) | – |

결과를 읽을 때:

- 점수 표에는 하네스 시간과 종료 방식(스스로 끝냄 / 시간 종료)을 같이 적음. Claude Code 는 "1시간 예산"에서 실제로는
  20초 안팎만 씀 → 하네스 비교는 "같은 예산을 받았을 때 하네스가 어떻게 쓰는가"까지 포함한 결과로 읽음
- 429 대기도 같이 적음. Codex 는 호출마다 컨텍스트 전체(약 10만 토큰)를 보내서 대기가 가장 김 (P0: 4~7분) → 그만큼 불리
- 한도 상향은 워크스페이스 쪽 일 (kaggle-master#37 에 남김). 한도가 바뀌면 그 전후 실행을 따로 표시

구현 (`fab mleb agent run|list|peek`, `fab/agent/__init__.py` + 원격 `fab/agent/runner.py`, 회수·채점은 정리 작업 `fab/remote/reconcile.py`):

- 전용 SP 와 전용 scratch Volume (SP 권한: USE CATALOG·USE SCHEMA·READ/WRITE VOLUME 만. TFM 실행용 Volume 은
  메타데이터만 보임, 위치는 `databricks.local.yaml` 의 `mleb_agent`). 작업은 `jobs submit` + `run_as` + `ai_runtime_task`
- FM API 시스템 엔드포인트에는 ACL 을 못 검 → 실행기 안 로컬 프록시가 모델을 설정값으로 고정하고 요청마다 토큰을 기록 (`proxy.jsonl`).
  SP 토큰은 요청마다 SDK 에서 받아 1시간을 넘겨도 만료 안 됨. AIDE 는 모델 이름이 `gpt-` 로 시작해야 OpenAI 백엔드를 써서 별칭(`gpt-km-*`)
- 지시문: MLE-bench `environment/instructions.txt` 그대로 + `fab/agent/additional_notes.txt` (AIDE 의 추가 안내에서 단계 수 줄을 빼고
  "시간이 되면 종료되니 유효한 제출을 늘 저장" 한 문장 추가) + 대회 description. 세 하네스가 같은 글을 받음
- 형식 검사 서버 `localhost:5000/validate` (sample_submission 과 열·행 수·id·결측만 비교, 메시지 영어)
- 하네스 버전 고정: AIDE `thesofakillers/aideml@v6.3.3` (MLE-bench 가 쓰는 포크), Claude Code 2.1.319, Codex 0.167.0
- 로컬 확인 (2026-11-09): 프록시의 Anthropic Messages·Responses 스트리밍, Chat Completions, 모델 고정, 사용량 기록, 형식 검사.
  이 노트북의 Claude Code·Codex 는 회사 관리 설정이 공급자를 강제해서 로컬 하네스 시험은 대표성이 없음 → 원격에서 확인
- 사후 검사 (정리 작업, 예전 `fetch`): 코드·기록의 TFM 이름, `pip` 로 새로 설치한 패키지, Kaggle 풀이·토론 URL 수, 토큰 합계, 다른 모델 요청 여부
- 하네스가 시간을 다 안 쓰고 스스로 끝낼 때 (2026-11-10 결정, kaggle-master#35. 위 "하네스 공통 규칙" 의 "스스로 끝냄" 은 본 결과의 규칙,
  여기에 별도 설정을 더함):
  - **본 결과(공식 설정)는 그대로**: 이어 가지 않음. 결과 표에 "스스로 끝냄"·하네스 시간을 같이 적음. P0 의 Claude Code
    (18~20초·4턴)는 이 설정의 결과로 둠
  - **별도 설정 `<config>+cont`** (`fab mleb agent run ... --continue`): 하네스가 정상 종료(exit 0)했는데 2분 이상 남았으면
    같은 세션을 이어 감 (Claude Code `-p --resume <session_id>`, Codex `exec resume <thread_id>`). 보내는 글은 세 하네스
    같음: "시간 한도에 아직 안 닿았고 N분 M초 남았다, 남은 시간으로 점수를 올려라, 유효한 제출을 늘 저장해 두라"
    (`fab/agent/runner.py` 의 `CONTINUE_MESSAGE`). 도구를 하나도 안 쓰고 끝난 구간이 연속 3번이면 멈추고, 전체로는 최대 40번
    (2026-11-11, P0 은 전체 20번만. 근거는 위 P0 결과의 "+cont")
    벽시계 1시간은 처음부터 셈 (이어 간 시간도 포함)
  - AIDE 는 해당 없음: 자기 `time_limit` 까지 돌아서 일찍 끝나지 않고 이어 갈 세션도 없음 → `--continue` 를 거부.
    Codex 는 58~59분을 써서 규칙이 거의 안 걸림. 실제로 걸리는 건 Claude Code
  - 기록: `continues` (이어 간 횟수), `first_exit_seconds` (공식 설정이었다면 끝났을 시각), meta.json 의 `segments`
    (구간마다 시작·길이·종료 코드·`tools` = 도구 호출 수, AIDE 는 null)·`continue_stop` (멈춘 까닭: `idle`·`max`·`min_left`·
    `timed_out`·`exit N`·`no_session`). 이 열 이전의 공식 설정 행은 0 과 harness_seconds
  - 세션 id 를 출력에서 못 찾으면: Claude Code 는 `--continue` (cwd 의 최근 대화), Codex 는 이어 가지 않음 (`--last` 와 글을 같이
    주면 글이 SESSION_ID 자리로 읽힐 수 있음). Codex 의 resume 은 원격에서 아직 안 돌아 봄 (로컬에서 인자 해석·JSON 출력만 확인)
  - 왜 둘 다: 공식 설정만 두면 Claude Code 는 "1시간 예산" 비교에 사실상 1분짜리로 들어가고, +cont 만 두면 하네스 그대로의
    동작이 안 보임. 둘을 나란히 보면 "Claude Code 가 약한지, 일찍 끝내서인지" 가 갈림. 두 설정은 결과 표에서 섞지 않음
  - P0 추가분: Claude Code × {spaceship, nomad} +cont 1회씩 (2회 × 1시간)
- 첫 원격 실행에서 고친 것: Volume 의 command_path 옆에 `env_vars.json`·`secret_env_vars.json`(빈 배열)이 있어야 런처가
  시작함, 코드 tarball 은 최상위 폴더 하나로 묶어야 함, `run_as` 에는 SP 의 Manager 와 별도로 **Use** 역할이 필요,
  프록시가 `Accept-Encoding` 을 넘기면 gzip 에러 본문이 깨져서 뺌

- FM API 호환 처리 (2026-11-09, 프록시·하네스 설정. 두 LLM 에 똑같이 적용):
  - 샘플링 파라미터(`temperature`·`top_p`·`top_k`)를 뺌: Sonnet 5.5 는 둘 다 거부, GPT-6.1 Terra 는 temperature 1 만 허용.
    AIDE 는 temperature 를 늘 보냄 (MLE-bench 의 o1-preview 설정도 기본값 1)
  - 429 (워크스페이스 분당 입력 토큰 한도): 프록시가 기다렸다 다시 보냄. 기다린 시간도 1시간에 들어감
    (`retries_429`·`waited_429` 로 기록). 같은 엔드포인트를 쓰는 실행은 동시에 돌리지 않음 (한도를 나눠 씀).
    대기는 1→2→4→8→15→30초(상한 30초, 최대 15번, 합 약 330초)에 ±25% 흔들기. 429 에 `Retry-After` 가 있으면 그보다 일찍 보내지 않음 (60초에서 자르고 0~25% 늦춰 흔듦,
    짧은 값으로 기본 간격보다 줄이지는 않음 → 합 약 330초 유지).
    P0 은 5→60초(8번, 합 315초)였음 → 2026-11-10 에 줄임 (아래 "429 측정")

### 429 측정 (2026-11-10, kaggle-master#37)

- P0 에서 429 를 받은 건 **Codex × GPT 만** (1시간 입력 4,200만~4,500만 토큰, 다른 설정은 5만~55만). AIDE × GPT 도 0번
  → 모델보다 하네스의 사용량 문제. 한도는 엔드포인트별 (오류에 `rate limit for databricks-gpt-6-1-terra`)
- Codex 의 첫 429 는 28~31분 (호출당 입력이 약 19만 토큰을 넘고 5분 평균 분당 100만~120만 토큰일 때). 처음 15분은 0번
- 429 받은 요청 대부분이 첫 재시도(5초 뒤)에 통과, 재시도당 평균 대기 8~11초 → 대기 간격을 줄임
- 직접 부하 (GPT-6.1 Terra, 요청당 14.5만 토큰·캐시 100%, 분당 0.5→2M 단계, 워크스페이스 2곳 동시):
  두 워크스페이스 결과가 같음 — 1.5M/분까지 429 없음, 2M/분 단계에서 429, 60초 창 최대 처리 1.89M.
  → 두 워크스페이스가 같은 티어, 한도는 워크스페이스마다 따로 (동시에 둘 다 1.75M/분 처리)
- 설명 안 되는 점: Codex 는 직전 60초 입력 0.43M 에서도 429 를 받았고 2.4M 에서도 통과함, 동시 요청도 대부분 1개.
  한도가 실제 입력이 아닌 추정치(예: 입력 + 최대 출력)로 세거나 1분 창이 아닌 방식일 수 있음 (확인 안 함)
- 근본 해결은 GPT 엔드포인트 티어 상향. 모델은 바꾸지 않음 (다른 GPT 엔드포인트도 같은 티어일 가능성이 크고, LLM 축이 깨짐)
  - AIDE: `agent.convert_system_to_user=True` (Claude 가 시스템 메시지만 있는 요청을 거부. MLE-bench 도 o1-preview 에서 씀)
  - AIDE 피드백 단계의 함수 호출: Sonnet 은 강제 tool_choice·json_schema 를, GPT 는 Chat Completions 함수 도구를 거부 →
    프록시가 도구를 "이 스키마의 JSON 만 답하라" 지시로 바꾸고 답을 함수 호출 응답으로 감쌈 (최대 3번). 원래 AIDE 는 네이티브 함수 호출
  - Claude Code: `ENABLE_TOOL_SEARCH=false` (kaggle-master#40). 도구 검색(일부 도구를 나중에 불러오는 기능)이 켜지면 첫 요청에
    `anthropic-beta: advanced-tool-use-2025-11-20` 이 붙고 FM API 가 400 으로 거부함 (P0 두 실행 모두 첫 요청 1회, Claude Code 가
    0.2초 뒤 다시 보내 결과에는 영향 없었음). 프록시에서 헤더만 빼면 본문의 지연 로드 도구가 검색 없이 남을 수 있어 기능 자체를 끔
    → 처음부터 모든 도구를 넘김. 프록시가 요청마다 `beta`(anthropic-beta 값)·`n_tools` 를 `proxy.jsonl` 에 기록해 다음 실행에서 확인
- 노드 사양 (kaggle-master#32): 실행기가 시작할 때 `meta.json` 의 `node` 에 CPU 수(`os.cpu_count`·affinity·cgroup 한도), 메모리(물리·
  `/proc/meminfo`·cgroup 한도), `/tmp`·HOME 디스크 여유, GPU(이름·메모리)를 적음. 정리 작업(예전 `fetch`)이 `node_cpus`·`node_mem_gb`·
  `node_tmp_free_gb`·`node_gpu` 열로 옮김 (CPU·메모리는 호스트·affinity·cgroup 한도 중 가장 작은 값.
  항목마다 따로 재서 하나가 실패해도 나머지는 남음, `nvidia-smi` 는 30초 제한) → 큰 대회(taxi·tps-dec-2021·tps-may-2022) 판단에 씀
- 제외한 실행 (`fab mleb agent exclude`, 사유는 Volume 의 state/agent/<key>.done.json, 2026-11-10 이전은 runs.json): 런처 시작 실패 4회, 프록시 Accept-Encoding 버그 1회,
  AIDE temperature 거부·빈 messages 2회, Codex 429 로 27분에 중단 1회 (점수 0.8092)

## 에이전트 실행 환경 (트랙 B) 후보 (결정 전 검토)

| | 로컬 Docker (MLE-bench 공식 환경) | Databricks 서버리스 GPU 작업 (추천) |
|---|---|---|
| 같은 조건 | 공식 컨테이너·지시문·검증 서버 그대로 | 지시문(`environment/instructions.txt`)·검증 스크립트를 옮겨 씀 |
| GPU | 없음 (Mac) | A10 (공식 설정과 같은 GPU) |
| 정답 격리 | 컨테이너 마운트로 | private 정답은 로컬에만 있고 원격엔 public 만 올라감 → 구조적으로 격리 |
| LLM 인증 | API 키를 컨테이너에 넘겨야 함 | 워크스페이스 Foundation Model API 를 쓰면 작업 안 인증으로 해결 (모델 제공 여부 확인 필요) |
| 하네스 | AIDE·OpenHands·MLAB 은 공식 agents/ 에 있음, Claude Code·Codex 는 추가 | 셋 다 작업 안에서 설치 |

## 해석 시 주의

- 조건이 다름: 에이전트는 시간·GPU 를 마음대로 씀 → "TFM 한 번 추론 vs 에이전트 N시간"으로 읽음
- 메달 기준선은 원래 Kaggle LB, 채점은 다시 나눈 test → 근사. tabular playground 는 아주 작은 차이로 메달이 갈림
- LLM 이 공개 Kaggle 풀이를 학습했을 가능성 (TFM 엔 없음)
