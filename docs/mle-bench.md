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
| tabular-playground-series-may-2022 | low | 큼, 위와 같음 |
| new-york-city-taxi-fare-prediction | low | 아주 큼 → 컨텍스트 샘플링 필수 |
| leaf-classification | low | 99 클래스 → TFM 클래스 수 제한 확인 필요 |

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
2. **동일성 확인**: 추출 피처로 에이전트가 쓴 모델을 다시 돌려 에이전트 점수와 비슷한지 확인, 크게 다르면 버림
3. **누수**: 타깃을 쓰는 피처(target encoding, OOF 스태킹)는 지금 형식(train+test 전체 계산)에선 CV 누수 → 1차는 타깃 비의존 피처만,
   2차에 fold 별 fit 형식(`build(train, test, y)`)으로 확장
4. **조건**: F0 원본 최소 피처 / F1 사람 피처 / F2… 에이전트 피처 × {tabpfn_v2, tabicl, kumo} + 대조군 {lgbm, catboost}
5. **지표**: (TFM 의 F0 대비 향상) − (GBDT 의 F0 대비 향상). 양수일 때만 "TFM 에 특히 도움". 피처 수와 `tfm_max_features` 적용 여부도 기록.
   점수는 같은 fold 의 CV + MLE-bench test 채점 둘 다.

## 저장소 변경

- `config.yaml` 에 `source: mlebench` → `fab download` 가 `mlebench prepare` 를 부르고 public 데이터만 `data/` 에 둠 (private 정답은 에이전트 작업 공간 밖)
- `fab run ... --features <variant>` → `competitions/<slug>/features/<variant>.py`, 결과에 `features` 기록
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
- 상한(전체 20번)은 그대로 둠: 이 2회는 이 규칙으로 돈 결과라 바꾸면 다시 돌려야 함. 다만 "올려도 소용없다" 는 확실치 않음 —
  nomad 는 거절 5번 연속(8~12번째) 뒤 13번째에 다시 작업함. 상한을 "연속 거절 N번" 으로 바꾸면 N=5 일 때 nomad 의 그 작업을
  놓침. 바꿀지는 P2 전에 정함 (바꾸면 +cont 2회 재실행). 토큰: spaceship 109만/7천, nomad 144만/1.2만

## 트랙 B 실행 환경 (2026-11-09 결정)

| 항목 | 결정 |
|---|---|
| 위치 | Databricks 서버리스 GPU 작업, **A10 1장** (MLE-bench 공식 설정과 같은 GPU) |
| LLM | 워크스페이스 FM API: `databricks-claude-sonnet-5-5`, `databricks-gpt-6-1-terra` |
| 하네스 × LLM | AIDE × {Sonnet, GPT} (Chat Completions), Claude Code × Sonnet (Anthropic Messages), Codex CLI × GPT (Responses) |
| 패키지 | 일반 ML 패키지만 (MLE-bench 공식 이미지 수준). TFM 은 미리 안 깖 → 에이전트가 직접 설치해 썼는지 기록 |
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
    (`fab/agent/runner.py` 의 `CONTINUE_MESSAGE`). 이어 가기는 최대 20번 (모델이 매번 바로 끝내도 남은 시간 내내 부르지 않게).
    벽시계 1시간은 처음부터 셈 (이어 간 시간도 포함)
  - AIDE 는 해당 없음: 자기 `time_limit` 까지 돌아서 일찍 끝나지 않고 이어 갈 세션도 없음 → `--continue` 를 거부.
    Codex 는 58~59분을 써서 규칙이 거의 안 걸림. 실제로 걸리는 건 Claude Code
  - 기록: `continues` (이어 간 횟수), `first_exit_seconds` (공식 설정이었다면 끝났을 시각), meta.json 의 `segments`·
    `continue_stop` (멈춘 까닭: `max`·`min_left`·`timed_out`·`exit N`·`no_session`). 이 열 이전의 공식 설정 행은 0 과 harness_seconds
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
