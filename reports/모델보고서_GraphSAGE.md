# 모델 보고서 — GraphSAGE (그래프 모델)

- 작성자: 안윤지 · 작성일: 2026-07-23
- 파이프라인 위치: [experiments/train_gnn.py](../src/experiments/train_gnn.py) (`model=graphsage`), 모델/학습은 `clear.gnn`
- 가중치 저장: `--save_model` →
  `outputs[/{scope}]/models/graphsage_{edge}[_mode][_blind][_mb]_seed{N}.pt`
  (예측 덤프와 같은 이름 규칙). `state_dict`와
  함께 하이퍼파라미터·**특성 열 이름 순서**·blind 여부·스코프·`calibrated: False`를 넣고,
  `clear.gnn.load_checkpoint`가 열 순서를 대조해 불일치하면 죽는다. **커밋하지 않는다**
  (클론만 한 사람은 그래프도 특성도 없어 쓸 수 없다)
- 벤치마크/캐비엇 공통 문서: [모델벤치마크_선행연구비교.md](모델벤치마크_선행연구비교.md)

## 1. 역할 — "그래프가 도움이 되는가"의 본체

GraphSAGE는 사건을 노드로, **지역·시기·수법 유사도로 건 엣지**를 통해 이웃 사건의
맥락을 임베딩에 집계하는 GNN이다. CLEAR "예측" 단계의 본체이자, XGBoost baseline을
넘는지가 프로젝트 가설의 핵심 검정이다.

## 2. 설정 (기본값 = config.GNN_*)

| 항목 | 값 | 근거 |
|---|---|---|
| 엣지 | **geo** (State+City 블로킹, k=20) | `config.GNN_DEFAULT_EDGE_TYPE`, `K_NEIGHBORS` |
| 구조 | hidden 64, 2 layers, dropout 0.3, aggr=mean | `config.GNN_HIDDEN_DIM` 등 |
| 최적화 | lr 0.005, weight_decay 5e-4, max 200 ep, patience 20 | ablation로 승격된 값 |
| 조기종료 | **validation MCC** 기준 (val loss 아님) | 재가중 BCE가 MCC와 1:1 아님 |
| 분할 | XGBoost와 **동일 test split** + val 15% 추가 분리 | `clear.data.get_split`, `GNN_VAL_SIZE` |
| seed 반복 | 42/43/44 (split 고정, torch seed만 변경) | `config.GNN_SEEDS` |

## 3. 우리 결과 (기본설정, seed 3회 mean±std)

| AUC | MCC | F1 | Sensitivity | Specificity | Balanced Acc | Precision |
|---|---|---|---|---|---|---|
| 0.712 ± 0.000 | **0.287 ± 0.001** | 0.739 ± 0.004 | 0.693 ± 0.008 | 0.609 ± 0.009 | 0.651 ± 0.001 | 0.792 ± 0.002 |

- 독립 재실행(untagged) 3회도 MCC 0.285±0.000, AUC 0.712±0.001로 **일관** →
  안정 지표는 재현성 높음.
- **판정(baseline 대비)**: 안정 지표에서 XGBoost를 이김 — MCC 0.287 vs 0.274,
  AUC 0.712 vs 0.703. 이 마진은 GNN 자체 seed 노이즈(MCC std ~0.001)의 ~10배라
  **신호로 인정**.
- **주의**: F1/Sensitivity의 GNN>baseline 순서는 seed 노이즈(Sens std ~0.008~0.02)
  **안**이라 승리로 주장하지 않음 — seed 반복을 넣은 이유가 바로 이 착시를 드러내기
  위함. 출처: `outputs/results.csv` / `clear.results.summarize`.

### 3-1. 전국 스코프 (2026-07-28 추가, 미니배치 5시드)

위 표는 3개 주(CA+TX+MI, full-batch)다. 전국(638,454행 / 51개 주)은 그래프가
25.0M 엣지라 full-batch로 안 올라가 `NeighborLoader` 미니배치로 학습했다.

| 조건 | AUC | MCC | Balanced Acc | Precision |
|---|---|---|---|---|
| 전국 sighted | 0.7403 ± 0.0009 | 0.3282 ± 0.0018 | **0.6754 ± 0.0007** | 0.8237 ± 0.0041 |
| 전국 blind | 0.7295 ± 0.0006 | 0.3133 ± 0.0006 | 0.6665 ± 0.0016 | 0.8152 ± 0.0044 |
| (3개 주 blind, 미니배치) | 0.6972 ± 0.0004 | 0.2666 ± 0.0015 | 0.6399 ± 0.0013 | — |

**전국이 3개 주보다 크게 좋지만 "데이터가 3.4배라 좋아졌다"로 읽으면 안 된다.**
`State` 더미가 3개에서 51개로 늘었고 주별 검거율이 34~93%로 벌어져 있으므로, 전국
모델은 3개 주 모델이 가질 수 없던 신호를 입력으로 받는다. **과제 자체가 다르다.**

**미니배치는 full-batch와 직접 비교 불가**라, 3개 주에서 미니배치를 재현해 다리를
놓았다: AUC Δ+0.00007(시드 std의 0.06배), MCC Δ+0.00069(1.45배), Balanced Accuracy
Δ−0.00253(3.90배). 정확도는 사실상 그대로다 — 이 과제에서 그래프의 값어치가
'관련성'이 아니라 '블록 특성 분포의 비편향 표본'이므로 이웃 40개 중 25개만 뽑아도
추정치의 성격이 유지되기 때문이다. 다만 **작동점은 옮겨가므로 공정성 수치는
이전되지 않는다**(공정성진단 보고서 §10-9).

**전국 평면 baseline은 없다.** `train_baseline.py`가 저장소에서 제거돼 있어(커밋
`96d1dd1`) **"전국에서 GNN이 평면 모델을 이기는가"는 측정되지 않았다.** 아래 §4의
상대 비교는 전부 3개 주 값이다.

## 4. 선행연구 대비 (공유지표 Balanced Acc · Precision)

- **같은 축 비교**: 우리 GraphSAGE BA **0.651±0.001** / Precision **0.792±0.002** vs
  Campedelli(2022) 전국 XGBoost BA **0.767** / Precision **0.863**. 절대 BA는 낮지만
  이유는 **입력 특성 차이**(Circumstance 부재 + offender-count 누수 배제 + 스코프) —
  모델 열세가 아님([벤치마크 §4](모델벤치마크_선행연구비교.md)).
- 준거 논문에는 **GNN이 없다**(9개 알고리즘 전부 flat). 최근(2024–2026)에도 살인 *검거*를
  GNN/분류기로 재-벤치마크한 논문은 없음 → **"그래프로 검거예측"은 문헌에서 거의 빈 자리**
  = 우리 차별화 지점([벤치마크 §6](모델벤치마크_선행연구비교.md)).
- 따라서 GraphSAGE 성능 판정의 정당한 기준은 **문헌 절대치가 아니라 우리 XGBoost
  baseline과의 상대 비교**다: BA 0.651 vs 0.646, MCC 0.287 vs 0.274, AUC 0.712 vs 0.703 —
  안정지표에서 그래프가 baseline을 앞섬.

## 5. 주의사항

- 임계값 의존 지표(F1/Sens/Spec)는 run마다 ±3~4점 흔들림 → **반드시 mean±std로** 보고,
  단일 run으로 baseline과 대면 금지.
- `--k_neighbors N` 사용 시 `04_build_graph.py --k N`을 먼저 실행해야 함(없으면 파일
  없음 오류, 조용한 fallback 아님).
- 타깃 누수 제약은 GNN에도 동일 적용(`LEAKAGE_COLS`). `import torch` 전
  `KMP_DUPLICATE_LIB_OK=TRUE` 필요(config.py가 처리).
