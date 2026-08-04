# 모델 보고서 — Logistic Regression (성능 하한)

- 작성자: 안윤지 · 작성일: 2026-07-23
- 파이프라인 위치: [experiments/train_baseline.py](../src/experiments/train_baseline.py)
  (`model=logreg`). 한동안 저장소에서 제거돼 있었고 2026-08-04에 커밋 `96d1dd1`에서
  **복원**했다(전국 평면 baseline이 필요해졌기 때문). 3개 주 덤프
  `outputs/predictions/logreg[_blind].csv`는 그 사이 재생성이 불가능해 커밋돼 있고,
  **재실행으로 덮어쓰지 않는다** — xgboost 2.x에서 3.x로 바뀌어 예측이 달라지면
  헤드라인 대조(+0.82)가 흔들린다. 복원한 학습기는 전국 스코프에서만 돌렸다
- 벤치마크/캐비엇 공통 문서: [모델벤치마크_선행연구비교.md](모델벤치마크_선행연구비교.md)


> **스코프 주의 (2026-08-04 갱신)**: 이 보고서의 본문 수치는 3개 주(CA+TX+MI) 표본이다. **전국 수치는 2026-08-04에 생겼다** — `train_baseline.py`를 커밋 `96d1dd1`에서 복원해 전국(638,454행)에서 다시 학습했고, 덤프는 `outputs/predictions/national/`에, 지표는 `results.csv`(params.scope=national)에 있다. 3개 주 수치와 **직접 비교하지 말 것** — 표본도 작동점도 다르다.

## 1. 역할 — "성능 하한(floor)"

로지스틱 회귀는 그래프도, 비선형 상호작용도 쓰지 않는 **가장 단순한 선형 분류기**다.
CLEAR에서의 역할은 이기는 모델이 아니라 **바닥선**이다: "노드 특성만 선형으로 넣으면
최소 이만큼은 나온다"를 고정해, XGBoost·GNN의 이득이 실제 신호인지 판별하는 기준.

## 2. 설정

| 항목 | 값 | 근거 |
|---|---|---|
| 입력 | 노드 특성 One-Hot 전면 인코딩 (민감속성 `sens__*` 제외) | `clear.data.load_xy` |
| 표본 | California + Texas + Michigan | `config.SAMPLE_STATES` |
| 분할 | 70/30 층화, 공유 test split | `config.TEST_SIZE`, `clear.data.get_split` |
| 튜닝 | 5-fold 층화 CV, scoring=MCC | `config.CV_FOLDS` |
| 누수 제거 | 가해자 열·관계 전면 제외 | `config.LEAKAGE_COLS` |

## 3. 우리 결과 (seed 42, 결정적)

| AUC | MCC | F1 | Sensitivity | Specificity | Balanced Acc | Precision |
|---|---|---|---|---|---|---|
| 0.673 | **0.233** | 0.688 | 0.614 | 0.636 | 0.625 | 0.783 |

- 대표 스칼라는 **MCC(0.233)** — ~32/68 불균형에서 정확도보다 신뢰할 수 있음.
- XGBoost(MCC 0.274) 대비 확실히 낮고, GraphSAGE(0.287)와는 더 벌어짐 → **하한 역할
  충실**. Balanced Acc·Precision은 논문 대조용 부가지표(§4). 출처: `outputs/results.csv`.

## 4. 선행연구 대비 (공유지표 Balanced Acc · Precision)

- 이제 **같은 축에서 비교 가능**: 우리 logreg BA **0.625** / Precision **0.783** vs
  Campedelli(2022) 전국 선형계열(Ridge/LASSO/EN) BA **0.751** / Precision **0.854**.
- **순위 방향 일치**: 논문도 선형계열이 트리계열(XGB 0.767)보다 낮음 — 우리 logreg <
  xgboost < graphsage와 정성적으로 같은 순서.
- 절대 BA가 논문보다 낮은 이유(Circumstance 특성 부재 + offender-count 누수 배제 +
  스코프 차이)는 [벤치마크 문서 §4](모델벤치마크_선행연구비교.md)에서 분석. 우리 5지표
  (AUC/MCC/…)는 논문이 보고하지 않아 그 축은 여전히 비교 불가.

## 5. 주의사항

- 타깃 누수(가해자 열) 재유입 시 AUC ~0.99로 치솟음 — logreg에서도 동일. 입력에서
  `LEAKAGE_COLS`가 빠졌는지 항상 확인.
- 선형 모델이라 seed 간 분산이 사실상 없음(결정적) → mean±std 없이 1행.
