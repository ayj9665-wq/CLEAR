# 모델 보고서 — XGBoost (flat baseline)

- 작성자: 안윤지 · 작성일: 2026-07-23
- 파이프라인 위치: [experiments/train_baseline.py](../src/experiments/train_baseline.py) (`model=xgboost`)
- 벤치마크/캐비엇 공통 문서: [모델벤치마크_선행연구비교.md](모델벤치마크_선행연구비교.md)

## 1. 역할 — "GNN이 넘어야 할 기준선"

XGBoost는 **그래프를 무시하고 노드 특성만으로** 학습하는 flat(비그래프) 모델이다.
CLEAR의 핵심 질문 "그래프가 실제로 도움이 되는가"는 곧 **GraphSAGE가 이 XGBoost를
넘느냐**로 환원된다. 즉 XGBoost는 성능 상한이 아니라 **GNN 이득을 재는 기준선**이며,
동시에 준거 논문(Campedelli 2022)이 9개 알고리즘 중 최고로 꼽은 모델이라 **문헌 정합의
접점**이기도 하다.

## 2. 설정

| 항목 | 값 | 근거 |
|---|---|---|
| 입력 | 노드 특성 One-Hot 전면 (민감속성 제외) | `clear.data.load_xy` |
| 표본 | California + Texas + Michigan | `config.SAMPLE_STATES` |
| 분할 | 70/30 층화, 공유 test split (GNN과 동일) | `clear.data.get_split` |
| 튜닝 | 5-fold 층화 CV, `scoring="matthews_corrcoef"` | `config.CV_FOLDS` |
| 버전 | xgboost>=2.1.4 (sklearn>=1.6 `__sklearn_tags__` 호환) | `requirements.txt` |

## 3. 우리 결과 (seed 42, 결정적)

| AUC | MCC | F1 | Sensitivity | Specificity | Balanced Acc | Precision |
|---|---|---|---|---|---|---|
| 0.703 | **0.274** | 0.726 | 0.670 | 0.621 | 0.646 | 0.791 |

- 대표 스칼라 **MCC 0.274**. logreg(0.233)보다 확실히 높아 비선형·상호작용이 신호를
  더 잡음. GraphSAGE(0.287)에는 **안정 지표(MCC·AUC)에서 근소하게 뒤짐** →
  "그래프가 도움이 된다"의 근거가 됨(자세한 판정은 GraphSAGE 보고서).
- 출처: `outputs/results.csv`. 결정적이라 1행.

## 4. 선행연구 대비 (공유지표 Balanced Acc · Precision)

- **같은 축 직접 비교**: 우리 XGBoost BA **0.646** / Precision **0.791** vs
  Campedelli(2022) 전국 XGBoost BA **0.767** / Precision **0.863**, California BA
  **0.802**. 우리가 BA에서 **약 0.12~0.16 낮다.**
- **이 격차는 모델 열세가 아니라 입력 특성 차이다.** 논문의 최상위 SHAP 예측변수
  `Circumstance`(32범주)가 우리 Kaggle 데이터엔 **아예 없고**, 2위 `Number of Offenders`
  (=Perpetrator Count)는 우리가 **누수로 배제**한다. 즉 우리는 더 좁고 엄격한 특성집합으로
  같은 데이터셋의 **다른(더 어려운) 문제**를 푼다. 전체 분석:
  [벤치마크 문서 §4](모델벤치마크_선행연구비교.md).
- 정성적 정합: 논문도 **XGBoost가 9개 중 최고** — 우리가 flat 대표 baseline으로 XGBoost를
  택한 것과 일치. 같은 MAP·같은 One-Hot·같은 70/30 분할이라 존재하는 가장 가까운 준거.
- 우리 5지표(AUC/MCC/F1/Sens/Spec)는 논문 미보고 → 그 축은 여전히 비교 불가.

## 5. 주의사항

- **타깃 누수가 이 데이터의 최대 함정**: 가해자 열 재유입 시 AUC ~0.99. `LEAKAGE_COLS`
  제외 상태 유지 필수.
- 평가 지표는 논문의 Balanced Acc+Precision이 아니라 AUC/MCC/F1/Sens/Spec —
  **의도적 후속 이탈**(CLAUDE.md). 문헌과 데이터·분할은 맞추되 지표는 우리 선택.
