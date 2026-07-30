# CLEAR — Clearance Learning & Equity Assessment on gRaphs

미국 살인사건 데이터(Murder Accountability Project, 1980–2014, 63.8만 건)를 사건
그래프로 재구성해 **검거 여부(clearance)를 예측**하고, 피해자 인종·성별에 따른
**검거율 격차를 진단**하고, 완화기법으로 그 격차를 **얼마나 되갚을 수 있는지 처방**한
뒤, 되갚고 남은 잔차가 **어느 카운티에 몰려 있는지 적용**한다.

하나의 질문("무엇이 검거를 결정하는가")을 **예측 → 진단 → 처방 → 적용** 네 단계로
파고든다. 뒤 단계는 앞 단계가 답하지 못한 것 때문에 존재한다.

## 결론 한 줄

> 사건을 지역으로 묶는 그래프는 검거 예측 정확도를 올리고, **동시에 인종 격차의 잔여
> 전달 경로다.** 그 격차는 정확도 0.016을 내고 되갚을 수 있으며, 되갚은 뒤 남는 잔차는
> 특정 카운티에 집중되고 그 집중은 인종 구성과 상관된다 — 다만 상관의 강도는 카운티
> 규모에 따라 커진다.

전체 서술은 **[최종보고서](reports/최종보고서_CLEAR.md)** 에 있다.

| 단계 | 결과 |
|---|---|
| 예측 | GraphSAGE(geo) > XGBoost > LogReg. 단 마진이 시드 노이즈를 넘는 지표는 MCC·AUC·Balanced Accuracy **뿐**이다 |
| 진단 | 성별 격차는 **직접 열**이 전부(blind에서 소멸), 인종 격차는 일부가 **그래프로 전달**(페어드 +0.82 [+0.65, +1.03]) |
| 처방 | 손실 벌점으로 전국 인종 증폭비 1.805 → 0.337, 비용 −0.0156 MCC. 엣지 재배선은 **실패**했고 그 실패가 누수 위치를 찾아냈다 |
| 적용 | 카운티 1,803개 판정, cold 74개. 잔차 ↔ 흑인비중 **+0.290 [+0.245, +0.331]** |

## 데이터

- `dataset/kaggle_homicide_Reports_1980_2014.csv` (약 63.8만 행) — 대용량이라 git에 없다(`.gitignore`).
- 타깃: `Crime Solved` (Yes=검거 / No=미해결). 불균형 약 68/32.
- **누수 열이 이 데이터의 지배적 제약이다.** 가해자 열(`Perpetrator *`, `Relationship`)은
  미해결 사건에서 90–99%가 `Unknown`이라 넣으면 타깃이 자명해진다(AUC ~0.99).
  `config.LEAKAGE_COLS`가 이를 고정한다. 자세히는 [데이터카드](reports/데이터카드_CLEAR.md).

## 스코프

`CLEAR_SCOPE` 환경변수 하나가 표본과 출력 경로를 함께 결정한다.

| 스코프 | 행 | 주 | 용도 |
|---|---|---|---|
| `ca_tx_mi` (기본) | 190,326 | 3 | 방법론 개발 · 기제 규명 |
| `national` | 638,454 | 51 | 최종 산출물 · 전국 지도 |

전국 값과 3개 주 값은 **직접 비교되지 않는다** — 작동점이 다르고 모든 공정성 수치가
임계값 의존이다.

## 실행

```bash
pip install -r requirements.txt
cd src
export CLEAR_SCOPE=national          # 미설정이면 3개 주

# 파이프라인 — 순서 있음(각 단계가 앞 단계 산출물을 읽는다)
python 01_clean.py
python 02_sample.py
python 03_features.py
python 04_build_graph.py --k 20

# 실험 — experiments/ 아래는 서로 의존하지 않는다
python -m experiments.train_gnn --blind --minibatch
python -m experiments.mitigate_loss --minibatch --alphas 0 10 25 50 100
python -m experiments.diagnose_fairness
python -m experiments.crossfit_predictions --minibatch        # ~76분
python -m experiments.detect_cold_blocks --model graphsage_fairloss_a100_mb_cv5 \
        --min_n 20 50 100 --out cold_blocks_cv5.csv
python -m experiments.crossfit_compare
python -m experiments.build_web_map --src cold_blocks_cv5.csv \
        --out map_national.html --simplify_km 1.5
```

전체 명령 목록과 각 인자의 의미는 `CLAUDE.md`에 있다. 테스트 스위트와 빌드/린트
단계는 없다.

**전국 규모에는 `--minibatch`가 필수다** — `geo` 그래프가 방향 엣지 2,504만 개로
full-batch에 약 20GB가 필요하다(가용 12.9GB).

## 주요 산출물

| 산출물 | 위치 |
|---|---|
| 전국 웹 지도 (대표) | `outputs/national/web/map_national.html` — 자체 완결 HTML, 외부 요청 0 |
| 전국 웹 지도 (test 분할 참조) | `outputs/national/web/map_national_test.html` |
| 통합 실험 원장 (long format) | `outputs/results.csv` |
| 공정성 진단 4표 | `outputs/[national/]fairness_{group_metrics,gaps,model_contrasts,gaps_standardized}.csv` |
| 카운티 이상탐지 | `outputs/national/cold_blocks{,_cv5}.csv` |
| 포스터·지도 그림 | `outputs/{poster,map}_fig*.png` |

원장이 하나인 것은 설계다 — `family`/`metric` 두 열로 "GNN이 기준선을 이겼나"가 파일
간 비교가 아니라 groupby 한 번이 된다. **새 지표는 새 행이고, 결코 새 열이 아니다.**

## 문서

읽는 순서를 권한다면:

1. **[최종보고서](reports/최종보고서_CLEAR.md)** — 네 단계 전체, 부정 결과, 한계
2. [문제정의서](reports/문제정의서_CLEAR.md) — 문제 설정과 윤리 게이트
3. [공정성진단 보고서](reports/공정성진단_보고서.md) — 격차 측정의 전말
4. [확장설계서](reports/확장설계서_CLEAR.md) — 사이드 트랙 · 이상탐지 · 전국 지도
5. [크로스피팅 계획서](reports/크로스피팅_계획서_CLEAR.md) — 사전 등록과 그 판정

그 외: [EDA](reports/EDA_보고서.md) · [개발계획서](reports/AI모델_개발계획서_살인사건검거_GNN.md) ·
[선행연구 비교](reports/모델벤치마크_선행연구비교.md) · [데이터카드](reports/데이터카드_CLEAR.md) ·
모델보고서 [GraphSAGE](reports/모델보고서_GraphSAGE.md) ·
[XGBoost](reports/모델보고서_XGBoost.md) · [LogReg](reports/모델보고서_LogReg.md)

## 방법론 기준 (선행연구)

Campedelli(2022, *Journal of Criminal Justice*)를 **의식적으로** 따른다 — 전면
One-Hot, 나이 5년 구간화, 70/30 무작위 층화 분할, 5-fold 층화 CV. 데이터·분할·인코딩
축에서 논문과 비교 가능하게 하기 위한 선택이다.

평가지표만 의도적으로 벗어난다: AUC/MCC/F1/Sensitivity/Specificity(프로젝트 선택)에
**Balanced Accuracy + Precision**(논문이 그 둘만 보고하므로 비교용)을 더한 7종.
**MCC는 선택 기준**이고("대표 지표"가 아니다), 논문 비교 맥락에서는 Balanced Accuracy와
Precision이 앞에 온다. 어느 것이 앞에 오든 모델 순위는 같다.

## 주의 (윤리)

이 저장소의 이상탐지 결과는 **동일범 판정이 아니다.** 데이터에 가해자 ID가 없으므로 그
주장은 원리적으로 불가능하다. 그리고 **모델은 카운티를 특성으로 갖지 않으므로** 잔차
z는 카운티 고유 효과 전체를 담는다 — 차별·수사자원·도시성·기록관행이 분리되지 않는다.
**z와 인종 구성의 상관을 인과로 읽으면 안 되고**, 이 문장은 지도를 포함한 모든 산출물에
함께 실린다.

## 이력

주차별 로드맵(데이터·EDA·베이스라인 → 그래프·GNN → 진단·처방 → 분석·시각화)은 4주차까지
완료됐고, 그 뒤 전국 확장 · 크로스피팅 · 층 표준화가 추가됐다. 초기 파이프라인 중
`eda.py`·`train_baseline.py`·`mitigate_threshold.py`·`ablation.py`는 결과를 얻은 뒤
저장소에서 제거했다(커밋 `96d1dd1`에서 복구 가능하며, 결과는 `outputs/results.csv`와
커밋된 예측 덤프에 남아 있다). 그 판단의 근거는 `CLAUDE.md`의 "Scope" 절에 있다.
