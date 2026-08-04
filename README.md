# CLEAR — Clearance Learning & Equity Assessment on gRaphs

미국 살인사건 데이터(Murder Accountability Project, 1980–2014, 638,454건)를 사건
그래프로 재구성해 **검거 여부를 예측**하고, 피해자 인종·성별에 따른 **격차를 진단**하고,
완화기법으로 그 격차를 **얼마나 되갚을 수 있는지 처방**한 뒤, 되갚고 남은 잔차가
**어느 카운티에 몰려 있는지 적용**한다.

하나의 질문("무엇이 검거를 결정하는가")을 **예측 → 진단 → 처방 → 적용** 네 단계로
파고든다. 뒤 단계는 앞 단계가 답하지 못한 것 때문에 존재한다.

---

## 빠른 시작 (3분)

**원본 데이터도, GPU도, 모델 가중치도 필요 없다.** 저장소에 커밋된 결과 CSV만으로
성능·공정성 대시보드가 브라우저에 뜬다.

```bash
# 1. 내려받는다
git clone <이 저장소 주소>
cd 2026_CLEAR

# 2. 설치한다 (pandas·numpy 둘뿐 — torch 없음)
pip install -r requirements-dashboard.txt

# 3. 실행한다 → 브라우저 창이 자동으로 열린다
cd src
python -m experiments.dashboard
```

`outputs/dashboard.html` 이 만들어지고 기본 브라우저로 열린다. 자체 완결 HTML이라
외부 요청이 0이고, 파일 하나만 남에게 보내도 그대로 열린다.

```bash
python -m experiments.dashboard --no_open          # 파일만 생성(CI·원격)
python -m experiments.dashboard --scope national   # 전국 결과만
python -m experiments.dashboard --verify_clone     # 클론 직후 실행이 되는지 검사
```

`--verify_clone` 은 **git이 추적하는 파일만 복사한 임시 트리에서** 대시보드를 실제로
빌드해 본다. "원본 데이터 없이 돌아간다"가 이 산출물의 유일한 계약이므로, 검사도 거기에
걸어 두었다.

### 대시보드가 보여 주는 것

| 패널 | 내용 |
|---|---|
| 01 예측 성능 | 모델 × 7지표, 시드 간 표준편차, **마진 ÷ 시드 노이즈** |
| 02 공정성 진단 | 증폭비, 짝지은 모델 대조, 주 단위 표준화 |
| 03 완화 트레이드오프 | 정확도와 격차를 같은 표에 |
| 04 그래프 진단 | 엣지가 민감속성의 프록시인가 / 실제로 관련 사건을 잇는가 |
| 05 적용 | 카운티 잔차 상위 표, 잔차 ↔ 인종 구성 상관 |
| 06 사후 감사 | 수사 인력 · 정황 기록 · 카운티 차분 · 우선순위 · 교차적합 |

---

## 성능 지표

아래 표는 `outputs/**.csv` 에서 자동 생성한다
(`python -m experiments.dashboard --emit_readme_tables`).
손으로 적지 않으므로 재실행 결과와 어긋나지 않는다.

<!-- METRICS:START -->
**3개 주 (CA·TX·MI)** — 예측 성능 (± 는 시드 간 표준편차)

| 모델 | MCC | AUC | Balanced Acc | Precision |
|---|---|---|---|---|
| GraphSAGE (geo) · fairness_handoff | 0.2856 ±0.0009 | 0.7114 ±0.0004 | 0.6520 ±0.0008 | 0.7996 ±0.0052 |
| GraphSAGE (geo) | 0.2854 ±0.0001 | 0.7117 ±0.0008 | 0.6513 ±0.0010 | 0.7957 ±0.0048 |
| XGBoost | 0.2744 | 0.7033 | 0.6456 | 0.7915 |
| GraphSAGE (geo) · blind · minibatch | 0.2666 ±0.0015 | 0.6972 ±0.0004 | 0.6399 ±0.0013 | 0.7825 ±0.0037 |
| GraphSAGE (geo) · blind | 0.2659 ±0.0005 | 0.6971 ±0.0012 | 0.6424 ±0.0006 | 0.7985 ±0.0051 |
| GraphSAGE (geo) · blind · rank_onehot_control | 0.2628 ±0.0006 | 0.6959 ±0.0003 | 0.6405 ±0.0004 | 0.7941 ±0.0026 |
| GraphSAGE (geo) · blind · rank_onehot | 0.2574 ±0.0006 | 0.6903 ±0.0009 | 0.6375 ±0.0003 | 0.7908 ±0.0021 |
| XGBoost · blind | 0.2547 | 0.6847 | 0.6355 | 0.7856 |
| GraphSAGE (temporal) · blind | 0.2494 ±0.0004 | 0.6819 ±0.0010 | 0.6320 ±0.0005 | 0.7803 ±0.0028 |
| LogReg | 0.2330 | 0.6726 | 0.6248 | 0.7834 |
| LogReg · blind | 0.2147 | 0.6613 | 0.6153 | 0.7810 |

**전국 51개 주** — 예측 성능 (± 는 시드 간 표준편차)

| 모델 | MCC | AUC | Balanced Acc | Precision |
|---|---|---|---|---|
| GraphSAGE (geo) | 0.3282 ±0.0018 | 0.7403 ±0.0009 | 0.6754 ±0.0007 | 0.8237 ±0.0041 |
| XGBoost | 0.3195 | 0.7363 | 0.6725 | 0.8270 |
| GraphSAGE (geo) · blind | 0.3133 ±0.0006 | 0.7295 ±0.0006 | 0.6665 ±0.0016 | 0.8152 ±0.0044 |
| XGBoost · blind | 0.2989 | 0.7224 | 0.6616 | 0.8200 |
| LogReg | 0.2719 | 0.7006 | 0.6482 | 0.8171 |
| LogReg · blind | 0.2638 | 0.6940 | 0.6437 | 0.8133 |

**전국** — 공정성 증폭비 (1.0이 기준선, 대괄호는 95% 신뢰구간)

| 민감속성 | 모델 | 원자료 격차 | 모델 격차 | 증폭비 |
|---|---|---|---|---|
| Victim Race | `graphsage_geo_blind_mb` | 0.0763 | 0.1378 | **1.805** [1.71, 1.90] |
| Victim Race | `graphsage_fairloss_a100_mb` | 0.0763 | 0.0257 | **0.337** [0.28, 0.40] |
| Victim Sex | `graphsage_geo_blind_mb` | 0.0826 | 0.1029 | **1.246** [1.17, 1.33] |
| Victim Sex | `graphsage_fairloss_a100_mb` | 0.0826 | 0.0755 | **0.915** [0.85, 0.99] |

**전국** — 적용 단계 (교차적합 예측, 최소 사건 수 20)

| 판정 카운티 | 기대보다 많이 잔존 | 기대보다 적게 잔존 | 잔차 ↔ 흑인 피해자 비중 |
|---|---|---|---|
| 1,803 | 74 | 276 | +0.288 |
<!-- METRICS:END -->

**±는 시드 간 표준편차이며, 이 프로젝트는 마진 자체가 아니라 마진을 그 표준편차로 나눈
값으로 성능 주장 여부를 정한다.** GPU 집계가 비결정적이라 임계값에 의존하는 지표는
실행마다 흔들리기 때문이다. 3개 주 기준 MCC·AUC·Balanced Accuracy만 그 선을 넘고
**Precision·Specificity·F1은 넘지 못한다** — 그래서 성능 주장에 쓰지 않았다. 비율은
대시보드 패널 01에서 확인할 수 있다.

---

## 무엇이 되고 무엇이 안 되나

| 하려는 일 | 필요한 것 | 걸리는 시간 |
|---|---|---|
| **결과 확인 · 성능 검증** | 클론만 | 즉시 (위 빠른 시작) |
| 재평가 · 재학습 | + Kaggle 원본 CSV + 파이프라인 실행 | 수 시간, 전국은 GPU 권장 |
| 새 사건을 넣고 검거 확률 예측 | — | **의도적으로 제공하지 않는다** |

마지막 줄은 성능 문제가 아니라 이 프로젝트가 내린 판단이다. 개별 사건 단위 출력은
(1) 재수사 결과 컬럼이 없어 **검증할 경로가 원리적으로 없고**, (2) MCC 0.30에서
잡음이 지배하며, (3) 모델이 카운티를 모르는 것이 블록 단위에서는 의도된 설계지만 사건
단위에서는 오염이 된다. 그래서 재수사 우선순위 목록은 **계산하되 공개하지 않고**
감사 통계만 낸다(`experiments/audit_priority.py`).

---

## 결론 한 줄

> 사건을 지역으로 묶는 그래프는 검거 예측 정확도를 올리고, **동시에 인종 격차의 잔여
> 전달 경로다.** 그 격차는 정확도 0.016을 내고 되갚을 수 있으며, 되갚은 뒤 남는 잔차는
> 특정 카운티에 집중되고 그 집중은 인종 구성과 상관된다 — 다만 상관의 강도는 카운티
> 규모에 따라 커진다.

| 단계 | 결과 |
|---|---|
| 예측 | GraphSAGE(geo) > XGBoost > LogReg. 단 마진이 시드 노이즈를 넘는 지표는 MCC·AUC·Balanced Accuracy **뿐**이다 |
| 진단 | 성별 격차는 **직접 열**이 전부(blind에서 소멸), 인종 격차는 일부가 **그래프로 전달** — 짝지은 차이 3개 주 +0.82 [+0.65, +1.03] / 전국 +0.173 [+0.135, +0.212]. 다만 전국에서는 평면 모델도 증폭을 멈추지 않는다(blind XGBoost 1.63) |
| 처방 | 손실 벌점으로 전국 인종 증폭비 1.805 → 0.337, 비용 −0.0156 MCC. 엣지 재배선은 **실패**했고 그 실패가 누수 위치를 찾아냈다 |
| 적용 | 카운티 1,803개 판정, 기대 초과 74개. 잔차 ↔ 흑인비중 **+0.288 [+0.244, +0.330]** |

전체 서술은 **[최종보고서](reports/최종보고서_CLEAR.md)** 에 있다.

---

## 전체 재현

### 데이터

- `dataset/kaggle_homicide_Reports_1980_2014.csv` (638,454행) — 대용량이라 git에
  없다(`.gitignore`). Kaggle "Homicide Reports, 1980-2014"에서 받는다.
- 타깃: `Crime Solved` (Yes=검거 / No=미해결). 불균형 약 68/32.
- **누수 열이 이 데이터의 지배적 제약이다.** 가해자 열(`Perpetrator *`,
  `Relationship`)은 미해결 사건에서 90–99%가 `Unknown`이라 넣으면 타깃이 자명해진다
  (AUC ~0.99). `config.LEAKAGE_COLS`가 이를 고정한다. 자세히는
  [데이터카드](reports/데이터카드_CLEAR.md).

### 스코프

`CLEAR_SCOPE` 환경변수 하나가 표본과 출력 경로를 함께 결정한다.

| 스코프 | 행 | 주 | 용도 |
|---|---|---|---|
| `ca_tx_mi` (기본) | 190,326 | 3 | 방법론 개발 · 기제 규명 |
| `national` | 638,454 | 51 | 최종 산출물 · 전국 지도 |

전국 값과 3개 주 값은 **직접 비교되지 않는다** — 작동점이 다르고 모든 공정성 수치가
임계값 의존이다. 대시보드도 두 스코프를 탭으로 분리해 둔다.

### 실행

```bash
pip install -r requirements.txt      # torch·PyG 포함(빠른 시작용과 다르다)
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

---

## 주요 산출물

| 산출물 | 위치 | git |
|---|---|---|
| **성능·공정성 대시보드** | `outputs/dashboard.html` | 재생성 |
| 프로젝트 소개 페이지 | `outputs/national/web/clear_story.html` | 재생성 |
| 전국 웹 지도 (대표) | `outputs/national/web/map_national.html` | 재생성 |
| 전국 웹 지도 (test 분할 참조) | `outputs/national/web/map_national_test.html` | 재생성 |
| 통합 실험 원장 (long format) | `outputs/results.csv` | **커밋** |
| 공정성 진단 표 | `outputs/[national/]fairness_*.csv` | **커밋** |
| 카운티 이상탐지 | `outputs/national/cold_blocks{,_cv5}.csv` | **커밋** |

**결과 CSV 43개(13MB)가 커밋돼 있는 것이 빠른 시작이 성립하는 근거다.** 그림·지도·중간
산출물은 재생성 가능하므로 제외하고 실험 기록인 작은 CSV는 남긴다는 규칙이
`.gitignore`에 이미 들어 있다.

원장이 하나인 것도 설계다 — `family`/`metric` 두 열로 "GNN이 기준선을 이겼나"가 파일
간 비교가 아니라 groupby 한 번이 된다. **새 지표는 새 행이고, 결코 새 열이 아니다.**

---

## 방법론 기준 (선행연구)

Campedelli(2022, *Journal of Criminal Justice*)를 **의식적으로** 따른다 — 전면
One-Hot, 나이 5년 구간화, 70/30 무작위 층화 분할, 5-fold 층화 CV. 데이터·분할·인코딩
축에서 논문과 비교 가능하게 하기 위한 선택이다.

평가지표만 의도적으로 벗어난다: AUC/MCC/F1/Sensitivity/Specificity(프로젝트 선택)에
**Balanced Accuracy + Precision**(논문이 그 둘만 보고하므로 비교용)을 더한 7종.
**MCC는 선택 기준**이고("대표 지표"가 아니다), 논문 비교 맥락에서는 Balanced Accuracy와
Precision이 앞에 온다. 어느 것이 앞에 오든 모델 순위는 같다.

---

## 주의 (윤리)

이 저장소의 이상탐지 결과는 **동일범 판정이 아니다.** 데이터에 가해자 ID가 없으므로 그
주장은 원리적으로 불가능하다. 그리고 **모델은 카운티를 특성으로 갖지 않으므로** 잔차
z는 카운티 고유 효과 전체를 담는다 — 차별·수사자원·도시성·기록관행이 분리되지 않는다.
**z와 인종 구성의 상관을 인과로 읽으면 안 되고**, 이 문장은 지도를 포함한 모든 산출물에
함께 실린다.

---

## 문서

읽는 순서를 권한다면:

1. **[최종보고서](reports/최종보고서_CLEAR.md)** — 네 단계 전체, 부정 결과, 한계
2. [문제정의서](reports/문제정의서_CLEAR.md) — 문제 설정과 윤리 게이트
3. [공정성진단 보고서](reports/공정성진단_보고서.md) — 격차 측정의 전말
4. [확장설계서](reports/확장설계서_CLEAR.md) — 사이드 트랙 · 이상탐지 · 전국 지도
5. [크로스피팅 계획서](reports/크로스피팅_계획서_CLEAR.md) — 사전 등록과 그 판정

사전 등록 계획서(각 문서가 계획과 그 판정을 함께 담는다):
[크로스피팅](reports/크로스피팅_계획서_CLEAR.md) ·
[카운티 차분](reports/카운티차분_계획서_CLEAR.md) ·
[짝지은 표준화 대조](reports/짝지은표준화대조_계획서_CLEAR.md) ·
[SHR 정황 조인](reports/SHR정황조인_계획서_CLEAR.md) ·
[로컬 배포](reports/로컬배포_계획서_CLEAR.md) ·
[포트폴리오 웹페이지](reports/포트폴리오웹페이지_계획서_CLEAR.md) ·
[Return A 교차검증](reports/ReturnA교차검증_계획서_CLEAR.md) ·
[층 표준화 벌점](reports/층표준화벌점_계획서_CLEAR.md)(**미실행** — 등록만)

그 외: [EDA](reports/EDA_보고서.md) · [개발계획서](reports/AI모델_개발계획서_살인사건검거_GNN.md) ·
[선행연구 비교](reports/모델벤치마크_선행연구비교.md) · [데이터카드](reports/데이터카드_CLEAR.md) ·
모델보고서 [GraphSAGE](reports/모델보고서_GraphSAGE.md) ·
[XGBoost](reports/모델보고서_XGBoost.md) · [LogReg](reports/모델보고서_LogReg.md)

---

## 라이선스

| 대상 | 라이선스 |
|---|---|
| `src/**` (코드) | MIT |
| `outputs/**`, `reports/**`, 문서, 생성 HTML | **CC BY-SA 4.0** — 원자료를 따른다 |
| 생성 HTML에 포함된 서체 서브셋 | SIL Open Font License 1.1 |

원자료는 Murder Accountability Project의
[Homicide Reports, 1980–2014](https://www.kaggle.com/datasets/murderaccountability/homicide-reports)(Kaggle)이고
**[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/deed.ko)** 이다. 정제·표본추출·
특성공학·모델 예측 부가·카운티 집계의 변경을 가했으며, **share-alike 조건에 따라 데이터
파생 산출물을 같은 라이선스로 배포한다.** 코드는 데이터를 담고 있지 않아 파생물이 아니므로
MIT다. `outputs/predictions/*.csv` 4개는 원자료의 열 값을 행 단위로 담으므로 명확한
파생물이며, 원본 CSV 자체는 재배포하지 않는다.

대시보드와 소개 페이지에는 나눔스퀘어_ac 서브셋이 base64로 들어 있다. OFL 1.1이
사본마다 요구하므로 **전문을 각 HTML 상단 주석에 함께 싣고**(외부 링크로 두면 "외부 요청
0"이라는 산출물 규율을 라이선스에서 깨게 된다), 서브셋은 예약 이름을 피해 내부 패밀리명을
바꾼다. 전문은 [OFL.txt](OFL.txt), 범위 전체는 [LICENSE](LICENSE)에 있다.

---

## 이력

주차별 로드맵(데이터·EDA·베이스라인 → 그래프·GNN → 진단·처방 → 분석·시각화)은 4주차까지
완료됐고, 그 뒤 전국 확장 · 크로스피팅 · 층 표준화가 추가됐다. 초기 파이프라인 중
`eda.py`·`train_baseline.py`·`mitigate_threshold.py`·`ablation.py`는 결과를 얻은 뒤
저장소에서 제거했다(커밋 `96d1dd1`에서 복구 가능하며, 결과는 `outputs/results.csv`와
커밋된 예측 덤프에 남아 있다). 그 판단의 근거는 `CLAUDE.md`의 "Scope" 절에 있다.
