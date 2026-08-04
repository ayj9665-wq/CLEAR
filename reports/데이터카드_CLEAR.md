# 데이터 카드 — CLEAR

## 1. 기본 정보

- 작성자: 안윤지
- 작성일: 2026-07-24
- 프로젝트 가제: CLEAR — Clearance Learning & Equity Assessment on gRaphs
  ("누가 잊히는가: 살인사건 검거 예측과 공정성")
- 관련 문서: [문제정의서_CLEAR.md](문제정의서_CLEAR.md) ·
  [AI모델_개발계획서_살인사건검거_GNN.md](AI모델_개발계획서_살인사건검거_GNN.md) ·
  [EDA_보고서.md](EDA_보고서.md)

## 2. 데이터 출처

- 데이터셋 이름: Kaggle "Homicide Reports, 1980–2014" (원자료: Murder Accountability
  Project, MAP — FBI Supplementary Homicide Report + FOIA 수집분)
- 출처 URL: https://www.kaggle.com/datasets/murderaccountability/homicide-reports
- 다운로드 날짜: 2026-07-20 (원본 CSV 파일 타임스탬프 기준)
- 라이선스/이용 조건: **CC BY-SA 4.0** (2026-08-04 Kaggle 데이터셋 페이지에서 확인).
  두 가지 의무가 따라온다. **BY** — 창작자·출처·라이선스명·링크와 **변경 사실**을
  명시한다(웹 지도·소개 페이지·대시보드 하단과 `LICENSE`에 기재). **SA** — 파생물을
  같은 라이선스로 배포한다. 그래서 이 저장소는 `outputs/**`·`reports/**`·문서·생성
  HTML을 CC BY-SA 4.0으로, `src/**` 코드를 MIT로 나눠 건다(`LICENSE` 참조).
  코드는 데이터를 담고 있지 않으므로 파생물이 아니다.
  share-alike가 실제로 걸리는 곳은 `outputs/predictions/*.csv` 4개다 — 원자료의 열
  값을 행 단위로 담는다(`y_true` = `Crime Solved`, 피해자 인종·성별, 57,098행).
  나머지 결과 CSV는 카운티·모델 단위 집계 통계라 법적 근거는 약하지만, 파일별로
  다투는 대신 같은 조건을 건다. 원본 CSV 자체는 재배포하지 않는다(`.gitignore`).
  MAP 원자료는 FBI SHR 공개자료를
  기반으로 하나, Kaggle 재배포본의 조건은 별도이므로 추정해서 적지 않는다.
  최소한 출처 표기(MAP + Kaggle)는 모든 산출물에 유지한다.
- 데이터가 담고 있는 기간/범위: 1980 ~ 2014년 (35년), 미국 50개 주 + DC.
  한 행 = 한 건의 살인/과실치사 사건 보고 레코드.

## 3. 데이터 구조

- 규모 (3단계 파이프라인이라 단계별로 기록):

| 단계 | 산출물 | 행 | 열 | 비고 |
|---|---|---|---|---|
| 원본 | `dataset/kaggle_homicide_Reports_1980_2014.csv` | 638,454 | 24 | ~107MB, git 미포함 |
| 정제 후 (`01_clean.py`) | `data/processed/clean.parquet` | **638,454** | **12** | 행 삭제 0건, 열만 13개 제거 + `solved` 추가 |
| 표본 추출 (`02_sample.py`) | `data/processed/sample.parquet` | **190,326** | 12 | California·Texas·Michigan 3개 주 |
| 특성 공학 (`03_features.py`) | `data/processed/features.parquet` | 190,326 | 76 | X 73열 + `solved` + `sens__*` 2열 |

**스코프 축 (2026-07-28 추가).** 위 표는 기본 스코프(`ca_tx_mi`)다. 전국 확장으로
`CLEAR_SCOPE=national` 축이 생겼고, 같은 스크립트가 별도 디렉터리에 산출물을 낸다.

| 단계 | 경로 | 행 | 열 | 비고 |
|---|---|---|---|---|
| 표본 추출 | `data/processed/national/sample.parquet` | **638,454** | 12 | 주 필터 없음, **51개 주** |
| 특성 공학 | `data/processed/national/features.parquet` | 638,454 | 126 | X **123열** + `solved` + `sens__*` 2열 |

X가 73 → 123열로 는 것은 `State` 더미가 3개에서 51개가 됐기 때문이다. `clean.parquet`은
**스코프 밖에 있다** — `01_clean.py`에 주 필터가 없어 산출물이 이미 전국이고 두
스코프가 공유한다.

두 스코프의 산출물을 반드시 분리해야 하는 이유는 편의가 아니다. 예측 덤프의
`row_index`는 `features.parquet`의 **행 위치**라, 표본이 바뀐 뒤 옛 덤프를 읽으면
민감속성이 엉뚱한 행과 조인되면서도 **에러 없이 그럴듯한 표**가 나온다.

**예측 덤프에는 두 종류가 있다(2026-07-29 추가).** 대부분은 test 분할 예측이지만,
cross-fitting 덤프는 **전체 표본**을 담는다 — 그래서 행 수와 `row_index` 범위가 다르다.

| 덤프 | 행 | `row_index` 범위 | 성격 |
|---|---|---|---|
| `graphsage_*.csv` 등 | 191,537 (전국 test) | test 인덱스 부분집합 | 학습에 안 쓰인 30% |
| `graphsage_fairloss_a100_mb_cv5.csv` | **638,454** | 0..638,453 **전부** | 5-fold **out-of-fold** 예측 |

후자는 라벨이 out-of-fold이므로 낙관 편향이 없지만, transductive 학습이라 held-out
노드의 **특성**은 메시지 전달로 학습에 참여한다(라벨은 아니다). 그래서 "완전한
out-of-sample"이 아니라 **"out-of-fold 라벨 예측"**이라고 부른다. 행 수가 다르므로
`assert_same_test_set`이 다른 덤프와 나란히 놓는 것을 (정당하게) 거부하고,
`predictions.discover()`가 `_cv5` 라벨을 자동 수집에서 제외한다.

  → 정제 단계에서 **행은 한 건도 버리지 않았다.** 결측이 `NaN`이 아니라
  `"Unknown"` 문자열/`998` 코드로 인코딩돼 있어서, 삭제 대상이 아니라 하나의
  범주값으로 살려 두는 편이 정보 손실이 적다고 판단했기 때문이다(4절 참조).
  행이 638,454 → 190,326으로 준 것은 품질 문제가 아니라 **분석 범위를 3개 주로
  좁힌 설계 결정**이다(6절 참조).

- y 타깃 열: `solved` — 원본 `Crime Solved`(Yes/No)를 이진 인코딩한 값.
  **1 = 검거(사건 해결), 0 = 미해결.**
  분포: 전국 70.2% / 29.8% → CA·TX·MI 표본에서 **68.2% / 31.8%**.
  약 7:3의 중간 수준 불균형 (정확도 대신 MCC·Balanced Accuracy로 평가하는 이유).

- X 주요 입력 열:

| 열 이름 | 의미 | 타입 | 처리 |
|---|---|---|---|
| `Victim Race` | 피해자 인종 (5종) | 범주 | One-Hot **+ `sens__` 사본** (민감속성) |
| `Victim Sex` | 피해자 성별 (3종) | 범주 | One-Hot **+ `sens__` 사본** (민감속성) |
| `Victim Age` | 피해자 나이 | 정수 → 범주 | 5년 단위 구간화 후 One-Hot (미상은 `Unknown` 구간) |
| `Weapon` | 사용 흉기 (16종) | 범주 | One-Hot. 검거율 51.9~87.9%로 변별력 큼 |
| `State` / `City` | 발생 주 / 도시 | 범주 | `State`만 One-Hot. `City`는 X에 없지만 **그래프 엣지의 블로킹 키**로 사용 |
| `Year` / `Month` | 발생 연·월 | 정수 / 범주 | `Year`→10년 단위 `Decade` 범주화, `Month` One-Hot |
| `Agency Type` | 신고 기관 유형 (7종) | 범주 | One-Hot |
| `Victim Count` | 추가 피해자 수 (0–10) | 정수 | 유일한 연속형 입력. 91.8%가 0 |

  → 논문(Campedelli 2022) 기준을 따라 **임베딩·순서형 인코딩 없이 전면 One-Hot**,
  나이는 5년 구간화. 최종 X = 73열 (수치 1 + 더미 72).

## 4. 품질 진단 결과

> 발견한 문제를 그대로 기록합니다.

**발견한 문제 1 — 타깃 누수(target leakage). 이 데이터셋의 가장 큰 함정.**
가해자 관련 열(`Perpetrator Sex/Age/Race/Ethnicity/Count`, `Relationship`)의
`Unknown` 비율이 타깃과 거의 완벽하게 대응한다:

| 열 | 미해결(No) 사건 중 Unknown | 해결(Yes) 사건 중 Unknown |
|---|---|---|
| `Perpetrator Sex` | 99.97% | 0.03% |
| `Perpetrator Race` | 99.85% | 1.35% |
| `Relationship` | 93.6% | 21.2% |

미해결 사건은 정의상 가해자를 특정하지 못했으므로, 이 열들은 상관이 아니라
**결정적 대응** 관계다. 넣고 학습하면 AUC가 0.99 근처로 뜨는데 이는 성능이
아니라 정답을 베낀 것이다.

**발견한 문제 2 — `NaN`이 0개인데 결측이 없는 게 아니다.**
`pandas.isna()` 기준 결측 0건. 그러나 실제로는 `"Unknown"` 문자열로 위장돼 있다:
`Victim Ethnicity` 57.7%, `Relationship` 42.8%, `Victim Race` 1.05%(6,676건),
`Victim Sex` 0.15%(984건). 결측 점검을 `isna()`로만 했다면 전부 놓쳤을 항목이다.

**발견한 문제 3 — 숫자 코드로 위장된 결측.**
`Victim Age`에 미상 코드 `998`이 974건, 100세 이상 이상치가 974건(사실상 동일 집합).
별개로 `Victim Age == 0`이 8,444건(표본 기준 2,269건) 있는데, **실제 영아 피해자인지
미상을 0으로 잘못 채운 것인지 구분되지 않는다.** `Perpetrator Age`는 dtype이
`object`인데 값의 33.9%가 문자열 `"0"`이다.

**발견한 문제 4 — 주(州) 이름 오타.**
`State` 열에 `"Rhodes Island"` 1,211건 (정식 명칭 Rhode Island). 그래프 엣지를
`State+City` 블로킹으로 구성하므로, 표준화하지 않으면 별개의 주로 잘못 분리돼
엣지가 끊긴다.

**발견한 문제 5 — 민감속성 소수 그룹의 극심한 희소성.**
표본(CA·TX·MI) 기준 `Victim Race` 분포:
White 113,296 / Black 69,062 / Asian·Pacific Islander 4,705 / Unknown 2,625 /
**Native American·Alaska Native 638**. 상위 2개 그룹이 96%를 차지한다.
공정성 지표를 max−min으로 계산하면 **n=638짜리 그룹이 헤드라인 숫자를 결정해
버린다.** 개발계획서 Plan B 리스크표의 "민감속성 소수 그룹 희소" 항목이 실제로
실현된 사례.

**발견한 문제 6 — 저정보·식별자 열.**
`Crime Type` 98.6% 단일값, `Incident`는 관할·월 내 재사용 일련번호라 전역 의미
없음, `Agency Code`/`Agency Name`은 12,003 / 9,216 유일값의 고카디널리티 식별자
(One-Hot 시 차원 폭발).

## 5. 정제 로그 (무엇을 / 왜 / 어떻게)

| 무엇을 | 왜 | 어떻게 | 영향 |
|---|---|---|---|
| 가해자 6개 열 (`Perpetrator Sex/Age/Race/Ethnicity/Count`, `Relationship`) | 타깃 누수 — 미해결 사건에서 99% Unknown이라 정답 노출 | `config.LEAKAGE_COLS`로 지정해 `01_clean.py`에서 제거 | 24열 → 18열. **AUC가 0.99에서 0.70대로 정상화** |
| `Crime Solved` (Yes/No) | 문자열 타깃은 학습 불가 | `(== "Yes").astype(int)` → `solved` 열로 대체, 원본 삭제 | 열 종류 유지, 타입 object → int |
| `Record ID`, `Agency Code`, `Agency Name` | 식별자·고카디널리티 (유일값 12,003 / 9,216) | `config.DROP_COLS`로 제거. 지리 정보는 `State`/`City`/`Agency Type`으로 일반화 | 18열 → 15열. 차원 폭발 회피 |
| `Crime Type` | 98.6%가 단일값이라 분산 없음 | 제거 | 학습 무의미 열 1개 감소 |
| `Incident` | 관할·월 내 재사용 일련번호, 사건 간 비교 의미 없음 | 제거 | — |
| `Record Source` | FBI 96.6% / FOIA 3.4%, 수집 경로 메타데이터 | 제거 | 최종 12열 |
| `Victim Age`의 `998` 및 100세 이상 | MAP 미상 코드 + 이상치. 나이 998로 학습하면 왜곡 | `NaN` 처리 후, `03_features.py`에서 `Unknown` 구간으로 One-Hot | **행 삭제 0건.** 표본 기준 199건이 `Unknown` 나이 구간으로 |
| `Victim Age` 연속값 | 검거율이 나이에 U자형(0–10세 89.7%, 20–30세 67.4%)이라 선형 가정이 안 맞음 | 5년 단위 구간화 후 One-Hot (논문 기준) | 연속 1열 → 더미 21열 |
| `Year` | 35년치 연도를 그대로 더미화하면 35열 | 10년 단위 `Decade`로 묶음 | 더미 4열 |
| 범주형 8종 전체 | 논문 기준 — 임베딩/순서형 아닌 전면 One-Hot | `pd.get_dummies(prefix_sep="=")` | **최종 X 73열** |
| `"Unknown"` 문자열 결측 | 삭제하면 `Victim Ethnicity`만으로 57.7%가 날아감. 게다가 "기록되지 않았다"는 사실 자체가 신호일 수 있음 | **대체·삭제하지 않고 하나의 범주값으로 유지** | 행 손실 0. 단, 공정성 격차 계산에서는 `Unknown` 그룹 제외 (인구집단이 아니라 기록 누락 코드이므로) |
| `Victim Race`·`Victim Sex` | 공정성 진단에 원본 값이 필요한데 One-Hot 후엔 복원이 번거로움 | 인코딩 **전에** `sens__Victim Race` / `sens__Victim Sex`로 사본 저장 (행 순서 보존) | 열 +2. 진단 단계가 모델 재학습 없이 CSV만 읽으면 되게 함 |
| 주 필터: California·Texas·Michigan | 6절 참조 (품질 문제가 아니라 설계 결정) | `config.SAMPLE_STATES` | 638,454행 → **190,326행 (29.8%)** |
| `"Rhodes Island"` 오타 | 표준화 필요하다고 EDA에서 진단 | ❌ **미조치** — 표본이 CA·TX·MI라 당장 영향은 없으나, 전국 확장 시 반드시 선행되어야 함 | 미해결 (6절) |

## 6. 특이사항 및 한계

**(1) 표본은 전국이 아니라 3개 주다 — 결과를 "미국 전체"로 일반화하면 안 된다.**
전국을 그대로 풀링하지 않은 이유는 메모리가 아니라 **심슨의 역설 위험**이다.
주별 검거율 편차가 매우 크고(예: SC 90.8% vs NY 54.1%), 인종 구성도 주마다 다르다.
전국을 합치면 "인종 격차"로 보이는 것이 실은 "주 구성 차이"일 수 있다.
그래서 논문(Campedelli 2022)이 실제로 분석한 3개 주 조합을 택했다.
다만 이 표본의 검거율(68.2%)은 전국(70.2%)보다 낮고, 특히 California 단독은
**63.6%**로 전국 대비 6.6%p 낮다 (Texas 76.4%, Michigan 66.8%).
→ **보고서에 "CA·TX·MI 한정"을 항상 명시할 것.**

**(2) 민감속성이 X에서 제거된 게 아니다 — `sens__`는 사본이지 이동이 아니다.**
가장 오해하기 쉬운 지점이라 명시한다. `03_features.py`가 `sens__Victim Race`를
만들지만, `config.CATEGORICAL_COLS`에도 `Victim Race`/`Victim Sex`/`Victim Ethnicity`가
그대로 들어 있어서 **`Victim Race=Black` 같은 더미 11개가 X(73열)에 남아 있다.**
즉 기본 설정의 모델은 인종·성별을 **직접 보고** 학습한다.
`load_xy(blind=True)`(`--blind` 플래그)를 써야 그 더미가 빠진다.
`Victim Ethnicity`도 blind 목록에 포함했는데, 히스패닉 여부가 인종의 강한 직접
프록시라 남겨두면 "모델이 인종을 못 본다"는 조건이 성립하지 않기 때문이다.

**(3) `Victim Age == 0`(표본 2,269건)의 해석이 여전히 모호하다.**
실제 영아 피해자인지 미상 오코딩인지 미검증. 0–10세 구간의 검거율이 89.7%로
가장 높은데, 이 구간에 미상이 섞여 있다면 그 수치는 부풀려진 것일 수 있다.
→ 후속 검증 권장 (`Relationship`·`Weapon`과 교차 확인).

**(4) `"Rhodes Island"` 오타 미조치.**
현 표본에는 포함되지 않아 영향이 없지만, `01_clean.py`에 주명 표준화 매핑이
없다는 사실은 남는다. **전국으로 확장하기 전에 반드시 선행 조치.**

**(5) 소수 인종 그룹으로는 신뢰할 만한 공정성 비교가 불가능하다.**
Native American/Alaska Native n=638, Asian/Pacific Islander n=4,705.
그래서 공정성 진단은 (a) 격차를 **두 개의 그룹 최소표본 기준**으로 항상 병기하고,
(b) **White vs Black 이원 비교**를 주축 지표로 삼는다.
max−min은 노이즈가 낀 추정치들의 최대값이라 **상향 편향**된 통계량이며,
부트스트랩 신뢰구간은 그 구간을 보여줄 뿐 편향을 제거하지 않는다.

**(6) 데이터 자체에 실질적 격차가 존재한다 — 이건 결함이 아니라 연구 대상이다.**
표본 기준 검거율: White 70.2% vs **Black 65.6%** (4.6%p),
Female 74.9% vs **Male 66.6%** (8.3%p).
이 격차는 정제로 없앨 대상이 아니라 프로젝트가 진단하려는 현상 그 자체다.
다만 **모델이 이 격차를 그대로 재현하는지, 증폭하는지**는 별개 문제이며,
그것을 재는 지표가 증폭비(모델 격차 ÷ 데이터 실제 격차)다.
동시에, 원자료의 격차는 교란변수(흉기·관계·지역)를 통제하지 않은 단순 교차표이므로
**그 자체로 인과를 주장하지 않는다.**

**(7) 이 데이터는 "사건"이 아니라 "경찰의 보고"를 기록한 것이다.**
`Crime Solved`는 객관적 진실이 아니라 **경찰이 해결로 분류했는지**를 뜻한다.
신고되지 않은 사건, 보고서를 제출하지 않은 기관의 사건은 애초에 이 데이터에
존재하지 않는다(MAP가 FOIA로 일부 보완했으나 완전하지 않다).
따라서 발견되는 격차에는 **실제 수사 자원 배분 격차 + 보고 관행 격차**가
분리되지 않은 채 섞여 있다. 이 데이터로는 둘을 구분할 수 없다.

**(8) 선행 연구의 핵심 예측 변수 2개가 이 CSV에 없다.**
Campedelli(2022)의 SHAP 상위 2개 변수가 `Circumstance`(Kaggle CSV에 아예 부재)와
`Number of Offenders`(= `Perpetrator Count`, **누수라서 의도적으로 제외**)다.
그래서 우리 Balanced Accuracy(XGB ≈ 0.65)가 논문(전국 0.767, California 0.802)보다
낮은 것은 구현 결함이 아니라 **입력 정보량의 차이**다.

## 7. 최종 파일

| 파일 | 내용 | git 추적 |
|---|---|---|
| `dataset/kaggle_homicide_Reports_1980_2014.csv` | 원본 (638,454 × 24, ~107MB) | ❌ (용량) |
| `data/processed/clean.parquet` | 정제본 (638,454 × 12) | ❌ (재생성 가능) |
| `data/processed/sample.parquet` | 3개 주 표본 (190,326 × 12) | ❌ |
| `data/processed/features.parquet` | 학습 입력 (190,326 × 76) | ❌ |

- **원본 보존 여부: ✅** — `01_clean.py`는 원본 CSV를 절대 덮어쓰지 않고,
  매 단계가 **새 parquet를 쓰는 단방향 파이프라인**이다. 각 중간 산출물이 남아
  있어 어느 단계에서 무엇이 바뀌었는지 되짚을 수 있다.
- `data/processed/`는 전체가 gitignore 대상이다. 대신 **재현 경로를 보장**한다 —
  각 단계가 앞 단계의 산출물을 읽으므로 **한 줄씩, 순서대로** 실행할 것
  (`&&`로 잇지 않는다: Windows PowerShell 5.1에는 `&&`가 없어 줄 전체가
  파서 오류로 죽고, `;`로 바꾸면 앞 단계가 실패해도 다음 단계가 낡은 parquet를
  읽어 조용히 진행된다):

  ```
  cd src
  python 01_clean.py
  python 02_sample.py
  python 03_features.py
  ```

  모든 경로·상수·난수 시드(`RANDOM_STATE = 42`)가 `src/config.py` 한 곳에 있어
  같은 원본 CSV에서 항상 동일한 산출물이 나온다.
- 결과 CSV(`outputs/*.csv`)는 실험 기록이므로 **git으로 추적한다**
  (예측 덤프 `predictions_*.csv`와 PNG는 용량 문제로 제외).
