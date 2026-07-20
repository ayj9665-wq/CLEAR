# CLEAR — Clearance Learning & Equity Assessment on gRaphs

미국 살인사건 데이터(Murder Accountability Project, 1980–2014)를 사건 그래프로
재구성해 **검거 여부(clearance)를 예측**하고, 피해자 인종·성별에 따른
**검거율 격차를 진단**하며, 완화기법으로 그 격차를 **얼마나 줄일 수 있는지 처방**한다.

하나의 질문("무엇이 검거를 결정하는가")을 **예측 → 진단 → 처방**으로 파고든다.
그래프(GNN)는 이 전 단계를 관통하는 예측 도구다.

## 데이터
- `dataset/kaggle_homicide_Reports_1980_2014.csv` (약 63.8만 행)
- 타깃: `Crime Solved` (Yes=검거 / No=미해결)
- 대용량이라 git 에는 커밋하지 않음(`.gitignore`).

## 파이프라인 (1주차 구현)
```
01_clean.py      데이터 정제(누수 열 제거, 타깃 인코딩)
02_sample.py     주(州) 단위 표본 추출(기본 California)
03_features.py   특성 공학(전면 One-Hot, 나이 5년 구간화)
eda.py           검거율·인종/성별 격차 사전 확인
05_train_baseline.py  flat 베이스라인(XGBoost/Logistic) — GNN 비교 기준선
```

## 실행
```bash
pip install -r requirements.txt
cd src
python 01_clean.py
python 02_sample.py
python 03_features.py
python eda.py
python 05_train_baseline.py
```

## 방법론 기준 (선행연구)
Campedelli(2022, *Journal of Criminal Justice*) 기준을 따름:
전면 One-Hot, 나이 5년 구간화, 70/30 무작위 층화 분할, 5-fold 층화 CV,
평가지표 Balanced Accuracy + Precision.

## 로드맵
- **1주차(현재)**: 데이터·EDA·베이스라인 ← 구현 완료
- 2주차: 그래프 구성 + GNN 예측
- 3주차: 공정성 진단 + 완화기법(처방)
- 4주차: 분석·시각화·포스터
