"""CLEAR 공용 라이브러리.

번호로 시작하는 파이프라인 스크립트(05_train_baseline.py 등)는 파일명이
숫자로 시작해 import가 불가능하다(`import 05_train_baseline`는 문법 오류) —
그래서 그동안 공유 로직(데이터 로딩·분할·평가지표)이 스크립트마다 복붙됐다.
이 패키지가 그 공유 로직의 단일 출처다. src/에서 실행하면
`from clear.data import load_xy, get_split` / `from clear.metrics import evaluate`
로 쓴다.
"""
