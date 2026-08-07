# docs/ — GitHub Pages 배포용 사본

여기 있는 `index.html`은 **생성물의 사본**이다. 손으로 고치지 말 것 — 다음 빌드 때 덮어써진다.

원본과 재생성 방법:

```bash
cd src
CLEAR_SCOPE=national python -m experiments.build_story_page
# -> outputs/national/web/clear_story.html
```

그 뒤 사본을 갱신한다(리포지터리 루트에서):

```powershell
Copy-Item outputs\national\web\clear_story.html docs\index.html -Force
```

## 왜 이 디렉터리만 gitignore 예외인가

`.gitignore`는 `outputs/web/`·`outputs/*/web/`·`outputs/dashboard.html`을 제외한다.
생성 HTML은 커밋된 결과 CSV에서 재생성되는 산출물이고, 그게 clone-and-run 계약이기
때문이다. 그 규칙은 그대로 두고, **배포 경로만 따로 둔다**:

- GitHub Pages는 브랜치의 파일을 그대로 서빙하므로 어딘가에는 커밋된 사본이 있어야 한다.
- CI에서 빌드할 수는 없다. `build_story_page`가 `dataset/geo/`의 county GeoJSON을 읽는데
  그 디렉터리는 gitignore 대상(재다운로드 가능한 참조 데이터)이라 러너에 존재하지 않는다.

그래서 `outputs/` 아래는 계속 "재생성물, 커밋하지 않음"이고, `docs/` 아래만
"배포된 스냅샷, 커밋함"이다. 두 규칙이 섞이지 않도록 경로로 분리한 것이다.

## `.nojekyll`

Pages의 Jekyll 처리를 끈다. 이 페이지는 완결된 단일 HTML이라 빌드 단계가 필요 없고,
Jekyll이 파일을 건드릴 여지를 없애는 편이 안전하다.

## 라이선스

페이지 자체는 데이터 파생 산출물이므로 **CC BY-SA 4.0**(출처: Murder Accountability
Project / Kaggle, "Homicide Reports, 1980-2014"). 본문 서체 나눔스퀘어_ac는 SIL OFL 1.1이며
전문이 파일 첫 주석에 포함되어 있다. 자세한 내용은 리포지터리 루트의 라이선스 고지 참조.
