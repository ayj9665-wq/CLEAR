/*
 * experiments/_domprobe.js -- 배포 HTML을 jsdom에서 실제로 실행하고 DOM을 조회한다.
 *
 * 왜 필요한가. 이 저장소가 배포까지 흘려보낸 유일한 JS 결함은 build_web_map의
 * legend()가 정의된 적 없는 D를 참조해 ReferenceError를 낸 것이었다. paint()가
 * 카운티를 다 칠한 **직후** 거기서 멈췄으므로 지도는 멀쩡해 보였고, 범례와 순위표만
 * 통째로 비어 있었다. 문법 검사(_htmlcheck)는 통과시킨다 -- 문법은 유효했다.
 * 방출된 path/fill을 되읽는 검증기도 이 코드를 건드리지 않는다.
 * **실행 중인 DOM을 조회해야만 잡힌다.**
 *
 * 하는 일은 조회뿐이다. 판정은 하지 않는다 -- 관측값을 JSON으로 stdout에 내보내고,
 * 단언과 CSV 대조는 verify_html.py(파이썬)가 한다. 데이터를 아는 쪽이 데이터를
 * 판정해야 하기 때문이다.
 *
 * 사용: node _domprobe.js <html경로> <kind: map|story|dashboard>
 */
'use strict';
const fs = require('fs');
const path = require('path');
const { JSDOM } = require(path.join(__dirname, '..', '..', 'node_modules', 'jsdom'));

const [, , file, kind] = process.argv;

// 페이지가 던지는 모든 것을 모은다. 이것이 이 검사의 핵심이고, 나머지는 보강이다.
const errors = [];

function textOf(el) { return el ? (el.textContent || '').trim() : null; }

/* tableSel: 순위표의 CSS 선택자. build_web_map은 '#t', 소개 페이지는 '#ct'다.
 *
 * 이 인자가 생긴 이유를 남겨 둔다. 처음에는 '#t'를 박아 두었고, 소개 페이지에서
 * tableRows가 모든 수준에서 0으로 나왔다. **표가 비어 있다**로 읽히는 값이었지만
 * 실제로는 선택자가 안 맞았을 뿐이다 -- 검사기 자신이 "그럴듯한데 틀린" 답을 낸 것이다.
 * 그래서 지금은 tbody 요소의 **존재 여부를 따로 보고**한다: 요소가 없는 것과 요소가
 * 있는데 행이 0인 것은 완전히 다른 사건이고, 검사기는 그 둘을 섞으면 안 된다.
 */
function probeMap(doc, win, tableSel = '#t') {
  const tbody = doc.querySelector(tableSel + ' tbody');
  // 한 수준에서의 관측치. 슬라이더를 옮길 때마다 다시 부른다.
  const snap = () => ({
    level: textOf(doc.getElementById('mnv')),
    legendSwatches: doc.querySelectorAll('#leg .sw').length,
    legendSpans: doc.querySelectorAll('#leg > span').length,
    tableFound: !!tbody,
    tableRows: tbody ? tbody.querySelectorAll('tr').length : -1,
    // 유의 카운티는 paint()가 .sig 클래스를 건다(색만으로 의미를 싣지 않으려고).
    sigPaths: doc.querySelectorAll('#g path.sig').length,
    hatched: [...doc.querySelectorAll('#g path')]
      .filter(p => (p.getAttribute('fill') || '').includes('url(#na)')).length,
    painted: doc.querySelectorAll('#g path').length,
  });

  const slider = doc.getElementById('mn');
  const levels = [];
  const nLevels = slider ? (+slider.max + 1) : 0;
  for (let i = 0; i < nLevels; i++) {
    slider.value = String(i);
    slider.dispatchEvent(new win.Event('input'));   // paint() -> legend() + table()
    levels.push(snap());
  }

  // 레이어 토글. res 이외의 레이어에서도 legend()가 살아 있어야 한다.
  const layers = {};
  for (const b of doc.querySelectorAll('[data-layer]')) {
    b.dispatchEvent(new win.MouseEvent('click', { bubbles: true }));
    layers[b.dataset.layer] = {
      legendSwatches: doc.querySelectorAll('#leg .sw').length,
      tableRows: tbody ? tbody.querySelectorAll('tr').length : -1,
      pressed: b.getAttribute('aria-pressed'),
    };
  }

  // 표 접기 토글
  const wrap = doc.getElementById('tablewrap');
  const before = wrap ? wrap.hidden : null;
  const tbtn = doc.getElementById('tbl');
  if (tbtn) tbtn.dispatchEvent(new win.MouseEvent('click', { bubbles: true }));
  const after = wrap ? wrap.hidden : null;

  return { levels, layers, tableToggle: { before, after } };
}

/*
 * jsdom이 구현하지 않는 것을 최소한으로 채운다.
 *
 * 셰임이 정당한 이유는 **이 페이지가 그 API를 무엇에 쓰는가**에 달려 있다.
 * IntersectionObserver는 스크롤 리빌에만 쓰인다 -- 관측 대상에 'in' 클래스를 붙이고
 * unobserve하는 것이 전부다. 그래서 "즉시 교차했다"고 답하는 셰임은 **끝까지 스크롤한
 * 상태**와 같고, 우리가 단언하고 싶은 것도 바로 그 상태다.
 *
 * getBoundingClientRect는 셰임이 필요 없다. 소개 페이지에서의 유일한 용도가
 * `void c.getBoundingClientRect()` -- 반환값을 버리는 리플로우 강제라, jsdom이 0을
 * 돌려줘도 무해하다. requestAnimationFrame은 페이지가 일부러 **안 쓴다**(프레임이
 * 스로틀되면 확대가 통째로 빠진다는 주석이 코드에 있다).
 *
 * 즉 여기서 흉내내는 것은 레이아웃이 아니라 **가시성 이벤트 하나**뿐이다. 레이아웃에
 * 의존하는 단언을 새로 추가하려면 그때는 진짜 브라우저가 필요하다.
 */
function installShims(win) {
  win.IntersectionObserver = class {
    constructor(cb) { this._cb = cb; }
    observe(el) { this._cb([{ target: el, isIntersecting: true }], this); }
    unobserve() {}
    disconnect() {}
    takeRecords() { return []; }
  };
}

function probeStory(doc, win) {
  const sections = [...doc.querySelectorAll('.sec, #mapsec, footer')];
  return {
    // 스크립트가 끝까지 돌았는지의 카나리아. 이 클래스는 스크립트 첫머리에서 붙고,
    // CSS는 .js가 붙었을 때만 본문을 숨긴다 -- 스크립트가 죽으면 페이지가 빈 화면이
    // 되는 대신 리빌 없이 그냥 보이도록 설계돼 있다(점진적 향상).
    jsClass: doc.documentElement.classList.contains('js'),
    sections: sections.length,
    revealed: sections.filter(s => s.classList.contains('in')).length,
    counters: [...doc.querySelectorAll('[data-count]')].map(b => ({
      want: String(b.dataset.count || ''),
      text: (b.textContent || '').trim(),
    })),
    hero: !!doc.getElementById('hero'),
    // 지도 절은 build_web_map과 같은 골격이므로 같은 조회를 재사용한다.
    map: probeMap(doc, win, '#ct'),
  };
}

function probeDashboard(doc, win) {
  const tabs = [...doc.querySelectorAll('[data-scope]')];
  const paneState = () => [...doc.querySelectorAll('[data-pane]')].map(p => ({
    scope: p.dataset.pane,
    hidden: p.hidden,
    panels: p.querySelectorAll('.pnl').length,
    tables: [...p.querySelectorAll('table')].map(t => t.querySelectorAll('tbody tr').length),
  }));

  const onLoad = paneState();
  const clicks = {};
  for (const b of tabs) {
    b.dispatchEvent(new win.MouseEvent('click', { bubbles: true }));
    clicks[b.dataset.scope] = {
      panes: paneState(),
      pressed: tabs.map(t => t.getAttribute('aria-pressed')),
    };
  }
  return { tabs: tabs.map(t => t.dataset.scope), onLoad, clicks };
}

const html = fs.readFileSync(file, 'utf8');
const dom = new JSDOM(html, {
  runScripts: 'dangerously',
  pretendToBeVisual: true,          // requestAnimationFrame 제공
  beforeParse: installShims,        // 페이지 스크립트가 돌기 **전에** 채워야 한다
  virtualConsole: new (require(path.join(__dirname, '..', '..',
    'node_modules', 'jsdom')).VirtualConsole)()
    .on('jsdomError', e => errors.push(String(e && e.message || e)))
    .on('error', (...a) => errors.push('console.error: ' + a.join(' '))),
});

const win = dom.window;
const doc = win.document;
win.addEventListener('error', e => errors.push('window.onerror: ' + e.message));

let result = {};
try {
  const probes = { map: probeMap, story: probeStory, dashboard: probeDashboard };
  if (!probes[kind]) throw new Error('알 수 없는 kind: ' + kind);
  result = probes[kind](doc, win);
} catch (e) {
  errors.push('probe threw: ' + (e && e.stack || e));
}

process.stdout.write(JSON.stringify({ errors, result }, null, 1));
