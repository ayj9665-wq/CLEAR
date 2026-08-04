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
 * 사용: node _domprobe.js <html경로> <kind: map|dashboard>
 */
'use strict';
const fs = require('fs');
const path = require('path');
const { JSDOM } = require(path.join(__dirname, '..', '..', 'node_modules', 'jsdom'));

const [, , file, kind] = process.argv;

// 페이지가 던지는 모든 것을 모은다. 이것이 이 검사의 핵심이고, 나머지는 보강이다.
const errors = [];

function textOf(el) { return el ? (el.textContent || '').trim() : null; }

function probeMap(doc, win) {
  // 한 수준에서의 관측치. 슬라이더를 옮길 때마다 다시 부른다.
  const snap = () => ({
    level: textOf(doc.getElementById('mnv')),
    legendSwatches: doc.querySelectorAll('#leg .sw').length,
    legendSpans: doc.querySelectorAll('#leg > span').length,
    tableRows: doc.querySelectorAll('#t tbody tr').length,
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
      tableRows: doc.querySelectorAll('#t tbody tr').length,
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
  result = kind === 'map' ? probeMap(doc, win) : probeDashboard(doc, win);
} catch (e) {
  errors.push('probe threw: ' + (e && e.stack || e));
}

process.stdout.write(JSON.stringify({ errors, result }, null, 1));
