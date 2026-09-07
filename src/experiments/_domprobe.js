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
 * 사용: node _domprobe.js <html경로> <kind: map|story|story2|dashboard>
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
  /* jsdom에는 레이아웃이 없어 scrollIntoView도 없다. 여기서는 **호출을 기록만**
   * 한다. 어디까지 스크롤됐는지는 레이아웃 문제라 jsdom이 답할 수 없지만, "어느
   * 요소로 스크롤하려 했는가"는 동작이고 그건 단언할 수 있다 -- ver2에서 지도
   * 클릭이 표의 **엉뚱한 행**으로 스크롤하는 결함은 화면상 아무 이상이 없어 보인다.
   * 아무것도 안 하는 셰임을 넣으면 그 결함이 통째로 안 보인다. */
  win.__scrolled = [];
  win.Element.prototype.scrollIntoView = function () { win.__scrolled.push(this); };

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

/* ver2(clear_story_v2.html). v1의 관측에 ver2가 새로 약속한 넷을 더한다:
 * 아키텍처 단계 선택 / 접이식 상세 / 카운티 상세 패널 / 지도<->표<->키보드 연동.
 *
 * 순서가 중요하다. **스크롤 진입 안내를 가장 먼저 본다** -- 설계상 사용자가 무엇이든
 * 조작하면 즉시 중단되므로, 슬라이더 하나만 먼저 건드려도 관측 대상이 사라진다.
 *
 * 페이지의 데이터(S/PL/LV)는 var 전역이라 여기서 그대로 읽을 수 있다. 그래서 패널에
 * 찍힌 문자열을 **그 카운티의 실제 값과 직접 대조**한다 -- '패널이 채워졌다'가 아니라
 * '패널이 맞는 값으로 채워졌다'를 봐야 한다. 그럴듯한 표가 틀린 수를 담는 것이 이
 * 저장소가 반복해 당한 실패다.
 */
function probeStory2(doc, win) {
  const sections = [...doc.querySelectorAll('.sec, #mapsec, footer')];
  const M = (t, o) => new win.MouseEvent(t, Object.assign({ bubbles: true }, o || {}));
  const K = k => new win.KeyboardEvent('keydown', { key: k, bubbles: true });

  // 1) 안내 -- 어떤 조작보다 먼저
  const spotlight = {
    outlines: doc.querySelectorAll('#spt path').length,
    caption: textOf(doc.getElementById('spotcap')),
    skip: !!doc.getElementById('spotskip'),
  };

  // 2) 패널 초기 상태(선택 전)
  const panelHint = doc.querySelectorAll('#panel .phint').length;

  // 3) 아키텍처 단계 선택
  const atabs = [...doc.querySelectorAll('.atab')];
  const arch = { tabs: atabs.length, panes: doc.querySelectorAll('.apane').length,
                 zones: doc.querySelectorAll('.azone').length, picks: [] };
  atabs.forEach((t, i) => {
    t.dispatchEvent(M('click'));
    arch.picks.push({
      clicked: i,
      selected: atabs.map(x => x.getAttribute('aria-selected')).indexOf('true'),
      openPanes: [...doc.querySelectorAll('.apane')].filter(p => !p.hidden).length,
      openPane: [...doc.querySelectorAll('.apane')]
        .map(p => !p.hidden).indexOf(true),
      zoneOn: [...doc.querySelectorAll('.azone')]
        .map(z => z.classList.contains('on')).indexOf(true),
    });
  });

  // 4) 접이식 상세
  const folds = [...doc.querySelectorAll('.foldb')].map(b => {
    const c = doc.getElementById('fold-' + b.dataset.fold);
    const before = c ? c.hidden : null;
    b.dispatchEvent(M('click'));
    const opened = c ? c.hidden : null;
    const expanded = b.getAttribute('aria-expanded');
    b.dispatchEvent(M('click'));
    return { id: b.dataset.fold, found: !!c, before, opened, expanded,
             closed: c ? c.hidden : null };
  });

  // 5) 지도 절 -- v1과 같은 골격이므로 같은 조회를 재사용한다
  const map = probeMap(doc, win, '#ct');

  // 6) 상세 패널과 연동. 슬라이더/레이어를 처음 상태로 되돌린 뒤 본다.
  const slider = doc.getElementById('mn');
  if (slider) { slider.value = '0'; slider.dispatchEvent(new win.Event('input')); }
  const res = doc.querySelector('[data-layer="res"]');
  if (res) res.dispatchEvent(M('click'));
  const tw = doc.getElementById('tablewrap');
  if (tw && tw.hidden) doc.getElementById('tbl').dispatchEvent(M('click'));

  const lv = win.LV[0], sArr = win.PL[lv].s;
  const idx = sArr.findIndex(v => v === 2);          // 유의 판정된 첫 카운티
  const st = idx >= 0 ? win.S.stat[idx] : null;
  const paths = [...doc.querySelectorAll('#g path')];

  const readPanel = () => ({
    name: textOf(doc.querySelector('#panel h4')),
    flag: textOf(doc.querySelector('#panel .pflag')),
    verdict: textOf(doc.querySelector('#panel .pverdict')),
    bars: doc.querySelectorAll('#panel .pbrow').length,
    stats: [...doc.querySelectorAll('#panel dl.pstats dd')].map(d => textOf(d)),
    close: !!doc.getElementById('pclose'),
    toTable: !!doc.getElementById('ptable'),
    outlines: doc.querySelectorAll('#sel path').length,
    hint: doc.querySelectorAll('#panel .phint').length,
  });

  const panel = { hintOnLoad: panelHint, idx,
                  expect: st ? { county: st[1], state: st[0], n: st[2], obs: st[3],
                                 exp: st[4], smr: st[5], z: st[6], black: st[7],
                                 q: win.PL[lv].q[idx] } : null };
  if (idx >= 0) {
    paths[idx].dispatchEvent(M('click'));
    panel.afterMapClick = readPanel();
    panel.rowSelected = !!doc.querySelector('#ct tbody tr.sel');
    const last = win.__scrolled[win.__scrolled.length - 1];
    panel.scrolledToRow = last && last.dataset ? +last.dataset.i : null;
    panel.rowSelectedIsRight =
      (doc.querySelector('#ct tbody tr.sel') || {}).dataset === undefined
        ? false : +doc.querySelector('#ct tbody tr.sel').dataset.i === idx;

    // 지도 호버 -> 표 행 강조
    paths[idx].dispatchEvent(M('pointermove', { clientX: 10, clientY: 10 }));
    const hl = doc.querySelector('#ct tbody tr.hl');
    panel.hoverMapMarksRow = !!hl && +hl.dataset.i === idx;

    // 표 호버 -> 지도 강조
    const other = doc.querySelector('#ct tbody tr:not([data-i="' + idx + '"])');
    if (other) {
      other.dispatchEvent(M('pointerover'));
      panel.hoverRowMarksMap = doc.querySelectorAll('#hl path').length;
    }

    // 표 행 클릭 -> 패널
    const row = doc.querySelector('#ct tbody tr');
    if (row) {
      row.dispatchEvent(M('click'));
      panel.afterRowClick = readPanel();
      panel.rowClickIdx = +row.dataset.i;
    }

    // 키보드: roving tabindex + Enter
    const rows = [...doc.querySelectorAll('#ct tbody tr')];
    panel.firstTabIndex = rows[0] ? rows[0].tabIndex : null;
    rows[0].focus();
    rows[0].dispatchEvent(K('ArrowDown'));
    panel.afterArrowDown = { first: rows[0].tabIndex, second: rows[1].tabIndex,
                             focused: doc.activeElement === rows[1] };
    rows[1].dispatchEvent(K('Enter'));
    panel.afterEnter = readPanel();
    panel.enterIdx = +rows[1].dataset.i;

    // 닫기(Escape)
    doc.dispatchEvent(K('Escape'));
    panel.afterEscape = readPanel();
  }

  return {
    jsClass: doc.documentElement.classList.contains('js'),
    sections: sections.length,
    revealed: sections.filter(s => s.classList.contains('in')).length,
    counters: [...doc.querySelectorAll('[data-count]')].map(b => ({
      want: String(b.dataset.count || ''),
      text: (b.textContent || '').trim(),
    })),
    hero: !!doc.getElementById('hero2'),
    spotlight, arch, folds, panel, map,
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
  const probes = { map: probeMap, story: probeStory, story2: probeStory2,
                   dashboard: probeDashboard };
  if (!probes[kind]) throw new Error('알 수 없는 kind: ' + kind);
  result = probes[kind](doc, win);
} catch (e) {
  errors.push('probe threw: ' + (e && e.stack || e));
}

process.stdout.write(JSON.stringify({ errors, result }, null, 1));
