// Page behaviour: theme toggle, section highlighting in the nav, the figure lightbox, the BibTeX copy button, and the two
// data-driven tables (Table 1 main results, Table 2 compute) rendered from results-data.js. Inlined by build.py.
(function () {
  'use strict';
  var doc = document;
  function $(id) { return doc.getElementById(id); }
  function h(tag, props) {
    var el = doc.createElement(tag);
    if (props) {
      Object.keys(props).forEach(function (k) {
        var v = props[k];
        if (v === null || v === undefined || v === false) { return; }
        if (k === 'class') { el.className = v; }
        else if (k === 'html') { el.innerHTML = v; }
        else if (k === 'dataset') { Object.keys(v).forEach(function (d) { el.dataset[d] = v[d]; }); }
        else if (k === 'hidden') { el.hidden = !!v; }
        else { el.setAttribute(k, v === true ? '' : v); }
      });
    }
    for (var i = 2; i < arguments.length; i++) { append(el, arguments[i]); }
    return el;
  }
  function append(el, c) {
    if (c === null || c === undefined || c === false) { return; }
    if (Array.isArray(c)) { c.forEach(function (x) { append(el, x); }); return; }
    el.appendChild(typeof c === 'string' || typeof c === 'number' ? doc.createTextNode(String(c)) : c);
  }

  /* ------------------------------------------------------------ theme */
  var root = doc.documentElement;
  var themeBtn = $('theme-btn');
  function systemDark() { return window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches; }
  function currentDark() { var t = root.getAttribute('data-theme'); return t ? t === 'dark' : systemDark(); }
  try { var saved = localStorage.getItem('lb-theme'); if (saved === 'dark' || saved === 'light') { root.setAttribute('data-theme', saved); } } catch (e) { /* storage unavailable */ }
  if (themeBtn) {
    themeBtn.addEventListener('click', function () {
      var next = currentDark() ? 'light' : 'dark';
      root.setAttribute('data-theme', next);
      try { localStorage.setItem('lb-theme', next); } catch (e) { /* ignore */ }
    });
  }

  /* ------------------------------------------------------------ nav: current section */
  var links = Array.prototype.slice.call(doc.querySelectorAll('.nav__links a'));
  var targets = links.map(function (a) { return $(a.getAttribute('href').slice(1)); }).filter(Boolean);
  function markCurrent() {
    var y = window.pageYOffset + 120, cur = null;
    targets.forEach(function (t) { if (t.offsetTop <= y) { cur = t.id; } });
    links.forEach(function (a) { var on = a.getAttribute('href') === '#' + cur; if (on) { a.setAttribute('aria-current', 'true'); } else { a.removeAttribute('aria-current'); } });
  }
  window.addEventListener('scroll', markCurrent, { passive: true }); markCurrent();

  /* ------------------------------------------------------------ lightbox */
  var dlg = $('lightbox');
  if (dlg && typeof dlg.showModal === 'function') {
    var lbImg = $('lightbox-img'), lbTitle = $('lightbox-title'), lbCap = $('lightbox-caption');
    function openFig(id) {
      var fig = $(id); if (!fig) { return; }
      var img = fig.querySelector('.fig__media img'), label = fig.querySelector('.fig__label'), cap = fig.querySelector('.fig__caption');
      lbImg.src = img.getAttribute('src'); lbImg.alt = img.alt;
      lbTitle.innerHTML = label ? label.innerHTML : ''; lbCap.innerHTML = cap ? cap.innerHTML : '';
      dlg.showModal();
    }
    doc.addEventListener('click', function (e) {
      var t = e.target.closest ? e.target.closest('[data-zoom]') : null;
      if (t) { e.preventDefault(); openFig(t.dataset.zoom); }
    });
    $('lightbox-close').addEventListener('click', function () { dlg.close(); });
    dlg.addEventListener('click', function (e) { if (e.target === dlg) { dlg.close(); } });
    dlg.addEventListener('close', function () { lbImg.src = ''; });
  }

  /* ------------------------------------------------------------ bibtex */
  var copy = $('copybib'), bib = $('bib');
  if (copy && bib) {
    copy.addEventListener('click', function () {
      function done() { copy.textContent = 'Copied'; setTimeout(function () { copy.textContent = 'Copy BibTeX'; }, 1500); }
      function select() { var r = doc.createRange(); r.selectNodeContents(bib); var s = window.getSelection(); s.removeAllRanges(); s.addRange(r); copy.textContent = 'Select and copy'; }
      if (navigator.clipboard && navigator.clipboard.writeText) { navigator.clipboard.writeText(bib.textContent).then(done).catch(select); } else { select(); }
    });
  }

  /* ------------------------------------------------------------ data tables */
  var D = window.LOOPBACK_DATA;
  if (!D) { return; }
  var MODEL = {}; D.models.forEach(function (m) { MODEL[m.key] = m; });
  var MINUS = '−';
  function fmt(v, d) {
    if (typeof v !== 'number') { return String(v); }
    var parts = v.toFixed(d).split('.');
    parts[0] = parts[0].replace(/\B(?=(\d{3})+(?!\d))/g, ',');
    return parts.join('.');
  }
  function signed(d, decimals) {
    var r = Number(d.toFixed(decimals));
    return { text: (r > 0 ? '+' : r < 0 ? MINUS : '±') + Math.abs(r).toFixed(decimals), kind: r > 0 ? 'pos' : r < 0 ? 'neg' : 'zero' };
  }

  /* data check: the printed means must agree with the per-task numbers (rounding of the per-task values allows ~0.01) */
  function checkMeans() {
    var bad = [];
    D.scales.forEach(function (s) {
      D.groups.forEach(function (g) { g.metrics.forEach(function (m) {
        if (!m.meanOf) { return; }
        s.models.forEach(function (key, i) {
          var sum = 0; m.meanOf.forEach(function (t) { sum += s.values[t][i]; });
          var mean = sum / m.meanOf.length, printed = s.values[m.key][i];
          if (Math.abs(mean - printed) > 0.015) { bad.push(s.key + ' ' + key + ' ' + m.key + ': ' + mean.toFixed(3) + ' vs ' + printed); }
        });
      }); });
    });
    console.assert(bad.length === 0, '[results] printed means disagree with per-task values:', bad);
    return bad;
  }
  window.LOOPBACK_CHECK = checkMeans();

  function segControl(items, onPick, label) {
    var btns = [];
    var seg = h('div', { 'class': 'seg', role: 'radiogroup', 'aria-label': label });
    items.forEach(function (it, i) {
      var b = h('button', { type: 'button', 'class': 'seg__btn', role: 'radio', 'aria-checked': i === 0 ? 'true' : 'false', tabindex: i === 0 ? '0' : '-1', dataset: { key: it.key } }, it.label);
      b.addEventListener('click', function () { pick(it.key); });
      b.addEventListener('keydown', function (e) {
        var j = i;
        if (e.key === 'ArrowRight' || e.key === 'ArrowDown') { j = (i + 1) % items.length; }
        else if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') { j = (i + items.length - 1) % items.length; }
        else if (e.key === 'Home') { j = 0; } else if (e.key === 'End') { j = items.length - 1; } else { return; }
        e.preventDefault(); pick(items[j].key); btns[j].focus();
      });
      btns.push(b); seg.appendChild(b);
    });
    function pick(key) {
      btns.forEach(function (b) { var on = b.dataset.key === key; b.setAttribute('aria-checked', on ? 'true' : 'false'); b.setAttribute('tabindex', on ? '0' : '-1'); });
      onPick(key);
    }
    return seg;
  }
  function switchControl(text, onToggle) {
    var on = false;
    var b = h('button', { type: 'button', 'class': 'rt-switch', role: 'switch', 'aria-checked': 'false' },
      h('span', { 'class': 'rt-switch__track', 'aria-hidden': 'true' }, h('span', { 'class': 'rt-switch__thumb' })), h('span', null, text));
    b.addEventListener('click', function () { on = !on; b.setAttribute('aria-checked', on ? 'true' : 'false'); onToggle(on); });
    return b;
  }
  function scrollHint(scroller, hint) {
    function sync() { hint.classList.toggle('is-on', scroller.scrollWidth - scroller.clientWidth > 1 && scroller.scrollLeft < scroller.scrollWidth - scroller.clientWidth - 1); }
    scroller.addEventListener('scroll', sync, { passive: true });
    if ('ResizeObserver' in window) { new ResizeObserver(sync).observe(scroller); } else { window.addEventListener('resize', sync); }
    sync();
  }

  /* ---------- Table 1: main results */
  var mount = $('results-table');
  if (mount) {
    var state = { scale: '374M', delta: false };
    var scroller = h('div', { 'class': 'rt-scroll', role: 'region', 'aria-label': 'Table 1, main results' });
    var hint = h('p', { 'class': 'rt-hint', 'aria-hidden': 'true' }, 'Scroll sideways for all models →');
    var title = h('span', { 'class': 'rt-caption' });
    var seg = segControl(D.scales.map(function (s) { return { key: s.key, label: s.label }; }), function (k) { state.scale = k; render(); }, 'Model scale');
    seg.querySelectorAll('.seg__btn').forEach(function (b) { var on = b.dataset.key === state.scale; b.setAttribute('aria-checked', on ? 'true' : 'false'); b.setAttribute('tabindex', on ? '0' : '-1'); });
    var sw = switchControl('Δ vs Standard', function (on) { state.delta = on; render(); });
    var card = h('div', { 'class': 'rt-card' },
      h('div', { 'class': 'rt-head' },
        h('p', { 'class': 'rt-label' }, h('b', null, 'Table 1'), ' · Main results'),
        h('div', { 'class': 'rt-controls' }, h('span', { 'class': 'rt-controls__name', 'aria-hidden': 'true' }, 'Scale'), seg, sw)),
      hint, scroller,
      h('div', { 'class': 'rt-foot' },
        h('ul', { 'class': 'legend', 'aria-label': 'Table key' },
          h('li', null, h('span', { 'class': 'key key--best' }, '1.23'), 'best in the row at this scale (lowest for loss and perplexity)'),
          h('li', null, h('span', { 'class': 'key key--ours', 'aria-hidden': 'true' }), 'shaded: Loopback (ours)'),
          h('li', null, h('span', { 'class': 'key' }, 'Δ'), 'Loopback minus Standard at the same scale, computed from the table; green is better')),
        title,
        h('p', { 'class': 'rt-notes' }, 'Validation loss is next-token cross-entropy on a held-out ClimbMix shard packed as in training. Accuracies are zero-shot, and every prompt starts with the beginning-of-sequence token, as in training; HellaSwag, OpenBookQA and both ARC sets use length-normalized accuracy. The mean covers all thirteen tasks; WikiText perplexity is not in it.')));
    mount.innerHTML = ''; mount.appendChild(card);
    scrollHint(scroller, hint);

    function render() {
      var s = null; D.scales.forEach(function (x) { if (x.key === state.scale) { s = x; } });
      var keys = s.models, dense = keys.indexOf('dense');
      var table = h('table', { 'class': 'rt-table' });
      table.appendChild(h('caption', { 'class': 'sr-only' }, 'Table 1: main results at ' + s.key + '. Benchmarks in rows, models in columns.'));
      // family groups in column order
      var runs = []; keys.forEach(function (k) { var f = MODEL[k].family; if (!runs.length || runs[runs.length - 1].family !== f) { runs.push({ family: f, n: 0 }); } runs[runs.length - 1].n++; });
      var trG = h('tr'); trG.appendChild(h('th', { scope: 'col', rowspan: '2', 'class': 'rt-corner' }, 'Benchmark'));
      runs.forEach(function (r) { trG.appendChild(h('th', { scope: 'colgroup', colspan: String(r.n), 'class': 'rt-grp rt-grp--' + r.family }, h('span', null, D.families[r.family]))); });
      var trM = h('tr');
      keys.forEach(function (k) {
        var m = MODEL[k];
        var sub = k === 'dense' ? s.blocks + ' blocks' : m.family === 'ours' ? s.topology : null;
        trM.appendChild(h('th', { scope: 'col', 'class': 'rt-model' + (m.family === 'ours' ? ' rt-ours' : '') }, m.name, sub ? h('small', null, sub) : null));
      });
      var thead = h('thead'); thead.appendChild(trG); thead.appendChild(trM); table.appendChild(thead);

      D.groups.forEach(function (g) {
        var tb = h('tbody');
        if (g.name !== 'Setting') { tb.appendChild(h('tr', { 'class': 'rt-sec' }, h('th', { scope: 'rowgroup', colspan: String(keys.length + 1) }, g.name))); }
        g.metrics.forEach(function (met) {
          var vals = s.values[met.key];
          var best = null;
          if (met.mark !== false) { best = met.lower ? Math.min.apply(null, vals) : Math.max.apply(null, vals); }
          var tr = h('tr', { 'class': met.meanOf ? 'rt-row-mean' : null });
          tr.appendChild(h('th', { scope: 'row', 'class': 'rt-bench' }, met.strong ? h('strong', null, met.label) : met.label, met.unit ? h('small', null, met.unit) : null));
          keys.forEach(function (k, i) {
            var m = MODEL[k], v = vals[i], isBest = best !== null && v === best;
            var td = h('td', { 'class': 'rt-num' + (m.family === 'ours' ? ' rt-ours' : '') },
              h('span', { 'class': 'rt-v' + (isBest ? ' is-best' : '') }, fmt(v, met.fmt)),
              isBest ? h('span', { 'class': 'sr-only' }, ' (best in row)') : null);
            if (state.delta && m.family === 'ours' && dense >= 0 && met.key !== 'params') {
              var d = signed(v - vals[dense], met.fmt);
              var better = met.lower ? d.kind === 'neg' : d.kind === 'pos';
              td.appendChild(h('span', { 'class': 'rt-delta', dataset: { sign: d.kind === 'zero' ? 'zero' : better ? 'pos' : 'neg' } }, h('span', { 'class': 'sr-only' }, ', difference from Standard: '), d.text));
            }
            tr.appendChild(td);
          });
          tb.appendChild(tr);
        });
        table.appendChild(tb);
      });
      scroller.innerHTML = ''; scroller.appendChild(table); scroller.scrollLeft = 0;
      title.innerHTML = '<strong>' + s.title + '.</strong> ' + (keys.length > 2
        ? 'The standard Transformer and eight residual rules; all looped arms share the topology, the token budget and the learning rate, and the standard Transformer keeps the same number of block applications.'
        : 'The standard Transformer and Loopback at the same executed depth and token budget.');
      hint.classList.remove('is-on'); setTimeout(function () { scroller.dispatchEvent(new Event('scroll')); }, 0);
    }
    render();
  }

  /* ---------- Table 2: compute */
  var cm = $('compute-table');
  if (cm && D.compute) {
    var C = D.compute, cstate = { scale: '374M' };
    var cscroller = h('div', { 'class': 'rt-scroll', role: 'region', 'aria-label': 'Table 2, compute and efficiency' });
    var chint = h('p', { 'class': 'rt-hint', 'aria-hidden': 'true' }, 'Scroll sideways for all columns →');
    var ctitle = h('span', { 'class': 'rt-caption' });
    var cseg = segControl(C.scales.map(function (s) { return { key: s.key, label: s.key }; }), function (k) { cstate.scale = k; crender(); }, 'Model scale');
    cseg.querySelectorAll('.seg__btn').forEach(function (b) { var on = b.dataset.key === cstate.scale; b.setAttribute('aria-checked', on ? 'true' : 'false'); b.setAttribute('tabindex', on ? '0' : '-1'); });
    var ccard = h('div', { 'class': 'rt-card' },
      h('div', { 'class': 'rt-head' },
        h('p', { 'class': 'rt-label' }, h('b', null, 'Table 2'), ' · Compute and efficiency'),
        h('div', { 'class': 'rt-controls' }, h('span', { 'class': 'rt-controls__name', 'aria-hidden': 'true' }, 'Scale'), cseg)),
      chint, cscroller,
      h('div', { 'class': 'rt-foot' }, ctitle,
        h('p', { 'class': 'rt-notes' }, 'Parameters are Transformer-only; extra forward FLOPs are relative to the standard Transformer. Training throughput and peak memory are per GPU on 8 H100s; hours are estimated from throughput. Prefill uses one H100. The underline marks the best looped rule in the column.')));
    cm.innerHTML = ''; cm.appendChild(ccard);
    scrollHint(cscroller, chint);
    function crender() {
      var s = null; C.scales.forEach(function (x) { if (x.key === cstate.scale) { s = x; } });
      var table = h('table', { 'class': 'rt-table' });
      table.appendChild(h('caption', { 'class': 'sr-only' }, 'Table 2: compute and efficiency at ' + s.key + '. Models in rows.'));
      var tr = h('tr'); tr.appendChild(h('th', { scope: 'col', 'class': 'rt-corner' }, 'Model'));
      C.columns.forEach(function (c) { tr.appendChild(h('th', { scope: 'col', 'class': 'rt-model' }, c.label)); });
      table.appendChild(h('thead', null, tr));
      var tb = h('tbody');
      // best among the looped rules per numeric column (the standard Transformer is the reference, not a competitor)
      var bestOf = {};
      C.columns.forEach(function (c, ci) {
        if (!c.lower && !c.higher) { return; }
        var vals = s.rows.filter(function (r) { return r[0] !== 'dense'; }).map(function (r) { var v = r[ci + 1]; return typeof v === 'number' ? v : parseFloat(String(v).replace(/[^\d.]/g, '')); });
        bestOf[c.key] = c.lower ? Math.min.apply(null, vals) : Math.max.apply(null, vals);
      });
      s.rows.forEach(function (r) {
        var m = MODEL[r[0]], ours = m.family === 'ours';
        var row = h('tr', { 'class': ours ? 'rt-row-mean' : null });
        row.appendChild(h('th', { scope: 'row', 'class': 'rt-bench' }, ours ? h('strong', null, m.name) : m.name, m.note ? h('small', null, m.note) : null));
        C.columns.forEach(function (c, ci) {
          var v = r[ci + 1], num = typeof v === 'number' ? v : parseFloat(String(v).replace(/[^\d.]/g, ''));
          var isBest = r[0] !== 'dense' && bestOf[c.key] !== undefined && num === bestOf[c.key];
          row.appendChild(h('td', { 'class': 'rt-num' + (ours ? ' rt-ours' : '') }, h('span', { 'class': 'rt-v' + (isBest ? ' is-best' : '') }, typeof v === 'number' ? fmt(v, c.fmt) : v)));
        });
        tb.appendChild(row);
      });
      table.appendChild(tb);
      cscroller.innerHTML = ''; cscroller.appendChild(table); cscroller.scrollLeft = 0;
      ctitle.innerHTML = '<strong>' + s.title + '.</strong> Loopback adds ' + (s.key === '778M' ? '0.04%' : '0.08%') + ' to the standard Transformer’s forward FLOPs and pays for the read in memory traffic: ' + (s.key === '191M' ? '25%' : s.key === '374M' ? '23%' : '16%') + ' fewer training tokens per second.';
      chint.classList.remove('is-on'); setTimeout(function () { cscroller.dispatchEvent(new Event('scroll')); }, 0);
    }
    crender();
  }
})();
