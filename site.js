// The interactive Loopback demo (the loop stage, the three-rule ledger and the chart). Page chrome (theme, nav, lightbox,
// BibTeX copy, data tables) lives in page.js; both are inlined by build.py.
// ---------------------------------------------------------------------------------------------------------------
// Interactive demo: one token's hidden state circles the shared blocks; each application appends an update to the
// history and the reader forms the next state from the input (coefficient one) and the stored updates (prior set
// by visit-end decay, allocation by attention). Colours follow the paper's figures: blue / peach shared blocks,
// green reader, grey prefix, coda and input, rose shades for coefficient magnitude (lighter = smaller).
// ---------------------------------------------------------------------------------------------------------------
(function () {
  'use strict';
  var NS = 'http://www.w3.org/2000/svg';
  var C = { ink: '#202020', muted: '#6B6B6B', faint: '#9A9A9A', grid: '#E2E2E2', border: '#4A5560', ghost: '#C4C9D0',
            blueF: '#D3E4F1', blueS: '#6F93B4', peachF: '#F1D9CD', peachS: '#C98F74', greenF: '#DFEBD3', greenS: '#6F9A5E',
            grayF: '#EAEAEA', navy: '#0F4D92', dense: '#767676', rose: '#B64342', ramp: ['#F6CFCB', '#E9A6A1', '#CF7F7A', '#B64342'] };
  var SERIF = "'Times New Roman', Times, serif";
  var reduce = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  function $(id) { return document.getElementById(id); }
  function el(n, a, p) { var e = document.createElementNS(NS, n); for (var k in a) if (a[k] !== undefined && a[k] !== null) e.setAttribute(k, a[k]); if (p) p.appendChild(e); return e; }
  function txt(p, x, y, s, a) { var o = { x: x, y: y, 'font-family': SERIF, 'font-size': 13, fill: C.ink }; for (var k in (a || {})) o[k] = a[k]; var t = el('text', o, p); t.textContent = s; return t; }
  function halo(t) { t.setAttribute('paint-order', 'stroke'); t.setAttribute('stroke', '#fff'); t.setAttribute('stroke-width', 3.5); t.setAttribute('stroke-linejoin', 'round'); return t; }
  // box of a text element in its own user space (0s when the svg is not rendered, so callers simply skip the backing)
  function bbox(t) { try { var b = t.getBBox(); return { x: b.x, y: b.y, w: b.width, h: b.height }; } catch (e) { return { x: 0, y: 0, w: 0, h: 0 }; } }
  // halo in the colour of whatever lies behind a label (wide enough to close the gaps between digits), so lines
  // passing under the label stop short of its glyphs
  function rim(t, color) { t.setAttribute('paint-order', 'stroke'); t.setAttribute('stroke', color); t.setAttribute('stroke-width', 3.6); t.setAttribute('stroke-linejoin', 'round'); return t; }
  // a card label against the liquid level whose top edge is at y (glyphs span t0..t1 vertically): above the level it is
  // ink on the card colour, on the level it is white when the level is deep (w > 0.45, as before) and ink otherwise.
  // Each part has a thin halo in its own background, so the dashed prior outline stops short of the glyphs and no halo
  // shows as a rim; a label the level's edge runs through is drawn twice, each copy clipped to its side of the edge.
  function levelLabel(p, id, make, t0, t1, y, w, k, x, size) {
    function part(fill, halo, clip) { var t = rim(make(fill), halo); if (clip) t.setAttribute('clip-path', 'url(#' + clip + ')'); return t; }
    var pad = 2, onFill = w > 0.45 ? '#fff' : C.ink;
    if (y >= t1 + pad) return part(C.ink, blockFill(k));
    if (y <= t0 - pad) return part(onFill, rampColor(w));
    var a = 'lb-cut-' + id + 'a', b = 'lb-cut-' + id + 'b';
    el('rect', { x: x, y: t0 - 8, width: size, height: y - t0 + 8 }, el('clipPath', { id: a }, p));
    el('rect', { x: x, y: y, width: size, height: t1 + 8 - y }, el('clipPath', { id: b }, p));
    part(C.ink, blockFill(k), a); part(onFill, rampColor(w), b);
  }
  // italic symbol with an optional subscript, e.g. δ7, h0, f3
  function sym(p, x, y, base, sub, a) {
    var t = txt(p, x, y, '', a);
    var b = el('tspan', { 'font-style': 'italic' }, t); b.textContent = base;
    if (sub !== undefined && sub !== null && sub !== '') { var s = el('tspan', { dy: '0.32em', 'font-size': '72%' }, t); s.textContent = sub; }
    return t;
  }
  function hex(c) { return [parseInt(c.slice(1, 3), 16), parseInt(c.slice(3, 5), 16), parseInt(c.slice(5, 7), 16)]; }
  function mix(a, b, t) { var A = hex(a), B = hex(b); return 'rgb(' + A.map(function (v, i) { return Math.round(v + (B[i] - v) * t); }).join(',') + ')'; }
  function rampColor(v) { var t = Math.max(0, Math.min(1, Math.sqrt(Math.max(0, v)))) * 3, i = Math.min(2, Math.floor(t)); return mix(C.ramp[i], C.ramp[i + 1], t - i); }
  function blockFill(k) { return k % 2 ? C.peachF : C.blueF; }
  function blockStroke(k) { return k % 2 ? C.peachS : C.blueS; }
  function fmt(v, d) { return v.toFixed(d === undefined ? 3 : d); }
  var tip = $('tip');
  function hover(node, label) {
    node.addEventListener('mousemove', function (e) { tip.textContent = typeof label === 'function' ? label() : label; tip.style.left = (e.clientX + 12) + 'px'; tip.style.top = (e.clientY - 30) + 'px'; tip.style.opacity = 1; });
    node.addEventListener('mouseleave', function () { tip.style.opacity = 0; });
  }

  // ---------- model ----------
  var P = { K: 4, R: 4, lam: 0.25, focus: 0, gain: 2, speed: 1, s: 0 };
  function N() { return P.K * P.R; }
  function bound() { return P.lam < 1 ? P.K * P.lam / (1 - P.lam) : Infinity; }
  // read after application s: decay prior (preDecay = the appended 1 before the visit-end decay) and the attention-reallocated coefficients
  function coeffs(s, preDecay) {
    var K = P.K, n = s + 1, ends = preDecay ? Math.floor(s / K) : Math.floor((s + 1) / K), prior = [], z = [], i;
    for (i = 0; i < n; i++) { prior.push(Math.pow(P.lam, ends - Math.floor(i / K))); z.push(P.focus > 0 && Math.floor(i / K) === P.focus - 1 ? P.gain : 0); }
    var B = prior.reduce(function (a, b) { return a + b; }, 0);
    var logits = z.map(function (v, i) { return v + Math.log(prior[i]); }), m = Math.max.apply(null, logits);
    var ex = logits.map(function (v) { return Math.exp(v - m); }), Z = ex.reduce(function (a, b) { return a + b; }, 0);
    return { prior: prior, post: ex.map(function (v) { return B * v / Z; }), B: B };
  }

  // ---------- layout ----------
  var L = {};
  function layout() {
    var K = P.K, R = P.R;
    L.ring = { cx: 176, cy: 238, r: 110 };
    L.block = { w: 60, h: 30 };
    L.reader = { cx: 404, cy: 238, r: 24 };
    // the visit labels sit to the right of the grid, outside the fan of read threads; the grid region stops short of
    // the right edge to leave them room (K = 8 still gets 45-unit cards, so every card keeps its three-decimal value)
    var gx0 = 544, gx1 = 956, gy0 = 74, gy1 = 404;
    var cell = Math.floor(Math.min(62, (gx1 - gx0) / K, (gy1 - gy0) / R)), gap = Math.max(5, Math.round(cell * 0.12)), card = cell - gap;
    var gw = K * cell - gap, gh = R * cell - gap;
    L.grid = { x0: gx0 + Math.floor((gx1 - gx0 - gw) / 2), y0: gy0 + Math.floor((gy1 - gy0 - gh) / 2), cell: cell, card: card, gap: gap, w: gw, h: gh };
    L.h0 = { x: 458, y: L.reader.cy - card / 2, card: card };
    L.blockPos = [];
    for (var k = 0; k < K; k++) { var a = -Math.PI / 2 + 2 * Math.PI * k / K; L.blockPos.push({ x: L.ring.cx + L.ring.r * Math.cos(a), y: L.ring.cy + L.ring.r * Math.sin(a), a: a }); }
  }
  function cellPos(i) { var k = i % P.K, v = Math.floor(i / P.K); return { x: L.grid.x0 + k * L.grid.cell, y: L.grid.y0 + v * L.grid.cell }; }
  function ringPoint(a) { return { x: L.ring.cx + L.ring.r * Math.cos(a), y: L.ring.cy + L.ring.r * Math.sin(a) }; }

  // ---------- stage ----------
  var stage = $('lb-stage'), G = {};
  function buildStage() {
    stage.innerHTML = '';
    var defs = el('defs', {}, stage);
    function marker(id, color) { var m = el('marker', { id: id, viewBox: '0 0 10 10', refX: 9, refY: 5, markerWidth: 7, markerHeight: 7, orient: 'auto-start-reverse' }, defs); el('path', { d: 'M0 0 L10 5 L0 10 z', fill: color }, m); }
    marker('lb-arrow-ink', C.ink); marker('lb-arrow-navy', C.navy); marker('lb-arrow-faint', C.ghost);
    var glow = el('filter', { id: 'lb-glow', x: '-40%', y: '-40%', width: '180%', height: '180%' }, defs);
    el('feGaussianBlur', { stdDeviation: 3, result: 'b' }, glow);
    var mg = el('feMerge', {}, glow); el('feMergeNode', { 'in': 'b' }, mg); el('feMergeNode', { 'in': 'SourceGraphic' }, mg);
    ['caption', 'ringStatic', 'threads', 'grid', 'readerG', 'dot', 'fly', 'badge', 'labels'].forEach(function (n) { G[n] = el('g', { 'data-layer': n }, stage); });
  }

  function drawRing() {
    var g = G.ringStatic; g.innerHTML = '';
    var K = P.K, rg = L.ring, bw = L.block.w, bh = L.block.h;
    // loop ring with direction arrows between the blocks
    el('circle', { cx: rg.cx, cy: rg.cy, r: rg.r, fill: 'none', stroke: '#CFD4DA', 'stroke-width': 2.2 }, g);
    for (var k = 0; k < K; k++) {
      var a0 = L.blockPos[k].a, a1 = a0 + 2 * Math.PI / K, am = (a0 + a1) / 2;
      var p0 = ringPoint(am - 0.02), p1 = ringPoint(am + 0.02);
      el('path', { d: 'M' + p0.x + ' ' + p0.y + ' L' + p1.x + ' ' + p1.y, stroke: '#CFD4DA', 'stroke-width': 2.2, 'marker-end': 'url(#lb-arrow-faint)' }, g);
    }
    // prefix in, coda out (through the top of the ring)
    var top = ringPoint(-Math.PI / 2);
    el('rect', { x: 14, y: top.y - 15, width: 46, height: 30, rx: 4, fill: C.grayF, stroke: C.border, 'stroke-width': 1 }, g);
    txt(g, 37, top.y + 4, 'P', { 'text-anchor': 'middle', 'font-size': 14 });
    txt(g, 37, top.y + 30, 'prefix', { 'text-anchor': 'middle', 'font-size': 11, fill: C.muted });
    el('path', { d: 'M60 ' + top.y + ' L' + (top.x - bw / 2 - 2) + ' ' + top.y, stroke: C.ink, 'stroke-width': 1.3, 'marker-end': 'url(#lb-arrow-ink)' }, g);
    G.exit = el('path', { d: 'M' + (top.x + bw / 2 + 2) + ' ' + top.y + ' L292 ' + top.y, stroke: C.ghost, 'stroke-width': 1.3, 'marker-end': 'url(#lb-arrow-faint)' }, g);
    el('rect', { x: 294, y: top.y - 15, width: 46, height: 30, rx: 4, fill: C.grayF, stroke: C.border, 'stroke-width': 1 }, g);
    txt(g, 317, top.y + 4, 'C', { 'text-anchor': 'middle', 'font-size': 14 });
    txt(g, 317, top.y + 30, 'coda', { 'text-anchor': 'middle', 'font-size': 11, fill: C.muted });
    // the shared blocks
    G.blocks = [];
    for (k = 0; k < K; k++) {
      var p = L.blockPos[k], bg = el('g', { transform: 'translate(' + p.x + ' ' + p.y + ')' }, g);
      var rect = el('rect', { x: -bw / 2, y: -bh / 2, width: bw, height: bh, rx: 5, fill: blockFill(k), stroke: C.border, 'stroke-width': 1 }, bg);
      sym(bg, 0, 5, 'f', String(k), { 'text-anchor': 'middle', 'font-size': 15 });
      G.blocks.push({ g: bg, rect: rect, x: p.x, y: p.y });
      hover(rect, 'shared block f' + k + ': applied once per visit; its update δ is appended to the history');
    }
    txt(g, rg.cx, rg.cy - 8, 'shared blocks', { 'text-anchor': 'middle', 'font-size': 12, fill: C.muted });
    txt(g, rg.cx, rg.cy + 10, 'K = ' + K + ', visited R = ' + P.R + ' times', { 'text-anchor': 'middle', 'font-size': 12, fill: C.muted });
    // reader
    var rd = G.readerG; rd.innerHTML = '';
    G.readerCircle = el('circle', { cx: L.reader.cx, cy: L.reader.cy, r: L.reader.r, fill: C.greenF, stroke: C.greenS, 'stroke-width': 1.2 }, rd);
    G.readerLabel = sym(rd, L.reader.cx, L.reader.cy + 5, 'R', '', { 'text-anchor': 'middle', 'font-size': 16 });
    halo(txt(rd, L.reader.cx, L.reader.cy + L.reader.r + 16, 'reader', { 'text-anchor': 'middle', 'font-size': 11, fill: C.muted }));
    hover(G.readerCircle, function () { return 'read after application ' + P.s + ': next state = input + Σ c·δ over ' + (P.s + 1) + ' stored update' + (P.s ? 's' : ''); });
    // reader output back into the loop
    var xEnd = rg.cx + rg.r + 4;
    for (k = 0; k < K; k++) if (Math.abs(L.blockPos[k].y - rg.cy) < bh) xEnd = Math.max(xEnd, L.blockPos[k].x + bw / 2 + 4);
    el('path', { d: 'M' + (L.reader.cx - L.reader.r - 3) + ' ' + rg.cy + ' L' + (xEnd + 2) + ' ' + rg.cy, stroke: C.navy, 'stroke-width': 1.6, 'marker-end': 'url(#lb-arrow-navy)' }, rd);
    var hx = (L.reader.cx - L.reader.r + xEnd) / 2;
    G.nextLabel = sym(rd, hx, rg.cy - 8, 'h', '', { 'text-anchor': 'middle', 'font-size': 13, fill: C.navy });
    // input card h0
    var h = L.h0, hg = el('g', {}, rd);
    el('rect', { x: h.x, y: h.y, width: h.card, height: h.card, rx: 5, fill: C.grayF, stroke: C.border, 'stroke-width': 1 }, hg);
    el('rect', { x: h.x + 5, y: h.y + h.card * 0.42, width: h.card - 10, height: h.card * 0.58 - 5, rx: 2, fill: C.ramp[3] }, hg);
    sym(hg, h.x + 6, h.y + 15, 'h', '0', { 'font-size': 13 });
    // with many visits the lower read threads leave h0 right under its caption: give the caption a paper backing
    var inLab = halo(txt(hg, h.x + h.card / 2, h.y + h.card + 14, 'input · 1', { 'text-anchor': 'middle', 'font-size': 11, fill: C.muted })), ib = bbox(inLab);
    if (ib.w) hg.insertBefore(el('rect', { x: ib.x - 3, y: ib.y + 0.5, width: ib.w + 6, height: ib.h, rx: 3, fill: '#fff' }), inLab);
    hover(hg, 'the input h0 enters every read with coefficient 1: it is never decayed and never shares a softmax with the updates');
    // history grid frame: column headers, row labels, ghost cells
    var gr = L.grid;
    txt(g, gr.x0 + gr.w / 2, gr.y0 - 26, 'history of updates (one card per application)', { 'text-anchor': 'middle', 'font-size': 12, fill: C.muted });
    for (k = 0; k < K; k++) sym(g, gr.x0 + k * gr.cell + gr.card / 2, gr.y0 - 9, 'f', String(k), { 'text-anchor': 'middle', 'font-size': 12, fill: blockStroke(k) });
    // row labels on the right of the grid: the threads from the reader converge on the left edge of each row
    G.labels.innerHTML = ''; G.rowLabels = G.labels; for (var v = 0; v < P.R; v++) halo(txt(G.rowLabels, gr.x0 + gr.w + 8, gr.y0 + v * gr.cell + gr.card / 2 + 4, 'visit ' + (v + 1), { 'text-anchor': 'start', 'font-size': 11, fill: C.muted }));
    // visit pips under the ring
    G.pips = [];
    var px = rg.cx - (P.R * 16 - 4) / 2, py = rg.cy + rg.r + 22;
    for (v = 0; v < P.R; v++) G.pips.push(el('rect', { x: px + v * 16, y: py, width: 12, height: 12, rx: 2, fill: '#fff', stroke: C.border, 'stroke-width': 1 }, g));
    G.pipLabel = txt(g, rg.cx, py + 30, '', { 'text-anchor': 'middle', 'font-size': 12, fill: C.muted });
  }

  function setBlockHighlight(k) {
    G.blocks.forEach(function (b, j) { b.rect.setAttribute('stroke', j === k ? C.navy : C.border); b.rect.setAttribute('stroke-width', j === k ? 2.4 : 1); b.rect.setAttribute('filter', j === k ? 'url(#lb-glow)' : null); });
  }
  function exitPoint(e) { var top = ringPoint(-Math.PI / 2); return { x: top.x + L.block.w / 2 + 2 + (292 - top.x - L.block.w / 2 - 8) * e, y: top.y }; }
  function drawDot(angle, exiting) {
    var g = G.dot; g.innerHTML = '';
    var p = exiting ? exiting : ringPoint(angle);
    el('circle', { cx: p.x, cy: p.y, r: 13, fill: C.navy, opacity: 0.18 }, g);
    el('circle', { cx: p.x, cy: p.y, r: 6.5, fill: C.navy, stroke: '#fff', 'stroke-width': 1.5 }, g);
  }

  // history cards, read threads, pips, captions for the coefficient arrays currently shown
  function drawDynamic(prior, post, s) {
    var gr = L.grid, K = P.K, n = N(), i, gg = G.grid, gt = G.threads; gg.innerHTML = ''; gt.innerHTML = '';
    var rx = L.reader.cx + L.reader.r, ry = L.reader.cy;
    // input thread
    el('path', { d: 'M' + L.h0.x + ' ' + L.reader.cy + ' L' + (rx + 2) + ' ' + ry, stroke: C.ink, 'stroke-width': 4, 'stroke-linecap': 'round', opacity: 0.85 }, gt);
    var maxc = Math.max(0.001, Math.max.apply(null, post.length ? post : [0]));
    for (i = 0; i < n; i++) {
      var c = cellPos(i), k = i % K, v = Math.floor(i / K);
      if (i > s) { el('rect', { x: c.x, y: c.y, width: gr.card, height: gr.card, rx: 5, fill: 'none', stroke: C.ghost, 'stroke-width': 1, 'stroke-dasharray': '3 3' }, gg); continue; }
      var w = post[i], pr = prior[i];
      // thread, weight-coded
      var mx = (c.x + rx) / 2;
      el('path', { d: 'M' + (rx + 2) + ' ' + ry + ' C' + mx + ' ' + ry + ' ' + mx + ' ' + (c.y + gr.card / 2) + ' ' + c.x + ' ' + (c.y + gr.card / 2), fill: 'none', stroke: C.navy, 'stroke-width': 0.6 + 4.5 * Math.min(1, w), opacity: 0.12 + 0.55 * Math.min(1, w / maxc), 'stroke-linecap': 'round' }, gt);
      // card
      var cg = el('g', {}, gg);
      el('rect', { x: c.x, y: c.y, width: gr.card, height: gr.card, rx: 5, fill: blockFill(k), stroke: i === s ? C.navy : C.border, 'stroke-width': i === s ? 2.2 : 1 }, cg);
      var inner = gr.card - 10, lvl = Math.min(1, w), hgt = Math.max(0, inner * lvl);
      el('rect', { x: c.x + 5, y: c.y + 5 + inner - hgt, width: inner, height: hgt, rx: 2, fill: rampColor(w) }, cg);
      // prior outline (decay only), visible when attention has moved weight
      if (Math.abs(pr - w) > 0.004) { var ph = Math.max(0, inner * Math.min(1, pr)); el('rect', { x: c.x + 5, y: c.y + 5 + inner - ph, width: inner, height: ph, rx: 2, fill: 'none', stroke: C.rose, 'stroke-width': 1, 'stroke-dasharray': '2 2' }, cg); }
      // card labels sit inside the level's area, clear of the dashed prior outline's edges, and take their colours from
      // whatever lies behind them (levelLabel)
      var barTop = c.y + 5 + inner - hgt;
      if (gr.card >= 34) {
        var fd = gr.card >= 48 ? 12.5 : 11, dy = c.y + 16.5;
        levelLabel(cg, 'd' + i, function (fill) { return sym(cg, c.x + 7.5, dy, 'δ', String(i), { 'font-size': fd, fill: fill }); }, dy - 0.76 * fd, dy + 0.32 * fd, barTop, w, k, c.x, gr.card);
      }
      if (gr.card >= 40) {
        var fn = gr.card >= 50 ? 10.5 : 9.5, ny = c.y + gr.card - 7, val = fmt(w, gr.card >= 44 ? 3 : 2);
        levelLabel(cg, 'n' + i, function (fill) { return txt(cg, c.x + gr.card - 8.5, ny, val, { 'text-anchor': 'end', 'font-size': fn, fill: fill }); }, ny - 0.74 * fn, ny, barTop, w, k, c.x, gr.card);
      }
      (function (i, k, v) { hover(cg, function () { var cc = coeffs(P.s, false); return 'δ' + i + ' from block f' + k + ', visit ' + (v + 1) + ': decay prior ' + fmt(cc.prior[i]) + ', after attention ' + fmt(cc.post[i]); }); })(i, k, v);
    }
    // visit pips
    var cur = Math.floor(s / K);
    G.pips.forEach(function (p, v) { p.setAttribute('fill', v < cur ? C.navy : v === cur ? C.peachF : '#fff'); p.setAttribute('stroke', v <= cur ? C.navy : C.border); });
    G.pipLabel.textContent = 'visit ' + (cur + 1) + ' of ' + P.R;
    // reader subscript and next-state label
    G.readerLabel.innerHTML = ''; var b = el('tspan', { 'font-style': 'italic' }, G.readerLabel); b.textContent = 'R'; var sb = el('tspan', { dy: '0.32em', 'font-size': '72%' }, G.readerLabel); sb.textContent = String(s);
    G.nextLabel.innerHTML = ''; b = el('tspan', { 'font-style': 'italic' }, G.nextLabel); b.textContent = 'h'; sb = el('tspan', { dy: '0.32em', 'font-size': '72%' }, G.nextLabel); sb.textContent = String(s + 1);
    // caption
    var cap = G.caption; cap.innerHTML = '';
    var B = prior.reduce(function (a, b) { return a + b; }, 0);
    txt(cap, 14, 26, 'application s = ' + s + ' of ' + (n - 1) + '  ·  visit ' + (cur + 1) + ', block f' + (s % K) + ' writes δ' + s + '  ·  read: input 1 + ' + (s + 1) + ' update' + (s ? 's' : '') + ' with total B = ' + fmt(B) + (isFinite(bound()) ? '  (visit-end bound Kλ/(1−λ) = ' + fmt(bound()) + ')' : '  (λ = 1: no decay, no bound)'), { 'font-size': 14 });
    G.exit.setAttribute('stroke', s === n - 1 ? C.navy : C.ghost); G.exit.setAttribute('marker-end', s === n - 1 ? 'url(#lb-arrow-navy)' : 'url(#lb-arrow-faint)');
  }

  // ---------- ledger and chart ----------
  function ledger(s, B) {
    var n = N(), scale = n + 1;
    function row(id, inp, tot, inW, totW) { $(id + '-in').textContent = inp; $(id + '-tot').textContent = tot; $(id + '-mi').style.width = (100 * inW / scale) + '%'; $(id + '-mu').style.width = (100 * totW / scale) + '%'; }
    row('lg-add', '1', String(s + 1), 1, s + 1);
    row('lg-sm', '1/' + (s + 2) + ' = ' + fmt(1 / (s + 2)), fmt((s + 1) / (s + 2)), 1 / (s + 2), (s + 1) / (s + 2));
    row('lg-lb', '1', fmt(B) + (isFinite(bound()) ? ' ≤ ' + fmt(bound()) : ''), 1, B);
  }
  var chart = $('lb-chart'), CH = {};
  function chartStatic() {
    chart.innerHTML = ''; var n = N(), W = 1000, H = 210, i;
    var allB = []; for (i = 0; i < n; i++) allB.push(coeffs(i, false).B);
    var panels = [{ x0: 62, x1: 470, title: 'input coefficient', ymax: 1.12 }, { x0: 560, x1: 985, title: 'total update weight B of the stored updates', ymax: isFinite(bound()) ? Math.max.apply(null, allB) * 1.14 : n * 1.06 }];
    // header: the panel titles, then one row for the right panel's annotations (bound key on the right, off-scale note on
    // the left), above the plot frame, so that no curve can run through them
    CH.panels = panels; CH.title = 17; CH.ann = 32; CH.top = 41; CH.bot = 176;
    panels.forEach(function (p, j) {
      p.px = function (s) { return p.x0 + (p.x1 - p.x0) * (n > 1 ? s / (n - 1) : 0); };
      p.py = function (v) { return CH.bot - (CH.bot - CH.top) * Math.min(v, p.ymax) / p.ymax; };
      el('clipPath', { id: 'lb-clip' + j }, el('defs', {}, chart)).appendChild(el('rect', { x: p.x0 - 2, y: CH.top - 2, width: p.x1 - p.x0 + 4, height: CH.bot - CH.top + 4 }));
      txt(chart, p.x0, CH.title, p.title, { 'font-size': 13, 'font-weight': 'bold' });
      el('rect', { x: p.x0, y: CH.top, width: p.x1 - p.x0, height: CH.bot - CH.top, fill: 'none', stroke: C.ink, 'stroke-width': 0.8 }, chart);
      var ticks = j === 0 ? [0, 0.5, 1] : (p.ymax <= 3 ? [0, 0.5, 1, 1.5, 2, 2.5, 3] : null);
      if (!ticks) { ticks = []; var step = p.ymax > 20 ? 10 : p.ymax > 8 ? 4 : p.ymax > 4 ? 2 : 1; for (i = 0; i <= p.ymax + 1e-9; i += step) ticks.push(i); }
      ticks.forEach(function (v) { if (v > p.ymax) return; var y = p.py(v); el('line', { x1: p.x0, x2: p.x1, y1: y, y2: y, stroke: C.grid, 'stroke-dasharray': '1 3' }, chart); txt(chart, p.x0 - 6, y + 4, String(v), { 'text-anchor': 'end', 'font-size': 11, fill: C.muted }); });
      var xt = n <= 12 ? 1 : n <= 24 ? 4 : 8;
      for (i = 0; i < n; i += xt) { var x = p.px(i); el('line', { x1: x, x2: x, y1: CH.bot, y2: CH.bot + 4, stroke: C.ink, 'stroke-width': 0.8 }, chart); txt(chart, x, CH.bot + 16, String(i), { 'text-anchor': 'middle', 'font-size': 11, fill: C.muted }); }
      txt(chart, (p.x0 + p.x1) / 2, CH.bot + 31, 'application s', { 'text-anchor': 'middle', 'font-size': 11.5, fill: C.muted });
      if (j === 1 && isFinite(bound())) {
        var yb = p.py(bound());
        el('line', { x1: p.x0, x2: p.x1, y1: yb, y2: yb, stroke: C.navy, 'stroke-width': 1, 'stroke-dasharray': '5 4' }, chart);
        // its label as a key in the header row: a short sample of the same dashed line, then the text
        var key = txt(chart, p.x1, CH.ann, 'visit-end total ≤ Kλ/(1−λ) = ' + fmt(bound()), { 'text-anchor': 'end', 'font-size': 11, fill: C.navy }), kw = bbox(key).w || 175;
        el('line', { x1: p.x1 - kw - 29, x2: p.x1 - kw - 6, y1: CH.ann - 3.5, y2: CH.ann - 3.5, stroke: C.navy, 'stroke-width': 1, 'stroke-dasharray': '5 4' }, chart);
      }
    });
    CH.g = el('g', {}, chart);
    // full curves
    CH.add = []; CH.sm = []; CH.lb = []; CH.in_sm = [];
    for (i = 0; i < n; i++) { CH.add.push(i + 1); CH.sm.push((i + 1) / (i + 2)); CH.lb.push(coeffs(i, false).B); CH.in_sm.push(1 / (i + 2)); }
  }
  function chartUpdate(s) {
    var g = CH.g; g.innerHTML = ''; var p0 = CH.panels[0], p1 = CH.panels[1], i;
    function line(p, vals, color, width, dash, clip) {
      var d = ''; for (i = 0; i <= s; i++) d += (i ? ' L' : 'M') + p.px(i) + ' ' + p.py(vals[i]);
      el('path', { d: d, fill: 'none', stroke: color, 'stroke-width': width, 'stroke-dasharray': dash || null, 'stroke-linejoin': 'round', 'clip-path': clip }, g);
      el('circle', { cx: p.px(s), cy: p.py(vals[s]), r: 4, fill: color, stroke: '#fff', 'stroke-width': 1.2, 'clip-path': clip }, g);
    }
    var ones = CH.add.map(function () { return 1; });
    line(p0, ones, C.dense, 4, null, 'url(#lb-clip0)'); line(p0, ones, C.navy, 2, null, 'url(#lb-clip0)'); line(p0, CH.in_sm, C.rose, 2, null, 'url(#lb-clip0)');
    line(p1, CH.add, C.dense, 2, null, 'url(#lb-clip1)'); line(p1, CH.sm, C.rose, 2, null, 'url(#lb-clip1)');
    // Loopback: the within-visit reads as a thin line, the visit-end totals (what the bound is about) emphasised
    line(p1, CH.lb, C.navy, 1.2, null, 'url(#lb-clip1)');
    var d = '', first = true; for (i = 0; i <= s; i++) if ((i + 1) % P.K === 0 || P.lam >= 1) { d += (first ? 'M' : ' L') + p1.px(i) + ' ' + p1.py(CH.lb[i]); first = false; el('circle', { cx: p1.px(i), cy: p1.py(CH.lb[i]), r: 3.2, fill: C.navy }, g); }
    if (d) el('path', { d: d, fill: 'none', stroke: C.navy, 'stroke-width': 2.4, 'stroke-linejoin': 'round' }, g);
    // the addition line leaves the Loopback-scaled panel through its top edge: say so in the header row above that edge
    for (i = 0; i <= s; i++) if (CH.add[i] > p1.ymax) { txt(g, p1.x0, CH.ann, 'residual addition: ' + (s + 1) + ', off the scale', { 'font-size': 11, fill: C.dense }); break; }
  }

  // ---------- animation ----------
  var shown = { prior: [], post: [] }, running = null, playing = false, dotAngle = -Math.PI / 2;
  function ease(u) { return 1 - Math.pow(1 - u, 3); }
  function cleanup() { G.fly.innerHTML = ''; G.badge.innerHTML = ''; G.threads.removeAttribute('class'); if (G.blocks) G.blocks.forEach(function (b) { b.g.setAttribute('transform', 'translate(' + b.x + ' ' + b.y + ')'); }); if (G.readerCircle) { G.readerCircle.setAttribute('r', L.reader.r); G.readerCircle.removeAttribute('filter'); } }
  function cancel() { if (running) { running.alive = false; running = null; } cleanup(); }
  function timeline(steps, done) {
    cancel(); var idx = 0, t0 = null, h = { alive: true }; running = h;
    function frame(ts) {
      if (!h.alive) return;
      var st = steps[idx];
      if (t0 === null) { t0 = ts; if (st.start) st.start(); }
      var dur = reduce ? 0 : st.dur / P.speed, u = dur ? Math.min(1, (ts - t0) / dur) : 1;
      if (st.tick) st.tick(ease(u), u);
      if (u >= 1) { if (st.end) st.end(); idx++; t0 = null; if (idx >= steps.length) { running = null; if (done) done(); return; } }
      requestAnimationFrame(frame);
    }
    requestAnimationFrame(frame);
  }
  function lerpArrays(from, to, e) { return to.map(function (v, i) { var f = i < from.length ? from[i] : 0; return f + (v - f) * e; }); }
  function tweenStep(target, dur) { // coefficient arrays -> drawDynamic
    var from = { prior: shown.prior.slice(), post: shown.post.slice() };
    return { dur: dur, tick: function (e) { shown.prior = lerpArrays(from.prior, target.prior, e); shown.post = lerpArrays(from.post, target.post, e); drawDynamic(shown.prior, shown.post, P.s); },
             end: function () { shown.prior = target.prior.slice(); shown.post = target.post.slice(); drawDynamic(shown.prior, shown.post, P.s); } };
  }
  function dotStep(toAngle, dur) { var from = dotAngle; return { dur: dur, tick: function (e) { dotAngle = from + (toAngle - from) * e; drawDot(dotAngle); }, end: function () { dotAngle = toAngle; drawDot(dotAngle); } }; }
  function pulseStep(k, dur) { var b = G.blocks[k]; return { dur: dur, tick: function (e, u) { var sc = 1 + 0.10 * Math.sin(Math.PI * u); b.g.setAttribute('transform', 'translate(' + b.x + ' ' + b.y + ') scale(' + sc + ')'); }, end: function () { b.g.setAttribute('transform', 'translate(' + b.x + ' ' + b.y + ')'); } }; }
  function flyStep(i, dur) {
    var k = i % P.K, b = G.blocks[k], c = cellPos(i), card = L.grid.card, g;
    return { dur: dur,
      start: function () { g = el('g', {}, G.fly); el('rect', { x: -card / 2, y: -card / 2, width: card, height: card, rx: 5, fill: blockFill(k), stroke: C.navy, 'stroke-width': 1.5, filter: 'url(#lb-glow)' }, g); sym(g, -card / 2 + 7.5, -card / 2 + 16.5, 'δ', String(i), { 'font-size': 12 }); },
      tick: function (e) { var x = b.x + (c.x + card / 2 - b.x) * e, y = b.y - 28 * Math.sin(Math.PI * e) + (c.y + card / 2 - b.y) * e, sc = 0.55 + 0.45 * e; g.setAttribute('transform', 'translate(' + x + ' ' + y + ') scale(' + sc + ')'); g.setAttribute('opacity', 0.5 + 0.5 * e); },
      end: function () { G.fly.innerHTML = ''; } };
  }
  function readerPulse(dur) { return { dur: dur, start: function () { G.threads.setAttribute('class', 'lb-flow'); }, tick: function (e, u) { G.readerCircle.setAttribute('r', L.reader.r * (1 + 0.14 * Math.sin(Math.PI * u))); G.readerCircle.setAttribute('filter', u < 1 ? 'url(#lb-glow)' : null); }, end: function () { G.readerCircle.setAttribute('r', L.reader.r); G.readerCircle.removeAttribute('filter'); G.threads.removeAttribute('class'); } }; }
  function badgeStep(dur, text) {
    var g, gr = L.grid, w = 330, x = gr.x0 + gr.w / 2 - w / 2, y = gr.y0 - 24;
    return { dur: dur,
      start: function () { g = el('g', {}, G.badge); el('rect', { x: x, y: y - 17, width: w, height: 30, rx: 15, fill: C.rose }, g); txt(g, x + w / 2, y + 4, text, { 'text-anchor': 'middle', fill: '#fff', 'font-size': 13.5 }); },
      tick: function (e, u) { g.setAttribute('opacity', u < 0.15 ? u / 0.15 : u > 0.8 ? (1 - u) / 0.2 : 1); g.setAttribute('transform', 'translate(0 ' + (-(1 - Math.min(1, u / 0.15)) * 6) + ')'); },
      end: function () { G.badge.innerHTML = ''; } };
  }
  function wait(dur) { return { dur: dur }; }

  // jump to application s (scrub, step without choreography, parameter changes)
  function show(s, animate, done) {
    P.s = Math.max(0, Math.min(N() - 1, s));
    var k = P.s % P.K, fin = coeffs(P.s, false);
    setBlockHighlight(k); readout(fin);
    var target = L.blockPos[k].a;
    if (!animate) { shown.prior = fin.prior.slice(); shown.post = fin.post.slice(); dotAngle = target; if (P.s === N() - 1) drawDot(0, exitPoint(1)); else drawDot(dotAngle); drawDynamic(shown.prior, shown.post, P.s); if (done) done(); return; }
    while (target < dotAngle - Math.PI) target += 2 * Math.PI; while (target > dotAngle + Math.PI) target -= 2 * Math.PI;
    var st = tweenStep(fin, 320), dt = dotStep(target, 320), last = P.s === N() - 1;
    timeline([{ dur: 320, tick: function (e, u) { st.tick(e, u); if (last) drawDot(0, exitPoint(1)); else dt.tick(e, u); }, end: function () { st.end(); if (last) { dotAngle = target; drawDot(0, exitPoint(1)); } else dt.end(); } }], done);
  }
  // the choreographed application s -> s+1
  function advance(done) {
    var s = P.s + 1; if (s > N() - 1) { if (done) done(); return; }
    var k = s % P.K, v = Math.floor(s / P.K), visitEnd = (s + 1) % P.K === 0 && P.lam < 1;
    P.s = s;
    var pre = coeffs(s, true), fin = coeffs(s, false), target = L.blockPos[k].a;
    while (target <= dotAngle) target += 2 * Math.PI;
    var steps = [dotStep(target, 340), { dur: 10, end: function () { setBlockHighlight(k); } }, pulseStep(k, 240), flyStep(s, 380), tweenStep(pre, 300), readerPulse(260)];
    if (visitEnd) { var bd = badgeStep(900, 'visit ' + (v + 1) + ' ends: every stored weight × λ = ' + fmt(P.lam, 2)), tw = tweenStep(fin, 520); steps.push({ dur: 900, start: function () { bd.start(); }, tick: function (e, u) { bd.tick(e, u); if (u > 0.25) tw.tick(ease(Math.min(1, (u - 0.25) / 0.6)), 0); }, end: function () { bd.end(); tw.end(); } }); }
    steps.push({ dur: 10, end: function () { readout(fin); } });
    if (s === N() - 1) { var tTop = -Math.PI / 2; while (tTop <= target) tTop += 2 * Math.PI; steps.push(wait(120), dotStep(tTop, 300), { dur: 380, tick: function (e) { drawDot(0, exitPoint(e)); } }); }
    steps.push(wait(260));
    timeline(steps, done);
  }
  function readout(c) {
    var n = N(); $('so').value = P.s; $('smax').value = n - 1; $('s').max = n - 1; $('s').value = P.s;
    ledger(P.s, c.B); chartUpdate(P.s);
  }
  function stop() { playing = false; $('btn-play').innerHTML = '&#9654; Play'; }
  function play() {
    if (playing) { stop(); cancel(); show(P.s, false); return; }
    playing = true; $('btn-play').innerHTML = '&#10074;&#10074; Pause';
    var start = P.s >= N() - 1 ? function (f) { show(0, false); f(); } : function (f) { f(); };
    start(function () { (function next() { if (!playing) return; if (P.s >= N() - 1) { stop(); return; } advance(next); })(); });
  }
  function rebuildFocus() {
    var sel = $('focus'), cur = +sel.value, opts = '<option value="0">none (zero queries)</option>', i;
    for (i = 1; i <= P.R; i++) opts += '<option value="' + i + '">visit ' + i + '</option>';
    sel.innerHTML = opts; sel.value = cur <= P.R ? cur : 0; P.focus = +sel.value;
  }
  function readParams() {
    P.K = +$('k').value; P.R = +$('r').value; P.lam = +$('lam').value; P.gain = +$('gain').value; P.speed = +$('speed').value;
    $('ko').value = P.K; $('ro').value = P.R; $('lamo').value = P.lam.toFixed(2); $('gaino').value = P.gain.toFixed(1); $('speedo').value = P.speed.toFixed(2).replace(/0$/, '') + '×';
    rebuildFocus();
  }
  function rebuild(keepS) { layout(); drawRing(); chartStatic(); show(keepS ? Math.min(P.s, N() - 1) : 0, false); }
  ['k', 'r', 'lam', 'gain'].forEach(function (id) { $(id).addEventListener('input', function () { stop(); cancel(); readParams(); rebuild(true); }); });
  $('speed').addEventListener('input', function () { readParams(); });
  $('focus').addEventListener('change', function () { P.focus = +$('focus').value; chartStatic(); show(P.s, true); });
  $('s').addEventListener('input', function () { stop(); cancel(); show(+$('s').value, true); });
  $('btn-play').addEventListener('click', play);
  $('btn-step').addEventListener('click', function () { stop(); cancel(); if (P.s >= N() - 1) show(0, true); else advance(); });
  $('btn-reset').addEventListener('click', function () { stop(); cancel(); show(0, true); });
  buildStage(); readParams(); rebuild(false);
})();
