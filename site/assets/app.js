/* Static modern dashboard — renders window.DASHBOARD_DATA with ECharts.
   No framework: a tiny hash router + per-view renderers. Theme-aware charts. */
(() => {
  const D = window.DASHBOARD_DATA || { runs: [], dataset: {} };
  const RUNS = D.runs || [];
  const byId = Object.fromEntries(RUNS.map(r => [r.id, r]));
  const PALETTE = ['#4f46e5', '#2563eb', '#06b6d4', '#16a34a', '#d97706', '#db2777', '#9333ea', '#64748b'];
  const charts = [];   // live ECharts instances (disposed on re-render)

  const $ = sel => document.querySelector(sel);
  const fmt = (v, d = 3) => (v == null ? '—' : (+v).toFixed(d));
  const isDark = () => document.documentElement.dataset.theme === 'dark';
  const tc = () => isDark()
    ? { ink: '#eef2f8', muted: '#98a4b8', grid: '#1e2738', axis: '#2a3548' }
    : { ink: '#0f1729', muted: '#5b6678', grid: '#eef1f6', axis: '#dde1ec' };

  // ── Chart helpers ───────────────────────────────────────────────────────────
  function mount(el, option) {
    const c = echarts.init(el, null, { renderer: 'canvas' });
    const t = tc();
    option.textStyle = { fontFamily: 'Inter, sans-serif', color: t.muted };
    option.grid = Object.assign({ left: 48, right: 22, top: 30, bottom: 36, containLabel: true }, option.grid || {});
    c.setOption(option);
    charts.push(c);
    return c;
  }
  const axisX = (extra = {}) => Object.assign({
    type: 'category', axisLine: { lineStyle: { color: tc().axis } },
    axisTick: { show: false }, axisLabel: { color: tc().muted, fontSize: 11 },
  }, extra);
  const axisY = (extra = {}) => Object.assign({
    type: 'value', splitLine: { lineStyle: { color: tc().grid } },
    axisLabel: { color: tc().muted, fontSize: 11 },
  }, extra);
  const tip = () => ({ trigger: 'axis', backgroundColor: isDark() ? '#1a2436' : '#fff',
    borderColor: tc().axis, textStyle: { color: tc().ink, fontSize: 12 },
    extraCssText: 'box-shadow:0 8px 24px rgba(0,0,0,.15);border-radius:10px;' });

  function lineChart(el, x, series) {
    mount(el, {
      tooltip: tip(),
      legend: { top: 0, right: 0, icon: 'roundRect', itemWidth: 11, itemHeight: 11,
        textStyle: { color: tc().muted, fontSize: 12 } },
      xAxis: axisX({ data: x, boundaryGap: false }),
      yAxis: axisY(),
      series: series.map((s, i) => ({
        name: s.name, type: 'line', data: s.data, smooth: 0.35, showSymbol: false,
        lineStyle: { width: 2.6, color: s.color || PALETTE[i] },
        itemStyle: { color: s.color || PALETTE[i] },
        areaStyle: s.area ? { color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
          { offset: 0, color: (s.color || PALETTE[i]) + '33' }, { offset: 1, color: (s.color || PALETTE[i]) + '00' }]) } : null,
      })),
    });
  }

  function barH(el, cats, vals, opts = {}) {
    mount(el, {
      tooltip: Object.assign(tip(), { trigger: 'item' }),
      grid: { left: opts.left || 150 },
      xAxis: axisY({ max: opts.max }),
      yAxis: axisX({ type: 'category', data: cats, axisLabel: { color: tc().muted, fontSize: 11, width: (opts.left || 150) - 14, overflow: 'truncate' } }),
      series: [{
        type: 'bar', data: vals.map((v, i) => ({ value: v, itemStyle: { color: opts.colors ? opts.colors[i] : PALETTE[0], borderRadius: [0, 5, 5, 0] } })),
        barWidth: opts.barWidth || '62%',
        label: opts.showLabel ? { show: true, position: 'right', color: tc().muted, fontSize: 11,
          formatter: o => (opts.labelFmt ? opts.labelFmt(o.value) : o.value) } : null,
        markLine: opts.refLine != null ? { silent: true, symbol: 'none',
          lineStyle: { type: 'dashed', color: tc().muted }, label: { formatter: opts.refLabel || '', color: tc().muted, fontSize: 11 },
          data: [{ xAxis: opts.refLine }] } : null,
      }],
    });
  }

  function donut(el, data) {
    mount(el, {
      tooltip: Object.assign(tip(), { trigger: 'item', formatter: '{b}: {c} ({d}%)' }),
      legend: { bottom: 0, left: 'center', icon: 'circle', textStyle: { color: tc().muted, fontSize: 12 } },
      series: [{ type: 'pie', radius: ['52%', '78%'], center: ['50%', '44%'], avoidLabelOverlap: true,
        itemStyle: { borderColor: isDark() ? '#131a29' : '#fff', borderWidth: 3, borderRadius: 6 },
        label: { show: false }, data: data.map((d, i) => ({ name: d.name, value: d.value, itemStyle: { color: PALETTE[i % PALETTE.length] } })) }],
    });
  }

  // ── Views ─────────────────────────────────────────────────────────────────────
  const VIEWS = {};

  VIEWS.overview = () => {
    setHead('Overview', 'The distributed-training study at a glance.');
    const ranked = RUNS.filter(r => r.best_f1 != null).sort((a, b) => b.best_f1 - a.best_f1);
    const top = ranked[0];
    const envs = new Set(RUNS.map(r => r.env)), models = new Set(RUNS.map(r => r.model));
    const c = $('#content');
    c.innerHTML = `
      <div class="hero fade">
        <div><h2>BigEarthNet-S2 · Vision Transformer</h2>
          <p>Single-GPU, distributed (DDP), model-parallel and heterogeneous training — compared,
             with a feasibility model that predicts speedup before you run it.</p></div>
        <div class="hero-stat"><div class="v">${top ? fmt(top.best_f1) : '—'}</div>
          <div class="l">Best Val F1 · ${top ? top.model : ''}</div></div>
      </div>
      <div class="kpis fade">
        ${kpi('Runs', RUNS.length)}
        ${kpi('Best Val F1', top ? fmt(top.best_f1) : '—')}
        ${kpi('Models', models.size)}
        ${kpi('Environments', envs.size)}
      </div>
      <div class="grid grid-2 fade">
        <div class="card"><div class="card-title">Best Val F1 by run (top 8)</div><div class="chart" id="ovBars" style="height:300px"></div></div>
        <div class="card"><div class="card-title">Runs by strategy</div><div class="chart" id="ovDonut" style="height:300px"></div></div>
      </div>
      <div class="card fade" style="margin-top:18px">
        <div class="card-title">All runs · ${RUNS.length}</div>
        <div style="overflow:auto">${leaderboard(ranked.concat(RUNS.filter(r => r.best_f1 == null)))}</div>
      </div>`;
    const tt = ranked.slice(0, 8).reverse();
    barH($('#ovBars'), tt.map(r => r.model + ' · ' + r.date.slice(0, 5)), tt.map(r => r.best_f1),
      { max: 1, showLabel: true, labelFmt: v => v.toFixed(3), left: 170,
        colors: tt.map(r => r.best_f1 >= 0.6 ? '#16a34a' : r.best_f1 >= 0.4 ? '#d97706' : '#4f46e5') });
    const counts = {}; RUNS.forEach(r => counts[r.mode_label] = (counts[r.mode_label] || 0) + 1);
    donut($('#ovDonut'), Object.entries(counts).map(([name, value]) => ({ name, value })));
    bindRows();
  };

  VIEWS.runs = () => {
    setHead('Run results', 'Curves and per-class metrics for one run.');
    const withCurve = RUNS.filter(r => (r.curve.val_f1 || []).length);
    const sel = state.runId && byId[state.runId] ? byId[state.runId] : withCurve[0];
    state.runId = sel ? sel.id : null;
    const opts = withCurve.map(r => `<option value="${r.id}" ${r === sel ? 'selected' : ''}>${r.label}</option>`).join('');
    $('#topbarActions').innerHTML = `<select id="runPick">${opts}</select>`;
    const c = $('#content');
    if (!sel) { c.innerHTML = empty('No runs with epoch metrics.'); return; }
    const cu = sel.curve, best = sel.best_f1;
    c.innerHTML = `
      <div class="kpis fade">
        ${kpi('Best Val F1', fmt(best))}
        ${kpi('Best epoch', sel.best_epoch ?? '—')}
        ${kpi('Epochs', sel.epochs)}
        ${kpi('Duration', sel.duration_min != null ? sel.duration_min + ' <small>min</small>' : '—')}
      </div>
      <div class="row fade" style="margin-bottom:14px">
        ${badge(sel.mode_label, 'accent')} ${badge(sel.precision.toUpperCase(), sel.precision !== 'fp32' ? 'amp' : '')}
        ${badge(sel.env)} ${badge(sel.model)}
      </div>
      <div class="grid grid-2 fade">
        <div class="card"><div class="card-title">F1 (macro)</div><div class="chart" id="rcF1" style="height:280px"></div></div>
        <div class="card"><div class="card-title">Loss</div><div class="chart" id="rcLoss" style="height:280px"></div></div>
      </div>
      ${sel.perclass.length ? `<div class="card fade" style="margin-top:18px">
        <div class="card-title">Per-class F1 — last epoch (sorted)</div>
        <div class="chart" id="rcPC" style="height:${Math.max(360, sel.perclass.length * 24 + 60)}px"></div></div>` : ''}`;
    lineChart($('#rcF1'), cu.epoch, [
      { name: 'Train', data: cu.train_f1, color: '#4f46e5' },
      { name: 'Val', data: cu.val_f1, color: '#06b6d4', area: true }]);
    lineChart($('#rcLoss'), cu.epoch, [
      { name: 'Train', data: cu.train_loss, color: '#4f46e5' },
      { name: 'Val', data: cu.val_loss, color: '#db2777' }]);
    if (sel.perclass.length) {
      const pc = [...sel.perclass].sort((a, b) => a.f1 - b.f1);
      barH($('#rcPC'), pc.map(p => p.cls), pc.map(p => p.f1),
        { max: 1, showLabel: true, labelFmt: v => v.toFixed(2), left: 230,
          colors: pc.map(p => p.f1 >= 0.6 ? '#16a34a' : p.f1 >= 0.3 ? '#d97706' : '#dc2626') });
    }
    $('#runPick').onchange = e => { state.runId = e.target.value; render(); };
  };

  VIEWS.compare = () => {
    setHead('Compare', 'Overlay any runs and see per-class differences.');
    $('#topbarActions').innerHTML = '';
    if (!state.cmp) {
      const latest = RUNS[0];
      state.cmp = RUNS.filter(r => r.env === latest.env && r.id.slice(0, 8) === latest.id.slice(0, 8))
        .slice(0, 6).map(r => r.id);
      if (state.cmp.length < 2) state.cmp = RUNS.slice(0, 2).map(r => r.id);
    }
    const c = $('#content');
    c.innerHTML = `
      <div class="card fade"><div class="card-title">Select runs</div>
        <div class="chips" id="cmpChips">${RUNS.map(r =>
          `<span class="chip ${state.cmp.includes(r.id) ? 'on' : ''}" data-id="${r.id}">${r.label}</span>`).join('')}</div></div>
      <div class="grid grid-2 fade" style="margin-top:18px">
        <div class="card"><div class="card-title">Val F1 across epochs</div><div class="chart" id="cmpF1" style="height:300px"></div></div>
        <div class="card"><div class="card-title">Best Val F1</div><div class="chart" id="cmpBar" style="height:300px"></div></div>
      </div>
      <div id="cmpDumb"></div>`;
    const sel = state.cmp.map(id => byId[id]).filter(Boolean);
    const allEp = [...new Set(sel.flatMap(r => r.curve.epoch || []))].sort((a, b) => a - b);
    lineChart($('#cmpF1'), allEp, sel.map((r, i) => ({ name: shortLabel(r), data: r.curve.val_f1, color: PALETTE[i % PALETTE.length] })));
    const sb = [...sel].filter(r => r.best_f1 != null).sort((a, b) => a.best_f1 - b.best_f1);
    barH($('#cmpBar'), sb.map(shortLabel), sb.map(r => r.best_f1), { max: 1, showLabel: true, labelFmt: v => v.toFixed(3), left: 180,
      colors: sb.map((_, i) => PALETTE[i % PALETTE.length]) });
    dumbbell(sel);
    document.querySelectorAll('#cmpChips .chip').forEach(ch => ch.onclick = () => {
      const id = ch.dataset.id;
      state.cmp = state.cmp.includes(id) ? state.cmp.filter(x => x !== id) : [...state.cmp, id].slice(0, 8);
      render();
    });
  };

  function dumbbell(sel) {
    const wpc = sel.filter(r => r.perclass.length);
    if (wpc.length < 2) return;
    const A = wpc[0], B = wpc[1];
    const ma = Object.fromEntries(A.perclass.map(p => [p.cls, p.f1]));
    const mb = Object.fromEntries(B.perclass.map(p => [p.cls, p.f1]));
    const cls = Object.keys(ma).filter(c => c in mb).sort((x, y) => (mb[x] - ma[x]) - (mb[y] - ma[y]));
    const data = cls.map((c, i) => [i, ma[c], mb[c]]);
    const better = cls.filter(c => mb[c] - ma[c] > 0.01).length, worse = cls.filter(c => mb[c] - ma[c] < -0.01).length;
    $('#cmpDumb').innerHTML = `<div class="card fade" style="margin-top:18px">
      <div class="card-title">Per-class F1 · ${shortLabel(A)} → ${shortLabel(B)}</div>
      <div class="chart" id="dumb" style="height:${Math.max(360, cls.length * 24 + 60)}px"></div>
      <div class="verdict">B improves <b>${better}</b> class(es) and loses ${worse}
        — macro F1 ${fmt(avg(Object.values(ma)))} → ${fmt(avg(Object.values(mb)))}.</div></div>`;
    const acc = isDark() ? '#6366f1' : '#4f46e5';
    mount($('#dumb'), {
      tooltip: { trigger: 'item', backgroundColor: isDark() ? '#1a2436' : '#fff', borderColor: tc().axis,
        textStyle: { color: tc().ink }, formatter: p => `${cls[p.value[0]]}<br>A ${p.value[1].toFixed(3)} · B ${p.value[2].toFixed(3)}` },
      grid: { left: 230 },
      xAxis: axisY({ max: 1 }),
      yAxis: axisX({ type: 'category', data: cls, axisLabel: { color: tc().muted, fontSize: 11, width: 216, overflow: 'truncate' } }),
      series: [{
        type: 'custom', encode: { x: [1, 2], y: 0 },
        renderItem: (params, api) => {
          const cat = api.value(0), a = api.coord([api.value(1), cat]), b = api.coord([api.value(2), cat]);
          const up = api.value(2) >= api.value(1), col = Math.abs(api.value(2) - api.value(1)) < 0.01 ? '#94a3b8' : (up ? '#16a34a' : '#dc2626');
          return { type: 'group', children: [
            { type: 'line', shape: { x1: a[0], y1: a[1], x2: b[0], y2: b[1] }, style: { stroke: col, lineWidth: 2.5 } },
            { type: 'circle', shape: { cx: a[0], cy: a[1], r: 5 }, style: { fill: '#94a3b8' } },
            { type: 'circle', shape: { cx: b[0], cy: b[1], r: 5.5 }, style: { fill: acc } } ] };
        }, data,
      }],
    });
  }

  VIEWS.dataset = () => {
    setHead('Dataset', 'BigEarthNet-S2 — splits, classes and the imbalance.');
    $('#topbarActions').innerHTML = '';
    const ds = D.dataset || {}, sp = ds.splits || {}, cls = ds.classes || [];
    const c = $('#content');
    c.innerHTML = `
      <div class="kpis fade">
        ${kpi('Train', (sp.train || 0).toLocaleString())}
        ${kpi('Validation', (sp.val || 0).toLocaleString())}
        ${kpi('Test', (sp.test || 0).toLocaleString())}
        ${kpi('Classes', cls.length || 19)}
      </div>
      <div class="grid grid-2 fade">
        <div class="card"><div class="card-title">Class frequency — the imbalance that caps macro-F1</div>
          <div class="chart" id="dsTree" style="height:340px"></div></div>
        <div class="card"><div class="card-title">Classes by train frequency</div>
          <div class="chart" id="dsBar" style="height:340px"></div></div>
      </div>`;
    if (cls.length) {
      mount($('#dsTree'), {
        tooltip: { trigger: 'item', formatter: p => `${p.name}<br>${(p.value || 0).toLocaleString()} patches` },
        series: [{ type: 'treemap', roam: false, nodeClick: false, breadcrumb: { show: false },
          itemStyle: { borderColor: isDark() ? '#0a0e18' : '#fff', borderWidth: 2, gapWidth: 2 },
          levels: [{ color: PALETTE, colorMappingBy: 'value' }],
          label: { fontSize: 11, color: '#fff', formatter: p => p.value > 0 ? p.name : '' },
          data: cls.map(d => ({ name: d.cls, value: d.count })) }],
      });
      const sorted = [...cls].sort((a, b) => a.count - b.count);
      barH($('#dsBar'), sorted.map(d => d.cls), sorted.map(d => d.count), { left: 230, colors: sorted.map(() => '#2563eb') });
    } else { $('#dsTree').parentElement.parentElement.innerHTML = empty('Dataset metadata not available.'); }
  };

  // ── Small components ────────────────────────────────────────────────────────
  const kpi = (l, v) => `<div class="kpi"><div class="l">${l}</div><div class="v">${v}</div></div>`;
  const badge = (t, cls = '') => t && t !== '—' ? `<span class="badge ${cls}">${t}</span>` : '';
  const empty = m => `<div class="card" style="text-align:center;color:var(--muted);padding:48px">${m}</div>`;
  const avg = a => a.length ? a.reduce((s, v) => s + v, 0) / a.length : 0;
  const shortLabel = r => `${r.model} · ${r.mode_label}${r.precision !== 'fp32' ? ' ' + r.precision : ''}`;
  function leaderboard(rows) {
    return `<table class="tbl"><thead><tr><th>Run</th><th>Model</th><th>Strategy</th><th>Precision</th><th>Env</th><th>Epochs</th><th>Best Val F1</th></tr></thead><tbody>${
      rows.map(r => `<tr data-id="${r.id}" class="${r.id === state.runId ? 'sel' : ''}">
        <td><div style="font-weight:600">${r.date}</div></td>
        <td>${r.model}</td><td>${badge(r.mode_label, 'accent')}</td>
        <td>${badge(r.precision.toUpperCase(), r.precision !== 'fp32' ? 'amp' : '')}</td>
        <td class="muted">${r.env}</td><td class="num">${r.epochs}</td>
        <td><div class="bar-cell"><div class="bar-track"><div class="bar-fill" style="width:${(r.best_f1 || 0) * 100}%"></div></div>
          <span class="num">${fmt(r.best_f1)}</span></div></td></tr>`).join('')}</tbody></table>`;
  }
  function bindRows() {
    document.querySelectorAll('.tbl tbody tr').forEach(tr => tr.onclick = () => {
      state.runId = tr.dataset.id; location.hash = 'runs';
    });
  }

  // ── Router + theme ──────────────────────────────────────────────────────────
  const state = { runId: null, cmp: null };
  function setHead(title, sub) { $('#pageTitle').textContent = title; $('#pageSub').textContent = sub || ''; $('#topbarActions').innerHTML = ''; }
  function render() {
    charts.forEach(c => c.dispose()); charts.length = 0;
    const view = (location.hash.replace('#', '') || 'overview');
    document.querySelectorAll('.nav-item').forEach(n => n.classList.toggle('active', n.dataset.view === view));
    (VIEWS[view] || VIEWS.overview)();
  }
  window.addEventListener('hashchange', render);
  window.addEventListener('resize', () => charts.forEach(c => c.resize()));
  document.querySelectorAll('.nav-item').forEach(n => n.onclick = () => { location.hash = n.dataset.view; });

  const root = document.documentElement;
  root.dataset.theme = localStorage.getItem('theme') || 'light';
  $('#themeToggle').onclick = () => {
    root.dataset.theme = isDark() ? 'light' : 'dark';
    localStorage.setItem('theme', root.dataset.theme); render();
  };
  $('#genStamp').textContent = 'Updated ' + (D.generated || '');
  render();
})();
