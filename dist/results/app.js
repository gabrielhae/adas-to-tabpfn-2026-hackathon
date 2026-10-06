
const $ = s => document.querySelector(s);
const state = { clips: [], cur: null, tel: null, plots: [], filters: {} };

/* ---------------- helpers ---------------- */
const fmtNum = (v, d = 2) => (v === null || v === undefined || Number.isNaN(v)) ? 'â€“' : (+v).toFixed(d);

function panelDefs(tel) {
  const s = tel.series;
  const hz = tel.native_hz;
  const has = k => s[k] && s[k].some(v => v !== null && !Number.isNaN(v));
  // `dig` pins the legend precision so values can't reflow and flicker.
  // `scale` selects the y-axis ('y' left, 'y2' right).
  const defs = [
    // ---- hero: full-width longitudinal context -------------------------
    { title: 'Speed & cruise set', unit: 'm/s', native: hz.carState, hero: true,
      series: [
        has('vEgo') && { key: 'vEgo', label: 'vEgo', color: '#58a6ff', width: 2, dig: 2 },
        has('cruiseState.speed') && { key: 'cruiseState.speed', label: 'cruise set', color: '#8b949e', dash: [4, 3], dig: 2 },
        has('vCruise') && { key: 'vCruise', label: 'vCruise', color: '#bc8cff', dash: [2, 3], dig: 2 },
      ].filter(Boolean) },

    // ---- 3 x 2 grid ----------------------------------------------------
    { title: 'Acceleration', unit: 'm/sÂ²', native: hz.carState,
      series: [
        has('aEgo') && { key: 'aEgo', label: 'aEgo', color: '#3fb950', width: 2, dig: 2 },
        has('aTarget') && { key: 'aTarget', label: 'aTarget', color: '#d29922', dash: [4, 3], dig: 2 },
        has('actuators.accel') && { key: 'actuators.accel', label: 'cmd accel', color: '#bc8cff', dash: [2, 3], dig: 2 },
      ].filter(Boolean) },
    { title: 'Steering', unit: 'deg / NÂ·m', native: hz.carState,
      series: [
        has('steeringAngleDeg') && { key: 'steeringAngleDeg', label: 'angle', color: '#f0883e', width: 2, dig: 1 },
        has('steeringTorque') && { key: 'steeringTorque', label: 'torque', color: '#ec6cb9', dig: 0 },
      ].filter(Boolean) },
    { title: 'Driver inputs', unit: 'on / off', native: hz.carState, step: true, height: 116,
      series: [
        has('brakePressed') && { key: 'brakePressed', label: 'brake', color: '#f85149', width: 2, dig: 0 },
        has('gasPressed') && { key: 'gasPressed', label: 'gas', color: '#3fb950', dig: 0 },
        has('steeringPressed') && { key: 'steeringPressed', label: 'steer', color: '#58a6ff', dig: 0 },
      ].filter(Boolean) },
    { title: 'ADAS engagement', unit: 'on / off', native: hz.controlsState, step: true, height: 116,
      series: [
        has('adas_engaged') && { key: 'adas_engaged', label: 'engaged (OR)', color: '#1f6feb', width: 2, dig: 0 },
        has('enabled') && { key: 'enabled', label: 'ctrl.enabled', color: '#bc8cff', dash: [3, 3], dig: 0 },
        has('active') && { key: 'active', label: 'active', color: '#3fb950', dash: [3, 3], dig: 0 },
        has('latActive') && { key: 'latActive', label: 'latActive', color: '#f0883e', dash: [2, 2], dig: 0 },
        has('longActive') && { key: 'longActive', label: 'longActive', color: '#d29922', dash: [2, 2], dig: 0 },
      ].filter(Boolean) },
    // dual axis: distance/speed on the left, safety margins (seconds) on the right
    { title: 'Lead vehicle & safety margins', unit: 'm Â· m/s | s',
      native: hz.radarState, height: 116,
      series: [
        has('leadOne.dRel') && { key: 'leadOne.dRel', label: 'dRel', color: '#58a6ff', width: 2, dig: 1 },
        has('leadTwo.dRel') && { key: 'leadTwo.dRel', label: 'dRelâ‚‚', color: '#6e7681', dash: [3, 3], dig: 1 },
        has('leadOne.vRel') && { key: 'leadOne.vRel', label: 'vRel', color: '#f0883e', dig: 1 },
        has('ttc') && { key: 'ttc', label: 'TTC', color: '#f85149', width: 2, dig: 2, scale: 'y2' },
        has('thw') && { key: 'thw', label: 'THW', color: '#3fb950', dig: 2, scale: 'y2' },
      ].filter(Boolean) },
    { title: 'Lane perception & planner', unit: 'prob | 1/m',
      native: hz.drivingModelData, height: 116,
      series: [
        has('laneLineMeta.leftProb') && { key: 'laneLineMeta.leftProb', label: 'leftProb', color: '#58a6ff', dig: 2 },
        has('laneLineMeta.rightProb') && { key: 'laneLineMeta.rightProb', label: 'rightProb', color: '#3fb950', dig: 2 },
        has('curvature') && { key: 'curvature', label: 'curvature', color: '#f0883e', dash: [4, 3], dig: 4 },
        has('desiredCurvature') && { key: 'desiredCurvature', label: 'desired', color: '#bc8cff', dash: [2, 3], dig: 4 },
      ].filter(Boolean) },
  ];
  return defs.filter(d => d.series.length);
}

// Widths must follow the CSS grid, not the width measured at construction time
// (which is often 0 or stale because the panel was just inserted).
const RO = new ResizeObserver(entries => {
  for (const e of entries) {
    const rec = e.target.__rec;
    if (!rec) continue;
    const w = Math.floor(e.contentRect.width);
    if (w > 60 && Math.abs(w - (rec.u.width || 0)) > 1) {
      rec.u.setSize({ width: w, height: rec.height });
    }
  }
});

function destroyPlots() {
  RO.disconnect();
  state.plots.forEach(({ u, el }) => {
    try { el && (el.__rec = null); u.destroy(); } catch (e) {}
  });
  state.plots = [];
}

const AXIS = { stroke: '#8b949e', grid: { stroke: '#1e252e' }, ticks: { stroke: '#2a3340' } };
const LEGEND = { show: true, live: true };

function makePanel(def, host, x, tel) {
  const el = document.createElement('div');
  el.className = 'panel' + (def.hero ? ' hero' : '');
  const head = document.createElement('div');
  head.className = 'h';
  head.innerHTML = `<span>${def.title}</span>` +
    `<em>${def.unit} Â· native ${fmtNum(def.native, 1)} Hz</em>`;
  const box = document.createElement('div');
  el.appendChild(head); el.appendChild(box); host.appendChild(el);

  const usesY2 = def.series.some(s => s.scale === 'y2');
  const height = def.height || (def.hero ? 190 : 132);

  const data = [x, ...def.series.map(s => tel.series[s.key])];
  const u = new uPlot({
    width: Math.floor(box.getBoundingClientRect().width) || 900, height,
    cursor: { sync: { key: 'adas' }, x: true, y: false, points: { size: 6 } },
    legend: LEGEND,
    scales: usesY2
      ? { x: { time: false }, y: { auto: true }, y2: { auto: true } }
      : { x: { time: false }, y: { auto: true } },
    axes: usesY2
      ? [
          { ...AXIS, values: (u, vs) => vs.map(v => v.toFixed(0) + 's') },
          { ...AXIS, size: def.hero ? 52 : 46 },
          { ...AXIS, scale: 'y2', side: 1, grid: { show: false }, size: def.hero ? 52 : 46 },
        ]
      : [
          { ...AXIS, values: (u, vs) => vs.map(v => v.toFixed(0) + 's') },
          { ...AXIS, size: def.hero ? 52 : 46 },
        ],
    series: [
      // `class` is copied onto the legend row by uPlot, so this is an exact,
      // position-independent handle on the x-series row.
      { label: 't', class: 'u-xrow' },
      ...def.series.map(s => ({
        label: s.label,
        stroke: s.color,
        width: s.width || 1,
        dash: s.dash,
        scale: s.scale || 'y',
        points: { show: false },
        // fixed precision -> the legend never changes width mid-playback
        value: (self, v) => (v === null || v === undefined || Number.isNaN(v))
          ? 'â€“' : v.toFixed(s.dig),
        ...(def.step ? { paths: uPlot.paths.stepped({ align: 1 }) } : {}),
      })),
    ],
    hooks: {
      draw: [self => {
        const ctx = self.ctx, { left, top, height: hh } = self.bbox;
        const xs = self.valToPos(0, 'x', true);
        ctx.save();
        ctx.strokeStyle = '#f85149';
        ctx.lineWidth = def.hero ? 1.75 : 1.25;
        ctx.setLineDash([]);
        ctx.beginPath(); ctx.moveTo(xs, top); ctx.lineTo(xs, top + hh); ctx.stroke();
        ctx.restore();
      }],
    },
  }, data);

  box.appendChild(u.root);
  box.__rec = { u, height };
  RO.observe(box);
  // force an immediate correct width once layout has settled
  requestAnimationFrame(() => {
    const w = Math.floor(box.getBoundingClientRect().width);
    if (w > 60) u.setSize({ width: w, height });
  });
  u.over.addEventListener('click', e => {
    const rect = u.over.getBoundingClientRect();
    seek(u.posToVal(e.clientX - rect.left, 'x') + 10);
  });
  state.plots.push({ u, el: box, height });
  return el;
}

function renderCharts(tel) {
  const host = $('#charts');
  destroyPlots();
  lastT = NaN;
  host.innerHTML = '';
  const x = tel.t;
  const defs = panelDefs(tel);
  if (!defs.length) { host.innerHTML = '<div id="loading">no plottable series</div>'; return; }

  const hero = defs[0];
  const rest = defs.slice(1);

  const wrap = document.createElement('div');
  makePanel(hero, wrap, x, tel);              // full width, on its own
  host.appendChild(wrap);

  if (rest.length) {
    const grid = document.createElement('div');
    grid.className = 'grid3x2';               // 3 rows x 2 columns
    host.appendChild(grid);
    rest.forEach(d => makePanel(d, grid, x, tel));
  }
  moveCursor(true);
}

// (sizing is handled by the ResizeObserver above; window resize needs no handler)

/* ---------------- video sync ---------------- */
const video = $('#video');
let rafId = null;
let lastT = NaN;

function moveCursor(force = false) {
  const t = (video.currentTime || 0) - 10;      // telemetry x = videoTime - 10

  // The hero panel and the grid panels have DIFFERENT pixel widths, so a single
  // pixel offset cannot be shared across them - each plot must convert the shared
  // time value into its own pixel space. That is why the marker only tracked on
  // the full-width panel before.
  //
  // Driving this from rAF (not `timeupdate`, which fires ~4x/s) is what makes the
  // motion smooth; the time-epsilon check keeps it cheap while still sub-pixel accurate.
  if (state.plots.length && (force || !(Math.abs(t - lastT) < 0.002))) {
    lastT = t;
    for (const p of state.plots) {
      p.u.setCursor({ left: p.u.valToPos(t, 'x') });
    }
  }

  const dur = video.duration || 20;
  $('#playhead').style.left =
    Math.min(100, Math.max(0, (video.currentTime / dur) * 100)) + '%';
  $('#vt').textContent =
    `${video.currentTime.toFixed(2)}s / ${dur.toFixed(2)}s  Â·  t=${t >= 0 ? '+' : ''}${t.toFixed(2)}s`;
  $('#hud').innerHTML = `<b>t = ${t >= 0 ? '+' : ''}${t.toFixed(2)} s</b>` +
    (Math.abs(t) < 0.25 ? ' &nbsp;â† TAKEOVER' : '');
}

function tick() {
  moveCursor();
  rafId = requestAnimationFrame(tick);
}
function startSync() { lastT = NaN; if (rafId === null) rafId = requestAnimationFrame(tick); }
function stopSync() {
  if (rafId !== null) { cancelAnimationFrame(rafId); rafId = null; }
  moveCursor(true);                    // settle exactly on the paused frame
}

// smooth while playing; hand cursor control back to the mouse when paused
video.addEventListener('play', startSync);
video.addEventListener('playing', startSync);
video.addEventListener('pause', stopSync);
video.addEventListener('ended', stopSync);
video.addEventListener('seeked', () => moveCursor(true));
video.addEventListener('loadedmetadata', () => moveCursor(true));
video.addEventListener('timeupdate', () => { if (video.paused) moveCursor(); });

function seek(t) {
  lastT = NaN;
  video.currentTime = Math.max(0, Math.min(video.duration || 20, t));
  moveCursor(true);
}

/* ---------------- data loading ---------------- */
async function loadClip(key) {
  state.cur = key;
  document.querySelectorAll('.clip').forEach(e => e.classList.toggle('active', e.dataset.key === key));
  // reset previous clip's UI so nothing accumulates across selections
  destroyPlots();
  $('#charts').innerHTML = '<div id="loading">loading telemetryâ€¦</div>';
  $('#info').innerHTML = '<span class="muted">loadingâ€¦</span>';
  $('#predout').innerHTML = '<span class="muted">â€“</span>';
  $('#modelname').textContent = '';
  const r = await fetch(`/api/clip/${key}`);
  if (!r.ok) { $('#charts').innerHTML = '<div id="loading">failed: ' + r.status + '</div>'; return; }
  const d = await r.json();
  state.tel = d.telemetry;
  video.src = `/api/media/${key}`;
  video.load();

  const m = d.meta;
  const trig = m.primary_trigger || 'â€” (no labels: clip not re-identified)';
  const warn = d.telemetry.warnings || [];
  const warnHtml = warn.length
    ? `<div class="warn">${warn.map(w => 'â€¢ ' + w).join('<br/>')}</div>` : '';
  $('#info').innerHTML = `
    <div class="kv">
      <div class="k">clip</div><div>${key}</div>
      <div class="k">brand / model</div><div>${m.brand || 'â€“'} Â· ${m.car_model}</div>
      <div class="k">powertrain</div><div>${m.powertrain || 'â€“'}</div>
      <div class="k">log</div><div>${m.log_kind} @ ${m.log_hz} Hz Â· cam ${m.camera_fps} fps</div>
      <div class="k">trigger</div><div><b>${trig}</b></div>
      <div class="k">scenario</div><div>${m.scenario || 'â€“'}</div>
      <div class="k">post maneuver</div><div>${m.post_maneuver_type || 'â€“'}</div>
      <div class="k">risk / maneuver</div><div>${fmtNum(m.risk_score,3)} / ${fmtNum(m.maneuver_score,3)}</div>
      <div class="k">ADAS engaged</div><div>${fmtNum(d.telemetry.adas_engaged_pct,1)} % of clip</div>
    </div>` + warnHtml;

  // engagement scrubber
  const eng = d.telemetry.series.adas_engaged || [];
  const n = eng.length; let i = 0, pct = 50;
  for (; i < n; i++) if (eng[i] < 0.5) { pct = 100 * i / n; break; }
  $('#seg_eng').style.width = pct + '%';
  $('#seg_man').style.left = pct + '%'; $('#seg_man').style.width = (100 - pct) + '%';

  renderCharts(d.telemetry);
  moveCursor();
  if (d.labelled) runPredict(key);
}

async function runPredict(key) {
  const tgt = 'post_maneuver_type';
  $('#predout').innerHTML = 'runningâ€¦';
  $('#modelname').textContent = '';
  try {
    const r = await fetch(`/api/predict?car_model=${encodeURIComponent(key.split('/')[0])}` +
      `&driver=${key.split('/')[1]}&route=${key.split('/')[2]}&clip_id=${key.split('/')[3]}&target=${tgt}`);
    if (!r.ok) { $('#predout').textContent = 'unavailable (' + r.status + ')'; return; }
    const d = await r.json();
    $('#modelname').textContent = `Â· ${d.model}`;
    let html = `<div>target <b>${d.target}</b> Â· leave-driver-out</div>`;
    if (d.probabilities) {
      const rows = Object.entries(d.probabilities).sort((a, b) => b[1] - a[1]);
      html += '<table class="pr">';
      for (const [c, p] of rows) {
        const mark = (c === d.actual) ? ' âœ”' : '';
        html += `<tr><td>${c}${mark}</td><td>${(p*100).toFixed(1)}%</td></tr>
                 <tr><td colspan="2"><div class="bar"><i style="width:${(p*100).toFixed(1)}%"></i></div></td></tr>`;
      }
      html += '</table>';
      html += `<div class="muted" style="margin-top:6px">actual: <b>${d.actual}</b></div>`;
    } else {
      html += `<div>predicted <b>${fmtNum(d.prediction,3)}</b></div>`;
    }
    html += d.degenerate ? '<div class="warn">small training set â€” treat with caution</div>' : '';
    $('#predout').innerHTML = html;
  } catch (e) { $('#predout').textContent = 'error: ' + e; }
}

/* ---------------- filters / list ---------------- */
function opts(sel, obj, label) {
  const el = $(sel); el.innerHTML = `<option value="">all ${label}</option>`;
  Object.entries(obj || {}).sort((a, b) => b[1] - a[1]).forEach(([k, v]) => {
    const o = document.createElement('option'); o.value = k; o.textContent = `${k} (${v})`; el.appendChild(o);
  });
}
async function loadList() {
  const f = state.filters;
  const p = new URLSearchParams();
  Object.entries(f).forEach(([k, v]) => { if (v) p.set(k, v); });
  p.set('limit', '400');
  const r = await fetch('/api/index?' + p);
  const d = await r.json();
  state.clips = d.clips;
  $('#cliplist').innerHTML = d.clips.map(c => `
    <div class="clip" data-key="${c.key}">
      <div class="k">${c.car_model}</div>
      <div class="s">${c.driver}/${c.route}#${c.clip_id_pub}</div>
      <div>${c.primary_trigger ? `<span class="pill">${c.primary_trigger}</span>` : ''}
           <span class="pill ${c.log_kind === 'qlog' ? 'q' : 'r'}">${c.log_kind}</span></div>
    </div>`).join('') || '<div id="loading">no clips</div>';
  document.querySelectorAll('.clip').forEach(e =>
    e.addEventListener('click', () => loadClip(e.dataset.key)));
  $('#subtitle').textContent = `${d.total} clips match Â· ${VIEW_TOTAL} total`;
}

let VIEW_TOTAL = 0;
async function boot() {
  const [fac, mdl] = await Promise.all([
    fetch('/api/facets').then(r => r.json()),
    fetch('/api/model').then(r => r.json()),
  ]);
  VIEW_TOTAL = fac.n_clips;
  $('#subtitle').textContent = `${fac.n_clips} clips Â· ${fac.n_labelled} labelled Â· TabPFN ${mdl.tabpfn_available ? 'ready' : 'needs token'}`;
  opts('#f_brand', fac.brand, 'brands');
  opts('#f_trigger', fac.primary_trigger, 'triggers');
  opts('#f_log', fac.log_kind, 'log kinds');
  opts('#f_powertrain', fac.powertrain, 'powertrains');

  $('#f_brand').onchange = e => { state.filters.brand = e.target.value; loadList(); };
  $('#f_trigger').onchange = e => { state.filters.trigger = e.target.value; loadList(); };
  $('#f_log').onchange = e => { state.filters.log_kind = e.target.value; loadList(); };
  $('#f_powertrain').onchange = e => { state.filters.powertrain = e.target.value; loadList(); };
  let t; $('#q').oninput = e => { clearTimeout(t); t = setTimeout(() => { state.filters.q = e.target.value; loadList(); }, 250); };
  $('#c_labelled').onclick = e => {
    e.target.classList.toggle('on');
    state.filters.labelled_only = e.target.classList.contains('on') ? 'true' : '';
    loadList();
  };
  $('#c_reset').onclick = () => { state.filters = {}; ['#f_brand','#f_trigger','#f_log','#f_powertrain','#q'].forEach(s=>$(s).value=''); $('#c_labelled').classList.remove('on'); loadList(); };

  $('#btn_play').onclick = () => { video.paused ? video.play() : video.pause(); };
  $('#btn_pre').onclick = () => seek(video.currentTime - 1);
  $('#btn_post').onclick = () => seek(video.currentTime + 1);
  $('#btn_event').onclick = () => seek(10);
  $('#scrub').onclick = e => {
    const r = e.currentTarget.getBoundingClientRect();
    seek((e.clientX - r.left) / r.width * (video.duration || 20));
  };
  video.addEventListener('play', () => $('#btn_play').textContent = 'âšâš pause');
  video.addEventListener('pause', () => $('#btn_play').textContent = 'â–¶ play');

  await loadList();
  // open a labelled clip with a lead vehicle so the demo is interesting
  const first = state.clips.find(c => c.primary_trigger) || state.clips[0];
  if (first) loadClip(first.key);
}
boot();

