// Shared SVG chart helpers for the drafts. Colors come from CSS variables of each direction.
window.CH = (() => {
  const D = window.DEMO, pct = v => (v * 100).toFixed(1);
  const ICON_OK = '<svg viewBox="0 0 16 16" aria-hidden="true"><circle cx="8" cy="8" r="6.6" fill="none" stroke="currentColor" stroke-width="1.6"/><path d="m5 8.2 2.1 2.1L11.2 6" fill="none" stroke="currentColor" stroke-width="1.7"/></svg>';
  const ICON_WARN = '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M8 1.5 15 14H1z" fill="none" stroke="currentColor" stroke-width="1.6"/><path d="M8 6v4M8 11.6v.4" stroke="currentColor" stroke-width="1.6"/></svg>';
  const badge = (abn, cls = 'badge') => `<span class="${cls} ${abn ? 'abn' : 'nor'}">${abn ? ICON_WARN + 'Abnormal' : ICON_OK + 'Normal'}</span>`;

  function scoreBar(m) {
    const dec = m.type === 'decision', p = D.pos(m.score, m.type), t = D.pos(m.thr, m.type);
    const ticks = dec ? [-3, -2, -1, 0, 1, 2, 3] : [0, .25, .5, .75, 1];
    return `<div class="sb">
      <div class="sb-head"><span>${dec ? 'Decision score (positive = abnormal)' : 'Probability of abnormal'}</span>
        <span class="sb-num"><b>${m.score.toFixed(4)}</b> · threshold ${m.thr.toFixed(dec ? 0 : 2)}</span></div>
      <div class="sb-track" role="img" aria-label="score ${m.score.toFixed(4)}, threshold ${m.thr}">
        <span class="sb-norm" style="width:${t}%"></span><span class="sb-abn" style="left:${t}%"></span>
        <span class="sb-thr" style="left:${t}%"></span><span class="sb-mark" style="left:${p}%"></span></div>
      <div class="sb-ticks">${ticks.map(v => `<span style="left:${D.pos(v, m.type)}%">${dec && v > 0 ? '+' : ''}${v}</span>`).join('')}</div>
      <div class="sb-sides"><span>&larr; normal side</span><span>abnormal side &rarr;</span></div></div>`;
  }

  function reliability(m) {
    const row = (k, l) => `<div class="rl-row"><span>${l}</span><span class="rl-bar"><i style="left:${(m.m[k][0] - m.m[k][1]) * 100}%;width:${m.m[k][1] * 200}%" class="rl-sd"></i><i style="left:${m.m[k][0] * 100}%" class="rl-pt"></i></span><b>${pct(m.m[k][0])} ± ${pct(m.m[k][1])}%</b></div>`;
    return `<div class="rl"><div class="rl-title">3-fold cross-validated test metrics <span>(not the confidence of this prediction)</span></div>
      ${row('recall', 'Test recall')}${row('spec', 'Specificity')}${row('bal', 'Balanced accuracy')}</div>`;
  }

  const distTag = m => {
    const rule = m.type === 'decision' ? 'Rule: |decision score| >= 1.0 means far' : 'Rule: |P(abnormal) - threshold| >= 0.15 means far';
    return `<span class="dist ${D.far(m) ? 'far' : 'close'}" title="${rule}. Describes distance only, not a probability of being correct.">${D.far(m) ? 'Far from threshold' : 'Close to threshold'}</span>`;
  };
  const links = () => '<a href="#">3-fold results in MLflow ↗</a><a href="#">train vs test ↗</a><a href="#">analysis in MLflow ↗</a>';

  function waveform(w = 900, h = 200) {
    const pts = D.wave(), dur = D.file.dur, x = t => 44 + t / dur * (w - 60), y = v => h / 2 - 8 - v * (h / 2 - 30);
    const path = pts.map(([t, v], i) => `${i ? 'L' : 'M'}${x(t).toFixed(1)} ${y(v).toFixed(1)}`).join('');
    let ticks = '';
    for (let s = 0; s <= Math.floor(dur); s++) ticks += `<line class="ax" x1="${x(s)}" x2="${x(s)}" y1="${h - 30}" y2="${h - 25}"/><text x="${x(s)}" y="${h - 10}" text-anchor="middle">${s} s</text>`;
    return `<svg class="wave" viewBox="0 0 ${w} ${h}" role="img" aria-label="Waveform, time axis in seconds">
      <line class="ax" x1="44" x2="${w - 16}" y1="${h - 30}" y2="${h - 30}"/><line class="zero" x1="44" x2="${w - 16}" y1="${y(0)}" y2="${y(0)}"/>
      <path class="sig" d="${path}"/>${ticks}<text x="6" y="${y(1) + 4}">+1</text><text x="6" y="${y(-1) + 4}">−1</text>
      <line class="ph" data-ph x1="44" x2="44" y1="14" y2="${h - 30}"/></svg>`;
  }

  function grouped(w = 860, h = 260) {
    const ks = [['recall', 'Test recall'], ['bal', 'Balanced acc.'], ['f1', 'F1']], gw = (w - 50) / D.models.length, bw = gw / 4.2;
    const sy = v => h - 34 - (v - 0.5) * 2 * (h - 60);
    let s = `<svg class="chart" viewBox="0 0 ${w} ${h}" role="img" aria-label="Grouped test metrics with std error bars">`;
    [.5, .6, .7, .8, .9, 1].forEach(v => s += `<line class="grid" x1="40" x2="${w}" y1="${sy(v)}" y2="${sy(v)}"/><text x="4" y="${sy(v) + 4}">${v.toFixed(1)}</text>`);
    D.models.forEach((m, i) => {
      const gx = 46 + i * gw + gw * 0.12;
      ks.forEach(([k], j) => {
        const [mu, sd] = m.m[k], x = gx + j * bw;
        s += `<rect class="b${j}" x="${x}" y="${sy(mu)}" width="${bw - 4}" height="${h - 34 - sy(mu)}"><title>${m.short} ${ks[j][1]}: ${mu.toFixed(3)} ± ${sd.toFixed(3)}</title></rect><line class="err" x1="${x + bw / 2 - 2}" x2="${x + bw / 2 - 2}" y1="${sy(mu + sd)}" y2="${sy(mu - sd)}"/>`;
      });
      s += `<text x="${gx + bw * 1.5}" y="${h - 14}" text-anchor="middle">${m.short}</text>`;
    });
    return s + `</svg><div class="legend">${ks.map(([, l], j) => `<span><i class="b${j}"></i>${l}</span>`).join('')}<span>error bar = std over 3 test folds · y-axis starts at 0.5</span></div>`;
  }

  function trainTest(w = 860, h = 210) {
    const gw = (w - 50) / D.models.length, bw = gw / 3.2, sy = v => h - 34 - (v - 0.5) * 2 * (h - 54);
    let s = `<svg class="chart" viewBox="0 0 ${w} ${h}" role="img" aria-label="Train vs test balanced accuracy">`;
    [.5, .75, 1].forEach(v => s += `<line class="grid" x1="40" x2="${w}" y1="${sy(v)}" y2="${sy(v)}"/><text x="4" y="${sy(v) + 4}">${v}</text>`);
    D.models.forEach((m, i) => {
      const gx = 46 + i * gw + gw * .18;
      [[m.trainBal, 'tr'], [m.m.bal[0], 'te']].forEach(([v, c], j) => s += `<rect class="${c}" x="${gx + j * bw}" y="${sy(v)}" width="${bw - 4}" height="${h - 34 - sy(v)}"><title>${c === 'tr' ? 'train' : 'test'} ${v.toFixed(3)}</title></rect><text class="val" x="${gx + j * bw + bw / 2 - 2}" y="${sy(v) - 5}" text-anchor="middle">${v.toFixed(2)}</text>`);
      s += `<text x="${gx + bw}" y="${h - 20}" text-anchor="middle">${m.short}</text><text x="${gx + bw}" y="${h - 5}" text-anchor="middle">gap ${(m.trainBal - m.m.bal[0]).toFixed(3)}</text>`;
    });
    return s + '</svg><div class="legend"><span><i class="tr"></i>Train</span><span><i class="te"></i>Test</span><span>balanced accuracy, mean of 3 folds</span></div>';
  }

  const heatmap = () => `<table class="heat"><thead><tr><th>Test recall</th><th>Fold 1</th><th>Fold 2</th><th>Fold 3</th></tr></thead><tbody>${D.models.map(m => `<tr><th>${m.short}</th>${m.foldsR.map(v => `<td style="--v:${((v - .65) / .3).toFixed(2)}">${v.toFixed(3)}</td>`).join('')}</tr>`).join('')}</tbody></table>`;

  const confusion = () => `<div class="cm"><div class="cm-sel"><label>Model <select><option>SVM linear</option><option>SVM RBF</option><option>Tree</option><option>CNN</option><option>VGG16</option></select></label><label>Fold <select><option>Sum of 3 test folds</option><option>Fold 1</option><option>Fold 2</option><option>Fold 3</option></select></label></div>
    <table><tr><td></td><th>Pred. normal</th><th>Pred. abnormal</th></tr><tr><th>True normal</th><td class="ph-cell">TN</td><td class="ph-cell">FP</td></tr><tr><th>True abnormal</th><td class="ph-cell">FN</td><td class="ph-cell">TP</td></tr></table>
    <p class="note">Placeholder: counts will come from the new read-only confusion-matrix field (MLflow artifacts).</p></div>`;

  function playhead() {
    const a = document.querySelector('[data-play]'); if (!a) return;
    let t0 = null, run = false, el = 0;
    a.addEventListener('click', () => { run = !run; a.textContent = run ? 'Pause' : 'Play'; t0 = null; if (run) requestAnimationFrame(step); });
    function step(ts) {
      if (!run) return;
      if (t0 == null) t0 = ts - el * 1000;
      el = ((ts - t0) / 1000) % D.file.dur; const f = el / D.file.dur;
      document.querySelectorAll('[data-ph]').forEach(l => {
        if (l.tagName === 'line') { const x = 44 + f * (900 - 60); l.setAttribute('x1', x); l.setAttribute('x2', x); } else l.style.left = (f * 100) + '%';
      });
      const c = document.querySelector('[data-time]'); if (c) c.textContent = el.toFixed(1) + ' s / ' + D.file.dur.toFixed(2) + ' s';
      requestAnimationFrame(step);
    }
  }

  function tabs() {
    document.querySelectorAll('[data-tab]').forEach(b => b.addEventListener('click', () => {
      document.querySelectorAll('[data-tab]').forEach(x => x.classList.toggle('on', x === b));
      document.querySelectorAll('[data-view]').forEach(v => v.hidden = v.dataset.view !== b.dataset.tab);
    }));
  }
  return { badge, scoreBar, reliability, distTag, links, waveform, grouped, trainTest, heatmap, confusion, playhead, tabs, pct };
})();
