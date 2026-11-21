import { expect, test, type Page } from '@playwright/test';
import AxeBuilder from '@axe-core/playwright';
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

// End-to-end checks of the new monitor UI against the RUNNING docker stack (app :3000, API :8000, MLflow :5000).
//   npx playwright test -c playwright.docker.config.ts e2e/monitor_v3.spec.ts
const here = dirname(fileURLToPath(import.meta.url));
const root = resolve(here, '../..');
const goldenDir = resolve(root, 'tests/fixtures/golden_v2');
const golden = JSON.parse(readFileSync(resolve(goldenDir, 'expected.json'), 'utf-8'));
const API = process.env.API_URL || 'http://localhost:8000';
const MLFLOW = process.env.MLFLOW_URL || 'http://localhost:5000';
const KEYS = ['svm_linear', 'svm_rbf', 'decision_tree', 'cnn_custom', 'vgg16_transfer'] as const;
const reportDir = resolve(root, 'docs/report/verification');
const figDir = resolve(root, 'docs/report/figures/ui');
mkdirSync(reportDir, { recursive: true }); mkdirSync(figDir, { recursive: true });

type Row = { check: string; expected: string; got: string; ok: boolean };
const report: Row[] = [];
const row = (check: string, expected: unknown, got: unknown, ok: boolean) => report.push({ check, expected: String(expected), got: String(got), ok });
test.afterAll(() => {
  writeFileSync(resolve(reportDir, 'e2e_report.json'), JSON.stringify(report, null, 2));
  const md = ['| check | expected | got | result |', '|---|---|---|---|', ...report.map(r => `| ${r.check} | ${r.expected} | ${r.got} | ${r.ok ? 'OK' : 'FAIL'} |`)];
  writeFileSync(resolve(reportDir, 'e2e_report.md'), `# Playwright e2e (monitor UI, docker stack)\n\n${report.filter(r => r.ok).length}/${report.length} OK\n\n${md.join('\n')}\n`);
});

async function analyzeFile(page: Page, path: string) {
  const resp = page.waitForResponse(r => r.url().endsWith('/analyze_pcg') && r.request().method() === 'POST', { timeout: 300_000 });
  await page.locator('input[type=file]').setInputFiles(path);
  await page.getByRole('button', { name: /analyze recording/i }).click();
  const body = await (await resp).json();
  await expect(page.getByText('ANALYSIS COMPLETE')).toBeVisible();
  return body;
}

async function cardValues(page: Page, key: string) {
  const card = page.locator(`article.model-card[data-model-key="${key}"]`);
  return {
    card,
    label: (await card.locator('.prediction').innerText()).trim(),
    score: (await card.getByTestId('score').innerText()).trim(),
    thr: (await card.getByTestId('threshold').innerText()).trim(),
  };
}

test('golden wavs: every displayed label / score / threshold matches the v3 snapshot', async ({ page }) => {
  test.setTimeout(1_200_000);
  await page.goto('/');
  await expect(page.getByText('API and models ready')).toBeVisible({ timeout: 300_000 });
  await expect(page.getByRole('note')).toContainText('not a medical device');
  const wavs = readdirSync(resolve(goldenDir, 'wavs')).filter(f => f.endsWith('.wav')).sort();
  for (const wav of wavs) {
    const exp = golden.recordings[wav].models;
    await analyzeFile(page, resolve(goldenDir, 'wavs', wav));
    for (const key of KEYS) {
      const v = await cardValues(page, key);
      const expLabel = exp[key].prediction ? 'Abnormal' : 'Normal';
      const expScore = Number(exp[key].score), expThr = Number(exp[key].threshold);
      row(`${wav} ${key} label`, expLabel, v.label, v.label === expLabel);
      // display has 4 decimals: allow half a display unit plus the snapshot tolerance
      row(`${wav} ${key} score`, expScore.toFixed(4), v.score, Math.abs(Number(v.score) - expScore) <= 5e-5 + 1e-4);
      row(`${wav} ${key} threshold`, expThr.toFixed(4), v.thr, v.thr === expThr.toFixed(4));
      const tableRow = page.locator(`tr[data-row-key="${key}"]`);
      const tScore = (await tableRow.locator('[data-col="score"]').innerText()).trim();
      row(`${wav} ${key} table = card`, v.score, tScore, tScore === v.score);
    }
  }
  expect(report.filter(r => !r.ok), JSON.stringify(report.filter(r => !r.ok), null, 1)).toEqual([]);
});

test('sample buttons analyse the two sample files and the screen equals the API response', async ({ page, request }) => {
  test.setTimeout(600_000);
  const samples = (await (await request.get(`${API}/samples`)).json()).samples;
  row('samples available', 2, samples.length, samples.length === 2);
  await page.goto('/');
  for (const id of ['normal', 'abnormal']) {
    const resp = page.waitForResponse(r => r.url().endsWith('/analyze_pcg') && r.request().method() === 'POST', { timeout: 300_000 });
    await page.getByTestId(`sample-${id}`).click();
    const body = await (await resp).json();
    await expect(page.getByText('ANALYSIS COMPLETE')).toBeVisible();
    for (const r of body.results) {
      const v = await cardValues(page, r.model_key);
      row(`sample ${id} ${r.model_key} label`, r.prediction_label, v.label, v.label === r.prediction_label);
      row(`sample ${id} ${r.model_key} score`, Number(r.score).toFixed(4), v.score, v.score === Number(r.score).toFixed(4));
      row(`sample ${id} ${r.model_key} threshold`, Number(r.threshold).toFixed(4), v.thr, v.thr === Number(r.threshold).toFixed(4));
    }
  }
  expect(report.filter(r => !r.ok && r.check.startsWith('sample')), 'sample rows').toEqual([]);
});

test('every MLflow link points to the intended run', async ({ page, request }) => {
  test.setTimeout(600_000);
  await page.goto('/');
  const body = await analyzeFile(page, resolve(goldenDir, 'wavs', readdirSync(resolve(goldenDir, 'wavs')).filter(f => f.endsWith('.wav')).sort()[0]));
  const runInfo = async (url: string) => {
    const id = url.split('/runs/')[1].split('/')[0];
    const r = await (await request.get(`${MLFLOW}/api/2.0/mlflow/runs/get?run_id=${id}`)).json();
    return Object.fromEntries((r.run?.data?.tags ?? []).map((t: { key: string; value: string }) => [t.key, t.value]));
  };
  for (const r of body.results) {
    const card = page.locator(`article.model-card[data-model-key="${r.model_key}"]`);
    const cv = await card.getByRole('link', { name: /3-fold results in MLflow/ }).getAttribute('href');
    const tvt = await card.getByRole('link', { name: /train vs test/ }).getAttribute('href');
    const an = await card.getByRole('link', { name: /analysis in MLflow/ }).getAttribute('href');
    row(`${r.model_key} 3-fold link = API`, r.benchmark_run_url, cv, cv === r.benchmark_run_url);
    row(`${r.model_key} train/test link = API`, r.benchmark_train_vs_test_url, tvt, tvt === r.benchmark_train_vs_test_url);
    row(`${r.model_key} analysis link = API`, body.mlflow_run_url, an, an === body.mlflow_run_url);
    const tags = await runInfo(cv!);
    row(`${r.model_key} 3-fold run tags`, `model=${r.model_key} v3 selected`, `model=${tags.model} ${tags.benchmark_version} ${tags.selected_for_app}`,
      tags.model === r.model_key && tags.benchmark_version === 'v3' && tags.selected_for_app === 'true');
  }
  const an = await runInfo(body.mlflow_run_url);
  row('analysis run exists', 'run with tags', Object.keys(an).length, Object.keys(an).length > 0);
  // benchmark tab buttons
  const bench = await (await request.get(`${API}/benchmark`)).json();
  await page.getByRole('button', { name: 'Benchmark', exact: true }).click();
  await expect(page.getByTestId('benchmark-table')).toBeVisible();
  for (const m of bench.models) {
    const href = await page.locator(`[data-chart-key="${m.model_key}"] a.cv-button`).getAttribute('href');
    row(`benchmark ${m.model_key} button`, m.benchmark_run_url, href, href === m.benchmark_run_url);
  }
  expect(report.filter(r => !r.ok && /link|run|button/.test(r.check)), 'link rows').toEqual([]);
});

test('playhead moves on waveform and spectrogram while the audio plays', async ({ page }) => {
  test.setTimeout(300_000);
  await page.goto('/');
  await analyzeFile(page, resolve(root, 'samples/physionet_a0296_abnormal.wav'));
  const wave0 = Number(await page.getByTestId('wave-playhead').getAttribute('data-x'));
  const mel0 = Number(await page.getByTestId('mel-playhead').getAttribute('data-left'));
  await page.evaluate(async () => { const a = document.querySelector('audio')!; a.muted = true; await a.play(); });
  await page.waitForTimeout(2500);
  const t = await page.evaluate(() => document.querySelector('audio')!.currentTime);
  const wave1 = Number(await page.getByTestId('wave-playhead').getAttribute('data-x'));
  const mel1 = Number(await page.getByTestId('mel-playhead').getAttribute('data-left'));
  row('audio advanced', '> 1 s', t.toFixed(2), t > 1);
  row('waveform playhead moved', `> ${wave0}`, wave1, wave1 > wave0);
  row('spectrogram playhead moved', `> ${mel0}`, mel1, mel1 > mel0);
  expect(t).toBeGreaterThan(1); expect(wave1).toBeGreaterThan(wave0); expect(mel1).toBeGreaterThan(mel0);
});

for (const scheme of ['light', 'dark'] as const) {
  for (const [w, h, tag] of [[1440, 900, 'desktop'], [820, 1180, 'tablet'], [390, 844, 'phone']] as const) {
    test(`layout ${scheme} ${tag}: no horizontal scroll, no text overflow, screenshots`, async ({ page }) => {
      test.setTimeout(300_000);
      await page.emulateMedia({ colorScheme: scheme });
      await page.setViewportSize({ width: w, height: h });
      await page.goto('/');
      await analyzeFile(page, resolve(root, 'samples/physionet_a0296_abnormal.wav'));
      const cols = await page.evaluate(() => getComputedStyle(document.querySelector('[data-testid="channel-grid"]')!).gridTemplateColumns.split(' ').length);
      const expCols = tag === 'desktop' ? 3 : tag === 'tablet' ? 2 : 1;
      row(`${scheme} ${tag} grid columns`, expCols, cols, cols === expCols);
      for (const name of ['Analyze', 'Benchmark', 'Models']) {
        if (name !== 'Analyze') await page.getByRole('button', { name, exact: true }).click();
        if (name === 'Benchmark') await expect(page.getByTestId('benchmark-table')).toBeVisible({ timeout: 120_000 });
        await page.waitForTimeout(300);
        const over = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
        const clipped = await page.evaluate(() => [...document.querySelectorAll('.ch, .ch *')].filter(e => {
          const el = e as HTMLElement; return el.scrollWidth > el.clientWidth + 1 && getComputedStyle(el).overflowX === 'visible' && el.clientWidth > 0 && !el.closest('.ts');
        }).length);
        row(`${scheme} ${tag} ${name} horizontal scroll`, 0, over, over <= 0);
        row(`${scheme} ${tag} ${name} overflowing elements in cards`, 0, clipped, clipped === 0);
        if (tag !== 'tablet') await page.screenshot({ path: resolve(figDir, `${name.toLowerCase()}_${scheme}_${tag}.png`), fullPage: true });
      }
      if (scheme === 'light' && tag !== 'tablet') {
        await page.getByRole('button', { name: 'Analyze', exact: true }).click();
        await page.emulateMedia({ media: 'print' });
        await page.screenshot({ path: resolve(figDir, `print_report_${tag}.png`), fullPage: true });
        if (tag === 'desktop') await page.pdf({ path: resolve(figDir, 'print_report.pdf'), format: 'A4', printBackground: true });
        await page.emulateMedia({ media: 'screen' });
      }
      expect(report.filter(r => !r.ok && r.check.startsWith(`${scheme} ${tag}`))).toEqual([]);
    });
  }
}

for (const scheme of ['light', 'dark'] as const) {
  test(`accessibility (axe, WCAG 2 A/AA) ${scheme}: no serious or critical violations`, async ({ page }) => {
    test.setTimeout(300_000);
    await page.emulateMedia({ colorScheme: scheme });
    await page.goto('/');
    await analyzeFile(page, resolve(root, 'samples/physionet_a0296_abnormal.wav'));
    const out: { view: string; id: string; impact: string; nodes: number; help: string }[] = [];
    for (const name of ['Analyze', 'Benchmark', 'History', 'Models']) {
      if (name !== 'Analyze') await page.getByRole('button', { name, exact: true }).click();
      if (name === 'Benchmark') await expect(page.getByTestId('benchmark-table')).toBeVisible({ timeout: 120_000 });
      await page.waitForTimeout(400);
      const res = await new AxeBuilder({ page }).withTags(['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa']).analyze();
      for (const v of res.violations) out.push({ view: name, id: v.id, impact: v.impact ?? '', nodes: v.nodes.length, help: v.help });
      const serious = res.violations.filter(v => v.impact === 'serious' || v.impact === 'critical');
      row(`axe ${scheme} ${name} serious/critical`, 0, serious.map(v => `${v.id}(${v.nodes.length})`).join(' ') || 0, serious.length === 0);
      row(`axe ${scheme} ${name} passes`, '>0', res.passes.length, res.passes.length > 0);
    }
    // badges carry an icon AND text
    await page.getByRole('button', { name: 'Analyze', exact: true }).click();
    const badges = await page.locator('.badge').evaluateAll(els => els.map(e => ({ icon: !!e.querySelector('svg'), text: (e.textContent ?? '').trim() })));
    row(`axe ${scheme} badges icon+text`, 'all', badges.filter(b => b.icon && b.text).length + '/' + badges.length, badges.every(b => b.icon && b.text));
    writeFileSync(resolve(reportDir, `axe_${scheme}.json`), JSON.stringify(out, null, 2));
    expect(report.filter(r => !r.ok && r.check.startsWith(`axe ${scheme}`))).toEqual([]);
  });
}
