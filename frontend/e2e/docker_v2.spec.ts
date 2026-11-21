import { expect, test } from '@playwright/test';
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const testDir = dirname(fileURLToPath(import.meta.url));
const root = resolve(testDir, '../..');
const goldenDir = resolve(root, 'tests/fixtures/golden_v2');
const golden = JSON.parse(readFileSync(resolve(goldenDir, 'expected.json'), 'utf-8'));
const shots = resolve(testDir, 'screenshots/docker_v2');
const API = process.env.API_URL || 'http://localhost:8000';
const MLFLOW = process.env.MLFLOW_URL || 'http://localhost:5000';
const KEYS = ['svm_linear', 'svm_rbf', 'decision_tree', 'cnn_custom', 'vgg16_transfer'] as const;
mkdirSync(shots, { recursive: true });

type Row = { check: string; expected: string; got: string; ok: boolean };
const report: Row[] = [];
const row = (check: string, expected: unknown, got: unknown, ok: boolean) => report.push({ check, expected: String(expected), got: String(got), ok });

test.afterAll(() => {
  writeFileSync(resolve(shots, 'report.json'), JSON.stringify(report, null, 2));
});

test('golden wavs: displayed values match the snapshot', async ({ page }) => {
  await page.goto('/');
  await expect(page.getByText('API and models ready')).toBeVisible({ timeout: 300_000 });
  await expect(page.getByText(/Educational project only/)).toBeVisible();
  const wavs = readdirSync(resolve(goldenDir, 'wavs')).filter(f => f.endsWith('.wav')).sort();
  for (const [i, wav] of wavs.entries()) {
    const exp = golden.recordings[wav].models;
    const responsePromise = page.waitForResponse(r => r.url().endsWith('/analyze_pcg') && r.request().method() === 'POST', { timeout: 300_000 });
    await page.locator('input[type=file]').setInputFiles(resolve(goldenDir, 'wavs', wav));
    await page.getByRole('button', { name: /analyze recording/i }).click();
    const body = await (await responsePromise).json();
    await expect(page.getByText('ANALYSIS COMPLETE')).toBeVisible();
    for (const key of KEYS) {
      const card = page.locator(`article.model-card[data-model-key="${key}"]`);
      const label = (await card.locator('.prediction').innerText()).trim();
      const score = (await card.getByTestId('score').innerText()).trim();
      const thr = (await card.getByTestId('threshold').innerText()).trim();
      const expLabel = exp[key].prediction ? 'Abnormal' : 'Normal';
      const expScore = Number(exp[key].score).toFixed(4);
      const scoreOk = Math.abs(Number(score) - Number(exp[key].score)) <= 1e-4 + 5e-5;
      row(`${wav} ${key} label`, expLabel, label, label === expLabel);
      row(`${wav} ${key} score`, expScore, score, scoreOk);
      row(`${wav} ${key} threshold`, Number(exp[key].threshold).toFixed(4), thr, thr === Number(exp[key].threshold).toFixed(4));
      const api = body.results.find((r: { model_key: string }) => r.model_key === key);
      await expect(card.getByRole('link', { name: /3-fold results in MLflow/ })).toHaveAttribute('href', api.benchmark_run_url);
    }
    if (i === 0) await page.screenshot({ path: resolve(shots, '01_results.png'), fullPage: true });
  }
  expect(report.filter(r => !r.ok), JSON.stringify(report.filter(r => !r.ok), null, 1)).toEqual([]);
});

test('every 3-fold button opens the right parent run with 3 child runs and the figures', async ({ page, context, request }) => {
  const bench = await (await request.get(`${API}/benchmark`)).json();
  await page.goto('/');
  await page.getByRole('button', { name: 'Benchmark', exact: true }).click();
  await expect(page.getByTestId('benchmark-table')).toBeVisible();
  await page.screenshot({ path: resolve(shots, '02_benchmark_tab.png'), fullPage: true });
  for (const model of bench.models) {
    const button = page.locator(`[data-chart-key="${model.model_key}"] a.cv-button`);
    const [popup] = await Promise.all([context.waitForEvent('page'), button.click()]);
    await popup.waitForLoadState('domcontentloaded');
    const opened = popup.url();
    row(`${model.model_key} button url`, model.benchmark_run_url, opened, opened === model.benchmark_run_url);
    await expect(popup.getByText(model.model_key, { exact: true }).first()).toBeVisible({ timeout: 60_000 });
    // structure through the MLflow REST API of the same server
    const runId = opened.split('/runs/')[1];
    const parent = await (await request.get(`${MLFLOW}/api/2.0/mlflow/runs/get?run_id=${runId}`)).json();
    const tags = Object.fromEntries(parent.run.data.tags.map((t: { key: string; value: string }) => [t.key, t.value]));
    row(`${model.model_key} parent tags`, 'v2 / selected_for_app=true', `${tags.benchmark_version} / selected_for_app=${tags.selected_for_app}`, tags.benchmark_version === 'v2' && tags.selected_for_app === 'true');
    const kids = await (await request.post(`${MLFLOW}/api/2.0/mlflow/runs/search`, { data: { experiment_ids: [parent.run.info.experiment_id], filter: `tags.mlflow.parentRunId = '${runId}'` } })).json();
    row(`${model.model_key} child runs`, 3, kids.runs?.length ?? 0, (kids.runs?.length ?? 0) === 3);
    const arts = await (await request.get(`${MLFLOW}/api/2.0/mlflow/artifacts/list?run_id=${runId}&path=figures`)).json();
    const files = (arts.files ?? []).map((f: { path: string }) => f.path).sort().join(',');
    row(`${model.model_key} figures`, 'figures/folds_overview.png,figures/train_vs_test.png', files, files === 'figures/folds_overview.png,figures/train_vs_test.png');
    if (model.model_key === 'svm_rbf') {
      await popup.waitForTimeout(3000);
      await popup.screenshot({ path: resolve(shots, '03_mlflow_parent_run.png'), fullPage: true });
      await popup.goto(model.train_vs_test_url);
      await popup.waitForTimeout(4000);
      await popup.screenshot({ path: resolve(shots, '04_mlflow_train_vs_test.png'), fullPage: true });
    }
    await popup.close();
  }
  expect(report.filter(r => !r.ok), JSON.stringify(report.filter(r => !r.ok), null, 1)).toEqual([]);
});
