import { expect, test } from '@playwright/test';
import { mkdirSync, readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const testDir = dirname(fileURLToPath(import.meta.url));
const projectRoot = resolve(testDir, '../..');
const golden = JSON.parse(
  readFileSync(resolve(projectRoot, 'tests/fixtures/golden/current_predictions.json'), 'utf-8'),
);
const modelKeys = ['svm_linear', 'svm_rbf', 'decision_tree', 'cnn_custom'] as const;
const probabilityModels = new Set(['decision_tree', 'cnn_custom']);
const displayPrecision = 4;

test('React UI predictions match golden WAV snapshots, MLflow and history', async ({ page, request }) => {
  const report: { file: string; model: string; expected: string; displayed: string; ok: boolean }[] = [];
  const createdRunIds: string[] = [];
  const screenshotDir = resolve(testDir, 'screenshots');
  mkdirSync(screenshotDir, { recursive: true });

  await page.goto('/');
  await expect(page.getByText('API and models ready')).toBeVisible({ timeout: 180_000 });
  await expect(page.getByText(/Educational project only/)).toBeVisible();

  for (let fileIndex = 0; fileIndex < golden.files.length; fileIndex++) {
    const entry = golden.files[fileIndex];
    const apiResponsePromise = page.waitForResponse(
      response => response.url().endsWith('/analyze_pcg') && response.request().method() === 'POST',
    );
    await page.locator('input[type=file]').setInputFiles(resolve(projectRoot, entry.file));
    await page.getByRole('button', { name: /analyze recording/i }).click();
    const apiResponse = await apiResponsePromise;
    expect(apiResponse.ok()).toBeTruthy();

    const apiBody = await apiResponse.json();
    createdRunIds.push(apiBody.mlflow_run_id);
    await expect(page.getByText('ANALYSIS COMPLETE')).toBeVisible();

    for (const modelKey of modelKeys) {
      const expected = entry.models[modelKey];
      const card = page.locator(`[data-model-key="${modelKey}"]`);
      await expect(card).toBeVisible();

      const prediction = (await card.locator('.prediction').innerText()).trim();
      const scoreText = (await card.locator('.score-line strong').innerText()).trim();
      const actualScore = Number(scoreText);
      const expectedScore = probabilityModels.has(modelKey)
        ? expected.probability_abnormal
        : expected.score;
      const expectedType = probabilityModels.has(modelKey)
        ? 'Probability of abnormal'
        : 'Decision score';
      const scoreMatches = actualScore === Number(expectedScore.toFixed(displayPrecision));
      const labelMatches = prediction.toLowerCase() === expected.prediction_label.toLowerCase();
      const displayed = `${prediction}/${scoreText}`;
      const expectedText = `${expected.prediction_label}/${expectedScore.toFixed(displayPrecision)}`;

      report.push({ file: entry.file, model: modelKey, expected: expectedText, displayed, ok: labelMatches && scoreMatches });
      expect(labelMatches, `${entry.file} ${modelKey} prediction`).toBeTruthy();
      expect(scoreMatches, `${entry.file} ${modelKey} displayed score`).toBeTruthy();
      await expect(card.locator('.score-line span')).toHaveText(expectedType);
      await expect(card.locator('.metric-grid')).toContainText(expected.threshold.toFixed(2));
    }

    const mlflowResponse = await request.get(
      `http://127.0.0.1:5001/api/2.0/mlflow/runs/get?run_id=${apiBody.mlflow_run_id}`,
    );
    expect(mlflowResponse.ok()).toBeTruthy();
    const run = (await mlflowResponse.json()).run;
    for (const result of apiBody.results) {
      const expectedMetrics = [
        ['prediction', result.prediction],
        ['confidence_or_score', result.confidence_or_score],
        ['score', result.score],
        ['threshold', result.threshold],
        ['latency_ms', result.latency_ms],
      ] as const;
      for (const [metricName, responseValue] of expectedMetrics) {
        const metric = run.data.metrics.find(
          (item: { key: string }) => item.key === `${result.model_key}.${metricName}`,
        );
        expect(metric, `${result.model_key} ${metricName} metric exists`).toBeTruthy();
        expect(metric.value).toBeCloseTo(responseValue, 5);
        if (metricName === 'score') {
          expect(Number(metric.value).toFixed(displayPrecision)).toBe(
            Number(result.score).toFixed(displayPrecision),
          );
        }
      }
    }

    if (fileIndex === 0) {
      const link = page.getByRole('link', { name: /open this analysis in mlflow/i });
      await expect(link).toHaveAttribute('href', new RegExp(apiBody.mlflow_run_id));
      const [runPage] = await Promise.all([
        page.context().waitForEvent('page'),
        link.click(),
      ]);
      await runPage.waitForLoadState('domcontentloaded');
      expect(runPage.url()).toContain(apiBody.mlflow_run_id);
      await runPage.close();
      await page.screenshot({ path: resolve(screenshotDir, 'analyze.png'), fullPage: true });
    }
  }

  await page.getByRole('button', { name: 'Models' }).click();
  await expect(page.getByRole('heading', { name: 'Models and inputs' })).toBeVisible();
  await expect(page.getByText('Accuracy').first()).toBeVisible();
  await page.screenshot({ path: resolve(screenshotDir, 'models.png'), fullPage: true });

  await page.getByRole('button', { name: 'History' }).click();
  await expect(page.getByRole('heading', { name: 'History' })).toBeVisible();
  const historyRows = page.locator('tbody tr');
  await expect(historyRows).toHaveCount(golden.files.length);
  for (let rowIndex = 0; rowIndex < golden.files.length; rowIndex++) {
    const expectedEntry = golden.files[golden.files.length - 1 - rowIndex];
    const row = historyRows.nth(rowIndex);
    const shownFilename = (await row.locator('td').nth(1).innerText()).trim();
    expect(shownFilename).toBe(expectedEntry.file.split('/').pop());
    for (let modelIndex = 0; modelIndex < modelKeys.length; modelIndex++) {
      const key = modelKeys[modelIndex];
      const displayedLabel = (await row.locator('td').nth(modelIndex + 2).innerText()).trim();
      const expectedLabel = expectedEntry.models[key].prediction === 1 ? 'Abnormal' : 'Normal';
      expect(displayedLabel).toBe(expectedLabel);
    }
  }
  await page.screenshot({ path: resolve(screenshotDir, 'history.png'), fullPage: true });

  const [historyRunPage] = await Promise.all([
    page.context().waitForEvent('page'),
    historyRows.first().click(),
  ]);
  await historyRunPage.waitForLoadState('domcontentloaded');
  expect(historyRunPage.url()).toContain(createdRunIds[createdRunIds.length - 1]);
  await historyRunPage.close();

  console.log('File | Model | Expected label/score | Displayed | Status');
  for (const row of report) {
    console.log(`${row.file} | ${row.model} | ${row.expected} | ${row.displayed} | ${row.ok ? 'OK' : 'FAIL'}`);
  }
});
