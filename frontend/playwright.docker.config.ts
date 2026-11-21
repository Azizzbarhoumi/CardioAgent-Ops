import { defineConfig } from '@playwright/test';

// End-to-end checks against the RUNNING docker compose stack (no servers started here).
//   docker compose up -d --build   then   npx playwright test -c playwright.docker.config.ts
export default defineConfig({
  testDir: './e2e', testMatch: /docker_v2\.spec\.ts/, timeout: 600_000, expect: { timeout: 60_000 }, workers: 1, reporter: 'list',
  use: { baseURL: process.env.APP_URL || 'http://localhost:3000', channel: 'msedge', headless: true, trace: 'retain-on-failure', viewport: { width: 1360, height: 900 } },
});
