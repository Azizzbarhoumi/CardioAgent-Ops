import { defineConfig } from '@playwright/test';
// New monitor UI against the RUNNING docker stack:  npx playwright test -c playwright.monitor.config.ts
export default defineConfig({
  testDir: './e2e', testMatch: /monitor_v3\.spec\.ts/, timeout: 600_000, expect: { timeout: 60_000 }, workers: 1, reporter: 'list',
  use: { baseURL: process.env.APP_URL || 'http://localhost:3000', channel: 'msedge', headless: true, trace: 'off', viewport: { width: 1440, height: 900 },
    launchOptions: { args: ['--autoplay-policy=no-user-gesture-required'] } },
});
