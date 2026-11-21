# Frontend redesign — verification (2026-11-21, benchmark v3)

| Check | Result | File |
|---|---|---|
| Python tests (pipeline + API, incl. confusion matrices, /samples) | 38/38 pass | `pytest tests/model tests/api` |
| Golden snapshot v3 vs live API (8 wavs x 5 models) | 39/40 OK. 1 FAIL: `pascal_normal_296_…_A1.wav` svm_linear score 0.516088 vs 0.516091 (3e-6 > 1e-6 tol); label, threshold and 4-dp display identical. Not adjusted. | `golden_api_table.md` |
| Vitest component tests | 36/36 | `vitest.txt` |
| Playwright e2e on docker stack (:3000) | 12/12 tests, 280/280 checks | `e2e_report.md` |
| Accessibility (axe, WCAG 2.1 A/AA, 4 tabs, light + dark) | 0 violations | `axe_light.json`, `axe_dark.json` |
| Screenshots (desktop + phone, light + dark, print) | saved | `../figures/ui/` |

## Commands
```
docker compose build audio-engine react-frontend && docker compose up -d     # app :3000, API :8000, MLflow :5000
.venv/Scripts/python.exe -m pytest tests/model tests/api -q                     # backend
cd frontend && npx vitest run                                                   # components
cd frontend && npx playwright test -c playwright.monitor.config.ts             # e2e + axe against the running stack
```

## Assumptions / not verified
- Demo samples (PhysioNet a0238 normal, a0296 abnormal; labels from the official REFERENCE.csv) live in git-ignored `samples/`; licence for redistribution not confirmed, so no WAV is committed.
- 2 kHz WAVs are re-encoded x4 (sample repeat) in the browser for listening only; the analysed file is the original.
- Distance-to-threshold cut-offs: 1.0 (SVM decision score), 0.15 (probability), in `frontend/src/lib/display.ts`.
- The old `e2e/docker_v2.spec.ts` checks `benchmark_version = v2` tags and the old `golden.spec.ts` uses the v1 fixture; both predate v3 and were not updated here.
- Old UI kept as `frontend/src/LegacyApp.tsx` (+ old components/tests) until you decide to remove it.
