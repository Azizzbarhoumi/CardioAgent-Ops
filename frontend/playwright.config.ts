import { defineConfig } from '@playwright/test';
import { mkdtempSync } from 'node:fs';
import { rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const frontendDir=dirname(fileURLToPath(import.meta.url));
const root=resolve(frontendDir,'..');
const py=process.env.PYTHON || resolve(root,'.venv/Scripts/python.exe');
const dataDir=mkdtempSync(join(tmpdir(),'cardio-ui-e2e-')).replaceAll('\\','/');
const database=`sqlite:///${dataDir}/mlflow.db`;
const artifacts=`${dataDir}/artifacts`;
process.on('exit',()=>{try{rmSync(dataDir,{recursive:true,force:true});}catch{/* Browser servers may retain a final file handle briefly. */}});

export default defineConfig({
  testDir:'./e2e', timeout:180_000, expect:{timeout:30_000}, workers:1, reporter:'list',
  use:{baseURL:'http://127.0.0.1:5173',channel:'msedge',headless:true,trace:'retain-on-failure'},
  webServer:[
    {command:`"${py}" -m mlflow server --backend-store-uri "${database}" --artifacts-destination "${artifacts}" --host 127.0.0.1 --port 5001 --workers 1`,cwd:root,url:'http://127.0.0.1:5001/health',reuseExistingServer:false,timeout:120_000,env:{...process.env,MLFLOW_ALLOW_FILE_STORE:'true'}},
    {command:`"${py}" -m uvicorn api.main:app --host 127.0.0.1 --port 8000`,cwd:root,url:'http://127.0.0.1:8000/health',reuseExistingServer:false,timeout:180_000,env:{...process.env,MLFLOW_TRACKING_URI:database,MLFLOW_PUBLIC_URL:'http://127.0.0.1:5001',CORS_ALLOWED_ORIGINS:'http://127.0.0.1:5173',CNN_DECISION_THRESHOLD:process.env.CNN_DECISION_THRESHOLD||'0.5'}},
    {command:'npm run dev -- --host 127.0.0.1',cwd:frontendDir,url:'http://127.0.0.1:5173',reuseExistingServer:false,timeout:60_000,env:{...process.env,VITE_API_URL:'http://127.0.0.1:8000'}},
  ],
});
