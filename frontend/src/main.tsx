import React from 'react';
import { createRoot } from 'react-dom/client';
import App from './App';
// legacy styles.css is kept for LegacyApp; the new UI imports monitor/monitor.css

createRoot(document.getElementById('root')!).render(<React.StrictMode><App /></React.StrictMode>);
