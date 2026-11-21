const { chromium } = require('playwright');
const path = require('path');
(async () => {
  const b = await chromium.launch({ channel: 'msedge' });
  const errs = [];
  for (const name of ['E2-monitor-bright']) {
    const url = 'file:///' + path.resolve(__dirname, name + '.html').replace(/\\/g, '/');
    for (const scheme of ['light', 'dark']) {
      for (const [vw, tag] of [[1440, 'desktop'], [390, 'phone']]) {
        const p = await b.newPage({ viewport: { width: vw, height: 900 }, colorScheme: scheme });
        p.on('pageerror', e => errs.push(name + ': ' + e.message));
        await p.goto(url); await p.waitForTimeout(600);
        const over = await p.evaluate(() => document.documentElement.scrollWidth - innerWidth);
        if (over > 0) errs.push(`${name} ${scheme} ${tag}: horizontal overflow ${over}px`);
        await p.screenshot({ path: path.join(__dirname, 'shots', `${name}_${scheme}_${tag}_analyze.png`), fullPage: true });
        if (tag === 'desktop') { await p.click('[data-tab=bm]'); await p.waitForTimeout(200);
          await p.screenshot({ path: path.join(__dirname, 'shots', `${name}_${scheme}_${tag}_benchmark.png`), fullPage: true }); }
        await p.close();
      }
    }
  }
  console.log(errs.length ? errs.join('\n') : 'no errors, no overflow');
  await b.close();
})();
