const { chromium } = require('playwright'); const path = require('path');
(async () => { const b = await chromium.launch({ channel: 'msedge' }); const errs=[];
 const url = 'file:///' + path.resolve(__dirname, 'E3-monitor-pastel.html').split(path.sep).join('/');
 for (const pal of ['blue','red']) for (const [vw,tag] of [[1440,'desktop'],[390,'phone']]) {
  const p = await b.newPage({ viewport:{width:vw,height:900}, colorScheme:'light' }); p.on('pageerror',e=>errs.push(e.message));
  await p.goto(url); await p.evaluate(x=>document.documentElement.dataset.pal=x,pal); await p.waitForTimeout(500);
  const o=await p.evaluate(()=>document.documentElement.scrollWidth-innerWidth); if(o>0)errs.push(`${pal} ${tag} overflow ${o}`);
  await p.screenshot({path:path.join(__dirname,'shots',`E3-${pal}_${tag}_analyze.png`),fullPage:true});
  if(tag==='desktop'){await p.click('[data-tab=bm]');await p.screenshot({path:path.join(__dirname,'shots',`E3-${pal}_${tag}_benchmark.png`),fullPage:true});}
  await p.close(); }
 console.log(errs.join('\n')||'no errors, no overflow'); await b.close(); })();
