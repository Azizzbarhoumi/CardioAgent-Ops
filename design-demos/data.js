// Sample values for the drafts. Scores/thresholds from the screenshot case; 3-fold metrics from live /benchmark (2026-11-21).
// Inference times are illustrative placeholders. Confusion matrices are not in the API yet (shown as placeholders).
window.DEMO = {
  file: { name: 'pascal_normal_246_1309199278902_D.wav', dur: 7.42, src: 44100, rs: 2000, total: 812 },
  models: [
    { key:'svm_linear', name:'SVM · linear kernel', short:'SVM linear', group:'svm', type:'decision', score:-2.0044, thr:0, ms:1.8, fold:3,
      m:{recall:[.766,.027],spec:[.820,.023],bal:[.793,.003],f1:[.836,.013]}, trainBal:.804, foldsR:[.795,.730,.772] },
    { key:'svm_rbf', name:'SVM · RBF kernel', short:'SVM RBF', group:'svm', type:'decision', score:-1.0741, thr:0, ms:3.1, fold:2,
      m:{recall:[.855,.005],spec:[.863,.025],bal:[.859,.010],f1:[.897,.002]}, trainBal:.901, foldsR:[.861,.850,.853] },
    { key:'decision_tree', name:'Decision Tree', short:'Tree', group:'tree', type:'prob', score:0.0323, thr:0.84, ms:0.9, fold:3,
      m:{recall:[.824,.053],spec:[.835,.048],bal:[.830,.002],f1:[.874,.023]}, trainBal:.888, foldsR:[.899,.784,.790] },
    { key:'cnn_custom', name:'CNN', short:'CNN', group:'cnn', type:'prob', score:0.3954, thr:0.81, ms:42.6, fold:3,
      m:{recall:[.708,.039],spec:[.963,.024],bal:[.835,.008],f1:[.822,.022]}, trainBal:.839, foldsR:[.763,.680,.680] },
    { key:'vgg16_transfer', name:'Transfer Learning (VGG16)', short:'VGG16', group:'vgg', type:'prob', score:0.1641, thr:0.46, ms:118.4, fold:3,
      m:{recall:[.778,.038],spec:[.873,.034],bal:[.826,.003],f1:[.852,.019]}, trainBal:.845, foldsR:[.816,.726,.793] },
  ],
  // distance-to-threshold rule (item 14 preview): SVM |score| >= 1.0 far; prob |score-thr| >= 0.15 far
  far(m){ return m.type==='decision' ? Math.abs(m.score-m.thr) >= 1.0 : Math.abs(m.score-m.thr) >= 0.15; },
  // marker position 0..100 on the bar. SVM axis linear -3..+3 centered on 0; prob axis 0..1.
  pos(v,type){ return type==='decision' ? Math.max(0,Math.min(100,(v+3)/6*100)) : Math.max(0,Math.min(100,v*100)); },
  wave(n=600){ // synthetic PCG-like envelope for the draft only (real app uses waveform_points)
    const out=[]; for(let i=0;i<n;i++){const t=i/n*7.42, ph=(t%0.82)/0.82;
      const s1=Math.exp(-((ph-0.06)**2)/0.0008), s2=0.7*Math.exp(-((ph-0.42)**2)/0.0006);
      out.push([t,(s1+s2)*Math.sin(i*1.7)*(0.8+0.2*Math.sin(i*.13))+0.03*Math.sin(i*5.1)]);} return out; },
};
