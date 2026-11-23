'use strict';
const $=id=>document.getElementById(id);
const COLORS=['#48e7ff','#b8ff4b','#ffbd5b','#ff5ebc'];
const fmt=value=>Number.isFinite(Number(value))?Number(value).toPrecision(5):'—';

function lineChart(target,series){
  const points=series.flatMap(item=>item.points).filter(point=>Number.isFinite(point.x)&&Number.isFinite(point.y));
  if(!points.length){target.innerHTML='<div class="empty">기록된 값이 없습니다.</div>';return;}
  const W=720,H=300,L=62,R=18,T=20,B=38;
  let xmin=Math.min(...points.map(p=>p.x)),xmax=Math.max(...points.map(p=>p.x));if(xmin===xmax)xmax=xmin+1;
  let ymin=Math.min(...points.map(p=>p.y)),ymax=Math.max(...points.map(p=>p.y));if(ymin===ymax)ymax=ymin+Math.max(Math.abs(ymin)*.1,1e-6);
  const pad=(ymax-ymin)*.08;ymin=Math.max(0,ymin-pad);ymax+=pad;
  const sx=x=>L+(x-xmin)/(xmax-xmin)*(W-L-R),sy=y=>H-B-(y-ymin)/(ymax-ymin)*(H-T-B);
  let svg=`<svg viewBox="0 0 ${W} ${H}" role="img">`;
  for(let i=0;i<=4;i++){const y=ymin+(ymax-ymin)*i/4,py=sy(y);svg+=`<line x1="${L}" y1="${py}" x2="${W-R}" y2="${py}" stroke="#1e2b3b" stroke-dasharray="4 4"/><text x="${L-8}" y="${py+4}" text-anchor="end">${y.toPrecision(3)}</text>`;}
  series.forEach((item,index)=>{const sorted=item.points.slice().sort((a,b)=>a.x-b.x);svg+=`<polyline points="${sorted.map(p=>`${sx(p.x)},${sy(p.y)}`).join(' ')}" fill="none" stroke="${item.color||COLORS[index]}" stroke-width="2"/>`;});
  svg+='</svg><div class="gpuList">'+series.map((item,index)=>`<span class="gpu" style="color:${item.color||COLORS[index]}">${item.name}</span>`).join('')+'</div>';
  target.innerHTML=svg;
}

function imageCards(target,items,emptyText){
  if(!items.length){target.innerHTML=`<div class="empty">${emptyText}</div>`;return;}
  target.innerHTML=items.map(item=>`<figure class="card"><img src="${item.url}" alt="${item.label}" loading="lazy"><figcaption class="caption">${item.label}</figcaption></figure>`).join('');
}

function render(data){
  const status=data.status||{},latest=status.latest||{};
  $('state').textContent=status.state||'prepared';$('step').textContent=(status.step||0).toLocaleString();
  $('loss').textContent=fmt(latest.loss);$('dataMse').textContent=fmt(latest.data_mse);$('kl').textContent=fmt(latest.kl);$('beta').textContent=fmt(latest.beta);
  const history=data.history?.loss||[];
  lineChart($('lossChart'),[
    {name:'Total',color:COLORS[0],points:history.map(p=>({x:p.step,y:Number(p.loss)}))},
    {name:'Data MSE',color:COLORS[1],points:history.map(p=>({x:p.step,y:Number(p.data_mse)}))},
  ]);
  lineChart($('klChart'),[
    {name:'e4',color:COLORS[0],points:history.map(p=>({x:p.step,y:Number(p.kl_e4)}))},
    {name:'e8',color:COLORS[2],points:history.map(p=>({x:p.step,y:Number(p.kl_e8)}))},
    {name:'e16',color:COLORS[3],points:history.map(p=>({x:p.step,y:Number(p.kl_e16)}))},
  ]);
  imageCards($('liveBatch'),data.live_batch||[],'학습 batch가 아직 없습니다.');
  const groups=new Map();(data.samples||[]).forEach(item=>{const key=`NFE ${item.nfe} · T ${item.temperature}`;if(!groups.has(key))groups.set(key,[]);groups.get(key).push(item);});
  if(!groups.size)$('samples').innerHTML='<div class="empty">생성 샘플이 아직 없습니다.</div>';
  else $('samples').innerHTML=[...groups].map(([name,items])=>`<section class="sampleGroup"><h3>${name}</h3><div class="sampleGrid">${items.map(item=>`<figure class="card"><img src="${item.url}" alt="${item.label}" loading="lazy"><figcaption class="caption">${item.label}</figcaption></figure>`).join('')}</div></section>`).join('');
  $('gpus').innerHTML=(data.gpus||[]).map(gpu=>`<span class="gpu">GPU ${gpu.index} · ${gpu.utilization}% · ${gpu.memory_used}/${gpu.memory_total} MiB</span>`).join('');
}

async function refresh(){
  try{const response=await fetch('/api/status',{cache:'no-store',signal:AbortSignal.timeout(4500)});if(!response.ok)throw new Error(`HTTP ${response.status}`);const data=await response.json();render(data);$('liveDot').classList.remove('off');$('connection').textContent='5초마다 갱신 · '+new Date().toLocaleTimeString('ko-KR',{hour12:false});}
  catch(error){$('liveDot').classList.add('off');$('connection').textContent='상태 연결 안 됨';}
  setTimeout(refresh,5000);
}
refresh();
