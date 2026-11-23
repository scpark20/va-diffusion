"use strict";
let beforeFirstRun = true;
let chosenStep = null, sampleSteps = [], lightboxItems = [], lightboxIndex = 0;
let latest = null, liveShown = -1;
const levels = [["e4","#48e7ff"],["e8","#ffbd5b"],["e16","#ff80bb"]];
const phaseLabel = {training:"학습 중", sampling:"체크포인트 샘플 생성 중", baseline:"STEP 0 샘플 생성 중",
  starting:"학습 시작 중", stopping:"저장 후 중지 중", stopped:"학습 중지", failed:"학습 오류"};

function graph(target, series, range, height, title) {
  const width = 640, left = 49, right = 579, top = 14, bottom = height - 28;
  const [xmin,xmax,peak] = range;
  const sx = x => left+(right-left)*(x-xmin)/Math.max(1,xmax-xmin);
  const sy = y => bottom-(bottom-top)*y/Math.max(1e-8,peak);
  let svg = '<svg viewBox="0 0 '+width+' '+height+'" role="img" aria-label="'+title+'"><title>'+title+'</title>';
  for(let i=0;i<=3;i++){
    const y=peak*i/3, py=sy(y);
    svg+='<path d="M'+left+' '+py+'H'+right+'" stroke="#223144"/><text x="'+(left-7)+'" y="'+(py+4)+'" text-anchor="end">'+Number(y.toPrecision(2))+'</text>';
  }
  const labels=[];
  for(const item of series){
    if(!item.xy.length) continue;
    const path=item.xy.map(([x,y],i)=>(i?"L":"M")+sx(x).toFixed(2)+" "+sy(y).toFixed(2)).join(" ");
    const [x,y]=item.xy.at(-1);
    svg+='<path data-series="'+item.key+'" d="'+path+'" fill="none" stroke="'+item.color+'" stroke-width="'+(item.dash?2:3)+'" '+(item.dash?'stroke-dasharray="6 5"':'')+'/>';
    if(!item.dash){
      svg+='<circle cx="'+sx(x)+'" cy="'+sy(y)+'" r="3.3" fill="'+item.color+'"/>';
      if(item.label)labels.push({text:item.label,py:sy(y),color:item.color});
    }
  }
  labels.sort((a,b)=>a.py-b.py);
  let last=top-14;
  for(const l of labels){l.textY=Math.max(l.py,last+14);last=l.textY;}
  let limit=bottom;
  for(const l of labels.reverse()){
    l.textY=Math.min(limit,l.textY);limit=l.textY-14;
    svg+='<path d="M'+(right+3)+' '+l.py+'L'+(right+14)+' '+l.textY+'" stroke="'+l.color+'"/><text x="'+(right+17)+'" y="'+(l.textY+4)+'" style="fill:'+l.color+'">'+l.text+'</text>';
  }
  svg+='<text x="'+left+'" y="'+(height-6)+'">STEP '+fmt(xmin)+'</text><text x="'+right+'" y="'+(height-6)+'" text-anchor="end">STEP '+fmt(xmax)+'</text></svg>';
  $(target).innerHTML=svg;
}

function renderTrainingPlots(data) {
  const rows=data.history.filter(r=>Number.isFinite(r.step) && levels.every(([k])=>Number.isFinite(r[k])));
  const m=rows.at(-1), keyBox=$("#klKeys");
  keyBox.replaceChildren();
  for(const [key,color] of levels){
    const item=make("b"), mark=make("i");mark.style.background=color;
    item.append(mark,document.createTextNode(key+" "+fixed(m?.[key],4)));keyBox.append(item);
  }
  if(m){
    const peak=Math.max(...rows.map(p=>Math.max(p.e4,p.e8,p.e16)),1e-6);
    graph("#klPlot",levels.map(([key,color])=>({key,label:key,color,xy:rows.map(p=>[p.step,p[key]])})),
          [rows[0].step,m.step,peak*1.12],225,"e4, e8, e16 계층별 KL 추이");
    const biggest=levels.reduce((a,b)=>m[a[0]]>=m[b[0]]?a:b)[0];
    set("#klText","현재 "+biggest+"의 KL이 가장 큽니다. 오른쪽으로 갈수록 최근 학습입니다.");
  } else {
    empty($("#klPlot"),"첫 학습 기록을 기다리고 있습니다.");set("#klText","");
  }
  const cfg=data.config, end=Math.max(cfg.plateau,(m?.step || 1)-1);
  const future=cfg.schedule.slice();if(end>cfg.plateau)future.push([end,1]);
  graph("#betaPlot",[{key:"plan",color:"#8896a8",dash:true,xy:future},
                     {key:"beta",color:"#b8ff4b",xy:rows.map(p=>[p.step-1,p.beta])}],
        [0,end,1.05],130,"β warm-up 예정값과 실제 적용값");
  set("#beta",m?fixed(m.beta,4):"초기 설정 "+cfg.schedule[0][1]);
  set("#betaText",m?.beta===1?"β가 1에 도달했습니다.":"초기값 0.0001에서 시작해 업데이트 전 STEP 5,042에서 1에 도달합니다.");
}

function tile(items, i) {
  const item=items[i], button=make("button","pic"), img=make("img");
  button.type="button";img.src=item.url;img.alt=item.label;img.loading="lazy";
  button.setAttribute("aria-label",item.label+" 확대");
  button.append(img,make("span","",item.label));
  button.onclick=()=>{lightboxItems=items;showLightbox(i);};
  return button;
}

function chooseStep(step) {
  if(!sampleSteps.includes(step))return;
  chosenStep=step;
  renderStepTools();
  const cfg=latest.config, grid=$("#samples"), items=[];
  grid.replaceChildren();
  for(const n of cfg.nfes) for(const t of cfg.temps){
    const card=make("article","condition"),quad=make("div","quad"),str=t.toFixed(1);
    card.append(make("h3","","NFE "+n+" · T "+str),quad);grid.append(card);
    for(const [i,label] of cfg.classes.entries()){
      const name="n"+n+"_t"+String(Math.round(t*10)).padStart(2,"0")+"_"+i+".png";
      items.push({step,url:"/samples/"+step+"/"+name,label:"Class "+label});
      quad.append(tile(items,items.length-1));
    }
  }
  set("#sampleMeta","STEP "+fmt(step)+(step===0?" · 학습 전 기준선":" · 저장한 addon")+" · Euler · CFG 1.5 · 36장");
}

function render(data) {
  latest=data;
  const s=data.status, m=s.latest||{}, cfg=data.config;
  beforeFirstRun=!s.step;
  set("#liveText",phaseLabel[s.phase]||s.phase);$("#liveDot").classList.toggle("off",["failed","stopped"].includes(s.phase));
  pills([["ImageNet 256"],["frozen SiT-S/2"],["GPU 1 · batch "+cfg.batch],["LR "+cfg.lr]]);
  set("#step","STEP "+fmt(s.step||0));set("#loss",fixed(m.loss,5));set("#mse",fixed(m.data_mse,5));set("#totalKl",fixed(m.kl,5));
  const live=data.live, liveStep=live.step??0;
  if(liveStep!==liveShown){
    liveShown=liveStep;const grid=$("#liveGrid");grid.replaceChildren();
    if(!live.count)empty(grid,"첫 optimizer 업데이트 후 실제 입력 4장을 표시합니다.");
    else {
      const items=Array.from({length:live.count},(_,i)=>({label:"입력 "+(i+1),url:"/live/"+liveStep+"/"+i+".png"}));
      items.forEach((item,i)=>grid.append(tile(items,i)));
    }
    set("#liveStep",live.count?"STEP "+fmt(liveStep)+" · 4장":"대기 중");
  }
  renderTrainingPlots(data);
  const newest=sampleSteps.at(-1),changed=String(data.sample_steps)!==String(sampleSteps);
  sampleSteps=data.sample_steps;
  if(changed && sampleSteps.length){
    if(chosenStep==null || chosenStep===newest)chooseStep(sampleSteps.at(-1));
    else renderStepTools();
  }else if(!sampleSteps.length){
    empty($("#samples"),"완료된 샘플을 기다리는 중입니다."); renderStepTools();
  }
  renderCheckpoint({...s, server_time:data.server_time}, !!s.step, s.phase==="training");
  set("#lossRange",data.history.length?"STEP "+fmt(data.history[0].step)+" – "+fmt(data.history.at(-1).step):"첫 측정 대기");
  drawChart(data.history);renderHardware(data.cpu,data.gpus);drawGpuHistory(data.gpu_history);
}

$("#lightboxClose").onclick=closeLightbox;$("#lightboxImage").onclick=closeLightbox;
$("#lightboxPrev").onclick=()=>showLightbox(lightboxIndex-1);$("#lightboxNext").onclick=()=>showLightbox(lightboxIndex+1);
$("#lightbox").onclick=e=>{if(e.target===$("#lightbox"))closeLightbox();};
document.addEventListener("keydown",e=>{
  if(!$("#lightbox").classList.contains("open"))return;
  if(e.key==="Escape")closeLightbox();if(e.key==="ArrowLeft")showLightbox(lightboxIndex-1);if(e.key==="ArrowRight")showLightbox(lightboxIndex+1);
});
document.addEventListener("click",e=>{if(!e.target.closest(".sampleTools")){
  $("#sampleTools .stepMenu")?.classList.remove("open");$("#sampleTools .stepToggle")?.setAttribute("aria-expanded","false");}});
window.addEventListener("resize",()=>{if(latest){drawChart(latest.history);drawGpuHistory(latest.gpu_history);}});
async function refresh(){
  try{
    const response=await fetch("/api/status",{cache:"no-store",signal:AbortSignal.timeout(6000)});
    if(!response.ok)throw new Error("HTTP "+response.status);render(await response.json());
  }catch(e){$("#liveDot").classList.add("off");set("#liveText","상태 연결 실패");}
  finally{setTimeout(refresh,3000);}
}
refresh();
