// Adapted from the server-67 PRX 8802 template, SHA-256 898b6993ff8e40e7a6a9eb93c2344acabd37e90507d97d8b619f4cc57a871198.
const $ = selector => document.querySelector(selector);
const fmt = value => value != null && Number.isFinite(Number(value)) ? Number(value).toLocaleString("ko-KR") : "—";
const fixed = (value, places = 4) => value != null && Number.isFinite(Number(value)) ? Number(value).toFixed(places) : "—";
const elapsed = seconds => {
  const value = Math.max(0, Math.floor(Number(seconds) || 0));
  if (value < 60) return value + "초";
  if (value < 3600) return Math.floor(value / 60) + "분 " + String(value % 60).padStart(2, "0") + "초";
  if (value < 86400) return Math.floor(value / 3600) + "시간 " + Math.floor(value % 3600 / 60) + "분";
  return Math.floor(value / 86400) + "일 " + Math.floor(value % 86400 / 3600) + "시간";
};
const percent = value => Math.max(0, Math.min(100, Number(value) || 0));
const stamp = value => value != null && Number.isFinite(Number(value))
  ? new Date(Number(value) * 1000).toLocaleString("ko-KR", {
      timeZone: "Asia/Seoul", year: "numeric", month: "2-digit", day: "2-digit",
      hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false
    }) + " KST"
  : "—";
const make = (tag, className, value) => {
  const element = document.createElement(tag);
  if (className) element.className = className;
  if (value != null) element.textContent = String(value);
  return element;
};
const set = (selector, value) => { $(selector).textContent = value; };
const empty = (parent, message) => { parent.replaceChildren(make("div", "empty", message)); };
let beforeFirstRun = true, chosenStep = null, sampleSteps = [], currentHistory = [], currentGpuHistory = null;
let lightboxItems = [], lightboxIndex = 0;
function pills(items) {
  const target = $("#runConfig");
  target.replaceChildren();
  for (const item of items) {
    target.append(make("span", "configPill" + (item[1] ? " " + item[1] : ""), item[0]));
  }
}

function drawChart(rawItems) {
  const canvas = $("#chart");
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  if (!width || !height) return;
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.round(width * ratio);
  canvas.height = Math.round(height * ratio);
  const ctx = canvas.getContext("2d");
  ctx.scale(ratio, ratio);
  ctx.clearRect(0, 0, width, height);
  const pad = {left: 82, right: 22, top: 28, bottom: 38};
  const items = (Array.isArray(rawItems) ? rawItems : []).map(item => ({
    step: Number(item.step), loss: Number(item.loss), ema: Number(item.ema)
  })).filter(item => Number.isFinite(item.step) && Number.isFinite(item.loss) && Number.isFinite(item.ema));
  if (!items.length) {
    ctx.fillStyle = "#9fb0c5";
    ctx.font = "13px ui-monospace";
    ctx.fillText(beforeFirstRun ? "학습 시작 전 · 아직 loss 이력이 없습니다" : "첫 loss 로그를 기다리는 중입니다", pad.left + 10, height / 2);
    return;
  }
  const values = items.flatMap(item => [item.loss, item.ema]);
  const low = values.reduce((a, b) => Math.min(a, b), Infinity);
  const high = values.reduce((a, b) => Math.max(a, b), -Infinity);
  const extent = Math.max(high - low, 0.005);
  const min = low - extent * 0.12;
  const max = high + extent * 0.12;
  const graphHeight = height - pad.top - pad.bottom;
  ctx.font = "12px ui-monospace";
  ctx.textAlign = "right";
  ctx.textBaseline = "middle";
  for (let tick = 0; tick <= 8; tick++) {
    const y = pad.top + graphHeight * tick / 8;
    const value = max - (max - min) * tick / 8;
    ctx.strokeStyle = tick === 0 || tick === 8 ? "#33465a" : "#223244";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(pad.left, y);
    ctx.lineTo(width - pad.right, y);
    ctx.stroke();
    ctx.fillStyle = "#a9bbd0";
    ctx.fillText(value.toFixed(4), pad.left - 10, y);
  }
  const x = position => pad.left + (width - pad.left - pad.right) * (items.length === 1 ? 1 : position / (items.length - 1));
  const y = value => pad.top + graphHeight * (1 - (value - min) / (max - min));
  const latestAverage = items[items.length - 1].ema;
  ctx.save();
  ctx.setLineDash([8, 6]);
  ctx.strokeStyle = "#ffbd5bcc";
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.moveTo(pad.left, y(latestAverage));
  ctx.lineTo(width - pad.right, y(latestAverage));
  ctx.stroke();
  ctx.restore();
  for (const series of [["loss", "#48e7ff"], ["ema", "#ffbd5b"]]) {
    ctx.strokeStyle = series[1];
    ctx.lineWidth = 2;
    ctx.beginPath();
    items.forEach((item, position) => {
      if (position) ctx.lineTo(x(position), y(item[series[0]]));
      else ctx.moveTo(x(position), y(item[series[0]]));
    });
    ctx.stroke();
  }
  ctx.fillStyle = "#91a4bb";
  ctx.font = "11px ui-monospace";
  ctx.textBaseline = "alphabetic";
  ctx.textAlign = "left";
  ctx.fillText("step " + fmt(items[0].step), pad.left, height - 10);
  ctx.textAlign = "right";
  ctx.fillText("step " + fmt(items[items.length - 1].step), width - pad.right, height - 10);
  ctx.textAlign = "left";
  ctx.fillStyle = "#48e7ff";
  ctx.fillRect(pad.left, pad.top - 13, 16, 2);
  ctx.fillStyle = "#c5d2e2";
  ctx.fillText("loss", pad.left + 22, pad.top - 9);
  ctx.fillStyle = "#ffbd5b";
  ctx.fillRect(pad.left + 70, pad.top - 13, 16, 2);
  ctx.fillStyle = "#c5d2e2";
  ctx.fillText("AVG", pad.left + 92, pad.top - 9);
}

function drawGpuHistory(history) {
  const canvas = $("#gpuHistoryChart");
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  if (!width || !height) return;
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.round(width * ratio);
  canvas.height = Math.round(height * ratio);
  const ctx = canvas.getContext("2d");
  ctx.scale(ratio, ratio);
  ctx.clearRect(0, 0, width, height);
  const pad = {left: 58, right: 26, top: 33, bottom: 38};
  const last = Math.max(Date.now() / 1000, Number(history?.last_timestamp) || 0);
  const first = last - 600;
  const raw = Array.isArray(history?.samples) ? history.samples : [];
  const points = raw.map(item => ({
    timestamp: Number(item.timestamp),
    gap: item.gap === true,
    utilization: item.utilization || {}
  })).filter(item => Number.isFinite(item.timestamp) && item.timestamp >= first && item.timestamp <= last);
  if (!points.length) {
    ctx.fillStyle = "#9fb0c5";
    ctx.font = "13px ui-monospace";
    ctx.fillText("GPU 이용률 기록을 기다리는 중입니다", pad.left + 12, height / 2);
    return;
  }
  const span = 600;
  const plotWidth = Math.max(width - pad.left - pad.right, 1);
  const plotHeight = Math.max(height - pad.top - pad.bottom, 1);
  const x = value => pad.left + plotWidth * Math.max(0, Math.min(1, (value - first) / span));
  const y = value => pad.top + plotHeight * (1 - value / 100);
  ctx.font = "11px ui-monospace";
  ctx.textBaseline = "middle";
  for (const tick of [0, 25, 50, 75, 100]) {
    const position = y(tick);
    ctx.strokeStyle = tick === 0 || tick === 100 ? "#34485c" : "#223244";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(pad.left, position);
    ctx.lineTo(width - pad.right, position);
    ctx.stroke();
    ctx.fillStyle = "#91a4bb";
    ctx.textAlign = "right";
    ctx.fillText(tick + "%", pad.left - 9, position);
  }
  const timeLabel = value => new Date(value * 1000).toLocaleString("ko-KR", {
    timeZone: "Asia/Seoul", month: "2-digit", day: "2-digit",
    hour: "2-digit", minute: "2-digit", hour12: false
  });
  ctx.textBaseline = "alphabetic";
  for (let tick = 0; tick <= 4; tick++) {
    const position = pad.left + plotWidth * tick / 4;
    ctx.fillStyle = "#91a4bb";
    ctx.textAlign = tick === 0 ? "left" : tick === 4 ? "right" : "center";
    ctx.fillText(timeLabel(first + span * tick / 4), position, height - 11);
  }
  const gpuIndices = [...new Set(points.flatMap(item => Object.keys(item.utilization)))].sort((a, b) => Number(a) - Number(b));
  const colors = ["#48e7ff", "#ffbd5b", "#ff5ebc", "#b8ff4b"];
  for (const [series, index] of gpuIndices.entries()) {
    ctx.strokeStyle = colors[series % colors.length];
    ctx.lineWidth = 1.8;
    ctx.beginPath();
    let connected = false;
    let lastPoint = null;
    for (const point of points) {
      const utilization = Number(point.utilization[index]);
      if (point.gap || !Number.isFinite(utilization) || utilization < 0 || utilization > 100) {
        connected = false;
        continue;
      }
      if (connected) ctx.lineTo(x(point.timestamp), y(utilization));
      else ctx.moveTo(x(point.timestamp), y(utilization));
      connected = true;
      lastPoint = {timestamp: point.timestamp, utilization};
    }
    ctx.stroke();
    if (lastPoint) {
      ctx.beginPath();
      ctx.arc(x(lastPoint.timestamp), y(lastPoint.utilization), 2.8, 0, Math.PI * 2);
      ctx.fillStyle = colors[series % colors.length];
      ctx.fill();
    }
    ctx.fillStyle = colors[series % colors.length];
    ctx.fillRect(pad.left + series * 90, 12, 15, 3);
    ctx.fillStyle = "#c5d2e2";
    ctx.textAlign = "left";
    ctx.font = "11px ui-monospace";
    ctx.fillText("GPU " + index, pad.left + series * 90 + 20, 18);
  }
}

function meter(value, className) {
  const track = make("div", "meter" + (className ? " " + className : ""));
  const fill = make("i");
  fill.style.width = percent(value) + "%";
  track.append(fill);
  return track;
}

function renderHardware(cpu, gpus) {
  const cpuTarget = $("#cpuList");
  cpuTarget.replaceChildren();
  const cpuCard = make("div", "gpu");
  const cpuTop = make("div", "gpuTop");
  cpuTop.append(make("span", "", "CPU · " + fmt(cpu.logical_cpus) + " THREADS"),
                make("span", "", Number.isFinite(Number(cpu.usage_percent)) ? fixed(cpu.usage_percent, 1) + "%" : "측정 중"));
  const cpuMeters = make("div", "meters");
  cpuMeters.append(meter(cpu.usage_percent), meter(cpu.temperature_c, "mem"));
  const cpuMeta = make("div", "gpuMeta");
  cpuMeta.append(make("span", "", "LOAD " + fixed(cpu.load_average, 2)),
                 make("span", "", cpu.temperature_c == null ? "온도 정보 없음" : fixed(cpu.temperature_c, 1) + "°C"));
  cpuCard.append(cpuTop, cpuMeters, cpuMeta);
  cpuTarget.append(cpuCard);

  const gpuTarget = $("#gpuList");
  gpuTarget.replaceChildren();
  if (!Array.isArray(gpus) || !gpus.length) {
    empty(gpuTarget, "GPU 정보 없음");
    return;
  }
  for (const gpu of gpus) {
    const card = make("div", "gpu");
    const top = make("div", "gpuTop");
    top.append(make("span", "", "GPU " + fmt(gpu.index) + " · " + String(gpu.name || "").replace("NVIDIA GeForce ", "")),
               make("span", "", fixed(gpu.utilization_percent, 0) + "%"));
    const bars = make("div", "meters");
    const memoryPercent = Number(gpu.memory_total_mib) > 0 ? 100 * Number(gpu.memory_used_mib) / Number(gpu.memory_total_mib) : 0;
    bars.append(meter(gpu.utilization_percent), meter(memoryPercent, "mem"));
    const meta = make("div", "gpuMeta");
    meta.append(make("span", "", fmt(gpu.memory_used_mib) + " / " + fmt(gpu.memory_total_mib) + " MiB"),
                make("span", "", fixed(gpu.temperature_c, 0) + "°C · " + fixed(gpu.power_w, 0) + "W"));
    card.append(top, bars, meta);
    gpuTarget.append(card);
  }
}

function renderStepTools() {
  const target = $("#sampleTools");
  target.replaceChildren();
  if (!sampleSteps.length) {
    set("#sampleMeta", beforeFirstRun ? "STEP 0 샘플 대기" : "저장 결과 대기 중");
    return;
  }
  const position = sampleSteps.indexOf(chosenStep);
  const nav = make("div", "stepNav");
  const previous = make("button", "stepArrow", "‹");
  previous.type = "button";
  previous.disabled = position <= 0;
  previous.setAttribute("aria-label", "이전 step");
  previous.onclick = () => chooseStep(sampleSteps[position - 1]);
  const toggle = make("button", "stepToggle");
  toggle.type = "button";
  toggle.setAttribute("aria-expanded", "false");
  toggle.append(make("span", "", chosenStep === 0 ? "STEP 0 · BASELINE" : "STEP " + fmt(chosenStep)), make("span", "chevron", "▾"));
  const next = make("button", "stepArrow", "›");
  next.type = "button";
  next.disabled = position < 0 || position === sampleSteps.length - 1;
  next.setAttribute("aria-label", "다음 step");
  next.onclick = () => chooseStep(sampleSteps[position + 1]);
  nav.append(previous, toggle, next);
  const menu = make("div", "stepMenu");
  for (const step of [...sampleSteps].reverse()) {
    const choice = make("button", "stepOption" + (step === chosenStep ? " active" : ""), step === 0 ? "STEP 0 · BASELINE" : "STEP " + fmt(step));
    choice.type = "button";
    choice.setAttribute("aria-pressed", step === chosenStep ? "true" : "false");
    choice.onclick = () => chooseStep(step);
    menu.append(choice);
  }
  toggle.onclick = event => {
    event.stopPropagation();
    toggle.setAttribute("aria-expanded", menu.classList.toggle("open") ? "true" : "false");
  };
  target.append(nav, menu);
}
function showLightbox(index) {
  if (!lightboxItems.length || index < 0 || index >= lightboxItems.length) return;
  lightboxIndex = index;
  const item = lightboxItems[index];
  const lightbox = $("#lightbox");
  lightbox.classList.add("open");
  lightbox.setAttribute("aria-hidden", "false");
  $("#lightboxImage").src = item.url;
  set("#lightboxCaption", item.label + " · " + (index + 1) + "/" + lightboxItems.length);
  $("#lightboxPrev").disabled = index === 0;
  $("#lightboxNext").disabled = index === lightboxItems.length - 1;
  document.body.style.overflow = "hidden";
  if (item.step != null && item.step !== chosenStep) chooseStep(item.step);
}

function closeLightbox() {
  $("#lightbox").classList.remove("open");
  $("#lightbox").setAttribute("aria-hidden", "true");
  $("#lightboxImage").removeAttribute("src");
  document.body.style.overflow = "";
}

$("#lightboxClose").onclick = closeLightbox;
$("#lightboxImage").onclick = closeLightbox;
$("#lightboxPrev").onclick = () => showLightbox(lightboxIndex - 1);
$("#lightboxNext").onclick = () => showLightbox(lightboxIndex + 1);
$("#lightbox").onclick = event => { if (event.target === $("#lightbox")) closeLightbox(); };
document.addEventListener("keydown", event => {
  if (!$("#lightbox").classList.contains("open")) return;
  if (event.key === "Escape") closeLightbox();
  if (event.key === "ArrowLeft") showLightbox(lightboxIndex - 1);
  if (event.key === "ArrowRight") showLightbox(lightboxIndex + 1);
});
document.addEventListener("click", event => {
  if (!event.target.closest(".sampleTools")) {
    $("#sampleTools .stepMenu")?.classList.remove("open");
    $("#sampleTools .stepToggle")?.setAttribute("aria-expanded", "false");
  }
});

function renderCheckpoint(d, started, running) {
  const last = d.last_checkpoint_at == null ? null : stamp(d.last_checkpoint_at);
  const now = Number(d.server_time) || Date.now() / 1000;
  const nextAt = d.next_checkpoint_at == null ? null : Number(d.next_checkpoint_at);
  const next = Number.isFinite(nextAt) ? stamp(nextAt) : null;
  set("#lastCheckpointLarge", last ? elapsed(now - Number(d.last_checkpoint_at)) + " 전" : "아직 없음");
  set("#lastCheckpointDetail", last
    ? "STEP " + fmt(d.last_checkpoint_step) + " · " + last + " · addon.safetensors 저장"
    : started ? "첫 체크포인트 대기" : "본 학습 체크포인트 없음");
  set("#lastCheckpoint", last
    ? "STEP " + fmt(d.last_checkpoint_step) + " · " + last
    : "본 학습 체크포인트 아직 없음");
  set("#nextCheckpointLarge", next
    ? nextAt > now ? elapsed(nextAt - now) + " 후" : "저장 예정 시각 지남"
    : "학습 대기");
  set("#nextCheckpointDetail", next
    ? next + " · 30분 주기 · addon 가중치 및 Euler 샘플"
    : running && started ? "다음 저장 시각 확인 중" : "본 학습 시작 후 저장 일정 표시");
  set("#nextCheckpoint", next || (running && started ? "일정 계산 중" : "본 학습 대기"));
}

let lastData = null, liveVersion = null;
function tile(item, items, index) {
  const button = make("button", "tile");
  button.type = "button";
  const img = make("img");
  img.loading = "lazy";
  img.src = item.url;
  img.alt = item.label;
  button.append(img, make("span", "tag", item.label));
  button.onclick = () => { lightboxItems = items; showLightbox(index); };
  return button;
}
function chooseStep(step) {
  if (!sampleSteps.includes(step)) return;
  chosenStep = step;
  renderStepTools();
  const target = $("#sampleGrid");
  target.replaceChildren();
  const folder = "/samples/step_" + String(step).padStart(8, "0") + "/";
  const items = [];
  for (const nfe of [1, 3, 5]) {
    for (const tenth of [0, 3, 6]) {
      const label = "NFE " + nfe + " · T " + (tenth / 10).toFixed(1);
      const card = make("article", "sample");
      const row = make("div", "images");
      card.append(make("div", "sampleName", label), row);
      for (const [i, cls] of [207, 360, 387, 974].entries()) {
        const item = {step, label: "Class " + cls + " · " + label, url: folder + "nfe" + nfe + "_t" + String(tenth).padStart(2, "0") + "_" + i + ".png"};
        items.push(item);
        row.append(tile(item, items, items.length - 1));
      }
      target.append(card);
    }
  }
  set("#sampleMeta", "STEP " + fmt(step) + (step === 0 ? " · 학습 전 기준선" : "") + " · Euler · CFG 1.5 · 9조건 × 4장");
}
const klLevels = [["kl4", "e4", "#48e7ff"], ["kl8", "e8", "#ffbd5b"], ["kl16", "e16", "#ff80bb"]];
function drawLevelRates(history) {
  const rows = history.filter(p => Number.isFinite(p.step) && klLevels.every(([key]) => Number.isFinite(p[key])));
  const legend = $("#klLegend");
  legend.replaceChildren();
  for (const [key, label, color] of klLevels) {
    const item = make("span"), line = make("i");
    line.style.background = color;
    item.append(line, make("span", "", label + " " + fixed(rows.at(-1)?.[key], 5)));
    legend.append(item);
  }
  const target = $("#klPlot");
  if (!rows.length) { empty(target, "첫 KL 측정값을 기다리고 있습니다."); return; }
  const width = 620, height = 184, left = 65, right = width - 18, top = 14, bottom = height - 29;
  const peak = Math.max(1e-8, ...rows.map(row => Math.max(...klLevels.map(([key]) => row[key])))) * 1.05;
  const first = rows[0].step, last = rows.at(-1).step;
  const x = step => left + (right-left) * (step-first) / Math.max(1,last-first);
  const y = value => bottom - (bottom-top) * value / peak;
  let svg = '<svg viewBox="0 0 620 184" role="img" aria-label="Optimizer step별 e4, e8, e16 KL rate">';
  for (let i=0; i<=3; i++) {
    const v = peak * i / 3, py = y(v);
    svg += '<path d="M'+left+' '+py+'H'+right+'" stroke="#283a50"/><text x="'+(left-7)+'" y="'+(py+4)+'" text-anchor="end">'+v.toPrecision(2)+'</text>';
  }
  for (const [key,,color] of klLevels) {
    const path = rows.map((row,i) => (i ? 'L':'M') + x(row.step).toFixed(2)+' '+y(row[key]).toFixed(2)).join(' ');
    svg += '<path data-kl="'+key+'" d="'+path+'" fill="none" stroke="'+color+'" stroke-width="2"/><circle cx="'+x(last)+'" cy="'+y(rows.at(-1)[key])+'" r="2.5" fill="'+color+'"/>';
  }
  svg += '<text x="'+left+'" y="179">STEP '+first+'</text>';
  if (last !== first) svg += '<text x="'+right+'" y="179" text-anchor="end">STEP '+last+'</text>';
  target.innerHTML = svg + '</svg>';
}
function render(d) {
  const s = d.status;
  const m = s.latest || {};
  beforeFirstRun = !s.step;
  lastData = d;
  currentHistory = d.history;
  currentGpuHistory = d.gpu_history;
  pills([["ImageNet 256"], ["Frozen SiT-S/2"], ["CNN addon"], ["Euler · CFG 1.5"]]);
  set("#state", s.state);
  set("#step", fmt(s.step));
  set("#stepHint", s.epoch == null ? "첫 업데이트 전" : "Epoch " + fmt(s.epoch) + " · 연속 학습");
  set("#age", s.updated_at ? stamp(s.updated_at) : "STEP 0");
  set("#loss", fixed(m.loss, 5));
  set("#ema", "LOSS AVG " + fixed(currentHistory.at(-1)?.ema, 5));
  set("#lr", s.lr == null ? "—" : String(s.lr));
  set("#globalBatch", fmt(s.batch_size));
  set("#dataMse", fixed(m.data_mse, 6));
  set("#klRate", fixed(m.kl, 6));
  set("#weightedKl", "β × KL " + fixed(Number.isFinite(m.beta) && Number.isFinite(m.kl) ? m.beta * m.kl : null, 6));
  set("#klBeta", m.beta == null ? "0.0001" : fixed(m.beta, 6));
  set("#betaHint", (m.beta == null ? "초기 설정" : "STEP " + fmt(m.step) + " 적용값") + " · s=5,042부터 1");
  set("#liveStep", d.live.length ? "STEP " + fmt(d.live_step) + " · " + fmt(d.live.length) + "장" : "첫 배치 대기");
  drawLevelRates(currentHistory);
  set("#chartRange", currentHistory.length ? "STEP " + fmt(currentHistory[0].step) + " – " + fmt(currentHistory.at(-1).step) : "첫 loss 대기");
  drawChart(currentHistory);
  renderHardware(d.cpu, d.gpus);
  drawGpuHistory(currentGpuHistory);
  set("#gpuHistoryRange", "GPU 1 · 최근 10분");
  renderCheckpoint({...s, server_time:d.server_time}, !beforeFirstRun, s.state === "training");
  const previousNewest = sampleSteps.at(-1);
  const changed = JSON.stringify(sampleSteps) !== JSON.stringify(d.steps);
  sampleSteps = d.steps;
  const newest = sampleSteps.at(-1);
  set("#sampleStep", fmt(newest));
  if (changed && newest != null) {
    if (chosenStep == null || chosenStep === previousNewest) chooseStep(newest);
    else renderStepTools();
  }
  if (d.live_step !== liveVersion) {
    liveVersion = d.live_step;
    const tiles = $("#liveBatch");
    tiles.replaceChildren();
    if (!d.live.length) empty(tiles, "첫 실제 학습 배치를 기다리고 있습니다.");
    const items = d.live.map((url,i) => ({url,label:"Live " + (i+1) + " · STEP " + fmt(d.live_step)}));
    items.forEach((item,i) => tiles.append(tile(item,items,i)));
  }
  set("#footer", "VA-Diffusion · ImageNet 256 · SERVER 67");
  $("#liveDot").classList.remove("off");
  set("#liveText", "연결됨 · 3초 갱신");
}
async function refresh() {
  try {
    const response = await fetch("/api/status", {cache:"no-store", signal:AbortSignal.timeout(5000)});
    if (!response.ok) throw new Error("HTTP " + response.status);
    render(await response.json());
  } catch (e) {
    $("#liveDot").classList.add("off");
    set("#liveText", "연결 실패");
    set("#age", String(e.message));
  } finally { setTimeout(refresh, 3000); }
}
refresh();
window.addEventListener("resize", () => { if (lastData) { drawChart(currentHistory); drawGpuHistory(currentGpuHistory); } });
