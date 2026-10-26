/** Live queue charts and the final comparison cards. It only displays numbers the server measured. */

const QUEUE_DIRECTIONS = ['north', 'south', 'east', 'west'];
const LIVE_CHART_POINTS = 120;
const FIXED_TIME_COLOR = '#64748b';
const LEARNED_COLOR = '#4f8ef7';

class MetricsDashboard {
  constructor() {
    this.series = { fixed_time: [], learned: [] };
    this.liveCharts = {
      fixed_time: this.createLiveChart('fixed-time-chart', '#94a3b8'),
      learned: this.createLiveChart('learned-chart', '#7eb3ff'),
    };
    this.comparisonCanvas = document.getElementById('comparison-chart');
    this.comparisonContext = this.comparisonCanvas.getContext('2d');
  }

  createLiveChart(canvasId, color) {
    const canvas = document.getElementById(canvasId);
    return { canvas, context: canvas.getContext('2d'), color };
  }

  /** Append the total queue of one controller to its rolling live chart. */
  addStepData(controller, queues) {
    const chart = this.liveCharts[controller];
    if (!chart) return;
    const total = QUEUE_DIRECTIONS.reduce((sum, direction) => sum + (Number((queues || {})[direction]) || 0), 0);
    const values = this.series[controller];
    values.push(total);
    if (values.length > LIVE_CHART_POINTS) values.shift();
    this.drawLiveChart(chart, values);
  }

  drawLiveChart(chart, values) {
    const { width, height } = prepareCanvas(chart.canvas, chart.context);
    chart.context.clearRect(0, 0, width, height);
    if (values.length < 2) return;
    const maximum = Math.max(5, ...values);
    chart.context.strokeStyle = chart.color;
    chart.context.lineWidth = 2;
    chart.context.beginPath();
    values.forEach((value, index) => {
      const x = index / (values.length - 1) * width;
      const y = height - value / maximum * (height - 4) - 2;
      if (index === 0) chart.context.moveTo(x, y);
      else chart.context.lineTo(x, y);
    });
    chart.context.stroke();
  }

  showComparison(fixedTime, learned) {
    const fixed = fixedTime || {};
    const policy = learned || {};
    const definitions = [
      { key: 'average_wait_experienced', label: 'Average wait experienced', unit: 's', direction: 'lower' },
      { key: 'total_departed', label: 'Vehicles served', unit: 'vehicles', direction: 'higher' },
      { key: 'average_queue_length', label: 'Average queue', unit: 'vehicles', direction: 'lower' },
      { key: 'signal_switches', label: 'Signal changes', unit: 'changes', direction: 'neutral' },
    ];
    const grid = document.getElementById('metrics-grid');
    grid.replaceChildren();
    for (const definition of definitions) {
      const fixedValue = finiteNumber(fixed[definition.key]);
      const learnedValue = finiteNumber(policy[definition.key]);
      const comparison = compareValues(fixedValue, learnedValue, definition.direction);
      const card = document.createElement('article');
      card.className = `metric-card${comparison.outcome ? ` ${comparison.outcome}` : ''}`;
      card.innerHTML = `
        <div class="metric-name">${definition.label}</div>
        <div class="metric-row"><span class="metric-source">Fixed-time</span>
          <span class="metric-value fixed-time-value">${formatMetric(fixedValue)} <small>${definition.unit}</small></span></div>
        <div class="metric-row"><span class="metric-source">Learned policy</span>
          <span class="metric-value learned-value">${formatMetric(learnedValue)} <small>${definition.unit}</small></span></div>
        <div class="metric-improvement ${comparison.className}">${comparison.text}</div>`;
      grid.append(card);
    }
    document.getElementById('comparison-summary').textContent = waitSummary(
      finiteNumber(fixed.average_wait_experienced), finiteNumber(policy.average_wait_experienced));
    this.drawComparisonChart(fixed, policy, definitions);
  }

  drawComparisonChart(fixed, policy, definitions) {
    const context = this.comparisonContext;
    const { width, height } = prepareCanvas(this.comparisonCanvas, context);
    context.clearRect(0, 0, width, height);
    const left = 34, right = 20, top = 38, bottom = 38;
    const chartHeight = Math.max(20, height - top - bottom);
    const groupWidth = Math.max(20, width - left - right) / definitions.length;
    const barWidth = Math.max(5, Math.min(26, groupWidth * 0.26));
    context.textAlign = 'center';
    definitions.forEach((definition, index) => {
      const fixedValue = finiteNumber(fixed[definition.key]);
      const learnedValue = finiteNumber(policy[definition.key]);
      const scaleMaximum = Math.max(fixedValue, learnedValue, 1);
      const centerX = left + groupWidth * (index + 0.5);
      const baseline = top + chartHeight;
      for (const [value, x, color] of [[fixedValue, centerX - barWidth - 2, FIXED_TIME_COLOR], [learnedValue, centerX + 2, LEARNED_COLOR]]) {
        const barHeight = value / scaleMaximum * chartHeight;
        context.fillStyle = color;
        context.fillRect(x, baseline - barHeight, barWidth, barHeight);
        context.fillStyle = '#cbd5e1';
        context.font = '9px ui-monospace, monospace';
        context.fillText(compactMetric(value), x + barWidth / 2, Math.max(11, baseline - barHeight - 4));
      }
      context.fillStyle = '#94a3b8';
      context.font = '10px system-ui, sans-serif';
      context.fillText(shortLabel(definition.label), centerX, height - 11);
    });
    context.textAlign = 'left';
    context.font = '10px system-ui, sans-serif';
    for (const [x, color, label] of [[width - 160, FIXED_TIME_COLOR, 'Fixed-time'], [width - 82, LEARNED_COLOR, 'Learned policy']]) {
      context.fillStyle = color; context.fillRect(x, 8, 10, 10);
      context.fillStyle = '#cbd5e1'; context.fillText(label, x + 15, 17);
    }
  }

  reset() {
    this.series = { fixed_time: [], learned: [] };
    const targets = [...Object.values(this.liveCharts), { canvas: this.comparisonCanvas, context: this.comparisonContext }];
    for (const { canvas, context } of targets) {
      const { width, height } = prepareCanvas(canvas, context);
      context.clearRect(0, 0, width, height);
    }
  }
}

/** Match a canvas's pixel buffer to its CSS size and the screen density so lines stay sharp. */
function prepareCanvas(canvas, context) {
  const bounds = canvas.getBoundingClientRect();
  const cssWidth = Math.max(1, bounds.width || canvas.width);
  const cssHeight = Math.max(1, bounds.height || canvas.height);
  const pixelRatio = Math.max(1, window.devicePixelRatio || 1);
  const pixelWidth = Math.round(cssWidth * pixelRatio);
  const pixelHeight = Math.round(cssHeight * pixelRatio);
  if (canvas.width !== pixelWidth || canvas.height !== pixelHeight) {
    canvas.width = pixelWidth;
    canvas.height = pixelHeight;
  }
  context.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
  return { width: cssWidth, height: cssHeight };
}

function finiteNumber(value) {
  const number = Number(value);
  return Number.isFinite(number) ? number : 0;
}

function formatMetric(value) {
  return value.toLocaleString('en-US', { maximumFractionDigits: 1, minimumFractionDigits: 0 });
}

function compactMetric(value) {
  return value >= 1000 ? `${(value / 1000).toFixed(1)}k` : value.toFixed(value < 10 && value % 1 ? 1 : 0);
}

function shortLabel(label) {
  return { 'Average wait experienced': 'Wait', 'Vehicles served': 'Served', 'Average queue': 'Queue', 'Signal changes': 'Changes' }[label] || label;
}

/** Describe the learned value relative to fixed-time and mark whether the change is an improvement. */
function compareValues(fixedValue, learnedValue, direction) {
  if (direction === 'neutral') return { outcome: '', className: 'improvement-neutral', text: 'Behavior measure' };
  if (fixedValue === learnedValue) return { outcome: '', className: 'improvement-neutral', text: 'No change' };
  if (fixedValue === 0) {
    const better = direction === 'higher' && learnedValue > 0;
    return { outcome: better ? 'better' : 'worse', className: better ? 'improvement-better' : 'improvement-worse',
             text: better ? 'Increase from zero' : 'No zero-baseline percentage' };
  }
  const better = direction === 'lower' ? learnedValue < fixedValue : learnedValue > fixedValue;
  const percentage = Math.abs((learnedValue - fixedValue) / Math.abs(fixedValue) * 100);
  return { outcome: better ? 'better' : 'worse', className: better ? 'improvement-better' : 'improvement-worse',
           text: `${percentage.toFixed(1)}% ${learnedValue < fixedValue ? 'lower' : 'higher'}` };
}

/** Write the headline sentence in the direction the numbers actually moved, never as a negative reduction. */
function waitSummary(fixedWait, learnedWait) {
  if (fixedWait === learnedWait) return `Both controllers had the same experienced wait: ${formatMetric(learnedWait)} seconds.`;
  if (fixedWait === 0) return `Fixed-time experienced wait was zero seconds; the learned policy's was ${formatMetric(learnedWait)} seconds.`;
  const percentage = Math.abs((learnedWait - fixedWait) / fixedWait * 100);
  return `The learned policy's experienced wait was ${percentage.toFixed(1)}% ${learnedWait < fixedWait ? 'lower' : 'higher'} than fixed-time (${formatMetric(learnedWait)} s versus ${formatMetric(fixedWait)} s).`;
}

if (typeof module !== 'undefined') module.exports = { compareValues, waitSummary, formatMetric, compactMetric };
