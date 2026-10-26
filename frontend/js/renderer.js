/**
 * Top-down intersection renderer. It draws queues and events it receives and never decides traffic behavior.
 * Vehicles sit in the lane and slot the simulator assigned them, and animations show arrivals and departures.
 */

const CAR_COLORS = ['#4f8ef7', '#22c55e', '#a855f7', '#f59e0b', '#ef4444', '#06b6d4', '#f97316', '#84cc16', '#ec4899', '#8b5cf6'];
const DIRECTIONS = ['north', 'south', 'east', 'west'];
const DIRECTION_UPPER = { north: 'N', south: 'S', east: 'E', west: 'W' };
const LANE_WIDTH = 26;
const CAR_SPACING = 22;
const STOP_LINE_GAP = 18;
const MAX_QUEUE_SLOTS_PER_LANE = 8;
const MAX_ANIMATIONS = 72;
const DEFAULT_ROAD = { lanes: { north: 1, south: 1, east: 1, west: 1 }, name: 'Single-lane four-way intersection' };
// The unrotated car points up; these angles turn its nose along the direction each approach travels.
const CAR_ANGLES = { north: Math.PI, south: 0, east: -Math.PI / 2, west: Math.PI / 2 };
const PHASE_LABELS = ['NORTH-SOUTH GREEN', 'EAST-WEST GREEN', 'YELLOW', 'ALL RED'];
const PHASE_COLORS = ['#22c55e', '#22c55e', '#f59e0b', '#ef4444'];

class IntersectionRenderer {
  constructor(canvasId, roadConfiguration = null) {
    this.canvas = document.getElementById(canvasId);
    this.context = this.canvas.getContext('2d');
    this.roadConfiguration = roadConfiguration || DEFAULT_ROAD;
    this.queues = { north: 0, south: 0, east: 0, west: 0 };
    this.queueVehicles = { north: [], south: [], east: [], west: [] };
    this.phase = 0;
    this.simulationTime = 0;
    this.totalDeparted = 0;
    this.animations = [];
    this.colorIndex = 0;
    this.playbackSpeed = 10;
    this.layout = null;
    this.resizeCanvas();
    this.resizeObserver = 'ResizeObserver' in window ? new ResizeObserver(() => this.resizeCanvas()) : null;
    if (this.resizeObserver) this.resizeObserver.observe(this.canvas);
    else window.addEventListener('resize', () => this.resizeCanvas());
    this.lastFrameTime = performance.now();
    this.animationFrame = null;
    this.drawFrame = this.drawFrame.bind(this);
    this.start();
  }

  // Layout: each approach owns half of its road, so lane counts can differ between approaches.

  resizeCanvas() {
    const bounds = this.canvas.getBoundingClientRect();
    const width = Math.max(1, bounds.width || 400);
    const height = Math.max(1, bounds.height || width);
    const pixelRatio = Math.max(1, window.devicePixelRatio || 1);
    const pixelWidth = Math.round(width * pixelRatio);
    const pixelHeight = Math.round(height * pixelRatio);
    if (this.canvas.width !== pixelWidth || this.canvas.height !== pixelHeight) {
      this.canvas.width = pixelWidth;
      this.canvas.height = pixelHeight;
    }
    this.context.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
    this.layout = computeLayout(width, height, this.roadConfiguration);
  }

  setRoadConfiguration(roadConfiguration) {
    this.roadConfiguration = roadConfiguration || DEFAULT_ROAD;
    this.resizeCanvas();
  }

  setPlaybackSpeed(speed) {
    this.playbackSpeed = Math.min(50, Math.max(1, Number(speed) || 10));
  }

  update(data) {
    if (data.queues) this.queues = data.queues;
    if (data.queueVehicles) this.queueVehicles = data.queueVehicles;
    if (data.phase !== undefined) this.phase = data.phase;
    if (data.time !== undefined) this.simulationTime = data.time;
    if (data.departed) this.totalDeparted += Number(data.departed) || 0;
  }

  /** Slide a new vehicle from the road edge to the queue slot the simulator gave it. */
  animateVehicleEntry(direction, vehicleIdentifier) {
    if (this.animations.length >= MAX_ANIMATIONS || !this.layout.laneCenters[direction]) return;
    const slot = queueSlot(this.queueVehicles[direction] || [], vehicleIdentifier);
    if (!slot) return;
    const target = this.slotPosition(direction, slot.laneIndex, slot.position);
    const edge = this.layout.edgeStart(direction, target);
    this.animations.push({ direction, startX: edge.x, startY: edge.y, endX: target.x, endY: target.y, progress: 0, color: this.nextColor() });
  }

  /** Drive a served vehicle straight through the intersection and out the opposite arm. */
  animateVehicleExit(direction, laneIndex = 0) {
    if (this.animations.length >= MAX_ANIMATIONS || !this.layout.laneCenters[direction]) return;
    const centers = this.layout.laneCenters[direction];
    const lane = centers[Math.min(laneIndex, centers.length - 1)];
    const { left, right, top, bottom, width, height } = this.layout;
    const paths = {
      north: [lane, top, lane, height], south: [lane, bottom, lane, 0],
      east: [right, lane, 0, lane], west: [left, lane, width, lane],
    };
    const [startX, startY, endX, endY] = paths[direction];
    this.animations.push({ direction, startX, startY, endX, endY, progress: 0, color: '#e2e8f0' });
  }

  reset() {
    this.queues = { north: 0, south: 0, east: 0, west: 0 };
    this.queueVehicles = { north: [], south: [], east: [], west: [] };
    this.phase = 0;
    this.simulationTime = 0;
    this.totalDeparted = 0;
    this.animations = [];
    this.colorIndex = 0;
  }

  nextColor() {
    return CAR_COLORS[this.colorIndex++ % CAR_COLORS.length];
  }

  slotPosition(direction, laneIndex, position) {
    const { left, right, top, bottom, laneCenters } = this.layout;
    const lane = laneCenters[direction][Math.min(laneIndex, laneCenters[direction].length - 1)];
    const offset = STOP_LINE_GAP + position * CAR_SPACING;
    if (direction === 'north') return { x: lane, y: top - offset };
    if (direction === 'south') return { x: lane, y: bottom + offset };
    if (direction === 'east') return { x: right + offset, y: lane };
    return { x: left - offset, y: lane };
  }

  start() {
    if (!this.animationFrame) this.animationFrame = requestAnimationFrame(this.drawFrame);
  }

  stop() {
    if (this.animationFrame) cancelAnimationFrame(this.animationFrame);
    this.animationFrame = null;
    if (this.resizeObserver) this.resizeObserver.disconnect();
  }

  drawFrame(timestamp) {
    const elapsedMilliseconds = Math.min(timestamp - this.lastFrameTime, 50);
    this.lastFrameTime = timestamp;
    this.draw(elapsedMilliseconds);
    this.animationFrame = requestAnimationFrame(this.drawFrame);
  }

  draw(elapsedMilliseconds) {
    const { width, height } = this.layout;
    this.context.clearRect(0, 0, width, height);
    this.drawBackground();
    this.drawRoads();
    this.drawLaneMarkings();
    this.drawTrafficLights();
    this.drawQueues();
    this.drawAnimations(elapsedMilliseconds);
    this.drawOverlay();
  }

  drawBackground() {
    const { width, height } = this.layout;
    this.context.fillStyle = '#0c0e18';
    this.context.fillRect(0, 0, width, height);
  }

  drawRoads() {
    const context = this.context;
    const { width, height, left, right, top, bottom } = this.layout;
    context.fillStyle = '#1e2235';
    context.fillRect(left, 0, right - left, height);
    context.fillRect(0, top, width, bottom - top);
    context.fillStyle = '#242843';
    context.fillRect(left, top, right - left, bottom - top);
    context.strokeStyle = '#ffffff';
    context.lineWidth = 2.5;
    context.beginPath();
    context.moveTo(left, top); context.lineTo(right, top);
    context.moveTo(left, bottom); context.lineTo(right, bottom);
    context.moveTo(left, top); context.lineTo(left, bottom);
    context.moveTo(right, top); context.lineTo(right, bottom);
    context.stroke();
  }

  drawLaneMarkings() {
    const context = this.context;
    const { width, height, left, right, top, bottom, laneCenters, lanes } = this.layout;
    context.strokeStyle = 'rgba(255,255,255,0.25)';
    context.lineWidth = 1.5;
    context.setLineDash([14, 14]);
    const dividers = (centers, drawLine) => {
      for (let index = 0; index < centers.length - 1; index++) drawLine((centers[index] + centers[index + 1]) / 2);
    };
    dividers(laneCenters.north, x => this.line(x, 0, x, top));
    dividers(laneCenters.south, x => this.line(x, bottom, x, height));
    dividers(laneCenters.east, y => this.line(right, y, width, y));
    dividers(laneCenters.west, y => this.line(0, y, left, y));
    context.setLineDash([]);
    context.strokeStyle = 'rgba(255,255,220,0.3)';
    context.lineWidth = 2;
    const centerX = left + lanes.north * LANE_WIDTH;
    const centerY = top + lanes.east * LANE_WIDTH;
    this.line(centerX, 0, centerX, top);
    this.line(centerX, bottom, centerX, height);
    this.line(0, centerY, left, centerY);
    this.line(right, centerY, width, centerY);
  }

  line(startX, startY, endX, endY) {
    this.context.beginPath();
    this.context.moveTo(startX, startY);
    this.context.lineTo(endX, endY);
    this.context.stroke();
  }

  lightColor(direction) {
    if (this.phase === 2) return '#f59e0b';
    const served = this.phase === 0 ? ['north', 'south'] : this.phase === 1 ? ['east', 'west'] : [];
    return served.includes(direction) ? '#22c55e' : '#ef4444';
  }

  drawTrafficLights() {
    const context = this.context;
    const { left, right, top, bottom } = this.layout;
    const offset = 14;
    const positions = {
      north: [left - offset, top - offset], south: [right + offset, bottom + offset],
      east: [right + offset, top - offset], west: [left - offset, bottom + offset],
    };
    for (const [direction, [x, y]] of Object.entries(positions)) {
      const color = this.lightColor(direction);
      context.fillStyle = '#0a0c15';
      context.beginPath(); context.roundRect(x - 8, y - 8, 16, 16, 4); context.fill();
      context.beginPath(); context.arc(x, y, 5.5, 0, Math.PI * 2);
      context.fillStyle = color;
      context.fill();
    }
  }

  drawQueues() {
    for (const direction of DIRECTIONS) {
      const vehicles = this.queueVehicles[direction] || [];
      const perLane = {};
      for (const vehicle of vehicles) {
        const laneIndex = vehicle.lane_index || 0;
        const position = perLane[laneIndex] = (perLane[laneIndex] ?? -1) + 1;
        if (position >= MAX_QUEUE_SLOTS_PER_LANE) continue;
        const point = this.slotPosition(direction, laneIndex, position);
        this.drawCar(point.x, point.y, direction, queueColor(vehicle.vehicle_id));
      }
      const hidden = (this.queues[direction] || 0) - Object.values(perLane).reduce((total, count) => total + Math.min(count + 1, MAX_QUEUE_SLOTS_PER_LANE), 0);
      if (hidden > 0) this.drawOverflowLabel(direction, hidden);
    }
  }

  drawOverflowLabel(direction, hidden) {
    const point = this.slotPosition(direction, 0, MAX_QUEUE_SLOTS_PER_LANE);
    this.context.fillStyle = 'rgba(226,232,240,0.85)';
    this.context.font = '11px monospace';
    this.context.textAlign = 'center';
    this.context.fillText(`+${hidden}`, Math.min(Math.max(point.x, 14), this.layout.width - 14), Math.min(Math.max(point.y, 14), this.layout.height - 14));
  }

  /** Draw a top-down car that points along the direction of travel of its approach. */
  drawCar(x, y, direction, color) {
    const context = this.context;
    const carWidth = 12;
    const carLength = 20;
    context.save();
    context.translate(x, y);
    context.rotate(CAR_ANGLES[direction] || 0);
    context.fillStyle = color;
    context.beginPath(); context.roundRect(-carWidth / 2, -carLength / 2, carWidth, carLength, 3); context.fill();
    context.fillStyle = 'rgba(160,220,255,0.65)';
    context.fillRect(-carWidth / 2 + 2, -carLength / 2 + 4, carWidth - 4, 4);
    context.fillStyle = 'rgba(0,0,0,0.35)';
    context.fillRect(-carWidth / 2 + 1.5, -carLength / 2 + 8, carWidth - 3, 6);
    context.fillStyle = '#fef3c7';
    context.fillRect(-carWidth / 2 + 1, -carLength / 2, 3, 2.5);
    context.fillRect(carWidth / 2 - 4, -carLength / 2, 3, 2.5);
    context.fillStyle = '#ff5252';
    context.fillRect(-carWidth / 2 + 1, carLength / 2 - 2.5, 3, 2.5);
    context.fillRect(carWidth / 2 - 4, carLength / 2 - 2.5, 3, 2.5);
    context.restore();
  }

  drawAnimations(elapsedMilliseconds) {
    // Progress is measured in simulated seconds, so a movement lasts one simulated second at any speed.
    const progressStep = elapsedMilliseconds * this.playbackSpeed / 1000;
    for (let index = this.animations.length - 1; index >= 0; index--) {
      const animation = this.animations[index];
      animation.progress += progressStep;
      if (animation.progress >= 1) { this.animations.splice(index, 1); continue; }
      const eased = easeInOut(animation.progress);
      this.context.save();
      this.context.globalAlpha = 0.85;
      this.drawCar(animation.startX + (animation.endX - animation.startX) * eased,
                   animation.startY + (animation.endY - animation.startY) * eased, animation.direction, animation.color);
      this.context.restore();
    }
  }

  drawOverlay() {
    const context = this.context;
    const { width, height, centerX, centerY } = this.layout;
    context.fillStyle = 'rgba(9,11,19,0.7)';
    context.beginPath(); context.roundRect(8, height - 32, 150, 24, 6); context.fill();
    context.beginPath(); context.roundRect(width - 68, height - 32, 60, 24, 6); context.fill();
    context.font = '11px monospace';
    context.textAlign = 'left';
    context.fillStyle = PHASE_COLORS[this.phase] || '#ffffff';
    context.fillText(PHASE_LABELS[this.phase] || '', 14, height - 16);
    context.textAlign = 'right';
    context.fillStyle = '#94a3b8';
    context.fillText(`t=${this.simulationTime.toFixed(0)}s`, width - 12, height - 16);
    context.fillStyle = 'rgba(148,163,184,0.8)';
    context.font = 'bold 13px monospace';
    context.textAlign = 'center';
    context.fillText('N', centerX, 16);
    context.fillText('S', centerX, height - 38);
    context.textAlign = 'left'; context.fillText('W', 4, centerY + 5);
    context.textAlign = 'right'; context.fillText('E', width - 4, centerY + 5);
  }
}

function computeLayout(width, height, roadConfiguration) {
  const configured = (roadConfiguration && roadConfiguration.lanes) || {};
  const lanes = Object.fromEntries(DIRECTIONS.map(direction => [direction, Math.max(1, Math.floor(Number(configured[direction]) || 1))]));
  const centerX = width / 2;
  const centerY = height / 2;
  const left = centerX - (lanes.north + lanes.south) * LANE_WIDTH / 2;
  const right = centerX + (lanes.north + lanes.south) * LANE_WIDTH / 2;
  const top = centerY - (lanes.east + lanes.west) * LANE_WIDTH / 2;
  const bottom = centerY + (lanes.east + lanes.west) * LANE_WIDTH / 2;
  const spread = (start, offset, count) => Array.from({ length: count }, (_, index) => start + (offset + index + 0.5) * LANE_WIDTH);
  // Traffic keeps to the right: southbound (north approach) uses the west half of the road, and so on.
  const laneCenters = {
    north: spread(left, 0, lanes.north), south: spread(left, lanes.north, lanes.south),
    east: spread(top, 0, lanes.east), west: spread(top, lanes.east, lanes.west),
  };
  const edgeStart = (direction, target) => ({
    north: { x: target.x, y: 0 }, south: { x: target.x, y: height }, east: { x: width, y: target.y }, west: { x: 0, y: target.y },
  })[direction];
  return { width, height, centerX, centerY, left, right, top, bottom, lanes, laneCenters, edgeStart };
}

/** Return a vehicle's lane and its position among the vehicles waiting in that lane, or null if it is not queued. */
function queueSlot(vehicles, vehicleIdentifier) {
  const target = vehicles.find(vehicle => vehicle.vehicle_id === vehicleIdentifier);
  if (!target) return null;
  const laneIndex = target.lane_index || 0;
  const position = vehicles.filter(vehicle => (vehicle.lane_index || 0) === laneIndex && vehicle.vehicle_id < vehicleIdentifier).length;
  return { laneIndex, position: Math.min(position, MAX_QUEUE_SLOTS_PER_LANE - 1) };
}

function queueColor(vehicleIdentifier) {
  const palette = ['#3d5fa0', '#2d4a78', '#4a6fa5', '#384e75', '#2a4060', '#4f6d9e'];
  return palette[Math.abs(Number(vehicleIdentifier) || 0) % palette.length];
}

function easeInOut(progress) {
  return progress < 0.5 ? 4 * progress ** 3 : 1 - Math.pow(-2 * progress + 2, 3) / 2;
}

if (typeof module !== 'undefined') module.exports = { computeLayout, queueSlot, easeInOut };
