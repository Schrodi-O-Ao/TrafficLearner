document.addEventListener('DOMContentLoaded', () => {
  // On Render the page and API share one origin; GitHub Pages sets window.TRAFFIC_API_BASE in config.js.
  const apiBase = (window.TRAFFIC_API_BASE || window.location.origin).replace(/\/$/, '') + '/';
  const apiUrl = path => new URL(path.replace(/^\//, ''), apiBase).toString();
  const webSocketUrl = path => {
    const url = new URL(path.replace(/^\//, ''), apiBase);
    url.protocol = url.protocol === 'https:' ? 'wss:' : 'ws:';
    return url.toString();
  };
  const byId = id => document.getElementById(id);
  const CONTROLLERS = ['fixed_time', 'learned'];
  const CONTROLLER_PREFIX = { fixed_time: 'fixed-time', learned: 'learned' };
  const PHASE_NAMES = ['North-south green', 'East-west green', 'Yellow', 'All red'];
  const MAX_UPLOAD_BYTES = 2 * 1024 * 1024;
  const MAX_LOG_ENTRIES = 60;

  const scenarioSelect = byId('scenario-select');
  const uploadInput = byId('csv-upload');
  const durationSlider = byId('episode-length');
  const speedSlider = byId('playback-speed');
  const startButton = byId('start-button');
  const errorMessage = byId('scenario-error');
  const progressBar = byId('progress-bar');
  const progressTrack = byId('progress-track');
  const progressLabel = byId('progress-label');
  const statusText = byId('simulation-status');
  const statusDot = byId('status-dot');
  const logArea = byId('log-area');

  const renderers = { fixed_time: new IntersectionRenderer('fixed-time-canvas'), learned: new IntersectionRenderer('learned-canvas') };
  const metrics = new MetricsDashboard();
  let activeSocket = null;
  let runInProgress = false;
  let comparisonFinished = false;

  enableRevealOnScroll();
  durationSlider.addEventListener('input', () => { byId('duration-value').textContent = durationSlider.value; });
  speedSlider.addEventListener('input', () => { byId('speed-value').textContent = speedSlider.value; });
  startButton.addEventListener('click', createAndRunScenario);
  byId('run-again-button').addEventListener('click', resetForAnotherRun);
  loadScenarios();

  async function loadScenarios() {
    scenarioSelect.replaceChildren(new Option('Choose a preset...', ''));
    try {
      const scenarios = await requestJson('api/scenarios');
      for (const scenario of scenarios) {
        const option = new Option(formatName(scenario.name), scenario.name);
        if (scenario.description) option.title = scenario.description;
        scenarioSelect.add(option);
      }
      if (scenarios.length > 0) scenarioSelect.value = scenarios[0].name;
    } catch (error) {
      showError(`Could not load the scenario list. ${error.message}`);
    }
  }

  async function createAndRunScenario() {
    if (runInProgress) return;
    clearError();
    startButton.disabled = true;
    setStatus('Preparing the selected traffic scenario...', 'running');
    const duration = Number.parseInt(durationSlider.value, 10);
    const speed = Number.parseFloat(speedSlider.value);
    const selectedFile = uploadInput.files[0];
    try {
      let response;
      let scenarioName;
      if (selectedFile) {
        if (selectedFile.size > MAX_UPLOAD_BYTES) throw new Error('The CSV must be 2 MB or smaller.');
        const formData = new FormData();
        formData.append('file', selectedFile);
        response = await requestJson('api/upload', { method: 'POST', body: formData });
        scenarioName = selectedFile.name;
      } else {
        if (!scenarioSelect.value) throw new Error('Choose a preset or select a CSV file first.');
        response = await requestJson('api/preset', {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ name: scenarioSelect.value, seed: 42, episode_length: duration }),
        });
        scenarioName = formatName(scenarioSelect.value);
      }
      beginPlayback(response.scenario_id, scenarioName, duration, speed);
    } catch (error) {
      showError(error.message);
      setStatus('The scenario could not be started. Check the message above.', 'error');
      startButton.disabled = false;
    }
  }

  function beginPlayback(scenarioId, scenarioName, duration, speed) {
    resetRunDisplay();
    for (const controller of CONTROLLERS) {
      renderers[controller].setPlaybackSpeed(speed);
      byId(`${CONTROLLER_PREFIX[controller]}-scenario-label`).textContent = scenarioName;
    }
    runInProgress = true;
    comparisonFinished = false;
    setStatus('Connected. Preparing the two controller runs...', 'running');
    const socket = new WebSocket(webSocketUrl(
      `ws/${encodeURIComponent(scenarioId)}?speed=${encodeURIComponent(speed)}&episode_length=${encodeURIComponent(duration)}`));
    activeSocket = socket;
    socket.addEventListener('open', () => addLog('Connected to the simulation service.'));
    socket.addEventListener('error', () => addLog('The live connection reported an error.', 'error'));
    socket.addEventListener('close', () => {
      if (runInProgress && !comparisonFinished) {
        runInProgress = false;
        startButton.disabled = false;
        setStatus('The connection ended before the comparison finished. You can try again.', 'error');
        addLog('Connection closed before results arrived.', 'error');
      }
    });
    socket.addEventListener('message', event => handleServerMessage(event, duration, scenarioName));
  }

  function handleServerMessage(event, duration, scenarioName) {
    let data;
    try {
      data = JSON.parse(event.data);
    } catch {
      addLog('Received a message that was not valid JSON.', 'error');
      return;
    }
    switch (data.type) {
      case 'road_config':
        for (const controller of CONTROLLERS) {
          renderers[controller].setRoadConfiguration(data.road_config);
          byId(`${CONTROLLER_PREFIX[controller]}-scenario-label`).textContent = data.scenario_name ? formatName(data.scenario_name) : scenarioName;
        }
        break;
      case 'sim_start':
        addLog(`Paired playback is starting for ${scenarioName}.`, 'sim-start');
        setStatus('Both controllers are advancing through the same arrival schedule.', 'running');
        break;
      case 'tick': {
        const timestamp = Number(data.t) || 0;
        for (const controller of CONTROLLERS) updateControllerDisplay(controller, data.controllers && data.controllers[controller], timestamp);
        setProgress(timestamp / duration * 100);
        statusText.textContent = `Simulation time ${timestamp.toFixed(0)} of ${duration} seconds.`;
        break;
      }
      case 'comparison_ready':
        metrics.showComparison(data.fixed_time, data.learned);
        setProgress(100);
        setStatus('Both runs are complete. Compare their measures below.', 'done');
        addLog('The comparison results are ready.', 'sim-done');
        comparisonFinished = true;
        runInProgress = false;
        startButton.disabled = false;
        if (activeSocket && activeSocket.readyState === WebSocket.OPEN) activeSocket.close(1000, 'Comparison complete');
        byId('results').scrollIntoView({ block: 'start', behavior: scrollBehavior() });
        break;
      case 'error':
        showError(data.message || 'The server could not run this scenario.');
        setStatus('The server reported an error.', 'error');
        addLog(data.message || 'The server reported an error.', 'error');
        runInProgress = false;
        startButton.disabled = false;
        break;
      default:
        // Unknown message types are ignored so a new backend message does not break the page.
        break;
    }
  }

  function updateControllerDisplay(controller, snapshot, timestamp) {
    if (!snapshot) return;
    const renderer = renderers[controller];
    renderer.update({ queues: snapshot.queues, queueVehicles: snapshot.queue_vehicles, phase: snapshot.phase, time: timestamp, departed: snapshot.departed });
    for (const vehicleEvent of snapshot.events || []) {
      if (vehicleEvent.type === 'vehicle_entered') renderer.animateVehicleEntry(vehicleEvent.direction, vehicleEvent.vehicle_id);
      else if (vehicleEvent.type === 'vehicle_exited') renderer.animateVehicleExit(vehicleEvent.direction, vehicleEvent.lane_index);
    }
    metrics.addStepData(controller, snapshot.queues);
    const statistics = byId(`${CONTROLLER_PREFIX[controller]}-statistics`);
    const queues = snapshot.queues;
    statistics.querySelector('.time-value').textContent = timestamp.toFixed(0);
    statistics.querySelector('.queue-value').textContent = `North ${queues.north} South ${queues.south} East ${queues.east} West ${queues.west}`;
    statistics.querySelector('.served-value').textContent = renderer.totalDeparted.toLocaleString();
    statistics.querySelector('.phase-value').textContent = PHASE_NAMES[Number(snapshot.phase)] || 'Unknown';
  }

  async function requestJson(path, options = {}) {
    const response = await fetch(apiUrl(path), options);
    const responseText = await response.text();
    let payload = {};
    try {
      payload = responseText ? JSON.parse(responseText) : {};
    } catch {
      payload = { detail: responseText };
    }
    if (!response.ok) throw new Error(payload.detail || `Request failed with status ${response.status}.`);
    return payload;
  }

  function setStatus(message, state = '') {
    statusText.textContent = message;
    statusDot.className = `status-dot${state ? ` ${state}` : ''}`;
  }

  function setProgress(value) {
    const percent = Math.max(0, Math.min(100, Number(value) || 0));
    progressBar.style.width = `${percent}%`;
    progressTrack.setAttribute('aria-valuenow', String(Math.round(percent)));
    progressLabel.textContent = `${Math.round(percent)}%`;
  }

  function addLog(message, className = 'info') {
    const entry = document.createElement('p');
    entry.className = `log-entry ${className}`;
    entry.textContent = `${new Date().toLocaleTimeString()} - ${message}`;
    logArea.append(entry);
    while (logArea.children.length > MAX_LOG_ENTRIES) logArea.firstElementChild.remove();
    logArea.scrollTop = logArea.scrollHeight;
  }

  function showError(message) {
    errorMessage.textContent = message;
    errorMessage.classList.remove('hidden');
  }

  function clearError() {
    errorMessage.textContent = '';
    errorMessage.classList.add('hidden');
  }

  function resetRunDisplay() {
    for (const controller of CONTROLLERS) {
      renderers[controller].reset();
      const statistics = byId(`${CONTROLLER_PREFIX[controller]}-statistics`);
      statistics.querySelector('.time-value').textContent = '0';
      statistics.querySelector('.queue-value').textContent = 'North 0 South 0 East 0 West 0';
      statistics.querySelector('.served-value').textContent = '0';
      statistics.querySelector('.phase-value').textContent = PHASE_NAMES[0];
    }
    metrics.reset();
    logArea.replaceChildren();
    setProgress(0);
    byId('comparison-summary').textContent = 'Run a comparison to see how the two controllers performed.';
    byId('metrics-grid').innerHTML = '<p class="empty-state">The comparison metrics will appear here after a run.</p>';
  }

  function resetForAnotherRun() {
    if (activeSocket && activeSocket.readyState < WebSocket.CLOSING) activeSocket.close(1000, 'Reset for another run');
    activeSocket = null;
    comparisonFinished = false;
    runInProgress = false;
    startButton.disabled = false;
    clearError();
    resetRunDisplay();
    setStatus('Choose a scenario and run the comparison.');
    byId('experiment').scrollIntoView({ block: 'start', behavior: scrollBehavior() });
  }

  function scrollBehavior() {
    return window.matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth';
  }

  function formatName(value) {
    return String(value || '').replace(/_/g, ' ').replace(/\b\w/g, character => character.toUpperCase());
  }

  function enableRevealOnScroll() {
    const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    const elements = document.querySelectorAll('.reveal, .section:not(.hero-section) .section-heading');
    if (reducedMotion || !('IntersectionObserver' in window)) {
      elements.forEach(element => element.classList.add('is-visible'));
      return;
    }
    document.documentElement.classList.add('js-motion');
    const observer = new IntersectionObserver((entries, currentObserver) => {
      for (const entry of entries) {
        if (entry.isIntersecting) {
          entry.target.classList.add('is-visible');
          currentObserver.unobserve(entry.target);
        }
      }
    }, { threshold: 0.12, rootMargin: '0px 0px -5% 0px' });
    elements.forEach(element => observer.observe(element));
  }
});
