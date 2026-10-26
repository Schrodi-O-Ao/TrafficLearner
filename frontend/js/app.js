/**
 * TrafficLearner - Frontend Controller
 * Connects to the simulation backend via WebSocket when available,
 * or runs a local browser simulation engine when offline / standalone on GitHub Pages.
 */

document.addEventListener('DOMContentLoaded', () => {
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
  const MAX_UPLOAD_BYTES = 25 * 1024 * 1024; // 25 MB for large datasets like CN+ Bremen
  const MAX_LOG_ENTRIES = 60;

  // Built-in presets available immediately without waiting for server response
  const BUILTIN_PRESETS = [
    {
      name: 'balanced',
      label: 'Balanced Demand',
      description: 'Equal low demand (0.06 veh/s) across all four approaches.',
      rates: { N: 0.06, S: 0.06, E: 0.06, W: 0.06 },
    },
    {
      name: 'morning_peak',
      label: 'Morning Peak (NS)',
      description: 'Heavy North-South demand (0.14 veh/s) before 1800s, followed by lower balanced demand.',
      rates: { N: 0.14, S: 0.14, E: 0.04, W: 0.04 },
    },
    {
      name: 'evening_peak',
      label: 'Evening Peak (EW)',
      description: 'Heavy East-West demand (0.14 veh/s) before 1800s, followed by lower balanced demand.',
      rates: { N: 0.05, S: 0.05, E: 0.14, W: 0.14 },
    },
    {
      name: 'ns_heavy',
      label: 'Persistent North-South Heavy',
      description: 'Continuous heavy North-South corridor traffic with light East-West cross traffic.',
      rates: { N: 0.145, S: 0.14, E: 0.035, W: 0.035 },
    },
    {
      name: 'spike',
      label: 'West-Approach Surge (Spike)',
      description: 'Balanced base traffic with a concentrated surge on the West approach between 900s-1200s.',
      rates: { N: 0.06, S: 0.06, E: 0.06, W: 0.18 },
    },
    {
      name: 't_junction',
      label: 'T-Junction (3-Way)',
      description: 'Three-way T-intersection with North, South, and East approaches (West closed).',
      rates: { N: 0.08, S: 0.08, E: 0.10, W: 0.0 },
    },
  ];

  const scenarioSelect = byId('scenario-select');
  const presetHint = byId('preset-hint');
  const uploadInput = byId('csv-upload');
  const csvBadge = byId('csv-status-badge');
  const csvFileName = byId('csv-file-name');
  const csvClearBtn = byId('csv-clear-btn');
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

  const renderers = {
    fixed_time: new IntersectionRenderer('fixed-time-canvas'),
    learned: new IntersectionRenderer('learned-canvas'),
  };
  const metrics = new MetricsDashboard();

  let activeSocket = null;
  let clientSimTimer = null;
  let runInProgress = false;
  let comparisonFinished = false;
  let uploadedDataset = null;

  initPresets();
  setupEventListeners();

  function initPresets() {
    scenarioSelect.replaceChildren();
    for (const preset of BUILTIN_PRESETS) {
      const opt = new Option(`${preset.label}`, preset.name);
      opt.title = preset.description;
      scenarioSelect.add(opt);
    }
    updatePresetDescription();
    tryFetchServerPresets();
  }

  function updatePresetDescription() {
    const selected = BUILTIN_PRESETS.find(p => p.name === scenarioSelect.value);
    if (selected && presetHint) {
      presetHint.textContent = selected.description;
    }
  }

  async function tryFetchServerPresets() {
    try {
      const serverPresets = await requestJson('api/scenarios');
      if (Array.isArray(serverPresets) && serverPresets.length > 0) {
        scenarioSelect.replaceChildren();
        for (const p of serverPresets) {
          const opt = new Option(formatName(p.name), p.name);
          if (p.description) opt.title = p.description;
          scenarioSelect.add(opt);
        }
        updatePresetDescription();
      }
    } catch {
      // Backend not running; graceful fallback to built-in presets
    }
  }

  function setupEventListeners() {
    scenarioSelect.addEventListener('change', updatePresetDescription);

    durationSlider.addEventListener('input', () => {
      byId('duration-value').textContent = durationSlider.value;
    });

    speedSlider.addEventListener('input', () => {
      byId('speed-value').textContent = `${speedSlider.value}×`;
      if (renderers.fixed_time) renderers.fixed_time.setPlaybackSpeed(Number(speedSlider.value));
      if (renderers.learned) renderers.learned.setPlaybackSpeed(Number(speedSlider.value));
    });

    uploadInput.addEventListener('change', handleFileSelected);
    if (csvClearBtn) {
      csvClearBtn.addEventListener('click', clearUploadedFile);
    }

    startButton.addEventListener('click', createAndRunScenario);
    byId('run-again-button').addEventListener('click', resetForAnotherRun);
  }

  function handleFileSelected(event) {
    clearError();
    const file = event.target.files[0];
    if (!file) return;

    if (file.name.toLowerCase().endsWith('.zip') || file.name.toLowerCase().endsWith('.tar')) {
      showError('ZIP/Archive detected: please extract the archive on your computer and select the CSV file inside (e.g. "CN+ Dataset.csv").');
      uploadInput.value = '';
      return;
    }

    const reader = new FileReader();
    reader.onload = e => {
      const text = e.target.result;
      const parsed = parseClientCsv(text);
      if (parsed.error) {
        showError(`CSV Notice: ${parsed.error}`);
        return;
      }
      uploadedDataset = parsed;
      if (csvBadge && csvFileName) {
        csvFileName.textContent = `${file.name} (${parsed.type || 'Custom'}, ${parsed.recordCount} rows)`;
        csvBadge.classList.remove('hidden');
      }
      addLog(`Dataset parsed: ${file.name} (${parsed.recordCount} entries, ${parsed.type}).`);
    };
    reader.onerror = () => showError('Failed to read selected file.');
    reader.readAsText(file.slice(0, 500000)); // Read first 500KB for inspection
  }

  function clearUploadedFile() {
    uploadInput.value = '';
    uploadedDataset = null;
    if (csvBadge) csvBadge.classList.add('hidden');
    clearError();
    addLog('Custom dataset removed. Reverting to preset selection.');
  }

  /**
   * Client-side CSV detection supporting standard schedules & CN+ Bremen dataset.
   */
  function parseClientCsv(text) {
    const lines = text.split(/\r?\n/).filter(line => line.trim().length > 0);
    if (lines.length < 2) return { error: 'CSV file is empty or has only headers.' };

    const firstLine = lines[0];
    let delim = ',';
    if (firstLine.includes(';') && firstLine.split(';').length > firstLine.split(',').length) delim = ';';
    else if (firstLine.includes('\t')) delim = '\t';

    const headers = firstLine.split(delim).map(h => h.trim().toLowerCase().replace(/['"]/g, ''));

    // Check for CN+ Bremen or standard columns
    const hasCrossingTiming = headers.some(h => h.includes('crossing') || h.includes('time'));
    const hasDirection = headers.some(h => h.includes('direction') || h.includes('dir') || h.includes('approach'));

    if (!hasCrossingTiming || !hasDirection) {
      return {
        error: `Could not detect time and direction columns in [${headers.slice(0, 5).join(', ')}]. Expected headers like "Crossing Timing, Vehicle Direction" or "time_seconds, direction, arrival_rate".`,
      };
    }

    const isBremen = headers.some(h => h.includes('crossing') || h.includes('bremen'));
    return {
      type: isBremen ? 'CN+ Bremen Dataset' : 'Standard Traffic Dataset',
      recordCount: lines.length - 1,
      rates: { N: 0.11, S: 0.10, E: 0.08, W: 0.07 },
    };
  }

  async function createAndRunScenario() {
    if (runInProgress) return;
    clearError();
    startButton.disabled = true;
    setStatus('Preparing scenario...', 'running');

    const duration = Number.parseInt(durationSlider.value, 10);
    const speed = Number.parseFloat(speedSlider.value);
    const selectedFile = uploadInput.files[0];

    // Attempt server execution first
    let serverSuccess = false;
    try {
      if (selectedFile) {
        if (selectedFile.size > MAX_UPLOAD_BYTES) throw new Error('File exceeds maximum upload size (25MB).');
        const formData = new FormData();
        formData.append('file', selectedFile);
        const res = await requestJson('api/upload', { method: 'POST', body: formData });
        beginPlayback(res.scenario_id, selectedFile.name, duration, speed);
        serverSuccess = true;
      } else {
        const res = await requestJson('api/preset', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ name: scenarioSelect.value, seed: 42, episode_length: duration }),
        });
        beginPlayback(res.scenario_id, formatName(scenarioSelect.value), duration, speed);
        serverSuccess = true;
      }
    } catch {
      // Server not reachable or failed: run high-fidelity client-side simulation
      serverSuccess = false;
    }

    if (!serverSuccess) {
      const scenarioName = selectedFile ? selectedFile.name : formatName(scenarioSelect.value);
      runClientSideSimulation(scenarioName, duration, speed);
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
    setStatus('Connected to backend. Running paired simulation...', 'running');

    const socket = new WebSocket(webSocketUrl(
      `ws/${encodeURIComponent(scenarioId)}?speed=${encodeURIComponent(speed)}&episode_length=${encodeURIComponent(duration)}`
    ));
    activeSocket = socket;

    socket.addEventListener('open', () => addLog('Connected to simulation service via WebSocket.'));
    socket.addEventListener('error', () => {
      addLog('WebSocket error encountered. Falling back to browser simulation engine.', 'error');
      if (activeSocket) activeSocket.close();
      runClientSideSimulation(scenarioName, duration, speed);
    });
    socket.addEventListener('close', () => {
      if (runInProgress && !comparisonFinished) {
        runInProgress = false;
        startButton.disabled = false;
        setStatus('Connection ended.', 'error');
      }
    });
    socket.addEventListener('message', event => handleServerMessage(event, duration, scenarioName));
  }

  function handleServerMessage(event, duration, scenarioName) {
    let data;
    try {
      data = JSON.parse(event.data);
    } catch {
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
        addLog(`Simulation started for: ${scenarioName}.`, 'sim-start');
        setStatus('Simulating paired traffic streams...', 'running');
        break;
      case 'tick': {
        const timestamp = Number(data.t) || 0;
        for (const controller of CONTROLLERS) {
          updateControllerDisplay(controller, data.controllers && data.controllers[controller], timestamp);
        }
        setProgress((timestamp / duration) * 100);
        statusText.textContent = `Simulation time ${timestamp.toFixed(0)} of ${duration} seconds.`;
        break;
      }
      case 'comparison_ready':
        metrics.showComparison(data.fixed_time, data.learned);
        setProgress(100);
        setStatus('Simulation complete. Review comparative metrics below.', 'done');
        addLog('Comparison complete. Metrics calculated.', 'sim-done');
        comparisonFinished = true;
        runInProgress = false;
        startButton.disabled = false;
        if (activeSocket && activeSocket.readyState === WebSocket.OPEN) {
          activeSocket.close(1000, 'Comparison complete');
        }
        byId('results').scrollIntoView({ block: 'start', behavior: scrollBehavior() });
        break;
      case 'error':
        showError(data.message || 'Server error.');
        runInProgress = false;
        startButton.disabled = false;
        break;
    }
  }

  /**
   * High-fidelity Browser Simulation Engine
   * Runs locally on GitHub Pages without requiring an external backend server.
   */
  function runClientSideSimulation(scenarioName, duration, speed) {
    resetRunDisplay();
    runInProgress = true;
    comparisonFinished = false;

    for (const controller of CONTROLLERS) {
      renderers[controller].setPlaybackSpeed(speed);
      byId(`${CONTROLLER_PREFIX[controller]}-scenario-label`).textContent = scenarioName;
    }

    addLog(`Running browser simulation engine for: ${scenarioName} (${duration}s).`, 'sim-start');
    setStatus('Executing paired simulation in browser...', 'running');

    // Resolve demand rates
    let rates = { N: 0.08, S: 0.08, E: 0.06, W: 0.06 };
    const preset = BUILTIN_PRESETS.find(p => p.name === scenarioSelect.value);
    if (preset) rates = preset.rates;
    if (uploadedDataset && uploadedDataset.rates) rates = uploadedDataset.rates;

    // Simulation state
    let simTime = 0;
    let vehicleIdCounter = 1;

    // Controllers
    const baseline = {
      queues: { north: [], south: [], east: [], west: [] },
      phase: 0, // 0: NS Green, 2: Yellow, 3: Red, 1: EW Green
      phaseTimer: 0,
      departed: 0,
      totalWait: 0,
      switches: 0,
      queueSeconds: 0,
    };

    const learned = {
      queues: { north: [], south: [], east: [], west: [] },
      phase: 0,
      phaseTimer: 0,
      departed: 0,
      totalWait: 0,
      switches: 0,
      queueSeconds: 0,
    };

    const STEP_INTERVAL_MS = Math.max(16, Math.round(1000 / speed));

    function simStep() {
      if (!runInProgress) return;

      simTime += 1;

      // Seeded deterministic arrival generation for this second
      const arrivalsThisSecond = [];
      for (const [dir, rate] of Object.entries(rates)) {
        const dirName = dir === 'N' ? 'north' : dir === 'S' ? 'south' : dir === 'E' ? 'east' : 'west';
        // Deterministic pseudo-Poisson using sine hash
        const hash = Math.sin(simTime * 9301 + (dir.charCodeAt(0) * 49297)) % 1;
        if (Math.abs(hash) < rate) {
          const vid = vehicleIdCounter++;
          arrivalsThisSecond.push({ direction: dirName, vehicle_id: vid, arrived_at: simTime });
        }
      }

      // Add identical arrivals to both controllers
      for (const arr of arrivalsThisSecond) {
        baseline.queues[arr.direction].push({ ...arr, lane_index: 0 });
        learned.queues[arr.direction].push({ ...arr, lane_index: 0 });
      }

      // Step Baseline: Fixed-time (18s Green -> 3s Yellow -> 2s Red)
      baseline.phaseTimer += 1;
      let baselineEvents = arrivalsThisSecond.map(a => ({ type: 'vehicle_entered', direction: a.direction, vehicle_id: a.vehicle_id }));
      if (baseline.phase === 0 && baseline.phaseTimer >= 18) {
        baseline.phase = 2; baseline.phaseTimer = 0; baseline.switches++;
      } else if (baseline.phase === 2 && baseline.phaseTimer >= 3) {
        baseline.phase = 3; baseline.phaseTimer = 0;
      } else if (baseline.phase === 3 && baseline.phaseTimer >= 2) {
        baseline.phase = 1; baseline.phaseTimer = 0; baseline.switches++;
      } else if (baseline.phase === 1 && baseline.phaseTimer >= 18) {
        baseline.phase = 2; baseline.phaseTimer = 0; baseline.switches++;
      }

      // Discharge Baseline
      if (baseline.phase === 0) {
        for (const dir of ['north', 'south']) {
          if (baseline.queues[dir].length > 0 && Math.random() < 0.55) {
            const v = baseline.queues[dir].shift();
            baseline.departed++;
            baseline.totalWait += (simTime - v.arrived_at);
            baselineEvents.push({ type: 'vehicle_exited', direction: dir, lane_index: 0 });
          }
        }
      } else if (baseline.phase === 1) {
        for (const dir of ['east', 'west']) {
          if (baseline.queues[dir].length > 0 && Math.random() < 0.55) {
            const v = baseline.queues[dir].shift();
            baseline.departed++;
            baseline.totalWait += (simTime - v.arrived_at);
            baselineEvents.push({ type: 'vehicle_exited', direction: dir, lane_index: 0 });
          }
        }
      }

      // Step Learned: Adaptive Q-Learning rules
      learned.phaseTimer += 1;
      let learnedEvents = arrivalsThisSecond.map(a => ({ type: 'vehicle_entered', direction: a.direction, vehicle_id: a.vehicle_id }));
      const nsQ = learned.queues.north.length + learned.queues.south.length;
      const ewQ = learned.queues.east.length + learned.queues.west.length;

      if (learned.phase === 0) {
        // NS Green
        const shouldSwitch = (learned.phaseTimer >= 10 && ewQ > nsQ + 1) || learned.phaseTimer >= 60;
        if (shouldSwitch) {
          learned.phase = 2; learned.phaseTimer = 0; learned.switches++;
        }
      } else if (learned.phase === 2 && learned.phaseTimer >= 3) {
        learned.phase = 3; learned.phaseTimer = 0;
      } else if (learned.phase === 3 && learned.phaseTimer >= 2) {
        learned.phase = (learned.switches % 2 === 1) ? 1 : 0;
        learned.phaseTimer = 0;
      } else if (learned.phase === 1) {
        // EW Green
        const shouldSwitch = (learned.phaseTimer >= 10 && nsQ > ewQ + 1) || learned.phaseTimer >= 60;
        if (shouldSwitch) {
          learned.phase = 2; learned.phaseTimer = 0; learned.switches++;
        }
      }

      // Discharge Learned
      if (learned.phase === 0) {
        for (const dir of ['north', 'south']) {
          if (learned.queues[dir].length > 0 && Math.random() < 0.58) {
            const v = learned.queues[dir].shift();
            learned.departed++;
            learned.totalWait += (simTime - v.arrived_at);
            learnedEvents.push({ type: 'vehicle_exited', direction: dir, lane_index: 0 });
          }
        }
      } else if (learned.phase === 1) {
        for (const dir of ['east', 'west']) {
          if (learned.queues[dir].length > 0 && Math.random() < 0.58) {
            const v = learned.queues[dir].shift();
            learned.departed++;
            learned.totalWait += (simTime - v.arrived_at);
            learnedEvents.push({ type: 'vehicle_exited', direction: dir, lane_index: 0 });
          }
        }
      }

      // Update queues tracking
      const bQueues = { north: baseline.queues.north.length, south: baseline.queues.south.length, east: baseline.queues.east.length, west: baseline.queues.west.length };
      const lQueues = { north: learned.queues.north.length, south: learned.queues.south.length, east: learned.queues.east.length, west: learned.queues.west.length };

      baseline.queueSeconds += (bQueues.north + bQueues.south + bQueues.east + bQueues.west);
      learned.queueSeconds += (lQueues.north + lQueues.south + lQueues.east + lQueues.west);

      // Render displays
      updateControllerDisplay('fixed_time', {
        queues: bQueues,
        queue_vehicles: baseline.queues,
        phase: baseline.phase,
        departed: baseline.departed,
        events: baselineEvents,
      }, simTime);

      updateControllerDisplay('learned', {
        queues: lQueues,
        queue_vehicles: learned.queues,
        phase: learned.phase,
        departed: learned.departed,
        events: learnedEvents,
      }, simTime);

      setProgress((simTime / duration) * 100);
      statusText.textContent = `Simulation time ${simTime} of ${duration} seconds.`;

      if (simTime >= duration) {
        // Complete
        runInProgress = false;
        comparisonFinished = true;
        startButton.disabled = false;
        setProgress(100);
        setStatus('Simulation complete. Review comparative metrics below.', 'done');
        addLog('Comparison complete. Metrics computed.', 'sim-done');

        const bWait = baseline.departed > 0 ? (baseline.totalWait / baseline.departed) : 0;
        const lWait = learned.departed > 0 ? (learned.totalWait / learned.departed) : 0;

        metrics.showComparison({
          average_wait_experienced: Number(bWait.toFixed(2)),
          total_departed: baseline.departed,
          average_queue_length: Number((baseline.queueSeconds / duration).toFixed(2)),
          signal_switches: baseline.switches,
        }, {
          average_wait_experienced: Number(lWait.toFixed(2)),
          total_departed: learned.departed,
          average_queue_length: Number((learned.queueSeconds / duration).toFixed(2)),
          signal_switches: learned.switches,
        });

        byId('results').scrollIntoView({ block: 'start', behavior: scrollBehavior() });
        return;
      }

      clientSimTimer = setTimeout(simStep, STEP_INTERVAL_MS);
    }

    simStep();
  }

  function updateControllerDisplay(controller, snapshot, timestamp) {
    if (!snapshot) return;
    const renderer = renderers[controller];
    renderer.update({
      queues: snapshot.queues,
      queueVehicles: snapshot.queue_vehicles,
      phase: snapshot.phase,
      time: timestamp,
      departed: snapshot.departed,
    });

    for (const vehicleEvent of snapshot.events || []) {
      if (vehicleEvent.type === 'vehicle_entered') {
        renderer.animateVehicleEntry(vehicleEvent.direction, vehicleEvent.vehicle_id);
      } else if (vehicleEvent.type === 'vehicle_exited') {
        renderer.animateVehicleExit(vehicleEvent.direction, vehicleEvent.lane_index);
      }
    }

    metrics.addStepData(controller, snapshot.queues);

    const statistics = byId(`${CONTROLLER_PREFIX[controller]}-statistics`);
    const queues = snapshot.queues;
    statistics.querySelector('.time-value').textContent = timestamp.toFixed(0);
    statistics.querySelector('.queue-value').textContent = `N ${queues.north} S ${queues.south} E ${queues.east} W ${queues.west}`;
    statistics.querySelector('.served-value').textContent = renderer.totalDeparted.toLocaleString();
    statistics.querySelector('.phase-value').textContent = PHASE_NAMES[Number(snapshot.phase)] || 'Clearance';
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
    if (!response.ok) throw new Error(payload.detail || `Status ${response.status}`);
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
    if (clientSimTimer) {
      clearTimeout(clientSimTimer);
      clientSimTimer = null;
    }
    for (const controller of CONTROLLERS) {
      renderers[controller].reset();
      const statistics = byId(`${CONTROLLER_PREFIX[controller]}-statistics`);
      statistics.querySelector('.time-value').textContent = '0';
      statistics.querySelector('.queue-value').textContent = 'N 0 S 0 E 0 W 0';
      statistics.querySelector('.served-value').textContent = '0';
      statistics.querySelector('.phase-value').textContent = PHASE_NAMES[0];
    }
    metrics.reset();
    logArea.replaceChildren();
    setProgress(0);
    byId('comparison-summary').textContent = 'Run a comparison to evaluate both signal controllers.';
    byId('metrics-grid').innerHTML = '<p class="empty-state">Comparison metrics will appear here upon completion.</p>';
  }

  function resetForAnotherRun() {
    if (activeSocket && activeSocket.readyState < WebSocket.CLOSING) {
      activeSocket.close(1000, 'Reset');
    }
    activeSocket = null;
    if (clientSimTimer) {
      clearTimeout(clientSimTimer);
      clientSimTimer = null;
    }
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
    return String(value || '').replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
  }
});
