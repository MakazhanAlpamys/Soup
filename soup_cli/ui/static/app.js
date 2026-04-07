/* Soup Web UI — Frontend Application */

const API = '';  // same origin

// --- State ---
let currentPage = 'dashboard';
let runsData = [];
let systemInfo = null;
let chatMessages = [];
let chatEndpoint = null;

// --- Navigation ---
function navigate(page) {
  currentPage = page;
  document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
  document.getElementById('page-' + page).classList.add('active');
  document.querySelector(`[data-page="${page}"]`).classList.add('active');

  if (page === 'dashboard') loadDashboard();
  else if (page === 'training') loadTrainingPage();
  else if (page === 'data') { /* loaded on demand */ }
  else if (page === 'chat') loadChatPage();
}

// --- API Helpers ---
async function api(path, opts = {}) {
  const resp = await fetch(API + path, {
    headers: { 'Content-Type': 'application/json' },
    ...opts,
  });
  if (!resp.ok) {
    const err = await resp.json().catch(() => ({ detail: resp.statusText }));
    throw new Error(err.detail || 'API error');
  }
  return resp.json();
}

function formatDuration(secs) {
  if (!secs) return '-';
  if (secs < 60) return `${Math.round(secs)}s`;
  if (secs < 3600) return `${Math.floor(secs / 60)}m ${Math.round(secs % 60)}s`;
  return `${Math.floor(secs / 3600)}h ${Math.floor((secs % 3600) / 60)}m`;
}

function formatDate(iso) {
  if (!iso) return '-';
  const d = new Date(iso);
  return d.toLocaleDateString() + ' ' + d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

function statusBadge(status) {
  const map = {
    completed: 'badge-success',
    failed: 'badge-danger',
    running: 'badge-warning',
  };
  return `<span class="badge ${map[status] || 'badge-info'}">${escapeHtml(status)}</span>`;
}

function truncate(str, len = 30) {
  if (!str) return '-';
  return str.length > len ? str.substring(0, len) + '...' : str;
}

// --- Dashboard ---
async function loadDashboard() {
  try {
    const [runsResp, sysResp] = await Promise.all([
      api('/api/runs?limit=100'),
      api('/api/system'),
    ]);
    runsData = runsResp.runs;
    systemInfo = sysResp;
    renderDashboard();
  } catch (err) {
    document.getElementById('dashboard-content').innerHTML =
      `<div class="empty-state"><div class="empty-state-text">Error loading dashboard: ${escapeHtml(err.message)}</div></div>`;
  }
}

function renderDashboard() {
  const completed = runsData.filter(r => r.status === 'completed');
  const failed = runsData.filter(r => r.status === 'failed');
  const running = runsData.filter(r => r.status === 'running');
  const bestLoss = completed.length
    ? Math.min(...completed.map(r => r.final_loss).filter(Boolean)).toFixed(4)
    : '-';

  document.getElementById('dashboard-content').innerHTML = `
    <div class="stats-row">
      <div class="card stat-card">
        <div class="stat-value">${runsData.length}</div>
        <div class="stat-label">Total Runs</div>
      </div>
      <div class="card stat-card">
        <div class="stat-value">${completed.length}</div>
        <div class="stat-label">Completed</div>
      </div>
      <div class="card stat-card">
        <div class="stat-value">${running.length}</div>
        <div class="stat-label">Running</div>
      </div>
      <div class="card stat-card">
        <div class="stat-value">${bestLoss}</div>
        <div class="stat-label">Best Loss</div>
      </div>
    </div>

    <div class="card">
      <div class="card-title">System</div>
      <div style="font-size:0.9rem; color: var(--text-dim)">
        Device: <strong style="color:var(--text)">${escapeHtml(systemInfo.device_name)}</strong> &nbsp;|&nbsp;
        GPU Memory: <strong style="color:var(--text)">${escapeHtml(systemInfo.gpu_info.memory_total)}</strong> &nbsp;|&nbsp;
        Python: <strong style="color:var(--text)">${escapeHtml(systemInfo.python_version)}</strong> &nbsp;|&nbsp;
        Soup: <strong style="color:var(--text)">v${escapeHtml(systemInfo.version)}</strong>
      </div>
    </div>

    <div class="card">
      <div class="card-title">Recent Runs</div>
      ${runsData.length === 0
        ? '<div class="empty-state"><div class="empty-state-text">No runs yet</div><div class="empty-state-hint">Start training with "soup train" or use the New Training page</div></div>'
        : renderRunsTable(runsData.slice(0, 20))
      }
    </div>
  `;
}

function renderRunsTable(runs) {
  return `
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Run ID</th>
            <th>Name</th>
            <th>Model</th>
            <th>Task</th>
            <th>Status</th>
            <th>Loss</th>
            <th>Duration</th>
            <th>Date</th>
            <th></th>
          </tr>
        </thead>
        <tbody>
          ${runs.map(r => `
            <tr style="cursor:pointer" onclick="showRunDetail('${escapeHtml(r.run_id)}')">
              <td><code style="font-size:0.8rem">${escapeHtml(r.run_id.substring(0, 20))}...</code></td>
              <td>${escapeHtml(r.experiment_name || '-')}</td>
              <td>${escapeHtml(truncate(r.base_model))}</td>
              <td>${escapeHtml(r.task || 'sft')}</td>
              <td>${statusBadge(r.status)}</td>
              <td>${r.final_loss ? r.final_loss.toFixed(4) : '-'}</td>
              <td>${formatDuration(r.duration_secs)}</td>
              <td>${formatDate(r.created_at)}</td>
              <td>
                <button class="btn btn-danger btn-sm" onclick="event.stopPropagation(); deleteRun('${escapeHtml(r.run_id)}')">Delete</button>
              </td>
            </tr>
          `).join('')}
        </tbody>
      </table>
    </div>
  `;
}

async function deleteRun(runId) {
  if (!confirm('Delete this run and all its metrics?')) return;
  try {
    await api(`/api/runs/${runId}`, { method: 'DELETE' });
    loadDashboard();
  } catch (err) {
    alert('Error: ' + err.message);
  }
}

// --- Run Detail Modal ---
let lossChart = null;

async function showRunDetail(runId) {
  const modal = document.getElementById('run-modal');
  const body = document.getElementById('run-modal-body');
  modal.classList.add('active');

  body.innerHTML = '<div style="text-align:center;padding:2rem;color:var(--text-dim)">Loading...</div>';

  try {
    const [run, metricsResp, evalResp] = await Promise.all([
      api(`/api/runs/${runId}`),
      api(`/api/runs/${runId}/metrics`),
      api(`/api/runs/${runId}/eval`),
    ]);

    const config = run.config_json ? JSON.parse(run.config_json) : {};
    const metrics = metricsResp.metrics;

    body.innerHTML = `
      <div class="grid-2" style="margin-bottom:1rem">
        <div>
          <div class="form-label">Run ID</div>
          <div><code>${escapeHtml(run.run_id)}</code></div>
        </div>
        <div>
          <div class="form-label">Status</div>
          <div>${statusBadge(run.status)}</div>
        </div>
        <div>
          <div class="form-label">Model</div>
          <div>${escapeHtml(run.base_model || '-')}</div>
        </div>
        <div>
          <div class="form-label">Task</div>
          <div>${escapeHtml(run.task || 'sft')}</div>
        </div>
        <div>
          <div class="form-label">Device</div>
          <div>${escapeHtml(run.device_name || run.device || '-')}</div>
        </div>
        <div>
          <div class="form-label">Duration</div>
          <div>${formatDuration(run.duration_secs)}</div>
        </div>
        <div>
          <div class="form-label">Initial Loss</div>
          <div>${run.initial_loss ? run.initial_loss.toFixed(4) : '-'}</div>
        </div>
        <div>
          <div class="form-label">Final Loss</div>
          <div>${run.final_loss ? run.final_loss.toFixed(4) : '-'}</div>
        </div>
      </div>

      ${metrics.length > 0 ? `
        <div class="chart-grid">
          <div class="card">
            <div class="card-title">Loss</div>
            <div class="chart-container"><canvas id="loss-chart"></canvas></div>
          </div>
          <div class="card">
            <div class="card-title">Learning Rate</div>
            <div class="chart-container"><canvas id="lr-chart"></canvas></div>
          </div>
          <div class="card">
            <div class="card-title">Gradient Norm</div>
            <div class="chart-container"><canvas id="gradnorm-chart"></canvas></div>
          </div>
          <div class="card">
            <div class="card-title">Throughput</div>
            <div class="chart-container"><canvas id="speed-chart"></canvas></div>
          </div>
        </div>
        ${metrics.some(m => m.gpu_mem) ? `
          <div class="card">
            <div class="card-title">GPU Memory</div>
            <div class="chart-container"><canvas id="gpumem-chart"></canvas></div>
          </div>
        ` : ''}
      ` : ''}

      ${evalResp.eval_results && evalResp.eval_results.length > 0 ? `
        <div class="card">
          <div class="card-title">Eval Results</div>
          <div class="table-wrap">
            <table class="eval-table">
              <thead><tr><th>Benchmark</th><th>Score</th><th>Details</th></tr></thead>
              <tbody>
                ${evalResp.eval_results.map(er => `
                  <tr>
                    <td>${escapeHtml(er.benchmark)}</td>
                    <td>${typeof er.score === 'number' ? er.score.toFixed(4) : escapeHtml(String(er.score))}</td>
                    <td><code style="font-size:0.75rem">${er.details_json ? escapeHtml(String(er.details_json).substring(0, 100)) : '-'}</code></td>
                  </tr>
                `).join('')}
              </tbody>
            </table>
          </div>
        </div>
      ` : ''}

      <div class="card">
        <div class="card-title">Config</div>
        <pre style="font-size:0.8rem;color:var(--text-dim);white-space:pre-wrap;max-height:300px;overflow-y:auto">${JSON.stringify(config, null, 2)}</pre>
      </div>
    `;

    if (metrics.length > 0) {
      renderCharts(metrics);
    }
  } catch (err) {
    body.innerHTML = `<div class="empty-state"><div class="empty-state-text">Error: ${escapeHtml(err.message)}</div></div>`;
  }
}

function renderCharts(metrics) {
  const steps = metrics.map(m => m.step);
  const losses = metrics.map(m => m.loss);
  const lrs = metrics.map(m => m.lr);
  const gradNorms = metrics.map(m => m.grad_norm || 0);
  const speeds = metrics.map(m => m.speed || 0);

  const chartOpts = (yLabel) => ({
    responsive: true,
    maintainAspectRatio: false,
    plugins: { legend: { display: false } },
    scales: {
      x: { title: { display: true, text: 'Step', color: '#a09088' }, ticks: { color: '#a09088' }, grid: { color: 'rgba(58,48,64,0.5)' } },
      y: { title: { display: true, text: yLabel, color: '#a09088' }, ticks: { color: '#a09088' }, grid: { color: 'rgba(58,48,64,0.5)' } },
    },
  });

  const makeDataset = (label, data, color) => ({
    label, data, borderColor: color,
    backgroundColor: color.replace(')', ', 0.1)').replace('rgb', 'rgba'),
    fill: true, tension: 0.3, pointRadius: 0, borderWidth: 2,
  });

  // Loss chart
  const lossCtx = document.getElementById('loss-chart');
  if (lossCtx) {
    if (lossChart) lossChart.destroy();
    lossChart = new Chart(lossCtx, {
      type: 'line',
      data: { labels: steps, datasets: [makeDataset('Loss', losses, 'rgb(192, 81, 45)')] },
      options: chartOpts('Loss'),
    });
  }

  // LR chart
  const lrCtx = document.getElementById('lr-chart');
  if (lrCtx) {
    new Chart(lrCtx, {
      type: 'line',
      data: { labels: steps, datasets: [makeDataset('LR', lrs, 'rgb(232, 151, 90)')] },
      options: chartOpts('LR'),
    });
  }

  // Gradient Norm chart
  const gnCtx = document.getElementById('gradnorm-chart');
  if (gnCtx) {
    new Chart(gnCtx, {
      type: 'line',
      data: { labels: steps, datasets: [makeDataset('Grad Norm', gradNorms, 'rgb(100, 180, 220)')] },
      options: chartOpts('Grad Norm'),
    });
  }

  // Speed chart
  const spCtx = document.getElementById('speed-chart');
  if (spCtx) {
    new Chart(spCtx, {
      type: 'line',
      data: { labels: steps, datasets: [makeDataset('Speed', speeds, 'rgb(120, 200, 130)')] },
      options: chartOpts('Tokens/sec'),
    });
  }

  // GPU Memory chart (optional — parse numeric values from strings like "4.2GB")
  const gmCtx = document.getElementById('gpumem-chart');
  if (gmCtx) {
    const gpuVals = metrics.map(m => {
      if (!m.gpu_mem) return 0;
      const match = String(m.gpu_mem).match(/([\d.]+)/);
      return match ? parseFloat(match[1]) : 0;
    });
    new Chart(gmCtx, {
      type: 'line',
      data: { labels: steps, datasets: [makeDataset('GPU Mem', gpuVals, 'rgb(200, 130, 200)')] },
      options: chartOpts('GPU Memory (GB)'),
    });
  }
}

function closeModal() {
  document.getElementById('run-modal').classList.remove('active');
}

// --- New Training Page ---
async function loadTrainingPage() {
  try {
    const [templatesResp, statusResp, recipesResp] = await Promise.all([
      api('/api/templates'),
      api('/api/train/status'),
      api('/api/recipes'),
    ]);
    window._recipes = recipesResp.recipes || [];
    renderTrainingPage(templatesResp.templates, statusResp);
  } catch (err) {
    document.getElementById('training-content').innerHTML =
      `<div class="empty-state"><div class="empty-state-text">Error: ${escapeHtml(err.message)}</div></div>`;
  }
}

function renderTrainingPage(templates, status) {
  const templateNames = Object.keys(templates);
  const editorId = 'config-editor';

  document.getElementById('training-content').innerHTML = `
    <div class="grid-2">
      <div>
        <div class="card">
          <div class="card-title">Template / Recipe</div>
          <div class="grid-2" style="margin-bottom:0">
            <div class="form-group" style="margin-bottom:0">
              <label class="form-label" style="font-size:0.8rem">Template</label>
              <select id="template-select" onchange="loadTemplate()">
                <option value="">-- Template --</option>
                ${templateNames.map(t => `<option value="${escapeHtml(t)}">${escapeHtml(t)}</option>`).join('')}
              </select>
            </div>
            <div class="form-group" style="margin-bottom:0">
              <label class="form-label" style="font-size:0.8rem">Recipe</label>
              <select id="recipe-select" onchange="loadRecipe()">
                <option value="">-- Recipe --</option>
                ${(window._recipes || []).map(r => `<option value="${escapeHtml(r.name)}">${escapeHtml(r.name)} (${escapeHtml(r.task)})</option>`).join('')}
              </select>
            </div>
          </div>
        </div>

        <div class="card">
          <div class="card-title">Config (YAML)</div>
          <textarea id="${editorId}" rows="22" placeholder="Paste your soup.yaml config here or select a template...">${templates[templateNames[0]] || ''}</textarea>
        </div>

        <div style="display:flex; gap:0.75rem; margin-top:0.75rem">
          <button class="btn btn-primary" onclick="validateConfig()">Validate</button>
          <button class="btn btn-primary" onclick="startTraining()">Start Training</button>
        </div>
        <div id="config-status" style="margin-top:0.75rem; font-size:0.85rem"></div>
      </div>

      <div>
        <div class="card">
          <div class="card-title">Training Status</div>
          <div id="train-status-panel">
            ${status.running
              ? `<div><span class="badge badge-warning">Running</span> PID: ${escapeHtml(String(status.pid))}</div>
                 <button class="btn btn-danger btn-sm" style="margin-top:0.75rem" onclick="stopTraining()">Stop Training</button>`
              : '<div style="color:var(--text-dim)">No training in progress</div>'
            }
          </div>
        </div>

        <div class="card">
          <div class="card-title">Quick Reference</div>
          <div style="font-size:0.85rem; color:var(--text-dim); line-height:1.8">
            <strong>Tasks:</strong> sft, dpo, grpo<br>
            <strong>Backends:</strong> transformers, unsloth<br>
            <strong>Modalities:</strong> text, vision<br>
            <strong>Quantization:</strong> 4bit, 8bit, none<br>
            <strong>Formats:</strong> alpaca, sharegpt, chatml, dpo, llava, sharegpt4v<br>
          </div>
        </div>
      </div>
    </div>
  `;

  // Store templates globally
  window._templates = templates;
}

function loadTemplate() {
  const sel = document.getElementById('template-select');
  const editor = document.getElementById('config-editor');
  if (sel.value && window._templates[sel.value]) {
    editor.value = window._templates[sel.value];
  }
}

function loadRecipe() {
  const sel = document.getElementById('recipe-select');
  const editor = document.getElementById('config-editor');
  if (!sel.value || !window._recipes) return;
  const recipe = window._recipes.find(r => r.name === sel.value);
  if (recipe && recipe.yaml) {
    editor.value = recipe.yaml;
  }
}

async function validateConfig() {
  const yaml = document.getElementById('config-editor').value;
  const statusEl = document.getElementById('config-status');
  try {
    const result = await api('/api/config/validate', {
      method: 'POST',
      body: JSON.stringify({ yaml }),
    });
    if (result.valid) {
      statusEl.innerHTML = '<span style="color:var(--accent)">Config is valid!</span>';
    } else {
      statusEl.innerHTML = `<span style="color:var(--danger)">Invalid: ${escapeHtml(result.error)}</span>`;
    }
  } catch (err) {
    statusEl.innerHTML = `<span style="color:var(--danger)">Error: ${escapeHtml(err.message)}</span>`;
  }
}

async function startTraining() {
  const yaml = document.getElementById('config-editor').value;
  if (!yaml.trim()) {
    alert('Please enter a config');
    return;
  }
  if (!confirm('Start training with this config?')) return;

  try {
    const result = await api('/api/train/start', {
      method: 'POST',
      body: JSON.stringify({ config_yaml: yaml }),
    });
    document.getElementById('config-status').innerHTML =
      `<span style="color:var(--accent)">Training started! PID: ${escapeHtml(String(result.pid))}</span>`;
    // Refresh status
    loadTrainingPage();
  } catch (err) {
    document.getElementById('config-status').innerHTML =
      `<span style="color:var(--danger)">Error: ${escapeHtml(err.message)}</span>`;
  }
}

async function stopTraining() {
  if (!confirm('Stop the current training run?')) return;
  try {
    await api('/api/train/stop', { method: 'POST' });
    loadTrainingPage();
  } catch (err) {
    alert('Error: ' + err.message);
  }
}

// --- Data Explorer ---
async function inspectData() {
  const path = document.getElementById('data-path').value;
  if (!path.trim()) { alert('Enter a file path'); return; }

  const limit = parseInt(document.getElementById('data-limit').value) || 50;
  const content = document.getElementById('data-content');
  content.innerHTML = '<div style="text-align:center;padding:2rem;color:var(--text-dim)">Loading...</div>';

  try {
    const result = await api('/api/data/inspect', {
      method: 'POST',
      body: JSON.stringify({ path, limit }),
    });
    renderDataResults(result);
  } catch (err) {
    content.innerHTML = `<div class="empty-state"><div class="empty-state-text">Error: ${escapeHtml(err.message)}</div></div>`;
  }
}

function renderDataResults(data) {
  const content = document.getElementById('data-content');

  content.innerHTML = `
    <div class="stats-row" style="margin-bottom:1rem">
      <div class="card stat-card">
        <div class="stat-value">${data.total}</div>
        <div class="stat-label">Total Entries</div>
      </div>
      <div class="card stat-card">
        <div class="stat-value" style="font-size:1.5rem">${data.format}</div>
        <div class="stat-label">Detected Format</div>
      </div>
      <div class="card stat-card">
        <div class="stat-value">${data.keys.length}</div>
        <div class="stat-label">Fields</div>
      </div>
    </div>

    <div class="card">
      <div class="card-title">Fields: ${data.keys.join(', ')}</div>
    </div>

    <div class="card">
      <div class="card-title">Sample Data (${data.sample.length} of ${data.total})</div>
      ${data.sample.map((entry, idx) => `
        <div class="data-entry">
          <div style="font-size:0.75rem; color:var(--text-dim); margin-bottom:0.5rem">#${idx + 1}</div>
          ${Object.entries(entry).map(([key, val]) => `
            <div class="data-entry-field">
              <span class="data-entry-key">${key}:</span>
              <span>${typeof val === 'object' ? JSON.stringify(val).substring(0, 200) : String(val).substring(0, 200)}</span>
            </div>
          `).join('')}
        </div>
      `).join('')}
    </div>
  `;
}

// --- Model Chat ---
let chatAbortController = null;

function loadChatPage() {
  renderChatMessages();
}

function renderChatMessages() {
  const container = document.getElementById('chat-messages');
  if (!container) return;

  if (chatMessages.length === 0) {
    container.innerHTML = `
      <div class="empty-state">
        <div class="empty-state-text">No messages yet</div>
        <div class="empty-state-hint">Enter a server URL and start chatting</div>
      </div>
    `;
    return;
  }

  container.innerHTML = chatMessages.map(msg => `
    <div class="chat-msg ${msg.role}">
      <div class="chat-msg-role">${msg.role}</div>
      <div class="chat-msg-content chat-markdown">${msg.role === 'assistant' ? renderMarkdown(msg.content) : escapeHtml(msg.content)}</div>
    </div>
  `).join('');

  container.scrollTop = container.scrollHeight;
}

function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}

function renderMarkdown(text) {
  let html = escapeHtml(text);
  // Code blocks (``` ... ```)
  html = html.replace(/```(\w*)\n([\s\S]*?)```/g, '<pre class="code-block"><code>$2</code></pre>');
  // Inline code
  html = html.replace(/`([^`]+)`/g, '<code>$1</code>');
  // Bold
  html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
  // Italic
  html = html.replace(/\*(.+?)\*/g, '<em>$1</em>');
  // List items
  html = html.replace(/^- (.+)$/gm, '<li>$1</li>');
  // Line breaks
  html = html.replace(/\n/g, '<br>');
  return html;
}

async function sendChatMessage() {
  const input = document.getElementById('chat-input');
  const serverUrl = document.getElementById('chat-server').value.trim();
  const msg = input.value.trim();
  if (!msg) return;
  if (!serverUrl) { alert('Enter a server URL'); return; }

  // Build messages with optional system prompt
  const systemPrompt = document.getElementById('chat-system')?.value?.trim();
  const allMessages = [];
  if (systemPrompt) {
    allMessages.push({ role: 'system', content: systemPrompt });
  }
  chatMessages.push({ role: 'user', content: msg });
  allMessages.push(...chatMessages.map(m => ({ role: m.role, content: m.content })));
  input.value = '';
  renderChatMessages();

  // Show typing indicator, switch buttons
  const typing = document.getElementById('typing-indicator');
  const sendBtn = document.getElementById('chat-send-btn');
  const cancelBtn = document.getElementById('chat-cancel-btn');
  if (typing) typing.style.display = 'flex';
  if (sendBtn) sendBtn.style.display = 'none';
  if (cancelBtn) cancelBtn.style.display = 'inline-flex';

  chatAbortController = new AbortController();

  const temperature = parseFloat(document.getElementById('chat-temperature')?.value || '0.7');
  const maxTokens = parseInt(document.getElementById('chat-max-tokens')?.value || '512');
  const topP = parseFloat(document.getElementById('chat-top-p')?.value || '0.9');
  const adapter = document.getElementById('chat-adapter')?.value?.trim() || undefined;

  try {
    const resp = await fetch(API + '/api/chat/send', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer ' + (window._authToken || ''),
      },
      body: JSON.stringify({
        messages: allMessages,
        endpoint: serverUrl,
        temperature: temperature,
        max_tokens: maxTokens,
        top_p: topP,
        adapter: adapter,
      }),
      signal: chatAbortController.signal,
    });

    // Stream tokens
    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let assistantMsg = '';
    chatMessages.push({ role: 'assistant', content: '' });

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      const chunk = decoder.decode(value);
      const lines = chunk.split('\n');
      for (const line of lines) {
        if (line.startsWith('data: ')) {
          try {
            const data = JSON.parse(line.slice(6));
            if (data.done) break;
            if (data.delta) {
              assistantMsg += data.delta;
              chatMessages[chatMessages.length - 1].content = assistantMsg;
              renderChatMessages();
            }
            if (data.error) {
              assistantMsg += '[Error: ' + data.error + ']';
              chatMessages[chatMessages.length - 1].content = assistantMsg;
              renderChatMessages();
            }
          } catch (parseErr) { /* ignore malformed lines */ }
        }
      }
    }
  } catch (err) {
    if (err.name !== 'AbortError') {
      chatMessages.push({ role: 'assistant', content: `[Error: ${err.message}]` });
      renderChatMessages();
    }
  } finally {
    chatAbortController = null;
    if (typing) typing.style.display = 'none';
    if (sendBtn) sendBtn.style.display = 'inline-flex';
    if (cancelBtn) cancelBtn.style.display = 'none';
  }
}

function cancelChat() {
  if (chatAbortController) chatAbortController.abort();
}

function clearChat() {
  chatMessages = [];
  renderChatMessages();
}

function exportChat() {
  if (chatMessages.length === 0) return;
  const blob = new Blob([JSON.stringify(chatMessages, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'chat-export.json';
  a.click();
  URL.revokeObjectURL(url);
}

function handleChatKey(event) {
  if (event.key === 'Enter' && !event.shiftKey) {
    event.preventDefault();
    sendChatMessage();
  }
}

// --- Training Live Monitor (SSE) ---
let logEventSource = null;
let metricsEventSource = null;
const LOG_MAX_LINES = 500;

function connectTrainingSSE() {
  disconnectTrainingSSE();

  const logPanel = document.getElementById('train-log-panel');
  const progressPanel = document.getElementById('train-progress');
  const liveBadge = document.getElementById('live-badge');
  if (!logPanel) return;

  logPanel.style.display = 'block';
  progressPanel.style.display = 'block';
  if (liveBadge) liveBadge.style.display = 'inline-flex';

  // Connect to log SSE
  logEventSource = new EventSource(API + '/api/train/logs');
  logEventSource.onmessage = function(ev) {
    const data = JSON.parse(ev.data);
    appendLogLine(data.line);
  };
  logEventSource.addEventListener('done', function() {
    appendLogLine('[Training finished]');
    disconnectTrainingSSE();
  });
  logEventSource.onerror = function() {
    setTimeout(function() {
      if (logEventSource && logEventSource.readyState === EventSource.CLOSED) {
        disconnectTrainingSSE();
      }
    }, 5000);
  };
}

function disconnectTrainingSSE() {
  if (logEventSource) { logEventSource.close(); logEventSource = null; }
  if (metricsEventSource) { metricsEventSource.close(); metricsEventSource = null; }
  const liveBadge = document.getElementById('live-badge');
  if (liveBadge) liveBadge.style.display = 'none';
}

function appendLogLine(text) {
  const output = document.getElementById('log-output');
  if (!output) return;
  output.textContent += text + '\n';
  // Ring buffer: trim to max lines
  const lines = output.textContent.split('\n');
  if (lines.length > LOG_MAX_LINES) {
    output.textContent = lines.slice(lines.length - LOG_MAX_LINES).join('\n');
  }
  // Auto-scroll
  const autoScroll = document.getElementById('log-autoscroll');
  if (autoScroll && autoScroll.checked) {
    output.scrollTop = output.scrollHeight;
  }
}

function updateProgressBar(step, total, elapsed, eta) {
  const fill = document.getElementById('progress-fill');
  const stepEl = document.getElementById('progress-step');
  const elapsedEl = document.getElementById('progress-elapsed');
  const etaEl = document.getElementById('progress-eta');
  if (!fill) return;

  const pct = total > 0 ? Math.min(100, (step / total) * 100) : 0;
  fill.style.width = pct.toFixed(1) + '%';
  if (stepEl) stepEl.textContent = 'Step ' + step + (total ? '/' + total : '');
  if (elapsedEl) elapsedEl.textContent = 'Elapsed: ' + formatDuration(elapsed);
  if (etaEl) etaEl.textContent = 'ETA: ' + formatDuration(eta);
}

// --- Init ---
document.addEventListener('DOMContentLoaded', () => {
  navigate('dashboard');
});
