/* LifeOps Agent — dashboard client.
   Talks to the FastAPI backend only; no keys and no third-party calls here. */

const STEPS = [
  { tool: 'parse_user_context', label: 'Understanding request', hint: 'Normalises times, tasks and defaults' },
  { tool: 'load_saved_places', label: 'Loading saved places', hint: 'Reads the places saved in this browser' },
  { tool: 'check_tomorrow_pressure', label: 'Checking tomorrow pressure', hint: 'Required wake-up and latest healthy bedtime' },
  { tool: 'estimate_task_priority', label: 'Ranking tasks', hint: 'What to drop first when the evening is full' },
  { tool: 'get_weather_context', label: 'Checking weather', hint: 'Open-Meteo forecast, no API key needed' },
  { tool: 'estimate_travel_time', label: 'Estimating travel', hint: 'Distance and average city speed' },
  { tool: 'create_evening_plan', label: 'Calling planner tool', hint: 'Deterministic scheduler owns every clock time' },
  { tool: 'compare_alternative_plans', label: 'Comparing alternatives', hint: 'Sleep-first vs productivity, real numbers' },
  { tool: 'recommendation', label: 'Generating recommendation', hint: 'Trade-off explanation written by the model' },
];

const EXAMPLES = [
  {
    label: 'Client meeting tomorrow',
    text: "I get home at 20:15. I need to cook, do laundry and review tomorrow's presentation. I have a 9 AM client meeting so I want to sleep early.",
  },
  {
    label: 'Quiet evening',
    text: 'Home at 19:00 after the gym. Dinner, dishes, 30 minutes of French practice, and I would like to watch one episode. Nothing before 10 AM tomorrow.',
  },
  {
    label: 'Late train',
    text: 'I arrive at Gare de Lyon at 22:40 and it is 25 minutes home. I still need to eat and shower, and I start work at 08:30.',
  },
  {
    label: 'High energy',
    text: 'Home at 18:30 with high energy. I want a run, cook properly, call my parents, and prepare a workshop for 09:30 tomorrow in La Défense.',
  },
];

// The key is kept apart from the other prefs so "forget key" is one clean delete.
const STORE = { places: 'lifeops.places', prefs: 'lifeops.prefs', key: 'lifeops.apikey' };

const $ = (id) => document.getElementById(id);
const state = { energy: 'medium', places: {}, busy: false, entries: [], serverHasKey: false };

/* ── tiny DOM helper ─────────────────────────────────────────── */
function el(tag, props = {}, children = []) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(props)) {
    if (value === null || value === undefined || value === false) continue;
    if (key === 'class') node.className = value;
    else if (key === 'text') node.textContent = value;
    else if (key === 'html') node.innerHTML = value;
    else if (key.startsWith('on')) node.addEventListener(key.slice(2), value);
    else node.setAttribute(key, value);
  }
  for (const child of [].concat(children)) {
    if (child) node.append(child.nodeType ? child : document.createTextNode(child));
  }
  return node;
}

function icon(name, cls = 'ico') {
  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  svg.setAttribute('class', cls);
  const use = document.createElementNS('http://www.w3.org/2000/svg', 'use');
  use.setAttribute('href', `#i-${name}`);
  svg.append(use);
  return svg;
}

/* ── persistence ─────────────────────────────────────────────── */
function loadStore(key, fallback) {
  try {
    return JSON.parse(localStorage.getItem(key)) ?? fallback;
  } catch {
    return fallback;
  }
}

function saveStore(key, value) {
  try {
    localStorage.setItem(key, JSON.stringify(value));
  } catch { /* private mode: keep working in memory */ }
}

function savePrefs() {
  saveStore(STORE.prefs, {
    energy: state.energy,
    location: $('location').value,
    message: $('message').value,
  });
}

/* ── API key (this browser only, never sent anywhere but our own backend) ── */
function apiKey() {
  return $('api-key').value.trim();
}

function saveKey() {
  const key = apiKey();
  try {
    if (key) localStorage.setItem(STORE.key, key);
    else localStorage.removeItem(STORE.key);
  } catch { /* private mode: keep it in the field only */ }
  renderKeyState();
}

function renderKeyState() {
  const set = !!apiKey();
  const label = $('key-requirement');
  $('key-field').classList.toggle('key-field--set', set);
  label.classList.toggle('is-set', set);
  label.classList.toggle('is-required', !set && !state.serverHasKey);
  if (set) label.textContent = 'saved in this browser';
  else if (state.serverHasKey) label.textContent = 'optional — server has one';
  else label.textContent = 'required';
}

/* ── saved places ────────────────────────────────────────────── */
function renderPlaces() {
  const list = $('places');
  list.replaceChildren();
  for (const [name, value] of Object.entries(state.places)) {
    list.append(
      el('li', {}, [
        el('span', { class: 'place-key', text: name }),
        el('span', { class: 'place-val', text: value, title: value }),
        el(
          'button',
          {
            class: 'icon-btn',
            type: 'button',
            title: `Remove ${name}`,
            'aria-label': `Remove ${name}`,
            onclick: () => {
              delete state.places[name];
              saveStore(STORE.places, state.places);
              renderPlaces();
            },
          },
          [icon('x')]
        ),
      ])
    );
  }
}

/* ── agent trace ─────────────────────────────────────────────── */
function stepCard(step, entry, index, isActive) {
  const status = entry ? entry.status : isActive ? 'active' : 'pending';
  const detail = entry ? entry.detail : step.hint;
  const hasPayload = !!entry && (Object.keys(entry.arguments || {}).length || Object.keys(entry.result || {}).length);

  const badge = el('span', { class: 'step-index' });
  if (status === 'complete') badge.append(icon('check'));
  else if (status === 'active') badge.append();
  else badge.textContent = String(index + 1);

  const meta = el('span', { class: 'step-meta' }, [
    el('span', { class: 'step-tool', text: step.tool }),
    entry && entry.elapsed_ms ? el('span', { text: `${(entry.elapsed_ms / 1000).toFixed(1)}s` }) : null,
    status === 'skipped' ? el('span', { text: 'not needed' }) : null,
  ]);

  const head = el('button', { class: 'step-head', type: 'button' }, [
    badge,
    el('span', { class: 'step-text' }, [
      el('div', { class: 'step-name', text: step.label }),
      el('div', { class: 'step-detail', text: detail }),
      meta,
    ]),
  ]);

  const card = el('li', { class: `step step--${status}${hasPayload ? ' has-detail' : ''}` }, [head]);

  if (hasPayload) {
    const payload = el('div', { class: 'step-payload', hidden: 'hidden' });
    if (Object.keys(entry.arguments || {}).length) {
      payload.append(el('h4', { text: 'Tool input' }), el('pre', { text: JSON.stringify(entry.arguments, null, 2) }));
    }
    if (Object.keys(entry.result || {}).length) {
      payload.append(el('h4', { text: 'Tool output' }), el('pre', { text: JSON.stringify(entry.result, null, 2) }));
    }
    card.append(payload);
    head.addEventListener('click', () => {
      payload.hidden = !payload.hidden;
    });
  }
  return card;
}

function renderTrace() {
  const list = $('trace');
  const byTool = new Map();
  const repeats = [];
  for (const entry of state.entries) {
    if (byTool.has(entry.tool)) repeats.push(entry);
    else byTool.set(entry.tool, entry);
  }

  const cards = [];
  let activeMarked = false;
  STEPS.forEach((step, index) => {
    const entry = byTool.get(step.tool);
    const isActive = !entry && state.busy && !activeMarked;
    if (isActive) activeMarked = true;
    cards.push(stepCard(step, entry, index, isActive));
  });
  repeats.forEach((entry, index) => {
    const step = STEPS.find((s) => s.tool === entry.tool) || { tool: entry.tool, label: entry.step, hint: '' };
    cards.push(stepCard(step, entry, STEPS.length + index, false));
  });
  list.replaceChildren(...cards);
}

/* ── plan dashboard ──────────────────────────────────────────── */
function timeline(blocks) {
  return el(
    'ul',
    { class: 'timeline' },
    blocks.map((block) =>
      el('li', { 'data-type': block.type }, [
        el('span', { class: 'tl-time', text: block.start }),
        el('span', { class: 'tl-rail' }),
        el('span', { class: 'tl-task', text: block.task }),
        el('span', { class: 'tl-len', text: `${block.minutes} min` }),
      ])
    )
  );
}

function stat(label, iconName, value, sub) {
  return el('div', { class: 'stat' }, [
    el('div', { class: 'stat-label' }, [icon(iconName), label]),
    el('div', { class: 'stat-value', text: value }),
    sub ? el('div', { class: 'stat-sub', text: sub }) : null,
  ]);
}

function note(kind, iconName, text) {
  return el('li', { class: `note note--${kind}` }, [icon(iconName), el('span', { text })]);
}

function section(title, iconName, body) {
  return el('div', { class: 'section' }, [el('h3', {}, [icon(iconName), title]), body]);
}

function readinessBlock(score) {
  const band = score >= 75 ? '' : score >= 55 ? 'readiness--warn' : 'readiness--risk';
  const headline = score >= 75 ? 'Ready for tomorrow' : score >= 55 ? 'Tight, but workable' : 'Tomorrow is at risk';
  const circumference = 2 * Math.PI * 25;

  // innerHTML on an HTML parent puts <svg> in the right namespace; createElement would not.
  const ring = el('div', {
    class: 'ring-wrap',
    html:
      `<svg class="ring" viewBox="0 0 62 62" aria-hidden="true">` +
      `<circle class="ring-bg" cx="31" cy="31" r="25"></circle>` +
      `<circle class="ring-fg" cx="31" cy="31" r="25" stroke-dasharray="${circumference}" ` +
      `stroke-dashoffset="${circumference * (1 - Math.max(0, Math.min(100, score)) / 100)}"></circle>` +
      `</svg>`,
  });
  ring.append(el('span', { class: 'ring-num', text: String(score) }));

  return el('div', { class: `readiness ${band}` }, [
    ring,
    el('div', { class: 'readiness-text' }, [
      el('h3', { text: headline }),
      el('p', { text: 'Tomorrow readiness score — sleep against your target, minus warnings and constraint violations.' }),
    ]),
  ]);
}

function alternativesBlock(alternatives, currentMode) {
  return el(
    'div',
    { class: 'alts' },
    alternatives.map((alt) =>
      el('div', { class: `alt${alt.mode === currentMode ? ' is-current' : ''}` }, [
        el('div', { class: 'alt-head' }, [
          el('span', { class: 'alt-name', text: alt.name }),
          alt.mode === currentMode ? el('span', { class: 'alt-badge', text: 'Recommended' }) : null,
        ]),
        el('div', { class: 'alt-summary', text: alt.summary }),
        el('div', { class: 'alt-stats' }, [
          el('span', {}, [el('b', { text: alt.bedtime }), ' bedtime']),
          el('span', {}, [el('b', { text: alt.sleep_duration }), ' sleep']),
          el('span', {}, [el('b', { text: `${alt.tomorrow_readiness_score}/100` }), ' readiness']),
        ]),
      ])
    )
  );
}

function renderPlan(data) {
  const host = $('plan-content');
  host.replaceChildren();
  $('plan-empty').hidden = true;
  host.hidden = false;

  if (data.status === 'error' || !data.recommended_plan) {
    const message = data.error ? data.error.message : 'The agent did not return a plan.';
    host.append(
      el('div', { class: 'banner' }, [
        icon('alert'),
        el('div', {}, [
          el('strong', { text: data.error && data.error.code === 'missing_api_key' ? 'Configuration error' : 'Could not plan' }),
          el('p', { class: 'prose', style: 'margin-top:4px;color:inherit', text: message }),
        ]),
      ])
    );
    if (data.summary && data.summary !== message) {
      host.append(el('p', { class: 'prose', style: 'margin-top:12px', text: data.summary }));
    }
    return;
  }

  const plan = data.recommended_plan;
  const target = plan.target_sleep_minutes;
  const deficit = target - plan.sleep_minutes;

  host.append(
    el('div', { class: 'stat-grid' }, [
      stat('Bedtime', 'moon', plan.bedtime, `home at ${plan.arrival_time}`),
      stat('Wake-up', 'sun', plan.wake_time, plan.tomorrow.first_event ? `first event ${plan.tomorrow.first_event}` : 'usual departure'),
      stat(
        'Sleep',
        'clock',
        plan.sleep_duration,
        deficit > 0 ? `${Math.round(deficit)} min under target` : 'target met'
      ),
    ])
  );

  host.append(readinessBlock(data.tomorrow_readiness_score));

  if (data.summary) {
    host.append(section('Agent summary', 'sparkle', el('p', { class: 'prose', text: data.summary })));
  }

  host.append(section('Tonight', 'moon', timeline(plan.evening_schedule)));
  host.append(section('Tomorrow morning', 'sun', timeline(plan.morning_schedule)));

  const alerts = [];
  plan.violations.forEach((text) => alerts.push(note('risk', 'alert', text)));
  plan.warnings.forEach((text) => alerts.push(note('warn', 'alert', text)));
  (data.coach_notes || []).forEach((text) => alerts.push(note('tip', 'sparkle', text)));
  if (data.weather && data.weather.available) {
    const w = data.weather;
    alerts.push(
      note('info', 'cloud', `${w.location}: ${w.evening.description}, ${w.evening.temperature_c}°C tonight, ${w.evening.precipitation_probability}% rain — ${w.advice}.`)
    );
  }
  if (data.travel && data.travel.minutes) {
    alerts.push(note('info', 'route', `Travel estimate: ${data.travel.minutes} min by ${data.travel.mode}${data.travel.distance_km ? ` (${data.travel.distance_km} km)` : ''} — ${data.travel.confidence} confidence.`));
  }
  if (!alerts.length) alerts.push(note('info', 'check', 'No constraint violations. Everything you asked for fits tonight.'));
  host.append(section('Warnings & context', 'alert', el('ul', { class: 'notes' }, alerts)));

  if (data.tradeoff_explanation) {
    host.append(section('Trade-offs', 'scale', el('p', { class: 'prose', text: data.tradeoff_explanation })));
  }
  if (data.alternatives && data.alternatives.length) {
    host.append(section('Alternatives', 'route', alternativesBlock(data.alternatives, 'balanced')));
  }

  const rules = el('details', { class: 'section' }, [
    el('summary', { class: 'prose', style: 'cursor:pointer', text: 'Planner rules applied' }),
    el(
      'ul',
      { class: 'notes', style: 'margin-top:8px' },
      plan.rules_applied.map((rule) => note('info', 'check', rule))
    ),
  ]);
  host.append(rules);
}

/* ── request flow ────────────────────────────────────────────── */
function setBusy(busy) {
  state.busy = busy;
  $('generate').disabled = busy;
  $('generate-label').textContent = busy ? 'Agent is working…' : 'Plan my evening';
  $('trace-hint').textContent = busy ? 'running' : 'Step 2';
}

function payload() {
  return {
    message: $('message').value.trim(),
    energy_level: state.energy,
    location: $('location').value.trim() || null,
    saved_places: state.places,
    api_key: apiKey() || null,
  };
}

async function streamPlan(body) {
  const response = await fetch('/api/plan/stream', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!response.ok || !response.body) throw new Error(`stream unavailable (${response.status})`);

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let result = null;

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const chunks = buffer.split('\n\n');
    buffer = chunks.pop();
    for (const chunk of chunks) {
      const line = chunk.split('\n').find((l) => l.startsWith('data:'));
      if (!line) continue;
      let event;
      try {
        event = JSON.parse(line.slice(5));
      } catch {
        continue;
      }
      if (event.type === 'trace') {
        state.entries.push(event.entry);
        renderTrace();
      } else if (event.type === 'result') {
        result = event.payload;
      }
    }
  }
  if (!result) throw new Error('stream ended without a result');
  return result;
}

async function postPlan(body) {
  const response = await fetch('/api/plan', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  return response.json();
}

async function generate() {
  if (state.busy) return;
  const body = payload();
  if (!body.message) {
    $('message').focus();
    return;
  }
  if (!body.api_key && !state.serverHasKey) {
    $('api-key').focus();
    $('key-field').classList.add('key-field--nudge');
    setTimeout(() => $('key-field').classList.remove('key-field--nudge'), 1200);
    return;
  }

  savePrefs();
  state.entries = [];
  setBusy(true);
  renderTrace();

  let data;
  try {
    data = await streamPlan(body);
  } catch (streamError) {
    try {
      data = await postPlan(body);
    } catch (error) {
      data = { status: 'error', error: { code: 'network', message: `Could not reach the agent: ${error.message}` } };
    }
  }

  setBusy(false);
  // The final trace is authoritative: it also carries the steps the agent skipped.
  if (data.agent_trace) state.entries = data.agent_trace;
  renderTrace();
  renderPlan(data);
  if (window.innerWidth <= 820) {
    document.getElementById('plan-content').scrollIntoView({ behavior: 'smooth', block: 'start' });
  }
}

/* ── boot ────────────────────────────────────────────────────── */
function init() {
  const prefs = loadStore(STORE.prefs, {});
  state.places = loadStore(STORE.places, {});
  state.energy = prefs.energy || 'medium';
  if (prefs.location) $('location').value = prefs.location;
  if (prefs.message) $('message').value = prefs.message;

  for (const button of $('energy').querySelectorAll('button')) {
    button.classList.toggle('is-active', button.dataset.value === state.energy);
    button.setAttribute('aria-checked', String(button.dataset.value === state.energy));
    button.addEventListener('click', () => {
      state.energy = button.dataset.value;
      for (const other of $('energy').querySelectorAll('button')) {
        other.classList.toggle('is-active', other === button);
        other.setAttribute('aria-checked', String(other === button));
      }
      savePrefs();
    });
  }

  $('examples').replaceChildren(
    ...EXAMPLES.map((example) =>
      el('button', {
        type: 'button',
        text: example.label,
        title: example.text,
        onclick: () => {
          $('message').value = example.text;
          $('message').focus();
          savePrefs();
        },
      })
    )
  );

  $('place-form').addEventListener('submit', (event) => {
    event.preventDefault();
    const name = $('place-name').value.trim();
    const value = $('place-value').value.trim();
    if (!name || !value) return;
    state.places[name] = value;
    saveStore(STORE.places, state.places);
    $('place-name').value = '';
    $('place-value').value = '';
    renderPlaces();
  });

  $('api-key').value = loadStore(STORE.key, '') || '';
  $('api-key').addEventListener('change', saveKey);
  $('api-key').addEventListener('input', renderKeyState);
  $('key-clear').addEventListener('click', () => {
    $('api-key').value = '';
    saveKey();
    $('api-key').focus();
  });

  $('generate').addEventListener('click', generate);
  $('message').addEventListener('input', savePrefs);
  $('location').addEventListener('input', savePrefs);
  $('message').addEventListener('keydown', (event) => {
    if ((event.metaKey || event.ctrlKey) && event.key === 'Enter') generate();
  });

  renderPlaces();
  renderTrace();
  renderKeyState();

  fetch('/api/health')
    .then((r) => r.json())
    .then((health) => {
      state.serverHasKey = !!health.llm_configured;
      const dot = $('status-chip').querySelector('.dot');
      dot.className = 'dot dot--ok';
      $('status-text').textContent = health.llm_configured
        ? `Agent online · ${health.model}`
        : `Bring your own key · ${health.model}`;
      renderKeyState();
    })
    .catch(() => {
      $('status-text').textContent = 'Backend unreachable';
    });
}

document.addEventListener('DOMContentLoaded', init);
