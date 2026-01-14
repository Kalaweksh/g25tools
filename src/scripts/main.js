/*
  Modern Admixture JS — Tabbed refactor

  Goals:
  - Vahaduo-style workflow separation:
      DATA (upload/parse) → MODEL (distance/mixture) → ANALYSIS (PCA/clustering)
  - Single shared state (no re-parsing across tabs)
  - Dependency-free (no external JS libs)

  Notes:
  - Mixture modelling uses a simplified slot-based Monte-Carlo local search inspired by Vahaduo.

*/
import * as distance from './model/distance.js'
import * as mixture from './model/mixture.js'




  // ----------------------------
  // Global state
  // ----------------------------
  export const state = {
    loaded: false,
    dimensions: 0,
    source: null, // {rows, names, vectors}
    target: null, // {rows, names, vectors}
    view: {
      tab: 'data',
      modelSubtab: 'distance',

    },
  };

  // ----------------------------
  // DOM helpers
  // ----------------------------
  export const $ = (id) => document.getElementById(id);
  const qsa = (sel) => Array.from(document.querySelectorAll(sel));

  export function setHTML(el, html) {
    el.innerHTML = html;
    // Apply Tailwind styling to any injected form controls / tables
    try {
      styleFormControls(el);
      el.querySelectorAll('table').forEach((table) => {
        styleTableEl(table);
        makeTableSortable(table);
      });
    } catch (e) {}
  }

  function show(el) { el.style.display = ''; }
  function hide(el) { el.style.display = 'none'; }

  export function clamp(x, a, b) { return Math.max(a, Math.min(b, x)); }

  // ----------------------------
  // Parsing
  // ----------------------------
  function parseCSV(text) {
    const lines = text.trim().split(/\r?\n/);
    const rows = [];
    let cols = null;
    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed) continue;
      const parts = trimmed.split(',');
      if (!cols) cols = parts.length;
      if (parts.length !== cols) throw new Error('Inconsistent column count (expected ' + cols + ', got ' + parts.length + ').');
      const name = parts[0].trim();
      if (!name) throw new Error('Empty sample name encountered.');
      const vec = parts.slice(1).map((v) => {
        const n = Number(v);
        if (!Number.isFinite(n)) throw new Error('Non-numeric value: ' + v);
        return n;
      });
      rows.push([name, ...vec]);
    }
    if (!rows.length) throw new Error('No rows found.');
    return rows;
  }

  function rowsToData(rows) {
    const names = rows.map(r => r[0]);
    const vectors = rows.map(r => r.slice(1));
    return { rows, names, vectors };
  }

  function validateWorkspace(sourceRows, targetRows) {
    const sDim = sourceRows[0].length - 1;
    const tDim = targetRows[0].length - 1;
    if (sDim !== tDim) {
      throw new Error(`Dimension mismatch: source has ${sDim}, target has ${tDim}.`);
    }
    if (sDim < 2) throw new Error('Need at least 2 dimensions.');
    return sDim;
  }

  // Aggregate names by splitting on ':' (Vahaduo convention)
  export function aggregateByColon(names, values) {
    const map = new Map();
    for (let i = 0; i < names.length; i++) {
      const key = String(names[i]).split(':')[0];
      map.set(key, (map.get(key) || 0) + values[i]);
    }
    const aggNames = Array.from(map.keys());
    const aggValues = aggNames.map(n => map.get(n));
    return { names: aggNames, values: aggValues };
  }

  // ----------------------------
  // File binding
  // ----------------------------
  function bindFileInput(fileInputId, textAreaId, previewId) {
    const fileInput = $(fileInputId);
    const textArea = $(textAreaId);
    const preview = $(previewId);
    fileInput.addEventListener('change', (e) => {
      const file = e.target.files && e.target.files[0];
      if (!file) return;
      const reader = new FileReader();
      reader.onload = (ev) => {
        textArea.value = String(ev.target.result || '').trim();
        renderPreview(textArea.value, preview);
      };
      reader.readAsText(file);
    });
    textArea.addEventListener('input', () => {
      renderPreview(textArea.value, preview);
    });
  }

  function renderPreview(text, el) {
    const t = (text || '').trim();
    if (!t) { setHTML(el, '<span class="muted">(empty)</span>'); return; }
    try {
      const rows = parseCSV(t);
      const head = rows.slice(0, 5);
      const cols = head[0].length;
      let html = `<div class="muted">${rows.length} rows • ${cols - 1} dims</div>`;
      html += '<div class="mini"><table><thead><tr>';
      html += '<th>Name</th>';
      html += '<th>…</th>';
      html += '</tr></thead><tbody>';
      for (const r of head) {
        html += `<tr><td>${escapeHTML(r[0])}</td><td class="muted">${escapeHTML(r.slice(1, Math.min(r.length, 4)).join(', '))}${r.length > 4 ? ', …' : ''}</td></tr>`;
      }
      html += '</tbody></table></div>';
      setHTML(el, html);
    } catch (err) {
      setHTML(el, `<span class="muted">Preview error:</span> <span class="notice-inline">${escapeHTML(err.message)}</span>`);
    }
  }

  export function escapeHTML(s) {
    return String(s)
      .replaceAll('&', '&amp;')
      .replaceAll('<', '&lt;')
      .replaceAll('>', '&gt;')
      .replaceAll('"', '&quot;')
      .replaceAll("'", '&#39;');
  }

  // ----------------------------
  // Tabs
  // ----------------------------
  function setActiveTab(tabName) {
    state.view.tab = tabName;
    qsa('.tab-btn').forEach(btn => {
      const active = btn.dataset.tab === tabName;
      btn.classList.toggle('active', active);
      twActiveTabStyles(btn, active);
      btn.setAttribute('aria-selected', active ? 'true' : 'false');
    });
    qsa('.tab-panel').forEach(panel => panel.classList.remove('active'));
    $('tab-' + tabName).classList.add('active');
  }

  function setActiveSubtab(group, subName) {
    if (group === 'model') state.view.modelSubtab = subName;
    if (group === 'analysis') state.view.analysisSubtab = subName;

    const btnSel = group === 'model' ? '#tab-model .subtab-btn' : '#tab-analysis .subtab-btn';
    const panelPrefix = group === 'model' ? 'subtab-' : 'subtab-';
    const panelContainer = group === 'model' ? $('tab-model') : $('tab-analysis');

    qsa(btnSel).forEach(btn => {
      const active = btn.dataset.subtab === subName;
      btn.classList.toggle('active', active);
      twActiveSubtabStyles(btn, active);
    });
    qsa(group === 'model' ? '#tab-model .subtab-panel' : '#tab-analysis .subtab-panel').forEach(p => p.classList.remove('active'));
    $(panelPrefix + subName).classList.add('active');


  }
  
  // ----------------------------
  // Presets
  // ----------------------------
  function applyMixturePreset(preset) {
    // Vahaduo semantics:
    //   Slots = number of sources in the mixture (small, 1–3)
    //   Cycles = iteration multiplier (large)
    const cyclesMultEl = $('cycles');   // UI label: Cycles
    const slotsEl = $('slots');       // UI label: Slots
    const permsEl = $('mixImpPerms');
    if (!cyclesMultEl || !slotsEl) return;

    if (preset === 'fast') {
      cyclesMultEl.value = '2';
      slotsEl.value = '2500';
      if (permsEl) permsEl.value = '100';
    } else if (preset === 'thorough') {
      cyclesMultEl.value = '2000';
      slotsEl.value = '12000';
      if (permsEl) permsEl.value = '4';
    } else if (preset === 'balanced') {
      cyclesMultEl.value = '400';
      slotsEl.value = '6000';
      if (permsEl) permsEl.value = '3';
    }
  }

// ----------------------------
  // Workspace load / reset
  // ----------------------------
  function loadWorkspace() {
    hide($('dataErrors'));
    try {
      const sText = $('sourceData').value;
      const tText = $('targetData').value;
      const sRows = parseCSV(sText);
      const tRows = parseCSV(tText);
      const dims = validateWorkspace(sRows, tRows);
      state.dimensions = dims;
      state.source = rowsToData(sRows);
      state.target = rowsToData(tRows);
      state.loaded = true;
      updateWorkspaceUI();
      distance.hydrateModelControls();
      // Gate panels
      hide($('modelGate'));
      show($('modelContent'));

      // Update status badge
      const badge = $('dataStatus');
      badge.classList.remove('badge-warn');
      badge.classList.add('badge-ok');
      badge.textContent = 'Data: loaded';
      badge.title = `Loaded: ${state.source.names.length} sources, ${state.target.names.length} targets, ${state.dimensions} dims`;
    } catch (err) {
      const el = $('dataErrors');
      show(el);
      setHTML(el, `<strong>Could not load workspace:</strong> ${escapeHTML(err.message)}`);
      state.loaded = false;
      updateWorkspaceUI();
    }
  }

  function resetWorkspace() {
    state.loaded = false;
    state.dimensions = 0;
    state.source = null;
    state.target = null;
    updateWorkspaceUI();

    // Gate panels
    show($('modelGate'));
    hide($('modelContent'));


    // Badge
    const badge = $('dataStatus');
    badge.classList.remove('badge-ok');
    badge.classList.add('badge-warn');
    badge.textContent = 'Data: not loaded';
    badge.title = 'Load data in the DATA tab';

    // Clear outputs
    setHTML($('distanceOutput'), '');
    setHTML($('mixtureOutput'), '');
    clearCanvas();
  }

  function updateWorkspaceUI() {
    const nS = state.source ? state.source.names.length : null;
    const nT = state.target ? state.target.names.length : null;
    const d = state.dimensions || null;

    $('statSources').textContent = (nS == null) ? '—' : String(nS);
    $('statTargets').textContent = (nT == null) ? '—' : String(nT);
    $('statDims').textContent = (d == null) ? '—' : String(d);
    $('statStatus').textContent = state.loaded ? 'Loaded' : 'Not loaded';

    const dot = $('statDot');
    if (dot) {
      dot.classList.remove('bg-cyan-400/90', 'bg-rose-400/90', 'bg-amber-300/90');
      if (!state.loaded) dot.classList.add('bg-rose-400/90');
      else if (state.dimensions && state.dimensions !== 25) dot.classList.add('bg-amber-300/90');
      else dot.classList.add('bg-cyan-400/90');
    }
  }





  
  


  // ----------------------------
  // Async / cooperative variants (prevents UI freezing)
  // ----------------------------
  export function _now() { return (typeof performance !== 'undefined' ? performance.now() : Date.now()); }

  export async function yieldToUI() {
    // Yield control back to the browser so it can repaint / handle input.
    await new Promise(requestAnimationFrame);
  }

 


  // ----------------------------
  // Wire up UI
  // ----------------------------
  function init() {
    styleFormControls();
    // File bindings + live preview
    bindFileInput('sourceFile', 'sourceData', 'sourcePreview');
    bindFileInput('targetFile', 'targetData', 'targetPreview');
    renderPreview($('sourceData').value, $('sourcePreview'));
    renderPreview($('targetData').value, $('targetPreview'));

    // Primary tabs
    qsa('.tab-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        setActiveTab(btn.dataset.tab);
      });
    });

    // Subtabs
    qsa('#tab-model .subtab-btn').forEach(btn => {
      btn.addEventListener('click', () => setActiveSubtab('model', btn.dataset.subtab));
    });
    qsa('#tab-analysis .subtab-btn').forEach(btn => {
      btn.addEventListener('click', () => setActiveSubtab('analysis', btn.dataset.subtab));
    });

    // Data tab buttons
    $('clearSource').addEventListener('click', () => { $('sourceData').value = ''; renderPreview('', $('sourcePreview')); });
    $('clearTarget').addEventListener('click', () => { $('targetData').value = ''; renderPreview('', $('targetPreview')); });
    $('loadWorkspace').addEventListener('click', loadWorkspace);
    $('resetWorkspace').addEventListener('click', resetWorkspace);

    // Model: distance
    $('runDistance').addEventListener('click', distance.runDistance);
    $('clearDistance').addEventListener('click', () => setHTML($('distanceOutput'), ''));

    // Model: mixture
    $('runSingle').addEventListener('click', () => mixture.runMixture(true));
    $('runMulti').addEventListener('click', () => mixture.runMixture(false));
    $('clearMixture').addEventListener('click', () => setHTML($('mixtureOutput'), ''));



    // Initial gates
    updateWorkspaceUI();
    show($('modelGate'));
    hide($('modelContent'));



    // Mixture preset
    const mp = $('mixPreset');
    if (mp) {
      mp.addEventListener('change', () => {
        const v = mp.value;
        if (v !== 'custom') applyMixturePreset(v);
      });
    }
    // When user edits solver knobs, flip preset to custom
    ['slots','cycles','mixImpPerms'].forEach(id => {
      const el = $(id);
      if (!el) return;
      el.addEventListener('input', () => {
        const mp2 = $('mixPreset');
        if (mp2) mp2.value = 'custom';
      });
    });

}

  // ----------------------------
  // Tailwind styling helpers
  // ----------------------------
  function twActiveTabStyles(btn, active) {
    // Primary tabs
    btn.classList.toggle('bg-cyan-400', active);
    btn.classList.toggle('text-zinc-950', active);
    btn.classList.toggle('shadow', active);

    btn.classList.toggle('text-zinc-200', !active);
    btn.classList.toggle('bg-transparent', !active);
  }

  function twActiveSubtabStyles(btn, active) {
    btn.classList.toggle('bg-zinc-900', active);
    btn.classList.toggle('text-zinc-100', active);
    btn.classList.toggle('shadow', active);

    btn.classList.toggle('text-zinc-200', !active);
    btn.classList.toggle('bg-transparent', !active);
  }

  function styleFormControls(root=document) {
    const cls = 'w-full rounded-xl border border-zinc-700 bg-zinc-950/60 px-3 py-2 text-sm text-zinc-100 placeholder:text-zinc-500 focus:outline-none focus:ring-2 focus:ring-cyan-400/60 focus:border-cyan-400/60';
    root.querySelectorAll('input[type="text"], input[type="number"], input[type="file"], textarea, select').forEach(el => {
      // Don't style checkboxes/radios like text inputs
      if (el.type === 'checkbox' || el.type === 'radio') return;
      el.classList.add(...cls.split(' '));
    });
  }

  export function styleTableEl(table) {
  if (!table) return;

  const apply = () => {
    table.classList.add('w-full', 'text-sm', 'overflow-hidden', 'rounded-2xl', 'border', 'border-zinc-800', 'table-compact');
    table.querySelectorAll('thead th').forEach(th => {
      th.classList.add('bg-zinc-950', 'text-zinc-300', 'uppercase', 'tracking-wider', 'text-xs', 'font-semibold');
    });
    table.querySelectorAll('tbody td').forEach(td => {
      td.classList.add('text-zinc-100');
    });
    table.querySelectorAll('tr').forEach(tr => {
      tr.classList.add('border-b', 'border-zinc-800');
    });
  };

  // Apply now (in case the table is already populated)…
  apply();
  // …and again after the current call stack, to catch cells appended later.
  queueMicrotask(apply);
}

  function getSortableHeaderRow(table) {
    const head = table?.tHead;
    if (!head) return null;
    const rows = Array.from(head.rows);
    for (let i = rows.length - 1; i >= 0; i--) {
      const row = rows[i];
      const cells = Array.from(row.cells);
      if (cells.length <= 1) continue;
      if (cells.some(cell => cell.colSpan > 1)) continue;
      return row;
    }
    return null;
  }

  function getCellValue(cell) {
    if (!cell) return '';
    if (cell.dataset.sortValue != null) return cell.dataset.sortValue;
    return cell.textContent.trim();
  }

  function detectNumeric(values) {
    return values.every(val => val !== '' && !Number.isNaN(Number.parseFloat(val)));
  }

  export function makeTableSortable(table, options = {}) {
    if (!table || table.dataset.sortableInitialized) return;

    const fixedColumns = options.fixedColumns ?? Number(table.dataset.fixedColumns ?? 0);
    const rowSortable = options.rowSortable ?? table.dataset.rowSortable === 'true';
    const rankColumn = options.rankColumn ?? (table.dataset.rankColumn ? Number(table.dataset.rankColumn) : null);

    const headerRow = getSortableHeaderRow(table);
    if (!headerRow) return;

    const headers = Array.from(headerRow.cells);
    headers.forEach((th, index) => {
      const label = th.textContent.trim();
      if (!label) return;
      th.classList.add('sortable');
      th.setAttribute('aria-sort', 'none');
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'sort-btn';
      btn.innerHTML = `<span>${escapeHTML(label)}</span><span class="sort-indicator" aria-hidden="true">↕</span>`;
      th.textContent = '';
      th.appendChild(btn);
      btn.addEventListener('click', () => sortByColumn(index));
    });

    const tbody = table.tBodies[0];
    if (!tbody) return;

    const sortByColumn = (index) => {
      const rows = Array.from(tbody.rows);
      const fixedRows = rows.filter(row => row.dataset.fixed === 'true' || row.classList.contains('is-summary'));
      const sortableRows = rows.filter(row => !fixedRows.includes(row));

      const values = sortableRows.map(row => getCellValue(row.cells[index]));
      const isNumeric = detectNumeric(values);

      const currentDir = headerRow.dataset.sortDir || 'none';
      const currentIndex = headerRow.dataset.sortIndex ? Number(headerRow.dataset.sortIndex) : null;
      const nextDir = currentIndex === index && currentDir === 'desc' ? 'asc' : 'desc';

      sortableRows.sort((a, b) => {
        const aVal = getCellValue(a.cells[index]);
        const bVal = getCellValue(b.cells[index]);
        if (isNumeric) {
          const diff = Number.parseFloat(aVal) - Number.parseFloat(bVal);
          return nextDir === 'asc' ? diff : -diff;
        }
        return nextDir === 'asc'
          ? aVal.localeCompare(bVal)
          : bVal.localeCompare(aVal);
      });

      sortableRows.forEach(row => tbody.appendChild(row));
      fixedRows.forEach(row => tbody.appendChild(row));

      if (rankColumn != null) {
        const updatedRows = Array.from(tbody.rows).filter(row => !row.classList.contains('is-summary'));
        updatedRows.forEach((row, idx) => {
          const cell = row.cells[rankColumn];
          if (cell) cell.textContent = String(idx + 1);
        });
      }

      headers.forEach((th, idx) => {
        const dir = idx === index ? nextDir : 'none';
        th.setAttribute('aria-sort', dir === 'none' ? 'none' : dir === 'asc' ? 'ascending' : 'descending');
        const indicator = th.querySelector('.sort-indicator');
        if (indicator) indicator.textContent = dir === 'none' ? '↕' : dir === 'asc' ? '↑' : '↓';
      });

      headerRow.dataset.sortIndex = String(index);
      headerRow.dataset.sortDir = nextDir;
    };

    if (rowSortable) {
      tbody.addEventListener('click', (event) => {
        const btn = event.target.closest('.row-sort-btn');
        if (!btn) return;
        const row = btn.closest('tr');
        if (!row) return;

        const currentDir = row.dataset.sortDir || 'none';
        const nextDir = currentDir === 'desc' ? 'asc' : 'desc';
        row.dataset.sortDir = nextDir;
        btn.setAttribute('aria-pressed', nextDir === 'asc' ? 'true' : 'false');

        const baseCells = Array.from(row.cells).slice(fixedColumns);
        const values = baseCells.map(cell => getCellValue(cell));
        const isNumeric = detectNumeric(values);

        const order = baseCells.map((cell, idx) => ({
          idx: idx + fixedColumns,
          value: getCellValue(cell)
        }));

        order.sort((a, b) => {
          if (isNumeric) {
            const diff = Number.parseFloat(a.value) - Number.parseFloat(b.value);
            return nextDir === 'asc' ? diff : -diff;
          }
          return nextDir === 'asc'
            ? String(a.value).localeCompare(String(b.value))
            : String(b.value).localeCompare(String(a.value));
        });

        const rowsToUpdate = [
          ...(table.tHead ? Array.from(table.tHead.rows) : []),
          ...Array.from(tbody.rows)
        ];

        rowsToUpdate.forEach((rowEl) => {
          const cells = Array.from(rowEl.cells);
          if (cells.length <= fixedColumns) return;
          const fixed = cells.slice(0, fixedColumns);
          const sorted = order.map(item => cells[item.idx]);
          rowEl.replaceChildren(...fixed, ...sorted);
        });

        headers.forEach((th) => {
          th.setAttribute('aria-sort', 'none');
          const indicator = th.querySelector('.sort-indicator');
          if (indicator) indicator.textContent = '↕';
        });
      });
    }

    table.dataset.sortableInitialized = 'true';
  }


init();

