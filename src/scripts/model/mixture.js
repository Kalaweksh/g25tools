  // ----------------------------
  // Mixture modelling (slots)
  // ----------------------------
import {$, state, clamp, yieldToUI, aggregateByColon, styleTableEl, escapeHTML, makeTableSortable} from '../main.js'

function fastMonteCarloSolver(targetVecScaled, sourceVecsScaled, slots, cyclesMult) {
  const nSources = sourceVecsScaled.length;
  const dim = targetVecScaled.length;

  const diffs = sourceVecsScaled.map(src => src.map((v, j) => v - targetVecScaled[j]));

  const currentSlots = Array.from({length: slots}, () => Math.floor(Math.random() * nSources));
  const currentPoint = new Array(dim).fill(0);
  for (let i = 0; i < slots; i++) {
    const d = diffs[currentSlots[i]];
    for (let j = 0; j < dim; j++) currentPoint[j] += d[j];
  }

  let currentDist = 0;
  for (let j = 0; j < dim; j++) currentDist += currentPoint[j] * currentPoint[j];

  const cycles = Math.max(1, Math.ceil(nSources * cyclesMult / 4));

  for (let c = 0; c < cycles; c++) {
    for (let s = 0; s < slots; s++) {
      const oldIdx = currentSlots[s];
      let newIdx = Math.floor(Math.random() * nSources);
      if (nSources > 1) while (newIdx === oldIdx) newIdx = Math.floor(Math.random() * nSources);

      const oldVec = diffs[oldIdx];
      const newVec = diffs[newIdx];
      let newDist = 0;
      for (let j = 0; j < dim; j++) {
        const v = currentPoint[j] - oldVec[j] + newVec[j];
        newDist += v * v;
      }

      if (newDist < currentDist) {
        for (let j = 0; j < dim; j++) currentPoint[j] = currentPoint[j] - oldVec[j] + newVec[j];
        currentDist = newDist;
        currentSlots[s] = newIdx;
      }
    }
  }

  const weights = new Array(nSources).fill(0);
  for (let i = 0; i < slots; i++) weights[currentSlots[i]]++;
  for (let i = 0; i < nSources; i++) weights[i] /= slots;
  return { weights, distance: Math.sqrt(currentDist) };
}

function getMixtureConfig() {
  const slotsRaw = $('slots')?.value ?? '10000';
  const cyclesRaw = $('cycles')?.value ?? '6000';

  return {
    slots: clamp(Number.parseInt(slotsRaw, 10), 100, 100000),
    cycles: clamp(Number.parseInt(cyclesRaw, 10), 100, 100000),
    printZeroes: $('printZeroes')?.value === 'yes',
    doAgg: $('aggregate')?.value === 'yes',
    doImportance: $('mixImportance')?.checked ?? false,
    impPerms: clamp(Number.parseInt($('mixImpPerms')?.value ?? '3', 10), 1, 100000),
    usedOnly: ($('mixImpUsedOnly')?.value ?? 'yes') === 'yes',
    isVerbose: $('verbose')?.value === 'yes',
    singleIdx: clamp(Number.parseInt($('mixtureTarget')?.value ?? '0', 10), 0, Math.max(0, state.target.names.length - 1))
  };
}

function createRunWrapper(config) {
  const runWrap = document.createElement('div');
  runWrap.className = 'run-block';
  if (config.isVerbose) {
    runWrap.innerHTML = `<div class="muted">Mixture run • ${escapeHTML(new Date().toLocaleString())} • slots=${config.slots} • cycles=${config.cycles} • aggregate=${config.doAgg ? 'yes' : 'no'} • printZeroes=${config.printZeroes ? 'yes' : 'no'}</div>`;
  }
  return runWrap;
}

function renderSingleTable(result, options) {
  const {printZeroes, showImpCol} = options;
  const table = document.createElement('table');
  styleTableEl(table);

  const thead = document.createElement('thead');
  const headRow = document.createElement('tr');
  const th = document.createElement('th');
  th.colSpan = showImpCol ? 3 : 2;
  th.innerHTML = `Target: <strong>${escapeHTML(result.target)}</strong> • Distance: ${result.distance.toFixed(8)}`;
  headRow.appendChild(th);
  thead.appendChild(headRow);

  const labelRow = document.createElement('tr');
  labelRow.innerHTML = showImpCol
    ? `<th>Mix</th><th>Source</th><th class="number">Imp Δdist</th>`
    : `<th>Mix</th><th>Source</th>`;
  thead.appendChild(labelRow);

  table.appendChild(thead);
  const tbody = document.createElement('tbody');

  for (const p of result.pairs) {
    if (!printZeroes && p.value === 0) continue;

    const tr = document.createElement('tr');
    const tdMix = document.createElement('td');
    tdMix.className = 'mix-cell';
    tdMix.dataset.sortValue = String(p.value);
    const pct = document.createElement('span');
    pct.className = 'mix-value';
    pct.textContent = (p.value * 100).toFixed(1) + '%';
    const outer = document.createElement('div');
    outer.className = 'mix-bar';
    const inner = document.createElement('div');
    inner.className = 'mix-bar-fill';
    inner.style.width = (clamp(p.value, 0, 1) * 100) + '%';
    outer.appendChild(inner);
    tdMix.appendChild(pct);
    tdMix.appendChild(outer);

    const tdName = document.createElement('td');
    tdName.textContent = p.name;

    tr.appendChild(tdMix);
    tr.appendChild(tdName);

    if (showImpCol) {
    const tdImp = document.createElement('td');
    tdImp.className = 'number';
    if (p.delta == null) {
      tdImp.textContent = '—';
    } else {
      tdImp.textContent = p.delta.toFixed(8);
      tdImp.dataset.sortValue = String(p.delta);
    }
    tr.appendChild(tdImp);
    }

    tbody.appendChild(tr);
  }

  table.appendChild(tbody);
  makeTableSortable(table);
  return table;
}

function renderMultiSummary({distances, matrix, colNames, colOrder, avg, avgDist}) {
  const table = document.createElement('table');
  table.dataset.fixedColumns = '2';
  table.dataset.rowSortable = 'true';
  styleTableEl(table);

  const thead = document.createElement('thead');
  const hr = document.createElement('tr');
  hr.innerHTML = `<th>Target</th><th>Distance</th>` + colOrder.map(i => `<th>${escapeHTML(colNames[i])}</th>`).join('');
  thead.appendChild(hr);
  table.appendChild(thead);

  const tbody = document.createElement('tbody');
  for (let t = 0; t < matrix.length; t++) {
    const tr = document.createElement('tr');
    const targetCell = document.createElement('td');
    const targetName = document.createElement('span');
    targetName.textContent = state.target.names[t];
    const rowSortBtn = document.createElement('button');
    rowSortBtn.type = 'button';
    rowSortBtn.className = 'row-sort-btn';
    rowSortBtn.setAttribute('aria-pressed', 'false');
    rowSortBtn.setAttribute('title', 'Sort this row by values');
    rowSortBtn.setAttribute('aria-label', 'Sort this row by values');
    rowSortBtn.innerHTML = '<span aria-hidden="true">⇄</span>';
    targetCell.appendChild(targetName);
    targetCell.appendChild(rowSortBtn);
    const distCell = document.createElement('td');
    distCell.className = 'number';
    distCell.textContent = distances[t].toFixed(8);
    tr.appendChild(targetCell);
    tr.appendChild(distCell);
    for (const i of colOrder) {
      const val = matrix[t][i];
      const td = document.createElement('td');
      td.className = 'number';
      td.textContent = (val * 100).toFixed(1);
      td.dataset.sortValue = String(val);
      const pct = clamp(val, 0, 1);
      td.style.backgroundColor = `rgba(11,94,215,${0.06 + pct * 0.35})`;
      td.style.color = pct > 0.55 ? '#fff' : '';
      tr.appendChild(td);
    }
    tbody.appendChild(tr);
  }

  const trAvg = document.createElement('tr');
  trAvg.classList.add('is-summary');
  trAvg.dataset.fixed = 'true';
  trAvg.innerHTML = `<td><strong>Average</strong></td><td class="number"><strong>${avgDist.toFixed(8)}</strong></td>`;
  for (const i of colOrder) {
    const val = avg[i];
    const td = document.createElement('td');
    td.className = 'number';
    td.textContent = (val * 100).toFixed(1);
    td.dataset.sortValue = String(val);
    const pct = clamp(val, 0, 1);
    td.style.backgroundColor = `rgba(11,94,215,${0.06 + pct * 0.35})`;
    td.style.color = pct > 0.55 ? '#fff' : '';
    trAvg.appendChild(td);
  }
  tbody.appendChild(trAvg);
  table.appendChild(tbody);

  const wrap = document.createElement('div');
  wrap.className = 'card';
  wrap.innerHTML = '<h3>Multi summary matrix</h3>';
  wrap.appendChild(table);
  makeTableSortable(table, {fixedColumns: 2, rowSortable: true});
  return wrap;
}

async function computeMixtureResults(single, config, jobId, onImportance) {
  const sNamesRaw = state.source.names.slice();
  const sVecsScaled = state.source.vectors.map(v => v.map(x => x / config.slots));
  const targetRows = single ? [config.singleIdx] : state.target.names.map((_, i) => i);
  const results = [];
  const distances = [];
  const weightsByTarget = [];

  for (const tIdx of targetRows) {
    const tName = state.target.names[tIdx];
    const tVecScaled = state.target.vectors[tIdx].map(x => x / config.slots);
    const res = fastMonteCarloSolver(tVecScaled, sVecsScaled, config.slots, config.cycles);

    let names = sNamesRaw.slice();
    let values = res.weights.slice();
    weightsByTarget.push(res.weights.slice());

    let deltaVals = null;
    let deltaCnt = null;

    if (config.doImportance && single) {
      if (jobId !== state._mixJobId) return null;
      const usedMask = config.usedOnly ? res.weights.map(w => w > 1e-6) : null;
      if (onImportance) onImportance(`Computing permutation importance… perms=${config.impPerms} ${config.usedOnly ? '(used sources only)' : '(all sources)'}`);
      await yieldToUI();

      const pi = await computePermutationImportance(tVecScaled, sVecsScaled, config.slots, config.cycles, res.distance, config.impPerms, usedMask);
      if (jobId !== state._mixJobId) return null;

      deltaVals = new Array(sNamesRaw.length).fill(0);
      deltaCnt = new Array(sNamesRaw.length).fill(0);
      for (let i = 0; i < sNamesRaw.length; i++) {
        if (pi.cnt[i] === 0) continue;
        deltaVals[i] = pi.imp[i];
        deltaCnt[i] = 1;
      }
      if (onImportance) onImportance(`Permutation importance computed (perms=${config.impPerms}).`);
    }

    if (config.doAgg) {
      const agg = aggregateByColon(names, values);
      names = agg.names;
      values = agg.values;

      if (deltaVals && deltaCnt) {
        const aD = aggregateByColon(sNamesRaw.slice(), deltaVals);
        const aC = aggregateByColon(sNamesRaw.slice(), deltaCnt);
        deltaVals = aD.values;
        deltaCnt = aC.values;
      }
    }

    const pairs = names.map((n, i) => ({
      name: n,
      value: values[i],
      delta: (deltaVals && deltaCnt && deltaCnt[i] > 0) ? deltaVals[i] : null
    }));
    pairs.sort((a, b) => b.value - a.value);

    results.push({ target: tName, pairs, distance: res.distance, showImpCol: !!(deltaVals && deltaCnt) });
    distances.push(res.distance);
  }

  return { results, distances, weightsByTarget, sNamesRaw };
}

function buildMultiSummary(weightsByTarget, sNamesRaw, distances, doAgg) {
  let colNames = sNamesRaw.slice();
  let matrix = weightsByTarget.map(row => row.slice());

  if (doAgg) {
    const map = new Map();
    for (let i = 0; i < colNames.length; i++) {
      const key = colNames[i].split(':')[0];
      if (!map.has(key)) map.set(key, []);
      map.get(key).push(i);
    }

    const newNames = Array.from(map.keys());
    const newMatrix = matrix.map(row => {
      const aggRow = new Array(newNames.length).fill(0);
      newNames.forEach((nm, j) => {
        let s = 0;
        for (const ii of map.get(nm)) s += row[ii];
        aggRow[j] = s;
      });
      return aggRow;
    });

    colNames = newNames;
    matrix = newMatrix;
  }

  const nT = matrix.length;
  const nS = colNames.length;
  const avg = new Array(nS).fill(0);
  for (let t = 0; t < nT; t++) {
    for (let i = 0; i < nS; i++) avg[i] += matrix[t][i];
  }
  for (let i = 0; i < nS; i++) avg[i] /= nT;

  const colOrder = Array.from({ length: nS }, (_, i) => i);
  colOrder.sort((a, b) => avg[b] - avg[a]);

  const avgDist = distances.reduce((a, b) => a + b, 0) / Math.max(1, distances.length);
  return { matrix, colNames, colOrder, avg, avgDist };
}

export async function runMixture(single) {
  if (!state.loaded) return;

  state._mixJobId = (state._mixJobId || 0) + 1;
  const jobId = state._mixJobId;
  const config = getMixtureConfig();

  const out = $('mixtureOutput');
  const runWrap = createRunWrapper(config);
  let impInfoEl = null;
  const onImportance = config.isVerbose ? (text) => {
    if (!impInfoEl) {
      impInfoEl = document.createElement('div');
      impInfoEl.className = 'muted';
      runWrap.appendChild(impInfoEl);
    }
    impInfoEl.textContent = text;
  } : null;

  const resultSet = await computeMixtureResults(single, config, jobId, onImportance);
  if (!resultSet) return;

  for (const result of resultSet.results) {
    if (single) {
      runWrap.appendChild(renderSingleTable(result, {printZeroes: config.printZeroes, showImpCol: result.showImpCol}));
    }
  }

  if (!single) {
    const summary = buildMultiSummary(resultSet.weightsByTarget, resultSet.sNamesRaw, resultSet.distances, config.doAgg);
    out.prepend(renderMultiSummary({
      distances: resultSet.distances,
      matrix: summary.matrix,
      colNames: summary.colNames,
      colOrder: summary.colOrder,
      avg: summary.avg,
      avgDist: summary.avgDist
    }));
  }

  out.prepend(runWrap);
}


   // ----------------------------
  // Mixture permutation importance
  // ----------------------------
function lcg(seed) {
  let s = seed >>> 0;
  return function() {
    s = (1664525 * s + 1013904223) >>> 0;
    return s / 4294967296;
  };
}

function permuteVector(v, rand) {
  const a = v.slice();
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    const tmp = a[i];
    a[i] = a[j];
    a[j] = tmp;
  }
  return a;
}

async function computePermutationImportance(tVecScaled, sVecsScaled, slots, cycles, baseDist, perms, usedOnlyMask) {
  const nS = sVecsScaled.length;
  const imp = new Array(nS).fill(0);
  const cnt = new Array(nS).fill(0);

  for (let i = 0; i < nS; i++) {
    if (usedOnlyMask && !usedOnlyMask[i]) continue;

    const rand = lcg(123456789 + i * 97);
    let sum = 0;
    for (let p = 0; p < perms; p++) {
      const sv = sVecsScaled.slice();
      sv[i] = permuteVector(sVecsScaled[i], rand);

      const r = fastMonteCarloSolver(tVecScaled, sv, slots, cycles);
      sum += (r.distance - baseDist);

      if ((p % 2) === 1) await yieldToUI();
    }
    imp[i] = sum / perms;
    cnt[i] = perms;
    await yieldToUI();
  }

  return { imp, cnt };
}
