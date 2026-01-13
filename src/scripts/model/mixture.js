  // ----------------------------
  // Mixture modelling (slots)
  // ----------------------------
import {$, state, clamp, yieldToUI, aggregateByColon, styleTableEl, escapeHTML} from '../main.js'
  function fastMonteCarloSolver(targetVecScaled, sourceVecsScaled, slots, cyclesMult,) {
    
    const nSources = sourceVecsScaled.length;
    const dim = targetVecScaled.length;

    // diff[i] = source[i] - target
    const diffs = new Array(nSources);
    for (let i = 0; i < nSources; i++) {
      const src = sourceVecsScaled[i];
      const diff = new Array(dim);
      for (let j = 0; j < dim; j++) diff[j] = src[j] - targetVecScaled[j];
      diffs[i] = diff;
    }

    // Initialise random slot assignment
    const currentSlots = new Array(slots);
    for (let i = 0; i < slots; i++) currentSlots[i] = Math.floor(Math.random() * nSources);

    // currentPoint = sum(diffs[currentSlots[i]])
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
        if (nSources > 1) {
          while (newIdx === oldIdx) newIdx = Math.floor(Math.random() * nSources);
        }

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

export async function runMixture(single) {
    if (!state.loaded) return;

    state._mixJobId = (state._mixJobId || 0) + 1;
    const jobId = state._mixJobId;
    // NOTE: UI labels match Vahaduo: Slots = result granularity (~1000); Cycles = iteration multiplier (large)
    const slots = (parseInt(($('slots') ? $('slots').value : '10000') || '3', 10), 100, 100000);
    const rawValue = $('cycles')?.value ?? '6000';
    const cycles = Math.max(100, Math.min(100000, Number.parseInt(rawValue, 10)));

    const printZeroes = $('printZeroes').value === 'yes';
    const doAgg = $('aggregate').value === 'yes';
    const doImportance = $('mixImportance') ? $('mixImportance').checked : false;
    const impPerms = Math.max(1, parseInt(($('mixImpPerms') ? $('mixImpPerms').value : '3') || '3', 10));
    const usedOnly = ($('mixImpUsedOnly') ? $('mixImpUsedOnly').value : 'yes') === 'yes';
    const isVerbose = $('verbose').value === "yes";

    // scale by slots (Vahaduo prepares by dividing vectors by slots)
    const sNamesRaw = state.source.names.slice();
    const sVecsScaled = state.source.vectors.map(v => v.map(x => x / slots));

    const out = $('mixtureOutput');
    // Accumulate outputs; Clear button removes all previous runs.
    const runWrap = document.createElement('div');
    runWrap.className = 'run-block';
    if (isVerbose) {
      runWrap.innerHTML = `<div class="muted">Mixture run • ${escapeHTML(new Date().toLocaleString())} • slots=${slots} • cycles=${cycles} • aggregate=${doAgg ? 'yes' : 'no'} • printZeroes=${printZeroes ? 'yes' : 'no'}</div>`;
    }
    

    const singleIdx = clamp(parseInt(($('mixtureTarget') ? $('mixtureTarget').value : '0') || '0', 10), 0, Math.max(0, state.target.names.length - 1));
    const targetRows = single ? [singleIdx] : state.target.names.map((_, i) => i);
    const allResults = [];
    const distances = [];

    for (const tIdx of targetRows) {
      const tName = state.target.names[tIdx];
      const tVecScaled = state.target.vectors[tIdx].map(x => x / slots);
      const res = fastMonteCarloSolver(tVecScaled, sVecsScaled, slots, cycles);

      let names = sNamesRaw.slice();
      let values = res.weights.slice();

      // Optional permutation importance (single runs only): Δdist per source
      let deltaVals = null;   // numeric array aligned to *raw* sources initially
      let deltaCnt = null;    // 1 if computed for that source, else 0
      let impInfoEl = null;

      if (doImportance && single) {
        if (jobId !== state._mixJobId) return;

        const usedMask = usedOnly ? res.weights.map(w => w > 1e-6) : null;

        impInfoEl = document.createElement('div');
        impInfoEl.className = 'muted';
        impInfoEl.textContent = `Computing permutation importance… perms=${impPerms} ${usedOnly ? '(used sources only)' : '(all sources)'}`;
        if (isVerbose) {runWrap.appendChild(impInfoEl);}
        await yieldToUI();

        const pi = await computePermutationImportance(tVecScaled, sVecsScaled, slots, cycles, res.distance, impPerms, usedMask);
        if (jobId !== state._mixJobId) return;

        deltaVals = new Array(sNamesRaw.length).fill(0);
        deltaCnt = new Array(sNamesRaw.length).fill(0);
        for (let i = 0; i < sNamesRaw.length; i++) {
          if (pi.cnt[i] === 0) continue;
          deltaVals[i] = pi.imp[i];
          deltaCnt[i] = 1;
        }

        impInfoEl.textContent = `Permutation importance computed (perms=${impPerms}).`;
      }

      if (doAgg) {
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

      // Sort descending (by mixture weight)
      const pairs = names.map((n, i) => ({
        name: n,
        value: values[i],
        delta: (deltaVals && deltaCnt && deltaCnt[i] > 0) ? deltaVals[i] : null
      }));
      pairs.sort((a, b) => b.value - a.value);

      // Render
      const showImpCol = !!(deltaVals && deltaCnt);
      const table = document.createElement('table');
      styleTableEl(table);

      const thead = document.createElement('thead');

      // summary row
      const headRow = document.createElement('tr');
      const th = document.createElement('th');
      th.colSpan = showImpCol ? 4 : 3;
      th.innerHTML = `Target: <strong>${escapeHTML(tName)}</strong> • Distance: ${res.distance.toFixed(8)}`;
      headRow.appendChild(th);
      thead.appendChild(headRow);

      // column labels
      const labelRow = document.createElement('tr');
      labelRow.innerHTML = showImpCol
        ? `<th class="number">%</th><th>Source</th><th>Bar</th><th class="number">Imp Δdist</th>`
        : `<th class="number">%</th><th>Source</th><th>Bar</th>`;
      thead.appendChild(labelRow);

      table.appendChild(thead);

      const tbody = document.createElement('tbody');
      for (const p of pairs) {
        if (!printZeroes && p.value === 0) continue;

        const tr = document.createElement('tr');

        const tdVal = document.createElement('td');
        tdVal.className = 'number';
        tdVal.textContent = (p.value * 100).toFixed(1);

        const tdName = document.createElement('td');
        tdName.textContent = p.name;

        const tdBar = document.createElement('td');
        const outer = document.createElement('div');
        outer.className = 'w-full h-3 rounded-full bg-zinc-800/80 overflow-hidden';
        const inner = document.createElement('div');
        inner.className = 'h-full bg-cyan-400';
        inner.style.width = (clamp(p.value, 0, 1) * 100) + '%';
        outer.appendChild(inner);
        tdBar.appendChild(outer);
tr.appendChild(tdVal);
        tr.appendChild(tdName);
        tr.appendChild(tdBar);

        if (showImpCol) {
          const tdImp = document.createElement('td');
          tdImp.className = 'number';
          tdImp.textContent = (p.delta == null) ? '—' : p.delta.toFixed(8);
          tr.appendChild(tdImp);
        }

        tbody.appendChild(tr);
      }
      table.appendChild(tbody);
      if (single){runWrap.appendChild(table);}
      
      allResults.push({ target: tName, pairs, distance: res.distance });
      distances.push(res.distance);
    }

    // Multi summary matrix (if !single)
    if (!single) {
      // Build matrix: rows targets, columns sources (sorted by avg contribution)
      // Re-run aggregation at matrix level for consistency
      const doAggMatrix = doAgg;
      const namesBase = sNamesRaw.slice();
      const weightsByTarget = [];
      for (let t = 0; t < state.target.names.length; t++) {
        const tVecScaled = state.target.vectors[t].map(x => x / slots);
        const r = fastMonteCarloSolver(tVecScaled, sVecsScaled, slots, cycles);
        weightsByTarget.push(r.weights);
      }

      // Aggregate matrix if requested
      let colNames = namesBase;
      let matrix = weightsByTarget;
      if (doAggMatrix) {
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
            const idxs = map.get(nm);
            let s = 0;
            for (const ii of idxs) s += row[ii];
            aggRow[j] = s;
          });
          return aggRow;
        });
        colNames = newNames;
        matrix = newMatrix;
      }

      // Average weights
      const nT = matrix.length;
      const nS = colNames.length;
      const avg = new Array(nS).fill(0);
      for (let t = 0; t < nT; t++) {
        for (let i = 0; i < nS; i++) avg[i] += matrix[t][i];
      }
      for (let i = 0; i < nS; i++) avg[i] /= nT;

      // sort columns by average desc
      const colOrder = Array.from({ length: nS }, (_, i) => i);
      colOrder.sort((a, b) => avg[b] - avg[a]);

      // render
      const table = document.createElement('table');
      styleTableEl(table);
      const thead = document.createElement('thead');
      const hr = document.createElement('tr');
      hr.innerHTML = `<th>Target</th><th>Distance</th>` + colOrder.map(i => `<th>${escapeHTML(colNames[i])}</th>`).join('');
      thead.appendChild(hr);
      table.appendChild(thead);
      const tbody = document.createElement('tbody');
      for (let t = 0; t < nT; t++) {
        const tr = document.createElement('tr');
        tr.innerHTML = `<td>${escapeHTML(state.target.names[t])}</td><td class="number">${distances[t].toFixed(8)}</td>`;
        for (const i of colOrder) {
          const val = matrix[t][i];
          const td = document.createElement('td');
          td.className = 'number';
          td.textContent = (val * 100).toFixed(1);
          const pct = clamp(val, 0, 1);
          // subtle shading
          td.style.backgroundColor = `rgba(11,94,215,${0.06 + pct * 0.35})`;
          td.style.color = pct > 0.55 ? '#fff' : '';
          tr.appendChild(td);
        }
        tbody.appendChild(tr);
      }
      // average row
      const avgDist = distances.reduce((a, b) => a + b, 0) / Math.max(1, distances.length);
      const trAvg = document.createElement('tr');
      trAvg.innerHTML = `<td><strong>Average</strong></td><td class="number"><strong>${avgDist.toFixed(8)}</strong></td>`;
      for (const i of colOrder) {
        const val = avg[i];
        const td = document.createElement('td');
        td.className = 'number';
        td.textContent = (val * 100).toFixed(1);
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
      out.prepend(wrap);
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
