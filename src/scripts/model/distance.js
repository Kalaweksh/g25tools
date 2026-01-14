// ----------------------------
// Distance modelling
// ----------------------------
import {$, state, clamp, styleTableEl, escapeHTML, makeTableSortable} from '../main.js'
export function euclidean(a, b) {
let s = 0;
for (let i = 0; i < a.length; i++) {
    const d = a[i] - b[i];
    s += d * d;
}
return Math.sqrt(s);
}

export function hydrateModelControls() {
// Populate target dropdown for distance
const sel = $('distanceTarget');
if (sel) {
    sel.innerHTML = '';
    for (const name of state.target.names) {
    const opt = document.createElement('option');
    opt.value = name;
    opt.textContent = name;
    sel.appendChild(opt);
    }
}

// Populate target dropdown for mixture (single runs)
const selM = $('mixtureTarget');
if (selM) {
    const prev = selM.value;
    selM.innerHTML = '';
    for (let i = 0; i < state.target.names.length; i++) {
    const opt = document.createElement('option');
    opt.value = String(i);
    opt.textContent = state.target.names[i];
    selM.appendChild(opt);
    }
    if (prev && prev !== '') selM.value = prev;
}
}

export function runDistance() {
if (!state.loaded) return;
const targetName = $('distanceTarget').value;
const topN = clamp(parseInt($('distanceTopN').value || '25', 10), 1, 100000);
const doAgg = $('distanceAggregate').value === 'yes';

const tIdx = state.target.names.indexOf(targetName);
if (tIdx < 0) return;
const tVec = state.target.vectors[tIdx];

// Distances to each source
let names = state.source.names.slice();
let dists = state.source.vectors.map(sv => euclidean(tVec, sv));

// Optional aggregation by ':' — for distances we aggregate by min distance (closest representative)
if (doAgg) {
    const map = new Map(); // name -> min dist
    for (let i = 0; i < names.length; i++) {
    const key = names[i].split(':')[0];
    const prev = map.get(key);
    const di = dists[i];
    map.set(key, prev == null ? di : Math.min(prev, di));
    }
    names = Array.from(map.keys());
    dists = names.map(n => map.get(n));
}

const pairs = names.map((n, i) => ({ name: n, dist: dists[i] }));
pairs.sort((a, b) => a.dist - b.dist);

const shown = pairs.slice(0, Math.min(topN, pairs.length));
const table = document.createElement('table');
styleTableEl(table);
const thead = document.createElement('thead');
thead.innerHTML = `<tr><th colspan="3">Target: <strong>${escapeHTML(targetName)}</strong> • Showing ${shown.length}/${pairs.length}</th></tr>`;
const labelRow = document.createElement('tr');
labelRow.innerHTML = '<th class="number">Rank</th><th>Source</th><th class="number">Distance</th>';
thead.appendChild(labelRow);
table.appendChild(thead);

const tbody = document.createElement('tbody');
const distValues = shown.map(p => p.dist);
const minDist = Math.min(...distValues);
const maxDist = Math.max(...distValues);
const span = Math.max(1e-9, maxDist - minDist);
for (const p of shown) {
    const tr = document.createElement('tr');
    const tdRank = document.createElement('td');
    tdRank.className = 'number';
    tdRank.textContent = String(shown.indexOf(p) + 1);
    const tdName = document.createElement('td');
    tdName.textContent = p.name;
    const tdDist = document.createElement('td');
    tdDist.className = 'number';
    tdDist.textContent = p.dist.toFixed(8);
    const ratio = (p.dist - minDist) / span;
    const baseAlpha = 0.08 + ratio * 0.2;
    const strongAlpha = 0.15 + ratio * 0.45;
    const gradientBase = `linear-gradient(90deg, rgba(34,211,238,${baseAlpha}), rgba(14,116,144,${baseAlpha}))`;
    tdRank.style.backgroundImage = gradientBase;
    tdName.style.backgroundImage = gradientBase;
    tdDist.style.backgroundImage = `linear-gradient(90deg, rgba(34,211,238,0.08), rgba(14,116,144,${strongAlpha}))`;
    tr.appendChild(tdRank);
    tr.appendChild(tdName);
    tr.appendChild(tdDist);
    tbody.appendChild(tr);
}
table.appendChild(tbody);
makeTableSortable(table);
const out = $('distanceOutput');
// Accumulate outputs; Clear button removes all previous runs.
const wrap = document.createElement('div');
wrap.className = 'run-block';
wrap.innerHTML = `<div class="muted">Distance run • ${escapeHTML(new Date().toLocaleString())} • target=${escapeHTML(targetName)} • topN=${topN} • aggregate=${doAgg ? 'yes' : 'no'}</div>`;
wrap.appendChild(table);
out.prepend(wrap);
}
