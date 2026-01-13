// ----------------------------
// Distance modelling
// ----------------------------
import {$, state, clamp, styleTableEl, escapeHTML} from '../main.js'
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
table.appendChild(thead);

const tbody = document.createElement('tbody');
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
    tr.appendChild(tdRank);
    tr.appendChild(tdName);
    tr.appendChild(tdDist);
    tbody.appendChild(tr);
}
table.appendChild(tbody);
const out = $('distanceOutput');
// Accumulate outputs; Clear button removes all previous runs.
const wrap = document.createElement('div');
wrap.className = 'run-block';
wrap.innerHTML = `<div class="muted">Distance run • ${escapeHTML(new Date().toLocaleString())} • target=${escapeHTML(targetName)} • topN=${topN} • aggregate=${doAgg ? 'yes' : 'no'}</div>`;
wrap.appendChild(table);
out.prepend(wrap);
}
