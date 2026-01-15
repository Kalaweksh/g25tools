// ----------------------------
// Gaussian Mixture Model (EM)
// ----------------------------
import {$, state, clamp, styleTableEl, escapeHTML, makeTableSortable, yieldToUI} from '../main.js'

const DEFAULTS = {
  k: 3,
  maxIter: 100,
  tol: 1e-4,
  covarianceType: 'diag',
  varianceFloor: 1e-6,
  criterion: 'none',
  minK: 2,
  maxK: 8,
  weightPrior: 1.1,
  meanPriorStrength: 0.01,
  covPriorScale: 1,
  covPriorDf: 2,
};

function randChoice(arr) {
  return arr[Math.floor(Math.random() * arr.length)];
}

function deepCopyMatrix(mat) {
  return mat.map(row => row.slice());
}

function meanVector(points) {
  const n = points.length;
  const d = points[0].length;
  const mean = new Array(d).fill(0);
  for (const p of points) {
    for (let j = 0; j < d; j++) mean[j] += p[j];
  }
  for (let j = 0; j < d; j++) mean[j] /= n;
  return mean;
}

function varianceVector(points, mean, floor) {
  const n = points.length;
  const d = points[0].length;
  const vars = new Array(d).fill(0);
  for (const p of points) {
    for (let j = 0; j < d; j++) {
      const diff = p[j] - mean[j];
      vars[j] += diff * diff;
    }
  }
  for (let j = 0; j < d; j++) vars[j] = Math.max(vars[j] / n, floor);
  return vars;
}

function covarianceMatrix(points, mean, floor) {
  const n = points.length;
  const d = points[0].length;
  const cov = Array.from({length: d}, () => new Array(d).fill(0));
  for (const p of points) {
    const diff = new Array(d);
    for (let j = 0; j < d; j++) diff[j] = p[j] - mean[j];
    for (let r = 0; r < d; r++) {
      for (let c = 0; c < d; c++) {
        cov[r][c] += diff[r] * diff[c];
      }
    }
  }
  for (let r = 0; r < d; r++) {
    for (let c = 0; c < d; c++) cov[r][c] /= n;
    cov[r][r] = Math.max(cov[r][r], floor);
  }
  return cov;
}

function kmeansPlusPlusInit(points, k) {
  const centers = [];
  centers.push(randChoice(points).slice());
  while (centers.length < k) {
    const distances = points.map(p => {
      let best = Infinity;
      for (const c of centers) {
        let s = 0;
        for (let j = 0; j < p.length; j++) {
          const d = p[j] - c[j];
          s += d * d;
        }
        if (s < best) best = s;
      }
      return best;
    });
    const total = distances.reduce((a, b) => a + b, 0);
    let r = Math.random() * total;
    let idx = 0;
    for (let i = 0; i < distances.length; i++) {
      r -= distances[i];
      if (r <= 0) { idx = i; break; }
    }
    centers.push(points[idx].slice());
  }
  return centers;
}

function invertMatrixAndLogDet(matrix) {
  const n = matrix.length;
  const a = matrix.map(row => row.slice());
  const inv = Array.from({length: n}, (_, i) => {
    const row = new Array(n).fill(0);
    row[i] = 1;
    return row;
  });
  let logDet = 0;

  for (let i = 0; i < n; i++) {
    let pivotRow = i;
    let pivotVal = Math.abs(a[i][i]);
    for (let r = i + 1; r < n; r++) {
      const val = Math.abs(a[r][i]);
      if (val > pivotVal) {
        pivotVal = val;
        pivotRow = r;
      }
    }

    if (pivotVal === 0) {
      a[i][i] = 1e-12;
      pivotVal = Math.abs(a[i][i]);
    }

    if (pivotRow !== i) {
      [a[i], a[pivotRow]] = [a[pivotRow], a[i]];
      [inv[i], inv[pivotRow]] = [inv[pivotRow], inv[i]];
    }

    const pivot = a[i][i];
    logDet += Math.log(Math.abs(pivot));

    for (let c = 0; c < n; c++) {
      a[i][c] /= pivot;
      inv[i][c] /= pivot;
    }

    for (let r = 0; r < n; r++) {
      if (r === i) continue;
      const factor = a[r][i];
      if (factor === 0) continue;
      for (let c = 0; c < n; c++) {
        a[r][c] -= factor * a[i][c];
        inv[r][c] -= factor * inv[i][c];
      }
    }
  }

  return {inverse: inv, logDet};
}

function logGaussianDiag(x, mean, variance) {
  const d = x.length;
  let quad = 0;
  let logDet = 0;
  for (let j = 0; j < d; j++) {
    const diff = x[j] - mean[j];
    quad += (diff * diff) / variance[j];
    logDet += Math.log(variance[j]);
  }
  return -0.5 * (d * Math.log(2 * Math.PI) + logDet + quad);
}

function logGaussianFull(x, mean, covInv, logDet) {
  const d = x.length;
  const diff = new Array(d);
  for (let j = 0; j < d; j++) diff[j] = x[j] - mean[j];

  const temp = new Array(d).fill(0);
  for (let r = 0; r < d; r++) {
    let s = 0;
    for (let c = 0; c < d; c++) s += covInv[r][c] * diff[c];
    temp[r] = s;
  }
  let quad = 0;
  for (let j = 0; j < d; j++) quad += diff[j] * temp[j];

  return -0.5 * (d * Math.log(2 * Math.PI) + logDet + quad);
}

function expectation(points, weights, means, covariances, covarianceType, cachedInverses) {
  const n = points.length;
  const k = weights.length;
  const responsibilities = Array.from({length: n}, () => new Array(k).fill(0));
  let logLikelihood = 0;

  for (let i = 0; i < n; i++) {
    const x = points[i];
    const logProbs = new Array(k);
    let maxLog = -Infinity;

    for (let c = 0; c < k; c++) {
      let logGauss = 0;
      if (covarianceType === 'full') {
        logGauss = logGaussianFull(x, means[c], cachedInverses[c].inverse, cachedInverses[c].logDet);
      } else {
        logGauss = logGaussianDiag(x, means[c], covariances[c]);
      }
      const logProb = Math.log(weights[c]) + logGauss;
      logProbs[c] = logProb;
      if (logProb > maxLog) maxLog = logProb;
    }

    let sumExp = 0;
    for (let c = 0; c < k; c++) sumExp += Math.exp(logProbs[c] - maxLog);
    const logSum = Math.log(sumExp) + maxLog;
    logLikelihood += logSum;

    for (let c = 0; c < k; c++) {
      responsibilities[i][c] = Math.exp(logProbs[c] - logSum);
    }
  }

  return {responsibilities, logLikelihood};
}

function maximization(points, responsibilities, covarianceType, varianceFloor, priors) {
  const n = points.length;
  const d = points[0].length;
  const k = responsibilities[0].length;

  const nk = new Array(k).fill(0);
  for (let i = 0; i < n; i++) {
    for (let c = 0; c < k; c++) nk[c] += responsibilities[i][c];
  }

  const means = Array.from({length: k}, () => new Array(d).fill(0));
  const xbars = Array.from({length: k}, () => new Array(d).fill(0));
  for (let i = 0; i < n; i++) {
    const x = points[i];
    for (let c = 0; c < k; c++) {
      const r = responsibilities[i][c];
      if (r === 0) continue;
      for (let j = 0; j < d; j++) xbars[c][j] += r * x[j];
    }
  }
  for (let c = 0; c < k; c++) {
    const denom = nk[c] || 1;
    for (let j = 0; j < d; j++) xbars[c][j] /= denom;
  }

  let covariances = null;
  const kappa0 = priors.meanPriorStrength;
  const mu0 = priors.meanPrior;
  const alphaWeight = priors.weightPrior;
  if (covarianceType === 'full') {
    covariances = Array.from({length: k}, () => Array.from({length: d}, () => new Array(d).fill(0)));
    const scatter = Array.from({length: k}, () => Array.from({length: d}, () => new Array(d).fill(0)));
    for (let i = 0; i < n; i++) {
      const x = points[i];
      for (let c = 0; c < k; c++) {
        const r = responsibilities[i][c];
        if (r === 0) continue;
        for (let rIdx = 0; rIdx < d; rIdx++) {
          const diffR = x[rIdx] - xbars[c][rIdx];
          for (let cIdx = 0; cIdx < d; cIdx++) {
            scatter[c][rIdx][cIdx] += r * diffR * (x[cIdx] - xbars[c][cIdx]);
          }
        }
      }
    }
    for (let c = 0; c < k; c++) {
      const denom = nk[c] || 0;
      const kappaN = kappa0 + denom;
      for (let j = 0; j < d; j++) {
        means[c][j] = (kappa0 * mu0[j] + denom * xbars[c][j]) / Math.max(1e-12, kappaN);
      }
      const diff = new Array(d);
      for (let j = 0; j < d; j++) diff[j] = xbars[c][j] - mu0[j];
      const scale = (kappa0 * denom) / Math.max(1e-12, kappaN);
      const psi = Array.from({length: d}, (_, rIdx) => {
        const row = new Array(d).fill(0);
        for (let cIdx = 0; cIdx < d; cIdx++) {
          row[cIdx] = priors.covPrior[rIdx][cIdx] + scatter[c][rIdx][cIdx] + scale * diff[rIdx] * diff[cIdx];
        }
        return row;
      });
      const nuN = priors.covPriorNu + denom;
      const denomCov = Math.max(1e-12, nuN + d + 1);
      for (let rIdx = 0; rIdx < d; rIdx++) {
        for (let cIdx = 0; cIdx < d; cIdx++) covariances[c][rIdx][cIdx] = psi[rIdx][cIdx] / denomCov;
        covariances[c][rIdx][rIdx] = Math.max(covariances[c][rIdx][rIdx], varianceFloor);
      }
    }
  } else {
    covariances = Array.from({length: k}, () => new Array(d).fill(varianceFloor));
    const scatter = Array.from({length: k}, () => new Array(d).fill(0));
    for (let i = 0; i < n; i++) {
      const x = points[i];
      for (let c = 0; c < k; c++) {
        const r = responsibilities[i][c];
        if (r === 0) continue;
        for (let j = 0; j < d; j++) {
          const diff = x[j] - xbars[c][j];
          scatter[c][j] += r * diff * diff;
        }
      }
    }
    for (let c = 0; c < k; c++) {
      const denom = nk[c] || 0;
      const kappaN = kappa0 + denom;
      for (let j = 0; j < d; j++) {
        means[c][j] = (kappa0 * mu0[j] + denom * xbars[c][j]) / Math.max(1e-12, kappaN);
      }
      for (let j = 0; j < d; j++) {
        const diff = xbars[c][j] - mu0[j];
        const alphaN = priors.varPriorAlpha + denom / 2;
        const betaN = priors.varPriorBeta[j]
          + 0.5 * scatter[c][j]
          + (kappa0 * denom / Math.max(1e-12, kappaN)) * (diff * diff) / 2;
        const mode = betaN / Math.max(1e-12, alphaN + 1);
        covariances[c][j] = Math.max(mode, varianceFloor);
      }
    }
  }

  const weights = nk.map(val => Math.max((val + alphaWeight) / (n + k * alphaWeight), 1e-12));
  return {means, covariances, weights};
}

function computeModelParamsCount(k, d, covarianceType) {
  const meanParams = k * d;
  const weightParams = k - 1;
  if (covarianceType === 'full') {
    const covParams = k * (d * (d + 1) / 2);
    return meanParams + covParams + weightParams;
  }
  const covParams = k * d;
  return meanParams + covParams + weightParams;
}

export function runGMM(points, options = {}) {
  const opts = {...DEFAULTS, ...options};
  const n = points.length;
  const d = points[0].length;
  const k = clamp(Number.parseInt(opts.k, 10), 1, n);

  let means = kmeansPlusPlusInit(points, k);
  const overallMean = meanVector(points);
  let covariances = null;
  const baseVar = varianceVector(points, overallMean, opts.varianceFloor);
  const priorVar = baseVar.map(v => Math.max(v * opts.covPriorScale, opts.varianceFloor));
  const varPriorAlpha = Math.max(2, opts.covPriorDf + 2);
  const varPriorBeta = priorVar.map(v => v * (varPriorAlpha + 1));
  const baseCov = covarianceMatrix(points, overallMean, opts.varianceFloor);
  const priorNu = Math.max(d + 2, d + opts.covPriorDf + 2);
  const covPrior = baseCov.map(row => row.map(v => v * opts.covPriorScale * (priorNu + d + 1)));
  const priors = {
    weightPrior: Math.max(1e-6, opts.weightPrior),
    meanPriorStrength: Math.max(1e-6, opts.meanPriorStrength),
    meanPrior: overallMean,
    varPriorAlpha,
    varPriorBeta,
    covPrior,
    covPriorNu: priorNu,
  };
  if (opts.covarianceType === 'full') {
    covariances = Array.from({length: k}, () => deepCopyMatrix(baseCov));
  } else {
    covariances = Array.from({length: k}, () => baseVar.slice());
  }
  let weights = new Array(k).fill(1 / k);

  let prevLogLik = -Infinity;
  let responsibilities = null;
  let converged = false;
  let iterations = 0;

  for (let iter = 0; iter < opts.maxIter; iter++) {
    const cachedInverses = opts.covarianceType === 'full'
      ? covariances.map(cov => invertMatrixAndLogDet(cov))
      : null;

    const eStep = expectation(points, weights, means, covariances, opts.covarianceType, cachedInverses || []);
    responsibilities = eStep.responsibilities;

    const mStep = maximization(points, responsibilities, opts.covarianceType, opts.varianceFloor, priors);
    means = mStep.means;
    covariances = mStep.covariances;
    weights = mStep.weights;

    const logLik = eStep.logLikelihood;
    iterations = iter + 1;
    if (Math.abs(logLik - prevLogLik) < opts.tol) {
      converged = true;
      prevLogLik = logLik;
      break;
    }
    prevLogLik = logLik;
  }

  const params = computeModelParamsCount(k, d, opts.covarianceType);
  const bic = -2 * prevLogLik + params * Math.log(n);
  const aic = -2 * prevLogLik + 2 * params;

  const labels = responsibilities.map(row => {
    let best = 0;
    let bestVal = row[0];
    for (let i = 1; i < row.length; i++) {
      if (row[i] > bestVal) { bestVal = row[i]; best = i; }
    }
    return best;
  });

  return {
    labels,
    responsibilities,
    means,
    covariances,
    weights,
    bic,
    aic,
    logLikelihood: prevLogLik,
    iterations,
    converged,
    k,
    covarianceType: opts.covarianceType,
    bayesian: true,
    priors: {
      weightPrior: priors.weightPrior,
      meanPriorStrength: priors.meanPriorStrength,
      covPriorScale: opts.covPriorScale,
      covPriorDf: opts.covPriorDf,
    },
  };
}

function getGmmConfig() {
  return {
    k: clamp(Number.parseInt($('gmmK')?.value ?? DEFAULTS.k, 10), 1, 100000),
    minK: clamp(Number.parseInt($('gmmMinK')?.value ?? DEFAULTS.minK, 10), 1, 100000),
    maxK: clamp(Number.parseInt($('gmmMaxK')?.value ?? DEFAULTS.maxK, 10), 1, 100000),
    covarianceType: $('gmmCovType')?.value ?? DEFAULTS.covarianceType,
    maxIter: clamp(Number.parseInt($('gmmMaxIter')?.value ?? DEFAULTS.maxIter, 10), 5, 100000),
    tol: clamp(Number.parseFloat($('gmmTol')?.value ?? DEFAULTS.tol), 1e-8, 1),
    criterion: $('gmmCriterion')?.value ?? 'none',
    weightPrior: clamp(Number.parseFloat($('gmmWeightPrior')?.value ?? DEFAULTS.weightPrior), 1e-6, 1e6),
    meanPriorStrength: clamp(Number.parseFloat($('gmmMeanPrior')?.value ?? DEFAULTS.meanPriorStrength), 1e-6, 1e6),
    covPriorScale: clamp(Number.parseFloat($('gmmCovPriorScale')?.value ?? DEFAULTS.covPriorScale), 1e-6, 1e6),
    covPriorDf: clamp(Number.parseFloat($('gmmCovPriorDf')?.value ?? DEFAULTS.covPriorDf), 0, 1e6),
  };
}

function collectPoints() {
  const names = [];
  const points = [];

  state.source.names.forEach((name, idx) => {
    names.push(`${name} (source)`);
    points.push(state.source.vectors[idx]);
  });
  state.target.names.forEach((name, idx) => {
    names.push(`${name} (target)`);
    points.push(state.target.vectors[idx]);
  });

  return {names, points};
}

function renderGmmSummary({sampleNames, responsibilities, labels, avg, colOrder, maxProb, bic, aic, logLikelihood}) {
  const table = document.createElement('table');
  table.dataset.fixedColumns = '2';
  table.dataset.rowSortable = 'true';
  styleTableEl(table);

  const thead = document.createElement('thead');
  const hr = document.createElement('tr');
  hr.innerHTML = `<th>Sample</th><th>Cluster</th>` + colOrder.map(i => `<th>C${i + 1}</th>`).join('');
  thead.appendChild(hr);
  table.appendChild(thead);

  const tbody = document.createElement('tbody');
  const displayIndexByComponent = new Map();
  colOrder.forEach((compIdx, displayIdx) => displayIndexByComponent.set(compIdx, displayIdx + 1));

  for (let i = 0; i < responsibilities.length; i++) {
    const tr = document.createElement('tr');
    const nameCell = document.createElement('td');
    const rowSortBtn = document.createElement('button');
    rowSortBtn.type = 'button';
    rowSortBtn.className = 'row-sort-btn';
    rowSortBtn.setAttribute('aria-pressed', 'false');
    rowSortBtn.setAttribute('title', 'Sort this row by values');
    rowSortBtn.setAttribute('aria-label', 'Sort this row by values');
    rowSortBtn.innerHTML = '<span aria-hidden="true">⇄</span>';
    nameCell.appendChild(document.createTextNode(sampleNames[i]));
    nameCell.appendChild(rowSortBtn);

    const labelCell = document.createElement('td');
    const displayLabel = displayIndexByComponent.get(labels[i]) || (labels[i] + 1);
    labelCell.className = 'number';
    labelCell.textContent = `C${displayLabel} (${(maxProb[i] * 100).toFixed(1)}%)`;

    tr.appendChild(nameCell);
    tr.appendChild(labelCell);

    for (const compIdx of colOrder) {
      const val = responsibilities[i][compIdx];
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

  const avgRow = document.createElement('tr');
  avgRow.classList.add('is-summary');
  avgRow.dataset.fixed = 'true';
  avgRow.innerHTML = `<td><strong>Average</strong></td><td class="number"><strong>—</strong></td>`;
  for (const compIdx of colOrder) {
    const val = avg[compIdx];
    const td = document.createElement('td');
    td.className = 'number';
    td.textContent = (val * 100).toFixed(1);
    td.dataset.sortValue = String(val);
    const pct = clamp(val, 0, 1);
    td.style.backgroundColor = `rgba(11,94,215,${0.06 + pct * 0.35})`;
    td.style.color = pct > 0.55 ? '#fff' : '';
    avgRow.appendChild(td);
  }
  tbody.appendChild(avgRow);

  table.appendChild(tbody);

  const wrap = document.createElement('div');
  wrap.className = 'card';
  wrap.innerHTML = `<h3>GMM summary matrix</h3><div class="muted">logL=${logLikelihood.toFixed(3)} • BIC=${bic.toFixed(2)} • AIC=${aic.toFixed(2)}</div>`;
  wrap.appendChild(table);
  makeTableSortable(table, {fixedColumns: 2, rowSortable: true});
  return wrap;
}

function computeAverages(responsibilities) {
  const n = responsibilities.length;
  const k = responsibilities[0].length;
  const avg = new Array(k).fill(0);
  for (let i = 0; i < n; i++) {
    for (let c = 0; c < k; c++) avg[c] += responsibilities[i][c];
  }
  for (let c = 0; c < k; c++) avg[c] /= n;
  return avg;
}

function computeMaxProb(responsibilities, labels) {
  return responsibilities.map((row, i) => row[labels[i]] ?? 0);
}

function chooseBestByCriterion(results, criterion) {
  if (criterion === 'aic') {
    return results.reduce((best, cur) => (cur.aic < best.aic ? cur : best));
  }
  if (criterion === 'bic') {
    return results.reduce((best, cur) => (cur.bic < best.bic ? cur : best));
  }
  return results[0];
}

export function hydrateModelControls() {
  const total = state.source.names.length + state.target.names.length;
  ['gmmK', 'gmmMinK', 'gmmMaxK'].forEach(id => {
    const el = $(id);
    if (!el) return;
    el.max = String(total);
    const val = Number.parseInt(el.value, 10);
    if (Number.isFinite(val) && val > total) el.value = String(total);
  });
}

export async function runGmmModel() {
  if (!state.loaded) return;
  const config = getGmmConfig();
  const {points, names} = collectPoints();

  const out = $('gmmOutput');
  const wrap = document.createElement('div');
  wrap.className = 'run-block';
  wrap.innerHTML = `<div class="muted">GMM run • ${escapeHTML(new Date().toLocaleString())} • cov=${escapeHTML(config.covarianceType)} • criterion=${escapeHTML(config.criterion)} • Bayesian priors</div>`;

  let result = null;
  if (config.criterion === 'aic' || config.criterion === 'bic') {
    const minK = Math.min(config.minK, config.maxK);
    const maxK = Math.max(config.minK, config.maxK);
    const results = [];
    for (let k = minK; k <= maxK; k++) {
      if (k > points.length) break;
      const res = runGMM(points, {
        k,
        maxIter: config.maxIter,
        tol: config.tol,
        covarianceType: config.covarianceType,
        weightPrior: config.weightPrior,
        meanPriorStrength: config.meanPriorStrength,
        covPriorScale: config.covPriorScale,
        covPriorDf: config.covPriorDf,
      });
      results.push(res);
      await yieldToUI();
    }
    if (!results.length) return;
    result = chooseBestByCriterion(results, config.criterion);
  } else {
    result = runGMM(points, {
      k: config.k,
      maxIter: config.maxIter,
      tol: config.tol,
      covarianceType: config.covarianceType,
      weightPrior: config.weightPrior,
      meanPriorStrength: config.meanPriorStrength,
      covPriorScale: config.covPriorScale,
      covPriorDf: config.covPriorDf,
    });
  }

  const avg = computeAverages(result.responsibilities);
  const colOrder = Array.from({length: result.k}, (_, i) => i);
  colOrder.sort((a, b) => avg[b] - avg[a]);
  const maxProb = computeMaxProb(result.responsibilities, result.labels);

  state.cluster = {
    algorithm: 'gmm',
    k: result.k,
    labels: result.labels,
    softLabels: result.responsibilities,
    means: result.means,
    covariances: result.covariances,
    weights: result.weights,
    bic: result.bic,
    aic: result.aic,
    logLikelihood: result.logLikelihood,
    covarianceType: result.covarianceType,
    bayesian: result.bayesian,
    priors: result.priors,
    sampleNames: names,
  };

  wrap.appendChild(renderGmmSummary({
    sampleNames: names,
    responsibilities: result.responsibilities,
    labels: result.labels,
    avg,
    colOrder,
    maxProb,
    bic: result.bic,
    aic: result.aic,
    logLikelihood: result.logLikelihood,
  }));

  out.prepend(wrap);
}
