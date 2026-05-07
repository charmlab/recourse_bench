const LOCAL_DATA_PATH = "data/default_results.csv";
const COMPATIBILITY_PATH = "data/compatibility.json";

const METRIC_DEFS = [
  { id: "validity", label: "Validity", column: "validity", direction: "max" },
  { id: "proximity", label: "Proximity (L2)", column: "distance_l2", direction: "min" },
  { id: "sparsity", label: "Sparsity (L0)", column: "distance_l0", direction: "min" },
  { id: "plausibility", label: "Plausibility (YNN)", column: "ynn", direction: "max" },
  { id: "runtime", label: "Runtime", column: "runtime_seconds", direction: "min" },
];

const METRIC_PRESETS = {
  balanced: METRIC_DEFS.map((m) => m.id),
  quality: ["validity", "plausibility", "sparsity"],
  speed: ["runtime", "proximity", "sparsity"],
};

const state = {
  rows: [],
  options: {
    datasets: [],
    models: [],
    methods: [],
  },
  compatibility: {
    methods: {},
  },
  selectedDataset: "",
  selectedModel: "",
  useAllMethods: true,
  selectedMethods: new Set(),
  selectedMetrics: new Set(METRIC_DEFS.map((m) => m.id)),
  activePreset: "balanced",
  metricBounds: {},
  latestVisuals: null,
};

const els = {
  datasetSelect: document.getElementById("datasetSelect"),
  modelSelect: document.getElementById("modelSelect"),
  methodDropdown: document.getElementById("methodDropdown"),
  methodChecklist: document.getElementById("methodChecklist"),
  metricPalette: document.getElementById("metricPalette"),
  modeAllMethodsBtn: document.getElementById("modeAllMethodsBtn"),
  modeCustomMethodsBtn: document.getElementById("modeCustomMethodsBtn"),
  selectAllMethodsBtn: document.getElementById("selectAllMethodsBtn"),
  clearMethodsBtn: document.getElementById("clearMethodsBtn"),
  selectAllMetricsBtn: document.getElementById("selectAllMetricsBtn"),
  clearMetricsBtn: document.getElementById("clearMetricsBtn"),
  presetBtns: [...document.querySelectorAll(".presetBtn")],
  rankBtn: document.getElementById("rankBtn"),
  resetBtn: document.getElementById("resetBtn"),
  reloadBtn: document.getElementById("reloadBtn"),
  selectionHeadline: document.getElementById("selectionHeadline"),
  methodSelectedCount: document.getElementById("methodSelectedCount"),
  metricSelectedCount: document.getElementById("metricSelectedCount"),
  selectionPills: document.getElementById("selectionPills"),
  selectionStats: document.getElementById("selectionStats"),
  statusMessage: document.getElementById("statusMessage"),
  visualSection: document.getElementById("visualSection"),
  tableSection: document.getElementById("tableSection"),
  tradeoffPlot: document.getElementById("tradeoffPlot"),
  radarPlot: document.getElementById("radarPlot"),
  summary: document.getElementById("summary"),
  tableMeta: document.getElementById("tableMeta"),
  tableHead: document.querySelector("#resultTable thead"),
  tableBody: document.querySelector("#resultTable tbody"),
};

let radarChart = null;
let visualResizeTimer = null;

function parseNumber(value) {
  if (value === null || value === undefined) return NaN;
  if (typeof value === "string" && value.trim() === "") return NaN;
  const n = Number(value);
  return Number.isFinite(n) ? n : NaN;
}

function mean(values) {
  const valid = values.filter((v) => Number.isFinite(v));
  if (!valid.length) return NaN;
  return valid.reduce((acc, v) => acc + v, 0) / valid.length;
}

function uniqueSorted(values) {
  return [...new Set(values)].sort((a, b) => String(a).localeCompare(String(b)));
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function formatCell(value, digits = 4) {
  const n = parseNumber(value);
  return Number.isFinite(n) ? n.toFixed(digits) : "-";
}

function isSuccessful(row) {
  if (row.status === undefined || row.status === null || String(row.status).trim() === "") return true;
  const status = String(row.status).trim().toLowerCase();
  return status === "success" || status === "completed";
}

function pickBenchmarkRows(rows) {
  const hasRowTypes = rows.some((row) => row && typeof row === "object" && Object.prototype.hasOwnProperty.call(row, "row_type"));
  if (!hasRowTypes) return rows;

  const summaryRows = rows.filter((row) => row && String(row.row_type ?? "").trim().toLowerCase() === "summary");
  if (summaryRows.length) return summaryRows;

  return rows.filter((row) => {
    if (!row) return false;
    const rowType = String(row.row_type ?? "").trim().toLowerCase();
    return !rowType || rowType === "detail";
  });
}

function normalizeBenchmarkRow(row) {
  const clean = { ...row };

  clean.dataset = String(clean.dataset ?? clean.dataset_name ?? "").trim();
  clean.model = String(clean.model ?? clean.model_name ?? "").trim();
  clean.method = String(clean.method ?? clean.method_name ?? "").trim();
  clean.status = String(clean.status ?? "").trim();

  if (String(clean.ynn ?? "").trim() === "" && String(clean.knn_5 ?? "").trim() !== "") {
    clean.ynn = clean.knn_5;
  }
  if (String(clean.runtime_seconds ?? "").trim() === "" && String(clean.run_duration_seconds ?? "").trim() !== "") {
    clean.runtime_seconds = clean.run_duration_seconds;
  }
  if (String(clean.runtime_seconds ?? "").trim() === "" && String(clean.elapsed_seconds ?? "").trim() !== "") {
    clean.runtime_seconds = clean.elapsed_seconds;
  }

  return clean;
}

function hideResultPanels() {
  els.visualSection.classList.add("hidden");
  els.tableSection.classList.add("hidden");
}

function showResultPanels() {
  els.visualSection.classList.remove("hidden");
  els.tableSection.classList.remove("hidden");
}

function clearResults(message) {
  els.statusMessage.textContent = message;
  els.summary.textContent = "";
  if (els.tableMeta) els.tableMeta.innerHTML = "";
  if (els.tradeoffPlot) els.tradeoffPlot.innerHTML = "";
  if (radarChart) { radarChart.destroy(); radarChart = null; }
  if (els.radarPlot) els.radarPlot.innerHTML = "";
  els.tableHead.innerHTML = "";
  els.tableBody.innerHTML = "";
  state.latestVisuals = null;
  hideResultPanels();
}

function markResultsStale() {
  const wasVisible = !els.visualSection.classList.contains("hidden") || !els.tableSection.classList.contains("hidden");
  if (wasVisible) {
    clearResults("Configuration changed. Click Show Benchmark to refresh visuals and ranking.");
  }
}

function normalizeCompatibility(data) {
  const methods = {};
  const source = data && typeof data === "object" && data.methods && typeof data.methods === "object"
    ? data.methods
    : {};

  Object.entries(source).forEach(([method, config]) => {
    const allowedModels = Array.isArray(config?.allowed_models)
      ? config.allowed_models.map((item) => String(item).trim()).filter(Boolean)
      : [];
    const allowedDatasets = Array.isArray(config?.allowed_datasets)
      ? config.allowed_datasets.map((item) => String(item).trim()).filter(Boolean)
      : [];

    methods[String(method).trim()] = {
      allowedModels,
      allowedDatasets,
    };
  });

  return { methods };
}

function getMethodCompatibility(method) {
  return state.compatibility.methods?.[method] ?? null;
}

function isMethodCompatibleWithScope(method, dataset = state.selectedDataset, model = state.selectedModel) {
  const compatibility = getMethodCompatibility(method);
  if (!compatibility) return true;

  if (model && compatibility.allowedModels.length && !compatibility.allowedModels.includes(model)) {
    return false;
  }

  if (dataset && compatibility.allowedDatasets.length && !compatibility.allowedDatasets.includes(dataset)) {
    return false;
  }

  return true;
}

function getMethodCompatibilityReason(method, dataset = state.selectedDataset, model = state.selectedModel) {
  const compatibility = getMethodCompatibility(method);
  if (!compatibility) return "";

  if (model && compatibility.allowedModels.length && !compatibility.allowedModels.includes(model)) {
    return `Unavailable for model: ${model}`;
  }

  if (dataset && compatibility.allowedDatasets.length && !compatibility.allowedDatasets.includes(dataset)) {
    return `Unavailable for dataset: ${dataset}`;
  }

  return "";
}

function areSelectedMethodsCompatible(dataset = state.selectedDataset, model = state.selectedModel) {
  if (state.useAllMethods || !state.selectedMethods.size) return true;
  return [...state.selectedMethods].every((method) => isMethodCompatibleWithScope(method, dataset, model));
}

function getScopeCompatibilityReason(type, option) {
  if (state.useAllMethods || !state.selectedMethods.size) return "";
  if (type === "dataset") {
    return `Unavailable for selected method set at dataset: ${option}`;
  }
  if (type === "model") {
    return `Unavailable for selected method set at model: ${option}`;
  }
  return "";
}

function isScopeOptionEnabled(type, option) {
  if (type === "dataset") {
    return areSelectedMethodsCompatible(option, state.selectedModel);
  }
  if (type === "model") {
    return areSelectedMethodsCompatible(state.selectedDataset, option);
  }
  return true;
}

function getCompatibleMethods(dataset = state.selectedDataset, model = state.selectedModel) {
  return state.options.methods.filter((method) => isMethodCompatibleWithScope(method, dataset, model));
}

function getCompatibleMethodSet(dataset = state.selectedDataset, model = state.selectedModel) {
  return new Set(getCompatibleMethods(dataset, model));
}

function getEffectiveSelectedMethods() {
  if (state.useAllMethods) return new Set();
  const compatibleMethodSet = getCompatibleMethodSet();
  return new Set([...state.selectedMethods].filter((method) => compatibleMethodSet.has(method)));
}

function syncSelectedMethodsToEffectiveSelection() {
  if (state.useAllMethods) {
    if (state.selectedMethods.size) {
      state.selectedMethods.clear();
    }
    return;
  }

  const effectiveSelectedMethods = getEffectiveSelectedMethods();
  const hasMismatch = effectiveSelectedMethods.size !== state.selectedMethods.size
    || [...state.selectedMethods].some((method) => !effectiveSelectedMethods.has(method));

  if (hasMismatch) {
    state.selectedMethods = effectiveSelectedMethods;
  }
}

function pruneSelectedMethods() {
  if (state.useAllMethods || !state.selectedMethods.size) return;

  const compatibleMethodSet = getCompatibleMethodSet();
  const compatibleSelected = [...state.selectedMethods].filter((method) => compatibleMethodSet.has(method));
  state.selectedMethods = new Set(compatibleSelected);
}

function normalizeMethodSelectionState() {
  if (state.useAllMethods) {
    state.selectedMethods.clear();
    return;
  }

  pruneSelectedMethods();
  if (!state.selectedMethods.size) {
    ensureCustomSelectionSeed();
  }
  syncSelectedMethodsToEffectiveSelection();
}

function renderScopeSelect(selectEl, options, selectedValue, type, placeholder) {
  selectEl.innerHTML = "";

  const placeholderOption = document.createElement("option");
  placeholderOption.value = "";
  placeholderOption.textContent = placeholder;
  placeholderOption.selected = !selectedValue;
  selectEl.appendChild(placeholderOption);

  options.forEach((option) => {
    const optionEl = document.createElement("option");
    optionEl.value = option;
    const enabled = isScopeOptionEnabled(type, option);
    optionEl.disabled = !enabled;
    optionEl.selected = selectedValue === option;
    optionEl.textContent = enabled ? option : `${option} [unavailable]`;
    selectEl.appendChild(optionEl);
  });
}

function renderMethodSelect() {
  syncSelectedMethodsToEffectiveSelection();
  const compatibleMethodSet = getCompatibleMethodSet();
  const effectiveSelectedMethods = state.useAllMethods ? compatibleMethodSet : getEffectiveSelectedMethods();
  const compatibleMethods = getCompatibleMethods();

  if (!els.methodChecklist) return;
  els.methodChecklist.innerHTML = "";

  state.options.methods.forEach((method) => {
    const compatible = isMethodCompatibleWithScope(method);
    const checked = compatible && effectiveSelectedMethods.has(method);
    const reason = compatible ? (checked ? "Included in current comparison" : "Available in current scope") : getMethodCompatibilityReason(method);

    const row = document.createElement("label");
    row.className = `methodOption${compatible ? "" : " is-disabled"}`;
    row.title = compatible ? "" : reason;
    row.innerHTML = `
      <input type="checkbox" data-method="${escapeHtml(method)}" ${checked ? "checked" : ""} ${compatible ? "" : "disabled"}>
      <span class="methodOptionText">
        <span class="methodOptionName">${escapeHtml(method)}</span>
        <span class="methodOptionMeta">${escapeHtml(reason)}</span>
      </span>
    `;
    els.methodChecklist.appendChild(row);
  });

  if (els.methodDropdown) {
    const summary = els.methodDropdown.querySelector(".dropdownSummary");
    if (summary) {
      summary.textContent = state.useAllMethods
        ? "All compatible methods"
        : "Choose methods";
    }
  }
}

function renderConfigurationControls() {
  renderScopeSelect(els.datasetSelect, state.options.datasets, state.selectedDataset, "dataset", "Choose dataset");
  renderScopeSelect(els.modelSelect, state.options.models, state.selectedModel, "model", "Choose model");
  renderMethodSelect();
}

function renderMetricPalette() {
  els.metricPalette.innerHTML = "";

  METRIC_DEFS.forEach((metric) => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = `token${state.selectedMetrics.has(metric.id) ? " active" : ""}`;
    btn.dataset.metricId = metric.id;
    btn.innerHTML = `${escapeHtml(metric.label)} <span class="dir">(${metric.direction === "max" ? "max" : "min"})</span>`;
    els.metricPalette.appendChild(btn);
  });

  els.presetBtns.forEach((button) => {
    button.classList.toggle("active", button.dataset.preset === state.activePreset);
  });
}

function selectedMetricConfig() {
  return METRIC_DEFS
    .filter((metric) => state.selectedMetrics.has(metric.id))
    .map((metric) => ({
      metricId: metric.id,
      label: metric.label,
      column: metric.column,
      direction: metric.direction,
    }));
}

function getMethodSubset() {
  if (state.useAllMethods) return getCompatibleMethods();
  normalizeMethodSelectionState();
  syncSelectedMethodsToEffectiveSelection();
  const effectiveSelectedMethods = getEffectiveSelectedMethods();
  return state.options.methods.filter((method) => effectiveSelectedMethods.has(method));
}

function ensureCustomSelectionSeed() {
  if (state.useAllMethods) return;
  if (state.selectedMethods.size) return;
  const compatibleMethods = getCompatibleMethods();
  if (compatibleMethods.length) {
    state.selectedMethods = new Set([compatibleMethods[0]]);
  }
}

function updateSelectionUI() {
  normalizeMethodSelectionState();
  syncSelectedMethodsToEffectiveSelection();
  const compatibleMethods = getCompatibleMethods();
  const effectiveSelectedMethods = getEffectiveSelectedMethods();
  const methodCount = state.useAllMethods ? compatibleMethods.length : effectiveSelectedMethods.size;
  const disabledMethodCount = Math.max(0, state.options.methods.length - compatibleMethods.length);
  const scopeReady = Boolean(state.selectedDataset && state.selectedModel);

  els.modeAllMethodsBtn.classList.toggle("active", state.useAllMethods);
  els.modeCustomMethodsBtn.classList.toggle("active", !state.useAllMethods);

  els.methodSelectedCount.textContent = state.useAllMethods
    ? "All Compatible"
    : "Custom Selection";

  els.metricSelectedCount.textContent = `${state.selectedMetrics.size} selected`;

  els.selectionHeadline.textContent = state.selectedDataset && state.selectedModel
    ? `Scope: ${state.selectedDataset} / ${state.selectedModel}`
    : "Choose a dataset and model to begin.";

  const methodLabel = state.useAllMethods ? "Methods: all compatible" : "Methods: custom selection";
  const metricLabel = state.selectedMetrics.size ? `Metrics: ${state.selectedMetrics.size}` : "Metrics: none";

  els.selectionPills.innerHTML = [
    `<span class="selectionPill">Dataset: ${escapeHtml(state.selectedDataset || "not selected")}</span>`,
    `<span class="selectionPill">Model: ${escapeHtml(state.selectedModel || "not selected")}</span>`,
    `<span class="selectionPill">${escapeHtml(methodLabel)}</span>`,
    `<span class="selectionPill">${escapeHtml(metricLabel)}</span>`,
  ].join("");

  if (els.selectionStats) {
    els.selectionStats.innerHTML = [
      {
        label: "Scope",
        value: scopeReady ? `${state.selectedDataset} / ${state.selectedModel}` : "Not Set",
        sub: scopeReady ? "Active comparison surface" : "Pick dataset and model",
      },
      {
        label: "Compatible Methods",
        value: String(compatibleMethods.length),
        sub: disabledMethodCount ? `${disabledMethodCount} unavailable in current scope` : "All visible methods available",
      },
      {
        label: "Selection Mode",
        value: state.useAllMethods ? "All Compatible" : "Custom",
        sub: state.useAllMethods ? "Auto-uses valid methods only" : `${effectiveSelectedMethods.size} method${effectiveSelectedMethods.size === 1 ? "" : "s"} selected`,
      },
      {
        label: "Metric Utility",
        value: String(state.selectedMetrics.size),
        sub: state.selectedMetrics.size ? "Higher is better after transform" : "Choose at least one metric",
      },
    ].map((card) => `
      <div class="statCard">
        <span class="statLabel">${escapeHtml(card.label)}</span>
        <span class="statValue">${escapeHtml(card.value)}</span>
        <span class="statSub">${escapeHtml(card.sub)}</span>
      </div>
    `).join("");
  }

  const canRun = Boolean(
    state.selectedDataset
    && state.selectedModel
    && state.selectedMetrics.size > 0
    && methodCount > 0
  );
  els.rankBtn.disabled = !canRun;
}

function aggregateByMethod(rows, metricColumns) {
  const groups = new Map();

  rows.forEach((row) => {
    const key = String(row.method);
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(row);
  });

  const aggregated = [];
  groups.forEach((methodRows, method) => {
    const item = { method, count: methodRows.length };
    metricColumns.forEach((column) => {
      item[column] = mean(methodRows.map((r) => parseNumber(r[column])));
    });
    aggregated.push(item);
  });

  return aggregated;
}

function computeMetricBounds(rows) {
  const bounds = {};

  METRIC_DEFS.forEach((metric) => {
    const values = rows.map((r) => parseNumber(r[metric.column])).filter((v) => Number.isFinite(v));
    const observedMax = values.length ? Math.max(...values) : NaN;

    if (metric.column === "validity" || metric.column === "ynn") {
      bounds[metric.column] = { min: 0, max: 1 };
      return;
    }

    const max = Number.isFinite(observedMax) && observedMax > 0 ? observedMax : 1;
    bounds[metric.column] = { min: 0, max };
  });

  return bounds;
}

function computeMetricTransforms(rows, metricConfig) {
  const transforms = {};
  metricConfig.forEach((cfg) => {
    const fixed = state.metricBounds[cfg.column];
    const values = rows.map((r) => parseNumber(r[cfg.column])).filter((v) => Number.isFinite(v));
    if (!values.length) {
      transforms[cfg.column] = null;
      return;
    }

    const min = fixed && Number.isFinite(fixed.min) ? fixed.min : Math.min(...values);
    const max = fixed && Number.isFinite(fixed.max) ? fixed.max : Math.max(...values);

    const shiftedValues = values
      .map((value) => value - min)
      .filter((value) => Number.isFinite(value) && value > 0)
      .sort((a, b) => a - b);

    let scale = 1;
    if (shiftedValues.length) {
      scale = shiftedValues[Math.floor(shiftedValues.length / 2)];
    } else if (Number.isFinite(max - min) && max > min) {
      scale = max - min;
    }

    if (!Number.isFinite(scale) || scale <= 0) {
      scale = 1;
    }

    let denominator = 1;
    if (cfg.direction === "max") {
      const span = Math.max(0, max - min);
      denominator = span > 0 ? (1 - Math.exp(-span / scale)) : 1;
      if (!Number.isFinite(denominator) || denominator <= 0) {
        denominator = 1;
      }
    }

    transforms[cfg.column] = {
      min,
      max,
      scale,
      denominator,
      direction: cfg.direction,
    };
  });
  return transforms;
}

function normalizeMetricValue(value, direction, transform) {
  if (!transform || !Number.isFinite(value)) return NaN;

  const shifted = Math.max(0, value - transform.min);
  if (direction === "max") {
    const utility = (1 - Math.exp(-shifted / transform.scale)) / transform.denominator;
    return Math.max(0, Math.min(1, utility));
  }

  const utility = Math.exp(-shifted / transform.scale);
  return Math.max(0, Math.min(1, utility));
}

function scoreRows(rows, metricConfig, transforms) {
  rows.forEach((row) => {
    let total = 0;
    let used = 0;

    metricConfig.forEach((cfg) => {
      const value = parseNumber(row[cfg.column]);
      const normalized = normalizeMetricValue(value, cfg.direction, transforms[cfg.column]);
      if (!Number.isFinite(normalized)) return;
      total += normalized;
      used += 1;
    });

    row.score = used ? total / used : NaN;
  });

  rows.sort((a, b) => {
    const av = Number.isFinite(a.score) ? a.score : -Infinity;
    const bv = Number.isFinite(b.score) ? b.score : -Infinity;
    return bv - av;
  });

  rows.forEach((row, index) => {
    row.rank = index + 1;
  });

  return rows;
}

function pickTradeoffMetrics() {
  return {
    x: { metricId: "proximity", label: "Proximity (L2)", column: "distance_l2", direction: "min" },
    y: { metricId: "validity", label: "Validity", column: "validity", direction: "max" },
  };
}

function renderTradeoffPlot(rows) {
  const pair = pickTradeoffMetrics();

  const width = 700;
  const height = 340;
  const pad = { left: 64, right: 20, top: 18, bottom: 56 };

  const xValues = rows.map((r) => parseNumber(r[pair.x.column])).filter((v) => Number.isFinite(v));
  const yValues = rows.map((r) => parseNumber(r[pair.y.column])).filter((v) => Number.isFinite(v));

  if (!xValues.length || !yValues.length) {
    els.tradeoffPlot.innerHTML = '<div class="tradeoffEmpty">Validity or Proximity (L2) values are missing for this selection.</div>';
    return;
  }

  const xBound = state.metricBounds?.[pair.x.column];
  const yBound = state.metricBounds?.[pair.y.column];

  const xMin = xBound && Number.isFinite(xBound.min) ? xBound.min : Math.min(...xValues);
  const xMaxRaw = xBound && Number.isFinite(xBound.max) ? xBound.max : Math.max(...xValues);
  const yMin = yBound && Number.isFinite(yBound.min) ? yBound.min : Math.min(...yValues);
  const yMaxRaw = yBound && Number.isFinite(yBound.max) ? yBound.max : Math.max(...yValues);

  const xMax = xMaxRaw > xMin ? xMaxRaw : xMin + 1;
  const yMax = yMaxRaw > yMin ? yMaxRaw : yMin + 1;

  const xSpan = xMax - xMin;
  const ySpan = yMax - yMin;

  const chartW = width - pad.left - pad.right;
  const chartH = height - pad.top - pad.bottom;

  const toX = (xv) => pad.left + ((xv - xMin) / xSpan) * chartW;
  const toY = (yv) => pad.top + chartH - ((yv - yMin) / ySpan) * chartH;

  const points = rows.map((row, index) => {
    const xv = parseNumber(row[pair.x.column]);
    const yv = parseNumber(row[pair.y.column]);
    if (!Number.isFinite(xv) || !Number.isFinite(yv)) return "";

    const cx = toX(xv);
    const cy = toY(yv);
    const score = Number.isFinite(row.score) ? row.score : 0;
    const fill = `hsl(${170 - score * 55} 70% ${44 - score * 8}%)`;

    const label = `${row.method}`;
    const approxLabelWidth = label.length * 7;
    const lx = Math.max(cx - 12, pad.left + approxLabelWidth + 4);
    const ly = Math.min(Math.max(cy + 4, pad.top + 12), pad.top + chartH - 4);

    return `
      <g>
        <circle cx="${cx.toFixed(2)}" cy="${cy.toFixed(2)}" r="7.5" fill="${fill}" stroke="#ffffff" stroke-width="1.4" opacity="0.95">
          <title>${escapeHtml(row.method)} | ${pair.x.label}: ${formatCell(xv, 3)} | ${pair.y.label}: ${formatCell(yv, 3)} | score: ${formatCell(row.score, 3)}</title>
        </circle>
        <text class="pointLabel" text-anchor="end" x="${lx.toFixed(2)}" y="${ly.toFixed(2)}">${escapeHtml(label)}</text>
      </g>
    `;
  }).join("");

  const axisColor = "#9d9788";
  const tickColor = "#8d8779";
  const gridColor = "#e1d8c8";
  const tickCount = 5;

  const fmtTick = (value) => {
    if (!Number.isFinite(value)) return "";
    if (Math.abs(value) >= 10) return value.toFixed(1);
    return value.toFixed(2);
  };

  const xTicks = Array.from({ length: tickCount + 1 }, (_, i) => xMin + (xSpan * i) / tickCount);
  const yTicks = Array.from({ length: tickCount + 1 }, (_, i) => yMin + (ySpan * i) / tickCount);

  const xTickMarkup = xTicks.map((tick) => {
    const x = toX(tick);
    return `
      <g>
        <line x1="${x.toFixed(2)}" y1="${pad.top}" x2="${x.toFixed(2)}" y2="${(pad.top + chartH).toFixed(2)}" stroke="${gridColor}" stroke-width="1" />
        <text x="${x.toFixed(2)}" y="${(pad.top + chartH + 16).toFixed(2)}" text-anchor="middle" font-size="11" fill="${tickColor}">${fmtTick(tick)}</text>
      </g>
    `;
  }).join("");

  const yTickMarkup = yTicks.map((tick) => {
    const y = toY(tick);
    return `
      <g>
        <line x1="${pad.left}" y1="${y.toFixed(2)}" x2="${(pad.left + chartW).toFixed(2)}" y2="${y.toFixed(2)}" stroke="${gridColor}" stroke-width="1" />
        <text x="${(pad.left - 8).toFixed(2)}" y="${(y + 4).toFixed(2)}" text-anchor="end" font-size="11" fill="${tickColor}">${fmtTick(tick)}</text>
      </g>
    `;
  }).join("");

  els.tradeoffPlot.innerHTML = `
    <svg viewBox="0 0 ${width} ${height}" width="100%" height="340" role="img" aria-label="Tradeoff scatter plot with raw metric values">
      ${xTickMarkup}
      ${yTickMarkup}
      <line x1="${pad.left}" y1="${pad.top + chartH}" x2="${pad.left + chartW}" y2="${pad.top + chartH}" stroke="${axisColor}" />
      <line x1="${pad.left}" y1="${pad.top}" x2="${pad.left}" y2="${pad.top + chartH}" stroke="${axisColor}" />
      <text x="${(pad.left + chartW / 2).toFixed(2)}" y="${(height - 16).toFixed(2)}" text-anchor="middle" font-size="12" fill="#595f58">${escapeHtml(pair.x.label)}</text>
      <text x="16" y="${(pad.top + chartH / 2).toFixed(2)}" text-anchor="middle" transform="rotate(-90, 16, ${(pad.top + chartH / 2).toFixed(2)})" font-size="12" fill="#595f58">${escapeHtml(pair.y.label)}</text>
      ${points}
    </svg>
  `;
}
function polygonArea(points) {
  if (!points.length) return 0;
  let sum = 0;
  for (let i = 0; i < points.length; i += 1) {
    const a = points[i];
    const b = points[(i + 1) % points.length];
    sum += a.x * b.y - b.x * a.y;
  }
  return Math.abs(sum) / 2;
}

function renderRadarFallbackSvg(series, metricConfig) {
  const size = 360;
  const center = size / 2;
  const radius = 132;
  const rings = [0.25, 0.5, 0.75, 1];

  const axes = metricConfig.map((cfg, index) => {
    const angle = -Math.PI / 2 + (2 * Math.PI * index) / metricConfig.length;
    const x = center + Math.cos(angle) * radius;
    const y = center + Math.sin(angle) * radius;
    const lx = center + Math.cos(angle) * (radius + 16);
    const ly = center + Math.sin(angle) * (radius + 16);
    return { label: cfg.label, angle, x, y, lx, ly };
  });

  const ringMarkup = rings.map((level) => {
    const points = axes.map((axis) => {
      const x = center + Math.cos(axis.angle) * radius * level;
      const y = center + Math.sin(axis.angle) * radius * level;
      return `${x.toFixed(2)},${y.toFixed(2)}`;
    }).join(" ");
    return `<polygon points="${points}" fill="none" stroke="#d8d0c1" stroke-width="1"></polygon>`;
  }).join("");

  const axisMarkup = axes.map((axis) => `
    <line x1="${center}" y1="${center}" x2="${axis.x.toFixed(2)}" y2="${axis.y.toFixed(2)}" stroke="#b9b09f" stroke-width="1"></line>
  `).join("");

  const labelMarkup = axes.map((axis) => `
    <text x="${axis.lx.toFixed(2)}" y="${axis.ly.toFixed(2)}" text-anchor="middle" dominant-baseline="middle" font-size="11" fill="#4e5a52">${escapeHtml(axis.label)}</text>
  `).join("");

  const polygons = series.map((item) => {
    const points = item.values.map((value, index) => {
      const axis = axes[index];
      const v = Math.max(0, Math.min(1, value));
      const x = center + Math.cos(axis.angle) * radius * v;
      const y = center + Math.sin(axis.angle) * radius * v;
      return `${x.toFixed(2)},${y.toFixed(2)}`;
    }).join(" ");

    return `
      <polygon points="${points}" fill="${item.fill}" stroke="${item.stroke}" stroke-width="2" opacity="0.9">
        <title>${escapeHtml(item.method)} (area ${item.areaPct.toFixed(1)}%)</title>
      </polygon>
    `;
  }).join("");

  const legendItems = series.map((item) => `
    <div class="radarLegendItem">
      <span class="radarSwatch" style="background:${item.stroke}"></span>
      <span class="radarName">${escapeHtml(item.method)}</span>
      <span class="radarArea">area ${item.areaPct.toFixed(1)}%</span>
    </div>
  `).join("");

  els.radarPlot.innerHTML = `
    <div class="radarWrap">
      <div class="radarCanvasWrap">
        <svg viewBox="0 0 ${size} ${size}" width="100%" height="100%" role="img" aria-label="Top method radar profile">
          ${ringMarkup}
          ${axisMarkup}
          ${polygons}
          ${labelMarkup}
        </svg>
      </div>
      <div class="radarLegend">${legendItems}</div>
    </div>
  `;
}

function renderRadarPlot(rows, metricConfig, transforms) {
  if (!els.radarPlot) return;

  if (radarChart) {
    radarChart.destroy();
    radarChart = null;
  }

  if (metricConfig.length < 2) {
    els.radarPlot.innerHTML = '<div class="tradeoffEmpty">Select at least 2 metrics to show a radar profile.</div>';
    return;
  }

  const top = rows.slice(0, Math.min(3, rows.length));
  if (!top.length) {
    els.radarPlot.innerHTML = '<div class="tradeoffEmpty">No methods available for radar plot.</div>';
    return;
  }

  const labels = metricConfig.map((m) => m.label);

  const palette = [
    { stroke: "#0f7a56", fill: "rgba(15,122,86,0.20)" },
    { stroke: "#126b8a", fill: "rgba(18,107,138,0.18)" },
    { stroke: "#da8e2f", fill: "rgba(218,142,47,0.20)" },
  ];

  const maxPolygonPoints = metricConfig.map((_, i) => {
    const angle = -Math.PI / 2 + (2 * Math.PI * i) / metricConfig.length;
    return { x: Math.cos(angle), y: Math.sin(angle) };
  });
  const maxArea = polygonArea(maxPolygonPoints) || 1;

  const datasets = [];
  const series = [];

  top.forEach((row, idx) => {
    const style = palette[idx % palette.length];
    const values = metricConfig.map((cfg) => {
      const raw = parseNumber(row[cfg.column]);
      const norm = normalizeMetricValue(raw, cfg.direction, transforms[cfg.column]);
      return Number.isFinite(norm) ? norm : 0;
    });

    datasets.push({
      label: row.method,
      data: values,
      rawValues: metricConfig.map((cfg) => parseNumber(row[cfg.column])),
      borderColor: style.stroke,
      backgroundColor: style.fill,
      pointBackgroundColor: style.stroke,
      pointBorderColor: "#ffffff",
      pointBorderWidth: 1,
      pointRadius: 3,
      borderWidth: 2,
      fill: true,
      tension: 0,
    });

    const pointsObj = values.map((v, i) => {
      const angle = -Math.PI / 2 + (2 * Math.PI * i) / values.length;
      return { x: Math.cos(angle) * v, y: Math.sin(angle) * v };
    });
    const areaPct = (polygonArea(pointsObj) / maxArea) * 100;

    series.push({
      method: row.method,
      values,
      stroke: style.stroke,
      fill: style.fill,
      areaPct,
    });
  });

  const hasChartLibrary = typeof window !== "undefined" && typeof window.Chart === "function";
  if (!hasChartLibrary) {
    renderRadarFallbackSvg(series, metricConfig);
    return;
  }

  const legendItems = series.map((item) => `
    <div class="radarLegendItem">
      <span class="radarSwatch" style="background:${item.stroke}"></span>
      <span class="radarName">${escapeHtml(item.method)}</span>
      <span class="radarArea">area ${item.areaPct.toFixed(1)}%</span>
    </div>
  `).join("");

  els.radarPlot.innerHTML = `
    <div class="radarWrap">
      <div class="radarCanvasWrap">
        <canvas id="radarCanvas"></canvas>
      </div>
      <div class="radarLegend">${legendItems}</div>
    </div>
  `;

  const canvas = document.getElementById("radarCanvas");
  if (!canvas || !canvas.getContext) {
    renderRadarFallbackSvg(series, metricConfig);
    return;
  }

  try {
    radarChart = new window.Chart(canvas.getContext("2d"), {
      type: "radar",
      data: {
        labels,
        datasets,
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            enabled: true,
            callbacks: {
              label: (context) => {
                const metric = metricConfig[context.dataIndex];
                const raw = context.dataset.rawValues?.[context.dataIndex];
                const utility = context.parsed?.r;
                return `${context.dataset.label} | ${metric?.label ?? "metric"}: utility ${formatCell(utility, 3)} | raw ${formatCell(raw, 3)}`;
              },
            },
          },
        },
        scales: {
          r: {
            min: 0,
            max: 1,
            ticks: {
              stepSize: 0.25,
              showLabelBackdrop: false,
              color: "#6a716a",
            },
            grid: { color: "#d8d0c1" },
            angleLines: { color: "#b9b09f" },
            pointLabels: {
              color: "#4e5a52",
              font: { size: 11, weight: "600" },
            },
          },
        },
      },
    });
  } catch (error) {
    renderRadarFallbackSvg(series, metricConfig);
    return;
  }

  if (typeof requestAnimationFrame === "function") {
    requestAnimationFrame(() => {
      if (radarChart) {
        radarChart.resize();
        radarChart.update("none");
      }
    });
  }
}
function renderTable(rows, metricConfig, transforms) {
  const headers = [
    { label: "rank", className: "stickyCol rankCol" },
    { label: "method", className: "stickyCol methodCol" },
    { label: "score", className: "" },
    ...metricConfig.map((cfg) => ({ label: cfg.label, className: "" })),
  ];

  els.tableHead.innerHTML = `<tr>${headers.map((h) => `<th class="${h.className}">${escapeHtml(h.label)}</th>`).join("")}</tr>`;
  els.tableBody.innerHTML = rows.map((row) => {
    const scorePct = Number.isFinite(row.score) ? Math.max(0, Math.min(100, row.score * 100)) : 0;

    const metricCells = metricConfig.map((cfg) => {
      const value = parseNumber(row[cfg.column]);
      const normalized = normalizeMetricValue(value, cfg.direction, transforms[cfg.column]);
      const strength = Number.isFinite(normalized) ? 0.12 + normalized * 0.58 : 0.06;
      return `<td><div class="heatmapCell" style="--strength:${strength.toFixed(3)}">${formatCell(value, 3)}</div></td>`;
    }).join("");

    return `
      <tr>
        <td class="stickyCol rankCol"><span class="rankBadge">#${row.rank}</span></td>
        <td class="stickyCol methodCol">
          <div class="methodCell">
            <span class="methodName">${escapeHtml(row.method)}</span>
            <span class="methodMeta">${row.count} benchmark row${row.count === 1 ? "" : "s"}</span>
          </div>
        </td>
        <td>
          <div class="scoreCell">
            <div class="scoreTop"><span>${formatCell(row.score, 3)}</span><span>${scorePct.toFixed(1)}%</span></div>
            <div class="miniTrack"><div class="miniFill" style="width:${scorePct.toFixed(2)}%"></div></div>
          </div>
        </td>
        ${metricCells}
      </tr>
    `;
  }).join("");
}

function rerenderResponsiveVisuals() {
  if (!state.latestVisuals || !els.radarPlot || els.visualSection.classList.contains("hidden")) {
    return;
  }

  const { rows, metricConfig, transforms } = state.latestVisuals;
  renderRadarPlot(rows, metricConfig, transforms);
}

function rankMethods() {
  try {
    const dataset = state.selectedDataset;
    const model = state.selectedModel;
    const metricConfig = selectedMetricConfig();

    if (!dataset || !model) {
      clearResults("Select both dataset and model first.");
      return;
    }

    if (!metricConfig.length) {
      clearResults("Select at least one metric.");
      return;
    }

    const metricColumns = metricConfig.map((m) => m.column);
    const aggregateColumns = [...new Set([...metricColumns, "validity", "distance_l2"])];

    const contextRows = state.rows
      .filter((r) => isSuccessful(r))
      .filter((r) => String(r.dataset) === dataset)
      .filter((r) => String(r.model) === model);

    if (!contextRows.length) {
      clearResults("No successful rows found for this dataset/model.");
      return;
    }

    const methodSubset = new Set(getMethodSubset());

    const filteredRows = contextRows.filter((r) => methodSubset.has(String(r.method)));
    if (!filteredRows.length) {
      clearResults("No rows remain after method filtering.");
      return;
    }

    const baselineAggregated = aggregateByMethod(contextRows, aggregateColumns);
    const aggregated = aggregateByMethod(filteredRows, aggregateColumns);

    if (!aggregated.length) {
      clearResults("No methods available for ranking.");
      return;
    }

    const transforms = computeMetricTransforms(baselineAggregated, metricConfig);
    const ranked = scoreRows(aggregated, metricConfig, transforms);

    els.statusMessage.textContent = "Benchmark updated.";
    els.summary.textContent = `Rows evaluated: ${filteredRows.length}. Methods ranked: ${ranked.length}. Scoring uses exponential utility transforms per metric, then averages the resulting higher-is-better values.`;
    if (els.tableMeta) {
      els.tableMeta.innerHTML = [
        `<span class="metaChip">Compatible set: ${methodSubset.size}</span>`,
        `<span class="metaChip">Ranked methods: ${ranked.length}</span>`,
        `<span class="metaChip">Raw values shown, utility drives score</span>`,
      ].join("");
    }

    state.latestVisuals = { rows: ranked, metricConfig, transforms };
    showResultPanels();
    renderRadarPlot(ranked, metricConfig, transforms);
    renderTable(ranked, metricConfig, transforms);
  } catch (error) {
    clearResults(`Ranking error: ${error.message}`);
  }
}

function setMethodsModeAll() {
  state.useAllMethods = true;
  state.selectedMethods.clear();
  renderConfigurationControls();
  updateSelectionUI();
  markResultsStale();
}

function setMethodsModeCustom() {
  state.useAllMethods = false;
  normalizeMethodSelectionState();
  renderConfigurationControls();
  updateSelectionUI();
  markResultsStale();
}

function toggleMethodSelection(method) {
  if (!isMethodCompatibleWithScope(method)) return;

  if (state.useAllMethods) {
    state.useAllMethods = false;
    state.selectedMethods = new Set([method]);
  } else if (state.selectedMethods.has(method)) {
    state.selectedMethods.delete(method);
    if (!state.selectedMethods.size) {
      state.useAllMethods = true;
    }
  } else {
    state.selectedMethods.add(method);
  }

  normalizeMethodSelectionState();
  renderConfigurationControls();
  updateSelectionUI();
  markResultsStale();
}

function selectAllMethodsCustom() {
  state.useAllMethods = false;
  state.selectedMethods = getCompatibleMethodSet();
  normalizeMethodSelectionState();
  renderConfigurationControls();
  updateSelectionUI();
  markResultsStale();
}

function clearMethodsCustom() {
  state.useAllMethods = false;
  state.selectedMethods.clear();
  normalizeMethodSelectionState();
  renderConfigurationControls();
  updateSelectionUI();
  markResultsStale();
}

function selectAllMetrics() {
  state.selectedMetrics = new Set(METRIC_DEFS.map((m) => m.id));
  state.activePreset = "balanced";
  renderMetricPalette();
  updateSelectionUI();
  markResultsStale();
}

function clearMetrics() {
  state.selectedMetrics.clear();
  state.activePreset = "";
  renderMetricPalette();
  updateSelectionUI();
  markResultsStale();
}

function applyMetricPreset(name) {
  const preset = METRIC_PRESETS[name];
  if (!preset) return;
  state.selectedMetrics = new Set(preset);
  state.activePreset = name;
  renderMetricPalette();
  updateSelectionUI();
  markResultsStale();
}

function toggleMetric(metricId) {
  if (state.selectedMetrics.has(metricId)) {
    state.selectedMetrics.delete(metricId);
  } else {
    state.selectedMetrics.add(metricId);
  }
  state.activePreset = "";
  renderMetricPalette();
  updateSelectionUI();
  markResultsStale();
}

function resetFilters() {
  state.selectedDataset = "";
  state.selectedModel = "";
  state.useAllMethods = true;
  state.selectedMethods.clear();
  state.selectedMetrics = new Set(METRIC_DEFS.map((m) => m.id));
  state.activePreset = "balanced";

  renderConfigurationControls();
  renderMetricPalette();
  updateSelectionUI();
  clearResults("Filters reset. Choose scope and click Show Benchmark.");
}

async function parseCompatibility() {
  const response = await fetch(COMPATIBILITY_PATH, { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`Compatibility manifest request failed (${response.status})`);
  }

  return normalizeCompatibility(await response.json());
}

function parseLocalCsv() {
  return new Promise((resolve, reject) => {
    if (!window.Papa) {
      reject(new Error("PapaParse is not loaded."));
      return;
    }

    Papa.parse(LOCAL_DATA_PATH, {
      download: true,
      header: true,
      skipEmptyLines: true,
      complete: (result) => {
        if (result.errors && result.errors.length) {
          reject(new Error(result.errors[0].message));
          return;
        }
        resolve(result.data || []);
      },
      error: reject,
    });
  });
}

function sanitizeRows(rows) {
  return pickBenchmarkRows(rows)
    .filter((row) => row && typeof row === "object")
    .map(normalizeBenchmarkRow)
    .filter((row) => row.dataset && row.model && row.method);
}

function initFromRows(rows, compatibility) {
  state.rows = sanitizeRows(rows);
  state.options.datasets = uniqueSorted(state.rows.map((r) => r.dataset));
  state.options.models = uniqueSorted(state.rows.map((r) => r.model));
  state.options.methods = uniqueSorted(state.rows.map((r) => r.method));
  state.compatibility = normalizeCompatibility(compatibility);
  state.metricBounds = computeMetricBounds(state.rows.filter((r) => isSuccessful(r)));

  state.selectedDataset = "";
  state.selectedModel = "";
  state.useAllMethods = true;
  state.selectedMethods.clear();
  state.selectedMetrics = new Set(METRIC_DEFS.map((m) => m.id));
  state.activePreset = "balanced";
  state.latestVisuals = null;

  renderConfigurationControls();
  renderMetricPalette();
  updateSelectionUI();
  clearResults("Select dataset/model and click Show Benchmark.");
}

async function loadData() {
  try {
    const [rows, compatibility] = await Promise.all([
      parseLocalCsv(),
      parseCompatibility(),
    ]);
    initFromRows(rows, compatibility);
  } catch (error) {
    clearResults(`Failed to load live data: ${error.message}`);
  }
}

els.rankBtn.addEventListener("click", rankMethods);
els.resetBtn.addEventListener("click", resetFilters);
els.reloadBtn.addEventListener("click", loadData);

els.modeAllMethodsBtn.addEventListener("click", setMethodsModeAll);
els.modeCustomMethodsBtn.addEventListener("click", setMethodsModeCustom);
els.selectAllMethodsBtn.addEventListener("click", selectAllMethodsCustom);
els.clearMethodsBtn.addEventListener("click", clearMethodsCustom);

els.selectAllMetricsBtn.addEventListener("click", selectAllMetrics);
els.clearMetricsBtn.addEventListener("click", clearMetrics);

els.datasetSelect.addEventListener("change", (event) => {
  state.selectedDataset = String(event.target.value || "");
  normalizeMethodSelectionState();
  renderConfigurationControls();
  updateSelectionUI();
  markResultsStale();
});

els.modelSelect.addEventListener("change", (event) => {
  state.selectedModel = String(event.target.value || "");
  normalizeMethodSelectionState();
  renderConfigurationControls();
  updateSelectionUI();
  markResultsStale();
});

els.methodChecklist.addEventListener("change", (event) => {
  const input = event.target.closest("input[type='checkbox'][data-method]");
  if (!input) return;

  const compatibleMethodSet = getCompatibleMethodSet();
  const nextSelected = state.useAllMethods
    ? new Set(compatibleMethodSet)
    : new Set(getEffectiveSelectedMethods());

  const method = String(input.dataset.method || "").trim();
  if (!method) return;

  if (input.checked) {
    nextSelected.add(method);
  } else {
    nextSelected.delete(method);
  }

  if (nextSelected.size === compatibleMethodSet.size) {
    state.useAllMethods = true;
    state.selectedMethods.clear();
  } else {
    state.useAllMethods = false;
    state.selectedMethods = nextSelected;
  }
  normalizeMethodSelectionState();
  renderConfigurationControls();
  updateSelectionUI();
  markResultsStale();
});

els.metricPalette.addEventListener("click", (event) => {
  const button = event.target.closest("button.token");
  if (!button || !button.dataset.metricId) return;
  toggleMetric(button.dataset.metricId);
});

els.presetBtns.forEach((button) => {
  button.addEventListener("click", () => applyMetricPreset(button.dataset.preset));
});

window.addEventListener("resize", () => {
  if (visualResizeTimer) {
    clearTimeout(visualResizeTimer);
  }

  visualResizeTimer = setTimeout(() => {
    rerenderResponsiveVisuals();
  }, 120);
});

loadData();
















































