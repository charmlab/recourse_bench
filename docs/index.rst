RecourseBench documentation
============================

**RecourseBench** is a modular framework for reproducible algorithmic recourse
evaluation. It composes five extensible component types — datasets,
preprocessing steps, target models, recourse methods, and evaluation metrics —
into experiments that are configured as data and run end to end.

Live example
------------

Edit the factual case and select a recourse method to see the kind of result
RecourseBench evaluates: a counterfactual that changes the model prediction with
a small set of feature edits.

.. raw:: html

   <style>
     .rb-home-demo {
       border: 1px solid var(--pst-color-border);
       border-radius: 8px;
       margin: 1rem 0 1.75rem;
       overflow: hidden;
       background: var(--pst-color-surface);
       box-shadow: 0 8px 24px rgba(15, 23, 42, 0.08);
     }
     .rb-home-demo__top {
       display: flex;
       justify-content: space-between;
       gap: 1rem;
       padding: 1rem;
       border-bottom: 1px solid var(--pst-color-border);
       background: color-mix(in srgb, var(--pst-color-primary) 8%, transparent);
     }
     .rb-home-demo__title {
       margin: 0;
       font-size: 1rem;
       font-weight: 700;
     }
     .rb-home-demo__select {
       min-width: 12rem;
       border: 1px solid var(--pst-color-border);
       border-radius: 6px;
       padding: 0.45rem 0.6rem;
       background: var(--pst-color-background);
       color: var(--pst-color-text-base);
     }
     .rb-home-demo__body {
       display: grid;
       grid-template-columns: minmax(0, 0.9fr) minmax(0, 1.1fr);
     }
     .rb-home-demo__panel {
       padding: 1rem;
     }
     .rb-home-demo__panel + .rb-home-demo__panel {
       border-left: 1px solid var(--pst-color-border);
     }
     .rb-home-demo__label {
       color: var(--pst-color-text-muted);
       font-size: 0.78rem;
       font-weight: 700;
       letter-spacing: 0.03em;
       margin-bottom: 0.55rem;
       text-transform: uppercase;
     }
     .rb-home-demo__row {
       display: flex;
       justify-content: space-between;
       gap: 1rem;
       border-bottom: 1px dashed var(--pst-color-border);
       padding: 0.35rem 0;
       align-items: center;
     }
     .rb-home-demo__row span:first-child {
       color: var(--pst-color-text-muted);
     }
     .rb-home-demo__input {
       width: 7.25rem;
       border: 1px solid var(--pst-color-border);
       border-radius: 6px;
       padding: 0.3rem 0.45rem;
       background: var(--pst-color-background);
       color: var(--pst-color-text-base);
       font: inherit;
       text-align: right;
     }
     .rb-home-demo__input:focus,
     .rb-home-demo__select:focus {
       outline: 2px solid color-mix(in srgb, var(--pst-color-primary) 34%, transparent);
       outline-offset: 1px;
     }
     .rb-home-demo__prediction {
       display: inline-flex;
       align-items: center;
       border-radius: 999px;
       padding: 0.18rem 0.6rem;
       font-weight: 700;
       white-space: nowrap;
     }
     .rb-home-demo__prediction--denied {
       background: color-mix(in srgb, #dc2626 12%, transparent);
       color: #b91c1c;
     }
     .rb-home-demo__prediction--approved {
       background: color-mix(in srgb, #16a34a 14%, transparent);
       color: #15803d;
     }
     .rb-home-demo__actions {
       display: flex;
       align-items: center;
       gap: 0.75rem;
       margin-top: 0.85rem;
       flex-wrap: wrap;
     }
     .rb-home-demo__button {
       border: 1px solid var(--pst-color-primary);
       border-radius: 6px;
       padding: 0.45rem 0.7rem;
       background: var(--pst-color-primary);
       color: #fff;
       font: inherit;
       font-weight: 700;
       cursor: pointer;
     }
     .rb-home-demo__button:hover {
       filter: brightness(0.95);
     }
     .rb-home-demo__status {
       color: var(--pst-color-text-muted);
       font-size: 0.86rem;
     }
     .rb-home-demo__outcome {
       border: 1px solid color-mix(in srgb, #16a34a 34%, transparent);
       border-radius: 6px;
       margin-bottom: 0.75rem;
       padding: 0.75rem;
       background: color-mix(in srgb, #16a34a 12%, transparent);
       font-weight: 700;
     }
     .rb-home-demo__changes {
       display: grid;
       gap: 0.5rem;
       margin-bottom: 0.75rem;
     }
     .rb-home-demo__change {
       display: grid;
       grid-template-columns: 1fr auto;
       gap: 0.75rem;
       align-items: center;
       border: 1px solid var(--pst-color-border);
       border-radius: 6px;
       padding: 0.55rem 0.65rem;
       background: var(--pst-color-background);
     }
     .rb-home-demo__change strong,
     .rb-home-demo__metric strong {
       display: block;
     }
     .rb-home-demo__change small,
     .rb-home-demo__metric span {
       color: var(--pst-color-text-muted);
     }
     .rb-home-demo__badge {
       border-radius: 999px;
       padding: 0.2rem 0.55rem;
       background: color-mix(in srgb, var(--pst-color-primary) 14%, transparent);
       color: var(--pst-color-primary);
       font-size: 0.78rem;
       font-weight: 700;
       white-space: nowrap;
     }
     .rb-home-demo__metrics {
       display: grid;
       grid-template-columns: repeat(3, minmax(0, 1fr));
       gap: 0.5rem;
     }
     .rb-home-demo__metric {
       border: 1px solid var(--pst-color-border);
       border-radius: 6px;
       padding: 0.65rem;
       background: var(--pst-color-background);
     }
     @media (max-width: 760px) {
       .rb-home-demo__top,
       .rb-home-demo__body {
         display: block;
       }
       .rb-home-demo__select {
         width: 100%;
         margin-top: 0.75rem;
       }
       .rb-home-demo__panel + .rb-home-demo__panel {
         border-left: 0;
         border-top: 1px solid var(--pst-color-border);
       }
       .rb-home-demo__metrics {
         grid-template-columns: 1fr;
       }
     }
   </style>
   <section class="rb-home-demo" aria-label="Interactive RecourseBench live example">
     <div class="rb-home-demo__top">
       <div>
         <p class="rb-home-demo__title">Toy credit recourse run</p>
         <div>Editable applicant, same toy model, different recourse method.</div>
       </div>
       <label>
         <span class="sr-only">Select method</span>
         <select id="rbHomeDemoMethod" class="rb-home-demo__select">
           <option value="wachter">Wachter</option>
           <option value="dice">DiCE</option>
           <option value="gs">Growing Spheres</option>
         </select>
       </label>
     </div>
     <div class="rb-home-demo__body">
       <div class="rb-home-demo__panel">
         <div class="rb-home-demo__label">Factual case</div>
         <div class="rb-home-demo__row"><span>Prediction</span><strong id="rbHomeDemoPrediction" class="rb-home-demo__prediction"></strong></div>
         <label class="rb-home-demo__row"><span>Income ($k)</span><input id="rbHomeDemoIncome" class="rb-home-demo__input" type="number" min="20" max="120" step="1" value="42"></label>
         <label class="rb-home-demo__row"><span>Savings ($k)</span><input id="rbHomeDemoSavings" class="rb-home-demo__input" type="number" min="0" max="60" step="0.1" value="3.2"></label>
         <label class="rb-home-demo__row"><span>Debt ratio (%)</span><input id="rbHomeDemoDebt" class="rb-home-demo__input" type="number" min="5" max="75" step="1" value="41"></label>
         <label class="rb-home-demo__row"><span>Late payments</span><input id="rbHomeDemoLate" class="rb-home-demo__input" type="number" min="0" max="6" step="1" value="2"></label>
         <div class="rb-home-demo__actions">
           <button id="rbHomeDemoApply" class="rb-home-demo__button" type="button">Apply edits and check</button>
           <span id="rbHomeDemoStatus" class="rb-home-demo__status"></span>
         </div>
       </div>
       <div class="rb-home-demo__panel">
         <div class="rb-home-demo__label">Counterfactual result</div>
         <div id="rbHomeDemoOutcome" class="rb-home-demo__outcome"></div>
         <div id="rbHomeDemoChanges" class="rb-home-demo__changes"></div>
         <div class="rb-home-demo__metrics">
           <div class="rb-home-demo__metric"><strong id="rbHomeDemoValidity"></strong><span>validity</span></div>
           <div class="rb-home-demo__metric"><strong id="rbHomeDemoDistance"></strong><span>L1 distance</span></div>
           <div class="rb-home-demo__metric"><strong id="rbHomeDemoChanged"></strong><span>features changed</span></div>
         </div>
       </div>
     </div>
   </section>
   <script>
     (function () {
       const select = document.getElementById("rbHomeDemoMethod");
       const prediction = document.getElementById("rbHomeDemoPrediction");
       const inputs = {
         income: document.getElementById("rbHomeDemoIncome"),
         savings: document.getElementById("rbHomeDemoSavings"),
         debt: document.getElementById("rbHomeDemoDebt"),
         late: document.getElementById("rbHomeDemoLate")
       };
       const applyButton = document.getElementById("rbHomeDemoApply");
       const status = document.getElementById("rbHomeDemoStatus");
       const outcome = document.getElementById("rbHomeDemoOutcome");
       const changes = document.getElementById("rbHomeDemoChanges");
       const validity = document.getElementById("rbHomeDemoValidity");
       const distance = document.getElementById("rbHomeDemoDistance");
       const changed = document.getElementById("rbHomeDemoChanged");
       if (!select || !prediction || !applyButton || !status || !outcome || !changes || !validity || !distance || !changed) return;
       function escapeHtml(value) {
         return String(value).replace(/[&<>"']/g, function (char) {
           return {"&": "&amp;", "<": "&lt;", ">": "&gt;", "\"": "&quot;", "'": "&#39;"}[char];
         });
       }
       function clamp(value, min, max) {
         return Math.min(max, Math.max(min, value));
       }
       function readInput(input, fallback) {
         const value = Number.parseFloat(input.value);
         const min = Number.parseFloat(input.min);
         const max = Number.parseFloat(input.max);
         return Number.isFinite(value) ? clamp(value, min, max) : fallback;
       }
       function score(row) {
         return (0.055 * row.income) + (0.11 * row.savings) - (0.075 * row.debt) - (0.65 * row.late) - 1.15;
       }
       function label(row) {
         return score(row) >= 0 ? "Approved" : "Denied";
       }
       function clone(row) {
         return {income: row.income, savings: row.savings, debt: row.debt, late: row.late};
       }
       function applyDelta(row, key, delta) {
         const ranges = {
           income: [20, 120],
           savings: [0, 60],
           debt: [5, 75],
           late: [0, 6]
         };
         row[key] = clamp(row[key] + delta, ranges[key][0], ranges[key][1]);
       }
       function improve(row, sequence) {
         const cf = clone(row);
         let guard = 0;
         while (score(cf) < 0.12 && guard < 120) {
           const before = JSON.stringify(cf);
           sequence.forEach(function (step) {
             if (score(cf) < 0.12) applyDelta(cf, step[0], step[1]);
           });
           if (JSON.stringify(cf) === before) break;
           guard += 1;
         }
         cf.income = Math.round(cf.income);
         cf.savings = Math.round(cf.savings * 10) / 10;
         cf.debt = Math.round(cf.debt);
         cf.late = Math.round(cf.late);
         return cf;
       }
       function recourse(row, method) {
         if (score(row) >= 0) return clone(row);
         if (method === "dice") {
           return improve(row, [["income", 2], ["savings", 1.2], ["late", -1], ["debt", -2]]);
         }
         if (method === "gs") {
           return improve(row, [["savings", 1.8], ["debt", -2], ["income", 1]]);
         }
         return improve(row, [["debt", -2], ["income", 2], ["savings", 0.8]]);
       }
       function formatFeature(key, value) {
         if (key === "income") return "$" + Math.round(value) + "k";
         if (key === "savings") return "$" + value.toFixed(1) + "k";
         if (key === "debt") return Math.round(value) + "%";
         return String(Math.round(value));
       }
       function featureName(key) {
         return {income: "Income", savings: "Savings", debt: "Debt ratio", late: "Late payments"}[key];
       }
       function deltaLabel(key, from, to) {
         const delta = to - from;
         const sign = delta > 0 ? "+" : "";
         if (key === "income") return sign + Math.round(delta) + "k";
         if (key === "savings") return sign + delta.toFixed(1) + "k";
         if (key === "debt") return sign + Math.round(delta) + " pts";
         return sign + Math.round(delta);
       }
       function distanceL1(row, cf) {
         return (
           Math.abs(cf.income - row.income) / 100 +
           Math.abs(cf.savings - row.savings) / 60 +
           Math.abs(cf.debt - row.debt) / 70 +
           Math.abs(cf.late - row.late) / 6
         );
       }
       function readRow() {
         return {
           income: readInput(inputs.income, 42),
           savings: readInput(inputs.savings, 3.2),
           debt: readInput(inputs.debt, 41),
           late: readInput(inputs.late, 2)
         };
       }
       function updatePrediction(row) {
         prediction.textContent = label(row);
         prediction.className = "rb-home-demo__prediction rb-home-demo__prediction--" + label(row).toLowerCase();
       }
       function markPending() {
         updatePrediction(readRow());
         status.textContent = "Edits changed. Click Apply edits and check.";
       }
       function render() {
         const row = readRow();
         const cf = recourse(row, select.value);
         const isApproved = label(row) === "Approved";
         const cfApproved = label(cf) === "Approved";
         const changedItems = ["income", "savings", "debt", "late"].filter(function (key) {
           return Math.abs(cf[key] - row[key]) > 0.0001;
         });
         updatePrediction(row);
         outcome.textContent = isApproved
           ? "Already approved; no counterfactual edits needed"
           : (cfApproved ? "Approved after " + changedItems.length + " feature edits" : "No approved point found within the toy bounds");
         validity.textContent = cfApproved ? "1.00" : "0.00";
         distance.textContent = distanceL1(row, cf).toFixed(2);
         changed.textContent = String(changedItems.length);
         status.textContent = "Checked current edits.";
         changes.innerHTML = changedItems.length
           ? changedItems.map(function (key) {
               return '<div class="rb-home-demo__change"><div><strong>' + escapeHtml(featureName(key)) +
                 '</strong><small>' + escapeHtml(formatFeature(key, row[key])) + ' -> ' + escapeHtml(formatFeature(key, cf[key])) +
                 '</small></div><span class="rb-home-demo__badge">' + escapeHtml(deltaLabel(key, row[key], cf[key])) + '</span></div>';
             }).join("")
           : '<div class="rb-home-demo__change"><div><strong>No edits</strong><small>The factual case is already predicted approved.</small></div><span class="rb-home-demo__badge">0</span></div>';
       }
       select.addEventListener("change", markPending);
       applyButton.addEventListener("click", render);
       Object.keys(inputs).forEach(function (key) {
         if (inputs[key]) inputs[key].addEventListener("input", markPending);
       });
       render();
     }());
   </script>

.. grid:: 1 2 2 2
   :gutter: 3

   .. grid-item-card:: Getting started
      :link: getting_started
      :link-type: doc

      Install the package, run your first experiment from a YAML config, and
      use the Python API.

   .. grid-item-card:: API reference
      :link: reference/index
      :link-type: doc

      Full description of every public class and function: arguments, return
      values, and behaviour.

   .. grid-item-card:: Methods
      :link: methods
      :link-type: doc

      Browse all registered recourse methods and open the RecourseBench paper.

   .. grid-item-card:: Extending the framework
      :link: extending
      :link-type: doc

      Register new datasets, preprocessing, models, methods, and metrics.

   .. grid-item-card:: Agent tools
      :link: agent_tools
      :link-type: doc

      Use the RecourseBench skills and MCP server from coding agents without
      giving up reproducibility or safety.

.. toctree::
   :maxdepth: 2
   :hidden:

   getting_started
   methods
   extending
   agent_tools
   reference/index
