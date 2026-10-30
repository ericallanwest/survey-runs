"use strict";
// Same road, different day: the change map with route profiles, the scorecard, the odometer
// check and the repeatability panel. Data in docs/data, built by `survey-runs site`.

(async function main() {
  const css = getComputedStyle(document.documentElement);
  const token = (name) => css.getPropertyValue(name).trim();
  const dark = matchMedia("(prefers-color-scheme: dark)").matches;
  const fmt = new Intl.NumberFormat("en-US");
  const pct = (v, d = 1) => `${(v * 100).toFixed(d)}%`;
  const signed = (v, d = 1) => (v == null ? "–" : `${v > 0 ? "+" : v < 0 ? "−" : ""}${Math.abs(v).toFixed(d)}`);
  const esc = (s) => String(s).replace(/[&<>"']/g, (c) => `&#${c.charCodeAt(0)};`);
  const svgNS = "http://www.w3.org/2000/svg";
  const getJSON = (url) => fetch(url).then((r) => {
    if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`);
    return r.json();
  });

  const site = await getJSON("data/site.json");
  const S = site.scores;
  const YEARS = site.program.years;
  const METHODS = [   // [key, label, in a sentence, color token, dashed]
    ["gps", "GPS", "GPS", "--s1", false],
    ["chainage", "Odometer", "odometer", "--s2", false],
    ["lrs", "LRS measure", "LRS measure", "--s3", false],
    ["oracle", "True position", "true position", "--context", true],
  ];
  const METHOD = Object.fromEntries(METHODS.map(([k, label, inline, color, dash]) =>
    [k, { label, inline, color: token(color), dash }]));
  const VEHICLE = { SV1: token("--v1"), SV2: token("--v2") };

  // "060-KY-0007-000" -> "KY 7"
  const routeLabel = (route) => {
    const [, prefix, number] = route.split("-");
    return `${prefix} ${Number(number.replace(/\D/g, ""))}${number.replace(/^\d+/, "")}`;
  };

  fillText();
  renderKpis();
  renderScores();
  renderDmi();
  initRepeat();
  initMap();

  // ======================= text and headline numbers =======================
  function fillText() {
    document.getElementById("nMiles").textContent = fmt.format(Math.round(site.network.mainline_miles));
    document.getElementById("nSegments").textContent = `${(site.program.segments / 1e6).toFixed(1)} million`;
    document.getElementById("nRuns").textContent = fmt.format(site.program.network_runs);
    document.getElementById("fetched").textContent = site.network.fetched;
  }

  function renderKpis() {
    const m = S.matching, cd = S.change_detection;
    const kpis = [
      [fmt.format(Math.round(site.network.mainline_miles)),
        `miles of US and KY routes, surveyed in both directions over ${YEARS.length} years`],
      [`${(site.program.segments / 1e6).toFixed(1)}M`, `26.4-ft segments from ${fmt.format(site.program.network_runs)} runs`],
      [pct(m.route_exact, 2), "of segments matched to the carriageway actually driven"],
      [pct(cd.precision), `of flagged resurfacing is real, and ${pct(cd.recall)} of it is found`],
    ];
    document.getElementById("kpis").innerHTML = kpis.map(([v, l]) =>
      `<div class="kpi"><div class="value">${v}</div><div class="label">${l}</div></div>`).join("");
  }

  function renderScores() {
    const m = S.matching, c = S.cells, d = S.dmi, cd = S.change_detection;
    const found = d.vehicle_years_flagged.filter((v) => d.vehicle_years_miscalibrated.includes(v));
    const falseAlarms = d.vehicle_years_flagged.length - found.length;
    const e = m.measure_error_ft;
    const rows = [
      ["Matching each segment to the road"],
      ["Matched to the route actually driven", pct(m.route_exact, 2)],
      ["Not matched at all", pct(m.unmatched, 2)],
      ["Direction of travel right", pct(m.direction_right, 4)],
      ["Measure error: median, 90th, 99th percentile", `${e.p50.toFixed(1)}, ${e.p90.toFixed(1)}, ${e.p99.toFixed(1)} ft`],
      ["Placed by odometer where GPS froze", `${fmt.format(m.interpolated)} segments`],
      ["Year-over-year grid"],
      ["Reading in its true 0.005-mi cell", `${pct(c.exact_cell)} (${pct(c.within_one_cell, 2)} within one cell)`],
      ["Divided roads' far carriageway, in its true cell", `${pct(c.twin_rows_exact)} (${pct(c.twin_rows_within_one_cell, 2)} within one)`],
      ["Odometer check"],
      ["Miscalibrated vehicle-years found", `${found.length} of ${d.vehicle_years_miscalibrated.length}, ` +
        `${falseAlarms ? `${falseAlarms} false alarm${falseAlarms > 1 ? "s" : ""}` : "no false alarms"}`],
      ["Worst vehicle-year estimate", `off by ${(d.vehicle_year_abs_error_max * 100).toFixed(3)} points of a percent`],
      ["Finding resurfacing"],
      ["Flagged stretches really resurfaced (precision)", pct(cd.precision)],
      ["Resurfaced stretches flagged (recall)", pct(cd.recall)],
    ];
    document.getElementById("scoreTable").innerHTML = `<thead><tr><th>Check</th><th>Result</th></tr></thead><tbody>` +
      rows.map((r) => r.length === 1 ? `<tr><td class="group" colspan="2">${r[0]}</td></tr>`
        : `<tr><td>${r[0]}</td><td>${r[1]}</td></tr>`).join("") + "</tbody>";
  }

  // ======================= a small SVG chart kit =======================
  function svgIn(host, W, H, aria) {
    const svg = document.createElementNS(svgNS, "svg");
    svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
    svg.setAttribute("role", "img");
    svg.setAttribute("aria-label", aria);
    host.appendChild(svg);
    const add = (tag, attrs, text, parent = svg) => {
      const el = document.createElementNS(svgNS, tag);
      for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
      if (text !== undefined) el.textContent = text;
      parent.appendChild(el);
      return el;
    };
    return { svg, add };
  }

  // nearest index in an ascending array
  function nearest(xs, v) {
    let lo = 0, hi = xs.length - 1;
    while (hi - lo > 1) {
      const mid = (lo + hi) >> 1;
      if (xs[mid] < v) lo = mid; else hi = mid;
    }
    return Math.abs(xs[lo] - v) <= Math.abs(xs[hi] - v) ? lo : hi;
  }

  // Line chart: series {name, color, xs, ys (null = gap), dash, width, label}; spans shade x ranges,
  // bars mark x ranges along the top; tip(i0, x) returns the hover tooltip's HTML.
  function lineChart(host, o) {
    const W = o.W || 820, H = o.H || 250;
    const m = { l: 44, r: o.labels === false ? 16 : 96, t: o.bars?.length ? 18 : 12, b: 32, ...o.m };
    const X = (v) => m.l + ((v - o.x.min) / (o.x.max - o.x.min)) * (W - m.l - m.r);
    const Y = (v) => m.t + (1 - (v - o.y.min) / (o.y.max - o.y.min)) * (H - m.t - m.b);
    const { svg, add } = svgIn(host, W, H, o.aria);
    for (const t of o.y.ticks) {
      add("line", { class: "grid", x1: m.l, x2: W - m.r, y1: Y(t), y2: Y(t) });
      add("text", { x: m.l - 6, y: Y(t) + 4, "text-anchor": "end" }, o.y.fmt ? o.y.fmt(t) : t);
    }
    for (const t of o.x.ticks) add("text", { x: X(t), y: H - m.b + 16, "text-anchor": "middle" }, o.x.fmt ? o.x.fmt(t) : t);
    if (o.x.label) add("text", { x: W - m.r, y: H - 2, "text-anchor": "end" }, o.x.label);
    if (o.y.label) add("text", { x: m.l, y: m.t - 3 }, o.y.label);
    add("line", { class: "axis", x1: m.l, x2: W - m.r, y1: H - m.b, y2: H - m.b });
    for (const r of o.refs || []) {
      add("line", { class: "ref", x1: m.l, x2: W - m.r, y1: Y(r.y), y2: Y(r.y) });
      add("text", { class: "ref-label", x: m.l + 6, y: Y(r.y) - 5 }, r.label);
    }
    for (const s of o.spans || []) {
      const x0 = X(Math.max(s.x0, o.x.min)), x1 = X(Math.min(s.x1, o.x.max));
      if (x1 > x0) add("rect", { x: x0, y: m.t, width: Math.max(x1 - x0, 1), height: H - m.t - m.b, fill: s.fill, opacity: s.opacity ?? 0.14 });
    }
    for (const b of o.bars || []) {
      const x0 = X(Math.max(b.x0, o.x.min)), x1 = X(Math.min(b.x1, o.x.max));
      if (x1 > x0) add("rect", { x: x0, y: m.t - 10, width: Math.max(x1 - x0, 2), height: 4, rx: 2, fill: b.fill });
    }
    for (const s of o.series) {
      let d = "", pen = false;
      s.xs.forEach((x, i) => {
        const y = s.ys[i];
        if (y == null) { pen = false; return; }
        d += `${pen ? "L" : "M"}${X(x).toFixed(1)},${Y(y).toFixed(1)}`;
        pen = true;
      });
      add("path", { class: "line", d, stroke: s.color, "stroke-width": s.width || 2, ...(s.dash ? { "stroke-dasharray": "5 4" } : {}) });
    }
    // direct labels at each labelled series' last point, nudged apart
    const ends = o.series.filter((s) => s.label).map((s) => {
      let i = s.ys.length - 1;
      while (i >= 0 && s.ys[i] == null) i--;
      return i < 0 ? null : { s, y: Y(s.ys[i]) };
    }).filter(Boolean).sort((a, b) => a.y - b.y);
    for (let k = 1; k < ends.length; k++) ends[k].y = Math.max(ends[k].y, ends[k - 1].y + 13);
    for (const e of ends) add("text", { class: "end-label", x: W - m.r + 6, y: e.y + 4 }, e.s.label);

    // hover: crosshair, a dot per series, and the tooltip
    const cross = add("line", { class: "cross", y1: m.t, y2: H - m.b, visibility: "hidden" });
    const dots = o.series.map((s) => add("circle", { r: 3.5, fill: s.color, stroke: token("--surface"), "stroke-width": 1.5, visibility: "hidden" }));
    const hit = add("rect", { x: m.l, y: m.t, width: W - m.l - m.r, height: H - m.t - m.b, fill: "transparent" });
    const tip = document.createElement("div");
    tip.className = "tip";
    tip.hidden = true;
    host.appendChild(tip);
    const ref = o.series[0].xs;
    hit.addEventListener("pointermove", (evt) => {
      const box = svg.getBoundingClientRect();
      const sx = ((evt.clientX - box.left) / box.width) * W;
      const xv = o.x.min + ((sx - m.l) / (W - m.l - m.r)) * (o.x.max - o.x.min);
      const i0 = nearest(ref, xv);
      const px = X(ref[i0]);
      cross.setAttribute("x1", px); cross.setAttribute("x2", px); cross.setAttribute("visibility", "visible");
      o.series.forEach((s, k) => {
        const i = nearest(s.xs, ref[i0]), y = s.ys[i];
        dots[k].setAttribute("visibility", y == null ? "hidden" : "visible");
        if (y != null) { dots[k].setAttribute("cx", X(s.xs[i])); dots[k].setAttribute("cy", Y(y)); }
      });
      tip.hidden = false;
      tip.innerHTML = o.tip(i0, ref[i0]);
      const left = (px / W) * box.width + 14;
      tip.style.left = `${left + tip.offsetWidth > box.width ? left - tip.offsetWidth - 28 : left}px`;
      tip.style.top = `${12 + (m.t / H) * box.height}px`;
    });
    hit.addEventListener("pointerleave", () => {
      cross.setAttribute("visibility", "hidden");
      dots.forEach((dt) => dt.setAttribute("visibility", "hidden"));
      tip.hidden = true;
    });
  }

  function niceTicks(lo, hi, step) {
    const out = [];
    for (let k = Math.ceil(lo / step - 1e-9); k * step <= hi + 1e-9; k++) out.push(+(k * step).toFixed(6));
    return out;
  }
  function mileStep(span) {
    return span > 40 ? 10 : span > 16 ? 5 : span > 6 ? 2 : span > 2.5 ? 0.5 : span > 1 ? 0.25 : 0.05;
  }

  function cardIn(host, title, sub) {
    const card = document.createElement("div");
    card.className = "chart";
    card.innerHTML = `<h4>${title}</h4>${sub ? `<p class="sub">${sub}</p>` : ""}`;
    host.appendChild(card);
    return card;
  }

  function tableToggle(btnId, tableId) {
    const btn = document.getElementById(btnId), table = document.getElementById(tableId);
    btn.addEventListener("click", () => {
      table.hidden = !table.hidden;
      btn.setAttribute("aria-expanded", String(!table.hidden));
      btn.textContent = table.hidden ? "Show as a table" : "Hide the table";
    });
  }

  // ======================= odometer check =======================
  function renderDmi() {
    const rows = S.dmi.table;
    const host = document.getElementById("dmiChart");
    host.innerHTML = `<h4>Odometer scale by vehicle and year</h4>
      <p class="sub">Ring: the simulated truth. Dot: the estimate from GPS. Outside the band is flagged.</p>`;
    const W = 820, H = 230, m = { l: 52, r: 120, t: 14, b: 30 };
    const vals = rows.flatMap((r) => [r.true * 100, r.gps * 100]);
    const lo = Math.min(-0.4, Math.floor(Math.min(...vals) * 10) / 10 - 0.1);
    const hi = Math.max(0.4, Math.ceil(Math.max(...vals) * 10) / 10 + 0.1);
    const X = (year, v) => m.l + ((year - YEARS[0] + 0.5 + (v === "SV1" ? -0.12 : 0.12)) / YEARS.length) * (W - m.l - m.r);
    const Y = (p) => m.t + (1 - (p - lo) / (hi - lo)) * (H - m.t - m.b);
    const { svg, add } = svgIn(host, W, H, `Odometer scale error per vehicle and year. ` +
      rows.map((r) => `${r.vehicle} ${r.year}: true ${(r.true * 100).toFixed(2)}%, estimated ${(r.gps * 100).toFixed(2)}%`).join("; "));
    add("rect", { x: m.l, y: Y(0.3), width: W - m.l - m.r, height: Y(-0.3) - Y(0.3), fill: token("--grid"), opacity: 0.55 });
    for (const t of niceTicks(lo, hi, 0.2)) {
      add("line", { class: "grid", x1: m.l, x2: W - m.r, y1: Y(t), y2: Y(t) });
      add("text", { x: m.l - 6, y: Y(t) + 4, "text-anchor": "end" }, `${t > 0 ? "+" : t < 0 ? "−" : ""}${Math.abs(t).toFixed(1)}%`);
    }
    add("text", { x: W - m.r + 8, y: Y(0) + 4, class: "end-label" }, "within ±0.3%");
    YEARS.forEach((y) => add("text", { x: X(y, ""), y: H - m.b + 18, "text-anchor": "middle" }, y));
    const tip = document.createElement("div");
    tip.className = "tip";
    tip.hidden = true;
    host.appendChild(tip);
    const flagged = new Set(S.dmi.vehicle_years_flagged);
    for (const r of rows) {
      const cx = X(r.year, r.vehicle), key = `${r.vehicle} ${r.year}`;
      add("circle", { cx, cy: Y(r.true * 100), r: 7, fill: "none", stroke: token("--ink-2"), "stroke-width": 1.5 });
      add("circle", { cx, cy: Y(r.gps * 100), r: 4, fill: VEHICLE[r.vehicle], stroke: token("--surface"), "stroke-width": 1.5 });
      if (flagged.has(key)) add("text", { x: cx + 12, y: Y(r.gps * 100) + 4, class: "end-label" }, `${key} flagged`);
      const hit = add("circle", { cx, cy: Y(r.gps * 100), r: 12, fill: "transparent" });
      hit.addEventListener("pointerenter", () => {
        const box = svg.getBoundingClientRect();
        tip.hidden = false;
        tip.innerHTML = `<b>${key}</b><br>Truth ${signed(r.true * 100, 3)}%<br>GPS estimate ${signed(r.gps * 100, 3)}%` +
          (flagged.has(key) ? "<br>Flagged" : "");
        tip.style.left = `${(cx / W) * box.width + 14}px`;
        tip.style.top = `${(Y(r.gps * 100) / H) * box.height - 10}px`;
      });
      hit.addEventListener("pointerleave", () => { tip.hidden = true; });
    }
    for (const v of ["SV1", "SV2"]) {
      const first = rows.find((r) => r.vehicle === v);
      add("text", { x: X(first.year, v), y: Math.min(Y(first.true * 100), Y(first.gps * 100)) - 12, "text-anchor": "middle", class: "end-label" }, v);
    }
    document.getElementById("dmiTable").innerHTML = `<thead><tr><th>Vehicle</th><th>Year</th><th>Truth</th><th>GPS estimate</th><th>Flagged</th></tr></thead><tbody>` +
      rows.map((r) => `<tr><td>${r.vehicle}</td><td>${r.year}</td><td>${signed(r.true * 100, 3)}%</td><td>${signed(r.gps * 100, 3)}%</td>` +
        `<td>${flagged.has(`${r.vehicle} ${r.year}`) ? "yes" : ""}</td></tr>`).join("") + "</tbody>";
    tableToggle("dmiTableToggle", "dmiTable");
  }

  // ======================= repeatability =======================
  function initRepeat() {
    const bv = site.repeat.summary.between_vehicles, wv = site.repeat.summary.within_vehicle;
    document.getElementById("repeatSummary").innerHTML =
      `Over all ${bv.gps.baskets} site-years, comparing both vehicles' runs, pairing by odometer gets ` +
      `<b>${pct(bv.chainage.paired_right)}</b> of pairs right, by GPS <b>${pct(bv.gps.paired_right)}</b> and by LRS ` +
      `measure <b>${pct(bv.lrs.paired_right)}</b> (${pct(bv.oracle.paired_right)} by true position). Within one ` +
      `vehicle the odometers mostly agree, and odometer pairing gets ${pct(wv.chainage.paired_right)}. A pair is ` +
      `right when the run's segment lies within half a segment (13.2 ft) of the station's.`;

    const siteSel = document.getElementById("siteSel"), yearSel = document.getElementById("yearSel");
    siteSel.innerHTML = site.sites.map((s) => `<option value="${s.site}">${s.site}: ${routeLabel(s.route)}, ${s.county} County</option>`).join("");
    yearSel.innerHTML = YEARS.map((y) => `<option>${y}</option>`).join("");
    // open on the site-year where odometer pairing costs ICC the most
    const both = site.repeat.baskets.filter((b) => b.basket === "both");
    const gap = (b) => both.find((o) => o.site === b.site && o.year === b.year && o.method === "oracle").ICC2k - b.ICC2k;
    const worst = both.filter((b) => b.method === "chainage").sort((a, b) => gap(b) - gap(a))[0];
    siteSel.value = worst.site;
    yearSel.value = worst.year;

    const seg = document.getElementById("methodSeg");
    seg.innerHTML = METHODS.map(([k, label]) => `<button type="button" data-k="${k}" aria-pressed="${k === "chainage"}">${label}</button>`).join("");
    let method = "chainage", current = null;
    seg.addEventListener("click", (e) => {
      const b = e.target.closest("button");
      if (!b) return;
      method = b.dataset.k;
      seg.querySelectorAll("button").forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
      if (current) renderZoom(current, method);
    });
    const load = async () => {
      current = await getJSON(`data/repeat/${siteSel.value}_${yearSel.value}.json`);
      renderSite(current);
      renderZoom(current, method);
    };
    siteSel.addEventListener("change", load);
    yearSel.addEventListener("change", load);
    tableToggle("zoomTableToggle", "zoomTable");
    load();
  }

  function renderSite(rep) {
    const s = site.sites.find((x) => x.site === rep.site);
    const odo = ["SV1", "SV2"].map((v) => {
      const p = rep.runs.filter((r) => r.vehicle === v).map((r) => r.dmi_pct);
      return `${v} ${signed(Math.min(...p), 2)}% to ${signed(Math.max(...p), 2)}%`;
    });
    document.getElementById("siteNote").textContent =
      `${rep.site} is ${rep.length_mi.toFixed(2)} mi of ${routeLabel(s.route)}, milepoints ${s.m_start.toFixed(2)} to ` +
      `${s.m_end.toFixed(2)}. Odometer scale this day (simulated): ${odo.join(", ")}.`;

    const host = document.getElementById("spreadChart");
    host.innerHTML = "<h4>How far off each pairing is, along the site</h4>" +
      "<p class=\"sub\">Distance on the road between each station and the worst-paired run's segment (from the truth), " +
      "median per 264-ft block</p>";
    const series = METHODS.map(([k]) => {
      const pts = rep.methods[k].block_off;
      return { name: METHOD[k].label, label: METHOD[k].label, color: METHOD[k].color, dash: METHOD[k].dash,
        width: k === "oracle" ? 1.5 : 2, xs: pts.map((p) => p[0]), ys: pts.map((p) => p[1]) };
    });
    const top = Math.max(26.4, ...series.flatMap((x) => x.ys.filter((v) => v != null)));
    const yMax = Math.ceil(top / 20) * 20;
    lineChart(host, {
      H: 250, aria: `Distance between stations and the worst-paired run's segment along ${rep.site} in ${rep.year}, for each pairing method.`,
      x: { min: 0, max: rep.length_mi, ticks: niceTicks(0, rep.length_mi, 0.5), fmt: (t) => t, label: "miles along the site" },
      y: { min: 0, max: yMax, ticks: niceTicks(0, yMax, yMax > 100 ? 40 : 20), label: "feet" },
      refs: [{ y: 13.2, label: "half a segment" }],
      series,
      tip: (i, x) => `<b>${x.toFixed(2)}–${(x + 0.05).toFixed(2)} mi</b><br>` + series.map((sr) => {
        const v = sr.ys[nearest(sr.xs, x)];
        return `<i style="background:${sr.color}"></i>${sr.name} ${v == null ? "–" : `${v.toFixed(0)} ft`}`;
      }).join("<br>"),
    });

    const rows = site.repeat.baskets.filter((b) => b.site === rep.site && b.year === rep.year);
    const get = (basket, k) => rows.find((b) => b.basket === basket && b.method === k);
    document.getElementById("repeatTable").innerHTML =
      `<thead><tr><th>${rep.site}, ${rep.year}, paired by</th><th>Pairs right, both vehicles</th><th>Pairs right, one vehicle</th>` +
      `<th>BRSD</th><th>ICC(2,k)</th><th>ICC(3,k)</th></tr></thead><tbody>` +
      METHODS.map(([k]) => {
        const b = get("both", k), one = (get("SV1", k).paired_right + get("SV2", k).paired_right) / 2;
        return `<tr><td>${METHOD[k].label}</td><td>${pct(b.paired_right)}</td><td>${pct(one)}</td>` +
          `<td>${b.BRSD.toFixed(3)}</td><td>${b.ICC2k.toFixed(3)}</td><td>${b.ICC3k.toFixed(3)}</td></tr>`;
      }).join("") + "</tbody>";
  }

  function renderZoom(rep, method) {
    const z = rep.methods[method];
    const host = document.getElementById("zoomChart");
    host.innerHTML = `<h4>The six runs over the site's last 0.2 mi, paired by ${METHOD[method].inline}</h4>
      <div class="legend-inline" aria-hidden="true">${["SV1", "SV2"].map((v) =>
        `<span><i style="background:${VEHICLE[v]}"></i>${v}'s three runs</span>`).join("")}</div>`;
    if (!z.zoom_x.length) { host.insertAdjacentHTML("beforeend", "<p class=\"empty\">No paired positions here.</p>"); return; }
    const series = rep.runs.map((r, k) => ({ name: r.run, color: VEHICLE[r.vehicle], xs: z.zoom_x, ys: z.zoom[k], width: 1.5 }));
    const all = series.flatMap((s) => s.ys.filter((v) => v != null));
    const lo = Math.floor(Math.min(...all) / 5) * 5, hi = Math.ceil(Math.max(...all) / 5) * 5;
    const x0 = z.zoom_x[0], x1 = z.zoom_x[z.zoom_x.length - 1];
    lineChart(host, {
      H: 260, labels: false,
      aria: `The six runs' readings over the last 0.2 mi of ${rep.site} in ${rep.year}, paired by ${METHOD[method].label}.`,
      x: { min: x0, max: x1, ticks: niceTicks(x0, x1, 0.05), fmt: (t) => t.toFixed(2), label: "miles along the site" },
      y: { min: lo, max: hi, ticks: niceTicks(lo, hi, 5) },
      series,
      tip: (i, x) => `<b>${x.toFixed(3)} mi</b><br>` + series.map((s) =>
        `<i style="background:${s.color}"></i>${s.name} ${s.ys[i] == null ? "–" : s.ys[i].toFixed(1)}`).join("<br>"),
    });
    document.getElementById("zoomTable").innerHTML =
      `<thead><tr><th>Miles</th>${rep.runs.map((r) => `<th>${r.run}</th>`).join("")}</tr></thead><tbody>` +
      z.zoom_x.map((x, i) => `<tr><td>${x.toFixed(3)}</td>${z.zoom.map((col) => `<td>${col[i] == null ? "–" : col[i].toFixed(1)}</td>`).join("")}</tr>`).join("") +
      "</tbody>";
  }

  // ======================= change map and route profiles =======================
  function initMap() {
    const state = { year: site.featured.year, route: null, map: null, ready: false };
    const yearSeg = document.getElementById("yearSeg");
    yearSeg.innerHTML = YEARS.slice(1).map((y) => `<button type="button" data-y="${y}" aria-pressed="${y === state.year}">${y}</button>`).join("");
    yearSeg.addEventListener("click", (e) => {
      const b = e.target.closest("button");
      if (b) setYear(Number(b.dataset.y));
    });

    const colors = [0, 1, 2, 3, 4].map((i) => token(`--r${i}`));
    const legend = document.getElementById("legend");
    const labels = ["Up 8 or more: resurfaced", "Up 3 to 8", "Within 3", "Down 3 to 8", "Down 8 or more"];
    legend.innerHTML = `<div class="title">Change since the previous survey</div>` +
      labels.map((t, i) => `<div class="row"><span class="swatch" style="background:${colors[i]}"></span>${t}</div>`).join("") +
      `<div class="row"><span class="swatch thin" style="background:${token("--axis")}"></span>Not compared this year</div>`;

    const key = (dir) => `${dir}${state.year % 100}`;
    const colorBy = (k) => ["step", ["get", k], colors[4], -8, colors[3], -3, colors[2], 3, colors[1], 8, colors[0]];
    const offset = (sign) => ["interpolate", ["linear"], ["zoom"], 8, sign * 1, 11, sign * 1.8, 14, sign * 3.2];
    const width = ["interpolate", ["linear"], ["zoom"], 8, 1.3, 11, 2.2, 14, 3.6];

    const map = new maplibregl.Map({
      container: "map",
      style: `https://basemaps.cartocdn.com/gl/${dark ? "dark-matter" : "positron"}-gl-style/style.json`,
      bounds: site.network.bounds,
      fitBoundsOptions: { padding: 24 },
      attributionControl: { compact: true },
      cooperativeGestures: true,   // the page scrolls past the map; ctrl/two-finger to zoom it
    });
    state.map = map;
    map.addControl(new maplibregl.NavigationControl({ showCompass: false }), "top-left");

    map.on("load", () => {
      map.addSource("chg", {
        type: "geojson", data: new URL("data/changes.geojson", location.href).href,
        attribution: `<a href="${esc(site.network.service.replace(/\/query$/, ""))}">KYTC All Roads</a>; simulated readings`,
      });
      const firstSymbol = map.getStyle().layers.find((l) => l.type === "symbol")?.id;
      map.addLayer({
        id: "sel", type: "line", source: "chg", filter: ["==", ["get", "r"], ""],
        layout: { "line-cap": "round", "line-join": "round" },
        paint: { "line-color": token("--ink"), "line-opacity": 0.22, "line-width": ["interpolate", ["linear"], ["zoom"], 8, 7, 14, 14] },
      }, firstSymbol);
      for (const [dir, sign] of [["i", 1], ["n", -1]]) {
        map.addLayer({
          id: `${dir}-none`, type: "line", source: "chg", filter: ["!", ["has", key(dir)]],
          layout: { "line-cap": "round", "line-join": "round" },
          paint: { "line-color": token("--axis"), "line-width": ["interpolate", ["linear"], ["zoom"], 8, 0.6, 14, 1.4], "line-offset": offset(sign) },
        }, firstSymbol);
        map.addLayer({
          id: `${dir}-data`, type: "line", source: "chg", filter: ["has", key(dir)],
          layout: { "line-cap": "round", "line-join": "round" },
          paint: { "line-color": colorBy(key(dir)), "line-width": width, "line-offset": offset(sign) },
        }, firstSymbol);
      }
      state.ready = true;
      highlight(state.route);
    });

    const layers = ["i-data", "n-data", "i-none", "n-none"];
    const pickBox = (pt, pad = 6) => [[pt.x - pad, pt.y - pad], [pt.x + pad, pt.y + pad]];
    const popup = new maplibregl.Popup({ maxWidth: "300px" });
    map.on("mousemove", (e) => {
      map.getCanvas().style.cursor = state.ready && map.queryRenderedFeatures(pickBox(e.point), { layers }).length ? "pointer" : "";
    });
    map.on("click", (e) => {
      if (!state.ready) return;
      const feats = map.queryRenderedFeatures(pickBox(e.point), { layers });
      if (!feats.length) return;
      const p = feats[0].properties;
      const show = (dir, label) => `<div><span class="k">${label}</span> ${p[key(dir)] == null ? "not compared" : signed(p[key(dir)])}</div>`;
      popup.setLngLat(e.lngLat).setHTML(`<div class="popup"><div class="big">${esc(routeLabel(p.r))}, mp ${Number(p.m).toFixed(1)}–${(Number(p.m) + 0.1).toFixed(1)}</div>` +
        `<div class="k">Change in ${state.year} since the previous survey</div>${show("i", "With the milepoints:")}${show("n", "Against them:")}` +
        `<div class="k" style="margin-top:4px">Profile below</div></div>`).addTo(map);
      showRoute(p.r);
    });

    function highlight(route) {
      if (state.ready) map.setFilter("sel", ["==", ["get", "r"], route || ""]);
    }

    function setYear(y) {
      state.year = y;
      yearSeg.querySelectorAll("button").forEach((b) => b.setAttribute("aria-pressed", String(Number(b.dataset.y) === y)));
      if (state.ready) {
        for (const dir of ["i", "n"]) {
          map.setFilter(`${dir}-none`, ["!", ["has", key(dir)]]);
          map.setFilter(`${dir}-data`, ["has", key(dir)]);
          map.setPaintProperty(`${dir}-data`, "line-color", colorBy(key(dir)));
        }
      }
      popup.remove();
      if (state.route) showRoute(state.route);
    }

    const routes = new Map();
    async function showRoute(route) {
      state.route = route;
      highlight(route);
      if (!routes.has(route)) routes.set(route, getJSON(`data/routes/${route}.json`).catch((err) => { routes.delete(route); throw err; }));
      const r = await routes.get(route);
      if (state.route !== route) return;
      renderProfile(r, state.year);
    }

    showRoute(site.featured.route);
  }

  function renderProfile(r, year) {
    const name = r.name && !r.name.replace(/\D/g, "").includes(String(Number(r.route.split("-")[2].replace(/\D/g, ""))))
      ? ` (${r.name.toLowerCase().replace(/\b\w/g, (c) => c.toUpperCase())})` : "";
    document.getElementById("profileTitle").textContent = `${routeLabel(r.route)}${name}, ${r.county} County`;
    document.getElementById("profileSub").textContent =
      `The ${year} survey against each direction's previous one, milepoints ${r.beg.toFixed(2)} to ${r.end.toFixed(2)}`;
    const cur = token("--accent"), prev = token("--context");
    document.getElementById("profileLegend").innerHTML =
      `<span><i style="background:${cur}"></i>${year}</span><span><i style="background:${prev}"></i>Previous survey</span>` +
      `<span><i class="band" style="background:${cur};opacity:.25"></i>Flagged as resurfaced</span>` +
      `<span><i class="bar" style="background:${token("--ink-2")}"></i>Resurfaced in the simulation (truth)</span>`;
    const host = document.getElementById("profileCharts");
    host.innerHTML = "";
    for (const [slot, label] of [["inv", "With the milepoints"], ["non", "Against the milepoints"]]) {
      const have = Object.keys(r.profile[slot]).map(Number).sort();
      const before = have.filter((y) => y < year);
      const p = before.length ? before[before.length - 1] : null;
      const card = cardIn(host, label, r.profile[slot][year] ? (p ? `${year} against ${p}` : `${year}, the first survey in this direction`) : "");
      if (!r.profile[slot][year]) {
        card.insertAdjacentHTML("beforeend", `<p class="empty">Not surveyed in this direction in ${year}.</p>`);
        continue;
      }
      // average down to at most ~700 points so long routes stay legible
      const n = r.profile[slot][year].length, k = Math.max(1, Math.ceil(n / 700));
      const xs = [], down = (vals) => {
        const out = [];
        for (let i = 0; i < n; i += k) {
          const chunk = vals.slice(i, i + k).filter((v) => v != null);
          out.push(chunk.length ? chunk.reduce((a, b) => a + b, 0) / chunk.length : null);
        }
        return out;
      };
      for (let i = 0; i < n; i += k) xs.push(r.x0 + (i + k / 2) * r.dx);
      const series = [];
      if (p) series.push({ name: String(p), color: prev, xs, ys: down(r.profile[slot][p]), width: 1.5 });
      series.push({ name: String(year), color: cur, xs, ys: down(r.profile[slot][year]), width: 1.5 });
      const all = series.flatMap((s) => s.ys.filter((v) => v != null));
      const lo = Math.floor(Math.min(...all) / 10) * 10, hi = Math.ceil(Math.max(...all) / 10) * 10;
      const xMax = r.x0 + n * r.dx;
      lineChart(card, {
        H: 230, labels: false,
        aria: `${label}: readings along ${routeLabel(r.route)} in ${year}${p ? ` and ${p}` : ""}.`,
        x: { min: r.x0, max: xMax, ticks: niceTicks(r.x0, xMax, mileStep(xMax - r.x0)), label: "milepoint" },
        y: { min: lo, max: hi, ticks: niceTicks(lo, hi, 10) },
        series,
        spans: (r.flagged[slot][year] || []).map(([a, b]) => ({ x0: a, x1: b, fill: cur, opacity: 0.16 })),
        bars: r.treated.filter(([, , y]) => y <= year && (p == null || y > p)).map(([a, b]) => ({ x0: a, x1: b, fill: token("--ink-2") })),
        tip: (i, x) => {
          const vals = series.map((s) => s.ys[i]);
          const delta = vals.length === 2 && vals[0] != null && vals[1] != null ? vals[1] - vals[0] : null;
          return `<b>mp ${x.toFixed(2)}</b><br>` + series.map((s, j) =>
            `<i style="background:${s.color}"></i>${s.name} ${vals[j] == null ? "–" : vals[j].toFixed(1)}`).join("<br>") +
            (delta == null ? "" : `<br>Change ${signed(delta)}`);
        },
      });
    }
  }
})();
