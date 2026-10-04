"use strict";
const $ = (id) => document.getElementById(id);
const esc = (v) => String(v ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const fmt = (v, d = 2) => (v === null || v === undefined ? "—" : Number(v).toLocaleString(undefined, { maximumFractionDigits: d }));
const row = (k, v) => `<div class="row"><span>${esc(k)}</span><strong>${v}</strong></div>`;
const bar = (pct) => `<div class="bar"><span style="width:${Math.max(0, Math.min(100, pct)).toFixed(1)}%"></span></div>`;

async function setMode(mode) {
  const r = await fetch("/api/admin/mode", { method: "POST", headers: { "Content-Type": "application/json", "X-JFE": "1" }, body: JSON.stringify({ mode }) });
  if (!r.ok) alert((await r.json()).detail);
  refresh();
}

function spark(points) {
  const W = 900, H = 120, P = 22, n = points.length;
  const max = Math.max(1, ...points.map((p) => Math.max(p.drafts, p.runs)));
  const x = (i) => P + (i * (W - 2 * P)) / Math.max(1, n - 1);
  const y = (v) => H - P - (v * (H - 2 * P)) / max;
  const line = (k, col) => `<polyline fill="none" stroke="${col}" stroke-width="2" points="${points.map((p, i) => `${x(i)},${y(p[k])}`).join(" ")}"/>`;
  const labels = points.map((p, i) => (i % 5 === 0 || i === n - 1 ? `<text x="${x(i)}" y="${H - 6}" text-anchor="middle">${esc(p.t)}</text>` : "")).join("");
  return `<svg viewBox="0 0 ${W} ${H}" width="100%" role="img" aria-label="Queries and runs per minute"><line x1="${P}" x2="${W - P}" y1="${H - P}" y2="${H - P}" stroke="#ccc"/><text x="2" y="${y(max) + 4}">${max}</text>${line("drafts", "#0066cc")}${line("runs", "#151515")}${labels}</svg>`;
}

function latRow(label, s, unit = "s") {
  if (!s || !s.n) return row(label, "no data");
  return row(`${label} (n=${s.n})`, `p50 ${fmt(s.p50)}${unit} · p95 ${fmt(s.p95)}${unit} · max ${fmt(s.max)}${unit}`);
}

async function refresh() {
  let d;
  try {
    const r = await fetch("/api/admin/stats", { cache: "no-store" });
    if (!r.ok) throw new Error(r.status);
    d = await r.json();
  } catch (e) {
    $("status").innerHTML = `<span class="err">stats unavailable (${esc(e.message)})</span>`;
    return;
  }
  $("status").textContent = `mode ${d.mode} · updated ${d.now} · ${d.sessions.active} active sessions (${d.sessions.new_1h} new in last hour)`;
  $("modes").innerHTML = d.modes.map((m) => `<button class="${m === d.mode ? "on" : ""}" data-mode="${m}">${m.replace("_", " ")}</button>`).join("");
  $("modes").querySelectorAll("button").forEach((b) => (b.onclick = () => setMode(b.dataset.mode)));

  const ev = (kind) => d.events.filter((e) => e.kind === kind);
  const sum = (arr, k) => arr.reduce((a, e) => a + Number(e[k]), 0);
  const drafts = ev("draft"), explains = ev("explain");
  $("queries").innerHTML = `<div class="big">${fmt(sum(drafts, "n") + sum(explains, "n"), 0)}</div><div class="muted">total LLM queries · ${fmt(sum(drafts, "n_1h") + sum(explains, "n_1h"), 0)} in the last hour</div>` +
    drafts.map((e) => row(`draft: ${e.status}`, `${e.n} <span class="muted">(${e.n_1h} 1h)</span>`)).join("") +
    explains.map((e) => row(`explain: ${e.status}`, `${e.n} <span class="muted">(${e.n_1h} 1h)</span>`)).join("") +
    ev("rate_limited").map((e) => row(`rate limited: ${e.status}`, e.n)).join("");

  const total = sum(d.runs, "n");
  $("runs").innerHTML = `<div class="big">${fmt(total, 0)}</div><div class="muted">runs · ${d.queued_now} queued now (limit ${d.limits.max_waiting_runs})</div>` +
    d.runs.map((r) => row(r.status, `${r.n} <span class="muted">(${r.n_1h} 1h)</span>`)).join("");

  $("latency").innerHTML = latRow("LLM interpretation", d.latency.llm_draft) + latRow("Grounded explanation", d.latency.explain) +
    latRow("Queue wait", d.latency.queue_wait) + latRow("Compute (fresh)", d.latency.compute) + latRow("Submit → complete", d.latency.end_to_end);

  const w = d.worker;
  if (w && w.gpu) {
    const g = w.gpu, memPct = (100 * g.mem_used_mb) / g.mem_total_mb;
    $("gpu").innerHTML = `<div class="muted">${esc(g.name)}</div>` + row("Utilisation", `${fmt(g.util_pct, 0)}%`) + bar(g.util_pct) +
      row("VRAM", `${fmt(g.mem_used_mb / 1024, 1)} / ${fmt(g.mem_total_mb / 1024, 1)} GB`) + bar(memPct) +
      row("Temperature", `${fmt(g.temp_c, 0)} °C`) + row("Power", `${fmt(g.power_w, 0)} / ${fmt(g.power_limit_w, 0)} W`) +
      row("Worker backend", `${esc(w.backend)} (GPU for ≥${w.gpu_min_draws} draws)`) + row("Heartbeat", `${fmt(w.heartbeat_age_s, 0)} s ago`);
  } else {
    $("gpu").innerHTML = w ? `<span class="err">${esc(w.gpu_error || "no GPU telemetry")}</span>` : `<span class="err">worker heartbeat missing</span>`;
  }
  $("host").innerHTML = w ? row("CPU load (1 min / 5 min)", `${fmt(w.load_1m)} / ${fmt(w.load_5m)} on ${w.cpu_count} cores`) + bar((100 * w.load_1m) / w.cpu_count) +
    row("RAM used", `${fmt(w.ram_total_gb - w.ram_available_gb, 1)} / ${fmt(w.ram_total_gb, 0)} GB`) + bar((100 * (w.ram_total_gb - w.ram_available_gb)) / w.ram_total_gb) : "—";

  const l = d.llm;
  $("llm").innerHTML = row("Status", l.up ? "up" : `<span class="err">down</span>`) + row("Model", esc(l.model)) +
    row("Requests running / waiting", `${fmt(l.running ?? 0, 0)} / ${fmt(l.waiting ?? 0, 0)}`) + row("API concurrency limit", l.active_limit) +
    (l.kv_cache_usage !== undefined ? row("KV cache", `${fmt(100 * l.kv_cache_usage, 1)}%`) + bar(100 * l.kv_cache_usage) : "") +
    row("Tokens in / out (total)", `${fmt(l.prompt_tokens_total ?? 0, 0)} / ${fmt(l.generation_tokens_total ?? 0, 0)}`);

  $("spark").innerHTML = spark(d.per_minute);
  $("recent").innerHTML = `<table><tr><th>at</th><th>run</th><th>status</th><th>mode</th><th class="n">draws</th><th>backend</th><th>cache</th><th class="n">queue s</th><th class="n">compute s</th><th class="n">total s</th><th>operator</th></tr>` +
    d.recent_runs.map((r) => `<tr><td>${esc(r.at)}</td><td>…${esc(r.id)}</td><td><span class="pill ${esc(r.status)}">${esc(r.status)}</span></td><td>${esc(r.mode)}</td><td class="n">${fmt(r.draws, 0)}</td><td>${esc(r.backend)}</td><td>${esc(r.cache_status)}</td><td class="n">${fmt(r.queue_s)}</td><td class="n">${fmt(r.compute_s, 4)}</td><td class="n">${fmt(r.total_s)}</td><td>${r.is_operator ? "yes" : ""}</td></tr>`).join("") + `</table>`;
}

refresh();
setInterval(refresh, 3000);
