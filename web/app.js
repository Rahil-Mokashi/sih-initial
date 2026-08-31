// Static demo viewer. No live inference, no model server, no external
// network calls -- reads only local JSON/PNG files under public/demo/.
// Every step is defensive: a missing or malformed file shows an inline
// error message instead of breaking the page.

const DEMO_DIR = "public/demo";

function showError(container, message) {
  const box = document.createElement("div");
  box.className = "error-box";
  box.textContent = message;
  container.appendChild(box);
}

function fmtPct(value) {
  if (typeof value !== "number" || Number.isNaN(value)) return "n/a";
  return value.toFixed(1) + "%";
}

function fmtScore(value) {
  if (typeof value !== "number" || Number.isNaN(value)) return "n/a";
  return value.toFixed(3);
}

async function fetchJson(path) {
  const res = await fetch(path, { cache: "no-store" });
  if (!res.ok) {
    throw new Error(`HTTP ${res.status} for ${path}`);
  }
  return res.json();
}

function renderSummary(summary) {
  const panel = document.getElementById("summary");
  panel.innerHTML = "";

  const heading = document.createElement("h2");
  heading.textContent = `Baseline: ${summary.checkpoint || "unknown checkpoint"} @ threshold ${summary.threshold ?? "?"}`;
  panel.appendChild(heading);

  const stats = summary.baseline_oil_tiles_only || {};
  const statsRow = document.createElement("div");
  statsRow.className = "summary-stats";

  const entries = [
    ["Oil IoU", fmtScore(stats.iou)],
    ["Dice", fmtScore(stats.dice)],
    ["Precision", fmtScore(stats.precision)],
    ["Recall", fmtScore(stats.recall)],
  ];
  for (const [label, value] of entries) {
    const stat = document.createElement("div");
    stat.className = "stat";
    const v = document.createElement("div");
    v.className = "value";
    v.textContent = value;
    const l = document.createElement("div");
    l.className = "label";
    l.textContent = label;
    stat.appendChild(v);
    stat.appendChild(l);
    statsRow.appendChild(stat);
  }
  panel.appendChild(statsRow);

  if (stats.note) {
    const note = document.createElement("p");
    note.style.color = "var(--muted)";
    note.style.fontSize = "0.78rem";
    note.style.marginTop = "10px";
    note.textContent = stats.note;
    panel.appendChild(note);
  }
}

function renderCard(entry) {
  const card = document.createElement("div");
  card.className = "card";

  const title = document.createElement("div");
  title.className = "card-title";
  const nameSpan = document.createElement("span");
  nameSpan.textContent = entry.image_id || "unknown";
  const badge = document.createElement("span");
  badge.className = "badge " + (entry.label || "");
  badge.textContent = entry.label || "?";
  title.appendChild(nameSpan);
  title.appendChild(badge);
  card.appendChild(title);

  const row = document.createElement("div");
  row.className = "image-row";
  const views = [
    ["raw", "Input (SAR)"],
    ["gt", "Ground truth"],
    ["pred", "Model prediction"],
  ];
  for (const [suffix, caption] of views) {
    const figure = document.createElement("figure");
    const img = document.createElement("img");
    img.src = `${DEMO_DIR}/${entry.image_id}_${suffix}.png`;
    img.alt = `${entry.image_id} ${suffix}`;
    img.onerror = () => {
      img.replaceWith(document.createTextNode("(image failed to load)"));
    };
    const cap = document.createElement("figcaption");
    cap.textContent = caption;
    figure.appendChild(img);
    figure.appendChild(cap);
    row.appendChild(figure);
  }
  card.appendChild(row);

  const metricsGrid = document.createElement("div");
  metricsGrid.className = "metrics-grid";
  const metricEntries = [
    ["IoU", fmtScore(entry.iou)],
    ["Precision", fmtScore(entry.precision)],
    ["Recall", fmtScore(entry.recall)],
    ["Predicted oil", fmtPct(entry.predicted_oil_pct)],
    ["True oil", fmtPct(entry.true_oil_pct)],
    ["Dice", fmtScore(entry.dice)],
  ];
  for (const [label, value] of metricEntries) {
    const wrap = document.createElement("div");
    const l = document.createElement("div");
    l.className = "m-label";
    l.textContent = label;
    const v = document.createElement("div");
    v.className = "m-value";
    v.textContent = value;
    wrap.appendChild(l);
    wrap.appendChild(v);
    metricsGrid.appendChild(wrap);
  }
  card.appendChild(metricsGrid);

  return card;
}

async function main() {
  const summaryPanel = document.getElementById("summary");
  const gallery = document.getElementById("gallery");

  try {
    const summary = await fetchJson(`${DEMO_DIR}/summary.json`);
    renderSummary(summary);

    gallery.innerHTML = "";
    if (!Array.isArray(summary.images) || summary.images.length === 0) {
      showError(gallery, "summary.json loaded but contains no images.");
      return;
    }
    for (const entry of summary.images) {
      try {
        gallery.appendChild(renderCard(entry));
      } catch (err) {
        showError(gallery, `Failed to render ${entry && entry.image_id ? entry.image_id : "an image"}: ${err.message}`);
      }
    }
  } catch (err) {
    summaryPanel.innerHTML = "";
    showError(summaryPanel, `Could not load demo data: ${err.message}. Make sure you're serving this page over HTTP (not opening index.html directly as a file://) and that web/public/demo/summary.json exists.`);
    gallery.innerHTML = "";
  }
}

main();
