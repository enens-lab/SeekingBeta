async function getJSON(url) {
    const res = await fetch(url, { cache: "no-store" });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return res.json();
  }
  
  function fmt(n) {
    if (n === null || n === undefined) return "";
    return Number(n).toLocaleString(undefined, { maximumFractionDigits: 2 });
  }
  
  function setHealth(status) {
    const el = document.getElementById("health");
    el.classList.remove("health--ok", "health--bad", "health--unknown");
    if (status.ok) {
      el.classList.add("health--ok");
      el.textContent = "healthy";
    } else {
      el.classList.add("health--bad");
      el.textContent = status.message || "unhealthy";
    }
    document.getElementById("dataSource").textContent = status.data_source || "—";
    document.getElementById("uniSize").textContent = status.universe_size ?? "—";
  }
  
  function buildTable(items) {
    const tbody = document.querySelector("#metricsTable tbody");
    tbody.innerHTML = "";
    items.forEach(row => {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td>${row.ts}</td>
        <td>${fmt(row.equity)}</td>
        <td>${fmt(row.cash)}</td>
        <td>${fmt(row.pnl)}</td>
        <td>${fmt(row.win_rate)}</td>
        <td>${fmt(row.turnover)}</td>
        <td>${fmt(row.max_drawdown)}</td>
      `;
      tbody.appendChild(tr);
    });
  }
  
  let chart;
  function buildChart(items) {
    const ctx = document.getElementById("equityChart");
    const labels = items.map(x => x.ts).reverse();
    const data = items.map(x => x.equity).reverse();
    if (chart) chart.destroy();
    chart = new Chart(ctx, {
      type: "line",
      data: { labels, datasets: [{ label: "Equity", data, fill: false }] },
      options: {
        responsive: true,
        interaction: { mode: "index", intersect: false },
        scales: { x: { display: true }, y: { display: true } }
      }
    });
  }
  
  async function refreshAll() {
    try {
      const status = await getJSON("/healthz");
      setHealth(status);
    } catch (e) {
      setHealth({ ok: false, message: "healthz failed" });
      console.error(e);
    }
  
    try {
      const { items } = await getJSON("/paper/metrics?limit=200");
      buildTable(items || []);
      if (items && items.length) buildChart(items);
    } catch (e) {
      console.error("metrics fetch failed:", e);
    }
  }
  
  window.addEventListener("DOMContentLoaded", () => {
    document.getElementById("refreshBtn").addEventListener("click", refreshAll);
    refreshAll();
  });
  