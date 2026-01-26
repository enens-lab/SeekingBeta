// Analyze page functionality

let userFeatures = null;
let availableStocks = [];
let stockCategories = {};
let selectedStocks = [];
let availableModels = [];
let availableTasks = [];
let resultsChart = null;

async function init() {
    if (!Auth.isLoggedIn()) {
        showLoginRequired();
        return;
    }

    await loadUserFeatures();
    await loadUniverse();
    await loadModels();
    updateSelectedDisplay();
}

function showLoginRequired() {
    document.getElementById('main-content').classList.add('hidden');
    document.getElementById('login-required').classList.remove('hidden');
    document.getElementById('login-required-btn').addEventListener('click', () => {
        document.getElementById('login-modal').classList.remove('hidden');
    });
}

async function loadUserFeatures() {
    try {
        const res = await Auth.fetchWithAuth('/api/user/features');
        if (!res.ok) throw new Error('Failed to load features');
        userFeatures = await res.json();
        updateUserInfo();
        updateRateLimitDisplay();
    } catch (err) {
        console.error('Failed to load user features:', err);
        userFeatures = {
            tier: 'free',
            features: {},
            limits: {
                daily_requests: 10,
                requests_used: 0,
                requests_remaining: 10,
                max_stocks_per_request: 5,
                max_historical_days: 30
            }
        };
    }
}

async function loadUniverse() {
    try {
        const res = await Auth.fetchWithAuth('/api/universe');
        if (!res.ok) throw new Error('Failed to load universe');
        const data = await res.json();
        availableStocks = data.stocks;
        stockCategories = data.categories;
        renderStockCategories();
    } catch (err) {
        console.error('Failed to load universe:', err);
    }
}

async function loadModels() {
    try {
        const res = await Auth.fetchWithAuth('/api/analyze/models');
        if (!res.ok) throw new Error('Failed to load models');
        const data = await res.json();
        availableModels = data.models;
        availableTasks = data.tasks;
        renderModelOptions();
    } catch (err) {
        console.error('Failed to load models:', err);
    }
}

function updateUserInfo() {
    document.getElementById('user-tier').textContent = userFeatures.tier.toUpperCase();
}

function updateRateLimitDisplay() {
    const remaining = document.getElementById('requests-remaining');
    const banner = document.getElementById('rate-limit-banner');
    const bannerText = document.getElementById('rate-limit-text');

    if (userFeatures.limits.daily_requests === null) {
        remaining.textContent = 'Unlimited';
        banner.classList.add('hidden');
    } else {
        const used = userFeatures.limits.requests_used || 0;
        const total = userFeatures.limits.daily_requests;
        const left = total - used;
        remaining.textContent = `${left} remaining`;
        bannerText.textContent = `${used}/${total} requests used today`;

        if (used >= total * 0.8) {
            banner.classList.remove('hidden');
        }
    }
}

function renderStockCategories() {
    const container = document.getElementById('stock-categories');
    container.innerHTML = '';

    for (const [category, stocks] of Object.entries(stockCategories)) {
        const div = document.createElement('div');
        div.className = 'category';
        div.innerHTML = `
            <div class="category-header" data-category="${category}">
                <span>${category}</span>
                <span class="category-count">${stocks.length}</span>
            </div>
            <div class="category-stocks collapsed" id="cat-${category.replace(/\s+/g, '-')}">
                ${stocks.map(s => `
                    <label class="stock-checkbox">
                        <input type="checkbox" value="${s}" ${selectedStocks.includes(s) ? 'checked' : ''} />
                        ${s}
                    </label>
                `).join('')}
            </div>
        `;
        container.appendChild(div);
    }

    // Add toggle handlers
    container.querySelectorAll('.category-header').forEach(header => {
        header.addEventListener('click', () => {
            const category = header.dataset.category;
            const stocksDiv = document.getElementById(`cat-${category.replace(/\s+/g, '-')}`);
            stocksDiv.classList.toggle('collapsed');
        });
    });

    // Add checkbox handlers
    container.querySelectorAll('input[type="checkbox"]').forEach(cb => {
        cb.addEventListener('change', (e) => {
            const ticker = e.target.value;
            if (e.target.checked) {
                if (selectedStocks.length >= userFeatures.limits.max_stocks_per_request) {
                    e.target.checked = false;
                    alert(`Maximum ${userFeatures.limits.max_stocks_per_request} stocks for ${userFeatures.tier} tier`);
                    return;
                }
                selectedStocks.push(ticker);
            } else {
                selectedStocks = selectedStocks.filter(s => s !== ticker);
            }
            updateSelectedDisplay();
        });
    });
}

function renderModelOptions() {
    const modelSelect = document.getElementById('model-select');
    const taskSelect = document.getElementById('task-select');

    // All possible models
    const allModels = [
        { value: 'gradient_boosting', label: 'Gradient Boosting' },
        { value: 'linear_regression', label: 'Linear Regression' },
        { value: 'random_forest', label: 'Random Forest' },
        { value: 'lstm', label: 'LSTM' }
    ];

    modelSelect.innerHTML = allModels.map(m => {
        const disabled = !availableModels.includes(m.value);
        return `<option value="${m.value}" ${disabled ? 'disabled' : ''}>${m.label}${disabled ? ' (PRO)' : ''}</option>`;
    }).join('');

    // Tasks
    taskSelect.querySelectorAll('option').forEach(opt => {
        opt.disabled = !availableTasks.includes(opt.value);
    });

    // Period options based on tier
    const periodSelect = document.getElementById('period-select');
    const maxDays = userFeatures.limits.max_historical_days;
    periodSelect.querySelectorAll('option').forEach(opt => {
        const periodDays = { '1M': 30, '3M': 90, '6M': 180, '1Y': 365 };
        opt.disabled = periodDays[opt.value] > maxDays;
        if (opt.disabled && !opt.textContent.includes('(PRO)')) {
            opt.textContent += ' (PRO)';
        }
    });
}

function updateSelectedDisplay() {
    const list = document.getElementById('selected-list');
    const limit = document.getElementById('stock-limit');

    if (selectedStocks.length === 0) {
        list.innerHTML = '<span class="placeholder">No stocks selected</span>';
    } else {
        list.innerHTML = selectedStocks.map(s => `
            <span class="stock-chip">
                ${s}
                <span class="remove" data-ticker="${s}">&times;</span>
            </span>
        `).join('');

        // Add remove handlers
        list.querySelectorAll('.remove').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const ticker = e.target.dataset.ticker;
                selectedStocks = selectedStocks.filter(s => s !== ticker);
                // Uncheck the checkbox
                const cb = document.querySelector(`input[value="${ticker}"]`);
                if (cb) cb.checked = false;
                updateSelectedDisplay();
            });
        });
    }

    limit.textContent = `${selectedStocks.length}/${userFeatures.limits.max_stocks_per_request} selected`;
}

async function runAnalysis() {
    if (selectedStocks.length === 0) {
        alert('Please select at least one stock');
        return;
    }

    const model = document.getElementById('model-select').value;
    const task = document.getElementById('task-select').value;
    const period = document.getElementById('period-select').value;
    const horizon = document.getElementById('horizon-select').value;

    const btn = document.getElementById('run-analysis');
    btn.disabled = true;
    btn.textContent = 'Analyzing...';

    try {
        const res = await Auth.fetchWithAuth('/api/analyze', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                tickers: selectedStocks,
                model,
                task,
                period,
                horizon
            })
        });

        if (res.status === 429) {
            alert('Rate limit exceeded. Please upgrade your plan or try again tomorrow.');
            return;
        }

        if (!res.ok) {
            const error = await res.json();
            throw new Error(error.detail || 'Analysis failed');
        }

        const data = await res.json();
        renderResults(data);

        // Refresh user features to update rate limit
        await loadUserFeatures();
    } catch (err) {
        alert(`Analysis failed: ${err.message}`);
    } finally {
        btn.disabled = false;
        btn.textContent = 'Run Analysis';
    }
}

function renderResults(data) {
    document.getElementById('results-empty').classList.add('hidden');
    document.getElementById('results-content').classList.remove('hidden');

    const tbody = document.querySelector('#results-table tbody');
    tbody.innerHTML = data.results.map(r => `
        <tr>
            <td><strong>${r.ticker}</strong></td>
            <td>$${r.last_close ? r.last_close.toFixed(2) : '-'}</td>
            <td>${r.prob_up !== null ? (r.prob_up * 100).toFixed(1) + '%' : '-'}</td>
            <td class="signal-${r.signal || 'hold'}">${r.signal ? r.signal.toUpperCase() : '-'}</td>
            <td>${r.predicted_return !== null ? (r.predicted_return * 100).toFixed(2) + '%' : '-'}</td>
        </tr>
    `).join('');

    renderResultsChart(data.results);

    // Show export button for users with that feature
    if (userFeatures.features.export_csv) {
        document.getElementById('export-csv').classList.remove('hidden');
    }
}

function renderResultsChart(results) {
    const ctx = document.getElementById('resultsChart');
    const labels = results.map(r => r.ticker);
    const data = results.map(r => r.prob_up !== null ? r.prob_up * 100 : 0);
    const colors = results.map(r => {
        if (r.signal === 'buy') return 'rgba(22, 163, 74, 0.8)';
        if (r.signal === 'sell') return 'rgba(220, 38, 38, 0.8)';
        return 'rgba(107, 114, 128, 0.8)';
    });

    if (resultsChart) resultsChart.destroy();

    resultsChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels,
            datasets: [{
                label: 'Probability Up (%)',
                data,
                backgroundColor: colors,
                borderRadius: 4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,
                    title: { display: true, text: 'Probability (%)' }
                }
            }
        }
    });
}

function exportCSV() {
    const table = document.getElementById('results-table');
    const rows = Array.from(table.querySelectorAll('tr'));
    const csv = rows.map(row => {
        const cells = Array.from(row.querySelectorAll('th, td'));
        return cells.map(c => `"${c.textContent.trim()}"`).join(',');
    }).join('\n');

    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `pythia_analysis_${new Date().toISOString().split('T')[0]}.csv`;
    a.click();
    URL.revokeObjectURL(url);
}

// Stock search filter
document.getElementById('stock-search')?.addEventListener('input', (e) => {
    const query = e.target.value.toLowerCase();
    document.querySelectorAll('.stock-checkbox').forEach(label => {
        const ticker = label.textContent.trim().toLowerCase();
        label.style.display = ticker.includes(query) ? '' : 'none';
    });

    // Expand all categories when searching
    if (query) {
        document.querySelectorAll('.category-stocks').forEach(div => {
            div.classList.remove('collapsed');
        });
    }
});

// Event listeners
document.getElementById('run-analysis')?.addEventListener('click', runAnalysis);
document.getElementById('export-csv')?.addEventListener('click', exportCSV);

// Initialize on page load
window.addEventListener('DOMContentLoaded', init);
