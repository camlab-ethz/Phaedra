// Operator-learning results on the project page: variable tabs for the token video, dataset tabs for
// the comparison figure, and the per-timestep error chart (Chart.js). The chart data is embedded in the
// page (<script id="results-data">) by scripts/website/make_site_assets.py from docs/results.
(function () {
  'use strict';

  function setupGroup(id, onSelect) {
    var group = document.getElementById(id);
    if (!group) return;
    var items = group.querySelectorAll('[data-value]');
    items.forEach(function (el) {
      el.addEventListener('click', function (e) {
        e.preventDefault();
        var value = el.getAttribute('data-value');
        items.forEach(function (other) {
          var on = other === el;
          other.classList.toggle('is-active', on);
          if (other.tagName === 'BUTTON') {
            other.classList.toggle('is-link', on);
            other.setAttribute('aria-pressed', on ? 'true' : 'false');
          }
        });
        onSelect(value);
      });
    });
  }

  // ------------------------------------------------------------------ token video (one mp4 per variable)
  function setupVideo() {
    var video = document.getElementById('token-video');
    if (!video) return;
    setupGroup('video-var-tabs', function (v) {
      var source = video.querySelector('source');
      video.pause();
      video.setAttribute('poster', 'static/videos/phaedra_tokens_' + v + '_poster.jpg');
      source.setAttribute('src', 'static/videos/phaedra_tokens_' + v + '.mp4');
      video.load();
      var p = video.play();
      if (p && typeof p.catch === 'function') p.catch(function () { /* autoplay blocked: the poster stays */ });
    });
  }

  // ------------------------------------------------------------------ comparison figure (one per dataset)
  function setupCompare() {
    var img = document.getElementById('compare-img');
    if (!img) return;
    setupGroup('compare-ds-tabs', function (ds) {
      img.src = 'static/images/operators/compare_' + ds + '.jpg';
      img.alt = 'Ground truth and every operator at t = 0.7 (' + ds.toUpperCase() + ', first test trajectory)';
    });
  }

  // ------------------------------------------------------------------ per-timestep chart
  var COLORS = { 'Phaedra': '#1f4e8c', 'FSQ': '#e07b39', 'VQ-VAE-2': '#9467bd', 'Continuous': '#d62728',
                 'FNO': '#2ca02c', 'CNO': '#8c564b', 'ViT': '#17a2b8' };
  var DASH = { 'Phaedra': [], 'FSQ': [7, 4], 'VQ-VAE-2': [7, 4], 'Continuous': [7, 4],
               'FNO': [2, 3], 'CNO': [2, 3], 'ViT': [2, 3] };
  var LABEL = { 'Phaedra': 'Phaedra tokens (ours)', 'FSQ': 'FSQ tokens', 'VQ-VAE-2': 'VQ-VAE-2 tokens',
                'Continuous': 'continuous latents', 'FNO': 'FNO', 'CNO': 'CNO', 'ViT': 'ViT' };

  function setupChart() {
    var canvas = document.getElementById('timestep-chart');
    var note = document.getElementById('timestep-chart-note');
    var holder = document.getElementById('results-data');
    if (!canvas || !holder) return;
    if (typeof Chart === 'undefined') {
      if (note) note.textContent = 'The interactive chart could not be loaded; the numbers are in docs/results/per_timestep_*.csv.';
      return;
    }
    var data = JSON.parse(holder.textContent);
    var state = { dataset: 'KH', strategy: 'direct', log: true };
    var chart = null;

    function render() {
      var s = data.strategies[state.strategy];
      var series = s.series[state.dataset];
      var datasets = data.models.filter(function (m) { return series[m]; }).map(function (m) {
        var ours = m === 'Phaedra';
        return {
          label: LABEL[m] || m,
          data: series[m].map(function (y, i) { return { x: s.t[i], y: y }; }).filter(function (p) { return p.y !== null; }),
          borderColor: COLORS[m], backgroundColor: COLORS[m], borderDash: DASH[m] || [],
          borderWidth: ours ? 3.5 : 1.8, pointRadius: ours ? 4 : 2.5, pointHoverRadius: 6, order: ours ? 0 : 1
        };
      });
      if (chart) chart.destroy();
      chart = new Chart(canvas, {
        type: 'line',
        data: { datasets: datasets },
        options: {
          responsive: true, maintainAspectRatio: false, animation: false,
          interaction: { mode: 'nearest', axis: 'x', intersect: false },
          scales: {
            x: { type: 'linear', min: 0, max: 0.72,
                 ticks: { stepSize: 0.1, callback: function (v) { return Number(v).toFixed(1); } },
                 title: { display: true, text: 'time t (predicted from t = 0)' } },
            y: { type: state.log ? 'logarithmic' : 'linear', beginAtZero: !state.log,
                 title: { display: true, text: 'relative L1 error (%), mean over ρ, u, v, p' } }
          },
          plugins: {
            legend: { position: 'bottom', labels: { usePointStyle: true, boxWidth: 8 } },
            tooltip: { callbacks: {
              title: function (items) { return items.length ? 't = ' + items[0].parsed.x.toFixed(2) : ''; },
              label: function (ctx) { return ctx.dataset.label + ': ' + ctx.parsed.y.toFixed(2) + ' %'; } } }
          }
        }
      });
    }

    setupGroup('chart-ds', function (v) { state.dataset = v; render(); });
    setupGroup('chart-strategy', function (v) { state.strategy = v; render(); });
    var logBox = document.getElementById('chart-log');
    if (logBox) logBox.addEventListener('change', function () { state.log = logBox.checked; render(); });
    render();
  }

  document.addEventListener('DOMContentLoaded', function () {
    setupVideo();
    setupCompare();
    setupChart();
  });
})();
