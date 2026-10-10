// Script Interaktivitas Web Prediksi Penjualan Mobil Toyota (Time Series MLP)

document.addEventListener('DOMContentLoaded', function () {
  // 1. Filter Pencarian pada Tabel Dataset
  const searchInput = document.getElementById('tableSearchInput');
  const table = document.getElementById('datasetTable');

  if (searchInput && table) {
    searchInput.addEventListener('keyup', function () {
      const filter = searchInput.value.toLowerCase();
      const rows = table.getElementsByTagName('tr');

      for (let i = 1; i < rows.length; i++) {
        const text = rows[i].textContent || rows[i].innerText;
        if (text.toLowerCase().indexOf(filter) > -1) {
          rows[i].style.display = '';
        } else {
          rows[i].style.display = 'none';
        }
      }
    });
  }
});

// Helper pembuatan grafik garis interaktif Chart.js
window.renderSalesTrendChart = function (canvasId, labels, dataPoints) {
  const ctx = document.getElementById(canvasId);
  if (!ctx) return;

  new Chart(ctx, {
    type: 'line',
    data: {
      labels: labels,
      datasets: [{
        label: 'Penjualan Toyota (Unit)',
        data: dataPoints,
        borderColor: '#06b6d4',
        backgroundColor: 'rgba(6, 182, 212, 0.12)',
        borderWidth: 2,
        fill: true,
        tension: 0.25,
        pointRadius: 2,
        pointHoverRadius: 6,
        pointBackgroundColor: '#ef4444'
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          labels: { color: '#94a3b8', font: { family: 'Plus Jakarta Sans' } }
        },
        tooltip: {
          backgroundColor: '#0f172a',
          titleColor: '#f8fafc',
          bodyColor: '#38bdf8',
          borderColor: 'rgba(255,255,255,0.1)',
          borderWidth: 1,
          callbacks: {
            label: function (context) {
              return ' ' + context.parsed.y.toLocaleString('id-ID') + ' Unit';
            }
          }
        }
      },
      scales: {
        x: {
          grid: { color: 'rgba(255, 255, 255, 0.04)' },
          ticks: { color: '#64748b', maxTicksLimit: 14 }
        },
        y: {
          grid: { color: 'rgba(255, 255, 255, 0.04)' },
          ticks: {
            color: '#64748b',
            callback: function (val) {
              return val.toLocaleString('id-ID');
            }
          }
        }
      }
    }
  });
};

// Grafik Perbandingan Aktual vs Prediksi pada Data Uji dengan Fitur Toggle Switcher
let evalChartInstance = null;

window.initEvalChartSwitcher = function (canvasId, testData, fullData) {
  const ctx = document.getElementById(canvasId);
  if (!ctx) return;

  function buildConfig(mode) {
    const isTest = mode === 'test';
    const labels = isTest ? testData.labels : fullData.labels;
    const actual = isTest ? testData.actual : fullData.actual;
    const pred = isTest ? testData.pred : fullData.pred;

    return {
      type: 'line',
      data: {
        labels: labels,
        datasets: [
          {
            label: isTest ? 'Data Aktual (GAIKINDO)' : 'Data Historis GAIKINDO (Aktual)',
            data: actual,
            borderColor: '#10b981',
            backgroundColor: isTest ? 'rgba(16, 185, 129, 0.15)' : 'rgba(16, 185, 129, 0.08)',
            borderWidth: isTest ? 3 : 2,
            tension: 0.2,
            pointRadius: isTest ? 6 : 1,
            pointHoverRadius: 6,
            pointBackgroundColor: '#10b981',
            fill: !isTest
          },
          {
            label: isTest ? 'Prediksi Model MLP' : 'Tebakan Model MLP (6 Bulan Data Uji)',
            data: pred,
            borderColor: '#ef4444',
            borderDash: [5, 5],
            backgroundColor: 'transparent',
            borderWidth: 3,
            tension: 0.2,
            pointRadius: isTest ? 6 : 4,
            pointHoverRadius: 7,
            pointBackgroundColor: '#ef4444',
            spanGaps: false
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            labels: { color: '#e2e8f0', font: { family: 'Plus Jakarta Sans', weight: '600' } }
          },
          tooltip: {
            backgroundColor: '#0f172a',
            callbacks: {
              label: function (context) {
                if (context.parsed.y === null || context.parsed.y === undefined) return '';
                return context.dataset.label + ': ' + context.parsed.y.toLocaleString('id-ID') + ' Unit';
              }
            }
          }
        },
        scales: {
          x: {
            grid: { color: 'rgba(255, 255, 255, 0.05)' },
            ticks: {
              color: '#94a3b8',
              maxTicksLimit: isTest ? 10 : 16
            }
          },
          y: {
            grid: { color: 'rgba(255, 255, 255, 0.05)' },
            ticks: {
              color: '#94a3b8',
              callback: function (val) {
                return val.toLocaleString('id-ID');
              }
            }
          }
        }
      }
    };
  }

  // Inisialisasi awal mode test (6 bulan)
  evalChartInstance = new Chart(ctx, buildConfig('test'));

  // Handler pergantian mode
  window.switchEvalChart = function (mode) {
    const btnTest = document.getElementById('btnModeTest');
    const btnFull = document.getElementById('btnModeFull');
    const title = document.getElementById('evalChartTitle');
    const subtitle = document.getElementById('evalChartSubtitle');

    if (mode === 'test') {
      if (btnTest) btnTest.classList.add('btn-toggle-active');
      if (btnFull) btnFull.classList.remove('btn-toggle-active');
      if (title) title.textContent = 'Grafik Perbandingan: Data Aktual vs Prediksi Model MLP (Fokus Data Uji 6 Bulan)';
      if (subtitle) subtitle.textContent = 'Garis hijau menunjukkan angka riil GAIKINDO, garis merah putus-putus menunjukkan tebakan model MLP.';
    } else {
      if (btnFull) btnFull.classList.add('btn-toggle-active');
      if (btnTest) btnTest.classList.remove('btn-toggle-active');
      if (title) title.textContent = 'Grafik Linimasa Penuh: Riwayat Penjualan (2011–2023) & Posisi Data Uji Model MLP';
      if (subtitle) subtitle.textContent = 'Garis hijau adalah data historis (149 bulan). Di ujung kanan (Des 2022 s.d. Mei 2023) tampak perbandingan prediksi model MLP.';
    }

    if (evalChartInstance) {
      evalChartInstance.destroy();
    }
    evalChartInstance = new Chart(ctx, buildConfig(mode));
  };
};

window.renderActualVsPredChart = function (canvasId, labels, actualData, predData) {
  // Fallback kompatibilitas
  window.initEvalChartSwitcher(canvasId, { labels, actual: actualData, pred: predData }, { labels, actual: actualData, pred: predData });
};
