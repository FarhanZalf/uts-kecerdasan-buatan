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

// Grafik Perbandingan Aktual vs Prediksi pada Data Uji
window.renderActualVsPredChart = function (canvasId, labels, actualData, predData) {
  const ctx = document.getElementById(canvasId);
  if (!ctx) return;

  new Chart(ctx, {
    type: 'line',
    data: {
      labels: labels,
      datasets: [
        {
          label: 'Data Aktual (GAIKINDO)',
          data: actualData,
          borderColor: '#10b981',
          backgroundColor: 'rgba(16, 185, 129, 0.15)',
          borderWidth: 3,
          tension: 0.2,
          pointRadius: 6,
          pointBackgroundColor: '#10b981'
        },
        {
          label: 'Prediksi Model MLP',
          data: predData,
          borderColor: '#ef4444',
          borderDash: [5, 5],
          backgroundColor: 'transparent',
          borderWidth: 3,
          tension: 0.2,
          pointRadius: 6,
          pointBackgroundColor: '#ef4444'
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
              return context.dataset.label + ': ' + context.parsed.y.toLocaleString('id-ID') + ' Unit';
            }
          }
        }
      },
      scales: {
        x: {
          grid: { color: 'rgba(255, 255, 255, 0.05)' },
          ticks: { color: '#94a3b8' }
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
  });
};
