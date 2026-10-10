// Interaktivitas UI Klasifikasi Iris UTS

document.addEventListener('DOMContentLoaded', function () {
  // 1. Sinkronisasi slider dan input number pada form prediksi
  const features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width'];

  features.forEach(feat => {
    const numInput = document.getElementById(`input_${feat}`);
    const sliderInput = document.getElementById(`slider_${feat}`);

    if (numInput && sliderInput) {
      numInput.addEventListener('input', function () {
        sliderInput.value = numInput.value;
      });

      sliderInput.addEventListener('input', function () {
        numInput.value = sliderInput.value;
      });
    }
  });

  // 2. Preset buttons untuk memudahkan pengujian data baru
  const presets = {
    setosa: { sl: 5.1, sw: 3.5, pl: 1.4, pw: 0.2 },
    versicolor: { sl: 5.9, sw: 3.0, pl: 4.2, pw: 1.5 },
    virginica: { sl: 6.5, sw: 3.0, pl: 5.5, pw: 2.0 }
  };

  window.applyPreset = function (species) {
    const data = presets[species];
    if (!data) return;

    if (document.getElementById('input_sepal_length')) {
      document.getElementById('input_sepal_length').value = data.sl;
      document.getElementById('slider_sepal_length').value = data.sl;

      document.getElementById('input_sepal_width').value = data.sw;
      document.getElementById('slider_sepal_width').value = data.sw;

      document.getElementById('input_petal_length').value = data.pl;
      document.getElementById('slider_petal_length').value = data.pl;

      document.getElementById('input_petal_width').value = data.pw;
      document.getElementById('slider_petal_width').value = data.pw;
    }
  };

  // 3. Pencarian / Filter sederhana di tabel Dataset
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
