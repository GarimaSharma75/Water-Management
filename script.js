// Dummy Data
const waterData = [60, 70, 80];
const soilData = [30, 25, 40];
const predictions = [
  { field_id: 1, crop: 'Corn', predicted: 60, lower: 55, upper: 65 },
  { field_id: 2, crop: 'Wheat', predicted: 50, lower: 45, upper: 55 },
  { field_id: 3, crop: 'Soybeans', predicted: 70, lower: 65, upper: 75 },
];

// Update Cards
document.getElementById('total-fields').textContent = predictions.length;
document.getElementById('average-water').textContent = (
  waterData.reduce((a, b) => a + b) / waterData.length
).toFixed(2) + ' mm';
document.getElementById('critical-fields').textContent = predictions.filter(p => p.predicted > 65).length;

// Populate Predictions Table
const tableBody = document.getElementById('prediction-table');
predictions.forEach(pred => {
  const row = `
    <tr>
      <td>${pred.field_id}</td>
      <td>${pred.crop}</td>
      <td>${pred.predicted}</td>
      <td>${pred.lower}</td>
      <td>${pred.upper}</td>
    </tr>`;
  tableBody.innerHTML += row;
});

// Water Requirement Chart
const ctxWater = document.getElementById('waterChart').getContext('2d');
new Chart(ctxWater, {
  type: 'line',
  data: {
    labels: ['Field 1', 'Field 2', 'Field 3'],
    datasets: [{
      label: 'Water Requirement (mm)',
      data: waterData,
      borderColor: 'rgba(75, 192, 192, 1)',
      backgroundColor: 'rgba(75, 192, 192, 0.2)',
      fill: true,
    }],
  },
});

// Soil Moisture Chart
const ctxSoil = document.getElementById('soilChart').getContext('2d');
new Chart(ctxSoil, {
  type: 'bar',
  data: {
    labels: ['Field 1', 'Field 2', 'Field 3'],
    datasets: [{
      label: 'Soil Moisture (%)',
      data: soilData,
      backgroundColor: ['#4caf50', '#2196f3', '#ff9800'],
    }],
  },
});
