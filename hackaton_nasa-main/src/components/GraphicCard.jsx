// src/components/GraphicCard.jsx
import { Bar } from 'react-chartjs-2';
import { Chart as ChartJS, CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend } from 'chart.js';

ChartJS.register(CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend);

const GraphicCard = ({ data }) => {
  const chartData = data?.features ? {
    labels: data.features.map(f => f.properties?.NM_BAIRRO?.slice(0, 8) || 'N/A'),  // Abrevia nomes
    datasets: [{
      label: 'Índice de Vulnerabilidade (HVI)',
      data: data.features.map(f => f.properties?.indice_vulnerabilidade || 0),
      backgroundColor: data.features.map(f => {
        const hvi = f.properties?.indice_vulnerabilidade || 0;
        return hvi > 0.7 ? 'rgba(220, 38, 38, 0.6)' : hvi > 0.4 ? 'rgba(245, 158, 11, 0.6)' : 'rgba(16, 185, 129, 0.6)';
      }),
      borderColor: 'rgba(255, 255, 255, 1)',
      borderWidth: 1
    }]
  } : { labels: [], datasets: [] };

  const options = {
    responsive: true,
    plugins: {
      legend: { position: 'top', labels: { color: 'white' } },
      title: { display: true, text: 'HVI por Bairro', color: 'white' }
    },
    scales: {
      y: { beginAtZero: true, ticks: { color: 'white' }, grid: { color: 'rgba(255,255,255,0.1)' } },
      x: { ticks: { color: 'white' }, grid: { color: 'rgba(255,255,255,0.1)' } }
    }
  };

  return (
    <div className="bg-black p-4 rounded shadow-lg">
      <Bar data={chartData} options={options} />
    </div>
  );
};

export default GraphicCard;