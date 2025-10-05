import './App.css'
import RankCard from './components/RankCard'
import FilterCard from './components/FilterCard'
import GraphicCard from './components/GraphicCard'
import Map from './components/Map'
import Header from './components/Header'
import SelectionBar from './components/SelectionBar'
import PopUpSuggestion from './components/PopUpSuggestion'
import PopUpNotification from './components/PopUpNotification'
import { useState, useEffect } from "react";  // Adicionei useEffect pra fetch

function App() {
  const [showSuggestion, setShowSuggestion] = useState(false);
  const [showNotification, setShowNotification] = useState(false);
  const [vulnerabilityData, setVulnerabilityData] = useState(null);  // State pros dados backend
  const [loading, setLoading] = useState(true);  // Loading
  const [selectedSector, setSelectedSector] = useState(null);  // Setor clicado pra análise
  const [analysisText, setAnalysisText] = useState('');  // Texto da LLM

  // Fetch inicial do backend
  useEffect(() => {
    const fetchData = async () => {
      try {
        const response = await fetch('http://localhost:5001/api/data');
        if (!response.ok) throw new Error('Erro no backend');
        const data = await response.json();
        setVulnerabilityData(data);  // GeoJSON/features
        setLoading(false);
      } catch (error) {
        console.error('Falha fetch:', error);
        setLoading(false);
        // Fallback mock se quiser: setVulnerabilityData(mockData);
      }
    };
    fetchData();
  }, []);

  const handleFilterClick = (filterType) => {  // Passe filterType como prop
    setShowNotification(true);
    // Opcional: Refetch com filtro, ex: /api/data?filter=heat
  };

  const handleSuggestionClick = async () => {
    setShowNotification(false);
    if (selectedSector) {
      try {
        const response = await fetch('http://localhost:5001/api/llm', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ ...selectedSector, filter: 'Heat' })  // Envia setor + contexto
        });
        const result = await response.json();
        setAnalysisText(result.analysis_text || 'Análise gerada pela IA.');
        setShowSuggestion(true);
      } catch (error) {
        console.error('Erro LLM:', error);
        setAnalysisText('Erro na análise. Tente novamente.');
        setShowSuggestion(true);
      }
    } else {
      setAnalysisText('Selecione um setor no mapa primeiro!');
      setShowSuggestion(true);
    }
  };

  const handleCloseNotification = () => {
    setShowNotification(false);
    setShowSuggestion(false);
  };

  if (loading) {
    return <div className="flex justify-center items-center h-screen text-white">Carregando dados do backend...</div>;
  }

  // Top 5 distritos por HVI
  const rankedDistricts = vulnerabilityData?.features
    ?.sort((a, b) => (b.properties?.indice_vulnerabilidade || 0) - (a.properties?.indice_vulnerabilidade || 0))
    ?.slice(0, 5) || [];

  return (
    <div className='relative bg-space-background min-h-screen'>
      <Header />
      
      <main className='my-2'>
        <SelectionBar />
        <div className='grid grid-cols-4 h-auto my-2'>
          <div className='flex flex-col col-span-3 ml-2'>
            <div className='flex justify-center items-center text-white h-2/3 w-full bg-blue-950'>
              <Map geoJsonData={vulnerabilityData} onSectorSelect={setSelectedSector} />
            </div>

            <div className='grid grid-cols-1 gap-2 h-1/2 w-full mt-2'> 
              <GraphicCard data={vulnerabilityData} />
            </div>
          </div>
          
          <div className='col-span-1 h-auto mx-2'>
            <div className='h-auto mt-4 mx-2'>
              <div className='grid grid-cols-2 gap-1'>
                <h3 className='col-span-2 font-bold text-lg text-white py-2 border-b-2 border-white/20 mb-4'> Filtragens </h3>
                <FilterCard 
                  title="Heat • ECOSTRESS" 
                  description="Temperatura, calor urbano, ilhas térmicas, conforto, energia" 
                  colorClass="bg-our-orange" 
                  onClick={() => handleFilterClick('heat')} 
                />
                <FilterCard 
                  title="Flood • SRTM+GPM" 
                  description="Alagamentos, chuva, relevo, drenagem, vulnerabilidade" 
                  colorClass="bg-our-purple" 
                  onClick={() => handleFilterClick('flood')} 
                />
                <FilterCard 
                  title="Green • NDVI" 
                  description="Vegetação, cobertura verde, saúde urbana, sustentabilidade" 
                  colorClass="bg-our-green" 
                  onClick={() => handleFilterClick('green')} 
                />
                <FilterCard 
                  title="Air • OMI/TEMPO" 
                  description="Poluição, qualidade, emissões, atmosfera, saúde" 
                  colorClass="bg-our-white" 
                  onClick={() => handleFilterClick('air')} 
                />
              </div>
            </div>

            <div className='flex flex-col h-auto'>
              <h3 className='font-bold text-lg text-white mx-2 py-2 mt-4 mb-1 border-b-2 border-white/20'> Regiões mais atingidas </h3>
              {rankedDistricts.map((district, index) => (
                <RankCard 
                  key={index} 
                  rank={index + 1}
                  name={district.properties?.NM_BAIRRO || 'Desconhecido'}
                  score={district.properties?.indice_vulnerabilidade || 0}
                  onClick={() => setSelectedSector(district.properties)}  // Seleciona pra LLM
                />
              ))}
            </div>
          </div>
        </div>

        {showNotification && (
          <PopUpNotification onClick={handleSuggestionClick} onClose={handleCloseNotification} />
        )}
        {showSuggestion && (
          <PopUpSuggestion onClose={handleCloseNotification} analysisText={analysisText} />  // Passe texto da LLM
        )}
      </main>

      <footer className='flex justify-center items-center bg-slate-500 w-full h-[100px]'>
        <p> RODAPÉ </p>
      </footer>
    </div>
  );
}

export default App;