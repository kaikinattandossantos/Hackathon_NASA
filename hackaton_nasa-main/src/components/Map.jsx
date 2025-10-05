// src/components/Map.jsx
import React, { useEffect, useState } from 'react';
import { MapContainer, TileLayer, GeoJSON } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import L from 'leaflet';

const Map = ({ geoJsonData, onSectorSelect }) => {
  const [geoJsonLayer, setGeoJsonLayer] = useState(null);
  const initialPosition = [-8.0578, -34.8827];
  const zoomLevel = 12;
  const darkModeTiles = 'https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png';

  // Estilo choropleth por HVI
  const style = (feature) => {
    const hvi = feature.properties?.indice_vulnerabilidade || 0;
    const color = hvi > 0.7 ? '#dc2626' : hvi > 0.4 ? '#f59e0b' : '#10b981';  // Vermelho alto, verde baixo
    return {
      fillColor: color,
      weight: 2,
      opacity: 1,
      color: 'white',
      dashArray: '3',
      fillOpacity: 0.7
    };
  };

  // Clique no setor
  const onEachFeature = (feature, layer) => {
    layer.on({
      click: () => onSectorSelect(feature.properties)  // Envia pro App pra LLM
    });
    layer.bindPopup(
      `<b>${feature.properties?.NM_BAIRRO || 'Setor'}</b><br>
       HVI: ${(feature.properties?.indice_vulnerabilidade || 0).toFixed(2)}<br>
       Temp: ${feature.properties?.temperatura_media_estimada?.toFixed(1)}°C<br>
       Densidade: ${feature.properties?.densidade_pop?.toFixed(0)} hab/km²`
    );
  };

  useEffect(() => {
    if (geoJsonData?.features) {
      const layer = L.geoJSON(geoJsonData, { style, onEachFeature });
      setGeoJsonLayer(layer);
    }
  }, [geoJsonData]);

  return (
    <MapContainer center={initialPosition} zoom={zoomLevel} scrollWheelZoom={true} style={{ height: '100%', width: '100%' }} className="rounded-lg shadow-xl">
      <TileLayer attribution='&copy; OpenStreetMap & CARTO' url={darkModeTiles} />
      {geoJsonLayer && <GeoJSON data={geoJsonLayer.toGeoJSON()} style={style} onEachFeature={onEachFeature} />}
    </MapContainer>
  );
};

export default Map;