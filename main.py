import pandas as pd
import geopandas as gpd
import requests
import os
import numpy as np
from scipy.interpolate import griddata
from rasterio.transform import from_origin
from rasterstats import zonal_stats
from dotenv import load_dotenv
from flask import Flask, jsonify, request
from flask_cors import CORS
import json

# --- Carrega as variáveis de ambiente do ficheiro .env ---
load_dotenv()

# --- 1. CONFIGURAÇÃO GERAL ---
PATH_MALHA_2022_SHP = r'C:\PE_setores_CD2022\PE_setores_CD2022.shp'
PATH_POPULACAO_2022_CSV = r'c:\Agregados_preliminares_por_setores_censitarios_PE.csv'
PATH_RENDA_2010_CSV = r'C:\PE_20231030\Base_informaçoes_setores2010_universo_PE\CSV\PessoaRenda_PE.csv'
CODIGO_MUNICIPIO_RECIFE = '2611606'
OUTPUT_GEOJSON_FILE = 'recife_vulnerabilidade.geojson'

FATOR_CORRECAO_RENDA = 2.03 

DISTRICT_COORDS = {
    'Norte': (-7.98, -34.88), 'Sul': (-8.12, -34.92),
    'Leste': (-8.05, -34.85), 'Oeste': (-8.05, -34.95),
    'Central': (-8.05, -34.90)
}

# --- 2. FUNÇÕES DE ANÁLISE DE DADOS (O "Cérebro") ---

def _normalize(series):
    """Função auxiliar para normalizar dados para a escala 0-1."""
    min_val, max_val = series.min(), series.max()
    if pd.isna(min_val) or pd.isna(max_val) or max_val == min_val:
        return pd.Series(0.5, index=series.index)
    return (series - min_val) / (max_val - min_val)

def run_full_analysis():
    """Executa a análise completa e salva um ficheiro GeoJSON com os resultados."""
    print("--- INICIANDO ANÁLISE COMPLETA DOS DADOS ---")
    
    # --- 1. Processamento IBGE ---
    print("--- 1. Processando e Unindo Dados do IBGE ---")
    try:
        gdf_malha = gpd.read_file(PATH_MALHA_2022_SHP, dtype={'CD_SETOR': str})
        df_pop_2022 = pd.read_csv(PATH_POPULACAO_2022_CSV, sep=';', encoding='latin-1', decimal=',', dtype={'CD_SETOR': str})
        df_renda_2010 = pd.read_csv(PATH_RENDA_2010_CSV, sep=';', encoding='latin-1', decimal=',', dtype={'Cod_setor': str})
    except FileNotFoundError as e:
        print(f"❌ ERRO: Ficheiro do IBGE não encontrado! {e}")
        return False

    # Padronização e limpeza dos IDs
    for df, col_name in [(gdf_malha, 'CD_SETOR'), (df_pop_2022, 'CD_SETOR'), (df_renda_2010, 'Cod_setor')]:
        df.rename(columns={col_name: 'CD_CENSITARIO'}, inplace=True)
        df['CD_CENSITARIO'] = df['CD_CENSITARIO'].str.extract(r'(\d{15})').fillna('')

    df_pop_sel = df_pop_2022[['CD_CENSITARIO', 'v0001', 'AREA_KM2']].rename(columns={'v0001': 'populacao_total', 'AREA_KM2': 'area_km2'})
    df_renda_sel = df_renda_2010[['CD_CENSITARIO', 'V003']].rename(columns={'V003': 'renda_media_2010'})
    
    gdf_final = gdf_malha.merge(df_pop_sel, on='CD_CENSITARIO', how='left')
    gdf_final = gdf_final.merge(df_renda_sel, on='CD_CENSITARIO', how='left')
    
    gdf_recife = gdf_final[gdf_final['CD_MUN'] == CODIGO_MUNICIPIO_RECIFE].copy()
    
    if gdf_recife.empty:
        print("❌ Nenhum dado para Recife encontrado após o filtro.")
        return False

    # Tratamento de dados faltantes (Imputação)
    for col in ['populacao_total', 'area_km2', 'renda_media_2010']:
        gdf_recife[col] = pd.to_numeric(gdf_recife[col], errors='coerce')
        gdf_recife[col] = gdf_recife[col].fillna(gdf_recife[col].median())

    gdf_recife['renda_corrigida'] = gdf_recife['renda_media_2010'] * FATOR_CORRECAO_RENDA
    gdf_recife = gdf_recife[gdf_recife['area_km2'] > 0].copy()
    gdf_recife['densidade_pop'] = gdf_recife['populacao_total'] / gdf_recife['area_km2']
    
    if 'NM_BAIRRO' not in gdf_recife.columns: gdf_recife['NM_BAIRRO'] = 'Recife'
    print(f"✅ Dados do IBGE para {len(gdf_recife)} setores de Recife processados.")

    # --- 2. Processamento Temperatura API ---
    print("\n--- 2. Obtendo e Interpolando Temperaturas ---")
    api_key = os.getenv('OWM_API_KEY')
    points, values = [], []
    if api_key:
        for city, (lat, lon) in DISTRICT_COORDS.items():
            url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric"
            try:
                resp = requests.get(url, timeout=5).json()
                if 'main' in resp:
                    points.append([lon, lat]); values.append(resp['main']['temp'])
                    print(f"✅ Temp {city}: {resp['main']['temp']}°C")
            except Exception as e:
                print(f"⚠️ Falha na API para {city}: {e}")
    
    if len(points) >= 4:
        gdf_wgs = gdf_recife.to_crs(epsg=4326)
        minx, miny, maxx, maxy = gdf_wgs.total_bounds
        grid_x, grid_y = np.meshgrid(np.linspace(minx, maxx, 100), np.linspace(miny, maxy, 100))
        grid_temp = griddata(points, values, (grid_x, grid_y), method='cubic', fill_value=np.mean(values))
        transform = from_origin(minx, maxy, (maxx - minx) / 100, (maxy - miny) / 100)
        stats = zonal_stats(gdf_wgs, grid_temp, affine=transform, stats="mean", nodata=np.nan)
        gdf_recife['temperatura_media_estimada'] = [s.get('mean') for s in stats]
        gdf_recife['temperatura_media_estimada'] = gdf_recife['temperatura_media_estimada'].fillna(gdf_recife['temperatura_media_estimada'].median())
    else:
        print("⚠️ Pontos de temperatura insuficientes. A usar valor fixo de 28°C.")
        gdf_recife['temperatura_media_estimada'] = 28.0

    # --- 3. Cálculo do Índice e Salvamento ---
    print("\n--- 3. Calculando Índice de Vulnerabilidade e Salvando Resultados ---")
    temp_norm = _normalize(gdf_recife['temperatura_media_estimada'])
    dens_norm = _normalize(gdf_recife['densidade_pop'])
    renda_norm_inv = 1 - _normalize(gdf_recife['renda_corrigida'])
    gdf_recife['indice_vulnerabilidade'] = (temp_norm * 0.5) + (renda_norm_inv * 0.3) + (dens_norm * 0.2)

    colunas_frontend = ['CD_CENSITARIO', 'NM_BAIRRO', 'indice_vulnerabilidade', 'densidade_pop', 'renda_corrigida', 'temperatura_media_estimada', 'geometry']
    gdf_recife_4326 = gdf_recife.to_crs(epsg=4326)
    gdf_recife_4326[colunas_frontend].to_file(OUTPUT_GEOJSON_FILE, driver='GeoJSON')
    print(f"✅ Ficheiro de dados final '{OUTPUT_GEOJSON_FILE}' salvo com sucesso.")
    
    print("\n--- ANÁLISE COMPLETA CONCLUÍDA ---")
    return True

# --- 3. CONFIGURAÇÃO DO SERVIDOR FLASK ---
app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "http://localhost:5173"}})

# Cache para os dados para evitar recálculos desnecessários
gdf_cache = None

def get_data():
    """Função para carregar e fazer cache dos dados processados."""
    global gdf_cache
    if gdf_cache is None:
        if not os.path.exists(OUTPUT_GEOJSON_FILE):
            return None
        gdf_cache = gpd.read_file(OUTPUT_GEOJSON_FILE)
    return gdf_cache

@app.route('/api/data')
def get_vulnerability_data():
    """Endpoint principal para servir todos os dados GeoJSON."""
    data = get_data()
    if data is None:
        return jsonify({"error": "Ficheiro de dados não encontrado. A análise inicial pode ter falhado."}), 500
    return jsonify(json.loads(data.to_json()))

@app.route('/api/dashboard/top_bairros')
def get_top_bairros():
    """Endpoint para o dashboard: ranking dos 10 bairros mais vulneráveis."""
    data = get_data()
    if data is None: return jsonify({"error": "Dados não disponíveis."}), 500
    
    bairros = data.groupby('NM_BAIRRO')['indice_vulnerabilidade'].mean().sort_values(ascending=False).head(10)
    return jsonify(bairros.to_dict())

@app.route('/api/dashboard/correlacao')
def get_correlacao_data():
    """Endpoint para o dashboard: dados para o gráfico de dispersão (Renda vs. Temperatura)."""
    data = get_data()
    if data is None: return jsonify({"error": "Dados não disponíveis."}), 500
    
    # Amostra de 500 pontos para não sobrecarregar o gráfico
    sample_data = data.sample(n=min(500, len(data)))
    correlacao = sample_data[['renda_corrigida', 'temperatura_media_estimada']]
    return jsonify(correlacao.to_dict('records'))

@app.route('/api/llm', methods=['POST'])
def get_llm_analysis():
    """Endpoint para o chatbot: atua como proxy para a API da Gemini."""
    sector_data = request.json
    gemini_api_key = os.getenv('GEMINI_API_KEY')
    if not gemini_api_key:
        return jsonify({"error": "Chave da API Gemini não configurada no backend."}), 500
        
    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key={gemini_api_key}"
    
    system_prompt = "Aja como um planeador urbano especialista em justiça climática. A sua resposta deve ser em português."
    user_prompt = f"""Analise os dados do seguinte setor censitário em Recife:
- Bairro: {sector_data.get('NM_BAIRRO', 'N/A')}
- Índice de Vulnerabilidade (0 a 1): {sector_data.get('indice_vulnerabilidade', 0):.2f}
- Temperatura Média Estimada: {sector_data.get('temperatura_media_estimada', 0):.1f}°C
- Densidade Populacional: {sector_data.get('densidade_pop', 0):.0f} hab/km²
- Renda Média Corrigida (Estimativa): R$ {sector_data.get('renda_corrigida', 0):.0f}

Com base nestes dados:
1. Faça um diagnóstico conciso da criticidade da área.
2. Proponha UMA intervenção de pequena escala e alto impacto (ex: parque de bolso, corredor verde, pintura de telhados com cores claras).
3. Estime o impacto da sua sugestão. Por exemplo, assuma que a intervenção reduz a temperatura local em 2°C e recalcule o índice de vulnerabilidade (apresente o cálculo de forma simplificada).
Use **negrito** para destacar a recomendação e a melhoria percentual."""

    try:
        response = requests.post(api_url, json={"contents": [{"parts": [{"text": user_prompt}]}], "systemInstruction": {"parts": [{"text": system_prompt}]}})
        response.raise_for_status()
        result = response.json()
        text = result['candidates'][0]['content']['parts'][0]['text']
        return jsonify({"analysis_text": text})
    except Exception as e:
        return jsonify({"error": f"Falha na comunicação com a API da Gemini: {str(e)}"}), 500

# --- 4. EXECUÇÃO PRINCIPAL ---
if __name__ == '__main__':
    if not os.path.exists(OUTPUT_GEOJSON_FILE):
        run_full_analysis()
    else:
        print("Ficheiro de dados já existe. A saltar a análise inicial.")
        
    print("\n--- INICIANDO SERVIDOR FLASK NA PORTA 5001 ---")
    print("Pronto para receber conexões do frontend React em http://localhost:5173")
    app.run(debug=True, port=5001)

