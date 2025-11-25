import signal
import sys
from fastapi import FastAPI, HTTPException
from typing import Dict, Any
import numpy as np
import random
import json
# GeoPandas는 실제 지리공간 처리 시 필요.

from fastapi.middleware.cors import CORSMiddleware
import folium
from fastapi.responses import HTMLResponse
import geopandas as gpd

app = FastAPI(title="LRI Engine Backend Prototype")

# allow CORS for local frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:3000", "http://localhost:3000", "http://127.0.0.1:3000/"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# LRI 모델 상수 및 파라미터 (회전익 기준)
W_W, W_N, W_T = 0.45, 0.35, 0.20  # 가중치 (w_W, w_N, w_T)
AL_H, AL_V = 40, 50              # 항법 한계 (AL_H, AL_V) - APV-I 기준
TAU_RED, TAU_YELLOW = 60, 80     # 임계값 (tau) -- FIXED: red < yellow

# ----------------------------------------------------------------------
# 헬퍼 함수: LRI 수식 구현
# ----------------------------------------------------------------------

def calculate_lri(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    LRI 모델 수식을 기반으로 위험도를 계산합니다.
    """
    try:
        # 1. 기상 W 계산 (W = 100 * min(1, p/p_req) * (1 - α_cloud))
        p = data.get('actual_visibility', 50)
        p_req = data.get('required_visibility', 30)
        alpha_cloud = data.get('alpha_cloud', 0.1)
        W = 100 * min(1, p / p_req) * (1 - alpha_cloud)

        # 2. 항법무결성 N 계산 (N = 100 - [50 * max(...)])
        HPL = data.get('HPL', 35)
        VPL = data.get('VPL', 45)
        N = 100 - (50 * max(0, (HPL - AL_H) / 20) + 50 * max(0, (VPL - AL_V) / 20))
        
        # 3. 지형 T 계산 (T = 100 * w * (1 - α_terrain) - 40 * r_{OCH<0})
        alpha_terrain = data.get('alpha_terrain', 0.05)
        r_och_neg = data.get('r_och_neg', 0.0)
        T = 100 * W_T * (1 - alpha_terrain) - 40 * r_och_neg

        # 4. 최종 위험도 LRI 계산 (LRI = 100 / (w_W/W + w_N/N + w_T/T'))
        T_norm = T / W_T if W_T != 0 else 0
        T_safe = max(5, T_norm) 
        LRI = 100 / (W_W / W + W_N / N + W_T / T_safe)
        LRI = round(min(100, LRI), 2)

        # 5. 하드 스톱 조건
        CTBT = data.get('CTBT', 273)
        delta_sigma_0 = data.get('delta_sigma_0', 0.0)
        core_percent = data.get('core_percent', 0)
        
        # Require both horizontal and vertical protection limits to be exceeded
        # to avoid single-parameter false positives triggering a hard stop.
        is_hard_stop = (CTBT < 235) or \
                       ((HPL > AL_H) and (VPL > AL_V)) or \
                       (delta_sigma_0 > 3.0 and core_percent >= 30)

        # 6. 등급 판단
        if is_hard_stop:
            grade = "RED (HARD STOP)"
        elif LRI < TAU_RED:
            grade = "RED (SEVERE)"
        elif LRI < TAU_YELLOW:
            grade = "YELLOW (WARNING)"
        else:
            grade = "GREEN (VERY GOOD)"
        
        return {
            "LRI": LRI,
            "Grade": grade,
            "W_score": round(W, 2),
            "N_score": round(N, 2),
            "T_score": round(T, 2),
            "HardStop": is_hard_stop
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def _generate_data_from_coords(lat: float, lon: float) -> Dict[str, Any]:
    """
    Generates mock risk data based on latitude and longitude.
    This simulates fetching real-world data for a specific point.
    """
    # Use coordinates to seed the random number generator for deterministic results
    # at the same location.
    seed = int(lat * 1000) + int(lon * 1000)
    rng = random.Random(seed)

    # --- W (Weather) Simulation ---
    # Simulate a weather front moving across the map
    weather_factor = (np.sin(lat / 10) + np.cos(lon / 10)) / 2  # Value between -1 and 1
    alpha_cloud = np.interp(weather_factor, [-1, 1], [0.05, 0.7]) # Map to cloud cover
    # Make CTBT riskier in a specific "storm" area
    ctbt = 230 if 34 < lat < 36 and 126 < lon < 128 else 273

    # --- N (Navigation) Simulation ---
    # 5% chance of HPL/VPL hard-stop, 20% chance of degraded values
    nav_roll = rng.random()
    if nav_roll < 0.05: # 5% Hard Stop
        hpl = AL_H + rng.uniform(1, 15)
        vpl = AL_V + rng.uniform(1, 10)
    elif nav_roll < 0.25: # 20% Degraded
        hpl = AL_H - rng.uniform(10, 20)
        vpl = AL_V - rng.uniform(10, 20)
    else: # 75% Normal
        hpl = rng.uniform(10, AL_H - 5)
        vpl = rng.uniform(10, AL_V - 5)

    # --- T (Terrain) Simulation ---
    # Simulate a mountainous region in the east
    if lon > 128.5:
        alpha_terrain = rng.uniform(0.1, 0.3)
        r_och_neg = rng.uniform(0.05, 0.1)
    else: # Plains in the west
        alpha_terrain = rng.uniform(0.01, 0.05)
        r_och_neg = rng.uniform(0.0, 0.01)

    return {
        "actual_visibility": rng.uniform(25, 60),
        "required_visibility": 30,
        "alpha_cloud": alpha_cloud,
        "CTBT": ctbt,
        "HPL": hpl,
        "VPL": vpl,
        "alpha_terrain": alpha_terrain,
        "r_och_neg": r_och_neg,
        "delta_sigma_0": rng.uniform(0.0, 4.0),
        "core_percent": rng.choice([5, 30, 40])
    }


# def _generate_scenario(name: str) -> Dict[str, Any]:
    """
    Return a mock data dict for a named scenario.
    Supported names: "very_good", "severe", "warning", "hard_stop", "random"
    """
    if name == "very_good":
        return {
            "actual_visibility": 60.0,
            "required_visibility": 30.0,
            "alpha_cloud": 0.0,
            "HPL": AL_H,
            "VPL": AL_V,
            "alpha_terrain": 0.01,
            "r_och_neg": 0.0,
            "delta_sigma_0": 0.0,
            "core_percent": 0,
            "CTBT": 273
        }
    if name == "severe":
        return {
            "actual_visibility": 0.05,
            "required_visibility": 30.0,
            "alpha_cloud": 0.99,
            "HPL": 59.6,
            "VPL": 50.0,
            "alpha_terrain": 0.9,
            "r_och_neg": 0.1,
            "delta_sigma_0": 2.0,
            "core_percent": 10,
            "CTBT": 273
        }
    if name == "warning":
        return {
            "actual_visibility": 1.0,
            "required_visibility": 30.0,
            "alpha_cloud": 0.892,
            "HPL": 78.0,
            "VPL": 50.0,
            "alpha_terrain": 0.01,
            "r_och_neg": 0.0,
            "delta_sigma_0": 0.5,
            "core_percent": 5,
            "CTBT": 273
        }
    if name == "hard_stop":
        return {
            "actual_visibility": 50,
            "required_visibility": 30,
            "alpha_cloud": 0.05,
            "HPL": AL_H + 20,
            "VPL": AL_V,
            "alpha_terrain": 0.05,
            "r_och_neg": 0.0,
            "delta_sigma_0": 4.0,
            "core_percent": 40,
            "CTBT": 230
        }
    # random/default
    return {
        "actual_visibility": random.uniform(25, 60),
        "required_visibility": 30,
        "alpha_cloud": random.uniform(0.05, 0.4),
        "CTBT": random.choice([273, 273, 273, 230]),
        "HPL": 40 + random.choice([0, 0, 0, 15]),
        "VPL": 50 + random.choice([0, 0, 0, 10]),
        "alpha_terrain": random.uniform(0.01, 0.3),
        "r_och_neg": random.uniform(0.0, 0.1),
        "delta_sigma_0": random.uniform(0.0, 4.0),
        "core_percent": random.choice([5, 30, 40])
    }

@app.post("/api/calculate_lri")
async def calculate_lri_endpoint(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Accepts JSON: { "lat": <float>, "lon": <float> }
    Calculates LRI based on coordinate-dependent data.
    """
    lat = payload.get("lat")
    lon = payload.get("lon")

    if lat is None or lon is None:
        raise HTTPException(status_code=400, detail="Latitude (lat) and Longitude (lon) are required.")

    # Generate data based on coordinates instead of a fixed scenario
    dynamic_data = _generate_data_from_coords(lat, lon)

    result = calculate_lri(dynamic_data)

    result['location'] = f"Lat: {lat:.4f}, Lon: {lon:.4f}"
    result['Evidence'] = [
        f"기상(W:{result['W_score']}): 구름 감쇠계수({dynamic_data['alpha_cloud']:.2f}) 적용.",
        f"항법(N:{result['N_score']}): HPL={dynamic_data['HPL']:.1f}m, VPL={dynamic_data['VPL']:.1f}m.",
        f"지형(T:{result['T_score']}): 지형 복잡도({dynamic_data['alpha_terrain']:.2f}) 및 장애물({dynamic_data['r_och_neg']:.2f}) 반영."
    ]

    return result

@app.get("/api/map", response_class=HTMLResponse)
async def get_map():
    """
    Generates a Leaflet map using GeoPandas and Folium.
    """
    # Lazy import for faster startup
    import folium
    from folium.elements import Element

    # 1. 위성 타일 레이어 변경
    m = folium.Map(
        location=[35.5, 128.0], 
        zoom_start=6, 
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community"
    )

    # Add BeautifyMarker plugin resources to the map header
    m.get_root().header.add_child(folium.CssLink("https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css"))
    m.get_root().header.add_child(folium.CssLink("https://cdn.jsdelivr.net/npm/leaflet-beautify-marker@1.0.9/dist/leaflet-beautify-marker.min.css"))
    m.get_root().header.add_child(folium.JavascriptLink("https://cdn.jsdelivr.net/npm/leaflet-beautify-marker@1.0.9/dist/leaflet-beautify-marker.min.js"))


    # 2. 인천 FIR 경계선 GeoJSON 파일 로드 및 지도에 추가
    try:
        fir_gdf = gpd.read_file("back/incheon_fir.geojson")
        folium.GeoJson(
            fir_gdf,
            name='Incheon FIR',
            style_function=lambda x: {'color': 'yellow', 'weight': 2, 'fillOpacity': 0}
        ).add_to(m)
    except Exception as e:
        # GeoJSON 파일이 없어도 맵은 로드되도록 예외 처리
        print(f"Could not load FIR GeoJSON: {e}")

    # 3. 부모 창과 통신하기 위한 스크립트 추가
    # - 지도 클릭 시 좌표를 부모 창으로 전송
    # - 부모 창에서 받은 좌표로 마커(비행기 아이콘) 추가
    script = """
        <script>
            // Increase the drag threshold to make clicks more reliable. Default is 5.
            L.Draggable.DRAGGING_THRESHOLD = 15;

            // Use L.BeautifyIcon to create a plane icon
            var planeIcon = L.BeautifyIcon.icon({
                icon: 'plane',
                iconShape: 'circle',
                borderColor: 'gray',
                textColor: 'black',
                backgroundColor: 'transparent'
            });
            var marker;

            // 지도 클릭 시 부모 창으로 좌표 전송
            this.on('click', function(e) {
                parent.postMessage({
                    type: 'MAP_CLICK',
                    lat: e.latlng.lat,
                    lon: e.latlng.lng
                }, '*');
            });

            // 부모 창으로부터 메시지 수신 (마커 추가용)
            window.addEventListener('message', function(event) {
                const { type, lat, lon } = event.data;
                if (type === 'ADD_MARKER') {
                    if (marker) {
                        this.removeLayer(marker);
                    }
                    marker = L.marker([lat, lon], {icon: planeIcon}).addTo(this);
                    this.setView([lat, lon], 8); // 마커 위치로 뷰 이동
                }
            }.bind(this));
        </script>
    """
    m.get_root().html.add_child(Element(script))

    # Render the map as HTML
    return m._repr_html_()

# Signal handler for graceful shutdown
def signal_handler(sig, frame):
    print("\nShutting down backend server gracefully...")
    sys.exit(0)

# Register signal handlers for SIGINT and SIGTERM
signal.signal(signal.SIGINT, signal_handler)  # Handles Ctrl+C
try:
    signal.signal(signal.SIGTSTP, signal_handler)  # Handles Ctrl+Z (suspend)
except AttributeError:
    # SIGTSTP may not exist on Windows; fall back to SIGTERM
    signal.signal(signal.SIGTERM, signal_handler)