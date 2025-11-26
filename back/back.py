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
import os

app = FastAPI(title="LRI Engine Backend Prototype")

# Get allowed origins from environment variable, with a fallback for local dev
allowed_origins = os.environ.get("ALLOWED_ORIGINS", "http://127.0.0.1:3000,http://localhost:8000").split(",")

# Add your EC2 domain here for production
# For example: "http://your-ec2-domain.com"
# You can also use a wildcard for testing: ["*"]
allowed_origins.append("http://52.78.12.152")


# allow CORS for local frontend and production
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# LRI 모델 상수 및 파라미터 (회전익 기준)
W_W, W_N, W_T = 0.45, 0.35, 0.20  # 가중치 (w_W, w_N, w_T)
AL_H, AL_V = 40, 50              # 항법 한계 (AL_H, AL_V) - APV-I 기준
TAU_RED, TAU_YELLOW, TAU_BLUE = 60, 80, 90 # 임계값 (Severe, Warning, Good)

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

        # 6. 등급 판단 (4-level system)
        if is_hard_stop:
            grade = "RED (HARD STOP)"
        elif LRI < TAU_RED: # < 60
            grade = "YELLOW (SEVERE)"
        elif LRI < TAU_YELLOW: # < 80
            grade = "BLUE (WARNING)"
        else: # >= 80
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

@app.get("/api/lri_grid")
async def get_lri_grid(lat: float, lon: float, lri: float):
    """
    Generates a GeoJSON grid around a central point to simulate a choropleth map.
    The LRI value decays from the center.
    """
    grid_size = 11  # Grid dismensions (must be odd)
    cell_size = 0.01  # Degrees per cell
    
    features = []
    center_offset = (grid_size - 1) / 2

    for i in range(grid_size):
        for j in range(grid_size):
            # Calculate distance from center
            dist_x = i - center_offset
            dist_y = j - center_offset
            distance = np.sqrt(dist_x**2 + dist_y**2)

            # Decay LRI based on distance (simple linear decay)
            max_dist = np.sqrt(2 * center_offset**2)
            decay_factor = max(0, 1 - (distance / max_dist))
            cell_lri = lri * decay_factor

            # Define polygon for the grid cell
            min_lon = lon + (j - center_offset) * cell_size - (cell_size / 2)
            min_lat = lat + (i - center_offset) * cell_size - (cell_size / 2)
            max_lon = min_lon + cell_size
            max_lat = min_lat + cell_size
            
            polygon = [[
                [min_lon, min_lat], [max_lon, min_lat],
                [max_lon, max_lat], [min_lon, max_lat],
                [min_lon, min_lat]
            ]]

            features.append({
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": polygon
                },
                "properties": {
                    "lri": cell_lri
                }
            })

    return {"type": "FeatureCollection", "features": features}


@app.get("/api/map")
async def get_map(lat: float = None, lon: float = None, lri: float = None, grade: str = None):
    """
    Generates a Folium map with FIR boundary, optional marker, and optional choropleth.
    """
    import folium
    from folium import plugins
    import branca.colormap as cm

    # Determine initial map settings based on provided coordinates
    if lat is not None and lon is not None:
        initial_location = [lat, lon]
        initial_zoom = 14  # Closer zoom when a point is specified
    else:
        initial_location = [35.5, 128.0]
        initial_zoom = 7   # Default overview zoom

    # Create base map with the correct initial settings
    m = folium.Map(
        location=initial_location,
        zoom_start=initial_zoom,
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery"
    )

    # Add FIR boundary
    try:
        fir_gdf = gpd.read_file("back/incheon_fir.geojson")
        folium.GeoJson(
            fir_gdf,
            name='Incheon FIR',
            style_function=lambda x: {
                'color': 'yellow', 
                'weight': 2, 
                'fillOpacity': 0,
                'interactive': False  # This allows clicks to pass through
            }
        ).add_to(m)
    except Exception as e:
        print(f"[LRI DEBUG] FIR load failed: {e}")

    # Add plane marker if coordinates provided
    if lat is not None and lon is not None:
        try:
            # Create plane icon using BeautifyIcon plugin
            icon_plane = folium.plugins.BeautifyIcon(
                icon="plane",
                border_color="#666666",
                text_color="#666666",
                icon_shape="triangle"
            )
            
            folium.Marker(
                location=[lat, lon],
                icon=icon_plane,
                popup=f"Selected Location<br>Lat: {lat:.4f}<br>Lon: {lon:.4f}"
            ).add_to(m)
            
            # The location and zoom are now set during map initialization, so no need to change them here.
            print(f"[LRI DEBUG] Plane marker added at ({lat}, {lon})")
        except Exception as e:
            print(f"[LRI ERROR] Failed to add plane marker: {e}")

    # Add choropleth if LRI provided
    if lat is not None and lon is not None and lri is not None:
        try:
            # Generate grid data
            grid_size = 7 # choropleth grid size
            cell_size = 0.01  # change to make center cell size  
            center_offset = (grid_size - 1) / 2
            
            features = []
            for i in range(grid_size):
                for j in range(grid_size):
                    dist_x = i - center_offset
                    dist_y = j - center_offset
                    distance = np.sqrt(dist_x**2 + dist_y**2)
                    
                    max_dist = np.sqrt(2 * center_offset**2)
                    decay_factor = max(0, 1 - (distance / max_dist))
                    cell_lri = lri * decay_factor
                    
                    min_lon = lon + (j - center_offset) * cell_size - (cell_size / 2)
                    min_lat = lat + (i - center_offset) * cell_size - (cell_size / 2)
                    max_lon = min_lon + cell_size
                    max_lat = min_lat + cell_size
                    
                    polygon = [[
                        [min_lon, min_lat], [max_lon, min_lat],
                        [max_lon, max_lat], [min_lon, max_lat],
                        [min_lon, min_lat]
                    ]]
                    
                    features.append({
                        "type": "Feature",
                        "geometry": {"type": "Polygon", "coordinates": polygon},
                        "properties": {"lri": cell_lri}
                    })
            
            geojson_data = {"type": "FeatureCollection", "features": features}
            
            # Define color function to match the 4-level grading system
            def style_function(feature):
                lri_val = feature['properties']['lri']
                
                # Determine color based on the LRI of the specific grid cell
                if lri_val < TAU_RED: # < 60
                    color = '#fdfd96'  # Pastel Yellow for Severe
                elif lri_val < TAU_YELLOW: # < 80
                    color = '#aec6cf'  # Pastel Blue for Warning
                else: # >= 80
                    color = '#77dd77'  # Pastel Green for Very Good

                # Override color for the center cell if it's a Hard Stop case
                # The center cell has the highest LRI, equal to the initial 'lri'
                is_center_cell = abs(lri_val - lri) < 0.01
                if is_center_cell and grade and "HARD STOP" in grade.upper():
                    color = '#ff6961' # Pastel Red for Hard Stop

                return {
                    'fillColor': color,
                    'color': 'white',
                    'weight': 0.3,
                    'fillOpacity': 0.5
                }
            
            folium.GeoJson(
                geojson_data,
                style_function=style_function,
                name='LRI Risk Map'
            ).add_to(m)
            print(f"[LRI DEBUG] Choropleth added with {len(features)} cells, LRI={lri}")
        except Exception as e:
            print(f"[LRI ERROR] Failed to add choropleth: {e}")

        # Add legend using Folium's colormap
        try:
            from branca.element import Template, MacroElement
            
            legend_template = '''
            {% macro html(this, kwargs) %}
            <div style="
                position: fixed; 
                bottom: 50px; 
                right: 50px; 
                width: 200px; 
                background-color: white; 
                border: 2px solid grey; 
                z-index: 9999; 
                font-size: 14px;
                padding: 15px;
                border-radius: 5px;
                box-shadow: 0 0 10px rgba(0,0,0,0.2);
            ">
                <p style="margin: 0 0 10px 0; font-weight: bold; font-size: 16px; text-align: center;">LRI Risk Scale</p>
                <div style="margin: 8px 0; display: flex; align-items: center;">
                    <span style="background-color: #77dd77; width: 30px; height: 20px; display: inline-block; border-radius: 3px; margin-right: 10px;"></span>
                    <span style="flex: 1;">Very Good</span>
                </div>
                <div style="margin: 8px 0; display: flex; align-items: center;">
                    <span style="background-color: #aec6cf; width: 30px; height: 20px; display: inline-block; border-radius: 3px; margin-right: 10px;"></span>
                    <span style="flex: 1;">Warning</span>
                </div>
                <div style="margin: 8px 0; display: flex; align-items: center;">
                    <span style="background-color: #fdfd96; width: 30px; height: 20px; display: inline-block; border-radius: 3px; margin-right: 10px;"></span>
                    <span style="flex: 1;">Severe</span>
                </div>
                <div style="margin: 8px 0; display: flex; align-items: center;">
                    <span style="background-color: #ff6961; width: 30px; height: 20px; display: inline-block; border-radius: 3px; margin-right: 10px;"></span>
                    <span style="flex: 1;">Hard Stop</span>
                </div>
            </div>
            {% endmacro %}
            '''
            
            legend_macro = MacroElement()
            legend_macro._template = Template(legend_template)
            m.get_root().add_child(legend_macro)
            print("[LRI DEBUG] Legend added successfully")
        except Exception as e:
            print(f"[LRI ERROR] Failed to add legend: {e}")

    # Add click handler and custom pin mode button via JavaScript
    try:
        click_script = '''
        <script>
        var map = null;
        
        function findMap() {
            var keys = Object.keys(window).filter(k => window[k] && window[k]._leaflet_id);
            return keys.length > 0 ? window[keys[0]] : null;
        }
        
        function setupMapInteractions() {
            map = findMap();
            if (map) {
                // --- Custom Pin Mode Control ---
                var isPinModeActive = false;
                var PinControl = L.Control.extend({
                    onAdd: function(map) {
                        var container = L.DomUtil.create('div', 'leaflet-bar leaflet-control');
                        var button = L.DomUtil.create('a', 'leaflet-control-button', container);
                        button.innerHTML = '📌'; // Pin emoji
                        button.href = '#';
                        button.role = 'button';
                        button.title = 'Toggle Pin Mode';
                        
                        L.DomEvent.on(button, 'click', L.DomEvent.stop).on(button, 'click', function() {
                            isPinModeActive = !isPinModeActive;
                            if (isPinModeActive) {
                                L.DomUtil.addClass(button, 'active');
                                map.dragging.disable();
                                map.getContainer().style.cursor = 'crosshair';
                            } else {
                                L.DomUtil.removeClass(button, 'active');
                                map.dragging.enable();
                                map.getContainer().style.cursor = '';
                            }
                        });
                        
                        return container;
                    }
                });
                map.addControl(new PinControl({ position: 'topleft' }));

                // --- Map Click Handler ---
                map.on('click', function(e) {
                    // Always send click coordinates to parent for analysis
                    parent.postMessage({
                        type: 'MAP_CLICK',
                        lat: e.latlng.lat,
                        lon: e.latlng.lng
                    }, '*');
                });

                // --- Notify Parent that Map is Ready ---
                parent.postMessage({ type: 'MAP_READY' }, '*');
            } else {
                setTimeout(setupMapInteractions, 100);
            }
        }
        
        // --- Custom CSS for the button ---
        var style = document.createElement('style');
        style.innerHTML = `
            .leaflet-control-button {
                font-size: 1.4em;
                line-height: 28px;
                text-align: center;
                width: 30px;
                height: 30px;
                background-color: #fff;
                border-radius: 4px;
            }
            .leaflet-control-button.active {
                background-color: #d4edff;
            }
        `;
        document.head.appendChild(style);

        // --- Initializer ---
        if (document.readyState === 'complete') {
            setupMapInteractions();
        } else {
            window.addEventListener('load', setupMapInteractions);
        }
        </script>
        '''
        m.get_root().html.add_child(folium.Element(click_script))
        print("[LRI DEBUG] Click handler and Pin Mode script added successfully")
    except Exception as e:
        print(f"[LRI ERROR] Failed to add custom map script: {e}")

    try:
        return HTMLResponse(content=m._repr_html_())
    except Exception as e:
        print(f"[LRI ERROR] Failed to generate HTML response: {e}")
        raise HTTPException(status_code=500, detail=f"Map generation failed: {str(e)}")

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