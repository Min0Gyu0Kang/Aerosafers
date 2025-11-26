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

def _generate_data_from_coords(lat: float, lon: float, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generates mock risk data based on latitude, longitude, and aircraft parameters.
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

    # Example of using the new params: rotary wings might be more sensitive to terrain
    if params.get("wing_type") == "rotary":
        alpha_terrain *= 1.2

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
    Accepts JSON: { "lat": <float>, "lon": <float>, "uam_type": <str>, "wing_type": <str> }
    Calculates LRI based on coordinate-dependent data.
    """
    lat = payload.get("lat")
    lon = payload.get("lon")
    params = {
        "uam_type": payload.get("uam_type", "eVTOL"),
        "wing_type": payload.get("wing_type", "rotary")
    }

    if lat is None or lon is None:
        raise HTTPException(status_code=400, detail="Latitude (lat) and Longitude (lon) are required.")

    # Generate data based on coordinates and new parameters
    dynamic_data = _generate_data_from_coords(lat, lon, params)

    result = calculate_lri(dynamic_data)

    result['location'] = f"Lat: {lat:.4f}, Lon: {lon:.4f}"
    result['Evidence'] = {
        "KASS 정보 (Navigation N)": {
            "HPL": f"{dynamic_data['HPL']:.1f}",
            "VPL": f"{dynamic_data['VPL']:.1f}",
            "N_score": f"{result['N_score']:.2f}"
        },
        "아리랑 정보 (Terrain T)": {
            "alpha_terrain": f"{dynamic_data['alpha_terrain']:.2f}",
            "r_och_neg": f"{dynamic_data['r_och_neg']:.2f}",
            "T_score": f"{result['T_score']:.2f}"
        },
        "천리안 정보 (Weather W)": {
            "alpha_cloud": f"{dynamic_data['alpha_cloud']:.2f}",
            "actual_visibility": f"{dynamic_data['actual_visibility']:.1f}",
            "required_visibility": f"{dynamic_data.get('required_visibility', 30)}",
            "CTBT": f"{dynamic_data['CTBT']}",
            "W_score": f"{result['W_score']:.2f}"
        }
    }

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
        attr="Tiles © Esri — Source: Esri, Maxar, Earthstar Geographics, and the GIS User Community",
        control_scale=True
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

    # Server-side marker is removed. All marker logic is now handled by the client-side script below.

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

            # Recompute center LRI/Grade on backend to avoid trusting client-supplied values
            try:
                center_params = {"uam_type": "eVTOL", "wing_type": "rotary"}
                center_data = _generate_data_from_coords(lat, lon, center_params)
                center_result = calculate_lri(center_data)
                center_lri = center_result.get("LRI", lri)
                center_grade = center_result.get("Grade", (grade or "").upper())
                print(f"[LRI DEBUG] Center recomputed LRI={center_lri}, Grade={center_grade}")
            except Exception as ex:
                # if recompute fails, fall back to provided values but warn
                print(f"[LRI WARN] Failed to recompute center LRI: {ex}")
                center_lri = lri
                center_grade = (grade or "").upper()

            # Only search for a safer spot when the center is not Very Good
            try:
                if center_grade and "VERY GOOD" not in center_grade.upper():
                    print(f"[LRI DEBUG] Searching for a safer spot near ({lat}, {lon}) based on recomputed center grade {center_grade}")

                    # STEP 0: selection operates on the generated geojson 'features'
                    very_good_cells = []
                    warning_cells = []

                    try:
                        for feat in features:
                            cell_lri = feat['properties'].get('lri', 0)
                            # skip exact center cell
                            is_center = abs(cell_lri - lri) < 0.0001
                            if is_center:
                                continue

                            # compute cell center from polygon coords: [ [ [lon,lat], ... ] ]
                            try:
                                poly = feat['geometry']['coordinates'][0]
                                lons = [p[0] for p in poly[:-1]]
                                lats = [p[1] for p in poly[:-1]]
                                cell_center_lon = sum(lons) / len(lons)
                                cell_center_lat = sum(lats) / len(lats)
                            except Exception as exc:
                                print(f"[LRI WARN] Failed to compute cell center: {exc}")
                                continue

                            if cell_lri >= TAU_YELLOW:  # >= 80 -> Very Good
                                very_good_cells.append((cell_center_lat, cell_center_lon, cell_lri))
                            elif cell_lri < TAU_YELLOW:  # < 80 -> Warning/Severe
                                warning_cells.append((cell_center_lat, cell_center_lon, cell_lri))
                    except Exception as ex_features:
                        print(f"[LRI WARN] Error iterating features for safer-spot selection: {ex_features}")

                    # STEP 1: choose a candidate (prefer very_good_cells)
                    best_spot = None
                    try:
                        import random as _random
                        if very_good_cells:
                            best_spot = _random.choice(very_good_cells)
                        elif warning_cells:
                            best_spot = _random.choice(warning_cells)
                        # normalize to dict if found
                        if best_spot:
                            best_spot = {"lat": best_spot[0], "lon": best_spot[1], "lri": best_spot[2]}
                            print(f"[LRI DEBUG] Candidate safer spot chosen at ({best_spot['lat']:.5f}, {best_spot['lon']:.5f}) with LRI {best_spot['lri']:.2f}")
                        else:
                            print("[LRI DEBUG] No candidate safer cells found in grid.")
                    except Exception as ex_choice:
                        print(f"[LRI WARN] Failed to pick safer spot: {ex_choice}")
                        best_spot = None

                    # STEP 2 / 3: create polyline arrow and NOTAM modal if best_spot exists
                    if best_spot:
                        alt_dot_created = False
                        arrow_created = False
                        notam_created = False

                        try:
                            # emphasize alternative dot (darker green for very good, darker blue for warning)
                            if best_spot['lri'] >= TAU_YELLOW:
                                alt_dot_color = '#2e8b57'  # darker green
                            else:
                                alt_dot_color = '#4b7380'  # darker blue

                            folium.CircleMarker(
                                location=[best_spot['lat'], best_spot['lon']],
                                radius=6,
                                color=alt_dot_color,
                                fill=True,
                                fill_color=alt_dot_color,
                                fill_opacity=0.95,
                                tooltip=f"Alternative: LRI {best_spot['lri']:.2f}"
                            ).add_to(m)
                            alt_dot_created = True
                            print(f"[LRI DEBUG] Alt dot created with color {alt_dot_color}")
                        except Exception as ex_dot:
                            print(f"[LRI WARN] Alt dot creation failed: {ex_dot}")

                        try:
                            line_color = alt_dot_color if 'alt_dot_color' in locals() and alt_dot_color else '#4b7380'
                            line = folium.PolyLine(
                                locations=[[lat, lon], [best_spot['lat'], best_spot['lon']]],
                                color=line_color,
                                weight=3,
                                opacity=0.9
                            ).add_to(m)
                            arrow_created = True
                            print("[LRI DEBUG] Direction line drawn")

                            # try adding an arrow glyph (best-effort)
                            try:
                                from folium.plugins import PolyLineTextPath
                                PolyLineTextPath(line, ' ▶ ', repeat=False, offset=7,
                                                 attributes={'fill': line_color, 'font-weight': 'bold', 'font-size': '18'}).add_to(m)
                                print("[LRI DEBUG] Arrow glyph added along line")
                            except Exception as ex_glyph:
                                print(f"[LRI WARN] Arrow glyph not available: {ex_glyph}")
                        except Exception as ex_line:
                            print(f"[LRI WARN] Failed to draw direction line: {ex_line}")

                        try:
                            mid_lat = (lat + best_spot['lat']) / 2.0
                            mid_lon = (lon + best_spot['lon']) / 2.0
                            notam_html = (
                                '<div style="'
                                'background: rgba(255,255,255,0.95);'
                                'border: 2px solid #ff6961;'
                                'color: #000;'
                                'padding: 8px 12px;'
                                'border-radius: 6px;'
                                'font-weight: 700;'
                                'box-shadow: 0 3px 8px rgba(0,0,0,0.25);'
                                'max-width: 260px; text-align: center;">'
                                'NOTAM: 해당 장소는 착륙하기 매우 어려울 것으로 예상됩니다!<br>우회 착륙지를 권장드립니다.'
                                '</div>'
                            )
                            folium.map.Marker(
                                [mid_lat, mid_lon],
                                icon=folium.DivIcon(html=notam_html, icon_size=(250,60), class_name='notam-divicon')
                            ).add_to(m)
                            notam_created = True
                            print("[LRI DEBUG] NOTAM DivIcon added")
                        except Exception as ex_notam:
                            print(f"[LRI WARN] Failed to add NOTAM DivIcon: {ex_notam}")

                        # final logs
                        if not alt_dot_created:
                            print("[LRI WARN] Alternative dot was NOT created; check earlier errors.")
                        if not arrow_created:
                            print("[LRI WARN] Direction arrow was NOT created; check earlier errors.")
                        if not notam_created:
                            print("[LRI WARN] NOTAM modal/popup was NOT created; check earlier errors.")
                    else:
                        print("[LRI DEBUG] No safer spot available after grid evaluation.")
                else:
                    print(f"[LRI DEBUG] Center grade is VERY GOOD ({center_grade}) — skipping safer-spot selection.")
            except Exception as e:
                print(f"[LRI ERROR] Failed during safer-spot selection: {e}")

            # End of map-generation try-block: ensure any uncaught exceptions are handled and a response is returned
        except Exception as e:
            print(f"[LRI ERROR] Unhandled error in get_map: {e}")
            try:
                return HTMLResponse(content=f"<html><body><h3>Map generation failed: {e}</h3></body></html>")
            except Exception:
                return HTMLResponse(content="<html><body><h3>Map generation failed.</h3></body></html>")

    # If lat/lon/lri not provided, or after successful map generation, return the map HTML
    try:
        return HTMLResponse(content=m._repr_html_())
    except Exception as e:
        print(f"[LRI ERROR] Failed to render map HTML: {e}")
        return HTMLResponse(content="<html><body><h3>Map rendering failed.</h3></body></html>")
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