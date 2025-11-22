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
W_V, W_N, W_S = 0.45, 0.35, 0.20  # 가중치 (w_V, w_N, w_S)
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
        # 1. 가시성 V 계산 (V = 100 * min(1, p/p_req) * (1 - α_cloud))
        p = data.get('actual_visibility', 50)
        p_req = data.get('required_visibility', 30)
        alpha_cloud = data.get('alpha_cloud', 0.1)
        V = 100 * min(1, p / p_req) * (1 - alpha_cloud)

        # 2. 항법무결성 N 계산 (N = 100 - [50 * max(...)])
        HPL = data.get('HPL', 35)
        VPL = data.get('VPL', 45)
        N = 100 - (50 * max(0, (HPL - AL_H) / 20) + 50 * max(0, (VPL - AL_V) / 20))
        
        # 3. 표면·지형 S 계산 (S = 100 * w * (1 - α_terrain) - 40 * r_{OCH<0})
        alpha_terrain = data.get('alpha_terrain', 0.05)
        r_och_neg = data.get('r_och_neg', 0.0)
        S = 100 * W_S * (1 - alpha_terrain) - 40 * r_och_neg

        # 4. 최종 위험도 LRI 계산 (LRI = 100 / (w_V/V + w_N/N + w_S/S'))
        S_norm = S / W_S if W_S != 0 else 0
        S_safe = max(5, S_norm) 
        LRI = 100 / (W_V / V + W_N / N + W_S / S_safe)
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
            "V_score": round(V, 2),
            "N_score": round(N, 2),
            "S_score": round(S, 2),
            "HardStop": is_hard_stop
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def _generate_scenario(name: str) -> Dict[str, Any]:
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
    Accepts JSON: { "lat": <float>, "lon": <float>, "scenario": "<name>" }
    scenario is optional; defaults to "random".
    """
    lat = payload.get("lat")
    lon = payload.get("lon")
    scenario = (payload.get("scenario") or "random").lower()
    # build mock_data from scenario
    mock_data = _generate_scenario(scenario)

    result = calculate_lri(mock_data)

    result['location'] = f"Lat: {lat:.4f}, Lon: {lon:.4f}" if (lat is not None and lon is not None) else "unknown"
    result['Evidence'] = [
        f"가시성({result['V_score']}): 구름 감쇠계수({mock_data['alpha_cloud']:.2f})로 인해 {round(100*mock_data['actual_visibility']/mock_data['required_visibility'], 1)}% 충족에서 감점.",
        f"항법({result['N_score']}): HPL={mock_data['HPL']:.1f}m (AL_H={AL_H}m)으로 항법 무결성 {('유지' if result['N_score'] >= 90 else '저하')} 상태.",
        f"지형({result['S_score']}): 장애물 비율({mock_data['r_och_neg']*100:.1f}%) 반영. 표면 습윤/결빙 위험 Δσ⁰={mock_data['delta_sigma_0']:.1f}dB."
    ]

    return result

@app.get("/api/map", response_class=HTMLResponse)
async def get_map():
    """
    Generates a Leaflet map using GeoPandas and Folium.
    """
    # Example GeoDataFrame (replace with your actual GeoPandas data)
    gdf = gpd.GeoDataFrame(
        {
            "name": ["Point A", "Point B"],
            "geometry": gpd.points_from_xy([127.0, 127.1], [37.5, 37.6]),
        }
    )

    # Create a Folium map centered on the first point
    m = folium.Map(location=[37.5, 127.0], zoom_start=12)

    # Add GeoPandas data to the map
    for _, row in gdf.iterrows():
        folium.Marker(
            location=[row.geometry.y, row.geometry.x],
            popup=row["name"],
        ).add_to(m)

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