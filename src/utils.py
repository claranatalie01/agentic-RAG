import aiohttp
import math
from datetime import datetime
from typing import Optional, Dict, Any

async def fetch_all_libraries() -> list:
    url = "https://sls.hkpl.gov.hk/api/cfm-admin-service/open-api/library/selectLibraryPageInfoForPSI"
    params = {"language": "en-US", "sizePerPage": "9999"}
    async with aiohttp.ClientSession() as session:
        async with session.get(url, params=params) as resp:
            if resp.status == 200:
                data = await resp.json()
                return data if isinstance(data, list) else []
            return []

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1-a))

COORD_MAP = {
    "HKCL": (22.284, 114.150),
    "STPL": (22.381, 114.188),
    "TSTPL": (22.296, 114.172),
}

async def resolve_current_library(lat: float, lon: float) -> Optional[Dict[str, Any]]:
    branches = await fetch_all_libraries()
    if not branches:
        return None
    nearest = None
    min_dist = float('inf')
    for branch in branches:
        code = branch.get("libraryCode")
        if code in COORD_MAP:
            lib_lat, lib_lon = COORD_MAP[code]
            dist = haversine(lat, lon, lib_lat, lib_lon)
            if dist < min_dist:
                min_dist = dist
                nearest = {
                    "code": code,
                    "name": branch.get("libraryDisplayName", code),
                    "distance": dist
                }
    return nearest

def get_current_datetime():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")