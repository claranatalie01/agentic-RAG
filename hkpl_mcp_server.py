#!/usr/bin/env python3
"""
MCP Server for HKPL Library Assistant
Provides tools for:
- Live workstation availability
- Library branch details (address, phone, hours)
- Library catalog search (books)
- Find nearby libraries by district or coordinates
- Book availability (simplified)
- Library count & statistics
"""

import aiohttp
import math
from typing import Optional, List, Dict, Any
from mcp.server.fastmcp import FastMCP

# Create MCP server instance
mcp = FastMCP("HKPL Library Tools")

# Base URL for HKPL public APIs (from DATA.GOV.HK)
HKPL_API_BASE = "https://sls.hkpl.gov.hk/api/cfm-admin-service/open-api/library/selectLibraryPageInfoForPSI"

# ----------------------------------------------------------------------
# Helper functions (not exposed as tools)
# ----------------------------------------------------------------------

async def fetch_all_libraries() -> List[Dict[str, Any]]:
    """Fetch all library branches from HKPL API."""
    params = {"language": "en-US", "sizePerPage": "9999"}
    async with aiohttp.ClientSession() as session:
        async with session.get(HKPL_API_BASE, params=params) as resp:
            if resp.status == 200:
                data = await resp.json()
                # API returns a list of library objects
                return data if isinstance(data, list) else []
            else:
                return []

def compute_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Haversine distance between two coordinates (in km)."""
    R = 6371
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda/2)**2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1-a))

# ----------------------------------------------------------------------
# Exposed Tools
# ----------------------------------------------------------------------

@mcp.tool()
async def get_live_workstation_availability(library_code: Optional[str] = None) -> str:
    """
    Get real‑time workstation availability.
    Args:
        library_code: Optional branch code (e.g., 'HKCL', 'STPL'). If omitted, returns summary for all branches.
    """
    params = {"language": "en-US", "sizePerPage": "9999"}
    if library_code:
        params["libraryCode"] = library_code

    async with aiohttp.ClientSession() as session:
        async with session.get(HKPL_API_BASE, params=params) as resp:
            if resp.status != 200:
                return f"Unable to fetch availability (HTTP {resp.status})."
            data = await resp.json()
            if not isinstance(data, list):
                return "Unexpected data format."

            if library_code:
                # Expect a single library object
                if not data:
                    return f"No library found with code '{library_code}'."
                lib = data[0]
                total_available = 0
                for sess in lib.get("sessionList", []):
                    for grp in sess.get("workstationGroup", []):
                        total_available += grp.get("availableWktNumber", 0)
                return f"{lib.get('libraryDisplayName', library_code)} has {total_available} available workstations."
            else:
                # Summary for all branches
                total_branches = len(data)
                total_workstations = 0
                for lib in data:
                    for sess in lib.get("sessionList", []):
                        for grp in sess.get("workstationGroup", []):
                            total_workstations += grp.get("availableWktNumber", 0)
                return f"Live HKPL data: {total_branches} branches open. Total workstations available: {total_workstations}."


@mcp.tool()
async def get_library_details(library_code: str) -> str:
    """
    Get detailed information about a specific library branch.
    Args:
        library_code: Branch code (e.g., 'HKCL' for Hong Kong Central Library, 'STPL' for Shatin).
    """
    params = {"libraryCode": library_code, "language": "en-US", "sizePerPage": "1"}
    async with aiohttp.ClientSession() as session:
        async with session.get(HKPL_API_BASE, params=params) as resp:
            if resp.status != 200:
                return f"Could not retrieve details for {library_code} (HTTP {resp.status})."
            data = await resp.json()
            if not data:
                return f"No library found with code '{library_code}'."
            lib = data[0]
            name = lib.get("libraryDisplayName", "Unknown")
            address = lib.get("address", "Address not available")
            phone = lib.get("telephone", "Phone not available")
            email = lib.get("email", "Email not available")
            is_open = "Open" if lib.get("isOpen") else "Closed"
            # compute hours from sessionList
            sessions = lib.get("sessionList", [])
            hours = "; ".join([f"{s['sessionStart']} – {s['sessionEnd']}" for s in sessions]) if sessions else "Hours not specified"
            return (f"{name} (Code: {library_code})\n"
                    f"Address: {address}\n"
                    f"Phone: {phone}\n"
                    f"Email: {email}\n"
                    f"Status: {is_open}\n"
                    f"Hours: {hours}")


@mcp.tool()
async def search_library_catalog(query: str, limit: int = 5) -> str:
    """
    Search for books in the library catalog (using OpenLibrary API).
    Args:
        query: Book title, author, or keyword.
        limit: Maximum number of results to return (default 5).
    """
    url = "https://openlibrary.org/search.json"
    params = {"q": query, "limit": limit}
    async with aiohttp.ClientSession() as session:
        async with session.get(url, params=params) as resp:
            if resp.status != 200:
                return f"Catalog search failed (HTTP {resp.status})."
            data = await resp.json()
            docs = data.get("docs", [])
            if not docs:
                return f"No results found for '{query}'."
            results = []
            for i, doc in enumerate(docs[:limit], 1):
                title = doc.get("title", "Unknown title")
                authors = ", ".join(doc.get("author_name", ["Unknown author"]))
                year = doc.get("first_publish_year", "N/A")
                results.append(f"{i}. {title} by {authors} ({year})")
            return "\n".join(results)


@mcp.tool()
async def find_nearby_libraries(district: Optional[str] = None, lat: Optional[float] = None, lon: Optional[float] = None, radius_km: float = 5.0) -> str:
    """
    Find libraries by district or proximity to coordinates.
    Args:
        district: Hong Kong district name (e.g., 'Sha Tin', 'Central and Western').
        lat: Latitude (if using coordinate search).
        lon: Longitude (if using coordinate search).
        radius_km: Search radius in kilometers (default 5.0).
    """
    libraries = await fetch_all_libraries()
    if not libraries:
        return "Unable to retrieve library data."

    if district:
        # Filter by district (case‑insensitive substring match)
        district_lower = district.lower()
        matches = [lib for lib in libraries if district_lower in lib.get("district", "").lower()]
        if not matches:
            return f"No libraries found in district '{district}'."
        result_lines = [f"Libraries in {district}:"]
        for lib in matches:
            name = lib.get("libraryDisplayName", "Unknown")
            addr = lib.get("address", "Address not available")
            result_lines.append(f"- {name}: {addr}")
        return "\n".join(result_lines)

    elif lat is not None and lon is not None:
        # Filter by distance
        # We need coordinates; the API does not provide lat/lon directly, so we use a mapping or fallback.
        # For demonstration, we use a simplified approach: assume we have a pre‑loaded mapping of branch codes to coordinates.
        # Here we use a small static mapping (you can extend).
        coord_map = {
            "HKCL": (22.284, 114.150),   # Hong Kong Central Library
            "STPL": (22.381, 114.188),   # Shatin Public Library (approximate)
            "TSTPL": (22.296, 114.172),  # Tsim Sha Tsui
            # Add more as needed
        }
        nearby = []
        for lib in libraries:
            code = lib.get("libraryCode")
            if code in coord_map:
                lib_lat, lib_lon = coord_map[code]
                dist = compute_distance(lat, lon, lib_lat, lib_lon)
                if dist <= radius_km:
                    nearby.append((dist, lib))
        nearby.sort(key=lambda x: x[0])
        if not nearby:
            return f"No libraries found within {radius_km} km of ({lat}, {lon})."
        result_lines = [f"Libraries within {radius_km} km:"]
        for dist, lib in nearby:
            name = lib.get("libraryDisplayName", "Unknown")
            result_lines.append(f"- {name} ({dist:.2f} km)")
        return "\n".join(result_lines)
    else:
        return "Please provide either a district name or coordinates (lat, lon)."


@mcp.tool()
async def check_book_availability(title: str, author: Optional[str] = None) -> str:
    """
    Simplified book availability check (simulated – in production would call HKPL catalog API).
    Args:
        title: Book title.
        author: Optional author name.
    """
    # For now, we simulate by searching OpenLibrary and then assuming the book is available at HKPL.
    # In a real implementation, you would call the actual HKPL catalog API.
    url = "https://openlibrary.org/search.json"
    params = {"title": title}
    if author:
        params["author"] = author
    params["limit"] = 1
    async with aiohttp.ClientSession() as session:
        async with session.get(url, params=params) as resp:
            if resp.status != 200:
                return "Unable to check availability."
            data = await resp.json()
            docs = data.get("docs", [])
            if not docs:
                return f"No record found for '{title}'."
            # Simulate that the book is available at most HKPL branches
            return f"The book '{title}' is likely available at HKPL branches. Please visit the catalog or ask a librarian for exact location."


@mcp.tool()
async def get_library_opening_hours(library_code: str) -> str:
    """
    Get opening hours for a specific library branch.
    Args:
        library_code: Branch code (e.g., 'HKCL').
    """
    params = {"libraryCode": library_code, "language": "en-US", "sizePerPage": "1"}
    async with aiohttp.ClientSession() as session:
        async with session.get(HKPL_API_BASE, params=params) as resp:
            if resp.status != 200:
                return f"Could not retrieve hours for {library_code}."
            data = await resp.json()
            if not data:
                return f"No library found with code '{library_code}'."
            lib = data[0]
            sessions = lib.get("sessionList", [])
            if not sessions:
                return f"No session information for {lib.get('libraryDisplayName', library_code)}."
            hours = "\n".join([f"{s['sessionStart'][:16]} – {s['sessionEnd'][:16]}" for s in sessions])
            return f"Opening hours for {lib.get('libraryDisplayName', library_code)}:\n{hours}"


@mcp.tool()
async def get_library_count() -> str:
    """
    Get the total number of HKPL libraries (static, mobile, self‑service).
    """
    libraries = await fetch_all_libraries()
    if not libraries:
        return "Unable to retrieve library count."
    total = len(libraries)
    # Count mobile libraries (if data indicates type; otherwise fallback)
    mobile = sum(1 for lib in libraries if "mobile" in lib.get("libraryDisplayName", "").lower())
    static = total - mobile
    return f"HKPL has {total} library branches ({static} static, {mobile} mobile)."


# ----------------------------------------------------------------------
# Run the server
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Use streamable HTTP transport for easy integration
    mcp.run(transport="streamable-http")