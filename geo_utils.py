"""
Shared geographic utilities.

WHY THIS FILE EXISTS:
The original submission hardcoded "Moscow, Russia" as the *only* distant
location ever used for the impossible-travel pattern, and never computed an
actual distance/time (geo-velocity) anywhere in the pipeline. That means the
model wasn't detecting "impossible travel" behaviourally -- it was just
memorizing one unique string that happened to correlate perfectly with one
label. This module gives every synthetic location a REAL latitude/longitude,
so profiler.py can compute a genuine geo-velocity (km/h) feature -- which is
also the exact example given in the hackathon's own deliverables list
("flagged due to geo-velocity + new device fingerprint").
"""
import math
import random
from typing import Optional, Tuple

# A small curated set of real cities spanning every continent, so that
# "home" cities and injected "impossible travel" destinations can be
# guaranteed to be either close (normal, believable daily jitter) or very
# far apart (thousands of km -- genuinely impossible to cover in minutes).
CITY_COORDS = {
    "New York, USA": (40.7128, -74.0060),
    "Los Angeles, USA": (34.0522, -118.2437),
    "Chicago, USA": (41.8781, -87.6298),
    "Toronto, Canada": (43.6532, -79.3832),
    "Mexico City, Mexico": (19.4326, -99.1332),
    "Sao Paulo, Brazil": (-23.5505, -46.6333),
    "Buenos Aires, Argentina": (-34.6037, -58.3816),
    "Bogota, Colombia": (4.7110, -74.0721),
    "London, UK": (51.5074, -0.1278),
    "Paris, France": (48.8566, 2.3522),
    "Berlin, Germany": (52.5200, 13.4050),
    "Madrid, Spain": (40.4168, -3.7038),
    "Rome, Italy": (41.9028, 12.4964),
    "Moscow, Russia": (55.7558, 37.6173),
    "Amsterdam, Netherlands": (52.3676, 4.9041),
    "Cairo, Egypt": (30.0444, 31.2357),
    "Lagos, Nigeria": (6.5244, 3.3792),
    "Nairobi, Kenya": (-1.2921, 36.8219),
    "Johannesburg, South Africa": (-26.2041, 28.0473),
    "Tokyo, Japan": (35.6762, 139.6503),
    "Mumbai, India": (19.0760, 72.8777),
    "Singapore, Singapore": (1.3521, 103.8198),
    "Beijing, China": (39.9042, 116.4074),
    "Seoul, South Korea": (37.5665, 126.9780),
    "Dubai, UAE": (25.2048, 55.2708),
    "Bangkok, Thailand": (13.7563, 100.5018),
    "Jakarta, Indonesia": (-6.2088, 106.8456),
    "Sydney, Australia": (-33.8688, 151.2093),
    "Melbourne, Australia": (-37.8136, 144.9631),
    "Auckland, New Zealand": (-36.8485, 174.7633),
}

CITY_NAMES = list(CITY_COORDS.keys())


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two (lat, lon) points, in kilometers."""
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def haversine_km_vec(lat1, lon1, lat2, lon2):
    """Vectorized haversine over numpy arrays. Propagates NaN (does not
    crash) when a coordinate is missing/unresolvable, and clips the
    intermediate term to [0, 1] to avoid a rare arcsin domain error from
    floating-point rounding on near-identical or near-antipodal points."""
    import numpy as np
    R = 6371.0
    lat1r, lon1r, lat2r, lon2r = (np.radians(lat1), np.radians(lon1),
                                   np.radians(lat2), np.radians(lon2))
    dlat = lat2r - lat1r
    dlon = lon2r - lon1r
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2) ** 2
    a = np.clip(a, 0.0, 1.0)
    return 2 * R * np.arcsin(np.sqrt(a))


def pick_far_city(home_city: str, min_km: float = 2000.0) -> str:
    """Pick a random city guaranteed to be at least `min_km` from home_city --
    used to inject a genuinely-impossible-to-reach location rather than
    always hardcoding the same single destination."""
    home_lat, home_lon = CITY_COORDS[home_city]
    candidates = [
        c for c in CITY_NAMES
        if c != home_city and haversine_km(home_lat, home_lon, *CITY_COORDS[c]) >= min_km
    ]
    return random.choice(candidates) if candidates else random.choice(CITY_NAMES)


def lookup_coords(geo_location: str) -> Optional[Tuple[float, float]]:
    """Returns (lat, lon) for a generated geo_location string, or None if it's
    not one of our known cities (e.g. the deliberately-anonymized
    'Unknown, Unknown' used for credential-stuffing attacker traffic)."""
    return CITY_COORDS.get(geo_location)