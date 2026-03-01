"""
Kinexica — Logistics Optimization Model v2.0
════════════════════════════════════════════
Physics-grounded logistics engine that integrates with PINN shelf-life
predictions to compute:

  1. URGENCY SCORE         — hours-remaining × spoilage-rate → dispatch priority
  2. HAVERSINE ROUTING     — real geodesic distance (km) between nodes
  3. ETA COMPUTATION       — traffic-aware ETA with road-factor correction
  4. COST MODEL            — fuel + driver + cold-chain + carbon offset per leg
  5. MULTI-TIER DISPATCH   — Tier-1 Emergency / Tier-2 Distress / Tier-3 Markdown
  6. TIME-DECAY PRICING    — non-linear markdown curve based on PIDR
  7. ROUTE OPTIMISATION    — nearest-warehouse greedy TSP for multi-stop
  8. CARBON FOOTPRINT      — CO₂e per km per vehicle class
  9. API-READY outputs     — structured dicts ready for FastAPI endpoints

Authors: Kinexica R&D Team
"""
# pylint: disable=invalid-name, too-many-arguments

import math
import time
from dataclasses import dataclass, field
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

EARTH_RADIUS_KM = 6371.0

# kg CO₂e per km per vehicle class (DEFRA 2023)
EMISSION_FACTOR = {
    "motorcycle":  0.0113,
    "van_small":   0.1605,
    "van_large":   0.2110,
    "truck_7t":    0.4020,
    "truck_14t":   0.5500,
    "truck_26t":   0.6850,
    "refrigerated": 0.7200,   # reefer truck — higher due to refrigeration unit
    "electric_van": 0.0410,   # grid average India 0.82 kgCO₂/kWh × 0.05 kWh/km
}

# Average speed km/h by road class + time of day
ROAD_SPEED = {
    "highway":  80,
    "city":     28,
    "rural":    45,
    "lastmile": 18,
}

# Fuel cost ₹/km by vehicle class (India 2024 average)
FUEL_COST_INR_PER_KM = {
    "motorcycle":   2.5,
    "van_small":    7.0,
    "van_large":    9.5,
    "truck_7t":    14.0,
    "truck_14t":   18.0,
    "truck_26t":   23.0,
    "refrigerated": 28.0,
    "electric_van":  3.2,
}

# Time-decay pricing tiers (remaining shelf hours → markdown %)
MARKDOWN_TIERS = [
    (72,  0.00),   # > 72h  → no markdown
    (48,  0.10),   # 48-72h → 10% off
    (36,  0.20),   # 36-48h → 20% off
    (24,  0.35),   # 24-36h → 35% off
    (18,  0.50),   # 18-24h → 50% off
    (12,  0.65),   # 12-18h → 65% off
    (6,   0.80),   # 6-12h  → 80% off
    (0,   0.92),   # < 6h   → 92% off (biotech divert)
]

# Dispatch tiers
DISPATCH_TIER = {
    "TIER_1_EMERGENCY": {"max_hours": 18,  "sla_minutes": 30,  "priority": 1},
    "TIER_2_DISTRESS":  {"max_hours": 48,  "sla_minutes": 120, "priority": 2},
    "TIER_3_MARKDOWN":  {"max_hours": 72,  "sla_minutes": 240, "priority": 3},
    "TIER_4_STABLE":    {"max_hours": 999, "sla_minutes": 480, "priority": 4},
}


# ─────────────────────────────────────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GeoNode:
    """A supply chain node with geographic coordinates."""
    node_id:   str
    name:      str
    lat:       float
    lon:       float
    node_type: str = "warehouse"   # farm / warehouse / cold_hub / retail / biotech_lab


@dataclass
class LogisticsResult:
    """Complete logistics decision for a batch pickup/delivery."""

    asset_id:            str = ""
    origin:              GeoNode = field(
        default_factory=lambda: GeoNode("", "", 0.0, 0.0))
    destination:         GeoNode = field(
        default_factory=lambda: GeoNode("", "", 0.0, 0.0))

    # Route metrics
    distance_km:         float = 0.0
    eta_minutes:         float = 0.0
    road_class:          str = "city"
    vehicle_class:       str = "van_small"

    # Cost breakdown (INR)
    fuel_cost_inr:       float = 0.0
    driver_cost_inr:     float = 0.0
    cold_chain_cost_inr: float = 0.0
    carbon_offset_inr:   float = 0.0
    total_cost_inr:      float = 0.0
    total_cost_usd:      float = 0.0

    # Cargo metrics
    mass_kg:             float = 0.0
    remaining_shelf_h:   float = 0.0
    pidr:                float = 0.0
    urgency_score:       float = 0.0
    dispatch_tier:       str = "TIER_4_STABLE"

    # Pricing
    base_price_inr:      float = 0.0
    markdown_pct:        float = 0.0
    effective_price_inr: float = 0.0
    recovery_ratio:      float = 0.0   # effective_price / base_price

    # Environmental
    co2e_kg:             float = 0.0

    # Metadata
    recommended_action:  str = ""
    dispatch_sla_min:    int = 60
    timestamp:           float = 0.0

    def to_dict(self) -> dict:
        return {
            "asset_id":            self.asset_id,
            "origin":              self.origin.name,
            "destination":         self.destination.name,
            "distance_km":         round(self.distance_km, 2),
            "eta_minutes":         round(self.eta_minutes, 1),
            "road_class":          self.road_class,
            "vehicle_class":       self.vehicle_class,
            "fuel_cost_inr":       round(self.fuel_cost_inr,       2),
            "driver_cost_inr":     round(self.driver_cost_inr,     2),
            "cold_chain_cost_inr": round(self.cold_chain_cost_inr, 2),
            "carbon_offset_inr":   round(self.carbon_offset_inr,   2),
            "total_cost_inr":      round(self.total_cost_inr,      2),
            "total_cost_usd":      round(self.total_cost_usd,      2),
            "mass_kg":             self.mass_kg,
            "remaining_shelf_h":   round(self.remaining_shelf_h, 2),
            "pidr":                round(self.pidr, 6),
            "urgency_score":       round(self.urgency_score, 4),
            "dispatch_tier":       self.dispatch_tier,
            "base_price_inr":      round(self.base_price_inr,      2),
            "markdown_pct":        round(self.markdown_pct * 100,  2),
            "effective_price_inr": round(self.effective_price_inr, 2),
            "recovery_ratio":      round(self.recovery_ratio,       4),
            "co2e_kg":             round(self.co2e_kg,              4),
            "recommended_action":  self.recommended_action,
            "dispatch_sla_min":    self.dispatch_sla_min,
            "timestamp":           self.timestamp,
        }


# ─────────────────────────────────────────────────────────────────────────────
# CORE MATHEMATICS
# ─────────────────────────────────────────────────────────────────────────────

def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Compute the geodesic (great-circle) distance between two GPS coordinates.
    Returns distance in kilometres.
    """
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (math.sin(dphi / 2) ** 2
         + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2)
    return 2 * EARTH_RADIUS_KM * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def infer_road_class(distance_km: float) -> str:
    """Infer road class from distance — inner city, city, rural, highway."""
    if distance_km < 5:
        return "lastmile"
    if distance_km < 25:
        return "city"
    if distance_km < 80:
        return "rural"
    return "highway"


def road_factor(road_class: str) -> float:
    """
    Convert straight-line (Haversine) distance to actual road distance.
    Detour factor varies by road class.
    """
    factors = {"highway": 1.10, "rural": 1.25, "city": 1.40, "lastmile": 1.60}
    return factors.get(road_class, 1.30)


def compute_eta(road_km: float, road_class: str) -> float:
    """ETA in minutes, accounting for congestion factor."""
    speed = ROAD_SPEED[road_class]
    # Peak-hour congestion penalty (city/lastmile)
    hour = time.localtime().tm_hour
    if road_class in ("city", "lastmile") and 8 <= hour <= 20:
        speed = max(speed * 0.70, 12)   # 30% congestion hit
    return (road_km / speed) * 60.0


def urgency_score(remaining_h: float, pidr: float, mass_kg: float) -> float:
    """
    Composite urgency score (0–1, higher = more urgent).
    Combines shelf-life pressure, decay rate, and economic mass.
    """
    shelf_pressure = max(0.0, 1.0 - remaining_h /
                         200.0)   # normalise over 200h
    pidr_norm = min(pidr * 100, 1.0)                   # PIDR ×100 clamped
    mass_norm = min(mass_kg / 5000.0, 1.0)             # normalise over 5t
    return round(0.50 * shelf_pressure + 0.35 * pidr_norm + 0.15 * mass_norm, 4)


def dispatch_tier_for(remaining_h: float) -> tuple[str, int]:
    """Return (tier_name, SLA_minutes) based on remaining shelf hours."""
    for tier, params in DISPATCH_TIER.items():
        if remaining_h <= params["max_hours"]:
            return tier, params["sla_minutes"]
    return "TIER_4_STABLE", 480


def markdown_for(remaining_h: float) -> float:
    """Return the markdown fraction (0.0–0.92) for the given remaining hours."""
    for threshold, pct in MARKDOWN_TIERS:
        if remaining_h > threshold:
            return pct
    return MARKDOWN_TIERS[-1][1]


def select_vehicle(mass_kg: float, remaining_h: float) -> str:
    """
    Select optimal vehicle class based on cargo mass and urgency.
    Emergency < 12h → motorcycle courier for < 100 kg; else refrigerated truck.
    """
    if remaining_h < 12 and mass_kg < 100:
        return "motorcycle"
    if remaining_h < 24:
        return "refrigerated"
    if mass_kg < 300:
        return "van_small"
    if mass_kg < 1000:
        return "van_large"
    if mass_kg < 7000:
        return "truck_7t"
    if mass_kg < 14000:
        return "truck_14t"
    return "truck_26t"


def cost_model(
    road_km:       float,
    vehicle:       str,
    mass_kg:       float,
    cold_chain:    bool = False,
    usd_to_inr:    float = 84.50,
) -> dict:
    """
    Compute full cost breakdown in INR and USD.

    Returns dict with: fuel, driver, cold_chain, carbon_offset, total_inr, total_usd
    """
    fuel_inr = road_km * FUEL_COST_INR_PER_KM.get(vehicle, 9.5)
    # Driver cost: ₹150/hr @ average speed
    speed = ROAD_SPEED.get(infer_road_class(road_km), 40)
    eta_h = road_km / speed
    driver_inr = eta_h * 150.0

    # Cold-chain surcharge: ₹2.50/km for refrigerated units
    cold_inr = road_km * 2.50 if cold_chain else 0.0

    # Carbon offset at ₹900/tCO₂e (India voluntary market average)
    co2e_kg = road_km * EMISSION_FACTOR.get(vehicle, 0.21)
    offset_inr = (co2e_kg / 1000.0) * 900.0

    total_inr = fuel_inr + driver_inr + cold_inr + offset_inr
    total_usd = total_inr / usd_to_inr

    return {
        "fuel_inr":   round(fuel_inr,   2),
        "driver_inr": round(driver_inr, 2),
        "cold_inr":   round(cold_inr,   2),
        "offset_inr": round(offset_inr, 2),
        "total_inr":  round(total_inr,  2),
        "total_usd":  round(total_usd,  2),
        "co2e_kg":    round(co2e_kg,    4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# RECOMMENDED ACTION ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def recommended_action(
    tier: str,
    remaining_h: float,
    markdown_pct: float,
    destination: GeoNode,
) -> str:
    """Return a clear, actionable dispatch recommendation."""
    if tier == "TIER_1_EMERGENCY":
        return (
            f"🚨 EMERGENCY DISPATCH — SLA 30 min | "
            f"Route to {destination.name} ({destination.node_type}) | "
            f"Markdown: {markdown_pct*100:.0f}% | "
            f"If biotech_lab: divert for fermentation substrate."
        )
    if tier == "TIER_2_DISTRESS":
        return (
            f"🟠 DISTRESS PICKUP — SLA 2 hrs | "
            f"Route to {destination.name} | "
            f"Markdown {markdown_pct*100:.0f}% — Dynamic ESL update required."
        )
    if tier == "TIER_3_MARKDOWN":
        return (
            f"🟡 RETAIL MARKDOWN — {markdown_pct*100:.0f}% discount | "
            f"Update electronic shelf label | No immediate dispatch needed."
        )
    return (
        f"✅ STABLE — Standard cold-chain transit to {destination.name} | "
        f"No markdown required (>{remaining_h:.0f}h remaining)."
    )


# ─────────────────────────────────────────────────────────────────────────────
# MULTI-STOP ROUTE OPTIMISER (Nearest-Neighbour Greedy TSP)
# ─────────────────────────────────────────────────────────────────────────────

def optimise_route(
    origin: GeoNode,
    waypoints: list[GeoNode],
) -> tuple[list[GeoNode], float]:
    """
    Nearest-Neighbour greedy TSP for multi-stop delivery routing.
    Returns (ordered_route, total_road_km).
    """
    unvisited = list(waypoints)
    route = [origin]
    total_km = 0.0
    current = origin

    while unvisited:
        # Find nearest unvisited node
        nearest = min(
            unvisited,
            key=lambda n: haversine(current.lat, current.lon, n.lat, n.lon),
        )
        dist_km = haversine(current.lat, current.lon, nearest.lat, nearest.lon)
        road_cls = infer_road_class(dist_km)
        total_km += dist_km * road_factor(road_cls)
        route.append(nearest)
        unvisited.remove(nearest)
        current = nearest

    return route, total_km


# ─────────────────────────────────────────────────────────────────────────────
# PRIMARY PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def plan_dispatch(
    asset_id:        str,
    origin:          GeoNode,
    destination:     GeoNode,
    mass_kg:         float,
    remaining_shelf_h: float,
    pidr:            float = 0.0,
    base_price_inr:  float = 0.0,
    cold_chain:      bool = True,
    usd_to_inr:      float = 84.50,
) -> LogisticsResult:
    """
    Full logistics plan for a single batch pickup/delivery.

    Parameters
    ----------
    asset_id         : unique batch identifier
    origin           : GeoNode where asset currently is
    destination      : GeoNode for delivery target
    mass_kg          : cargo weight in kg
    remaining_shelf_h: PINN-predicted hours remaining
    pidr             : Physics-Informed Decay Rate from PINN
    base_price_inr   : original market value (INR)
    cold_chain       : whether refrigerated transport is needed
    usd_to_inr       : live exchange rate (default fallback)

    Returns
    -------
    LogisticsResult  — fully populated, serialisable via .to_dict()
    """
    result = LogisticsResult(
        asset_id=asset_id,
        origin=origin,
        destination=destination,
        mass_kg=mass_kg,
        remaining_shelf_h=remaining_shelf_h,
        pidr=pidr,
        base_price_inr=base_price_inr,
        timestamp=time.time(),
    )

    # ── Distance + road class ──────────────────────────────────
    straight_km = haversine(origin.lat, origin.lon,
                            destination.lat, destination.lon)
    road_cls = infer_road_class(straight_km)
    road_km = straight_km * road_factor(road_cls)

    result.distance_km = road_km
    result.road_class = road_cls

    # ── ETA ───────────────────────────────────────────────────
    result.eta_minutes = compute_eta(road_km, road_cls)

    # ── Vehicle selection ──────────────────────────────────────
    vehicle = select_vehicle(mass_kg, remaining_shelf_h)
    result.vehicle_class = vehicle

    # ── Cost model ─────────────────────────────────────────────
    costs = cost_model(road_km, vehicle, mass_kg, cold_chain, usd_to_inr)
    result.fuel_cost_inr = costs["fuel_inr"]
    result.driver_cost_inr = costs["driver_inr"]
    result.cold_chain_cost_inr = costs["cold_inr"]
    result.carbon_offset_inr = costs["offset_inr"]
    result.total_cost_inr = costs["total_inr"]
    result.total_cost_usd = costs["total_usd"]
    result.co2e_kg = costs["co2e_kg"]

    # ── Urgency + tier ─────────────────────────────────────────
    result.urgency_score = urgency_score(remaining_shelf_h, pidr, mass_kg)
    tier, sla = dispatch_tier_for(remaining_shelf_h)
    result.dispatch_tier = tier
    result.dispatch_sla_min = sla

    # ── Pricing ────────────────────────────────────────────────
    md = markdown_for(remaining_shelf_h)
    result.markdown_pct = md
    result.effective_price_inr = base_price_inr * (1.0 - md)
    result.recovery_ratio = (1.0 - md) if base_price_inr > 0 else 0.0

    # ── Action ─────────────────────────────────────────────────
    result.recommended_action = recommended_action(
        tier, remaining_shelf_h, md, destination)

    return result


def plan_multi_stop_dispatch(
    asset_id:          str,
    origin:            GeoNode,
    stops:             list[GeoNode],
    mass_kg:           float,
    remaining_shelf_h: float,
    pidr:              float = 0.0,
    base_price_inr:    float = 0.0,
    cold_chain:        bool = True,
    usd_to_inr:        float = 84.50,
) -> dict:
    """
    Plan an optimised multi-stop route and return aggregate logistics data.
    Uses Nearest-Neighbour greedy TSP for route ordering.
    """
    ordered_route, total_km = optimise_route(origin, stops)

    vehicle = select_vehicle(mass_kg, remaining_shelf_h)
    road_cls = infer_road_class(total_km / max(len(stops), 1))
    costs = cost_model(total_km, vehicle, mass_kg, cold_chain, usd_to_inr)
    eta_min = compute_eta(total_km, road_cls)
    tier, sla = dispatch_tier_for(remaining_shelf_h)
    md = markdown_for(remaining_shelf_h)

    return {
        "asset_id":         asset_id,
        "route_order":      [n.name for n in ordered_route],
        "total_km":         round(total_km, 2),
        "eta_minutes":      round(eta_min, 1),
        "vehicle_class":    vehicle,
        "dispatch_tier":    tier,
        "dispatch_sla_min": sla,
        "total_cost_inr":   costs["total_inr"],
        "total_cost_usd":   costs["total_usd"],
        "co2e_kg":          costs["co2e_kg"],
        "markdown_pct":     round(md * 100, 1),
        "effective_price_inr": round(base_price_inr * (1.0 - md), 2),
        "urgency_score":    urgency_score(remaining_shelf_h, pidr, mass_kg),
    }


# ─────────────────────────────────────────────────────────────────────────────
# KNOWN INDIA SUPPLY CHAIN NODES (seed data)
# ─────────────────────────────────────────────────────────────────────────────

INDIA_NODES: dict[str, GeoNode] = {
    "azadpur_mandi":   GeoNode("azadpur",      "Azadpur APMC Mandi",         28.7141, 77.1739, "warehouse"),
    "vashi_mandi":     GeoNode("vashi",        "Vashi APMC Mandi Mumbai",    19.0771, 72.9988, "warehouse"),
    "koyambedu":       GeoNode("koyambedu",    "Koyambedu APMC Chennai",     13.0694, 80.1947, "warehouse"),
    "madiwala":        GeoNode("madiwala",     "Madiwala APMC Bengaluru",    12.9216, 77.6247, "warehouse"),
    "gultekdi":        GeoNode("gultekdi",     "Gultekdi Market Pune",       18.4855, 73.8563, "warehouse"),
    "bigbasket_hyd":   GeoNode("bb_hyd",       "BigBasket DC Hyderabad",     17.4065, 78.4772, "warehouse"),
    "coldex_delhi":    GeoNode("coldex_dlh",   "COLDEX Reefer Hub Delhi",    28.6692, 77.4538, "cold_hub"),
    "snowman_mum":     GeoNode("snowman_mum",  "Snowman Logistics Mumbai",   19.1136, 72.8697, "cold_hub"),
    "biotech_pune":    GeoNode("bt_pune",      "Zyus Fermentation Lab Pune", 18.5204, 73.8567, "biotech_lab"),
    "biotech_hyd":     GeoNode("bt_hyd",       "Bharat Biotech Campus",      17.5359, 78.5713, "biotech_lab"),
    "zomato_dark_blr": GeoNode("zomato_blr",   "Zomato Dark Kitchen BLR",    12.9716, 77.5946, "retail"),
    "swiggy_del":      GeoNode("swiggy_del",   "Swiggy DC Delhi",            28.6139, 77.2090, "retail"),
}


def nearest_node(origin: GeoNode, node_type: Optional[str] = None) -> GeoNode:
    """Return the nearest known India supply chain node (filtered by type)."""
    candidates = list(INDIA_NODES.values())
    if node_type:
        candidates = [n for n in candidates if n.node_type == node_type]
    if not candidates:
        candidates = list(INDIA_NODES.values())
    return min(
        candidates,
        key=lambda n: haversine(origin.lat, origin.lon, n.lat, n.lon),
    )


# ─────────────────────────────────────────────────────────────────────────────
# CLI DEMO
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json

    farm = GeoNode("farm1", "Nashik Tomato Farm", 19.9975, 73.7898, "farm")
    dest = nearest_node(farm, "cold_hub")

    result = plan_dispatch(
        asset_id="Pallet-T8-Tomatoes",
        origin=farm,
        destination=dest,
        mass_kg=2500,
        remaining_shelf_h=22.0,
        pidr=0.0078,
        base_price_inr=125000.0,
        cold_chain=True,
    )
    print(json.dumps(result.to_dict(), indent=2))
