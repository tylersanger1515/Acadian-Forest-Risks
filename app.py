# ---------- IMPORTS ----------
from __future__ import annotations
import os, json, base64, re, io, csv, math
from typing import Dict, Any, Optional, Tuple, List

import requests
import streamlit as st
import streamlit.components.v1 as components

# ---------- CONFIG ----------
IMG_PATH = "assets/images/fokabs image.jpg"
PAGE_ICON = IMG_PATH if os.path.exists(IMG_PATH) else "🌲"

st.set_page_config(
    page_title="SAFER — Sustainable Acadian Forests & Environmental Risks",
    page_icon=PAGE_ICON,
    layout="wide",
)

# ---------- SECRETS / CONFIG HELPERS ----------
def _get_secret(name: str, default: str | None = None) -> str | None:
    try:
        return st.secrets[name]
    except Exception:
        return os.getenv(name, default)

def load_config():
    return {
        "FIRE_URL":        _get_secret("N8N_FIRES_URL", ""),
        "RISK_URL":        _get_secret("N8N_RISK_URL", ""),
        "SUBSCRIBE_URL":   _get_secret("N8N_SUBSCRIBE_URL", ""),
        "SHARED_SECRET":   _get_secret("N8N_SHARED_SECRET", ""),
        "OPENCAGE_KEY":    _get_secret("OPENCAGE_API_KEY", ""),
        "GOOGLE_KEY":      _get_secret("GOOGLE_GEOCODING_API_KEY", ""),
        "TIMEOUT_SEC":     int(_get_secret("REQUEST_TIMEOUT_SEC", "60")),
    }

cfg = load_config()
fires_url     = cfg["FIRE_URL"]            # Tab 1 (n8n HTML summary)
risk_url      = cfg["RISK_URL"]            # not used on Tab 2 anymore
subscribe_url = cfg["SUBSCRIBE_URL"]
shared_secret = cfg["SHARED_SECRET"]
timeout_sec   = cfg["TIMEOUT_SEC"]
opencage_key  = cfg["OPENCAGE_KEY"]
google_key    = cfg["GOOGLE_KEY"]

# Provinces used in UI
PROVINCE_CHOICES = ["ALL", "NB", "NS", "PE", "NL"]

# ---------- STYLE & HEADER ----------
_STYLES = """
<style>
:root{ --beige:#f6f2ea; --ink:#1f2937; --pine:#0f5132; --pine-2:#2d8a4f; }
.stApp{ background:var(--beige); }
header[data-testid="stHeader"]{ background:var(--beige); box-shadow:none; min-height:32px; height:32px; }
.block-container{ max-width:1200px; margin:0 auto; padding-top:1rem; padding-bottom:2rem; }
[data-baseweb="tab-list"] button[role="tab"][aria-selected="true"]{ border-bottom:2px solid var(--pine); }
.s-header{ margin-top:6px; padding:8px 0 16px; margin-bottom:8px; border-bottom:1px solid #e6e0d4; }
.s-title{ display:flex; align-items:center; gap:12px; }
.s-acronym{ font-weight:800; font-size:56px; letter-spacing:.3px; color:var(--pine); }
.s-sub{ font-size:20px; color:#374151; margin-top:0; }
.s-tag{ font-size:16px; font-style:italic; color:var(--pine-2); margin-top:4px; }
@media (max-width:700px){ .s-acronym{ font-size:42px; } }
</style>
"""

def _fokabs_logo(height=44) -> str:
    if os.path.exists(IMG_PATH):
        ext = os.path.splitext(IMG_PATH)[1].lower()
        mime = "image/png" if ext == ".png" else "image/jpeg"
        b64 = base64.b64encode(open(IMG_PATH, "rb").read()).decode("ascii")
        return (f'<img src="data:{mime};base64,{b64}" alt="FOKABS" '
                f'style="height:{height}px;width:auto;vertical-align:baseline;border-radius:10px;'
                f'padding:6px 10px;background:#fff;box-shadow:0 1px 3px rgba(0,0,0,.08);" />')
    return '🌐'

_HEADER = (
    '<div class="s-header"><div class="s-title">'
    f'<span class="s-acronym">SAFER</span>{_fokabs_logo(44)}'
    '</div><div class="s-sub">Sustainable Acadian Forests &amp; Environmental Risks</div>'
    '<div class="s-tag">Monitor, Maintain, Move Forward</div></div>'
)

st.markdown(_STYLES, unsafe_allow_html=True)
st.markdown(_HEADER, unsafe_allow_html=True)

# ---------- GENERIC HELPERS ----------
def _headers(secret: Optional[str]) -> Dict[str, str]:
    h = {"Content-Type": "application/json"}
    if secret:
        h["X-API-KEY"] = secret
    return h

def post_json(url: str, body: Dict[str, Any], secret: Optional[str], timeout: int = 60) -> Dict[str, Any]:
    r = requests.post(url, headers=_headers(secret), json=body, timeout=timeout)
    r.raise_for_status()
    try:
        return r.json()
    except Exception:
        return {"summary": r.text}

def _valid_email(x: str) -> bool:
    return bool(re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", x or ""))

# ---- Province-specific bounds (for geocoding) ----
PROVINCE_BOUNDS = {
    "NB": {"south": 44.5, "west": -69.1, "north": 48.1, "east": -63.7},
    "NS": {"south": 43.3, "west": -66.5, "north": 47.0, "east": -59.3},
    "PE": {"south": 45.9, "west": -64.4, "north": 47.1, "east": -61.9},
    "NL": {"south": 46.5, "west": -59.5, "north": 53.8, "east": -52.0},
}

def _norm(s: str) -> str:
    s2 = re.sub(r"[^\w\s]", " ", s or "")
    s2 = re.sub(r"\s+", " ").strip().upper()
    return s2

def pick_bounds_from_address(address: str) -> Optional[Dict[str, float]]:
    a = " " + _norm(address) + " "
    if " NEW BRUNSWICK " in a or " NB " in a: return PROVINCE_BOUNDS["NB"]
    if " NOVA SCOTIA " in a or " NS " in a: return PROVINCE_BOUNDS["NS"]
    if " PRINCE EDWARD ISLAND " in a or " PEI " in a or " PE " in a: return PROVINCE_BOUNDS["PE"]
    if " NEWFOUNDLAND " in a or " LABRADOR " in a or " NL " in a: return PROVINCE_BOUNDS["NL"]
    return None

# ---------- GEOCODING (Google first, OpenCage fallback, then postal centroid) ----------
def _opencage_geocode(address: str, api_key: str) -> Optional[Dict[str, Any]]:
    if not api_key or not address.strip(): return None
    url = "https://api.opencagedata.com/geocode/v1/json"
    params = {"q": address.strip(), "key": api_key, "limit": 1, "countrycode": "ca",
              "no_annotations": 1, "pretty": 0}
    b = pick_bounds_from_address(address)
    if b: params["bounds"] = f"{b['west']},{b['south']}|{b['east']},{b['north']}"
    r = requests.get(url, params=params, timeout=25); r.raise_for_status()
    data = r.json(); results = (data or {}).get("results") or []
    return results[0] if results else None

def _google_geocode(address: str, api_key: str) -> Optional[Dict[str, Any]]:
    if not api_key or not address.strip(): return None
    url = "https://maps.googleapis.com/maps/api/geocode/json"
    params = {"address": address.strip(), "region":"ca", "key": api_key}
    b = pick_bounds_from_address(address)
    if b: params["bounds"] = f"{b['south']},{b['west']}|{b['north']},{b['east']}"
    r = requests.get(url, params=params, timeout=25); r.raise_for_status()
    data = r.json(); results = (data or {}).get("results") or []
    return results[0] if results else None

POSTAL_PREFIX_TO_COORD = {
    # NL examples
    "A1A": (47.5606, -52.7126),
    "A1B": (47.5769, -52.7342),
    "A1X": (47.5120, -52.9480),
    # NB/NS/PE common prefixes
    "E1C": (46.087, -64.778),   # Moncton
    "B3J": (44.647, -63.573),   # Halifax
    "C1A": (46.238, -63.129),   # Charlottetown
}

def _normalize_address(addr: str) -> str:
    s = (addr or "").strip()
    if not s: return s
    tokens = s.split()
    repl = {"ext":"extension","st":"street","rd":"road","ave":"avenue","dr":"drive"}
    tokens = [repl.get(t.lower(), t) for t in tokens]
    s = " ".join(tokens)
    su = " " + s.upper() + " "
    if not any(p in su for p in [" NB "," NS "," NL "," PE "," PEI "," NEW BRUNSWICK "," NOVA SCOTIA "," NEWFOUNDLAND "," LABRADOR "," PRINCE EDWARD ISLAND "]):
        s += ", New Brunswick, Canada"
    elif " CANADA" not in su:
        s += ", Canada"
    return s

def _postal_centroid(addr: str):
    m = re.search(r"\b([A-Za-z]\d[A-Za-z])\s?(\d[A-Za-z]\d)\b", addr or "", re.I)
    if not m: return None
    prefix = m.group(1).upper()
    return POSTAL_PREFIX_TO_COORD.get(prefix)

def geocode_address(address: str, oc_key: str, g_key: Optional[str] = None) -> Optional[Tuple[float,float,str,str]]:
    try:
        address = _normalize_address(address)
        if g_key:
            gg = _google_geocode(address, g_key)
            if gg:
                loc = (gg.get("geometry") or {}).get("location") or {}
                return (float(loc.get("lat")), float(loc.get("lng")), gg.get("formatted_address") or address, "google")
        if oc_key:
            oc = _opencage_geocode(address, oc_key)
            if oc:
                lat = float(oc["geometry"]["lat"]); lon = float(oc["geometry"]["lng"])
                return (lat, lon, oc.get("formatted") or address, "opencage")
        pc = _postal_centroid(address)
        if pc:
            lat, lon = pc
            return (lat, lon, f"{address} (postal centroid)", "local")
        return None
    except Exception:
        return None

# =====================================================================
# TAB 2: SHORT‑TERM LOCAL "AI AGENT" FOR FIRE QUESTIONS (Q2 + Q10)
# =====================================================================
ACTIVE_FIRES_URL = "https://cwfis.cfs.nrcan.gc.ca/downloads/activefires/activefires.csv"

# Mock industrial/flare points — replace with real list later
FLARE_SITES = [
    {"name": "Saint John Refinery", "lat": 45.291, "lon": -66.025},
    {"name": "Come By Chance Refinery", "lat": 47.812, "lon": -53.967},
    {"name": "Halifax Industrial", "lat": 44.655, "lon": -63.600},
]
SUPPRESS_RADIUS_KM = 3.0  # 3 km


def _hav_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return 2 * R * math.asin(math.sqrt(a))


def _nearest_flare_km(lat: float, lon: float) -> Tuple[float, str]:
    best = None
    for s in FLARE_SITES:
        d = _hav_m(lat, lon, s["lat"], s["lon"]) / 1000.0
        if best is None or d < best[0]:
            best = (d, s["name"])
    return best or (float("inf"), "")


def fetch_active_fires_csv() -> List[Dict[str, Any]]:
    r = requests.get(ACTIVE_FIRES_URL, timeout=30)
    r.raise_for_status()
    text = r.text
    rows = list(csv.DictReader(io.StringIO(text)))
    return rows


def filter_by_province(rows: List[Dict[str, Any]], province: str | None) -> List[Dict[str, Any]]:
    if not province or province.upper() in ("ALL", "ANY"):
        return rows
    prov_cols = ["PROVINCE", "PROV_TERR", "province", "prov_terr"]
    out = []
    for r in rows:
        val = None
        for c in prov_cols:
            if c in r and r[c]:
                val = str(r[c]).strip().upper(); break
        if not val: continue
        if val == province.upper(): out.append(r)
    return out


def normalize_fire_row(r: Dict[str, Any]) -> Dict[str, Any]:
    lat = r.get("LATITUDE") or r.get("LAT") or r.get("latitude")
    lon = r.get("LONGITUDE") or r.get("LON") or r.get("longitude")
    fid = r.get("FIRE_ID") or r.get("FIRENUMBER") or r.get("id") or r.get("FIRE_NUM")
    prov = r.get("PROVINCE") or r.get("PROV_TERR") or r.get("province")
    conf = r.get("CONFIDENCE") or r.get("confidence") or 0.7
    try:
        lat = float(lat); lon = float(lon)
    except Exception:
        return {}
    try:
        conf = float(conf)
    except Exception:
        conf = 0.7
    return {
        "id": str(fid or f"{lat:.4f},{lon:.4f}"),
        "province": (prov or "").strip() or None,
        "lat": lat,
        "lon": lon,
        "confidence": conf,
    }


def _eval_hotspot(lat: float, lon: float) -> Dict[str, Any]:
    d_km, name = _nearest_flare_km(lat, lon)
    likely_false = d_km <= SUPPRESS_RADIUS_KM
    verdict = "LIKELY FALSE POSITIVE (industrial flare)" if likely_false else "not near known flares"
    lines = [
        f"<b>Hotspot check</b> at {lat:.4f}, {lon:.4f}",
        f"• Nearest flare site: {name or '—'} ({d_km:.2f} km)",
        f"• Verdict: <b>{verdict}</b>",
    ]
    incidents = [{
        "id": f"{lat:.4f},{lon:.4f}",
        "status": "update",
        "confidence": 0.5 if likely_false else 0.8,
        "priority": 50 if likely_false else 80,
        "location": {"lat": lat, "lon": lon},
        "province": None,
        "sources": ["Provincial"],
        "forecast": {"h8": {}, "h24": {}, "h72": {}},
        "actions_taken": ["verified_basic"],
        "likely_false_positive": likely_false,
        "near_flare_km": round(d_km, 2),
        "suppressed_reason": f"Near {name} ({d_km:.2f} km)" if likely_false else None,
    }]
    return {"summary_html": "<br>".join(lines), "incidents": incidents}


def build_fire_summary(province: str | None) -> Dict[str, Any]:
    # Try live CSV, but guarantee output even if empty
    try:
        raw = fetch_active_fires_csv()
    except Exception:
        raw = []
    raw = filter_by_province(raw, province)
    norm = [normalize_fire_row(r) for r in raw]
    norm = [r for r in norm if r]

    # Fallback mock rows if nothing
    if not norm:
        norm = [
            {"id":"NB001","province":"NB","lat":46.0868,"lon":-64.7782,"confidence":0.83},  # Moncton (active)
            {"id":"NB002","province":"NB","lat":45.3000,"lon":-66.0300,"confidence":0.71},  # near refinery (will suppress)
        ]
        if province and province.upper() not in ("ALL","NB"):  # keep demo consistent if not NB
            # move mock to selected province but far from our flare sites
            if province.upper()=="NS": norm[0].update({"province":"NS","lat":44.682,"lon":-63.744})
            if province.upper()=="PE": norm[0].update({"province":"PE","lat":46.286,"lon":-63.126})
            if province.upper()=="NL": norm[0].update({"province":"NL","lat":47.5606,"lon":-52.7126})

    active, suppressed = [], []
    for j in norm:
        d_km, near_name = _nearest_flare_km(j["lat"], j["lon"])
        j["near_flare_km"] = round(d_km, 2)
        j["likely_false_positive"] = d_km <= SUPPRESS_RADIUS_KM
        if j["likely_false_positive"]:
            j["suppressed_reason"] = f"Near {near_name} ({j['near_flare_km']} km)"
            j["confidence"] = min(j.get("confidence", 0.7), 0.5)
            suppressed.append(j)
        else:
            active.append(j)

    reg = f"{province.upper()}" if province and province.upper() != "ALL" else "the Acadian region"
    lines = [f"<b>Active fires in {reg}</b>: {len(active)}"]
    if active:
        top = sorted(active, key=lambda x: x.get("confidence", 0), reverse=True)[:5]
        for t in top:
            lines.append(f"• <b>{t['id']}</b> ({t.get('province') or '—'}) – conf {t['confidence']:.2f} at {t['lat']:.3f}, {t['lon']:.3f}")
    if suppressed:
        lines.append(f"<hr><i>{len(suppressed)} detections suppressed as likely non-wildfire (e.g., industrial flare).</i>")

    incidents = [{
        "id": j["id"],
        "status": "update",
        "confidence": j.get("confidence", 0.7),
        "priority": int(round(j.get("confidence", 0.7) * 100)),
        "location": {"lat": j["lat"], "lon": j["lon"]},
        "province": j.get("province"),
        "sources": ["Provincial"],
        "forecast": {"h8": {}, "h24": {}, "h72": {}},
        "actions_taken": ["verified_basic"],
    } for j in active]

    return {"summary_html": "<br>".join(lines), "incidents": incidents, "next_actions": [
        "Replace mock flare points with real layers",
        "Add smoke/lightning cross-checks",
    ], "action_error": None}

# ---------- TABS ----------
t1, t2, t3 = st.tabs(["🔥 Active Fires", "🤖 AI Agent", "🚨 SAFER Fire Alert"])

# ===== TAB 1: ACTIVE FIRES (n8n-backed) =====
with t1:
    st.subheader("Active Fires in the Acadian Region")
    if st.button("Fetch Active Fires", type="primary", disabled=not bool(fires_url)):
        try:
            data = post_json(fires_url, {"from": "streamlit"}, shared_secret or None, timeout=timeout_sec)
            html = data.get("summary_html") or data.get("html")
            if isinstance(html, str) and html.strip():
                components.html(html, height=820, scrolling=True)
            else:
                st.write(data.get("summary") or data.get("summary_text") or "(No summary returned)")
            st.success("Received response from n8n")
        except requests.HTTPError as e:
            st.error(f"HTTP error: {e.response.status_code} {e.response.text[:400]}")
        except Exception as e:
            st.error(f"Failed: {e}")

# ===== TAB 2: 🤖 AI AGENT (LOCAL — answers Q2 & Q10) =====
with t2:
    st.subheader("Ask the AI about fires in the Acadian region")
    st.caption("This agent summarizes active fires and applies a basic false‑positive check near industrial flares (3 km).")

    colq, colp = st.columns([3,1])
    question = colq.text_input("Your request", value="Summarize active fires", placeholder="e.g., Is the hotspot near 45.29 -66.02 a false positive?")
    province = colp.selectbox("Province filter", PROVINCE_CHOICES, index=1)  # default NB

    if st.button("Ask AI", type="primary"):
        try:
            # Q2 direct check if lat/lon found in the question
            m = re.search(r"(-?\d+(?:\.\d+)?)\s*,?\s+(-?\d+(?:\.\d+)?)", (question or ""))
            if m:
                lat, lon = float(m.group(1)), float(m.group(2))
                resp = _eval_hotspot(lat, lon)
                st.markdown(resp["summary_html"], unsafe_allow_html=True)
                with st.expander("Show incidents JSON"):
                    st.json(resp["incidents"])
                st.success("Hotspot check complete (Q2).")
            else:
                # Q10 summary (with built‑in mock fallback)
                resp = build_fire_summary(None if province == "ALL" else province)
                st.markdown(resp.get("summary_html", "<i>No summary produced.</i>"), unsafe_allow_html=True)
                with st.expander("Show incidents JSON"):
                    st.json(resp.get("incidents", []))
                st.success("Summary generated (Q10).")
        except Exception as e:
            st.error(f"AI Agent failed: {e}")

# ===== TAB 3: SAFER Fire Alert =====
with t3:
    st.subheader("Be SAFER in the Acadian region with a fire alert response for your home address")
    st.write("Enter your **address** (optional). We’ll geocode it to coordinates, or you can set lat/lon manually.")

    if not subscribe_url:
        st.warning("Subscribe webhook URL is not configured. Add N8N_SUBSCRIBE_URL in **App → Settings → Secrets**.")

    ss = st.session_state
    ss.setdefault("sub_email", "")
    ss.setdefault("sub_address", "")
    ss.setdefault("sub_lat", 46.1675)
    ss.setdefault("sub_lon", -64.7508)
    ss.setdefault("sub_radius", 10)
    ss.setdefault("alerts_active", False)

    with st.form("sub_form", clear_on_submit=False):
        email = st.text_input("Email", value=ss["sub_email"], placeholder="you@example.com")
        c_addr = st.columns([4, 1])
        address = c_addr[0].text_input("Address (optional)", value=ss["sub_address"], placeholder="123 Main St, Halifax, NS B3H 2Y9")
        geocode_clicked = c_addr[1].form_submit_button("Geocode", use_container_width=True, disabled=not bool(opencage_key or google_key))
        colA, colB = st.columns(2)
        lat = colA.number_input("Latitude", value=float(ss["sub_lat"]), step=0.0001, format="%.6f")
        lon = colB.number_input("Longitude", value=float(ss["sub_lon"]), step=0.0001, format="%.6f")
        radius = st.number_input("Radius (km)", min_value=1, max_value=250, value=int(ss["sub_radius"]), step=1)
        btn_label = "Cancel Alerts" if ss.get("alerts_active") else "Activate Alerts"
        toggle_clicked = st.form_submit_button(btn_label, type="primary", disabled=not bool(subscribe_url))

    ss["sub_email"], ss["sub_address"] = email, address
    ss["sub_lat"], ss["sub_lon"], ss["sub_radius"] = float(lat), float(lon), int(radius)

    if geocode_clicked:
        if not (opencage_key or google_key):
            st.error("Please add at least one geocoding key (OpenCage or Google) in **App → Settings → Secrets**.")
        elif not address.strip():
            st.error("Please enter an address to geocode.")
        else:
            g = geocode_address(address, opencage_key, google_key)
            if not g:
                st.error("No coordinates found for that address.")
            else:
                g_lat, g_lon, g_fmt, g_src = g
                ss["sub_lat"], ss["sub_lon"], ss["sub_address"] = g_lat, g_lon, g_fmt
                st.success(f"Coordinates filled from address (via {g_src}).")
                st.rerun()

# --- subscriber helpers ---
def _subscribe():
    errs = []
    if not _valid_email(st.session_state["sub_email"]): errs.append("Please enter a valid email.")
    if abs(float(st.session_state['sub_lat'])) > 90: errs.append("Latitude must be between -90 and 90.")
    if abs(float(st.session_state['sub_lon'])) > 180: errs.append("Longitude must be between -180 and 180.")
    if not (1 <= int(st.session_state['sub_radius']) <= 250): errs.append("Radius must be 1–250 km.")
    if errs:
        for e in errs: st.error(e); return
    lat_val, lon_val = float(st.session_state["sub_lat"]), float(st.session_state["sub_lon"])
    addr_val = st.session_state["sub_address"].strip()
    if (opencage_key or google_key) and addr_val:
        g = geocode_address(addr_val, opencage_key, google_key)
        if g:
            lat_val, lon_val, fmt, _ = g
            st.session_state["sub_lat"], st.session_state["sub_lon"], st.session_state["sub_address"] = lat_val, lon_val, fmt
            addr_val = fmt
    body = {"email": st.session_state["sub_email"].strip(), "lat": lat_val, "lon": lon_val,
            "radius_km": int(st.session_state["sub_radius"]), "address": addr_val, "active": True, "from": "streamlit"}
    resp = post_json(subscribe_url, body, shared_secret or None, timeout=timeout_sec)
    st.success(f'Alerts activated for "{st.session_state["sub_email"].strip()}".'); st.json(resp)
    st.session_state["alerts_active"] = True; st.rerun()


def _unsubscribe():
    if not _valid_email(st.session_state["sub_email"]):
        st.error("Please enter a valid email to cancel alerts."); return
    body = {"email": st.session_state["sub_email"].strip(), "active": False, "from": "streamlit"}
    resp = post_json(subscribe_url, body, shared_secret or None, timeout=timeout_sec)
    st.success(f'Alerts canceled for "{st.session_state["sub_email"].strip()}".'); st.json(resp)
    st.session_state["alerts_active"] = False; st.rerun()

if 'toggle_clicked' in locals() and toggle_clicked and subscribe_url:
    if st.session_state.get("alerts_active"): _unsubscribe()
    else: _subscribe()
