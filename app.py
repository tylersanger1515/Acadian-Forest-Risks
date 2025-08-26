# ---------- IMPORTS ----------
from __future__ import annotations
import os, json, base64, re
from typing import Dict, Any, Optional, Tuple

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
    # Prefer Streamlit Secrets; then environment; then default
    try:
        return st.secrets[name]
    except Exception:
        return os.getenv(name, default)

def load_config():
    return {
        "FIRE_URL":       _get_secret("N8N_FIRES_URL", ""),
        "RISK_URL":       _get_secret("N8N_RISK_URL", ""),
        "SUBSCRIBE_URL":  _get_secret("N8N_SUBSCRIBE_URL", ""),
        "SHARED_SECRET":  _get_secret("N8N_SHARED_SECRET", ""),       # optional
        "OPENCAGE_KEY":   _get_secret("OPENCAGE_API_KEY", ""),
        "GOOGLE_KEY":     _get_secret("GOOGLE_GEOCODING_API_KEY", ""),# optional
        "TIMEOUT_SEC":    int(_get_secret("REQUEST_TIMEOUT_SEC", "60")),
    }

cfg = load_config()
fires_url     = cfg["FIRE_URL"]
risk_url      = cfg["RISK_URL"]
subscribe_url = cfg["SUBSCRIBE_URL"]
shared_secret = cfg["SHARED_SECRET"]
timeout_sec   = cfg["TIMEOUT_SEC"]
opencage_key  = cfg["OPENCAGE_KEY"]
google_key    = cfg["GOOGLE_KEY"]

DEFAULT_CITIES = [
    "Fredericton,CA","Moncton,CA","Saint John,CA","Bathurst,CA","Miramichi,CA",
    "Charlottetown,CA","Summerside,CA","Halifax,CA","Dartmouth,CA",
    "Sydney,CA","Yarmouth,CA","Truro,CA",
]

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

# ---------- HELPERS ----------
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

# ---- Province-specific bounds (Option B) ----
# west, south, east, north (OpenCage format: "west,south|east,north")
PROVINCE_BOUNDS = {
    "NB": {"south": 44.5, "west": -69.1, "north": 48.1, "east": -63.7},  # New Brunswick
    "NS": {"south": 43.3, "west": -66.5, "north": 47.0, "east": -59.3},  # Nova Scotia
    "PE": {"south": 45.9, "west": -64.4, "north": 47.1, "east": -61.9},  # Prince Edward Island
    "NL": {"south": 46.5, "west": -59.5, "north": 53.8, "east": -52.0},  # NL island & east Labrador
}

def _norm(s: str) -> str:
    s2 = re.sub(r"[^\w\s]", " ", s or "")
    s2 = re.sub(r"\s+", " ", s2).strip().upper()
    return s2

def pick_bounds_from_address(address: str) -> Optional[Dict[str, float]]:
    a = " " + _norm(address) + " "
    if " NEW BRUNSWICK " in a or " NB " in a:
        return PROVINCE_BOUNDS["NB"]
    if " NOVA SCOTIA " in a or " NS " in a:
        return PROVINCE_BOUNDS["NS"]
    if " PRINCE EDWARD ISLAND " in a or " PEI " in a or " PE " in a:
        return PROVINCE_BOUNDS["PE"]
    if " NEWFOUNDLAND " in a or " LABRADOR " in a or " NL " in a:
        return PROVINCE_BOUNDS["NL"]
    return None

# ---------- GEOCODING (OpenCage + Google fallback) ----------
def _opencage_geocode(address: str, api_key: str) -> Optional[Dict[str, Any]]:
    if not api_key or not address.strip():
        return None
    url = "https://api.opencagedata.com/geocode/v1/json"
    params = {
        "q": address.strip(),
        "key": api_key,
        "limit": 1,
        "countrycode": "ca",
        "no_annotations": 1,
        "pretty": 0,
    }
    b = pick_bounds_from_address(address)
    if b:
        params["bounds"] = f"{b['west']},{b['south']}|{b['east']},{b['north']}"
    r = requests.get(url, params=params, timeout=25)
    r.raise_for_status()
    data = r.json()
    results = (data or {}).get("results") or []
    return results[0] if results else None

def _opencage_is_precise(res: Dict[str, Any]) -> bool:
    comp = res.get("components") or {}
    if any(k in comp for k in ("house_number", "house")):
        return True
    if "road" in comp and ("postcode" in comp or "suburb" in comp):
        return True
    conf = res.get("confidence")
    if isinstance(conf, (int, float)) and conf >= 8:
        return True
    return False

def _google_geocode(address: str, api_key: str) -> Optional[Dict[str, Any]]:
    if not api_key or not address.strip():
        return None
    url = "https://maps.googleapis.com/maps/api/geocode/json"
    params = {"address": address.strip(), "region": "ca", "key": api_key}
    b = pick_bounds_from_address(address)
    if b:
        params["bounds"] = f"{b['south']},{b['west']}|{b['north']},{b['east']}"
    r = requests.get(url, params=params, timeout=25)
    r.raise_for_status()
    data = r.json()
    results = (data or {}).get("results") or []
    return results[0] if results else None

def geocode_address(address: str,
                    oc_key: str,
                    g_key: Optional[str] = None) -> Optional[Tuple[float, float, str, str]]:
    """
    Try OpenCage first (province-bounded). If coarse, fall back to Google.
    Returns: (lat, lon, formatted_address, source) or None
    """
    try:
        oc = _opencage_geocode(address, oc_key) if oc_key else None
        if oc:
            lat = float(oc["geometry"]["lat"])
            lon = float(oc["geometry"]["lng"])
            fmt = oc.get("formatted") or address.strip()
            if _opencage_is_precise(oc) or not g_key:
                return (lat, lon, fmt, "opencage")

        if g_key:
            gg = _google_geocode(address, g_key)
            if gg:
                loc = ((gg.get("geometry") or {}).get("location") or {})
                fmt = gg.get("formatted_address") or address.strip()
                lat, lon = float(loc.get("lat")), float(loc.get("lng"))
                return (lat, lon, fmt, "google")

        if oc:
            return (lat, lon, fmt, "opencage")
        return None
    except Exception:
        return None

# ---------- TABS ----------
t1, t2, t3 = st.tabs(["🔥 Active Fires", "🧭 Risk Summary", "🚨 SAFER Fire Alert"])

# ===== TAB 1: ACTIVE FIRES =====
with t1:
    st.subheader("Active Fires in the Acadian Region")
    if st.button("Fetch Active Fires", type="primary", disabled=not bool(fires_url)):
        try:
            data = post_json(fires_url, {"from": "streamlit"}, shared_secret or None, timeout=timeout_sec)
            html = data.get("summary_html")
            if isinstance(html, str) and html.strip():
                components.html(html, height=820, scrolling=True)
            else:
                st.write(data.get("summary") or data.get("summary_text") or "(No summary returned)")
            st.success("Received response from n8n")
        except requests.HTTPError as e:
            st.error(f"HTTP error: {e.response.status_code} {e.response.text[:400]}")
        except Exception as e:
            st.error(f"Failed: {e}")

# ===== TAB 2: RISK SUMMARY =====
with t2:
    st.subheader("Discover how your cities weather is affecting the Acadian Forest today!")
    cities = st.multiselect("Cities", DEFAULT_CITIES, default=["Fredericton,CA"])
    if st.button("Get risk summary", type="primary", disabled=not bool(risk_url)):
        try:
            payload = {"cities": cities, "detail": "detailed", "from": "streamlit"}
            data = post_json(risk_url, payload, shared_secret or None, timeout=max(60, timeout_sec))
            title = data.get("title") or data.get("subject")
            if title: st.markdown(f"### {title}")
            html = data.get("summary_html") or data.get("html")
            if isinstance(html, str) and html.strip():
                components.html(html, height=820, scrolling=True)
            else:
                st.write(data.get("summary_text") or data.get("summary") or "(No summary returned)")
            st.success("Received response from n8n")
        except requests.HTTPError as e:
            st.error(f"HTTP error: {e.response.status_code} {e.response.text[:400]}")
        except Exception as e:
            st.error(f"Failed: {e}")

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

    # ---- form ----
    with st.form("sub_form", clear_on_submit=False):
        email = st.text_input("Email", value=ss["sub_email"], placeholder="you@example.com")

        c_addr = st.columns([4, 1])
        address = c_addr[0].text_input(
            "Address (optional)",
            value=ss["sub_address"],
            placeholder="123 Main St, Halifax, NS B3H 2Y9",
        )
        geocode_clicked = c_addr[1].form_submit_button(
            "Geocode",
            use_container_width=True,
            disabled=not bool(opencage_key or google_key),
        )

        colA, colB = st.columns(2)
        lat = colA.number_input("Latitude", value=float(ss["sub_lat"]), step=0.0001, format="%.6f")
        lon = colB.number_input("Longitude", value=float(ss["sub_lon"]), step=0.0001, format="%.6f")
        radius = st.number_input("Radius (km)", min_value=1, max_value=250, value=int(ss["sub_radius"]), step=1)

        btn_label = "Cancel Alerts" if ss.get("alerts_active") else "Activate Alerts"
        toggle_clicked = st.form_submit_button(btn_label, type="primary", disabled=not bool(subscribe_url))

    # persist current inputs (still under `with t3:`; outside the form)
    ss["sub_email"], ss["sub_address"] = email, address
    ss["sub_lat"], ss["sub_lon"], ss["sub_radius"] = float(lat), float(lon), int(radius)

    # geocode button handler (also under `with t3:`)
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

# --- keep function defs at left margin (not indented under the tab) ---
def _subscribe():
    errs = []
    if not _valid_email(email):
        errs.append("Please enter a valid email.")
    if abs(float(ss['sub_lat'])) > 90:
        errs.append("Latitude must be between -90 and 90.")
    if abs(float(ss['sub_lon'])) > 180:
        errs.append("Longitude must be between -180 and 180.")
    if not (1 <= int(ss['sub_radius']) <= 250):
        errs.append("Radius must be 1–250 km.")
    if errs:
        for e in errs:
            st.error(e)
        return

    # Start with current lat/lon/address
    lat_val, lon_val = float(ss["sub_lat"]), float(ss["sub_lon"])
    addr_val = ss["sub_address"].strip()

    # Only try geocoding if an address was entered
    if (opencage_key or google_key) and address.strip():
        g = geocode_address(address, opencage_key, google_key)
        if g:
            lat_val, lon_val, fmt, _src = g
            ss["sub_lat"], ss["sub_lon"], ss["sub_address"] = lat_val, lon_val, fmt
            addr_val = fmt

    body = {
        "email": email.strip(),
        "lat": lat_val,
        "lon": lon_val,
        "radius_km": int(ss["sub_radius"]),
        "address": addr_val,
        "active": True,
        "from": "streamlit",
    }
    resp = post_json(subscribe_url, body, shared_secret or None, timeout=timeout_sec)

    # Friendly message
    st.success(f'Alerts activated for "{email.strip()}".')
    st.json(resp)

    ss["alerts_active"] = True
    st.rerun()  # flip the button label immediately


def _unsubscribe():
    if not _valid_email(email):
        st.error("Please enter a valid email to cancel alerts.")
        return

    body = {
        "email": email.strip(),
        "active": False,
        "from": "streamlit",
    }
    resp = post_json(subscribe_url, body, shared_secret or None, timeout=timeout_sec)

    # Friendly message
    st.success(f'Alerts canceled for "{email.strip()}".')
    st.json(resp)

    ss["alerts_active"] = False
    st.rerun()  # flip the button label immediately

# --- click handlers must also be at left margin ---
if toggle_clicked and subscribe_url:
    if ss.get("alerts_active"):
        _unsubscribe()
    else:
        _subscribe()

# ---------- FOOTER ----------
st.markdown("""
---
**Notes**
- Address is required for subscribing. We geocode via OpenCage (province bounds) and fall back to Google for house-level precision when needed. You can still fine-tune lat/lon after.
- Put keys in **App → Settings → Secrets** on Streamlit Cloud (recommended). For local dev only, you can also use `.streamlit/secrets.toml`.
""")
