# ---------- IMPORTS ----------
from __future__ import annotations
import os, json, base64, re
from typing import Dict, Any, Optional, Tuple, List

import requests
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import altair as alt

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
        "AI_RISK_URL":     _get_secret("N8N_AI_RISK_URL", ""),
        "RISK_URL":        _get_secret("N8N_RISK_URL", ""),           # fallback only
        "SUBSCRIBE_URL":   _get_secret("N8N_SUBSCRIBE_URL", ""),
        "SHARED_SECRET":   _get_secret("N8N_SHARED_SECRET", ""),
        "OPENCAGE_KEY":    _get_secret("OPENCAGE_API_KEY", ""),
        "GOOGLE_KEY":      _get_secret("GOOGLE_GEOCODING_API_KEY", ""),
        "TIMEOUT_SEC":     int(_get_secret("REQUEST_TIMEOUT_SEC", "60")),
    }

cfg = load_config()
fires_url     = cfg["FIRE_URL"]
risk_url      = cfg["AI_RISK_URL"] or cfg["RISK_URL"]   # Tab 2 uses AI URL only
subscribe_url = cfg["SUBSCRIBE_URL"]
shared_secret = cfg["SHARED_SECRET"]
timeout_sec   = cfg["TIMEOUT_SEC"]
opencage_key  = cfg["OPENCAGE_KEY"]
google_key    = cfg["GOOGLE_KEY"]

# Provinces shown in the UI
PROVINCE_CHOICES = ["NB", "NS", "PE", "NL"]

# Cities per province (kept in sync with your n8n Edit Fields "cities" list)
PROVINCE_CITIES = {
    "NB": ["Fredericton", "Moncton", "Saint John", "Bathurst", "Miramichi"],
    "NS": ["Halifax", "Dartmouth", "Sydney", "Truro"],
    "PE": ["Charlottetown", "Summerside"],
    "NL": ["St. John's", "Corner Brook", "Gander", "Grand Falls-Windsor"],
}

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

def post_map_html(url: str, body: Dict[str, Any], secret: Optional[str], timeout: int = 60) -> str:
    """Call the AI webhook's MAP branch (returns HTML)."""
    h = _headers(secret)
    h["Accept"] = "text/html"
    r = requests.post(url, headers=h, json=body, timeout=timeout)
    r.raise_for_status()
    ct = r.headers.get("Content-Type","")
    if "text/html" in ct:
        return r.text
    # Fallback if the workflow returned JSON with an html field
    try:
        j = r.json()
        return j.get("map_html") or j.get("html") or ""
    except Exception:
        return r.text

def _valid_email(x: str) -> bool:
    return bool(re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", x or ""))

def _results_to_df(results: List[Dict[str, Any]]) -> pd.DataFrame:
    df = pd.json_normalize(results)
    # Standardize a few common column names if present
    rename = {
        "city": "City", "province": "Province",
        "fire_score":"FireScore", "flood_score":"FloodScore",
        "wind_kph":"WindKPH", "temp_c":"TempC", "humidity":"Humidity",
    }
    df = df.rename(columns={k:v for k,v in rename.items() if k in df.columns})
    return df

# ---- Province-specific bounds (for geocoding in Tab 3) ----
PROVINCE_BOUNDS = {
    "NB": {"south": 44.5, "west": -69.1, "north": 48.1, "east": -63.7},
    "NS": {"south": 43.3, "west": -66.5, "north": 47.0, "east": -59.3},
    "PE": {"south": 45.9, "west": -64.4, "north": 47.1, "east": -61.9},
    "NL": {"south": 46.5, "west": -59.5, "north": 53.8, "east": -52.0},
}
def _norm(s: str) -> str:
    s2 = re.sub(r"[^\w\s]", " ", s or "")
    s2 = re.sub(r"\s+", " ", s2).strip().upper()
    return s2
def pick_bounds_from_address(address: str) -> Optional[Dict[str, float]]:
    a = " " + _norm(address) + " "
    if " NEW BRUNSWICK " in a or " NB " in a: return PROVINCE_BOUNDS["NB"]
    if " NOVA SCOTIA " in a or " NS " in a: return PROVINCE_BOUNDS["NS"]
    if " PRINCE EDWARD ISLAND " in a or " PEI " in a or " PE " in a: return PROVINCE_BOUNDS["PE"]
    if " NEWFOUNDLAND " in a or " LABRADOR " in a or " NL " in a: return PROVINCE_BOUNDS["NL"]
    return None

# ---------- GEOCODING (OpenCage + Google fallback) ----------
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
def _opencage_is_precise(res: Dict[str, Any]) -> bool:
    comp = res.get("components") or {}
    if any(k in comp for k in ("house_number","house")): return True
    if "road" in comp and ("postcode" in comp or "suburb" in comp): return True
    conf = res.get("confidence")
    return isinstance(conf,(int,float)) and conf >= 8
def _google_geocode(address: str, api_key: str) -> Optional[Dict[str, Any]]:
    if not api_key or not address.strip(): return None
    url = "https://maps.googleapis.com/maps/api/geocode/json"
    params = {"address": address.strip(), "region":"ca", "key": api_key}
    b = pick_bounds_from_address(address)
    if b: params["bounds"] = f"{b['south']},{b['west']}|{b['north']},{b['east']}"
    r = requests.get(url, params=params, timeout=25); r.raise_for_status()
    data = r.json(); results = (data or {}).get("results") or []
    return results[0] if results else None
def geocode_address(address: str, oc_key: str, g_key: Optional[str] = None) -> Optional[Tuple[float,float,str,str]]:
    try:
        if g_key:
            gg = _google_geocode(address, g_key)
            if gg:
                loc = (gg.get("geometry") or {}).get("location") or {}
                return (float(loc.get("lat")), float(loc.get("lng")), gg.get("formatted_address") or address.strip(), "google")
        if oc_key:
            oc = _opencage_geocode(address, oc_key)
            if oc:
                lat = float(oc["geometry"]["lat"]); lon = float(oc["geometry"]["lng"])
                return (lat, lon, oc.get("formatted") or address.strip(), "opencage")
        return None
    except Exception:
        return None

# ---------- TABS ----------
t1, t2, t3 = st.tabs(["🔥 Active Fires", "🤖 AI Agent", "🚨 SAFER Fire Alert"])

# ===== TAB 1: ACTIVE FIRES =====
with t1:
    st.subheader("Active Fires in the Acadian Region")
    if st.button("Fetch Active Fires", type="primary", disabled=not bool(fires_url)):
        try:
            data = post_json(fires_url, {"from":"streamlit"}, shared_secret or None, timeout=timeout_sec)
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

# ===== TAB 2: AI AGENT (AI risk analysis only) =====
with t2:
    st.subheader("Ask the AI about risk in the Acadian region")
    st.caption("Explain what you want in plain English. The agent can return summaries, **maps**, or **charts** for cities across NB, NS, PE, NL.")

    colL, colR = st.columns([3,2])
    with colL:
        question = st.text_area("Your request", height=110, placeholder="e.g., Which NB cities have the highest fire risk today and why?")
    with colR:
        province = st.selectbox("Province filter", ["ALL", "NB", "NS", "PE", "NL"], index=0)
        since_days = st.number_input("Lookback (days) for 'new' fires (optional)", min_value=0, max_value=30, value=7, step=1)
        output = st.radio("Output", ["Auto", "Map", "Bar chart", "Line chart", "Table"], horizontal=True)

    # Small examples to help users
    with st.expander("Examples you can ask"):
        st.markdown("""
- *"Rank NB cities by fire risk today and show a bar chart."*  
- *"Map the current fires in NS with popup details."*  
- *"Compare Halifax vs Moncton wind speeds and humidity, line chart."*  
- *"List PE cities with fire risk ≥ 3 as a table."*  
- *"Give a short narrative summary for all provinces today."*
        """)

    ask = st.button("Ask AI", type="primary", disabled=not bool(risk_url))
    if ask:
        try:
            fmt = {
                "Auto": "",
                "Map": "map",
                "Bar chart": "chart:bar",
                "Line chart": "chart:line",
                "Table": "table",
            }[output]

            payload = {
                "question": (question or "").strip(),
                "province": province,
                "since_days": int(since_days),
                "format": fmt,
                "detail": "detailed",
                "from": "streamlit",
            }

            if fmt == "map":
                html = post_map_html(risk_url, payload, shared_secret or None, timeout=max(60, timeout_sec))
                if not html.strip():
                    st.warning("No map returned.")
                else:
                    components.html(html, height=820, scrolling=True)
            else:
                data = post_json(risk_url, payload, shared_secret or None, timeout=max(60, timeout_sec))

                # Prefer server-provided HTML if available
                title = data.get("title") or data.get("subject")
                if title: st.markdown(f"### {title}")
                html = data.get("summary_html") or data.get("html")
                if isinstance(html, str) and html.strip():
                    components.html(html, height=820, scrolling=True)

                # If we receive rows, draw charts/tables
                results = data.get("results") or []
                if isinstance(results, list) and results:
                    df = _results_to_df(results)

                    if output == "Table" or (output == "Auto" and len(df.columns) <= 4):
                        st.dataframe(df, use_container_width=True)
                    elif output in ("Bar chart","Line chart") or output == "Auto":
                        # Pick a y metric automatically if Auto
                        y_candidates = [c for c in ["FireScore","FloodScore","WindKPH","TempC","Humidity"] if c in df.columns]
                        y = y_candidates[0] if y_candidates else df.columns[-1]
                        x = "City" if "City" in df.columns else df.columns[0]

                        if output == "Bar chart" or (output == "Auto" and y == "FireScore"):
                            chart = alt.Chart(df).mark_bar().encode(x=alt.X(x, sort='-y'), y=y, tooltip=list(df.columns)).properties(height=400)
                        else:
                            chart = alt.Chart(df).mark_line(point=True).encode(x=x, y=y, tooltip=list(df.columns)).properties(height=400)

                        st.altair_chart(chart, use_container_width=True)

                    # Download CSV helper
                    csv = df.to_csv(index=False).encode("utf-8")
                    st.download_button("Download results as CSV", csv, file_name="risk_results.csv", mime="text/csv")

                if not (html or results):
                    st.write(data.get("summary_text") or data.get("summary") or "(No response returned)")

            st.success("Received response from AI workflow")
        except requests.HTTPError as e:
            st.error(f"HTTP error: {e.response.status_code} {e.response.text[:400]}")
        except Exception as e:
            st.error(f"Failed: {e}")

# ===== TAB 3: SAFER Fire Alert (unchanged label & behavior) =====
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

# --- subscriber helpers (same behavior) ---
def _subscribe():
    errs = []
    if not _valid_email(st.session_state["sub_email"]): errs.append("Please enter a valid email.")
    if abs(float(st.session_state['sub_lat'])) > 90: errs.append("Latitude must be between -90 and 90.")
    if abs(float(st.session_state['sub_lon'])) > 180: errs.append("Longitude must be between -180 and 180.")
    if not (1 <= int(st.session_state['sub_radius']) <= 250): errs.append("Radius must be 1–250 km.")
    if errs:
        for e in errs: st.error(e)
        return
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
