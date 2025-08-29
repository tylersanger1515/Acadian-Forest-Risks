# ---------- IMPORTS ----------
from __future__ import annotations
import os, json, base64, re, math, datetime as dt
from typing import Dict, Any, Optional, Tuple, List

import requests
import streamlit as st
import streamlit.components.v1 as components
import pydeck as pdk

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
        "FIRE_URL":       _get_secret("N8N_FIRES_URL", ""),
        "RISK_URL":       _get_secret("N8N_RISK_URL", ""),
        "SUBSCRIBE_URL":  _get_secret("N8N_SUBSCRIBE_URL", ""),
        "SHARED_SECRET":  _get_secret("N8N_SHARED_SECRET", ""),
        "OPENCAGE_KEY":   _get_secret("OPENCAGE_API_KEY", ""),
        "GOOGLE_KEY":     _get_secret("GOOGLE_GEOCODING_API_KEY", ""),
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
    "Fredericton, NB, Canada", "Moncton, NB, Canada", "Saint John, NB, Canada",
    "Bathurst, NB, Canada", "Miramichi, NB, Canada", "Charlottetown, PE, Canada",
    "Summerside, PE, Canada", "Halifax, NS, Canada", "Dartmouth, NS, Canada",
    "Sydney, NS, Canada", "Yarmouth, NS, Canada", "Truro, NS, Canada",
    "Gander, NL, Canada", "Corner Brook, NL, Canada", "St. John’s, NL, Canada"
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

# ---------- HTTP / GEOCODE HELPERS ----------
def _headers(secret: Optional[str]) -> Dict[str, str]:
    h = {"Content-Type": "application/json"}
    if secret: h["X-API-KEY"] = secret
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

# Province bounds (OpenCage format hints)
PROVINCE_BOUNDS = {
    "NB": {"south": 44.5, "west": -69.1, "north": 48.1, "east": -63.7},
    "NS": {"south": 43.3, "west": -66.5, "north": 47.0, "east": -59.3},
    "PE": {"south": 45.9, "west": -64.4, "north": 47.1, "east": -61.9},
    "NL": {"south": 46.5, "west": -59.5, "north": 53.8, "east": -52.0},
}

def _norm_upper(s: str) -> str:
    s2 = re.sub(r"[^\w\s]", " ", s or "")
    s2 = re.sub(r"\s+", " ", s2).strip().upper()
    return s2

def pick_bounds_from_address(address: str) -> Optional[Dict[str, float]]:
    a = " " + _norm_upper(address) + " "
    if " NEW BRUNSWICK " in a or " NB " in a: return PROVINCE_BOUNDS["NB"]
    if " NOVA SCOTIA " in a or " NS " in a:   return PROVINCE_BOUNDS["NS"]
    if " PRINCE EDWARD ISLAND " in a or " PEI " in a or " PE " in a: return PROVINCE_BOUNDS["PE"]
    if " NEWFOUNDLAND " in a or " LABRADOR " in a or " NL " in a:   return PROVINCE_BOUNDS["NL"]
    return None

def _opencage_geocode(address: str, api_key: str) -> Optional[Dict[str, Any]]:
    if not api_key or not address.strip(): return None
    url = "https://api.opencagedata.com/geocode/v1/json"
    params = {"q": address.strip(), "key": api_key, "limit": 1, "countrycode": "ca", "no_annotations": 1, "pretty": 0}
    b = pick_bounds_from_address(address)
    if b: params["bounds"] = f"{b['west']},{b['south']}|{b['east']},{b['north']}"
    r = requests.get(url, params=params, timeout=25); r.raise_for_status()
    res = (r.json() or {}).get("results") or []
    return res[0] if res else None

def _google_geocode(address: str, api_key: str) -> Optional[Dict[str, Any]]:
    if not api_key or not address.strip(): return None
    url = "https://maps.googleapis.com/maps/api/geocode/json"
    params = {"address": address.strip(), "region": "ca", "key": api_key}
    b = pick_bounds_from_address(address)
    if b: params["bounds"] = f"{b['south']},{b['west']}|{b['north']},{b['east']}"
    r = requests.get(url, params=params, timeout=25); r.raise_for_status()
    res = (r.json() or {}).get("results") or []
    return res[0] if res else None

def geocode_address(address: str,
                    oc_key: str,
                    g_key: Optional[str] = None) -> Optional[Tuple[float, float, str, str]]:
    """Prefer Google, fall back to OpenCage. Returns (lat, lon, formatted, source)."""
    try:
        if g_key:
            gg = _google_geocode(address, g_key)
            if gg:
                loc = (gg.get("geometry") or {}).get("location") or {}
                return (float(loc["lat"]), float(loc["lng"]), gg.get("formatted_address") or address, "google")
        if oc_key:
            oc = _opencage_geocode(address, oc_key)
            if oc:
                return (float(oc["geometry"]["lat"]), float(oc["geometry"]["lng"]), oc.get("formatted") or address, "opencage")
        return None
    except Exception:
        return None

# ---------- TABS ----------
t1, t2, t3 = st.tabs(["🔥 Active Fires", "🧾 Incident Brief", "🚨 SAFER Fire Alert"])

# ===== TAB 1: ACTIVE FIRES =====
with t1:
    st.subheader("Active Fires in the Acadian Region")

    ss = st.session_state
    ss.setdefault("fires_payload", None)
    ss.setdefault("fires_html", None)
    ss.setdefault("city_cache", {})  # label -> (lat, lon)

    left, right = st.columns([1, 1])

    # ---------------- LEFT: summary table ----------------
    with left:
        if st.button("Fetch Active Fires", type="primary", disabled=not bool(fires_url)):
            try:
                data = post_json(fires_url, {"from": "streamlit"}, shared_secret or None, timeout=timeout_sec)
                html = data.get("summary_html")
                if isinstance(html, str) and html.strip():
                    components.html(html, height=820, scrolling=True)
                else:
                    st.write(data.get("summary") or data.get("summary_text") or "(No summary returned)")
                ss["fires_payload"] = data
                ss["fires_html"] = html
                st.success("Received response from n8n")
            except requests.HTTPError as e:
                st.error(f"HTTP error: {e.response.status_code} {e.response.text[:400]}")
            except Exception as e:
                st.error(f"Failed: {e}")

        if ss.get("fires_html"):
            components.html(ss["fires_html"], height=820, scrolling=True)

    # ---------------- RIGHT: Q&A + SAFETY CHECK ----------------
    with right:
        st.markdown("#### Ask about today’s fires")

        # ----- Examples & inputs (kept inside right column) -----
        examples = [
            "which fires are out of control?",
            "top 4 largest in NB",
            "fires within 40 km of Halifax",
            "what place is closest to fire 68586?",
            "when did fire 68622 start?",
        ]
        st.selectbox("Examples", options=examples, index=0, key="examples_q")

        def _copy_example_to_input():
            st.session_state["q_fires"] = st.session_state.get("examples_q", "")

        use_ex = st.button("Use example", key="use_example", on_click=_copy_example_to_input)

        q = st.text_input(
            "Your question",
            key="q_fires",
            placeholder=("e.g., fires near Halifax • within 40 km of Truro • closest to Moncton • "
                         "top 4 largest in NB • totals by province • where is fire 68622 • "
                         "how far is fire 68622 from Halifax • started last 7 days • older than 3 days"),
        )

        ask = st.button("Ask", key="ask_fires", disabled=not bool(fires_url))

        # ---------- small helpers ----------
        def _prov(f): return (f.get("agency") or "").strip().upper()
        def _ctrl_text(f): return (f.get("control") or f.get("status") or "").strip().lower()
        def _sizeha(f):
            try: return float(f.get("size_ha") or 0.0)
            except: return 0.0
        def _started_s(f): return (str(f.get("started") or "")[:10]).strip()
        def _date_iso(s):
            try: return dt.date.fromisoformat((s or "")[:10])
            except Exception: return None
        def _lat(f):
            try: return float(f.get("lat"))
            except: return None
        def _lon(f):
            try: return float(f.get("lon"))
            except: return None

        def haversine_km(lat1, lon1, lat2, lon2):
            if None in (lat1, lon1, lat2, lon2):
                return None
            R = 6371.0; p = math.pi/180.0
            dlat = (lat2-lat1)*p; dlon = (lon2-lon1)*p
            a = math.sin(dlat/2)**2 + math.cos(lat1*p)*math.cos(lat2*p)*math.sin(dlon/2)**2
            return R*2*math.asin(math.sqrt(a))

        def fmt_fire_line(f, show_km=False):
            base = f"- {f.get('name','(id?)')} — {_prov(f)} · {_sizeha(f):,.1f} ha · {f.get('control','—')}"
            if show_km and f.get('_dist_km') is not None:
                base += f" · {f['_dist_km']:.1f} km"
            base += f" · Started {_started_s(f) or '—'}"
            return base

        # numbers in words → ints
        _NUM_WORDS = {"one":1,"two":2,"three":3,"four":4,"five":5,"six":6,"seven":7,"eight":8,"nine":9,"ten":10,
                      "eleven":11,"twelve":12,"thirteen":13,"fourteen":14,"fifteen":15,"sixteen":16,"seventeen":17,
                      "eighteen":18,"nineteen":19,"twenty":20}
        def _num_from_words(t:str) -> str:
            return re.sub(r'\b('+"|".join(_NUM_WORDS.keys())+r')\b',
                          lambda m: str(_NUM_WORDS[m.group(1)]), t or "")

        # size phrases
        def parse_size_range(text):
    t = (text or "").lower()
    # capture numbers with or without unit; allow "100ha", "100 ha", "100"
    nums = [float(x) for x in re.findall(r'(\d+(?:\.\d+)?)\s*(?:ha)?', t)]
    # Normalize phrasing
    if "between" in t and "and" in t and len(nums) >= 2:
        a, b = sorted(nums[:2])
        return ("between", a, b)
    if any(w in t for w in [
        "over", "more than", ">", ">=", "at least", "minimum", "greater than", "bigger than", "larger than", "above"
    ]):
        return ("min", nums[0]) if nums else None
    if any(w in t for w in [
        "under", "less than", "<", "<=", "at most", "maximum", "below", "smaller than", "lesser than"
    ]):
        return ("max", nums[0]) if nums else None
    # patterns like ">= 50ha", "<=100"
    m = re.search(r'([<>]=?)\s*(\d+(?:\.\d+)?)\s*(?:ha)?', t)
    if m:
        op, val = m.group(1), float(m.group(2))
        if op.startswith(">"):
            return ("min", val)
        else:
            return ("max", val)
    return None

**If a wildfire is near you (≤40 km):**
- **Call 911** if you see fire/smoke threatening people or property, or if told to evacuate.
- **Contact your provincial forestry / wildfire line** to report details if safe.
- **Prepare a go-bag:** ID, meds/prescriptions, chargers, cash, water/snacks, clothing, sturdy shoes, flashlight, important docs, pet supplies, masks (N95), glasses.
- **Get ready to leave quickly:** keep your vehicle fueled, park facing the road, know **two ways out**, share plans with family.
- **Harden your home (only if safe):** close windows/vents, remove flammables from deck, wet vegetation, keep hoses ready.
- **Activate the SAFER fire alert for critical proximity alerts.**
""")

        if check_btn:
            try:
                raw = ss.get("fires_payload") or post_json(
                    fires_url, {"from": "safety"}, shared_secret or None, timeout=timeout_sec
                )
                fires = raw.get("fires") or []
                if not loc_in.strip():
                    st.info("Enter a community or coordinates first.")
                else:
                    anchor = None
                    ll = _parse_latlon(loc_in)
                    if ll:
                        anchor = (ll[0], ll[1], f"{ll[0]:.4f}, {ll[1]:.4f}")
                    else:
                        g = geocode_place(loc_in)
                        if g: anchor = (float(g[0]), float(g[1]), g[2])

                    if not anchor:
                        st.info("Couldn't locate that place. Try including province or postal code (e.g., 'Halifax B3H 1X1').")
                    else:
                        lat0, lon0, place_lbl = anchor
                        cands = [f for f in fires if _lat(f) is not None and _lon(f) is not None]
                        for f in cands:
                            f["_dist_km"] = haversine_km(lat0, lon0, _lat(f), _lon(f))
                        nearby = [f for f in cands if (f["_dist_km"] is not None and f["_dist_km"] <= RADIUS_KM)]
                        nearby.sort(key=lambda x: x["_dist_km"] if x["_dist_km"] is not None else 9e9)

                        if nearby:
                            st.error(f"⚠️ {len(nearby)} active fire(s) within {RADIUS_KM:.0f} km of {place_lbl}")
                            for f in nearby:
                                st.write(f"- {f.get('name')} — {_prov(f)} · {_sizeha(f):,.1f} ha · {f.get('control','—')} · "
                                         f"{f['_dist_km']:.1f} km away · Started {_started_s(f) or '—'}")
                            _guidance_block()
                        else:
                            st.success(f"No active fires within {RADIUS_KM:.0f} km of {place_lbl} right now.")
                            with st.expander("Be prepared anyway (quick tips)"):
                                _guidance_block()
            except requests.HTTPError as e:
                st.error(f"HTTP error: {e.response.status_code} {e.response.text[:300]}")
            except Exception as e:
                st.error(f"Check failed: {e}")

# ===== TAB 2: INCIDENT BRIEF =====
with t2:
    st.subheader("Incident Briefs")

    if not risk_url:
        st.warning("Incident Brief webhook URL is not configured. Set N8N_RISK_URL in **App → Settings → Secrets** to your n8n /webhook/ai/incident-brief URL.")
    else:
        ss = st.session_state

        # Persist the radio selection across reruns
        mode = st.radio("Find by", ["Fire ID", "Location"], horizontal=True, key="brief_mode")

        # ---- Input form (ID or Lat/Lon) ----
        with st.form("brief_form", clear_on_submit=False):
            payload = None
            if mode == "Fire ID":
                fire_id = st.text_input("Fire ID (e.g., 68622)", key="brief_id")
                if fire_id.strip():
                    payload = {"id": fire_id.strip()}
            else:
                c1, c2, c3 = st.columns(3)
                with c1:
                    lat = st.number_input("Lat", value=47.4851, format="%.6f")
                with c2:
                    lon = st.number_input("Lon", value=-65.5618, format="%.6f")
                with c3:
                    radius = st.number_input("Radius (km)", min_value=1, value=30, step=1)
                payload = {"lat": float(lat), "lon": float(lon), "radius_km": int(radius)}

            submitted = st.form_submit_button("Get Brief", type="primary")

        # ---- Fetch on submit; cache the result so UI survives reruns ----
        if submitted and payload:
            try:
                data = post_json(risk_url, payload, shared_secret or None, timeout=timeout_sec)
                incident = (data or {}).get("incident") or {}
                brief_md = (data or {}).get("brief_md") or "_No brief returned_"

                ss["brief_data"] = data
                ss["brief_incident"] = incident
                ss["brief_md"] = brief_md
            except requests.HTTPError as e:
                st.error(f"HTTP error: {e.response.status_code} {e.response.text[:400]}")
            except Exception as e:
                st.error(f"Error fetching brief: {e}")
        elif submitted and not payload:
            st.warning("Enter a Fire ID or a location first.")

        # ---- Render from cached values (so other buttons don’t clear the view) ----
        data = ss.get("brief_data")
        incident = ss.get("brief_incident", {})
        brief_md = ss.get("brief_md")

        if data and incident and brief_md:
            # Top metrics
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Tier", data.get("tier_main", data.get("tier", "—")), data.get("tier_sub", ""))
            m2.metric("Control", incident.get("control", "—"))
            try:
                m3.metric("Size (ha)", int(incident.get("size_ha") or 0))
            except Exception:
                m3.metric("Size (ha)", incident.get("size_ha", "—"))
            m4.metric("Started", incident.get("started", "—"))

            # Brief text
            st.markdown(brief_md)

            # Map + Nearest places (Google)
            if "lat" in incident and "lon" in incident:
                st.markdown("**Map**")
                view = pdk.ViewState(
                    latitude=float(incident["lat"]),
                    longitude=float(incident["lon"]),
                    zoom=8, pitch=0
                )
                layer = pdk.Layer(
                    "ScatterplotLayer",
                    data=[{"lat": incident["lat"], "lon": incident["lon"]}],
                    get_position=["lon", "lat"],
                    get_radius=1000,
                    radius_min_pixels=6,
                    radius_max_pixels=60,
                    pickable=False,
                )
                st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view))

                # --- Nearest places (Google Places) ---
                def _places_nearby(lat, lon, radius_m, api_key, types=("locality", "sublocality", "neighborhood")):
                    url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
                    out, seen = [], set()
                    for t in types:
                        params = {"location": f"{lat},{lon}", "radius": int(radius_m), "type": t, "key": api_key}
                        r = requests.get(url, params=params, timeout=20); r.raise_for_status()
                        for it in (r.json() or {}).get("results", []):
                            pid = it.get("place_id")
                            if not pid or pid in seen:
                                continue
                            seen.add(pid)
                            loc = (it.get("geometry") or {}).get("location") or {}
                            plat, plon = float(loc.get("lat", 0)), float(loc.get("lng", 0))
                            # haversine (km)
                            R = 6371.0; p = math.pi/180.0
                            dlat = (plat - float(lat)) * p
                            dlon = (plon - float(lon)) * p
                            a = math.sin(dlat/2)**2 + math.cos(float(lat)*p)*math.cos(plat*p)*math.sin(dlon/2)**2
                            dist_km = 2 * R * math.asin(math.sqrt(a))
                            out.append({
                                "name": it.get("name") or "(unnamed)",
                                "types": it.get("types") or [],
                                "lat": plat, "lon": plon,
                                "distance_km": round(dist_km, 2),
                            })
                    out.sort(key=lambda x: x["distance_km"])
                    return out

                st.divider()
                st.markdown("**Nearest places (Google)**")

                if not google_key:
                    st.info("Add GOOGLE_GEOCODING_API_KEY in App → Settings → Secrets to enable this.")
                else:
                    c1, c2 = st.columns([2, 2])
                    with c1:
                        radius_choices = st.multiselect(
                            "Show closest within (km)",
                            options=[1, 5, 10, 20, 30, 40, 80],
                            default=[20, 40],
                        )
                    with c2:
                        type_choices = st.multiselect(
                            "Place types",
                            options=["locality", "sublocality", "neighborhood", "point_of_interest", "establishment"],
                            default=["locality", "sublocality", "neighborhood"],
                        )

                    if st.button("Find nearest places"):
                        lat_i, lon_i = float(incident["lat"]), float(incident["lon"])
                        max_r = (max(radius_choices) if radius_choices else 40) * 1000  # meters
                        places = _places_nearby(lat_i, lon_i, max_r, google_key, tuple(type_choices))

                        if not places:
                            st.info("No places returned by Google for these settings.")
                        else:
                            for R in sorted(radius_choices):
                                subset = [p for p in places if p["distance_km"] <= float(R)]
                                st.markdown(f"**≤ {R} km** — {len(subset)} place(s)")
                                if subset:
                                    for p in subset[:12]:
                                        t = ", ".join(p.get("types", [])[:3])
                                        st.write(f"- {p['name']} — {p['distance_km']:.2f} km" + (f" · _{t}_" if t else ""))
                                else:
                                    st.caption("none")

            # Quick links / details
            cA, cB = st.columns(2)
            if (data or {}).get("map_link"):
                cA.link_button("Open in Google Maps", data["map_link"])
            with cB:
                with st.popover("Details"):
                    st.json(incident)

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

    # persist
    ss["sub_email"], ss["sub_address"] = email, address
    ss["sub_lat"], ss["sub_lon"], ss["sub_radius"] = float(lat), float(lon), int(radius)

    # geocode button handler
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

# --- subscribe/unsubscribe helpers kept at left margin ---
def _subscribe():
    errs = []
    if not _valid_email(st.session_state.get("sub_email","")):
        errs.append("Please enter a valid email.")
    if abs(float(st.session_state['sub_lat'])) > 90:
        errs.append("Latitude must be between -90 and 90.")
    if abs(float(st.session_state['sub_lon'])) > 180:
        errs.append("Longitude must be between -180 and 180.")
    if not (1 <= int(st.session_state['sub_radius']) <= 250):
        errs.append("Radius must be 1–250 km.")
    if errs:
        for e in errs: st.error(e)
        return

    email = st.session_state["sub_email"].strip()
    address = st.session_state["sub_address"].strip()
    lat_val, lon_val = float(st.session_state["sub_lat"]), float(st.session_state["sub_lon"])

    if (opencage_key or google_key) and address:
        g = geocode_address(address, opencage_key, google_key)
        if g:
            lat_val, lon_val, fmt, _ = g
            st.session_state["sub_lat"], st.session_state["sub_lon"], st.session_state["sub_address"] = lat_val, lon_val, fmt
            address = fmt

    body = {
        "email": email, "lat": lat_val, "lon": lon_val,
        "radius_km": int(st.session_state["sub_radius"]),
        "address": address, "active": True, "from": "streamlit",
    }
    resp = post_json(subscribe_url, body, shared_secret or None, timeout=timeout_sec)
    st.success(f'Alerts activated for "{email}".'); st.json(resp)
    st.session_state["alerts_active"] = True; st.rerun()

def _unsubscribe():
    email = st.session_state.get("sub_email","").strip()
    if not _valid_email(email):
        st.error("Please enter a valid email to cancel alerts."); return
    body = {"email": email, "active": False, "from": "streamlit"}
    resp = post_json(subscribe_url, body, shared_secret or None, timeout=timeout_sec)
    st.success(f'Alerts canceled for "{email}".'); st.json(resp)
    st.session_state["alerts_active"] = False; st.rerun()

# click handlers
if 'toggle_clicked' in globals() and toggle_clicked and subscribe_url:
    if st.session_state.get("alerts_active"): _unsubscribe()
    else: _subscribe()

# ---------- FOOTER ----------
st.markdown("""
---
**Notes**
- Add your keys in **App → Settings → Secrets**. We’ll use Google when available and fall back to OpenCage (bounded by province) for faster, locality-correct results.
""")
