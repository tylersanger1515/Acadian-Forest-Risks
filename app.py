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

# ---------- HTTP / UTIL HELPERS ----------

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

# ---------- GEOCODING HELPERS ----------
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
    if " NEW BRUNSWICK " in a or " NB " in a:
        return PROVINCE_BOUNDS["NB"]
    if " NOVA SCOTIA " in a or " NS " in a:
        return PROVINCE_BOUNDS["NS"]
    if " PRINCE EDWARD ISLAND " in a or " PEI " in a or " PE " in a:
        return PROVINCE_BOUNDS["PE"]
    if " NEWFOUNDLAND " in a or " LABRADOR " in a or " NL " in a:
        return PROVINCE_BOUNDS["NL"]
    return None


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
    res = (r.json() or {}).get("results") or []
    return res[0] if res else None


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
    res = (r.json() or {}).get("results") or []
    return res[0] if res else None


def geocode_address(address: str, oc_key: str, g_key: Optional[str] = None) -> Optional[Tuple[float, float, str, str]]:
    """Prefer Google, fall back to OpenCage. Returns (lat, lon, formatted, source)."""
    try:
        if g_key:
            gg = _google_geocode(address, g_key)
            if gg:
                loc = (gg.get("geometry") or {}).get("location") or {}
                return (
                    float(loc["lat"]),
                    float(loc["lng"]),
                    gg.get("formatted_address") or address,
                    "google",
                )
        if oc_key:
            oc = _opencage_geocode(address, oc_key)
            if oc:
                return (
                    float(oc["geometry"]["lat"]),
                    float(oc["geometry"]["lng"]),
                    oc.get("formatted") or address,
                    "opencage",
                )
        return None
    except Exception:
        return None


# ---------- GENERIC HELPERS ----------

def _parse_latlon(s: str) -> Optional[Tuple[float, float]]:
    if not s:
        return None
    m = re.search(r"(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)", s)
    if not m:
        return None
    try:
        return float(m.group(1)), float(m.group(2))
    except Exception:
        return None


def haversine_km(lat1, lon1, lat2, lon2):
    if None in (lat1, lon1, lat2, lon2):
        return None
    R = 6371.0
    p = math.pi / 180.0
    dlat = (lat2 - lat1) * p
    dlon = (lon2 - lon1) * p
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1 * p) * math.cos(lat2 * p) * math.sin(dlon / 2) ** 2
    return R * 2 * math.asin(math.sqrt(a))


def _guidance_block():
    st.markdown(
        """
**If a wildfire is near you (≤ 40 km):**
- **Call 911** if you see fire/smoke threatening people or property, or if told to evacuate.
- **Contact your provincial forestry / wildfire line** to report details if safe.
- **Prepare a go-bag:** ID, meds/prescriptions, chargers, cash, water/snacks, clothing, sturdy shoes, flashlight, important docs, pet supplies, masks (N95), glasses.
- **Get ready to leave quickly:** keep your vehicle fueled, park facing the road, know **two ways out**, share plans with family.
- **Harden your home (only if safe):** close windows/vents, remove flammables from deck, wet vegetation, keep hoses ready.
- **Activate the SAFER fire alert** for critical proximity alerts.
        """
    )


# ---------- TABS ----------
t1, t2, t3 = st.tabs(["🔥 Active Fires", "🧾 Incident Brief", "🚨 SAFER Fire Alert"])

# ===== TAB 1: ACTIVE FIRES =====
with t1:
    st.subheader("Active Fires in the Acadian Region")

    ss = st.session_state
    ss.setdefault("fires_payload", None)
    ss.setdefault("fires_html", None)

    left, right = st.columns([1, 1])

    # ---------------- LEFT: summary table ----------------
    with left:
        if st.button("Fetch Active Fires", type="primary", disabled=not bool(fires_url)):
            try:
                data = post_json(
                    fires_url, {"from": "streamlit"}, shared_secret or None, timeout=timeout_sec
                )
                html = data.get("summary_html")
                if isinstance(html, str) and html.strip():
                    # cache & rerun so we render in one place below
                    ss["fires_payload"] = data
                    ss["fires_html"] = html
                    st.success("Received response from n8n")
                    st.rerun()
                else:
                    st.write(
                        data.get("summary") or data.get("summary_text") or "(No summary returned)"
                    )
                ss["fires_payload"] = data
                ss["fires_html"] = html
                st.success("Received response from n8n")
            except requests.HTTPError as e:
                st.error(f"HTTP error: {e.response.status_code} {e.response.text[:400]}")
            except Exception as e:
                st.error(f"Failed: {e}")

        if ss.get("fires_html"):
            components.html(ss["fires_html"], height=1200, scrolling=True)

    # ---------------- RIGHT: Q&A ----------------
    with right:
        st.markdown("#### Ask about today’s fires")

        # Helpers for fires list
        def _prov(f):
            return (f.get("agency") or "").strip().upper()

        def _ctrl_text(f):
            return (f.get("control") or f.get("status") or "").strip().lower()

        def _sizeha(f):
            try:
                return float(f.get("size_ha") or 0.0)
            except Exception:
                return 0.0

        def _started_s(f):
            return (str(f.get("started") or "")[:10]).strip()
        def _status_to_color(ctrl: str | None):
            s = (ctrl or "").lower()
            if "out" in s:
                return [220, 38, 38]  # red
            if "held" in s:
                return [234, 179, 8]  # yellow
            if "under" in s:
                return [16, 185, 129]  # green
            return [107, 114, 128]  # gray

        def _date_iso(s):
            try:
                return dt.date.fromisoformat((s or "")[:10])
            except Exception:
                return None

        def fmt_fire_line(f, show_km=False):
            base = f"- {f.get('name','(id?)')} — {_prov(f)} · {_sizeha(f):,.1f} ha · {f.get('control','—')}"
            if show_km and f.get('_dist_km') is not None:
                base += f" · {f['_dist_km']:.1f} km"
            base += f" · Started {_started_s(f) or '—'}"
            return base

        # numbers in words → ints
        _NUM_WORDS = {
            "one": 1,
            "two": 2,
            "three": 3,
            "four": 4,
            "five": 5,
            "six": 6,
            "seven": 7,
            "eight": 8,
            "nine": 9,
            "ten": 10,
            "eleven": 11,
            "twelve": 12,
            "thirteen": 13,
            "fourteen": 14,
            "fifteen": 15,
            "sixteen": 16,
            "seventeen": 17,
            "eighteen": 18,
            "nineteen": 19,
            "twenty": 20,
        }

        def _num_from_words(t: str) -> str:
            return re.sub(
                r"\b(" + "|".join(_NUM_WORDS.keys()) + r")\b",
                lambda m: str(_NUM_WORDS[m.group(1)]),
                t or "",
            )

        # size phrases
        def parse_size_range(text: str):
            t = (text or "").lower()
            # capture numbers with or without unit; allow "100ha", "100 ha", "100"
            nums = [float(x) for x in re.findall(r"(\d+(?:\.\d+)?)\s*(?:ha)?", t)]
            # Normalize phrasing
            if "between" in t and "and" in t and len(nums) >= 2:
                a, b = sorted(nums[:2])
                return ("between", a, b)
            if any(
                w in t
                for w in [
                    "over",
                    "more than",
                    ">",
                    ">=",
                    "at least",
                    "minimum",
                    "greater than",
                    "bigger than",
                    "larger than",
                    "above",
                ]
            ):
                return ("min", nums[0]) if nums else None
            if any(
                w in t
                for w in [
                    "under",
                    "less than",
                    "<",
                    "<=",
                    "at most",
                    "maximum",
                    "below",
                    "smaller than",
                    "lesser than",
                ]
            ):
                return ("max", nums[0]) if nums else None
            # patterns like ">= 50ha", "<=100"
            m = re.search(r"([<>]=?)\s*(\d+(?:\.\d+)?)\s*(?:ha)?", t)
            if m:
                op, val = m.group(1), float(m.group(2))
                if op.startswith(">"):
                    return ("min", val)
                else:
                    return ("max", val)
            return None

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

        st.button("Use example", key="use_example", on_click=_copy_example_to_input)

        q = st.text_input(
            "Your question",
            key="q_fires",
            placeholder=(
                "e.g., fires near Halifax • within 40 km of Truro • closest to Moncton • "
                "top 4 largest in NB • totals by province • where is fire 68622 • "
                "how far is fire 68622 from Halifax • started last 7 days • older than 3 days"
            ),
        )

        ask = st.button("Ask", key="ask_fires", disabled=not bool(fires_url))

        def _ensure_fires() -> List[Dict[str, Any]]:
            raw = ss.get("fires_payload")
            if not raw:
                raw = post_json(
                    fires_url, {"from": "qa"}, shared_secret or None, timeout=timeout_sec
                )
                ss["fires_payload"] = raw
            return raw.get("fires") or []

        def _geo_place(place: str) -> Optional[Tuple[float, float, str]]:
            g = geocode_address(place, opencage_key, google_key)
            if not g:
                return None
            return float(g[0]), float(g[1]), g[2]

        def _find_fire_by_id(fires: List[Dict[str, Any]], fid: str) -> Optional[Dict[str, Any]]:
            fid = fid.strip()
            for f in fires:
                if str(f.get("id") or f.get("name") or "").strip() == fid:
                    return f
            # Also try substring match of name
            for f in fires:
                if fid in str(f.get("name") or ""):
                    return f
            return None

        def _maybe_filter_province(fires: List[Dict[str, Any]], ql: str) -> List[Dict[str, Any]]:
            prov_map = {
                "NB": ["nb", "new brunswick"],
                "NS": ["ns", "nova scotia"],
                "PE": ["pe", "pei", "prince edward"],
                "NL": ["nl", "newfoundland", "labrador"],
            }
            for code, keys in prov_map.items():
                if any(k in ql for k in keys):
                    return [f for f in fires if _prov(f) == code]
            return fires

        def answer_fire_question(q: str):
            if not q.strip():
                st.info("Type a question first.")
                return
            fires = _ensure_fires()
            if not fires:
                st.warning("No fire data returned.")
                return

            qn = _num_from_words(q).strip()
            ql = qn.lower()
            out: List[Dict[str, Any]] = []

            # Filters
            fires2 = _maybe_filter_province(fires, ql)

            # Out of control
            if "out of control" in ql or "out-of-control" in ql or "uncontrolled" in ql:
                out = [f for f in fires2 if "out" in _ctrl_text(f)]
                if not out:
                    st.write("No out-of-control fires found in the selection.")
                    return
                out.sort(key=lambda f: -_sizeha(f))
                for f in out[:20]:
                    st.markdown(fmt_fire_line(f))
                return

            # Top N largest
            m_top = re.search(r"top\s+(\d+)", ql)
            if m_top:
                n = int(m_top.group(1))
                fires2.sort(key=lambda f: -_sizeha(f))
                for f in fires2[: max(1, n)]:
                    st.markdown(fmt_fire_line(f))
                return

            # Size filters
            rng = parse_size_range(ql)
            if rng:
                kind = rng[0]
                if kind == "between":
                    a, b = rng[1], rng[2]
                    out = [f for f in fires2 if a <= _sizeha(f) <= b]
                elif kind == "min":
                    a = rng[1]
                    out = [f for f in fires2 if _sizeha(f) >= a]
                elif kind == "max":
                    a = rng[1]
                    out = [f for f in fires2 if _sizeha(f) <= a]
                out.sort(key=lambda f: -_sizeha(f))
                if not out:
                    st.write("No fires matched that size filter.")
                else:
                    for f in out[:50]:
                        st.markdown(fmt_fire_line(f))
                return

            # Started last/older than X days
            m_last = re.search(r"started\s+last\s+(\d+)\s+day", ql)
            m_old = re.search(r"older\s+than\s+(\d+)\s+day", ql)
            today = dt.date.today()
            if m_last:
                d = int(m_last.group(1))
                cutoff = today - dt.timedelta(days=d)
                out = [f for f in fires2 if (_date_iso(_started_s(f)) or today) >= cutoff]
                for f in out:
                    st.markdown(fmt_fire_line(f))
                return
            if m_old:
                d = int(m_old.group(1))
                cutoff = today - dt.timedelta(days=d)
                out = [f for f in fires2 if (_date_iso(_started_s(f)) or today) < cutoff]
                for f in out:
                    st.markdown(fmt_fire_line(f))
                return

            # within X km of PLACE / closest to PLACE
            m_within = re.search(r"within\s+(\d+)\s*km\s+of\s+(.+)$", ql)
            m_close = re.search(r"closest\s+to\s+(.+)$", ql)
            if m_within or m_close:
                if m_within:
                    R_km = float(m_within.group(1))
                    place = m_within.group(2).strip()
                else:
                    R_km = float("inf")
                    place = m_close.group(1).strip()
                g = _geo_place(place)
                if not g:
                    st.warning("Could not geocode that place. Try adding province/postal code.")
                    return
                lat0, lon0, label = g
                cands = []
                for f in fires2:
                    try:
                        lat, lon = float(f.get("lat")), float(f.get("lon"))
                    except Exception:
                        continue
                    dkm = haversine_km(lat0, lon0, lat, lon)
                    if dkm is None:
                        continue
                    if dkm <= R_km:
                        f2 = dict(f)
                        f2["_dist_km"] = dkm
                        cands.append(f2)
                cands.sort(key=lambda x: x.get("_dist_km", 9e9))
                if not cands:
                    st.write(f"No active fires within {R_km:.0f} km of {label}.")
                    return
                st.write(f"Closest fires to **{label}**:")
                for f in cands[:20]:
                    st.markdown(fmt_fire_line(f, show_km=True))
                # quick map
                view = pdk.ViewState(latitude=lat0, longitude=lon0, zoom=7)
                plot_data = [
                    {
                        "lat": f.get("lat"),
                        "lon": f.get("lon"),
                        "color": _status_to_color(f.get("control")),
                    }
                    for f in cands[:200]
                ]
                layer = pdk.Layer(
                    "ScatterplotLayer",
                    data=plot_data,
                    get_position=["lon", "lat"],
                    get_fill_color="color",
                    get_radius=1500,
                    radius_min_pixels=4,
                    radius_max_pixels=32,
                )
                st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view), use_container_width=True)
                st.caption("Legend: 🔴 Out of Control · 🟡 Being Held · 🟢 Under Control")

            # Quick links / details
            cA, cB = st.columns(2)
            if (data or {}).get("map_link"):
                cA.link_button("Open in Google Maps", data["map_link"])
            with cB:
                with st.popover("Details"):
                    st.json(incident)

# ===== TAB 3: SAFER Fire Alert =====
with t3:
    st.subheader(
        "Be SAFER in the Acadian region with a fire alert response for your home address"
    )
    st.write(
        "Enter your **address** (optional). We’ll geocode it to coordinates, or you can set lat/lon manually."
    )

    if not subscribe_url:
        st.warning(
            "Subscribe webhook URL is not configured. Add N8N_SUBSCRIBE_URL in **App → Settings → Secrets**."
        )

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
        lat = colA.number_input(
            "Latitude", value=float(ss["sub_lat"]), step=0.0001, format="%.6f"
        )
        lon = colB.number_input(
            "Longitude", value=float(ss["sub_lon"]), step=0.0001, format="%.6f"
        )
        radius = st.number_input(
            "Radius (km)", min_value=1, max_value=250, value=int(ss["sub_radius"]), step=1
        )

        btn_label = "Cancel Alerts" if ss.get("alerts_active") else "Activate Alerts"
        toggle_clicked = st.form_submit_button(
            btn_label, type="primary", disabled=not bool(subscribe_url)
        )

    # persist
    ss["sub_email"], ss["sub_address"] = email, address
    ss["sub_lat"], ss["sub_lon"], ss["sub_radius"] = float(lat), float(lon), int(radius)

    # geocode button handler
    if geocode_clicked:
        if not (opencage_key or google_key):
            st.error(
                "Please add at least one geocoding key (OpenCage or Google) in **App → Settings → Secrets**."
            )
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
        if not _valid_email(st.session_state.get("sub_email", "")):
            errs.append("Please enter a valid email.")
        if abs(float(st.session_state["sub_lat"])) > 90:
            errs.append("Latitude must be between -90 and 90.")
        if abs(float(st.session_state["sub_lon"])) > 180:
            errs.append("Longitude must be between -180 and 180.")
        if not (1 <= int(st.session_state["sub_radius"]) <= 250):
            errs.append("Radius must be 1–250 km.")
        if errs:
            for e in errs:
                st.error(e)
            return

        email = st.session_state["sub_email"].strip()
        address = st.session_state["sub_address"].strip()
        lat_val, lon_val = float(st.session_state["sub_lat"]), float(
            st.session_state["sub_lon"]
        )

        if (opencage_key or google_key) and address:
            g = geocode_address(address, opencage_key, google_key)
            if g:
                lat_val, lon_val, fmt, _ = g
                st.session_state["sub_lat"], st.session_state["sub_lon"], st.session_state[
                    "sub_address"
                ] = lat_val, lon_val, fmt
                address = fmt

        body = {
            "email": email,
            "lat": lat_val,
            "lon": lon_val,
            "radius_km": int(st.session_state["sub_radius"]),
            "address": address,
            "active": True,
            "from": "streamlit",
        }
        resp = post_json(subscribe_url, body, shared_secret or None, timeout=timeout_sec)
        st.success(f'Alerts activated for "{email}".')
        st.json(resp)
        st.session_state["alerts_active"] = True
        st.rerun()

    def _unsubscribe():
        email = st.session_state.get("sub_email", "").strip()
        if not _valid_email(email):
            st.error("Please enter a valid email to cancel alerts.")
            return
        body = {"email": email, "active": False, "from": "streamlit"}
        resp = post_json(subscribe_url, body, shared_secret or None, timeout=timeout_sec)
        st.success(f'Alerts canceled for "{email}".')
        st.json(resp)
        st.session_state["alerts_active"] = False
        st.rerun()

    # click handlers
    if toggle_clicked and subscribe_url:
        if st.session_state.get("alerts_active"):
            _unsubscribe()
        else:
            _subscribe()
