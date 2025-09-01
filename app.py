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


def load_config() -> Dict[str, Any]:
    return {
        "FIRE_URL": _get_secret("N8N_FIRES_URL", ""),
        "AGENT_URL": _get_secret("N8N_AGENT_URL", ""),
        "RISK_URL": _get_secret("N8N_RISK_URL", ""),
        "SUBSCRIBE_URL": _get_secret("N8N_SUBSCRIBE_URL", ""),
        "SHARED_SECRET": _get_secret("N8N_SHARED_SECRET", ""),
        "OPENCAGE_KEY": _get_secret("OPENCAGE_API_KEY", ""),
        "GOOGLE_KEY": _get_secret("GOOGLE_GEOCODING_API_KEY", ""),
        "TIMEOUT_SEC": int(_get_secret("REQUEST_TIMEOUT_SEC", "60")),
    }


cfg = load_config()
fires_url = cfg["FIRE_URL"]
agent_url = cfg["AGENT_URL"]
risk_url = cfg["RISK_URL"]
subscribe_url = cfg["SUBSCRIBE_URL"]
shared_secret = cfg["SHARED_SECRET"]
timeout_sec = cfg["TIMEOUT_SEC"]
opencage_key = cfg["OPENCAGE_KEY"]
google_key = cfg["GOOGLE_KEY"]

DEFAULT_CITIES = [
    "Fredericton, NB, Canada",
    "Moncton, NB, Canada",
    "Saint John, NB, Canada",
    "Bathurst, NB, Canada",
    "Miramichi, NB, Canada",
    "Charlottetown, PE, Canada",
    "Summerside, PE, Canada",
    "Halifax, NS, Canada",
    "Dartmouth, NS, Canada",
    "Sydney, NS, Canada",
    "Yarmouth, NS, Canada",
    "Truro, NS, Canada",
    "Gander, NL, Canada",
    "Corner Brook, NL, Canada",
    "St. John’s, NL, Canada",
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


def _fokabs_logo(height: int = 44) -> str:
    if os.path.exists(IMG_PATH):
        ext = os.path.splitext(IMG_PATH)[1].lower()
        mime = "image/png" if ext == ".png" else "image/jpeg"
        b64 = base64.b64encode(open(IMG_PATH, "rb").read()).decode("ascii")
        return (
            f'<img src="data:{mime};base64,{b64}" alt="FOKABS" '
            f'style="height:{height}px;width:auto;vertical-align:baseline;border-radius:10px;'
            f'padding:6px 10px;background:#fff;box-shadow:0 1px 3px rgba(0,0,0,.08);" />'
        )
    return "🌐"


_HEADER = (
    '<div class="s-header"><div class="s-title">'
    f'<span class="s-acronym">SAFER</span>{_fokabs_logo(44)}'
    "</div><div class=\"s-sub\">Sustainable Acadian Forests &amp; Environmental Risks</div>"
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
                return float(loc["lat"]), float(loc["lng"]), gg.get("formatted_address") or address, "google"
        if oc_key:
            oc = _opencage_geocode(address, oc_key)
            if oc:
                return float(oc["geometry"]["lat"]), float(oc["geometry"]["lng"]), oc.get("formatted") or address, "opencage"
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


def haversine_km(lat1, lon1, lat2, lon2) -> Optional[float]:
    if None in (lat1, lon1, lat2, lon2):
        return None
    R = 6371.0
    p = math.pi / 180.0
    dlat = (lat2 - lat1) * p
    dlon = (lon2 - lon1) * p
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1 * p) * math.cos(lat2 * p) * math.sin(dlon / 2) ** 2
    return R * 2 * math.asin(math.sqrt(a))


def _status_to_color(ctrl: Optional[str]) -> List[int]:
    s = (ctrl or "").lower()
    if "out" in s:  # out of control
        return [220, 38, 38]  # red
    if "held" in s:
        return [234, 179, 8]  # yellow
    if "under" in s:  # under control
        return [16, 185, 129]  # green
    return [107, 114, 128]  # gray


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


# ---------- UI TABS ----------
t1, t2, t3 = st.tabs(["🔥 Active Fires", "🧾 Incident Brief", "🚨 SAFER Fire Alert"])

# ===== TAB 1: ACTIVE FIRES =====
with t1:
    st.subheader("Active Fires in the Acadian Region")

    ss = st.session_state
    ss.setdefault("fires_payload", None)
    ss.setdefault("fires_html", None)

    left, right = st.columns([1, 1])

    # ---------------- LEFT: summary + map ----------------
    with left:
        if st.button("Fetch Active Fires", type="primary", disabled=not bool(fires_url)):
            try:
                data = post_json(fires_url, {"from": "streamlit"}, shared_secret or None, timeout=timeout_sec)
                html = data.get("summary_html")
                ss["fires_payload"], ss["fires_html"] = data, html
                if isinstance(html, str) and html.strip():
                    components.html(html, height=820, scrolling=True)
                else:
                    st.write(data.get("summary") or data.get("summary_text") or "(No summary returned)")
                st.success("Received response from n8n")
            except requests.HTTPError as e:
                st.error(f"HTTP error: {e.response.status_code} {e.response.text[:400]}")
            except Exception as e:
                st.error(f"Failed: {e}")

        if ss.get("fires_html"):
            components.html(ss["fires_html"], height=820, scrolling=True)

        st.divider()
        st.markdown("**Map of active fires (colored pins)**")
        # Build map data from payload if present
        fires_list: List[Dict[str, Any]] = []
        payload = ss.get("fires_payload") or {}
        if isinstance(payload, dict):
            if isinstance(payload.get("fires"), list):
                fires_list = payload["fires"]
            elif isinstance(payload.get("items"), list):
                fires_list = payload["items"]
        # Normalize & filter for mappable points
        map_rows = []
        for f in fires_list or []:
            try:
                lat = float(f.get("lat"))
                lon = float(f.get("lon"))
            except Exception:
                continue
            ctrl = f.get("control") or f.get("status") or ""
            color = _status_to_color(ctrl)
            size_ha = f.get("size_ha")
            try:
                size_ha = float(size_ha) if size_ha is not None else None
            except Exception:
                size_ha = None
            map_rows.append({
                "lat": lat,
                "lon": lon,
                "name": f.get("name") or f.get("id") or "(id?)",
                "control": ctrl,
                "size_ha": size_ha if size_ha is not None else "—",
                "color": color,
            })
        if map_rows:
            center_lat = sum(r["lat"] for r in map_rows) / len(map_rows)
            center_lon = sum(r["lon"] for r in map_rows) / len(map_rows)
            view = pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=5.8)
            layer = pdk.Layer(
                "ScatterplotLayer",
                data=map_rows,
                get_position=["lon", "lat"],
                get_fill_color="color",
                get_radius=1600,
                radius_min_pixels=4,
                radius_max_pixels=40,
                pickable=True,
            )
            deck = pdk.Deck(
                layers=[layer],
                initial_view_state=view,
                tooltip={"text": "{name}\n{control}\n{size_ha} ha"},
            )
            st.pydeck_chart(deck)
            st.caption("Legend: 🔴 Out of Control · 🟡 Being Held · 🟢 Under Control · ⚪ Unknown")
        else:
            st.caption("Fetch fires to populate the map.")

    # ---------------- RIGHT: Q&A ----------------
    with right:
        st.markdown("#### Ask about today’s fires")

        # Helpers for fires list
        def _prov(f: Dict[str, Any]) -> str:
            return (f.get("agency") or "").strip().upper()

        def _ctrl_text(f: Dict[str, Any]) -> str:
            return (f.get("control") or f.get("status") or "").strip().lower()

        def _sizeha(f: Dict[str, Any]) -> float:
            try:
                return float(f.get("size_ha") or 0.0)
            except Exception:
                return 0.0

        def _started_s(f: Dict[str, Any]) -> str:
            return (str(f.get("started") or "")[:10]).strip()

        def _date_iso(s: Optional[str]) -> Optional[dt.date]:
            try:
                return dt.date.fromisoformat((s or "")[:10])
            except Exception:
                return None

        def fmt_fire_line(f: Dict[str, Any], show_km: bool = False) -> str:
            base = f"- {f.get('name','(id?)')} — {_prov(f)} · {_sizeha(f):,.1f} ha · {f.get('control','—')}"
            if show_km and f.get("_dist_km") is not None:
                base += f" · {f['_dist_km']:.1f} km"
            base += f" · Started {_started_s(f) or '—'}"
            return base

        # numbers in words → ints
        _NUM_WORDS = {
            "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
            "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
            "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
            "nineteen": 19, "twenty": 20,
        }

        def _num_from_words(t: str) -> str:
            return re.sub(r"\b(" + "|".join(_NUM_WORDS.keys()) + r")\b", lambda m: str(_NUM_WORDS[m.group(1)]), t or "")

        # size phrases
        def parse_size_range(text: str):
            t = (text or "").lower()
            nums = [float(x) for x in re.findall(r"(\d+(?:\.\d+)?)\s*(?:ha)?", t)]
            if "between" in t and "and" in t and len(nums) >= 2:
                a, b = sorted(nums[:2]); return ("between", a, b)
            if any(w in t for w in ["over","more than",">=",">","at least","minimum","greater than","bigger than","larger than","above"]):
                return ("min", nums[0]) if nums else None
            if any(w in t for w in ["under","less than","<=","<","at most","maximum","below","smaller than","lesser than"]):
                return ("max", nums[0]) if nums else None
            m = re.search(r"([<>]=?)\s*(\d+(?:\.\d+)?)\s*(?:ha)?", t)
            if m:
                op, val = m.group(1), float(m.group(2))
                return ("min", val) if op.startswith(">") else ("max", val)
            return None

        # date windows
        def parse_days_window(text: str):
            s2 = _num_from_words((text or "").lower())
            today = dt.date.today()
            m = re.search(r"started\s+last\s+(\d+)\s+day", s2)
            if m:
                d = int(m.group(1)); cutoff = today - dt.timedelta(days=d)
                return ("recent", cutoff)
            m = re.search(r"older\s+than\s+(\d+)\s+day", s2)
            if m:
                d = int(m.group(1)); cutoff = today - dt.timedelta(days=d)
                return ("older", cutoff)
            return (None, None)

        # GEO intent parsing
        def parse_geo(text: str):
            t = (text or "").lower().strip()
            m = re.search(r"within\s+(\d+(?:\.\d+)?)\s*km\s+(?:of|from)\s+(.+)", t)
            if m: return ("within", float(m.group(1)), m.group(2).strip())
            m = re.search(r"(?:top\s+(\d+)\s+)?(?:closest|nearest)(?:\s+fire)?\s+(?:to|near)\s+(.+)", t)
            if m: return ("closest", int(m.group(1) or 1), m.group(2).strip())
            m = re.search(r"(?:near|nearby|close to)\s+(.+)", t)
            if m: return ("within", 40.0, m.group(1).strip())
            return (None, None, None)

        def geocode_place(name: str) -> Optional[Tuple[float, float, str]]:
            qtext = (name or "").strip()
            if not qtext:
                return None
            # allow direct lat,lon
            ll = _parse_latlon(qtext)
            if ll:
                return float(ll[0]), float(ll[1]), f"{ll[0]:.4f}, {ll[1]:.4f}"
            # scrub phrases like "within 40 km"
            qtext = re.sub(r"\bwithin\s+\d+(?:\.\d+)?\s*km\b", "", qtext, flags=re.I).strip(",.;: ")
            try:
                g = geocode_address(qtext, opencage_key, google_key)
                if g:
                    return float(g[0]), float(g[1]), g[2]
            except Exception:
                return None
            return None

        # province filter
        def _maybe_filter_province(fires: List[Dict[str, Any]], ql: str) -> List[Dict[str, Any]]:
            if any(w in ql for w in [" in nb", " in new brunswick"]):
                return [f for f in fires if _prov(f) == "NB"]
            if any(w in ql for w in [" in ns", " in nova scotia"]):
                return [f for f in fires if _prov(f) == "NS"]
            if any(w in ql for w in [" in pe", " in pei", " in prince edward"]):
                return [f for f in fires if _prov(f) == "PE"]
            if any(w in ql for w in [" in nl", " in newfoundland", " in labrador"]):
                return [f for f in fires if _prov(f) == "NL"]
            return fires

        def _find_fire_by_id(fires: List[Dict[str, Any]], fid: str) -> Optional[Dict[str, Any]]:
            fid = re.sub(r"\D", "", str(fid or ""))
            return next((x for x in fires if str(x.get("name")) == fid or str(x.get("id")) == fid), None)

        # UI: examples
        examples = [
            "which fires are out of control?",
            "top 4 largest in NB",
            "fires within 40 km of Halifax",
            "what place is closest to fire 68586?",
            "when did fire 68622 start?",
            "largest fire in NS",
            "fires older than 3 days",
            "fires between 100 and 500 ha in NL",
        ]
        ex_sel = st.selectbox("Examples", options=examples, index=0, key="examples_q")
        if st.button("Use example", key="use_example"):
            st.session_state["q_fires"] = ex_sel
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

        # Retrieve fires list from payload when needed
        def _get_fires_from_payload() -> List[Dict[str, Any]]:
            payload = st.session_state.get("fires_payload") or {}
            fires = []
            if isinstance(payload, dict):
                if isinstance(payload.get("fires"), list):
                    fires = payload["fires"]
                elif isinstance(payload.get("items"), list):
                    fires = payload["items"]
            return [f for f in fires if isinstance(f, dict)]

        def answer_fire_question(q: str):
            fires = _get_fires_from_payload()
            if not fires:
                st.info("Fetch Active Fires first.")
                return

            qn = _num_from_words(q or "").strip()
            ql = qn.lower()

            fires2 = _maybe_filter_province(fires, ql)

            # 1) status filters
            if any(k in ql for k in ["out of control", "out-of-control", "uncontrolled"]):
                out = [f for f in fires2 if "out" in _ctrl_text(f)]
                if not out:
                    st.write("No out-of-control fires found in the selection.")
                    return
                out.sort(key=lambda f: -_sizeha(f))
                for f in out[:20]:
                    st.markdown(fmt_fire_line(f))
                return

            # 2) top N largest
            m_top = re.search(r"top\s+(\d+)", ql)
            if m_top:
                n = max(1, int(m_top.group(1)))
                fires2.sort(key=lambda f: -_sizeha(f))
                for f in fires2[: n]:
                    st.markdown(fmt_fire_line(f))
                return

            # 3) size range
            rng = parse_size_range(ql)
            if rng:
                kind = rng[0]
                out: List[Dict[str, Any]] = []
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

            # 4) recency
            kind, cutoff = parse_days_window(ql)
            if kind == "recent":
                out = [f for f in fires2 if (_date_iso(_started_s(f)) or dt.date.today()) >= cutoff]
                for f in out:
                    st.markdown(fmt_fire_line(f))
                return
            if kind == "older":
                out = [f for f in fires2 if (_date_iso(_started_s(f)) or dt.date.today()) < cutoff]
                for f in out:
                    st.markdown(fmt_fire_line(f))
                return

            # 5) geo — within X km of PLACE / closest to PLACE
            g_kind, g_a, g_b = parse_geo(ql)
            if g_kind in ("within", "closest"):
                R_km = float(g_a) if g_kind == "within" else float("inf")
                place = g_b
                g = geocode_place(place)
                if not g:
                    st.warning("Could not geocode that place. Try adding province/postal code.")
                    return
                lat0, lon0, label = g
                cands: List[Dict[str, Any]] = []
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
                if not cands and g_kind == "within":
                    st.write(f"No active fires within {R_km:.0f} km of {label}.")
                    return
                if not cands and g_kind == "closest":
                    st.write(f"No active fires found to compute closeness for {label}.")
                    return
                st.write(f"Closest fires to **{label}**:")
                for f in cands[:20]:
                    st.markdown(fmt_fire_line(f, show_km=True))
                # simple map for this subset
                view = pdk.ViewState(latitude=lat0, longitude=lon0, zoom=7)
                subset_rows = []
                for f in cands[:200]:
                    try:
                        subset_rows.append({
                            "lat": float(f.get("lat")),
                            "lon": float(f.get("lon")),
                            "name": f.get("name") or f.get("id") or "(id?)",
                            "color": _status_to_color(f.get("control") or f.get("status")),
                        })
                    except Exception:
                        pass
                if subset_rows:
                    layer = pdk.Layer(
                        "ScatterplotLayer",
                        data=subset_rows,
                        get_position=["lon", "lat"],
                        get_fill_color="color",
                        get_radius=1200,
                        radius_min_pixels=4,
                        radius_max_pixels=30,
                    )
                    st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view, tooltip={"text": "{name}"}))
                return

            # 6) how far is fire <id> from <place>
            m_howfar = re.search(r"how\s+far\s+(?:is\s+)?(?:fire\s+)?(\d{3,6})\s+(?:from|to)\s+(.+)", ql)
            if m_howfar:
                fid, place = m_howfar.group(1), m_howfar.group(2).strip()
                f = _find_fire_by_id(fires2, fid)
                if not f:
                    st.write("Couldn't find that fire.")
                    return
                g = geocode_place(place)
                if not g:
                    st.write("Couldn't geocode that place.")
                    return
                lat0, lon0, label = g
                try:
                    dkm = haversine_km(lat0, lon0, float(f.get("lat")), float(f.get("lon")))
                except Exception:
                    dkm = None
                st.write(
                    f"Fire {fid} is {dkm:.1f} km from {label}." if dkm is not None else "Distance unavailable."
                )
                st.markdown(fmt_fire_line(f))
                return

            # 7) when did fire <id> start
            m_start = re.search(r"(?:when\s+did\s*(?:fire\s*)?([0-9\-\s]+)\s*start|start\s*date\s*for\s*([0-9\-\s]+))", ql)
            if m_start:
                fid = re.sub(r"\D", "", (m_start.group(1) or m_start.group(2) or ""))
                f = _find_fire_by_id(fires2, fid)
                if not f:
                    st.write(f"Fire {fid} not found.")
                else:
                    st.markdown(f"**Start date for fire {fid}: {_started_s(f) or '—'}**")
                    st.write(fmt_fire_line(f))
                return

            # fallback — counts by province
            by_p: Dict[str, int] = {}
            for f in fires2:
                by_p.setdefault(_prov(f) or "—", 0)
                by_p[_prov(f) or "—"] += 1
            st.write("Try one of the example questions above. Here's a quick count by province:")
            st.json(by_p)

        agent_url = _get_secret("N8N_AGENT_URL", "")  # new secret for the AI Agent webhook

if ask:
    try:
        if agent_url:
            # Send the user’s text to the agent
            res = post_json(agent_url, {"q": q, "question": q}, shared_secret or None, timeout=timeout_sec)
            md = res.get("answer_md") or "_No answer_"
            st.markdown(md)

            # If agent returns map markers, plot them
            marks = res.get("markers") or []
            if marks:
                view = pdk.ViewState(latitude=float(marks[0]["lat"]),
                                     longitude=float(marks[0]["lon"]),
                                     zoom=6)
                layer = pdk.Layer(
                    "ScatterplotLayer",
                    data=[{"lat": m["lat"], "lon": m["lon"], "name": m["label"]} for m in marks],
                    get_position=["lon", "lat"],
                    get_radius=1200,
                    radius_min_pixels=4,
                    radius_max_pixels=30,
                    pickable=True,
                )
                st.pydeck_chart(pdk.Deck(layers=[layer],
                                         initial_view_state=view,
                                         tooltip={"text": "{name}"}))
        else:
            # fallback to the old local parser
            answer_fire_question(q)
    except requests.HTTPError as e:
        st.error(f"HTTP error: {e.response.status_code} {e.response.text[:400]}")
    except Exception as e:
        st.error(f"Failed to answer: {e}")

        st.divider()
        st.markdown("#### Safety check (40 km)")
        loc_in = st.text_input(
            "Your community or coordinates",
            placeholder=(
                "e.g. Halifax NS  |  Moncton  |  44.65,-63.57  • Tip: include your postal code for best accuracy (e.g. B3H 1X1)"
            ),
            key="safety_place",
        )
        if st.button("Check proximity", disabled=not bool(ss.get("fires_payload"))):
            fires = _get_fires_from_payload()
            g = None
            if _parse_latlon(loc_in or ""):
                lat, lon = _parse_latlon(loc_in)
                g = (float(lat), float(lon), f"{lat:.4f}, {lon:.4f}")
            elif (loc_in or "").strip():
                gg = geocode_address(loc_in, opencage_key, google_key)
                if gg:
                    g = (float(gg[0]), float(gg[1]), gg[2])
            if not g:
                st.warning("Couldn't locate that place.")
            else:
                lat0, lon0, label = g
                near = []
                for f in fires:
                    try:
                        dkm = haversine_km(lat0, lon0, float(f.get("lat")), float(f.get("lon")))
                        if dkm is not None and dkm <= 40.0:
                            f2 = dict(f); f2["_dist_km"] = dkm; near.append(f2)
                    except Exception:
                        pass
                if not near:
                    st.success(f"No active fires within 40 km of {label}.")
                else:
                    st.error(f"{len(near)} fire(s) within 40 km of {label}:")
                    near.sort(key=lambda x: x.get("_dist_km", 9e9))
                    for f in near:
                        st.markdown(fmt_fire_line(f, show_km=True))
                    _guidance_block()

# ===== TAB 2: INCIDENT BRIEF =====
with t2:
    st.subheader("Incident Brief")

    if not risk_url:
        st.warning("Incident brief webhook is not configured. Set N8N_RISK_URL in App → Settings → Secrets.")
    else:
        ss = st.session_state
        mode = st.radio("Find by", ["Fire ID", "Location"], horizontal=True, key="brief_mode")
        with st.form("brief_form", clear_on_submit=False):
            payload: Optional[Dict[str, Any]] = None
            if mode == "Fire ID":
                fire_id = st.text_input("Fire ID (e.g. 68622)", key="brief_id")
                if fire_id.strip():
                    payload = {"id": re.sub(r"\D", "", fire_id.strip())}
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

        if submitted and payload:
            try:
                data = post_json(risk_url, payload, shared_secret or None, timeout=timeout_sec)
                incident = (data or {}).get("incident") or {}
                brief_md = (data or {}).get("brief_md") or "_No brief returned_"
                ss["brief_data"], ss["brief_incident"], ss["brief_md"] = data, incident, brief_md
            except requests.HTTPError as e:
                st.error(f"HTTP error: {e.response.status_code} {e.response.text[:400]}")
            except Exception as e:
                st.error(f"Error fetching brief: {e}")
        elif submitted and not payload:
            st.warning("Enter a Fire ID or a location first.")

        data = ss.get("brief_data")
        incident = ss.get("brief_incident", {})
        brief_md = ss.get("brief_md")

        if data and incident and brief_md:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Tier", data.get("tier_main", data.get("tier", "—")), data.get("tier_sub", ""))
            m2.metric("Control", incident.get("control", "—"))
            try:
                m3.metric("Size (ha)", int(incident.get("size_ha") or 0))
            except Exception:
                m3.metric("Size (ha)", incident.get("size_ha", "—"))
            m4.metric("Started", incident.get("started", "—"))

            st.markdown(brief_md)

            if "lat" in incident and "lon" in incident:
                st.markdown("**Map**")
                view = pdk.ViewState(latitude=float(incident["lat"]), longitude=float(incident["lon"]), zoom=8, pitch=0)
                _col = _status_to_color(incident.get("control"))
                layer = pdk.Layer(
                    "ScatterplotLayer",
                    data=[{"lat": incident["lat"], "lon": incident["lon"], "color": _col}],
                    get_position=["lon", "lat"],
                    get_fill_color="color",
                    get_radius=1600,
                    radius_min_pixels=6,
                    radius_max_pixels=60,
                    pickable=False,
                )
                st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view))

                # Optional: Nearest places (Google Places)
                def _places_nearby(lat: float, lon: float, radius_m: int, api_key: str, types: Tuple[str, ...] = ("locality", "sublocality", "neighborhood")) -> List[Dict[str, Any]]:
                    url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
                    out: List[Dict[str, Any]] = []
                    seen: set = set()
                    for t in types:
                        params = {"location": f"{lat},{lon}", "radius": int(radius_m), "type": t, "key": api_key}
                        r = requests.get(url, params=params, timeout=20)
                        r.raise_for_status()
                        for it in (r.json() or {}).get("results", []):
                            pid = it.get("place_id")
                            if not pid or pid in seen:
                                continue
                            seen.add(pid)
                            loc = (it.get("geometry") or {}).get("location") or {}
                            plat, plon = float(loc.get("lat", 0)), float(loc.get("lng", 0))
                            # haversine
                            R = 6371.0; p = math.pi/180.0
                            dlat = (plat - float(lat)) * p
                            dlon = (plon - float(lon)) * p
                            a = math.sin(dlat/2)**2 + math.cos(float(lat)*p)*math.cos(plat*p)*math.sin(dlon/2)**2
                            dist_km = 2 * R * math.asin(math.sqrt(a))
                            out.append({"name": it.get("name") or "(unnamed)", "types": it.get("types") or [], "lat": plat, "lon": plon, "distance_km": round(dist_km, 2)})
                    out.sort(key=lambda x: x["distance_km"])
                    return out

                st.divider()
                st.markdown("**Nearest places (Google)**")
                if not google_key:
                    st.info("Add GOOGLE_GEOCODING_API_KEY in App → Settings → Secrets to enable this.")
                else:
                    c1, c2 = st.columns([2, 2])
                    with c1:
                        radius_choices = st.multiselect("Show closest within (km)", options=[1, 5, 10, 20, 30, 40, 80], default=[20, 40])
                    with c2:
                        type_choices = st.multiselect("Place types", options=["locality", "sublocality", "neighborhood", "point_of_interest", "establishment"], default=["locality", "sublocality", "neighborhood"])
                    if st.button("Find nearest places"):
                        lat_i, lon_i = float(incident["lat"]), float(incident["lon"])
                        max_r = (max(radius_choices) if radius_choices else 40) * 1000
                        places = _places_nearby(lat_i, lon_i, int(max_r), google_key, tuple(type_choices))
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
            "Address (optional)", value=ss["sub_address"], placeholder="123 Main St, Halifax, NS B3H 2Y9",
        )
        geocode_clicked = c_addr[1].form_submit_button(
            "Geocode", use_container_width=True, disabled=not bool(opencage_key or google_key),
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

    # toggle alerts
    if toggle_clicked and subscribe_url:
        if not _valid_email(email):
            st.error("Please enter a valid email.")
        else:
            try:
                body = {
                    "email": email,
                    "lat": float(ss["sub_lat"]),
                    "lon": float(ss["sub_lon"]),
                    "radius_km": int(ss["sub_radius"]),
                    "active": not bool(ss.get("alerts_active")),
                }
                res = post_json(subscribe_url, body, shared_secret or None, timeout=timeout_sec)
                ok = bool(res) and (res.get("ok") is True or res.get("status") in ("ok", "success"))
                ss["alerts_active"] = not ss.get("alerts_active") if ok else ss.get("alerts_active")
                if ok:
                    st.success("Alerts " + ("activated" if ss["alerts_active"] else "canceled") + ".")
                else:
                    st.warning("Request sent, but the server did not confirm success.")
            except requests.HTTPError as e:
                st.error(f"HTTP error: {e.response.status_code} {e.response.text[:400]}")
            except Exception as e:
                st.error(f"Failed to toggle alerts: {e}")
