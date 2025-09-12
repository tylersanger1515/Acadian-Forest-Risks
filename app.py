# ---------- IMPORTS ----------
from __future__ import annotations
import os, json, base64, re, math, datetime as dt
from typing import Dict, Any, Optional, Tuple, List

import requests
import math
import streamlit as st
import streamlit.components.v1 as components
import pydeck as pdk

# ---------- CONFIG ----------
IMG_PATH = "assets/images/fokabs image.jpg"
PAGE_ICON = IMG_PATH if os.path.exists(IMG_PATH) else "🌲"

st.set_page_config(
    page_title="SAFER — Sustainable Acadian Fires & Emergency Risks",
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
    "</div><div class=\"s-sub\">Sustainable Acadian Fires &amp; Emergency Risks</div>"
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
t1, t2, t3 = st.tabs(["🔥 Active Fires", "📍 Fire Map", "🚨 SAFER Fire Alert"])

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
                st.success("Received response from n8n")
            except requests.HTTPError as e:
                st.error(f"HTTP error: {e.response.status_code} {e.response.text[:400]}")
            except Exception as e:
                st.error(f"Failed: {e}")

        if ss.get("fires_html"):
            components.html(ss["fires_html"], height=820, scrolling=True)

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
            "largest in NB",
            "total hectares burned in NS",
            "fires older than 10 days",
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
        ask = st.button("Ask", key="ask_fires", disabled=not bool(agent_url))

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

        # --- send question to the Agent (right column) ---
        if ask:
            try:
                if agent_url:
                    # Send the user’s text to the agent
                    res = post_json(agent_url, {"q": q, "question": q}, shared_secret or None, timeout=timeout_sec)

                    # If a raw string came back, parse it
                    if isinstance(res, dict) and "response" in res and not res.get("answer_md"):
                        try:
                            res = json.loads(res["response"])
                        except Exception:
                            pass

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
            "Your community or coordinates. TIP Use postal code for best accuracy",
            placeholder=(
                "e.g. Halifax NS  |  Moncton  |  44.65,-63.57"
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

# ===== TAB 2: FIRE MAP & LOCATOR =====
with t2:
    st.subheader("Fire Map & Locator")

    # Pull fires from payload fetched in Tab 1
    payload = st.session_state.get("fires_payload") or {}
    fires_list: List[Dict[str, Any]] = []
    if isinstance(payload, dict):
        fires_list = payload.get("fires") or payload.get("items") or []

    if not fires_list:
        st.info("Click **Fetch Active Fires** in the first tab to load today’s fires.")
        st.stop()

    # --- helpers -------------------------------------------------------------
    def _norm_id(x: Any) -> str:
        return re.sub(r"\D", "", str(x or ""))

    def _as_float(v, default=None):
        try:
            return float(v)
        except Exception:
            return default

    # hectares -> marker radius (meters)
    # NOTE: requires `import math` in your imports block.
    def _radius_from_ha(size_ha: float, k: float = 0.6) -> float:
        try:
            if size_ha is None or float(size_ha) <= 0:
                return 800
            r_m = math.sqrt((float(size_ha) * 10000.0) / math.pi)  # 1 ha = 10,000 m²
            return max(600, min(r_m * float(k), 12000))
        except Exception:
            return 800

    # Normalize status to buckets for filtering
    def _status_key(ctrl: Optional[str]) -> str:
        s = (ctrl or "").lower()
        if "out" in s:   return "ooc"  # Out of Control
        if "held" in s:  return "bh"   # Being Held
        if "under" in s: return "uc"   # Under Control
        return "unk"                 # Unknown / not reported

    # Build base rows (no radius yet)
    base_rows: List[Dict[str, Any]] = []
    for f in fires_list:
        lat = _as_float(f.get("lat"))
        lon = _as_float(f.get("lon"))
        if lat is None or lon is None:
            continue
        ctrl = f.get("control") or f.get("status") or ""
        color = _status_to_color(ctrl)
        size_ha = _as_float(f.get("size_ha"))
        base_rows.append({
            "lat": lat,
            "lon": lon,
            "name": f.get("name") or f.get("id") or "(id?)",
            "id_norm": _norm_id(f.get("name") or f.get("id")),
            "control": ctrl,
            "status_key": _status_key(ctrl),
            "size_ha": size_ha if size_ha is not None else 0.0,
            "color": color,
        })

    # -------- Region clamp + “outside” dim ----------------------------------
    _west  = min(b["west"]  for b in PROVINCE_BOUNDS.values())
    _south = min(b["south"] for b in PROVINCE_BOUNDS.values())
    _east  = max(b["east"]  for b in PROVINCE_BOUNDS.values())
    _north = max(b["north"] for b in PROVINCE_BOUNDS.values())
    BBOX = (_west, _south, _east, _north)

    mask_geojson = {
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "properties": {},
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [[-180, -90], [180, -90], [180, 90], [-180, 90], [-180, -90]],
                    [[BBOX[0], BBOX[1]], [BBOX[0], BBOX[3]], [BBOX[2], BBOX[3]], [BBOX[2], BBOX[1]], [BBOX[0], BBOX[1]]]
                ]
            }
        }]
    }
    mask_layer = pdk.Layer(
        "GeoJsonLayer",
        data=mask_geojson,
        stroked=False,
        filled=True,
        get_fill_color=[255, 255, 255, 150],
        pickable=False,
    )

    # Province border rectangles (approx)
    border_paths = []
    for b in PROVINCE_BOUNDS.values():
        rect = [[b["west"], b["south"]], [b["west"], b["north"]],
                [b["east"], b["north"]], [b["east"], b["south"]], [b["west"], b["south"]]]
        border_paths.append({"path": rect})
    borders = pdk.Layer(
        "PathLayer",
        data=border_paths,
        get_path="path",
        get_width=2,
        get_color=[255, 255, 255],
        width_min_pixels=1,
        pickable=False,
    )

    controller = {
        "dragRotate": False,
        "minZoom": 5.2,
        "maxZoom": 11,
        "bounds": [[BBOX[0], BBOX[1]], [BBOX[2], BBOX[3]]],
    }

    map_style = "mapbox://styles/mapbox/satellite-streets-v12" if (
        os.getenv("MAPBOX_API_KEY")
        or os.getenv("MAPBOX_TOKEN")
        or (hasattr(st, "secrets") and ("MAPBOX_API_KEY" in st.secrets or "MAPBOX_TOKEN" in st.secrets))
    ) else None

    col_map, col_side = st.columns([3, 2], gap="large")

    # ---------- RIGHT: Finder + Controls ------------------------------------
    with col_side:
        st.markdown("### Find by Fire Name")
        in_id = st.text_input("Fire Name (e.g. 68622)", key="map_fire_id")

        picked = st.session_state.get("selected_fire") or None
        if st.button("Get Brief", type="primary"):
            want = re.sub(r"\D", "", in_id or "")
            picked = next((r for r in base_rows if r["id_norm"] == want), None)
            st.session_state["selected_fire"] = picked or {}

        # Map Options
        st.markdown("### Map Options")
        scale_by_size = st.checkbox("Scale dots by size (ha)", value=True)
        max_ha = max((r["size_ha"] for r in base_rows if isinstance(r["size_ha"], (int, float))), default=0.0)
        min_size = st.slider("Minimum size (ha) to show", 0.0, float(max(max_ha, 10.0)), 0.0, step=1.0)
        exaggeration = st.slider("Size exaggeration", 0.2, 1.5, 0.6, step=0.05)

        # NEW: Status Filters
        st.markdown("### Status Filters")
        show_ooc = st.checkbox("Out of Control", value=True)
        show_bh  = st.checkbox("Being Held", value=True)
        show_uc  = st.checkbox("Under Control", value=True)
        show_unk = st.checkbox("Unknown", value=True)
        allowed_status = {k for k, v in [
            ("ooc", show_ooc), ("bh", show_bh), ("uc", show_uc), ("unk", show_unk)
        ] if v}

        # Stats block for the selected fire
        def _stat(label: str, value: str):
            st.markdown(
                f"""
                <div style="margin:10px 0">
                  <div style="font-size:13px;color:#6b7280;margin-bottom:4px">{label}</div>
                  <div style="font-size:36px;font-weight:700;line-height:1.1;white-space:normal">{value}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        if picked:
            st.markdown("#### Selected Fire")
            _stat("Control", str(picked.get("control") or "—"))
            _stat("Size (ha)", f'{picked["size_ha"]:,.0f}')
            started_str = next(
                (str(f.get("started") or "—") for f in fires_list
                 if re.sub(r"\D","",str(f.get("name") or f.get("id"))) == picked["id_norm"]), "—"
            )
            _stat("Started", started_str)
            st.write(
                f'- **ID**: {picked.get("name")}\n'
                f'- **Agency**: {next(((f.get("agency") or "—").upper() for f in fires_list if re.sub(r"\\D","",str(f.get("name") or f.get("id"))) == picked["id_norm"]), "—")}\n'
                f'- **Location**: {picked["lat"]}, {picked["lon"]}'
            )

    # ---------- LEFT: Map ----------------------------------------------------
    # Apply filters + attach radius
    map_rows: List[Dict[str, Any]] = []
    for r in base_rows:
        # filter by size
        if isinstance(r["size_ha"], (int, float)) and r["size_ha"] < min_size:
            continue
        # filter by status
        if r.get("status_key") not in allowed_status:
            continue
        r2 = dict(r)
        r2["radius"] = (_radius_from_ha(r["size_ha"], exaggeration) if scale_by_size else 1600)
        map_rows.append(r2)

    # View
    if picked:
        view = pdk.ViewState(latitude=float(picked["lat"]), longitude=float(picked["lon"]), zoom=8)
    else:
        view = pdk.ViewState(
            latitude=sum(r["lat"] for r in map_rows)/len(map_rows),
            longitude=sum(r["lon"] for r in map_rows)/len(map_rows),
            zoom=5.8,
        )

    # Scatter layer with per-point radius
    all_layer = pdk.Layer(
        "ScatterplotLayer",
        data=map_rows,
        get_position=["lon", "lat"],
        get_fill_color="color",
        get_radius="radius",
        radius_min_pixels=3,
        radius_max_pixels=60,
        pickable=True,
    )
    layers = [mask_layer, all_layer, borders]

    # Selection highlight
    if picked:
        layers.append(
            pdk.Layer(
                "ScatterplotLayer",
                data=[picked],
                get_position=["lon", "lat"],
                get_fill_color=[255, 255, 255],
                get_radius=3800,
                radius_min_pixels=6,
                radius_max_pixels=60,
                stroked=True,
                get_line_color=[0, 0, 0],
                line_width_min_pixels=2,
                pickable=False,
            )
        )

    with col_map:
        st.pydeck_chart(
            pdk.Deck(
                layers=layers,
                initial_view_state=view,
                views=[pdk.View(type="MapView", controller=controller)],
                tooltip={"html": "<b>{name}</b><br/>{control}<br/>{size_ha} ha"},
                map_style=map_style,
            ),
            use_container_width=True,
        )

        # Legend
        st.markdown(
            """
            <div style="margin-top:.5rem;font-size:0.95rem;line-height:1.5">
              <div><b>Legend</b></div>
              <div>
                <span style="color:#dc2626">●</span> Out of Control &nbsp;·&nbsp;
                <span style="color:#eab308">●</span> Being Held &nbsp;·&nbsp;
                <span style="color:#10b981">●</span> Under Control &nbsp;·&nbsp;
                <span style="color:#6b7280">●</span> Unknown
              </div>

              <div style="margin-top:4px;">
                <span>Dot size ≈ fire size (ha). Examples:</span>
                <span style="margin-left:6px">~10 ha = small</span> ·
                <span>~100 ha = medium</span> ·
                <span>~1000 ha = large</span>
              </div>

              <!-- NEW: quick hectare comparison line -->
              <div style="margin-top:2px;color:#6b7280">
                1 ha = 10,000 m² ≈ 2.47 acres (≈ a 100 m × 100 m square)
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ----------------------- TAB 3 — SAFER Fire Alert ---------------------------
with t3:
    # Use session state alias
    ss = st.session_state

    # Defaults (do not change UI)
    ss.setdefault("sub_email", "")
    ss.setdefault("sub_channel", "Email")            # Email | Telegram | Both
    ss.setdefault("sub_telegram_chat_id", "")
    ss.setdefault("sub_address", "")
    ss.setdefault("sub_lat", 46.167500)
    ss.setdefault("sub_lon", -64.750800)
    ss.setdefault("sub_radius_km", 10)
    ss.setdefault("alerts_active", False)

    # ------------------ FORM (UI unchanged) ------------------
    with st.form("sub_form", clear_on_submit=False):
        email = st.text_input(
            "Email",
            placeholder="you@example.com",
            key="sub_email"
        )

        channel = st.radio(
            "Channel",
            ["Email", "Telegram", "Both"],
            horizontal=True,
            index=["Email", "Telegram", "Both"].index(ss.get("sub_channel", "Email")),
            key="sub_channel"
        )

        tg_cols = st.columns([3, 1.2, 1.2])
        with tg_cols[0]:
            telegram_chat_id = st.text_input(
                "Telegram Chat ID",
                placeholder="e.g. 8436906519",
                key="sub_telegram_chat_id",
                help=(
                    "Open @SaferAlertsBot and type /start to get proximity alerts. "
                    "If you don't know your chat ID, open @UserInfoBot and type /start, "
                    "then copy the ID it shows for your Telegram account."
                )
            )
        with tg_cols[1]:
            st.markdown(
                '<div style="margin-top:30px;"><a href="https://t.me/SaferAlertsBot" target="_blank">Open @SaferAlertsBot</a></div>',
                unsafe_allow_html=True,
            )
        with tg_cols[2]:
            st.markdown(
                '<div style="margin-top:30px;"><a href="https://t.me/userinfobot" target="_blank">Open @UserInfoBot</a></div>',
                unsafe_allow_html=True,
            )

        addr_cols = st.columns([3, 1])
        with addr_cols[0]:
            address = st.text_input(
                "Address (optional)",
                placeholder="67a Long Shore Rd, Conception Bay South, NL A1X 6A6, Canada",
                key="sub_address",
            )
        with addr_cols[1]:
            geocode_clicked = st.form_submit_button("Geocode", key="geocode_btn")

        # IMPORTANT: no keys on these number_inputs to avoid key-write conflicts
        lat_col, lon_col = st.columns(2)
        with lat_col:
            lat = st.number_input(
                "Latitude",
                value=float(ss.get("sub_lat", 46.167500)),
                step=0.0001,
                format="%.6f"
            )
        with lon_col:
            lon = st.number_input(
                "Longitude",
                value=float(ss.get("sub_lon", -64.750800)),
                step=0.0001,
                format="%.6f"
            )

        radius_km = st.number_input(
            "Radius (km)",
            value=int(ss.get("sub_radius_km", 10)),
            step=1
        )

        # Button (left) + schedule blurb (right) — emojis unchanged
        col_btn, col_info = st.columns([1, 2])
        with col_btn:
            btn_label = "Cancel Alerts" if ss.get("alerts_active") else "Activate Alerts"
            toggle_clicked = st.form_submit_button(
                btn_label,
                key="alerts_btn",
                type="primary",
                disabled=not bool(subscribe_url),
            )
        with col_info:
            st.markdown(
                """
                <div style="margin-top:6px;font-size:.95rem;line-height:1.4">
                  🕒 <b>Proximity Alerts</b> active every hour<br/>
                  🕒 <b>Daily Fire table/CSV file</b> active every day at 12 noon Atlantic Standard Time
                </div>
                """,
                unsafe_allow_html=True,
            )

    # Persist user edits from number_inputs after the form renders
    ss["sub_lat"] = float(lat)
    ss["sub_lon"] = float(lon)
    ss["sub_radius_km"] = int(radius_km)

    # ------------------ HANDLERS ------------------

    # Geocode: OpenCage first, then Google; updates session state safely
    if geocode_clicked:
        addr = (ss.get("sub_address") or "").strip()
        if not addr:
            st.warning("Type an address first, then click Geocode.")
        else:
            try:
                lat_val = None
                lon_val = None
                formatted = None
                source = None

                oc_key = st.secrets.get("OPENCAGE_API_KEY")
                if oc_key and lat_val is None:
                    r = requests.get(
                        "https://api.opencagedata.com/geocode/v1/json",
                        params={"q": addr, "key": oc_key, "limit": 1},
                        timeout=15,
                    )
                    j = r.json()
                    if j.get("results"):
                        g = j["results"][0]
                        lat_val = float(g["geometry"]["lat"])
                        lon_val = float(g["geometry"]["lng"])
                        formatted = g.get("formatted")
                        source = "OpenCage"

                g_key = st.secrets.get("GOOGLE_MAPS_API_KEY")
                if g_key and lat_val is None:
                    r = requests.get(
                        "https://maps.googleapis.com/maps/api/geocode/json",
                        params={"address": addr, "key": g_key},
                        timeout=15,
                    )
                    j = r.json()
                    if j.get("status") == "OK" and j.get("results"):
                        g = j["results"][0]
                        loc = g["geometry"]["location"]
                        lat_val = float(loc["lat"])
                        lon_val = float(loc["lng"])
                        formatted = g.get("formatted_address")
                        source = "Google"

                if lat_val is not None and lon_val is not None:
                    ss.update({
                        "sub_lat": round(lat_val, 6),
                        "sub_lon": round(lon_val, 6),
                        "sub_address": formatted or ss.get("sub_address"),
                    })
                    st.success(f"Geocoded via {source or 'provider'}")
                else:
                    st.error("Geocoding failed. Add province/postal code and try again.")
            except Exception as e:
                st.error(f"Geocoding error: {e}")

    # Activate / Cancel webhook call (n8n)
    if toggle_clicked:
        if not subscribe_url:
            st.error("Subscription webhook URL is not set.")
        else:
            try:
                chan = ss.get("sub_channel", "Email")
                want_email = chan in ("Email", "Both")
                want_tg    = chan in ("Telegram", "Both")

                if want_email and not (ss.get("sub_email") or "").strip():
                    st.error("Please enter an email address.")
                elif want_tg and not (ss.get("sub_telegram_chat_id") or "").strip():
                    st.error("Please enter your Telegram Chat ID.")
                else:
                    payload = {
                        "channel": chan,
                        "email": (ss.get("sub_email") or "").strip() or None,
                        "telegram_chat_id": (ss.get("sub_telegram_chat_id") or "").strip() or None,
                        "address": (ss.get("sub_address") or "").strip() or None,
                        "lat": float(ss.get("sub_lat")),
                        "lon": float(ss.get("sub_lon")),
                        "radius_km": int(ss.get("sub_radius_km") or 10),
                        "source": "streamlit-tab3",
                    }

                    headers = {}
                    secret = st.secrets.get("N8N_SHARED_SECRET")
                    if secret:
                        headers["X-API-KEY"] = secret

                    if ss.get("alerts_active"):
                        r = requests.post(f"{subscribe_url}/cancel", json=payload, headers=headers, timeout=20)
                        if r.ok:
                            ss["alerts_active"] = False
                            st.success("Alerts cancelled.")
                        else:
                            st.error(f"Cancel failed: {r.status_code} {r.text[:300]}")
                    else:
                        r = requests.post(subscribe_url, json=payload, headers=headers, timeout=20)
                        if r.ok:
                            ss["alerts_active"] = True
                            st.success("Alerts activated.")
                        else:
                            st.error(f"Activate failed: {r.status_code} {r.text[:300]}")
            except Exception as e:
                st.error(f"Request error: {e}")
