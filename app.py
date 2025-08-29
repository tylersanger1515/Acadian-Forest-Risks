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
            nums = [float(x) for x in re.findall(r'(\d+(?:\.\d+)?)\s*ha?', t)]
            if "between" in t and "and" in t and len(nums) >= 2:
                a, b = sorted(nums[:2]); return ("between", a, b)
            if any(w in t for w in ["over","more than",">=","at least","minimum","greater than","bigger than","larger than",">"]):
                return ("min", nums[0]) if nums else None
            if any(w in t for w in ["under","less than","<=","at most","maximum","below","smaller than","lesser than","<"]):
                return ("max", nums[0]) if nums else None
            return None

        # time window
        def parse_days_window(text: str):
            s2 = _num_from_words((text or "").lower())
            end = _date_iso(st.session_state.get("fires_payload", {}).get("date")) or dt.date.today()
            m = re.search(r'(?:last|past)\s+(\d+)\s+day', s2)
            if m: d = int(m.group(1));  return (end - dt.timedelta(days=d), end, "past")
            m = re.search(r'(?:last|past)\s+(\d+)\s+week', s2)
            if m: w = int(m.group(1));  return (end - dt.timedelta(days=7*w), end, "past")
            m = re.search(r'(?:older\s+than|at\s+least|>=)\s+(\d+)\s+day(?:s)?(?:\s+ago)?', s2)
            if m: d = int(m.group(1));  return (None, end - dt.timedelta(days=d), "older")
            m = re.search(r'(\d+)\s+day(?:s)?\s+ago', s2)
            if m: d = int(m.group(1));  return (None, end - dt.timedelta(days=d), "older")
            if "last week" in s2 or "past week" in s2: return (end - dt.timedelta(days=7), end, "past")
            if "last 7 days" in s2: return (end - dt.timedelta(days=7), end, "past")
            return (None, None, "")

        # GEO intent parsing
        def parse_geo(text):
            t = (text or "").lower().strip()
            m = re.search(r'within\s+(\d+(?:\.\d+)?)\s*km\s+(?:of|from)\s+(.+)', t)
            if m: return ("within", float(m.group(1)), m.group(2).strip())
            m = re.search(r'(\d+(?:\.\d+)?)\s*km\s+(?:near|around|about)\s+(.+)', t)
            if m: return ("within", float(m.group(1)), m.group(2).strip())
            m = re.search(r'(?:any\s+fires\s+)?(\d+(?:\.\d+)?)\s*km\s+(?:away\s+from|from|of)\s+(.+)', t)
            if m: return ("within", float(m.group(1)), m.group(2).strip())
            m = re.search(r'(?:top\s+(\d+)\s+)?(?:closest|nearest)(?:\s+fire)?\s+(?:to|near)\s+(.+)', t)
            if m: return ("closest", int(m.group(1) or 1), m.group(2).strip())
            m = re.search(r'\b(?:near|nearby|close to)\s+(.+)', t)
            if m: return ("within", 40.0, m.group(1).strip())
            return (None, None, None)

        def geocode_place(name: str) -> Optional[Tuple[float, float, str]]:
            qtext = (name or "").strip()
            if not qtext: return None
            qtext = re.sub(r'\bwithin\s+\d+(?:\.\d+)?\s*km\b', '', qtext, flags=re.I).strip(",.;: ")
            try:
                g = geocode_address(qtext, opencage_key, google_key)
                if g: return float(g[0]), float(g[1]), g[2]
            except Exception:
                pass
            return None

        def attach_distances(fs, where):
            g = geocode_place(where)
            if not g:
                return fs, False, where
            lat0, lon0, label = g
            for f in fs:
                lat, lon = _lat(f), _lon(f)
                d = haversine_km(lat0, lon0, lat, lon) if (lat is not None and lon is not None) else None
                f["_dist_km"] = d
            return fs, True, label

        def _digits(s: str) -> str:
            return re.sub(r"\D", "", s or "")

        # geocode & cache cities for “nearest city to fire <id>”
        def _geocode_city(label: str) -> Optional[Tuple[float, float]]:
            cache = ss["city_cache"]
            if label in cache: return cache[label]
            g = geocode_place(label)
            if not g: return None
            cache[label] = (g[0], g[1])
            return cache[label]

        def _nearest_place_to_fire(fid: str) -> Optional[Tuple[str, float]]:
            raw = ss.get("fires_payload") or {}
            fires = raw.get("fires") or []
            f = next((x for x in fires if str(x.get("name")) == fid), None)
            if not f or _lat(f) is None or _lon(f) is None:
                return None
            lat, lon = _lat(f), _lon(f)
            best = None
            for city in DEFAULT_CITIES:
                ll = _geocode_city(city)
                if not ll: continue
                d = haversine_km(lat, lon, ll[0], ll[1])
                if d is None: 
                    continue
                if not best or d < best[1]:
                    best = (city, d)
            return best

        # ---------- Q&A engine ----------
        def render_qna(q_text: str):
            try:
                raw = ss.get("fires_payload") or post_json(
                    fires_url, {"from": "streamlit"}, shared_secret or None, timeout=timeout_sec
                )
                ss["fires_payload"] = raw
                fires = raw.get("fires") or []
                date_label = raw.get("date", "today")

                text = (q_text or "").lower().strip()

                # province filters (support multiple)
                want = []
                if re.search(r'\bnb\b|new brunswick', text): want.append("NB")
                if re.search(r'\bns\b|nova scotia', text):  want.append("NS")
                if re.search(r'\bnl\b|newfoundland|labrador', text): want.append("NL")
                subset = [f for f in fires if (_prov(f) in want)] if want else list(fires)

                # control synonyms (OC / UC / BH)
                want_ctrl = None
                tnorm = text.replace("-", " ")
                if re.search(r"\bout of control\b|\booc\b|out of control", tnorm):
                    want_ctrl = "out of control"
                elif re.search(r"\bbeing held\b|\bbh\b|\bheld\b", tnorm):
                    want_ctrl = "being held"
                elif re.search(r"\bunder control\b|\buc\b|\bcontrolled\b|\bcontained\b", tnorm):
                    want_ctrl = "under control"
                if want_ctrl:
                    subset = [f for f in subset if want_ctrl in _ctrl_text(f)]

                # size range
                rng = parse_size_range(text)
                if rng:
                    if rng[0] == "between":
                        a, b = rng[1], rng[2]; subset = [f for f in subset if a <= _sizeha(f) <= b]
                    elif rng[0] == "min":
                        subset = [f for f in subset if _sizeha(f) >= rng[1]]
                    elif rng[0] == "max":
                        subset = [f for f in subset if _sizeha(f) <= rng[1]]

                # time window
                d_from, d_to, _tag = parse_days_window(text)
                if d_from or d_to:
                    def in_window(f):
                        d = _date_iso(_started_s(f))
                        if not d: return False
                        if d_from and d < d_from: return False
                        if d_to   and d > d_to:   return False
                        return True
                    subset = [f for f in subset if in_window(f)]

                # new today
                if any(w in text for w in ["new fire", "new fires", "started today", "today only"]):
                    today = _date_iso(date_label) or dt.date.today()
                    sf = [f for f in fires if (_date_iso(_started_s(f)) == today)]
                    if not sf:
                        st.info(f"No new fires for {date_label}.")
                    else:
                        st.markdown(f"**New fires — {date_label} ({len(sf)}):**")
                        for f in sf: st.write(fmt_fire_line(f))
                    return

                # totals by province (hectares)
                if (any(w in text for w in ["hectare","hectares","ha","burnt","burned"])
                    and ("by province" in text or "per province" in text or ("province" in text and "total" in text))):
                    totals = {}
                    for f in fires:
                        p = _prov(f)
                        if p: totals[p] = totals.get(p, 0.0) + _sizeha(f)
                    if totals:
                        st.markdown(f"**Total hectares by province — {date_label}:**")
                        for p, v in sorted(totals.items()):
                            st.write(f"- {p}: {v:,.1f} ha")
                    else:
                        st.info("No data.")
                    return

                # overall total hectares
                if (any(w in text for w in ["hectare","hectares","ha"]) and "total" in text and "province" not in text):
                    tot = sum(_sizeha(f) for f in fires)
                    st.markdown(f"**Total area across all active fires — {date_label}: {tot:,.1f} ha**")
                    return

                # counts by province
                if (("how many" in text or "count" in text or "number of" in text or "total fires" in text)
                    and ("by province" in text or "per province" in text or "each province" in text)):
                    from collections import Counter
                    counts = Counter(_prov(f) for f in fires if _prov(f))
                    if counts:
                        st.markdown(f"**Active fires by province — {date_label}:**")
                        for p, c in counts.most_common():
                            st.write(f"- {p}: {c}")
                    else:
                        st.info("No active fires found.")
                    return

                # province with most
                if "most" in text and "province" in text and "fire" in text:
                    from collections import Counter
                    counts = Counter(_prov(f) for f in fires if _prov(f))
                    if counts:
                        p, c = counts.most_common(1)[0]
                        st.markdown(f"**{p}** has the most active fires today ({c}).")
                    else:
                        st.info("No active fires found.")
                    return

                # largest per province
                if "largest" in text and ("per province" in text or "by province" in text):
                    best = {}
                    for f in fires:
                        p = _prov(f)
                        if not p: continue
                        if p not in best or _sizeha(f) > _sizeha(best[p]): best[p] = f
                    if best:
                        st.markdown(f"**Largest fire per province — {date_label}:**")
                        for p in ("NB","NL","NS"):
                            if p in best: st.write(fmt_fire_line(best[p]))
                    else:
                        st.info("No data.")
                    return

                # top N largest (current subset)
                if "top" in text and "largest" in text:
                    m = re.search(r'top\s+(\d+)', text); k = int(m.group(1)) if m else 5
                    biggest = sorted(subset, key=_sizeha, reverse=True)[:k]
                    st.markdown(f"**Top {k} largest fires — {date_label}:**" if biggest else "No fires found.")
                    for f in biggest: st.write(fmt_fire_line(f))
                    return

                # largest in subset (e.g., “largest in NB”)
                if "largest" in text and (" in " in text or want):
                    if not subset: st.info("No matching fires."); return
                    f = max(subset, key=_sizeha)
                    st.markdown(f"**Largest fire — {date_label}:**"); st.write(fmt_fire_line(f)); return

                # where is / location of fire <id>
                m_id = re.search(r'\b(\d{3,6})\b', text)
                if any(k in text for k in ["where is","what location","location of","loc of","coords of"]) and m_id:
                    fid = m_id.group(1)
                    f = next((x for x in fires if str(x.get("name")) == fid), None)
                    if not f: 
                        st.info(f"Fire {fid} not found.")
                    else:
                        lat, lon = _lat(f), _lon(f)
                        coords = f"{lat:.4f}, {lon:.4f}" if (lat is not None and lon is not None) else "—"
                        st.markdown(f"**Location for fire {fid} — {date_label}:**")
                        st.write(f"- {_prov(f)} · {_sizeha(f):,.1f} ha · {f.get('control','—')} · "
                                 f"coords: {coords} · Started {_started_s(f) or '—'}")
                    return

                # when did fire <id> start? (hyphen/space tolerant)
                m_start = re.search(r'(?:when\s+did\s*(?:fire\s*)?([0-9\-\s]+)\s*start|start\s*date\s*for\s*([0-9\-\s]+))', text)
                if m_start:
                    fid = _digits(m_start.group(1) or m_start.group(2))
                    f = next((x for x in fires if str(x.get("name")) == fid), None)
                    if not f: st.info(f"Fire {fid} not found.")
                    else:
                        st.markdown(f"**Start date for fire {fid}: {_started_s(f) or '—'}**")
                        st.write(fmt_fire_line(f))
                    return

                # how far is fire <id> from <place>
                m = re.search(r'how\s+far\s+(?:is\s+)?(?:fire\s+)?(\d{3,6})\s+(?:from|to)\s+(.+)', text)
                if m:
                    fid, place = m.group(1), m.group(2).strip()
                    f = next((x for x in fires if str(x.get("name")) == fid), None)
                    if not f: st.info(f"Fire {fid} not found."); return
                    if _lat(f) is None or _lon(f) is None:
                        st.info(f"Coordinates for fire {fid} are not available."); return
                    g = geocode_place(place)
                    if not g: st.info(f"Couldn’t locate “{place}”. Try including province or postal code."); return
                    lat0, lon0, lbl = g
                    d = haversine_km(lat0, lon0, _lat(f), _lon(f))
                    if d is None:
                        st.info("Couldn’t compute distance for that query."); return
                    st.markdown(f"**Distance — fire {fid} to {lbl}: {d:.1f} km**"); st.write(fmt_fire_line(f)); return

                # nearest place to fire <id> (using built-in city list)
                m = re.search(r'(?:nearest|closest)\s+(?:place|city)\s+(?:to|for)\s+(?:fire\s*)?(\d{3,6})', text)
                if m:
                    fid = m.group(1)
                    best = _nearest_place_to_fire(fid)
                    if not best: st.info(f"Fire {fid} not found or no reference places were geocoded yet.")
                    else:
                        st.markdown(f"**Closest place to fire {fid}: {best[0]} — {best[1]:.1f} km**")
                    return

                # “still 0.1 ha after N days”
                if "0.1 ha" in text and ("after" in text or "older than" in text or "later" in text or "still" in text):
                    m = re.search(r'after\s+(\d+)\s+day', _num_from_words(text))
                    n = int(m.group(1)) if m else 2
                    cutoff = dt.date.today() - dt.timedelta(days=n)
                    sf = [f for f in subset if abs(_sizeha(f) - 0.1) < 1e-6 and (_date_iso(_started_s(f)) or dt.date(1900,1,1)) <= cutoff]
                    if not sf:
                        st.info(f"No fires still 0.1 ha after {n} days.")
                    else:
                        st.markdown(f"**Matches — {date_label} ({len(sf)}):**")
                        for f in sf: st.write(fmt_fire_line(f))
                        st.caption(
                            f"Total size: {sum(_sizeha(x) for x in sf):,.1f} ha · "
                            f"Earliest start: {min((_date_iso(_started_s(x)) for x in sf if _date_iso(_started_s(x))), default='—')} · "
                            f"Newest start: {max((_date_iso(_started_s(x)) for x in sf if _date_iso(_started_s(x))), default='—')}"
                        )
                    return

                # GEO: within / closest
                mode, amount, place = parse_geo(text)
                if mode == "within" and place:
                    fs, show_km, label = attach_distances(subset[:], place)
                    if not show_km:
                        st.info(f"Couldn’t locate “{place}”. Try including the province or a postal code.")
                        return
                    radius = float(amount or 40)
                    fs = [f for f in fs if f.get("_dist_km") is not None and f["_dist_km"] <= radius]
                    if not fs:
                        st.info(f"No active fires within {radius:.0f} km of {label}."); return
                    st.markdown(f"**Matches within {radius:.0f} km of {label} — {date_label} ({len(fs)}):**")
                    for f in sorted(fs, key=lambda x: x.get("_dist_km") or 9e9): st.write(fmt_fire_line(f, show_km=True))
                    return

                if mode == "closest" and place:
                    fs, show_km, label = attach_distances(subset[:], place)
                    if not show_km:
                        st.info(f"Couldn’t locate “{place}”. Try “closest to Truro, NS”."); return
                    k = int(amount or 1)
                    fs = [f for f in fs if f.get("_dist_km") is not None]
                    fs = sorted(fs, key=lambda x: x["_dist_km"])[:k]
                    if not fs: st.info(f"No fires found near {label}."); return
                    title = (
                        f"Closest to {label} — {date_label}"
                        if k == 1
                        else f"Closest {k} to {label} — {date_label}"
                    )
                    st.markdown(f"**{title}:**")
                    for f in fs:
                        st.write(fmt_fire_line(f, show_km=True))
                    return

                # list if a filter matched
                if (d_from or d_to) or want_ctrl or want:
                    if not subset: st.info("No matching fires.")
                    else:
                        st.markdown(f"**Matches — {date_label} ({len(subset)}):**")
                        for f in subset: st.write(fmt_fire_line(f))
                    return

                # fallback snapshot
                ongoing = int(raw.get("count_ongoing") or len(fires))
                st.markdown(f"**Snapshot for {date_label}:** {ongoing} ongoing fire(s).")
                st.write("Try: *fires over 20 ha in NB* · *closest to Truro* · *within 40 km of Halifax* · "
                         "*top 5 largest in NS* · *started last 7 days* · *older than 3 days*")
            except requests.HTTPError as e:
                st.error(f"HTTP error: {e.response.status_code} {e.response.text[:400]}")
            except Exception as e:
                st.warning(f"Couldn’t answer that right now ({e}).")

        # trigger Q&A
        if ask or use_ex:
            render_qna(q)

        # ---------------- SAFETY CHECK (kept in right column) ----------------
        st.divider()
        st.markdown("#### Safety check (40 km)")

        RADIUS_KM = 40.0
        loc_in = st.text_input(
            "Your community or coordinates",
            placeholder="e.g., Halifax NS  |  Moncton  |  44.65,-63.57  • Tip: include your postal code for best accuracy (e.g., B3H 1X1)",
            key="safety_place",
        )
        st.caption("Tip: For the most accurate location, include your postal code (e.g., 'B3H 1X1', 'E1C 1A1').")

        colA, colB = st.columns([1, 3])
        check_btn = colA.button("Check 40 km", key="safety_check", disabled=not bool(fires_url))

        def _parse_latlon(s: str):
            try:
                a, b = [t.strip() for t in (s or "").split(",")]
                return float(a), float(b)
            except Exception:
                return None

        def _guidance_block():
            st.markdown("""
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
        mode = st.radio("Find by", ["Fire ID", "Location"], horizontal=True)

        with st.form("brief_form", clear_on_submit=False):
            payload = None
            if mode == "Fire ID":
                fire_id = st.text_input("Fire ID (e.g., 68622)", key="brief_id")
                if fire_id.strip():
                    payload = {"id": fire_id.strip()}
            else:
                c1, c2, c3 = st.columns(3)
                with c1: lat = st.number_input("Lat", value=47.4851, format="%.6f")
                with c2: lon = st.number_input("Lon", value=-65.5618, format="%.6f")
                with c3: radius = st.number_input("Radius (km)", min_value=1, value=30, step=1)
                payload = {"lat": float(lat), "lon": float(lon), "radius_km": int(radius)}
            submitted = st.form_submit_button("Get Brief", type="primary")

        if submitted and payload:
            try:
                data = post_json(risk_url, payload, shared_secret or None, timeout=timeout_sec)
                incident = (data or {}).get("incident") or {}
                brief_md = (data or {}).get("brief_md") or "_No brief returned_"

                # --- Top metrics
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Tier", data.get("tier", "—"))
                m2.metric("Control", incident.get("control", "—"))
                try:
                    m3.metric("Size (ha)", int(incident.get("size_ha") or 0))
                except Exception:
                    m3.metric("Size (ha)", incident.get("size_ha", "—"))
                m4.metric("Started", incident.get("started", "—"))

                # --- Brief text
                st.markdown(brief_md)

                # --- Small map
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

                # --- Quick links / details
                c1, c2 = st.columns(2)
                if (data or {}).get("map_link"):
                    c1.link_button("Open in Google Maps", data["map_link"])
                with c2:
                    with st.popover("Details"):
                        st.json(incident)

            except requests.HTTPError as e:
                st.error(f"HTTP error: {e.response.status_code} {e.response.text[:400]}")
            except Exception as e:
                st.error(f"Error fetching brief: {e}")
        elif submitted and not payload:
            st.warning("Enter a Fire ID or a location first.")

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
