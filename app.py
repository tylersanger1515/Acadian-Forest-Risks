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
    Prefer Google. Fall back to OpenCage only if Google returns nothing.
    Returns: (lat, lon, formatted_address, source)
    """
    try:
        # 1) Google first
        if g_key:
            gg = _google_geocode(address, g_key)
            if gg:
                loc = ((gg.get("geometry") or {}).get("location") or {})
                fmt = gg.get("formatted_address") or address.strip()
                return (float(loc.get("lat")), float(loc.get("lng")), fmt, "google")

        # 2) OpenCage fallback
        if oc_key:
            oc = _opencage_geocode(address, oc_key)
            if oc:
                lat = float(oc["geometry"]["lat"])
                lon = float(oc["geometry"]["lng"])
                fmt = oc.get("formatted") or address.strip()
                return (lat, lon, fmt, "opencage")

        return None
    except Exception:
        return None

# ---------- TABS ----------
t1, t2, t3 = st.tabs(["🔥 Active Fires", "🧭 Risk Summary", "🚨 SAFER Fire Alert"])

# ===== TAB 1: ACTIVE FIRES =====
with t1:
    st.subheader("Active Fires in the Acadian Region")

    # cache so table remains visible after pressing Ask
    ss = st.session_state
    ss.setdefault("fires_payload", None)
    ss.setdefault("fires_html", None)

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
                # cache for re-render
                ss["fires_payload"] = data
                ss["fires_html"] = html
                st.success("Received response from n8n")
            except requests.HTTPError as e:
                st.error(f"HTTP error: {e.response.status_code} {e.response.text[:400]}")
            except Exception as e:
                st.error(f"Failed: {e}")

        # always re-render last table if we have it
        if ss.get("fires_html"):
            components.html(ss["fires_html"], height=820, scrolling=True)

# ===================== TAB 1 — RIGHT (Q&A only) =====================
with right:
    import re, math, datetime as dt, requests

    st.markdown("#### Ask about today’s fires")
    q = st.text_input(
        "Your question",
        placeholder=(
            "e.g., Which fires are Out of Control? | within 40 km of Halifax | "
            "fires over 25 ha in NB | top 5 largest | largest per province | "
            "still 0.1 ha after 5 days | older than eight days | started at least 5 days ago"
        ),
        key="q_fires",
    )
    ask = st.button("Ask", key="ask_fires", disabled=not bool(fires_url))

    # ---------- helpers ----------
    def _norm(s): return (s or "").strip()
    def _prov(f): return _norm((f.get("agency") or "").upper())
    def _ctrl_text(f): return _norm((f.get("control") or f.get("status") or "")).lower()
    def _sizeha(f):
        try: return float(f.get("size_ha") or 0.0)
        except: return 0.0
    def _started_s(f): return _norm(str(f.get("started") or ""))[:10]
    def _lat(f):
        try: return float(f.get("lat"))
        except: return None
    def _lon(f):
        try: return float(f.get("lon"))
        except: return None
    def _date_iso(s):
        try: return dt.date.fromisoformat((s or "")[:10])
        except Exception: return None

    def haversine_km(lat1, lon1, lat2, lon2):
        R = 6371.0; p = math.pi/180.0
        dlat = (lat2-lat1)*p; dlon = (lon2-lon1)*p
        a = math.sin(dlat/2)**2 + math.cos(lat1*p)*math.cos(lat2*p)*math.sin(dlon/2)**2
        return R*2*math.asin(math.sqrt(a))

    # word→digit so “eight days” works
    _NUMWORDS = {
        "zero":0,"one":1,"two":2,"three":3,"four":4,"five":5,"six":6,"seven":7,"eight":8,"nine":9,"ten":10,
        "eleven":11,"twelve":12,"thirteen":13,"fourteen":14,"fifteen":15,"sixteen":16,"seventeen":17,"eighteen":18,"nineteen":19,"twenty":20
    }
    def _words_to_digits(s: str) -> str:
        return re.sub(r'\b(' + '|'.join(_NUMWORDS.keys()) + r')\b',
                      lambda m: str(_NUMWORDS[m.group(1)]), s)

    # robust geocoder (uses app-level geocode_address if available)
    def geocode_place(name: str):
        qtext = (name or "").strip()
        if not qtext: return None
        # strip trailing “within 40 km” if user wrote “near Halifax within 40 km”
        qtext = re.sub(r'\bwithin\s+\d+(?:\.\d+)?\s*km\b', '', qtext, flags=re.I).strip(",.;: ")

        try:
            _ = geocode_address
            use_helper = True
        except NameError:
            use_helper = False

        gkey = google_key
        ockey = opencage_key

        variants = [
            qtext,
            f"{qtext}, Canada",
            f"{qtext}, NS, Canada",
            f"{qtext}, Nova Scotia, Canada",
            f"{qtext} Regional Municipality, NS, Canada",
            f"{qtext} Regional Municipality, Nova Scotia, Canada",
        ]
        for v in variants:
            try:
                if use_helper:
                    g = geocode_address(v, ockey, gkey)
                    if g:
                        lat, lon, fmt, _src = g
                        return float(lat), float(lon), fmt
                if gkey:
                    r = requests.get(
                        "https://maps.googleapis.com/maps/api/geocode/json",
                        params={"address": v, "region": "ca", "key": gkey},
                        timeout=10,
                    ).json()
                    if r.get("status") == "OK" and r.get("results"):
                        res = r["results"][0]
                        loc = res["geometry"]["location"]
                        return float(loc["lat"]), float(loc["lng"]), res.get("formatted_address", v)
                if ockey:
                    r = requests.get(
                        "https://api.opencagedata.com/geocode/v1/json",
                        params={"q": v, "key": ockey, "limit": 1, "countrycode": "ca", "no_annotations": 1},
                        timeout=10,
                    ).json()
                    if r.get("results"):
                        res = r["results"][0]
                        loc = res["geometry"]
                        return float(loc["lat"]), float(loc["lng"]), res.get("formatted", v)
            except Exception:
                pass
        return None

    def parse_days_window(text: str):
        """Return (d_from, d_to, tag) for:
           'last/past N days|weeks', 'older than / at least N days (ago)', 'N days ago'."""
        s2 = (text or "").lower()
        end = _date_iso(st.session_state.get("fires_payload", {}).get("date")) or dt.date.today()

        # last / past N days|weeks
        m = re.search(r'(?:last|past)\s+(\d+)\s+day(?:s)?', s2)
        if m: d = int(m.group(1));  return (end - dt.timedelta(days=d), end, "past")
        m = re.search(r'(?:last|past)\s+(\d+)\s+week(?:s)?', s2)
        if m: w = int(m.group(1));  return (end - dt.timedelta(days=7*w), end, "past")

        # older than / at least N days (optional 'ago')
        m = re.search(r'(?:older\s+than|at\s+least|>=)\s+(\d+)\s+day(?:s)?(?:\s+ago)?', s2)
        if m: d = int(m.group(1));  return (None, end - dt.timedelta(days=d), "older")

        # N days ago
        m = re.search(r'(\d+)\s+day(?:s)?\s+ago', s2)
        if m: d = int(m.group(1));  return (None, end - dt.timedelta(days=d), "older")

        return (None, None, "")

    if ask:
        try:
            raw = st.session_state.get("fires_payload") or post_json(
                fires_url, {"from": "streamlit"}, shared_secret or None, timeout=timeout_sec
            )
            fires = raw.get("fires") or []
            date_label = raw.get("date", "today")

            # --- parse query text (normalize number words) ---
            text = _words_to_digits((q or "").strip().lower())

            # province filters
            want_provs = []
            if re.search(r'\b(nb|new brunswick)\b', text): want_provs.append("NB")
            if re.search(r'\b(ns|nova scotia)\b', text):   want_provs.append("NS")
            if re.search(r'\b(nl|newfoundland)\b', text):  want_provs.append("NL")

            # control filters
            want_oc = ("out of control" in text) or re.search(r'\boc\b', text)
            want_bh = ("being held" in text) or re.search(r'\bbh\b', text)
            want_uc = ("under control" in text) or (re.search(r'\buc\b', text) and "out of" not in text)

            # size filters
            size_min = size_max = None
            m = re.search(r'over\s+(\d+(?:\.\d+)?)\s*ha', text)
            if m: size_min = float(m.group(1))
            m = re.search(r'under\s+(\d+(?:\.\d+)?)\s*ha', text)
            if m: size_max = float(m.group(1))
            m = re.search(r'between\s+(\d+(?:\.\d+)?)\s*ha\s*(?:and|-|to)\s*(\d+(?:\.\d+)?)\s*ha', text)
            if m:
                a, b = float(m.group(1)), float(m.group(2))
                size_min, size_max = min(a, b), max(a, b)

            # date windows
            d_from, d_to, _tag = parse_days_window(text)
            m = re.search(r'after\s+(\d{4}-\d{2}-\d{2})', text)
            after_date = _date_iso(m.group(1)) if m else None
            m = re.search(r'before\s+(\d{4}-\d{2}-\d{2})', text)
            before_date = _date_iso(m.group(1)) if m else None

            # "still 0.1 ha after 5 days"
            still_small = bool(re.search(r'\bstill\b', text)) and bool(re.search(r'\bafter\s+\d+\s+day(?:s)?\b', text))
            m = re.search(r'still\s+(\d+(?:\.\d+)?)\s*ha', text)
            small_target = float(m.group(1)) if m else None
            m = re.search(r'after\s+(\d+)\s+day(?:s)?', text)
            days_min = int(m.group(1)) if m else None

            # top N / "N of the largest"
            topN = None
            m = re.search(r'top\s+(\d+)', text)
            if m: topN = int(m.group(1))
            if topN is None:
                m = re.search(r'(\d+)\s+of\s+the\s+largest', text)
                if m: topN = int(m.group(1))

            # distance / proximity
            DEFAULT_NEAR_KM = 40.0
            want_within_km = None
            anchor = None
            place_label = ""
            asked_closest = bool(re.search(r'\b(closest|nearest)\b', text))

            m_rad = re.search(r'within\s+(\d+(?:\.\d+)?)\s*km', text)
            if m_rad: want_within_km = float(m_rad.group(1))

            m_place = (
                re.search(r'within\s+\d+(?:\.\d+)?\s*km\s+of\s+(.+)', text) or
                re.search(r'(?:closest|nearest)\s+(?:to\s+)?(.+)', text) or
                re.search(r'(?:near|close to)\s+(.+)', text)
            )
            if m_place:
                ptxt = re.sub(r'\bwithin\s+\d+(?:\.\d+)?\s*km\b', '', m_place.group(1), flags=re.I).strip(",.;: ")
                anchor = geocode_place(ptxt)
                place_label = (anchor[2] if anchor else ptxt)

            if anchor and want_within_km is None and ("near" in text or "close to" in text or asked_closest):
                want_within_km = None  # “closest to …” without radius => show nearest few
            names_only = "names only" in text or "just names" in text

            # ---------- filtering pipeline ----------
            subset = list(fires)

            if want_provs:
                subset = [f for f in subset if _prov(f) in want_provs]

            if want_oc or want_bh or want_uc:
                def _match_ctrl(f):
                    s = _ctrl_text(f)
                    return ((want_oc and "out of control" in s) or
                            (want_bh and "being held" in s) or
                            (want_uc and "under control" in s))
                subset = [f for f in subset if _match_ctrl(f)]

            if size_min is not None:
                subset = [f for f in subset if _sizeha(f) >= size_min]
            if size_max is not None:
                subset = [f for f in subset if _sizeha(f) <= size_max]

            if d_from or d_to:
                sub = []
                for f in subset:
                    d = _date_iso(_started_s(f))
                    if not d: 
                        continue
                    if d_from and d < d_from: 
                        continue
                    if d_to and d > d_to:
                        continue
                    sub.append(f)
                subset = sub

            if after_date:
                subset = [f for f in subset if (_date_iso(_started_s(f)) or dt.date.min) > after_date]
            if before_date:
                subset = [f for f in subset if (_date_iso(_started_s(f)) or dt.date.max) < before_date]

            if (d_from or d_to or after_date or before_date) and not subset:
                st.info("No fires match that time window.")
                st.stop()

            if still_small and small_target is not None and days_min is not None:
                today = _date_iso(date_label) or dt.date.today()
                tol = max(0.05, 0.01 * small_target)
                subset = [
                    f for f in subset
                    if abs(_sizeha(f) - small_target) <= tol
                    and (today - (_date_iso(_started_s(f)) or today)).days >= days_min
                ]

            if anchor:
                lat0, lon0, place_label = float(anchor[0]), float(anchor[1]), anchor[2]
                cands = [f for f in subset if _lat(f) is not None and _lon(f) is not None]
                for f in cands:
                    f["_dist_km"] = haversine_km(lat0, lon0, _lat(f), _lon(f))
                if want_within_km is not None:
                    cands = [f for f in cands if f["_dist_km"] <= want_within_km]
                cands.sort(key=lambda x: x["_dist_km"])
                # If they asked for “closest …” without a radius, cap to the nearest 12
                if asked_closest and want_within_km is None:
                    cands = cands[:12]
                subset = cands
                if want_within_km is not None and not subset:
                    st.info(f"No active fires within {want_within_km:.0f} km of {place_label}.")
                    st.stop()

            # ---------- special summaries (place ABOVE generic listing) ----------
            # 1) counts by province
            if ("how many" in text or "count" in text) and ("by province" in text or "per province" in text):
                from collections import Counter
                counts = Counter(_prov(f) for f in fires if _prov(f))
                if counts:
                    st.markdown(f"**Active fires by province — {date_label}:**")
                    for p, c in counts.most_common():
                        st.write(f"- {p}: {c}")
                else:
                    st.info("No active fires found.")
                st.stop()

            # 2) most hectares by province
            if "most hectares" in text or ("hectares" in text and "most" in text):
                totals = {}
                for f in fires:
                    p = _prov(f)
                    if p: totals[p] = totals.get(p, 0.0) + _sizeha(f)
                if totals:
                    leader = max(totals, key=totals.get)
                    st.markdown(f"**Most hectares burning:** {leader} — {totals[leader]:,.1f} ha — {date_label}")
                    st.caption(", ".join(f"{p}: {v:,.1f} ha" for p, v in sorted(totals.items(), key=lambda x: x[1], reverse=True)))
                else:
                    st.info("No data.")
                st.stop()

            # 3) largest per province
            if ("largest" in text or "biggest" in text or "max" in text) and ("per province" in text or "each province" in text):
                pool = subset if want_provs else fires
                by_p = {}
                for f in pool:
                    p = _prov(f)
                    if not p: continue
                    if p not in by_p or _sizeha(f) > _sizeha(by_p[p]):
                        by_p[p] = f
                if not by_p:
                    st.info("No matching fires.")
                else:
                    st.markdown(f"**Largest fire per province — {date_label}:**")
                    for p in sorted(by_p.keys()):
                        f = by_p[p]
                        st.write(f"- {p}: {f.get('name')} · {_sizeha(f):,.1f} ha · {f.get('control','—')} · Started {_started_s(f) or '—'}")
                st.stop()

            # ---------- names only ----------
            if names_only:
                if not subset:
                    st.info("No matching fires.")
                else:
                    st.markdown("**Matching fire names:**")
                    st.write(", ".join(sorted({_norm(f.get('name')) for f in subset})))
                st.stop()

            # ---------- top N ----------
            if topN is not None and topN > 0:
                subset = sorted(subset, key=_sizeha, reverse=True)[:topN]

            # ---------- default render ----------
            if not subset:
                ongoing = int(raw.get("count_ongoing") or len(fires))
                st.markdown(f"**Snapshot for {date_label}:** {ongoing} ongoing fire(s).")
                st.write("Try: *fires over 20 ha in NB* · *closest to Truro* · *within 40 km of Halifax* · *top 5 largest in NS* · *started last 7 days* · *older than 3 days* · *still 0.1 ha after 5 days*")
                st.stop()

            st.markdown(f"**Matches — {date_label} ({len(subset)}):**")
            for f in subset:
                # distance shown BEFORE the date (your request)
                dist = f" · {f['_dist_km']:.1f} km" if "_dist_km" in f else ""
                st.write(
                    f"- {f.get('name')} — {_prov(f)} · {_sizeha(f):,.1f} ha · {f.get('control','—')}{dist} · Started {_started_s(f) or '—'}"
                )

            total = sum(_sizeha(f) for f in subset)
            earliest = min((_date_iso(_started_s(f)) for f in subset if _date_iso(_started_s(f))), default=None)
            newest   = max((_date_iso(_started_s(f)) for f in subset if _date_iso(_started_s(f))), default=None)
            st.caption(
                f"Total size: {total:,.1f} ha · "
                f"Earliest start: {earliest or '—'} · "
                f"Newest start: {newest or '—'}"
            )

        except requests.HTTPError as e:
            st.error(f"HTTP error: {e.response.status_code} {e.response.text[:400]}")
        except Exception as e:
            st.error(f"Failed: {e}")

    # ---------------- SAFETY CHECK (40 km) ----------------
    st.divider()
    st.markdown("#### Safety check (40 km)")

    RADIUS_KM = 40.0
    loc_in = st.text_input(
        "Your community or coordinates",
        placeholder="e.g., Halifax NS  |  Moncton  |  44.65,-63.57  • Tip: include your postal code for best accuracy (e.g., B3H 1X1)",
        key="safety_place",
    )
    st.caption("Tip: For the most accurate location, include your postal code (e.g., 'Halifax B3H 1X1', 'Moncton E1C 1A1').")

    colA, colB = st.columns([1, 3])
    check_btn = colA.button("Check 40 km", key="safety_check", disabled=not bool(fires_url))

    def _parse_latlon(s: str):
        try:
            a, b = [t.strip() for t in (s or "").split(",")]
            return float(a), float(b)
        except Exception:
            return None

    def _guidance_block():
        st.markdown(
            """
**If a wildfire is near you (≤40 km):**
- **Call 911** if you see fire/smoke threatening people or property, or if told to evacuate.
- **Contact your provincial forestry / wildfire line** to report details if safe to do so.
- **Prepare a go-bag** for each person: ID, meds/prescriptions, phone/chargers, cash, water & snacks, clothing, sturdy shoes, flashlight, important docs (USB/photo), pet supplies, masks (N95), eyeglasses.
- **Get ready to leave quickly:** keep your vehicle fueled, park facing the road, know **two ways out**, share plans with family.
- **Harden your home (only if safe):** close windows/vents, remove flammables from deck, wet vegetation, keep hoses ready.
- **Activate the SAFER fire alert for critical proximity alerts.**
            """
        )
        with st.expander("Preparedness checklist (if not under immediate threat)"):
            st.markdown(
                """
- Make a family communication & evacuation plan (pets included).
- Assemble go-bags and keep copies of critical documents.
- Clear debris within 1–2 m of structures; trim dry brush; store fuel away from the house.
- Keep an N95 supply for smoke days; monitor air quality if sensitive.
                """
            )

    if check_btn:
        try:
            raw = st.session_state.get("fires_payload") or post_json(
                fires_url, {"from": "safety"}, shared_secret or None, timeout=timeout_sec
            )
            fires = raw.get("fires") or []
            if not loc_in.strip():
                st.info("Enter a community or coordinates first.")
                st.stop()

            anchor = None
            ll = _parse_latlon(loc_in)
            if ll:
                anchor = (ll[0], ll[1], f"{ll[0]:.4f}, {ll[1]:.4f}")
            else:
                g = geocode_place(loc_in)
                if g:
                    anchor = (float(g[0]), float(g[1]), g[2])

            if not anchor:
                st.info("Couldn't locate that place. Try including the province or your postal code, e.g., 'Halifax B3H 1X1'.")
                st.stop()

            lat0, lon0, place_lbl = anchor
            cands = [f for f in fires if _lat(f) is not None and _lon(f) is not None]
            for f in cands:
                f["_dist_km"] = haversine_km(lat0, lon0, _lat(f), _lon(f))
            nearby = [f for f in cands if f["_dist_km"] <= RADIUS_KM]
            nearby.sort(key=lambda x: x["_dist_km"])

            if nearby:
                st.error(f"⚠️ {len(nearby)} active fire(s) within {RADIUS_KM:.0f} km of {place_lbl}")
                for f in nearby:
                    st.write(
                        f"- {f.get('name')} — {_prov(f)} · {_sizeha(f):,.1f} ha · {f.get('control','—')} · "
                        f"Started {_started_s(f) or '—'} · {f['_dist_km']:.1f} km away"
                    )
                _guidance_block()
            else:
                st.success(f"No active fires within {RADIUS_KM:.0f} km of {place_lbl} right now.")
                with st.expander("Be prepared anyway (quick tips)"):
                    _guidance_block()

        except requests.HTTPError as e:
            st.error(f"HTTP error: {e.response.status_code} {e.response.text[:300]}")
        except Exception as e:
            st.error(f"Check failed: {e}")

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
