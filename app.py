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

  # ---------------- RIGHT: rich Q&A over fires[] ----------------
with right:
    st.markdown("#### Ask about today’s fires")

    ss = st.session_state
    q = st.text_input(
        "Your question",
        placeholder=(
            "fires over 20 ha in NB • closest to Truro • within 30 km of Moncton • "
            "top 5 largest in NS • started last 7 days • older than 3 days • "
            "still 0.1 ha after 2 days • fires to worry about near Halifax • list names"
        ),
    )
    ask = st.button("Ask", key="ask_fires", disabled=not bool(fires_url))

    if ask:
        try:
            # Use cache if present; otherwise fetch once
            raw = ss.get("fires_payload") or post_json(
                fires_url, {"from": "qna"}, shared_secret or None, timeout=timeout_sec
            )
            fires = raw.get("fires") or []
            date_label = raw.get("date", "today")

            # ---------- helpers ----------
            from collections import Counter
            import datetime as dt, re, math
            from math import radians, sin, cos, asin, sqrt

            def norm(s): return (s or "").strip()
            def prov(f): return norm((f.get("agency") or "").upper())
            def sizeha(f):
                try: return float(f.get("size_ha") or 0.0)
                except: return 0.0
            def started_s(f):
                s = norm(f.get("started"));  return s[:10] if len(s) >= 10 else s
            def parse_date(s):
                try: return dt.date.fromisoformat((s or "")[:10])
                except: return None
            def control_code(f):
                code = norm((f.get("control_code") or "")).upper()
                if code in ("OC","BH","UC"): return code
                txt = norm((f.get("control") or f.get("status") or "")).lower()
                if "out of control" in txt: return "OC"
                if "being held" in txt:     return "BH"
                if "under control" in txt:  return "UC"
                return ""

            def haversine_km(lat1, lon1, lat2, lon2):
                R = 6371.0
                dlat = radians(lat2 - lat1); dlon = radians(lon2 - lon1)
                a = sin(dlat/2)**2 + cos(radians(lat1))*cos(radians(lat2))*sin(dlon/2)**2
                return 2 * R * asin(sqrt(a))

            # Use your existing geocoder (Google first, OpenCage fallback)
            def geocode_place(name: str):
                g = geocode_address(name, opencage_key, google_key)
                if not g: return None
                lat, lon, fmt, _src = g
                return float(lat), float(lon), fmt

            def parse_num(s: str):
                m = re.search(r'(-?\d+(?:\.\d+)?)', s or '')
                return float(m.group(1)) if m else None

            def parse_range(s: str):
                # "between 100 and 500", "100-500", ">= 50", "< 10", "over 20", "under 5", "= 0.1"
                s2 = (s or "").lower().replace(",", "")
                m = re.search(r'between\s+(-?\d+(?:\.\d+)?)\s+and\s+(-?\d+(?:\.\d+)?)', s2)
                if m: return float(m.group(1)), float(m.group(2))
                m = re.search(r'(-?\d+(?:\.\d+)?)\s*-\s*(-?\d+(?:\.\d+)?)', s2)
                if m: return float(m.group(1)), float(m.group(2))
                if "over" in s2 or ">=" in s2 or ">"  in s2: return (parse_num(s2),  math.inf)
                if "under" in s2 or "<=" in s2 or "<"  in s2: return (-math.inf, parse_num(s2))
                if any(k in s2 for k in ["= ", " ="," exact"," is "]):
                    n = parse_num(s2); return (n, n)
                return (None, None)

            def parse_days_window(s: str):
                # "last 7 days", "last 2 weeks", "older than 3 days"
                s2 = (s or "").lower()
                m = re.search(r'last\s+(\d+)\s+day', s2)
                if m:
                    d = int(m.group(1)); end = parse_date(date_label) or dt.date.today()
                    return (end - dt.timedelta(days=d), end, "last")
                m = re.search(r'last\s+(\d+)\s+week', s2)
                if m:
                    w = int(m.group(1)); end = parse_date(date_label) or dt.date.today()
                    return (end - dt.timedelta(days=7*w), end, "last")
                m = re.search(r'older\s+than\s+(\d+)\s+day', s2)
                if m:
                    d = int(m.group(1)); end = parse_date(date_label) or dt.date.today()
                    return (None, end - dt.timedelta(days=d), "older")
                return (None, None, "")

            # ---------- intent parse ----------
            text = (q or "").lower().strip()

            province_map = {
                " nb": "NB", "new brunswick": "NB",
                " ns": "NS", "nova scotia": "NS",
                " nl": "NL", "newfoundland": "NL",
                " pei": "PE", " prince edward island": "PE",
            }
            want_provs = [v for k,v in province_map.items() if k in (" " + text)]
            want_provs = list(dict.fromkeys(want_provs))  # unique

            want_ctrl = set()
            if "out of control" in text or " ooc" in (" "+text): want_ctrl.add("OC")
            if "being held"   in text or " bh"  in (" "+text): want_ctrl.add("BH")
            if "under control" in text or " uc" in (" "+text): want_ctrl.add("UC")

            lo, hi = parse_range(text)  # None/None if not present
            d_from, d_to, win_tag = parse_days_window(text)

            d_on = None
            m = re.search(r'(\d{4}-\d{2}-\d{2})', text)
            if m: d_on = parse_date(m.group(1))
            if "before" in text and d_on: d_from, d_to, d_on = (None, d_on, "")
            if "after"  in text and d_on: d_from, d_to, d_on = (d_on, None, "")

            # Distance / proximity
            want_within_km = None; anchor = None; place_label = ""
            m = re.search(r'within\s+(\d+(?:\.\d+)?)\s*km\s+of\s+(.+)', text)
            if m:
                want_within_km = float(m.group(1))
                anchor = geocode_place(m.group(2))
                place_label = (anchor[2] if anchor else m.group(2))
            if not anchor:
                m = re.search(r'(closest|nearest)\s+(?:to\s+)?(.+)', text)
                if m:
                    anchor = geocode_place(m.group(2))
                    place_label = (anchor[2] if anchor else m.group(2))
            if not anchor:
                m = re.search(r'(?:near|close to)\s+(.+)', text)
                if m:
                    anchor = geocode_place(m.group(1))
                    place_label = (anchor[2] if anchor else m.group(1))
                    if anchor: want_within_km = want_within_km or 30.0  # default radius

            # Heuristics:
            worry = ("worry" in text or "concern" in text) and ("near" in text or "close" in text)
            still_small = False; small_target = None; days_min = None
            m = re.search(r'still\s+(\d+(?:\.\d+)?)\s*ha', text)
            if m:
                small_target = float(m.group(1))
                m2 = re.search(r'after\s+(\d+)\s+day', text)
                if m2:
                    days_min = int(m2.group(1))
                    still_small = True

            names_only = ("list names" in text) or ("just names" in text)

            topN = None; smallest = False
            m = re.search(r'top\s+(\d+)', text)
            if m: topN = int(m.group(1))
            if "smallest" in text: smallest = True
            if "largest" in text:  smallest = False

            # ---------- apply filters ----------
            subset = fires[:]

            if want_provs:
                subset = [f for f in subset if prov(f) in want_provs]

            if want_ctrl:
                subset = [f for f in subset if control_code(f) in want_ctrl]

            if lo is not None or hi is not None:
                if lo is None: lo = -math.inf
                if hi is None: hi =  math.inf
                subset = [f for f in subset if lo <= sizeha(f) <= hi]

            if d_on:
                subset = [f for f in subset if parse_date(started_s(f)) == d_on]
            else:
                if d_from:
                    subset = [f for f in subset if (parse_date(started_s(f)) or dt.date(1900,1,1)) >= d_from]
                if d_to:
                    subset = [f for f in subset if (parse_date(started_s(f)) or dt.date(9999,1,1)) <= d_to]

            if still_small and small_target is not None and days_min is not None:
                today = parse_date(date_label) or dt.date.today()
                subset = [
                    f for f in subset
                    if abs(sizeha(f) - small_target) < 1e-9
                    and (today - (parse_date(started_s(f)) or today)).days >= days_min
                ]

            if anchor:
                lat0, lon0, _ = anchor
                subset = [f for f in subset if isinstance(f.get("lat"), (int,float)) and isinstance(f.get("lon"), (int,float))]
                for f in subset:
                    f["_dist_km"] = haversine_km(lat0, lon0, float(f["lat"]), float(f["lon"]))
                if want_within_km is not None:
                    subset = [f for f in subset if f.get("_dist_km", 1e9) <= want_within_km]

            # sorting
            if smallest:
                subset.sort(key=sizeha)
            elif "largest" in text or (topN and not smallest):
                subset.sort(key=sizeha, reverse=True)
            elif anchor and ("closest" in text or "nearest" in text or worry):
                subset.sort(key=lambda x: x.get("_dist_km", 1e9))
            elif "newest" in text:
                subset.sort(key=lambda f: parse_date(started_s(f)) or dt.date.min, reverse=True)
            elif "oldest" in text:
                subset.sort(key=lambda f: parse_date(started_s(f)) or dt.date.max)

            if worry and anchor:
                # within 25km, not UC, rank by size desc
                subset = [f for f in subset if f.get("_dist_km", 1e9) <= 25 and control_code(f) in ("OC","BH")]
                subset.sort(key=sizeha, reverse=True)

            if topN:
                subset = subset[:topN]

            # ---------- special summaries ----------
            if "counts by province" in text or ("by province" in text and "count" in text):
                counts = Counter(prov(f) for f in fires if prov(f))
                if counts:
                    st.markdown(f"**Active fires by province — {date_label}:**")
                    for p,c in counts.most_common():
                        st.write(f"- {p}: {c}")
                else:
                    st.info("No active fires found.")
                return

            if "most hectares" in text or ("hectares" in text and "most" in text):
                totals = {}
                for f in fires:
                    p = prov(f)
                    if p: totals[p] = totals.get(p, 0.0) + sizeha(f)
                if totals:
                    leader = max(totals, key=totals.get)
                    st.markdown(f"**Most hectares burning:** {leader} — {totals[leader]:,.1f} ha — {date_label}")
                    st.caption(", ".join(f"{p}: {v:,.1f} ha" for p,v in sorted(totals.items(), key=lambda x: x[1], reverse=True)))
                else:
                    st.info("No data.")
                return

            if names_only:
                if not subset:
                    st.info("No matching fires.")
                else:
                    st.markdown("**Matching fire names:**")
                    st.write(", ".join(sorted(set(norm(f.get('name')) for f in subset))))
                return

            # ---------- default response ----------
            if not subset:
                ongoing = int(raw.get("count_ongoing") or len(fires))
                st.markdown(f"**Snapshot for {date_label}:** {ongoing} ongoing fire(s).")
                st.write(
                    "Try: *fires over 20 ha in NB* · *closest to Truro* · *within 30 km of Moncton* · "
                    "*top 5 largest in NS* · *started last 7 days* · *older than 3 days* · "
                    "*still 0.1 ha after 2 days* · *fires to worry about near Halifax*"
                )
            else:
                hdr = f"**Matches — {date_label} ({len(subset)}):**"
                if anchor:
                    p = place_label or "that place"
                    if want_within_km is not None:
                        hdr = f"**Matches within {want_within_km:g} km of {p} — {date_label} ({len(subset)}):**"
                    elif "closest" in text or "nearest" in text:
                        hdr = f"**Closest to {p} — {date_label} (top {len(subset)}):**"
                st.markdown(hdr)

                for f in subset:
                    extra = []
                    if anchor and f.get("_dist_km") is not None:
                        extra.append(f"{f['_dist_km']:.1f} km")
                    extra_txt = f" · {', '.join(extra)}" if extra else ""
                    st.write(
                        f"- {f.get('name')} — {prov(f)} · {sizeha(f):,.1f} ha · "
                        f"{f.get('control','—')} · Started {started_s(f) or '—'}{extra_txt}"
                    )

                total = sum(sizeha(f) for f in subset)
                dates = [parse_date(started_s(f)) for f in subset if parse_date(started_s(f))]
                earliest = min(dates).isoformat() if dates else "—"
                latest   = max(dates).isoformat() if dates else "—"
                st.caption(f"Total size: {total:,.1f} ha · Earliest start: {earliest} · Newest start: {latest}")

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
