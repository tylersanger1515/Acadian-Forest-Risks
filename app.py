# ---------- IMPORTS ----------
from __future__ import annotations
import os, json, base64, re
from typing import List, Dict, Any, Optional

import requests
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

# ---------- CONFIG ----------
IMG_PATH = "assets/images/fokabs image.jpg"  # your uploaded image
PAGE_ICON = IMG_PATH if os.path.exists(IMG_PATH) else "🌲"

st.set_page_config(
    page_title="SAFER — Sustainable Acadian Forests & Environmental Risks",
    page_icon=PAGE_ICON,
    layout="wide",
)

# ---------- SECRETS / DEFAULTS ----------
N8N_FIRES_URL_DEFAULT = st.secrets.get("N8N_FIRES_URL", os.getenv("N8N_FIRES_URL", ""))
N8N_RISK_URL_DEFAULT  = st.secrets.get("N8N_RISK_URL",  os.getenv("N8N_RISK_URL",  ""))
N8N_SUBSCRIBE_URL_DEFAULT = st.secrets.get("N8N_SUBSCRIBE_URL", os.getenv("N8N_SUBSCRIBE_URL", ""))
N8N_SHARED_SECRET_DEFAULT = st.secrets.get("N8N_SHARED_SECRET", os.getenv("N8N_SHARED_SECRET", ""))

DEFAULT_CITIES = [
    "Fredericton,CA", "Moncton,CA", "Saint John,CA", "Bathurst,CA", "Miramichi,CA",
    "Charlottetown,CA", "Summerside,CA", "Halifax,CA", "Dartmouth,CA",
    "Sydney,CA", "Yarmouth,CA", "Truro,CA"
]

# ---------- STYLES ----------
_STYLES = """
<style>
:root{
  --beige:#f6f2ea; --ink:#1f2937;
  --pine:#0f5132;      /* darker green for SAFER */
  --pine-2:#2d8a4f; --bark:#7a3e1a;
}
.stApp{ background:var(--beige); }

/* Slim Streamlit chrome */
header[data-testid="stHeader"]{ background:var(--beige); box-shadow:none; min-height:32px; height:32px; padding:0; }
header[data-testid="stHeader"] > div{ height:32px; }
div[data-testid="stDecoration"]{ display:none; }

/* Layout + tabs polish */
.block-container{ max-width:1200px; margin:0 auto; padding-top:1rem; padding-bottom:2rem; }
[data-baseweb="tab-list"]{ margin-top:4px; }
[data-baseweb="tab-list"] button[role="tab"][aria-selected="true"]{ border-bottom:2px solid var(--pine); }

/* SAFER header (no pine icon) */
.s-header{ margin-top:6px; padding:8px 0 16px; margin-bottom:8px; border-bottom:1px solid #e6e0d4; }
.s-title{ display:flex; align-items:center; gap:12px; }  /* left block holds text */
.s-text{ display:block; }
.s-line1{ display:flex; align-items:baseline; gap:12px; } /* SAFER + FOKABS */
.s-acronym{ font-weight:800; font-size:56px; letter-spacing:.3px; color:var(--pine); } /* dark green */
.s-sub{ font-size:20px; color:#374151; margin-top:0; }  /* sits directly under the S of SAFER */
.s-tag{ font-size:16px; font-style:italic; color:var(--pine-2); margin-top:4px; }

@media (max-width:700px){
  .s-acronym{ font-size:42px; }
}
</style>
"""

def _fokabs_logo(height=44) -> str:
    """Image shown AFTER 'SAFER' — base64 inlined to prevent path issues."""
    if os.path.exists(IMG_PATH):
        ext = os.path.splitext(IMG_PATH)[1].lower()
        mime = "image/png" if ext == ".png" else "image/jpeg"
        b64 = base64.b64encode(open(IMG_PATH, "rb").read()).decode("ascii")
        return (
            f'<img src="data:{mime};base64,{b64}" alt="FOKABS" '
            f'style="height:{height}px;width:auto;vertical-align:baseline;border-radius:10px;padding:6px 10px;background:#fff;box-shadow:0 1px 3px rgba(0,0,0,.08);" />'
        )
    return '<span style="font-size:28px;vertical-align:baseline">🌐</span>'

# Header HTML (no leading spaces so Markdown won't render a code block)
_HEADER = (
    '<div class="s-header">'
    '  <div class="s-title">'
    '    <div class="s-text">'
    '      <div class="s-line1">'
    '        <span class="s-acronym">SAFER</span>'
    f'        {_fokabs_logo(44)}'
    '      </div>'
    '      <div class="s-sub">Sustainable Acadian Forests &amp; Environmental Risks</div>'
    '      <div class="s-tag">Monitor, Maintain, Move Forward</div>'
    '    </div>'
    '  </div>'
    '</div>'
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

def extract_fires_payload(d: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "summary_html": d.get("summary_html"),
        "summary_text": d.get("summary") or d.get("summary_text"),
        "has_new": d.get("has_new_fires"),
        "fires_today": d.get("fires_today"),
        "csv_base64": d.get("csv_base64"),
        "csv_filename": d.get("csv_filename", "fires.csv"),
    }

def extract_risk_payload(d: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "summary_html": d.get("summary_html") or d.get("html"),
        "summary_text": d.get("summary_text") or d.get("summary") or d.get("reply") or d.get("message"),
        "title": d.get("title") or d.get("subject"),
    }

def _valid_email(x: str) -> bool:
    return bool(re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", x or ""))

# ---------- SIDEBAR ----------
st.sidebar.title("🔌 Connections")
fires_url = st.sidebar.text_input("Active Fires webhook URL", value=N8N_FIRES_URL_DEFAULT)
risk_url  = st.sidebar.text_input("Forest Risk webhook URL", value=N8N_RISK_URL_DEFAULT)
subscribe_url = st.sidebar.text_input("Subscribe webhook URL", value=N8N_SUBSCRIBE_URL_DEFAULT)
shared_secret = st.sidebar.text_input("Optional shared secret (X-API-KEY)", value=N8N_SHARED_SECRET_DEFAULT, type="password")
timeout_sec = st.sidebar.slider("Request timeout (seconds)", 10, 120, 60)
st.sidebar.caption("Store secrets in `.streamlit/secrets.toml` (N8N_FIRES_URL, N8N_RISK_URL, N8N_SUBSCRIBE_URL, N8N_SHARED_SECRET).")

# ---------- TABS ----------
t1, t2, t3 = st.tabs(["🔥 Active Fires", "🧭 Risk Summary", "📬 Subscribe"])

# ===== TAB 1: ACTIVE FIRES =====
with t1:
    st.subheader("Active Fires in the Acadian Region")
    st.write("Fetch latest data, summary, and CSV for NB / NS / PE (plus nearby if applicable).")

    run = st.button("Fetch Active Fires", type="primary", disabled=not bool(fires_url))
    if not fires_url:
        st.warning("Add the Active Fires webhook URL in the sidebar to enable this.")

    if run and fires_url:
        with st.spinner("Requesting active fires report…"):
            try:
                data = post_json(fires_url, {"from": "streamlit"}, shared_secret or None, timeout=timeout_sec)
                p = extract_fires_payload(data)

                # Summary (prefer HTML)
                if isinstance(p["summary_html"], str) and p["summary_html"].strip():
                    components.html(p["summary_html"], height=820, scrolling=True)
                else:
                    st.write(p["summary_text"] or "(No summary text returned)")

                # Metrics
                cols = st.columns(3)
                if p["fires_today"] is not None: cols[0].metric("Fires today", p["fires_today"])
                if p["has_new"] is not None: cols[1].metric("New fires?", "Yes" if p["has_new"] else "No")

                # CSV
                if p["csv_base64"]:
                    try:
                        csv_bytes = base64.b64decode(p["csv_base64"])
                        st.download_button("Download filtered CSV", csv_bytes, file_name=p["csv_filename"], mime="text/csv")
                    except Exception:
                        st.info("CSV attachment was provided but could not be decoded.")

                st.success("Received response from n8n")
                st.toast("Active fires report loaded", icon="✅")
            except requests.HTTPError as e:
                st.error(f"HTTP error: {e.response.status_code} {e.response.text[:400]}")
            except requests.RequestException as e:
                st.error(f"Request failed: {e}")
            except Exception as e:
                st.error(f"Unexpected error: {e}")

# ===== TAB 2: RISK SUMMARY =====
with t2:
    st.subheader("Sustainable Management & Risk Summary")
    st.write("Ask for a sustainability, fire/flood risk, or environmental insight summary.")

    cities = st.multiselect("Cities", options=DEFAULT_CITIES, default=["Fredericton,CA"])
    detail = st.radio("Detail level", ["short", "detailed"], index=0, horizontal=True)

    go = st.button("Get risk summary", type="primary", disabled=not bool(risk_url))
    if not risk_url:
        st.warning("Add the Forest Risk webhook URL in the sidebar to enable this.")

    if go and risk_url:
        with st.spinner("Requesting risk summary…"):
            try:
                # Removed focus_topics/question. Keep it simple.
                payload = {"cities": cities, "detail": detail, "from": "streamlit"}
                data = post_json(risk_url, payload, shared_secret or None, timeout=max(60, timeout_sec))
                p = extract_risk_payload(data)

                if p["title"]:
                    st.markdown(f"### {p['title']}")
                if isinstance(p["summary_html"], str) and p["summary_html"].strip():
                    components.html(p["summary_html"], height=820, scrolling=True)
                else:
                    st.write(p["summary_text"] or "(No summary text returned)")

                st.success("Received response from n8n")
                st.toast("Risk summary ready", icon="🌲")
            except requests.HTTPError as e:
                st.error(f"HTTP error: {e.response.status_code} {e.response.text[:400]}")
            except requests.RequestException as e:
                st.error(f"Request failed: {e}")
            except Exception as e:
                st.error(f"Unexpected error: {e}")


# ===== TAB 3: SUBSCRIBE =====
with t3:
    st.subheader("SAFER Fire Alerts")
    st.write("Enter your location and radius to receive **critical proximity alerts** and the **daily Acadian fires update**.")

    if not subscribe_url:
        st.warning("Add the Subscribe webhook URL in the sidebar to enable this form.")

    with st.form("sub_form", clear_on_submit=False):
        email = st.text_input("Email", placeholder="you@example.com")
        colA, colB = st.columns(2)
        lat = colA.number_input("Latitude", value=46.1675, step=0.0001, format="%.6f")
        lon = colB.number_input("Longitude", value=-64.7508, step=0.0001, format="%.6f")
        radius = st.number_input("Radius (km)", min_value=1, max_value=250, value=10, step=1)
        active = st.checkbox("Activate alerts", value=True)
        submitted = st.form_submit_button("Save subscription", type="primary", disabled=not bool(subscribe_url))

    if submitted and subscribe_url:
        errs = []
        if not _valid_email(email): errs.append("Please enter a valid email.")
        if abs(lat) > 90: errs.append("Latitude must be between -90 and 90.")
        if abs(lon) > 180: errs.append("Longitude must be between -180 and 180.")
        if not (1 <= int(radius) <= 250): errs.append("Radius must be 1–250 km.")
        if errs:
            for e in errs: st.error(e)
        else:
            body = {
                "email": email.strip(),
                "lat": float(lat),
                "lon": float(lon),
                "radius_km": int(radius),
                "active": bool(active),
                "from": "streamlit"
            }
            with st.spinner("Saving subscription…"):
                try:
                    resp = post_json(subscribe_url, body, shared_secret or None, timeout=timeout_sec)
                    msg = resp.get("message") or resp.get("msg") or resp.get("status") or "Subscription saved."
                    st.success(msg)
                    st.json(resp)
                except requests.HTTPError as e:
                    st.error(f"HTTP error: {e.response.status_code} {e.response.text[:400]}")
                except requests.RequestException as e:
                    st.error(f"Request failed: {e}")
                except Exception as e:
                    st.error(f"Unexpected error: {e}")

# ---------- FOOTER ----------
st.markdown("""
---
**Notes**
- Active Fires tab prefers HTML from your n8n summary (falls back to plain text).
- Keep n8n behind auth and validate `X-API-KEY` in your workflows for security.
- Subscribe tab posts to your n8n Webhook (Subscribe-Manage) with `email`, `lat`, `lon`, `radius_km`, and `active`.
""")
