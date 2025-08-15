# ---------- IMPORTS ----------
import json
import base64
import time
from datetime import datetime
from typing import Any, Dict, Optional

import requests
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

# ---------- CONFIG ----------
st.set_page_config(
    page_title="SAFER — Sustainable Acadian Forests & Environmental Risks",
    page_icon="🌲",
    layout="wide",
)

# ---------- STYLES ----------
_STYLES = """
<style>
:root {
  --beige:#f6f2ea;
  --ink:#1f2937;
  --pine:#1b6b3a;
  --pine-2:#2d8a4f;
  --bark:#7a3e1a;
}
.stApp { background:var(--beige); }

/* Header */
.s-header { 
  padding:6px 0 16px; 
  margin:0 0 8px; 
  border-bottom:1px solid #e6e0d4; 
  box-shadow: 0 1px 0 rgba(0,0,0,0.04);
}
.s-logo { display:flex; align-items:center; gap:16px; }
.s-wordmark { line-height:1.1; }
.s-acronym { font-weight:800; font-size:44px; letter-spacing:.3px; color:var(--ink); }
.s-sub { font-size:18px; color:#374151; margin-top:2px; }
.s-tag { font-size:16px; font-style:italic; color:var(--pine); margin-top:4px; }

/* Larger screens */
@media (min-width:900px){
  .s-acronym{ font-size:56px; }
  .s-sub{ font-size:20px; }
}

/* ---- Extra app-wide polish ---- */

/* Center content and set comfortable width; tighten padding */
.block-container { 
  max-width: 1200px; 
  margin: 0 auto; 
  padding-top: 1.25rem; 
  padding-bottom: 2rem; 
}

/* Space between header and first tab list */
[data-baseweb="tab-list"] { margin-top: 4px; }

/* Brand primary buttons */
div.stButton > button {
  background: var(--pine);
  color: #fff;
  border: 1px solid #165a31;
}
div.stButton > button:hover {
  background: #175f35;
  border-color: #134e2b;
}

/* Tab underline + hover */
[data-baseweb="tab-list"] button[role="tab"][aria-selected="true"] {
  border-bottom: 2px solid var(--pine);
}
[data-baseweb="tab-list"] button[role="tab"]:hover {
  background: rgba(27,107,58,0.06);
}

/* Code blocks blend with beige */
.stCodeBlock, pre, code {
  background: #f2ecdf !important;
}

/* Links use pine color */
a { color: var(--pine); }
a:hover { text-decoration: underline; }
</style>
"""

# ---------- HEADER ----------
_HEADER = """
<div class="s-header">
  <div class="s-logo">
    <!-- Leaning white pine -->
    <svg width="96" height="72" viewBox="0 0 120 90" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
      <!-- Canopy layers -->
      <path d="M58 28 C35 45, 20 50, 8 52 C27 36, 56 20, 98 22 C84 28, 72 32, 58 28 Z" fill="var(--pine)"/>
      <path d="M63 22 C38 38, 22 46, 12 48 C32 32, 60 16, 105 18 C90 25, 76 28, 63 22 Z" fill="var(--pine-2)"/>
      <!-- Trunk (slight lean) -->
      <rect x="49" y="40" width="8" height="30" rx="3" fill="var(--bark)" transform="skewX(-10)"/>
    </svg>

    <div class="s-wordmark">
      <div class="s-acronym">SAFER</div>
      <div class="s-sub">Sustainable Acadian Forests &amp; Environmental Risks</div>
      <div class="s-tag">Monitor, Maintain, Move Forward</div>
    </div>
  </div>
</div>
"""

st.markdown(_STYLES, unsafe_allow_html=True)
st.markdown(_HEADER, unsafe_allow_html=True)

# ---------- HELPERS ----------
def _headers(secret: Optional[str]) -> Dict[str, str]:
    h = {"Content-Type": "application/json"}
    if secret:
        # If you pass an API token in secrets, add it as Bearer
        h["Authorization"] = f"Bearer {secret}"
    return h

def post_json(url: str, body: Dict[str, Any], secret: Optional[str], timeout: int = 60) -> Dict[str, Any]:
    """POST JSON and return dict. If response isn't JSON, fall back to {'summary': text}."""
    resp = requests.post(url, headers=_headers(secret), json=body, timeout=timeout)
    resp.raise_for_status()
    try:
        return resp.json()
    except Exception:
        return {"summary": resp.text}

def _b64_download_link(csv_text: str, filename: str = "report.csv", link_label: str = "Download CSV") -> str:
    b64 = base64.b64encode(csv_text.encode("utf-8")).decode("utf-8")
    return f'<a href="data:text/csv;base64,{b64}" download="{filename}">{link_label}</a>'

def extract_fires_payload(data: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize possible keys from your n8n 'active fires' webhook."""
    # expected keys (adjust if your webhook differs)
    date = data.get("date") or data.get("run_date")
    has_new = data.get("hasNew")
    count_new = data.get("count_new") or data.get("new_count")
    count_ongoing = data.get("count_ongoing") or data.get("ongoing_count")
    summary_html = data.get("summary_html")
    summary_text = data.get("summary_text") or data.get("summary")
    subject = data.get("subject")

    # CSV: support CSV returned as text OR via url
    csv_text = data.get("csv_text")
    csv_url = data.get("csv_url")
    return {
        "date": date,
        "has_new": has_new,
        "count_new": count_new,
        "count_ongoing": count_ongoing,
        "summary_html": summary_html,
        "summary_text": summary_text,
        "subject": subject,
        "csv_text": csv_text,
        "csv_url": csv_url,
    }

def extract_risk_payload(data: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize possible keys from your risk summary webhook."""
    summary_html = data.get("summary_html") or data.get("html")
    summary_text = data.get("summary_text") or data.get("summary")
    title = data.get("title") or data.get("subject")
    return {
        "summary_html": summary_html,
        "summary_text": summary_text,
        "title": title,
    }

# ---------- SIDEBAR ----------
st.sidebar.header("Configuration")
fires_url = st.sidebar.text_input(
    "Active Fires Webhook URL",
    value=st.secrets.get("FIRES_WEBHOOK", "")
)
risk_url = st.sidebar.text_input(
    "Risk Summary Webhook URL",
    value=st.secrets.get("RISK_WEBHOOK", "")
)
api_secret = st.sidebar.text_input(
    "Auth Token (optional)",
    value=st.secrets.get("API_TOKEN", ""),
    type="password"
)
timeout_sec = st.sidebar.slider("Request timeout (seconds)", 10, 120, 60)

st.sidebar.markdown("---")
st.sidebar.caption("Tip: store values in `.streamlit/secrets.toml` as FIRES_WEBHOOK, RISK_WEBHOOK, API_TOKEN")

# ---------- TABS ----------
tab1, tab2 = st.tabs(["🔥 Active Fires", "🧭 Risk Summary"])

# ===== TAB 1: ACTIVE FIRES =====
with tab1:
    st.subheader("Active Fires in the Acadian Region")
    st.write("Fetch latest data, summary, and CSV for NB / NS / PE (plus nearby if applicable).")

    col_a, col_b = st.columns([1,1])
    with col_a:
        run_fires = st.button("Fetch Active Fires")
    with col_b:
        today_str = datetime.now().strftime("%Y-%m-%d %H:%M")

    if run_fires:
        if not fires_url:
            st.error("Please set your **Active Fires Webhook URL** in the sidebar.")
        else:
            with st.spinner("Requesting active fires report…"):
                try:
                    body = {"from": "streamlit"}  # add parameters if your workflow expects any
                    data = post_json(fires_url, body, api_secret, timeout=timeout_sec)
                    payload = extract_fires_payload(data)

                    # Metrics / status row
                    m1, m2, m3 = st.columns(3)
                    m1.metric("New fires", payload.get("count_new", 0))
                    m2.metric("Ongoing fires", payload.get("count_ongoing", 0))
                    m3.metric("Report time", payload.get("date") or today_str)

                    # HTML summary preferred; fallback to text
                    html = payload.get("summary_html")
                    if isinstance(html, str) and html.strip():
                        components.html(html, height=820, scrolling=True)
                    else:
                        txt = payload.get("summary_text") or "(No summary text returned)"
                        st.write(txt)

                    # CSV handling
                    csv_text = payload.get("csv_text")
                    csv_url = payload.get("csv_url")
                    if csv_text and isinstance(csv_text, str):
                        st.markdown(_b64_download_link(csv_text, filename="acadian_fires.csv"), unsafe_allow_html=True)
                        try:
                            df = pd.read_csv(pd.compat.StringIO(csv_text))
                            st.dataframe(df, use_container_width=True)
                        except Exception:
                            pass
                    elif csv_url and isinstance(csv_url, str):
                        st.markdown(f"[Download CSV]({csv_url})")

                    st.success("Received response from n8n")
                    st.toast("Active fires report loaded", icon="✅")

                except requests.HTTPError as e:
                    st.error(f"HTTP error: {e.response.status_code} {e.response.text[:500]}")
                except requests.RequestException as e:
                    st.error(f"Request failed: {e}")
                except Exception as e:
                    st.error(f"Unexpected error: {e}")

# ===== TAB 2: RISK SUMMARY =====
with tab2:
    st.subheader("Sustainable Management & Risk Summary")
    st.write("Ask for a focused sustainability, fire/flood risk, or environmental insight summary.")

    default_topics = "fire risk, flood risk, sustainability impacts, weather outlook"
    topics = st.text_input("Focus topics (comma-separated)", value=default_topics)

    col1, col2 = st.columns([1,1])
    with col1:
        ask_risk = st.button("Generate Summary")
    with col2:
        st.caption("The response can include HTML or plain text; both are supported.")

    if ask_risk:
        if not risk_url:
            st.error("Please set your **Risk Summary Webhook URL** in the sidebar.")
        else:
            with st.spinner("Requesting risk summary…"):
                try:
                    body = {"focus_topics": [t.strip() for t in topics.split(",") if t.strip()]}
                    data = post_json(risk_url, body, api_secret, timeout=max(60, timeout_sec))
                    payload = extract_risk_payload(data)

                    title = payload.get("title")
                    if title:
                        st.markdown(f"### {title}")

                    html = payload.get("summary_html")
                    if isinstance(html, str) and html.strip():
                        components.html(html, height=820, scrolling=True)
                    else:
                        txt = payload.get("summary_text") or "(No summary text returned)"
                        st.write(txt)

                    st.success("Received response from n8n")
                    st.toast("Risk summary ready", icon="🌲")

                except requests.HTTPError as e:
                    st.error(f"HTTP error: {e.response.status_code} {e.response.text[:500]}")
                except requests.RequestException as e:
                    st.error(f"Request failed: {e}")
                except Exception as e:
                    st.error(f"Unexpected error: {e}")

# ---------- NOTES ----------
# If your webhooks return different keys, adjust extract_fires_payload / extract_risk_payload above.
# For CSV: if n8n sends the CSV as a base64 string instead, decode and use the same download link pattern.
