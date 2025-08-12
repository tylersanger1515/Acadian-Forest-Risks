"""
Streamlit front‑end that talks to TWO n8n workflows:
1) Active Fires (returns daily status + optional CSV)
2) Forest Risk Summary (weather + AI summary per city)

How to use:
- Put this file in your repo as app.py
- Add the two webhook URLs as Streamlit secrets or environment variables:
    N8N_FIRES_URL, N8N_RISK_URL
  (optionally add N8N_SHARED_SECRET if your workflows check a header)
- requirements.txt should include: streamlit, requests, pandas
- Deploy on Streamlit Cloud and set the secrets under app settings.
"""
from __future__ import annotations
import os
import json
import base64
from typing import List, Dict, Any

import requests
import pandas as pd
import streamlit as st

# ---------- CONFIG ----------
st.set_page_config(page_title="Acadian Forest – Fire & Risk Assistant", page_icon="🌲", layout="wide")

# Read endpoints from Streamlit secrets first, fall back to env vars, then manual sidebar input.
N8N_FIRES_URL_DEFAULT = st.secrets.get("N8N_FIRES_URL", os.getenv("N8N_FIRES_URL", ""))
N8N_RISK_URL_DEFAULT = st.secrets.get("N8N_RISK_URL", os.getenv("N8N_RISK_URL", ""))
N8N_SHARED_SECRET_DEFAULT = st.secrets.get("N8N_SHARED_SECRET", os.getenv("N8N_SHARED_SECRET", ""))

# A small, curated list of cities for quick selection (you can tweak freely)
DEFAULT_CITIES = [
    "Fredericton,CA", "Moncton,CA", "Saint John,CA", "Bathurst,CA", "Miramichi,CA",
    "Charlottetown,CA", "Summerside,CA",
    "Halifax,CA", "Dartmouth,CA", "Sydney,CA", "Yarmouth,CA", "Truro,CA"
]

# ---------- HELPERS ----------
def _headers(secret: str | None) -> Dict[str, str]:
    h = {"Content-Type": "application/json"}
    if secret:
        # Your n8n workflow can check this header to accept or reject requests
        h["X-API-KEY"] = secret
    return h

@st.cache_data(show_spinner=False)
def _cities_options() -> List[str]:
    return DEFAULT_CITIES

def call_fires_endpoint(url: str, secret: str | None) -> Dict[str, Any]:
    """POST to the Active Fires workflow. Expected JSON return shape:
    {
      "has_new_fires": bool,          # optional
      "fires_today": int,             # optional
      "summary": "...",               # text for UI
      "csv_base64": "...",            # optional base64 CSV
      "csv_filename": "filtered_acadian_fires.csv"  # optional
    }
    """
    resp = requests.post(url, headers=_headers(secret), json={"from": "streamlit"}, timeout=60)
    resp.raise_for_status()
    return resp.json()

def call_risk_endpoint(url: str, cities: List[str], question: str, detail: str, secret: str | None) -> Dict[str, Any]:
    """POST to the Forest Risk Summary workflow. Expected JSON return shape:
    {
      "reply": "AI written summary",
      "cities": ["Fredericton,CA", ...],
      "metrics": [
          {"city": "Fredericton,CA", "temp": 22, "humidity": 55, "precip": 1.2,
           "fire_risk": "Moderate", "flood_risk": "Low"}, ...
      ]
    }
    """
    payload = {
        "cities": cities,
        "question": question,
        "detail": detail,
        "from": "streamlit"
    }
    resp = requests.post(url, headers=_headers(secret), json=payload, timeout=90)
    resp.raise_for_status()
    return resp.json()

# ---------- SIDEBAR ----------
st.sidebar.title("🔌 Connections")
fires_url = st.sidebar.text_input("Active Fires webhook URL", value=N8N_FIRES_URL_DEFAULT, type="default", help="The Production URL from your n8n Webhook node in the Active Fires workflow.")
risk_url = st.sidebar.text_input("Forest Risk webhook URL", value=N8N_RISK_URL_DEFAULT, type="default", help="The Production URL from your n8n Webhook node in the Risk Summary workflow.")
shared_secret = st.sidebar.text_input("Optional shared secret (X-API-KEY)", value=N8N_SHARED_SECRET_DEFAULT, type="password")

st.sidebar.markdown("—")
st.sidebar.caption("Tip: keep secrets in Streamlit → App → Settings → Secrets. Fields above simply let you override at runtime.")

# ---------- UI ----------
st.title("🌲 Acadian Forest – Fire & Risk Assistant")
st.caption("Chat-style front‑end that calls your n8n workflows for live data and AI summaries.")

fires_tab, risk_tab = st.tabs(["🔥 Active Fires", "🛰️ Forest Risk Summary"])

with fires_tab:
    st.subheader("Check today's active fires in the Acadian region")
    st.write("This calls your n8n *Active Fires* workflow and shows the summary. Your daily email workflow remains separate.")
    disabled = not bool(fires_url)
    if disabled:
        st.warning("Add the Active Fires webhook URL in the sidebar to enable this.")
    if st.button("Run active fires check", type="primary", disabled=disabled):
        with st.spinner("Contacting n8n …"):
            try:
                data = call_fires_endpoint(fires_url, shared_secret or None)
            except Exception as e:
                st.error(f"Request failed: {e}")
            else:
                st.success("Received response from n8n")
                st.write(data.get("summary", "(No summary text returned)"))
                # Show metrics if present
                cols = st.columns(3)
                if "fires_today" in data:
                    cols[0].metric("Fires today", data.get("fires_today"))
                if "has_new_fires" in data:
                    cols[1].metric("New fires?", "Yes" if data.get("has_new_fires") else "No")
                # CSV download if provided
                if data.get("csv_base64"):
                    try:
                        csv_bytes = base64.b64decode(data["csv_base64"])  # bytes
                        csv_name = data.get("csv_filename", "fires.csv")
                        st.download_button("Download filtered CSV", csv_bytes, file_name=csv_name, mime="text/csv")
                    except Exception:
                        st.info("CSV attachment was provided but could not be decoded.")

with risk_tab:
    st.subheader("Ask for risk summaries by city")
    left, right = st.columns([0.55, 0.45])
    with left:
        cities = st.multiselect("Cities", options=_cities_options(), default=["Fredericton,CA", "Halifax,CA"])
        detail = st.radio("Detail level", ["short", "detailed"], horizontal=True)
        question = st.text_area("Optional question for the AI (e.g., 'Focus on wind and precipitation'). Leave blank for a general summary.", height=80)
        go = st.button("Get risk summary", type="primary", disabled=not bool(risk_url))
    with right:
        if not risk_url:
            st.warning("Add the Forest Risk webhook URL in the sidebar to enable this.")
        if go:
            if not cities:
                st.error("Pick at least one city.")
            else:
                with st.spinner("Asking n8n for weather + AI summary …"):
                    try:
                        data = call_risk_endpoint(risk_url, cities, question, detail, shared_secret or None)
                    except Exception as e:
                        st.error(f"Request failed: {e}")
                    else:
                        # Text reply from the AI agent
                        st.write(data.get("reply", "(No reply text returned)"))
                        # Tabular metrics if provided
                        if isinstance(data.get("metrics"), list) and data["metrics"]:
                            try:
                                df = pd.DataFrame(data["metrics"])  # expect columns: city,temp,humidity,precip,fire_risk,flood_risk
                                st.dataframe(df, use_container_width=True)
                            except Exception:
                                st.info("Metrics were returned but could not be rendered as a table.")

# ---------- FOOTER ----------
st.markdown("""
---
**Notes**
- This front‑end is read‑only: it calls your n8n webhooks and displays responses. Your existing scheduled emails remain unchanged.
- For security, prefer to keep your n8n behind authentication and validate an `X-API-KEY` header in the workflow.
""")
