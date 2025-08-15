"""
Streamlit front-end that talks to TWO n8n workflows:
1) Active Fires (returns daily status + optional CSV + HTML summary)
2) Forest Risk Summary (weather + AI summary per city)

How to use:
- Put this file in your repo as app.py
- Add the webhook URLs as Streamlit secrets or environment variables:
    N8N_FIRES_URL, N8N_RISK_URL, N8N_SHARED_SECRET (optional)
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
import streamlit.components.v1 as components

# ---------- CONFIG ----------
st.set_page_config(page_title="SAFER - Sustainable Acadian Forests & Enviromental Risks", page_icon="🌲", layout="wide")

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
        h["X-API-KEY"] = secret
    return h

@st.cache_data(show_spinner=False)
def _cities_options() -> List[str]:
    return DEFAULT_CITIES

def call_fires_endpoint(url: str, secret: str | None) -> Dict[str, Any]:
    """POST to the Active Fires workflow. Expected JSON return shape:
    {
      "has_new_fires": bool,          # optional
      "fires_today": int,             # optional (we map to active fires)
      "summary": "...",               # text for UI
      "summary_html": "...",          # optional HTML designed for email-like layout
      "csv_base64": "...",            # optional base64 CSV
      "csv_filename": "filtered_acadian_fires.csv"  # optional
    }
    """
    resp = requests.post(url, headers=_headers(secret), json={"from": "streamlit"}, timeout=60)
    resp.raise_for_status()
    return resp.json()

# ✅ UPDATED: accept `focus` list and include it in the payload
def call_risk_endpoint(
    url: str,
    cities: List[str],
    question: str,
    detail: str,
    focus: List[str],                # <— NEW PARAM
    secret: str | None
) -> Dict[str, Any]:
    payload = {
        "cities": cities,
        "question": question,        # optional free text
        "detail": detail,
        "focus": focus,              # <— SEND FOCUS AS JSON ARRAY
        "from": "streamlit",
    }
    resp = requests.post(url, headers=_headers(secret), json=payload, timeout=90)
    resp.raise_for_status()
    # Some n8n nodes return plain text; keep robust
    try:
        return resp.json()
    except Exception:
        return {"reply": resp.text}

# ---------- SIDEBAR ----------
st.sidebar.title("🔌 Connections")
fires_url = st.sidebar.text_input("Active Fires webhook URL", value=N8N_FIRES_URL_DEFAULT, type="default")
risk_url = st.sidebar.text_input("Forest Risk webhook URL", value=N8N_RISK_URL_DEFAULT, type="default")
shared_secret = st.sidebar.text_input("Optional shared secret (X-API-KEY)", value=N8N_SHARED_SECRET_DEFAULT, type="password")

st.sidebar.markdown("—")
st.sidebar.caption("Tip: keep secrets in Streamlit → App → Settings → Secrets. Fields above simply let you override at runtime.")

# ---------- UI ----------
# ---- SAFER header (styles + markup) ----
st.markdown("""
<style>
:root {
  --beige: #f6f2ea;
  --ink: #1f2937;
  --pine: #1b6b3a;
  --pine-2: #2d8a4f;
  --bark: #7a3e1a;
}
.stApp { background: var(--beige); }

/* Header layout */
.s-header { padding: 6px 0 16px; margin: 0 0 8px; border-bottom: 1px solid #e6e0d4; }
.s-logo { display: flex; align-items: center; gap: 16px; }
.s-wordmark { line-height: 1.1; }
.s-acronym { font-weight: 800; font-size: 44px; letter-spacing: .3px; color: var(--ink); }
.s-sub { font-size: 18px; color: #374151; margin-top: 2px; }
.s-tag { font-size: 16px; font-style: italic; color: var(--pine); margin-top: 4px; }

/* Larger screens */
@media (min-width: 900px) {
  .s-acronym { font-size: 56px; }
  .s-sub { font-size: 20px; }
}
</style>
<div class="s-header">
  <div class="s-logo">
    <!-- Leaning white pine canopy -->
    <svg width="96" height="72" viewBox="0 0 120 90" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
      <!-- canopy layers -->
      <path d="M58 28 C35 45, 20 50, 8 52 C27 36, 56 20, 98 22 C84 28, 72 32, 58 28 Z" fill="var(--pine)"/>
      <path d="M63 22 C38 38, 22 46, 12 48 C32 32, 60 16, 105 18 C90 25, 76 28, 63 22 Z" fill="var(--pine-2)"/>
      <!-- trunk (slight lean) -->
      <rect x="49" y="40" width="8" height="30" rx="3" fill="var(--bark)" transform="skewX(-10)"/>
    </svg>

    <div class="s-wordmark">
      <div class="s-acronym">SAFER</div>
      <div class="s-sub">Sustainable Acadian Forests &amp; Environmental Risks</div>
      <div class="s-tag">Monitor, Maintain, Move Forward</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

fires_tab, risk_tab = st.tabs(["🔥 New/Ongoing Fires", "🛰️ Forest Risk Summary Today"])

with fires_tab:
    st.subheader("Check today's active fires in the Acadian region")
    st.write("This calls your n8n *Active Fires* workflow and shows a nicely formatted report. Your daily email workflow remains separate.")
    disabled = not bool(fires_url)
    if disabled:
        st.warning("Add the Active Fires webhook URL in the sidebar to enable this.")

    run = st.button("Run active fires check", type="primary", disabled=disabled)
    if run:
        with st.spinner("Contacting n8n …"):
            try:
                data = call_fires_endpoint(fires_url, shared_secret or None)
            except Exception as e:
                st.error(f"Request failed: {e}")
            else:
                st.success("Received response from n8n")

                # 1) Render email-style HTML if provided, otherwise show plain text
                if isinstance(data.get("summary_html"), str) and data["summary_html"].strip():
                    components.html(data["summary_html"], height=620, scrolling=True)
                else:
                    st.write(data.get("summary", "(No summary text returned)"))

                # 2) Metrics row
                cols = st.columns(3)
                if "fires_today" in data:
                    cols[0].metric("Fires today", data.get("fires_today"))
                if "has_new_fires" in data:
                    cols[1].metric("New fires?", "Yes" if data.get("has_new_fires") else "No")

                # 3) CSV download if provided
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
        # Cities from your helper; default to a single city
        cities = st.multiselect(
            "Cities",
            options=_cities_options(),
            default=["Fredericton,CA"]
        )

        detail = st.radio("Detail level", ["short", "detailed"], index=0, horizontal=True)

        # Explicit focus picker (no free text required)
        focus_topics = st.multiselect(
            "Focus on (choose one or more, or leave blank for a general summary)",
            options=["Precipitation", "Humidity", "Temperature", "Fire risk", "Flood risk"],
            help="If you pick topics, the AI will focus ONLY on these. Otherwise it gives a general summary."
        )

        # Optional: still build a strict instruction string for extra clarity
        if focus_topics:
            question = "Focus ONLY on: " + ", ".join(focus_topics) + ". Do NOT include other topics."
        else:
            question = ""

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
                        # ✅ UPDATED: pass `focus_topics` as an array
                        data = call_risk_endpoint(
                            risk_url, cities, question, detail, focus_topics, shared_secret or None
                        )
                    except Exception as e:
                        st.error(f"Request failed: {e}")
                    else:
                        # Prefer any of these keys; render as Markdown to keep headings/sections
                        reply = (
                            data.get("reply")
                            or data.get("text")
                            or data.get("summary")
                            or data.get("message")
                            or "(No reply text returned)"
                        )
                        st.markdown(reply)

                        # Optional: show returned metrics table if present
                        if isinstance(data.get("metrics"), list) and data["metrics"]:
                            try:
                                df = pd.DataFrame(data["metrics"])
                                st.dataframe(df, use_container_width=True)
                            except Exception:
                                st.info("Metrics were returned but could not be rendered as a table.")

# ---------- FOOTER ----------
st.markdown("""
---
**Notes**
- The Active Fires tab renders HTML from your n8n summary so it matches your email styling. If HTML is missing, it falls back to plain text.
- For security, prefer to keep n8n behind authentication and validate an `X-API-KEY` header in the workflows.
""")
