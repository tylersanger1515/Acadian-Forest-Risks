import os, io, json
import requests
import pandas as pd
import streamlit as st
import altair as alt

st.set_page_config(page_title="SAFER", layout="wide")

# ---------- Secrets ----------
FIRES_URL = st.secrets.get("N8N_FIRES_URL", "")
AI_RISK   = st.secrets.get("N8N_AI_RISK_URL", "")
AI_FIRES  = st.secrets.get("N8N_AI_FIRES_URL", "")
SUB_URL   = st.secrets.get("N8N_SUBSCRIBE_URL", "")

# ---------- HTTP helpers ----------
def post_any(url: str, payload: dict | None = None):
    """POST and return one of: dict (JSON), str (HTML), bytes (CSV)."""
    r = requests.post(url, json=payload or {}, timeout=90)
    r.raise_for_status()
    ctype = (r.headers.get("content-type") or "").lower()
    if "text/html" in ctype:
        return r.text                      # html map
    if "text/csv" in ctype or "application/csv" in ctype:
        return r.content                   # raw csv
    try:
        return r.json()                    # json
    except Exception:
        return r.text

def to_df_from_any(x):
    """Best-effort normalize n8n responses into a DataFrame."""
    if x is None:
        return pd.DataFrame([])
    if isinstance(x, bytes):
        return pd.read_csv(io.BytesIO(x))
    if isinstance(x, str):
        # raw CSV as string?
        if "," in x and "\n" in x:
            return pd.read_csv(io.StringIO(x))
        return pd.DataFrame([{"message": x[:4000]}])
    if isinstance(x, dict):
        if "fires" in x and isinstance(x["fires"], list):
            return pd.json_normalize(x["fires"])
        if "results" in x and isinstance(x["results"], list):
            return pd.json_normalize(x["results"])
        # flatten generic dict
        return pd.json_normalize(x)
    if isinstance(x, list):
        return pd.json_normalize(x)
    return pd.DataFrame([])

# ---------- AI helpers ----------
def detect_dataset(prompt: str, forced_output: str):
    p = (prompt or "").lower()
    fire_words = ("active fire","active fires","hotspot","hot spots",
                  "out of control","size","hectare","agency","frp","map")
    if forced_output == "map" or any(w in p for w in fire_words):
        return "fires"
    return "risk"

def detect_metric(prompt: str):
    p = (prompt or "").lower()
    if "wind" in p:     return "wind_kph"
    if "temp" in p:     return "temp_c"
    if "humid" in p:    return "humidity"
    if "cloud" in p:    return "cloud_pct"
    if "flood" in p:    return "flood_score"
    return "fire_score"

def risk_summary(df: pd.DataFrame, province: str):
    if df.empty or "fire_score" not in df:
        st.info("No risk rows returned.")
        return
    scope = f"in {province}" if province and province != "ALL" else "across the region"
    avg = df["fire_score"].mean()
    top = df.sort_values("fire_score", ascending=False).head(3)
    tops = ", ".join(f"{r.City} ({r.fireRisk})" if "City" in df.columns else f"{r.get('city','')}"
                     for _, r in top.iterrows())
    st.markdown(f"**Average fire score {scope}: {avg:.1f}.**  Top hot spots: **{tops}**.")

def risk_chart(df: pd.DataFrame, metric: str, chart_type: str):
    if df.empty:
        st.info("No data to chart.")
        return
    if metric not in df.columns:
        metric = "fire_score"
    base = alt.Chart(df).encode(
        x=alt.X("City:N" if "City" in df.columns else "city:N", sort="-y", title="City"),
        y=alt.Y(f"{metric}:Q", title=metric.replace("_"," ").title()),
        tooltip=[c for c in ["City","city","Province","province",metric] if c in df.columns]
    )
    st.altair_chart(base.mark_line(point=True) if chart_type=="line" else base.mark_bar(),
                    use_container_width=True)

def embed_html(html: str, height: int = 650):
    st.components.v1.html(html, height=height, scrolling=True)

# =====================  UI  =====================

tab1, tab2, tab3 = st.tabs(["🔥 Active Fires", "🤖 AI Agent", "🎆 SAFER Fire Alert"])

# --------------------- Tab 1 ---------------------
with tab1:
    st.header("Active Fires in the Acadian Region")

    st.caption("Press the button to fetch the latest active fires and explore them in a table or chart.")
    colA, colB = st.columns([1,5])
    with colA:
        run = st.button("Fetch Active Fires", type="primary")
    province_filter = st.selectbox("Province filter (optional)", ["ALL","NB","NS","PE","NL"], index=0)

    if run:
        if not FIRES_URL:
            st.error("N8N_FIRES_URL is not set in secrets.")
        else:
            # Ask your n8n flow for JSON/CSV; send a hint but handle anything it returns.
            resp = post_any(FIRES_URL, {"format": "json", "province": province_filter})
            df = to_df_from_any(resp)

            if df.empty:
                st.info("No fires returned.")
            else:
                # Basic cleanup of likely columns
                rename = {
                    "size_ha":"Size (ha)", "status":"Status", "agency":"Agency",
                    "name":"Name", "started":"Started", "lat":"Lat", "lon":"Lon",
                }
                for k,v in rename.items():
                    if k in df.columns: df.rename(columns={k:v}, inplace=True)

                st.dataframe(df, use_container_width=True, height=420)

                # Tiny summary & chart
                if "Status" in df.columns:
                    st.write("**By status**")
                    status_df = df["Status"].value_counts().reset_index()
                    status_df.columns = ["Status","Count"]
                    st.altair_chart(
                        alt.Chart(status_df).mark_bar().encode(x="Status:N", y="Count:Q"),
                        use_container_width=True
                    )

                st.download_button(
                    "Download CSV",
                    data=df.to_csv(index=False).encode("utf-8"),
                    file_name="active_fires.csv",
                    mime="text/csv",
                )

# --------------------- Tab 2 ---------------------
with tab2:
    st.header("Ask the AI about risk in the Acadian region")
    st.write("Explain what you want in plain English. The agent can return **summaries**, **maps**, or **charts** for NB, NS, PE, NL.")

    c1, c2 = st.columns([2,1])
    with c1:
        prompt = st.text_area("Your request", "Give a short narrative summary for all provinces today", height=100)
    with c2:
        province = st.selectbox("Province filter", ["ALL","NB","NS","PE","NL"], index=0)
        lookback = st.number_input("Lookback (days) for 'new' fires (optional)", 0, 60, 7, 1)

    st.write("Output")
    out = st.radio("", ["Auto","Map","Bar chart","Line chart","Table"], index=0, horizontal=True, label_visibility="collapsed")

    if st.button("Ask AI", type="primary"):
        forced = {"Auto":"auto","Map":"map","Bar chart":"bar","Line chart":"line","Table":"table"}[out]
        choice = detect_dataset(prompt, forced)

        if choice == "fires" or forced == "map":
            # Active fires path (returns HTML if you use the map branch)
            if not AI_FIRES:
                st.error("N8N_AI_FIRES_URL is not set.")
            else:
                payload = {"format": "map" if forced=="map" else "json",
                           "province": province, "since_days": int(lookback)}
                resp = post_any(AI_FIRES, payload)
                if isinstance(resp, str) and "<html" in resp.lower():
                    embed_html(resp)
                else:
                    df = to_df_from_any(resp)
                    st.dataframe(df, use_container_width=True)
        else:
            # Risk path
            if not AI_RISK:
                st.error("N8N_AI_RISK_URL is not set.")
            else:
                payload = {"province": province, "detail": "detailed", "question": prompt, "focus": []}
                resp = post_any(AI_RISK, payload)
                df = to_df_from_any(resp)

                if {"City","fire_score"}.issubset(df.columns) or "fire_score" in df.columns:
                    if forced == "auto":
                        risk_summary(df, province)
                        st.dataframe(df, use_container_width=True)
                    elif forced == "table":
                        st.dataframe(df, use_container_width=True)
                    else:
                        met = detect_metric(prompt)
                        risk_chart(df, met, "line" if forced=="line" else "bar")
                else:
                    # Fallback: show whatever came back
                    st.dataframe(df, use_container_width=True)

                if not df.empty:
                    st.download_button("Download results as CSV",
                        data=df.to_csv(index=False).encode("utf-8"),
                        file_name="risk_results.csv",
                        mime="text/csv")

# --------------------- Tab 3 ---------------------
with tab3:
    st.header("SAFER Fire Alert")
    st.caption("Subscribe to alerts. We’ll route to your n8n subscription webhook.")

    col1, col2 = st.columns(2)
    with col1:
        name  = st.text_input("Name")
        email = st.text_input("Email")
        addr  = st.text_input("Address (optional)")
        prov  = st.selectbox("Province", ["NB","NS","PE","NL"], index=0)
    with col2:
        radius = st.number_input("Radius (km)", 1, 250, 50)
        freq   = st.selectbox("Frequency", ["daily","weekly","instant"], index=0)
        notes  = st.text_input("Notes (optional)")

    if st.button("Subscribe", type="primary"):
        if not SUB_URL:
            st.error("N8N_SUBSCRIBE_URL is not set in secrets.")
        elif not email:
            st.warning("Email is required.")
        else:
            pay = {"name": name, "email": email, "address": addr, "province": prov,
                   "radius_km": int(radius), "frequency": freq, "notes": notes}
            try:
                resp = post_any(SUB_URL, pay)
                st.success("Subscription request sent.")
                with st.expander("Response (debug)"):
                    st.write(resp)
            except Exception as e:
                st.error(f"Subscription failed: {e}")
