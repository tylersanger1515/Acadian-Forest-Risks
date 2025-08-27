# app.py — SAFER
import os, json, io
import requests
import pandas as pd
import streamlit as st
import altair as alt

st.set_page_config(page_title="SAFER", layout="wide")

# ---------- Secrets ----------
RISK_URL  = st.secrets.get("N8N_AI_RISK_URL",  "")
FIRES_AI  = st.secrets.get("N8N_AI_FIRES_URL", "")
# (Keep any other secrets you already use in Tab 1 / Tab 3)

# ---------- Helpers ----------
def _post_json(url: str, payload: dict):
    r = requests.post(url, json=payload, timeout=60)
    r.raise_for_status()
    # Fires-map branch returns HTML, not JSON
    ctype = r.headers.get("content-type","").lower()
    if "text/html" in ctype:
        return {"_html": r.text}
    return r.json()

def detect_dataset(prompt: str, forced_output: str):
    p = (prompt or "").lower()
    # Any clear mention of active fires / hotspots OR user forces Map => fire webhook
    fire_words = ("active fire", "active fires", "hotspot", "hot spots",
                  "out of control", "size", "hectare", "agency", "frp")
    if forced_output == "map" or any(w in p for w in fire_words):
        return "fires"
    # Otherwise, default to risk
    return "risk"

def detect_metric(prompt: str):
    p = (prompt or "").lower()
    if "wind" in p:      return "wind_kph"
    if "temp" in p:      return "temp_c"
    if "humid" in p:     return "humidity"
    if "cloud" in p:     return "cloud_pct"
    if "flood" in p:     return "flood_score"
    # default (risk)
    return "fire_score"

def make_risk_df(results):
    # Flatten the per-city rows from risk webhook
    if not isinstance(results, list): 
        return pd.DataFrame([])
    rows = []
    for r in results:
        rows.append({
            "City":       r.get("city",""),
            "Province":   r.get("province",""),
            "temp_c":     r.get("temp_c"),
            "humidity":   r.get("humidity"),
            "rain_mm":    r.get("rain_mm"),
            "wind_kph":   r.get("wind_kph"),
            "cloud_pct":  r.get("cloud_pct"),
            "fire_score": r.get("fire_score"),
            "fireRisk":   r.get("fireRisk"),
            "flood_score":r.get("flood_score"),
            "floodRisk":  r.get("floodRisk"),
        })
    return pd.DataFrame(rows)

def auto_risk_summary(df: pd.DataFrame, province: str):
    if df.empty:
        st.info("No data returned.")
        return
    # Simple readable summary
    scope = f"in {province}" if province and province != "ALL" else "across the region"
    top = df.sort_values("fire_score", ascending=False).head(3)
    avg = df["fire_score"].mean() if "fire_score" in df else None
    lines = []
    if avg is not None:
        lines.append(f"Average fire score {scope}: **{avg:.1f}**.")
    if not top.empty:
        tops = ", ".join(f"{row.City} ({row.fireRisk})" for _, row in top.iterrows())
        lines.append(f"Top hot spots today: **{tops}**.")
    st.markdown("\n\n".join(lines))

def risk_charts(df: pd.DataFrame, metric: str, chart_type: str):
    if df.empty:
        st.info("No data to chart.")
        return
    if metric not in df.columns:
        st.info(f"Metric '{metric}' not present; defaulting to fire_score.")
        metric = "fire_score"
    base = alt.Chart(df).encode(
        x=alt.X("City:N", sort="-y"),
        y=alt.Y(f"{metric}:Q", title=metric.replace("_"," ").title()),
        tooltip=["City","Province",metric]
    )
    if chart_type == "bar":
        st.altair_chart(base.mark_bar(), use_container_width=True)
    elif chart_type == "line":
        st.altair_chart(base.mark_line(point=True), use_container_width=True)
    else:
        st.altair_chart(base.mark_bar(), use_container_width=True)

def render_fire_table(fires):
    if not isinstance(fires, list) or not fires:
        st.info("No active fires returned.")
        return
    # Normalize a handy subset
    rows = []
    for f in fires:
        rows.append({
            "Name": f.get("name"),
            "Agency": f.get("agency"),
            "Status": f.get("status"),
            "Size (ha)": f.get("size_ha"),
            "Started": f.get("started"),
            "Lat": f.get("lat"),
            "Lon": f.get("lon"),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

def embed_html(html: str, height: int = 650):
    # Show Leaflet map returned by n8n (build HTML map branch)
    st.components.v1.html(html, height=height, scrolling=True)

# =========================
# UI
# =========================

tabs = st.tabs(["🔥 Active Fires", "🤖 AI Agent", "🎆 SAFER Fire Alert"])

# -------------------------
# Tab 1: keep your existing Active Fires UI/content here
# (unchanged from your current app)
with tabs[0]:
    st.header("Active Fires")
    st.write("Your existing Tab 1 content stays as before.")
    st.caption("Use the new AI Agent tab for maps/charts driven by your prompt.")

# -------------------------
# Tab 2: AI Agent (updated)
with tabs[1]:
    st.header("Ask the AI about risk in the Acadian region")

    colL, colR = st.columns([2,1])
    with colL:
        prompt = st.text_area("Your request", "Give a short narrative summary for all provinces today", height=100)
    with colR:
        province = st.selectbox("Province filter", ["ALL","NB","NS","PE","NL"], index=0)
        since_days = st.number_input("Lookback (days) for 'new' fires (optional)", 0, 60, 7, 1)

    st.write("Output")
    out_col = st.columns(5)
    # Radio-like set with our own labels
    output = out_col[0].radio("", ["Auto","Map","Bar chart","Line chart","Table"], index=0,
                              label_visibility="collapsed")

    if st.button("Ask AI", type="primary"):
        # Decide which dataset to hit
        forced = output.lower()
        forced = {"auto":"auto","map":"map","bar chart":"bar","line chart":"line","table":"table"}[forced]
        dataset = detect_dataset(prompt, forced)

        if dataset == "fires" or forced == "map":
            # ---- ACTIVE FIRES WEBHOOK ----
            if not FIRES_AI:
                st.error("N8N_AI_FIRES_URL is not set in secrets.")
            else:
                fmt = "map" if forced == "map" else "json"
                payload = {"format": fmt, "province": province, "since_days": int(since_days)}
                resp = _post_json(FIRES_AI, payload)

                if "_html" in resp:
                    embed_html(resp["_html"])
                else:
                    # Show a simple table for JSON response from fire AI
                    render_fire_table(resp.get("fires"))
        else:
            # ---- RISK WEBHOOK ----
            if not RISK_URL:
                st.error("N8N_AI_RISK_URL is not set in secrets.")
            else:
                payload = {
                    "province": province,
                    "detail": "detailed",
                    "question": prompt,
                    "focus": []  # you can pass ["fire","wind"] etc. if you want
                }
                resp = _post_json(RISK_URL, payload)
                df = make_risk_df(resp.get("results", []))

                if forced == "auto":
                    auto_risk_summary(df, province)
                    st.dataframe(df, use_container_width=True)
                elif forced == "table":
                    st.dataframe(df, use_container_width=True)
                else:
                    metric = detect_metric(prompt)
                    chart_type = "bar" if forced == "bar" else "line"
                    risk_charts(df, metric, chart_type)
                # Download
                if not df.empty:
                    st.download_button("Download results as CSV",
                        data=df.to_csv(index=False).encode("utf-8"),
                        file_name="risk_results.csv",
                        mime="text/csv",
                    )

# -------------------------
# Tab 3: SAFER Fire Alert (leave your existing UI/logic here)
with tabs[2]:
    st.header("SAFER Fire Alert")
    st.write("Your existing Tab 3 subscriber workflow stays as before.")
