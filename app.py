import os, sys, subprocess
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import pydeck as pdk

st.set_page_config(page_title="NYC CityPulse",page_icon="🗽",layout="wide")
ROOT=os.path.dirname(os.path.abspath(__file__))
PROC=os.path.join(ROOT,"data","processed")
RAW=os.path.join(ROOT,"data","raw")
OUT=os.path.join(ROOT,"outputs")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html,body,[class*="css"]{font-family:Inter,sans-serif}
.stApp{background:radial-gradient(circle at 50% 0%,#0b1d35 0%,#050b17 55%,#030711 100%);color:#e8f4ff}
.block-container{padding-top:1rem;max-width:100%}
section[data-testid="stSidebar"]{background:#071222;border-right:1px solid #16314e}
h1,h2,h3{color:#eaf6ff!important}
div[data-testid="stMetric"]{background:#0c1c2e;border:1px solid #1b4262;padding:14px;border-radius:12px}
.stButton>button{background:linear-gradient(90deg,#174d78,#38256c);color:#e9f7ff;border:1px solid #267bb0;border-radius:9px}
</style>
""",unsafe_allow_html=True)

def read(name):
    p=os.path.join(PROC,name)
    return pd.read_csv(p,low_memory=False) if os.path.exists(p) else pd.DataFrame()

st.markdown("""
<div style="display:flex;justify-content:space-between;align-items:center;border-bottom:1px solid #183650;padding-bottom:12px">
<div><div style="font-size:27px;font-weight:800">🗽 NYC <span style="color:#27d7ff">CityPulse</span></div>
<div style="color:#83a9c7;font-size:11px;letter-spacing:1px">REAL-WORLD URBAN MOBILITY INTELLIGENCE · NEW YORK CITY</div></div>
<div style="color:#45d6b5;border:1px solid #1d665f;border-radius:20px;padding:8px 12px;font-size:11px">● REAL DATA WORKFLOW</div>
</div>
""",unsafe_allow_html=True)

st.info("Data are downloaded from MTA, TransitLab and Open-Meteo sources. This dashboard does not create synthetic replacements. Missing files are shown as missing.")
with st.sidebar:
    st.markdown("## DATA SOURCE CONFIGURATION")
    st.caption("Analysis city")
    st.selectbox("City", ["New York City · New York · USA"], index=0,
                 disabled=True, label_visibility="collapsed")
    st.divider()

    st.markdown("### ANALYSIS LAYER")
    layer=st.radio("Choose a map layer",
        ["Passenger boardings","Bus GPS positions","Bus route speeds","Stop network"],
        label_visibility="collapsed")
    st.divider()

    st.markdown("### TIME SETTINGS")
    analysis_date=st.date_input("Analysis date", value=pd.Timestamp("2026-10-06").date())
    selected_hour=st.slider("Hour of day", min_value=0, max_value=23, value=8, step=1)
    st.caption(f"Selected hour: {selected_hour:02d}:00–{(selected_hour+1)%24:02d}:00")
    st.caption("MTA ridership is published as monthly weekday/day-type/hour estimates, not individual daily passenger trips.")
    st.divider()

    st.markdown("### MAP DISPLAY")
    radius=st.slider("Stop marker radius",30,250,90)
    show_labels=st.checkbox("Show stop labels",False)
    show_hotspots=st.checkbox("Show busiest-stop ranking",True)
    st.divider()

    st.markdown("### DATA SOURCE STATUS")
    st.write("🚌 MTA · Bus stops & ridership")
    st.write("📍 TransitLab NYC · Vehicle positions & speeds")
    st.write("🌦️ Open-Meteo · Historical weather")
    st.caption("Only downloaded source records are displayed. No synthetic fallback data is generated.")

st.markdown("### Dataset readiness")
files={
"MTA bus stops":"mta_bus_stops.csv",
"MTA bus stop ridership":"mta_bus_stop_ridership.csv",
"Hourly NYC weather":"nyc_hourly_weather.csv",
"TransitLab route speeds":"nyc_bus_route_speeds.csv",
"Sampled bus GPS":"nyc_bus_gps_sample.csv"
}
cols=st.columns(5)
for c,(label,fn) in zip(cols,files.items()):
    exists=os.path.exists(os.path.join(RAW,fn))
    c.metric(label,"Ready" if exists else "Not downloaded")
    c.caption(fn)

stops=read("nyc_stops.csv")
rid=read("nyc_ridership.csv")
weather=read("nyc_weather.csv")
gps=read("nyc_gps_sample.csv")
routes=read("nyc_routes.csv")

if stops.empty or rid.empty:
    st.warning("The real NYC data have not been downloaded/prepared yet. Use the setup instructions in README.md, or run these commands in the project terminal:")
    st.code("py scripts/download_data.py\npy -m src.prepare_data")
    st.markdown("The download script fetches the current MTA stop records, a selected month of MTA stop-level boarding estimates, actual historical weather and TransitLab route speeds. It can also sample one real archived GPS day.")
    st.stop()

# Ensure numeric columns and merge route/stop locations.
for c in ["latitude","longitude"]:
    stops[c]=pd.to_numeric(stops[c],errors="coerce")
rid["estimated_boardings"]=pd.to_numeric(rid["estimated_boardings"],errors="coerce").fillna(0)
rid["hour"]=pd.to_numeric(rid["hour"],errors="coerce")
hour_rid = rid[rid["hour"] == selected_hour].copy()
if hour_rid.empty:
    hour_rid = rid.copy()
    hour_note = "No rows for the selected hour; showing the complete downloaded ridership extract."
else:
    hour_note = f"Showing ridership estimates for {selected_hour:02d}:00–{(selected_hour+1)%24:02d}:00."

stop_activity=hour_rid.groupby("stop_id",as_index=False).agg(
    estimated_boardings=("estimated_boardings","sum"),
    avg_boardings=("estimated_boardings","mean"),
    routes_serving=("route_id","nunique")
)
mapdata=stops.merge(stop_activity,on="stop_id",how="left").fillna({"estimated_boardings":0,"avg_boardings":0,"routes_serving":0})
route_count=hour_rid["route_id"].nunique()
gps_count=len(gps)
total_boardings=int(hour_rid["estimated_boardings"].sum())
m1,m2,m3,m4=st.columns(4)
m1.metric("Mapped bus stops",f"{len(stops):,}")
m2.metric("Ridership estimate rows",f"{len(hour_rid):,}")
m3.metric("Estimated boardings in selected extract",f"{total_boardings:,}")
m4.metric("Routes represented",f"{route_count:,}")

tabs=st.tabs(["City Overview","Machine Learning Lab","Dataset Explorer","Data Sources & Methods"])
with tabs[0]:
    left,center,right=st.columns([1,3,1.25])
    with left:
        st.markdown("#### Mobility snapshot")
        st.write(f"**City:** New York City")
        st.write(f"**Mapped stops:** {len(stops):,}")
        st.write(f"**Routes in ridership extract:** {route_count:,}")
        st.write(f"**GPS sample records:** {gps_count:,}")
        if not weather.empty:
            st.write(f"**Weather hours:** {len(weather):,}")
        st.markdown("#### Map legend")
        st.caption("Marker size and colour indicate estimated boardings.")
    with center:
        st.markdown("#### NYC spatial mobility map")
        if layer=="Passenger boardings":
            map_layer=pdk.Layer("ScatterplotLayer",data=mapdata, get_position="[longitude,latitude]",
                get_radius=radius, radius_min_pixels=3, radius_max_pixels=16,
                get_fill_color="[35,190,235,180]",pickable=True,auto_highlight=True)
            deckdata=mapdata
        elif layer=="Stop network":
            map_layer=pdk.Layer("ScatterplotLayer",data=stops,get_position="[longitude,latitude]",
                get_radius=radius,radius_min_pixels=3,get_fill_color="[20,205,225,190]",pickable=True)
            deckdata=stops
        elif layer=="Bus GPS positions" and not gps.empty:
            deckdata=gps.copy()
            deckdata["longitude"]=deckdata["longitude"]
            deckdata["latitude"]=deckdata["latitude"]
            map_layer=pdk.Layer("ScatterplotLayer",data=deckdata,get_position="[longitude,latitude]",
                get_radius=35,radius_min_pixels=2,get_fill_color="[255,115,70,180]",pickable=True)
        else:
            deckdata=mapdata
            map_layer=pdk.Layer("ScatterplotLayer",data=mapdata,get_position="[longitude,latitude]",
                get_radius=radius,radius_min_pixels=3,get_fill_color="[120,160,255,180]",pickable=True)
        view=pdk.ViewState(latitude=float(stops.latitude.mean()),longitude=float(stops.longitude.mean()),zoom=10.3,pitch=40)
        tooltip={"html":"<b>{stop_name}</b><br/>Stop ID: {stop_id}<br/>Estimated boardings: {estimated_boardings}"}
        if layer=="Bus GPS positions":
            tooltip={"html":"<b>Bus {vehicle_id}</b><br/>Route: {route_id}<br/>Time: {timestamp}<br/>Stop: {stop_id}"}
        st.pydeck_chart(pdk.Deck(layers=[map_layer],initial_view_state=view,
            map_style="https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
            tooltip=tooltip),use_container_width=True,height=560)
    with right:
        if show_hotspots:
            st.markdown("#### Busiest stops")
            top=mapdata.sort_values("estimated_boardings",ascending=False).head(12)
            for i,row in enumerate(top.itertuples(),1):
                st.write(f"**{i:02d}.** {row.stop_name} · {int(row.estimated_boardings):,}")
        if not weather.empty:
            st.markdown("#### Weather overview")
            st.write(f"Weather observations: {len(weather):,}")
            if "precipitation_mm" in weather:
                st.write(f"Mean hourly precipitation: {weather.precipitation_mm.mean():.2f} mm")
    st.caption(hour_note)
    st.markdown("#### Hourly passenger boarding profile")
    hourly=rid.groupby("hour",as_index=False)["estimated_boardings"].sum()
    fig=px.line(hourly,x="hour",y="estimated_boardings",markers=True,title="MTA estimated boardings by hour")
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",font_color="#cceaff")
    st.plotly_chart(fig,use_container_width=True)

with tabs[1]:
    st.markdown("### Unsupervised Machine Learning")
    st.write("Run clustering and anomaly detection on the downloaded real MTA extracts.")
    if st.button("Run all five unsupervised models",type="primary"):
        with st.spinner("Preparing features and fitting models..."):
            from src.ml_models import run_models
            run_models()
        st.success("Model execution completed. Results saved in outputs/.")
    metric_path=os.path.join(OUT,"nyc_model_metrics.csv")
    stop_path=os.path.join(OUT,"nyc_stop_clusters.csv")
    route_path=os.path.join(OUT,"nyc_route_clusters.csv")
    anom_path=os.path.join(OUT,"nyc_detected_anomalies.csv")
    if os.path.exists(metric_path):
        met=pd.read_csv(metric_path)
        st.markdown("#### Clustering evaluation")
        st.dataframe(met,use_container_width=True,hide_index=True)
        if os.path.exists(stop_path):
            sr=pd.read_csv(stop_path)
            st.markdown("#### K-Means passenger activity groups")
            fig=px.scatter(sr,x="longitude",y="latitude",color="activity_group",hover_name="stop_name",
                title="NYC bus stops grouped by estimated boarding patterns")
            st.plotly_chart(fig,use_container_width=True)
            st.markdown("#### DBSCAN geographical density clusters")
            fig=px.scatter(sr,x="longitude",y="latitude",color=sr.dbscan_cluster.astype(str),
                hover_name="stop_name",title="DBSCAN clusters (-1 = noise)")
            st.plotly_chart(fig,use_container_width=True)
            st.markdown("#### GMM membership")
            st.dataframe(sr[["stop_id","stop_name","gmm_cluster","gmm_confidence"]].head(100),use_container_width=True,hide_index=True)
        if os.path.exists(route_path):
            rr=pd.read_csv(route_path)
            st.markdown("#### Hierarchical route groups")
            st.dataframe(rr,use_container_width=True,hide_index=True)
        if os.path.exists(anom_path):
            aa=pd.read_csv(anom_path)
            st.markdown(f"#### Isolation Forest anomalies · {len(aa):,} flagged rows")
            st.dataframe(aa.head(200),use_container_width=True,hide_index=True)
        st.caption("Silhouette, Davies–Bouldin and Calinski–Harabasz are internal clustering metrics, not classification accuracy. Isolation Forest is assessed separately.")

with tabs[2]:
    st.markdown("### Dataset Explorer")
    available={}
    for label,fn in files.items():
        p=os.path.join(RAW,fn)
        if os.path.exists(p): available[label]=p
    if available:
        selected=st.selectbox("Choose source dataset",list(available))
        df=pd.read_csv(available[selected],nrows=100000,low_memory=False)
        st.caption(f"Previewing up to 100,000 rows from {selected}")
        st.dataframe(df.head(500),use_container_width=True,hide_index=True)
        st.download_button("Download preview CSV",df.to_csv(index=False),file_name=selected.replace(" ","_")+"_preview.csv",mime="text/csv")
    else:
        st.info("No source files downloaded yet.")

with tabs[3]:
    st.markdown("### Sources, provenance and methodological limits")
    st.markdown("""
    - **MTA Bus Stops:** route/direction-specific bus stop records. The source dataset includes historical bundle instances; the preparation step deduplicates physical stop IDs for map display.
    - **MTA Bus Stop Ridership Estimates:** estimated boardings by month, weekday/day type, hour, route, direction and stop. These are APC-derived estimates, not individual timestamped passenger journeys.
    - **TransitLab Bus Positions:** historical MTA GTFS-realtime vehicle-position archive. The download script samples one selected day to keep the local project manageable.
    - **TransitLab Bus Route Speeds:** route/direction/hour/day speed calculations derived from reported bus positions.
    - **Open-Meteo Historical Weather:** reanalysis-based hourly weather estimates at central NYC coordinates; not direct readings from every bus stop.

    The ridership table's monthly weekday/hour aggregates do not align one-to-one with individual GPS timestamps. Weather comparisons must be aggregated to compatible month/day-type/hour periods; do not imply causal effects from correlation.
    """)
    st.markdown("**Attribution:** Metropolitan Transportation Authority (MTA), TransitLab NYC (CC BY 4.0 for its collected datasets), and Open-Meteo. See README for links and terms.")

st.markdown("<div style='text-align:center;color:#64839e;font-size:10px;padding:16px'>NYC CITYPULSE · REAL-WORLD DATA ANALYTICS · ACADEMIC PROJECT</div>",unsafe_allow_html=True)
