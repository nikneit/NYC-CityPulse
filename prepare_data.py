import os, re
import pandas as pd
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")
PROC = os.path.join(ROOT, "data", "processed")

def choose(df, candidates, required=True):
    cols={str(c).lower().strip():c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols:
            return cols[cand.lower()]
    # tolerate labels such as stop_latitude / stop_longitude
    for cand in candidates:
        for low, original in cols.items():
            if cand.lower() in low:
                return original
    if required:
        raise ValueError(f"Could not identify one of {candidates}. Available columns: {list(df.columns)}")
    return None

def prepare():
    os.makedirs(PROC,exist_ok=True)
    stop_path=os.path.join(RAW,"mta_bus_stops.csv")
    rid_path=os.path.join(RAW,"mta_bus_stop_ridership.csv")
    weather_path=os.path.join(RAW,"nyc_hourly_weather.csv")
    if not all(os.path.exists(p) for p in [stop_path,rid_path,weather_path]):
        raise FileNotFoundError("Required raw files missing. Run: py scripts/download_data.py")
    s=pd.read_csv(stop_path,low_memory=False)
    stop_id=choose(s,["stop_id"])
    lat=choose(s,["latitude","stop_lat","lat"])
    lon=choose(s,["longitude","stop_lon","lon"])
    stop_name=choose(s,["stop_name","stop_name_text","name"],required=False)
    route=choose(s,["route_id"],required=False)
    s=s.rename(columns={stop_id:"stop_id",lat:"latitude",lon:"longitude"})
    if stop_name: s=s.rename(columns={stop_name:"stop_name"})
    else: s["stop_name"]=s["stop_id"].astype(str)
    if route: s=s.rename(columns={route:"route_id"})
    s["latitude"]=pd.to_numeric(s["latitude"],errors="coerce")
    s["longitude"]=pd.to_numeric(s["longitude"],errors="coerce")
    s=s.dropna(subset=["stop_id","latitude","longitude"])
    s=s[(s.latitude.between(40.45,40.95))&(s.longitude.between(-74.30,-73.65))]
    # One coordinate/name row per physical stop; route/direction instances retained separately in source.
    stops=s.sort_values("stop_id").drop_duplicates("stop_id").copy()
    stops[["stop_id","stop_name","latitude","longitude"]].to_csv(os.path.join(PROC,"nyc_stops.csv"),index=False)

    r=pd.read_csv(rid_path,low_memory=False)
    rid=choose(r,["estimated_ridership","ridership","boardings"])
    rid_stop=choose(r,["stop_id"])
    rid_route=choose(r,["route_id"])
    hour=choose(r,["hour"])
    month=choose(r,["month"],required=False)
    weekday=choose(r,["weekday","day_of_week"],required=False)
    daytype=choose(r,["day_type"],required=False)
    r=r.rename(columns={rid:"estimated_boardings",rid_stop:"stop_id",rid_route:"route_id",hour:"hour"})
    r["estimated_boardings"]=pd.to_numeric(r["estimated_boardings"],errors="coerce").fillna(0)
    r["hour"]=pd.to_numeric(r["hour"],errors="coerce")
    r=r.dropna(subset=["stop_id","route_id","hour"])
    if month: r=r.rename(columns={month:"month"})
    if weekday: r=r.rename(columns={weekday:"weekday"})
    if daytype: r=r.rename(columns={daytype:"day_type"})
    r.to_csv(os.path.join(PROC,"nyc_ridership.csv"),index=False)

    w=pd.read_csv(weather_path)
    w["time"]=pd.to_datetime(w["time"],errors="coerce")
    w=w.rename(columns={"temperature_2m":"temperature_c","relative_humidity_2m":"humidity_pct",
                        "precipitation":"precipitation_mm","wind_speed_10m":"wind_kmh"})
    w.to_csv(os.path.join(PROC,"nyc_weather.csv"),index=False)

    gps_path=os.path.join(RAW,"nyc_bus_gps_sample.csv")
    if os.path.exists(gps_path):
        g=pd.read_csv(gps_path,low_memory=False)
        g=g.rename(columns={"lat":"latitude","lon":"longitude","vehicle_ts":"timestamp"})
        g["timestamp"]=pd.to_datetime(g["timestamp"],errors="coerce",utc=True)
        g["latitude"]=pd.to_numeric(g["latitude"],errors="coerce")
        g["longitude"]=pd.to_numeric(g["longitude"],errors="coerce")
        g=g.dropna(subset=["latitude","longitude"])
        g.to_csv(os.path.join(PROC,"nyc_gps_sample.csv"),index=False)

    if "route_id" in s.columns:
        route_table=s[["route_id"]].dropna().drop_duplicates().copy()
        route_table.to_csv(os.path.join(PROC,"nyc_routes.csv"),index=False)
    print(f"Prepared {len(stops):,} physical stops and {len(r):,} ridership rows.")
    print("Processed files are in data/processed/")

if __name__=="__main__":
    prepare()
