import os, sys, argparse, time, json
from datetime import datetime
import requests
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(BASE, "data", "raw")
os.makedirs(RAW, exist_ok=True)
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "NYC-CityPulse-Academic-Research/1.0 contact: student-project"})

def fetch_socrata(dataset_id, filename, where=None, limit=50000, max_rows=None):
    """Download a Socrata dataset in pages, enforcing max_rows before each request."""
    url = f"https://data.ny.gov/resource/{dataset_id}.json"
    offset, total = 0, 0
    chunks = []
    while True:
        page_limit = limit
        if max_rows is not None:
            remaining = max_rows - total
            if remaining <= 0:
                break
            page_limit = min(limit, remaining)
        params = {"$limit": page_limit, "$offset": offset, "$order": ":id"}
        if where:
            params["$where"] = where
        response = SESSION.get(url, params=params, timeout=120)
        response.raise_for_status()
        batch = response.json()
        if not batch:
            break
        chunks.extend(batch)
        total += len(batch)
        print(f"  {filename}: {total:,} rows downloaded", flush=True)
        if len(batch) < page_limit or (max_rows is not None and total >= max_rows):
            break
        offset += len(batch)
        time.sleep(0.15)
    if not chunks:
        raise RuntimeError(f"No rows returned for {dataset_id}. Check query/field names.")
    df = pd.DataFrame(chunks)
    path = os.path.join(RAW, filename)
    df.to_csv(path, index=False)
    print(f"Saved {len(df):,} rows -> {path}")
    return df

def download_stops():
    # The published MTA stop table contains historical route/direction/bundle instances.
    # Restrict to records marked in_effect to avoid treating every historic instance as current.
    try:
        return fetch_socrata("2ucp-7wg5", "mta_bus_stops.csv",
                             where="in_effect = 'true'", limit=50000)
    except Exception as e:
        print("Current-stop filter failed; retrying a small unfiltered sample to inspect schema:", e)
        return fetch_socrata("2ucp-7wg5", "mta_bus_stops.csv", limit=50000, max_rows=50000)

def download_ridership(month_start, month_end):
    # Dataset is monthly x weekday/day type x hour x route x stop estimates,
    # not individual passenger transactions or exact dated events.
    where = f"month >= '{month_start}' AND month <= '{month_end}'"
    return fetch_socrata("fvdm-uavx", "mta_bus_stop_ridership.csv",
                         where=where, limit=50000, max_rows=500000)

def download_weather(start_date, end_date):
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": 40.7128, "longitude": -74.0060,
        "start_date": start_date, "end_date": end_date,
        "hourly": "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m,weather_code",
        "timezone": "America/New_York", "temperature_unit": "celsius",
        "wind_speed_unit": "kmh", "precipitation_unit": "mm"
    }
    r = SESSION.get(url, params=params, timeout=120)
    r.raise_for_status()
    payload = r.json()
    hourly = payload["hourly"]
    df = pd.DataFrame(hourly)
    df.insert(0, "city", "New York City")
    df.to_csv(os.path.join(RAW, "nyc_hourly_weather.csv"), index=False)
    print(f"Saved {len(df):,} hourly weather rows")
    return df

def download_route_speeds():
    url = "https://data.transitlab.nyc/v1/bus_route_speeds.parquet"
    path = os.path.join(RAW, "nyc_bus_route_speeds.parquet")
    r = SESSION.get(url, timeout=180)
    r.raise_for_status()
    with open(path, "wb") as f:
        f.write(r.content)
    print("Saved TransitLab bus route speeds:", path)
    try:
        import pandas as pd
        df = pd.read_parquet(path)
        df.to_csv(os.path.join(RAW, "nyc_bus_route_speeds.csv"), index=False)
        print(f"Route speed rows: {len(df):,}")
    except Exception as e:
        print("Parquet saved. Install pyarrow to read it:", e)

def download_gps_day(gps_date, max_rows=75000):
    """Fetch one actual daily Parquet archive and retain a reproducible sample."""
    try:
        import duckdb
    except ImportError:
        raise RuntimeError("Install duckdb first: py -m pip install duckdb")
    year, month = gps_date[:4], gps_date[5:7]
    url = f"https://data.transitlab.nyc/v1/bus_vehicle_positions/year={year}/month={month}/{gps_date}.parquet"
    out = os.path.join(RAW, "nyc_bus_gps_sample.parquet")
    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs;")
    # A deterministic sample keeps the student project laptop-friendly.
    query = f"""
        COPY (
          SELECT row_id, observed_at, event, vehicle_id, vehicle_ts, trip_id,
                 route_id, direction_id, start_date, lat, lon, bearing, stop_id, source
          FROM read_parquet('{url}')
          WHERE event <> 'removed'
          ORDER BY hash(vehicle_id, vehicle_ts)
          LIMIT {int(max_rows)}
        ) TO '{out}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """
    con.execute(query)
    df = con.execute(f"SELECT * FROM read_parquet('{out}')").df()
    df.to_csv(os.path.join(RAW, "nyc_bus_gps_sample.csv"), index=False)
    print(f"Saved {len(df):,} real GPS position records sampled from {gps_date}")
    con.close()

def main():
    parser = argparse.ArgumentParser(description="Download real NYC MTA, TransitLab and Open-Meteo data.")
    parser.add_argument("--start-date", default="2026-10-01", help="Weather start date YYYY-MM-DD")
    parser.add_argument("--end-date", default="2026-10-31", help="Weather end date YYYY-MM-DD")
    parser.add_argument("--ridership-month-start", default="2026-10-01")
    parser.add_argument("--ridership-month-end", default="2026-10-01")
    parser.add_argument("--gps-date", default="2026-10-06", help="A documented available GPS day, YYYY-MM-DD")
    parser.add_argument("--gps-sample-rows", type=int, default=75000)
    parser.add_argument("--skip-gps", action="store_true")
    args = parser.parse_args()
    if not 50000 <= args.gps_sample_rows <= 100000:
        parser.error("--gps-sample-rows must be between 50000 and 100000")

    print("NYC CityPulse real-data acquisition")
    print("No synthetic data is generated by this script.")
    download_stops()
    download_ridership(args.ridership_month_start, args.ridership_month_end)
    download_weather(args.start_date, args.end_date)
    download_route_speeds()
    if not args.skip_gps:
        download_gps_day(args.gps_date, args.gps_sample_rows)
    print("\nDownload complete. Review data/raw/DATA_PROVENANCE.md before analysis.")

if __name__ == "__main__":
    main()
