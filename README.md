# NYC CityPulse — Real-World Urban Mobility Analytics

An interactive Streamlit project using real New York City MTA / TransitLab public transport data and Open-Meteo historical weather. The project does not silently generate synthetic replacement data.

## What you get
- Real NYC bus-stop coordinates and route/direction metadata from MTA.
- MTA stop-level estimated boardings.
- A laptop-manageable sample of actual historical MTA bus GPS positions from TransitLab's archive.
- TransitLab route speeds derived from bus positions.
- Hourly historical weather for central NYC.
- Interactive NYC map, passenger hotspot view, GPS view, analytics and dataset explorer.
- Five unsupervised algorithms: K-Means, DBSCAN, Agglomerative Hierarchical Clustering, Gaussian Mixture Model and Isolation Forest.

## Important data-size and provenance note
The public source catalogs are much larger than a typical student laptop should download in full:
- MTA Bus Stops catalog: approximately 3.21 million route/direction/bundle stop-instance rows.
- MTA stop ridership: approximately 102 million records since 2024.
- TransitLab bus-position archive: approximately 2.87 billion rows / 54.3 GB, with gaps in its historical coverage.

Therefore, the default downloader obtains current/in-effect bus stop records, one selected ridership month (default October 2026), one selected GPS day sampled to 200,000 records, the compact route-speed table, and weather for the matching period. This is a real-data research extract, not every record in those multi-million/billion-row catalogs. You can change the date/month and GPS sample size in the command line. The downloader does not fabricate data.

## Sources
1. MTA Bus Stops: https://data.ny.gov/d/2ucp-7wg5
2. MTA Bus Stop Ridership Estimates (Beginning 2024): https://data.ny.gov/d/fvdm-uavx
3. TransitLab Bus Positions: https://docs.transitlab.nyc/datasets/bus-vehicle-positions/
4. TransitLab Bus Route Speeds: https://docs.transitlab.nyc/datasets/bus-route-speeds/
5. Open-Meteo Historical Weather API: https://open-meteo.com/en/docs/historical-weather-api

The TransitLab dataset documentation states that bus positions cover July 2017–November 2024 with gaps and from October 2026 onward; it has no records for 5 Nov 2024–1 Oct 2026. The selected example GPS day is 2026-10-06. Bus positions are UTC; MTA ridership hours are New York local time. Be careful with time-zone conversion.

## Run on Windows / VS Code
1. Extract the ZIP.
2. Open `NYC_CityPulse` in VS Code.
3. Open Terminal > New Terminal.
4. Install packages:
   `py -m pip install -r requirements.txt`
5. Download the real source data:
   `py scripts/download_data.py`
6. Prepare consistent analysis tables:
   `py -m src.prepare_data`
7. Start the dashboard:
   `py -m streamlit run app.py`
8. Open `http://localhost:8501`.

## Change the download window
Example: use September 2026 ridership/weather and a selected GPS date:
`py scripts/download_data.py --start-date 2026-09-01 --end-date 2026-09-30 --ridership-month-start 2026-09-01 --ridership-month-end 2026-09-01 --gps-date 2026-09-15`

Skip GPS if you only want stops, ridership and weather:
`py scripts/download_data.py --skip-gps`

Increase or reduce the sampled GPS records:
`py scripts/download_data.py --gps-sample-rows 50000`

## Models and interpretation
- K-Means: clusters stops using estimated boardings, peak boardings and route coverage.
- DBSCAN: groups geographically close stops using haversine distance; -1 means noise.
- Agglomerative Hierarchical Clustering: groups routes using boarding and stop/hour profiles.
- GMM: probabilistic stop-group membership and confidence.
- Isolation Forest: flags unusual stop-route-hour boarding observations.

Clustering evaluation uses Silhouette, Davies–Bouldin and Calinski–Harabasz scores where defined. These are not classification accuracy. Isolation Forest is anomaly detection and should be inspected separately.

## Methodological cautions
- MTA stop ridership is estimated boardings, not alightings, pedestrian footfall, or individual tap-level records.
- Stop ridership is an aggregate by month, weekday/day type and hour. It is not a timestamped daily passenger series.
- Weather is hourly reanalysis at one central coordinate, not a station-specific observation at every stop.
- GPS archive records are sampled; sample frequencies differ between archive and current collector periods.
- A correlation between weather and boardings does not establish causation.
- MTA datasets may have their own terms; TransitLab collected tables are CC BY 4.0. Check source terms and provide attribution in academic work.

## Output folders
- `data/raw/`: downloaded source files and provenance record
- `data/processed/`: normalized stop, ridership, weather and GPS tables
- `outputs/`: clustering, anomaly and evaluation CSV files
