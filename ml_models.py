import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC=os.path.join(ROOT,"data","processed")
OUT=os.path.join(ROOT,"outputs")

def metrics(name,X,labels):
    labels=np.asarray(labels)
    valid=labels!=-1
    groups=set(labels[valid])
    if len(groups)<2 or valid.sum()<=len(groups):
        return {"model":name,"clusters":len(groups),"silhouette":np.nan,
                "davies_bouldin":np.nan,"calinski_harabasz":np.nan,
                "note":"Not enough clusters for internal metrics"}
    Xv=np.asarray(X)[valid]; y=labels[valid]
    return {"model":name,"clusters":len(groups),
            "silhouette":silhouette_score(Xv,y),
            "davies_bouldin":davies_bouldin_score(Xv,y),
            "calinski_harabasz":calinski_harabasz_score(Xv,y),
            "note":"Noise excluded" if not valid.all() else "All samples"}

def run_models():
    os.makedirs(OUT,exist_ok=True)
    stops=pd.read_csv(os.path.join(PROC,"nyc_stops.csv"))
    rid=pd.read_csv(os.path.join(PROC,"nyc_ridership.csv"),low_memory=False)
    rid["estimated_boardings"]=pd.to_numeric(rid["estimated_boardings"],errors="coerce").fillna(0)
    rid["hour"]=pd.to_numeric(rid["hour"],errors="coerce").fillna(0)
    agg=rid.groupby("stop_id").agg(
        avg_boardings=("estimated_boardings","mean"),
        peak_boardings=("estimated_boardings","max"),
        std_boardings=("estimated_boardings","std"),
        active_route_count=("route_id","nunique"),
        hour_variability=("hour","nunique")
    ).reset_index().fillna(0)
    stop=stops.merge(agg,on="stop_id",how="inner")
    features=["avg_boardings","peak_boardings","std_boardings","active_route_count","hour_variability"]
    X=StandardScaler().fit_transform(stop[features])
    km=KMeans(n_clusters=4,random_state=42,n_init=15)
    stop["kmeans_cluster"]=km.fit_predict(X)
    # Label clusters by mean boardings.
    order=stop.groupby("kmeans_cluster")["avg_boardings"].mean().sort_values().index.tolist()
    labels=["Low","Moderate","High","Very high"]
    stop["activity_group"]=stop["kmeans_cluster"].map({c:labels[i] for i,c in enumerate(order)})
    gmm=GaussianMixture(n_components=4,random_state=42,n_init=3)
    stop["gmm_cluster"]=gmm.fit_predict(X)
    stop["gmm_confidence"]=gmm.predict_proba(X).max(axis=1)
    coords=np.radians(stop[["latitude","longitude"]].to_numpy())
    # 350-metre neighbourhood; adjust based on spatial silhouette and NYC stop density.
    db=DBSCAN(eps=0.35/6371.0,min_samples=4,metric="haversine")
    stop["dbscan_cluster"]=db.fit_predict(coords)
    route_features=rid.groupby("route_id").agg(
        avg_boardings=("estimated_boardings","mean"),
        peak_boardings=("estimated_boardings","max"),
        stop_count=("stop_id","nunique"),
        hour_coverage=("hour","nunique")
    ).reset_index()
    XR=StandardScaler().fit_transform(route_features[["avg_boardings","peak_boardings","stop_count","hour_coverage"]])
    if len(route_features)>=4:
        route_features["hierarchical_cluster"]=AgglomerativeClustering(n_clusters=4,linkage="ward").fit_predict(XR)
    else:
        route_features["hierarchical_cluster"]=0
    # Anomaly detection on stop-route-hour ridership rows.
    af=rid[["estimated_boardings","hour"]].copy()
    if "month" in rid.columns:
        af["month_num"]=pd.to_datetime(rid["month"],errors="coerce").dt.month.fillna(1)
    else: af["month_num"]=1
    Xa=StandardScaler().fit_transform(af)
    iso=IsolationForest(n_estimators=150,contamination=0.01,random_state=42,n_jobs=-1)
    rid["anomaly_flag"]=(iso.fit_predict(Xa)==-1).astype(int)
    rid["anomaly_score"]=iso.decision_function(Xa)
    anomaly=rid[rid.anomaly_flag==1].copy()
    scores=[
        metrics("K-Means",X,stop["kmeans_cluster"]),
        metrics("GMM",X,stop["gmm_cluster"]),
        metrics("DBSCAN",X,stop["dbscan_cluster"]),
        metrics("Hierarchical",XR,route_features["hierarchical_cluster"])
    ]
    metric_df=pd.DataFrame(scores)
    stop.to_csv(os.path.join(OUT,"nyc_stop_clusters.csv"),index=False)
    route_features.to_csv(os.path.join(OUT,"nyc_route_clusters.csv"),index=False)
    rid.to_csv(os.path.join(OUT,"nyc_ridership_anomaly_scores.csv"),index=False)
    anomaly.to_csv(os.path.join(OUT,"nyc_detected_anomalies.csv"),index=False)
    metric_df.to_csv(os.path.join(OUT,"nyc_model_metrics.csv"),index=False)
    return stop,route_features,anomaly,metric_df
