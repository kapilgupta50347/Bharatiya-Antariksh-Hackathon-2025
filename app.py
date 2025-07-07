import os
from flask import Flask, request, render_template, send_file
import pandas as pd
import numpy as np
import h5py
import xarray as xr
import joblib
import matplotlib.pyplot as plt


app = Flask(__name__)

model_path = "model_pm25.pkl"
model = joblib.load(model_path)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def upload_predict():
    try:
        # Get files
        aod_file = request.files["aod_file"]
        met_file = request.files["met_file"]
        pblh_file = request.files["pblh_file"]
        cpcb_file = request.files.get("cpcb_file")

        if not aod_file or not met_file or not pblh_file:
            return "Missing required files", 400

        # Save files
        aod_path = "temp_aod.h5"
        met_path = "temp_met.nc4"
        pblh_path = "temp_pblh.nc"
        aod_file.save(aod_path)
        met_file.save(met_path)
        pblh_file.save(pblh_path)

        # AOD
        with h5py.File(aod_path, "r") as f:
            aod_data = f["AOD"][:]
            lat = f["latitude"][:]
            lon = f["longitude"][:]
        lon_grid, lat_grid = np.meshgrid(lon, lat)
        df_aod = pd.DataFrame({
            "lat_aod": lat_grid.flatten(),
            "lon_aod": lon_grid.flatten(),
            "aod": aod_data.flatten()
        })
        df_aod["lat_round"] = df_aod["lat_aod"].round(1)
        df_aod["lon_round"] = df_aod["lon_aod"].round(1)

        # Met
        ds_met = xr.open_dataset(met_path)
        ps = ds_met["PS"].isel(time=0).squeeze().values.flatten()
        t2m = ds_met["T2M"].isel(time=0).squeeze().values.flatten()
        qv2m = ds_met["QV2M"].isel(time=0).squeeze().values.flatten()
        u10m = ds_met["U10M"].isel(time=0).squeeze().values.flatten()
        v10m = ds_met["V10M"].isel(time=0).squeeze().values.flatten()
        lat_met = ds_met["lat"].values
        lon_met = ds_met["lon"].values
        lon_grid_met, lat_grid_met = np.meshgrid(lon_met, lat_met)
        df_met = pd.DataFrame({
            "lat_met": lat_grid_met.flatten(),
            "lon_met": lon_grid_met.flatten(),
            "PS": ps,
            "T2M": t2m,
            "QV2M": qv2m,
            "U10M": u10m,
            "V10M": v10m
        })
        df_met["lat_round"] = df_met["lat_met"].round(1)
        df_met["lon_round"] = df_met["lon_met"].round(1)

        # PBLH
        ds_pblh = xr.open_dataset(pblh_path)
        pblh = ds_pblh["PBLH"].isel(time=0).squeeze().values.flatten()
        lat_pblh = ds_pblh["lat"].values
        lon_pblh = ds_pblh["lon"].values
        lon_grid_pblh, lat_grid_pblh = np.meshgrid(lon_pblh, lat_pblh)
        df_pblh = pd.DataFrame({
            "lat_pblh": lat_grid_pblh.flatten(),
            "lon_pblh": lon_grid_pblh.flatten(),
            "PBLH": pblh
        })
        df_pblh["lat_round"] = df_pblh["lat_pblh"].round(1)
        df_pblh["lon_round"] = df_pblh["lon_pblh"].round(1)

        # Merge
        df_merged = pd.merge(df_aod, df_met, on=["lat_round", "lon_round"], how="inner")
        df_merged = pd.merge(df_merged, df_pblh, on=["lat_round", "lon_round"], how="inner")

        df_merged = df_merged.dropna(subset=["aod", "PBLH", "PS", "T2M", "QV2M", "U10M", "V10M"])

        # Confirm feature order
        feature_cols = ["aod", "PBLH", "PS", "T2M", "QV2M", "U10M", "V10M"]
        df_predict = df_merged[feature_cols]
        print("Columns for prediction:", df_predict.columns.tolist())

        # Predict
        df_merged["pred_pm25"] = model.predict(df_predict)

        # Save CSV
        pm_map_path = "static/PM_Map_Final.csv"
        df_merged[["lat_aod", "lon_aod", "pred_pm25"]].to_csv(pm_map_path, index=False)

        # Create plot
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 8))
        sc = plt.scatter(df_merged["lon_aod"], df_merged["lat_aod"], c=df_merged["pred_pm25"],
                        cmap="jet", marker='s', s=30, edgecolor='none')
        plt.colorbar(sc, label="PM2.5 concentration (μg/m³)")
        plt.title("PM2.5 Concentration Map")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.grid(True)

        plot_path = "static/PM_Map_1deg.png"
        plt.savefig(plot_path)
        plt.close()

        # Save CPCB if provided
        if cpcb_file and cpcb_file.filename != "":
            cpcb_df = pd.read_csv(cpcb_file)
            cpcb_df.to_csv("static/CPCB_Uploaded.csv", index=False)

        return render_template("index.html", plot_url=plot_path)

    except Exception as e:
        return f"An error occurred: {str(e)}", 500


if __name__ == '__main__':
    app.run(debug=True)
