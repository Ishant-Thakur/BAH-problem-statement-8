import pandas as pd
import rasterio
import numpy as np

# ==============================
# INPUT FILES
# ==============================

CRATER_CSV = "crater_candidates.csv"

VOL_TIF = "ch2_sar_ndxl_20250630my4rspwest_d_vol_xx_fp_xx_xxx.tif"
ODD_TIF = "ch2_sar_ndxl_20250630my4rspwest_d_odd_xx_fp_xx_xxx.tif"
EVN_TIF = "ch2_sar_ndxl_20250630my4rspwest_d_evn_xx_fp_xx_xxx.tif"
HELIX_TIF = "ch2_sar_ndxl_20250630my4rspwest_d_hlx_xx_fp_xx_xxx.tif"

OUTPUT_CSV = "crater_features.csv"

# ==============================
# Read crater candidates
# ==============================

craters = pd.read_csv(CRATER_CSV)

print(f"Loaded {len(craters)} crater candidates")

# ==============================
# Open rasters
# ==============================

vol = rasterio.open(VOL_TIF)
odd = rasterio.open(ODD_TIF)
evn = rasterio.open(EVN_TIF)
helix = rasterio.open(HELIX_TIF)

# ==============================
# Function to extract one pixel
# ==============================

def sample_raster(dataset, x, y):

    try:
        value = next(dataset.sample([(x, y)]))[0]

        if dataset.nodata is not None and value == dataset.nodata:
            return np.nan

        return float(value)

    except Exception:
        return np.nan

# ==============================
# Extract radar values
# ==============================

vol_values = []
odd_values = []
evn_values = []
helix_values = []

for i, (_, crater) in enumerate(craters.iterrows()):

    if i % 50 == 0:
        print(f"Processing crater {i}/{len(craters)}")

    x = crater["x_polar_m"]
    y = crater["y_polar_m"]

    vol_values.append(sample_raster(vol, x, y))
    odd_values.append(sample_raster(odd, x, y))
    evn_values.append(sample_raster(evn, x, y))
    helix_values.append(sample_raster(helix, x, y))

# ==============================
# Save features
# ==============================

craters["VOL"] = vol_values
craters["ODD"] = odd_values
craters["EVN"] = evn_values
craters["HELIX"] = helix_values

craters.to_csv(OUTPUT_CSV, index=False)

print("\nFinished!")

print(f"Saved {OUTPUT_CSV}")

print(craters.head())