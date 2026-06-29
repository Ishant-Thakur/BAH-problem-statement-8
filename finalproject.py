# ══════════════════════════════════════════════════════════════════
# MODULE 1 — PSR & Doubly Shadowed Crater Mapping
# LAPTOP SAFE VERSION — optimised for 8GB RAM
# ══════════════════════════════════════════════════════════════════

import rasterio
from rasterio.windows import Window
import numpy as np
from scipy.ndimage import minimum_filter, label
import matplotlib
matplotlib.use('Agg')  # no popup window — saves RAM
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
from rasterio.transform import xy
import gc
import os

# ─────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────
DEM_PATH        = 'data/LDEM_80S_80MPP_ADJ.TIF'
OUTPUT_MAP      = 'module1_output.png'
OUTPUT_CSV      = 'crater_candidates.csv'
PIXEL_SIZE_M    = 80
MIN_CRATER_PX   = 1    # accept even single pixel craters
PSR_PERCENTILE  = 20   # wider shadow net
FILTER_SIZE     = 10   # smaller window finds more depressions
CROP_SIZE       = 1500 # only load 1500x1500 pixels

# ─────────────────────────────────────────
# STEP 1: Load CROPPED DEM
# ─────────────────────────────────────────
print("=" * 55)
print(" MODULE 1 — Lunar South Pole PSR Mapping")
print(" (Laptop-safe version — 8GB RAM)")
print("=" * 55)
print()
print("[1/5] Loading DEM (cropped to save RAM)...")

if not os.path.exists(DEM_PATH):
    print(f"ERROR: File not found → {DEM_PATH}")
    print("Make sure LDEM_80S_80MPP_ADJ.TIF is inside your data/ folder")
    exit()

with rasterio.open(DEM_PATH) as src:
    full_h, full_w = src.height, src.width
    print(f"    Full DEM size  : {full_h} x {full_w} pixels")

    # Crop to center — south pole is at center of polar projection
    row_start = (full_h - CROP_SIZE) // 2
    col_start = (full_w - CROP_SIZE) // 2
    window    = Window(col_start, row_start, CROP_SIZE, CROP_SIZE)

    dem       = src.read(1, window=window).astype(np.float32)
    transform = src.window_transform(window)

print(f"    ✓ Cropped to   : {dem.shape[0]} x {dem.shape[1]} pixels")
print(f"    ✓ Area covered : {dem.shape[0]*PIXEL_SIZE_M/1000:.0f} km x {dem.shape[1]*PIXEL_SIZE_M/1000:.0f} km")
print(f"    ✓ Elev range   : {dem.min():.1f} m  to  {dem.max():.1f} m")
print(f"    ✓ RAM used     : ~{dem.nbytes / 1e6:.0f} MB")

# DEBUG — understand your data
print()
print("    DEBUG INFO:")
print("    Elevation percentiles:")
for p in [5, 10, 20, 30, 50]:
    print(f"      {p}th percentile = {np.percentile(dem, p):.1f} m")
print(f"    Negative elevations    : {(dem < 0).sum():,} pixels")
print(f"    Below -3000m           : {(dem < -3000).sum():,} pixels")
print(f"    Below -5000m           : {(dem < -5000).sum():,} pixels")

# ─────────────────────────────────────────
# STEP 2: PSR mask
# ─────────────────────────────────────────
print()
print("[2/5] Computing PSR mask...")

elev_threshold = np.percentile(dem, PSR_PERCENTILE)
psr_mask       = dem < elev_threshold
psr_area_km2   = psr_mask.sum() * (PIXEL_SIZE_M ** 2) / 1e6

print(f"    ✓ PSR threshold : below {elev_threshold:.1f} m")
print(f"    ✓ PSR pixels    : {psr_mask.sum():,}")
print(f"    ✓ PSR area      : {psr_area_km2:.1f} km²")
print(f"    ✓ PSR coverage  : {psr_mask.mean()*100:.1f}% of cropped area")

# ─────────────────────────────────────────
# STEP 3: Crater floor detection
# ─────────────────────────────────────────
print()
print("[3/5] Detecting crater floors...")

local_min     = minimum_filter(dem, size=FILTER_SIZE)
crater_floors = (dem == local_min)

del local_min
gc.collect()

print(f"    ✓ Raw crater pixels  : {crater_floors.sum():,}")
print(f"    ✓ Inside PSR too     : {(crater_floors & psr_mask).sum():,}")

# ─────────────────────────────────────────
# STEP 4: Doubly shadowed craters + filter
# ─────────────────────────────────────────
print()
print("[4/5] Finding doubly shadowed craters...")

doubly_shadowed_raw  = crater_floors & psr_mask
labeled, total_blobs = label(doubly_shadowed_raw)

del doubly_shadowed_raw
gc.collect()

print(f"    Total blobs found   : {total_blobs:,}")

real_crater_mask = np.zeros(dem.shape, dtype=bool)
crater_data      = []

for i in range(1, total_blobs + 1):
    pixels = np.where(labeled == i)
    size   = len(pixels[0])

    if size < MIN_CRATER_PX:
        continue

    real_crater_mask[pixels] = True

    center_row = int(np.mean(pixels[0]))
    center_col = int(np.mean(pixels[1]))
    x_m, y_m  = xy(transform, center_row, center_col)

    crater_data.append({
        'crater_id'         : i,
        'center_row'        : center_row,
        'center_col'        : center_col,
        'x_polar_m'         : round(x_m, 1),
        'y_polar_m'         : round(y_m, 1),
        'dist_from_pole_km' : round(np.sqrt(x_m**2 + y_m**2) / 1000, 2),
        'elevation_m'       : round(float(dem[center_row, center_col]), 1),
        'size_pixels'       : size,
        'area_km2'          : round(size * (PIXEL_SIZE_M**2) / 1e6, 4),
    })

del labeled
gc.collect()

# Sort deepest first
crater_data.sort(key=lambda x: x['elevation_m'])

print(f"    ✓ Real craters found : {len(crater_data)}")

# ─────────────────────────────────────────
# STEP 4b: Save CSV
# ─────────────────────────────────────────
df = pd.DataFrame(crater_data)

if len(df) == 0:
    print()
    print("    ⚠  WARNING: No craters found in this crop region.")
    print("    This can happen if the cropped area misses the deep craters.")
    print("    Try increasing CROP_SIZE to 2500 at the top of this file.")
    print("    Saving empty CSV and continuing to map...")
else:
    print(f"    ✓ Deepest crater    : {crater_data[0]['elevation_m']} m")
    print(f"    ✓ Dist from pole    : {crater_data[0]['dist_from_pole_km']} km")
    print(f"    ✓ Area              : {crater_data[0]['area_km2']} km²")
    print()
    print("    Top 5 deepest crater candidates:")
    cols_to_show = [c for c in
                    ['crater_id','elevation_m','area_km2','dist_from_pole_km']
                    if c in df.columns]
    print(df[cols_to_show].head(5).to_string(index=False))

df.to_csv(OUTPUT_CSV, index=False)
print(f"    ✓ Saved → {OUTPUT_CSV}")

# ─────────────────────────────────────────
# STEP 5: Visualize — no popup
# ─────────────────────────────────────────
print()
print("[5/5] Generating map (saving as PNG, no popup)...")

fig, axes = plt.subplots(1, 2, figsize=(16, 8))
fig.patch.set_facecolor('#0a0a0a')

# ── Left: terrain map ──
ax1 = axes[0]
ax1.set_facecolor('#0a0a0a')
ax1.imshow(dem, cmap='gray', origin='upper', alpha=0.9)

psr_display = np.where(psr_mask, 1.0, np.nan)
ax1.imshow(psr_display, cmap='Blues', alpha=0.35, origin='upper')

if real_crater_mask.any():
    rc_display = np.where(real_crater_mask, 1.0, np.nan)
    ax1.imshow(rc_display, cmap='Reds', alpha=0.9, origin='upper')

ax1.set_title('Lunar South Pole\nPSR + Doubly Shadowed Craters',
              color='white', fontsize=12, pad=8)
ax1.set_xlabel('Pixel Column', color='white', fontsize=9)
ax1.set_ylabel('Pixel Row',    color='white', fontsize=9)
ax1.tick_params(colors='white')
for sp in ax1.spines.values():
    sp.set_edgecolor('#444')

legend_handles = [
    mpatches.Patch(color='#888888', label='Terrain (LOLA DEM)'),
    mpatches.Patch(color='#4477CC', alpha=0.6,
                   label=f'PSR — {psr_area_km2:.0f} km²'),
    mpatches.Patch(color='#CC3333',
                   label=f'Doubly shadowed ({len(crater_data)})'),
]
ax1.legend(handles=legend_handles, loc='lower right',
           fontsize=8, facecolor='#1a1a1a',
           labelcolor='white', edgecolor='#444')

# ── Right: elevation histogram ──
ax2 = axes[1]
ax2.set_facecolor('#111111')

ax2.hist(dem.flatten(), bins=80, color='#555555',
         alpha=0.7, label='All terrain', density=True)
ax2.hist(dem[psr_mask].flatten(), bins=80, color='#4477CC',
         alpha=0.7, label='PSR pixels', density=True)

if crater_data:
    crat_elev = np.array([c['elevation_m'] for c in crater_data])
    ax2.hist(crat_elev, bins=30, color='#CC3333', alpha=0.9,
             label=f'Craters ({len(crater_data)})', density=True)

ax2.axvline(elev_threshold, color='cyan', linestyle='--',
            linewidth=1.5, label=f'PSR cutoff ({elev_threshold:.0f} m)')

ax2.set_title('Elevation Distribution', color='white', fontsize=12)
ax2.set_xlabel('Elevation (m)', color='white', fontsize=9)
ax2.set_ylabel('Density',       color='white', fontsize=9)
ax2.tick_params(colors='white')
ax2.legend(fontsize=8, facecolor='#1a1a1a',
           labelcolor='white', edgecolor='#444')
for sp in ax2.spines.values():
    sp.set_edgecolor('#444')

plt.suptitle(
    f'Module 1 — {len(crater_data)} Doubly Shadowed Crater Candidates',
    color='white', fontsize=13, y=1.01
)
plt.tight_layout()
plt.savefig(OUTPUT_MAP, dpi=100, bbox_inches='tight',
            facecolor='#0a0a0a')
plt.close()

del dem, psr_mask, crater_floors, real_crater_mask
gc.collect()

print(f"    ✓ Saved → {OUTPUT_MAP}")
print()
print("=" * 55)
print(" MODULE 1 COMPLETE")
print(f" → Open {OUTPUT_MAP}  to see your map")
print(f" → Open {OUTPUT_CSV}  to see crater table")
print("=" * 55)