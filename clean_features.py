import pandas as pd

# ===========================
# Input / Output
# ===========================

INPUT_CSV = "crater_features.csv"
OUTPUT_CSV = "crater_features_clean.csv"

# ===========================
# Read CSV
# ===========================

df = pd.read_csv(INPUT_CSV)

print(f"Original rows : {len(df)}")

# ===========================
# Remove rows with missing radar values
# ===========================

df_clean = df.dropna(subset=["VOL", "ODD", "EVN", "HELIX"]).copy()

print(f"Remaining rows: {len(df_clean)}")
print(f"Removed rows  : {len(df)-len(df_clean)}")

# ===========================
# Save cleaned CSV
# ===========================

df_clean.to_csv(OUTPUT_CSV, index=False)

print(f"\nSaved: {OUTPUT_CSV}")