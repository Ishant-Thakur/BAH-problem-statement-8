import pandas as pd
import numpy as np

INPUT = "crater_features_clean.csv"
OUTPUT = "crater_features_with_index.csv"

EPS = 1e-12

df = pd.read_csv(INPUT)

df["IceIndex"] = df["VOL"] / (df["ODD"] + df["EVN"] + EPS)

df.to_csv(OUTPUT, index=False)

print(df[["VOL", "ODD", "EVN", "IceIndex"]].head())

print("\nSaved:", OUTPUT)