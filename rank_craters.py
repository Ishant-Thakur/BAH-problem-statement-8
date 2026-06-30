import pandas as pd

# ==========================================
# INPUT / OUTPUT
# ==========================================

INPUT = "crater_features_with_index.csv"
OUTPUT = "ice_priority.csv"

# ==========================================
# READ DATA
# ==========================================

df = pd.read_csv(INPUT)

# ==========================================
# REMOVE INVALID VALUES
# ==========================================

df = df.dropna(subset=["IceIndex"]).copy()

print(f"Total Valid Craters : {len(df)}")

# ==========================================
# CALCULATE PERCENTILES
# ==========================================

Q1 = df["IceIndex"].quantile(0.25)
Q2 = df["IceIndex"].quantile(0.50)
Q3 = df["IceIndex"].quantile(0.75)

print("\nIce Index Statistics")
print("----------------------")
print(f"Q1 (25%) : {Q1:.4f}")
print(f"Median   : {Q2:.4f}")
print(f"Q3 (75%) : {Q3:.4f}")

# ==========================================
# CLASSIFICATION
# ==========================================

def classify(value):

    if value >= Q3:
        return "High Priority"

    elif value >= Q1:
        return "Medium Priority"

    else:
        return "Low Priority"

df["Priority"] = df["IceIndex"].apply(classify)

# ==========================================
# SORT
# ==========================================

df = df.sort_values(
    by="IceIndex",
    ascending=False
)

# ==========================================
# ADD RANK
# ==========================================

df.insert(
    0,
    "Rank",
    range(1, len(df)+1)
)

# ==========================================
# SAVE
# ==========================================

df.to_csv(
    OUTPUT,
    index=False
)

print("\nTop 20 Candidate Craters\n")

print(
    df[
        [
            "Rank",
            "IceIndex",
            "Priority"
        ]
    ].head(20)
)

print(f"\nSaved : {OUTPUT}")