import pandas as pd
import matplotlib.pyplot as plt

# ============================
# Read Data
# ============================

df = pd.read_csv("ice_priority.csv")

print(f"Loaded {len(df)} craters")

# ============================
# Split by Priority
# ============================

high = df[df["Priority"] == "High Priority"]
medium = df[df["Priority"] == "Medium Priority"]
low = df[df["Priority"] == "Low Priority"]

# ============================
# Plot
# ============================

plt.figure(figsize=(10,10))

plt.scatter(
    low["x_polar_m"],
    low["y_polar_m"],
    s=12,
    label="Low Priority"
)

plt.scatter(
    medium["x_polar_m"],
    medium["y_polar_m"],
    s=18,
    label="Medium Priority"
)

plt.scatter(
    high["x_polar_m"],
    high["y_polar_m"],
    s=25,
    label="High Priority"
)

# ============================
# Highlight Top 10
# ============================

top10 = df.head(10)

plt.scatter(
    top10["x_polar_m"],
    top10["y_polar_m"],
    s=80,
    marker="*",
    label="Top 10 Ice Targets"
)

# Add labels

for _, row in top10.iterrows():

    plt.text(
        row["x_polar_m"],
        row["y_polar_m"],
        str(row["Rank"]),
        fontsize=8
    )

plt.xlabel("X Coordinate (m)")
plt.ylabel("Y Coordinate (m)")

plt.title("Ranked Lunar South Pole Ice Candidate Craters")

plt.legend()

plt.grid(True)

plt.savefig(
    "ice_priority_map.png",
    dpi=300
)

plt.show()

print("\nSaved : ice_priority_map.png")