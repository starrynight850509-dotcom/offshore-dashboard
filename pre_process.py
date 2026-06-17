# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 13:22:20 2026

@author: ali.chang
"""

import pandas as pd
import numpy as np

# progress
df = pd.read_excel("progress.xlsx")

df["Date"] = pd.to_datetime(df["Date"])
df["Start"] = df["Date"]
df["End"] = df["Date"] + pd.Timedelta(hours=24)

df["Task"] = df["Task"].astype("category")
df["Category"] = df["Category"].astype("category")
df["Cluster"] = pd.to_numeric(df["Cluster"], errors="coerce")

df["Progress"] = (
    df["Progress"]
    .astype(str)
    .str.replace("%", "", regex=False)
)
df["Progress"] = pd.to_numeric(
    df["Progress"],
    errors="coerce"
).fillna(0)

if df["Progress"].max() <= 1:
    df["Progress"] *= 100

df["Progress"] = df["Progress"].round().astype(int)

df["Date_str"] = df["Date"].dt.strftime("%Y-%m-%d (%a)")
df["Cluster_hover"] = np.where(
    df["Cluster"].notna(),
    df["Cluster"].astype("Int64").astype(str),
    ""
)
df["Progress_hover"] = df["Progress"].astype(str)

for col in [
    "Supervisor",
    "Pilot",
    "Tether Manager",
    "Assistant",
    "Remark",
]:
    if col in df.columns:
        df[col] = df[col].fillna("")



# df["Lane"] = np.where(
#     df["Cluster"].notna(),
#     "Cluster "
#     + df["Cluster"].astype("Int64").astype(str)
#     + " | "
#     + df["Task"].astype(str),
#     df["Category"].astype(str)
#     + " | "
#     + df["Task"].astype(str),
# )
non_operational_categories = [
    "WOW(onshore)",
    "Delay",
    "Day off",
    "Data Processing"
]

df["Lane"] = np.where(
    df["Category"].isin(non_operational_categories),
    "Non-Offshore",
    "Cluster "
    + df["Cluster"].astype("Int64").astype(str)
    + " | "
    + df["Task"].astype(str)
)


# 存成 Pickle
df.to_pickle("progress.pkl")

# hourly
hourly_df = pd.read_excel("hourly.xlsx")
hourly_df["Date"] = pd.to_datetime(hourly_df["Date"])
hourly_df.to_pickle("hourly.pkl")


print("✅ Pickle files generated successfully!")



