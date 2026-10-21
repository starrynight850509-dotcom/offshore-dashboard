# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 13:22:20 2026

@author: ali.chang
"""

import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

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
    "Data Processing",
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

metadata = {
    "updated": pd.Timestamp.now(
        tz="Asia/Taipei"
    ).strftime("%Y-%m-%d %H:%M")
}
pd.to_pickle(
    metadata,
    "metadata.pkl"
)

print("✅ Pickle files generated successfully!")
# ============================================================
# PostgreSQL Schedule Data Upload
# ============================================================

def upload_schedule_to_neon():
    load_dotenv()

    database_url = os.getenv("DATABASE_URL")

    if not database_url:
        raise RuntimeError("DATABASE_URL is not configured")

    engine = create_engine(database_url)
    # Always load the latest processed data
    progress_df = pd.read_pickle("progress.pkl")
    hourly_upload_df = pd.read_pickle("hourly.pkl")
    # Convert DataFrames into JSON-compatible records
    progress_json = progress_df.to_json(
        orient="records",
        date_format="iso"
    )
    
    hourly_json = hourly_upload_df.to_json(
        orient="records",
        date_format="iso"
    )

    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS schedule_data (
                name TEXT PRIMARY KEY,
                data JSONB NOT NULL,
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
        """))

        conn.execute(
            text("""
                INSERT INTO schedule_data (name, data, updated_at)
                VALUES (:name, CAST(:data AS JSONB), NOW())
                ON CONFLICT (name)
                DO UPDATE SET
                    data = EXCLUDED.data,
                    updated_at = EXCLUDED.updated_at
            """),
            {
                "name": "progress",
                "data": progress_json
            }
        )

        conn.execute(
            text("""
                INSERT INTO schedule_data (name, data, updated_at)
                VALUES (:name, CAST(:data AS JSONB), NOW())
                ON CONFLICT (name)
                DO UPDATE SET
                    data = EXCLUDED.data,
                    updated_at = EXCLUDED.updated_at
            """),
            {
                "name": "hourly",
                "data": hourly_json
            }
        )

    engine.dispose()

    print("Schedule data uploaded to Neon successfully!")

#%% Main
if __name__ == "__main__":
    upload_schedule_to_neon()
