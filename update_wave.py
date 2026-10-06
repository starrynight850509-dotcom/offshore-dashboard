# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 15:00:19 2026

@author: ali.chang
"""


# update_wave.py
# 手動更新 F2 Windy Wave Forecast → PostgreSQL

import os
from pathlib import Path

import pandas as pd
import psycopg2
import requests
from dotenv import load_dotenv


# ============================================================
# 1. Environment
# ============================================================

load_dotenv(Path(__file__).resolve().parent / ".env")

DATABASE_URL = os.getenv("DATABASE_URL")
WINDY_API_KEY = os.getenv("WINDY_API_KEY")

WINDY_URL = "https://api.windy.com/api/point-forecast/v2"

F2_LAT = 24.70843
F2_LON = 120.75996
WINDY_MODEL = "gfsWave"


# ============================================================
# 2. Fetch Windy forecast
# ============================================================

def fetch_windy_wave():

    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL is not configured")

    if not WINDY_API_KEY:
        raise RuntimeError("WINDY_API_KEY is not configured")

    payload = {
        "lat": F2_LAT,
        "lon": F2_LON,
        "model": WINDY_MODEL,
        "levels": ["surface"],
        "parameters": ["waves"],
        "key": WINDY_API_KEY
    }

    response = requests.post(
        WINDY_URL,
        json=payload,
        headers={"Content-Type": "application/json"},
        timeout=20
    )

    response.raise_for_status()
    return response.json()


# ============================================================
# 3. Parse forecast
# ============================================================

def parse_windy_wave(data):

    ts = data.get("ts", [])

    if not ts:
        raise ValueError("Windy returned no forecast timestamps")

    wave_df = pd.DataFrame({
        "valid_time": pd.to_datetime(
            ts, unit="ms", utc=True
        ),
        "wave_height_m": data["waves_height-surface"],
        "wave_period_s": data["waves_period-surface"],
        "wave_direction_deg": data["waves_direction-surface"]
    })

    wave_df["model"] = WINDY_MODEL
    wave_df["lat"] = F2_LAT
    wave_df["lon"] = F2_LON

    return wave_df


# ============================================================
# 4. Save to PostgreSQL (UPSERT)
# ============================================================

def save_wave_forecast(wave_df):

    fetched_at = pd.Timestamp.now(tz="UTC")

    rows = []

    for _, r in wave_df.iterrows():
        rows.append((
            fetched_at.to_pydatetime(),
            r["valid_time"].to_pydatetime(),
            r["model"],
            float(r["lat"]),
            float(r["lon"]),
            None if pd.isna(r["wave_height_m"])
            else float(r["wave_height_m"]),
            None if pd.isna(r["wave_period_s"])
            else float(r["wave_period_s"]),
            None if pd.isna(r["wave_direction_deg"])
            else float(r["wave_direction_deg"])
        ))

    with psycopg2.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:

            cur.executemany("""
                INSERT INTO wave_forecast (
                    fetched_at,
                    valid_time,
                    model,
                    lat,
                    lon,
                    wave_height_m,
                    wave_period_s,
                    wave_direction_deg
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)

                ON CONFLICT (valid_time, model, lat, lon)
                DO UPDATE SET
                    fetched_at = EXCLUDED.fetched_at,
                    wave_height_m = EXCLUDED.wave_height_m,
                    wave_period_s = EXCLUDED.wave_period_s,
                    wave_direction_deg = EXCLUDED.wave_direction_deg
            """, rows)

        conn.commit()


# ============================================================
# 5. Manual update
# ============================================================

def main():

    print("Fetching Windy wave forecast...")

    data = fetch_windy_wave()
    wave_df = parse_windy_wave(data)

    save_wave_forecast(wave_df)

    print(
        f"Windy wave forecast updated: "
        f"{len(wave_df)} records"
    )

    print(
        "Last updated:",
        pd.Timestamp.now(tz="Asia/Taipei").strftime(
            "%Y-%m-%d %H:%M:%S"
        )
    )


if __name__ == "__main__":
    main()
