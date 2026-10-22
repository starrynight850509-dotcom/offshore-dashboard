"""Fetch F2 Open-Meteo Marine forecast and store each run in Neon PostgreSQL."""
import os
from datetime import datetime, timezone
import requests
import psycopg2
from psycopg2.extras import execute_values

URL = 'https://marine-api.open-meteo.com/v1/marine'
LAT, LON = 24.70843, 120.75996
MODEL = 'best_match'


def main():
    db_url = os.environ['DATABASE_URL']
    params = {
        'latitude': LAT, 'longitude': LON,
        'hourly': 'wave_height,wave_period,wave_direction',
        'models': MODEL, 'timezone': 'GMT', 'forecast_days': 7,
        'cell_selection': 'sea',
    }
    response = requests.get(URL, params=params, timeout=30)
    if not response.ok:
        raise RuntimeError(f'Open-Meteo HTTP {response.status_code}: {response.text[:500]}')
    data = response.json()
    hourly = data.get('hourly') or {}
    times = hourly.get('time') or []
    if not times:
        raise RuntimeError('Open-Meteo returned no hourly forecast')
    fields = ('wave_height', 'wave_period', 'wave_direction')
    if any(len(hourly.get(field, [])) != len(times) for field in fields):
        raise RuntimeError('Open-Meteo hourly arrays have inconsistent lengths')
    fetched_at = datetime.now(timezone.utc)
    rows = []
    for i, stamp in enumerate(times):
        valid_time = datetime.fromisoformat(stamp.replace('Z', '+00:00'))
        if valid_time.tzinfo is None:
            valid_time = valid_time.replace(tzinfo=timezone.utc)
        rows.append((fetched_at, valid_time, MODEL, LAT, LON,
                     hourly['wave_height'][i], hourly['wave_period'][i],
                     hourly['wave_direction'][i]))
    with psycopg2.connect(db_url) as conn:
        with conn.cursor() as cur:
            cur.execute('''
                CREATE TABLE IF NOT EXISTS open_meteo_wave_forecast (
                    fetched_at TIMESTAMPTZ NOT NULL,
                    valid_time TIMESTAMPTZ NOT NULL,
                    model TEXT NOT NULL,
                    lat DOUBLE PRECISION NOT NULL,
                    lon DOUBLE PRECISION NOT NULL,
                    wave_height_m DOUBLE PRECISION,
                    wave_period_s DOUBLE PRECISION,
                    wave_direction_deg DOUBLE PRECISION,
                    PRIMARY KEY (valid_time, model, lat, lon)
                )
            ''')
            execute_values(cur, '''
                INSERT INTO open_meteo_wave_forecast
                (fetched_at, valid_time, model, lat, lon, wave_height_m, wave_period_s, wave_direction_deg)
                VALUES %s
                ON CONFLICT (valid_time, model, lat, lon) DO UPDATE SET
                    fetched_at = EXCLUDED.fetched_at,
                    wave_height_m = EXCLUDED.wave_height_m,
                    wave_period_s = EXCLUDED.wave_period_s,
                    wave_direction_deg = EXCLUDED.wave_direction_deg
            ''', rows)
    print(f'SUCCESS: {len(rows)} forecast hours stored at {fetched_at.isoformat()} UTC')


if __name__ == '__main__':
    main()
