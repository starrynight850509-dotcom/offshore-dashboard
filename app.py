# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 10:24:21 2026

@author: ali.chang
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import os
import sqlite3
import hashlib
import secrets
import re
import psycopg2
import requests

from flask import Flask, request, Response, jsonify
from dash import Dash, dcc, html, Input, Output
from datetime import datetime
from zoneinfo import ZoneInfo
from dash import get_asset_url
from sqlalchemy import create_engine
from sqlalchemy import text
from functools import lru_cache
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from dash import State, no_update
from dotenv import load_dotenv
# Per-worker startup diagnostics (not persisted across Render restarts).
_STARTUP_BEGIN = time.perf_counter()
_STARTUP_STEPS = []
_STARTUP_LOCK = threading.Lock()

_STARTUP_INSTRUMENT_SCHEDULE = False
_STARTUP_INSTRUMENT_WAVE = False
_STARTUP_INSTRUMENT_OPEN_METEO = False

def record_startup_step(label, elapsed):
    if label.startswith("Wave forecast · ") and not _STARTUP_INSTRUMENT_WAVE:
        return
    if label.startswith("Open-Meteo · ") and not _STARTUP_INSTRUMENT_OPEN_METEO:
        return
    with _STARTUP_LOCK:
        _STARTUP_STEPS.append((label, elapsed))
    print(f"[STARTUP] {label}: {elapsed:.3f} s", flush=True)

def measure_startup(label, operation):
    started = time.perf_counter()
    try:
        return operation()
    finally:
        elapsed = time.perf_counter() - started
        record_startup_step(label, elapsed)

load_dotenv(".env")
print("DATABASE_URL loaded:", bool(os.getenv("DATABASE_URL")))
print("WINDY_API_KEY loaded:", bool(os.getenv("WINDY_API_KEY")))



#%% Load schedule data from Neon PostgreSQL

def load_schedule_data():
    """Load schedule using the shared, thread-safe SQLAlchemy connection pool."""
    connect_started = time.perf_counter()
    with engine.connect() as conn:
        if _STARTUP_INSTRUMENT_SCHEDULE:
            record_startup_step("Schedule · Connect", time.perf_counter() - connect_started)
        query_started = time.perf_counter()
        rows = conn.execute(text("""
            SELECT name, data, updated_at
            FROM schedule_data
            WHERE name IN ('progress', 'hourly')
        """)).fetchall()
        if _STARTUP_INSTRUMENT_SCHEDULE:
            record_startup_step("Schedule · Query + fetch", time.perf_counter() - query_started)

    conversion_started = time.perf_counter()
    schedule = {
        name: {
            "df": pd.DataFrame(data),
            "updated_at": updated_at
        }
        for name, data, updated_at in rows
    }

    if "progress" not in schedule or "hourly" not in schedule:
        raise RuntimeError("Schedule data is incomplete")

    progress_df = schedule["progress"]["df"]
    hourly_data = schedule["hourly"]["df"]

    for col in ["Date", "Start", "End"]:
        progress_df[col] = pd.to_datetime(progress_df[col])

    hourly_data["Date"] = pd.to_datetime(
        hourly_data["Date"]
    )

    last_updated = max(
        schedule["progress"]["updated_at"],
        schedule["hourly"]["updated_at"]
    )

    updated_text = (
        pd.Timestamp(last_updated)
        .tz_convert("Asia/Taipei")
        .strftime("%Y-%m-%d %H:%M")
    )

    if _STARTUP_INSTRUMENT_SCHEDULE:
        record_startup_step("Schedule · DataFrame conversion", time.perf_counter() - conversion_started)

    print("Schedule data loaded from Neon")

    return progress_df, hourly_data, updated_text

@lru_cache(maxsize=2)
def load_schedule_cached(version):
    return load_schedule_data()

def get_schedule_updated_at():
    with engine.connect() as conn:
        return conn.execute(text("""
            SELECT MAX(updated_at)
            FROM schedule_data
            WHERE name IN ('progress', 'hourly')
        """)).scalar()

# One Engine shared by all three startup readers. Each thread checks out
# its own connection from the same QueuePool; connections are never shared
# concurrently across threads.
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL is not configured")
SQLALCHEMY_URL = DATABASE_URL.replace("postgresql://", "postgresql+psycopg2://", 1)
engine = create_engine(SQLALCHEMY_URL, pool_pre_ping=True, pool_size=3, max_overflow=2)

# Begin the independent schedule query while wave initialization runs.
# The future is joined before any code accesses df/hourly_df.
_STARTUP_INSTRUMENT_SCHEDULE = True
_startup_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="neon-schedule")
_schedule_future = _startup_executor.submit(
    measure_startup, "Load schedule (Neon)", load_schedule_data
)

#%% Environment Variables
# ============================================================
# Local:
#   從專案資料夾的 .env 讀取環境變數
#
# Render:
#   Render Environment Variables 仍可由 os.getenv() 取得
# ============================================================

load_dotenv()

# PostgreSQL
DATABASE_URL = os.getenv("DATABASE_URL")

# Windy API
WINDY_API_KEY = os.getenv("WINDY_API_KEY")

# Analytics
ANALYTICS_PATH = "/admin"
ANALYTICS_USER = os.getenv("ANALYTICS_USER")
ANALYTICS_PASSWORD = os.getenv("ANALYTICS_PASSWORD")

SESSION_MINUTES = 30


# ------------------------------------------------------------
# PostgreSQL configuration check
# ------------------------------------------------------------
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL is not configured")


# SQLAlchemy PostgreSQL URL
SQLALCHEMY_URL = DATABASE_URL.replace(
    "postgresql://",
    "postgresql+psycopg2://",
    1
)

#%% F2 Wind Farm Tower Master (read-only)
def load_tower_master():
    """Return canonical F2 turbine IDs; schedule/progress remains independent."""
    with engine.connect() as conn:
        rows = conn.execute(text("SELECT tower_id FROM tower_master ORDER BY tower_id")).fetchall()
    towers = [str(row[0]).strip().upper() for row in rows]
    if not towers:
        raise RuntimeError("tower_master is empty")
    return towers

#%% Windy Point Forecast
# ============================================================
# Windy Point Forecast API
# 功能：
#   1. 從 Windy Point Forecast API 取得 F2 代表點的波浪預報
#   2. 將 API JSON 轉換成 pandas DataFrame
#   3. 將最新 wave forecast 儲存至 PostgreSQL
#   4. 讀取目前最新 forecast
#   5. 建立 Plotly 波高圖，供 Dashboard 顯示
#
# 注意：
#   - 此處資料為 Windy Forecast，不是現地觀測值
#   - fetched_at = 取得預報的時間
#   - valid_time = 預報所代表的時間
#   - 相同 valid_time 的 forecast 以最新資料覆寫
#   - fetched_at = 該 forecast 最後更新時間
# ============================================================

WINDY_API_KEY = os.getenv("WINDY_API_KEY")
WINDY_URL = "https://api.windy.com/api/point-forecast/v2"

# F2 wind farm representative point
F2_LAT = 24.70843
F2_LON = 120.75996

# Windy wave model
WINDY_MODEL = "gfsWave"


# ------------------------------------------------------------
# 1. 建立 Windy forecast PostgreSQL 資料表
# ------------------------------------------------------------
def _init_wave_db():
    """
    建立 wave_forecast table。

    IF NOT EXISTS：
    - 第一次執行時建立 table
    - 之後重新啟動 Dashboard 不會重建或刪除既有資料
    """

    connect_started = time.perf_counter()
    with psycopg2.connect(DATABASE_URL) as c:
        record_startup_step("Wave DB · Connect", time.perf_counter() - connect_started)
        ddl_started = time.perf_counter()
        with c.cursor() as cur:

            cur.execute("""
                CREATE TABLE IF NOT EXISTS wave_forecast (
                    fetched_at TIMESTAMPTZ NOT NULL,
                    valid_time TIMESTAMPTZ NOT NULL,
                    model TEXT NOT NULL,
                    lat DOUBLE PRECISION NOT NULL,
                    lon DOUBLE PRECISION NOT NULL,
                    wave_height_m DOUBLE PRECISION,
                    wave_period_s DOUBLE PRECISION,
                    wave_direction_deg DOUBLE PRECISION,

                    PRIMARY KEY (
                        valid_time,
                        model,
                        lat,
                        lon
                    )
                )
            """)

        c.commit()
        record_startup_step("Wave DB · DDL + commit", time.perf_counter() - ddl_started)


# ------------------------------------------------------------
# 2. 向 Windy Point Forecast API 取得波浪預報
# ------------------------------------------------------------
def fetch_windy_wave_forecast():
    """
    呼叫 Windy Point Forecast API，
    取得 F2 representative point 的 wave forecast。

    若 Windy 回傳 HTTP error，
    會先印出 Windy response body，
    方便判斷真正的 400 原因。
    """

    if not WINDY_API_KEY:
        raise RuntimeError("WINDY_API_KEY is not configured")

    # Windy Point Forecast request body
    payload = {
        "lat": F2_LAT,
        "lon": F2_LON,
        "model": WINDY_MODEL,

        # Wave data 使用 surface level
        "levels": ["surface"],

        # waves 會回傳：
        # - waves_height-surface
        # - waves_period-surface
        # - waves_direction-surface
        "parameters": ["waves"],

        "key": WINDY_API_KEY
    }

    # 明確指定 JSON Content-Type
    headers = {
        "Content-Type": "application/json"
    }

    response = requests.post(
        WINDY_URL,
        json=payload,
        headers=headers,
        timeout=20
    )

    # --------------------------------------------------------
    # Debug Windy API error
    # --------------------------------------------------------
    if not response.ok:
        print("Windy API status:", response.status_code)
        print("Windy API response:", response.text)

    # HTTP 4xx / 5xx → exception
    response.raise_for_status()

    return response.json()


# ------------------------------------------------------------
# 3. 將 Windy JSON 轉換成 pandas DataFrame
# ------------------------------------------------------------
def parse_windy_wave(data):
    """
    將 Windy API response 轉成 DataFrame。

    valid_time：
        Forecast 所代表的時間。

    注意：
        保留 Windy API 原始時間解析度，
        不在此處強制 resample 成 hourly。
    """

    ts = data.get("ts", [])

    height = data.get(
        "waves_height-surface",
        []
    )

    period = data.get(
        "waves_period-surface",
        []
    )

    direction = data.get(
        "waves_direction-surface",
        []
    )

    # API 沒有 timestamp → 視為沒有有效 forecast
    if not ts:
        return pd.DataFrame()

    wave_df = pd.DataFrame({
        "valid_time": pd.to_datetime(
            ts,
            unit="ms",
            utc=True
        ),
        "wave_height_m": height,
        "wave_period_s": period,
        "wave_direction_deg": direction
    })

    # 補上 metadata
    wave_df["model"] = WINDY_MODEL
    wave_df["lat"] = F2_LAT
    wave_df["lon"] = F2_LON

    return wave_df


# ------------------------------------------------------------
# 4. 將 Forecast Snapshot 存入 PostgreSQL
# ------------------------------------------------------------
def save_wave_forecast(wave_df):
    """
    儲存目前最新的 Windy forecast。

    fetched_at：
        本次向 Windy API 抓取資料的時間。

    valid_time：
        每一筆 forecast 所代表的時間。

    相同 valid_time / model / lat / lon 的 forecast
    以最新取得的資料覆寫。
    """

    if wave_df.empty:
        return

    # 整批資料共用同一個 fetched_at
    fetched_at = pd.Timestamp.now(tz="UTC")

    rows = []

    for _, r in wave_df.iterrows():

        rows.append((
            fetched_at.to_pydatetime(),
            r["valid_time"].to_pydatetime(),
            r["model"],
            float(r["lat"]),
            float(r["lon"]),

            None
            if pd.isna(r["wave_height_m"])
            else float(r["wave_height_m"]),

            None
            if pd.isna(r["wave_period_s"])
            else float(r["wave_period_s"]),

            None
            if pd.isna(r["wave_direction_deg"])
            else float(r["wave_direction_deg"])
        ))

    with psycopg2.connect(DATABASE_URL) as c:
        with c.cursor() as cur:

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
                VALUES (
                    %s, %s, %s, %s,
                    %s, %s, %s, %s
                )
                ON CONFLICT (valid_time, model, lat, lon)
                DO UPDATE SET
                    fetched_at = EXCLUDED.fetched_at,
                    wave_height_m = EXCLUDED.wave_height_m,
                    wave_period_s = EXCLUDED.wave_period_s,
                    wave_direction_deg = EXCLUDED.wave_direction_deg
            """, rows)

        c.commit()


# ------------------------------------------------------------
# 5. Windy Forecast 更新流程
# ------------------------------------------------------------
def update_windy_wave():
    """
    完整更新流程：

    Windy API
        ↓
    JSON
        ↓
    DataFrame
        ↓
    PostgreSQL

    Windy API 失敗時不讓整個 Dashboard crash。
    """

    try:

        data = fetch_windy_wave_forecast()

        wave_df = parse_windy_wave(data)

        save_wave_forecast(wave_df)

        print(
            f"Windy wave forecast updated: "
            f"{len(wave_df)} records"
        )

    except Exception as e:

        # Windy 是 Dashboard enhancement，
        # API 暫時失敗不應造成整個 Gantt 無法使用。
        print(
            f"Windy update failed: {e}"
        )


# ------------------------------------------------------------
# 6. 讀取最新一次 Forecast Snapshot
# ------------------------------------------------------------
def load_latest_wave_forecast():
    """
    從 PostgreSQL 取得 F2 representative point 目前保存的最新 forecast。

    相同 valid_time 只保留最新資料，因此不再依 fetched_at snapshot 篩選。
    """

    query = """
        SELECT
            fetched_at,
            valid_time,
            model,
            lat,
            lon,
            wave_height_m,
            wave_period_s,
            wave_direction_deg

        FROM wave_forecast

        WHERE model = %(model)s
          AND lat = %(lat)s
          AND lon = %(lon)s

        ORDER BY valid_time
    """

    connect_started = time.perf_counter()
    with engine.connect() as c:
        record_startup_step("Wave forecast · Connect", time.perf_counter() - connect_started)
        query_started = time.perf_counter()
        wave_df = pd.read_sql_query(
            query,
            c,
            params={
                "model": WINDY_MODEL,
                "lat": F2_LAT,
                "lon": F2_LON
            }
        )
        record_startup_step("Wave forecast · Query + fetch", time.perf_counter() - query_started)

    conversion_started = time.perf_counter()
    if not wave_df.empty:

        # PostgreSQL 儲存 UTC，
        # Dashboard 顯示轉為台灣時間。
        wave_df["valid_time"] = (
            pd.to_datetime(
                wave_df["valid_time"],
                utc=True
            )
            .dt.tz_convert("Asia/Taipei")
        )

        wave_df["fetched_at"] = (
            pd.to_datetime(
                wave_df["fetched_at"],
                utc=True
            )
            .dt.tz_convert("Asia/Taipei")
        )

    record_startup_step("Wave forecast · DataFrame conversion", time.perf_counter() - conversion_started)
    return wave_df


# ------------------------------------------------------------
# 7. 建立 Wave Forecast Plotly Chart
# ------------------------------------------------------------
# Open-Meteo forecast: read-only Neon source.
def load_open_meteo_forecast():
    connect_started = time.perf_counter()
    with engine.connect() as c:
        record_startup_step("Open-Meteo · Connect", time.perf_counter() - connect_started)
        query_started = time.perf_counter()
        data = pd.read_sql_query(text("""SELECT fetched_at, valid_time, model, lat, lon,
            wave_height_m, wave_period_s, wave_direction_deg
            FROM open_meteo_wave_forecast
            WHERE model=:model AND lat=:lat AND lon=:lon ORDER BY valid_time"""), c,
            params={"model":"best_match","lat":F2_LAT,"lon":F2_LON})
        record_startup_step("Open-Meteo · Query + fetch", time.perf_counter() - query_started)
    conversion_started = time.perf_counter()
    if not data.empty:
        for column in ("valid_time", "fetched_at"):
            data[column] = pd.to_datetime(data[column], utc=True).dt.tz_convert("Asia/Taipei")
    record_startup_step("Open-Meteo · DataFrame conversion", time.perf_counter() - conversion_started)
    return data

def get_open_meteo_last_updated():
    with engine.connect() as c:
        value = c.execute(text("""SELECT MAX(fetched_at) FROM open_meteo_wave_forecast
            WHERE model=:model AND lat=:lat AND lon=:lon"""),
            {"model":"best_match","lat":F2_LAT,"lon":F2_LON}).scalar()
    return pd.to_datetime(value,utc=True).tz_convert("Asia/Taipei") if value is not None else None

def wave_arrow(deg):
    # Windy direction = FROM
    # 顯示浪實際前進方向 = +180°
    travel_deg = (deg + 180) % 360

    arrows = ["↑", "↗", "→", "↘", "↓", "↙", "←", "↖"]
    idx = int((travel_deg + 22.5) // 45) % 8

    return arrows[idx]


def build_wave_chart():
    """
    建立 Dashboard 上方的 Windy Wave Forecast 圖。

    Y-axis：
        Forecast significant wave height (Hs)

    Hover：
        - Forecast Hs
        - Wave period
        - Wave direction
    """

    wave_df = load_latest_wave_forecast()
    fig = go.Figure()

    # --------------------------------------------------------
    # 沒有 forecast data
    # --------------------------------------------------------
    if wave_df.empty:

        fig.add_annotation(
            text="Wave forecast unavailable",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False
        )

    # --------------------------------------------------------
    # 有 forecast data
    # --------------------------------------------------------
    else:

        # Windy direction = FROM
        # 轉換成浪實際前進方向
        wave_df["wave_arrow"] = (
            wave_df["wave_direction_deg"].apply(wave_arrow)
        )

        # 例如：37° (to SW)
        wave_df["direction_text"] = (
            wave_df["wave_direction_deg"]
            .round(0)
            .astype(int)
            .astype(str)
            + "° (to "
            + wave_df["wave_arrow"]
            + ")"
        )

        customdata = wave_df[
            ["wave_period_s"]
        ].to_numpy()

        fig.add_trace(
            go.Scatter(
                x=wave_df["valid_time"],
                y=wave_df["wave_height_m"],

                mode="lines+markers",

                name="Forecast Hs",

                customdata=customdata,
                text=wave_df["direction_text"],

                hovertemplate=(
                    "<b>Windy Wave Forecast</b><br>"
                    "Time: %{x|%Y-%m-%d %H:%M}<br>"
                    "Forecast Hs: %{y:.2f} m<br>"
                    "Period: %{customdata[0]:.1f} s<br>"
                    "Direction: %{text}"
                    "<extra></extra>"
                )
            )
        )

    # --------------------------------------------------------
    # Chart layout
    # --------------------------------------------------------
    fig.update_layout(
        title=(
            "Windy Wave Forecast — "
            "F2 Representative Point"
        ),

        height=220,

        margin=dict(
            l=60,
            r=20,
            t=45,
            b=20
        ),

        showlegend=False,

        hovermode="x unified",

        # 保留使用者 zoom / pan 狀態
        uirevision="wave-keep"
    )

    fig.update_yaxes(
        title="Hs (m)",
        rangemode="tozero"
    )

    fig.update_xaxes(
        title=None,
        fixedrange=False
    )

    return fig


# ============================================================
# Windy startup initialization
# ============================================================

# Web App 啟動時只確認 PostgreSQL table 存在。
# 不在 startup 呼叫 Windy API，避免外部 API 延遲拖慢網站啟動。
# Forecast 更新改由獨立 scheduled job 執行。
measure_startup("Initialize wave DB", _init_wave_db)
# Schedule was already submitted above. Run both forecast reads concurrently
# with it; keep DB initialization ahead of the forecast queries.
_STARTUP_INSTRUMENT_WAVE = True
_STARTUP_INSTRUMENT_OPEN_METEO = True
_forecast_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="neon-forecast")
try:
    _windy_future = _forecast_executor.submit(
        measure_startup, "Load wave forecast (Neon)", load_latest_wave_forecast
    )
    _open_meteo_future = _forecast_executor.submit(
        measure_startup, "Load Open-Meteo forecast (Neon)", load_open_meteo_forecast
    )

    # Preserve the existing behavior: schedule and Windy errors are fatal,
    # whereas an unavailable Open-Meteo forecast does not stop the dashboard.
    # Always collect all futures before shutting down the executors.
    _startup_errors = []
    try:
        df, hourly_df, data_updated = _schedule_future.result()
    except Exception as exc:
        _startup_errors.append(exc)
    try:
        wave_df_status = _windy_future.result()
    except Exception as exc:
        _startup_errors.append(exc)
    try:
        open_meteo_df_status = _open_meteo_future.result()
    except Exception as exc:
        print(f"Open-Meteo startup unavailable: {exc}", flush=True)
        open_meteo_df_status = pd.DataFrame()
    if _startup_errors:
        raise _startup_errors[0]
finally:
    _STARTUP_INSTRUMENT_SCHEDULE = False
    _STARTUP_INSTRUMENT_WAVE = False
    _STARTUP_INSTRUMENT_OPEN_METEO = False
    _forecast_executor.shutdown(wait=True)
    _startup_executor.shutdown(wait=True)

if not wave_df_status.empty:
    last_updated = wave_df_status["fetched_at"].max()

    now_tw = pd.Timestamp.now(tz="Asia/Taipei")
    age_hours = (now_tw - last_updated).total_seconds() / 3600

    windy_updated_text = (
        f"Windy Forecast Updated {last_updated:%Y-%m-%d %H:%M}"
    )

    if age_hours > 6:
        windy_updated_text += " | ⚠ Forecast may be outdated"

else:
    windy_updated_text = "Windy Forecast Unavailable"
    
def get_windy_last_updated():
    """
    取得 F2 representative point 的 Windy forecast
    最後更新時間。
    只查 MAX(fetched_at)，不載入整批 forecast。
    """
    query = text("""
        SELECT MAX(fetched_at) AS last_updated
        FROM wave_forecast
        WHERE model = :model
          AND lat = :lat
          AND lon = :lon
    """)
    with engine.connect() as c:
        result = c.execute(
            query,
            {
                "model": WINDY_MODEL,
                "lat": F2_LAT,
                "lon": F2_LON
            }
        ).scalar()

    if result is None:
        return None

    last_updated = pd.Timestamp(result)

    if last_updated.tzinfo is None:
        last_updated = last_updated.tz_localize("UTC")

    return last_updated.tz_convert("Asia/Taipei")
    


#%% Site analytics
DATABASE_URL = os.getenv("DATABASE_URL")
ANALYTICS_PATH = "/admin"
ANALYTICS_USER = os.getenv("ANALYTICS_USER")
ANALYTICS_PASSWORD = os.getenv("ANALYTICS_PASSWORD")
SESSION_MINUTES = 30

OFFICE_ROLES = {
    "59.125.75.250": "IOG-TC",
    "210.242.183.103": "IOG-TC",
}

ADMIN_IPS = {
    # "59.125.75.250",
    "127.0.0.1",
}

if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL is not configured")

SQLALCHEMY_URL = DATABASE_URL.replace(
    "postgresql://",
    "postgresql+psycopg2://",
    1
)

def _adb():
    c = psycopg2.connect(DATABASE_URL)
    with c.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS page_views(
                ts TIMESTAMPTZ NOT NULL,
                visitor TEXT NOT NULL,
                device TEXT,
                browser TEXT,
                path TEXT
            )
        """)
    c.commit()
    return c


def _check_auth():
    auth = request.authorization
    if not auth or not ANALYTICS_USER or not ANALYTICS_PASSWORD:
        return False
    return (
        secrets.compare_digest(auth.username, ANALYTICS_USER)
        and secrets.compare_digest(auth.password, ANALYTICS_PASSWORD)
    )


def _auth_required():
    return Response(
        "Authentication required.", 401,
        {"WWW-Authenticate": 'Basic realm="Site Analytics"'}
    )


def _client_info():
    ua = request.headers.get("User-Agent", "")
    ip = request.headers.get(
        "X-Forwarded-For", request.remote_addr or ""
    ).split(",")[0].strip()

    u = ua.lower()

    device = (
        "Mobile"
        if any(x in u for x in ["mobile", "android", "iphone"])
        else "Desktop"
    )

    if "edg/" in u:
        browser = "Edge"
    elif "chrome/" in u:
        browser = "Chrome"
    elif "firefox/" in u:
        browser = "Firefox"
    elif "safari/" in u:
        browser = "Safari"
    else:
        browser = "Other"

    return ip, device, browser


def _track():
    if request.method != "GET" or request.path != "/":
        return

    ua = request.headers.get("User-Agent", "")
    u = ua.lower()

    blocked = [
        "python-requests", "curl/", "wget/",
        "bot", "crawler", "spider",
        "googlebot", "bingbot",
        "monitor", "healthcheck", "uptime"
    ]

    if not ua or any(x in u for x in blocked):
        return

    if not any(
        x in u for x in ["edg/", "chrome/", "firefox/", "safari/"]
    ):
        return

    visitor, device, browser = _client_info()

    try:
        with _adb() as c:
            with c.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO page_views
                    (ts, visitor, device, browser, path)
                    VALUES (%s, %s, %s, %s, %s)
                    """,
                    (
                        datetime.now(ZoneInfo("Asia/Taipei")),
                        visitor,
                        device,
                        browser,
                        "/"
                    )
                )
    except psycopg2.Error:
        pass


def _sessionize(v):
    if v.empty:
        v["session_id"] = pd.Series(dtype="object")
        return v

    v = v.sort_values(["visitor", "ts"]).copy()

    gap = v.groupby("visitor")["ts"].diff()

    new_session = (
        gap.isna()
        | (gap > pd.Timedelta(minutes=SESSION_MINUTES))
    )

    v["_session"] = (
        new_session.groupby(v["visitor"])
        .cumsum()
        .astype(int)
    )

    v["session_id"] = (
        v["visitor"] + "-" + v["_session"].astype(str)
    )

    return v.drop(columns="_session")


def _startup_diagnostics_html():
    from html import escape
    with _STARTUP_LOCK:
        steps = list(_STARTUP_STEPS)
    total = globals().get("_STARTUP_READY_SECONDS")
    if total is None:
        total = time.perf_counter() - _STARTUP_BEGIN

    groups = [
        ("Load schedule (Neon)", "Schedule · ", "#3b82f6"),
        ("Load wave forecast (Neon)", "Wave forecast · ", "#8b5cf6"),
        ("Load Open-Meteo forecast (Neon)", "Open-Meteo · ", "#0d9488"),
        ("Initialize wave DB", "Wave DB · ", "#f59e0b"),
    ]
    main_steps = {label: duration for label, duration in steps if " · " not in label}

    def row(label, elapsed, color, denominator, nested=False, top_detail=False):
        percent = elapsed / denominator * 100 if denominator > 0 else 0
        if nested:
            marker = ' <span class="startup-top-marker">Highest</span>' if top_detail else ''
            css_class = 'startup-detail startup-detail-top' if top_detail else 'startup-detail'
            return (
                f'<tr class="{css_class}"><td>↳ {escape(label)}{marker}</td>'
                f'<td class="duration">{elapsed:.3f} s</td>'
                f'<td class="duration">{percent:.1f}%</td><td></td></tr>'
            )
        width = min(100, max(0, percent))
        return (
            f'<tr class="startup-main"><td><strong>{escape(label)}</strong></td>'
            f'<td class="duration"><strong>{elapsed:.3f} s</strong></td>'
            f'<td class="duration"><strong>{percent:.1f}%</strong></td>'
            f'<td><div class="startup-track"><div class="startup-bar" '
            f'style="width:{width:.2f}%;background:{color}"></div></div></td></tr>'
        )

    rows = ""
    for main_label, detail_prefix, color in groups:
        if main_label not in main_steps:
            continue
        duration = main_steps[main_label]
        rows += row(main_label, duration, color, total)
        # Deduplicate startup sub-stages and highlight the slowest within each parent.
        details = {}
        for label, value in steps:
            if label.startswith(detail_prefix):
                short_label = label[len(detail_prefix):]
                if short_label not in details:
                    details[short_label] = value
        highest = max(details, key=details.get) if details else None
        for short_label, value in details.items():
            rows += row(short_label, value, color, duration, nested=True,
                        top_detail=(short_label == highest))

    grouped_names = {g[0] for g in groups}
    for label, elapsed in steps:
        if " · " not in label and label not in grouped_names:
            rows += row(label, elapsed, "#10b981", total)
    rows += (f'<tr class="startup-total"><td><strong>Python startup total</strong></td>'
             f'<td class="duration"><strong>{total:.3f} s</strong></td>'
             f'<td class="duration"><strong>100.0%</strong></td><td></td></tr>')
    return (
        '<section class="startup-diagnostics">'
        '<h3>Startup Diagnostics</h3>'
        '<p class="note">Current Python worker only · timings reset on restart. '
        'Render wake-up time before Python starts is not measured.</p>'
        '<p class="note">Main stages: % of Python startup total. '
        'Indented stages: % of their parent stage. Parallel stages may overlap.</p>'
        '<div class="startup-table-wrap"><table class="startup-table">'
        '<thead><tr><th>Startup stage</th><th>Duration</th><th>Share</th><th>Relative duration</th></tr></thead>'
        f'<tbody>{rows}</tbody></table></div></section>'
    )


def _analytics(fragment=False):
    with engine.connect() as c:
        v = pd.read_sql_query(
            """
            SELECT ts, visitor, device, browser, path
            FROM page_views
            ORDER BY ts
            """,
            c
        )

    if not v.empty:
        v["ts"] = (
            pd.to_datetime(v["ts"], utc=True)
            .dt.tz_convert("Asia/Taipei")
        )

        v = _sessionize(v)

        v["role"] = v["visitor"].apply(
            lambda ip: ("Admin" if ip in ADMIN_IPS
                        else OFFICE_ROLES.get(ip, "Visitor"))
        )
        visitors = v[v["role"] == "Visitor"].copy()

        now = pd.Timestamp.now(tz="Asia/Taipei")

        today = visitors[visitors["ts"] >= now.normalize()]
        d7 = visitors[visitors["ts"] >= now - pd.Timedelta(days=7)]
        d30 = visitors[visitors["ts"] >= now - pd.Timedelta(days=30)]

    else:
        v["role"] = pd.Series(dtype="object")
        today = d7 = d30 = v

    groups = [
        ("Today", today),
        ("7 Days", d7),
        ("30 Days", d30)
    ]

    cards = ""

    for title, data in groups:
        unique = (
            data["visitor"].nunique()
            if not data.empty else 0
        )

        sessions = (
            data["session_id"].nunique()
            if not data.empty else 0
        )

        cards += f"""
        <div class="section-title">{title}</div>

        <div class="card-row">
            <div class="card">
                <b>{sessions}</b><br>Sessions
            </div>

            <div class="card">
                <b>{unique}</b><br>Unique IPs
            </div>
        </div>
        """

    # One row per session, using its first recorded page view (session start).
    # The original page_views table remains unchanged.
    if not v.empty:
        session_rows = (
            v.sort_values("ts")
            .drop_duplicates(subset=["session_id"], keep="first")
            .sort_values("ts", ascending=False)
            .copy()
        )
        recent_data = session_rows.rename(
            columns={"visitor": "ip", "role": "Role"}
        ).copy()
        recent_data["ts"] = recent_data["ts"].dt.strftime("%Y-%m-%d %H:%M:%S")
        recent_data = recent_data[
            ["ts", "ip", "Role", "device", "browser", "session_id"]
        ]
        recent = recent_data.to_html(
            index=False, classes="analytics-table", escape=True
        )
    else:
        session_rows = v.copy()
        recent_data = pd.DataFrame(
            columns=["ts", "ip", "Role", "device", "browser", "session_id"]
        )
        recent = "<p>No sessions yet.</p>"

    # Add client-side filters to each table header (no database changes).
    from html import escape as html_escape
    import re

    columns = ["ts", "ip", "Role", "device", "browser", "session_id"]
    labels = ["Start Time", "IP", "Role", "Device", "Browser", "Session ID"]
    if not recent_data.empty:
        options_by_col = [
            sorted(recent_data[col].fillna("").astype(str).unique())
            for col in columns
        ]
    else:
        options_by_col = [[] for _ in columns]

    if "<table" not in recent:
        recent = '<table class="dataframe analytics-table"><thead><tr>' + ''.join(
            f'<th>{label}</th>' for label in labels
        ) + '</tr></thead><tbody></tbody></table>'

    header_html = []
    for idx, label in enumerate(labels):
        if idx == 5:
            header_html.append("<th>Session ID</th>")
            continue
        if idx == 0:
            choices = [('desc', 'Newest first'), ('asc', 'Oldest first')]
        else:
            choices = [('', 'All')] + [(item, item) for item in options_by_col[idx] if item]
        option_html = ''.join(
            f'<option value="{html_escape(value, quote=True)}">{html_escape(title)}</option>'
            for value, title in choices
        )
        header_html.append(
            f'<th><label class="filter-label" for="filter-{idx}">{label}</label>'
            f'<select class="column-filter" id="filter-{idx}" data-column="{idx}" autocomplete="off" '
            f'aria-label="Filter {label}">{option_html}</select></th>'
        )
    recent = re.sub(r'<thead>.*?</thead>',
                    '<thead><tr>' + ''.join(header_html) + '</tr></thead>',
                    recent, count=1, flags=re.DOTALL)

    if fragment:
        return {"cards": cards, "recent": recent, "session_count": len(session_rows)}

    return f"""
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 30px;
            background: #f8fafc;
            color: #111827;
        }}

        .section-title {{
            font-size: 16px;
            font-weight: bold;
            margin: 12px 0 4px;
        }}

        .card-row {{
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
        }}

        .card {{
            flex: 1;
            min-width: 0;
            background: white;
            border: 1px solid #ddd;
            border-radius: 10px;
            padding: 9px 10px;
            font-size: 13px;
            text-align: center;
            box-sizing: border-box;
        }}

        .card b {{
            font-size: 21px;
            color: #1f77b4;
        }}

        .note {{
            color: #6b7280;
            font-size: 13px;
        }}

        /* Two-column layout */

        .analytics-layout {{
            display: grid;
            grid-template-columns:
                minmax(0, 1fr) minmax(0, 1fr);
            gap: 30px;
            align-items: start;
        }}

        .analytics-summary,
        .analytics-history {{
            min-width: 0;
        }}

        .analytics-history h3, .analytics-summary h3 {{
            font-size: 18px;
            margin: 20px 0 5px;
        }}

        /* Recent Sessions */

        .table-container {{
            width: 100%;
            max-width: 100%;
            max-height: 350px;
            overflow-y: auto;
            overflow-x: hidden;
            border: 1px solid #ddd;
            border-radius: 10px;
            background: white;
            box-sizing: border-box;
            direction: ltr;
        }}

        .analytics-table {{
            width: 100%;
            table-layout: fixed;
            border-collapse: separate;
            border-spacing: 0;
            margin: 0;
            font-size: 13px;
        }}

        .analytics-table th,
        .analytics-table td {{
            padding: 10px 8px;
            border-right: 1px solid #e5e7eb;
            border-bottom: 1px solid #e5e7eb;
            text-align: left;
            overflow-wrap: anywhere;
            word-break: normal;
            vertical-align: middle;
        }}

        .analytics-table th {{
            position: sticky;
            top: 0;
            background: #f1f5f9;
            z-index: 2;
            font-weight: bold;
        }}

        .column-filter {{
            display: block;
            width: 100%;
            max-width: 100%;
            min-width: 0;
            box-sizing: border-box;
            margin-top: 5px;
            padding: 5px 2px;
            font-size: 13px;
        }}

        .filter-label {{ display: block; }}

        .analytics-table th:last-child,
        .analytics-table td:last-child {{
            border-right: none;
        }}

        .analytics-table tr:hover td {{
            background: #eff6ff;
        }}

        /* Six columns; Session ID is display-only */
        .analytics-table th:nth-child(1) {{ width: 25%; }}
        .analytics-table th:nth-child(2) {{ width: 18%; }}
        .analytics-table th:nth-child(3) {{ width: 11%; }}
        .analytics-table th:nth-child(4) {{ width: 12%; }}
        .analytics-table th:nth-child(5) {{ width: 12%; }}
        .analytics-table th:nth-child(6) {{ width: 22%; }}

        /* Vertical scrollbar */

        .table-container::-webkit-scrollbar {{
            width: 8px;
        }}

        .table-container::-webkit-scrollbar-thumb {{
            background: #94a3b8;
            border-radius: 6px;
        }}

        .table-container::-webkit-scrollbar-track {{
            background: #f1f5f9;
        }}

        /* Responsive layout */

        @media (max-width: 1200px) {{
            .analytics-layout {{
                grid-template-columns: 1fr;
            }}

            .analytics-summary,
            .analytics-history {{
                width: 100%;
            }}
        }}

        @media (max-width: 600px) {{
            body {{
                margin: 16px;
            }}

            .card-row {{
                gap: 6px;
            }}

            .card {{
                padding: 12px 5px;
                font-size: 12px;
            }}

            .card b {{
                font-size: 22px;
            }}

            .analytics-table {{
                font-size: 11px;
            }}

            .analytics-table th,
            .analytics-table td {{
                padding: 7px 4px;
            }}
        }}

        /* Startup timings: compact, independent of Recent Sessions table filters */
        .startup-diagnostics {{ margin-top: 0; }}
        .startup-diagnostics h3 {{ margin-bottom: 8px; }}
        .startup-table-wrap {{ width: 100%; overflow-x: auto; }}
        .startup-table {{ width: 100%; border-collapse: collapse; background: white; font-size: 13px; }}
        .startup-table th, .startup-table td {{ padding: 8px 7px; border-bottom: 1px solid #e5e7eb; text-align: left; }}
        .startup-table th {{ background: #f1f5f9; }}
        .startup-table .duration {{ text-align: right; white-space: nowrap; font-variant-numeric: tabular-nums; }}
        .startup-table th:nth-child(1) {{ width: 43%; }}
        .startup-table th:nth-child(2) {{ width: 17%; text-align: right; }}
        .startup-table th:nth-child(3) {{ width: 14%; text-align: right; }}
        .startup-table th:nth-child(4) {{ width: 26%; }}
        .startup-track {{ background: #e2e8f0; height: 10px; border-radius: 5px; overflow: hidden; }}
        .startup-bar {{ height: 100%; border-radius: 5px; }}
        .startup-main {{ background: #eef4fb; border-top: 2px solid #cbd5e1; }}
        .startup-main td {{ padding-top: 11px; padding-bottom: 11px; font-size: 13px; }}
        .startup-main td:first-child {{ color: #0f172a; }}
        .startup-detail {{ color: #64748b; background: #fff; }}
        .startup-detail td:first-child {{ padding-left: 22px; }}
        .startup-detail td {{ padding-top: 6px; padding-bottom: 6px; }}
        .startup-detail-top {{ background: #fff; color: #0f172a; font-weight: 600; }}
        .startup-detail-top td:first-child {{ border-left: 3px solid #0284c7; padding-left: 19px; }}
        .startup-top-marker {{ display: inline-block; margin-left: 6px; padding: 2px 5px;
            border-radius: 4px; background: #dbeafe; color: #1d4ed8; font-size: 11px; font-weight: 700; }}
        .startup-total {{ background: #e2e8f0; border-top: 2px solid #94a3b8; }}
    </style>

    <h1>Site Analytics</h1>

    <p>Asia/Taipei · Visitor = Public IP</p>

    <p class="note">
        Session = same IP activity grouped within
        {SESSION_MINUTES} minutes
    </p>

    <div class="analytics-layout">

        <!-- Left: Summary + Recent Sessions -->
        <div class="analytics-summary">
            <div id="analytics-cards">{cards}</div>

            <h3>Recent Sessions</h3>
            <div class="table-container" id="recent-container">
                {recent}
            </div>
            <p class="note" id="session-count">
                Showing all {len(session_rows)} sessions ·
                Scroll to view older records
            </p>
        </div>

        <!-- Right: Startup Diagnostics -->
        <div class="analytics-history">

            {_startup_diagnostics_html()}

        </div>

    </div>

    <script>
    (() => {{
        const refreshMs = 60000;
        const cards = document.getElementById('analytics-cards');
        const container = document.getElementById('recent-container');
        const count = document.getElementById('session-count');
        let busy = false;

        function applyFilters() {{
            const table = container.querySelector('.analytics-table');
            if (!table) return;
            const tbody = table.querySelector('tbody');
            const filters = Array.from(table.querySelectorAll('.column-filter'));
            const rows = Array.from(tbody.querySelectorAll('tr'));
            const sort = filters[0] ? filters[0].value : 'desc';
            rows.sort((a, b) => {{
                const at = a.cells[0].textContent.trim();
                const bt = b.cells[0].textContent.trim();
                return sort === 'asc' ? at.localeCompare(bt) : bt.localeCompare(at);
            }});
            const fragment = document.createDocumentFragment();
            rows.forEach(row => {{
                const visible = filters.slice(1).every(filter => {{
                    const idx = Number(filter.dataset.column);
                    return !filter.value || row.cells[idx].textContent.trim() === filter.value;
                }});
                row.style.display = visible ? '' : 'none';
                fragment.appendChild(row);
            }});
            tbody.appendChild(fragment);
        }}

        // Delegated listener also works after the table is replaced.
        container.addEventListener('change', event => {{
            if (event.target.matches('.column-filter')) applyFilters();
        }});
        applyFilters();

        async function refreshAnalytics() {{
            if (busy || document.hidden) return;
            busy = true;
            try {{
                const response = await fetch('/admin/data', {{cache: 'no-store', credentials: 'same-origin'}});
                if (!response.ok) throw new Error('Analytics refresh HTTP ' + response.status);
                const data = await response.json();
                const oldFilters = Array.from(container.querySelectorAll('.column-filter'));
                const selected = oldFilters.map(filter => filter.value);
                const scrollTop = container.scrollTop;
                const scrollLeft = container.scrollLeft;
                cards.innerHTML = data.cards;
                container.innerHTML = data.recent;
                const newFilters = Array.from(container.querySelectorAll('.column-filter'));
                newFilters.forEach((filter, index) => {{
                    if (Array.from(filter.options).some(option => option.value === selected[index])) {{
                        filter.value = selected[index];
                    }}
                }});
                count.textContent = 'Showing all ' + data.session_count + ' sessions · Scroll to view older records';
                applyFilters();
                container.scrollTop = scrollTop;
                container.scrollLeft = scrollLeft;
            }} catch (error) {{
                console.warn('Analytics auto-refresh failed:', error);
            }} finally {{
                busy = false;
            }}
        }}
        setInterval(refreshAnalytics, refreshMs);
    }})();
    </script>

    <p>
        <a href="/">&larr; Back to Dashboard</a>
    </p>
    """





F2_LEAFLET_MAP_HTML = '<!doctype html><html lang="zh-Hant"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>F2 Marine Wave Overlay — Prototype</title><link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"><style>body{margin:0;font:14px system-ui,-apple-system,sans-serif;background:#101827;color:#edf2f7}.wrap{max-width:1300px;margin:auto;padding:18px}.top{display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:12px}h2{margin:0 0 5px;font-size:21px}.muted{color:#aab8cc}#map{height:min(76vh,760px);min-height:460px;border-radius:12px;overflow:hidden;margin-top:8px}.controls{display:flex;align-items:center;gap:12px;flex-wrap:wrap;margin:10px 0}.controls button{background:#263c5a;color:white;border:1px solid #57708e;border-radius:6px;padding:8px 12px;cursor:pointer}input{accent-color:#52a4e6}#status{font-size:12px;padding:8px 0;color:#b9c8db}.tower-label{font-size:11px;font-weight:700;color:#102036;background:#fff;padding:2px 4px;border-radius:3px;box-shadow:0 1px 3px #0006;white-space:nowrap}.legend{background:white;color:#1b293b;padding:8px 10px;border-radius:5px;font:12px system-ui}.legend .bar{height:10px;width:175px;background:linear-gradient(90deg,#262f90,#2084c5,#32b88e,#ffe45e,#f28c39,#bd2d2d);margin:6px 0}.wave-key{background:#1b2b42;border:1px solid #48607e;border-radius:9px;padding:12px 14px;margin:12px 0 4px;max-width:620px}.wave-key-title{font-weight:700;margin-bottom:7px}.wave-key-bar{height:15px;border-radius:4px;background:linear-gradient(to right,#3b276d,#6c3f9a,#ae6595,#dd8b7d,#eeaa64,#f4d783,#f5f7a6)}.wave-key-ticks{display:flex;justify-content:space-between;font-size:12px;margin-top:5px}.wave-key-note{font-size:11px;color:#b9c8db;margin-top:7px;line-height:1.5}.note{line-height:1.6;font-size:12px;color:#aab8cc}\n.filter-row{margin-top:0}.filter-row .map-controls{max-width:none;box-shadow:none;background:transparent;border:0;padding:0;font-size:13px}.filter-row .map-controls input[type=range]{width:95px}\n/* Compact map-first interface. Controls are HTML overlays, not geographic layers. */\n#map{position:relative;height:min(82vh,800px);min-height:520px;margin-top:14px}\n.map-bottom{position:absolute;z-index:900;left:12px;right:12px;bottom:22px;display:flex;flex-direction:column;gap:8px;align-items:flex-start;pointer-events:none}\n.map-controls{display:flex;flex-wrap:wrap;align-items:center;gap:8px 12px;background:rgba(16,24,39,.93);color:#edf2f7;border:1px solid #56708c;border-radius:9px;padding:9px 11px;box-shadow:0 3px 14px #0006;pointer-events:auto;max-width:calc(100% - 22px);font-size:12px}\n.map-controls label{display:flex;align-items:center;gap:5px;white-space:nowrap}\n.map-controls input[type=range]{width:80px}\n.map-controls button{background:#263c5a;color:white;border:1px solid #57708e;border-radius:6px;padding:6px 9px;cursor:pointer}\n.map-bottom .wave-key{margin:0;width:min(420px,calc(100% - 22px));max-width:420px;box-sizing:border-box;background:rgba(16,24,39,.94);color:#edf2f7;border:1px solid #56708c;padding:10px 13px;pointer-events:auto}\n.wave-key-title{font-size:12px}.key-hint{font-size:10px;font-weight:400;color:#b9c8db}\n.wave-key-bar{height:14px;margin-top:6px;border-radius:4px;background:linear-gradient(90deg,#0638ff 0%,#0286b7 20%,#44a75d 40%,#c4cf52 58%,#ffbd36 77%,#ff3900 100%)}.wave-key-ticks{font-size:11px;margin-top:5px}.wave-key-title{font-weight:700}\n#status{margin-top:5px}\n@media(max-width:650px){.map-bottom{bottom:18px;left:7px;right:7px}.map-controls{gap:5px 8px;padding:7px;font-size:11px}.map-controls input[type=range]{width:65px}#map{min-height:570px}}\n/* Dashboard iframe: prevent nested scrollbars. */\nhtml,body{height:100%;overflow:hidden}\n.wrap{box-sizing:border-box;height:100vh;max-width:none;padding:12px 16px;display:flex;flex-direction:column;min-height:0}\n.filter-row{flex:0 0 auto}\n#map{flex:1 1 auto;height:auto!important;min-height:0!important;margin-top:10px}\n#status{flex:0 0 auto;padding:4px 0;margin:0}\n@media(max-width:650px){#map{min-height:0!important}}\n</style></head><body><div class="wrap"><div class="filter-row"><div class="map-controls"><label><input id="waves" type="checkbox" checked> Waves</label><label>Opacity <input id="opacity" type="range" min="0" max="100" value="68"> <span id="opacityLabel">68%</span></label><label><input id="smooth" type="checkbox" checked> Smooth</label><label>Blur <input id="blur" type="range" min="0" max="24" value="12"> <span id="blurLabel">12px</span></label><label><input id="labels" type="checkbox" checked> Tower IDs</label><button id="fit">Fit towers</button></div></div><div id="map"><div class="map-bottom"><section class="wave-key" aria-label="Wave height color scale"><div class="wave-key-title">Hs (m)</div><div class="wave-key-bar" aria-hidden="true"></div><div class="wave-key-ticks"><span>0.5</span><span>1.5</span><span>2.5</span><span>3.5</span><span>4.5</span><span>5.5</span><span>6.5</span></div><div class="wave-key-note">Approximate color guide · refer to Copernicus for exact values</div></section></div></div><div id="status">Initializing map…</div></div><script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script><script>\nconst towers = [{"id": "D01", "lat": 24.66391667000001, "lon": 120.7168333}, {"id": "D02", "lat": 24.65530555999999, "lon": 120.7318056}, {"id": "D03", "lat": 24.64747221999998, "lon": 120.7228333}, {"id": "D04", "lat": 24.64522221999999, "lon": 120.7052222}, {"id": "D05", "lat": 24.63769444, "lon": 120.7075278}, {"id": "D06", "lat": 24.62544444000002, "lon": 120.6911944}, {"id": "D07", "lat": 24.67158333000002, "lon": 120.7114167}, {"id": "D08", "lat": 24.65661111000002, "lon": 120.7069722}, {"id": "D09", "lat": 24.65463888999999, "lon": 120.6974167}, {"id": "D10", "lat": 24.64388889000002, "lon": 120.6948889}, {"id": "D11", "lat": 24.63536110999998, "lon": 120.6955278}, {"id": "D12", "lat": 24.63119443999999, "lon": 120.6858056}, {"id": "C01", "lat": 24.69377778000002, "lon": 120.7689167}, {"id": "C02", "lat": 24.70077778000003, "lon": 120.7594167}, {"id": "C03", "lat": 24.71761111000002, "lon": 120.7624722}, {"id": "C04", "lat": 24.71572222000001, "lon": 120.7511667}, {"id": "C05", "lat": 24.70413888999999, "lon": 120.7493333}, {"id": "C06", "lat": 24.70663889000001, "lon": 120.7420278}, {"id": "C07", "lat": 24.68630555999999, "lon": 120.7566389}, {"id": "C08", "lat": 24.67605556000003, "lon": 120.7424722}, {"id": "C09", "lat": 24.68166666999998, "lon": 120.7372778}, {"id": "C10", "lat": 24.69175000000003, "lon": 120.7472778}, {"id": "C11", "lat": 24.69522221999999, "lon": 120.7392222}, {"id": "B01", "lat": 24.70952777999999, "lon": 120.7708611}, {"id": "B02", "lat": 24.72827777999997, "lon": 120.7743056}, {"id": "B03", "lat": 24.75319444000003, "lon": 120.7887778}, {"id": "B04", "lat": 24.76647222, "lon": 120.7979722}, {"id": "B05", "lat": 24.77286111, "lon": 120.81475}, {"id": "B06", "lat": 24.77777778000002, "lon": 120.8079722}, {"id": "B07", "lat": 24.72738889, "lon": 120.7591111}, {"id": "B08", "lat": 24.73808333000001, "lon": 120.7606944}, {"id": "B09", "lat": 24.74394443999999, "lon": 120.7711944}, {"id": "B10", "lat": 24.76277778, "lon": 120.7811667}, {"id": "B11", "lat": 24.77411110999999, "lon": 120.7920278}, {"id": "B12", "lat": 24.78300000000001, "lon": 120.8006667}, {"id": "A01", "lat": 24.70427778000001, "lon": 120.7805}, {"id": "A02", "lat": 24.72011111, "lon": 120.7934167}, {"id": "A03", "lat": 24.72658333000002, "lon": 120.8059444}, {"id": "A04", "lat": 24.73397221999999, "lon": 120.7993056}, {"id": "A05", "lat": 24.74666667000001, "lon": 120.8159722}, {"id": "A06", "lat": 24.75655556, "lon": 120.8277778}, {"id": "A07", "lat": 24.71897222000001, "lon": 120.7725278}, {"id": "A08", "lat": 24.72305556000002, "lon": 120.7840556}, {"id": "A09", "lat": 24.73747222000001, "lon": 120.7859722}, {"id": "A10", "lat": 24.74416666999999, "lon": 120.79475}, {"id": "A11", "lat": 24.75399999999998, "lon": 120.8025}, {"id": "A12", "lat": 24.75513889, "lon": 120.8121944}];\nL.DomEvent.disableClickPropagation(document.querySelector(\'.map-bottom\'));L.DomEvent.disableScrollPropagation(document.querySelector(\'.map-bottom\'));\nconst map=L.map(\'map\',{preferCanvas:true}).setView([24.70843,120.75996],11);\nL.tileLayer(\'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png\',{maxZoom:19,attribution:\'© OpenStreetMap contributors\'}).addTo(map);\nconst group=L.featureGroup();const labels=[];\nfor(const t of towers){const marker=L.circleMarker([t.lat,t.lon],{radius:5,color:\'#142a4d\',weight:1.5,fillColor:\'#ffca36\',fillOpacity:1}).bindPopup(\'<b>\'+t.id+\'</b><br>Latitude: \'+t.lat.toFixed(8)+\'<br>Longitude: \'+t.lon.toFixed(8));marker.addTo(group);const label=L.marker([t.lat,t.lon],{icon:L.divIcon({className:\'\',html:\'<span class="tower-label">\'+t.id+\'</span>\',iconSize:[45,18],iconAnchor:[-8,9]}),interactive:false}).addTo(group);labels.push(label)}\ngroup.addTo(map);map.fitBounds(group.getBounds().pad(.12));\n// Copernicus Marine global wave forecast: dynamic WMTS tiles in Web Mercator.\n// Dataset IDs and supported matrix sets can change; validate against GetCapabilities.\nconst layerId=\'GLOBAL_ANALYSISFORECAST_WAV_001_027/cmems_mod_glo_wav_anfc_0.083deg_PT3H-i_202411/VHM0\';\nconst tileUrl=\'https://wmts.marine.copernicus.eu/teroWmts/?service=WMTS&version=1.0.0&request=GetTile&format=image%2Fpng&layer=\'+encodeURIComponent(layerId)+\'&tilematrixset=EPSG:3857&tilematrix={z}&tilecol={x}&tilerow={y}&STYLE=cmap:rainbow,range=0/5\';\nmap.createPane(\'wavePane\');map.getPane(\'wavePane\').style.zIndex=\'350\';map.getPane(\'wavePane\').style.pointerEvents=\'none\';\nconst waves=L.tileLayer(tileUrl,{pane:\'wavePane\',opacity:.68,maxNativeZoom:10,maxZoom:19,tileSize:256,attribution:\'Copernicus Marine Service\',errorTileUrl:\'\'});\nlet good=0,bad=0;const status=document.getElementById(\'status\');\nfunction updateStatus(){status.textContent=\'KML: \'+towers.length+\' turbines loaded · Copernicus tiles: \'+good+\' loaded, \'+bad+\' failed\'+(bad&&!good?\' · Check WMTS layer ID, availability, or browser network\':\'\');}\nwaves.on(\'tileload\',()=>{good++;updateStatus()});waves.on(\'tileerror\',()=>{bad++;updateStatus()});\nwaves.addTo(map);group.bringToFront();updateStatus();\nfunction updateSmoothing(){const active=document.getElementById(\'smooth\').checked;const px=Number(document.getElementById(\'blur\').value);map.getPane(\'wavePane\').style.filter=active?\'blur(\'+px+\'px)\':\'none\';document.getElementById(\'blurLabel\').textContent=px+\'px\';}\ndocument.getElementById(\'smooth\').onchange=updateSmoothing;document.getElementById(\'blur\').oninput=updateSmoothing;updateSmoothing();\ndocument.getElementById(\'waves\').onchange=e=>{if(e.target.checked){waves.addTo(map);group.bringToFront()}else map.removeLayer(waves)};\ndocument.getElementById(\'opacity\').oninput=e=>{waves.setOpacity(Number(e.target.value)/100);document.getElementById(\'opacityLabel\').textContent=e.target.value+\'%\'};\ndocument.getElementById(\'labels\').onchange=e=>{labels.forEach(l=>e.target.checked?group.addLayer(l):group.removeLayer(l))};\nfunction fitF2Towers(){\n  map.invalidateSize({pan:false});\n  if(group.getLayers().length){map.fitBounds(group.getBounds().pad(.15),{animate:false});}\n}\ndocument.getElementById(\'fit\').onclick=fitF2Towers;\n// The iframe is initially inside a hidden Dash tab. Re-fit once visible.\nconst mapResizeObserver=new ResizeObserver(entries=>{\n  const rect=entries[0].contentRect;\n  if(rect.width>0 && rect.height>0){requestAnimationFrame(fitF2Towers);}\n});\nmapResizeObserver.observe(document.getElementById(\'map\'));\nwindow.addEventListener(\'message\',event=>{\n  if(event.origin===window.location.origin && event.data===\'f2-copernicus-fit\'){\n    requestAnimationFrame(fitF2Towers);\n  }\n});\n\n</script></body></html>'

#%%Color map
color_map = {
    "Inspection": "#1f77b4",         #藍
    "Data Processing": "#2ca02c",    #綠
    "WOW(offshore)": "#d62728",      #紅
    "Day off": "#000000",            #黑
    "WOW(onshore)": "#bdbdbd",       #灰
    "Delay": "#ff7f0e",              #橘
}
category_icon = {
    "Inspection": "🟦",
    "Data Processing": "🟩",
    "WOW(offshore)": "🟥",
    "WOW(onshore)": "⬜",
    "Delay": "🟧",
    "Day off": "⬛",
    "Mob": "⚓",
    "Demob": "🏁",
    "Standby": "⏸️"
}
#%%Project Events
# project_events = pd.DataFrame([
#     {
#         "Date": pd.Timestamp("2026-04-06"),
#         "MarkerType": "PROJECT_EVENT",
#         "EventType": "MOB",
#         "Task": "⚓ MOB",
#         "Title": "Mobilization",
#         "Detail": "BeeX team mobilized for F2 offshore wind farm underwater inspection.",
#         "Remark": "Project start / preparation phase"
#     },
#     {
#         "Date": pd.Timestamp("2026-06-19"),
#         "MarkerType": "PROJECT_EVENT",
#         "EventType": "DEMOB",
#         "Task": "🏁 DEMOB",
#         "Title": "Demobilization",
#         "Detail": "BeeX team demobilized due to F2 crane issue.",
#         "Remark": "Operation suspended / team off-jacket"
#     },
#     {
#         "Date": pd.Timestamp("2026-06-20"),
#         "MarkerType": "PROJECT_EVENT",
#         "EventType": "STANDBY",
#         "Task": "⏸️ STANDBY",
#         "Title": "Standby Period",
#         "Detail": "Project remains on standby pending further instruction from F2 side.",
#         "Remark": "Standby period from 2026-06-20 to 2026-07-31"
#     },
#     {
#         "Date": pd.Timestamp("2026-08-01"),
#         "MarkerType": "PROJECT_EVENT",
#         "EventType": "REMOB",
#         "Task": "🔄 REMOB",
#         "Title": "Remobilization",
#         "Detail": "Expected remobilization for continued offshore inspection works.",
#         "Remark": "Tentative remobilization date"
#     }
# ])
#%%Task / Category list
# task_list = list(df["Task"].unique())
always_show_categories = [
    "WOW(onshore)",
    "Delay",
    "Day off",
    "Data Processing",
]

task_df = df[
    (~df["Category"].isin(always_show_categories))
    &
    (~df["Task"].isin([
        "Mobilization",
        "Demobilization",
        "Standby"
    ]))
].copy()
task_list = sorted(task_df["Task"].dropna().unique())

# Personnel attendance: match any role (OR across selected names).
PERSONNEL_COLUMNS = ["Supervisor", "Pilot", "Tether Manager", "Assistant"]
def split_personnel(value):
    if pd.isna(value):
        return []
    return [name.strip() for name in re.split(r"[,/、，;；]+", str(value))
            if name.strip() and name.strip().casefold() not in ("nan", "none", "-")]

def personnel_names(source_df):
    names = set()
    for column in PERSONNEL_COLUMNS:
        if column in source_df.columns:
            for value in source_df[column].dropna().astype(str):
                names.update(split_personnel(value))
    return sorted(names, key=lambda name: (name.casefold() == "beex teams", name.casefold()))

category_order = [
    "Inspection",
    "WOW(offshore)",
    "Data Processing",
    "WOW(onshore)",
    "Delay",
    "Day off",
]
cat_list = category_order
PROJECT_CODES = {2025: "S2507BEX50", 2026: "S2603BEX50"}
#%%Summary Table
projects = [
    {"name":"BeeX",
     "logo":"beex_logo.png",
     "start":datetime(2026,4,6).date()},
    {"name":"IOG",
     "logo":"iog_logo.png",
     "start":datetime(2026,4,18).date()}]
def build_summary_table():
    today = pd.Timestamp.now(tz="Asia/Taipei").date()
    rows = []
    for p in projects:
        days = (today - p["start"]).days + 1
        rows.append(
            html.Tr([
                html.Td(
                    html.Img(
                        src=get_asset_url(p["logo"]),
                        alt=p["name"],
                        title=p["name"],
                        style={"height":"25px", "maxWidth":"100%", "objectFit":"contain"}
                    )
                ),
                html.Td(str(p["start"])),
                html.Td(str(today)),
                html.Td(
                    str(days),
                    style={
                        "color":"red",
                        "fontSize":"20px",
                        "fontWeight":"bold"
                    }
                )
            ])
        )
    return html.Table(
        [
            html.Tr([
                html.Th("Company",style={"width":"25%"}),
                html.Th("Start",style={"width":"25%"}),
                html.Th("Today",style={"width":"25%"}),
                html.Th("Days",style={"width":"25%"})
            ])
        ] + rows,
        className="ocean-company-table",
        style={
            "width": "100%",                 # 表格寬度佔滿
            "tableLayout": "fixed",          # 固定欄寬，不依內容自動調整
            "textAlign": "center",           # 所有文字置中
            "border": "1px solid lightgray", # 表格外框為淺灰色 1px
            #"borderCollapse": "collapse",     # 合併相鄰儲存格邊框，避免雙線
            "borderRadius": "8px"
        }
    )
#%%KPI Dashboard
def calculate_kpi(source_df):
    today = pd.Timestamp.now(tz="Asia/Taipei").tz_localize(None).normalize()
    # KPI is independent of the selected historical year.
    beex_start = pd.Timestamp(projects[0]["start"])
    kpi_df = source_df.loc[
        source_df["Date"].dt.normalize().between(beex_start, today)
    ].copy()
    
    inspection_days = (
        kpi_df.loc[
            kpi_df["Category"] == "Inspection",
            "Date"
        ]
        .dt.normalize()
        .nunique()
    )
    
    data_days = (
        kpi_df.loc[
            kpi_df["Category"] == "Data Processing",
            "Date"
        ]
        .dt.normalize()
        .nunique()
    )
    
    wow_days = (
        kpi_df.loc[
            kpi_df["Category"].isin(
                ["WOW(offshore)", "WOW(onshore)"]
            ),
            "Date"
        ]
        .dt.normalize()
        .nunique()
    )
    
    delay_days = (
        kpi_df.loc[
            kpi_df["Category"] == "Delay",
            "Date"
        ]
        .dt.normalize()
        .nunique()
    )
    
    off_days = (
        kpi_df.loc[
            kpi_df["Category"] == "Day off",
            "Date"
        ]
        .dt.normalize()
        .nunique()
    )
    
    standby_days = (
        kpi_df.loc[
            kpi_df["Category"] == "Standby",
            "Date"
        ]
        .dt.normalize()
        .nunique()
    )
    
    # inspection_days = len(df[df["Category"]=="Inspection"]) - 1  ##6/1有兩隻塔
    # data_days = len(df[df["Category"]=="Data Processing"])
    # wow_days = len(df[df["Category"]=="WOW(offshore)"]) + len(df[df["Category"]=="WOW(onshore)"])
    # delay_days = len(df[df["Category"]=="Delay"])
    # off_days = len(df[df["Category"]=="Day off"])
    kpi_data = {
        "Inspection Days": inspection_days,
        "Data Processing Days": data_days,
        "WOW Days": wow_days,
        "Delay Days": delay_days,
        "Day Off": off_days,
        "Standby": standby_days}
    return kpi_data
kpi_data = measure_startup("Calculate KPI", lambda: calculate_kpi(df))
def build_card(title, value):
    ## 外框
    return html.Div([
            ##標題
            html.H2(value,style={
                            "margin":"0",
                            "fontSize":"20px",
                            "color":"#1f77b4"     #藍色
                            }),
            ##段落文字
            html.P(title,style={
                            "margin": "5px"
                            })],
            className="ocean-kpi-card",
            style={
                "border": "1px solid lightgray",   # 淺灰色 1px 邊框
                "padding": "10px",                 # 內距 15px（內容與邊框的距離）
                "width": "15%",                    # （parent container）寬度的 20%
                "display": "inline-block",         # 與其他元件並排顯示
                "margin": "5px",                  # 外距 10px（與其他元件的距離）
                "textAlign": "center",             # 文字置中
                "boxSizing": "border-box",
                "borderRadius": "8px"
                })
#%%upcoming_tasks
def build_upcoming_tasks(days=7, source_df=None):
    if source_df is None:
        source_df = df
    today = pd.Timestamp.now(tz="Asia/Taipei").tz_localize(None).normalize()
    end_day = today + pd.Timedelta(days=days)

    upcoming = source_df[
        (source_df["Start"] >= today) &
        (source_df["Start"] < end_day)
    ].copy()

    if upcoming.empty:
        return html.Div(
            "No upcoming tasks.",
            style={
                "fontSize": "12px",
                "color": "#6b7280",
                "padding": "10px"
            }
        )

    upcoming = upcoming.sort_values(["Start", "Task"])

    rows = []

    for _, r in upcoming.iterrows():
        rows.append(
            html.Tr([
                html.Td(r["Start"].strftime("%Y-%m-%d")),
                html.Td(r["Task"]),
                html.Td(html.Span(
                    f"{category_icon.get(r['Category'],'')} {r['Category']}")),
                html.Td("" if pd.isna(r.get("Supervisor")) else r["Supervisor"]),
                html.Td("" if pd.isna(r.get("Pilot")) else r["Pilot"]),
                html.Td("" if pd.isna(r.get("Tether Manager")) else r["Tether Manager"]),
                html.Td("" if pd.isna(r.get("Assistant")) else r["Assistant"]),
                html.Td("" if pd.isna(r.get("Remark")) else r["Remark"]),
            ])
        )

    return html.Div([
        html.H4(
            "Upcoming Tasks",
            style={
                "margin": "0 0 8px 0",
                "fontSize": "14px"
            }
        ),

        html.Table(
            [
                html.Tr([
                    html.Th("Date"),
                    html.Th("Task"),
                    html.Th("Category"),
                    html.Th("Supervisor"),
                    html.Th("Pilot"),
                    html.Th("Tether Manager"),
                    html.Th("Assistant"),
                    html.Th("Note")
                ])
            ] + rows,
            className="ocean-upcoming-table",
            style={
                "width": "100%",
                "tableLayout": "auto",
                "textAlign": "center",
                "borderCollapse": "collapse",
                "fontSize": "12px"
            }
        )
    ], className="ocean-upcoming", style={
        "backgroundColor": "#f9f9f9",
        "padding": "10px",
        "border": "1px solid lightgray",
        "borderRadius": "8px",
        "marginTop": "10px"
    })
#%%App
server = Flask(__name__)
server.before_request(_track)

@server.route(ANALYTICS_PATH)
def analytics_page():
    if not _check_auth():
        return _auth_required()
    return Response(_analytics(), mimetype="text/html")

@server.route("/admin/data")
def analytics_data():
    if not _check_auth():
        return _auth_required()
    response = jsonify(_analytics(fragment=True))
    response.headers["Cache-Control"] = "no-store"
    return response

@server.route("/f2-marine-wave-map")
def f2_marine_wave_map():
    return Response(F2_LEAFLET_MAP_HTML, mimetype="text/html")

app = Dash(__name__, server=server)

# Initial browser-only loading overlay. Progress counts observable milestones,
# not elapsed time; the Tux mascot follows the completed portion of the track.
# The overlay is created once per full page load, not on filter callbacks.
app.index_string = """<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            #initial-dashboard-loading {
                position: fixed; inset: 0; z-index: 99999;
                display: flex; align-items: center; justify-content: center;
                background: #f5f8fc; color: #193451;
                font-family: 'Segoe UI', Arial, sans-serif;
                transition: opacity .3s ease, visibility .3s ease;
            }
            #initial-dashboard-loading.ready { opacity: 0; visibility: hidden; pointer-events: none; }
            .initial-loading-content { text-align: center; padding: 24px; width: min(420px, 90vw); }
            .initial-loading-spinner {
                width: 38px; height: 38px; margin: 0 auto 20px;
                border: 3px solid #dbe5f0; border-top-color: #287bb5;
                border-radius: 50%; animation: initial-dashboard-spin .8s linear infinite;
            }
            .initial-loading-title { font-size: 18px; font-weight: 600; }
            .initial-loading-subtitle { margin-top: 8px; font-size: 13px; color: #64748b; }
            .initial-loading-status { display: flex; justify-content: space-between; gap: 12px;
                margin-top: 27px; color: #475569; font-size: 12px; text-align: left; }
            .initial-loading-status span:last-child { color: #287bb5; white-space: nowrap; font-weight: 600; }
            .initial-loading-track { position: relative; margin-top: 38px; height: 10px;
                background: #dce5ee; border-radius: 999px; }
            .initial-loading-fill { width: 0%; height: 100%; background: #287bb5;
                border-radius: 999px; transition: width .35s ease; }
            .initial-loading-tux { position: absolute; bottom: 6px; left: 0%;
                width: 68px; height: 94px; transform: translateX(-50%);
                transition: left .35s ease; }
            .initial-loading-tux .tux-sprite {
                display: block; width: 68px; height: 94px;
                background-image: url("data:image/png;base64,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");
                background-repeat: no-repeat; background-size: 544px 94px;
                animation: tux-walk-frames .75s steps(8) infinite;
                transform: scale(.64); transform-origin: 50% 100%;
            }
            .initial-loading-tux.still .tux-sprite { animation: none; }
            .initial-loading-footer { margin-top: 14px; color: #64748b; font-size: 12px; }
            @keyframes initial-dashboard-spin { to { transform: rotate(360deg); } }
            @keyframes tux-walk-frames { to { background-position: -544px 0; } }
            @media (prefers-reduced-motion: reduce) {
                .initial-loading-spinner, .initial-loading-tux .tux-sprite { animation: none; }
                #initial-dashboard-loading, .initial-loading-fill, .initial-loading-tux { transition: none; }
            }
        </style>
    </head>
    <body>
        <div id="initial-dashboard-loading" role="status" aria-live="polite">
            <div class="initial-loading-content">
                <div class="initial-loading-spinner" aria-hidden="true"></div>
                <div class="initial-loading-title">Loading Dashboard...</div>
                <div class="initial-loading-subtitle">Preparing F2 Offshore Dashboard</div>
                <div class="initial-loading-status"><span id="initial-loading-stage">Initializing dashboard...</span>
                    <span id="initial-loading-count">0 / 5</span></div>
                <div class="initial-loading-track" role="progressbar" aria-label="Dashboard initialization"
                     aria-valuemin="0" aria-valuemax="5" aria-valuenow="0">
                    <div class="initial-loading-fill" id="initial-loading-fill"></div>
                    <div class="initial-loading-tux" id="initial-loading-tux" aria-label="Linux Tux penguin">
                        <!-- Walking Tux frames from the user-provided sprite sheet. -->
                        <span class="tux-sprite" aria-hidden="true"></span>
                    </div>
                </div>
                <div class="initial-loading-footer">Tux is preparing your dashboard...</div>
            </div>
        </div>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
        <script>
        (function () {
            var overlay = document.getElementById('initial-dashboard-loading');
            if (!overlay) return;
            var finished = false, stage = 0, pendingFinish = false;
            var fill = document.getElementById('initial-loading-fill');
            var tux = document.getElementById('initial-loading-tux');
            var label = document.getElementById('initial-loading-stage');
            var count = document.getElementById('initial-loading-count');
            var track = overlay.querySelector('.initial-loading-track');
            var descriptions = [
                'Initializing dashboard...',
                'Dashboard interface ready',
                'Schedule data ready',
                'Summary and KPI cards ready',
                'Overall progress verified',
                'Dashboard Ready!'
            ];
            function advance(next) {
                if (next <= stage || finished) return;
                stage = next;
                fill.style.width = (stage * 20) + '%';
                tux.style.left = (stage * 20) + '%';
                label.textContent = descriptions[stage];
                count.textContent = stage + ' / 5';
                track.setAttribute('aria-valuenow', String(stage));
                if (stage === 5) tux.classList.add('still');
            }
            function finish() {
                if (finished) return;
                finished = true;
                overlay.classList.add('ready');
                window.setTimeout(function () { overlay.remove(); }, 350);
                observer.disconnect();
                window.clearInterval(poll);
                window.clearTimeout(watchdog);
            }
            function textOf(id) {
                var element = document.getElementById(id);
                return element ? (element.textContent || '').trim() : '';
            }
            function hasResolvedValue(id, placeholder) {
                var value = textOf(id);
                return value.length > 0 && value !== placeholder;
            }
            function checkDashboard() {
                if (finished || pendingFinish) return;
                // Each milestone is a prerequisite for the next. No timer-based progress.
                if (!document.getElementById('overview-page') ||
                    !document.getElementById('schedule-page')) return;
                advance(1);

                // The version store is populated only after the Neon timestamp callback.
                // Dash's dcc.Store data is React-managed, not a DOM attribute.
                // Use dependent callback outputs below as observable proof of data readiness.
                if (textOf('initial-schedule-check') !== 'ready') return;
                advance(2);

                var summary = document.querySelector('#overview-page .ocean-company-table');
                var cards = document.querySelectorAll('#kpi-container .ocean-kpi-card');
                if (!summary || !summary.querySelector('tr') || !cards.length) return;
                var cardsReady = Array.from(cards).every(function (card) {
                    var value = card.querySelector('h2');
                    return value && (value.textContent || '').trim() !== '';
                });
                if (!cardsReady || !document.getElementById('upcoming-container') ||
                    textOf('initial-kpi-check') !== 'ready') return;
                advance(3);

                // These start with em-dash placeholders and are filled by a separate callback.
                if (textOf('initial-progress-check') !== 'ready' ||
                    !hasResolvedValue('overall-progress-percent', '—') ||
                    !hasResolvedValue('overall-progress-count', '— / — Jackets') ||
                    !hasResolvedValue('project-completed-jackets', ' | Completed Jackets: —')) return;
                advance(4);

                var container = document.getElementById('gantt-chart');
                var graph = container && container.querySelector('.js-plotly-plot');
                if (!graph || !graph._fullLayout || !graph._fullData ||
                    !graph.querySelector('.main-svg, .gl-container')) return;
                pendingFinish = true;
                requestAnimationFrame(function () {
                    requestAnimationFrame(function () {
                        advance(5);
                        window.setTimeout(finish, 450);
                    });
                });
            }
            var observer = new MutationObserver(checkDashboard);
            observer.observe(document.body, {childList: true, subtree: true, characterData: true});
            var poll = window.setInterval(checkDashboard, 200);
            checkDashboard();
            // A failed callback must never be reported as 100% ready.
            // Keep the last verified stage visible and offer a manual retry.
            var watchdog = window.setTimeout(function () {
                if (finished || pendingFinish) return;
                tux.classList.add('still');
                label.textContent = 'Initialization incomplete: ' + ([
                    'Dashboard interface', 'Schedule data', 'KPI cards',
                    'Overall progress', 'Gantt Chart'
                ][Math.min(stage, 4)]);
                var subtitle = overlay.querySelector('.initial-loading-subtitle');
                if (subtitle) subtitle.textContent = 'A required component did not become ready. Check Render logs or retry.';
                var footer = overlay.querySelector('.initial-loading-footer');
                if (footer) {
                    footer.textContent = '';
                    var retry = document.createElement('button');
                    retry.textContent = 'Retry loading';
                    retry.style.cssText = 'margin-top:12px;padding:8px 16px;border:0;border-radius:6px;background:#287bb5;color:white;cursor:pointer';
                    retry.onclick = function () { window.location.reload(); };
                    footer.appendChild(retry);
                }
                observer.disconnect();
                window.clearInterval(poll);
            }, 30000);
        })();
        </script>
    </body>
</html>"""

app.layout = html.Div([
    # =========================================================
    # Schedule Data Refresh
    # =========================================================
    # Only change this store when Neon schedule data actually changes.
    dcc.Store(id="schedule-version-store"),
    html.Div(id="initial-schedule-check", style={"display": "none"}),
    html.Div(id="initial-kpi-check", style={"display": "none"}),
    html.Div(id="initial-progress-check", style={"display": "none"}),
    dcc.Interval(
        id="schedule-refresh-interval",
        interval=60 * 1000,
        n_intervals=0
    ),
    # =========================================================
    # HEADER (工程系統標題)
    # =========================================================
    html.Div([
        html.Div([
        html.H2(
            "🌊F2 Offshore Wind Farm Underwater Inspection",
            className="ocean-title",
            style={"marginBottom": "5px"}
        ),
    
        html.Div(
            f"Engineering Scheduling & Progress Tracking System | "
            f"v1.5.1 Beta | "
            f"Schedule Data Updated: {data_updated}",
            id="schedule-update-status",
            className="ocean-subtitle",
            style={
                "color": "#6b7280",
                "fontSize": "12px",
                "marginTop": "3px",
                "fontWeight": "400",
                "lineHeight": "1.5"
            }
        ),
    
    
        ], style={"flex": "4", "minWidth": 0, "textAlign": "center"}),
        html.Div([
            html.Div("Overall Progress", style={"fontSize": "12px", "color": "#cbd5e1"}),
            html.Div("—", id="overall-progress-percent", style={
                "fontSize": "26px", "fontWeight": "700", "color": "white", "lineHeight": "1.25"
            }),
            html.Div("— / — Jackets", id="overall-progress-count", style={
                "fontSize": "12px", "color": "#cbd5e1"
            })
        ], style={
            "flex": "1", "minWidth": "135px", "borderLeft": "1px solid #52647a",
            "display": "flex", "flexDirection": "column", "alignItems": "center",
            "justifyContent": "center", "textAlign": "center", "paddingLeft": "12px"
        })
    ], className="ocean-header", style={
        "display": "flex", "alignItems": "center", "gap": "12px",
        "marginBottom": "10px"
    }),
    # =========================================================
    # WINDY STATUS REFRESH
    # =========================================================
    dcc.Interval(
        id="windy-status-interval",
        interval=5 * 60 * 1000,
        n_intervals=0
    ),
    # Compact navigation: keep all pages mounted for existing callbacks.
    dcc.Tabs(
        id="main-navigation", value="overview",
        children=[
            dcc.Tab(label=label, value=value,
                    style={"padding": "7px 17px", "fontSize": "13px",
                           "fontWeight": "500", "backgroundColor": "#203249",
                           "color": "#b9cce5", "border": "1px solid #344b65",
                           "borderBottom": "2px solid transparent"},
                    selected_style={"padding": "7px 17px", "fontSize": "13px",
                                    "fontWeight": "600", "backgroundColor": "#29435f",
                                    "color": "#ffffff", "border": "1px solid #426b94",
                                    "borderBottom": "2px solid #69b8ff"})
            for label, value in [("Overview", "overview"),
                                 ("Schedule", "schedule"),
                                 ("Weather", "weather"),
                                 ("Resources", "resources")]
        ],
        className="ocean-navigation",
        style={"marginBottom": "10px", "width": "fit-content", "maxWidth": "100%"}
    ),
    # =========================================================
    # SUMMARY TABLE + KPI (控制面板區)
    # =========================================================
    html.Div([
        # 2026 project scope label (display only)
        html.Div([
            html.Span("2026 Project Overview — S2603BEX50", style={"fontWeight": "600", "color": "#ffffff"}),
            html.Span(" | Completed Jackets: —", id="project-completed-jackets", style={"fontWeight": "600", "color": "#ffffff"})
        ], style={"display": "flex", "justifyContent": "flex-start", "alignItems": "center", "marginBottom": "8px"}),
        # Summary Table
        build_summary_table(),
        # KPI Cards
        html.Div(
            [
                build_card(title, value)
                for title, value in kpi_data.items()
            ],
            id="kpi-container",
            className="ocean-kpi-grid",
            style={
                "display": "flex",
                "justifyContent": "center",
                "flexWrap": "wrap",
                "gap": "10px",
                "marginTop": "10px"
            }
        ),
        # Upcoming Tasks
        html.Div(
        id="upcoming-container",
        children=build_upcoming_tasks(days=7)
        )
    ], id="overview-page", className="ocean-summary", style={
        "backgroundColor": "#101D2D",
        "padding": "10px",
        "borderRadius": "8px",
        "marginBottom": "10px"
    }),
    html.Div([
        # Open-Meteo forecast status
        html.Div(id="open-meteo-update-status", className="ocean-windy-status",
            children="Open-Meteo Forecast | Checking... | Forecast Data – Reference Only",
            style={"color":"#6b7280","fontSize":"12px","marginTop":"3px","lineHeight":"1.5"}),
        # Windy forecast status
        html.Div(
            id="windy-update-status",
            className="ocean-windy-status",
            children="Windy Forecast | Checking... | Testing API – Not for Operational Use",
            style={
                "color": "#6b7280",
                "fontSize": "12px",
                "marginTop": "3px",
                "fontWeight": "400",
                "lineHeight": "1.5"
            }
        ),
    # =========================================================
    # TOOLBAR (工程 Filter 列)
    # =========================================================
    html.Div([
        html.Div([
            html.Label("Project", style={"fontWeight": "bold"}),
            dcc.Dropdown(
                id="project-filter",
                options=[{"label": "All Projects", "value": "all"}] + [
                    {"label": f"{year} | {PROJECT_CODES.get(year, str(year))}", "value": str(year)}
                    for year in sorted(df["Date"].dt.year.dropna().astype(int).unique(), reverse=True)
                ],
                value="all",
                clearable=False
            )
        ], style={"width": "24%", "display": "inline-block", "verticalAlign": "top"}),
        html.Div([
            html.Label("Personnel Filter", style={"fontWeight": "bold"}),
            dcc.Dropdown(
                id="personnel-filter",
                options=[{"label": name, "value": name} for name in personnel_names(df)],
                value=[],
                multi=True,
                placeholder="All Personnel"
            )
        ], style={"width": "24%", "display": "inline-block", "verticalAlign": "top"}),
        html.Div([
            html.Label("Task Filter", style={"fontWeight": "bold"}),
            dcc.Dropdown(
                task_list,
                value=list(task_list),
                multi=True,
                id="task-filter"
            )
        ], style={"width": "24%", "display": "inline-block", "verticalAlign": "top"}),
        html.Div([
            html.Label("Category Filter", style={"fontWeight": "bold"}),
            dcc.Dropdown(
                cat_list,
                value=list(cat_list),
                multi=True,
                id="cat-filter"
            )
        ], style={"width": "24%", "display": "inline-block", "verticalAlign": "top"}),
    ], className="ocean-toolbar", style={
        "marginBottom": "8px",
        "padding": "8px 12px",
        "backgroundColor": "#1d3047",
        "border": "1px solid #ddd",
        "borderRadius": "6px"
    }),    
    # =========================================================
    # GANTT VIEWER (主工程區)
    # =========================================================
    html.Div([
        # 左側
        html.Div([
            html.Div([
                dcc.Graph(
                    id="gantt-chart",
                    config={
                        "scrollZoom": True,
                        "displayModeBar": True,
                        "doubleClick": "reset",
                        "responsive": True
                    },
                    style={"width": "100%", "height": "100%"}
                )
            ], className="ocean-gantt-scroll")
        ], className="ocean-gantt-column",
           style={"minWidth": 0}),
        # 右側
        html.Div(
            id="detail-panel",
            children=[
                html.H4("Jump to Date",
                style={
                        "margin":"0px",
                        "padding":"0px"
                    }
                ),
                dcc.DatePickerSingle(
                    id="date-picker",
                    date=None,
                    display_format="YYYY-MM-DD",
                    placeholder="Select a date"
                ),
                html.Hr(style={"margin":"5px 0"}),
                #html.H3("Task Detail"),
                html.Div(
                    "Click a bar to view details",
                    id="detail-content"
                )
            ],
            className="ocean-detail-panel"
        )
    ], className="ocean-gantt-layout"),
    ], id="schedule-page"),
    # Weather source selector: exactly one map is visible at a time.
    html.Div([
        html.Div([
            html.Span("Map source", style={"fontWeight": "600"}),
            dcc.Dropdown(
                id="weather-map-source",
                options=[
                    {"label": "Ventusky", "value": "ventusky"},
                    {"label": "Windy", "value": "windy"},
                    {"label": "Copernicus Wave Forecast", "value": "copernicus"},
                ],
                value="ventusky",
                clearable=False,
                searchable=False,
                style={"width": "270px", "color": "#182333"},
            ),
        ], style={"display": "flex", "alignItems": "center",
                  "gap": "12px", "marginBottom": "14px", "flexWrap": "wrap"}),
        html.Div([
            html.Div([
                html.H3("Windy Interactive Map", style={"margin": "0"}),
                html.A("Open in Windy ↗",
                       href="https://www.windy.com/upload/69e9dfa66061bb27a7ae71cc",
                       target="_blank", rel="noopener noreferrer",
                       style={"color": "#9ed2ff", "fontSize": "14px", "textDecoration": "none"}),
            ], style={"display": "flex", "alignItems": "center", "gap": "16px",
                      "flexWrap": "wrap", "marginBottom": "12px"}),
            html.Iframe(
                src="https://embed.windy.com/embed2.html?lat=24.70843&lon=120.75996&detailLat=24.70843&detailLon=120.75996&width=900&height=600&zoom=9&level=surface&overlay=waves&product=ecmwf&menu=&message=true&marker=true&calendar=now&pressure=&type=map&location=coordinates&detail=&metricWind=m%2Fs&metricTemp=%C2%B0C&radarRange=-1",
                style={"width": "100%", "height": "min(75vh, 700px)", "minHeight": "480px",
                       "border": "0", "borderRadius": "8px"},
                title="Windy F2 Wind Farm Weather Map",
            ),
        ], id="weather-windy-view"),
        html.Div([
            html.H3("Copernicus Wave Forecast", style={"margin": "0 0 12px"}),
            html.Iframe(
                src="/f2-marine-wave-map",
                style={"width": "100%", "height": "min(85vh, 820px)", "minHeight": "560px",
                       "border": "0", "borderRadius": "8px"},
                title="Copernicus Marine F2 Wave Map",
            ),
        ], id="weather-copernicus-view", style={"display": "none"}),
        html.Div([
            html.Div([
                html.H3("Ventusky Interactive Map", style={"margin": "0"}),
                html.A("Open in Ventusky ↗",
                       href="https://www.ventusky.com/?p=24.70843;120.75996;9&l=wave",
                       target="_blank", rel="noopener noreferrer",
                       style={"color": "#9ed2ff", "fontSize": "14px", "textDecoration": "none"}),
            ], style={"display": "flex", "alignItems": "center", "gap": "16px",
                      "flexWrap": "wrap", "marginBottom": "12px"}),
            # The iframe is mounted only when Weather > Ventusky is visible.
            # Initializing it inside display:none can give Ventusky a zero-size canvas.
            html.Div(id="ventusky-iframe-container"),
        ], id="weather-ventusky-view", style={"display": "none"}),
    ], id="weather-page", style={"display": "none", "padding": "16px",
        "backgroundColor": "#203249", "borderRadius": "8px"}),
    # Dedicated Resources page: SharePoint shortcuts.
    html.Div([
        html.H3("Project Resources", style={"marginTop": "0", "marginBottom": "12px"}),
        html.A('Duty Roster', href='https://iovtec.sharepoint.com/sites/SV-SurveyDept2/Shared%20Documents/Forms/AllItems.aspx?id=%2Fsites%2FSV%2DSurveyDept2%2FShared%20Documents%2F%E6%8E%A2%E6%B8%AC%E9%83%A8%2D%E7%8F%BE%E5%A0%B4%E7%87%9F%E9%81%8B%E7%B5%84%2F%E7%8F%AD%E8%A1%A8&p=true&ga=1', target="_blank",
               rel="noopener noreferrer",
               style={"display": "block", "padding": "8px 0", "color": "#9ed2ff"}),
        html.A('DPR', href='https://iovtec-my.sharepoint.com/:f:/g/personal/ali_chang_iovtec_com_tw/IgDMHGM8MshVQqUP_kgzjh4LAfee-_a7WUBFEz2pLI2bgXE?e=E5T8b7', target="_blank",
               rel="noopener noreferrer",
               style={"display": "block", "padding": "8px 0", "color": "#9ed2ff"}),
        html.A('Survey Log', href='https://iovtec.sharepoint.com/sites/SV-SurveyDept2/Shared%20Documents/Forms/AllItems.aspx?id=%2Fsites%2FSV%2DSurveyDept2%2FShared%20Documents%2F%E6%8E%A2%E6%B8%AC%E9%83%A8%2D%E7%8F%BE%E5%A0%B4%E7%87%9F%E9%81%8B%E7%B5%84%2FSurvey%20Log&viewid=a9aaccb6%2Dfaa4%2D446d%2Db6e3%2Dbeafdbfc933e', target="_blank",
               rel="noopener noreferrer",
               style={"display": "block", "padding": "8px 0", "color": "#9ed2ff"}),
        html.A('Photo Archive', href='https://iovtec.sharepoint.com/sites/IOGWindProject/Shared%20Documents/Forms/AllItems.aspx?id=%2Fsites%2FIOGWindProject%2FShared%20Documents%2F01%5FOperations%2F04%5FOngoing%20Projects%2F4372%5F%20Manpower%20for%20BeeX%2F2%2E%20Project%20Documentation%2F%E4%BD%9C%E6%A5%AD%E7%85%A7%E7%89%87&p=true&ct=1782188226845&or=Teams%2DHL&ga=1&LOF=1', target="_blank",
               rel="noopener noreferrer",
               style={"display": "block", "padding": "8px 0", "color": "#9ed2ff"}),
    ], id="resources-page", style={"display": "none", "padding": "16px",
        "backgroundColor": "#203249", "borderRadius": "8px"})
], className="ocean-app")

@app.callback(
    Output("weather-windy-view", "style"),
    Output("weather-copernicus-view", "style"),
    Output("weather-ventusky-view", "style"),
    Output("ventusky-iframe-container", "children"),
    Input("weather-map-source", "value"),
    Input("main-navigation", "value"),
)
def switch_weather_map(source, page):
    hidden = {"display": "none"}
    # Match the standalone HTML test: create a real, sized iframe only
    # while the Ventusky map is selected and the Weather page is visible.
    ventusky_visible = source == "ventusky" and page == "weather"
    ventusky_iframe = html.Iframe(
        src="https://embed.ventusky.com/?p=24.70843;120.75996;9&l=wave&pin=24.663917;120.716833;dot;D01;24.655306;120.731806;dot;D02;24.647472;120.722833;dot;D03;24.645222;120.705222;dot;D04;24.637694;120.707528;dot;D05;24.625444;120.691194;dot;D06;24.671583;120.711417;dot;D07;24.656611;120.706972;dot;D08;24.654639;120.697417;dot;D09;24.643889;120.694889;dot;D10;24.635361;120.695528;dot;D11;24.631194;120.685806;dot;D12;24.693778;120.768917;dot;C01;24.700778;120.759417;dot;C02;24.717611;120.762472;dot;C03;24.715722;120.751167;dot;C04;24.704139;120.749333;dot;C05;24.706639;120.742028;dot;C06;24.686306;120.756639;dot;C07;24.676056;120.742472;dot;C08;24.681667;120.737278;dot;C09;24.691750;120.747278;dot;C10;24.695222;120.739222;dot;C11;24.709528;120.770861;dot;B01;24.728278;120.774306;dot;B02;24.753194;120.788778;dot;B03;24.766472;120.797972;dot;B04;24.772861;120.814750;dot;B05;24.777778;120.807972;dot;B06;24.727389;120.759111;dot;B07;24.738083;120.760694;dot;B08;24.743944;120.771194;dot;B09;24.762778;120.781167;dot;B10;24.774111;120.792028;dot;B11;24.783000;120.800667;dot;B12;24.704278;120.780500;dot;A01;24.720111;120.793417;dot;A02;24.726583;120.805944;dot;A03;24.733972;120.799306;dot;A04;24.746667;120.815972;dot;A05;24.756556;120.827778;dot;A06;24.718972;120.772528;dot;A07;24.723056;120.784056;dot;A08;24.737472;120.785972;dot;A09;24.744167;120.794750;dot;A10;24.754000;120.802500;dot;A11;24.755139;120.812194;dot;A12",
        style={"display": "block", "width": "100%", "height": "70vh",
               "minHeight": "480px", "border": "0", "borderRadius": "8px",
               "backgroundColor": "#444"},
        title="Ventusky F2 Weather Map",
    ) if ventusky_visible else None
    return (
        {} if source == "windy" else hidden,
        {} if source == "copernicus" else hidden,
        {} if source == "ventusky" else hidden,
        ventusky_iframe,
    )


@app.callback(
    Output("overview-page", "style"),
    Output("schedule-page", "style"),
    Output("weather-page", "style"),
    Output("resources-page", "style"),
    Input("main-navigation", "value")
)
def switch_main_page(page):
    hidden = {"display": "none"}
    overview = {"backgroundColor": "#101D2D", "padding": "10px",
                "borderRadius": "8px", "marginBottom": "10px"}
    schedule = {}
    weather = {"padding": "16px", "backgroundColor": "#203249", "borderRadius": "8px"}
    resources = {"padding": "16px", "backgroundColor": "#203249",
                 "borderRadius": "8px"}
    return (overview if page == "overview" else hidden,
            schedule if page == "schedule" else hidden,
            weather if page == "weather" else hidden,
            resources if page == "resources" else hidden)

# Cached chart inputs: project/filter changes should not repeat preprocessing or SQL reads.
@lru_cache(maxsize=4)
def prepared_chart_data(schedule_version):
    source, _, _ = load_schedule_cached(schedule_version)
    prepared = source.copy()
    if "Lane" in prepared.columns:
        non_offshore = prepared["Lane"].astype(str).str.strip().eq("Non-Offshore")
    else:
        non_offshore = pd.Series(False, index=prepared.index)
    task_codes = prepared["Task"].fillna("").astype(str).str.extract(
        r"(?i)(?<![A-Za-z0-9])([ABCD]\s*[-_]?\s*\d{1,2})(?!\d)", expand=False
    )
    lane_codes = prepared["Lane"].fillna("").astype(str).str.extract(
        r"(?i)(?<![A-Za-z0-9])([ABCD]\s*[-_]?\s*\d{1,2})(?!\d)", expand=False
    )
    codes = task_codes.fillna(lane_codes).str.replace(r"[^A-Za-z0-9]", "", regex=True).str.upper()
    formatted = codes.str.extract(r"^([ABCD])(\d{1,2})$", expand=True)
    canonical = formatted[0] + formatted[1].str.zfill(2)
    prepared["_chart_lane"] = prepared["Lane"].astype(str).where(canonical.isna(), canonical)
    prepared.loc[non_offshore, "_chart_lane"] = "Non-Offshore"
    return prepared

@lru_cache(maxsize=1)
def cached_master_towers():
    return load_tower_master()

@lru_cache(maxsize=3)
def cached_chart_wave(time_bucket):
    # Wave forecast is refreshed independently; 5-minute TTL prevents
    # repeated PostgreSQL reads while rapidly switching project filters.
    return load_latest_wave_forecast()

@lru_cache(maxsize=3)
def cached_chart_open_meteo(bucket):
    return load_open_meteo_forecast()

#%%Callback (dynamic update)
@app.callback(
    Output("schedule-version-store", "data"),
    Input("schedule-refresh-interval", "n_intervals"),
    State("schedule-version-store", "data")
)
def refresh_schedule_version(n_intervals, previous_version):
    # Poll the lightweight timestamp; avoid sending a new figure when unchanged.
    version = get_schedule_updated_at()
    current_version = None if version is None else pd.Timestamp(version).isoformat()
    if current_version == previous_version:
        return no_update
    return current_version

@app.callback(
    Output("gantt-chart", "figure"),
    Input("task-filter", "value"),
    Input("cat-filter", "value"),
    Input("personnel-filter", "value"),
    Input("project-filter", "value"),
    Input("date-picker", "date"),
    Input("schedule-version-store", "data")
)

def update_chart(selected_tasks, selected_cats, selected_personnel, selected_project, selected_date, schedule_version):
    # Figure redraws only on filter/date changes or a new Neon data version.
    if schedule_version is None:
        return no_update
    
    # Load schedule data only when version changes
    chart_df = prepared_chart_data(schedule_version)
    if selected_project not in (None, "all"):
        chart_df = chart_df.loc[chart_df["Date"].dt.year == int(selected_project)].copy()
    # =========================================================
    # SAFE DEFAULTS
    # =========================================================
    selected_tasks = selected_tasks or task_list
    selected_cats = selected_cats or cat_list
    now = pd.Timestamp.now(tz="Asia/Taipei").tz_localize(None)
    # =========================================================
    # FILTER
    # =========================================================
    # filtered = df.loc[
    #     (df["Task"].isin(selected_tasks)) &
    #     (df["Category"].isin(selected_cats))
    # ].copy()
    selected_tasks = selected_tasks or task_list
    selected_cats = selected_cats or cat_list
    
    always_show_mask = chart_df["Category"].isin(always_show_categories)
    
    task_filter_mask = (
        chart_df["Task"].isin(selected_tasks) &
        ~always_show_mask
    )
    
    filtered = chart_df.loc[
        (task_filter_mask | always_show_mask) &
        (chart_df["Category"].isin(selected_cats))
    ].copy()
    # Personnel attendance filter: any selected person in any role is a match.
    if selected_personnel:
        role_columns = [c for c in PERSONNEL_COLUMNS if c in filtered.columns]
        if role_columns:
            selected_set = {str(name).strip().casefold() for name in selected_personnel}
            personnel_mask = filtered[role_columns].fillna("").astype(str).apply(
                lambda col: col.map(lambda value: any(
                    part.casefold() in selected_set for part in split_personnel(value)
                ))
            ).any(axis=1)
            filtered = filtered.loc[personnel_mask].copy()
        else:
            filtered = filtered.iloc[0:0].copy()

    # =========================================================
    # ORDERING — use cached vectorized lane parsing
    # =========================================================
    filtered["Lane"] = filtered["_chart_lane"]
    def lane_key(lane):
        match = re.fullmatch(r"([ABCD])(\d+)", str(lane))
        if match:
            return (0, match.group(1), int(match.group(2)))
        if lane == "Non-Offshore":
            return (2, "", 0)
        return (1, str(lane), 0)
    lane_order = sorted(filtered["Lane"].dropna().unique().tolist(), key=lane_key)
    # All Projects: show the complete F2 wind farm, including empty lanes.
    # When personnel is selected, keep the personnel-only view.
    # Year-specific views continue to show only towers with schedule records.
    if selected_project in (None, "all") and not selected_personnel:
        master_towers = cached_master_towers()
        lane_order = sorted(set(lane_order).union(master_towers), key=lane_key)
    filtered["_lane_sort"] = filtered["Lane"].map({lane: i for i, lane in enumerate(lane_order)})
    filtered = filtered.sort_values(["_lane_sort", "Start"], na_position="last")
    # lane_order = ["Project Events"] + lane_order
    
    filtered["Lane"] = pd.Categorical(
        filtered["Lane"],
        categories=lane_order,
        ordered=True
    )
    # =========================================================
    # LANE PROGRESS — same Inspection rule as overall/project completion KPI.
    # Intentionally independent of task/category/personnel display filters.
    inspection_map = latest_inspection_progress(chart_df, cached_master_towers())
    # A tower is touched if it appears in ANY schedule record, regardless of Category.
    # Use full history, independent of selected year and display filters.
    all_history = prepared_chart_data(schedule_version)
    touched_towers = set(all_history["_chart_lane"].dropna())
    ticktext = []
    for lane in lane_order:
        if lane in ("Project Events", "Non-Offshore"):
            ticktext.append(lane)
            continue
        progress = inspection_map.get(lane, 0)
        color = ("#94a3b8" if lane not in touched_towers else
                 "red" if progress < 50 else "orange" if progress < 80 else "green")
        progress_text = f"{progress:g}"
        ticktext.append(f"{lane} <span style='color:{color};'><b>({progress_text}%)</b></span>")
    # =========================================================
    # GANTT
    # =========================================================
    hover_cols = [
    "Date_str",
    "Category",
    "Task",
    "Cluster_hover",
    "Supervisor",
    "Pilot",
    "Tether Manager",
    "Assistant",
    "Remark",
    "Progress_hover",
    "Date"
    ]
    fig = px.timeline(
        filtered,
        x_start="Start",
        x_end="End",
        y="Lane",
        color="Category",
        color_discrete_map=color_map,
        category_orders={
        "Category": category_order,
        "Lane": lane_order
        },
        custom_data=hover_cols
    )
    # Project Event
    # fig.add_trace(
    #     go.Scatter(
    #         x=project_events["Date"],
    #         y=["Project Events"] * len(project_events),
    #         mode="markers+text",
    #         text=project_events["Task"],
    #         textposition="top center",
    #         marker=dict(
    #             size=14,
    #             color="#4b5563"
    #         ),
    #         customdata=project_events[
    #             ["MarkerType", "EventType", "Title", "Detail", "Remark"]
    #         ],
    #         hovertemplate=
    #             "<b>%{customdata[2]}</b><br>"
    #             "Type: %{customdata[1]}<br>"
    #             "Detail: %{customdata[3]}<br>"
    #             "Remark: %{customdata[4]}"
    #             "<extra></extra>",
    #         name="Project Events"
    #     )
    # )
    fig.update_traces(
        hovertemplate=
        "<b>%{customdata[2]}</b><br>"
        "Date: %{customdata[0]}<br>"
        "Category: %{customdata[1]}<br>"
        "Progress: %{customdata[9]}%<br>"
        "Supervisor: %{customdata[4]}<br>"
        "Pilot: %{customdata[5]}<br>"
        "Tether: %{customdata[6]}<br>"
        "Assistant: %{customdata[7]}<br>"
        "Note: %{customdata[8]}"
        "<extra></extra>"
    )
    
    # =========================================================
    # WINDY WAVE FORECAST
    # =========================================================
    # 從 PostgreSQL 讀取最新一次 Windy forecast snapshot。
    #
    # Wave forecast 與 Gantt 共用 x-axis（Date / Time），
    # 但使用獨立的 yaxis2 顯示 Hs (m)。
    #
    # 因此：
    #   yaxis  = Gantt lanes
    #   yaxis2 = Forecast Hs
    #   xaxis  = 共用時間軸
    # =========================================================
    
    wave_df = cached_chart_wave(int(time.time() // 300)).copy()
    
    if not wave_df.empty:
    
        # Hover 額外顯示：
        #   customdata[0] = Wave Period
        #   customdata[1] = Wave Direction
        wave_df["wave_arrow"] = (
            wave_df["wave_direction_deg"].apply(wave_arrow)
        )

        wave_customdata = np.stack(
            [
                wave_df["wave_period_s"],
                wave_df["wave_direction_deg"],
                wave_df["wave_arrow"]
            ],
            axis=-1
        )
    
        fig.add_trace(
            go.Scatter(
                x=wave_df["valid_time"],
                y=wave_df["wave_height_m"],
    
                # Wave 使用第二個 Y 軸
                yaxis="y2",
    
                mode="lines+markers",
    
                name="Windy Forecast Hs",
    
                line=dict(
                    color="#2563EB",
                    width=0.5,
                    dash="5px,2px"
                ),
                
                marker=dict(
                    color="#2563EB",
                    size=4
                ),
    
                customdata=wave_customdata,
    
                hovertemplate=(
                    "<b>Windy Wave Forecast</b><br>"
                    "Time: %{x|%Y-%m-%d %H:%M}<br>"
                    "Forecast Hs: %{y:.2f} m<br>"
                    "Period: %{customdata[0]:.1f} s<br>"
                    "Direction: %{customdata[1]:.0f}° %{customdata[2]}"
                    "<extra></extra>"
                ),
    
                # 不加入 Category legend，
                # 避免跟 Inspection / WOW / Delay 混在一起
                showlegend=True,
                legend="legend2"
            )
        )
    try:
        meteo = cached_chart_open_meteo(int(time.time() // 300)).copy()
    except Exception as exc:
        print(f"Open-Meteo chart unavailable: {exc}")
        meteo = pd.DataFrame()
    if not meteo.empty:
        meteo["wave_arrow"] = meteo["wave_direction_deg"].apply(
            lambda d: wave_arrow(d) if pd.notna(d) else "—")
        meta = np.stack([meteo["wave_period_s"], meteo["wave_direction_deg"], meteo["wave_arrow"]],axis=-1)
        fig.add_trace(go.Scatter(x=meteo["valid_time"],y=meteo["wave_height_m"],
            yaxis="y2",mode="lines+markers",name="Open-Meteo Forecast Hs",
            line=dict(color="#0d9488",width=0.5,dash="5px,2px"),marker=dict(color="#0d9488",size=4),
            customdata=meta,showlegend=True,legend="legend2",
            hovertemplate=("<b>Open-Meteo Wave Forecast</b><br>Time: %{x|%Y-%m-%d %H:%M}<br>"
                "Forecast Hs: %{y:.2f} m<br>Period: %{customdata[0]:.1f} s<br>"
                "Direction: %{customdata[1]:.0f}° %{customdata[2]}<extra></extra>")))
        # ---------------------------------------------------------
        # Wave Hs horizontal reference lines
        # 0–10 m (same thin black solid-line style)
        # ---------------------------------------------------------
        for hs_level in range(0, 11):
            fig.add_shape(
                type="line",
        
                # 橫跨整個時間軸
                xref="paper",
                x0=0,
                x1=1,
        
                # 使用 Wave 的 Y-axis
                yref="y2",
                y0=hs_level,
                y1=hs_level,
        
                # 黑色細實線
                line=dict(
                    color="black",
                    width=0.3,
                    dash="solid"
                ),
        
                # 放在資料後方，不擋 Wave trace
                layer="below"
            )

        # ---------------------------------------------------------
        # Wave operational limit: Hs = 1.5 m
        # ---------------------------------------------------------
        fig.add_shape(
            type="line",
        
            # 橫跨整個時間軸
            xref="paper",
            x0=0,
            x1=1,
        
            # 使用 Wave 的 Y-axis
            yref="y2",
            y0=1.5,
            y1=1.5,
        
            line=dict(
                color="red",
                width=1,
                dash="solid"
            )
        )
        
        # Limit label
        fig.add_annotation(
            xref="paper",
            x=0.01,
        
            yref="y2",
            y=1.5,
        
            text="Hs Limit 1.5 m",
        
            showarrow=False,
        
            xanchor="left",
            yanchor="bottom",
        
            font=dict(
                color="red",
                size=11
            )
        )
    # =========================================================
    # TODAY LINE
    # =========================================================
    fig.add_shape(
        type="line",
        x0=now,
        x1=now,
        y0=0,
        y1=1,
        yref="paper",
        line=dict(color="red", width=2, dash="dot")
    )
    fig.add_annotation(
        x=now,
        y=0.88,
        yref="paper",
        text=(
            f"<b>TODAY</b><br>"
            f"{now.strftime('%Y-%m-%d %H:%M')}"
        ),
        showarrow=False,
        font=dict(color="red", size=12),
        bgcolor="rgba(255,255,255,0.85)",
        bordercolor="red",
        borderwidth=1
    )
    # =========================================================
    # MILESTONES: MOB / DEMOB / REMOB
    # =========================================================
    milestones = [
        ("2026-04-06", "MOB", "#4b5563"),
        ("2026-06-19", "DEMOB", "#4b5563"),
        ("2026-09-12", "REMOB", "#4b5563"),]
    for date, label, color in milestones:
        date = (pd.Timestamp(date)+ pd.Timedelta(hours=12))
        fig.add_shape(
            type="line",
            x0=date,
            x1=date,
            y0=0,
            y1=0.84,
            yref="paper",
            line=dict(color=color,width=2,dash="dot")
        )
    
        fig.add_annotation(
            x=date,
            y=0.88,
            yref="paper",
            text=(f"<b>{label}</b><br>"f"{date.strftime('%Y-%m-%d')}"),
            showarrow=False,
            font=dict(color=color,size=12),
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor=color,
            borderwidth=1
        )
    # =========================================================
    # STANDBY BACKGROUND
    # =========================================================
    fig.add_shape(
        type="rect",
        x0=pd.Timestamp("2026-06-20"),
        x1=pd.Timestamp("2026-09-12"),
        y0=0,
        y1=0.84,
        xref="x",
        yref="paper",
        fillcolor="#696969",
        opacity=0.40,
        layer="below",
        line_width=0
    )
    
    fig.add_annotation(
        x=pd.Timestamp("2026-07-31"),
        y=0.4,
        yref="paper",
        text="<b>STANDBY</b>",
        showarrow=False,
        font=dict(
            color="gray",
            size=12
        ),
        bgcolor="rgba(255,255,255,0.85)",
        bordercolor="gray",
        borderwidth=1
    )
    # =========================================================
    # LAYOUT
    # =========================================================
    chart_height = min(
        max(650, len(lane_order) * 30),
        700
    )
    fig.update_layout(
        title="Engineering Operations Scheduling Gantt Chart",
        legend_title_text="",
    
        legend=dict(
            orientation="h",
            x=0.55,              # 最右邊
            xanchor="right",    # 右對齊
            y=1.08,             # 再往上移
            yanchor="top",
            font=dict(size=8)
        ),
        legend2=dict(
            orientation="h",
            x=1, xanchor="right",
            y=1.01, yanchor="bottom",
            font=dict(size=10),
            bgcolor="rgba(255,255,255,0.75)",
            borderwidth=0,
            itemsizing="constant"
        ),
        height=chart_height,
        margin=dict(
            l=120,  # Reserve room for Non-Offshore and progress labels
            r=20,
            t=100,             # 上方空間增加
            b=5
        ),
        dragmode="pan",
        # Phase 2: preserve the approved white engineering workspace.
        paper_bgcolor="#ffffff",
        plot_bgcolor="#e7eef7",
        font=dict(family="Segoe UI, Arial, sans-serif", size=12, color="#193451"),
        title_font=dict(size=17, color="#193451"),
        hoverlabel=dict(
            bgcolor="#ffffff",
            bordercolor="#b8c9da",
            font=dict(family="Segoe UI, Arial, sans-serif", size=12, color="#193451")
        ),
        uirevision=f"project-{selected_project}"
    )
    # =========================================================
    # SELECTED DATE ZOOM + HIGHLIGHT
    # =========================================================
    # Automatically fit the visible project records to calendar-month boundaries.
    # Apply after all filters; an explicitly selected date retains priority below.
    if not selected_date and not filtered.empty:
        date_starts = pd.to_datetime(filtered["Start"], errors="coerce")
        date_ends = pd.to_datetime(filtered["End"], errors="coerce")
        visible_start = date_starts.min()
        visible_end = date_ends.max()
        if pd.notna(visible_start) and pd.notna(visible_end):
            month_start = visible_start.to_period("M").start_time
            month_end = (visible_end.to_period("M") + 1).start_time
            fig.update_xaxes(range=[month_start, month_end])
    if selected_date:
        center_date = pd.to_datetime(selected_date)
        # 自動 zoom 到選取日前後 3 天
        fig.update_xaxes(
            range=[
                center_date - pd.Timedelta(days=3),
                center_date + pd.Timedelta(days=3)
            ]
        )
        # 加 Selected Date 藍色虛線
        fig.add_shape(
            type="line",
            x0=center_date,
            x1=center_date,
            y0=0,
            y1=1,
            yref="paper",
            line=dict(color="blue", width=2, dash="dot")
        )
        fig.add_annotation(
            x=center_date,
            y=0.88,
            yref="paper",
            text=(
                f"<b>SELECTED</b><br>"
                f"{center_date.strftime('%Y-%m-%d')}"
            ),
            showarrow=False,
            font=dict(color="blue", size=12),
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="blue",
            borderwidth=1
        )   
    # =========================================================
    # Y AXES
    # =========================================================
    # 同一個 Plotly Figure 分成上下兩個垂直區域：
    #
    #   yaxis2 → Windy Wave Forecast    上方 20%
    #   yaxis  → Engineering Gantt      下方 76%
    #
    # 中間保留少量空間，避免 Wave 與 Gantt 擠在一起。
    # =========================================================

    # ---------------------------------------------------------
    # Gantt Y-axis
    # ---------------------------------------------------------
    fig.update_yaxes(
        autorange="reversed",
        categoryorder="array",
        categoryarray=lane_order,
        tickmode="array",
        tickvals=lane_order,
        ticktext=ticktext,
        showticklabels=False,  # Custom left-aligned lane labels below
        title=None,
        tickfont=dict(family="Segoe UI, Arial, sans-serif", size=11, color="#193451"),
        showgrid=False,
        zeroline=False,

        # Gantt 使用圖面下方約 76%
        domain=[0.00, 0.84]
    )

    # Plotly's native Y-axis tick labels are right-aligned by design.
    # Use annotations with a common left anchor instead; preserve the
    # original colored progress percentages and lane order.
    for lane, label in zip(lane_order, ticktext):
        fig.add_annotation(
            x=0,
            xref="paper",
            xshift=-73,  # Keep labels left-aligned; Non-Offshore ends ~10-12 px before plot
            xanchor="left",
            y=lane,
            yref="y",
            yanchor="middle",
            text=label,
            showarrow=False,
            align="left",
            font=dict(family="Segoe UI, Arial, sans-serif", size=11, color="#193451"),
            bgcolor="rgba(0,0,0,0)",
            borderpad=0,
        )

    # ---------------------------------------------------------
    # Windy Wave Y-axis
    # ---------------------------------------------------------
    fig.update_layout(
        yaxis2=dict(
            title="Hs (m)",
    
            # Wave 區域壓扁：
            # 原本 [0.82, 1.00] 約佔 18%
            # 現在只佔最上方約 12%
            domain=[0.88, 1.00],
    
            # 與 Gantt 共用 X 軸
            anchor="x",
    
            # 預設只顯示 0–5 m；允許向上查看，但不得拖到 0 m 以下
            range=[0, 5],
            minallowed=0,
    
            # 固定刻度，閱讀比較直觀
            tickmode="array",
            tickvals=list(range(0, 11)),  # 0–10 m，每 1 m 一條水平網格線
    
            # Reference lines are drawn as 0.3 px black shapes above; avoid duplicate gray gridlines.
            showgrid=False,
            zeroline=False,
            tickfont=dict(size=10, color="#193451"),
            title_font=dict(size=12, color="#193451"),
            fixedrange=False
        )
    )
    fig.update_xaxes(
        rangeslider_visible=True,
        rangeslider=dict(thickness=0.02, bgcolor="#f3f7fb", bordercolor="#d6e0ea", borderwidth=1),
        tickfont=dict(size=11, color="#193451"),
        gridcolor="#ffffff",
        gridwidth=1,
        zeroline=False,
        fixedrange=False
    )
    return fig
#%% 點擊 Gantt bar 後，更新右側 detail-content 面板
@app.callback(
    Output("detail-content", "children"),
    Input("gantt-chart", "clickData"),
    State("schedule-version-store", "data")
)
def show_detail(clickData, schedule_version):
    # =========================================================
    # 尚未點擊任何 bar 時，顯示預設提示文字
    # =========================================================
    if clickData is None:
        return html.Div(
            "Click a bar to view details",
            style={
                "fontSize": "12px",
                "color": "#6b7280"
            }
        )
    # =========================================================
    # Ignore non-Gantt traces
    # =========================================================
    # Windy Wave trace 也位於 gantt-chart figure 中，
    # 但它的 customdata 只有：
    #   [wave_period_s, wave_direction_deg]
    #
    # Gantt bar 則有完整的 11 個 customdata fields。
    #
    # 因此點擊 Wave marker 時，
    # 不進入 Gantt detail parser，避免 IndexError。
    # =========================================================
    
    point = clickData["points"][0]
    customdata = point.get("customdata")
    
    if customdata is None or len(customdata) < 11:
        return html.Div(
            "Click a Gantt bar to view task details",
            style={
                "fontSize": "12px",
                "color": "#6b7280"
            }
        )
    # =========================================================
    # 從 clickData 取得被點擊 bar 的資料
    # customdata 對應 update_chart() 裡的 hover_cols 順序
    # =========================================================
    row = clickData["points"][0]["customdata"]
    date_str = row[0]
    category = row[1]
    task = row[2]
    cluster = row[3]
    note = row[8]
    progress = int(row[9])
    # =========================================================
    # 根據 Progress 決定顏色與狀態 icon
    # =========================================================
    if progress < 50:
        progress_color = "#dc2626"
        progress_icon = "🔴"
    elif progress < 80:
        progress_color = "#f59e0b"
        progress_icon = "🟠"
    else:
        progress_color = "#16a34a"
        progress_icon = "🟢"
    date = pd.to_datetime(row[10]).normalize()
    # =========================================================
    # 依照 Task + Date 從 hourly_df 找出當天 Timeline
    # =========================================================
    # Reuse the current schedule version; no extra Neon timestamp query per click.
    if schedule_version is None:
        return html.Div("Schedule data is loading...")
    _, current_hourly_df, _ = load_schedule_cached(schedule_version)
    
    timeline = current_hourly_df[
        (current_hourly_df["Task"] == task) &
        (current_hourly_df["Date"].dt.normalize() == date)
    ].copy()
    # =========================================================
    # 如果沒有 hourly timeline 資料
    # =========================================================
    if timeline.empty:
        timeline_children = [
            html.Div(
                "No timeline data available.",
                style={
                    "fontSize": "12px",
                    "color": "#6b7280",
                    "padding": "8px"
                }
            )
        ]
    # =========================================================
    # 如果有 timeline 資料，建立每一列 Time + Event
    # =========================================================
    else:
        timeline_children = [
            html.Div([

                # Time 欄位
                html.Span(
                    "" if pd.isna(r["Time"]) else str(r["Time"]),
                    style={
                        "display": "inline-block",
                        "width": "70px",
                        "textAlign": "center",
                        "fontSize": "12px",
                        "fontWeight": "bold",
                        "color": "#2563eb",
                        "verticalAlign": "top"
                    }
                ),

                # Event 欄位
                html.Span(
                    "" if pd.isna(r["Event"]) else str(r["Event"]),
                    style={
                        "fontSize": "12px",
                        "color": "#374151",
                        "whiteSpace": "nowrap"
                    }
                )

            ], style={
                "padding": "6px",
                "borderBottom": "1px solid #eeeeee",
                "whiteSpace": "nowrap",
                "minWidth": "450px"
            })

            for _, r in timeline.iterrows()
        ]
    # =========================================================
    # 回傳右側 Detail Panel 的內容
    # =========================================================
    return html.Div([
        ## Task / Date / Progress 資訊卡
        html.Div([
            # Task + Category + Note
            html.Div([
                html.Div(
                    f"{task} ({str(cluster) if str(cluster).lower().startswith('cluster') else 'Cluster ' + str(cluster)})" if pd.notna(cluster) and str(cluster).strip() not in ("", "nan", "None") else str(task),
                    style={
                        "fontSize": "16px",
                        "fontWeight": "bold",
                        "color": "#111827",
                        "marginBottom": "2px"
                    }
                ),
        
                # Category
                html.Div(
                    f"{category_icon.get(category,'')} {category}",
                    style={
                        "fontSize": "11px",
                        "color": "#6b7280"
                    }
                ),
        
                # Note
                html.Div(
                    note,
                    style={
                        "fontSize": "9px",
                        "color": "#9ca3af",
                        "marginTop": "2px",
                        "fontStyle": "italic"
                    }
                ) if note else None
        
            ], style={
                "paddingBottom": "8px",
                "marginBottom": "8px",
                "borderBottom": "1px solid #e5e7eb"
            }),
            # Date
            html.Div([
                html.Span(
                    "Date : ",
                    style={
                        "fontWeight": "bold",
                        "color": "#6b7280"
                    }
                ),
                html.Span(date_str)
            ], style={
                "fontSize": "13px",
                "marginBottom": "8px"
            }),
            # Progress
            html.Div([
                html.Span(
                    "Progress : ",
                    style={
                        "fontWeight": "bold",
                        "color": "#6b7280"
                    }
                ),
                html.Span(
                    f"{progress_icon} {int(progress)}%",
                    style={
                        "color": progress_color,
                        "fontWeight": "bold"
                    }
                )
            ], style={
                "fontSize": "13px"
            })

        ], style={
            "padding": "10px",
            "border": "1px solid #dddddd",
            "borderRadius": "8px",
            "backgroundColor": "#white",
            "marginBottom": "12px"
        }),
        html.Hr(style={"margin":"5px 0"}),
        ##Timeline 區塊標題
        html.H4(
            "Timeline",
            style={
                "margin": "0 0 8px 0",
                "fontSize": "15px",
                "fontWeight": "bold",
                "color": "#111827"
            }
        ),
        ## Timeline 清單
        ## overflowX="auto" + whiteSpace="nowrap"
        ## 可保留左右拉桿，避免文字自動換行
        html.Div(
            timeline_children,
            style={
                "maxHeight": "260px",
                "overflowY": "auto",
                "overflowX": "auto",
                "border": "1px solid #dddddd",
                "borderRadius": "8px",
                "padding": "6px",
                "backgroundColor": "white",
                "whiteSpace": "nowrap"
            }
        )
    ])
#%%windy-update-status
# 每 5 分鐘檢查一次 PostgreSQL 的最新 Windy 資料時間。
# 灰色：90 分鐘內；橘色：超過 90 分鐘；紅色：資料不存在或查詢失敗。
# 注意：資料過舊不等於 GitHub Actions 已確認失敗。
@app.callback(
    Output("windy-update-status", "children"),
    Input("windy-status-interval", "n_intervals")
)
def update_windy_status(_):
    disclaimer = " | Testing API – Not for Operational Use"

    def status_line(message, color):
        return [
            "Windy Forecast | ",
            html.Span(message, style={"color": color, "fontWeight": "500"}),
            disclaimer
        ]

    try:
        last_updated = get_windy_last_updated()

        if last_updated is None:
            return status_line("Error: Data Unavailable", "#dc2626")

        now_tw = pd.Timestamp.now(tz="Asia/Taipei")
        age_minutes = max(0, (now_tw - last_updated).total_seconds() / 60)
        updated_label = f"{last_updated:%Y-%m-%d %H:%M}"

        if age_minutes > 90:
            hours, minutes = divmod(int(age_minutes), 60)
            return status_line(
                f"Stale: {hours}h {minutes:02d}m (Forecast Last Updated: {updated_label})",
                "#d97706"
            )

        return status_line(f"Forecast Updated: {updated_label}", "#6b7280")

    except Exception as e:
        print(f"Windy status check failed: {e}")
        return status_line("Error: Status Unavailable", "#dc2626")
@app.callback(Output("open-meteo-update-status","children"),
              Input("windy-status-interval","n_intervals"))
def update_open_meteo_status(_):
    def line(message,color):
        return ["Open-Meteo Forecast | ",html.Span(message,style={"color":color,"fontWeight":"500"}),
                " | Forecast Data – Reference Only"]
    try:
        updated=get_open_meteo_last_updated()
        if updated is None: return line("Error: Data Unavailable","#dc2626")
        age=max(0,(pd.Timestamp.now(tz="Asia/Taipei")-updated).total_seconds()/60)
        label=f"{updated:%Y-%m-%d %H:%M}"
        if age>90:
            h,m=divmod(int(age),60)
            return line(f"Stale: {h}h {m:02d}m (Forecast Last Updated: {label})","#d97706")
        return line(f"Forecast Updated: {label}","#6b7280")
    except Exception as exc:
        print(f"Open-Meteo status failed: {exc}")
        return line("Error: Status Unavailable","#dc2626")

#%% KPI Auto Refresh
@app.callback(
    Output("kpi-container", "children"),
    Output("initial-kpi-check", "children"),
    Input("schedule-version-store", "data")
)
def update_kpi(schedule_version):

    if schedule_version is None:
        return no_update, no_update
    latest_df, _, _ = load_schedule_cached(schedule_version)

    latest_kpi = calculate_kpi(latest_df)

    return [
        build_card(title, value)
        for title, value in latest_kpi.items()
    ], "ready"
#%% Upcoming Tasks Auto Refresh
@app.callback(
    Output("upcoming-container", "children"),
    Input("schedule-version-store", "data")
)
def update_upcoming_tasks(schedule_version):

    if schedule_version is None:
        return no_update
    latest_df, _, _ = load_schedule_cached(schedule_version)

    return build_upcoming_tasks(
        days=7,
        source_df=latest_df
    )
#%% Task Filter Auto Refresh
@app.callback(
    Output("task-filter", "options"),
    Output("task-filter", "value"),
    Input("schedule-version-store", "data"),
    State("task-filter", "options"),
    State("task-filter", "value")
)
def update_task_options(schedule_version, old_options, selected_tasks):

    if schedule_version is None:
        return no_update, no_update
    latest_df, _, _ = load_schedule_cached(schedule_version)

    latest_tasks = latest_df[
        (~latest_df["Category"].isin(always_show_categories)) &
        (~latest_df["Task"].isin([
            "Mobilization",
            "Demobilization",
            "Standby"
        ]))
    ]["Task"].dropna().unique()

    new_options = sorted(latest_tasks.tolist())

    old_tasks = {
        item["value"] if isinstance(item, dict) else item
        for item in (old_options or [])
    }

    selected_tasks = selected_tasks or []

    # 保留原本已選取的 Task
    new_selection = [
        task for task in selected_tasks
        if task in new_options
    ]

    # 自動選取新加入的 Task
    new_selection += [
        task for task in new_options
        if task not in old_tasks
        and task not in new_selection
    ]

    # Do not write identical dropdown props every 60 seconds: doing so can
    # trigger update_chart and reset the user's Gantt viewport.
    old_options_values = [
        item["value"] if isinstance(item, dict) else item
        for item in (old_options or [])
    ]
    options_result = no_update if old_options_values == new_options else new_options
    value_result = no_update if (selected_tasks == new_selection) else new_selection
    return options_result, value_result
#%% Project and Personnel Filter Auto Refresh
@app.callback(
    Output("project-filter", "options"),
    Output("project-filter", "value"),
    Input("schedule-version-store", "data"),
    State("project-filter", "value"),
    State("project-filter", "options")
)
def update_project_options(schedule_version, selected_project, old_options):
    if schedule_version is None:
        return no_update, no_update
    latest_df, _, _ = load_schedule_cached(schedule_version)
    years = sorted(latest_df["Date"].dt.year.dropna().astype(int).unique(), reverse=True)
    options = [{"label": "All Projects", "value": "all"}] + [
        {"label": f"{year} | {PROJECT_CODES.get(year, str(year))}", "value": str(year)}
        for year in years
    ]
    valid = {item["value"] for item in options}
    new_value = selected_project if selected_project in valid else "all"
    return (no_update if old_options == options else options,
            no_update if selected_project == new_value else new_value)

@app.callback(
    Output("personnel-filter", "options"),
    Output("personnel-filter", "value"),
    Input("schedule-version-store", "data"),
    State("personnel-filter", "value"),
    State("personnel-filter", "options")
)
def update_personnel_options(schedule_version, selected_personnel, old_options):
    if schedule_version is None:
        return no_update, no_update
    latest_df, _, _ = load_schedule_cached(schedule_version)
    names = personnel_names(latest_df)
    options = [{"label": name, "value": name} for name in names]
    selected = [name for name in (selected_personnel or []) if name in names]
    return (no_update if old_options == options else options,
            no_update if selected == (selected_personnel or []) else selected)

# Shared source of truth for jacket Inspection progress.
# Use the latest dated Inspection entry for each canonical jacket.
def latest_inspection_progress(source_df, master_towers):
    if source_df.empty:
        return {}
    prepared = source_df if "_chart_lane" in source_df.columns else _prepare_completion_lanes(source_df)
    valid = prepared.loc[
        prepared["Category"].fillna("").astype(str).str.strip().str.casefold().eq("inspection")
        & prepared["_chart_lane"].isin(set(master_towers))
    ].copy()
    if valid.empty:
        return {}
    valid["_completion_progress"] = pd.to_numeric(valid["Progress"], errors="coerce")
    valid = valid.dropna(subset=["_completion_progress"])
    sort_columns = [c for c in ("Date", "Start", "End") if c in valid.columns]
    valid = valid.sort_values(sort_columns, kind="stable") if sort_columns else valid
    latest = valid.drop_duplicates("_chart_lane", keep="last")
    return latest.set_index("_chart_lane")["_completion_progress"].to_dict()


def _prepare_completion_lanes(source_df):
    # Reuse the exact tower normalization used by the Gantt.
    task_codes = source_df["Task"].fillna("").astype(str).str.extract(
        r"(?i)(?<![A-Za-z0-9])([ABCD]\s*[-_]?\s*\d{1,2})(?!\d)", expand=False)
    lane_codes = source_df["Lane"].fillna("").astype(str).str.extract(
        r"(?i)(?<![A-Za-z0-9])([ABCD]\s*[-_]?\s*\d{1,2})(?!\d)", expand=False)
    codes = task_codes.fillna(lane_codes).str.replace(r"[^A-Za-z0-9]", "", regex=True).str.upper()
    parts = codes.str.extract(r"^([ABCD])(\d{1,2})$", expand=True)
    prepared = source_df.copy()
    prepared["_chart_lane"] = parts[0] + parts[1].str.zfill(2)
    prepared.loc[source_df["Lane"].fillna("").astype(str).str.strip().eq("Non-Offshore"), "_chart_lane"] = "Non-Offshore"
    return prepared


#%% Overall F2 tower completion (unaffected by display filters)
def calculate_overall_tower_completion(source_df, master_towers):
    """Count jackets whose latest Inspection progress is at least 100%."""
    progress = latest_inspection_progress(source_df, master_towers)
    return sum(value >= 100 for value in progress.values()), len(set(master_towers))


@lru_cache(maxsize=2)
def cached_completion_counts(schedule_version):
    """Compute both summary counts once per Neon schedule version."""
    latest_df, _, _ = load_schedule_cached(schedule_version)
    master_towers = cached_master_towers()
    overall_completed, total = calculate_overall_tower_completion(latest_df, master_towers)
    project_df = latest_df.loc[latest_df["Date"].dt.year.eq(2026)]
    project_completed, _ = calculate_overall_tower_completion(project_df, master_towers)
    return overall_completed, total, project_completed


@app.callback(
    Output("overall-progress-percent", "children"),
    Output("overall-progress-count", "children"),
    Output("project-completed-jackets", "children"),
    Output("initial-progress-check", "children"),
    Input("schedule-version-store", "data")
)
def update_overall_progress(schedule_version):
    if schedule_version is None:
        return no_update, no_update, no_update, no_update
    completed, total, project_completed = cached_completion_counts(schedule_version)
    percentage = (100 * completed / total) if total else 0
    return f"{percentage:.1f}%", f"{completed} / {total} Jackets", f" | Completed Jackets: {project_completed}", "ready"

#%% Schedule Updated Time Auto Refresh
@app.callback(
    Output("schedule-update-status", "children"),
    Output("initial-schedule-check", "children"),
    Input("schedule-version-store", "data")
)
def update_schedule_status(schedule_version):

    if schedule_version is None:
        return no_update, no_update
    _, _, latest_updated = load_schedule_cached(schedule_version)

    return (
        "Engineering Scheduling & Progress Tracking System | "
        "v1.5.1 Beta | "
        f"Schedule Data Updated: {latest_updated}"
    ), "ready"
# End of module initialization (one measurement per Python worker).
_STARTUP_READY_SECONDS = time.perf_counter() - _STARTUP_BEGIN
print(f"[STARTUP] Python startup total: {_STARTUP_READY_SECONDS:.3f} s", flush=True)

#%%Run server
#render佈署
if __name__ == "__main__":
    app.run(
        debug=False,
        host="0.0.0.0",
        port=8050
    )

#本機測試
# server = app.server
# if __name__ == "__main__":
#     app.run(
#         debug=True,
#         host="127.0.0.1",
#         port=8050
#     )    
    
    
#http://127.0.0.1:8050/