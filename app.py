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

from flask import Flask, request, Response, jsonify, session
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




def _health_data():
    """Read external liveness-check history; do not modify monitoring jobs."""
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT checked_at, status, http_status, total_seconds, error_message
            FROM health_checks ORDER BY checked_at DESC LIMIT 168
        """)).mappings().all()
    now = pd.Timestamp.now(tz="UTC")
    history = []
    for row in rows:
        stamp = pd.Timestamp(row["checked_at"])
        if stamp.tzinfo is None:
            stamp = stamp.tz_localize("UTC")
        stamp = stamp.tz_convert("Asia/Taipei")
        history.append({
            "time": stamp.strftime("%Y-%m-%d %H:%M:%S"),
            "status": str(row["status"] or "UNKNOWN").upper(),
            "http": row["http_status"],
            "seconds": float(row["total_seconds"]) if row["total_seconds"] is not None else None,
            "error": str(row["error_message"] or ""),
        })
    latest = history[0] if history else None
    if not rows:
        state = "NO DATA"
    else:
        checked = pd.Timestamp(rows[0]["checked_at"])
        if checked.tzinfo is None:
            checked = checked.tz_localize("UTC")
        state = "DELAYED" if (now - checked).total_seconds() > 90 * 60 else latest["status"]
    return {"state": state, "latest": latest, "history": history,
            "success_rate": round(100 * sum(r["status"] == "PASS" for r in history) / len(history), 1) if history else None}

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
        .admin-tabs {{ display:flex; gap:6px; flex-wrap:wrap; margin:14px 0 18px; }}
        .admin-tab {{ background:white; border:1px solid #cbd5e1; border-radius:7px;
            padding:8px 15px; font-size:13px; cursor:pointer; color:#334155; }}
        .admin-tab.active {{ background:#1d4ed8; color:white; border-color:#1d4ed8; }}
        .admin-panel[hidden] {{ display:none !important; }}
        .health-metrics {{ display:flex; gap:10px; flex-wrap:wrap; margin:14px 0; }}
        .health-metric {{ background:white; border:1px solid #ddd; border-radius:10px;
            padding:12px 16px; min-width:145px; flex:1; }}
        .health-metric strong {{ display:block; font-size:20px; margin-top:6px; }}
        .health-table-wrap {{ max-height:370px; overflow:auto; border:1px solid #ddd; border-radius:9px; }}
        .health-table {{ border-collapse:collapse; width:100%; font-size:13px; background:white; }}
        .health-table th,.health-table td {{ text-align:left; padding:9px; border-bottom:1px solid #e5e7eb; }}
        .health-table th {{ background:#f1f5f9; position:sticky; top:0; }}
        .health-plot {{ width:100%; height:240px; background:white; border:1px solid #ddd;
            border-radius:10px; box-sizing:border-box; }}
        .health-pass {{ color:#15803d; }} .health-fail {{ color:#b91c1c; }}
        .health-delayed {{ color:#b45309; }}
    </style>

    <h1>Admin Console</h1>
    <nav class="admin-tabs" aria-label="Admin sections">
      <button type="button" class="admin-tab active" data-panel="analytics">Analytics</button>
      <button type="button" class="admin-tab" data-panel="startup">Startup</button>
      <button type="button" class="admin-tab" data-panel="health">Health</button>
    </nav>
    <div class="admin-panel" id="panel-analytics">
      <h2>Site Analytics</h2>

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

    </div>
    </div>
    <div class="admin-panel" id="panel-startup" hidden>
      {_startup_diagnostics_html()}
    </div>
    <div class="admin-panel" id="panel-health" hidden>
      <h2>System Health</h2>
      <p class="note">External /healthz liveness checks · GitHub Actions · Neon history (up to 168 checks)</p>
      <div id="health-content" aria-live="polite">Loading health checks…</div>
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
        const tabs = Array.from(document.querySelectorAll('.admin-tab'));
        const panels = Array.from(document.querySelectorAll('.admin-panel'));
        const healthContent = document.getElementById('health-content');
        let healthBusy = false;
        function escapeHtml(value) {{
            return String(value == null ? '' : value).replace(/[&<>"']/g, c =>
                ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}})[c]);
        }}
        function renderHealth(data) {{
            const latest = data.latest;
            const state = data.state;
            const cls = state === 'PASS' ? 'health-pass' :
                state === 'DELAYED' ? 'health-delayed' : 'health-fail';
            const time = latest ? latest.time + ' (Asia/Taipei)' : '—';
            const duration = latest && latest.seconds != null ? latest.seconds.toFixed(3) + ' s' : '—';
            const rate = data.success_rate == null ? '—' : data.success_rate.toFixed(1) + '%';
            const metrics = `<div class="health-metrics">
                <div class="health-metric">Current Status<strong class="${{cls}}">${{escapeHtml(state)}}</strong></div>
                <div class="health-metric">Last Check<strong style="font-size:14px">${{escapeHtml(time)}}</strong></div>
                <div class="health-metric">Response Time<strong>${{escapeHtml(duration)}}</strong></div>
                <div class="health-metric">Pass Rate (loaded checks)<strong>${{escapeHtml(rate)}}</strong></div>
                </div>`;
            const history = data.history || [];
            const valid = history.filter(r => r.seconds != null).slice().reverse();
            let plot = '<p class="note">No response-time data yet.</p>';
            if (valid.length) {{
                const w = 900, h = 230, left = 56, right = 20, top = 18, bottom = 34;
                const max = Math.max(0.1, ...valid.map(r => r.seconds)) * 1.1;
                const x = i => left + i * (w-left-right) / Math.max(1, valid.length-1);
                const y = v => top + (h-top-bottom) * (1-v/max);
                const pts = valid.map((r,i) => `${{x(i).toFixed(1)}},${{y(r.seconds).toFixed(1)}}`).join(' ');
                const dots = valid.map((r,i) => `<circle cx="${{x(i).toFixed(1)}}" cy="${{y(r.seconds).toFixed(1)}}" r="3" fill="${{r.status === 'PASS' ? '#2563eb' : '#dc2626'}}"><title>${{escapeHtml(r.time)}} · ${{r.seconds.toFixed(3)}} s · ${{escapeHtml(r.status)}}</title></circle>`).join('');
                plot = `<svg class="health-plot" viewBox="0 0 ${{w}} ${{h}}" role="img" aria-label="Response time history in seconds">
                    <line x1="${{left}}" y1="${{h-bottom}}" x2="${{w-right}}" y2="${{h-bottom}}" stroke="#94a3b8"/>
                    <line x1="${{left}}" y1="${{top}}" x2="${{left}}" y2="${{h-bottom}}" stroke="#94a3b8"/>
                    <text x="8" y="${{top+5}}" font-size="12" fill="#475569">${{max.toFixed(2)}} s</text>
                    <text x="25" y="${{h-bottom}}" font-size="12" fill="#475569">0</text>
                    <text x="${{left}}" y="${{h-8}}" font-size="12" fill="#475569">Oldest</text>
                    <text x="${{w-right}}" y="${{h-8}}" text-anchor="end" font-size="12" fill="#475569">Latest</text>
                    <polyline fill="none" stroke="#2563eb" stroke-width="2" points="${{pts}}"/>${{dots}}</svg>`;
            }}
            const rows = history.map(r => `<tr><td>${{escapeHtml(r.time)}}</td><td>${{escapeHtml(r.status)}}</td><td>${{escapeHtml(r.http)}}</td><td>${{r.seconds == null ? '—' : r.seconds.toFixed(3) + ' s'}}</td><td>${{escapeHtml(r.error)}}</td></tr>`).join('');
            healthContent.innerHTML = metrics + '<h3>Response Time History</h3>' + plot +
                '<h3>Check History</h3><div class="health-table-wrap"><table class="health-table"><thead><tr><th>Time (TW)</th><th>Status</th><th>HTTP</th><th>Duration</th><th>Error</th></tr></thead><tbody>' +
                (rows || '<tr><td colspan="5">No checks recorded yet.</td></tr>') + '</tbody></table></div>';
        }}
        async function refreshHealth() {{
            if (healthBusy || document.hidden) return;
            healthBusy = true;
            try {{
                const response = await fetch('/admin/health-data', {{cache:'no-store', credentials:'same-origin'}});
                if (!response.ok) throw new Error('Health refresh HTTP ' + response.status);
                renderHealth(await response.json());
            }} catch (error) {{
                healthContent.textContent = 'Health data unavailable: ' + error.message;
            }} finally {{ healthBusy = false; }}
        }}
        tabs.forEach(tab => tab.addEventListener('click', () => {{
            const target = tab.dataset.panel;
            tabs.forEach(t => {{
                t.classList.toggle('active', t === tab);
                t.setAttribute('aria-selected', t === tab ? 'true' : 'false');
            }});
            panels.forEach(p => p.hidden = p.id !== 'panel-' + target);
            if (target === 'health') refreshHealth();
        }}));
        setInterval(refreshAnalytics, refreshMs);
        setInterval(() => {{
            if (!document.getElementById('panel-health').hidden) refreshHealth();
        }}, refreshMs);
    }})();
    </script>

    <p>
        <a href="/">&larr; Back to Dashboard</a>
    </p>
    """





F2_LEAFLET_MAP_HTML = '<!doctype html><html lang="zh-Hant"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>F2 Marine Wave Overlay — Prototype</title><link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"><style>body{margin:0;font:14px system-ui,-apple-system,sans-serif;background:#101827;color:#edf2f7}.wrap{max-width:1300px;margin:auto;padding:18px}.top{display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:12px}h2{margin:0 0 5px;font-size:21px}.muted{color:#aab8cc}#map{height:min(76vh,760px);min-height:460px;border-radius:12px;overflow:hidden;margin-top:8px}.controls{display:flex;align-items:center;gap:12px;flex-wrap:wrap;margin:10px 0}.controls button{background:#263c5a;color:white;border:1px solid #57708e;border-radius:6px;padding:8px 12px;cursor:pointer}input{accent-color:#52a4e6}#status{font-size:12px;padding:8px 0;color:#b9c8db}.tower-label{font-size:11px;font-weight:700;color:#102036;background:#fff;padding:2px 4px;border-radius:3px;box-shadow:0 1px 3px #0006;white-space:nowrap}.legend{background:white;color:#1b293b;padding:8px 10px;border-radius:5px;font:12px system-ui}.legend .bar{height:10px;width:175px;background:linear-gradient(90deg,#262f90,#2084c5,#32b88e,#ffe45e,#f28c39,#bd2d2d);margin:6px 0}.wave-key{background:#1b2b42;border:1px solid #48607e;border-radius:9px;padding:12px 14px;margin:12px 0 4px;max-width:620px}.wave-key-title{font-weight:700;margin-bottom:7px}.wave-key-bar{height:15px;border-radius:4px;background:linear-gradient(to right,#3b276d,#6c3f9a,#ae6595,#dd8b7d,#eeaa64,#f4d783,#f5f7a6)}.wave-key-ticks{display:flex;justify-content:space-between;font-size:12px;margin-top:5px}.wave-key-note{font-size:11px;color:#b9c8db;margin-top:7px;line-height:1.5}.note{line-height:1.6;font-size:12px;color:#aab8cc}\n.filter-row{margin-top:0}.filter-row .map-controls{max-width:none;box-shadow:none;background:transparent;border:0;padding:0;font-size:13px}.filter-row .map-controls input[type=range]{width:95px}\n/* Compact map-first interface. Controls are HTML overlays, not geographic layers. */\n#map{position:relative;height:min(82vh,800px);min-height:520px;margin-top:14px}\n.map-bottom{position:absolute;z-index:900;left:12px;right:12px;bottom:22px;display:flex;flex-direction:column;gap:8px;align-items:flex-start;pointer-events:none}\n.map-controls{display:flex;flex-wrap:wrap;align-items:center;gap:8px 12px;background:rgba(16,24,39,.93);color:#edf2f7;border:1px solid #56708c;border-radius:9px;padding:9px 11px;box-shadow:0 3px 14px #0006;pointer-events:auto;max-width:calc(100% - 22px);font-size:12px}\n.map-controls label{display:flex;align-items:center;gap:5px;white-space:nowrap}\n.map-controls input[type=range]{width:80px}\n.map-controls button{background:#263c5a;color:white;border:1px solid #57708e;border-radius:6px;padding:6px 9px;cursor:pointer}\n.map-bottom .wave-key{margin:0;width:min(420px,calc(100% - 22px));max-width:420px;box-sizing:border-box;background:rgba(16,24,39,.94);color:#edf2f7;border:1px solid #56708c;padding:10px 13px;pointer-events:auto}\n.wave-key-title{font-size:12px}.key-hint{font-size:10px;font-weight:400;color:#b9c8db}\n.wave-key-bar{height:14px;margin-top:6px;border-radius:4px;background:linear-gradient(90deg,#0638ff 0%,#0286b7 20%,#44a75d 40%,#c4cf52 58%,#ffbd36 77%,#ff3900 100%)}.wave-key-ticks{font-size:11px;margin-top:5px}.wave-key-title{font-weight:700}\n#status{margin-top:5px}\n@media(max-width:650px){.map-bottom{bottom:18px;left:7px;right:7px}.map-controls{gap:5px 8px;padding:7px;font-size:11px}.map-controls input[type=range]{width:65px}#map{min-height:570px}}\n/* Dashboard iframe: prevent nested scrollbars. */\nhtml,body{height:100%;overflow:hidden}\n.wrap{box-sizing:border-box;height:100vh;max-width:none;padding:12px 16px;display:flex;flex-direction:column;min-height:0}\n.filter-row{display:none!important;flex:0 0 auto}\n#map{flex:1 1 auto;height:auto!important;min-height:0!important;margin-top:0}\n#status{flex:0 0 auto;padding:0;margin:0}\n@media(max-width:650px){#map{min-height:0!important}}\n</style></head><body><div class="wrap"><div class="filter-row"><div class="map-controls"><label><input id="waves" type="checkbox" checked> Waves</label><label>Opacity <input id="opacity" type="range" min="0" max="100" value="100"> <span id="opacityLabel">100%</span></label><label><input id="smooth" type="checkbox" checked> Smooth</label><label>Blur <input id="blur" type="range" min="0" max="24" value="24"> <span id="blurLabel">24px</span></label><label><input id="labels" type="checkbox" checked> Tower IDs</label><button id="fit">Fit towers</button></div></div><div id="map"><div class="map-bottom"><section class="wave-key" aria-label="Wave height color scale"><div class="wave-key-title">Hs (m)</div><div class="wave-key-bar" aria-hidden="true"></div><div class="wave-key-ticks"><span>0.5</span><span>1.5</span><span>2.5</span><span>3.5</span><span>4.5</span><span>5.5</span><span>6.5</span></div><div class="wave-key-note">Approximate color guide · refer to Copernicus for exact values</div></section></div></div><div id="status">Initializing map…</div></div><script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script><script>\nconst towers = [{"id": "D01", "lat": 24.66391667000001, "lon": 120.7168333}, {"id": "D02", "lat": 24.65530555999999, "lon": 120.7318056}, {"id": "D03", "lat": 24.64747221999998, "lon": 120.7228333}, {"id": "D04", "lat": 24.64522221999999, "lon": 120.7052222}, {"id": "D05", "lat": 24.63769444, "lon": 120.7075278}, {"id": "D06", "lat": 24.62544444000002, "lon": 120.6911944}, {"id": "D07", "lat": 24.67158333000002, "lon": 120.7114167}, {"id": "D08", "lat": 24.65661111000002, "lon": 120.7069722}, {"id": "D09", "lat": 24.65463888999999, "lon": 120.6974167}, {"id": "D10", "lat": 24.64388889000002, "lon": 120.6948889}, {"id": "D11", "lat": 24.63536110999998, "lon": 120.6955278}, {"id": "D12", "lat": 24.63119443999999, "lon": 120.6858056}, {"id": "C01", "lat": 24.69377778000002, "lon": 120.7689167}, {"id": "C02", "lat": 24.70077778000003, "lon": 120.7594167}, {"id": "C03", "lat": 24.71761111000002, "lon": 120.7624722}, {"id": "C04", "lat": 24.71572222000001, "lon": 120.7511667}, {"id": "C05", "lat": 24.70413888999999, "lon": 120.7493333}, {"id": "C06", "lat": 24.70663889000001, "lon": 120.7420278}, {"id": "C07", "lat": 24.68630555999999, "lon": 120.7566389}, {"id": "C08", "lat": 24.67605556000003, "lon": 120.7424722}, {"id": "C09", "lat": 24.68166666999998, "lon": 120.7372778}, {"id": "C10", "lat": 24.69175000000003, "lon": 120.7472778}, {"id": "C11", "lat": 24.69522221999999, "lon": 120.7392222}, {"id": "B01", "lat": 24.70952777999999, "lon": 120.7708611}, {"id": "B02", "lat": 24.72827777999997, "lon": 120.7743056}, {"id": "B03", "lat": 24.75319444000003, "lon": 120.7887778}, {"id": "B04", "lat": 24.76647222, "lon": 120.7979722}, {"id": "B05", "lat": 24.77286111, "lon": 120.81475}, {"id": "B06", "lat": 24.77777778000002, "lon": 120.8079722}, {"id": "B07", "lat": 24.72738889, "lon": 120.7591111}, {"id": "B08", "lat": 24.73808333000001, "lon": 120.7606944}, {"id": "B09", "lat": 24.74394443999999, "lon": 120.7711944}, {"id": "B10", "lat": 24.76277778, "lon": 120.7811667}, {"id": "B11", "lat": 24.77411110999999, "lon": 120.7920278}, {"id": "B12", "lat": 24.78300000000001, "lon": 120.8006667}, {"id": "A01", "lat": 24.70427778000001, "lon": 120.7805}, {"id": "A02", "lat": 24.72011111, "lon": 120.7934167}, {"id": "A03", "lat": 24.72658333000002, "lon": 120.8059444}, {"id": "A04", "lat": 24.73397221999999, "lon": 120.7993056}, {"id": "A05", "lat": 24.74666667000001, "lon": 120.8159722}, {"id": "A06", "lat": 24.75655556, "lon": 120.8277778}, {"id": "A07", "lat": 24.71897222000001, "lon": 120.7725278}, {"id": "A08", "lat": 24.72305556000002, "lon": 120.7840556}, {"id": "A09", "lat": 24.73747222000001, "lon": 120.7859722}, {"id": "A10", "lat": 24.74416666999999, "lon": 120.79475}, {"id": "A11", "lat": 24.75399999999998, "lon": 120.8025}, {"id": "A12", "lat": 24.75513889, "lon": 120.8121944}];\nL.DomEvent.disableClickPropagation(document.querySelector(\'.map-bottom\'));L.DomEvent.disableScrollPropagation(document.querySelector(\'.map-bottom\'));\nconst map=L.map(\'map\',{preferCanvas:true}).setView([24.70843,120.75996],11);\nL.tileLayer(\'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png\',{maxZoom:19,attribution:\'© OpenStreetMap contributors\'}).addTo(map);\nconst group=L.featureGroup();const labels=[];\nfor(const t of towers){const marker=L.circleMarker([t.lat,t.lon],{radius:5,color:\'#142a4d\',weight:1.5,fillColor:\'#ffca36\',fillOpacity:1}).bindPopup(\'<b>\'+t.id+\'</b><br>Latitude: \'+t.lat.toFixed(8)+\'<br>Longitude: \'+t.lon.toFixed(8));marker.addTo(group);const label=L.marker([t.lat,t.lon],{icon:L.divIcon({className:\'\',html:\'<span class="tower-label">\'+t.id+\'</span>\',iconSize:[45,18],iconAnchor:[-8,9]}),interactive:false}).addTo(group);labels.push(label)}\ngroup.addTo(map);map.fitBounds(group.getBounds().pad(.12));\n// Copernicus Marine global wave forecast: dynamic WMTS tiles in Web Mercator.\n// Dataset IDs and supported matrix sets can change; validate against GetCapabilities.\nconst layerId=\'GLOBAL_ANALYSISFORECAST_WAV_001_027/cmems_mod_glo_wav_anfc_0.083deg_PT3H-i_202411/VHM0\';\nconst tileUrl=\'https://wmts.marine.copernicus.eu/teroWmts/?service=WMTS&version=1.0.0&request=GetTile&format=image%2Fpng&layer=\'+encodeURIComponent(layerId)+\'&tilematrixset=EPSG:3857&tilematrix={z}&tilecol={x}&tilerow={y}&STYLE=cmap:rainbow,range=0/5\';\nmap.createPane(\'wavePane\');map.getPane(\'wavePane\').style.zIndex=\'350\';map.getPane(\'wavePane\').style.pointerEvents=\'none\';\nconst waves=L.tileLayer(tileUrl,{pane:\'wavePane\',opacity:1,maxNativeZoom:10,maxZoom:19,tileSize:256,attribution:\'Copernicus Marine Service\',errorTileUrl:\'\'});\nlet good=0,bad=0;const status=document.getElementById(\'status\');\nfunction updateStatus(){const details=\'KML: \'+towers.length+\' turbines loaded · Copernicus tiles: \'+good+\' loaded, \'+bad+\' failed\';console.debug(details);status.textContent=(bad>0 && good===0)?\'Wave overlay unavailable · Check Copernicus service or network\':\'\';}\nwaves.on(\'tileload\',()=>{good++;updateStatus()});waves.on(\'tileerror\',()=>{bad++;updateStatus()});\nwaves.addTo(map);group.bringToFront();updateStatus();\nfunction updateSmoothing(){const active=document.getElementById(\'smooth\').checked;const px=Number(document.getElementById(\'blur\').value);map.getPane(\'wavePane\').style.filter=active?\'blur(\'+px+\'px)\':\'none\';document.getElementById(\'blurLabel\').textContent=px+\'px\';}\ndocument.getElementById(\'smooth\').onchange=updateSmoothing;document.getElementById(\'blur\').oninput=updateSmoothing;updateSmoothing();\ndocument.getElementById(\'waves\').onchange=e=>{if(e.target.checked){waves.addTo(map);group.bringToFront()}else map.removeLayer(waves)};\ndocument.getElementById(\'opacity\').oninput=e=>{waves.setOpacity(Number(e.target.value)/100);document.getElementById(\'opacityLabel\').textContent=e.target.value+\'%\'};\ndocument.getElementById(\'labels\').onchange=e=>{labels.forEach(l=>e.target.checked?group.addLayer(l):group.removeLayer(l))};\nfunction fitF2Towers(){\n  map.invalidateSize({pan:false});\n  if(group.getLayers().length){map.fitBounds(group.getBounds().pad(.15),{animate:false});}\n}\ndocument.getElementById(\'fit\').onclick=fitF2Towers;\n// The iframe is initially inside a hidden Dash tab. Re-fit once visible.\nconst mapResizeObserver=new ResizeObserver(entries=>{\n  const rect=entries[0].contentRect;\n  if(rect.width>0 && rect.height>0){requestAnimationFrame(fitF2Towers);}\n});\nmapResizeObserver.observe(document.getElementById(\'map\'));\nwindow.addEventListener(\'message\',event=>{\n  if(event.origin===window.location.origin && event.data===\'f2-copernicus-fit\'){\n    requestAnimationFrame(fitF2Towers);\n  }\n});\n\n</script></body></html>'

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
                        style={"height":"36px", "width":"140px", "maxWidth":"100%", "objectFit":"contain", "display":"block", "margin":"0 auto", "transform":"scale(0.85)" if p["name"] == "BeeX" else "scale(1.15)"}
                    ),
                    style={"textAlign":"center", "verticalAlign":"middle"}
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
# Configure a persistent random SECRET_KEY in Render / .env.
# A random fallback permits local testing but invalidates sessions on restart.
server.secret_key = os.getenv("FLASK_SECRET_KEY") or secrets.token_hex(32)
server.config.update(SESSION_COOKIE_HTTPONLY=True, SESSION_COOKIE_SAMESITE="Lax",
                     SESSION_COOKIE_SECURE=os.getenv("RENDER") is not None)
server.before_request(_track)

# Lightweight liveness endpoint for external health monitoring.
# Does not query Neon or call forecast APIs.
@server.get("/healthz")
def healthz():
    response = jsonify(status="ok")
    response.headers["Cache-Control"] = "no-store"
    return response

# Initial pilot access code. Replace with a server-side employee-ID allowlist
# before inviting staff; employee numbers alone are not strong authentication.
RESOURCE_CODES = {"Iog54503373"}
RESOURCE_SESSION_SECONDS = 8 * 60 * 60

def resource_access_valid():
    verified = session.get("resource_verified_at")
    return isinstance(verified, (int, float)) and time.time() - verified < RESOURCE_SESSION_SECONDS

RESOURCE_LINKS = [
    ("Duty Roster", "https://iovtec.sharepoint.com/sites/SV-SurveyDept2/Shared%20Documents/Forms/AllItems.aspx?id=%2Fsites%2FSV%2DSurveyDept2%2FShared%20Documents%2F%E6%8E%A2%E6%B8%AC%E9%83%A8%2D%E7%8F%BE%E5%A0%B4%E7%87%9F%E9%81%8B%E7%B5%84%2F%E7%8F%AD%E8%A1%A8&p=true&ga=1"),
    ("DPR", "https://iovtec-my.sharepoint.com/:f:/g/personal/ali_chang_iovtec_com_tw/IgDMHGM8MshVQqUP_kgzjh4LAfee-_a7WUBFEz2pLI2bgXE?e=E5T8b7"),
    ("Survey Log", "https://iovtec.sharepoint.com/sites/SV-SurveyDept2/Shared%20Documents/Forms/AllItems.aspx?id=%2Fsites%2FSV%2DSurveyDept2%2FShared%20Documents%2F%E6%8E%A2%E6%B8%AC%E9%83%A8%2D%E7%8F%BE%E5%A0%B4%E7%87%9F%E9%81%8B%E7%B5%84%2FSurvey%20Log&viewid=a9aaccb6%2Dfaa4%2D446d%2Db6e3%2Dbeafdbfc933e"),
    ("Photo Archive", "https://iovtec.sharepoint.com/sites/IOGWindProject/Shared%20Documents/Forms/AllItems.aspx?id=%2Fsites%2FIOGWindProject%2FShared%20Documents%2F01%5FOperations%2F04%5FOngoing%20Projects%2F4372%5F%20Manpower%20for%20BeeX%2F2%2E%20Project%20Documentation%2F%E4%BD%9C%E6%A5%AD%E7%85%A7%E7%89%87&p=true&ct=1782188226845&or=Teams%2DHL&ga=1&LOF=1"),
]

def resource_links_view():
    return [html.A(name, href=url, target="_blank", rel="noopener noreferrer",
                   style={"display":"block", "padding":"8px 0", "color":"#9ed2ff"})
            for name, url in RESOURCE_LINKS]


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

@server.get("/admin/health-data")
def admin_health_data():
    if not _check_auth():
        return _auth_required()
    try:
        response = jsonify(_health_data())
    except Exception as exc:
        server.logger.warning("Admin health data unavailable: %s", exc)
        response = jsonify(error="Health check history is temporarily unavailable")
        response.status_code = 503
    response.headers["Cache-Control"] = "no-store"
    return response

@server.route("/f2-marine-wave-map")
def f2_marine_wave_map():
    return Response(F2_LEAFLET_MAP_HTML, mimetype="text/html")

app = Dash(__name__, server=server, suppress_callback_exceptions=True)

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
        <script>
        // Capture Dash callback requests before Dash renderer initializes.
        // Background interval callbacks after initial readiness do not reopen the overlay.
        (function () {
            var state = window.__initialDashCallbacks = {active: 0, seen: 0, lastActivity: performance.now()};
            var nativeFetch = window.fetch;
            window.fetch = function () {
                var input = arguments[0];
                var url = typeof input === 'string' ? input : (input && input.url) || '';
                var isCallback = url.indexOf('_dash-update-component') !== -1;
                if (!isCallback) return nativeFetch.apply(this, arguments);
                state.active += 1;
                state.seen += 1;
                state.lastActivity = performance.now();
                try {
                    return Promise.resolve(nativeFetch.apply(this, arguments)).then(
                        function (response) {
                            state.active -= 1;
                            state.lastActivity = performance.now();
                            return response;
                        },
                        function (error) {
                            state.active -= 1;
                            state.lastActivity = performance.now();
                            throw error;
                        }
                    );
                } catch (error) {
                    state.active -= 1;
                    state.lastActivity = performance.now();
                    throw error;
                }
            };
        })();
        </script>
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
                margin-top: 27px; transform: translateY(-18px); color: #475569; font-size: 12px; text-align: left; }
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
                <div class="initial-loading-title">F2 Offshore Dashboard</div>
                <div class="initial-loading-subtitle">Initializing dashboard</div>
                <img alt="Octopus cat preparing offshore dashboard data" src="data:image/png;base64,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" style="display:block;width:min(320px,75vw);height:auto;margin:16px auto 0;" />
                <div class="initial-loading-status"><span id="initial-loading-stage">Initializing Dashboard Interface...</span>
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
            var finished = false, stage = 0;
            var ganttRendered = false;
            var callbackState = window.__initialDashCallbacks;
            var fill = document.getElementById('initial-loading-fill');
            var tux = document.getElementById('initial-loading-tux');
            var label = document.getElementById('initial-loading-stage');
            var count = document.getElementById('initial-loading-count');
            var track = overlay.querySelector('.initial-loading-track');
            // The stage label describes the CURRENT check, not the previous result.
            var descriptions = [
                'Initializing Dashboard Interface...',
                'Loading Schedule Data...',
                'Checking Summary Table & KPI Cards...',
                'Verifying Overall Progress & Completed Jackets...',
                'Rendering Gantt Chart...',
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
                if (finished) return;
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
                // The chart alone is not sufficient: all initial Dash callback
                // requests must finish, including those launched by other callbacks.
                if (!ganttRendered) {
                    ganttRendered = true;
                    requestAnimationFrame(function () {
                        requestAnimationFrame(checkDashboard);
                    });
                    return;
                }
                if (!callbackState || callbackState.seen === 0 || callbackState.active !== 0) return;
                // Dash may enqueue a dependent callback after the preceding request
                // resolves; wait for the renderer and a short idle window.
                if (document.title.indexOf('Updating...') !== -1 ||
                    performance.now() - callbackState.lastActivity < 700) return;
                advance(5);
                // Keep the completed loading screen visible for one second.
                // Only the visual transition is delayed; data loading is unchanged.
                window.setTimeout(function () {
                    if (callbackState.active === 0 &&
                        document.title.indexOf('Updating...') === -1 &&
                        performance.now() - callbackState.lastActivity >= 700) {
                        finish();
                    }
                }, 1000);
            }
            var observer = new MutationObserver(checkDashboard);
            observer.observe(document.body, {childList: true, subtree: true, characterData: true});
            var poll = window.setInterval(checkDashboard, 200);
            checkDashboard();
            // A failed callback must never be reported as 100% ready.
            // Keep the last verified stage visible and offer a manual retry.
            var watchdog = window.setTimeout(function () {
                if (finished || stage === 5) return;
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
        html.H2([
            html.Span("🌊", style={"display": "inline-block", "width": "48px"}),
            "F2 Offshore Wind Farm Underwater Inspection"
        ], className="ocean-title", style={"marginBottom": "5px"}),

        html.Div(
            f"Engineering Scheduling & Progress Tracking System | "
            f"v2.0.0-rc.2 | "
            f"Schedule Data Updated: {data_updated}",
            id="schedule-update-status",
            className="ocean-subtitle",
            style={
                "color": "#6b7280",
                "fontSize": "12px",
                "marginTop": "3px",
                "fontWeight": "400",
                "lineHeight": "1.5",
                "paddingLeft": "48px"
            }
        ),
    
    
        ], style={"flex": "4", "minWidth": 0, "textAlign": "left"}),
        html.Div([
            html.Div("Overall Completed (All Years)", style={"fontSize": "12px", "color": "#cbd5e1"}),
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
                                 ("Resources", "resources"),
                                 ("Licenses", "licenses")]
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
    # Resources: links are rendered only after server-side session verification.
    html.Div([
        html.H3("Project Resources", style={"marginTop": "0", "marginBottom": "12px"}),
        html.Div(id="resources-access-content"),
    ], id="resources-page", style={"display": "none", "padding": "16px",
        "backgroundColor": "#203249", "borderRadius": "8px"}),
    # Licenses & Attribution — source inventory; terms must be reviewed per deployment.
    html.Div([
        html.H3("Licenses & Attribution", style={"marginTop": "0", "marginBottom": "8px"}),
        html.P("Third-party software, data sources, attribution requirements and usage notices.",
               style={"color": "#cbd5e1"}),
        html.H4("Data Sources & Services"),
        html.Ul([
            html.Li([html.A(name, href=url, target="_blank", rel="noopener noreferrer",
                            style={"color": "#9ed2ff"}), " — ", note],
                    style={"textAlign": "justify", "textAlignLast": "left"})
            for name, url, note in [
                ("Windy Point Forecast API", "https://account.windy.com/agreements/windy-forecast-api-terms-of-use",
                 "Proprietary API terms. Trial access is for development only; the testing API returns modified/shuffled data. Terms restrict storing forecast data, derivative databases and redistribution. This application stores Windy forecasts in PostgreSQL: obtain explicit authorization before continuing that use. Provider and original data-source attribution may also be required."),
                ("Open-Meteo", "https://open-meteo.com/en/terms",
                 "Forecast data: CC BY 4.0 (credit, license link and indication of modifications). Free hosted API: non-commercial use only; business use requires an applicable commercial plan. The API data license and the API service terms are distinct. Credit should also appear near displayed forecast data."),
                ("Copernicus Marine Service", "https://help.marine.copernicus.eu/en/articles/4444611-citing-copernicus-marine-products-and-services",
                 "Copernicus Marine Licence Agreement applies. Suggested derived-product credit: 'Generated using E.U. Copernicus Marine Service Information' plus the DOI(s) of the actual product(s). Product IDs/DOIs and any additional notices remain to be verified."),
                ("Ventusky", "https://embed.ventusky.com/",
                 "External embedded weather map. Use the provider's official embed service and preserve its attribution; review embedding and commercial-use terms for this deployment."),
                ("OpenStreetMap", "https://www.openstreetmap.org/copyright",
                 "Geographic data: ODbL; credit © OpenStreetMap contributors on the map itself. If using OSM Foundation tile servers, follow the separate tile usage policy; tile imagery and other basemaps may have additional terms.")
            ]
        ], style={"lineHeight": "1.85"}),
        html.H4("Third-Party Software", style={"marginTop": "18px"}),
        html.P("Licenses below describe the named projects; deployed package versions, transitive dependencies, bundled assets and any required copyright notices must be checked separately.",
               style={"color": "#cbd5e1"}),
        html.Ul([
            html.Li([html.A(name, href=url, target="_blank", rel="noopener noreferrer",
                            style={"color": "#9ed2ff"}), " — ", license_name],
                    style={"textAlign": "justify", "textAlignLast": "left"})
            for name, license_name, url in [
                ("Dash", "MIT", "https://github.com/plotly/dash/blob/dev/LICENSE"),
                ("Flask", "BSD-3-Clause", "https://github.com/pallets/flask/blob/main/LICENSE.txt"),
                ("Leaflet 1.9.4", "BSD-2-Clause", "https://github.com/Leaflet/Leaflet/blob/v1.9.4/LICENSE"),
                ("NumPy", "BSD-3-Clause; bundled third-party notices may apply", "https://github.com/numpy/numpy/blob/main/LICENSE.txt"),
                ("pandas", "BSD-3-Clause", "https://github.com/pandas-dev/pandas/blob/main/LICENSE"),
                ("Plotly.py", "MIT", "https://github.com/plotly/plotly.py/blob/main/LICENSE.txt"),
                ("psycopg2", "LGPL with exceptions", "https://www.psycopg.org/docs/license.html"),
                ("python-dotenv", "BSD-3-Clause", "https://github.com/theskumar/python-dotenv/blob/main/LICENSE"),
                ("Requests", "Apache-2.0", "https://github.com/psf/requests/blob/main/LICENSE"),
                ("SQLAlchemy", "MIT", "https://github.com/sqlalchemy/sqlalchemy/blob/main/LICENSE"),
            ]
        ], style={"lineHeight": "1.9"}),
        html.H4("Usage & Disclaimer", style={"marginTop": "18px"}),
        html.P("Weather forecasts, map layers and visualizations are informational and may be delayed, incomplete or inaccurate.",
               style={"textAlign": "justify", "textAlignLast": "left"}),
        html.P("Testing Data / API – Not for operational use.",
               style={"fontWeight": "700", "color": "#fbbf24", "textAlign": "left"}),
        html.P("Do not use this dashboard as the sole basis for offshore operations, navigation or safety-critical decisions."),
        html.P("Third-party names do not imply sponsorship or endorsement. This page is an attribution inventory, not certification of compliance or permission to store, redistribute or commercially use third-party data. Confirm the actual subscriptions, data products, deployment dependencies, LICENSE/NOTICE obligations and any necessary provider permissions before production or distribution.",
               style={"color": "#cbd5e1", "textAlign": "justify", "textAlignLast": "left"}),
        html.P("Licensing references reviewed: 2026-10-23 · Outstanding checks: Windy storage permission; Open-Meteo commercial API plan; Copernicus product DOIs; deployed dependency inventory.",
               style={"fontSize": "12px", "color": "#94a3b8", "textAlign": "left"}),
    ], id="licenses-page", style={"display": "none", "padding": "16px",
        "backgroundColor": "#203249", "borderRadius": "8px", "lineHeight": "1.6",
        "textAlign": "justify"})
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


def resource_login_view(error=False):
    """Dark input with a show/hide toggle; no credentials are stored in the browser."""
    return html.Div([
        html.P("IOG Employee Verification", style={"color": "#cbd5e1"}),
        html.P("Invalid employee code. Please try again.",
               style={"color": "#ffb4b4"}) if error else None,
        html.Div([
            dcc.Input(
                id="resource-employee-code", type="password",
                placeholder="Employee Code", debounce=False, autoComplete="off",
                style={"flex": "1", "minWidth": "0", "padding": "10px 12px",
                       "backgroundColor": "#101820", "color": "#ffffff",
                       "border": "1px solid #526479", "borderRadius": "6px",
                       "fontSize": "14px", "outline": "none"},
            ),
            html.Button(
                "Show", id="resource-visibility-button", n_clicks=0,
                title="Show or hide employee code", **{"aria-label": "Show or hide employee code"},
                style={"padding": "9px 13px", "backgroundColor": "#101820",
                       "color": "#ffffff", "border": "1px solid #526479",
                       "borderRadius": "6px", "cursor": "pointer"},
            ),
        ], style={"display": "flex", "gap": "8px", "maxWidth": "420px",
                  "alignItems": "stretch"}),
        html.Button(
            "Continue", id="resource-unlock-button", n_clicks=0,
            style={"marginTop": "12px", "padding": "9px 16px",
                   "backgroundColor": "#101820", "color": "#ffffff",
                   "border": "1px solid #526479", "borderRadius": "6px",
                   "cursor": "pointer"},
        ),
    ])


@app.callback(
    Output("resources-access-content", "children"),
    Input("main-navigation", "value"),
)
def show_resources_access(page):
    if page != "resources":
        return []
    if resource_access_valid():
        return resource_links_view()
    return resource_login_view()


@app.callback(
    Output("resource-employee-code", "type"),
    Output("resource-visibility-button", "children"),
    Input("resource-visibility-button", "n_clicks"),
    prevent_initial_call=True,
)
def toggle_resource_code_visibility(n_clicks):
    visible = bool(n_clicks and n_clicks % 2)
    return ("text" if visible else "password",
            "Hide" if visible else "Show")


@app.callback(
    Output("resources-access-content", "children", allow_duplicate=True),
    Input("resource-unlock-button", "n_clicks"),
    State("resource-employee-code", "value"),
    prevent_initial_call=True,
)
def unlock_resources(n_clicks, code):
    if not n_clicks:
        return no_update
    if not code or not any(secrets.compare_digest(code.strip(), item) for item in RESOURCE_CODES):
        return resource_login_view(error=True)
    session["resource_verified_at"] = time.time()
    return resource_links_view()

@app.callback(
    Output("overview-page", "style"),
    Output("schedule-page", "style"),
    Output("weather-page", "style"),
    Output("resources-page", "style"),
    Output("licenses-page", "style"),
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
            resources if page == "resources" else hidden,
            resources if page == "licenses" else hidden)

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
        "v2.0.0-rc.1 | "
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
#    app.run(
#        debug=True,
#        host="127.0.0.1",
#        port=8050
#    )    
    
    
#http://127.0.0.1:8050/