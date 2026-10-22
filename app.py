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

def record_startup_step(label, elapsed):
    if label.startswith("Wave forecast · ") and not _STARTUP_INSTRUMENT_WAVE:
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
    
    database_url = os.getenv("DATABASE_URL")

    if not database_url:
        raise RuntimeError("DATABASE_URL is not configured")

    engine_started = time.perf_counter()
    schedule_engine = create_engine(
        database_url.replace(
            "postgresql://",
            "postgresql+psycopg2://",
            1
        ),
        pool_pre_ping=True
    )

    if _STARTUP_INSTRUMENT_SCHEDULE:
        record_startup_step("Schedule · Engine setup", time.perf_counter() - engine_started)

    try:
        connect_started = time.perf_counter()
        with schedule_engine.connect() as conn:
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

    finally:
        dispose_started = time.perf_counter()
        schedule_engine.dispose()
        if _STARTUP_INSTRUMENT_SCHEDULE:
            record_startup_step("Schedule · Dispose", time.perf_counter() - dispose_started)
@lru_cache(maxsize=2)
def load_schedule_cached(version):
    return load_schedule_data()

def get_schedule_updated_at():
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL is not configured")

    engine = create_engine(
        database_url.replace(
            "postgresql://",
            "postgresql+psycopg2://",
            1
        ),
        pool_pre_ping=True
    )

    try:
        with engine.connect() as conn:
            updated_at = conn.execute(text("""
                SELECT MAX(updated_at)
                FROM schedule_data
                WHERE name IN ('progress', 'hourly')
            """)).scalar()

        return updated_at
    finally:
        engine.dispose()
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

engine = create_engine(
    SQLALCHEMY_URL,
    pool_pre_ping=True
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
# ============================================================
# Windy forecast last update status
# ============================================================

_STARTUP_INSTRUMENT_WAVE = True
try:
    wave_df_status = measure_startup("Load wave forecast (Neon)", load_latest_wave_forecast)
finally:
    _STARTUP_INSTRUMENT_WAVE = False

# Both Neon startup branches have now run concurrently. Propagate schedule errors
# rather than letting the Dashboard start with missing or stale schedule data.
try:
    df, hourly_df, data_updated = _schedule_future.result()
finally:
    _STARTUP_INSTRUMENT_SCHEDULE = False
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

engine = create_engine(SQLALCHEMY_URL, pool_pre_ping=True)


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





F2_LEAFLET_MAP_HTML = '<!doctype html><html lang="zh-Hant"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>F2 Marine Wave Overlay — Prototype</title><link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"><style>body{margin:0;font:14px system-ui,-apple-system,sans-serif;background:#101827;color:#edf2f7}.wrap{max-width:1300px;margin:auto;padding:18px}.top{display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:12px}h2{margin:0 0 5px;font-size:21px}.muted{color:#aab8cc}#map{height:min(76vh,760px);min-height:460px;border-radius:12px;overflow:hidden;margin-top:8px}.controls{display:flex;align-items:center;gap:12px;flex-wrap:wrap;margin:10px 0}.controls button{background:#263c5a;color:white;border:1px solid #57708e;border-radius:6px;padding:8px 12px;cursor:pointer}input{accent-color:#52a4e6}#status{font-size:12px;padding:8px 0;color:#b9c8db}.tower-label{font-size:11px;font-weight:700;color:#102036;background:#fff;padding:2px 4px;border-radius:3px;box-shadow:0 1px 3px #0006;white-space:nowrap}.legend{background:white;color:#1b293b;padding:8px 10px;border-radius:5px;font:12px system-ui}.legend .bar{height:10px;width:175px;background:linear-gradient(90deg,#262f90,#2084c5,#32b88e,#ffe45e,#f28c39,#bd2d2d);margin:6px 0}.wave-key{background:#1b2b42;border:1px solid #48607e;border-radius:9px;padding:12px 14px;margin:12px 0 4px;max-width:620px}.wave-key-title{font-weight:700;margin-bottom:7px}.wave-key-bar{height:15px;border-radius:4px;background:linear-gradient(to right,#3b276d,#6c3f9a,#ae6595,#dd8b7d,#eeaa64,#f4d783,#f5f7a6)}.wave-key-ticks{display:flex;justify-content:space-between;font-size:12px;margin-top:5px}.wave-key-note{font-size:11px;color:#b9c8db;margin-top:7px;line-height:1.5}.note{line-height:1.6;font-size:12px;color:#aab8cc}\n.filter-row{margin-top:0}.filter-row .map-controls{max-width:none;box-shadow:none;background:transparent;border:0;padding:0;font-size:13px}.filter-row .map-controls input[type=range]{width:95px}\n/* Compact map-first interface. Controls are HTML overlays, not geographic layers. */\n#map{position:relative;height:min(82vh,800px);min-height:520px;margin-top:14px}\n.map-bottom{position:absolute;z-index:900;left:12px;right:12px;bottom:22px;display:flex;flex-direction:column;gap:8px;align-items:flex-start;pointer-events:none}\n.map-controls{display:flex;flex-wrap:wrap;align-items:center;gap:8px 12px;background:rgba(16,24,39,.93);color:#edf2f7;border:1px solid #56708c;border-radius:9px;padding:9px 11px;box-shadow:0 3px 14px #0006;pointer-events:auto;max-width:calc(100% - 22px);font-size:12px}\n.map-controls label{display:flex;align-items:center;gap:5px;white-space:nowrap}\n.map-controls input[type=range]{width:80px}\n.map-controls button{background:#263c5a;color:white;border:1px solid #57708e;border-radius:6px;padding:6px 9px;cursor:pointer}\n.map-bottom .wave-key{margin:0;width:min(420px,calc(100% - 22px));max-width:420px;box-sizing:border-box;background:rgba(16,24,39,.94);color:#edf2f7;border:1px solid #56708c;padding:10px 13px;pointer-events:auto}\n.wave-key-title{font-size:12px}.key-hint{font-size:10px;font-weight:400;color:#b9c8db}\n.wave-key-bar{height:14px;margin-top:6px;border-radius:4px;background:linear-gradient(90deg,#0638ff 0%,#0286b7 20%,#44a75d 40%,#c4cf52 58%,#ffbd36 77%,#ff3900 100%)}.wave-key-ticks{font-size:11px;margin-top:5px}.wave-key-title{font-weight:700}\n#status{margin-top:5px}\n@media(max-width:650px){.map-bottom{bottom:18px;left:7px;right:7px}.map-controls{gap:5px 8px;padding:7px;font-size:11px}.map-controls input[type=range]{width:65px}#map{min-height:570px}}\n/* Dashboard iframe: prevent nested scrollbars. */\nhtml,body{height:100%;overflow:hidden}\n.wrap{box-sizing:border-box;height:100vh;max-width:none;padding:12px 16px;display:flex;flex-direction:column;min-height:0}\n.filter-row{flex:0 0 auto}\n#map{flex:1 1 auto;height:auto!important;min-height:0!important;margin-top:10px}\n#status{flex:0 0 auto;padding:4px 0;margin:0}\n@media(max-width:650px){#map{min-height:0!important}}\n</style></head><body><div class="wrap"><div class="filter-row"><div class="map-controls"><label><input id="waves" type="checkbox" checked> Waves</label><label>Opacity <input id="opacity" type="range" min="0" max="100" value="68"> <span id="opacityLabel">68%</span></label><label><input id="smooth" type="checkbox" checked> Smooth</label><label>Blur <input id="blur" type="range" min="0" max="24" value="12"> <span id="blurLabel">12px</span></label><label><input id="labels" type="checkbox" checked> Tower IDs</label><button id="fit">Fit towers</button></div></div><div id="map"><div class="map-bottom"><section class="wave-key" aria-label="Wave height color scale"><div class="wave-key-title">Hs (m)</div><div class="wave-key-bar" aria-hidden="true"></div><div class="wave-key-ticks"><span>0.5</span><span>1.5</span><span>2.5</span><span>3.5</span><span>4.5</span><span>5.5</span><span>6.5</span></div><div class="wave-key-note">Approximate color guide · refer to Copernicus for exact values</div></section></div></div><div id="status">Initializing map…</div></div><script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script><script>\nconst towers = [{"id": "D01", "lat": 24.66391667000001, "lon": 120.7168333}, {"id": "D02", "lat": 24.65530555999999, "lon": 120.7318056}, {"id": "D03", "lat": 24.64747221999998, "lon": 120.7228333}, {"id": "D04", "lat": 24.64522221999999, "lon": 120.7052222}, {"id": "D05", "lat": 24.63769444, "lon": 120.7075278}, {"id": "D06", "lat": 24.62544444000002, "lon": 120.6911944}, {"id": "D07", "lat": 24.67158333000002, "lon": 120.7114167}, {"id": "D08", "lat": 24.65661111000002, "lon": 120.7069722}, {"id": "D09", "lat": 24.65463888999999, "lon": 120.6974167}, {"id": "D10", "lat": 24.64388889000002, "lon": 120.6948889}, {"id": "D11", "lat": 24.63536110999998, "lon": 120.6955278}, {"id": "D12", "lat": 24.63119443999999, "lon": 120.6858056}, {"id": "C01", "lat": 24.69377778000002, "lon": 120.7689167}, {"id": "C02", "lat": 24.70077778000003, "lon": 120.7594167}, {"id": "C03", "lat": 24.71761111000002, "lon": 120.7624722}, {"id": "C04", "lat": 24.71572222000001, "lon": 120.7511667}, {"id": "C05", "lat": 24.70413888999999, "lon": 120.7493333}, {"id": "C06", "lat": 24.70663889000001, "lon": 120.7420278}, {"id": "C07", "lat": 24.68630555999999, "lon": 120.7566389}, {"id": "C08", "lat": 24.67605556000003, "lon": 120.7424722}, {"id": "C09", "lat": 24.68166666999998, "lon": 120.7372778}, {"id": "C10", "lat": 24.69175000000003, "lon": 120.7472778}, {"id": "C11", "lat": 24.69522221999999, "lon": 120.7392222}, {"id": "B01", "lat": 24.70952777999999, "lon": 120.7708611}, {"id": "B02", "lat": 24.72827777999997, "lon": 120.7743056}, {"id": "B03", "lat": 24.75319444000003, "lon": 120.7887778}, {"id": "B04", "lat": 24.76647222, "lon": 120.7979722}, {"id": "B05", "lat": 24.77286111, "lon": 120.81475}, {"id": "B06", "lat": 24.77777778000002, "lon": 120.8079722}, {"id": "B07", "lat": 24.72738889, "lon": 120.7591111}, {"id": "B08", "lat": 24.73808333000001, "lon": 120.7606944}, {"id": "B09", "lat": 24.74394443999999, "lon": 120.7711944}, {"id": "B10", "lat": 24.76277778, "lon": 120.7811667}, {"id": "B11", "lat": 24.77411110999999, "lon": 120.7920278}, {"id": "B12", "lat": 24.78300000000001, "lon": 120.8006667}, {"id": "A01", "lat": 24.70427778000001, "lon": 120.7805}, {"id": "A02", "lat": 24.72011111, "lon": 120.7934167}, {"id": "A03", "lat": 24.72658333000002, "lon": 120.8059444}, {"id": "A04", "lat": 24.73397221999999, "lon": 120.7993056}, {"id": "A05", "lat": 24.74666667000001, "lon": 120.8159722}, {"id": "A06", "lat": 24.75655556, "lon": 120.8277778}, {"id": "A07", "lat": 24.71897222000001, "lon": 120.7725278}, {"id": "A08", "lat": 24.72305556000002, "lon": 120.7840556}, {"id": "A09", "lat": 24.73747222000001, "lon": 120.7859722}, {"id": "A10", "lat": 24.74416666999999, "lon": 120.79475}, {"id": "A11", "lat": 24.75399999999998, "lon": 120.8025}, {"id": "A12", "lat": 24.75513889, "lon": 120.8121944}];\nL.DomEvent.disableClickPropagation(document.querySelector(\'.map-bottom\'));L.DomEvent.disableScrollPropagation(document.querySelector(\'.map-bottom\'));\nconst map=L.map(\'map\',{preferCanvas:true}).setView([24.70843,120.75996],11);\nL.tileLayer(\'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png\',{maxZoom:19,attribution:\'© OpenStreetMap contributors\'}).addTo(map);\nconst group=L.featureGroup();const labels=[];\nfor(const t of towers){const marker=L.circleMarker([t.lat,t.lon],{radius:5,color:\'#142a4d\',weight:1.5,fillColor:\'#ffca36\',fillOpacity:1}).bindPopup(\'<b>\'+t.id+\'</b><br>Latitude: \'+t.lat.toFixed(8)+\'<br>Longitude: \'+t.lon.toFixed(8));marker.addTo(group);const label=L.marker([t.lat,t.lon],{icon:L.divIcon({className:\'\',html:\'<span class="tower-label">\'+t.id+\'</span>\',iconSize:[45,18],iconAnchor:[-8,9]}),interactive:false}).addTo(group);labels.push(label)}\ngroup.addTo(map);map.fitBounds(group.getBounds().pad(.12));\n// Copernicus Marine global wave forecast: dynamic WMTS tiles in Web Mercator.\n// Dataset IDs and supported matrix sets can change; validate against GetCapabilities.\nconst layerId=\'GLOBAL_ANALYSISFORECAST_WAV_001_027/cmems_mod_glo_wav_anfc_0.083deg_PT3H-i_202411/VHM0\';\nconst tileUrl=\'https://wmts.marine.copernicus.eu/teroWmts/?service=WMTS&version=1.0.0&request=GetTile&format=image%2Fpng&layer=\'+encodeURIComponent(layerId)+\'&tilematrixset=EPSG:3857&tilematrix={z}&tilecol={x}&tilerow={y}&STYLE=cmap:rainbow,range=0/5\';\nmap.createPane(\'wavePane\');map.getPane(\'wavePane\').style.zIndex=\'350\';map.getPane(\'wavePane\').style.pointerEvents=\'none\';\nconst waves=L.tileLayer(tileUrl,{pane:\'wavePane\',opacity:.68,maxNativeZoom:10,maxZoom:19,tileSize:256,attribution:\'Copernicus Marine Service\',errorTileUrl:\'\'});\nlet good=0,bad=0;const status=document.getElementById(\'status\');\nfunction updateStatus(){status.textContent=\'KML: \'+towers.length+\' turbines loaded · Copernicus tiles: \'+good+\' loaded, \'+bad+\' failed\'+(bad&&!good?\' · Check WMTS layer ID, availability, or browser network\':\'\');}\nwaves.on(\'tileload\',()=>{good++;updateStatus()});waves.on(\'tileerror\',()=>{bad++;updateStatus()});\nwaves.addTo(map);group.bringToFront();updateStatus();\nfunction updateSmoothing(){const active=document.getElementById(\'smooth\').checked;const px=Number(document.getElementById(\'blur\').value);map.getPane(\'wavePane\').style.filter=active?\'blur(\'+px+\'px)\':\'none\';document.getElementById(\'blurLabel\').textContent=px+\'px\';}\ndocument.getElementById(\'smooth\').onchange=updateSmoothing;document.getElementById(\'blur\').oninput=updateSmoothing;updateSmoothing();\ndocument.getElementById(\'waves\').onchange=e=>{if(e.target.checked){waves.addTo(map);group.bringToFront()}else map.removeLayer(waves)};\ndocument.getElementById(\'opacity\').oninput=e=>{waves.setOpacity(Number(e.target.value)/100);document.getElementById(\'opacityLabel\').textContent=e.target.value+\'%\'};\ndocument.getElementById(\'labels\').onchange=e=>{labels.forEach(l=>e.target.checked?group.addLayer(l):group.removeLayer(l))};\ndocument.getElementById(\'fit\').onclick=()=>map.fitBounds(group.getBounds().pad(.18));\n\n</script></body></html>'

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

app.layout = html.Div([
    # =========================================================
    # Schedule Data Refresh
    # =========================================================
    # Only change this store when Neon schedule data actually changes.
    dcc.Store(id="schedule-version-store"),
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
    
        # Windy forecast 最後更新時間
        # 內容由 callback 自動更新
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
        )
    
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
                    {"label": "Windy", "value": "windy"},
                    {"label": "Copernicus Marine WMTS", "value": "copernicus"},
                ],
                value="windy",
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
            html.H3("Copernicus Marine WMTS", style={"margin": "0 0 12px"}),
            html.Iframe(
                src="/f2-marine-wave-map",
                style={"width": "100%", "height": "min(85vh, 820px)", "minHeight": "560px",
                       "border": "0", "borderRadius": "8px"},
                title="Copernicus Marine F2 Wave Map",
            ),
        ], id="weather-copernicus-view", style={"display": "none"}),
    ], id="weather-page", style={"display": "none", "padding": "16px",
        "backgroundColor": "#203249", "borderRadius": "8px"}),
    # Dedicated Resources page: SharePoint shortcuts.
    html.Div([
        html.H3("Project Resources", style={"marginTop": "0", "marginBottom": "12px"}),
        html.A('Photo Archive', href='https://iovtec.sharepoint.com/sites/IOGWindProject/Shared%20Documents/Forms/AllItems.aspx?id=%2Fsites%2FIOGWindProject%2FShared%20Documents%2F01%5FOperations%2F04%5FOngoing%20Projects%2F4372%5F%20Manpower%20for%20BeeX%2F2%2E%20Project%20Documentation%2F%E4%BD%9C%E6%A5%AD%E7%85%A7%E7%89%87&p=true&ct=1782188226845&or=Teams%2DHL&ga=1&LOF=1', target="_blank",
               rel="noopener noreferrer",
               style={"display": "block", "padding": "8px 0", "color": "#9ed2ff"}),
        html.A('Duty Roster', href='https://iovtec.sharepoint.com/sites/SV-SurveyDept2/Shared%20Documents/Forms/AllItems.aspx?id=%2Fsites%2FSV%2DSurveyDept2%2FShared%20Documents%2F%E6%8E%A2%E6%B8%AC%E9%83%A8%2D%E7%8F%BE%E5%A0%B4%E7%87%9F%E9%81%8B%E7%B5%84%2F%E7%8F%AD%E8%A1%A8&p=true&ga=1', target="_blank",
               rel="noopener noreferrer",
               style={"display": "block", "padding": "8px 0", "color": "#9ed2ff"}),
        html.A('Survey Log', href='https://iovtec.sharepoint.com/sites/SV-SurveyDept2/Shared%20Documents/Forms/AllItems.aspx?id=%2Fsites%2FSV%2DSurveyDept2%2FShared%20Documents%2F%E6%8E%A2%E6%B8%AC%E9%83%A8%2D%E7%8F%BE%E5%A0%B4%E7%87%9F%E9%81%8B%E7%B5%84%2FSurvey%20Log&viewid=a9aaccb6%2Dfaa4%2D446d%2Db6e3%2Dbeafdbfc933e', target="_blank",
               rel="noopener noreferrer",
               style={"display": "block", "padding": "8px 0", "color": "#9ed2ff"}),
        html.A('DPR', href='https://iovtec-my.sharepoint.com/:f:/g/personal/ali_chang_iovtec_com_tw/IgDMHGM8MshVQqUP_kgzjh4LAfee-_a7WUBFEz2pLI2bgXE?e=E5T8b7', target="_blank",
               rel="noopener noreferrer",
               style={"display": "block", "padding": "8px 0", "color": "#9ed2ff"}),
    ], id="resources-page", style={"display": "none", "padding": "16px",
        "backgroundColor": "#203249", "borderRadius": "8px"})
], className="ocean-app")

@app.callback(
    Output("weather-windy-view", "style"),
    Output("weather-copernicus-view", "style"),
    Input("weather-map-source", "value"),
)
def switch_weather_map(source):
    return ({"display": "none"}, {}) if source == "copernicus" else ({}, {"display": "none"})


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
                    color="#111827",
                    width=1,
                    dash="3px,2px"
                ),
                
                marker=dict(
                    color="#2563EB",
                    size=3
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
                showlegend=False
            )
        )
        # ---------------------------------------------------------
        # Wave Hs horizontal reference lines
        # 0, 1, 2, 3, 4, 5 m
        # ---------------------------------------------------------
        for hs_level in [0, 1, 2, 3, 4, 5]:
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
    
            # 固定顯示 0–3 m
            range=[0, 5],
    
            # 固定刻度，閱讀比較直觀
            tickmode="array",
            tickvals=[0, 1, 2, 3, 4, 5],
    
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
#%% KPI Auto Refresh
@app.callback(
    Output("kpi-container", "children"),
    Input("schedule-version-store", "data")
)
def update_kpi(schedule_version):

    if schedule_version is None:
        return no_update
    latest_df, _, _ = load_schedule_cached(schedule_version)

    latest_kpi = calculate_kpi(latest_df)

    return [
        build_card(title, value)
        for title, value in latest_kpi.items()
    ]
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
    Input("schedule-version-store", "data")
)
def update_overall_progress(schedule_version):
    if schedule_version is None:
        return no_update, no_update, no_update
    completed, total, project_completed = cached_completion_counts(schedule_version)
    percentage = (100 * completed / total) if total else 0
    return f"{percentage:.1f}%", f"{completed} / {total} Jackets", f" | Completed Jackets: {project_completed}"

#%% Schedule Updated Time Auto Refresh
@app.callback(
    Output("schedule-update-status", "children"),
    Input("schedule-version-store", "data")
)
def update_schedule_status(schedule_version):

    if schedule_version is None:
        return no_update
    _, _, latest_updated = load_schedule_cached(schedule_version)

    return (
        "Engineering Scheduling & Progress Tracking System | "
        "v1.5.1 Beta | "
        f"Schedule Data Updated: {latest_updated}"
    )
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