# -*- coding: utf-8 -*-
"""Prepare F2 schedule data and upload progress, hourly and pie-chart data to Neon."""
import os
from datetime import datetime, time, timedelta
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

ACTIVITY_COLUMNS = [
    'Transit', 'Mob / Demob', 'ROV Operational in Water',
    'Downtime', 'Administration', 'ROV Maintenance'
]


def duration_minutes(value):
    """Interpret Excel clock-like HH:MM:SS entries as elapsed durations."""
    if pd.isna(value):
        return 0
    if isinstance(value, (datetime, time)):
        return value.hour * 60 + value.minute + value.second / 60
    if isinstance(value, (pd.Timedelta, timedelta)):
        return value.total_seconds() / 60
    if isinstance(value, (int, float, np.integer, np.floating)):
        # Excel numeric time fractions are expressed as fractions of a day.
        return float(value) * 1440
    value = str(value).strip()
    if not value:
        return 0
    parts = value.split(':')
    if len(parts) in (2, 3) and all(part.isdigit() for part in parts):
        hours, minutes = int(parts[0]), int(parts[1])
        seconds = int(parts[2]) if len(parts) == 3 else 0
        if minutes >= 60 or seconds >= 60:
            raise ValueError(f'Invalid duration: {value}')
        return hours * 60 + minutes + seconds / 60
    raise ValueError(f'Unsupported duration: {value!r}')


def prepare_data():
    df = pd.read_excel('progress.xlsx')
    df['Date'] = pd.to_datetime(df['Date'])
    df['Start'] = df['Date']
    df['End'] = df['Date'] + pd.Timedelta(hours=24)
    df['Task'] = df['Task'].astype('category')
    df['Category'] = df['Category'].astype('category')
    df['Cluster'] = pd.to_numeric(df['Cluster'], errors='coerce')
    df['Progress'] = pd.to_numeric(
        df['Progress'].astype(str).str.replace('%', '', regex=False), errors='coerce'
    ).fillna(0)
    if df['Progress'].max() <= 1:
        df['Progress'] *= 100
    df['Progress'] = df['Progress'].round().astype(int)
    df['Date_str'] = df['Date'].dt.strftime('%Y-%m-%d (%a)')
    df['Cluster_hover'] = np.where(
        df['Cluster'].notna(), df['Cluster'].astype('Int64').astype(str), ''
    )
    df['Progress_hover'] = df['Progress'].astype(str)
    for col in ['Supervisor', 'Pilot', 'Tether Manager', 'Assistant', 'Remark']:
        if col in df.columns:
            df[col] = df[col].fillna('')
    non_operational_categories = ['WOW(onshore)', 'Delay', 'Day off', 'Data Processing']
    df['Lane'] = np.where(
        df['Category'].isin(non_operational_categories),
        'Non-Offshore',
        'Cluster ' + df['Cluster'].astype('Int64').astype(str) + ' | ' + df['Task'].astype(str)
    )
    df.to_pickle('progress.pkl')

    hourly_df = pd.read_excel('hourly.xlsx')
    hourly_df['Date'] = pd.to_datetime(hourly_df['Date'])
    hourly_df.to_pickle('hourly.pkl')

    pie_df = pd.read_excel('pie_chart.xlsx')
    pie_df.columns = pie_df.columns.astype(str).str.strip()
    if 'Date' not in pie_df.columns and 'Unnamed: 0' in pie_df.columns:
        pie_df = pie_df.rename(columns={'Unnamed: 0': 'Date'})
    missing = [col for col in ['Date', *ACTIVITY_COLUMNS] if col not in pie_df.columns]
    if missing:
        raise ValueError(f'pie_chart.xlsx missing columns: {missing}')
    pie_df['Date'] = pd.to_datetime(pie_df['Date'], errors='raise')
    if pie_df['Date'].isna().any():
        raise ValueError('pie_chart.xlsx contains empty dates')
    if pie_df['Date'].duplicated().any():
        raise ValueError('pie_chart.xlsx contains duplicate dates')
    for col in ACTIVITY_COLUMNS:
        pie_df[col] = pie_df[col].map(duration_minutes).round(4)
    pie_df.to_pickle('pie_chart.pkl')

    pd.to_pickle({
        'updated': pd.Timestamp.now(tz='Asia/Taipei').strftime('%Y-%m-%d %H:%M')
    }, 'metadata.pkl')
    print(f'Pickle files generated successfully! Pie chart: {len(pie_df)} dates.')


def upload_schedule_to_neon():
    load_dotenv()
    database_url = os.getenv('DATABASE_URL')
    if not database_url:
        raise RuntimeError('DATABASE_URL is not configured')
    engine = create_engine(database_url)
    try:
        payloads = {
            name: pd.read_pickle(f'{name}.pkl').to_json(orient='records', date_format='iso')
            for name in ('progress', 'hourly', 'pie_chart')
        }
        with engine.begin() as conn:
            conn.execute(text('''
                CREATE TABLE IF NOT EXISTS schedule_data (
                    name TEXT PRIMARY KEY,
                    data JSONB NOT NULL,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
            '''))
            for name, json_data in payloads.items():
                conn.execute(text('''
                    INSERT INTO schedule_data (name, data, updated_at)
                    VALUES (:name, CAST(:data AS JSONB), NOW())
                    ON CONFLICT (name) DO UPDATE SET
                        data = EXCLUDED.data,
                        updated_at = EXCLUDED.updated_at
                '''), {'name': name, 'data': json_data})
    finally:
        engine.dispose()
    print('Progress, hourly and pie_chart uploaded to Neon successfully!')


if __name__ == '__main__':
    prepare_data()
    upload_schedule_to_neon()
