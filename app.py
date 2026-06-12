# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 10:24:21 2026

@author: ali.chang
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from dash import Dash, dcc, html, Input, Output
from datetime import datetime
from dash import get_asset_url


#%%Load data
df = pd.read_excel("progress.xlsx")
df["Date"] = pd.to_datetime(df["Date"])
df["Start"] = df["Date"] - pd.Timedelta(hours=12)
df["End"] = df["Date"] + pd.Timedelta(hours=12)
# Category dtype (Filter會快很多)
df["Task"] = df["Task"].astype("category")
df["Category"] = df["Category"].astype("category")
# Cluster
df["Cluster"] = pd.to_numeric(df["Cluster"],errors="coerce")
# Progress
df["Progress"] = (df["Progress"].astype(str).str.replace("%", "", regex=False))
df["Progress"] = (pd.to_numeric(df["Progress"],errors="coerce").fillna(0))
if df["Progress"].max() <= 1:
    df["Progress"] *= 100
df["Progress"] = (df["Progress"].round().astype(int))
# Date String
df["Date_str"] = (df["Date"].dt.strftime("%Y-%m-%d (%a)"))
# Hover
df["Cluster_hover"] = np.where(df["Cluster"].notna(),df["Cluster"].astype("Int64").astype(str),"")
df["Progress_hover"] = (df["Progress"].astype(str))
# Empty Text
for col in [
    "Supervisor",
    "Pilot",
    "Tether Manager",
    "Assistant",
    "Remark"]:
    if col in df.columns:
        df[col] = df[col].fillna("")
# Lane (預先建立)
df["Lane"] = np.where(
    df["Cluster"].notna(),
    "Cluster "
    + df["Cluster"].astype("Int64").astype(str)
    + " | "
    + df["Task"].astype(str),
    df["Category"].astype(str)
    + " | "
    + df["Task"].astype(str))
# print(df["Progress"].describe())
# print(df["Progress"].head(10))
#%%Task / Category list
task_list = list(df["Task"].unique())
cat_list = list(df["Category"].unique())
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
                        style={"height":"25px"}
                    )
                ),
                html.Td(p["name"]),
                html.Td(str(p["start"])),
                html.Td(str(today)),
                html.Td(
                    str(days),
                    style={
                        "color":"red",
                        "fontSize":"25px",
                        "fontWeight":"bold"
                    }
                )
            ])
        )
    return html.Table(
        [
            html.Tr([
                html.Th("Logo",style={"width":"20%"}),
                html.Th("Company",style={"width":"20%"}),
                html.Th("Start",style={"width":"20%"}),
                html.Th("Today",style={"width":"20%"}),
                html.Th("Days",style={"width":"20%"})
            ])
        ] + rows,
        style={
            "width":"100%",
            "tableLayout":"fixed",
            "textAlign":"center",
            "border":"1px solid black",
            "borderCollapse":"collapse"
        }
    )
#%%KPI Dashboard
inspection_days = len(df[df["Category"]=="Inspection"])
data_days = len(df[df["Category"]=="Data Processing"])
wow_days = len(df[df["Category"]=="WOW(offshore)"]) + len(df[df["Category"]=="WOW(onshore)"])
off_days = len(df[df["Category"]=="Day off"])
kpi_data = {
    "Inspection Days": inspection_days,
    "Data Processing Days": data_days,
    "WOW Days": wow_days,
    "Day Off Days": off_days}
def build_card(title, value):
    return html.Div([
    html.H2(
        value,
        style={
            "margin":"0",
            "fontSize":"20px",
            "color":"#1f77b4"
        }
    ),
    html.P(
        title,
        style={
            "margin":"5px 0 0 0"
        }
    )
    ],
    style={
        "border":"1px solid lightgray",
        "padding":"15px",
        "width":"200px",
        "display":"inline-block",
        "margin":"10px",
        "textAlign":"center"
    })
#%%App

app = Dash(__name__)
app.layout = html.Div([

    # =========================================================
    # 1. HEADER (工程系統標題)
    # =========================================================
    html.Div([
        html.H2("🌊S2603BEX50 F2 Offshore Wind Farm Underwater Inspection",
                style={"marginBottom": "5px"}),

        html.Div("Engineering Scheduling & Progress Tracking System",
                 style={"color": "gray", "fontSize": "14px"})
    ], style={"textAlign": "center", "marginBottom": "10px"}),

    # =========================================================
    # 2. SUMMARY TABLE + KPI (控制面板區)
    # =========================================================
    html.Div([

        # Summary Table
        build_summary_table(),

        # KPI Cards
        html.Div(
            [
                build_card(title, value)
                for title, value in kpi_data.items()
            ],
            style={
                "display": "flex",
                "justifyContent": "center",
                "flexWrap": "wrap",
                "gap": "10px",
                "marginTop": "10px"
            }
        )

    ], style={
        "backgroundColor": "#f9f9f9",
        "padding": "10px",
        "borderRadius": "8px",
        "marginBottom": "10px"
    }),

    # =========================================================
    # 3. TOOLBAR (工程 Filter 列)
    # =========================================================
    html.Div([

        html.Div([
            html.Label("Task Filter", style={"fontWeight": "bold"}),
            dcc.Dropdown(
                task_list,
                value=list(task_list),
                multi=True,
                id="task-filter"
            )
        ], style={"width": "49%", "display": "inline-block"}),

        html.Div([
            html.Label("Category Filter", style={"fontWeight": "bold"}),
            dcc.Dropdown(
                cat_list,
                value=list(cat_list),
                multi=True,
                id="cat-filter"
            )
        ], style={"width": "49%", "display": "inline-block"})

    ], style={
        "marginBottom": "10px",
        "padding": "10px",
        "backgroundColor": "white",
        "border": "1px solid #ddd",
        "borderRadius": "6px"
    }),

    # =========================================================
    # 4. GANTT VIEWER (主工程區)
    # =========================================================
    html.Div([

        dcc.Graph(
            id="gantt-chart",
            config={
                "displaylogo": False,
                "scrollZoom": True,
                "doubleClick": "reset"
            }
        )

    ], style={
        "backgroundColor": "white",
        "border": "1px solid #ddd",
        "borderRadius": "6px",
        "padding": "5px"
    })

])
#%%Color map
color_map = {
    "Inspection": "#1f77b4",         #藍
    "Data Processing": "#2ca02c",    #綠
    "WOW(offshore)": "#d62728",      #紅
    "Day off": "#000000",            #黑
    "WOW(onshore)": "#7f7f7f"        #灰
    
}
#%%Callback (dynamic update)
@app.callback(
    Output("gantt-chart", "figure"),
    Input("task-filter", "value"),
    Input("cat-filter", "value")
)
def update_chart(selected_tasks, selected_cats):

    # =========================================================
    # 0. SAFE DEFAULTS
    # =========================================================
    selected_tasks = selected_tasks or task_list
    selected_cats = selected_cats or cat_list

    now = pd.Timestamp.now(tz="Asia/Taipei").tz_localize(None)

    # =========================================================
    # 1. FILTER
    # =========================================================
    filtered = df.loc[
        (df["Task"].isin(selected_tasks)) &
        (df["Category"].isin(selected_cats))
    ].copy()


    # =========================================================
    # 4. ORDERING
    # =========================================================
    filtered["_cluster_sort"] = filtered["Cluster"].fillna(999)

    filtered = filtered.sort_values(
        ["_cluster_sort", "Task", "Start"],
        na_position="last"
    )

    lane_order = filtered["Lane"].drop_duplicates().tolist()

    filtered["Lane"] = pd.Categorical(
        filtered["Lane"],
        categories=lane_order,
        ordered=True
    )

    # =========================================================
    # 5. LANE PROGRESS
    # =========================================================
    filtered["_time_diff"] = (
        filtered["Start"]
        .sub(now)
        .abs()
    )
    
    idx = (
        filtered
        .groupby("Lane")["_time_diff"]
        .idxmin()
    )

    lane_summary = filtered.loc[idx]

    lane_progress_map = lane_summary.set_index("Lane")["Progress"].to_dict()

    ticktext = [
        f"{lane} ({lane_progress_map.get(lane,0)}%)"
        for lane in lane_order
    ]


    # =========================================================
    # 7. GANTT
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
    "Progress_hover"
    ]
    fig = px.timeline(
        filtered,
        x_start="Start",
        x_end="End",
        y="Lane",
        color="Category",
        color_discrete_map=color_map,
        custom_data=hover_cols
    )

    fig.update_traces(
        hovertemplate=
        "<b>%{customdata[2]}</b><br>"
        "Date: %{customdata[0]}<br>"
        "Category: %{customdata[1]}<br>"
        "Cluster: %{customdata[3]}<br>"
        "Progress: %{customdata[9]}%<br>"
        "Supervisor: %{customdata[4]}<br>"
        "Pilot: %{customdata[5]}<br>"
        "Tether: %{customdata[6]}<br>"
        "Assistant: %{customdata[7]}<br>"
        "Note: %{customdata[8]}"
        "<extra></extra>"
    )

    # =========================================================
    # 8. TODAY LINE
    # =========================================================
    fig.add_shape(
        type="line",
        x0=now,
        x1=now,
        y0=0,
        y1=1,
        yref="paper",
        line=dict(color="red", width=2, dash="dash")
    )

    fig.add_annotation(
        x=now,
        y=1.02,
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
    # 9. LAYOUT
    # =========================================================
    chart_height = min(
        max(650, len(lane_order) * 30),
        1600
    )

    fig.update_layout(
        title="Engineering Operations Scheduling Gantt Chart",
        height=chart_height,
        margin=dict(l=240, r=30, t=60, b=40),
        dragmode="pan",
        hoverlabel=dict(
            bgcolor="white",
            font_size=12
        ),
        uirevision="keep"
    )

    fig.update_yaxes(
        autorange="reversed",
        tickmode="array",
        tickvals=lane_order,
        ticktext=ticktext
    )

    return fig
#%%Run server

server = app.server

if __name__ == "__main__":
    app.run(
        debug=False,
        host="0.0.0.0",
        port=8050
    )
#http://127.0.0.1:8050/