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
from dash import Dash, dcc, html, Input, Output
from datetime import datetime
from zoneinfo import ZoneInfo
from dash import get_asset_url


#%%Load data
df = pd.read_pickle("progress.pkl")
hourly_df = pd.read_pickle("hourly.pkl")

data_updated = datetime.fromtimestamp(
    os.path.getmtime("progress.pkl"),
    tz=ZoneInfo("Asia/Taipei")
)
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

category_order = [
    "Inspection",
    "WOW(offshore)",
    "Data Processing",
    "WOW(onshore)",
    "Delay",
    "Day off",
]

cat_list = category_order
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
                        "fontSize":"20px",
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
            "width": "100%",                 # 表格寬度佔滿
            "tableLayout": "fixed",          # 固定欄寬，不依內容自動調整
            "textAlign": "center",           # 所有文字置中
            "border": "1px solid lightgray", # 表格外框為淺灰色 1px
            #"borderCollapse": "collapse",     # 合併相鄰儲存格邊框，避免雙線
            "borderRadius": "8px"
        }
    )
#%%KPI Dashboard
today = pd.Timestamp.now(tz="Asia/Taipei").tz_localize(None).normalize()
kpi_df = df[df["Date"].dt.normalize() <= today].copy()

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
def build_upcoming_tasks(days=7):
    today = pd.Timestamp.now(tz="Asia/Taipei").tz_localize(None).normalize()
    end_day = today + pd.Timedelta(days=days)

    upcoming = df[
        (df["Start"] >= today) &
        (df["Start"] < end_day)
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
                html.Td(r["Category"]),
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
            style={
                "width": "100%",
                "tableLayout": "auto",
                "textAlign": "center",
                "borderCollapse": "collapse",
                "fontSize": "12px"
            }
        )
    ], style={
        "backgroundColor": "#f9f9f9",
        "padding": "10px",
        "border": "1px solid lightgray",
        "borderRadius": "8px",
        "marginTop": "10px"
    })
#%%App
app = Dash(__name__)
app.layout = html.Div([
    # =========================================================
    # HEADER (工程系統標題)
    # =========================================================
    html.Div([
        html.H2("🌊S2603BEX50 F2 Offshore Wind Farm Underwater Inspection",
                style={"marginBottom": "5px"}),

        html.Div(f"Engineering Scheduling & Progress Tracking System | v1.0 Beta | Updated {data_updated:%Y-%m-%d %H:%M}",
                 style={"color": "gray", "fontSize": "14px"})
    ], style={"textAlign": "center", "marginBottom": "10px"}),
    # =========================================================
    # SUMMARY TABLE + KPI (控制面板區)
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
        ),
        # Upcoming Tasks
        build_upcoming_tasks(days=7)
    ], style={
        "backgroundColor": "#f9f9f9",
        "padding": "10px",
        "borderRadius": "8px",
        "marginBottom": "10px"
    }),
    # =========================================================
    # TOOLBAR (工程 Filter 列)
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
    # GANTT VIEWER (主工程區)
    # =========================================================
    html.Div([
        # 左側
        html.Div([
            dcc.Graph(
                id="gantt-chart",
                config={
                "scrollZoom": True,      # 滑鼠滾輪縮放
                "displayModeBar": True,  # 顯示 Plotly 工具列
                "doubleClick": "reset"   # 雙擊重設縮放
                }
            )
        ],
        style={
            "width":"80%",
            "display":"inline-block",
            "verticalAlign":"top"
        }),
        # 右側
        html.Div(
            id="detail-panel",
            children=[
                html.H4("Project Resources",
                style={
                        "margin":"0px",
                        "padding":"0px"
                    }
                ),
                html.A(
                    "Photo Archive",
                    href="https://iovtec.sharepoint.com/sites/IOGWindProject/Shared%20Documents/Forms/AllItems.aspx?id=%2Fsites%2FIOGWindProject%2FShared%20Documents%2F01%5FOperations%2F04%5FOngoing%20Projects%2F4372%5F%20Manpower%20for%20BeeX%2F2%2E%20Project%20Documentation%2F%E4%BD%9C%E6%A5%AD%E7%85%A7%E7%89%87&p=true&ct=1782188226845&or=Teams%2DHL&ga=1&LOF=1",
                    target="_blank",
                    style={
                        "display":"block",
                        "marginTop":"0px"
                    }
                ),
                html.Hr(style={"margin":"5px 0"}),
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
            style={
                "width": "16%",
                "display": "inline-block",
                "verticalAlign": "top",
                "padding": "15px",
                "backgroundColor": "#ffffff",
                "border": "1px solid #d1d5db",
                "borderRadius": "10px",
                "boxShadow": "0 1px 3px rgba(0,0,0,0.08)",
                "height": "550px",
                "overflowY": "auto",
                "marginTop": "70px"
            }
        )
    ])
])
#%%Color map
color_map = {
    "Inspection": "#1f77b4",         #藍
    "Data Processing": "#2ca02c",    #綠
    "WOW(offshore)": "#d62728",      #紅
    "Day off": "#000000",            #黑
    "WOW(onshore)": "#bdbdbd",       #灰
    "Delay": "#ff7f0e",              #橘
}
#%%Callback (dynamic update)
@app.callback(
    Output("gantt-chart", "figure"),
    Input("task-filter", "value"),
    Input("cat-filter", "value"),
    Input("date-picker", "date") )
def update_chart(selected_tasks, selected_cats, selected_date):
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
    
    always_show_mask = df["Category"].isin(always_show_categories)
    
    task_filter_mask = (
        df["Task"].isin(selected_tasks) &
        ~always_show_mask
    )
    
    filtered = df.loc[
        (task_filter_mask | always_show_mask) &
        (df["Category"].isin(selected_cats))
    ].copy()
    # =========================================================
    # ORDERING
    # =========================================================
    filtered["_cluster_sort"] = filtered["Cluster"].fillna(999)
    
    # Non-Operational 固定排最後
    filtered["_lane_sort"] = np.where(
        filtered["Lane"] == "Non-Offshore",
        9999,
        filtered["_cluster_sort"]
    )
    
    filtered = filtered.sort_values(
        ["_lane_sort", "Task", "Start"],
        na_position="last"
    )
    
    lane_order = filtered["Lane"].drop_duplicates().tolist()
    
    filtered["Lane"] = pd.Categorical(
        filtered["Lane"],
        categories=lane_order,
        ordered=True
    )
    # =========================================================
    # LANE PROGRESS
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
    
    no_progress_tasks = [
        "Non-Offshore",
    ]
    ticktext = []
    for lane in lane_order:
    
        if lane == "Non-Offshore":
            ticktext.append(lane)
            continue
    
        row = filtered.loc[
            filtered["Lane"] == lane
        ].iloc[0]
    
        progress = lane_progress_map.get(lane, 0)
    
        if progress < 50:
            color = "red"
        elif progress < 80:
            color = "orange"
        else:
            color = "green"
    
        ticktext.append(
            f"{lane} "
            f"<span style='color:{color};'><b>({progress}%)</b></span>"
        )
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
        "Category":category_order
        },
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
    # MILESTONES: MOB / DEMOB / REMOB
    # =========================================================
    milestones = [
        ("2026-04-06", "MOB", "#4b5563"),
        ("2026-06-19", "DEMOB", "#4b5563"),
        ("2026-08-01", "REMOB", "#4b5563"),]
    for date, label, color in milestones:
        date = (pd.Timestamp(date)+ pd.Timedelta(hours=12))
        fig.add_shape(
            type="line",
            x0=date,
            x1=date,
            y0=0,
            y1=1,
            yref="paper",
            line=dict(color=color,width=2,dash="dot")
        )
    
        fig.add_annotation(
            x=date,
            y=0.5,
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
        x1=pd.Timestamp("2026-07-31"),
        y0=0,
        y1=1,
        xref="x",
        yref="paper",
        fillcolor="#696969",
        opacity=0.40,
        layer="below",
        line_width=0
    )
    
    fig.add_annotation(
        x=pd.Timestamp("2026-07-11"),
        y=0.5,
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
            l=180,
            r=20,
            t=100,             # 上方空間增加
            b=5
        ),
        dragmode="pan",
    
        hoverlabel=dict(
            bgcolor="white",
            font_size=12
        ),
    
        uirevision="keep"
    )
    # =========================================================
    # SELECTED DATE ZOOM + HIGHLIGHT
    # =========================================================
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
            y=1.02,
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
    fig.update_yaxes(
    autorange="reversed",
    tickmode="array",
    tickvals=lane_order,
    ticktext=ticktext,
    title=None
    )
    fig.update_xaxes(
    rangeslider_visible=True,
    rangeslider=dict(thickness=0.02),
    fixedrange=False
    )
    return fig
#%% 點擊 Gantt bar 後，更新右側 detail-content 面板
@app.callback(
    Output("detail-content", "children"),
    Input("gantt-chart", "clickData")
)
def show_detail(clickData):
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
    # 從 clickData 取得被點擊 bar 的資料
    # customdata 對應 update_chart() 裡的 hover_cols 順序
    # =========================================================
    row = clickData["points"][0]["customdata"]
    date_str = row[0]
    category = row[1]
    task = row[2]
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
    timeline = hourly_df[
        (hourly_df["Task"] == task) &
        (hourly_df["Date"].dt.normalize() == date)
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
            # Task + Category
            html.Div([
                html.Div(
                    task,
                    style={
                        "fontSize": "16px",
                        "fontWeight": "bold",
                        "color": "#111827",
                        "marginBottom": "2px"
                    }
                ),
                html.Div(
                    category,
                    style={
                        "fontSize": "11px",
                        "color": "#6b7280"
                    }
                )
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
#%%Run server
# render佈署
server = app.server
if __name__ == "__main__":
    app.run(
        debug=False,
        host="0.0.0.0",
        port=8050
    )

# 本機測試
# server = app.server
# if __name__ == "__main__":
#     app.run(
#         debug=True,
#         host="127.0.0.1",
#         port=8050
#     )    
    
    
#http://127.0.0.1:8050/