# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 11:19:51 2026

@author: ali.chang
"""

import pandas as pd
import plotly.express as px
from dash import Dash, dcc, html, Input, Output
from datetime import datetime
from dash import get_asset_url
from dash import State

#%%Load data
df = pd.read_excel("progress.xlsx")
df["Date"] = pd.to_datetime(df["Date"])
df["Start"] = df["Date"] - pd.Timedelta(hours=12)
df["End"] = df["Date"] + pd.Timedelta(hours=12)
df["Lane_Type"] = df["Operation"].apply(lambda x: "WORK" if x == "offshore" else "EVENT")
df["Cluster"] = (pd.to_numeric(df["Cluster"], errors="coerce"))
df["Lane"] = (df["Cluster"].apply(lambda x: str(int(x)) if pd.notna(x) else "NoCluster")+
              " | " +df["Task"].fillna("").astype(str))
#%%Task / Category list
task_list = df["Task"].unique()
cat_list = df["Category"].unique()
#%%Summary Table
projects = [
    {"name":"BeeX",
     "logo":"beex_logo.png",
     "start":datetime(2026,4,6).date()},
    {"name":"IOG",
     "logo":"iog_logo.png",
     "start":datetime(2026,4,18).date()}]

def build_summary_table():
    today = datetime.today().date()
    rows = []
    for p in projects:
        days = (today - p["start"]).days
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
wow_days = len(df[df["Category"]=="WOW"])
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
    # Title
    html.H2("🌊S2603BEX50 F2 Offshore Wind Farm Underwater Inspection"),
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
            "flexWrap": "wrap"
        }
    ),
    html.Br(),
    # Filters
    html.Div([
        html.Div([
            html.Label("Task Filter"),
            dcc.Dropdown(
                task_list,
                value=list(task_list),
                multi=True,
                id="task-filter"
            )
        ], style={"width": "48%", "display": "inline-block"}),
        html.Div([
            html.Label("Category Filter"),
            dcc.Dropdown(
                cat_list,
                value=list(cat_list),
                multi=True,
                id="cat-filter"
            )
        ], style={"width": "48%", "display": "inline-block"})
    ]),
    # Chart
    dcc.Graph(
        id="gantt-chart"
    )
])
#%%Color map
color_map = {
    "Inspection": "#1f77b4",         #藍
    "Data Processing": "#2ca02c",    #綠
    "Delay": "#d62728",              #紅
    "Day off": "#000000",            #黑
    "WOW": "#7f7f7f"                 #灰
}
#%%Callback (dynamic update)
@app.callback(
    Output("gantt-chart", "figure"),
    Input("task-filter", "value"),
    Input("cat-filter", "value")
)
def update_chart(selected_tasks, selected_cats):

    # =========================================================
    # 1. FILTER
    # =========================================================
    filtered = df[
        (df["Task"].isin(selected_tasks)) &
        (df["Category"].isin(selected_cats))
    ].copy().reset_index(drop=True)

    # =========================================================
    # 2. CLEAN DATA
    # =========================================================
    filtered["Cluster"] = pd.to_numeric(filtered["Cluster"], errors="coerce")
    filtered["Date_str"] = filtered["Date"].dt.strftime("%Y-%m-%d (%a)")

    # =========================================================
    # 3. ENGINEERING LANE STRUCTURE
    # =========================================================
    def lane(row):
        if pd.notna(row["Cluster"]):
            return f"Cluster {int(row['Cluster'])} | {row['Task']}"
        return f"{row['Category']} | {row['Task']}"

    filtered["Lane"] = filtered.apply(lane, axis=1)

    # =========================================================
    # 4. ENGINEERING ORDERING (stable + deterministic)
    # =========================================================
    cluster_order = sorted(filtered["Cluster"].dropna().unique())

    lane_order = []

    # Cluster first (engineering grouping)
    for c in cluster_order:
        tasks = filtered.loc[filtered["Cluster"] == c, "Task"].dropna().unique()
        for t in tasks:
            lane_order.append(f"Cluster {int(c)} | {t}")

    # Event section (fixed order)
    event_order = ["Inspection", "Data Processing", "WOW", "Day off"]

    for e in event_order:
        tasks = filtered.loc[filtered["Category"] == e, "Task"].dropna().unique()
        for t in tasks:
            lane_order.append(f"{e} | {t}")

    filtered["Lane"] = pd.Categorical(filtered["Lane"], categories=lane_order, ordered=True)
    filtered = filtered.sort_values(["Lane", "Start"])

    # =========================================================
    # 5. HOVER CLEAN (engineering style)
    # =========================================================
    hover_cols = [
        "Date_str",
        "Category",
        "Task",
        "Cluster",
        "Supervisor",
        "Pilot",
        "Tether Manager",
        "Assistant",
        "Remark"
    ]

    filtered[hover_cols] = filtered[hover_cols].fillna("")

    # =========================================================
    # 6. GANTT PLOT (engineering style)
    # =========================================================
    fig = px.timeline(
        filtered,
        x_start="Start",
        x_end="End",
        y="Lane",
        color="Category",
        color_discrete_map=color_map,
        custom_data=hover_cols
    )

    # =========================================================
    # 7. CLEAN HOVER TEMPLATE (engineering card)
    # =========================================================
    fig.update_traces(
        hovertemplate=
        "<b>%{customdata[2]}</b><br>"
        "Date: %{customdata[0]}<br>"
        "Category: %{customdata[1]}<br>"
        "Cluster: %{customdata[3]}<br>"
        "Supervisor: %{customdata[4]}<br>"
        "Pilot: %{customdata[5]}<br>"
        "Tether: %{customdata[6]}<br>"
        "Assistant: %{customdata[7]}<br>"
        "Note: %{customdata[8]}"
        "<extra></extra>"
    )

    # =========================================================
    # 8. ENGINEERING TIMELINE (today reference line)
    # =========================================================
    now = pd.Timestamp.now()

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
        y=1.04,
        yref="paper",
        text="TODAY",
        showarrow=False,
        font=dict(color="red")
    )

    # =========================================================
    # 9. ENGINEERING LAYOUT
    # =========================================================
    fig.update_layout(
        title="Engineering Gantt Dashboard",
        height=min(max(500, len(filtered) * 25), 600),
        margin=dict(l=180, r=30, t=60, b=40),
        dragmode="pan",  # engineering mode default
        xaxis=dict(
            type="date",
            rangeslider=dict(visible=True),
        ),
        yaxis=dict(
            autorange="reversed"  # MS Project style
        ),
        hoverlabel=dict(
            bgcolor="white",
            font_size=12,
            font_family="Arial"
        )
    )

    return fig

#%%Run server

server = app.server

if __name__ == "__main__":
    app.run(
        debug=True,
        host="0.0.0.0",
        port=8050
    )
#http://127.0.0.1:8050/