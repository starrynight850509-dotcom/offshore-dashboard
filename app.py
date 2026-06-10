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


# =============================================================================
# Load data
# =============================================================================
df = pd.read_excel("progress.xlsx")
df["Date"] = pd.to_datetime(df["Date"])
df["Start"] = df["Date"] - pd.Timedelta(hours=12)
df["End"] = df["Date"] + pd.Timedelta(hours=12)
# =============================================================================
# Task / Category list
# =============================================================================
task_list = df["Task"].unique()
cat_list = df["Category"].unique()
#==============================================================================
#Summary Table
#==============================================================================
projects = [
    {
        "name":"BeeX",
        "logo":"beex_logo.png",
        "start":datetime(2026,4,6).date()
    },
    {
        "name":"IOG",
        "logo":"iog_logo.png",
        "start":datetime(2026,4,18).date()
    }
]

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
                        style={"height":"40px"}
                    )
                ),
                html.Td(p["name"]),
                html.Td(str(p["start"])),
                html.Td(str(today)),
                html.Td(
                    str(days),
                    style={
                        "color":"red",
                        "fontSize":"30px",
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
# =============================================================================
#KPI Dashboard
# =============================================================================
inspection_days = len(
    df[df["Category"]=="Inspection"]
)
data_days = len(
    df[df["Category"]=="Data Processing"]
)
wow_days = len(
    df[df["Category"]=="WOW"]
)
off_days = len(
    df[df["Category"]=="Day off"]
)

kpi_data = {
    "Inspection Days": inspection_days,
    "Data Processing Days": data_days,
    "WOW Days": wow_days,
    "Day Off Days": off_days
}

def build_card(title, value):
    return html.Div([
    html.H2(
        value,
        style={
            "margin":"0",
            "fontSize":"36px",
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
# =============================================================================
# App
# =============================================================================
app = Dash(__name__)
app.layout = html.Div([
    ##標題
    html.H2("🌊S2603BEX50 F2 Offshore Wind Farm Underwater Inspection"),
    ##Summery Table
    build_summary_table(),
    ##KPI Card
    html.Div(
    [
        build_card(title, value)
        for title, value in kpi_data.items()
    ],
    style={
        "display":"flex",
        "justifyContent":"center",
        "flexWrap":"wrap"
    }
    ),
    html.Br(),
    ##下拉選單
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
    ##圖表區
    dcc.Graph(id="gantt-chart")
])

# =============================================================================
# Color map
# =============================================================================
color_map = {
    "Inspection": "#1f77b4",         #藍
    "Data Processing": "#2ca02c",    #綠
    "Delay": "#d62728",              #紅
    "Day off": "#000000",            #黑
    "WOW": "#7f7f7f"                 #灰
}

# =============================================================================
# Callback (dynamic update)
# =============================================================================
@app.callback(
    Output("gantt-chart", "figure"),
    Input("task-filter", "value"),
    Input("cat-filter", "value")
)
def update_chart(selected_tasks, selected_cats):

    filtered = df[
        (df["Task"].isin(selected_tasks)) &
        (df["Category"].isin(selected_cats))
    ].copy()
    
    cols = [
    "Date","Category","Task","Cluster",
    "Supervisor","Pilot","Tether Manager",
    "Assistant","Remark"
    ]
    filtered[cols] = filtered[cols].fillna("")
    
    filtered["Date_str"] = (
    filtered["Date"]
    .dt.strftime("%Y-%m-%d (%a)")
    )
    
    fig = px.timeline(
        filtered,
        x_start="Start",
        x_end="End",
        y="Task",
        color="Category",
        color_discrete_map=color_map,
        custom_data=["Date_str", 
                    "Category", 
                    "Task",
                    "Cluster",
                    "Supervisor",
                    "Pilot",
                    "Tether Manager",
                    "Assistant",
                    "Remark",
                    "Start",
                    "End"
                    ]
    )

    fig.update_traces(
    hovertemplate=
    "Date: %{customdata[0]}<br>" +
    "Category: %{customdata[1]}<br>" +
    "Task: %{customdata[2]}<br>" +
    "Cluster: %{customdata[3]}<br>" +
    "Supervisor: %{customdata[4]}<br>" +
    "Pilot: %{customdata[5]}<br>" +
    "Tether Manager: %{customdata[6]}<br>" +
    "Assistant: %{customdata[7]}<br>" +
    "Remark: %{customdata[8]}<extra></extra>"
    )
    
    fig.update_yaxes(autorange="reversed")
    ##today line
    today_line = pd.Timestamp.now()
    fig.add_shape(
    type="line",
    x0=today_line,
    x1=today_line,
    y0=0,
    y1=1,
    yref="paper",
    line=dict(
        color="red",
        width=3,
        dash="dash"
    )
    )
    fig.add_annotation(
    x=today_line,
    y=1.03,
    yref="paper",
    text="Today",
    showarrow=False,
    font=dict(
        size=14,
        color="red"
    )
    )
    ##chart height control
    task_count = filtered["Task"].nunique()
    chart_height = min(
    max(650, task_count * 40),
    1200
    )
    
    fig.update_layout(
        title="Offshore Gantt Chart",
        height=chart_height,
        xaxis=dict(
            rangeslider=dict(
                visible=True
            )
        )        
    )
    return fig
# =============================================================================
# Run server
# =============================================================================
server = app.server

if __name__ == "__main__":
    app.run(
        debug=True,
        host="0.0.0.0",
        port=8050
    )
#http://127.0.0.1:8050/