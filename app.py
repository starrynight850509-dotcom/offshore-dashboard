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
today = datetime.today().date()

beex_start = datetime(2026,4,6).date()
iog_start  = datetime(2026,4,18).date()

beex_days = (today - beex_start).days
iog_days = (today - iog_start).days

summary_table = html.Table([
    html.Tr([
        html.Th(""),
        html.Th("Start"),
        html.Th("Today"),
        html.Th("Cumulative Days")
    ]),
    html.Tr([
        html.Td(
            html.Img(
                src=get_asset_url("beex_logo.png"),
                style={"height":"40px"}
            )
        ),
        html.Td("2026/4/6"),
        html.Td(str(today)),
        html.Td(
            str(beex_days),
            style={
                "color":"red",
                "fontSize":"30px",
                "fontWeight":"bold"
            }
        )
    ]),
    html.Tr([
        html.Td(
            html.Img(
                src=get_asset_url("iog_logo.png"),
                style={"height":"40px"}
            )
        ),
        html.Td("2026/4/18"),
        html.Td(str(today)),
        html.Td(
            str(iog_days),
            style={
                "color":"red",
                "fontSize":"30px",
                "fontWeight":"bold"
            }
        )
    ])
],
style={
    "width":"100%",
    "textAlign":"center",
    "border":"1px solid black"
})
# =============================================================================
# 1. Load data
# =============================================================================
df = pd.read_excel("progress.xlsx")
df["Date"] = pd.to_datetime(df["Date"])

# =============================================================================
# 2. Task / Category list
# =============================================================================
task_list = df["Task"].unique()
cat_list = df["Category"].unique()

# =============================================================================
# 3. App
# =============================================================================
app = Dash(__name__)

app.layout = html.Div([
    ##標題
    html.H2("🌊S2603BEX50 F2 Offshore Wind Farm Underwater Inspection"),
    summary_table,
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
# 4. Color map
# =============================================================================
color_map = {
    "Inspection": "#1f77b4",         #藍
    "Data Processing": "#2ca02c",    #綠
    "Delay": "#d62728",              #紅
    "Day off": "#000000",            #黑
    "WOW": "#7f7f7f"                 #灰
}

# =============================================================================
# 5. Callback (dynamic update)
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
    ]

    filtered["Start"] = filtered["Date"] - pd.Timedelta(hours=12)
    filtered["End"] = filtered["Date"] + pd.Timedelta(hours=12)

    fig = px.timeline(
        filtered,
        x_start="Start",
        x_end="End",
        y="Task",
        color="Category",
        color_discrete_map=color_map,
        hover_data={"Start": False,
                    "End": False,
                    "Date": True, 
                    "Category": True, 
                    "Task": True,
                    "Remark": True
                    }
    )

    fig.update_yaxes(autorange="reversed")

    fig.update_layout(
        title="Offshore Gantt Chart",
        height=650
    )

    return fig

# =============================================================================
# 6. Run server
# =============================================================================
server = app.server

if __name__ == "__main__":
    app.run(
        debug=True,
        host="0.0.0.0",
        port=8050
    )
#http://127.0.0.1:8050/