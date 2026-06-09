# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 11:19:51 2026

@author: ali.chang
"""

import pandas as pd
import plotly.express as px
from dash import Dash, dcc, html, Input, Output

# =========================
# 1. Load data
# =========================
df = pd.read_excel("progress.xlsx")
df["Date"] = pd.to_datetime(df["Date"])

# =========================
# 2. Task / Category list
# =========================
task_list = df["Task"].unique()
cat_list = df["Category"].unique()

# =========================
# 3. App
# =========================
app = Dash(__name__)

app.layout = html.Div([
    html.H2("🌊 Offshore Gantt Monitoring System"),

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

    dcc.Graph(id="gantt-chart")
])

# =========================
# 4. Color map
# =========================
color_map = {
    "Inspection": "#1f77b4",
    "Data Processing": "#2ca02c",
    "Delay": "#d62728",
    "Day off": "#000000",
    "WOW": "#7f7f7f"
}

# =========================
# 5. Callback (dynamic update)
# =========================
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

    filtered["Start"] = filtered["Date"]
    filtered["End"] = filtered["Date"] + pd.Timedelta(days=1)

    fig = px.timeline(
        filtered,
        x_start="Start",
        x_end="End",
        y="Task",
        color="Category",
        color_discrete_map=color_map,
        hover_data=["Date", "Category", "Task"]
    )

    fig.update_yaxes(autorange="reversed")

    fig.update_layout(
        title="Offshore Gantt Chart (Live Dashboard)",
        height=650
    )

    return fig

# =========================
# 6. Run server
# =========================
server = app.server

if __name__ == "__main__":
    app.run(
        debug=True,
        host="0.0.0.0",
        port=8050
    )