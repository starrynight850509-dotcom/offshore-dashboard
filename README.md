# Offshore Scheduling & Progress Dashboard

**Engineering Scheduling & Progress Tracking System**  
**Status:** v1.5.0 Beta · Internal project support tool

A Python-based dashboard for visualizing offshore inspection schedules, tracking project progress, and reviewing operational information in one place. Built to reduce repetitive spreadsheet work and make schedule information easier to explore.

> **Disclaimer:** This is a project-support and visualization tool under development. Wave forecasts are model predictions, not on-site measurements, and **must not be used as the sole basis for operational or safety decisions**.

## Features

- **Interactive Gantt chart** — Explore inspection activities and other schedule categories, filter tasks, zoom, pan, and jump to a date.
- **Task details** — Select a Gantt bar to inspect the associated information.
- **Project overview** — Summary table, KPI cards, and upcoming tasks.
- **Wave forecast** — Windy Point Forecast integration, including significant wave height, wave period, and direction; an indicative 1.5 m reference line is shown.
- **Automatic schedule refresh** — Check for schedule updates every 60 seconds and refresh relevant views when the underlying data version changes.
- **Site analytics** — Password-protected `/admin` page with visitor/session summaries and recent-session filters.

## Technology Stack

| Layer | Technology |
| --- | --- |
| Application | Python, Dash, Flask |
| Visualization | Plotly |
| Data processing | pandas, openpyxl |
| Database | PostgreSQL (Neon), SQLAlchemy, psycopg2 |
| Wave forecast | Windy Point Forecast API |
| Hosting | Render, Gunicorn |
| Version control | Git, GitHub |

## Data Flow

```text
Schedule Excel files
    │
    ▼
pre_process.py
    │  transform / serialize / upload
    ▼
Neon PostgreSQL ──────────────┐
                              │
Windy Point Forecast API      │
    │                         │
    ▼                         │
Wave forecast records ────────┤
                              ▼
                         Dash / Plotly
                              │
                              ▼
                      Render-hosted website
```

Schedule data and wave forecasts are stored in PostgreSQL. The dashboard reads these records for visualization rather than relying on the Excel workbooks at request time.

## Project Files

| File | Purpose |
| --- | --- |
| `app.py` | Main Dash application, charts, callbacks, and analytics |
| `pre_process.py` | Prepare schedule data and upload it to PostgreSQL |
| `update_wave.py` | Fetch and store Windy wave forecasts, where configured |
| `requirements.txt` | Python dependencies |
| `assets/` | Application CSS and image assets |
| `run_PreProcess.bat` | Local Windows convenience script; paths are machine-specific |

Some utility filenames may differ between local development and deployment. The application currently uses a primarily single-file architecture; modularization is a possible future improvement.

## Local Setup

1. Clone the repository and enter its directory.
2. Create and activate a Python virtual environment.
3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Create a local `.env` file with the required configuration:

   ```dotenv
   DATABASE_URL=your_postgresql_connection_url
   WINDY_API_KEY=your_windy_api_key
   ANALYTICS_USER=your_admin_username
   ANALYTICS_PASSWORD=your_admin_password
   ```

5. Ensure the PostgreSQL database contains the expected schedule data. The app loads the `progress` and `hourly` entries from `schedule_data` during startup.
6. Run the application:

   ```bash
   python app.py
   ```

7. Open `http://127.0.0.1:8050`.

**Important:** Never commit `.env`, database credentials, API keys, private workbooks, or operationally sensitive information. Keep secrets in local environment variables or the hosting platform's secret configuration.

## Updating Data

**Schedule:** Edit the source Excel files, run the preprocessing/upload workflow, and verify the resulting database update. The dashboard checks the schedule version periodically; data-only updates do not require a new application deployment.

**Wave forecast:** Run the configured wave updater to fetch and save new forecast records. The dashboard reads the stored forecast and checks its update status separately.

**Application code:** Changes to Python code, dependencies, or assets require a new deployment.

## Deployment

The application is designed to run on Render using:

```bash
gunicorn app:server
```

Configure the required environment variables in Render, install dependencies from `requirements.txt`, and deploy from the appropriate Git branch. Database credentials and API keys must remain outside version control.

## Refresh and Performance

- Schedule version polling: **60 seconds**.
- Wave forecast status polling: **5 minutes**.
- A shared schedule-version store coordinates dependent callbacks.
- Schedule data is cached by version to avoid unnecessary reloads.
- Unchanged schedule versions should not trigger a new Gantt figure.
- Gantt detail updates are triggered by chart interaction rather than periodic polling.

These mechanisms aim to reduce redundant database requests and preserve the user's zoom/pan state during routine checks.

## Limitations and Future Work

- Beta software: functionality and UI may change.
- Availability depends on the hosting platform, database, and external forecast API.
- IP-based analytics are approximate and do not uniquely identify individual people.
- Potential future work: modular application structure, improved observability, performance benchmarking, and additional automated tests.

## Access and Data Handling

This repository may be public, but that does **not** imply the underlying project data, company resources, or credentials are public. Share data, internal links, and project documents only with appropriate authorization.

---

*Developed to support offshore engineering schedule visualization and reduce repetitive manual workflows.*
