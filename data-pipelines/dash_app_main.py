#!/usr/bin/env python3

"""
Enhanced Cybersecurity Dashboard - Main Application

This is the main entry point for the dashboard application that:
1. Initializes the Dash application
2. Sets up the layout structure
3. Configures the necessary callbacks
4. Starts the server

Run with: python dash_app_main.py
"""

import os
import sys
import pandas as pd
import dash
from dash import dcc, html, Input, Output, State, callback
import dash_bootstrap_components as dbc
from datetime import datetime
import time

# Add parent directory to sys.path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the component and data provider modules
from dash_components import (
    create_header, create_footer, create_overview_tab, 
    create_intrusion_tab, create_rba_tab, create_text_threat_tab,
    create_realtime_tab
)
from dash_data_provider import (
    initialize_data_connections, get_intrusion_data, get_rba_data,
    get_text_threat_data, get_real_time_alerts, process_intrusion_data,
    process_rba_data, process_text_threat_data, get_ai_analytics_data,
    update_real_time_data
)

# Initialize the Dash app with Bootstrap theme
app = dash.Dash(
    __name__, 
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css'
    ],
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
    suppress_callback_exceptions=True,
)
app.title = "SecureVision Dashboard"
server = app.server

# Custom CSS for the application (moved from the original index_string)
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
        <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap" rel="stylesheet">
        <style>
            body {
                background-color: #f8fafc;
                font-family: 'Poppins', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', sans-serif;
                color: #1e293b;
            }
            .card {
                border-radius: 12px;
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
                transition: all 0.3s ease;
                border: none;
                background-color: white;
                margin-bottom: 20px;
            }
            .card:hover {
                box-shadow: 0 8px 16px rgba(0, 0, 0, 0.1);
                transform: translateY(-2px);
            }
            .stat-card {
                padding: 1.5rem;
                background: white;
                position: relative;
                overflow: hidden;
            }
            .stat-card::before {
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                width: 5px;
                height: 100%;
                background: #3f84f8;
            }
            .dashboard-container {
                max-width: 1500px;
                margin: 0 auto;
                padding: 2rem;
            }
            .stat-value {
                font-size: 2.5rem;
                font-weight: 700;
                margin-bottom: 0.5rem;
            }
            .stat-label {
                font-size: 0.9rem;
                text-transform: uppercase;
                letter-spacing: 0.05rem;
                font-weight: 600;
                color: #64748b;
                margin-bottom: 0;
            }
            .chart-container {
                padding: 1rem;
                height: 100%;
                background: white;
                border-radius: 12px;
            }
            .nav-link {
                font-weight: 500;
                color: #64748b;
                padding: 0.75rem 1rem;
                border-radius: 8px;
                margin-right: 0.5rem;
            }
            .nav-link.active {
                background-color: #e0e7ff;
                color: #4338ca;
                font-weight: 600;
            }
            .navbar {
                background-color: white !important;
                box-shadow: 0 2px 12px rgba(0, 0, 0, 0.05);
            }
            .navbar-brand {
                font-weight: 700;
                letter-spacing: 0.05rem;
                color: #3f84f8 !important;
            }
            .dashboard-title {
                font-weight: 700;
                margin-bottom: 2rem;
                color: #3f84f8;
                text-align: center;
                font-size: 2.2rem;
            }
            .card-header {
                background-color: white;
                border-bottom: 1px solid #f1f5f9;
                font-weight: 600;
                padding: 1rem 1.5rem;
                color: #334155;
            }
            .alert-icon {
                display: flex;
                align-items: center;
                justify-content: center;
                width: 40px;
                height: 40px;
                border-radius: 50%;
                margin-right: 15px;
            }
            .table {
                color: #334155;
            }
            .table thead th {
                font-weight: 600;
                color: #64748b;
                border-bottom-width: 1px;
            }
            .badge {
                font-weight: 500;
                padding: 0.4em 0.8em;
                border-radius: 6px;
            }
            .footer {
                background-color: white;
                padding: 1.5rem 0;
                border-top: 1px solid #f1f5f9;
            }
            .btn-primary {
                background-color: #3f84f8;
                border-color: #3f84f8;
            }
            .btn-primary:hover {
                background-color: #3b72df;
                border-color: #3b72df;
            }
            .live-dot {
                display: inline-block;
                width: 10px;
                height: 10px;
                border-radius: 50%;
                background-color: #ef4444;
                margin-right: 8px;
                animation: pulse 1.5s infinite;
            }
            @keyframes pulse {
                0% {
                    transform: scale(0.95);
                    box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.7);
                }
                70% {
                    transform: scale(1);
                    box-shadow: 0 0 0 6px rgba(239, 68, 68, 0);
                }
                100% {
                    transform: scale(0.95);
                    box-shadow: 0 0 0 0 rgba(239, 68, 68, 0);
                }
            }
            .tab-content {
                margin-top: 20px;
            }
            .nav-pills .nav-link.active {
                background-color: #4f46e5;
                color: white;
            }
            .nav-pills .nav-link {
                color: #4f46e5;
                border-radius: 8px;
                padding: 8px 16px;
                margin-right: 8px;
                font-weight: 500;
            }
            .activity-timeline {
                position: relative;
                padding-left: 30px;
            }
            .activity-timeline::before {
                content: '';
                position: absolute;
                left: 10px;
                top: 0;
                bottom: 0;
                width: 2px;
                background-color: #e2e8f0;
            }
            .activity-item {
                position: relative;
                margin-bottom: 20px;
            }
            .activity-item::before {
                content: '';
                position: absolute;
                left: -30px;
                top: 0;
                width: 12px;
                height: 12px;
                border-radius: 50%;
                background-color: #4f46e5;
                border: 2px solid white;
            }
            .system-status-indicator {
                display: inline-block;
                width: 12px;
                height: 12px;
                border-radius: 50%;
                margin-right: 8px;
            }
            .status-healthy {
                background-color: #10b981;
            }
            .status-warning {
                background-color: #f59e0b;
            }
            .status-critical {
                background-color: #ef4444;
            }
            .world-map-container {
                height: 400px;
                position: relative;
                background-color: #f8fafc;
                border-radius: 12px;
                overflow: hidden;
            }
            .worldmap-overlay {
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                display: flex;
                align-items: center;
                justify-content: center;
                background-color: rgba(255, 255, 255, 0.7);
                z-index: 10;
            }
            .spinner-container {
                display: flex;
                align-items: center;
                justify-content: center;
                height: 200px;
            }
            .data-table-container {
                max-height: 400px;
                overflow-y: auto;
                border-radius: 8px;
                border: 1px solid #e2e8f0;
            }
            .heatmap-tooltip {
                background-color: white;
                border: 1px solid #e2e8f0;
                border-radius: 8px;
                padding: 8px 12px;
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# Initialize data connections
initialize_data_connections()

# Define the app layout
app.layout = html.Div([
    # Header/Navbar
    create_header(),
    
    # Main content area with tabs
    dbc.Container(
        [
            dbc.Tabs(
                [
                    dbc.Tab(create_overview_tab(), label="Overview", tab_id="tab-overview"),
                    dbc.Tab(create_intrusion_tab(), label="Intrusion Detection", tab_id="tab-intrusion"),
                    dbc.Tab(create_rba_tab(), label="Risk-Based Authentication", tab_id="tab-rba"),
                    dbc.Tab(create_text_threat_tab(), label="Text-based Threats", tab_id="tab-text"),
                    dbc.Tab(create_realtime_tab(), label="Real-time Monitoring", tab_id="tab-realtime"),
                ],
                id="tabs",
                active_tab="tab-overview",
                className="mb-3",
            ),
        ],
        fluid=True,
        className="px-4 py-3",
    ),
    
    # Footer
    create_footer(),
    
    # Interval component for refreshing data
    dcc.Interval(
        id='interval-component',
        interval=300 * 1000,  # in milliseconds, refresh every 5 minutes
        n_intervals=0
    ),
    
    # Interval component for real-time updates
    dcc.Interval(
        id='realtime-interval-component',
        interval=5 * 1000,  # in milliseconds, update every 5 seconds
        n_intervals=0
    ),
    
    # Store last update time
    dcc.Store(id='last-update-time', data=datetime.now().isoformat()),
    
    # Store for AI analytics data
    dcc.Store(id='ai-analytics-data', data={}),
])

# Callback to refresh data and update timestamps
@app.callback(
    [
        Output("last-update", "children"),
        Output("footer-timestamp", "children"),
        Output("last-update-time", "data")
    ],
    [
        Input("refresh-btn", "n_clicks"),
        Input("interval-component", "n_intervals")
    ]
)
def update_timestamps(n_clicks, n_intervals):
    now = datetime.now()
    last_update_text = f"Last Updated: {now.strftime('%H:%M:%S')}"
    footer_text = f"Last updated: {now.strftime('%Y-%m-%d %H:%M:%S')}"
    return last_update_text, footer_text, now.isoformat()

# Callback for updating the real-time monitor timestamp
@app.callback(
    Output("realtime-timestamp", "children"),
    [Input("realtime-interval-component", "n_intervals")]
)
def update_realtime_timestamp(n_intervals):
    return datetime.now().strftime("%H:%M:%S")

# Callback for real-time data updates
@app.callback(
    Output("ai-analytics-data", "data"),
    [Input("realtime-interval-component", "n_intervals")]
)
def update_ai_analytics(n_intervals):
    return get_ai_analytics_data()

# Callback for navbar toggler
@app.callback(
    Output("navbar-collapse", "is_open"),
    [Input("navbar-toggler", "n_clicks")],
    [State("navbar-collapse", "is_open")],
)
def toggle_navbar_collapse(n, is_open):
    if n:
        return not is_open
    return is_open

# Callback to update system data on refresh
@app.callback(
    Output("system-status-container", "children"),
    [Input("realtime-interval-component", "n_intervals")]
)
def update_system_status(n_intervals):
    return update_real_time_data()

if __name__ == "__main__":
    # Get port from command line args or default to 8050
    import argparse
    parser = argparse.ArgumentParser(description='Cybersecurity Dashboard')
    parser.add_argument('--port', type=int, default=8050, help='Port to run the dashboard on')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to run the dashboard on')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    args = parser.parse_args()
    
    print(f"Starting dashboard on http://{args.host}:{args.port}/")
    app.run(host=args.host, port=args.port, debug=args.debug)