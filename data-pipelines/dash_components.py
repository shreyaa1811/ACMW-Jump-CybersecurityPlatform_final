#!/usr/bin/env python3

"""
Enhanced Cybersecurity Dashboard - UI Components

This module contains all the visualization components and UI elements for the dashboard:
- Header and footer
- Dashboard cards and charts
- Tab layouts and content
- Supporting UI elements
"""

import os
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import dcc, html, dash_table
import dash_bootstrap_components as dbc
from datetime import datetime, timedelta
import json

# Define color palette for UI consistency
COLORS = {
    'primary': '#4f46e5',   # Indigo
    'secondary': '#0ea5e9', # Sky blue
    'success': '#10b981',   # Emerald
    'danger': '#ef4444',    # Red
    'warning': '#f59e0b',   # Amber
    'info': '#06b6d4',      # Cyan
    'dark': '#1e293b',      # Slate dark
    'darker': '#0f172a',    # Slate darker
    'light': '#cbd5e1',     # Slate light
    'background': '#020617',   # Background
    'card': '#0f172a',      # Card background
}

# Modern graph layout settings
graph_layout = {
    'paper_bgcolor': 'rgba(0,0,0,0)',
    'plot_bgcolor': 'rgba(0,0,0,0)',
    'font': {'color': '#1e293b', 'family': 'Poppins, sans-serif'},
    'margin': {'l': 40, 'r': 40, 't': 40, 'b': 40},
    'hovermode': 'closest',
    'legend': {'orientation': 'h', 'y': -0.2},
    'colorway': [COLORS['primary'], COLORS['secondary'], COLORS['success'], COLORS['danger'], 
                 COLORS['warning'], COLORS['info']],
    'xaxis': {'gridcolor': 'rgba(0,0,0,0.05)', 'zerolinecolor': 'rgba(0,0,0,0.2)'},
    'yaxis': {'gridcolor': 'rgba(0,0,0,0.05)', 'zerolinecolor': 'rgba(0,0,0,0.2)'}
}

def create_custom_layout(**kwargs):
    """Create a custom layout by merging the global graph_layout with specific overrides."""
    custom_layout = graph_layout.copy()
    custom_layout.update(kwargs)
    return custom_layout

def create_header():
    """Create the dashboard header/navbar"""
    return dbc.Navbar(
        dbc.Container(
            [
                html.A(
                    dbc.Row(
                        [
                            dbc.Col(html.I(className="fas fa-shield-alt fa-2x", style={"color": "#3f84f8"}), width="auto"),
                            dbc.Col(dbc.NavbarBrand("SecureVision", className="ms-2", style={"fontSize": "1.4rem", "fontWeight": "700"})),
                        ],
                        align="center",
                        className="g-0",
                    ),
                    href="#",
                    style={"textDecoration": "none"},
                ),
                dbc.NavbarToggler(id="navbar-toggler", n_clicks=0),
                dbc.Collapse(
                    dbc.Row(
                        [
                            dbc.Col(
                                html.Div([
                                    html.Span(f"Last Updated: {datetime.now().strftime('%H:%M:%S')}", id="last-update", className="me-3 text-secondary"),
                                    dbc.Button([
                                        html.I(className="fas fa-sync-alt me-2"),
                                        "Refresh"
                                    ], id="refresh-btn", color="primary", size="sm", className="rounded-pill")
                                ], className="d-flex align-items-center")
                            ),
                        ],
                        className="ms-auto flex-nowrap mt-3 mt-md-0",
                        align="center",
                    ),
                    id="navbar-collapse",
                    is_open=False,
                    navbar=True,
                ),
            ],
            fluid=True,
        ),
        color="light",
        dark=False,
        sticky="top",
        className="mb-4 shadow-sm",
    )

def create_footer():
    """Create the dashboard footer"""
    return html.Footer(
        dbc.Container(
            [
                html.Hr(),
                dbc.Row([
                    dbc.Col(
                        html.P(
                            "SecureVision Security Dashboard © 2025",
                            className="text-muted",
                        ),
                        width={"size": 6}
                    ),
                    dbc.Col(
                        html.P(
                            f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                            id="footer-timestamp",
                            className="text-muted text-end",
                        ),
                        width={"size": 6}
                    ),
                ])
            ],
            fluid=True,
        ),
        className="mt-4 py-3 footer",
    )

def create_stat_card(title, value, icon, color=COLORS['primary']):
    """Create a modern stat card with icon and trend indicator"""
    return dbc.Card([
        dbc.CardBody([
            html.Div([
                html.Div([
                    html.I(className=f"fas fa-{icon} fa-2x", style={"color": color}),
                ], className="col-auto"),
                html.Div([
                    html.P(title, className="stat-label mb-1"),
                    html.H3(value, className="stat-value m-0", style={"color": color})
                ], className="col")
            ], className="row align-items-center")
        ])
    ], className="stat-card h-100")

def create_chart_card(title, chart, icon="chart-line"):
    """Create a card containing a chart with modern styling"""
    return dbc.Card([
        dbc.CardHeader([
            html.I(className=f"fas fa-{icon} me-2", style={"color": COLORS['primary']}),
            html.Span(title, style={"fontWeight": "500"})
        ], style={"backgroundColor": "rgba(0,0,0,0.1)", "border": "none"}),
        dbc.CardBody([
            chart
        ], className="chart-container")
    ], className="h-100 mb-4")

def create_overview_tab():
    """Create the overview dashboard tab with modern styling"""
    from dash_data_provider import (
        process_intrusion_data, process_rba_data, 
        process_text_threat_data, get_real_time_alerts
    )
    
    # Get visualization data
    intrusion_viz_data = process_intrusion_data()
    rba_viz_data = process_rba_data()
    text_viz_data = process_text_threat_data()
    
    # Calculate key metrics
    # (These would come from actual data, using placeholders here)
    total_sessions = 12500
    total_attacks = 378
    attack_rate = 3.02
    total_logins = 35689
    
    if 'attack_summary' in intrusion_viz_data and not intrusion_viz_data['attack_summary'].empty:
        total_sessions = intrusion_viz_data['attack_summary']['count'].sum()
        total_attacks = intrusion_viz_data['attack_summary'][intrusion_viz_data['attack_summary']['attack_detected'] == 1]['count'].sum()
        attack_rate = (total_attacks / total_sessions * 100) if total_sessions > 0 else 0
    
    # Create stat cards row
    stat_cards = dbc.Row([
        dbc.Col(create_stat_card("Total Sessions", f"{total_sessions:,}", "shield-alt", COLORS['primary']), width=3),
        dbc.Col(create_stat_card("Attacks Detected", f"{total_attacks:,}", "exclamation-triangle", COLORS['danger']), width=3),
        dbc.Col(create_stat_card("Attack Rate", f"{attack_rate:.2f}%", "percent", COLORS['warning']), width=3),
        dbc.Col(create_stat_card("Login Attempts", f"{total_logins:,}", "sign-in-alt", COLORS['info']), width=3),
    ], className="mb-4 g-3")
    
    # Attack distribution pie chart
    attack_pie = None
    if 'attack_summary' in intrusion_viz_data and not intrusion_viz_data['attack_summary'].empty:
        attack_pie = dcc.Graph(
            figure=px.pie(
                intrusion_viz_data['attack_summary'], 
                values='count', 
                names='label',
                color='label',
                color_discrete_map={'Normal': COLORS['success'], 'Attack': COLORS['danger']},
                hole=0.5
            ).update_layout(**create_custom_layout(title=""))
        )
    else:
        # Placeholder when data is not available
        attack_pie = html.Div("No attack data available", 
                             className="text-center p-5 text-muted")
    
    # Country bar chart
    country_bar = None
    if 'country_counts' in rba_viz_data and not rba_viz_data['country_counts'].empty:
        country_bar = dcc.Graph(
            figure=px.bar(
                rba_viz_data['country_counts'].head(10),
                x='country',
                y='count',
                text='count',
                labels={'count': 'Login Count', 'country': 'Country'},
                color_discrete_sequence=[COLORS['secondary']]
            ).update_layout(
                **create_custom_layout(
                    title="",
                    xaxis_title="",
                    yaxis_title=""
                )
            ).update_traces(
                texttemplate='%{text:,}',
                textposition='outside'
            )
        )
    else:
        # Placeholder when data is not available
        country_bar = html.Div("No country data available", 
                             className="text-center p-5 text-muted")
    
    # Browser distribution chart
    browser_chart = None
    if 'browser_summary' in intrusion_viz_data and not intrusion_viz_data['browser_summary'].empty:
        browser_df = intrusion_viz_data['browser_summary'].copy()
        browser_df_pivot = browser_df.pivot_table(
            index='browser_type', 
            columns='attack_detected', 
            values='count', 
            aggfunc='sum'
        ).fillna(0).reset_index()
        
        if 1 in browser_df_pivot.columns:
            browser_df_pivot['attack_pct'] = browser_df_pivot[1] / (browser_df_pivot[0] + browser_df_pivot[1]) * 100
        else:
            browser_df_pivot['attack_pct'] = 0
            
        browser_chart = dcc.Graph(
            figure=px.bar(
                browser_df_pivot,
                x='browser_type',
                y='attack_pct',
                labels={'attack_pct': 'Attack %', 'browser_type': 'Browser'},
                color_discrete_sequence=[COLORS['danger']]
            ).update_layout(
                **create_custom_layout(
                    title="",
                    xaxis_title="",
                    yaxis_title="Attack %"
                )
            )
        )
    else:
        # Placeholder when data is not available
        browser_chart = html.Div("No browser data available", 
                               className="text-center p-5 text-muted")
    
    # Protocol security chart
    protocol_chart = None
    if 'protocol_summary' in intrusion_viz_data and not intrusion_viz_data['protocol_summary'].empty:
        protocol_df = intrusion_viz_data['protocol_summary'].copy()
        protocol_df_pivot = protocol_df.pivot_table(
            index='protocol_type', 
            columns='attack_detected', 
            values='count', 
            aggfunc='sum'
        ).fillna(0).reset_index()
        
        if 1 in protocol_df_pivot.columns and 0 in protocol_df_pivot.columns:
            protocol_df_pivot['total'] = protocol_df_pivot[0] + protocol_df_pivot[1]
            protocol_df_pivot['attack_count'] = protocol_df_pivot[1]
            protocol_df_pivot['normal_count'] = protocol_df_pivot[0]
            protocol_df_pivot['attack_pct'] = protocol_df_pivot[1] / protocol_df_pivot['total'] * 100
        else:
            protocol_df_pivot['total'] = protocol_df_pivot.get(0, 0) + protocol_df_pivot.get(1, 0)
            protocol_df_pivot['attack_count'] = protocol_df_pivot.get(1, 0)
            protocol_df_pivot['normal_count'] = protocol_df_pivot.get(0, 0)
            protocol_df_pivot['attack_pct'] = 0
            
        protocol_chart = dcc.Graph(
            figure=go.Figure()
            .add_trace(go.Bar(
                x=protocol_df_pivot['protocol_type'],
                y=protocol_df_pivot['normal_count'],
                name='Normal',
                marker_color=COLORS['success'],
                hovertemplate='%{y:,} normal sessions'
            ))
            .add_trace(go.Bar(
                x=protocol_df_pivot['protocol_type'],
                y=protocol_df_pivot['attack_count'],
                name='Attack',
                marker_color=COLORS['danger'],
                hovertemplate='%{y:,} attack sessions'
            ))
            .update_layout(
                **create_custom_layout(
                    title="",
                    barmode='stack',
                    xaxis_title="",
                    yaxis_title="Session Count"
                )
            )
        )
    else:
        # Placeholder when data is not available
        protocol_chart = html.Div("No protocol data available", 
                                className="text-center p-5 text-muted")
    
    # Time pattern chart (hourly distribution)
    time_pattern_chart = None
    if 'time_pattern_df' in intrusion_viz_data and not intrusion_viz_data['time_pattern_df'].empty:
        time_pattern_chart = dcc.Graph(
            figure=go.Figure()
            .add_trace(go.Scatter(
                x=intrusion_viz_data['time_pattern_df']['hour'],
                y=intrusion_viz_data['time_pattern_df']['normal'],
                mode='lines+markers',
                name='Normal Sessions',
                line=dict(color=COLORS['success'], width=3),
                marker=dict(size=8)
            ))
            .add_trace(go.Scatter(
                x=intrusion_viz_data['time_pattern_df']['hour'],
                y=intrusion_viz_data['time_pattern_df']['attacks'],
                mode='lines+markers',
                name='Attack Sessions',
                line=dict(color=COLORS['danger'], width=3),
                marker=dict(size=8)
            ))
            .update_layout(
                **create_custom_layout(
                    title="",
                    xaxis_title="Hour of Day",
                    yaxis_title="Session Count",
                    hovermode="x unified"
                )
            )
        )
    else:
        # Generate placeholder data
        hours = np.arange(24)
        normal_data = [50 + 40 * np.sin(0.3 * h) + np.random.randint(-10, 10) for h in hours]
        attack_data = [8 + 6 * np.sin(0.3 * h + 1) + np.random.randint(-2, 3) for h in hours]
        
        time_pattern_chart = dcc.Graph(
            figure=go.Figure()
            .add_trace(go.Scatter(
                x=hours,
                y=normal_data,
                mode='lines+markers',
                name='Normal Sessions',
                line=dict(color=COLORS['success'], width=3),
                marker=dict(size=8)
            ))
            .add_trace(go.Scatter(
                x=hours,
                y=attack_data,
                mode='lines+markers',
                name='Attack Sessions',
                line=dict(color=COLORS['danger'], width=3),
                marker=dict(size=8)
            ))
            .update_layout(
                **create_custom_layout(
                    title="",
                    xaxis_title="Hour of Day",
                    yaxis_title="Session Count",
                    hovermode="x unified"
                )
            )
        )
    
    # Create visualization cards
    visualization_row_1 = dbc.Row([
        dbc.Col(create_chart_card("Traffic Security Status", attack_pie, "chart-pie"), width=6),
        dbc.Col(create_chart_card("Top Countries by Login Volume", country_bar, "globe"), width=6),
    ], className="mb-4 g-3")
    
    visualization_row_2 = dbc.Row([
        dbc.Col(create_chart_card("Browser Security Analysis", browser_chart, "browser"), width=6),
        dbc.Col(create_chart_card("Protocol Security Analysis", protocol_chart, "network-wired"), width=6),
    ], className="mb-4 g-3")
    
    visualization_row_3 = dbc.Row([
        dbc.Col(create_chart_card("Daily Activity Pattern", time_pattern_chart, "clock"), width=12),
    ], className="mb-4 g-3")
    
    # Get real-time alerts for the recent alerts card
    recent_alerts = get_real_time_alerts(5)
    
    alerts_card = dbc.Card([
        dbc.CardHeader([
            html.I(className="fas fa-bell me-2", style={"color": COLORS['warning']}),
            html.Span("Recent Security Alerts", style={"fontWeight": "500"})
        ], style={"backgroundColor": "rgba(0,0,0,0.1)", "border": "none"}),
        dbc.CardBody([
            html.Div([
                dbc.Row([
                    dbc.Col(html.Span(alert["timestamp"], className="text-muted"), width=2),
                    dbc.Col(
                        html.Span(
                            alert["type"], 
                            className="badge rounded-pill", 
                            style={"backgroundColor": COLORS['secondary'], "color": "white"}
                        ), 
                        width=2
                    ),
                    dbc.Col(
                        html.Span(
                            alert["status"], 
                            className="badge rounded-pill", 
                            style={
                                "backgroundColor": COLORS['danger'] if alert["status"] == "Critical" else 
                                                   COLORS['warning'] if alert["status"] == "Warning" else 
                                                   COLORS['info']
                            }
                        ), 
                        width=2
                    ),
                    dbc.Col(html.Span(alert["message"]), width=6),
                ], className="py-2 border-bottom border-light")
                for _, alert in recent_alerts.iterrows()
            ])
        ])
    ], className="mb-4")
    
    # Security posture overview
    security_metrics = [
        {"name": "Network Security", "score": 85, "color": COLORS['success']},
        {"name": "Authentication", "score": 68, "color": COLORS['warning']},
        {"name": "Encryption", "score": 92, "color": COLORS['success']},
        {"name": "Application Security", "score": 73, "color": COLORS['warning']},
    ]
    
    security_posture = dbc.Card([
        dbc.CardHeader([
            html.I(className="fas fa-shield-alt me-2", style={"color": COLORS['primary']}),
            html.Span("Security Posture Overview", style={"fontWeight": "500"})
        ], style={"backgroundColor": "rgba(0,0,0,0.1)", "border": "none"}),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.Div(className="d-flex justify-content-between align-items-center mb-1", children=[
                            html.Span(metric["name"], className="me-2"),
                            html.Span(f"{metric['score']}%", style={"color": metric["color"], "fontWeight": "600"})
                        ]),
                        dbc.Progress(
                            value=metric["score"], 
                            color="success" if metric["score"] >= 80 else "warning" if metric["score"] >= 60 else "danger",
                            className="mb-3",
                            style={"height": "8px"}
                        )
                    ])
                    for metric in security_metrics
                ], width=6),
                dbc.Col([
                    html.Div(className="text-center", children=[
                        html.H1("78%", className="display-4 mb-0", style={"color": COLORS['warning']}),
                        html.P("Overall Security Score", className="text-muted"),
                        html.Div(className="d-flex justify-content-center mt-3", children=[
                            html.Span("Previous: 72%", className="me-3 text-muted"),
                            html.Span([
                                html.I(className="fas fa-arrow-up me-1", style={"color": COLORS['success']}),
                                "6%"
                            ], style={"color": COLORS['success']})
                        ])
                    ])
                ], width=6)
            ])
        ])
    ])
    
    # AI Analytics Integration - This container will be populated by callbacks
    ai_analytics_container = html.Div([
        html.H4("AI Security Analysis", className="mt-4 mb-3"),
        html.Div(id="ai-analytics-overview", className="mb-3"),
    ])
    
    return html.Div([
        html.H2("Security Operations Center", className="dashboard-title mb-4"),
        
        # Key metrics row
        stat_cards,
        
        # Visualizations
        visualization_row_1,
        visualization_row_2,
        visualization_row_3,
        
        # AI Analytics Integration
        ai_analytics_container,
        
        # Alerts and security posture
        dbc.Row([
            dbc.Col(alerts_card, width=7),
            dbc.Col(security_posture, width=5),
        ], className="g-3"),
        
    ], className="dashboard-container")

def create_intrusion_tab():
    """Create the intrusion detection tab"""
    from dash_data_provider import process_intrusion_data
    
    # Get visualization data
    intrusion_viz_data = process_intrusion_data()
    
    if not intrusion_viz_data or all(df.empty for df in intrusion_viz_data.values() if isinstance(df, pd.DataFrame)):
        return html.Div([
            html.H3("Intrusion Detection Dataset", className="text-center"),
            html.P("No intrusion detection data available.", className="text-center text-warning")
        ], className="dashboard-container")
    
    # Calculate metrics
    total_sessions = 12500  # Default placeholder values
    total_attacks = 378
    attack_rate = 3.02
    
    if 'attack_summary' in intrusion_viz_data and not intrusion_viz_data['attack_summary'].empty:
        total_sessions = intrusion_viz_data['attack_summary']['count'].sum()
        total_attacks = intrusion_viz_data['attack_summary'][intrusion_viz_data['attack_summary']['attack_detected'] == 1]['count'].sum()
        attack_rate = (total_attacks / total_sessions * 100) if total_sessions > 0 else 0
    
    # Create metrics row
    metrics_row = dbc.Row([
        dbc.Col(create_stat_card("Total Sessions", f"{total_sessions:,}", "chart-line", COLORS['primary']), width=3),
        dbc.Col(create_stat_card("Attack Sessions", f"{total_attacks:,}", "bug", COLORS['danger']), width=3),
        dbc.Col(create_stat_card("Attack Rate", f"{attack_rate:.2f}%", "percent", COLORS['warning']), width=3),
        dbc.Col(create_stat_card("Unusual Time Access", f"{378:,}", "clock", COLORS['info']), width=3),
    ], className="mb-4 g-3")
    
    # Attack distribution pie chart
    attack_pie = None
    if 'attack_summary' in intrusion_viz_data and not intrusion_viz_data['attack_summary'].empty:
        attack_pie = dcc.Graph(
            figure=px.pie(
                intrusion_viz_data['attack_summary'], 
                values='count', 
                names='label',
                color='label',
                color_discrete_map={'Normal': COLORS['success'], 'Attack': COLORS['danger']},
                hole=0.6,
            ).update_layout(
                **graph_layout,
                title="",
                annotations=[dict(
                    text=f"{attack_rate:.1f}%<br>Attack Rate", 
                    x=0.5, y=0.5,
                    font_size=20,
                    font_color=COLORS['dark'],
                    showarrow=False
                )]
            )
        )
    else:
        # Placeholder when data is not available
        attack_pie = html.Div("No attack data available", 
                             className="text-center p-5 text-muted")
    
    # Protocol vs Attack visualization
    protocol_bar = None
    if 'protocol_summary' in intrusion_viz_data and not intrusion_viz_data['protocol_summary'].empty:
        protocol_df = intrusion_viz_data['protocol_summary'].pivot_table(
            values='count', 
            index='protocol_type', 
            columns='attack_detected', 
            aggfunc='sum'
        ).fillna(0).reset_index()
        
        if 0 in protocol_df.columns and 1 in protocol_df.columns:
            protocol_df['total'] = protocol_df[0] + protocol_df[1]
            protocol_df['attack_rate'] = (protocol_df[1] / protocol_df['total'] * 100)
        else:
            protocol_df['total'] = protocol_df.get(0, 0)
            protocol_df['attack_rate'] = 0
            
        protocol_bar = dcc.Graph(
            figure=go.Figure()
            .add_trace(go.Bar(
                x=protocol_df['protocol_type'],
                y=protocol_df.get(0, []),
                name='Normal Sessions',
                marker_color=COLORS['success'],
            ))
            .add_trace(go.Bar(
                x=protocol_df['protocol_type'],
                y=protocol_df.get(1, []),
                name='Attack Sessions',
                marker_color=COLORS['danger'],
            ))
            .update_layout(
                **graph_layout,
                title="",
                barmode='stack',
                xaxis_title="Protocol Type",
                yaxis_title="Session Count",
            )
        )
    else:
        # Placeholder when data is not available
        protocol_bar = html.Div("No protocol data available", 
                               className="text-center p-5 text-muted")
    
    # Create a scatter plot for failed logins vs. attacks
    failed_login_scatter = None
    if 'failed_login_summary' in intrusion_viz_data and not intrusion_viz_data['failed_login_summary'].empty:
        failed_login_scatter = dcc.Graph(
            figure=px.scatter(
                intrusion_viz_data['failed_login_summary'],
                x='failed_logins',
                y='count',
                size='count',
                color='attack_detected',
                color_discrete_map={0: COLORS['success'], 1: COLORS['danger']},
                hover_name='failed_logins',
                labels={'failed_logins': 'Failed Logins', 'count': 'Session Count', 'attack_detected': 'Attack'},
                title=""
            ).update_layout(**graph_layout)
        )
    else:
        # Placeholder
        failed_login_scatter = html.Div("No failed login data available", 
                                      className="text-center p-5 text-muted")
    
    # Encryption used bar chart
    encryption_bar = None
    if 'encryption_summary' in intrusion_viz_data and not intrusion_viz_data['encryption_summary'].empty:
        encryption_bar = dcc.Graph(
            figure=px.bar(
                intrusion_viz_data['encryption_summary'],
                x='encryption_used',
                y='count',
                color='attack_detected',
                color_discrete_map={0: COLORS['success'], 1: COLORS['danger']},
                barmode='group',
                labels={'attack_detected': 'Attack', 'count': 'Count', 'encryption_used': 'Encryption Type'}
            ).update_layout(**graph_layout, title="")
        )
    else:
        # Placeholder
        encryption_bar = html.Div("No encryption data available", 
                                 className="text-center p-5 text-muted")
    
    # Risk metrics card
    risk_metrics = dbc.Card([
        dbc.CardHeader([
            html.I(className="fas fa-exclamation-triangle me-2", style={"color": COLORS['warning']}),
            html.Span("Risk Metrics & Vulnerabilities", style={"fontWeight": "500"})
        ], style={"backgroundColor": "rgba(0,0,0,0.1)", "border": "none"}),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.Div(className="d-flex justify-content-between align-items-center mb-1", children=[
                            html.Div([
                                html.I(className="fas fa-lock-open me-2", style={"color": COLORS['danger']}),
                                html.Span("Unencrypted Sessions")
                            ]),
                            html.Span(f"{1254:,}", style={"color": COLORS['danger'], "fontWeight": "600"})
                        ]),
                        dbc.Progress(
                            value=34, 
                            color="danger",
                            className="mb-3",
                            style={"height": "6px"}
                        )
                    ]),
                    html.Div([
                        html.Div(className="d-flex justify-content-between align-items-center mb-1", children=[
                            html.Div([
                                html.I(className="fas fa-user-shield me-2", style={"color": COLORS['warning']}),
                                html.Span("Failed Login Attempts")
                            ]),
                            html.Span("1,287", style={"color": COLORS['warning'], "fontWeight": "600"})
                        ]),
                        dbc.Progress(
                            value=48, 
                            color="warning",
                            className="mb-3",
                            style={"height": "6px"}
                        )
                    ]),
                ], width=6),
                dbc.Col([
                    html.Div([
                        html.Div(className="d-flex justify-content-between align-items-center mb-1", children=[
                            html.Div([
                                html.I(className="fas fa-network-wired me-2", style={"color": COLORS['danger']}),
                                html.Span("Insecure Protocols")
                            ]),
                            html.Span("732", style={"color": COLORS['danger'], "fontWeight": "600"})
                        ]),
                        dbc.Progress(
                            value=42, 
                            color="danger",
                            className="mb-3",
                            style={"height": "6px"}
                        )
                    ]),
                    html.Div([
                        html.Div(className="d-flex justify-content-between align-items-center mb-1", children=[
                            html.Div([
                                html.I(className="fas fa-exclamation-circle me-2", style={"color": COLORS['warning']}),
                                html.Span("Overall Risk Score")
                            ]),
                            html.Span("67/100", style={"color": COLORS['warning'], "fontWeight": "600"})
                        ]),
                        dbc.Progress(
                            value=67, 
                            color="warning",
                            className="mb-3",
                            style={"height": "6px"}
                        )
                    ])
                ], width=6),
            ])
        ])
    ])
    
    # Browser type security analysis
    browser_table_card = None
    if 'browser_summary' in intrusion_viz_data and not intrusion_viz_data['browser_summary'].empty:
        browser_df = intrusion_viz_data['browser_summary'].copy()
        browser_df_pivot = browser_df.pivot_table(
            index='browser_type', 
            columns='attack_detected', 
            values='count', 
            aggfunc='sum'
        ).fillna(0).reset_index()
        
        if 1 in browser_df_pivot.columns:
            browser_df_pivot['attack_rate'] = browser_df_pivot[1] / (browser_df_pivot[0] + browser_df_pivot[1]) * 100
        else:
            browser_df_pivot['attack_rate'] = 0
            
        # Format the attack rate as a percentage string
        browser_display_df = browser_df_pivot[['browser_type', 'attack_rate']].sort_values('attack_rate', ascending=False).copy()
        browser_display_df['attack_rate'] = browser_display_df['attack_rate'].apply(lambda x: f"{x:.2f}%")
        
        browser_table = dbc.Table.from_dataframe(
            browser_display_df,
            striped=True,
            bordered=False,
            hover=True,
            responsive=True,
            className="mb-0"
        )
        
        browser_table_card = dbc.Card([
            dbc.CardHeader([
                html.I(className="fas fa-browser me-2", style={"color": COLORS['primary']}),
                html.Span("Browser Security Ranking", style={"fontWeight": "500"})
            ], style={"backgroundColor": "rgba(0,0,0,0.1)", "border": "none"}),
            dbc.CardBody([
                browser_table
            ], className="p-0")
        ])
    else:
        # Placeholder
        browser_table_card = dbc.Card([
            dbc.CardHeader([
                html.I(className="fas fa-browser me-2", style={"color": COLORS['primary']}),
                html.Span("Browser Security Ranking", style={"fontWeight": "500"})
            ], style={"backgroundColor": "rgba(0,0,0,0.1)", "border": "none"}),
            dbc.CardBody([
                html.P("No browser data available", className="text-center text-muted p-3")
            ], className="p-0")
        ])
    
    # AI Analytics Integration
    ai_analytics_section = html.Div([
        html.H4("AI-Powered Threat Analysis", className="mt-4 mb-3"),
        html.Div(id="intrusion-ai-analysis")
    ], className="mb-4")
    
    return html.Div([
        html.H2("Intrusion Detection Analysis", className="dashboard-title mb-4"),
        
        # Key metrics row
        metrics_row,
        
        # Tabs for different analysis views
        dbc.Tabs([
            dbc.Tab(
                dbc.Row([
                    # Main charts row
                    dbc.Row([
                        dbc.Col(create_chart_card("Traffic Security Status", attack_pie, "chart-pie"), width=5),
                        dbc.Col(create_chart_card("Protocol Security Analysis", protocol_bar, "network-wired"), width=7),
                    ], className="mb-4 g-3"),
                    
                    # Second row of charts
                    dbc.Row([
                        dbc.Col(create_chart_card("Failed Logins Impact Analysis", failed_login_scatter, "key"), width=6),
                        dbc.Col(create_chart_card("Encryption Method Analysis", encryption_bar, "lock"), width=6),
                    ], className="mb-4 g-3"),
                    
                    # AI Analytics integration
                    ai_analytics_section,
                    
                    # Risk metrics and browser table
                    dbc.Row([
                        dbc.Col(risk_metrics, width=7),
                        dbc.Col(browser_table_card, width=5),
                    ], className="g-3"),
                ]),
                label="Overview",
                tab_id="intrusion-tab-1",
            ),
            dbc.Tab(
                html.Div(id="intrusion-advanced-analysis", className="pt-4"),
                label="Advanced Analysis",
                tab_id="intrusion-tab-2",
            ),
        ], className="mb-4", active_tab="intrusion-tab-1"),
    ], className="dashboard-container")

def create_rba_tab():
    """Create the risk-based authentication tab"""
    from dash_data_provider import process_rba_data
    
    # Get visualization data
    rba_viz_data = process_rba_data()
    
    if not rba_viz_data or all(df.empty for df in rba_viz_data.values() if isinstance(df, pd.DataFrame)):
        return html.Div([
            html.H3("Risk-Based Authentication Dataset", className="text-center"),
            html.P("No RBA data available.", className="text-center text-warning")
        ], className="dashboard-container")
    
    # Stats for metrics - use placeholder values if data not available
    total_logins = 35689
    unique_countries = 127
    avg_rtt = 78.5
    unique_users = 4521
    
    # Create metrics row
    metrics_row = dbc.Row([
        dbc.Col(create_stat_card("Total Logins", f"{total_logins:,}", "sign-in-alt", COLORS['primary']), width=3),
        dbc.Col(create_stat_card("Unique Users", f"{unique_users:,}", "users", COLORS['secondary']), width=3),
        dbc.Col(create_stat_card("Unique Countries", f"{unique_countries}", "globe", COLORS['info']), width=3),
        dbc.Col(create_stat_card("Avg. Round-Trip Time", f"{avg_rtt:.2f} ms", "tachometer-alt", COLORS['success']), width=3),
    ], className="mb-4 g-3")
    
    # Country distribution bar chart
    country_bar = None
    if 'country_counts' in rba_viz_data and not rba_viz_data['country_counts'].empty:
        country_bar = dcc.Graph(
            figure=px.bar(
                rba_viz_data['country_counts'].head(15),
                x='country',
                y='count',
                title="",
                labels={'count': 'Login Count', 'country': 'Country'},
                color_discrete_sequence=[COLORS['primary']]
            ).update_layout(**create_custom_layout())
            .update_traces(texttemplate='%{y:,}', textposition='outside')
        )
    else:
        # Placeholder
        country_bar = html.Div("No country data available", 
                              className="text-center p-5 text-muted")
    
    # Login hour distribution chart
    hour_line = None
    if 'hour_counts' in rba_viz_data and not rba_viz_data['hour_counts'].empty:
        hour_layout = graph_layout.copy()
        hour_layout.update({
            'xaxis': dict(tickmode='linear', tick0=0, dtick=1, gridcolor='rgba(0,0,0,0.05)', zerolinecolor='rgba(0,0,0,0.2)'),
            'yaxis': dict(title='Login Count', gridcolor='rgba(0,0,0,0.05)', zerolinecolor='rgba(0,0,0,0.2)')
        })
        
        hour_line = dcc.Graph(
            figure=px.line(
                rba_viz_data['hour_counts'],
                x='hour',
                y='count',
                title="",
                labels={'count': 'Login Count', 'hour': 'Hour of Day'},
                markers=True,
                line_shape='spline',
                color_discrete_sequence=[COLORS['secondary']]
            ).update_layout(**create_custom_layout(**hour_layout))
        )
    else:
        # Generate placeholder data
        hours = list(range(24))
        counts = [200 + 150 * np.sin(np.pi * h / 12) + np.random.randint(-20, 20) for h in hours]
        hour_data = pd.DataFrame({'hour': hours, 'count': counts})
        
        hour_line = dcc.Graph(
            figure=px.line(
                hour_data,
                x='hour',
                y='count',
                title="",
                labels={'count': 'Login Count', 'hour': 'Hour of Day'},
                markers=True,
                line_shape='spline',
                color_discrete_sequence=[COLORS['secondary']]
            ).update_layout(**create_custom_layout())
        )
    
    # Risk assessment card
    risk_assessment = dbc.Card([
        dbc.CardHeader([
            html.I(className="fas fa-user-shield me-2", style={"color": COLORS['primary']}),
            html.Span("Login Risk Assessment", style={"fontWeight": "500"})
        ], style={"backgroundColor": "rgba(0,0,0,0.1)", "border": "none"}),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.Div(className="d-flex justify-content-between align-items-center mb-1", children=[
                            html.Div([
                                html.I(className="fas fa-globe me-2", style={"color": COLORS['warning']}),
                                html.Span("Geo-location Risk")
                            ]),
                            html.Span("Medium", style={"color": COLORS['warning'], "fontWeight": "600"})
                        ]),
                        dbc.Progress(
                            value=65, 
                            color="warning",
                            className="mb-3",
                            style={"height": "6px"}
                        )
                    ]),
                    html.Div([
                        html.Div(className="d-flex justify-content-between align-items-center mb-1", children=[
                            html.Div([
                                html.I(className="fas fa-clock me-2", style={"color": COLORS['info']}),
                                html.Span("Temporal Risk")
                            ]),
                            html.Span("Low", style={"color": COLORS['info'], "fontWeight": "600"})
                        ]),
                        dbc.Progress(
                            value=32, 
                            color="info",
                            className="mb-3",
                            style={"height": "6px"}
                        )
                    ]),
                    html.Div([
                        html.Div(className="d-flex justify-content-between align-items-center mb-1", children=[
                            html.Div([
                                html.I(className="fas fa-fingerprint me-2", style={"color": COLORS['danger']}),
                                html.Span("Identity Risk")
                            ]),
                            html.Span("High", style={"color": COLORS['danger'], "fontWeight": "600"})
                        ]),
                        dbc.Progress(
                            value=78, 
                            color="danger",
                            className="mb-3",
                            style={"height": "6px"}
                        )
                    ])
                ], width=6),
                dbc.Col([
                    html.Div([
                        html.Div(className="d-flex justify-content-between align-items-center mb-1", children=[
                            html.Div([
                                html.I(className="fas fa-network-wired me-2", style={"color": COLORS['warning']}),
                                html.Span("Network Risk")
                            ]),
                            html.Span("Medium", style={"color": COLORS['warning'], "fontWeight": "600"})
                        ]),
                        dbc.Progress(
                            value=58, 
                            color="warning",
                            className="mb-3",
                            style={"height": "6px"}
                        )
                    ]),
                    html.Div([
                        html.Div(className="d-flex justify-content-between align-items-center mb-1", children=[
                            html.Div([
                                html.I(className="fas fa-mobile-alt me-2", style={"color": COLORS['info']}),
                                html.Span("Device Risk")
                            ]),
                            html.Span("Low", style={"color": COLORS['info'], "fontWeight": "600"})
                        ]),
                        dbc.Progress(
                            value=29, 
                            color="info",
                            className="mb-3",
                            style={"height": "6px"}
                        )
                    ]),
                    html.Div([
                        html.Div(className="d-flex justify-content-between align-items-center mb-1", children=[
                            html.Div([
                                html.I(className="fas fa-shield-alt me-2", style={"color": COLORS['warning']}),
                                html.Span("Overall Risk Score")
                            ]),
                            html.Span("62/100", style={"color": COLORS['warning'], "fontWeight": "600"})
                        ]),
                        dbc.Progress(
                            value=62, 
                            color="warning",
                            className="mb-3",
                            style={"height": "6px"}
                        )
                    ])
                ], width=6),
            ])
        ])
    ])
    
    # AI Analytics Integration
    ai_analytics_section = html.Div([
        html.H4("AI-Powered Authentication Analysis", className="mt-4 mb-3"),
        html.Div(id="rba-ai-analysis")
    ], className="mb-4")
    
    # Get real-time alerts for suspicious logins
    from dash_data_provider import get_real_time_alerts
    suspicious_logins = get_real_time_alerts(4)
    
    # Color map for status
    status_colors = {
        "Critical": COLORS['danger'],
        "Warning": COLORS['warning'],
        "Info": COLORS['info']
    }
    
    suspicious_logins_card = dbc.Card([
        dbc.CardHeader([
            html.I(className="fas fa-exclamation-circle me-2", style={"color": COLORS['danger']}),
            html.Span("Recent Suspicious Logins", style={"fontWeight": "500"})
        ], style={"backgroundColor": "rgba(0,0,0,0.1)", "border": "none"}),
        dbc.CardBody([
            html.Div([
                dbc.Row([
                    dbc.Col(html.Span(alert["timestamp"], className="text-muted"), width=2),
                    dbc.Col(html.Span(alert.get("userid", "unknown")), width=2),
                    dbc.Col(html.Span(alert.get("source_ip", "unknown")), width=3),
                    dbc.Col(
                        html.Span(
                            alert.get("country", ""), 
                            className="badge rounded-pill", 
                            style={"backgroundColor": COLORS['secondary'], "color": "white"}
                        ), 
                        width=2
                    ),
                    dbc.Col(
                        html.Span(
                            alert["status"], 
                            className="badge rounded-pill", 
                            style={"backgroundColor": status_colors.get(alert["status"], COLORS['info'])}
                        ), 
                        width=3
                    ),
                ], className="py-2 border-bottom border-dark")
                for _, alert in suspicious_logins.iterrows()
            ])
        ], className="p-0")
    ])
    
    return html.Div([
        html.H2("Risk-Based Authentication Analysis", className="dashboard-title mb-4"),
        
        # Key metrics row
        metrics_row,
        
        # Tabs for different analysis views
        dbc.Tabs([
            dbc.Tab(
                dbc.Row([
                    # Main charts row
                    dbc.Row([
                        dbc.Col(create_chart_card("Geographic Login Distribution", country_bar, "globe"), width=6),
                        dbc.Col(create_chart_card("Login Time Distribution", hour_line, "clock"), width=6),
                    ], className="mb-4 g-3"),
                    
                    # AI Analytics integration
                    ai_analytics_section,
                    
                    # Risk assessment and suspicious logins
                    dbc.Row([
                        dbc.Col(risk_assessment, width=6),
                        dbc.Col(suspicious_logins_card, width=6),
                    ], className="g-3"),
                ]),
                label="Overview",
                tab_id="rba-tab-1",
            ),
            dbc.Tab(
                html.Div(id="rba-advanced-analysis", className="pt-4"),
                label="Advanced Analysis",
                tab_id="rba-tab-2",
            ),
        ], className="mb-4", active_tab="rba-tab-1"),
    ], className="dashboard-container")

def create_text_threat_tab():
    """Create the text-based threat detection tab"""
    from dash_data_provider import process_text_threat_data
    
    # Get visualization data
    text_viz_data = process_text_threat_data()
    
    if not text_viz_data or all(df.empty for df in text_viz_data.values() if isinstance(df, pd.DataFrame)):
        return html.Div([
            html.H3("Text-based Threat Detection Dataset", className="text-center"),
            html.P("No text threat data available.", className="text-center text-warning")
        ], className="dashboard-container")
    
    # Placeholder metrics (replace with actual calculations from data)
    total_threats = 845
    critical_threats = 56
    high_threats = 219
    
    metrics_row = dbc.Row([
        dbc.Col(create_stat_card("Total Threats", f"{total_threats:,}", "shield-virus", COLORS['primary']), width=3),
        dbc.Col(create_stat_card("Critical Threats", f"{critical_threats:,}", "radiation", COLORS['danger']), width=3),
        dbc.Col(create_stat_card("High Severity", f"{high_threats:,}", "exclamation-triangle", COLORS['warning']), width=3),
        dbc.Col(create_stat_card("Medium/Low Threats", f"{total_threats - critical_threats - high_threats:,}", "info-circle", COLORS['info']), width=3),
    ], className="mb-4 g-3")
    
    # AI Analytics Integration
    ai_analytics_section = html.Div([
        html.H4("AI-Powered Text Analysis", className="mt-4 mb-3"),
        html.Div(id="text-ai-analysis")
    ], className="mb-4")
    
    # Placeholder content for the text threat tab
    content = html.Div([
        html.H2("Text-based Threat Intelligence", className="dashboard-title mb-4"),
        
        # Key metrics row
        metrics_row,
        
        # AI Analytics integration
        ai_analytics_section,
        
        # Placeholder for text analysis
        dbc.Card([
            dbc.CardHeader([
                html.I(className="fas fa-file-alt me-2", style={"color": COLORS['primary']}),
                html.Span("Text Content Analysis", style={"fontWeight": "500"})
            ], style={"backgroundColor": "rgba(0,0,0,0.1)", "border": "none"}),
            dbc.CardBody([
                html.P("This tab displays text-based threat analysis.", className="mb-4"),
                html.Div(id="text-content-analysis")
            ])
        ]),
    ], className="dashboard-container")
    
    return content

def create_realtime_tab():
    """Create the real-time monitoring tab"""
    
    # Create metrics row with placeholder data
    metrics_row = dbc.Row([
        dbc.Col(create_stat_card("Active Alerts", "23", "bell", COLORS['danger']), width=3),
        dbc.Col(create_stat_card("Systems Monitored", "28", "server", COLORS['primary']), width=3),
        dbc.Col(create_stat_card("Avg. Response Time", "1.2s", "clock", COLORS['info']), width=3),
        dbc.Col(create_stat_card("Events Per Minute", "573", "tachometer-alt", COLORS['success']), width=3),
    ], className="mb-4 g-3")
    
    # Container for system status that will be updated via callback
    system_status_container = html.Div(id="system-status-container")
    
    # Container for real-time alerts that will be updated via callback
    realtime_alerts_container = html.Div(id="realtime-alerts-container")
    
    # Container for AI Analytics
    ai_analytics_container = html.Div([
        html.H4("AI Security Analysis - Real-time", className="mt-4 mb-3"),
        html.Div(id="realtime-ai-analysis")
    ], className="mb-4")
    
    return html.Div([
        html.H2("Real-time Security Monitoring", className="dashboard-title mb-4"),
        
        dbc.Row([
            html.Div([
                html.Span(className="live-dot"),
                html.Span("LIVE MONITORING ACTIVE", className="text-danger fw-bold ms-2"),
                html.Span("Last updated: ", className="ms-3 text-muted"),
                html.Span(id="realtime-timestamp", children=datetime.now().strftime("%H:%M:%S"))
            ], className="text-center mb-4")
        ]),
        
        # Key metrics row
        metrics_row,
        
        # System status and network traffic
        dbc.Row([
            dbc.Col(system_status_container, width=12),
        ], className="mb-4 g-3"),
        
        # AI Analytics
        ai_analytics_container,
        
        # Live alerts
        dbc.Row([
            dbc.Col(realtime_alerts_container, width=12),
        ], className="mb-4 g-3"),
        
    ], className="dashboard-container")