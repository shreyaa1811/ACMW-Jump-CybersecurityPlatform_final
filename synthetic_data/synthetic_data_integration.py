#!/usr/bin/env python3

"""
Enhanced Synthetic Security Data Integration Module

This module provides functions to integrate synthetic security data with the dashboard.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import datetime
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("SyntheticDataIntegration")

# Add parent directory for imports if needed
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Create output directory for generated data
output_dir = Path('synthetic_data')
output_dir.mkdir(exist_ok=True)

# Path for latest data
latest_data_path = output_dir / 'latest_data.json'
latest_analysis_path = output_dir / 'latest_analysis.json'

# Global cache variables
_latest_security_data = None
_latest_analysis_results = None
_last_update_time = datetime.datetime.now() - datetime.timedelta(hours=1)

def get_synthetic_security_data(hours=24, limit=100, use_fake=True, force_refresh=False):
    """Get synthetic security data for the dashboard"""
    global _latest_security_data, _last_update_time
    
    # Generate some mock data
    events = []
    now = datetime.datetime.now()
    
    for i in range(min(limit, 20)):
        event_time = now - datetime.timedelta(minutes=i*30)
        event = {
            "id": i + 1,
            "event_type": np.random.choice(["intrusion", "authentication", "data_access", "malware"]),
            "is_malicious": np.random.choice([0, 1], p=[0.7, 0.3]),
            "severity": np.random.choice(["critical", "high", "medium", "low", "info"]),
            "timestamp": event_time,
            "source_ip": f"192.168.{np.random.randint(1, 255)}.{np.random.randint(1, 255)}",
            "destination_ip": f"10.0.{np.random.randint(1, 255)}.{np.random.randint(1, 255)}",
            "description": f"Sample event {i+1}"
        }
        events.append(event)
    
    df = pd.DataFrame(events)
    _latest_security_data = df
    _last_update_time = now
    
    return df

def get_latest_threat_analysis():
    """Get the latest threat analysis for dashboard display"""
    analysis = {
        "timestamp": datetime.datetime.now().isoformat(),
        "summary": {
            "total_events_analyzed": 150,
            "total_threats_detected": 42,
            "threat_types": {
                "intrusion": 15,
                "authentication": 12,
                "data_access": 8,
                "malware": 7
            },
            "severity_distribution": {
                "critical": 5,
                "high": 12,
                "medium": 15,
                "low": 8,
                "info": 2
            }
        }
    }
    
    return analysis

def get_event_summary(events_df):
    """Generate summary statistics for security events"""
    if events_df is None or events_df.empty:
        return {}
    
    summary = {
        "total_events": len(events_df),
        "malicious_events": int(events_df["is_malicious"].sum() if "is_malicious" in events_df.columns else 0),
        "event_types": {}
    }
    
    if "event_type" in events_df.columns:
        for event_type in events_df["event_type"].unique():
            summary["event_types"][event_type] = int(len(events_df[events_df["event_type"] == event_type]))
    
    return summary

def get_events_by_time(events_df, interval='hour'):
    """Group events by time intervals for time series visualization"""
    if events_df is None or events_df.empty or "timestamp" not in events_df.columns:
        return pd.DataFrame()
    
    # Create a copy with a time_group column
    df_copy = events_df.copy()
    df_copy["time_group"] = pd.to_datetime(df_copy["timestamp"]).dt.floor('H')
    
    # Group by time
    time_groups = df_copy.groupby("time_group").size().reset_index(name="count")
    time_groups["is_malicious"] = "total"
    
    return time_groups

def get_threat_intel_summary(events_df):
    """Generate threat intelligence summary for dashboard"""
    if events_df is None or events_df.empty:
        return {
            "event_types": {},
            "top_attacks": [],
            "geographic_data": {}
        }
    
    # Simple mockup data
    return {
        "event_types": {
            "intrusion": {"total": 25, "by_severity": {"critical": 3, "high": 8, "medium": 10, "low": 4}},
            "authentication": {"total": 18, "by_severity": {"critical": 1, "high": 3, "medium": 8, "low": 6}},
            "data_access": {"total": 12, "by_severity": {"high": 2, "medium": 5, "low": 5}},
            "malware": {"total": 7, "by_severity": {"critical": 2, "high": 3, "medium": 2}}
        },
        "top_attacks": [
            {"attack_type": "brute force", "count": 12, "severity": "high"},
            {"attack_type": "SQL injection", "count": 8, "severity": "critical"},
            {"attack_type": "ransomware", "count": 5, "severity": "critical"}
        ],
        "geographic_data": {
            "US": 15,
            "CN": 8,
            "RU": 7,
            "UK": 5,
            "Internal": 23
        }
    }