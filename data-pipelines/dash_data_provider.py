#!/usr/bin/env python3

"""
Data Provider Module for Cybersecurity Dashboard

This module handles:
1. Database connections
2. Data retrieval from various sources
3. Data processing for visualization
4. Integration with AI analytics via API
"""

import os
import sys
import pandas as pd
import numpy as np
import pymysql
import pymongo
from pymongo import MongoClient
from dotenv import load_dotenv
from urllib.parse import quote_plus
import datetime
import random
import time
import json
from pathlib import Path

# Add parent directory to sys.path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Add the current directory to the Python path for the API client
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Import the API client
try:
    from api_client import (
        get_synthetic_security_data, 
        get_latest_threat_analysis,
        get_event_summary, 
        get_events_by_time,
        get_threat_intel_summary,
        get_ai_analytics_data,
        api_client
    )
    
    API_CLIENT_AVAILABLE = True
    print("✅ Successfully imported AI Analytics API client")
    
except ImportError as e:
    print(f"⚠️ AI Analytics API client not available: {e}")
    API_CLIENT_AVAILABLE = False
    
    # Define placeholder functions for when API client is not available
    def get_synthetic_security_data(hours=24, limit=100, use_fake=True, force_refresh=False):
        """Placeholder for synthetic data when API client is not available"""
        import pandas as pd
        # Return empty DataFrame
        return pd.DataFrame()
    
    def get_latest_threat_analysis():
        """Placeholder for threat analysis when API client is not available"""
        return {
            "timestamp": "",
            "summary": {
                "total_events_analyzed": 0,
                "total_threats_detected": 0,
                "threat_types": {},
                "severity_distribution": {
                    "critical": 0, "high": 0, "medium": 0, "low": 0, "info": 0
                }
            }
        }
    
    def get_event_summary(df):
        """Placeholder for event summary when API client is not available"""
        return {}
    
    def get_events_by_time(df, interval='hour'):
        """Placeholder for events by time when API client is not available"""
        import pandas as pd
        return pd.DataFrame()
    
    def get_threat_intel_summary(df):
        """Placeholder for threat intel summary when API client is not available"""
        return {
            "event_types": {},
            "top_attacks": [],
            "geographic_data": {}
        }
    
    def get_ai_analytics_data():
        """Placeholder for all AI analytics data"""
        return {
            "events": [],
            "analysis": {},
            "summary": {},
            "time_series": [],
            "threat_intel": {}
        }

# Load environment variables
load_dotenv()

# Define color scheme
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

# Database connections
mysql_conn = None
mongodb_db = None

# Create custom MySQL config for Shreyaa (who created the tables)
MYSQL_CONFIG = {
    'host': os.getenv('MYSQL_HOST'),
    'user': os.getenv('DEFAULT_MYSQL_USER'),
    'password': os.getenv('DEFAULT_MYSQL_PASSWORD'),
    'database': os.getenv('DEFAULT_MYSQL_DB')
}

# MongoDB config
MONGODB_CONFIG = {
    'uri': f"mongodb://{quote_plus(os.getenv('DEFAULT_MONGODB_USER'))}:{quote_plus(os.getenv('DEFAULT_MONGODB_PASSWORD'))}@{os.getenv('MONGODB_HOST')}:27017/{os.getenv('DEFAULT_MONGODB_DB')}?authSource={os.getenv('DEFAULT_MONGODB_DB')}",
    'database': os.getenv('DEFAULT_MONGODB_DB')
}

# Configure API URL from environment variable if available
if API_CLIENT_AVAILABLE and os.getenv('AI_ANALYTICS_API_URL'):
    api_client.base_url = os.getenv('AI_ANALYTICS_API_URL')

def initialize_data_connections():
    """Initialize connections to databases"""
    global mysql_conn, mongodb_db
    
    # Connect to MySQL
    mysql_conn = connect_mysql(MYSQL_CONFIG)
    
    # Connect to MongoDB
    mongodb_db = connect_mongodb(MONGODB_CONFIG)
    
    # Check AI Analytics API connection if available
    if API_CLIENT_AVAILABLE:
        try:
            health = api_client.check_health()
            print(f"AI Analytics API Status: {health.get('status', 'unknown')}")
        except Exception as e:
            print(f"Error connecting to AI Analytics API: {e}")
    
    return mysql_conn is not None and mongodb_db is not None

def connect_mysql(config):
    """Connect to MySQL database"""
    try:
        connection = pymysql.connect(
            host=config['host'],
            user=config['user'],
            password=config['password'],
            database=config['database'],
            charset='utf8mb4',
            cursorclass=pymysql.cursors.DictCursor
        )
        print(f"✅ Connected to MySQL: {config['user']}@{config['host']}/{config['database']}")
        return connection
    except Exception as e:
        print(f"❌ Error connecting to MySQL: {e}")
        return None

def connect_mongodb(config):
    """Connect to MongoDB database"""
    try:
        client = MongoClient(config['uri'], serverSelectionTimeoutMS=5000)
        db = client[config['database']]
        
        # Test the connection
        client.admin.command('ping')
        print(f"✅ Connected to MongoDB: {config['database']}")
        return db
    except Exception as e:
        print(f"❌ Error connecting to MongoDB: {e}")
        return None

def get_intrusion_data():
    """Fetch intrusion detection data from MySQL"""
    if mysql_conn is None:
        return pd.DataFrame()
    
    try:
        with mysql_conn.cursor() as cursor:
            cursor.execute("""
            SELECT 
                session_id,
                protocol_type, 
                encryption_used,
                browser_type, 
                login_attempts,
                failed_logins,
                unusual_time_access,
                attack_detected,
                ip_reputation_score,
                session_duration,
                network_packet_size,
                COUNT(*) as count
            FROM intrusion_detection
            GROUP BY 
                session_id,
                protocol_type, 
                encryption_used,
                browser_type,
                login_attempts,
                failed_logins,
                unusual_time_access,
                attack_detected,
                ip_reputation_score,
                session_duration,
                network_packet_size
            """)
            result = cursor.fetchall()
            return pd.DataFrame(result)
    except Exception as e:
        print(f"Error fetching intrusion data: {e}")
        return pd.DataFrame()

def get_rba_data(limit=10000):
    """Fetch risk-based authentication data from MySQL"""
    if mysql_conn is None:
        return pd.DataFrame()
    
    try:
        with mysql_conn.cursor() as cursor:
            cursor.execute(f"""
            SELECT 
                login_timestamp,
                userid,
                country,
                region,
                city,
                ip_address,
                round_trip_time_ms,
                asn,
                user_agent
            FROM rba_logins
            ORDER BY login_timestamp DESC
            LIMIT {limit}
            """)
            result = cursor.fetchall()
            return pd.DataFrame(result)
    except Exception as e:
        print(f"Error fetching RBA data: {e}")
        return pd.DataFrame()

def get_text_threat_data(limit=1000):
    """Fetch text-based threat data from MongoDB"""
    if mongodb_db is None:
        return pd.DataFrame()
    
    try:
        # Convert MongoDB documents to DataFrame
        cursor = mongodb_db.text_based_detection.find().limit(limit)
        data = list(cursor)
        return pd.DataFrame(data)
    except Exception as e:
        print(f"Error fetching text threat data: {e}")
        return pd.DataFrame()

def get_real_time_alerts(limit=10):
    """Get real-time alerts - use AI analytics API if available, otherwise generate mock data"""
    if API_CLIENT_AVAILABLE:
        try:
            # Fetch alerts from AI analytics API
            events_df = get_synthetic_security_data(hours=1, limit=limit, force_refresh=True)
            if not events_df.empty:
                # Convert to the format expected by the dashboard
                alerts = []
                for _, row in events_df.iterrows():
                    alert = {
                        "timestamp": row.get('timestamp', datetime.datetime.now()).strftime('%H:%M:%S'),
                        "type": row.get('event_type', 'Unknown'),
                        "status": row.get('severity', 'Info').title(),
                        "message": row.get('description', 'No description available'),
                        "source_ip": row.get('source_ip', 'Unknown'),
                        "country": row.get('country', 'Unknown')
                    }
                    alerts.append(alert)
                return pd.DataFrame(alerts)
        except Exception as e:
            print(f"Error getting real-time alerts from AI analytics API: {e}")
    
    # If API not available or error occurred, generate mock data
    alerts = []
    for i in range(limit):
        timestamp = datetime.datetime.now() - datetime.timedelta(minutes=random.randint(0, 60))
        alert_types = ["Login", "Traffic", "Auth", "Data", "System"]
        statuses = ["Critical", "Warning", "Info"]
        messages = [
            "Multiple failed logins from user admin",
            "DDoS attack detected from IP range",
            "Login from unusual location",
            "Port scan detected on public-facing servers",
            "Possible credential stuffing attack",
            "Unusual data transfer detected",
            "Firewall rule violation",
            "Malware signature detected",
            "Abnormal user behavior detected",
            "Configuration change detected"
        ]
        
        alerts.append({
            "timestamp": timestamp.strftime('%H:%M:%S'),
            "type": random.choice(alert_types),
            "status": random.choice(statuses),
            "message": random.choice(messages),
            "source_ip": f"192.168.{random.randint(1, 255)}.{random.randint(1, 255)}",
            "country": random.choice(["US", "CN", "RU", "GB", "DE", "BR", "IN"])
        })
    
    return pd.DataFrame(alerts).sort_values(by="timestamp", ascending=False)

def process_intrusion_data():
    """Process intrusion detection data for visualization"""
    intrusion_df = get_intrusion_data()
    
    if intrusion_df.empty:
        return {}
    
    # Attack distribution
    attack_summary = intrusion_df.groupby('attack_detected')['count'].sum().reset_index()
    attack_summary['label'] = attack_summary['attack_detected'].apply(
        lambda x: 'Attack' if x == 1 else 'Normal'
    )
    
    # Protocol distribution
    protocol_summary = intrusion_df.groupby(['protocol_type', 'attack_detected'])['count'].sum().reset_index()
    
    # Browser distribution
    browser_summary = intrusion_df.groupby(['browser_type', 'attack_detected'])['count'].sum().reset_index()
    
    # Encryption distribution
    encryption_summary = intrusion_df.groupby(['encryption_used', 'attack_detected'])['count'].sum().reset_index()
    
    # Failed logins vs attacks
    failed_login_summary = intrusion_df.groupby(['failed_logins', 'attack_detected'])['count'].sum().reset_index()
    
    # Packet size vs. duration scatter data
    packet_duration = intrusion_df[['network_packet_size', 'session_duration', 'attack_detected']].copy()
    
    # IP reputation score distribution
    ip_score_summary = intrusion_df.groupby(['ip_reputation_score', 'attack_detected'])['count'].sum().reset_index()
    
    # Time-based patterns (using random data for demo - replace with real timestamps in production)
    np.random.seed(42)  # For reproducibility
    hours = np.arange(24)
    attacks_by_hour = np.random.randint(5, 30, size=24)
    normal_by_hour = np.random.randint(50, 200, size=24)
    
    time_pattern_df = pd.DataFrame({
        'hour': hours,
        'attacks': attacks_by_hour,
        'normal': normal_by_hour
    })
    
    return {
        'attack_summary': attack_summary,
        'protocol_summary': protocol_summary,
        'browser_summary': browser_summary,
        'encryption_summary': encryption_summary,
        'failed_login_summary': failed_login_summary,
        'packet_duration': packet_duration,
        'ip_score_summary': ip_score_summary,
        'time_pattern_df': time_pattern_df
    }

def process_rba_data():
    """Process risk-based authentication data for visualization"""
    rba_df = get_rba_data()
    
    if rba_df.empty:
        return {}
    
    # Convert timestamp to datetime if it's not already
    if 'login_timestamp' in rba_df.columns and not pd.api.types.is_datetime64_any_dtype(rba_df['login_timestamp']):
        rba_df['login_timestamp'] = pd.to_datetime(rba_df['login_timestamp'], errors='coerce')
    
    # Country distribution
    country_counts = rba_df['country'].value_counts().reset_index()
    country_counts.columns = ['country', 'count']
    
    # RTT distribution
    rtt_df = rba_df[['round_trip_time_ms']].dropna().copy()
    
    # Login time distribution
    if 'login_timestamp' in rba_df.columns:
        rba_df['hour'] = rba_df['login_timestamp'].dt.hour
        hour_counts = rba_df['hour'].value_counts().sort_index().reset_index()
        hour_counts.columns = ['hour', 'count']
    else:
        hour_counts = pd.DataFrame(columns=['hour', 'count'])
    
    # User agent analysis
    if 'user_agent' in rba_df.columns:
        # Extract device/browser info (simplified for demo)
        def categorize_user_agent(ua):
            ua = str(ua).lower()
            if 'mobile' in ua or 'android' in ua or 'iphone' in ua:
                return 'Mobile'
            elif 'tablet' in ua or 'ipad' in ua:
                return 'Tablet'
            elif 'bot' in ua or 'crawler' in ua or 'spider' in ua:
                return 'Bot'
            else:
                return 'Desktop'
                
        rba_df['device_type'] = rba_df['user_agent'].apply(categorize_user_agent)
        device_counts = rba_df['device_type'].value_counts().reset_index()
        device_counts.columns = ['device_type', 'count']
    else:
        device_counts = pd.DataFrame(columns=['device_type', 'count'])
    
    # Login frequency by user (top 10)
    user_login_counts = rba_df['userid'].value_counts().head(10).reset_index()
    user_login_counts.columns = ['userid', 'login_count']
    
    # Login heatmap data (day of week vs hour)
    if 'login_timestamp' in rba_df.columns:
        rba_df['day_of_week'] = rba_df['login_timestamp'].dt.dayofweek
        rba_df['hour'] = rba_df['login_timestamp'].dt.hour
        
        # Group by day and hour
        heatmap_data = rba_df.groupby(['day_of_week', 'hour']).size().reset_index()
        heatmap_data.columns = ['day_of_week', 'hour', 'count']
        
        # Convert day numbers to names
        day_map = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 
                   4: 'Friday', 5: 'Saturday', 6: 'Sunday'}
        heatmap_data['day_name'] = heatmap_data['day_of_week'].map(day_map)
    else:
        heatmap_data = pd.DataFrame(columns=['day_of_week', 'hour', 'count', 'day_name'])
    
    return {
        'country_counts': country_counts,
        'rtt_df': rtt_df,
        'hour_counts': hour_counts,
        'device_counts': device_counts,
        'user_login_counts': user_login_counts,
        'heatmap_data': heatmap_data
    }

def process_text_threat_data():
    """Process text-based threat data for visualization"""
    text_df = get_text_threat_data()
    
    if text_df.empty:
        return {}
    
    # Sample threat types for visualization
    threat_types = [
        {"type": "Phishing", "count": 327, "severity": "High"},
        {"type": "Malware", "count": 205, "severity": "Critical"}, 
        {"type": "Ransomware", "count": 178, "severity": "Critical"},
        {"type": "Data Leak", "count": 156, "severity": "Medium"},
        {"type": "Social Engineering", "count": 143, "severity": "High"},
        {"type": "Insider Threat", "count": 98, "severity": "High"},
        {"type": "DDoS", "count": 87, "severity": "Medium"},
        {"type": "Zero-day", "count": 43, "severity": "Critical"},
    ]
    
    # Convert to DataFrame
    threat_df = pd.DataFrame(threat_types)
    
    # Severity distribution
    severity_counts = threat_df.groupby('severity')['count'].sum().reset_index()
    
    # Trend data (sample data - replace with your actual time series data)
    dates = pd.date_range(start='2025-04-01', end='2025-04-21')
    trend_data = pd.DataFrame({
        'date': dates,
        'phishing': np.random.randint(10, 30, size=len(dates)),
        'malware': np.random.randint(5, 25, size=len(dates)),
        'ransomware': np.random.randint(3, 15, size=len(dates)),
    })
    
    # Sample keyword data (replace with actual NLP results)
    keywords = [
        {"word": "password", "count": 312, "category": "Credentials"},
        {"word": "login", "count": 287, "category": "Credentials"},
        {"word": "account", "count": 245, "category": "Credentials"},
        {"word": "virus", "count": 198, "category": "Malware"},
        {"word": "malware", "count": 176, "category": "Malware"},
        {"word": "phishing", "count": 165, "category": "Attack"},
        {"word": "ransomware", "count": 142, "category": "Malware"},
        {"word": "vulnerability", "count": 132, "category": "System"},
        {"word": "encryption", "count": 118, "category": "Security"},
        {"word": "firewall", "count": 95, "category": "Security"},
    ]
    keywords_df = pd.DataFrame(keywords)
    
    return {
        'threat_df': threat_df,
        'severity_counts': severity_counts,
        'trend_data': trend_data,
        'keywords_df': keywords_df,
        'raw_data': text_df.head(100)  # Include some raw data for tables
    }

def update_real_time_data():
    """Get real-time monitoring data for dashboard"""
    # Try to get system status from API if available
    if API_CLIENT_AVAILABLE:
        try:
            # Check API health
            health = api_client.check_health()
            
            # If API is healthy, get threat analysis to update system status
            if health.get('status') == 'healthy':
                analysis = get_latest_threat_analysis()
                if analysis and 'summary' in analysis:
                    summary = analysis['summary']
                    total_threats = summary.get('total_threats_detected', 0)
                    
                    # Create system status based on API data
                    system_status = [
                        {"system": "AI Analytics Service", "status": "Healthy", "last_check": "just now", "uptime": "99.98%"},
                        {"system": "Intrusion Detection", "status": "Healthy" if total_threats < 5 else "Warning" if total_threats < 10 else "Critical", 
                         "last_check": "just now", "uptime": "99.99%"},
                        {"system": "Authentication Service", "status": "Healthy", "last_check": "1 min ago", "uptime": "99.95%"},
                        {"system": "Network Monitor", "status": "Healthy", "last_check": "2 min ago", "uptime": "99.97%"},
                        {"system": "Database Server", "status": "Healthy", "last_check": "1 min ago", "uptime": "99.99%"},
                    ]
                    
                    return system_status
        except Exception as e:
            print(f"Error getting system status from API: {e}")
    
    # Mock system status for demonstration (fallback)
    system_status = [
        {"system": "Intrusion Detection System", "status": "Healthy", "last_check": "2 min ago", "uptime": "99.98%"},
        {"system": "Firewall", "status": "Healthy", "last_check": "1 min ago", "uptime": "99.99%"},
        {"system": "Authentication Service", "status": "Warning", "last_check": "3 min ago", "uptime": "98.52%"},
        {"system": "Network Monitor", "status": "Healthy", "last_check": "1 min ago", "uptime": "99.95%"},
        {"system": "Database Server", "status": "Critical", "last_check": "5 min ago", "uptime": "97.33%"},
        {"system": "Log Collection", "status": "Healthy", "last_check": "2 min ago", "uptime": "99.91%"},
    ]
    
    return system_status

# Close database connections when app stops
def shutdown_db_connections():
    global mysql_conn
    try:
        if mysql_conn is not None:
            # Check if connection is still open
            try:
                mysql_conn.ping(reconnect=False)
                mysql_conn.close()
                print("MySQL connection closed")
            except:
                # Connection already closed
                pass
    except Exception as e:
        print(f"Error closing MySQL connection: {e}")