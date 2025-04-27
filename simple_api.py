#!/usr/bin/env python3

"""
AI Analytics API Service

This service provides endpoints for the dashboard to fetch security analytics data.
It integrates with the synthetic data generator and ML models for threat detection.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import datetime
import time
import logging
from flask import Flask, jsonify, request, Response
from pathlib import Path
import pymysql
from dotenv import load_dotenv
import xgboost as xgb

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='ai_analytics.log',
    filemode='a'
)
logger = logging.getLogger("AI_Analytics_API")

# Add the current directory and parent directory to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Try to import from synthetic data modules
try:
    # Import from synthetic_data_integration.py
    from synthetic_data.synthetic_data_integration import (
        get_synthetic_security_data as generate_synthetic_events,
        analyze_events
    )
    SYNTHETIC_DATA_AVAILABLE = True
    logger.info("✅ Successfully imported synthetic data modules")
except ImportError:
    # Try alternative imports
    try:
        # Make sure we're looking in the right place
        sys.path.append(os.path.join(current_dir, 'synthetic_data'))
        # Import from synthetic_data_generator.py
        from synthetic_data.synthetic_data_generator import (
            SecurityRAGGenerator,
            generate_batch as generate_synthetic_events,
            analyze_events
        )
        SYNTHETIC_DATA_AVAILABLE = True
        logger.info("✅ Successfully imported synthetic data generator")
    except ImportError as e:
        logger.warning(f"⚠️ Could not import synthetic data modules: {e}")
        SYNTHETIC_DATA_AVAILABLE = False
        
        # Define fallback functions
        def generate_synthetic_events(limit=100, event_types=None):
            """Generate synthetic security events if the actual module is not available"""
            logger.info(f"Generating {limit} synthetic events")
            events = []
            
            # Event types to generate
            if event_types is None:
                event_types = ["intrusion", "authentication", "data_access", "malware"]
                
            # Generate random events
            for i in range(limit):
                event_type = np.random.choice(event_types)
                timestamp = datetime.datetime.now() - datetime.timedelta(minutes=np.random.randint(0, 60))
                
                # Create base event
                event = {
                    "id": i,
                    "event_type": event_type,
                    "timestamp": timestamp.isoformat(),
                    "severity": np.random.choice(["critical", "high", "medium", "low", "info"], p=[0.05, 0.15, 0.3, 0.3, 0.2]),
                    "is_malicious": np.random.choice([True, False], p=[0.3, 0.7]),
                    "description": f"Simulated {event_type} event {i}"
                }
                
                # Add event-specific fields
                if event_type == "intrusion":
                    event.update({
                        "source_ip": f"192.168.{np.random.randint(1, 255)}.{np.random.randint(1, 255)}",
                        "destination_ip": f"10.0.{np.random.randint(1, 255)}.{np.random.randint(1, 255)}",
                        "protocol": np.random.choice(["TCP", "UDP", "HTTP", "HTTPS"]),
                        "port": np.random.randint(1, 65535),
                        "attack_type": np.random.choice(["port_scan", "brute_force", "dos", "sql_injection", "xss"])
                    })
                elif event_type == "authentication":
                    event.update({
                        "username": f"user{np.random.randint(1, 100)}",
                        "source_ip": f"192.168.{np.random.randint(1, 255)}.{np.random.randint(1, 255)}",
                        "success": np.random.choice([True, False], p=[0.7, 0.3]),
                        "browser": np.random.choice(["Chrome", "Firefox", "Safari", "Edge"]),
                        "location": np.random.choice(["US", "UK", "DE", "FR", "IN", "CN"])
                    })
                elif event_type == "data_access":
                    event.update({
                        "username": f"user{np.random.randint(1, 100)}",
                        "resource": f"resource_{np.random.randint(1, 20)}",
                        "action": np.random.choice(["read", "write", "delete", "modify"]),
                        "bytes_transferred": np.random.randint(1000, 1000000),
                        "success": np.random.choice([True, False], p=[0.9, 0.1])
                    })
                elif event_type == "malware":
                    event.update({
                        "hostname": f"host{np.random.randint(1, 50)}",
                        "malware_name": f"malware_{np.random.randint(1, 10)}",
                        "file_path": f"/var/log/file_{np.random.randint(1, 100)}.txt",
                        "malware_type": np.random.choice(["virus", "trojan", "ransomware", "spyware", "worm"])
                    })
                
                events.append(event)
            
            return events
        
        def analyze_events(events):
            """Analyze events if the actual module is not available"""
            if not events:
                return {"summary": {"total_events_analyzed": 0, "total_threats_detected": 0}}
            
            # Count threats by type and severity
            threat_types = {}
            severity_counts = {"critical": 0, "high": 0, "medium": 0, "low": 0, "info": 0}
            
            # Count malicious events
            malicious_count = 0
            for event in events:
                if event.get("is_malicious", False):
                    malicious_count += 1
                    
                    # Count by event type
                    event_type = event.get("event_type", "unknown")
                    threat_types[event_type] = threat_types.get(event_type, 0) + 1
                    
                    # Count by severity
                    severity = event.get("severity", "unknown").lower()
                    if severity in severity_counts:
                        severity_counts[severity] += 1
            
            return {
                "summary": {
                    "total_events_analyzed": len(events),
                    "total_threats_detected": malicious_count,
                    "threat_types": threat_types,
                    "severity_distribution": severity_counts
                }
            }

# Try to import ML models
try:
    # Add ML models directory to path - USE PROJECT-RELATIVE PATH
    ml_models_dir = os.path.join(current_dir, "ml_models")
    if os.path.exists(ml_models_dir):
        sys.path.append(ml_models_dir)
        
        # Import specific model prediction functions if available
        model_imports_successful = False
        
        # Try to import intrusion detection model
        try:
            from intrusion_detection.predict import predict_intrusions
            logger.info("✅ Successfully imported intrusion detection model")
            model_imports_successful = True
        except ImportError as e:
            logger.warning(f"⚠️ Could not import intrusion detection model: {e}")
            
        # Try to import RBA detection model
        try:
            from rba_detection.predict import detect_anomalies
            logger.info("✅ Successfully imported RBA detection model")
            model_imports_successful = True
        except ImportError as e:
            logger.warning(f"⚠️ Could not import RBA detection model: {e}")
            
        # Try to import text threat detection model
        try:
            sys.path.append(os.path.join(ml_models_dir, "text_threat_detection"))
            from text_threat_detection.predict import analyze_text
            logger.info("✅ Successfully imported text threat detection model")
            model_imports_successful = True
        except ImportError as e:
            logger.warning(f"⚠️ Could not import text threat detection model: {e}")
            
        ML_MODELS_AVAILABLE = model_imports_successful
    else:
        logger.warning(f"⚠️ ML models directory does not exist: {ml_models_dir}")
        ML_MODELS_AVAILABLE = False
except Exception as e:
    logger.error(f"❌ Error importing ML models: {e}")
    ML_MODELS_AVAILABLE = False

# Load environment variables
load_dotenv()

# Database connection settings
DB_CONFIGS = {
    "mysql": {
        "host": os.getenv("MYSQL_HOST"),
        "user": os.getenv("DEFAULT_MYSQL_USER"),
        "password": os.getenv("DEFAULT_MYSQL_PASSWORD"),
        "database": os.getenv("DEFAULT_MYSQL_DB")
    }
}

# In-memory cache for synthetic data
cache = {
    "security_data": None,
    "security_data_timestamp": None,
    "threat_analysis": None,
    "threat_analysis_timestamp": None,
    "all_analytics_data": None,
    "all_analytics_timestamp": None
}

# Create Flask app
app = Flask(__name__)

# Enable CORS
@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    return response

def connect_to_mysql():
    """Connect to MySQL database"""
    try:
        connection = pymysql.connect(
            host=DB_CONFIGS["mysql"]["host"],
            user=DB_CONFIGS["mysql"]["user"],
            password=DB_CONFIGS["mysql"]["password"],
            database=DB_CONFIGS["mysql"]["database"],
            charset='utf8mb4',
            cursorclass=pymysql.cursors.DictCursor
        )
        return connection
    except Exception as e:
        logger.error(f"Error connecting to MySQL: {e}")
        return None

def get_events_from_database(limit=100):
    """Get security events from MySQL database"""
    connection = connect_to_mysql()
    if not connection:
        return []
    
    try:
        with connection.cursor() as cursor:
            # Check if the rag_security_events table exists
            cursor.execute("""
            SELECT COUNT(*) as count
            FROM information_schema.tables
            WHERE table_schema = %s
            AND table_name = %s
            """, (DB_CONFIGS["mysql"]["database"], "rag_security_events"))
            
            result = cursor.fetchone()
            if result["count"] == 0:
                logger.warning("rag_security_events table does not exist")
                return []
            
            # Query events
            cursor.execute("""
            SELECT * FROM rag_security_events
            ORDER BY timestamp DESC
            LIMIT %s
            """, (limit,))
            
            events = cursor.fetchall()
            
            # Process events: parse JSON event_data field
            processed_events = []
            for event in events:
                try:
                    if "event_data" in event and event["event_data"]:
                        event_data = json.loads(event["event_data"])
                        # Merge parsed event_data with other fields
                        processed_event = {**event_data, "id": event["id"]}
                        processed_events.append(processed_event)
                    else:
                        processed_events.append(event)
                except:
                    # If parsing fails, just use the original event
                    processed_events.append(event)
            
            return processed_events
    except Exception as e:
        logger.error(f"Error getting events from database: {e}")
        return []
    finally:
        connection.close()

def get_synthetic_security_data(hours=24, limit=100, use_fake=True, force_refresh=False):
    """Get synthetic security data using the data generator"""
    global cache
    
    # Check if we have cached data and it's still fresh (less than 5 minutes old)
    if not force_refresh and cache["security_data"] is not None:
        cache_age = (datetime.datetime.now() - cache["security_data_timestamp"]).total_seconds()
        if cache_age < 30:  # Reduced from 300 seconds to 30 seconds for more frequent updates
            logger.info("Using cached security data")
            return cache["security_data"]
    
    try:
        # First try to get real data from the database
        db_events = get_events_from_database(limit=limit)
        
        if db_events and len(db_events) > 0:
            logger.info(f"Got {len(db_events)} events from database")
            # Convert to DataFrame
            df = pd.DataFrame(db_events)
            cache["security_data"] = df
            cache["security_data_timestamp"] = datetime.datetime.now()
            return df
        
        # If no database events or we're forcing fake data, generate synthetic events
        if not db_events or use_fake:
            logger.info("Generating synthetic events")
            # Generate synthetic events
            synthetic_data = generate_synthetic_events(limit=limit)
            
            # Check if synthetic_data is a DataFrame or a list
            if isinstance(synthetic_data, pd.DataFrame):
                df = synthetic_data
            else:
                # Convert to DataFrame if it's a list of dictionaries
                df = pd.DataFrame(synthetic_data)
            
            # Ensure datetime columns are properly formatted
            if not df.empty and 'timestamp' in df.columns:
                try:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                except:
                    pass
            
            cache["security_data"] = df
            cache["security_data_timestamp"] = datetime.datetime.now()
            return df
        else:
            logger.warning("No events found in database and not using fake data")
            return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error getting security data: {e}")
        return pd.DataFrame()

def get_latest_threat_analysis(events=None):
    """Get the latest threat analysis based on current data"""
    global cache
    
    # Check if we have cached analysis and it's still fresh (less than 5 minutes old)
    if cache["threat_analysis"] is not None:
        cache_age = (datetime.datetime.now() - cache["threat_analysis_timestamp"]).total_seconds()
        if cache_age < 30:  # Reduced from 300 seconds to 30 seconds
            logger.info("Using cached threat analysis")
            return cache["threat_analysis"]
    
    try:
        # Get the latest security data
        df = get_synthetic_security_data()
        
        if df.empty:
            logger.warning("No security data available for analysis")
            return {
                "timestamp": datetime.datetime.now().isoformat(),
                "summary": {
                    "total_events_analyzed": 0,
                    "total_threats_detected": 0,
                    "threat_types": {},
                    "severity_distribution": {
                        "critical": 0, "high": 0, "medium": 0, "low": 0, "info": 0
                    }
                }
            }
        
        # Convert DataFrame to list of dictionaries for analysis
        events = df.to_dict('records')
        
        # Analyze the events
        analysis = analyze_events(events)
        
        # Add timestamp
        analysis["timestamp"] = datetime.datetime.now().isoformat()
        
        # Cache the results
        cache["threat_analysis"] = analysis
        cache["threat_analysis_timestamp"] = datetime.datetime.now()
        
        return analysis
    except Exception as e:
        logger.error(f"Error getting threat analysis: {e}")
        return {
            "timestamp": datetime.datetime.now().isoformat(),
            "summary": {
                "total_events_analyzed": 0,
                "total_threats_detected": 0,
                "threat_types": {},
                "severity_distribution": {
                    "critical": 0, "high": 0, "medium": 0, "low": 0, "info": 0
                }
            }
        }

def get_event_summary(df=None):
    """Generate summary statistics for events"""
    if df is None or df.empty:
        df = get_synthetic_security_data()
    
    if df.empty:
        return {}
    
    try:
        # Count events by type
        event_types = {}
        if 'event_type' in df.columns:
            for event_type, count in df['event_type'].value_counts().items():
                event_types[event_type] = int(count)
        
        # Count by severity
        severity_counts = {}
        if 'severity' in df.columns:
            for severity, count in df['severity'].value_counts().items():
                severity_counts[severity] = int(count)
        
        # Count malicious events
        malicious_count = 0
        if 'is_malicious' in df.columns:
            malicious_count = int(df['is_malicious'].sum())
        
        return {
            "total_events": len(df),
            "event_types": event_types,
            "severity_counts": severity_counts,
            "malicious_count": malicious_count
        }
    except Exception as e:
        logger.error(f"Error generating event summary: {e}")
        return {}

def get_events_by_time(df=None, interval='hour'):
    """Group events by time intervals for time series visualization"""
    if df is None or df.empty:
        df = get_synthetic_security_data()
    
    if df.empty or 'timestamp' not in df.columns:
        return pd.DataFrame()
    
    try:
        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        
        # Group by time interval
        if interval == 'hour':
            df['time_group'] = df['timestamp'].dt.floor('H')
        elif interval == 'day':
            df['time_group'] = df['timestamp'].dt.floor('D')
        elif interval == 'minute':
            df['time_group'] = df['timestamp'].dt.floor('min')
        else:
            df['time_group'] = df['timestamp'].dt.floor('H')
        
        # Group by time and count events
        time_series = df.groupby('time_group').size().reset_index(name='count')
        time_series['time_group'] = time_series['time_group'].dt.strftime('%Y-%m-%d %H:%M:%S')
        
        # Count malicious events if available
        if 'is_malicious' in df.columns:
            malicious_df = df[df['is_malicious'] == True]
            if not malicious_df.empty:
                malicious_series = malicious_df.groupby('time_group').size().reset_index(name='malicious_count')
                malicious_series['time_group'] = malicious_series['time_group'].dt.strftime('%Y-%m-%d %H:%M:%S')
                
                # Merge with total counts
                time_series = pd.merge(time_series, malicious_series, on='time_group', how='left')
                time_series['malicious_count'] = time_series['malicious_count'].fillna(0).astype(int)
            else:
                time_series['malicious_count'] = 0
        
        return time_series
    except Exception as e:
        logger.error(f"Error grouping events by time: {e}")
        return pd.DataFrame()

def get_threat_intel_summary(df=None):
    """Generate threat intelligence summary"""
    if df is None or df.empty:
        df = get_synthetic_security_data()
    
    if df.empty:
        return {
            "event_types": {},
            "top_attacks": [],
            "geographic_data": {}
        }
    
    try:
        # Process by event type and severity
        event_types_intel = {}
        
        if 'event_type' in df.columns and 'severity' in df.columns:
            for event_type, group in df.groupby('event_type'):
                severity_counts = group['severity'].value_counts().to_dict()
                event_types_intel[event_type] = {
                    "total": len(group),
                    "by_severity": severity_counts
                }
        
        # Process attack types if available
        top_attacks = []
        if 'attack_type' in df.columns:
            attack_counts = df['attack_type'].value_counts().head(10).to_dict()
            for attack, count in attack_counts.items():
                severity = "high"  # Default severity
                if 'severity' in df.columns:
                    attack_df = df[df['attack_type'] == attack]
                    if not attack_df.empty:
                        most_common_severity = attack_df['severity'].value_counts().idxmax()
                        severity = most_common_severity
                
                top_attacks.append({
                    "attack_type": attack,
                    "count": int(count),
                    "severity": severity
                })
        
        # Geographic data if available
        geo_data = {}
        location_columns = [col for col in df.columns if col.lower() in ['country', 'location']]
        if location_columns:
            location_col = location_columns[0]
            geo_counts = df[location_col].value_counts().head(15).to_dict()
            geo_data = {country: int(count) for country, count in geo_counts.items()}
        
        return {
            "event_types": event_types_intel,
            "top_attacks": top_attacks,
            "geographic_data": geo_data
        }
    except Exception as e:
        logger.error(f"Error generating threat intel summary: {e}")
        return {
            "event_types": {},
            "top_attacks": [],
            "geographic_data": {}
        }

def get_all_analytics_data(force_refresh=False):
    """Get all analytics data in one call"""
    global cache
    
    # Check if we have cached data and it's still fresh (less than 2 minutes old)
    if not force_refresh and cache["all_analytics_data"] is not None:
        cache_age = (datetime.datetime.now() - cache["all_analytics_timestamp"]).total_seconds()
        if cache_age < 15:  # Reduced from 120 seconds to 15 seconds
            logger.info("Using cached all_analytics data")
            return cache["all_analytics_data"]
    
    try:
        # Get security data
        df = get_synthetic_security_data(force_refresh=force_refresh)
        
        # Convert DataFrame to list of dictionaries for the response
        events = df.to_dict('records') if not df.empty else []
        
        # Get threat analysis
        analysis = get_latest_threat_analysis()
        
        # Get event summary
        summary = get_event_summary(df)
        
        # Get time series data
        time_series = get_events_by_time(df).to_dict('records')
        
        # Get threat intel
        threat_intel = get_threat_intel_summary(df)
        
        # Add sample text threats for analysis
        sample_texts = [
            "Server login failed 5 times from IP 203.0.113.42",
            "Unusual data transfer detected: 500MB to external IP",
            "Potential SQL injection attempt detected in web logs",
            "Encrypted ransomware payload detected in email attachment",
            "Multiple authentication failures from different geographic locations",
            "Attempted access to admin console from unauthorized IP",
            "Malicious script detected in uploaded file",
            "Brute force SSH login attempts detected",
            "Data exfiltration attempt blocked by firewall",
            "Suspicious registry modifications detected on endpoint"
        ]
        
        text_threats = []
        for i, text in enumerate(sample_texts):
            severity = "high" if i < 3 else "medium" if i < 7 else "low"
            text_threats.append({
                "id": i + 1,
                "text": text,
                "timestamp": (datetime.datetime.now() - datetime.timedelta(minutes=i*5)).isoformat(),
                "severity": severity,
                "is_malicious": True if i < 7 else False
            })
        
        # Combine all data
        all_analytics = {
            "events": events,
            "analysis": analysis,
            "summary": summary,
            "time_series": time_series,
            "threat_intel": threat_intel,
            "text_threats": text_threats,
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        # Cache the results
        cache["all_analytics_data"] = all_analytics
        cache["all_analytics_timestamp"] = datetime.datetime.now()
        
        return all_analytics
    except Exception as e:
        logger.error(f"Error getting all analytics data: {e}")
        return {
            "events": [],
            "analysis": {},
            "summary": {},
            "time_series": [],
            "threat_intel": {},
            "text_threats": [],
            "timestamp": datetime.datetime.now().isoformat()
        }

# API Endpoints
@app.route('/health', methods=['GET'])
def health_check():
    """Simple health check endpoint"""
    logger.info("Health check request received")
    status = {
        "status": "healthy",
        "timestamp": datetime.datetime.now().isoformat()
    }
    return jsonify(status)

@app.route('/api/security-data', methods=['GET'])
def security_data_endpoint():
    """Endpoint to retrieve synthetic security data"""
    try:
        # Get parameters from request
        hours = request.args.get('hours', default=24, type=int)
        limit = request.args.get('limit', default=100, type=int)
        use_fake = request.args.get('use_fake', default='true', type=str).lower() == 'true'
        force_refresh = request.args.get('force_refresh', default='false', type=str).lower() == 'true'
        
        # Get security data
        df = get_synthetic_security_data(hours=hours, limit=limit, use_fake=use_fake, force_refresh=force_refresh)
        
        # Convert to serializable format
        data = df.to_dict('records') if not df.empty else []
        
        response = {
            "status": "success",
            "count": len(data),
            "data": data,
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        return jsonify(response)
    except Exception as e:
        logger.error(f"Error in /api/security-data endpoint: {e}")
        return jsonify({
            "status": "error",
            "message": str(e),
            "timestamp": datetime.datetime.now().isoformat()
        }), 500

@app.route('/api/threat-analysis', methods=['GET'])
def threat_analysis_endpoint():
    """Endpoint to retrieve threat analysis"""
    try:
        # Get threat analysis
        analysis = get_latest_threat_analysis()
        
        response = {
            "status": "success",
            "data": analysis,
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        return jsonify(response)
    except Exception as e:
        logger.error(f"Error in /api/threat-analysis endpoint: {e}")
        return jsonify({
            "status": "error",
            "message": str(e),
            "timestamp": datetime.datetime.now().isoformat()
        }), 500

@app.route('/api/event-summary', methods=['GET'])
def event_summary_endpoint():
    """Endpoint to retrieve event summary"""
    try:
        # Get parameters
        hours = request.args.get('hours', default=24, type=int)
        limit = request.args.get('limit', default=100, type=int)
        
        # Get security data
        df = get_synthetic_security_data(hours=hours, limit=limit)
        
        # Get summary
        summary = get_event_summary(df)
        
        response = {
            "status": "success",
            "data": summary,
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        return jsonify(response)
    except Exception as e:
        logger.error(f"Error in /api/event-summary endpoint: {e}")
        return jsonify({
            "status": "error",
            "message": str(e),
            "timestamp": datetime.datetime.now().isoformat()
        }), 500

@app.route('/api/events-by-time', methods=['GET'])
def events_by_time_endpoint():
    """Endpoint to retrieve events grouped by time"""
    try:
        # Get parameters
        hours = request.args.get('hours', default=24, type=int)
        limit = request.args.get('limit', default=500, type=int)
        interval = request.args.get('interval', default='hour', type=str)
        
        # Get security data
        df = get_synthetic_security_data(hours=hours, limit=limit)
        
        # Group by time
        time_series = get_events_by_time(df, interval=interval)
        
        response = {
            "status": "success",
            "data": time_series.to_dict('records'),
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        return jsonify(response)
    except Exception as e:
        logger.error(f"Error in /api/events-by-time endpoint: {e}")
        return jsonify({
            "status": "error",
            "message": str(e),
            "timestamp": datetime.datetime.now().isoformat()
        }), 500

@app.route('/api/threat-intel', methods=['GET'])
def threat_intel_endpoint():
    """Endpoint to retrieve threat intelligence summary"""
    try:
        # Get parameters
        hours = request.args.get('hours', default=24, type=int)
        limit = request.args.get('limit', default=500, type=int)
        
        # Get security data
        df = get_synthetic_security_data(hours=hours, limit=limit)
        
        # Get threat intel
        threat_intel = get_threat_intel_summary(df)
        
        response = {
            "status": "success",
            "data": threat_intel,
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        return jsonify(response)
    except Exception as e:
        logger.error(f"Error in /api/threat-intel endpoint: {e}")
        return jsonify({
            "status": "error",
            "message": str(e),
            "timestamp": datetime.datetime.now().isoformat()
        }), 500

@app.route('/api/all-analytics', methods=['GET'])
def all_analytics_endpoint():
    """Endpoint to retrieve all analytics data in one call"""
    try:
        # Get parameters
        force_refresh = request.args.get('force_refresh', default='false', type=str).lower() == 'true'
        
        # Get all analytics data
        all_analytics = get_all_analytics_data(force_refresh=force_refresh)
        
        response = {
            "status": "success",
            "data": all_analytics,
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        return jsonify(response)
    except Exception as e:
        logger.error(f"Error in /api/all-analytics endpoint: {e}")
        return jsonify({
            "status": "error",
            "message": str(e),
            "timestamp": datetime.datetime.now().isoformat()
        }), 500

@app.route('/api/ml-prediction', methods=['POST'])
def ml_prediction_endpoint():
    """Endpoint for ML model predictions"""
    try:
        data = request.json
        
        if not data:
            return jsonify({
                "status": "error",
                "message": "No data provided",
                "timestamp": datetime.datetime.now().isoformat()
            }), 400
        
        model_type = data.get('model_type', '')
        input_data = data.get('data', {})
        
        if model_type == 'intrusion':
            import pandas as pd
            import joblib
            
            input_df = pd.DataFrame([input_data])
            logger.info(f"Input data for intrusion detection: {input_data}")
            
            # Derived features
            input_df['login_failure_ratio'] = input_df.apply(
                lambda row: row['failed_logins'] / row['login_attempts'] if row['login_attempts'] > 0 else 0,
                axis=1
            )
            input_df['packet_duration_ratio'] = input_df['network_packet_size'] / (input_df['session_duration'] + 1)
            input_df['risk_score'] = (
                input_df['failed_logins'] * 0.3 +
                input_df['unusual_time_access'] * 0.4 +
                (1 - input_df['ip_reputation_score']) * 0.3
            )
            encryption_map = {'none': 0, 'basic': 1, 'advanced': 2}
            input_df['encryption_level'] = input_df['encryption_used'].map(encryption_map)
            
            model_dir = Path('model_artifacts')
            preprocessor_path = model_dir / 'intrusion_preprocessor.pkl'
            if not preprocessor_path.exists():
                return jsonify({
                    "status": "error",
                    "message": "Intrusion preprocessor not found",
                    "timestamp": datetime.datetime.now().isoformat()
                }), 404
            
            preprocessor = joblib.load(preprocessor_path)
            
            try:
                X_transformed = preprocessor.transform(input_df)

                # Heuristic prediction (since model compatibility issues)
                risk_indicators = 0
                if input_data.get('failed_logins', 0) > 2:
                    risk_indicators += 1
                if input_data.get('unusual_time_access', 0) == 1:
                    risk_indicators += 1
                if input_data.get('ip_reputation_score', 1.0) < 0.5:
                    risk_indicators += 1
                if input_data.get('encryption_used', '') == 'none':
                    risk_indicators += 1
                
                prediction = 1 if risk_indicators >= 2 else 0
                probability = min(0.95, risk_indicators * 0.25)
                
                result = {
                    "prediction": prediction,
                    "probability": probability,
                    "risk_level": "High" if probability > 0.7 else "Medium" if probability > 0.4 else "Low",
                    "risk_indicators": risk_indicators,
                    "analyzed_features": {
                        "failed_logins": input_data.get('failed_logins', 0),
                        "unusual_time_access": input_data.get('unusual_time_access', 0),
                        "ip_reputation_score": input_data.get('ip_reputation_score', 1.0),
                        "encryption_used": input_data.get('encryption_used', '')
                    },
                    "note": "This is a rule-based prediction due to XGBoost compatibility issues",
                    "timestamp": datetime.datetime.now().isoformat()
                }
                
                return jsonify({
                    "status": "success",
                    "result": result,
                    "timestamp": datetime.datetime.now().isoformat()
                })
            
            except Exception as e:
                logger.error(f"Error in transformation/prediction: {e}")
                return jsonify({
                    "status": "error",
                    "message": f"Data processing error: {str(e)}",
                    "timestamp": datetime.datetime.now().isoformat()
                }), 500
        
        elif model_type == 'text_threat':
            if 'text' not in input_data:
                return jsonify({
                    "status": "error",
                    "message": "No text provided in data",
                    "timestamp": datetime.datetime.now().isoformat()
                }), 400
            
            from ml_models.text_threat_detection.model import TextThreatDetectionModel
            
            model_dir = Path('model_artifacts')
            model_path = model_dir / 'text_threat_model.pkl'
            vectorizer_path = model_dir / 'text_threat_vectorizer.pkl'
            label_encoder_path = model_dir / 'text_threat_label_encoder.pkl'
            
            if model_path.exists() and vectorizer_path.exists() and label_encoder_path.exists():
                model = TextThreatDetectionModel(
                    model_path=model_path,
                    vectorizer_path=vectorizer_path,
                    label_encoder_path=label_encoder_path
                )
                
                result = model.predict([input_data['text']])[0]
                
                return jsonify({
                    "status": "success",
                    "prediction": result,
                    "timestamp": datetime.datetime.now().isoformat()
                })
            else:
                return jsonify({
                    "status": "error",
                    "message": "Text threat model files not found",
                    "timestamp": datetime.datetime.now().isoformat()
                }), 404
        
        elif model_type == 'rba':
            try:
                from ml_models.rba_detection.predict import detect_anomalies
                import pandas as pd

                required_fields = [
                    'User ID', 'Login Timestamp', 'Round-Trip Time [ms]',
                    'IP Address', 'Country', 'User Agent String'
                ]
                missing = [f for f in required_fields if f not in input_data]
                if missing:
                    return jsonify({
                        "status": "error",
                        "message": f"Missing required RBA fields: {missing}",
                        "timestamp": datetime.datetime.now().isoformat()
                    }), 400
                
                input_df = pd.DataFrame([input_data])
                results = detect_anomalies(data=input_df)
                
                if results is not None and not results.empty:
                    result_row = results.iloc[0]
                    result = {
                        "anomaly_detected": int(result_row.get("anomaly_detected", 0)),
                        "anomaly_score": float(result_row.get("anomaly_score", 0)),
                        "anomaly_probability": float(result_row.get("anomaly_probability", 0)),
                        "risk_category": str(result_row.get("risk_category", "")),
                        "User ID": result_row.get("User ID", ""),
                        "Country": result_row.get("Country", ""),
                        "ip_version": int(result_row.get("ip_version", 4)),
                        "device_type": result_row.get("device_type", ""),
                        "hour": int(result_row.get("hour", 0)),
                    }
                    return jsonify({
                        "status": "success",
                        "result": result,
                        "timestamp": datetime.datetime.now().isoformat()
                    })
                
                else:
                    return jsonify({
                        "status": "error",
                        "message": "No prediction generated",
                        "timestamp": datetime.datetime.now().isoformat()
                    }), 500
            
            except Exception as e:
                logger.error(f"Error in RBA prediction: {e}")
                return jsonify({
                    "status": "error",
                    "message": str(e),
                    "timestamp": datetime.datetime.now().isoformat()
                }), 500
        
        else:
            return jsonify({
                "status": "error",
                "message": f"Unknown model type: {model_type}",
                "timestamp": datetime.datetime.now().isoformat()
            }), 400
    
    except Exception as e:
        logger.error(f"Error in /api/ml-prediction endpoint: {e}")
        return jsonify({
            "status": "error",
            "message": str(e),
            "timestamp": datetime.datetime.now().isoformat()
        }), 500

@app.route('/api/analyze-text', methods=['POST'])
def analyze_text_endpoint():
    """Endpoint specifically for text threat analysis"""
    try:
        data = request.json
        if not data or 'text' not in data:
            return jsonify({
                "status": "error",
                "message": "No text provided for analysis",
                "timestamp": datetime.datetime.now().isoformat()
            }), 400
        
        text_input = data['text']
        logger.info(f"Analyzing text: {text_input[:50]}...")
        
        from ml_models.text_threat_detection.model import TextThreatDetectionModel
        
        model_dir = Path('model_artifacts')
        model_path = model_dir / 'text_threat_model.pkl'
        vectorizer_path = model_dir / 'text_threat_vectorizer.pkl'
        label_encoder_path = model_dir / 'text_threat_label_encoder.pkl'
        
        if model_path.exists() and vectorizer_path.exists() and label_encoder_path.exists():
            model = TextThreatDetectionModel(
                model_path=model_path,
                vectorizer_path=vectorizer_path,
                label_encoder_path=label_encoder_path
            )
            result = model.predict([text_input])[0]
            
            response = {
                "text": text_input,
                "prediction": result["predicted_threat"],
                "confidence": result["confidence"],
                "timestamp": datetime.datetime.now().isoformat()
            }
            
            return jsonify({
                "status": "success",
                "result": response,
                "timestamp": datetime.datetime.now().isoformat()
            })
        else:
            return jsonify({
                "status": "error",
                "message": f"Required model files not found in {model_dir}",
                "timestamp": datetime.datetime.now().isoformat()
            }), 500
    
    except Exception as e:
        logger.error(f"Error analyzing text: {e}")
        return jsonify({
            "status": "error",
            "message": str(e),
            "timestamp": datetime.datetime.now().isoformat()
        }), 500


@app.route('/api/text-threats', methods=['GET'])
def text_threats_endpoint():
    """Endpoint to retrieve sample text threats for analysis"""
    try:
        # Sample text threats for analysis
        sample_texts = [
            "Server login failed 5 times from IP 203.0.113.42",
            "Unusual data transfer detected: 500MB to external IP",
            "Potential SQL injection attempt detected in web logs",
            "Encrypted ransomware payload detected in email attachment",
            "Multiple authentication failures from different geographic locations",
            "Attempted access to admin console from unauthorized IP",
            "Malicious script detected in uploaded file",
            "Brute force SSH login attempts detected",
            "Data exfiltration attempt blocked by firewall",
            "Suspicious registry modifications detected on endpoint"
        ]
        
        threats = []
        for i, text in enumerate(sample_texts):
            severity = "high" if i < 3 else "medium" if i < 7 else "low"
            threats.append({
                "id": i + 1,
                "text": text,
                "timestamp": (datetime.datetime.now() - datetime.timedelta(minutes=i*5)).isoformat(),
                "severity": severity,
                "is_malicious": True if i < 7 else False
            })
        
        response = {
            "status": "success",
            "data": threats,
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        return jsonify(response)
    except Exception as e:
        logger.error(f"Error in /api/text-threats endpoint: {e}")
        return jsonify({
            "status": "error",
            "message": str(e),
            "timestamp": datetime.datetime.now().isoformat()
        }), 500

if __name__ == '__main__':
    # Report on the availability of components
    print(f"Synthetic Data Generator available: {SYNTHETIC_DATA_AVAILABLE}")
    print(f"ML Models available: {ML_MODELS_AVAILABLE}")
    
    # Start the API service
    print(f"Starting AI Analytics API on http://0.0.0.0:5000/")
    app.run(host='0.0.0.0', port=5000, debug=False)
