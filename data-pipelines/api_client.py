#!/usr/bin/env python3

"""
AI Analytics API Client

This module provides functions to fetch data from the AI Analytics API service.
It replaces direct imports from the ai-analytics project with API calls.
"""

import os
import sys
import json
import pandas as pd
import requests
from datetime import datetime
import logging
from urllib.parse import urljoin
import time
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("API_Client")

# Default API settings (can be overridden with environment variables)
API_BASE_URL = os.environ.get("AI_ANALYTICS_API_URL", "http://localhost:5000")
API_TIMEOUT = int(os.environ.get("AI_ANALYTICS_API_TIMEOUT", "10"))  # seconds
API_RETRY_COUNT = int(os.environ.get("AI_ANALYTICS_API_RETRY_COUNT", "3"))  # number of retries
API_RETRY_DELAY = int(os.environ.get("AI_ANALYTICS_API_RETRY_DELAY", "2"))  # seconds between retries

class APIClient:
    """Client for the AI Analytics API service"""
    
    def __init__(self, base_url=API_BASE_URL, timeout=API_TIMEOUT, retry_count=API_RETRY_COUNT, retry_delay=API_RETRY_DELAY):
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.retry_count = retry_count
        self.retry_delay = retry_delay
        self.session = requests.Session()
        
        # Cache for responses
        self.cache = {}
        self.cache_expiry = {}  # timestamp when cache expires
        self.cache_ttl = 60  # seconds
    
    def _make_request(self, endpoint, method='GET', params=None, data=None, use_cache=True):
        """Make an HTTP request to the API with retry logic"""
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        cache_key = f"{method}:{url}:{json.dumps(params)}:{json.dumps(data)}"
        
        # Check if we have a cached response that's not expired
        if use_cache and method == 'GET' and cache_key in self.cache:
            if datetime.now().timestamp() < self.cache_expiry.get(cache_key, 0):
                logger.debug(f"Using cached response for {url}")
                return self.cache[cache_key]
        
        # Not in cache or cache expired, make the request
        for attempt in range(self.retry_count):
            try:
                response = self.session.request(
                    method=method,
                    url=url,
                    params=params,
                    json=data,
                    timeout=self.timeout
                )
                
                # Raise exception for HTTP errors
                response.raise_for_status()
                
                # Parse JSON response
                result = response.json()
                
                # Cache the result if it's a GET request
                if method == 'GET' and use_cache:
                    self.cache[cache_key] = result
                    self.cache_expiry[cache_key] = datetime.now().timestamp() + self.cache_ttl
                
                return result
            
            except requests.exceptions.RequestException as e:
                logger.warning(f"API request error ({url}) attempt {attempt+1}/{self.retry_count}: {e}")
                if attempt < self.retry_count - 1:
                    # Wait before retrying
                    time.sleep(self.retry_delay)
                else:
                    # Last attempt failed, return error response
                    logger.error(f"API request failed after {self.retry_count} attempts: {e}")
                    return {
                        "status": "error",
                        "message": str(e),
                        "timestamp": datetime.now().isoformat()
                    }
    
    def clear_cache(self):
        """Clear the response cache"""
        self.cache = {}
        self.cache_expiry = {}
        logger.info("API client cache cleared")
    
    def check_health(self):
        """Check if the API is healthy"""
        try:
            return self._make_request('health', use_cache=False)
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {"status": "error", "message": str(e)}
    
    def get_security_data(self, hours=24, limit=100, use_fake=True, force_refresh=False):
        """Get synthetic security data from the API"""
        params = {
            'hours': hours,
            'limit': limit,
            'use_fake': str(use_fake).lower(),
            'force_refresh': str(force_refresh).lower()
        }
        
        response = self._make_request('api/security-data', params=params, use_cache=not force_refresh)
        
        if response.get('status') == 'success' and 'data' in response:
            # Convert to DataFrame
            df = pd.DataFrame(response['data'])
            
            # Convert string timestamps back to datetime objects
            for col in df.columns:
                if col in ['timestamp', 'login_timestamp', 'generated_at']:
                    try:
                        df[col] = pd.to_datetime(df[col])
                    except:
                        pass
            
            return df
        else:
            logger.warning(f"Failed to get security data: {response.get('message', 'Unknown error')}")
            return pd.DataFrame()
    
    def get_threat_analysis(self):
        """Get latest threat analysis from the API"""
        response = self._make_request('api/threat-analysis')
        
        if response.get('status') == 'success' and 'data' in response:
            return response['data']
        else:
            logger.warning(f"Failed to get threat analysis: {response.get('message', 'Unknown error')}")
            return {
                "timestamp": datetime.now().isoformat(),
                "summary": {
                    "total_events_analyzed": 0,
                    "total_threats_detected": 0,
                    "threat_types": {},
                    "severity_distribution": {
                        "critical": 0, "high": 0, "medium": 0, "low": 0, "info": 0
                    }
                }
            }
    
    def get_event_summary(self, hours=24, limit=100):
        """Get event summary from the API"""
        params = {
            'hours': hours,
            'limit': limit
        }
        
        response = self._make_request('api/event-summary', params=params)
        
        if response.get('status') == 'success' and 'data' in response:
            return response['data']
        else:
            logger.warning(f"Failed to get event summary: {response.get('message', 'Unknown error')}")
            return {}
    
    def get_events_by_time(self, hours=24, limit=500, interval='hour'):
        """Get events grouped by time from the API"""
        params = {
            'hours': hours,
            'limit': limit,
            'interval': interval
        }
        
        response = self._make_request('api/events-by-time', params=params)
        
        if response.get('status') == 'success' and 'data' in response:
            # Convert to DataFrame
            df = pd.DataFrame(response['data'])
            
            # Convert string timestamps back to datetime objects
            for col in df.columns:
                if col in ['time_group']:
                    try:
                        df[col] = pd.to_datetime(df[col])
                    except:
                        pass
            
            return df
        else:
            logger.warning(f"Failed to get events by time: {response.get('message', 'Unknown error')}")
            return pd.DataFrame()
    
    def get_threat_intel(self, hours=24, limit=500):
        """Get threat intelligence summary from the API"""
        params = {
            'hours': hours,
            'limit': limit
        }
        
        response = self._make_request('api/threat-intel', params=params)
        
        if response.get('status') == 'success' and 'data' in response:
            return response['data']
        else:
            logger.warning(f"Failed to get threat intel: {response.get('message', 'Unknown error')}")
            return {
                "event_types": {},
                "top_attacks": [],
                "geographic_data": {}
            }
    
    def get_all_analytics(self, hours=24, limit=100, force_refresh=False):
        """Get all analytics data in one call"""
        params = {
            'force_refresh': str(force_refresh).lower()
        }
        
        response = self._make_request('api/all-analytics', params=params, use_cache=not force_refresh)
        
        if response.get('status') == 'success' and 'data' in response:
            return response['data']
        else:
            logger.warning(f"Failed to get all analytics: {response.get('message', 'Unknown error')}")
            
            # Generate fallback data
            return self._generate_fallback_analytics_data(hours, limit)
    
    def make_ml_prediction(self, model_type, data):
        """Make a prediction using one of the ML models"""
        request_data = {
            'model_type': model_type,
            'data': data
        }
        
        response = self._make_request('api/ml-prediction', method='POST', data=request_data, use_cache=False)
        
        if response.get('status') == 'success':
            if 'predictions' in response:
                return response['predictions']
            elif 'prediction' in response:
                return response['prediction']
            else:
                return {}
        else:
            logger.warning(f"Failed to get ML prediction: {response.get('message', 'Unknown error')}")
            return {}
    
    def _generate_fallback_analytics_data(self, hours=24, limit=100):
        """Generate fallback data when API is unavailable"""
        logger.info("Generating fallback analytics data")
        
        # Generate synthetic events
        events = []
        now = datetime.now()
        
        for i in range(min(limit, 50)):
            event_time = now - pd.Timedelta(minutes=i*30)
            
            # Randomly select event type
            event_type = np.random.choice(["intrusion", "authentication", "data_access", "malware"])
            
            # Base event data
            event = {
                "id": i + 1,
                "event_type": event_type,
                "is_malicious": bool(np.random.choice([0, 1], p=[0.7, 0.3])),
                "severity": np.random.choice(["critical", "high", "medium", "low", "info"]),
                "timestamp": event_time.isoformat(),
                "source_ip": f"192.168.{np.random.randint(1, 255)}.{np.random.randint(1, 255)}",
                "country": np.random.choice(["US", "UK", "DE", "FR", "CN", "RU", "BR", "IN"]),
                "description": f"Fallback {event_type} event {i+1}"
            }
            
            # Add event-specific fields
            if event_type == "intrusion":
                event.update({
                    "protocol": np.random.choice(["TCP", "UDP", "HTTP", "HTTPS"]),
                    "port": np.random.randint(1, 65535),
                    "attack_type": np.random.choice(["port_scan", "dos", "sql_injection", "xss"])
                })
            elif event_type == "authentication":
                event.update({
                    "username": f"user{np.random.randint(1, 100)}",
                    "success": bool(np.random.choice([0, 1], p=[0.2, 0.8])),
                    "browser": np.random.choice(["Chrome", "Firefox", "Safari", "Edge"])
                })
            
            events.append(event)
        
        # Create a simple analysis summary
        malicious_count = sum(1 for e in events if e.get("is_malicious"))
        severity_counts = {"critical": 0, "high": 0, "medium": 0, "low": 0, "info": 0}
        event_types = {"intrusion": 0, "authentication": 0, "data_access": 0, "malware": 0}
        
        for event in events:
            severity = event.get("severity", "").lower()
            if severity in severity_counts:
                severity_counts[severity] += 1
            
            event_type = event.get("event_type", "")
            if event_type in event_types:
                event_types[event_type] += 1
        
        analysis = {
            "timestamp": now.isoformat(),
            "summary": {
                "total_events_analyzed": len(events),
                "total_threats_detected": malicious_count,
                "threat_types": event_types,
                "severity_distribution": severity_counts
            }
        }
        
        # Create time series data
        time_groups = {}
        for event in events:
            timestamp = datetime.fromisoformat(event["timestamp"].replace('Z', '+00:00'))
            hour_key = timestamp.replace(minute=0, second=0, microsecond=0).isoformat()
            
            if hour_key not in time_groups:
                time_groups[hour_key] = {"count": 0, "malicious_count": 0}
            
            time_groups[hour_key]["count"] += 1
            if event.get("is_malicious"):
                time_groups[hour_key]["malicious_count"] += 1
        
        time_series = []
        for time_group, counts in time_groups.items():
            time_series.append({
                "time_group": time_group,
                "count": counts["count"],
                "malicious_count": counts["malicious_count"]
            })
        
        # Create geographic data
        geo_data = {}
        for event in events:
            country = event.get("country", "Unknown")
            if country not in geo_data:
                geo_data[country] = 0
            geo_data[country] += 1
        
        # Create summary
        summary = {
            "total_events": len(events),
            "event_types": event_types,
            "severity_counts": severity_counts,
            "malicious_count": malicious_count
        }
        
        # Create threat intel
        threat_intel = {
            "event_types": {et: {"total": count} for et, count in event_types.items()},
            "top_attacks": [
                {"attack_type": "port_scan", "count": 5, "severity": "medium"},
                {"attack_type": "sql_injection", "count": 3, "severity": "high"},
                {"attack_type": "dos", "count": 2, "severity": "critical"}
            ],
            "geographic_data": geo_data
        }
        
        return {
            "events": events,
            "analysis": analysis,
            "summary": summary,
            "time_series": time_series,
            "threat_intel": threat_intel,
            "timestamp": now.isoformat(),
            "is_fallback_data": True
        }

# Create a singleton instance
api_client = APIClient()

# Convenience functions for direct imports replacement
def get_synthetic_security_data(hours=24, limit=100, use_fake=True, force_refresh=False):
    """Get synthetic security data"""
    return api_client.get_security_data(hours, limit, use_fake, force_refresh)

def get_latest_threat_analysis():
    """Get latest threat analysis"""
    return api_client.get_threat_analysis()

def get_event_summary(df=None):
    """Get event summary"""
    if df is not None and not df.empty:
        # If a DataFrame is provided, use the API with appropriate parameters
        return api_client.get_event_summary(limit=len(df))
    return api_client.get_event_summary()

def get_events_by_time(df=None, interval='hour'):
    """Get events by time"""
    if df is not None and not df.empty:
        # If a DataFrame is provided, use the API with appropriate parameters
        return api_client.get_events_by_time(limit=len(df), interval=interval)
    return api_client.get_events_by_time(interval=interval)

def get_threat_intel_summary(df=None):
    """Get threat intelligence summary"""
    if df is not None and not df.empty:
        # If a DataFrame is provided, use the API with appropriate parameters
        return api_client.get_threat_intel(limit=len(df))
    return api_client.get_threat_intel()

def get_ai_analytics_data():
    """Get all AI analytics data"""
    return api_client.get_all_analytics()

def predict_intrusion(data):
    """Make a prediction using the intrusion detection model"""
    return api_client.make_ml_prediction('intrusion', data)

def detect_rba_anomalies(data):
    """Make a prediction using the RBA anomaly detection model"""
    return api_client.make_ml_prediction('rba', data)

def analyze_text_threat(text):
    """Analyze text for threats using the text threat detection model"""
    return api_client.make_ml_prediction('text_threat', {'text': text})

if __name__ == "__main__":
    # Test the API client
    print("Testing API Client...")
    client = APIClient()
    
    health = client.check_health()
    print(f"API Health: {health}")
    
    if health.get('status') == 'healthy':
        # Test security data
        df = client.get_security_data(limit=5)
        print(f"Security Data: {len(df)} records")
        
        # Test threat analysis
        analysis = client.get_threat_analysis()
        print(f"Threat Analysis: {json.dumps(analysis.get('summary', {}), indent=2)}")
        
        # Test all analytics
        all_data = client.get_all_analytics(force_refresh=True)
        print(f"All Analytics: {len(all_data.get('events', []))} events")
    else:
        print("API is not healthy, generating fallback data")
        fallback_data = client._generate_fallback_analytics_data()
        print(f"Fallback Data: {len(fallback_data.get('events', []))} events")