#!/usr/bin/env python3

"""
AI Analytics Service Runner

This script starts/stops/manages the AI Analytics API service.
The service provides data and analysis for the security dashboard.

Usage:
    python run_analytics_service.py start    # Start the service
    python run_analytics_service.py stop     # Stop the service
    python run_analytics_service.py status   # Check service status
    python run_analytics_service.py restart  # Restart the service
"""

import os
import sys
import subprocess
import time
import requests
import argparse
import signal
import json
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='ai_analytics.log',
    filemode='a'
)
logger = logging.getLogger("AI_Analytics_Service")

# Service configuration
API_PORT = 5000
API_HOST = "0.0.0.0"
API_SCRIPT = "simple_api.py"
PID_FILE = "analytics_service.pid"

# Status directory
STATUS_DIR = Path('service_status')
STATUS_DIR.mkdir(exist_ok=True)
STATUS_FILE = STATUS_DIR / "service_status.json"

def update_status_file(status="running"):
    """Update status file with current service information"""
    try:
        # Basic status info
        status_data = {
            "status": status,
            "last_update": time.strftime("%Y-%m-%d %H:%M:%S"),
            "data_stats": {
                "total_events": 0,
                "latest_event_time": None
            },
            "analysis_stats": {
                "threats_detected": 0,
                "last_analysis": None
            }
        }
        
        # Try to get more detailed stats from the API
        try:
            response = requests.get(f"http://localhost:{API_PORT}/api/all-analytics", timeout=2)
            if response.status_code == 200:
                data = response.json()
                if 'data' in data:
                    analytics_data = data['data']
                    
                    # Update stats
                    if 'events' in analytics_data:
                        status_data["data_stats"]["total_events"] = len(analytics_data['events'])
                        if analytics_data['events'] and 'timestamp' in analytics_data['events'][0]:
                            status_data["data_stats"]["latest_event_time"] = analytics_data['events'][0]['timestamp']
                    
                    if 'analysis' in analytics_data and 'summary' in analytics_data['analysis']:
                        summary = analytics_data['analysis']['summary']
                        status_data["analysis_stats"]["threats_detected"] = summary.get('total_threats_detected', 0)
                        status_data["analysis_stats"]["last_analysis"] = analytics_data.get('timestamp')
        except Exception as e:
            logger.warning(f"Error getting detailed stats: {e}")
        
        # Write status to file
        with open(STATUS_FILE, 'w') as f:
            json.dump(status_data, f, indent=2)
            
        logger.info(f"Updated status file: {status}")
    except Exception as e:
        logger.error(f"Error updating status file: {e}")

def is_service_running():
    """Check if the service is already running."""
    # Check if PID file exists
    if not os.path.exists(PID_FILE):
        return False, None
    
    # Read PID from file
    try:
        with open(PID_FILE, 'r') as f:
            pid = int(f.read().strip())
    except (IOError, ValueError):
        return False, None
    
    # Check if process is running
    try:
        os.kill(pid, 0)  # Signal 0 checks if process exists
        return True, pid
    except OSError:
        # Process not running
        return False, None

def is_api_responding():
    """Check if the API is responding."""
    try:
        response = requests.get(f"http://localhost:{API_PORT}/health", timeout=2)
        return response.status_code == 200, response.json() if response.status_code == 200 else None
    except:
        return False, None

def start_service():
    """Start the AI Analytics API service"""
    logger.info("Starting AI Analytics API service...")
    print("Starting AI Analytics API service...")
    
    # Check if already running
    running, pid = is_service_running()
    if running:
        logger.info(f"Service already running with PID {pid}")
        print(f"Service already running with PID {pid}")
        
        # Check if the API is responding
        api_running, _ = is_api_responding()
        if api_running:
            print(f"API is responding at http://localhost:{API_PORT}/")
            update_status_file()
            return True
        else:
            print("Warning: Process is running but API is not responding")
            print("Stopping old process and starting a new one...")
            stop_service()
    
    # Clear any stale PID file
    if os.path.exists(PID_FILE):
        os.remove(PID_FILE)
    
    # Start the API as a background process
    try:
        # Make sure we're using the correct path to the API script
        api_script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), API_SCRIPT)
        
        # Start the process
        with open('ai_analytics.log', 'a') as log_file:
            log_file.write(f"\n\n=== Starting AI Analytics service at {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
            process = subprocess.Popen(
                [
                    sys.executable,
                    api_script_path,
                ],
                stdout=log_file,
                stderr=log_file,
                start_new_session=True  # Detach from parent process
            )
        
        # Save PID to file
        with open(PID_FILE, 'w') as f:
            f.write(str(process.pid))
        
        logger.info(f"Service started with PID {process.pid}")
        print(f"Service started with PID {process.pid}")
        
        # Wait for API to become available
        print("Waiting for API to start...")
        for _ in range(10):  # Wait up to 10 seconds
            time.sleep(1)
            api_running, api_status = is_api_responding()
            if api_running:
                logger.info(f"API is responding: {api_status}")
                print(f"API is responding at http://localhost:{API_PORT}/")
                update_status_file()
                return True
        
        # API didn't start in time
        logger.warning("API didn't start responding in the expected time")
        print("Warning: API didn't start responding in the expected time.")
        print("The service may still be starting. Check status in a moment.")
        return True
        
    except Exception as e:
        logger.error(f"Error starting service: {e}")
        print(f"Error starting service: {e}")
        return False

def stop_service():
    """Stop the AI Analytics API service"""
    logger.info("Stopping AI Analytics API service...")
    print("Stopping AI Analytics API service...")
    
    # Check if service is running
    running, pid = is_service_running()
    if not running:
        logger.info("Service is not running")
        print("Service is not running")
        
        # Remove stale PID file if it exists
        if os.path.exists(PID_FILE):
            os.remove(PID_FILE)
            
        # Update status file
        update_status_file(status="stopped")
        return True
    
    # Try to terminate gracefully
    try:
        logger.info(f"Sending SIGTERM to process {pid}")
        os.kill(pid, signal.SIGTERM)
        
        # Wait for process to terminate
        for _ in range(5):  # Wait up to 5 seconds
            time.sleep(1)
            try:
                os.kill(pid, 0)  # Check if process still exists
            except OSError:
                # Process has terminated
                break
        else:
            # Process didn't terminate, send SIGKILL
            logger.warning(f"Process {pid} didn't terminate gracefully, sending SIGKILL")
            print(f"Process {pid} didn't terminate gracefully, forcing kill...")
            os.kill(pid, signal.SIGKILL)
    except OSError as e:
        logger.error(f"Error stopping service: {e}")
        print(f"Error stopping service: {e}")
    
    # Remove PID file
    if os.path.exists(PID_FILE):
        os.remove(PID_FILE)
    
    logger.info("Service stopped")
    print("Service stopped")
    
    # Update status file
    update_status_file(status="stopped")
    return True

def check_status():
    """Check the status of the AI Analytics service"""
    print("Checking AI Analytics service status...")
    
    # Check if service is running
    running, pid = is_service_running()
    if running:
        print(f"✅ Process is running with PID {pid}")
    else:
        print("❌ Process is not running")
        
        # If PID file exists but process is not running, it's stale
        if os.path.exists(PID_FILE):
            print("Warning: Stale PID file exists")
    
    # Check if API is responding
    api_running, api_status = is_api_responding()
    if api_running:
        print(f"✅ API is responding at http://localhost:{API_PORT}/")
        print(f"   Status: {api_status}")
    else:
        print("❌ API is not responding")
    
    # Check status file
    if os.path.exists(STATUS_FILE):
        try:
            with open(STATUS_FILE, 'r') as f:
                status_data = json.load(f)
                print(f"\nService status: {status_data.get('status', 'unknown')}")
                print(f"Last update: {status_data.get('last_update', 'unknown')}")
                
                # Print data stats if available
                if 'data_stats' in status_data:
                    data_stats = status_data['data_stats']
                    print(f"\nData Statistics:")
                    print(f"  Total events: {data_stats.get('total_events', 'N/A')}")
                    print(f"  Latest event: {data_stats.get('latest_event_time', 'N/A')}")
                
                # Print analysis stats if available
                if 'analysis_stats' in status_data:
                    analysis_stats = status_data['analysis_stats']
                    print(f"\nAnalysis Statistics:")
                    print(f"  Threats detected: {analysis_stats.get('threats_detected', 'N/A')}")
                    print(f"  Last analysis: {analysis_stats.get('last_analysis', 'N/A')}")
        except Exception as e:
            print(f"Error reading status file: {e}")
    else:
        print("\nNo status file found")
    
    # Check log file
    log_file = 'ai_analytics.log'
    if os.path.exists(log_file):
        log_size = os.path.getsize(log_file) / 1024
        print(f"\nLog file size: {log_size:.1f} KB")
        
        # Check if log file contains recent errors
        try:
            with open(log_file, 'r') as f:
                # Get last 10 lines
                lines = f.readlines()[-10:]
                if any("ERROR" in line for line in lines):
                    print("⚠️  Log contains recent errors. Check ai_analytics.log for details.")
        except Exception:
            pass
    else:
        print("\nNo log file found")

def main():
    """Main function to parse arguments and run commands"""
    parser = argparse.ArgumentParser(description="AI Analytics Service Manager")
    parser.add_argument('action', choices=['start', 'stop', 'status', 'restart'], 
                       help='Action to perform')
    
    args = parser.parse_args()
    
    if args.action == 'start':
        start_service()
    elif args.action == 'stop':
        stop_service()
    elif args.action == 'restart':
        stop_service()
        time.sleep(2)
        start_service()
    elif args.action == 'status':
        check_status()

if __name__ == "__main__":
    main()