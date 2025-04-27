#!/usr/bin/env python3

"""
Dashboard Runner Script

This script helps run the dashboard together with the AI Analytics service.
It can start both services, stop them, or check their status.

Usage:
    python run_dashboard.py start   # Start both services
    python run_dashboard.py stop    # Stop both services
    python run_dashboard.py status  # Check services status
"""

import os
import sys
import subprocess
import time
import argparse
import signal
import requests
from pathlib import Path
import json

# Configuration
DASHBOARD_SCRIPT = "data-pipelines/dash_app_main.py"
DASHBOARD_PORT = 8050
DASHBOARD_HOST = "0.0.0.0"

# Path to AI analytics directory and service script
AI_ANALYTICS_DIR = os.path.expanduser("~/EigenvectorsAndChill-cybersecurity-platform")
AI_ANALYTICS_SERVICE = "run_analytics_service.py"
AI_ANALYTICS_API_PORT = 5000

# PID files
DASHBOARD_PID_FILE = "dashboard.pid"
AI_ANALYTICS_PID_FILE = os.path.join(AI_ANALYTICS_DIR, "analytics_service.pid")

# Status file
STATUS_DIR = os.path.join(AI_ANALYTICS_DIR, "service_status")
STATUS_FILE = os.path.join(STATUS_DIR, "service_status.json")

def get_script_dir():
    """Get the directory where this script is located"""
    return os.path.dirname(os.path.abspath(__file__))

def is_dashboard_running():
    """Check if the dashboard is already running"""
    pid_file = os.path.join(get_script_dir(), DASHBOARD_PID_FILE)
    
    if not os.path.exists(pid_file):
        return False, None
    
    # Read PID from file
    with open(pid_file, 'r') as f:
        pid = f.read().strip()
    
    if not pid:
        return False, None
    
    # Check if process is running
    try:
        pid = int(pid)
        os.kill(pid, 0)  # Send signal 0 to check if process exists
        return True, pid
    except (OSError, ValueError):
        # Process not running or PID file contains invalid data
        return False, None

def is_api_running():
    """Check if the AI Analytics API is running"""
    try:
        response = requests.get(f"http://localhost:{AI_ANALYTICS_API_PORT}/health", timeout=2)
        if response.status_code == 200:
            return True, response.json()
        return False, {"status": "API returned non-200 status code"}
    except Exception as e:
        return False, {"status": f"API not responding: {str(e)}"}

def start_ai_analytics():
    """Start the AI Analytics service"""
    print("Starting AI Analytics service...")
    
    # Simple check if API is already running
    is_running, status = is_api_running()
    if is_running:
        print("AI Analytics API is already running")
        return True
    
    # Check if we have a PID file and if that process is running
    if os.path.exists(AI_ANALYTICS_PID_FILE):
        try:
            with open(AI_ANALYTICS_PID_FILE, 'r') as f:
                pid = int(f.read().strip())
            # Try to check if process exists
            os.kill(pid, 0)
            print(f"AI Analytics service is already running with PID {pid}")
            return True
        except (OSError, ValueError):
            # Process not running or invalid PID, continue to start
            print("Found stale PID file, will start a new instance")
    
    # Start the service with a simple command
    try:
        os.chdir(AI_ANALYTICS_DIR)
        result = subprocess.run([sys.executable, AI_ANALYTICS_SERVICE, "start"], 
                              check=False, capture_output=True, text=True)
        print(result.stdout)
        
        # Return to original directory
        os.chdir(get_script_dir())
        
        # Wait for service to start and verify it's running
        for _ in range(10):  # Try for 10 seconds
            time.sleep(1)
            is_running, _ = is_api_running()
            if is_running:
                print("AI Analytics service started successfully!")
                return True
        
        print("AI Analytics service started but API is not responding")
        return False
    except Exception as e:
        print(f"Error starting AI Analytics service: {e}")
        # Return to original directory
        os.chdir(get_script_dir())
        return False

def stop_ai_analytics():
    """Stop the AI Analytics service"""
    print("Stopping AI Analytics service...")
    
    # Change directory to AI Analytics and run the service
    current_dir = os.getcwd()
    os.chdir(AI_ANALYTICS_DIR)
    
    # Stop the service
    try:
        result = subprocess.run(
            [sys.executable, AI_ANALYTICS_SERVICE, "stop"],
            check=True,
            capture_output=True,
            text=True
        )
        print(result.stdout)
        success = "stopped" in result.stdout.lower()
    except subprocess.CalledProcessError as e:
        print(f"Error stopping AI Analytics service: {e}")
        print(e.stdout)
        print(e.stderr)
        success = False
    finally:
        # Change back to original directory
        os.chdir(current_dir)
    
    # Verify the service has stopped
    is_running, _ = is_api_running()
    if is_running:
        print("Warning: AI Analytics API is still responding after stop command")
        success = False
    else:
        print("AI Analytics service stopped successfully")
    
    return success

def start_dashboard():
    """Start the dashboard application"""
    # Check if dashboard is already running
    running, pid = is_dashboard_running()
    if running:
        print(f"Dashboard is already running (PID: {pid})")
        return True
    
    # Set environment variable to point to AI Analytics API
    os.environ["AI_ANALYTICS_API_URL"] = f"http://localhost:{AI_ANALYTICS_API_PORT}"
    
    # Dashboard script path
    dashboard_script = os.path.join(get_script_dir(), DASHBOARD_SCRIPT)
    if not os.path.exists(dashboard_script):
        print(f"Error: Dashboard script not found at {dashboard_script}")
        return False
    
    print(f"Starting dashboard on http://{DASHBOARD_HOST}:{DASHBOARD_PORT}/...")
    
    # Start the dashboard as a background process
    with open("dashboard.log", 'a') as log_file:
        log_file.write(f"\n\n=== Starting dashboard at {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
        process = subprocess.Popen(
            [
                sys.executable, 
                dashboard_script, 
                "--host", DASHBOARD_HOST, 
                "--port", str(DASHBOARD_PORT)
            ],
            stdout=log_file,
            stderr=log_file,
            env=os.environ,
            start_new_session=True  # Detach process from parent
        )
    
    # Save PID to file
    with open(DASHBOARD_PID_FILE, 'w') as f:
        f.write(str(process.pid))
    
    print(f"Dashboard started with PID: {process.pid}")
    print(f"Dashboard URL: http://{DASHBOARD_HOST}:{DASHBOARD_PORT}/")
    print(f"Logs are being written to: dashboard.log")
    
    # Wait a moment and verify dashboard is running by checking the log for errors
    time.sleep(3)
    try:
        with open("dashboard.log", 'r') as log_file:
            log_content = log_file.read()
            if "Error" in log_content or "Traceback" in log_content:
                print("Warning: Dashboard may have encountered errors. Check dashboard.log for details.")
            else:
                print("Dashboard appears to be running without errors.")
    except Exception as e:
        print(f"Warning: Could not check dashboard log: {e}")
    
    return True

def stop_dashboard():
    """Stop the dashboard application"""
    running, pid = is_dashboard_running()
    
    if not running:
        print("Dashboard is not running")
        # Clean up PID file if it exists
        if os.path.exists(DASHBOARD_PID_FILE):
            os.remove(DASHBOARD_PID_FILE)
        return True
    
    print(f"Stopping dashboard (PID: {pid})...")
    
    try:
        # Try to terminate gracefully first
        os.kill(pid, signal.SIGTERM)
        
        # Give the process some time to terminate
        for _ in range(5):
            time.sleep(1)
            try:
                # Check if process still exists
                os.kill(pid, 0)
            except OSError:
                # Process has terminated
                break
        else:
            # Process didn't terminate, force kill
            print("Process didn't terminate gracefully, forcing kill...")
            os.kill(pid, signal.SIGKILL)
    
    except OSError as e:
        print(f"Error stopping dashboard: {e}")
    
    # Remove PID file
    if os.path.exists(DASHBOARD_PID_FILE):
        os.remove(DASHBOARD_PID_FILE)
    
    print("Dashboard stopped")
    return True

def get_analytics_service_info():
    """Get detailed information about the analytics service"""
    info = {
        "is_running": False,
        "pid": None,
        "status": "unknown",
        "data_stats": {},
        "analysis_stats": {}
    }
    
    # Check if we have a PID file
    if os.path.exists(AI_ANALYTICS_PID_FILE):
        try:
            with open(AI_ANALYTICS_PID_FILE, 'r') as f:
                pid = int(f.read().strip())
            # Try to check if process exists
            os.kill(pid, 0)
            info["is_running"] = True
            info["pid"] = pid
        except (OSError, ValueError):
            # Process not running or invalid PID
            info["is_running"] = False
    
    # Try to get API status
    is_api_running_flag, api_status = is_api_running()
    info["api_responding"] = is_api_running_flag
    
    if is_api_running_flag:
        info["status"] = api_status.get("status", "unknown")
    
    # Try to read status file for more detailed info
    if os.path.exists(STATUS_FILE):
        try:
            with open(STATUS_FILE, 'r') as f:
                status_data = json.load(f)
                if "data_stats" in status_data:
                    info["data_stats"] = status_data["data_stats"]
                if "analysis_stats" in status_data:
                    info["analysis_stats"] = status_data["analysis_stats"]
                if "last_update" in status_data:
                    info["last_update"] = status_data["last_update"]
        except Exception as e:
            print(f"Error reading service status file: {e}")
    
    return info

def check_status():
    """Check the status of both services"""
    print("Checking service status...\n")
    
    # Check AI Analytics API
    api_running, api_status = is_api_running()
    
    # Get more detailed info
    analytics_info = get_analytics_service_info()
    
    if api_running:
        print("✅ AI Analytics API is running")
        print(f"   Status: {api_status}")
        if "last_update" in analytics_info:
            print(f"   Last updated: {analytics_info['last_update']}")
        if "data_stats" in analytics_info and analytics_info["data_stats"]:
            data_stats = analytics_info["data_stats"]
            print(f"   Total events: {data_stats.get('total_events', 'N/A')}")
            print(f"   Latest event: {data_stats.get('latest_event_time', 'N/A')}")
        if "analysis_stats" in analytics_info and analytics_info["analysis_stats"]:
            analysis_stats = analytics_info["analysis_stats"]
            print(f"   Threats detected: {analysis_stats.get('threats_detected', 'N/A')}")
    else:
        print("❌ AI Analytics API is not running")
        if analytics_info["is_running"]:
            print(f"   Process is running (PID: {analytics_info['pid']}) but API is not responding")
    
    # Check Dashboard
    dashboard_running, dashboard_pid = is_dashboard_running()
    if dashboard_running:
        print(f"✅ Dashboard is running (PID: {dashboard_pid})")
        print(f"   URL: http://{DASHBOARD_HOST}:{DASHBOARD_PORT}/")
    else:
        print("❌ Dashboard is not running")
    
    # Check log files
    dashboard_log = os.path.join(get_script_dir(), "dashboard.log")
    if os.path.exists(dashboard_log):
        log_size = os.path.getsize(dashboard_log) / 1024
        print(f"\nDashboard log file: {log_size:.1f} KB")
        
        # Check last few lines for errors
        try:
            with open(dashboard_log, 'r') as f:
                # Read last 20 lines
                lines = f.readlines()[-20:]
                if any("Error" in line or "Traceback" in line for line in lines):
                    print("   ⚠️ Dashboard log contains recent errors. Check dashboard.log for details.")
        except Exception:
            pass
    
    ai_analytics_log = os.path.join(AI_ANALYTICS_DIR, "ai_analytics.log")
    if os.path.exists(ai_analytics_log):
        log_size = os.path.getsize(ai_analytics_log) / 1024
        print(f"AI Analytics log file: {log_size:.1f} KB")
        
        # Check last few lines for errors
        try:
            with open(ai_analytics_log, 'r') as f:
                # Read last 20 lines
                lines = f.readlines()[-20:]
                if any("Error" in line or "Traceback" in line for line in lines):
                    print("   ⚠️ AI Analytics log contains recent errors. Check ai_analytics.log for details.")
        except Exception:
            pass
    
    print("\nDone!")

def main():
    """Main function to parse arguments and run commands"""
    parser = argparse.ArgumentParser(description="Dashboard and AI Analytics Service Runner")
    parser.add_argument('action', choices=['start', 'stop', 'status', 'restart'],
                      help='Action to perform on the services')
    
    args = parser.parse_args()
    
    if args.action == 'start':
        # Start both services
        print("Starting services...\n")
        ai_success = start_ai_analytics()
        if ai_success:
            dashboard_success = start_dashboard()
            if dashboard_success:
                print("\nAll services started successfully!")
            else:
                print("\nWarning: Dashboard failed to start, but AI Analytics is running.")
        else:
            print("\nFailed to start AI Analytics. Not starting dashboard.")
    
    elif args.action == 'stop':
        # Stop both services
        print("Stopping services...\n")
        stop_dashboard()
        stop_ai_analytics()
        print("\nAll services stopped.")
    
    elif args.action == 'restart':
        # Restart both services
        print("Restarting services...\n")
        stop_dashboard()
        stop_ai_analytics()
        time.sleep(2)
        ai_success = start_ai_analytics()
        if ai_success:
            dashboard_success = start_dashboard()
            if dashboard_success:
                print("\nAll services restarted successfully!")
            else:
                print("\nWarning: Dashboard failed to restart, but AI Analytics is running.")
        else:
            print("\nFailed to restart AI Analytics. Not starting dashboard.")
    
    elif args.action == 'status':
        # Check status
        check_status()

if __name__ == "__main__":
    main()