#!/usr/bin/env python3

"""
Remote Dataset Access Test Script

This script tests if you can access the datasets directly from the remote server.
It uses SSH to connect to the server and perform basic EDA on each dataset.

Requirements:
- pip install paramiko pandas python-dotenv
- SSH key for accessing the server
- .env file with connection details
"""

import os
import io
import sys
import paramiko
import pandas as pd
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Server connection details from environment variables
SSH_HOST = os.getenv('SSH_HOST')
SSH_USER = os.getenv('SSH_USER')
SSH_KEY_PATH = os.path.expanduser(os.getenv('SSH_KEY_PATH'))

# Dataset paths on remote server
DATASET_PATHS = {
    'intrusion_detection': os.getenv('INTRUSION_DETECTION_DATASET'),
    'ai_enhanced_events': os.getenv('AI_ENHANCED_DATASET'),
    'text_based_detection': os.getenv('TEXT_BASED_DATASET'),
    'rba_dataset': os.getenv('RBA_DATASET')
}

def connect_to_server():
    """Connect to the remote server via SSH."""
    try:
        print(f"Connecting to {SSH_USER}@{SSH_HOST} using key {SSH_KEY_PATH}...")
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(
            hostname=SSH_HOST,
            username=SSH_USER,
            key_filename=SSH_KEY_PATH
        )
        print("✅ Connected to remote server")
        return client
    except Exception as e:
        print(f"❌ Error connecting to server: {e}")
        sys.exit(1)

def execute_command(client, command):
    """Execute a command on the remote server."""
    print(f"Executing: {command}")
    stdin, stdout, stderr = client.exec_command(command)
    stdout_str = stdout.read().decode('utf-8')
    stderr_str = stderr.read().decode('utf-8')
    
    if stderr_str:
        print(f"WARNING/ERROR: {stderr_str}")
    
    return stdout_str

def check_file_exists(client, path):
    """Check if a file exists on the remote server."""
    expanded_path = execute_command(client, f"echo {path}").strip()
    result = execute_command(client, f"test -f {expanded_path} && echo 'Exists' || echo 'Not found'")
    return 'Exists' in result

def get_file_size(client, path):
    """Get the size of a file on the remote server."""
    expanded_path = execute_command(client, f"echo {path}").strip()
    result = execute_command(client, f"du -h {expanded_path} | cut -f1")
    return result.strip()

def analyze_dataset(client, dataset_name, path, sample_size=1000):
    """Analyze a dataset on the remote server."""
    print("\n" + "="*50)
    print(f"ANALYZING DATASET: {dataset_name}")
    print("="*50)
    
    expanded_path = execute_command(client, f"echo {path}").strip()
    
    # Check if file exists
    if not check_file_exists(client, expanded_path):
        print(f"❌ Dataset not found at path: {expanded_path}")
        return
    
    # Get file size
    file_size = get_file_size(client, expanded_path)
    print(f"File size: {file_size}")
    
    # Count total lines
    line_count = execute_command(client, f"wc -l {expanded_path} | cut -d' ' -f1")
    print(f"Total lines: {line_count.strip()}")
    
    # Get column names
    headers = execute_command(client, f"head -1 {expanded_path}")
    print(f"Headers: {headers.strip()}")
    
    # Get a sample of data (first 5 lines)
    sample_data = execute_command(client, f"head -5 {expanded_path}")
    print(f"\nSample data (first 5 lines):\n{sample_data}")
    
    # If it's a very large file, analyze a limited sample using Python
    if dataset_name == "rba_dataset":
        print(f"\nThis is a large dataset. Analyzing only first {sample_size} rows...")
        
        # Stream the first N lines for analysis
        command = f"head -n {sample_size+1} {expanded_path}"  # +1 for header
        stdin, stdout, stderr = client.exec_command(command)
        
        # Read data into pandas
        df = pd.read_csv(io.StringIO(stdout.read().decode('utf-8')))
        
        # Basic stats
        print(f"\nDataFrame shape: {df.shape}")
        print(f"\nColumn data types:\n{df.dtypes}")
        
        # Sample data description
        print(f"\nNumeric columns summary:\n{df.describe().to_string()}")
        
        # Check for missing values
        missing = df.isnull().sum()
        if missing.sum() > 0:
            print(f"\nMissing values:\n{missing[missing > 0]}")
        else:
            print("\nNo missing values found in sample")
    
    print(f"\n✅ Successfully analyzed {dataset_name} dataset")

def main():
    start_time = time.time()
    
    # Connect to server
    client = connect_to_server()
    
    try:
        # Check all datasets
        for dataset_name, path in DATASET_PATHS.items():
            analyze_dataset(client, dataset_name, path)
    
    finally:
        # Close connection
        client.close()
        print("\nSSH connection closed")
    
    end_time = time.time()
    print(f"\nTotal execution time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()