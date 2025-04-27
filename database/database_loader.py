#!/usr/bin/env python3

"""
Cybersecurity Datasets Database Loader

This script loads datasets directly from the remote server into databases:
- Intrusion Detection Dataset -> MySQL
- AI-Enhanced Events Dataset -> PostgreSQL
- Text-based Threat Detection -> MongoDB
- RBA (Risk-Based Authentication) -> MySQL
"""

import os
import io
import sys
import pandas as pd
import numpy as np
import json
import pymysql
import psycopg2
from pymongo import MongoClient
import paramiko
from pathlib import Path
import warnings
import time
from tqdm import tqdm
from dotenv import load_dotenv
from urllib.parse import quote_plus

warnings.filterwarnings('ignore')

# Load environment variables from .env file
load_dotenv()

# SSH connection settings
SSH_HOST = os.getenv('SSH_HOST')
SSH_USER = os.getenv('SSH_USER')
SSH_KEY_PATH = os.path.expanduser(os.getenv('SSH_KEY_PATH'))

# Database configuration from environment variables
# Using DEFAULT_* variables that can be replaced by team members
MYSQL_CONFIG = {
    'host': os.getenv('MYSQL_HOST'),
    'user': os.getenv('DEFAULT_MYSQL_USER'),
    'password': os.getenv('DEFAULT_MYSQL_PASSWORD'),
    'database': os.getenv('DEFAULT_MYSQL_DB')
}

# PostgreSQL Config
POSTGRES_CONFIG = {
    'host': os.getenv('POSTGRES_HOST'),
    'user': os.getenv('DEFAULT_POSTGRES_USER'),
    'password': os.getenv('DEFAULT_POSTGRES_PASSWORD'),
    'database': os.getenv('DEFAULT_POSTGRES_DB'), 
    'connect_timeout': 10
}

# MongoDB Config with correct authSource parameter
MONGODB_CONFIG = {
    'uri': f"mongodb://{quote_plus(os.getenv('DEFAULT_MONGODB_USER'))}:{quote_plus(os.getenv('DEFAULT_MONGODB_PASSWORD'))}@{os.getenv('MONGODB_HOST')}:27017/{os.getenv('DEFAULT_MONGODB_DB')}?authSource={os.getenv('DEFAULT_MONGODB_DB')}",
    'database': os.getenv('DEFAULT_MONGODB_DB')
}

# Dataset paths on remote server
DATASET_PATHS = {
    'intrusion_detection': os.getenv('INTRUSION_DETECTION_DATASET'),
    'ai_enhanced_events': os.getenv('AI_ENHANCED_DATASET'),
    'text_based_detection': os.getenv('TEXT_BASED_DATASET'),
    'rba_dataset': os.getenv('RBA_DATASET')
}

# Create output directory for logs
log_dir = Path('db_loading_logs')
log_dir.mkdir(exist_ok=True)

# Connect to the remote server via SSH
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
        return None

# Read a dataset from the remote server
def read_dataset_from_server(ssh_client, dataset_path, chunksize=None, nrows=None):
    """Read a dataset from the remote server as a pandas DataFrame."""
    try:
        # Get expanded path
        stdin, stdout, stderr = ssh_client.exec_command(f"echo {dataset_path}")
        expanded_path = stdout.read().decode('utf-8').strip()
        
        print(f"Reading dataset from {expanded_path}")
        
        if chunksize is not None:
            # For large datasets, yield chunks
            command = f"cat {expanded_path}"
            stdin, stdout, stderr = ssh_client.exec_command(command)
            chunks = pd.read_csv(stdout, chunksize=chunksize)
            return chunks
        elif nrows is not None:
            # For limited rows (sample)
            command = f"head -n {nrows+1} {expanded_path}"  # +1 for header
            stdin, stdout, stderr = ssh_client.exec_command(command)
            df = pd.read_csv(stdout)
            return df
        else:
            # For full dataset
            command = f"cat {expanded_path}"
            stdin, stdout, stderr = ssh_client.exec_command(command)
            df = pd.read_csv(stdout)
            return df
    except Exception as e:
        print(f"❌ Error reading dataset: {e}")
        return None

# Function to connect to MySQL
def connect_mysql(config):
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

# Function to connect to PostgreSQL
def connect_postgres(config):
    try:
        connection = psycopg2.connect(
            host=config['host'],
            user=config['user'],
            password=config['password'],
            database=config['database'],
            # Add connect_timeout parameter
            connect_timeout=10
        )
        print(f"✅ Connected to PostgreSQL: {config['user']}@{config['host']}/{config['database']}")
        return connection
    except Exception as e:
        print(f"❌ Error connecting to PostgreSQL: {e}")
        return None

# Function to connect to MongoDB
def connect_mongodb(config):
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
    
# Load intrusion detection dataset to MySQL
def load_intrusion_detection_to_mysql(ssh_client, mysql_conn):
    try:
        if ssh_client is None or mysql_conn is None:
            return False
        
        print(f"Loading Intrusion Detection dataset from remote server")
        
        # Read dataset from remote server
        df = read_dataset_from_server(ssh_client, DATASET_PATHS['intrusion_detection'])
        if df is None:
            return False
            
        print(f"Dataset shape: {df.shape}")
        
        # Create table
        with mysql_conn.cursor() as cursor:
            cursor.execute("""
            DROP TABLE IF EXISTS intrusion_detection;
            """)
            
            cursor.execute("""
            CREATE TABLE intrusion_detection (
                id INT AUTO_INCREMENT PRIMARY KEY,
                session_id VARCHAR(20),
                network_packet_size INT,
                protocol_type VARCHAR(10),
                login_attempts INT,
                session_duration FLOAT,
                encryption_used VARCHAR(10),
                ip_reputation_score FLOAT,
                failed_logins INT,
                browser_type VARCHAR(20),
                unusual_time_access INT,
                attack_detected INT
            );
            """)
            
            # Insert data in batches
            batch_size = 1000
            total_batches = (len(df) + batch_size - 1) // batch_size
            
            with tqdm(total=total_batches, desc="Loading to MySQL") as pbar:
                for i in range(0, len(df), batch_size):
                    batch = df.iloc[i:i+batch_size]
                    
                    values = []
                    for _, row in batch.iterrows():
                        value = (
                            str(row['session_id']),
                            int(row['network_packet_size']),
                            str(row['protocol_type']),
                            int(row['login_attempts']),
                            float(row['session_duration']),
                            str(row['encryption_used']),
                            float(row['ip_reputation_score']),
                            int(row['failed_logins']),
                            str(row['browser_type']),
                            int(row['unusual_time_access']),
                            int(row['attack_detected'])
                        )
                        values.append(value)
                    
                    placeholders = ", ".join(["%s"] * 11)
                    sql = f"""
                    INSERT INTO intrusion_detection 
                    (session_id, network_packet_size, protocol_type, login_attempts, 
                    session_duration, encryption_used, ip_reputation_score, failed_logins, 
                    browser_type, unusual_time_access, attack_detected)
                    VALUES ({placeholders})
                    """
                    
                    cursor.executemany(sql, values)
                    mysql_conn.commit()
                    pbar.update(1)
        
        # Verify the data was loaded
        with mysql_conn.cursor() as cursor:
            cursor.execute("SELECT COUNT(*) as count FROM intrusion_detection")
            result = cursor.fetchone()
            print(f"✅ Loaded {result['count']} rows into MySQL intrusion_detection table")
        
        return True
    except Exception as e:
        print(f"❌ Error loading intrusion detection data to MySQL: {e}")
        return False
'''
# Load AI-enhanced events dataset to PostgreSQL
def load_ai_enhanced_to_postgres(ssh_client, postgres_conn):
    try:
        if ssh_client is None or postgres_conn is None:
            return False
        
        print(f"Loading AI-Enhanced Events dataset from remote server")
        
        # Read dataset from remote server
        df = read_dataset_from_server(ssh_client, DATASET_PATHS['ai_enhanced_events'])
        if df is None:
            return False
            
        print(f"Dataset shape: {df.shape}")
        
        # Create table
        with postgres_conn.cursor() as cursor:
            cursor.execute("""
            DROP TABLE IF EXISTS ai_enhanced_events;
            """)
            
            # Create table with appropriate columns
            columns_query = "CREATE TABLE ai_enhanced_events (id SERIAL PRIMARY KEY"
            
            for col in df.columns:
                col_name = col.replace(' ', '_').lower()
                
                # Determine data type
                if df[col].dtype == 'int64':
                    col_type = "INTEGER"
                elif df[col].dtype == 'float64':
                    col_type = "FLOAT"
                else:
                    col_type = "TEXT"
                
                columns_query += f", {col_name} {col_type}"
            
            columns_query += ");"
            cursor.execute(columns_query)
            
            # Insert data in batches
            batch_size = 1000
            total_batches = (len(df) + batch_size - 1) // batch_size
            
            with tqdm(total=total_batches, desc="Loading to PostgreSQL") as pbar:
                for i in range(0, len(df), batch_size):
                    batch = df.iloc[i:i+batch_size]
                    
                    # Prepare columns for insertion
                    columns = [col.replace(' ', '_').lower() for col in df.columns]
                    placeholders = ", ".join([f"%s"] * len(columns))
                    columns_str = ", ".join(columns)
                    
                    # Prepare values for insertion
                    for _, row in batch.iterrows():
                        values = []
                        for col in df.columns:
                            value = row[col]
                            if pd.isna(value):
                                value = None
                            elif isinstance(value, str):
                                value = value.replace("'", "''")  # Escape single quotes
                            values.append(value)
                        
                        # Build and execute query
                        cursor.execute(
                            f"INSERT INTO ai_enhanced_events ({columns_str}) VALUES ({placeholders})",
                            values
                        )
                    
                    postgres_conn.commit()
                    pbar.update(1)
        
        # Verify the data was loaded
        with postgres_conn.cursor() as cursor:
            cursor.execute("SELECT COUNT(*) FROM ai_enhanced_events")
            result = cursor.fetchone()
            print(f"✅ Loaded {result[0]} rows into PostgreSQL ai_enhanced_events table")
        
        return True
    except Exception as e:
        print(f"❌ Error loading AI-enhanced events data to PostgreSQL: {e}")
        return False'''

# Load text-based detection dataset to MongoDB
def load_text_based_to_mongodb(ssh_client, mongodb_db):
    try:
        if ssh_client is None or mongodb_db is None:
            return False
        
        print(f"Loading Text-based Threat Detection dataset from remote server")
        
        # Read dataset from remote server
        df = read_dataset_from_server(ssh_client, DATASET_PATHS['text_based_detection'])
        if df is None:
            return False
            
        print(f"Dataset shape: {df.shape}")
        
        # Drop existing collection
        mongodb_db.text_based_detection.drop()
        
        # Convert DataFrame to list of dictionaries
        records = df.to_dict(orient='records')
        
        # Process complex fields (like JSON strings in entries, relations)
        for record in records:
            for key, value in record.items():
                if isinstance(value, str) and (value.startswith('[') or value.startswith('{')):
                    try:
                        record[key] = json.loads(value)
                    except:
                        pass  # Keep as string if not valid JSON
        
        # Insert data in batches
        batch_size = 1000
        total_batches = (len(records) + batch_size - 1) // batch_size
        
        with tqdm(total=total_batches, desc="Loading to MongoDB") as pbar:
            for i in range(0, len(records), batch_size):
                batch = records[i:i+batch_size]
                mongodb_db.text_based_detection.insert_many(batch)
                pbar.update(1)
        
        # Verify the data was loaded
        count = mongodb_db.text_based_detection.count_documents({})
        print(f"✅ Loaded {count} documents into MongoDB text_based_detection collection")
        
        return True
    except Exception as e:
        print(f"❌ Error loading text-based detection data to MongoDB: {e}")
        return False

# TODO: Shreyaa to implement this function
def load_rba_dataset_to_database(ssh_client,mysql_conn):
    try :
        if ssh_client is None or mysql_conn is None :
            return False
        
        #Create Table
        with mysql_conn.cursor() as cursor :
            cursor.execute("""
            DROP TABLE IF EXISTS rba_logins;
            """)

            cursor.execute("""
            CREATE TABLE rba_logins (
            id INT AUTO_INCREMENT PRIMARY KEY,
            login_timestamp DATETIME(3),
            userid VARCHAR(25),
            round_trip_time_ms FLOAT,
            ip_address VARCHAR(45),
            country VARCHAR(5),
            region VARCHAR(100),
            city VARCHAR(100),
            asn VARCHAR(100),
            user_agent TEXT
            );
            """)
            mysql_conn.commit()


        #Read dataset from remote server
        chunks = read_dataset_from_server(ssh_client,DATASET_PATHS['rba_dataset'],chunksize=10000)
        if chunks is None :
            print(f"❌ Dataset could not be read")
            return False

        #Security state
        slow_login_count = {}
        ip_user_map = {}
        user_geo = {}
        
        total_inserted=0
        approx = 33000000 // 10000
        chunk_progress = tqdm(chunks, total = approx,desc="Loading Chunks", unit="chunk")


        for chunk in chunk_progress :
            rows_to_insert = []

            for _, row in chunk.iterrows() :
                try:
                    #Handle timestamp
                    ts = row['Login Timestamp']
                    if isinstance(ts,str):
                        login_timestamp = pd.to_datetime(row['Login Timestamp'], errors='coerce')
                    else:
                        login_timestamp=None

                    #Handle userid
                    user_id = str(row['User ID'])
                    
                    #Prepare data row
                    rows_to_insert.append((
                        login_timestamp,
                        user_id,
                        row['Round-Trip Time [ms]'] if not pd.isnull(row['Round-Trip Time [ms]']) else None,
                        row['IP Address']if not pd.isnull(row['IP Address']) else None,
                        row['Country'] if not pd.isnull(row['Country']) else None,
                        row['Region'] if not pd.isnull(row['Region']) else None,
                        row['City'] if not pd.isnull(row['City']) else None,
                        row['ASN'] if not pd.isnull(row['ASN']) else None,
                        row['User Agent String'] if not pd.isnull(row['User Agent String']) else None
                    ))
                except Exception as row_err :
                    print(f"⚠️ Skipping row due to error: {row_err}")
                
            #Insert chunk
            if rows_to_insert:
                with mysql_conn.cursor() as cursor :
                    cursor.executemany("""
                    INSERT INTO rba_logins(
                    login_timestamp, userid, round_trip_time_ms,
                    ip_address, country, region, city,
                    asn, user_agent) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s);
                    """, rows_to_insert)
                    mysql_conn.commit()
                
                total_inserted += len(rows_to_insert)
                chunk_progress.set_postfix(inserted=total_inserted)
                

        print("🎉 Dataset successfully loaded into MySQL.")
        return True

    except Exception as e:
        print(f"❌ Error in load_rba_dataset_to_database: {e}")
        return False


# TODO: Sivangi to implement this function
def create_data_visualization_pipeline(ssh_client):
    """
    This function prepares data for visualization dashboards.
    
    Your task:
    1. Create aggregated views of security data
    2. Prepare data for different chart types (pie, bar, line)
    3. Save data in a format ready for the Next.js frontend
    
    Hints:
    - Use pandas for data aggregation
    - Create summary statistics for dashboard visualizations
    - Save results as JSON files for easy frontend integration
    """
    print("\n=== TASK FOR SIVANGI: IMPLEMENT DATA VISUALIZATION PIPELINE ===")
    
    # Example code to get you started:
    """
    # Load a sample of the intrusion detection dataset
    df = read_dataset_from_server(ssh_client, DATASET_PATHS['intrusion_detection'])
    
    # Create a directory for dashboard data
    dashboard_dir = Path('dashboard_data')
    dashboard_dir.mkdir(exist_ok=True)
    
    # Create summary data for pie chart
    attack_summary = df['attack_detected'].value_counts().reset_index()
    attack_summary.columns = ['is_attack', 'count']
    
    # Add labels for the frontend
    attack_summary['label'] = attack_summary['is_attack'].apply(
        lambda x: 'Attack' if x == 1 else 'Normal'
    )
    
    # Create data structure for frontend
    pie_data = {
        'labels': attack_summary['label'].tolist(),
        'values': attack_summary['count'].tolist(),
        'colors': ['#ff6666', '#66b3ff']
    }
    
    # Save as JSON
    with open(dashboard_dir / 'attack_distribution.json', 'w') as f:
        json.dump(pie_data, f)
    
    print(f"✅ Created visualization data at {dashboard_dir}")
    """
    
    print("TODO: Implement data visualization pipeline (assigned to Sivangi)")
    return False

# Main function
def main():
    print("Starting data loading process...")
    start_time = time.time()
    
    # Connect to SSH server
    ssh_client = connect_to_server()
    if ssh_client is None:
        print("❌ Cannot continue without SSH connection")
        return
    
    try:
        # Connect to databases
        mysql_conn = connect_mysql(MYSQL_CONFIG)
        postgres_conn = connect_postgres(POSTGRES_CONFIG)
        mongodb_db = connect_mongodb(MONGODB_CONFIG)
        
        # Load datasets to appropriate databases
        print("\n1. Loading Intrusion Detection Dataset to MySQL...")
        intrusion_success = load_intrusion_detection_to_mysql(ssh_client, mysql_conn)
        
        '''print("\n2. Loading AI-Enhanced Events Dataset to PostgreSQL...")
        ai_enhanced_success = load_ai_enhanced_to_postgres(ssh_client, postgres_conn)'''
        
        print("\n3. Loading Text-based Detection Dataset to MongoDB...")
        text_based_success = load_text_based_to_mongodb(ssh_client, mongodb_db)
        
        print("\n4. Loading Risk-Based Authentication Dataset...")
        print("   This task is assigned to Shreyaa (TODO)")
        rba_success = load_rba_dataset_to_database(ssh_client,mysql_conn)
        
        print("\n5. Creating Data Visualization Pipeline...")
        print("   This task is assigned to Sivangi (TODO)")
        viz_success = create_data_visualization_pipeline(ssh_client)
        
        # Close connections
        if mysql_conn:
            mysql_conn.close()
        
        if postgres_conn:
            postgres_conn.close()
        
        # MongoDB connections are closed automatically
    
    finally:
        # Close SSH connection
        if ssh_client:
            ssh_client.close()
            print("SSH connection closed")
    
    end_time = time.time()
    duration = end_time - start_time
    
    # Write summary to log file
    with open(log_dir / "loading_summary.txt", "w") as f:
        f.write("Data Loading Summary\n")
        f.write("===================\n\n")
        f.write(f"Total duration: {duration:.2f} seconds\n\n")
        f.write("Status by dataset:\n")
        f.write(f"✅ Intrusion Detection Dataset: {'Loaded' if intrusion_success else 'Failed'}\n")
        #f.write(f"✅ AI-Enhanced Events Dataset: {'Loaded' if ai_enhanced_success else 'Failed'}\n")
        f.write(f"✅ Text-based Detection Dataset: {'Loaded' if text_based_success else 'Failed'}\n")
        f.write(f"🔜 Risk-Based Authentication Dataset: TODO (Shreyaa)\n")
        f.write(f"🔜 Data Visualization Pipeline: TODO (Sivangi)\n")
    
    print("\nData loading process completed!")
    print(f"Total duration: {duration:.2f} seconds")
    print(f"\nSummary log saved to {log_dir / 'loading_summary.txt'}")

if __name__ == "__main__":
    main()