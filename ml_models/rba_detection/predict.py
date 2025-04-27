#!/usr/bin/env python3

"""
Prediction module for RBA anomaly detection model

Provides functions to:
- Load the trained model
- Make predictions on new login data
- Interpret the anomaly detection results
"""

import os
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import pymysql
import json
from datetime import datetime, timedelta
import paramiko
from dotenv import load_dotenv
import sys

# Load environment variables
load_dotenv()

# SSH connection settings
SSH_HOST = os.getenv('SSH_HOST')
SSH_USER = os.getenv('SSH_USER')
SSH_KEY_PATH = os.path.expanduser(os.getenv('SSH_KEY_PATH'))

# Dataset path on remote server
RBA_DATASET = os.getenv('RBA_DATASET')

# Path to model artifacts
model_dir = Path('model_artifacts')
model_dir.mkdir(exist_ok=True)

def load_model():
    """Load the trained model and preprocessor"""
    model_path = model_dir / 'rba_detection_model.pkl'
    preprocessor_path = model_dir / 'rba_preprocessor.pkl'
    
    if not model_path.exists() or not preprocessor_path.exists():
        # Try to look in the parent directory
        parent_model_dir = Path('../model_artifacts')
        model_path = parent_model_dir / 'rba_detection_model.pkl'
        preprocessor_path = parent_model_dir / 'rba_preprocessor.pkl'
        
        if not model_path.exists() or not preprocessor_path.exists():
            raise FileNotFoundError(f"Model or preprocessor not found at {model_dir} or {parent_model_dir}")
    
    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)
    
    return model, preprocessor

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

def read_dataset_from_server(ssh_client, dataset_path, nrows=10000):
    """Read a limited sample from the dataset (for prediction)"""
    try:
        if ssh_client is None:
            raise Exception("SSH client is not connected")
            
        # Get expanded path
        stdin, stdout, stderr = ssh_client.exec_command(f"echo {dataset_path}")
        expanded_path = stdout.read().decode('utf-8').strip()
        
        print(f"Reading dataset sample from {expanded_path}")
        
        # For limited rows (sample)
        command = f"head -n {nrows+1} {expanded_path}"  # +1 for header
        stdin, stdout, stderr = ssh_client.exec_command(command)
        df = pd.read_csv(stdout)
        
        print(f"Read {len(df)} sample rows from dataset")
        return df
    except Exception as e:
        print(f"❌ Error reading dataset: {e}")
        return None

def preprocess_chunk(df):
    """Preprocess a chunk of RBA data - simplified version from model.py"""
    try:
        # Import preprocessing functions here to match model.py pattern
        from preprocessing import (
            extract_user_agent_features, 
            extract_time_features, 
            extract_ip_features, 
            create_risk_score
        )
        
        # Extract features - making sure to use 'User Agent String' not 'user_agent'
        df_processed = extract_user_agent_features(df)  # This should handle 'User Agent String'
        df_processed = extract_time_features(df_processed)
        df_processed = extract_ip_features(df_processed)
        
        # Create user profiles for behavior patterns (limited to the current chunk)
        user_profiles = {}
        
        # Group by user
        for userid, user_df in df_processed.groupby('User ID'):
            if len(user_df) < 2:  # Skip users with only one login
                continue
                
            # Get user's typical login patterns
            user_profiles[userid] = {
                'common_countries': user_df['Country'].value_counts().index.tolist(),
                'common_device_types': user_df['device_type'].value_counts().index.tolist(),
                'common_browsers': user_df['browser'].value_counts().index.tolist(),
                'avg_rtt': user_df['Round-Trip Time [ms]'].mean() if 'Round-Trip Time [ms]' in user_df else None,
                'typical_hours': user_df['hour'].value_counts().index.tolist() if 'hour' in user_df else [],
                'login_count': len(user_df)
            }
        
        # Add features based on user profiles
        def is_typical_country(row):
            userid = row['User ID']
            country = row['Country']
            if userid in user_profiles and len(user_profiles[userid]['common_countries']) > 0:
                return 1 if country in user_profiles[userid]['common_countries'][:1] else 0
            return 0
        
        def is_typical_device(row):
            userid = row['User ID']
            device = row['device_type']
            if userid in user_profiles and len(user_profiles[userid]['common_device_types']) > 0:
                return 1 if device in user_profiles[userid]['common_device_types'][:1] else 0
            return 0
        
        def is_typical_browser(row):
            userid = row['User ID']
            browser = row['browser']
            if userid in user_profiles and len(user_profiles[userid]['common_browsers']) > 0:
                return 1 if browser in user_profiles[userid]['common_browsers'][:1] else 0
            return 0
        
        def is_typical_hour(row):
            userid = row['User ID']
            if 'hour' not in row or pd.isna(row['hour']):
                return 0
            hour = row['hour']
            if userid in user_profiles and len(user_profiles[userid]['typical_hours']) > 0:
                return 1 if hour in user_profiles[userid]['typical_hours'][:3] else 0
            return 0
        
        def get_rtt_deviation(row):
            userid = row['User ID']
            rtt = row['Round-Trip Time [ms]']
            if pd.isna(rtt):
                return 0
            if userid in user_profiles and user_profiles[userid]['avg_rtt'] is not None:
                avg_rtt = user_profiles[userid]['avg_rtt']
                if avg_rtt > 0:
                    return abs(rtt - avg_rtt) / avg_rtt
            return 0
        
        def get_login_frequency(row):
            userid = row['User ID']
            if userid in user_profiles:
                return user_profiles[userid]['login_count']
            return 1  # Default to 1 for users not in profiles
        
        # Apply the functions
        df_processed['is_typical_country'] = df_processed.apply(is_typical_country, axis=1)
        df_processed['is_typical_device'] = df_processed.apply(is_typical_device, axis=1)
        df_processed['is_typical_browser'] = df_processed.apply(is_typical_browser, axis=1)
        df_processed['is_typical_hour'] = df_processed.apply(is_typical_hour, axis=1)
        df_processed['rtt_deviation'] = df_processed.apply(get_rtt_deviation, axis=1)
        df_processed['login_frequency'] = df_processed.apply(get_login_frequency, axis=1)
        
        # Create risk score
        df_processed = create_risk_score(df_processed)
        
        # Define feature columns
        feature_columns = [
            'Round-Trip Time [ms]', 'device_type', 'browser', 'os',
            'hour', 'day', 'day_of_week', 'month', 'is_weekend',
            'time_category', 'is_business_hours', 'ip_version',
            'is_private_ip', 'is_typical_country', 'is_typical_device',
            'is_typical_browser', 'is_typical_hour', 'rtt_deviation',
            'login_frequency', 'risk_score'
        ]
        
        # Define reference columns
        reference_columns = ['User ID', 'Login Timestamp', 'Country', 'IP Address']
        
        # Handle missing values
        for col in feature_columns:
            if col in df_processed.columns:
                if df_processed[col].dtype in ['float64', 'int64']:
                    df_processed[col] = df_processed[col].fillna(0)
                else:
                    df_processed[col] = df_processed[col].fillna('unknown')
        
        # Select columns for model features and reference data
        X = df_processed[[col for col in feature_columns if col in df_processed.columns]].copy()
        ref = df_processed[[col for col in reference_columns if col in df_processed.columns]].copy()
        
        return X, ref, feature_columns, reference_columns
    
    except Exception as e:
        print(f"Error preprocessing chunk: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None

def detect_anomalies(data=None, limit=10000):
    """
    Detect anomalies in login data using SSH connection and direct data reading
    
    Args:
        data: DataFrame containing login data, or None to fetch from server
        limit: Maximum number of rows to read from the dataset
        
    Returns:
        DataFrame with original data plus anomaly detection results
    """
    # Connect to server
    ssh_client = connect_to_server()
    if ssh_client is None:
        print("❌ Cannot continue without SSH connection")
        return None
    
    try:
        # Get data if not provided
        if data is None:
            data = read_dataset_from_server(ssh_client, RBA_DATASET, nrows=limit)
        
        if data is None or data.empty:
            print("No data available for anomaly detection")
            return pd.DataFrame()
        
        # Preprocess data
        X, reference_data, feature_columns, _ = preprocess_chunk(data)
        
        if X is None:
            print("Error preprocessing data")
            return pd.DataFrame()
            
        # Load model and preprocessor
        model, preprocessor = load_model()
        
        # Transform features using the preprocessor
        X_transformed = preprocessor.transform(X)
        
        # Predict anomalies
        raw_scores = model.predict(X_transformed)
        decision_scores = model.decision_function(X_transformed)
        
        # Convert to binary labels (1 for anomalies, 0 for normal)
        anomaly_labels = np.where(raw_scores == -1, 1, 0)
        
        # Calculate anomaly probability
        min_score = decision_scores.min()
        max_score = decision_scores.max()
        score_range = max_score - min_score
        
        if score_range > 0:
            anomaly_probs = 1 - ((decision_scores - min_score) / score_range)
        else:
            anomaly_probs = np.zeros_like(decision_scores)
        
        # Combine results with original data
        results_df = reference_data.copy()
        results_df['anomaly_detected'] = anomaly_labels
        results_df['anomaly_score'] = decision_scores
        results_df['anomaly_probability'] = anomaly_probs
        
        # Add risk categories
        results_df['risk_category'] = pd.cut(
            results_df['anomaly_probability'],
            bins=[0, 0.5, 0.8, 0.95, 1.0],
            labels=['Low', 'Medium', 'High', 'Critical']
        )
        
        # Add original feature values for reference
        for col in X.columns:
            results_df[col] = X[col].values
        
        print(f"✅ Anomaly detection complete for {len(results_df)} logins")
        return results_df
    except Exception as e:
        print(f"❌ Error in anomaly detection process: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()
    finally:
        # Close SSH connection
        if ssh_client:
            ssh_client.close()
            print("SSH connection closed")

def save_predictions_to_database(predictions_df, table_name='rba_anomalies', batch_size=100):
    """Save anomaly detection results to database in batches with progress reporting"""
    if predictions_df.empty:
        print("No predictions to save to database")
        return
        
    mysql_config = {
        'host': '40.76.125.54',  # Azure VM IP address
        'user': 'shreyaa',  # Username
        'password': 'Shreyaa@123',  # Password
        'database': 'shreyaa',  # Database name
        'charset': 'utf8mb4',
        'cursorclass': pymysql.cursors.DictCursor,
        'connect_timeout': 30,  # Add timeout
        'read_timeout': 30,
        'write_timeout': 30
    }
    
    try:
        connection = pymysql.connect(**mysql_config)
        print(f"✅ Connected to MySQL: {mysql_config['user']}@{mysql_config['host']}/{mysql_config['database']}")
        
        # Create the anomalies table if it doesn't exist
        with connection.cursor() as cursor:
            print("Creating/checking table structure...")
            cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                id INT AUTO_INCREMENT PRIMARY KEY,
                login_id VARCHAR(50),
                userid VARCHAR(50),
                login_timestamp DATETIME,
                country VARCHAR(50),
                ip_address VARCHAR(50),
                anomaly_detected TINYINT(1),
                anomaly_score FLOAT,
                anomaly_probability FLOAT,
                risk_category VARCHAR(20),
                risk_score FLOAT,
                detection_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """)
            connection.commit()
            print("Table structure confirmed.")
        
        # Insert the predictions in batches
        total_rows = len(predictions_df)
        print(f"Saving {total_rows} predictions to database in batches of {batch_size}...")
        
        # Process in batches
        batches = range(0, total_rows, batch_size)
        
        for i, batch_start in enumerate(batches):
            batch_end = min(batch_start + batch_size, total_rows)
            batch = predictions_df.iloc[batch_start:batch_end]
            
            print(f"Processing batch {i+1}/{len(batches)}: rows {batch_start}-{batch_end}...")
            
            with connection.cursor() as cursor:
                for _, row in batch.iterrows():
                    try:
                        cursor.execute(f"""
                        INSERT INTO {table_name} (
                            login_id,
                            userid,
                            login_timestamp,
                            country,
                            ip_address,
                            anomaly_detected,
                            anomaly_score,
                            anomaly_probability,
                            risk_category,
                            risk_score
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """, (
                            str(row['User ID']) if 'User ID' in row else None,
                            str(row['User ID']) if 'User ID' in row else None,
                            row['Login Timestamp'] if 'Login Timestamp' in row else None,
                            str(row['Country']) if 'Country' in row else None,
                            str(row['IP Address']) if 'IP Address' in row else None,
                            int(row['anomaly_detected']),
                            float(row['anomaly_score']),
                            float(row['anomaly_probability']),
                            str(row['risk_category']),
                            float(row['risk_score']) if 'risk_score' in row else 0.0
                        ))
                    except Exception as e:
                        print(f"Error inserting row: {e}")
                        continue
                connection.commit()
                print(f"Batch {i+1} committed.")
        
        print(f"✅ Saved all predictions to {table_name} table")
    except Exception as e:
        print(f"❌ Error saving predictions: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'connection' in locals() and connection:
            connection.close()
            print("Database connection closed.")

def main():
    """Main function to detect anomalies in a sample of the RBA dataset"""
    try:
        # Detect anomalies in a sample of the dataset
        predictions = detect_anomalies(limit=10000)
        
        if predictions is None or predictions.empty:
            print("No predictions available to save")
            return None
        
        # Save predictions to database
        save_predictions_to_database(predictions)
        
        # Print summary
        anomaly_count = predictions[predictions['anomaly_detected'] == 1].shape[0]
        
        print(f"\nAnomaly Detection Summary:")
        print(f"Total logins analyzed: {len(predictions)}")
        print(f"Anomalies detected: {anomaly_count} ({anomaly_count/len(predictions)*100:.2f}%)")
        
        # Print risk categories
        risk_summary = predictions['risk_category'].value_counts()
        print("\nRisk category breakdown:")
        for category, count in risk_summary.items():
            print(f"  {category}: {count} ({count/len(predictions)*100:.2f}%)")
        
        # Print high/critical risk logins
        high_risk = predictions[predictions['risk_category'].isin(['High', 'Critical'])]
        if not high_risk.empty:
            print(f"\nDetected {len(high_risk)} high/critical risk logins:")
            print(high_risk[['User ID', 'Country', 'anomaly_probability', 'risk_category']].head(10))
        
        return predictions
    except Exception as e:
        print(f"❌ Error in anomaly detection process: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    predictions = main()