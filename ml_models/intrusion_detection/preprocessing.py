#!/usr/bin/env python3

"""
Preprocessing for Intrusion Detection dataset
- Feature engineering
- Encoding categorical variables
- Scaling numerical features
- Train-test split
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import pymysql
from dotenv import load_dotenv
import joblib
from pathlib import Path

# Load environment variables
load_dotenv()

# Create directory for model artifacts
model_dir = Path('model_artifacts')
model_dir.mkdir(exist_ok=True)

# Update connect_to_database function
def connect_to_database():
    """Connect to MySQL database"""
    from config import MYSQL_CONFIG
    import pymysql
    
    try:
        connection = pymysql.connect(
            host=MYSQL_CONFIG['host'],
            user=MYSQL_CONFIG['user'],
            password=MYSQL_CONFIG['password'],
            database=MYSQL_CONFIG['database'],
            charset=MYSQL_CONFIG['charset'],
            cursorclass=pymysql.cursors.DictCursor
        )
        print(f"✅ Connected to MySQL: {MYSQL_CONFIG['user']}@{MYSQL_CONFIG['host']}/{MYSQL_CONFIG['database']}")
        return connection
    except Exception as e:
        print(f"❌ Error connecting to MySQL: {e}")
        return None

def load_data():
    """Load intrusion detection data from MySQL"""
    connection = connect_to_database()
    
    if connection is None:
        raise Exception("Failed to connect to database")
    
    try:
        with connection.cursor() as cursor:
            query = """
            SELECT 
                session_id,
                network_packet_size,
                protocol_type, 
                login_attempts,
                session_duration,
                encryption_used,
                ip_reputation_score,
                failed_logins,
                browser_type, 
                unusual_time_access,
                attack_detected
            FROM intrusion_detection
            """
            cursor.execute(query)
            data = cursor.fetchall()
            df = pd.DataFrame(data)
            print(f"✅ Loaded {len(df)} records from intrusion_detection table")
            return df
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        raise
    finally:
        connection.close()

def engineer_features(df):
    """Engineer additional features for improving model performance"""
    print("Engineering features...")
    
    # Create copy to avoid modifying original dataframe
    df_processed = df.copy()
    
    # Feature: login_failure_ratio (if login_attempts > 0)
    df_processed['login_failure_ratio'] = df_processed.apply(
        lambda row: row['failed_logins'] / row['login_attempts'] 
        if row['login_attempts'] > 0 else 0, 
        axis=1
    )
    
    # Feature: packet_duration_ratio
    df_processed['packet_duration_ratio'] = df_processed['network_packet_size'] / (df_processed['session_duration'] + 1)
    
    # Feature: risk_score (combined risk metric)
    df_processed['risk_score'] = (
        df_processed['failed_logins'] * 0.3 +
        df_processed['unusual_time_access'] * 0.4 +
        (1 - df_processed['ip_reputation_score']) * 0.3
    )
    
    # Feature: encryption_level (numeric representation)
    encryption_map = {'none': 0, 'basic': 1, 'advanced': 2}
    df_processed['encryption_level'] = df_processed['encryption_used'].map(encryption_map)
    
    # Drop session_id as it's just an identifier
    if 'session_id' in df_processed.columns:
        df_processed.drop('session_id', axis=1, inplace=True)
    
    return df_processed

def preprocess_data(df, test_size=0.2, random_state=42):
    """
    Preprocess intrusion detection data:
    - Handle categorical features
    - Scale numerical features
    - Split into train/test sets
    """
    print("Preprocessing data...")
    
    # First apply feature engineering
    df_processed = engineer_features(df)
    
    # Define feature types
    categorical_features = ['protocol_type', 'encryption_used', 'browser_type']
    numerical_features = [
        'network_packet_size', 'login_attempts', 'session_duration',
        'ip_reputation_score', 'failed_logins', 'unusual_time_access',
        'login_failure_ratio', 'packet_duration_ratio', 'risk_score',
        'encryption_level'
    ]
    
    # Define target
    X = df_processed.drop('attack_detected', axis=1)
    y = df_processed['attack_detected']
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Create preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ]
    )
    
    # Fit preprocessor on training data only
    preprocessor.fit(X_train)
    
    # Save the preprocessor for future use
    joblib.dump(preprocessor, model_dir / 'intrusion_preprocessor.pkl')
    
    # Transform both train and test data
    X_train_processed = preprocessor.transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    print(f"✅ Preprocessing complete. Train shape: {X_train_processed.shape}, Test shape: {X_test_processed.shape}")
    
    return X_train_processed, X_test_processed, y_train, y_test, preprocessor, list(X.columns)

if __name__ == "__main__":
    # Test the preprocessing pipeline
    print("Testing preprocessing pipeline...")
    df = load_data()
    X_train, X_test, y_train, y_test, preprocessor, feature_names = preprocess_data(df)
    print("Feature names:", feature_names)
    print("Preprocessor:", preprocessor)
    print("Preprocessing test successful!")