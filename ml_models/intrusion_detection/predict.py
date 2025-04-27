#!/usr/bin/env python3

"""
Prediction module for Intrusion Detection model

Provides functions to:
- Load the trained model
- Make predictions on new data
- Interpret the prediction results
"""

import os
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import pymysql
from dotenv import load_dotenv
import json

# Load environment variables
load_dotenv()

# Path to model artifacts
model_dir = Path('model_artifacts')

def load_model():
    """Load the trained model and preprocessor"""
    model_path = model_dir / 'intrusion_detection_model.pkl'
    preprocessor_path = model_dir / 'intrusion_preprocessor.pkl'
    
    if not model_path.exists() or not preprocessor_path.exists():
        raise FileNotFoundError(f"Model or preprocessor not found at {model_dir}")
    
    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)
    
    return model, preprocessor

def connect_to_database():
    """Connect to MySQL database"""
    mysql_config = {
        'host': os.getenv('MYSQL_HOST', '40.76.125.54'),
        'user': os.getenv('SHREYAA_MYSQL_USER', 'shreyaa'),
        'password': os.getenv('SHREYAA_MYSQL_PASSWORD', 'Shreyaa@123'),
        'database': os.getenv('SHREYAA_MYSQL_DB', 'shreyaa'),
        'charset': 'utf8mb4',
        'cursorclass': pymysql.cursors.DictCursor
    }
    
    try:
        connection = pymysql.connect(**mysql_config)
        print(f"✅ Connected to MySQL: {mysql_config['user']}@{mysql_config['host']}/{mysql_config['database']}")
        return connection
    except Exception as e:
        print(f"❌ Error connecting to MySQL: {e}")
        return None

def engineer_features(df):
    """Engineer additional features for prediction"""
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
    
    # Drop session_id as it's just an identifier (if present)
    if 'session_id' in df_processed.columns:
        df_processed.drop('session_id', axis=1, inplace=True)
    
    return df_processed

def get_latest_sessions(limit=100):
    """Get the latest sessions from the database for prediction"""
    connection = connect_to_database()
    
    if connection is None:
        raise Exception("Failed to connect to database")
    
    try:
        with connection.cursor() as cursor:
            query = f"""
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
                unusual_time_access
            FROM intrusion_detection
            ORDER BY id DESC
            LIMIT {limit}
            """
            cursor.execute(query)
            data = cursor.fetchall()
            df = pd.DataFrame(data)
            print(f"✅ Loaded {len(df)} latest sessions for prediction")
            return df
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        raise
    finally:
        connection.close()

def predict_intrusions(data=None, limit=100):
    """
    Make intrusion predictions on new data
    
    Args:
        data: DataFrame containing the sessions to predict
        limit: Number of latest sessions to predict if data is None
        
    Returns:
        DataFrame with original data plus predictions and probabilities
    """
    # Load model and preprocessor
    model, preprocessor = load_model()
    
    # Get data if not provided
    if data is None:
        data = get_latest_sessions(limit=limit)
    
    # Keep original data for the output
    result_df = data.copy()
    
    # Engineer features
    data_processed = engineer_features(data)
    
    # Preprocess data
    data_transformed = preprocessor.transform(data_processed)
    
    # Make predictions
    predictions = model.predict(data_transformed)
    probabilities = model.predict_proba(data_transformed)[:, 1]
    
    # Add predictions to result dataframe
    result_df['prediction'] = predictions
    result_df['attack_probability'] = probabilities
    
    # Add risk level based on probability
    result_df['risk_level'] = pd.cut(
        result_df['attack_probability'], 
        bins=[0, 0.2, 0.5, 0.8, 1.0],
        labels=['Low', 'Medium', 'High', 'Critical']
    )
    
    print(f"✅ Predictions made for {len(result_df)} sessions")
    return result_df

def save_predictions_to_database(predictions_df, table_name='intrusion_predictions'):
    """Save predictions to database"""
    connection = connect_to_database()
    
    if connection is None:
        raise Exception("Failed to connect to database")
    
    try:
        # Create the predictions table if it doesn't exist
        with connection.cursor() as cursor:
            cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                id INT AUTO_INCREMENT PRIMARY KEY,
                session_id VARCHAR(50),
                prediction TINYINT(1),
                attack_probability FLOAT,
                risk_level VARCHAR(20),
                prediction_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """)
            connection.commit()
        
        # Insert the predictions
        with connection.cursor() as cursor:
            for _, row in predictions_df.iterrows():
                cursor.execute(f"""
                INSERT INTO {table_name} (
                    session_id,
                    prediction,
                    attack_probability,
                    risk_level
                ) VALUES (%s, %s, %s, %s)
                """, (
                    row['session_id'],
                    int(row['prediction']),
                    float(row['attack_probability']),
                    row['risk_level']
                ))
            connection.commit()
        
        print(f"✅ Saved {len(predictions_df)} predictions to {table_name} table")
    except Exception as e:
        print(f"❌ Error saving predictions: {e}")
        raise
    finally:
        connection.close()

def main():
    """Main function to make predictions on latest sessions"""
    try:
        # Make predictions on the latest 100 sessions
        predictions = predict_intrusions(limit=100)
        
        # Save predictions to database
        save_predictions_to_database(predictions)
        
        # Print summary
        attack_count = predictions[predictions['prediction'] == 1].shape[0]
        normal_count = predictions[predictions['prediction'] == 0].shape[0]
        
        print(f"\nPrediction Summary:")
        print(f"Total sessions: {len(predictions)}")
        print(f"Attack sessions: {attack_count} ({attack_count/len(predictions)*100:.2f}%)")
        print(f"Normal sessions: {normal_count} ({normal_count/len(predictions)*100:.2f}%)")
        
        # Print high-risk sessions
        high_risk = predictions[predictions['risk_level'].isin(['High', 'Critical'])]
        if not high_risk.empty:
            print(f"\nDetected {len(high_risk)} high/critical risk sessions:")
            print(high_risk[['session_id', 'attack_probability', 'risk_level']].head(10))
        
        return predictions
    except Exception as e:
        print(f"❌ Error in prediction process: {e}")
        return None

if __name__ == "__main__":
    predictions = main()