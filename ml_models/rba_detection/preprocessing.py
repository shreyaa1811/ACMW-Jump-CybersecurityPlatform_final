#!/usr/bin/env python3

"""
Preprocessing for Risk-Based Authentication (RBA) dataset
- Feature engineering for login patterns
- Temporal features
- Geolocation-based features
- User behavior profiling
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pymysql
import joblib
from pathlib import Path
from datetime import datetime, timedelta
import ipaddress
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Create directory for model artifacts
model_dir = Path('model_artifacts')
model_dir.mkdir(exist_ok=True)

def extract_user_agent_features(df):
    """Extract features from user agent string"""
    # Create a copy to avoid modifying the original
    df_processed = df.copy()
    
    # Check if 'User Agent String' column exists and map it to 'user_agent'
    if 'User Agent String' in df_processed.columns:
        df_processed['user_agent'] = df_processed['User Agent String']
    elif 'user_agent' not in df_processed.columns:
        # If neither column exists, create a default
        print("Warning: No user agent column found. Using default values.")
        df_processed['user_agent'] = "Unknown"
    
    # Device type extraction
    def get_device_type(ua):
        ua = str(ua).lower() if not pd.isna(ua) else ""
        if 'mobile' in ua or 'android' in ua or 'iphone' in ua or 'ipad' in ua:
            return 'mobile'
        elif 'bot' in ua or 'crawler' in ua or 'spider' in ua:
            return 'bot'
        else:
            return 'desktop'
    
    # Browser extraction
    def get_browser(ua):
        ua = str(ua).lower() if not pd.isna(ua) else ""
        if 'chrome' in ua:
            return 'chrome'
        elif 'firefox' in ua:
            return 'firefox'
        elif 'safari' in ua and 'chrome' not in ua:  # Chrome includes Safari in UA
            return 'safari'
        elif 'edge' in ua:
            return 'edge'
        elif 'msie' in ua or 'trident' in ua:
            return 'ie'
        elif 'opera' in ua:
            return 'opera'
        else:
            return 'other'
    
    # OS extraction
    def get_os(ua):
        ua = str(ua).lower() if not pd.isna(ua) else ""
        if 'windows' in ua:
            return 'windows'
        elif 'mac os' in ua or 'macos' in ua:
            return 'macos'
        elif 'linux' in ua:
            return 'linux'
        elif 'android' in ua:
            return 'android'
        elif 'ios' in ua or 'iphone' in ua or 'ipad' in ua:
            return 'ios'
        else:
            return 'other'
    
    # Apply the extraction functions
    df_processed['device_type'] = df_processed['user_agent'].apply(get_device_type)
    df_processed['browser'] = df_processed['user_agent'].apply(get_browser)
    df_processed['os'] = df_processed['user_agent'].apply(get_os)
    
    return df_processed

def extract_time_features(df):
    """Extract temporal features from login timestamp"""
    # Create a copy to avoid modifying the original
    df_processed = df.copy()
    
    # Check if 'Login Timestamp' column exists and map it to 'login_timestamp'
    if 'Login Timestamp' in df_processed.columns:
        # Convert to datetime if needed
        try:
            df_processed['login_timestamp'] = pd.to_datetime(df_processed['Login Timestamp'], errors='coerce')
        except:
            print("Warning: Could not convert Login Timestamp to datetime")
            # Create a default timestamp
            df_processed['login_timestamp'] = pd.to_datetime('now')
    elif 'login_timestamp' not in df_processed.columns:
        # If neither column exists, create a default
        print("Warning: No timestamp column found. Using current time.")
        df_processed['login_timestamp'] = pd.to_datetime('now')
    
    # Extract basic time components
    df_processed['hour'] = df_processed['login_timestamp'].dt.hour
    df_processed['day'] = df_processed['login_timestamp'].dt.day
    df_processed['day_of_week'] = df_processed['login_timestamp'].dt.dayofweek
    df_processed['month'] = df_processed['login_timestamp'].dt.month
    df_processed['year'] = df_processed['login_timestamp'].dt.year
    
    # Weekend indicator
    df_processed['is_weekend'] = df_processed['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    
    # Time of day categories
    def get_time_category(hour):
        if 0 <= hour < 6:
            return 'night'
        elif 6 <= hour < 12:
            return 'morning'
        elif 12 <= hour < 18:
            return 'afternoon'
        else:  # 18 <= hour < 24
            return 'evening'
    
    df_processed['time_category'] = df_processed['hour'].apply(get_time_category)
    
    # Business hours indicator (9 AM to 5 PM, Monday to Friday)
    df_processed['is_business_hours'] = df_processed.apply(
        lambda row: 1 if 9 <= row['hour'] < 17 and row['day_of_week'] < 5 else 0, 
        axis=1
    )
    
    return df_processed

def extract_ip_features(df):
    """Extract features from IP address"""
    # Create a copy to avoid modifying the original
    df_processed = df.copy()
    
    # Check if 'IP Address' column exists and map it to 'ip_address'
    if 'IP Address' in df_processed.columns:
        df_processed['ip_address'] = df_processed['IP Address']
    elif 'ip_address' not in df_processed.columns:
        # If neither column exists, create a default
        print("Warning: No IP address column found. Using default values.")
        df_processed['ip_address'] = "192.168.1.1"
    
    # IP version
    def get_ip_version(ip):
        if pd.isna(ip):
            return 4  # Default to IPv4
        try:
            return ipaddress.ip_address(ip).version
        except:
            return 4  # Default to IPv4 on error
    
    # Private IP indicator
    def is_private_ip(ip):
        if pd.isna(ip):
            return 0  # Default to non-private
        try:
            return int(ipaddress.ip_address(ip).is_private)
        except:
            return 0  # Default to non-private on error
    
    df_processed['ip_version'] = df_processed['ip_address'].apply(get_ip_version)
    df_processed['is_private_ip'] = df_processed['ip_address'].apply(is_private_ip)
    
    return df_processed

def extract_user_behavior_features(df):
    """Extract user behavior pattern features"""
    # Create a copy to avoid modifying the original
    df_processed = df.copy()
    
    # Map column names if needed
    if 'User ID' in df_processed.columns and 'userid' not in df_processed.columns:
        df_processed['userid'] = df_processed['User ID']
    
    if 'Country' in df_processed.columns and 'country' not in df_processed.columns:
        df_processed['country'] = df_processed['Country']
        
    if 'Round-Trip Time [ms]' in df_processed.columns and 'round_trip_time_ms' not in df_processed.columns:
        df_processed['round_trip_time_ms'] = df_processed['Round-Trip Time [ms]']
    
    # Create user profile features
    user_profiles = {}
    
    # Group by user
    user_id_column = 'userid' if 'userid' in df_processed.columns else 'User ID'
    
    for userid, user_df in df_processed.groupby(user_id_column):
        if len(user_df) < 2:  # Skip users with only one login
            continue
            
        # Get user's typical login patterns
        country_col = 'country' if 'country' in user_df.columns else 'Country'
        rtt_col = 'round_trip_time_ms' if 'round_trip_time_ms' in user_df.columns else 'Round-Trip Time [ms]'
        
        user_profiles[userid] = {
            'common_countries': user_df[country_col].value_counts().index.tolist() if country_col in user_df else [],
            'common_device_types': user_df['device_type'].value_counts().index.tolist() if 'device_type' in user_df else [],
            'common_browsers': user_df['browser'].value_counts().index.tolist() if 'browser' in user_df else [],
            'avg_rtt': user_df[rtt_col].mean() if rtt_col in user_df else None,
            'typical_hours': user_df['hour'].value_counts().index.tolist() if 'hour' in user_df else [],
            'login_count': len(user_df)
        }
    
    # Add features based on user profiles
    def is_typical_country(row):
        userid = row[user_id_column]
        country_col = 'country' if 'country' in row else 'Country'
        if country_col not in row:
            return 0
        country = row[country_col]
        if userid in user_profiles and len(user_profiles[userid]['common_countries']) > 0:
            return 1 if country in user_profiles[userid]['common_countries'][:1] else 0
        return 0
    
    def is_typical_device(row):
        userid = row[user_id_column]
        if 'device_type' not in row:
            return 0
        device = row['device_type']
        if userid in user_profiles and len(user_profiles[userid]['common_device_types']) > 0:
            return 1 if device in user_profiles[userid]['common_device_types'][:1] else 0
        return 0
    
    def is_typical_browser(row):
        userid = row[user_id_column]
        if 'browser' not in row:
            return 0
        browser = row['browser']
        if userid in user_profiles and len(user_profiles[userid]['common_browsers']) > 0:
            return 1 if browser in user_profiles[userid]['common_browsers'][:1] else 0
        return 0
    
    def is_typical_hour(row):
        userid = row[user_id_column]
        if 'hour' not in row:
            return 0
        hour = row['hour']
        if userid in user_profiles and len(user_profiles[userid]['typical_hours']) > 0:
            return 1 if hour in user_profiles[userid]['typical_hours'][:3] else 0
        return 0
    
    def get_rtt_deviation(row):
        userid = row[user_id_column]
        rtt_col = 'round_trip_time_ms' if 'round_trip_time_ms' in row else 'Round-Trip Time [ms]'
        if rtt_col not in row or pd.isna(row[rtt_col]):
            return 0
        rtt = row[rtt_col]
        if userid in user_profiles and user_profiles[userid]['avg_rtt'] is not None:
            avg_rtt = user_profiles[userid]['avg_rtt']
            if avg_rtt > 0:
                return abs(rtt - avg_rtt) / avg_rtt
        return 0
    
    def get_login_frequency(row):
        userid = row[user_id_column]
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
    
    return df_processed

def create_risk_score(df):
    """Create a simple risk score based on features"""
    # Create a copy to avoid modifying the original
    df_processed = df.copy()
    
    # Initialize risk score
    df_processed['risk_score'] = 0
    
    # Add risk based on various factors
    risk_factors = [
        # Non-typical patterns
        ('is_typical_country', 0, 30),  # High weight for unusual country
        ('is_typical_device', 0, 15),   # Medium weight for unusual device
        ('is_typical_browser', 0, 10),  # Medium weight for unusual browser
        ('is_typical_hour', 0, 10),     # Medium weight for unusual hour
        
        # Time-based risk
        ('is_business_hours', 0, 5),    # Small weight for non-business hours
        ('is_weekend', 1, 5),           # Small weight for weekend logins
        
        # IP-based risk
        ('is_private_ip', 0, 5),        # Small weight for non-private IPs
    ]
    
    # Apply risk factors
    for feature, risk_value, weight in risk_factors:
        if feature in df_processed.columns:
            # Only apply risk if feature is not null and matches risk value
            mask = (df_processed[feature] == risk_value) & (~df_processed[feature].isna())
            df_processed.loc[mask, 'risk_score'] += weight
    
    # Add risk for high RTT deviation (if available)
    if 'rtt_deviation' in df_processed.columns:
        # Scale rtt_deviation to 0-20 range for those with > 50% deviation
        high_rtt_mask = (df_processed['rtt_deviation'] > 0.5) & (~df_processed['rtt_deviation'].isna())
        df_processed.loc[high_rtt_mask, 'risk_score'] += 10
    
    # Normalize risk score to 0-100 range
    max_possible_score = sum(weight for _, _, weight in risk_factors) + 10  # Add RTT weight
    df_processed['risk_score'] = (df_processed['risk_score'] / max_possible_score) * 100
    
    return df_processed

def preprocess_data(df):
    """
    Preprocess RBA data:
    - Extract features from user agent
    - Extract temporal features
    - Extract IP-based features
    - Extract user behavior patterns
    - Create risk score
    """
    print("Preprocessing data...")
    
    # Extract features
    df_processed = extract_user_agent_features(df)
    df_processed = extract_time_features(df_processed)
    df_processed = extract_ip_features(df_processed)
    df_processed = extract_user_behavior_features(df_processed)
    df_processed = create_risk_score(df_processed)
    
    # Define columns to keep for modeling
    feature_columns = [
        'round_trip_time_ms', 'device_type', 'browser', 'os',
        'hour', 'day', 'day_of_week', 'month', 'is_weekend',
        'time_category', 'is_business_hours', 'ip_version',
        'is_private_ip', 'is_typical_country', 'is_typical_device',
        'is_typical_browser', 'is_typical_hour', 'rtt_deviation',
        'login_frequency', 'risk_score'
    ]
    
    # Get feature columns that exist in the dataframe
    available_feature_columns = [col for col in feature_columns if col in df_processed.columns]
    
    # If we're missing any feature columns, try to find equivalent or create with defaults
    if len(available_feature_columns) < len(feature_columns):
        missing_cols = set(feature_columns) - set(available_feature_columns)
        print(f"Warning: Missing feature columns: {missing_cols}")
        
        # Map from original dataset columns to expected feature columns
        column_mapping = {
            'Round-Trip Time [ms]': 'round_trip_time_ms'
        }
        
        # Apply mappings and create default values
        for col in missing_cols:
            # Check if there's a mapping
            for orig_col, target_col in column_mapping.items():
                if col == target_col and orig_col in df_processed.columns:
                    print(f"Mapping {orig_col} to {target_col}")
                    df_processed[target_col] = df_processed[orig_col]
                    available_feature_columns.append(target_col)
                    break
            else:
                # No mapping found, create default
                if col not in df_processed.columns:
                    if col in ['device_type', 'browser', 'os', 'time_category']:
                        df_processed[col] = 'unknown'
                    else:
                        df_processed[col] = 0
                    available_feature_columns.append(col)
    
    # Keep reference columns (not used for modeling but useful for interpretation)
    reference_columns = ['id', 'userid', 'login_timestamp', 'country', 'ip_address']
    
    # Map original column names to reference columns
    column_mapping = {
        'User ID': 'userid',
        'Login Timestamp': 'login_timestamp',
        'Country': 'country',
        'IP Address': 'ip_address'
    }
    
    for orig_col, target_col in column_mapping.items():
        if target_col not in df_processed.columns and orig_col in df_processed.columns:
            df_processed[target_col] = df_processed[orig_col]
    
    available_reference_columns = [col for col in reference_columns if col in df_processed.columns]
    
    # Prepare final dataset
    df_model = df_processed[available_feature_columns + available_reference_columns].copy()
    
    # Handle missing values
    for col in available_feature_columns:
        if df_model[col].dtype.kind in 'if':  # numeric columns
            df_model[col] = df_model[col].fillna(0)
        else:  # categorical columns
            df_model[col] = df_model[col].fillna('unknown')
    
    # Create preprocessing pipeline
    numeric_features = [col for col in available_feature_columns if df_model[col].dtype.kind in 'if' 
                         and col != 'risk_score']  # Keep risk_score unscaled
    categorical_features = [col for col in available_feature_columns if df_model[col].dtype == 'object']
    
    print(f"Numeric features: {numeric_features}")
    print(f"Categorical features: {categorical_features}")
    
    # Create column transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
        remainder='drop'  # Drop the reference columns not used for modeling
    )
    
    # Fit the preprocessor
    preprocessor.fit(df_model)
    
    # Save the preprocessor
    joblib.dump(preprocessor, model_dir / 'rba_preprocessor.pkl')
    
    # Select X (features for model) and ref (reference data)
    X = df_model[available_feature_columns].copy()
    ref = df_model[available_reference_columns].copy()
    
    print(f"✅ Preprocessing complete. Dataset shape: {X.shape}")
    
    return X, ref, preprocessor, available_feature_columns, available_reference_columns

if __name__ == "__main__":
    # Test the preprocessing pipeline with example data
    print("Testing preprocessing pipeline with sample data...")
    
    # Create a sample dataframe with the original column names
    sample_data = {
        'User ID': ['user1', 'user2', 'user3'],
        'Login Timestamp': ['2023-04-20 10:30:00', '2023-04-20 14:45:00', '2023-04-20 20:15:00'],
        'Round-Trip Time [ms]': [120.5, 85.3, 450.2],
        'IP Address': ['192.168.1.1', '10.0.0.5', '203.0.113.4'],
        'Country': ['US', 'GB', 'CA'],
        'User Agent String': [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15',
            'Mozilla/5.0 (iPhone; CPU iPhone OS 14_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Mobile/15E148 Safari/604.1'
        ]
    }
    
    df = pd.DataFrame(sample_data)
    X, ref, preprocessor, feature_columns, reference_columns = preprocess_data(df)
    
    print("\nSample dataframe:")
    print(df.head())
    
    print("\nProcessed features:")
    print(X.head())
    
    print("\nReference data:")
    print(ref.head())
    
    print("\nPreprocessing test successful!")