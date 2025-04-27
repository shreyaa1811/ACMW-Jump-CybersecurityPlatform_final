#!/usr/bin/env python3

"""
Risk-Based Authentication (RBA) Anomaly Detection Model
- Isolation Forest for unsupervised anomaly detection
- Feature importance analysis
- Risk scoring
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import joblib
from pathlib import Path
import json
from dotenv import load_dotenv
import paramiko
from tqdm import tqdm
import gc
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import warnings
warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv()

# SSH connection settings
SSH_HOST = os.getenv('SSH_HOST')
SSH_USER = os.getenv('SSH_USER')
SSH_KEY_PATH = os.path.expanduser(os.getenv('SSH_KEY_PATH'))

# Dataset path on remote server
RBA_DATASET = os.getenv('RBA_DATASET')

# Create directory for model artifacts
model_dir = Path('model_artifacts')
model_dir.mkdir(exist_ok=True)

# Helper function to convert NumPy types to Python types for JSON serialization
def convert_to_serializable(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    else:
        return obj

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

class RBADetectionModel:
    """Isolation Forest model for RBA anomaly detection"""
    
    def __init__(self, model_path=None):
        """Initialize the model or load pretrained model"""
        if model_path:
            self.model = joblib.load(model_path)
            self.preprocessor = joblib.load(model_dir / 'rba_preprocessor.pkl')
            print(f"✅ Loaded pretrained model from {model_path}")
        else:
            self.model = None
            self.preprocessor = None
    
    def train(self, X, contamination=0.05, n_estimators=100, random_state=42):
        """Train the Isolation Forest model for anomaly detection"""
        print("Training RBA anomaly detection model...")
        
        # Transform the data if preprocessor is available
        if self.preprocessor is not None:
            print(f"Transforming data with preprocessor...")
            X_transformed = self.preprocessor.transform(X)
            print(f"Data transformation complete.")
        else:
            # Handle categorical variables manually
            X_transformed = X
        
        # Train Isolation Forest model
        print(f"Training Isolation Forest with {n_estimators} estimators...")
        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
            verbose=1,
            n_jobs=-1  # Use all available cores
        )
        
        self.model.fit(X_transformed)
        
        print("✅ Model training complete")
        
        # Save the model
        joblib.dump(self.model, model_dir / 'rba_detection_model.pkl')
        print(f"✅ Model saved to {model_dir / 'rba_detection_model.pkl'}")
        
        return self.model
    
    def evaluate(self, X, feature_names=None, reference_data=None):
        """Evaluate the anomaly detection model"""
        if self.model is None:
            raise Exception("Model not trained yet")
        
        print("Evaluating model performance...")
        
        # Transform the data if preprocessor is available
        if self.preprocessor is not None:
            X_transformed = self.preprocessor.transform(X)
        else:
            X_transformed = X
        
        # Get anomaly scores (-1 for anomalies, 1 for normal)
        raw_scores = self.model.predict(X_transformed)
        
        # Get decision function (distance from separating hyperplane)
        # Lower (more negative) values represent stronger anomalies
        decision_scores = self.model.decision_function(X_transformed)
        
        # Convert raw scores to binary labels (1 for anomalies, 0 for normal)
        anomaly_labels = np.where(raw_scores == -1, 1, 0)
        
        # Calculate anomaly probability (normalized decision scores)
        # Transform decision scores to 0-1 range where 1 is strongest anomaly
        min_score = decision_scores.min()
        max_score = decision_scores.max()
        score_range = max_score - min_score
        
        if score_range > 0:
            anomaly_probs = 1 - ((decision_scores - min_score) / score_range)
        else:
            anomaly_probs = np.zeros_like(decision_scores)
        
        # Calculate metrics
        anomaly_count = sum(anomaly_labels)
        anomaly_rate = anomaly_count / len(anomaly_labels)
        
        print(f"Detected anomalies: {anomaly_count} ({anomaly_rate:.2%})")
        
        # Create results dataframe
        results_df = pd.DataFrame({
            'anomaly_label': anomaly_labels,
            'anomaly_score': decision_scores,
            'anomaly_probability': anomaly_probs
        })
        
        # Add reference data if available
        if reference_data is not None:
            for col in reference_data.columns:
                results_df[col] = reference_data[col].values
        
        # Add original features if available
        if feature_names is not None:
            for col in feature_names:
                if col in X.columns:
                    results_df[col] = X[col].values
        
        # Save results
        results_df.to_csv(model_dir / 'rba_anomaly_results.csv', index=False)
        
        # Plot anomaly score distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(anomaly_probs, bins=50)
        plt.title('Anomaly Probability Distribution')
        plt.xlabel('Anomaly Probability')
        plt.ylabel('Frequency')
        plt.savefig(model_dir / 'rba_anomaly_distribution.png')
        
        # Plot top anomalies by country
        if 'Country' in results_df.columns:
            top_anomaly_countries = results_df[results_df['anomaly_label'] == 1]['Country'].value_counts().head(10)
            plt.figure(figsize=(12, 6))
            sns.barplot(x=top_anomaly_countries.index, y=top_anomaly_countries.values)
            plt.title('Top Countries with Anomalous Logins')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(model_dir / 'rba_anomaly_countries.png')
        
        # Plot anomaly rate by hour of day
        if 'hour' in results_df.columns:
            hour_anomaly_rate = results_df.groupby('hour')['anomaly_label'].mean().reset_index()
            plt.figure(figsize=(12, 6))
            sns.lineplot(x='hour', y='anomaly_label', data=hour_anomaly_rate, marker='o')
            plt.title('Anomaly Rate by Hour of Day')
            plt.xlabel('Hour of Day')
            plt.ylabel('Anomaly Rate')
            plt.xticks(range(0, 24))
            plt.savefig(model_dir / 'rba_anomaly_hours.png')
        
        # Optional: Try clustering anomalies to find patterns
        if anomaly_count > 10:
            try:
                # Use only numeric columns for clustering
                numeric_cols = [col for col in results_df.columns if results_df[col].dtype.kind in 'if']
                anomaly_features = results_df[results_df['anomaly_label'] == 1][numeric_cols].copy()
                
                # Fill any NaN values
                anomaly_features = anomaly_features.fillna(0)
                
                # Find optimal number of clusters
                max_clusters = min(5, anomaly_count - 1)
                silhouette_scores = []
                
                for n_clusters in range(2, max_clusters + 1):
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                    cluster_labels = kmeans.fit_predict(anomaly_features)
                    silhouette_avg = silhouette_score(anomaly_features, cluster_labels)
                    silhouette_scores.append(silhouette_avg)
                
                best_n_clusters = np.argmax(silhouette_scores) + 2  # +2 because we start from 2
                
                # Cluster the anomalies
                kmeans = KMeans(n_clusters=best_n_clusters, random_state=42, n_init=10)
                results_df.loc[results_df['anomaly_label'] == 1, 'anomaly_cluster'] = kmeans.fit_predict(anomaly_features)
                results_df['anomaly_cluster'] = results_df['anomaly_cluster'].fillna(-1).astype(int)
                
                # Save clustered results
                results_df.to_csv(model_dir / 'rba_anomaly_results_clustered.csv', index=False)
                
                # Analyze clusters
                cluster_analysis = results_df[results_df['anomaly_cluster'] != -1].groupby('anomaly_cluster').agg({
                    'anomaly_probability': 'mean',
                    'anomaly_score': 'mean',
                    'risk_score': 'mean' if 'risk_score' in results_df.columns else 'count',
                    'User ID': 'count' if 'User ID' in results_df.columns else 'count'
                }).rename(columns={'User ID': 'count'})
                
                cluster_analysis.to_csv(model_dir / 'rba_anomaly_clusters.csv')
                
                print(f"Identified {best_n_clusters} distinct anomaly patterns")
            except Exception as e:
                print(f"Error during anomaly clustering: {e}")
        
        # Save evaluation metrics (convert NumPy types to Python types)
        metrics = {
            'total_logins': int(len(anomaly_labels)),
            'anomaly_count': int(anomaly_count),
            'anomaly_rate': float(anomaly_rate),
            'avg_anomaly_probability': float(anomaly_probs.mean()),
            'high_risk_count': int(sum(anomaly_probs > 0.8))
        }
        
        with open(model_dir / 'rba_metrics.json', 'w') as f:
            json.dump(metrics, f)
        
        print(f"✅ Evaluation complete. Results saved to {model_dir}")
        return results_df, metrics
    
    def predict(self, X):
        """Predict anomalies in new data"""
        if self.model is None:
            raise Exception("Model not trained yet")
        
        # Transform the data if preprocessor is available
        if self.preprocessor is not None:
            X_transformed = self.preprocessor.transform(X)
        else:
            X_transformed = X
        
        # Get anomaly predictions (-1 for anomalies, 1 for normal)
        raw_scores = self.model.predict(X_transformed)
        
        # Get decision scores
        decision_scores = self.model.decision_function(X_transformed)
        
        # Convert raw scores to binary labels (1 for anomalies, 0 for normal)
        anomaly_labels = np.where(raw_scores == -1, 1, 0)
        
        # Calculate anomaly probability (normalized decision scores)
        min_score = decision_scores.min()
        max_score = decision_scores.max()
        score_range = max_score - min_score
        
        if score_range > 0:
            anomaly_probs = 1 - ((decision_scores - min_score) / score_range)
        else:
            anomaly_probs = np.zeros_like(decision_scores)
        
        return anomaly_labels, anomaly_probs, decision_scores

def extract_user_agent_features(df):
    """Extract features from user agent string"""
    # Create a copy to avoid modifying the original
    df_processed = df.copy()
    
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
    df_processed['device_type'] = df_processed['User Agent String'].apply(get_device_type)
    df_processed['browser'] = df_processed['User Agent String'].apply(get_browser)
    df_processed['os'] = df_processed['User Agent String'].apply(get_os)
    
    return df_processed

def extract_time_features(df):
    """Extract temporal features from login timestamp"""
    # Create a copy to avoid modifying the original
    df_processed = df.copy()
    
    # Convert to datetime if needed
    if 'Login Timestamp' in df_processed.columns:
        try:
            df_processed['Login Timestamp'] = pd.to_datetime(df_processed['Login Timestamp'], errors='coerce')
        except:
            print("Warning: Could not convert Login Timestamp to datetime")
    
    # Extract basic time components if datetime column exists
    if 'Login Timestamp' in df_processed.columns and pd.api.types.is_datetime64_any_dtype(df_processed['Login Timestamp']):
        df_processed['hour'] = df_processed['Login Timestamp'].dt.hour
        df_processed['day'] = df_processed['Login Timestamp'].dt.day
        df_processed['day_of_week'] = df_processed['Login Timestamp'].dt.dayofweek
        df_processed['month'] = df_processed['Login Timestamp'].dt.month
        df_processed['year'] = df_processed['Login Timestamp'].dt.year
        
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
    else:
        # If no timestamp, create placeholder columns
        for col in ['hour', 'day', 'day_of_week', 'month', 'year']:
            df_processed[col] = np.nan
        df_processed['is_weekend'] = np.nan
        df_processed['time_category'] = 'unknown'
        df_processed['is_business_hours'] = np.nan
    
    return df_processed

def extract_ip_features(df):
    """Extract features from IP address"""
    # Create a copy to avoid modifying the original
    df_processed = df.copy()
    
    import ipaddress
    
    # IP version
    def get_ip_version(ip):
        if pd.isna(ip):
            return None
        try:
            return ipaddress.ip_address(ip).version
        except:
            return None
    
    # Private IP indicator
    def is_private_ip(ip):
        if pd.isna(ip):
            return None
        try:
            return int(ipaddress.ip_address(ip).is_private)
        except:
            return None
    
    # Apply functions if IP column exists
    if 'IP Address' in df_processed.columns:
        df_processed['ip_version'] = df_processed['IP Address'].apply(get_ip_version)
        df_processed['is_private_ip'] = df_processed['IP Address'].apply(is_private_ip)
    else:
        df_processed['ip_version'] = np.nan
        df_processed['is_private_ip'] = np.nan
    
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

def preprocess_chunk(df):
    """Preprocess a chunk of RBA data"""
    try:
        # Extract features
        df_processed = extract_user_agent_features(df)
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

def train_and_evaluate():
    """Main function to train and evaluate the model using SSH connection"""
    try:
        # Connect to server
        ssh_client = connect_to_server()
        if ssh_client is None:
            print("❌ Cannot continue without SSH connection")
            return None, None, None
        
        try:
            # Use a smaller chunk size for initial preprocessing and model setup
            initial_chunk_size = 10000
            print(f"Reading initial chunk of {initial_chunk_size} rows to set up the model...")
            initial_df = read_dataset_from_server(ssh_client, RBA_DATASET, nrows=initial_chunk_size)
            
            if initial_df is None:
                print("❌ Failed to read initial data chunk")
                return None, None, None
            
            print(f"Preprocessing initial chunk...")
            X_initial, ref_initial, feature_columns, reference_columns = preprocess_chunk(initial_df)
            
            if X_initial is None:
                print("❌ Failed to preprocess initial data chunk")
                return None, None, None
            
            # Create and fit preprocessor on initial data
            print("Creating and fitting preprocessor...")
            numeric_features = [col for col in feature_columns if X_initial[col].dtype.kind in 'if' and col != 'risk_score']
            categorical_features = [col for col in feature_columns if X_initial[col].dtype == 'object']
            
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', StandardScaler(), numeric_features),
                    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
                ],
                remainder='drop'
            )
            
            # Fit on initial chunk
            preprocessor.fit(X_initial)
            
            # Save the preprocessor
            joblib.dump(preprocessor, model_dir / 'rba_preprocessor.pkl')
            print(f"✅ Preprocessor saved to {model_dir / 'rba_preprocessor.pkl'}")
            
            # Initialize model
            model = RBADetectionModel()
            model.preprocessor = preprocessor
            
            # Train model on initial chunk
            model.train(X_initial, contamination=0.05, n_estimators=100)
            
            # Now process larger chunks for continued training
            processing_chunk_size = 50000
            print(f"Now processing the full dataset in chunks of {processing_chunk_size} rows...")
            
            # Get chunks from the dataset
            chunks = read_dataset_from_server(ssh_client, RBA_DATASET, chunksize=processing_chunk_size)
            
            if chunks is None:
                print("❌ Failed to get data chunks")
                return model, None, None
            
            # Process each chunk
            total_processed = initial_chunk_size
            anomaly_counts = []
            
            # Try to estimate total rows by getting file size
            try:
                stdin, stdout, stderr = ssh_client.exec_command(f"wc -l {RBA_DATASET}")
                total_lines = int(stdout.read().decode('utf-8').strip().split()[0])
                print(f"Total lines in dataset: {total_lines:,}")
                # Create progress bar
                pbar = tqdm(total=total_lines, desc="Processing dataset")
                pbar.update(initial_chunk_size)
            except:
                print("Could not determine total dataset size, using incremental counter")
                pbar = None
            
            for i, chunk in enumerate(chunks):
                # Skip first chunk equivalent to initial data
                if i == 0 and total_processed > 0:
                    # Skip only if we've already processed some data
                    if pbar:
                        pbar.update(min(processing_chunk_size, len(chunk)))
                    continue
                
                chunk_num = i + 1
                print(f"\nProcessing chunk {chunk_num}")
                
                # Preprocess chunk
                X_chunk, ref_chunk, _, _ = preprocess_chunk(chunk)
                
                if X_chunk is None:
                    print(f"⚠️ Skipping chunk {chunk_num} due to processing error")
                    continue
                
                # Evaluate chunk with current model
                try:
                    # Transform chunk data
                    X_transformed = preprocessor.transform(X_chunk)
                    
                    # Get predictions
                    raw_scores = model.model.predict(X_transformed)
                    anomaly_labels = np.where(raw_scores == -1, 1, 0)
                    anomaly_count = sum(anomaly_labels)
                    anomaly_rate = anomaly_count / len(anomaly_labels)
                    
                    # Track anomalies
                    anomaly_counts.append(int(anomaly_count))  # Convert to regular Python int
                    
                    print(f"Chunk {chunk_num}: {len(chunk)} records, {anomaly_count} anomalies ({anomaly_rate:.2%})")
                    
                    # Add to total processed
                    total_processed += len(chunk)
                    if pbar:
                        pbar.update(len(chunk))
                    else:
                        print(f"Total processed: {total_processed:,}")
                    
                except Exception as e:
                    print(f"Error evaluating chunk {chunk_num}: {e}")
                    import traceback
                    traceback.print_exc()
            
            if pbar:
                pbar.close()
            
            # After processing all chunks, run a final evaluation on the initial chunk
            # This gives us visualization outputs
            results, metrics = model.evaluate(X_initial, feature_columns, ref_initial)
            
            # Update metrics with total processed
            metrics['total_processed'] = int(total_processed)
            metrics['chunks_processed'] = i + 1
            metrics['anomalies_by_chunk'] = anomaly_counts
            
            # Convert all NumPy types to Python types for JSON serialization
            metrics = convert_to_serializable(metrics)
            
            # Save updated metrics
            with open(model_dir / 'rba_full_metrics.json', 'w') as f:
                json.dump(metrics, f)
            
            print(f"\n✅ Full dataset processing complete! Processed {total_processed:,} records across {i+1} chunks")
            return model, results, metrics
            
        finally:
            # Close SSH connection
            if ssh_client:
                ssh_client.close()
                print("SSH connection closed")
    
    except Exception as e:
        print(f"❌ Error in train_and_evaluate: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

if __name__ == "__main__":
    # Train and evaluate model
    model, results, metrics = train_and_evaluate()
    if metrics:
        print("Final metrics:", metrics)