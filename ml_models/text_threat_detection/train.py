#!/usr/bin/env python3

"""
Training script for Text Threat Detection model

This script:
1. Connects to MongoDB to load the text-based threat detection dataset
2. Preprocesses the text data and prepares it for training
3. Trains a machine learning model for threat classification
4. Evaluates the model and saves it for later use
"""

import os
import sys
import pandas as pd
import numpy as np
import pymongo
from pymongo import MongoClient
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
import argparse
from pathlib import Path
import joblib
import time
import paramiko
import io
from urllib.parse import quote_plus
import json

# Add the parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from text_threat_detection.model import TextThreatDetectionModel

# Load environment variables
load_dotenv()

# MongoDB config from environment variables
MONGODB_CONFIG = {
    'uri': f"mongodb://{quote_plus(os.getenv('DEFAULT_MONGODB_USER'))}:{quote_plus(os.getenv('DEFAULT_MONGODB_PASSWORD'))}@{os.getenv('MONGODB_HOST')}:27017/{os.getenv('DEFAULT_MONGODB_DB')}?authSource={os.getenv('DEFAULT_MONGODB_DB')}",
    'database': os.getenv('DEFAULT_MONGODB_DB')
}

# SSH connection settings
SSH_HOST = os.getenv('SSH_HOST')
SSH_USER = os.getenv('SSH_USER')
SSH_KEY_PATH = os.path.expanduser(os.getenv('SSH_KEY_PATH'))

# Dataset path on remote server
TEXT_DATASET = os.getenv('TEXT_BASED_DATASET')

# Create directory for model artifacts
model_dir = Path('model_artifacts')
model_dir.mkdir(exist_ok=True)

def connect_to_mongodb():
    """Connect to MongoDB database"""
    try:
        client = MongoClient(MONGODB_CONFIG['uri'], serverSelectionTimeoutMS=5000)
        db = client[MONGODB_CONFIG['database']]
        
        # Test the connection
        client.admin.command('ping')
        print(f"✅ Connected to MongoDB: {MONGODB_CONFIG['database']}")
        return db
    except Exception as e:
        print(f"❌ Error connecting to MongoDB: {e}")
        return None

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

def read_dataset_from_server(ssh_client):
    """Read the dataset from the remote server."""
    try:
        if ssh_client is None:
            return None
        
        # Get expanded path
        stdin, stdout, stderr = ssh_client.exec_command(f"echo {TEXT_DATASET}")
        expanded_path = stdout.read().decode('utf-8').strip()
        
        print(f"Reading dataset from {expanded_path}")
        
        # Get file size first to check if it's manageable
        stdin, stdout, stderr = ssh_client.exec_command(f"du -h {expanded_path}")
        file_size = stdout.read().decode('utf-8').strip()
        print(f"File size: {file_size}")
        
        # Read the dataset
        command = f"cat {expanded_path}"
        stdin, stdout, stderr = ssh_client.exec_command(command)
        
        # Read into pandas dataframe
        df = pd.read_csv(stdout)
        return df
    except Exception as e:
        print(f"❌ Error reading dataset: {e}")
        return None

def load_data_from_mongodb():
    """Load text threat data from MongoDB"""
    mongodb_db = connect_to_mongodb()
    
    if mongodb_db is None:
        print("Failed to connect to MongoDB, trying SSH connection instead...")
        return None
    
    try:
        # Query the text_based_detection collection
        cursor = mongodb_db.text_based_detection.find()
        df = pd.DataFrame(list(cursor))
        
        print(f"Loaded {len(df)} records from MongoDB")
        return df
    except Exception as e:
        print(f"❌ Error loading data from MongoDB: {e}")
        return None

def identify_text_column(df):
    """Identify which column contains the text content"""
    potential_text_columns = ['text', 'content', 'message', 'body', 'description']
    text_col = None
    
    # Check for columns with 'text' in the name
    text_cols = [col for col in df.columns if 'text' in col.lower()]
    if text_cols:
        text_col = text_cols[0]
    
    # If no text column found, look for potential content columns
    if text_col is None:
        for col in potential_text_columns:
            if col in df.columns:
                text_col = col
                break
    
    # If still not found, use the column with the longest average string length
    if text_col is None:
        string_cols = df.select_dtypes(include=['object']).columns
        avg_lengths = {col: df[col].astype(str).apply(len).mean() for col in string_cols}
        
        if avg_lengths:
            text_col = max(avg_lengths.items(), key=lambda x: x[1])[0]
    
    return text_col

def identify_label_column(df):
    """Identify which column contains the threat label/category"""
    potential_label_columns = ['label', 'category', 'threat', 'threat_type', 'class', 'type']
    label_col = None
    
    # Check for columns with 'label' or 'category' in the name
    label_cols = [col for col in df.columns if any(term in col.lower() for term in potential_label_columns)]
    if label_cols:
        label_col = label_cols[0]
    
    # If no label column found, look for columns with a small number of unique values
    if label_col is None:
        object_cols = df.select_dtypes(include=['object']).columns
        unique_counts = {col: df[col].nunique() for col in object_cols}
        
        # Filter for columns with 2-20 unique values that aren't likely ID columns
        label_candidates = {col: count for col, count in unique_counts.items() 
                           if 2 <= count <= 20 and not col.lower().endswith('id')}
        
        if label_candidates:
            label_col = min(label_candidates.items(), key=lambda x: x[1])[0]
    
    return label_col

def preprocess_data(df):
    """Preprocess the data for training"""
    print("Preprocessing data...")
    
    # Identify text and label columns
    text_col = identify_text_column(df)
    label_col = identify_label_column(df)
    
    if text_col is None or label_col is None:
        print(f"❌ Could not identify text column ({text_col}) or label column ({label_col})")
        return None, None, None, None
    
    print(f"Using '{text_col}' as text column and '{label_col}' as label column")
    
    # Create a model instance to use its preprocessing method
    model = TextThreatDetectionModel()
    
    # Preprocess all text
    print("Preprocessing text data...")
    df['processed_text'] = df[text_col].apply(model.preprocess_text)
    
    # Remove rows with empty processed text
    df = df[df['processed_text'].str.strip() != '']
    
    # Make sure labels are strings
    df[label_col] = df[label_col].astype(str)
    
    # Examine the distribution of labels
    label_counts = df[label_col].value_counts()
    print(f"Label distribution:\n{label_counts.head(10)}")
    
    # Remove labels with very few examples (less than 5)
    min_examples = 5
    valid_labels = label_counts[label_counts >= min_examples].index
    df = df[df[label_col].isin(valid_labels)]
    
    # Split into train and test sets
    X = df['processed_text']
    y = df[label_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if len(y.unique()) > 1 else None
    )
    
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    
    # Save the test data for later evaluation
    test_data = pd.DataFrame({
        'text': df[text_col][X_test.index],
        'processed_text': X_test,
        'label': y_test
    })
    test_data.to_csv(model_dir / 'text_threat_test_data.csv', index=False)
    
    return X_train, X_test, y_train, y_test

def main(args):
    """Main function to train and evaluate the model"""
    start_time = time.time()
    
    print("\n===== Text Threat Detection Model Training =====")
    
    # Load data
    df = None
    
    # Try MongoDB first
    if not args.skip_mongodb:
        df = load_data_from_mongodb()
    
    # If MongoDB failed, try SSH
    if df is None or df.empty:
        print("MongoDB data not available or empty, trying SSH...")
        ssh_client = connect_to_server()
        
        if ssh_client:
            try:
                df = read_dataset_from_server(ssh_client)
            finally:
                ssh_client.close()
                print("SSH connection closed")
    
    if df is None or df.empty:
        print("❌ Failed to load data from both MongoDB and SSH")
        return
    
    print(f"Dataset loaded with {len(df)} rows and {len(df.columns)} columns")
    print(f"Columns: {df.columns.tolist()}")
    
    # Preprocess data
    X_train, X_test, y_train, y_test = preprocess_data(df)
    
    if X_train is None:
        print("❌ Failed to preprocess data")
        return
    
    # Train model
    model = TextThreatDetectionModel()
    model.train(X_train, y_train, model_type=args.model_type)
    
    # Evaluate model
    metrics = model.evaluate(X_test, y_test)
    
    # Sample predictions
    sample_texts = X_test[:5].tolist()
    predictions = model.predict(sample_texts)
    
    print("\nSample predictions:")
    for pred in predictions:
        print(f"Text: {pred['text']}")
        print(f"Predicted threat: {pred['predicted_threat']}")
        print(f"Confidence: {pred['confidence']}")
        print()
    
    # Print training summary
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"\n===== Training Summary =====")
    print(f"Model type: {args.model_type}")
    print(f"Training data size: {len(X_train)}")
    print(f"Test data size: {len(X_test)}")
    print(f"Number of threat categories: {len(model.label_encoder.classes_)}")
    print(f"Training duration: {duration:.2f} seconds")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    
    # Save model info
    model_info = {
        'model_type': args.model_type,
        'training_date': time.strftime('%Y-%m-%d %H:%M:%S'),
        'training_duration': duration,
        'training_size': len(X_train),
        'test_size': len(X_test),
        'num_categories': len(model.label_encoder.classes_),
        'categories': model.label_encoder.classes_.tolist(),
        'metrics': metrics
    }
    
    with open(model_dir / 'text_threat_model_info.json', 'w') as f:
        json.dump(model_info, f, indent=4)
    
    print(f"\n✅ Model training complete! Model artifacts saved to {model_dir.absolute()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Text Threat Detection Model")
    parser.add_argument("--model-type", choices=['rf', 'lr'], default='rf',
                       help="Model type: Random Forest (rf) or Logistic Regression (lr)")
    parser.add_argument("--skip-mongodb", action="store_true",
                       help="Skip MongoDB and use SSH connection directly")
    
    args = parser.parse_args()
    main(args)