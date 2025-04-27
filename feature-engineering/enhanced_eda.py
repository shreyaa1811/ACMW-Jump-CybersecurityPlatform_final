#!/usr/bin/env python3

"""
Enhanced Cybersecurity Datasets EDA Script

This script performs exploratory data analysis on cybersecurity datasets accessed directly 
from a remote server via SSH.
"""

import os
import io
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
from pathlib import Path
import warnings
import paramiko
from dotenv import load_dotenv
import time

warnings.filterwarnings('ignore')

# Load environment variables from .env file
load_dotenv()

# SSH connection settings
SSH_HOST = os.getenv('SSH_HOST')
SSH_USER = os.getenv('SSH_USER')
SSH_KEY_PATH = os.path.expanduser(os.getenv('SSH_KEY_PATH'))

# Set styling for better visualizations
plt.style.use('fivethirtyeight')
sns.set_palette('viridis')
plt.rcParams['figure.figsize'] = (14, 8)

# Create output directory for results
output_dir = Path('eda_results')
output_dir.mkdir(exist_ok=True)

# Dataset paths on remote server
DATASET_PATHS = {
    'intrusion_detection': os.getenv('INTRUSION_DETECTION_DATASET'),
    'ai_enhanced_events': os.getenv('AI_ENHANCED_DATASET'), 
    'text_based_detection': os.getenv('TEXT_BASED_DATASET'),
    'rba_dataset': os.getenv('RBA_DATASET')
}

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

# Function to create dataset-specific directories
def create_dataset_dir(dataset_name):
    dataset_dir = output_dir / dataset_name
    dataset_dir.mkdir(exist_ok=True)
    return dataset_dir

# Function to save figures
def save_fig(fig, dataset_dir, filename):
    fig.savefig(dataset_dir / f"{filename}.png", bbox_inches='tight', dpi=300)
    plt.close(fig)

# Function for basic dataset analysis
def analyze_basic_info(df, dataset_name, dataset_dir):
    print(f"\n{'='*50}")
    print(f"DATASET: {dataset_name}")
    print(f"{'='*50}")
    
    # Basic information
    print(f"Shape: {df.shape}")
    print(f"Memory usage: {df.memory_usage().sum() / (1024**2):.2f} MB")
    
    # Save the basic info to a file
    with open(dataset_dir / "basic_info.txt", "w") as f:
        f.write(f"DATASET: {dataset_name}\n")
        f.write(f"Shape: {df.shape}\n")
        f.write(f"Memory usage: {df.memory_usage().sum() / (1024**2):.2f} MB\n\n")
        
        f.write("Data Types:\n")
        for col, dtype in df.dtypes.items():
            f.write(f"{col}: {dtype}\n")
        
        f.write("\nMissing Values:\n")
        missing_data = df.isnull().sum()
        for col, count in missing_data.items():
            percentage = count / len(df) * 100
            f.write(f"{col}: {count} ({percentage:.2f}%)\n")
        
        f.write("\nSample Data (5 rows):\n")
        f.write(df.head(5).to_string())
    
    # Create missing values plot
    missing_values = df.isnull().sum() / len(df) * 100
    missing_values = missing_values[missing_values > 0].sort_values(ascending=False)
    
    if not missing_values.empty:
        fig, ax = plt.subplots(figsize=(12, 6))
        missing_values.plot(kind='bar', ax=ax)
        plt.title(f'Missing Values in {dataset_name}')
        plt.ylabel('Percentage (%)')
        plt.xlabel('Features')
        plt.tight_layout()
        save_fig(fig, dataset_dir, "missing_values")

# Function to analyze numerical features
def analyze_numerical_features(df, dataset_name, dataset_dir):
    # Identify numerical columns
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    if not numerical_cols:
        print("No numerical features found.")
        return
    
    # Save descriptive statistics
    with open(dataset_dir / "numerical_stats.txt", "w") as f:
        f.write(f"Numerical Features Statistics for {dataset_name}\n\n")
        f.write(df[numerical_cols].describe().to_string())
    
    # Generate histograms
    n_cols = min(4, len(numerical_cols))
    n_rows = (len(numerical_cols) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
    axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]
    
    for i, col in enumerate(numerical_cols):
        if i < len(axes):
            sns.histplot(df[col], kde=True, ax=axes[i])
            axes[i].set_title(f'Distribution of {col}')
    
    # Hide unused subplots
    for j in range(len(numerical_cols), len(axes)):
        if j < len(axes):
            fig.delaxes(axes[j])
    
    plt.tight_layout()
    save_fig(fig, dataset_dir, "numerical_distributions")
    
    # Create correlation heatmap
    if len(numerical_cols) > 1:
        corr_matrix = df[numerical_cols].corr()
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", 
                   cmap='coolwarm', center=0, linewidths=.5, ax=ax)
        plt.title(f'Correlation Matrix for {dataset_name}')
        plt.tight_layout()
        save_fig(fig, dataset_dir, "correlation_heatmap")

# Function to analyze categorical features
def analyze_categorical_features(df, dataset_name, dataset_dir):
    # Identify categorical columns (including object and categorical dtype)
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if not categorical_cols:
        print("No categorical features found.")
        return
    
    # Save category counts
    with open(dataset_dir / "categorical_stats.txt", "w") as f:
        f.write(f"Categorical Features Statistics for {dataset_name}\n\n")
        for col in categorical_cols:
            value_counts = df[col].value_counts().head(20)  # Top 20 values
            unique_count = df[col].nunique()
            f.write(f"{col} (Unique values: {unique_count}):\n")
            f.write(f"{value_counts.to_string()}\n\n")
            if unique_count > 20:
                f.write(f"... and {unique_count - 20} more values\n\n")
    
    # Generate bar plots for categorical features (limit to reasonable number of categories)
    for col in categorical_cols:
        unique_values = df[col].nunique()
        
        if unique_values <= 30:  # Only plot if the number of categories is reasonable
            plt.figure(figsize=(14, 6))
            if unique_values <= 10:
                counts = df[col].value_counts().sort_values(ascending=False)
                sns.barplot(x=counts.index, y=counts.values)
            else:
                # Get top 15 categories and group others as "Other"
                top_categories = df[col].value_counts().nlargest(15).index
                df_plot = df.copy()
                df_plot[col] = df_plot[col].apply(lambda x: x if x in top_categories else 'Other')
                counts = df_plot[col].value_counts().sort_values(ascending=False)
                sns.barplot(x=counts.index, y=counts.values)
            
            plt.title(f'Distribution of {col}')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(dataset_dir / f"cat_{col}.png", bbox_inches='tight', dpi=300)
            plt.close()

# Function to analyze the intrusion detection dataset
def analyze_intrusion_detection(ssh_client):
    """Analyze the Cybersecurity Intrusion Detection Dataset"""
    print("\nAnalyzing Cybersecurity Intrusion Detection Dataset...")
    
    # Create dataset directory
    dataset_name = "intrusion_detection"
    dataset_dir = create_dataset_dir(dataset_name)
    
    # Load from remote server
    df = read_dataset_from_server(ssh_client, DATASET_PATHS[dataset_name])
    
    if df is None:
        print(f"❌ Could not load {dataset_name} dataset")
        return
    
    # Handle data type for attack_detected column
    try:
        if 'attack_detected' in df.columns and df['attack_detected'].dtype == 'object':
            print("Converting 'attack_detected' column to numeric values...")
            df['attack_detected'] = pd.to_numeric(df['attack_detected'], errors='coerce')
            df['attack_detected'] = df['attack_detected'].fillna(0).astype(int)
    except Exception as e:
        print(f"Warning: Could not convert attack_detected column: {e}")
    
    # Basic analysis
    analyze_basic_info(df, dataset_name, dataset_dir)
    analyze_numerical_features(df, dataset_name, dataset_dir)
    analyze_categorical_features(df, dataset_name, dataset_dir)
    
    # Specific analysis for this dataset
    try:
        # Attack distribution
        if 'attack_detected' in df.columns:
            plt.figure(figsize=(10, 6))
            attack_counts = df['attack_detected'].value_counts()
            
            plt.pie(attack_counts, labels=['Normal', 'Attack'] if len(attack_counts) == 2 else None,
                autopct='%1.1f%%', startangle=90, colors=['#66b3ff', '#ff6666'])
            plt.title('Distribution of Attack vs Normal Traffic')
            plt.tight_layout()
            plt.savefig(dataset_dir / "attack_distribution.png", bbox_inches='tight', dpi=300)
            plt.close()
        
        # Protocol distribution by attack status
        if 'protocol_type' in df.columns and 'attack_detected' in df.columns:
            plt.figure(figsize=(12, 6))
            # Ensure both columns are properly typed
            protocol_attack = pd.crosstab(df['protocol_type'], 
                                          pd.to_numeric(df['attack_detected'], errors='coerce').fillna(0).astype(int))
            protocol_attack_pct = protocol_attack.div(protocol_attack.sum(axis=1), axis=0) * 100
            
            protocol_attack_pct.plot(kind='bar', stacked=True)
            plt.title('Protocol Distribution by Attack Status')
            plt.ylabel('Percentage')
            plt.xlabel('Protocol Type')
            plt.legend(['Normal', 'Attack'])
            plt.tight_layout()
            plt.savefig(dataset_dir / "protocol_by_attack.png", bbox_inches='tight', dpi=300)
            plt.close()
    except Exception as e:
        print(f"❌ Error in intrusion detection analysis: {e}")
    
    print(f"✅ Analysis complete. Results saved to {dataset_dir}")

# Function to analyze the AI-enhanced dataset
def analyze_ai_enhanced_dataset(ssh_client):
    """Analyze the AI-Enhanced Cybersecurity Events Dataset"""
    print("\nAnalyzing AI-Enhanced Cybersecurity Events Dataset...")
    
    # Create dataset directory
    dataset_name = "ai_enhanced_events"
    dataset_dir = create_dataset_dir(dataset_name)
    
    # Load from remote server
    df = read_dataset_from_server(ssh_client, DATASET_PATHS[dataset_name])
    
    if df is None:
        print(f"❌ Could not load {dataset_name} dataset")
        return
    
    # Basic analysis
    analyze_basic_info(df, dataset_name, dataset_dir)
    analyze_numerical_features(df, dataset_name, dataset_dir)
    analyze_categorical_features(df, dataset_name, dataset_dir)
    
    # Specific analysis for this dataset
    try:
        # Check if specific columns exist before analysis
        if 'Attack Type' in df.columns:
            # Event type distribution
            plt.figure(figsize=(14, 8))
            event_counts = df['Attack Type'].value_counts().head(15)  # Top 15 events
            
            sns.barplot(x=event_counts.values, y=event_counts.index)
            plt.title('Top 15 Attack Types')
            plt.xlabel('Count')
            plt.tight_layout()
            plt.savefig(dataset_dir / "attack_type_distribution.png", bbox_inches='tight', dpi=300)
            plt.close()
        
        # Check for severity or risk columns
        severity_cols = [col for col in df.columns if 'sever' in col.lower() or 'risk' in col.lower()]
        for col in severity_cols:
            plt.figure(figsize=(10, 6))
            df[col].value_counts().plot(kind='pie', autopct='%1.1f%%', colors=sns.color_palette("YlOrRd", df[col].nunique()))
            plt.title(f'Distribution of {col}')
            plt.ylabel('')  # Hide ylabel
            plt.tight_layout()
            plt.savefig(dataset_dir / f"{col.lower()}_distribution.png", bbox_inches='tight', dpi=300)
            plt.close()
    except Exception as e:
        print(f"❌ Error in AI enhanced dataset analysis: {e}")
    
    print(f"✅ Analysis complete. Results saved to {dataset_dir}")

# Function to analyze the text-based dataset
def analyze_text_based_dataset(ssh_client):
    """Analyze the Text-based Cyber Threat Detection dataset"""
    print("\nAnalyzing Text-based Cyber Threat Detection dataset...")
    
    # Create dataset directory
    dataset_name = "text_based_detection"
    dataset_dir = create_dataset_dir(dataset_name)
    
    # Load from remote server
    df = read_dataset_from_server(ssh_client, DATASET_PATHS[dataset_name])
    
    if df is None:
        print(f"❌ Could not load {dataset_name} dataset")
        return
    
    # Basic analysis
    analyze_basic_info(df, dataset_name, dataset_dir)
    analyze_numerical_features(df, dataset_name, dataset_dir)
    analyze_categorical_features(df, dataset_name, dataset_dir)
    
    # Specific analysis for text data
    try:
        if 'text' in df.columns:
            # Text length analysis
            df['text_length'] = df['text'].astype(str).apply(len)
            
            plt.figure(figsize=(12, 6))
            sns.histplot(df['text_length'], kde=True)
            plt.title('Distribution of Text Length')
            plt.xlabel('Text Length (characters)')
            plt.ylabel('Frequency')
            plt.tight_layout()
            plt.savefig(dataset_dir / "text_length_distribution.png", bbox_inches='tight', dpi=300)
            plt.close()
            
            # Save text length statistics
            with open(dataset_dir / "text_stats.txt", "w") as f:
                f.write("Text Length Statistics:\n")
                f.write(df['text_length'].describe().to_string())
                f.write("\n\nSample texts (first 5):\n")
                for i, text in enumerate(df['text'].head(5).values):
                    f.write(f"\n{i+1}. {text[:500]}...\n")  # First 500 chars
        
        # Analyze labels if they exist
        if 'label' in df.columns:
            plt.figure(figsize=(12, 6))
            label_counts = df['label'].value_counts()
            sns.barplot(x=label_counts.index, y=label_counts.values)
            plt.title('Distribution of Labels')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(dataset_dir / "label_distribution.png", bbox_inches='tight', dpi=300)
            plt.close()
    except Exception as e:
        print(f"❌ Error in text-based dataset analysis: {e}")
    
    print(f"✅ Analysis complete. Results saved to {dataset_dir}")

# Function to analyze the login dataset (memory-optimized)
def analyze_login_dataset(ssh_client, sample_size=10000):
    """Analyze the Login Data Set for Risk-Based Authentication"""
    print("\nAnalyzing Login Data Set for Risk-Based Authentication...")
    
    # Create dataset directory
    dataset_name = "rba_dataset"
    dataset_dir = create_dataset_dir(dataset_name)
    
    try:
        # For large dataset, analyze a sample
        print(f"Loading a sample of {sample_size} rows from RBA dataset...")
        df = read_dataset_from_server(ssh_client, DATASET_PATHS[dataset_name], nrows=sample_size)
        
        if df is None:
            print(f"❌ Could not load {dataset_name} dataset")
            return
        
        print(f"Sample created with {len(df)} rows.")
        
        # Now analyze the sample
        analyze_basic_info(df, f"{dataset_name} (Sample)", dataset_dir)
        analyze_numerical_features(df, f"{dataset_name} (Sample)", dataset_dir)
        analyze_categorical_features(df, f"{dataset_name} (Sample)", dataset_dir)
        
        # Specific analysis for this dataset
        try:
            # Login success vs failure
            if 'Login Successful' in df.columns:
                plt.figure(figsize=(10, 6))
                login_counts = df['Login Successful'].value_counts()
                
                labels = ['Failed', 'Successful'] if login_counts.index[0] == False else ['Successful', 'Failed']
                plt.pie(login_counts, labels=labels, autopct='%1.1f%%', 
                       startangle=90, colors=['#ff6666', '#66b3ff'])
                plt.title('Distribution of Successful vs Failed Logins')
                plt.tight_layout()
                plt.savefig(dataset_dir / "login_success_distribution.png", bbox_inches='tight', dpi=300)
                plt.close()
            
            # Geolocation analysis if available
            geo_cols = [col for col in df.columns if col.lower() in ['country', 'region', 'city']]
            for col in geo_cols:
                plt.figure(figsize=(14, 8))
                geo_counts = df[col].value_counts().head(20)  # Top 20 locations
                
                sns.barplot(x=geo_counts.index, y=geo_counts.values)
                plt.title(f'Top {col} Locations')
                plt.xticks(rotation=90)
                plt.tight_layout()
                plt.savefig(dataset_dir / f"{col.lower()}_distribution.png", bbox_inches='tight', dpi=300)
                plt.close()
        except Exception as e:
            print(f"❌ Error in login dataset specific analysis: {e}")
        
        print(f"✅ Analysis complete. Results saved to {dataset_dir}")
            
    except Exception as e:
        print(f"❌ Error analyzing login dataset: {e}")
        import traceback
        traceback.print_exc()

# TODO: Darshitha to implement this function
def feature_exploration(ssh_client):
    """
    Exploratory analysis of features in cybersecurity datasets.
    
    Your task:
    1. Analyze feature distributions for numerical variables
    2. Explore relationships between features and attack indicators
    3. Identify correlated features
    4. Analyze feature importance or relevance
    
    Use EDA techniques to understand the data better.
    """
    print("\n=== TODO: FEATURE EXPLORATION (assigned to Darshitha) ===")
    print("Implement feature exploration for better understanding of the datasets")
    return

# TODO: Sandra to implement this function
def text_data_analysis(ssh_client):
    """
    Exploratory analysis of text data in the cybersecurity datasets.
    
    Your task:
    1. Analyze text length distributions
    2. Explore common terms and phrases in the text
    3. Identify patterns in text related to security incidents
    4. Visualize text characteristics
    
    Use EDA techniques specifically for text data.
    """
    print("\n=== TODO: TEXT DATA ANALYSIS (assigned to Sandra) ===")
    print("Implement text data analysis for the text-based cyber threat detection dataset")
    return

# Generate a summary report
def generate_summary_report():
    print("\nGenerating summary report...")
    
    with open(output_dir / "summary_report.md", "w") as f:
        f.write("# Cybersecurity Datasets EDA Summary Report\n\n")
        f.write(f"*Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
        
        f.write("## Datasets Analyzed\n\n")
        f.write("1. **Cybersecurity Intrusion Detection Dataset**\n")
        f.write("   - Features: Network traffic and user behavior data\n")
        f.write("   - Target: Attack detection\n\n")
        
        f.write("2. **AI-Enhanced Cybersecurity Events Dataset**\n")
        f.write("   - Simulated cybersecurity incidents and events\n\n")
        
        f.write("3. **Text-based Cyber Threat Detection Dataset**\n")
        f.write("   - Text content with cyber threat descriptions\n")
        f.write("   - Includes sender-receiver relationships\n\n")
        
        f.write("4. **Login Data Set for Risk-Based Authentication**\n")
        f.write("   - Synthesized login data with over 33M login attempts\n")
        f.write("   - Features: IP, User-Agent, RTT, geographic info\n")
        f.write("   - Note: Analyzed using a sample due to its large size\n\n")
        
        f.write("## Key Findings\n\n")
        f.write("For detailed analysis results, see the respective dataset directories.\n\n")
        
        f.write("## Assignment Status\n\n")
        f.write("### Darshitha (ML Engineer)\n")
        f.write("- ✅ Basic feature analysis\n")
        f.write("- 🔜 TODO: Advanced feature exploration\n\n")
        
        f.write("### Sandra (LLM Engineer)\n")
        f.write("- ✅ Basic text analysis\n")
        f.write("- 🔜 TODO: Advanced text data analysis\n\n")
        
        f.write("## Next Steps\n\n")
        f.write("1. Complete the assigned EDA tasks for each team member\n")
        f.write("2. Use insights from EDA to guide further analysis\n")
        f.write("3. Identify important patterns in the data\n")
    
    print(f"✅ Summary report generated at {output_dir / 'summary_report.md'}")

# Main function
def main():
    print("Starting Exploratory Data Analysis for Cybersecurity Datasets...")
    start_time = time.time()
    
    # Connect to SSH server
    ssh_client = connect_to_server()
    if ssh_client is None:
        print("❌ Cannot continue without SSH connection")
        return
    
    try:
        # Run each dataset analysis
        analyze_intrusion_detection(ssh_client)
        analyze_ai_enhanced_dataset(ssh_client)
        analyze_text_based_dataset(ssh_client)
        analyze_login_dataset(ssh_client)
        
        # TODO EDA tasks
        feature_exploration(ssh_client)  # Darshitha
        text_data_analysis(ssh_client)   # Sandra
        
        # Generate summary report
        generate_summary_report()
        
    finally:
        # Close SSH connection
        if ssh_client:
            ssh_client.close()
            print("SSH connection closed")
    
    end_time = time.time()
    print(f"\n✅ EDA completed successfully in {end_time - start_time:.2f} seconds! Results saved to {output_dir.absolute()}")

if __name__ == "__main__":
    main()