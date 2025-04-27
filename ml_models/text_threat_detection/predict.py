#!/usr/bin/env python3

"""
Prediction script for Text Threat Detection model

This script:
1. Loads a trained text threat detection model
2. Takes new text as input (from file, database, or command line)
3. Analyzes the text to detect security threats
4. Provides detailed threat analysis with confidence scores and extracted keywords
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib
import argparse
from pathlib import Path
import json
import pymongo
from pymongo import MongoClient
from dotenv import load_dotenv
from urllib.parse import quote_plus
import datetime
import time

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

# Directory for model artifacts
model_dir = Path('model_artifacts')

def load_model():
    """Load the trained model and required components"""
    model_path = model_dir / 'text_threat_model.pkl'
    vectorizer_path = model_dir / 'text_threat_vectorizer.pkl'
    label_encoder_path = model_dir / 'text_threat_label_encoder.pkl'
    
    if not model_path.exists() or not vectorizer_path.exists() or not label_encoder_path.exists():
        raise FileNotFoundError(
            f"Model files not found in {model_dir}. "
            "Please train the model first using train.py."
        )
    
    model = TextThreatDetectionModel(
        model_path=model_path,
        vectorizer_path=vectorizer_path,
        label_encoder_path=label_encoder_path
    )
    
    return model

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

def load_texts_from_mongodb(limit=100):
    """Load recent texts from MongoDB for analysis"""
    db = connect_to_mongodb()
    
    if db is None:
        return []
    
    try:
        # Identify the text collection
        if 'text_based_detection' in db.list_collection_names():
            collection = db.text_based_detection
        else:
            print("❌ text_based_detection collection not found")
            return []
        
        # Identify which field has the text content
        sample_doc = collection.find_one()
        if sample_doc is None:
            print("❌ No documents found in collection")
            return []
        
        # Look for likely text fields
        text_fields = [field for field in sample_doc.keys() 
                      if isinstance(sample_doc[field], str) and len(sample_doc[field]) > 50
                      and field not in ['_id']]
        
        if not text_fields:
            print("❌ No suitable text fields found in documents")
            return []
        
        text_field = text_fields[0]
        print(f"Using field '{text_field}' for text content")
        
        # Query for recent documents
        documents = list(collection.find().sort([('_id', -1)]).limit(limit))
        
        texts = []
        for doc in documents:
            if text_field in doc and doc[text_field]:
                texts.append(doc[text_field])
        
        print(f"Loaded {len(texts)} texts from MongoDB")
        return texts
    
    except Exception as e:
        print(f"❌ Error loading texts from MongoDB: {e}")
        return []

def load_texts_from_file(file_path):
    """Load texts from a file (one text per line or CSV)"""
    file_path = Path(file_path)
    
    if not file_path.exists():
        print(f"❌ File not found: {file_path}")
        return []
    
    texts = []
    
    try:
        # Try to read as CSV first
        if file_path.suffix.lower() == '.csv':
            df = pd.read_csv(file_path)
            
            # Try to identify the text column
            text_cols = [col for col in df.columns if 'text' in col.lower()]
            if text_cols:
                text_col = text_cols[0]
            else:
                # Use the column with the longest average string length
                string_cols = df.select_dtypes(include=['object']).columns
                avg_lengths = {col: df[col].astype(str).apply(len).mean() for col in string_cols}
                
                if avg_lengths:
                    text_col = max(avg_lengths.items(), key=lambda x: x[1])[0]
                else:
                    # Fall back to first column
                    text_col = df.columns[0]
            
            print(f"Using column '{text_col}' from CSV file")
            texts = df[text_col].tolist()
        else:
            # Read as plain text file, one text per line
            with open(file_path, 'r', encoding='utf-8') as f:
                texts = [line.strip() for line in f if line.strip()]
        
        print(f"Loaded {len(texts)} texts from {file_path}")
        return texts
    
    except Exception as e:
        print(f"❌ Error loading texts from file: {e}")
        return []

def save_results(results, output_file=None):
    """Save analysis results to a file"""
    if not results:
        return
    
    if output_file:
        output_path = Path(output_file)
        
        # Create output directory if needed
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save as JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4)
        
        print(f"✅ Results saved to {output_path}")
    
    # Save to MongoDB
    save_results_to_mongodb(results)

def save_results_to_mongodb(results):
    """Save analysis results to MongoDB"""
    if not results:
        return
    
    db = connect_to_mongodb()
    
    if db is None:
        return
    
    try:
        # Check if the collection exists, create if it doesn't
        if 'threat_analysis_results' not in db.list_collection_names():
            db.create_collection('threat_analysis_results')
        
        # Add timestamp to each result
        timestamp = datetime.datetime.now()
        for result in results:
            result['analysis_timestamp'] = timestamp
        
        # Insert results
        db.threat_analysis_results.insert_many(results)
        
        print(f"✅ Results saved to MongoDB: threat_analysis_results collection")
    except Exception as e:
        print(f"❌ Error saving results to MongoDB: {e}")

def analyze_text(model, text, extract_keywords=True):
    """
    Analyze a single text for security threats
    
    Args:
        model: Trained TextThreatDetectionModel
        text: Text to analyze
        extract_keywords: Whether to extract keywords from the text
        
    Returns:
        Dictionary with threat analysis results
    """
    # Make prediction
    prediction = model.predict([text])[0]
    
    # Extract keywords if requested
    keywords = None
    if extract_keywords:
        keywords = model.extract_keywords(text)
    
    # Create result dictionary
    result = {
        'text': text[:100] + '...' if len(text) > 100 else text,
        'full_text': text,
        'predicted_threat': prediction['predicted_threat'],
        'confidence': prediction['confidence']
    }
    
    # Add keywords if extracted
    if keywords:
        result['keywords'] = [
            {'word': word, 'score': float(score)} for word, score in keywords
        ]
    
    # Add security risk assessment
    risk_level = 'Low'
    if prediction['confidence'] and prediction['confidence'] > 0.8:
        risk_level = 'High'
    elif prediction['confidence'] and prediction['confidence'] > 0.5:
        risk_level = 'Medium'
    
    result['risk_level'] = risk_level
    
    return result

def main(args):
    """Main function to run text threat detection"""
    print("\n===== Text Threat Detection Analysis =====")
    
    try:
        # Load the model
        model = load_model()
        print("✅ Model loaded successfully")
        
        # Get the text to analyze
        texts = []
        
        # From MongoDB
        if args.mongodb:
            texts = load_texts_from_mongodb(limit=args.limit)
        
        # From file
        elif args.file:
            texts = load_texts_from_file(args.file)
        
        # From command line
        elif args.text:
            texts = [args.text]
        
        # If no texts were provided
        if not texts:
            print("❌ No texts to analyze. Please provide text via --text, --file, or --mongodb.")
            return
        
        # Analyze the texts
        print(f"\nAnalyzing {len(texts)} text(s) for threats...")
        
        results = []
        for i, text in enumerate(texts):
            if i % 10 == 0 and i > 0:
                print(f"Processed {i}/{len(texts)} texts...")
            
            result = analyze_text(model, text, extract_keywords=args.keywords)
            results.append(result)
        
        # Display results
        print("\nAnalysis Results:")
        for i, result in enumerate(results[:5]):  # Show only first 5 results
            print(f"\nText {i+1}: {result['text']}")
            print(f"Predicted Threat: {result['predicted_threat']}")
            print(f"Confidence: {result['confidence']:.4f}" if result['confidence'] else "Confidence: N/A")
            print(f"Risk Level: {result['risk_level']}")
            
            if 'keywords' in result:
                print("Key Terms:")
                for kw in result['keywords'][:5]:  # Show only top 5 keywords
                    print(f"  - {kw['word']} ({kw['score']:.4f})")
        
        if len(results) > 5:
            print(f"\n... and {len(results) - 5} more results not shown")
        
        # Save results if output file is specified
        if args.output or args.save_mongodb:
            save_results(results, args.output)
        
        # Threat summary
        threat_counts = {}
        risk_counts = {'Low': 0, 'Medium': 0, 'High': 0}
        
        for result in results:
            threat = result['predicted_threat']
            risk = result['risk_level']
            
            threat_counts[threat] = threat_counts.get(threat, 0) + 1
            risk_counts[risk] += 1
        
        print("\nThreat Summary:")
        for threat, count in sorted(threat_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  - {threat}: {count} ({count/len(results)*100:.1f}%)")
        
        print("\nRisk Level Summary:")
        for risk, count in risk_counts.items():
            print(f"  - {risk}: {count} ({count/len(results)*100:.1f}%)")
        
        print(f"\n✅ Analysis complete for {len(results)} texts")
    
    except FileNotFoundError as e:
        print(f"❌ {e}")
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Text Threat Detection Prediction")
    
    # Input options
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument("--text", type=str, help="Text to analyze")
    input_group.add_argument("--file", type=str, help="Path to file containing texts to analyze")
    input_group.add_argument("--mongodb", action="store_true", help="Load recent texts from MongoDB")
    
    # Analysis options
    parser.add_argument("--keywords", action="store_true", default=True, 
                       help="Extract keywords from the text (default: True)")
    parser.add_argument("--limit", type=int, default=100,
                       help="Maximum number of texts to analyze when using MongoDB (default: 100)")
    
    # Output options
    parser.add_argument("--output", type=str, help="Path to save analysis results")
    parser.add_argument("--save-mongodb", action="store_true", 
                       help="Save analysis results to MongoDB")
    
    args = parser.parse_args()
    main(args)