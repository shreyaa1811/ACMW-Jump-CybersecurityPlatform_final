#!/usr/bin/env python3

"""
Text Threat Detection Model
- BERT-based model for detecting security threats in text content
- Fine-tuned classifier for threat type identification
- Feature importance analysis for explainability
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import joblib
from pathlib import Path
import json
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import warnings
warnings.filterwarnings('ignore')

# Download NLTK resources if not already present
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

# Create directory for model artifacts
model_dir = Path('model_artifacts')
model_dir.mkdir(exist_ok=True)

class TextThreatDetectionModel:
    """Text-based threat detection model using NLP techniques"""
    
    def __init__(self, model_path=None, vectorizer_path=None, label_encoder_path=None):
        """Initialize the model or load pretrained model"""
        if model_path and vectorizer_path and label_encoder_path:
            self.model = joblib.load(model_path)
            self.vectorizer = joblib.load(vectorizer_path)
            self.label_encoder = joblib.load(label_encoder_path)
            self.is_trained = True
            print(f"✅ Loaded pretrained model from {model_path}")
        else:
            self.model = None
            self.vectorizer = TfidfVectorizer(
                max_features=10000,
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.9
            )
            self.label_encoder = LabelEncoder()
            self.is_trained = False
            print("Initialized new model (not trained yet)")
    
    def preprocess_text(self, text):
        """Clean and preprocess text data"""
        if pd.isna(text):
            return ""
        
        # Convert to lowercase
        text = str(text).lower()
        
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', ' url ', text)
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Remove punctuation and special characters
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words]
        
        # Lemmatize
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(token) for token in tokens]
        
        # Join tokens back into text
        return ' '.join(tokens)
    
    def train(self, X_train, y_train, model_type='rf'):
        """Train the model on preprocessed text data"""
        print("Training text threat detection model...")
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y_train)
        
        # Create and fit vectorizer
        X_vectorized = self.vectorizer.fit_transform(X_train)
        
        if model_type == 'rf':
            # Train Random Forest
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=20,
                min_samples_split=10,
                min_samples_leaf=4,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            )
        else:
            # Train Logistic Regression (default fallback)
            self.model = LogisticRegression(
                C=1.0,
                max_iter=1000,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            )
        
        # Fit the model
        self.model.fit(X_vectorized, y_encoded)
        self.is_trained = True
        
        print("✅ Model training complete")
        
        # Save the model components
        joblib.dump(self.model, model_dir / 'text_threat_model.pkl')
        joblib.dump(self.vectorizer, model_dir / 'text_threat_vectorizer.pkl')
        joblib.dump(self.label_encoder, model_dir / 'text_threat_label_encoder.pkl')
        
        print(f"✅ Model saved to {model_dir / 'text_threat_model.pkl'}")
        
        return self.model
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance on test set"""
        if not self.is_trained:
            raise Exception("Model not trained yet")
        
        print("Evaluating model performance...")
        
        # Preprocess and vectorize test data
        X_test_vectorized = self.vectorizer.transform(X_test)
        
        # Encode test labels
        y_test_encoded = self.label_encoder.transform(y_test)
        
        # Make predictions
        y_pred = self.model.predict(X_test_vectorized)
        y_pred_proba = None
        
        if hasattr(self.model, 'predict_proba'):
            y_pred_proba = self.model.predict_proba(X_test_vectorized)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test_encoded, y_pred)
        precision = precision_score(y_test_encoded, y_pred, average='weighted')
        recall = recall_score(y_test_encoded, y_pred, average='weighted')
        f1 = f1_score(y_test_encoded, y_pred, average='weighted')
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        
        # Confusion matrix
        cm = confusion_matrix(y_test_encoded, y_pred)
        plt.figure(figsize=(10, 8))
        class_names = self.label_encoder.classes_
        
        # Create a more readable confusion matrix for many classes
        if len(class_names) > 10:
            # For many classes, make a simplified heatmap
            sns.heatmap(cm, annot=False, cmap='Blues')
            plt.title('Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
        else:
            # For fewer classes, show the full matrix with labels
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                      xticklabels=class_names, yticklabels=class_names)
            plt.title('Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(model_dir / 'text_threat_confusion_matrix.png')
        
        # Feature importance visualization (for Random Forest)
        if isinstance(self.model, RandomForestClassifier):
            feature_names = self.vectorizer.get_feature_names_out()
            importances = self.model.feature_importances_
            indices = np.argsort(importances)[-20:]  # Top 20 features
            
            plt.figure(figsize=(12, 8))
            plt.title('Feature Importances')
            plt.barh(range(len(indices)), importances[indices], align='center')
            plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
            plt.xlabel('Relative Importance')
            plt.tight_layout()
            plt.savefig(model_dir / 'text_threat_feature_importance.png')
        
        # Full classification report
        report = classification_report(y_test_encoded, y_pred, 
                                      target_names=class_names, output_dict=True)
        
        with open(model_dir / 'text_threat_classification_report.json', 'w') as f:
            json.dump(report, f, indent=4)
        
        # Save evaluation metrics
        metrics = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1)
        }
        
        with open(model_dir / 'text_threat_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=4)
        
        print(f"✅ Evaluation complete. Results saved to {model_dir}")
        return metrics
    
    def predict(self, texts):
        """Predict threat category for new texts"""
        if not self.is_trained:
            raise Exception("Model not trained yet")
        
        # Preprocess texts
        processed_texts = [self.preprocess_text(text) for text in texts]
        
        # Vectorize
        X_vectorized = self.vectorizer.transform(processed_texts)
        
        # Predict
        pred_encoded = self.model.predict(X_vectorized)
        
        # Get probability scores if available
        pred_proba = None
        if hasattr(self.model, 'predict_proba'):
            pred_proba = self.model.predict_proba(X_vectorized)
        
        # Convert predictions back to original labels
        predictions = self.label_encoder.inverse_transform(pred_encoded)
        
        # Create results
        results = []
        for i, pred in enumerate(predictions):
            result = {
                'text': texts[i][:100] + '...' if len(texts[i]) > 100 else texts[i],
                'predicted_threat': pred,
                'confidence': None
            }
            
            # Add confidence if probability is available
            if pred_proba is not None:
                result['confidence'] = float(pred_proba[i, pred_encoded[i]])
            
            results.append(result)
        
        return results
    
    def extract_keywords(self, text, top_n=10):
        """Extract key threat-related terms from text"""
        if not self.is_trained:
            raise Exception("Model not trained yet")
        
        # Preprocess the text
        processed_text = self.preprocess_text(text)
        
        # Get the TF-IDF vector for this text
        text_vector = self.vectorizer.transform([processed_text])
        
        # Get feature names
        feature_names = self.vectorizer.get_feature_names_out()
        
        # Get TF-IDF scores for this text
        tfidf_scores = text_vector.toarray()[0]
        
        # Create a dictionary of term: score
        term_scores = {feature_names[i]: tfidf_scores[i] for i in range(len(feature_names)) if tfidf_scores[i] > 0}
        
        # Sort by score and get top N
        top_terms = sorted(term_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
        
        return top_terms

def train_and_evaluate(X_train, y_train, X_test, y_test, model_type='rf'):
    """Train and evaluate the model in one step"""
    # Initialize model
    model = TextThreatDetectionModel()
    
    # Train model
    model.train(X_train, y_train, model_type=model_type)
    
    # Evaluate model
    metrics = model.evaluate(X_test, y_test)
    
    return model, metrics

if __name__ == "__main__":
    # This will be implemented in train.py
    print("This module defines the text threat detection model.")
    print("Use train.py to train the model and predict.py to make predictions.")