#!/usr/bin/env python3

"""
Intrusion Detection Model
- XGBoost classifier for binary classification
- Hyperparameter tuning with cross-validation
- Model evaluation
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    precision_recall_curve, roc_curve
)
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
import joblib
from pathlib import Path
import json
from preprocessing import load_data, preprocess_data

# Create directory for model artifacts
model_dir = Path('model_artifacts')
model_dir.mkdir(exist_ok=True)

class IntrusionDetectionModel:
    """XGBoost-based model for intrusion detection"""
    
    def __init__(self, model_path=None):
        """Initialize the model or load pretrained model"""
        if model_path:
            self.model = joblib.load(model_path)
            self.preprocessor = joblib.load(model_dir / 'intrusion_preprocessor.pkl')
            print(f"✅ Loaded pretrained model from {model_path}")
        else:
            self.model = None
            self.preprocessor = None
    
    def train(self, X_train, y_train, X_test=None, y_test=None, tune_hyperparams=True):
        """Train the XGBoost model with optional hyperparameter tuning"""
        print("Training intrusion detection model...")
        
        if tune_hyperparams and X_test is not None and y_test is not None:
            # Hyperparameter tuning with RandomizedSearchCV
            param_dist = {
                'n_estimators': [100, 200, 300, 400, 500],
                'max_depth': [3, 5, 7, 9],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0],
                'gamma': [0, 0.1, 0.2],
                'min_child_weight': [1, 3, 5]
            }
            
            xgb_model = xgb.XGBClassifier(
                objective='binary:logistic',
                eval_metric='auc',
                random_state=42
            )
            
            search = RandomizedSearchCV(
                xgb_model,
                param_distributions=param_dist,
                n_iter=5,  # Reduced to speed up training
                scoring='roc_auc',
                cv=5,
                verbose=1,
                random_state=42,
                n_jobs=-1
            )
            
            # Fit without early stopping
            search.fit(X_train, y_train)
            
            # Get best parameters and train final model
            best_params = search.best_params_
            print(f"Best hyperparameters: {best_params}")
            
            # Save best parameters
            with open(model_dir / 'intrusion_best_params.json', 'w') as f:
                json.dump(best_params, f)
            
            # Create final model with best parameters
            self.model = xgb.XGBClassifier(
                objective='binary:logistic',
                eval_metric='auc',
                random_state=42,
                **best_params
            )
            
            # Train the final model without early stopping
            self.model.fit(X_train, y_train)
            
        else:
            # Train with default parameters
            self.model = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.1,
                subsample=0.9,
                colsample_bytree=0.9,
                objective='binary:logistic',
                eval_metric='auc',
                random_state=42
            )
            
            # Simple training without early stopping
            self.model.fit(X_train, y_train)
        
        print("✅ Model training complete")
        
        # Save the model
        joblib.dump(self.model, model_dir / 'intrusion_detection_model.pkl')
        print(f"✅ Model saved to {model_dir / 'intrusion_detection_model.pkl'}")
        
        return self.model
    
    def evaluate(self, X_test, y_test, feature_names=None):
        """Evaluate model performance on test set"""
        if self.model is None:
            raise Exception("Model not trained yet")
        
        print("Evaluating model performance...")
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"ROC AUC: {auc:.4f}")
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(model_dir / 'intrusion_confusion_matrix.png')
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'AUC = {auc:.4f}')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc='lower right')
        plt.savefig(model_dir / 'intrusion_roc_curve.png')
        
        # Precision-Recall Curve
        precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_pred_proba)
        plt.figure(figsize=(8, 6))
        plt.plot(recall_curve, precision_curve)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.savefig(model_dir / 'intrusion_pr_curve.png')
        
        # Feature importance
        if feature_names is not None:
            # Get feature importances
            importances = self.model.feature_importances_
            
            # Get preprocessor feature names if one-hot encoding was applied
            if hasattr(self.preprocessor, 'transformers_'):
                # This is more complex due to the ColumnTransformer and one-hot encoding
                # We need to get the feature names after preprocessing
                # For now, we'll use the top importances
                sorted_idx = np.argsort(importances)[::-1]
                plt.figure(figsize=(10, 8))
                plt.barh(range(min(20, len(sorted_idx))), importances[sorted_idx][:20])
                plt.yticks(range(min(20, len(sorted_idx))), [f"Feature {i}" for i in sorted_idx[:20]])
                plt.title('Top 20 Feature Importances')
                plt.savefig(model_dir / 'intrusion_feature_importance.png')
            else:
                # If no complex preprocessing, we can use original feature names
                feature_importance = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': importances
                }).sort_values('Importance', ascending=False)
                
                plt.figure(figsize=(10, 8))
                sns.barplot(x='Importance', y='Feature', data=feature_importance.head(20))
                plt.title('Top 20 Feature Importances')
                plt.tight_layout()
                plt.savefig(model_dir / 'intrusion_feature_importance.png')
        
        # Full classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        with open(model_dir / 'intrusion_classification_report.json', 'w') as f:
            json.dump(report, f)
        
        # Save evaluation metrics
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc
        }
        
        with open(model_dir / 'intrusion_metrics.json', 'w') as f:
            json.dump(metrics, f)
        
        print(f"✅ Evaluation complete. Results saved to {model_dir}")
        return metrics
    
    def predict(self, X):
        """Make predictions using the trained model"""
        if self.model is None:
            raise Exception("Model not trained yet")
        
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Get probability predictions using the trained model"""
        if self.model is None:
            raise Exception("Model not trained yet")
        
        return self.model.predict_proba(X)[:, 1]

def train_and_evaluate():
    """Main function to train and evaluate the model"""
    # Load data
    df = load_data()
    
    # Preprocess data
    X_train, X_test, y_train, y_test, preprocessor, feature_names = preprocess_data(df)
    
    # Initialize and train model
    model = IntrusionDetectionModel()
    model.preprocessor = preprocessor
    model.train(X_train, y_train, X_test, y_test, tune_hyperparams=True)
    
    # Evaluate model
    metrics = model.evaluate(X_test, y_test, feature_names)
    
    return model, metrics

if __name__ == "__main__":
    # Train and evaluate model
    model, metrics = train_and_evaluate()
    print("Final metrics:", metrics)