#!/usr/bin/env python
"""
Classifier for AI-generated vs Real receipt images.
Uses the 14 Elective features extracted from text crops.
"""

import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve, average_precision_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# Paths
DATA_PATH = Path(r"C:\Users\dev\Workspace\school\elective\programs\thesis-classifier\elective3_features_900x900.csv")
MODEL_DIR = Path(r"C:\Users\dev\Workspace\school\elective\programs\thesis-classifier\models")
MODEL_DIR.mkdir(exist_ok=True)

# Feature columns (exclude metadata)
META_COLS = ['image_id', 'source_image', 'source_path', 'class_name', 'label', 'split', 'quality_status', 'include_default']
FEATURE_COLS = [
    'crop_count', 'crop_area_ratio_sum', 'crop_width_norm_avg', 'crop_height_norm_avg',
    'crop_aspect_ratio_avg', 'gray_mean_avg', 'gray_mean_std', 'gray_std_avg',
    'ink_fraction_avg', 'edge_density_avg', 'edge_density_std',
    'laplacian_variance_avg', 'local_variance_avg', 'entropy_avg'
]
TARGET_COL = 'label'


def load_data():
    """Load and prepare the dataset."""
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} samples")
    print(f"Class distribution:\n{df[TARGET_COL].value_counts().sort_index()}")
    
    X = df[FEATURE_COLS].values
    y = df[TARGET_COL].values
    
    # Check for NaN
    if np.any(np.isnan(X)):
        print("Warning: NaN values found in features, filling with median")
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(strategy='median')
        X = imputer.fit_transform(X)
    
    return X, y, df


def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate a trained model."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else model.decision_function(X_test)
    
    print(f"\n{'='*60}")
    print(f"Results for {model_name}")
    print(f"{'='*60}")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Real (0)', 'AI (1)']))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)
    tn, fp, fn, tp = cm.ravel()
    print(f"TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    
    # ROC AUC
    roc_auc = roc_auc_score(y_test, y_prob)
    print(f"\nROC AUC: {roc_auc:.4f}")
    
    # Average Precision (PR AUC)
    ap = average_precision_score(y_test, y_prob)
    print(f"Average Precision (PR AUC): {ap:.4f}")
    
    # Per-class accuracy
    acc_0 = tn / (tn + fp) if (tn + fp) > 0 else 0
    acc_1 = tp / (tp + fn) if (tp + fn) > 0 else 0
    print(f"Accuracy on Real (0): {acc_0:.4f}")
    print(f"Accuracy on AI (1): {acc_1:.4f}")
    print(f"Balanced Accuracy: {(acc_0 + acc_1) / 2:.4f}")
    
    return {
        'model_name': model_name,
        'roc_auc': roc_auc,
        'average_precision': ap,
        'balanced_accuracy': (acc_0 + acc_1) / 2,
        'confusion_matrix': cm.tolist(),
        'classification_report': classification_report(y_test, y_pred, target_names=['Real (0)', 'AI (1)'], output_dict=True)
    }


def train_and_evaluate():
    """Train multiple models and compare."""
    X, y, df = load_data()
    
    # Train/test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTrain size: {len(X_train)}, Test size: {len(X_test)}")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save scaler
    joblib.dump(scaler, MODEL_DIR / 'scaler.pkl')
    print(f"\nScaler saved to {MODEL_DIR / 'scaler.pkl'}")
    
    # Define models to try
    models = {
        'LogisticRegression': LogisticRegression(
            max_iter=1000, class_weight='balanced', random_state=42, C=1.0
        ),
        'RandomForest': RandomForestClassifier(
            n_estimators=500, max_depth=10, min_samples_split=5,
            min_samples_leaf=2, class_weight='balanced', random_state=42, n_jobs=-1
        ),
        'GradientBoosting': GradientBoostingClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.8, random_state=42
        ),
        'XGBoost': xgb.XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=1, random_state=42, n_jobs=-1,
            eval_metric='logloss', verbosity=0
        ),
        'SVM (RBF)': SVC(
            kernel='rbf', class_weight='balanced', probability=True,
            random_state=42, C=1.0, gamma='scale'
        ),
        'MLP': MLPClassifier(
            hidden_layer_sizes=(64, 32), max_iter=500,
            alpha=0.001, early_stopping=True, random_state=42
        ),
    }
    
    results = {}
    best_model = None
    best_score = 0
    
    # Cross-validation on training set
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    print("\n" + "="*60)
    print("CROSS-VALIDATION RESULTS (5-fold)")
    print("="*60)
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Use scaled data for models that need it
        if name in ['LogisticRegression', 'SVM (RBF)', 'MLP']:
            X_train_use = X_train_scaled
        else:
            X_train_use = X_train
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train_use, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)
        print(f"  CV ROC AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
        
        # Fit on full training set
        model.fit(X_train_use, y_train)
        
        # Evaluate on test set
        if name in ['LogisticRegression', 'SVM (RBF)', 'MLP']:
            X_test_use = X_test_scaled
        else:
            X_test_use = X_test
        
        result = evaluate_model(model, X_test_use, y_test, name)
        results[name] = result
        
        # Track best model by ROC AUC
        if result['roc_auc'] > best_score:
            best_score = result['roc_auc']
            best_model = (name, model, X_train_use is X_train_scaled)
    
    # Save best model
    best_name, best_model_obj, uses_scaled = best_model
    print(f"\n{'='*60}")
    print(f"BEST MODEL: {best_name} (ROC AUC: {best_score:.4f})")
    print(f"{'='*60}")
    
    # Save best model
    model_package = {
        'model': best_model_obj,
        'scaler': scaler if uses_scaled else None,
        'feature_names': FEATURE_COLS,
        'model_name': best_name,
        'uses_scaled': uses_scaled,
    }
    joblib.dump(model_package, MODEL_DIR / 'best_classifier.pkl')
    print(f"Best model saved to {MODEL_DIR / 'best_classifier.pkl'}")
    
    # Save all results
    with open(MODEL_DIR / 'evaluation_results.json', 'w') as f:
        # Convert numpy types to Python types for JSON serialization
        def convert(obj):
            if isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(v) for v in obj]
            return obj
        json.dump(convert(results), f, indent=2)
    print(f"Results saved to {MODEL_DIR / 'evaluation_results.json'}")
    
    # Feature importance for tree-based models
    if hasattr(best_model_obj, 'feature_importances_'):
        importances = best_model_obj.feature_importances_
        feat_imp = pd.DataFrame({
            'feature': FEATURE_COLS,
            'importance': importances
        }).sort_values('importance', ascending=False)
        print(f"\nFeature Importances ({best_name}):")
        print(feat_imp.to_string(index=False))
        feat_imp.to_csv(MODEL_DIR / 'feature_importances.csv', index=False)
    
    return results, best_model


def predict_new_image(image_features, model_path=None):
    """
    Predict on new image features.
    
    Args:
        image_features: dict or array with 14 features in FEATURE_COLS order
        model_path: path to saved model (uses best_classifier.pkl if None)
    
    Returns:
        dict with prediction, probability, and class label
    """
    if model_path is None:
        model_path = MODEL_DIR / 'best_classifier.pkl'
    
    package = joblib.load(model_path)
    model = package['model']
    scaler = package['scaler']
    uses_scaled = package['uses_scaled']
    
    # Prepare features
    if isinstance(image_features, dict):
        X = np.array([[image_features[col] for col in FEATURE_COLS]])
    else:
        X = np.array(image_features).reshape(1, -1)
    
    # Scale if needed
    if uses_scaled and scaler is not None:
        X = scaler.transform(X)
    
    # Predict
    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0, 1] if hasattr(model, 'predict_proba') else None
    
    return {
        'prediction': int(pred),
        'class_name': 'ai_generated' if pred == 1 else 'non_ai_generated',
        'probability_ai': float(prob) if prob is not None else None
    }


if __name__ == '__main__':
    print("Starting classifier training...")
    results, best_model = train_and_evaluate()
    
    print("\n\nSUMMARY OF ALL MODELS:")
    print("-" * 80)
    summary = []
    for name, res in results.items():
        summary.append({
            'Model': name,
            'ROC AUC': f"{res['roc_auc']:.4f}",
            'PR AUC': f"{res['average_precision']:.4f}",
            'Balanced Acc': f"{res['balanced_accuracy']:.4f}"
        })
    summary_df = pd.DataFrame(summary).sort_values('ROC AUC', ascending=False)
    print(summary_df.to_string(index=False))
    
    print("\nDone!")