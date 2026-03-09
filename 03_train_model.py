import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os

# Load the processed data
print("Loading processed data...")
df = pd.read_csv('../data/processed_features_2023.csv')
print(f"Loaded {len(df)} rows of data")

# Convert timedelta to seconds for practice pace
print("Converting practice pace to seconds...")
df['AvgPracticePace_seconds'] = pd.to_timedelta(df['AvgPracticePace']).dt.total_seconds()

# Create target variable: Podium (1) vs Not Podium (0)
print("Creating target variable...")
df['Podium'] = df['FinishPosition'].apply(lambda x: 1 if x <= 3 else 0)

# Check class distribution
podium_count = df['Podium'].sum()
total_races = len(df)
print(f"Podium finishes: {podium_count}/{total_races} ({podium_count/total_races:.1%})")

# Select features for the model
features = ['AvgPracticePace_seconds', 'QualifyingPos', 'GridPosition']
X = df[features]
y = df['Podium']

# Split data into training and testing sets
print("Splitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train Random Forest model
print("Training Random Forest model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate model
print("Evaluating model performance...")
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n=== MODEL PERFORMANCE ===")
print(f"Accuracy: {accuracy:.2%}")
print(f"Test set size: {len(y_test)} races")
print(f"Correct predictions: {accuracy_score(y_test, y_pred, normalize=False)}/{len(y_test)}")

print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Not Podium', 'Podium']))

# Show feature importance
print("\n=== FEATURE IMPORTANCE ===")
feature_importance = pd.DataFrame({
    'Feature': features,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

print(feature_importance)

# Save the model
print("\nSaving model...")
os.makedirs('../models', exist_ok=True)
with open('../models/podium_predictor_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model saved to '../models/podium_predictor_model.pkl'")

# Show some predictions
print("\n=== SAMPLE PREDICTIONS ===")
sample_data = X_test.head(5).copy()
sample_data['Actual_Podium'] = y_test.head(5).values
sample_data['Predicted_Podium'] = y_pred[:5]
sample_data['Prediction_Correct'] = (sample_data['Actual_Podium'] == sample_data['Predicted_Podium'])

print(sample_data)