import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import warnings
warnings.filterwarnings('ignore')

# Load the data
df = pd.read_csv('GLT_302_TEST_ADENIYI_BATCH_1.csv')

# Feature Engineering
df['submission_hour'] = pd.to_datetime(df['Timestamp']).dt.hour
df['submission_minute'] = pd.to_datetime(df['Timestamp']).dt.minute
df['time_of_day'] = df['submission_hour'].apply(
    lambda x: 'Early Morning' if 6 <= x <= 9 
    else 'Mid Morning' if 10 <= x <= 13 
    else 'Afternoon' if 14 <= x <= 17 
    else 'Evening'
)

# Create performance bands (target variable)
def categorize_score(score):
    if score >= 70:
        return 'Excellent'
    elif score >= 60:
        return 'Good'
    elif score >= 50:
        return 'Average'
    elif score >= 40:
        return 'Below Average'
    elif score >= 30:
        return 'Poor'
    elif score >= 20:
        return 'Very Poor'
    else:
        return 'Critical'

df['performance_band'] = df['Score (%)'].apply(categorize_score)

# Count multiple attempts per student
attempt_counts = df.groupby('Name').size().reset_index(name='num_attempts')
df = df.merge(attempt_counts, on='Name', how='left')

# Select features
features = ['Department', 'submission_hour', 'time_of_day', 'num_attempts']
X = df[features].copy()
y = df['performance_band']

# Encode categorical variables
le_dept = LabelEncoder()
le_time = LabelEncoder()
le_target = LabelEncoder()

X['Department'] = le_dept.fit_transform(X['Department'])
X['time_of_day'] = le_time.fit_transform(X['time_of_day'])
y_encoded = le_target.fit_transform(y)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print("=" * 60)
print("SCORE RANGE PREDICTION MODEL TRAINING")
print("=" * 60)
print(f"\nDataset Size: {len(df)} records")
print(f"Training Set: {len(X_train)} | Test Set: {len(X_test)}")
print(f"\nPerformance Band Distribution:")
print(df['performance_band'].value_counts().sort_index())

# Model 1: Random Forest
print("\n" + "=" * 60)
print("Training Random Forest Classifier...")
print("=" * 60)

rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    class_weight='balanced'
)

rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_pred)

print(f"\nRandom Forest Accuracy: {rf_accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, rf_pred, target_names=le_target.classes_))

# Cross-validation
cv_scores = cross_val_score(rf_model, X_train, y_train, cv=5)
print(f"\nCross-Validation Scores: {cv_scores}")
print(f"Mean CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# Feature Importance
print("\nFeature Importance:")
feature_importance = pd.DataFrame({
    'feature': features,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)
print(feature_importance)

# Model 2: Gradient Boosting (Alternative)
print("\n" + "=" * 60)
print("Training Gradient Boosting Classifier...")
print("=" * 60)

gb_model = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)

gb_model.fit(X_train, y_train)
gb_pred = gb_model.predict(X_test)
gb_accuracy = accuracy_score(y_test, gb_pred)

print(f"\nGradient Boosting Accuracy: {gb_accuracy:.4f}")

# Select best model
if rf_accuracy >= gb_accuracy:
    best_model = rf_model
    best_model_name = "Random Forest"
    best_accuracy = rf_accuracy
else:
    best_model = gb_model
    best_model_name = "Gradient Boosting"
    best_accuracy = gb_accuracy

print("\n" + "=" * 60)
print(f"BEST MODEL: {best_model_name} (Accuracy: {best_accuracy:.4f})")
print("=" * 60)

# Save the model and encoders
joblib.dump(best_model, 'score_predictor_model.pkl')
joblib.dump(le_dept, 'label_encoder_dept.pkl')
joblib.dump(le_time, 'label_encoder_time.pkl')
joblib.dump(le_target, 'label_encoder_target.pkl')

print("\n✓ Model saved as 'score_predictor_model.pkl'")
print("✓ Encoders saved successfully")

# Save feature names for reference
with open('feature_names.txt', 'w') as f:
    f.write(','.join(features))

print("\n" + "=" * 60)
print("MODEL TRAINING COMPLETE!")
print("=" * 60)