import numpy as np
import pandas as pd
import sqlite3
from datetime import datetime, timedelta
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, log_loss, classification_report
import warnings

# Suppress warnings for cleaner output in a self-contained script
warnings.filterwarnings('ignore')

# 1. Synthetic Dataset Generation
print("1. Generating Synthetic Dataset...")

np.random.seed(42)
num_users = 200
num_ads = 100
total_impressions = 15000 # Sufficient volume for sequential features

# User Profiles
user_profiles = pd.DataFrame({
    'user_id': range(num_users),
    'age': np.random.randint(18, 65, num_users),
    'gender': np.random.choice(['Male', 'Female', 'Other'], num_users, p=[0.45, 0.45, 0.1]),
    'location_category': np.random.choice(['Urban', 'Suburban', 'Rural', 'Exurban'], num_users, p=[0.4, 0.3, 0.2, 0.1])
})

# Ad Attributes
ad_attributes = pd.DataFrame({
    'ad_id': range(num_ads),
    'ad_category': np.random.choice(['Electronics', 'Fashion', 'HomeGoods', 'Automotive', 'Services', 'Travel', 'Food'], num_ads),
    'advertiser_id': np.random.randint(1, 30, num_ads),
    'ad_placement_type': np.random.choice(['Banner', 'Native', 'Video', 'Pop-up'], num_ads),
    'creative_type': np.random.choice(['Image', 'GIF', 'Text', 'Rich Media'], num_ads)
})

# Impression Logs
impression_data = []
start_time = datetime(2023, 1, 1, 0, 0, 0)
time_range_minutes = 60 * 24 * 60 # 60 days of impressions

for i in range(total_impressions):
    user_id = np.random.randint(0, num_users)
    ad_id = np.random.randint(0, num_ads)
    impression_timestamp = start_time + timedelta(minutes=np.random.randint(0, time_range_minutes))
    
    # Simulate click outcome based on some factors
    user_age = user_profiles.loc[user_profiles['user_id'] == user_id, 'age'].iloc[0]
    ad_cat = ad_attributes.loc[ad_attributes['ad_id'] == ad_id, 'ad_category'].iloc[0]
    
    base_click_prob = 0.03
    if user_age < 35:
        base_click_prob += 0.02
    if ad_cat in ['Electronics', 'Fashion']:
        base_click_prob += 0.015
    if np.random.rand() < 0.1: # Some random spikes
        base_click_prob += 0.05
    
    click_outcome = 1 if np.random.rand() < base_click_prob else 0
    
    impression_data.append({
        'impression_id': i,
        'user_id': user_id,
        'ad_id': ad_id,
        'impression_timestamp': impression_timestamp,
        'click_outcome': click_outcome
    })

impression_logs = pd.DataFrame(impression_data)

# Sort logs by user_id and timestamp for sequential features
impression_logs = impression_logs.sort_values(by=['user_id', 'impression_timestamp']).reset_index(drop=True)

# Merge all datasets
df = impression_logs.merge(user_profiles, on='user_id', how='left')
df = df.merge(ad_attributes, on='ad_id', how='left')

print("Synthetic Dataset Head:")
print(df.head())
print(f"\nDataset Shape: {df.shape}")
print(f"Total clicks: {df['click_outcome'].sum()} ({df['click_outcome'].mean():.2%} CTR)")

# 2. Advanced SQL-based Feature Engineering for Sequential Behavior
print("\n2. Performing SQL-based Feature Engineering...")

# Use sqlite3 for SQL window functions
conn = sqlite3.connect(':memory:')
# Ensure impression_timestamp is in a format SQLite understands for julianday
df['impression_timestamp'] = df['impression_timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
df.to_sql('impressions_with_attrs', conn, index=False, if_exists='replace')

sql_query = """
SELECT
    t1.*,
    (julianday(t1.impression_timestamp) - julianday(LAG(t1.impression_timestamp, 1) OVER (PARTITION BY t1.user_id ORDER BY t1.impression_timestamp))) * 24 * 60 * 60 AS time_since_last_impression_seconds,
    (julianday(t1.impression_timestamp) - julianday(LAG(CASE WHEN t1.click_outcome = 1 THEN t1.impression_timestamp END, 1) OVER (PARTITION BY t1.user_id ORDER BY t1.impression_timestamp))) * 24 * 60 * 60 AS time_since_last_click_seconds,
    COALESCE(SUM(1) OVER (PARTITION BY t1.user_id ORDER BY t1.impression_timestamp ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING), 0) AS user_impressions_count_prev,
    COALESCE(SUM(CASE WHEN t1.click_outcome = 1 THEN 1 ELSE 0 END) OVER (PARTITION BY t1.user_id ORDER BY t1.impression_timestamp ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING), 0) AS user_clicks_count_prev
FROM
    impressions_with_attrs t1
ORDER BY
    t1.user_id, t1.impression_timestamp
"""

df_features = pd.read_sql(sql_query, conn)
conn.close()

# Convert impression_timestamp back to datetime object for sorting
df_features['impression_timestamp'] = pd.to_datetime(df_features['impression_timestamp'])

# Calculate user_ctr_prev in pandas for robust division-by-zero handling
df_features['user_ctr_prev'] = df_features['user_clicks_count_prev'] / df_features['user_impressions_count_prev']
df_features['user_ctr_prev'] = df_features['user_ctr_prev'].fillna(0) # If no prior impressions, CTR is 0

# Impute NULLs for `time_since_last_impression_seconds` and `time_since_last_click_seconds`
# For first impression, `time_since_last_impression_seconds` will be NULL. Impute with a large value.
# For first click, `time_since_last_click_seconds` will be NULL. Impute with a large value.
# Use maximum observed value to set a reasonable "very long time ago" sentinel.
max_tsli = df_features['time_since_last_impression_seconds'].max()
max_tslc = df_features['time_since_last_click_seconds'].max()

df_features['time_since_last_impression_seconds'] = df_features['time_since_last_impression_seconds'].fillna(max_tsli * 1.5 if pd.notna(max_tsli) else 0)
df_features['time_since_last_click_seconds'] = df_features['time_since_last_click_seconds'].fillna(max_tslc * 1.5 if pd.notna(max_tslc) else 0)

print("Features after SQL engineering (head):")
print(df_features.head())
print("\nFeatures after SQL engineering (null counts):")
print(df_features.isnull().sum()) # Should show 0 nulls for the new features

# 3. Feature Preprocessing and Encoding
print("\n3. Performing Feature Preprocessing and Encoding...")

# Keep impression_timestamp temporarily for time-based split, drop other IDs
df_processed = df_features.drop(columns=['user_id', 'ad_id', 'impression_id'])

# Identify target and features
X = df_processed.drop('click_outcome', axis=1)
y = df_processed['click_outcome']

# Identify categorical and numerical features (excluding impression_timestamp from transformations)
features_for_preprocessing = X.drop(columns=['impression_timestamp']) 
categorical_features = features_for_preprocessing.select_dtypes(include='object').columns.tolist()
numerical_features = features_for_preprocessing.select_dtypes(include=np.number).columns.tolist()

# Create preprocessing pipelines for numerical and categorical features
numerical_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Create a column transformer to apply different transformations to different columns
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='passthrough' # Keep other columns if any, e.g. impression_timestamp
)

print(f"Categorical features identified: {categorical_features}")
print(f"Numerical features identified: {numerical_features}")

# 4. Model Selection and Time-Based Data Splitting
print("\n4. Splitting Data (Time-Based) and Model Selection...")

# Sort the dataset by impression_timestamp for time-based split
df_processed = df_processed.sort_values(by='impression_timestamp').reset_index(drop=True)

# Re-extract X and y after sorting, and now drop impression_timestamp from X for model input
X = df_processed.drop(columns=['click_outcome', 'impression_timestamp'])
y = df_processed['click_outcome']

# Apply the preprocessor to the feature matrix X
X_transformed = preprocessor.fit_transform(X)

# Get feature names after one-hot encoding for feature importance later
ohe_feature_names = preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(categorical_features)
final_feature_names = numerical_features + ohe_feature_names.tolist()

# Time-based split: Use first 80% for training, last 20% for testing
split_idx = int(len(X_transformed) * 0.8)
X_train_transformed, X_test_transformed = X_transformed[:split_idx], X_transformed[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

print(f"Training set size: {X_train_transformed.shape[0]} impressions")
print(f"Test set size: {X_test_transformed.shape[0]} impressions")
print(f"Proportion of clicks in training set: {y_train.mean():.4f}")
print(f"Proportion of clicks in test set: {y_test.mean():.4f}")

# Model Selection: Gradient Boosting Classifier
model = GradientBoostingClassifier(random_state=42)
print("Selected Model: GradientBoostingClassifier")

# 5. Model Training and Hyperparameter Tuning
print("\n5. Training Model (with basic tuning)...")

# Define a simpler parameter grid for GridSearch to keep runtime manageable
param_grid = {
    'n_estimators': [50, 100],
    'learning_rate': [0.05, 0.1],
    'max_depth': [3, 5]
}

# Use a small cross-validation (cv=2 or 3) for quick execution in a self-contained script
grid_search = GridSearchCV(model, param_grid, cv=3, scoring='roc_auc', n_jobs=-1, verbose=0)
grid_search.fit(X_train_transformed, y_train)

best_model = grid_search.best_estimator_
print(f"Best hyperparameters found: {grid_search.best_params_}")
print(f"Best ROC AUC score on training folds: {grid_search.best_score_:.4f}")

# 6. Model Evaluation and Interpretation
print("\n6. Evaluating Model Performance...")

y_pred_proba = best_model.predict_proba(X_test_transformed)[:, 1]
y_pred = best_model.predict(X_test_transformed)

roc_auc = roc_auc_score(y_test, y_pred_proba)
logloss = log_loss(y_test, y_pred_proba)

print(f"Test Set ROC AUC: {roc_auc:.4f}")
print(f"Test Set Log Loss: {logloss:.4f}")
print("\nClassification Report (Test Set):")
print(classification_report(y_test, y_pred))

# Feature Importance
print("\nFeature Importance (Top 10):")
feature_importances = best_model.feature_importances_

importance_df = pd.DataFrame({
    'feature': final_feature_names,
    'importance': feature_importances
}).sort_values(by='importance', ascending=False)

print(importance_df.head(10))

print("\n--- Script Finished ---")