import pandas as pd
import numpy as np
from datetime import timedelta
import random

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
)

# --- 1. Data Simulation ---

# Configuration for synthetic data generation
N_CUSTOMERS = 2000
START_DATE = pd.to_datetime('2022-01-01')
END_DATE = pd.to_datetime('2023-12-31')
# The window for early usage pattern analysis
FEATURE_WINDOW_DAYS = 30
# Overall churn rate for synthetic data
CHURN_RATE = 0.25
# Probability of a customer *not* having usage on a given day (simulates inactive days)
USAGE_SPARSITY = 0.3

# Generate subscriptions_df
customer_ids = np.arange(N_CUSTOMERS)
signup_dates = [START_DATE + timedelta(days=np.random.randint(0, (END_DATE - START_DATE).days)) for _ in range(N_CUSTOMERS)]
subscription_plans = np.random.choice(['Basic', 'Standard', 'Premium'], N_CUSTOMERS, p=[0.4, 0.4, 0.2])

churn_dates = []
for i in range(N_CUSTOMERS):
    if np.random.rand() < CHURN_RATE:
        # Churn happens between FEATURE_WINDOW_DAYS + 1 and 365 days after signup
        # This ensures churn prediction is based on early behavior before churn date
        churn_offset = np.random.randint(FEATURE_WINDOW_DAYS + 1, 365)
        churn_date = signup_dates[i] + timedelta(days=churn_offset)
        # Ensure churn_date doesn't exceed simulation end date
        churn_dates.append(min(churn_date, END_DATE + timedelta(days=30))) # allow some buffer
    else:
        churn_dates.append(pd.NaT) # pd.NaT represents NULL for datetime

subscriptions_df = pd.DataFrame({
    'customer_id': customer_ids,
    'signup_date': signup_dates,
    'subscription_plan': subscription_plans,
    'churn_date': churn_dates
})

# Generate usage_df
usage_records = []
for _, row in subscriptions_df.iterrows():
    customer_id = row['customer_id']
    signup_date = row['signup_date']
    churn_date = row['churn_date']

    # Generate usage data for a period relevant to the feature window
    # Extend slightly beyond the feature window to simulate data that is ignored
    # Also, usage should not exceed the churn date or the simulation end date
    usage_gen_end_date = min(signup_date + timedelta(days=FEATURE_WINDOW_DAYS + 30),
                             churn_date if pd.notna(churn_date) else END_DATE)

    for d_offset in range((usage_gen_end_date - signup_date).days):
        usage_date = signup_date + timedelta(days=d_offset)
        if np.random.rand() > USAGE_SPARSITY: # Simulate some days with no usage
            usage_records.append({
                'customer_id': customer_id,
                'usage_date': usage_date,
                'data_used_mb': np.random.uniform(10, 500), # MB used daily
                'calls_made': np.random.randint(0, 30),     # Number of calls
                'features_accessed': np.random.randint(0, 10) # Number of unique features accessed
            })

usage_df = pd.DataFrame(usage_records)

print(f"Generated {len(subscriptions_df)} subscriptions and {len(usage_df)} usage records.")
print(f"Simulated churn rate: {subscriptions_df['churn_date'].notna().mean():.2f}")
print("\nSubscriptions head:")
print(subscriptions_df.head())
print("\nUsage head:")
print(usage_df.head())

# --- 2. Time-Series Feature Engineering (First 30 Days) ---

# Merge subscriptions to usage to easily filter by signup_date for each customer
merged_usage_for_features = pd.merge(usage_df, subscriptions_df[['customer_id', 'signup_date']], on='customer_id', how='left')

# Filter usage data for the first `FEATURE_WINDOW_DAYS`
first_window_usage = merged_usage_for_features[
    (merged_usage_for_features['usage_date'] >= merged_usage_for_features['signup_date']) &
    (merged_usage_for_features['usage_date'] < merged_usage_for_features['signup_date'] + timedelta(days=FEATURE_WINDOW_DAYS))
].copy()

# Calculate aggregated features for each customer
engineered_features = first_window_usage.groupby('customer_id').agg(
    data_used_mb_sum=('data_used_mb', 'sum'),
    data_used_mb_mean=('data_used_mb', 'mean'),
    data_used_mb_max=('data_used_mb', 'max'),
    data_used_mb_min=('data_used_mb', 'min'),
    data_used_mb_std=('data_used_mb', 'std'),
    calls_made_sum=('calls_made', 'sum'),
    calls_made_mean=('calls_made', 'mean'),
    calls_made_max=('calls_made', 'max'),
    calls_made_min=('calls_made', 'min'),
    calls_made_std=('calls_made', 'std'),
    features_accessed_sum=('features_accessed', 'sum'),
    features_accessed_mean=('features_accessed', 'mean'),
    features_accessed_max=('features_accessed', 'max'),
    features_accessed_min=('features_accessed', 'min'),
    features_accessed_std=('features_accessed', 'std'),
    num_active_days=('usage_date', 'nunique')
).reset_index()

# Handle NaNs from standard deviation (occurs if only one usage day for a customer)
# Or if a customer has 0 usage, these aggregates would be NaN after groupby.
# We fill these with 0 as it represents no activity.
engineered_features.fillna(0, inplace=True)

# Derive frequency-based feature: ratio of active days to the total window
engineered_features['active_day_ratio'] = engineered_features['num_active_days'] / FEATURE_WINDOW_DAYS

# --- 3. Dataset Consolidation and Target Definition ---

# Merge engineered features back to subscriptions_df, ensuring all customers are included
# Use a left merge to include customers who might not have any usage in the first 30 days.
final_df = pd.merge(subscriptions_df, engineered_features, on='customer_id', how='left')

# Fill NaN values for engineered features for customers with no usage in the first window.
# These NaNs would result from the left merge if a customer was not in engineered_features.
feature_cols = [col for col in engineered_features.columns if col != 'customer_id']
final_df[feature_cols] = final_df[feature_cols].fillna(0)

# Define the binary target variable 'churn'
final_df['churn'] = final_df['churn_date'].notna().astype(int)

print("\nFinal DataFrame head after feature engineering and consolidation:")
print(final_df.head())
print(f"\nChurn distribution in consolidated dataset: \n{final_df['churn'].value_counts(normalize=True)}")

# --- 4. Data Preprocessing and Splitting ---

# Define features (X) and target (y)
X = final_df.drop(['customer_id', 'signup_date', 'churn_date', 'churn'], axis=1)
y = final_df['churn']

# Identify categorical and numerical features
categorical_features = ['subscription_plan']
numerical_features = [col for col in X.columns if col not in categorical_features]

# Create a preprocessor using ColumnTransformer for scaling and one-hot encoding
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), categorical_features)
    ],
    remainder='passthrough' # Keep other columns (if any) as they are
)

# Split data into training and testing sets, stratifying by 'churn' to maintain ratio
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"\nShape of training data: {X_train.shape}")
print(f"Shape of testing data: {X_test.shape}")
print(f"Churn distribution in training set: \n{y_train.value_counts(normalize=True)}")
print(f"Churn distribution in test set: \n{y_test.value_counts(normalize=True)}")

# --- 5. Model Training and Evaluation ---

# Create a pipeline with the preprocessor and a RandomForestClassifier
# Using class_weight='balanced' to handle potential class imbalance
model_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', RandomForestClassifier(random_state=42, class_weight='balanced'))])

# Train the model
model_pipeline.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model_pipeline.predict(X_test)
y_proba = model_pipeline.predict_proba(X_test)[:, 1] # Probabilities for the positive class (churn)

# Evaluate the model's performance
print("\n--- Model Evaluation ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision (Churn=1): {precision_score(y_test, y_pred, pos_label=1):.4f}")
print(f"Recall (Churn=1): {recall_score(y_test, y_pred, pos_label=1):.4f}")
print(f"F1-Score (Churn=1): {f1_score(y_test, y_pred, pos_label=1):.4f}")
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_proba):.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['No Churn (0)', 'Churn (1)']))

# Optional: Display top feature importances for interpretability
print("\n--- Feature Importance (Top 10) ---")
# Get feature names after one-hot encoding and scaling
ohe_feature_names = model_pipeline.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(categorical_features)
all_feature_names = numerical_features + list(ohe_feature_names)

# Get feature importances from the trained Random Forest classifier
feature_importances = model_pipeline.named_steps['classifier'].feature_importances_
importance_df = pd.DataFrame({'feature': all_feature_names, 'importance': feature_importances})
importance_df = importance_df.sort_values(by='importance', ascending=False)
print(importance_df.head(10))