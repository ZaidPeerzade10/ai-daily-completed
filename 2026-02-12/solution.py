import pandas as pd
import numpy as np
import datetime
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, classification_report

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# --- 1. Generate Synthetic Data (Pandas/Numpy) ---

np.random.seed(42)

# Helper function for random dates
def random_dates(start, end, n=10):
    start_u = start.value // 10**9
    end_u = end.value // 10**9
    return pd.to_datetime(np.random.randint(start_u, end_u, n), unit='s')

# 1.1 users_df
num_users = np.random.randint(500, 701)
users_df = pd.DataFrame({
    'user_id': range(1, num_users + 1),
    'signup_date': random_dates(pd.Timestamp.now() - pd.DateOffset(years=5), pd.Timestamp.now() - pd.DateOffset(days=90), num_users),
    'age': np.random.randint(18, 71, num_users),
    'region': np.random.choice(['North', 'South', 'East', 'West', 'Central'], num_users, p=[0.2, 0.2, 0.2, 0.2, 0.2]),
    'user_segment': np.random.choice(['New', 'Regular', 'Power_User', 'Churn_Risk'], num_users, p=[0.25, 0.45, 0.2, 0.1]),
})

# 1.2 pre_campaign_activity_df
num_activities = np.random.randint(3000, 5001)
activity_user_ids = np.random.choice(users_df['user_id'], num_activities, replace=True)
activity_dates = []
for user_id in activity_user_ids:
    user_signup_date = users_df[users_df['user_id'] == user_id]['signup_date'].iloc[0]
    # Activities must be after signup date and before a conceptual campaign launch date (e.g., 90 days ago from now)
    activity_dates.append(random_dates(user_signup_date, pd.Timestamp.now() - pd.DateOffset(days=90), 1)[0])

pre_campaign_activity_df = pd.DataFrame({
    'activity_id': range(1, num_activities + 1),
    'user_id': activity_user_ids,
    'activity_date': activity_dates,
    'activity_type': np.random.choice(['login', 'view_dashboard', 'edit_profile', 'access_report', 'settings_update'], num_activities, p=[0.4, 0.25, 0.1, 0.15, 0.1]),
    'duration_seconds': np.random.uniform(10, 600, num_activities),
})

# Sort for chronological consistency check later
pre_campaign_activity_df = pre_campaign_activity_df.sort_values(by=['user_id', 'activity_date']).reset_index(drop=True)

# 1.3 campaign_exposure_df
num_exposed_users = np.random.randint(300, 501)
exposed_user_ids = np.random.choice(users_df['user_id'], num_exposed_users, replace=False)
campaign_start_window = pd.Timestamp.now() - pd.DateOffset(days=60) # Campaign in last 2 months
campaign_end_window = pd.Timestamp.now() - pd.DateOffset(days=30)
exposure_dates = random_dates(campaign_start_window, campaign_end_window, num_exposed_users)

campaign_exposure_df = pd.DataFrame({
    'exposure_id': range(1, num_exposed_users + 1),
    'user_id': exposed_user_ids,
    'exposure_date': exposure_dates,
    'campaign_variant': np.random.choice(['Control', 'Variant_A', 'Variant_B'], num_exposed_users, p=[0.4, 0.3, 0.3]),
})

# 1.4 post_campaign_feature_usage_df
num_usage_events = np.random.randint(800, 1201)

# Simulate realistic patterns: higher adoption for Variant A/B, Power Users, and more active users
exposed_users_with_variant = campaign_exposure_df.merge(users_df[['user_id', 'user_segment']], on='user_id', how='left')
# Calculate pre-campaign activity level for each exposed user (simplified for generation)
pre_activity_counts = pre_campaign_activity_df.groupby('user_id').size().reset_index(name='activity_count')
exposed_users_with_variant = exposed_users_with_variant.merge(pre_activity_counts, on='user_id', how='left').fillna({'activity_count': 0})

usage_user_ids = []
usage_dates = []
feature_names = np.random.choice(['New_Dashboard_Analytics', 'Improved_Search', 'AI_Assistant', 'Premium_Feature_X'], 4)

for _ in range(num_usage_events):
    # Select a user based on probabilities
    # Base probability for usage, normalized by max activity count to scale between 0 and 1
    max_activity = exposed_users_with_variant['activity_count'].max()
    base_prob = exposed_users_with_variant['activity_count'] / max_activity if max_activity > 0 else 0.1
    base_prob = np.clip(base_prob, 0.1, 0.8) # Ensure some base level

    # Adjust probability by variant
    variant_boost = exposed_users_with_variant['campaign_variant'].map({'Control': 0.8, 'Variant_A': 1.2, 'Variant_B': 1.1}).fillna(1)
    # Adjust probability by user segment
    segment_boost = exposed_users_with_variant['user_segment'].map({'New': 0.8, 'Regular': 1.0, 'Power_User': 1.5, 'Churn_Risk': 0.5}).fillna(1)
    
    # Combined probability - ensure it's not too high or too low
    weights = base_prob * variant_boost * segment_boost
    weights = np.clip(weights, 0.05, 1.0) # Cap probabilities

    # Normalize weights to sum to 1 for np.random.choice
    if weights.sum() == 0: # Fallback if all weights are zero
        selected_user_row = exposed_users_with_variant.sample(1).iloc[0]
    else:
        selected_user_row = exposed_users_with_variant.sample(1, weights=weights / weights.sum()).iloc[0]

    user_id = selected_user_row['user_id']
    exposure_date = selected_user_row['exposure_date']
    
    # Usage date must be after exposure_date and within a reasonable window (e.g., up to 90 days after exposure)
    usage_start = exposure_date + pd.DateOffset(days=1)
    usage_end = exposure_date + pd.DateOffset(days=90)
    
    # Ensure usage_start is not in the future
    if usage_start > pd.Timestamp.now():
        usage_start = pd.Timestamp.now() - pd.DateOffset(days=1) # Fallback to avoid future dates
    if usage_end > pd.Timestamp.now():
        usage_end = pd.Timestamp.now()
    
    # If the window is invalid, skip or adjust
    if usage_start >= usage_end:
        usage_dates.append(pd.Timestamp.now() - pd.DateOffset(days=np.random.randint(1, 30))) # Fallback date
    else:
        usage_dates.append(random_dates(usage_start, usage_end, 1)[0])
    
    usage_user_ids.append(user_id)

post_campaign_feature_usage_df = pd.DataFrame({
    'usage_id': range(1, len(usage_user_ids) + 1),
    'user_id': usage_user_ids,
    'usage_date': usage_dates,
    'feature_name': np.random.choice(feature_names, len(usage_user_ids), replace=True),
})

# Display generated data info
print("--- Synthetic DataFrames Generated ---")
print(f"users_df: {len(users_df)} rows")
print(f"pre_campaign_activity_df: {len(pre_campaign_activity_df)} rows")
print(f"campaign_exposure_df: {len(campaign_exposure_df)} rows")
print(f"post_campaign_feature_usage_df: {len(post_campaign_feature_usage_df)} rows\n")

# --- 2. Load into SQLite & SQL Feature Engineering ---

conn = sqlite3.connect(':memory:')
users_df.to_sql('users', conn, index=False, if_exists='replace')
pre_campaign_activity_df.to_sql('pre_campaign_activity', conn, index=False, if_exists='replace') # Corrected table name
campaign_exposure_df.to_sql('campaign_exposure', conn, index=False, if_exists='replace') # Corrected table name

# Determine overall campaign launch date (not strictly used in the query as requested, but good for context)
campaign_launch_date_pd = campaign_exposure_df['exposure_date'].min()
print(f"Overall Campaign Launch Date: {campaign_launch_date_pd.strftime('%Y-%m-%d')}\n")

sql_query = """
SELECT
    e.user_id,
    u.age,
    u.region,
    u.user_segment,
    u.signup_date,
    e.exposure_date,
    e.campaign_variant,
    COUNT(CASE WHEN a.activity_type = 'login' AND JULIANDAY(a.activity_date) < JULIANDAY(e.exposure_date) THEN 1 END) AS num_pre_campaign_logins,
    SUM(CASE WHEN JULIANDAY(a.activity_date) < JULIANDAY(e.exposure_date) THEN a.duration_seconds ELSE 0 END) AS total_pre_campaign_duration,
    MAX(CASE WHEN JULIANDAY(a.activity_date) < JULIANDAY(e.exposure_date)
             THEN JULIANDAY(e.exposure_date) - JULIANDAY(a.activity_date)
             ELSE NULL END) AS days_since_last_pre_campaign_activity
FROM
    campaign_exposure AS e
LEFT JOIN
    users AS u ON e.user_id = u.user_id
LEFT JOIN
    pre_campaign_activity AS a ON e.user_id = a.user_id
GROUP BY
    e.user_id, u.age, u.region, u.user_segment, u.signup_date, e.exposure_date, e.campaign_variant;
"""

campaign_features_df = pd.read_sql_query(sql_query, conn)
conn.close()

print("--- SQL Feature Engineering Results (First 5 rows) ---")
print(campaign_features_df.head())
print(f"\nDataFrame shape after SQL: {campaign_features_df.shape}\n")

# --- 3. Pandas Feature Engineering & Binary Target Creation ---

# Handle NaN values
campaign_features_df['num_pre_campaign_logins'] = campaign_features_df['num_pre_campaign_logins'].fillna(0).astype(int)
campaign_features_df['total_pre_campaign_duration'] = campaign_features_df['total_pre_campaign_duration'].fillna(0)
campaign_features_df['days_since_last_pre_campaign_activity'] = campaign_features_df['days_since_last_pre_campaign_activity'].fillna(9999).astype(int)

# Convert dates
campaign_features_df['signup_date'] = pd.to_datetime(campaign_features_df['signup_date'])
campaign_features_df['exposure_date'] = pd.to_datetime(campaign_features_df['exposure_date'])

# Calculate account_age_at_exposure_days
campaign_features_df['account_age_at_exposure_days'] = (campaign_features_df['exposure_date'] - campaign_features_df['signup_date']).dt.days

# Create Binary Target: adopted_new_feature
# Ensure post_campaign_feature_usage_df dates are datetime
post_campaign_feature_usage_df['usage_date'] = pd.to_datetime(post_campaign_feature_usage_df['usage_date'])

# Merge exposure info with potential usage
temp_df = campaign_features_df[['user_id', 'exposure_date']].merge(
    post_campaign_feature_usage_df[['user_id', 'usage_date']],
    on='user_id',
    how='left'
)

# Filter usage that happened within the 60-day post-exposure window
temp_df['adoption_window_start'] = temp_df['exposure_date']
temp_df['adoption_window_end'] = temp_df['exposure_date'] + pd.Timedelta(days=60)

adopted_users = temp_df[
    (temp_df['usage_date'] >= temp_df['adoption_window_start']) &
    (temp_df['usage_date'] < temp_df['adoption_window_end'])
]['user_id'].unique()

campaign_features_df['adopted_new_feature'] = campaign_features_df['user_id'].isin(adopted_users).astype(int)

print("--- Pandas Feature Engineering & Target Creation ---")
print(f"Adoption rate: {campaign_features_df['adopted_new_feature'].mean():.2f}")
print(campaign_features_df[['user_id', 'exposure_date', 'account_age_at_exposure_days', 'adopted_new_feature']].head())
print(f"\nFinal DataFrame shape: {campaign_features_df.shape}\n")

# Define features X and target y
numerical_features = [
    'age',
    'account_age_at_exposure_days',
    'num_pre_campaign_logins',
    'total_pre_campaign_duration',
    'days_since_last_pre_campaign_activity'
]
categorical_features = [
    'region',
    'user_segment',
    'campaign_variant'
]

X = campaign_features_df[numerical_features + categorical_features]
y = campaign_features_df['adopted_new_feature']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}, y_test shape: {y_test.shape}\n")


# --- 4. Data Visualization ---

plt.style.use('seaborn-v0_8-darkgrid')
plt.figure(figsize=(15, 6))

# Plot 1: Stacked bar chart of adoption by campaign_variant
plt.subplot(1, 2, 1)
adoption_by_variant = campaign_features_df.groupby('campaign_variant')['adopted_new_feature'].value_counts(normalize=True).unstack().fillna(0)
adoption_by_variant.plot(kind='bar', stacked=True, ax=plt.gca(), cmap='viridis')
plt.title('Feature Adoption Rate by Campaign Variant')
plt.xlabel('Campaign Variant')
plt.ylabel('Proportion')
plt.xticks(rotation=45)
plt.legend(title='Adopted Feature', labels=['Not Adopted', 'Adopted'])

# Plot 2: Violin plot of account_age_at_exposure_days by adopted_new_feature
plt.subplot(1, 2, 2)
sns.violinplot(x='adopted_new_feature', y='account_age_at_exposure_days', data=campaign_features_df, palette='pastel', ax=plt.gca())
plt.title('Distribution of Account Age at Exposure by Feature Adoption')
plt.xlabel('Adopted New Feature (0=No, 1=Yes)')
plt.ylabel('Account Age at Exposure (Days)')
plt.tight_layout()
plt.show()

# --- 5. ML Pipeline & Evaluation (Binary Classification) ---

print("--- ML Pipeline & Evaluation ---")

# Preprocessing pipelines for numerical and categorical features
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Create a ColumnTransformer to apply different transformations to different columns
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Create the full pipeline
model_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                 ('classifier', GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42))])

# Train the pipeline
model_pipeline.fit(X_train, y_train)

# Predict probabilities for the positive class (class 1) on the test set
y_pred_proba = model_pipeline.predict_proba(X_test)[:, 1]

# Predict class labels for classification report
y_pred = model_pipeline.predict(X_test)

# Calculate and print evaluation metrics
roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"ROC AUC Score on Test Set: {roc_auc:.4f}")

print("\nClassification Report on Test Set:")
print(classification_report(y_test, y_pred))

print("\n--- Script Finished Successfully ---")