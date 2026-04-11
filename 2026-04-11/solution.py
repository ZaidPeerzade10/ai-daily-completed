import pandas as pd
import numpy as np
import datetime
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer # Corrected import path
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, classification_report

# Ensure reproducibility
np.random.seed(42)

# --- 1. Generate Synthetic Data ---

# User data generation
N_USERS = np.random.randint(500, 701)
signup_dates = pd.to_datetime('2021-01-01') + pd.to_timedelta(np.random.randint(0, 3 * 365, N_USERS), unit='D')
current_plans = np.random.choice(['Free', 'Basic', 'Pro'], size=N_USERS, p=[0.5, 0.3, 0.2])
regions = np.random.choice(['North', 'South', 'East', 'West'], size=N_USERS)
industries = np.random.choice(['Tech', 'Finance', 'Retail', 'Education'], size=N_USERS)

users_df = pd.DataFrame({
    'user_id': np.arange(N_USERS),
    'signup_date': signup_dates,
    'current_plan': current_plans,
    'region': regions,
    'industry': industries
})

# Define plan hierarchy for logic and mapping
plan_map = {'Free': 0, 'Basic': 1, 'Pro': 2}
plan_rev_map = {0: 'Free', 1: 'Basic', 2: 'Pro'} # For potential reverse mapping

# Simulate target plan with biases
def determine_target_plan(row):
    current_level = plan_map[row['current_plan']]
    
    # Pro users do not upgrade further
    if row['current_plan'] == 'Pro':
        return 'Pro'

    upgrade_chance = np.random.rand()
    
    if row['current_plan'] == 'Free':
        if upgrade_chance < 0.12:  # ~12% upgrade from Free
            return np.random.choice(['Basic', 'Pro'], p=[0.8, 0.2]) # Majority to Basic
        else:
            return 'Free'
    elif row['current_plan'] == 'Basic':
        if upgrade_chance < 0.08:  # ~8% upgrade from Basic
            return 'Pro'
        else:
            return 'Basic'
    return row['current_plan'] # Fallback, should not be reached

users_df['target_plan'] = users_df.apply(determine_target_plan, axis=1)

# Create a temporary 'will_upgrade' flag for app event generation bias
users_df['will_upgrade'] = users_df.apply(
    lambda row: plan_map[row['target_plan']] > plan_map[row['current_plan']], axis=1
)

# App event data generation
N_EVENTS = np.random.randint(15000, 25001)
all_events = []
event_id_counter = 0

# Base probabilities for event types
event_type_probs_base = {
    'dashboard_view': 0.45,
    'report_run': 0.2,
    'api_call': 0.1,
    'settings_change': 0.2,
    'support_chat': 0.05
}

for _, user_row in users_df.iterrows():
    user_id = user_row['user_id']
    signup_date = user_row['signup_date']
    current_plan = user_row['current_plan']
    will_upgrade = user_row['will_upgrade']
    
    # Number of events per user, biased by plan
    # Free users have fewer events, Pro users have more
    num_user_events = np.random.randint(10, 80) # Base number of events
    if current_plan == 'Pro':
        num_user_events = np.random.randint(40, 150)
    elif current_plan == 'Basic':
        num_user_events = np.random.randint(20, 100)
    
    # Generate events within a window after signup (e.g., up to 150 days)
    max_event_window_days = 150 
    
    for _ in range(num_user_events):
        event_timestamp_offset_days = np.random.randint(1, max_event_window_days + 1)
        timestamp = signup_date + pd.Timedelta(event_timestamp_offset_days, unit='D')
        
        event_type_probs = event_type_probs_base.copy()
        duration_seconds = 0
        
        # Bias event types based on current_plan
        if current_plan == 'Pro':
            event_type_probs['api_call'] += 0.15
            event_type_probs['report_run'] += 0.10
            event_type_probs['dashboard_view'] -= 0.25 
        elif current_plan == 'Free':
            event_type_probs['dashboard_view'] += 0.15
            event_type_probs['api_call'] -= 0.05
            event_type_probs['report_run'] -= 0.05
        
        # Normalize probabilities to ensure they sum to 1
        event_type_probs_sum = sum(event_type_probs.values())
        event_type_probs = {k: v / event_type_probs_sum for k, v in event_type_probs.items()}

        event_type = np.random.choice(list(event_type_probs.keys()), p=list(event_type_probs.values()))
        
        # Introduce bias for upgraders in their first 30 days
        is_first_30_days = (timestamp <= signup_date + pd.Timedelta(30, unit='D'))
        
        if will_upgrade and is_first_30_days:
            if event_type in ['report_run', 'api_call']:
                duration_seconds = np.random.randint(120, 600) # Higher duration for premium features
            elif event_type == 'support_chat':
                duration_seconds = np.random.randint(30, 300)
            else:
                duration_seconds = np.random.randint(5, 120) # Some duration for other events too
        else:
            if event_type in ['api_call', 'report_run']:
                duration_seconds = np.random.randint(30, 300)
            elif event_type == 'support_chat':
                duration_seconds = np.random.randint(10, 180)
            else:
                duration_seconds = 0 # Most other events have no explicit duration

        all_events.append({
            'event_id': event_id_counter,
            'user_id': user_id,
            'timestamp': timestamp,
            'event_type': event_type,
            'duration_seconds': duration_seconds
        })
        event_id_counter += 1

app_events_df = pd.DataFrame(all_events)
# Sort for easier sequential processing in theory, not strictly needed for the SQL query here
app_events_df = app_events_df.sort_values(by=['user_id', 'timestamp']).reset_index(drop=True)

# Drop the temporary 'will_upgrade' column from users_df before SQL load
users_df = users_df.drop(columns=['will_upgrade'])


# --- 2. Load into SQLite & SQL Feature Engineering (Early User Behavior) ---

# Create an in-memory SQLite database
conn = sqlite3.connect(':memory:')

# Load DataFrames into SQLite tables
users_df.to_sql('users', conn, index=False, if_exists='replace')
app_events_df.to_sql('app_events', conn, index=False, if_exists='replace')

# SQL query to aggregate early user behavior
sql_query = """
SELECT
    u.user_id,
    u.signup_date,
    u.current_plan,
    u.region,
    u.industry,
    u.target_plan,
    COALESCE(SUM(CASE WHEN e.event_id IS NOT NULL THEN 1 ELSE 0 END), 0) AS num_events_first_30d,
    COALESCE(SUM(e.duration_seconds), 0) AS total_engagement_duration_first_30d,
    COALESCE(SUM(CASE WHEN e.event_type = 'report_run' THEN 1 ELSE 0 END), 0) AS num_report_runs_first_30d,
    COALESCE(SUM(CASE WHEN e.event_type = 'api_call' THEN 1 ELSE 0 END), 0) AS num_api_calls_first_30d,
    COALESCE(COUNT(DISTINCT DATE(e.timestamp)), 0) AS days_with_activity_first_30d,
    COALESCE(MAX(CASE WHEN e.event_type = 'support_chat' THEN 1 ELSE 0 END), 0) AS has_used_support_first_30d
FROM
    users AS u
LEFT JOIN
    app_events AS e
ON
    u.user_id = e.user_id AND e.timestamp <= DATE(u.signup_date, '+30 days')
GROUP BY
    u.user_id, u.signup_date, u.current_plan, u.region, u.industry, u.target_plan
ORDER BY
    u.user_id
"""

user_early_features_df = pd.read_sql_query(sql_query, conn)
conn.close()

# Ensure correct data types after SQL fetch (COALESCE typically handles nulls to 0, but explicit cast for safety)
int_cols_to_convert = [
    'num_events_first_30d',
    'num_report_runs_first_30d',
    'num_api_calls_first_30d',
    'days_with_activity_first_30d',
    'has_used_support_first_30d'
]
for col in int_cols_to_convert:
    user_early_features_df[col] = user_early_features_df[col].astype(int)

# Ensure duration is float
user_early_features_df['total_engagement_duration_first_30d'] = user_early_features_df['total_engagement_duration_first_30d'].astype(float)


# --- 3. Pandas Feature Engineering & Binary Target Creation ---

# Convert signup_date to datetime objects
user_early_features_df['signup_date'] = pd.to_datetime(user_early_features_df['signup_date'])

# Calculate `activity_frequency_first_30d`
user_early_features_df['activity_frequency_first_30d'] = user_early_features_df['num_events_first_30d'] / 30.0
user_early_features_df['activity_frequency_first_30d'] = user_early_features_df['activity_frequency_first_30d'].fillna(0)

# Calculate `premium_feature_usage_ratio_first_30d` (add 1 to denominator to prevent division by zero)
user_early_features_df['premium_feature_usage_ratio_first_30d'] = \
    (user_early_features_df['num_report_runs_first_30d'] + user_early_features_df['num_api_calls_first_30d']) / \
    (user_early_features_df['num_events_first_30d'] + 1)
user_early_features_df['premium_feature_usage_ratio_first_30d'] = user_early_features_df['premium_feature_usage_ratio_first_30d'].fillna(0)

# Create the Binary Target `will_upgrade_90d`
user_early_features_df['will_upgrade_90d'] = user_early_features_df.apply(
    lambda row: 1 if plan_map[row['target_plan']] > plan_map[row['current_plan']] else 0,
    axis=1
)

# Define features (X) and target (y)
numerical_features = [
    'num_events_first_30d',
    'total_engagement_duration_first_30d',
    'num_report_runs_first_30d',
    'num_api_calls_first_30d',
    'days_with_activity_first_30d',
    'activity_frequency_first_30d',
    'premium_feature_usage_ratio_first_30d'
]
categorical_features = [
    'current_plan',
    'region',
    'industry',
    'has_used_support_first_30d' # This is binary, but OneHotEncoder handles it well without issue
]

X = user_early_features_df[numerical_features + categorical_features]
y = user_early_features_df['will_upgrade_90d']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

print(f"Total users: {len(user_early_features_df)}")
print(f"Upgrade rate: {y.mean():.2%}")
print(f"Train set size: {len(X_train)} (Upgrade rate: {y_train.mean():.2%})")
print(f"Test set size: {len(X_test)} (Upgrade rate: {y_test.mean():.2%})")

# --- 4. Data Visualization ---

plt.style.use('seaborn-v0_8-darkgrid')
plt.figure(figsize=(14, 6))

# Violin Plot: Engagement Duration Distribution by Upgrade Status
plt.subplot(1, 2, 1)
sns.violinplot(x='will_upgrade_90d', y='total_engagement_duration_first_30d', data=user_early_features_df, palette='pastel')
plt.title('Engagement Duration Distribution by Upgrade Status', fontsize=14)
plt.xlabel('Will Upgrade (0: No, 1: Yes)', fontsize=12)
plt.ylabel('Total Engagement Duration (seconds) - First 30 Days', fontsize=12)
plt.xticks([0, 1], ['No Upgrade', 'Upgrade'], fontsize=10)
plt.yticks(fontsize=10)

# Stacked Bar Chart: Upgrade Proportion by Current Plan
plt.subplot(1, 2, 2)
current_plan_upgrade_proportions = pd.crosstab(
    user_early_features_df['current_plan'],
    user_early_features_df['will_upgrade_90d'],
    normalize='index'
).sort_values(by=1, ascending=False) # Sort by upgrade rate for better comparison
current_plan_upgrade_proportions.plot(kind='bar', stacked=True, ax=plt.gca(), cmap='viridis')
plt.title('Upgrade Proportion by Current Plan', fontsize=14)
plt.xlabel('Current Plan', fontsize=12)
plt.ylabel('Proportion', fontsize=12)
plt.legend(title='Will Upgrade', labels=['No', 'Yes'], fontsize=10, title_fontsize=10)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(fontsize=10)

plt.tight_layout()
plt.show()


# --- 5. ML Pipeline & Evaluation (Binary Classification) ---

# Create a ColumnTransformer for preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ]), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='passthrough' # Keep any other columns, though none are expected here
)

# Create the ML pipeline
# HistGradientBoostingClassifier typically does not strictly require scaling,
# but it's included for numerical features as per task requirements and good practice.
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', HistGradientBoostingClassifier(random_state=42))
])

# Train the pipeline
print("\nTraining the ML model...")
model_pipeline.fit(X_train, y_train)
print("Model training complete.")

# Predict probabilities for the positive class (class 1: upgrade) on the test set
y_pred_proba = model_pipeline.predict_proba(X_test)[:, 1]

# Calculate and print ROC AUC Score
roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"\nROC AUC Score on Test Set: {roc_auc:.4f}")

# Predict classes for the classification report (using default threshold 0.5)
y_pred = model_pipeline.predict(X_test)

# Print Classification Report
print("\nClassification Report on Test Set:")
print(classification_report(y_test, y_pred, target_names=['No Upgrade', 'Upgrade']))