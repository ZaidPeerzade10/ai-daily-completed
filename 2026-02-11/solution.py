import pandas as pd
import numpy as np
import datetime
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, classification_report

# --- 1. Generate Synthetic Data (Pandas/Numpy) ---
print("--- 1. Generating Synthetic Data ---")

# Define base dates for data generation to ensure reproducibility and logical time flow
np.random.seed(42)
today = pd.to_datetime('2023-11-01') # Fixed 'today' for consistent analysis window
start_date_range_min = today - pd.Timedelta(days=5 * 365) # 5 years ago
start_date_range_max = today - pd.Timedelta(days=3 * 365) # 3 years ago

# Users DataFrame
num_users = np.random.randint(500, 701)
user_ids = np.arange(num_users)
signup_dates = [
    start_date_range_min + pd.Timedelta(days=np.random.randint(0, (start_date_range_max - start_date_range_min).days))
    for _ in range(num_users)
]
regions = np.random.choice(['North', 'South', 'East', 'West'], num_users)
initial_plan_types = np.random.choice(['Free', 'Basic', 'Premium'], num_users, p=[0.4, 0.4, 0.2]) # More free/basic
churn_risk_scores = np.random.rand(num_users) # Hidden churn risk

users_df = pd.DataFrame({
    'user_id': user_ids,
    'signup_date': signup_dates,
    'region': regions,
    'initial_plan_type': initial_plan_types,
    'churn_risk_score': churn_risk_scores
})

# Sessions DataFrame - Simulate realistic churn patterns
all_sessions = []
session_counter = 0

# Define a time window for "recent activity" for non-churners vs churners
# Churners will have few or no sessions occurring within the last 3-6 months
churn_inactive_window_start = today - pd.Timedelta(days=np.random.randint(90, 181)) 

for _, user in users_df.iterrows():
    user_id = user['user_id']
    signup_date = user['signup_date']
    churn_risk_score = user['churn_risk_score']

    if churn_risk_score > 0.7: # High churn risk users
        # Fewer sessions, shorter duration, fewer pages, concentrated earlier in their history
        num_sessions = np.random.randint(1, 6) # 1-5 sessions
        
        # Ensure sessions are before the churn_inactive_window_start and after signup
        # Sessions for churners are concentrated earlier, e.g., within their first year, but before the inactive window.
        effective_session_end = min(churn_inactive_window_start, signup_date + pd.Timedelta(days=365))
        
        # Adjust if signup is too recent to have early sessions before the inactive window
        if effective_session_end <= signup_date:
            effective_session_end = signup_date + pd.Timedelta(days=30) # at least 30 days
            if effective_session_end > churn_inactive_window_start:
                 continue # Skip user if no valid early session window for churn simulation
            
        session_dates = [
            signup_date + pd.Timedelta(days=np.random.randint(0, (effective_session_end - signup_date).days))
            for _ in range(num_sessions)
        ]
        duration_minutes = np.random.uniform(5, 30, num_sessions) # 5-30 min
        num_pages_viewed = np.random.randint(1, 6, num_sessions) # 1-5 pages
    else: # Low churn risk users
        # More sessions, longer duration, more pages, spread out including recent times
        num_sessions = np.random.randint(5, 21) # 5-20 sessions
        session_dates = [
            signup_date + pd.Timedelta(days=np.random.randint(0, (today - signup_date).days))
            for _ in range(num_sessions)
        ]
        duration_minutes = np.random.uniform(30, 120, num_sessions) # 30-120 min
        num_pages_viewed = np.random.randint(5, 21, num_sessions) # 5-20 pages

    for i in range(num_sessions):
        all_sessions.append({
            'session_id': session_counter,
            'user_id': user_id,
            'session_start_time': session_dates[i],
            'duration_minutes': duration_minutes[i],
            'num_pages_viewed': num_pages_viewed[i]
        })
        session_counter += 1

sessions_df = pd.DataFrame(all_sessions)
sessions_df['session_start_time'] = pd.to_datetime(sessions_df['session_start_time'])

print(f"Generated {len(users_df)} users.")
print(f"Generated {len(sessions_df)} sessions.")
print("Users head:\n", users_df.head())
print("Sessions head:\n", sessions_df.head())


# --- 2. Load into SQLite & SQL Feature Engineering ---
print("\n--- 2. Loading into SQLite & SQL Feature Engineering ---")

conn = sqlite3.connect(':memory:')
users_df.to_sql('users', conn, index=False, if_exists='replace')
sessions_df.to_sql('sessions', conn, index=False, if_exists='replace')

# Determine global_analysis_date and feature_cutoff_date
global_analysis_date = sessions_df['session_start_time'].max() + pd.Timedelta(days=30)
feature_cutoff_date = global_analysis_date - pd.Timedelta(days=60)

print(f"Global Analysis Date: {global_analysis_date.strftime('%Y-%m-%d')}")
print(f"Feature Cutoff Date: {feature_cutoff_date.strftime('%Y-%m-%d')}")

# SQL query for feature engineering
# Note: SQLite stores dates as strings, so comparisons need to be against string formats.
# `strftime` is used for date conversion in SQL.
# COALESCE handles NULLs from LEFT JOIN for aggregate functions.
# CASE WHEN handles NULL for days_since_last_session_pre_cutoff for users with no pre-cutoff sessions.
sql_query = f"""
SELECT
    u.user_id,
    u.region,
    u.initial_plan_type,
    u.signup_date,
    COALESCE(COUNT(s.session_id), 0) AS total_sessions_pre_cutoff,
    COALESCE(AVG(s.duration_minutes), 0.0) AS avg_session_duration_pre_cutoff,
    COALESCE(SUM(s.num_pages_viewed), 0) AS total_pages_viewed_pre_cutoff,
    CASE
        WHEN MAX(s.session_start_time) IS NULL THEN NULL
        ELSE JULIANDAY('{feature_cutoff_date.strftime('%Y-%m-%d %H:%M:%S')}') - JULIANDAY(MAX(s.session_start_time))
    END AS days_since_last_session_pre_cutoff
FROM
    users AS u
LEFT JOIN
    sessions AS s ON u.user_id = s.user_id
WHERE
    s.session_start_time IS NULL OR s.session_start_time < '{feature_cutoff_date.strftime('%Y-%m-%d %H:%M:%S')}'
GROUP BY
    u.user_id, u.region, u.initial_plan_type, u.signup_date
ORDER BY
    u.user_id;
"""

user_features_df = pd.read_sql_query(sql_query, conn)
conn.close() # Close SQLite connection after fetching data

print("User features DataFrame head (from SQL):\n", user_features_df.head())
print(f"Number of users in features DF: {len(user_features_df)}")


# --- 3. Pandas Feature Engineering & Binary Target Creation ---
print("\n--- 3. Pandas Feature Engineering & Binary Target Creation ---")

# Convert signup_date to datetime objects
user_features_df['signup_date'] = pd.to_datetime(user_features_df['signup_date'])

# Handle NaN values for features from SQL query
user_features_df['total_sessions_pre_cutoff'] = user_features_df['total_sessions_pre_cutoff'].fillna(0).astype(int)
user_features_df['total_pages_viewed_pre_cutoff'] = user_features_df['total_pages_viewed_pre_cutoff'].fillna(0).astype(int)
user_features_df['avg_session_duration_pre_cutoff'] = user_features_df['avg_session_duration_pre_cutoff'].fillna(0.0)

# Calculate account_age_at_cutoff_days
user_features_df['account_age_at_cutoff_days'] = (feature_cutoff_date - user_features_df['signup_date']).dt.days

# Fill days_since_last_session_pre_cutoff with a sentinel value
# For users with no sessions before cutoff, their 'days_since_last_session_pre_cutoff' will be NaN from SQL.
# A large sentinel value indicates a very long time since the last session, e.g., account age + 30 days.
user_features_df['days_since_last_session_pre_cutoff'] = user_features_df['days_since_last_session_pre_cutoff'].fillna(
    user_features_df['account_age_at_cutoff_days'] + 30
)
user_features_df['days_since_last_session_pre_cutoff'] = user_features_df['days_since_last_session_pre_cutoff'].astype(float)


# Create the Binary Target `is_churned_future`
# A user is considered churned (1) if they have NO sessions in the original `sessions_df`
# in the period BETWEEN `feature_cutoff_date` and `global_analysis_date`. Otherwise, 0.
future_sessions_df = sessions_df[
    (sessions_df['session_start_time'] >= feature_cutoff_date) &
    (sessions_df['session_start_time'] < global_analysis_date)
]
active_users_in_future = future_sessions_df['user_id'].unique()

user_features_df['is_churned_future'] = user_features_df['user_id'].apply(
    lambda x: 0 if x in active_users_in_future else 1
)

print("User features DataFrame with target:\n", user_features_df.head())
print("Churn status distribution:\n", user_features_df['is_churned_future'].value_counts(normalize=True))

# Define features X and target y
numerical_features = [
    'account_age_at_cutoff_days',
    'total_sessions_pre_cutoff',
    'avg_session_duration_pre_cutoff',
    'total_pages_viewed_pre_cutoff',
    'days_since_last_session_pre_cutoff'
]
categorical_features = ['region', 'initial_plan_type']

X = user_features_df[numerical_features + categorical_features]
y = user_features_df['is_churned_future']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"\nX_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")


# --- 4. Data Visualization ---
print("\n--- 4. Data Visualization ---")
plt.figure(figsize=(14, 6))

# Plot 1: Violin plot of total_sessions_pre_cutoff vs. is_churned_future
plt.subplot(1, 2, 1)
sns.violinplot(x='is_churned_future', y='total_sessions_pre_cutoff', data=user_features_df)
plt.title('Total Sessions Pre-Cutoff Distribution by Churn Status')
plt.xlabel('Is Churned Future (0=No, 1=Yes)')
plt.ylabel('Total Sessions Pre-Cutoff')

# Plot 2: Stacked bar chart of initial_plan_type vs. is_churned_future proportion
plt.subplot(1, 2, 2)
churn_by_plan = user_features_df.groupby('initial_plan_type')['is_churned_future'].value_counts(normalize=True).unstack()
churn_by_plan.plot(kind='bar', stacked=True, ax=plt.gca(), cmap='viridis')
plt.title('Churn Proportion by Initial Plan Type')
plt.xlabel('Initial Plan Type')
plt.ylabel('Proportion')
plt.legend(title='Is Churned Future', labels=['Not Churned', 'Churned'])
plt.xticks(rotation=45, ha='right')

plt.tight_layout()
plt.show()


# --- 5. ML Pipeline & Evaluation (Binary Classification) ---
print("\n--- 5. ML Pipeline & Evaluation (Binary Classification) ---")

# Create a ColumnTransformer for preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ]), numerical_features),
        ('cat', Pipeline([
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ]), categorical_features)
    ],
    remainder='passthrough' # Keep other columns (if any)
)

# Create the ML pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', GradientBoostingClassifier(random_state=42, n_estimators=100, learning_rate=0.1))
])

# Train the pipeline
print("Training the ML pipeline...")
pipeline.fit(X_train, y_train)
print("Pipeline trained.")

# Predict probabilities for the positive class (class 1) on the test set
y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

# Calculate ROC AUC Score
roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"\nROC AUC Score on Test Set: {roc_auc:.4f}")

# Convert probabilities to binary predictions for classification report (using default 0.5 threshold)
y_pred = (y_pred_proba > 0.5).astype(int)

# Print Classification Report
print("\nClassification Report on Test Set:")
print(classification_report(y_test, y_pred))

print("\n--- Script Finished ---")