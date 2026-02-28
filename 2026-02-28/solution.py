import numpy as np
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer # Corrected import based on feedback
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, classification_report

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')

# --- 1. Generate Synthetic Data ---

# Seed for reproducibility
np.random.seed(42)

# 1.1 users_df
num_users = np.random.randint(500, 701)
signup_dates = pd.to_datetime('now') - pd.to_timedelta(np.random.rand(num_users) * 5 * 365, unit='D')
user_segments = np.random.choice(['New', 'Regular', 'VIP'], num_users, p=[0.3, 0.5, 0.2])
device_types = np.random.choice(['Mobile', 'Desktop', 'Tablet'], num_users, p=[0.6, 0.3, 0.1])

users_df = pd.DataFrame({
    'user_id': np.arange(num_users),
    'signup_date': signup_dates,
    'user_segment': user_segments,
    'device_type': device_types
})

# 1.2 sessions_df
num_sessions = np.random.randint(3000, 5001)
session_ids = np.arange(num_sessions)
session_user_ids = np.random.choice(users_df['user_id'], num_sessions)
session_duration_seconds = np.random.randint(30, 1801, num_sessions) # 0.5 to 30 minutes

sessions_df = pd.DataFrame({
    'session_id': session_ids,
    'user_id': session_user_ids,
    'session_duration_seconds': session_duration_seconds
})

# Ensure session_start_date is after signup_date
sessions_df = sessions_df.merge(users_df[['user_id', 'signup_date']], on='user_id', how='left', suffixes=('', '_user'))

# Generate session start dates within a reasonable window after signup
max_days_after_signup = 365 * 4 # Sessions can start up to 4 years after signup
sessions_df['session_start_date'] = sessions_df.apply(
    lambda row: row['signup_date'] + pd.to_timedelta(np.random.rand() * max_days_after_signup, unit='D'),
    axis=1
)
# Ensure no session starts in the future, cap at current time
sessions_df['session_start_date'] = sessions_df['session_start_date'].apply(lambda d: min(d, pd.Timestamp.now()))
sessions_df = sessions_df.drop(columns=['signup_date']) # Clean up temp signup_date

# Generate is_high_value_session, simulating influence from user_segment and session_duration
sessions_df = sessions_df.merge(users_df[['user_id', 'user_segment']], on='user_id', how='left')
sessions_df['is_high_value_session'] = 0

# Base probability for high value session
base_prob = 0.05

# Influence by user segment
vip_idx = sessions_df['user_segment'] == 'VIP'
sessions_df.loc[vip_idx, 'is_high_value_session'] = np.random.binomial(1, base_prob * 3, size=vip_idx.sum())

# Influence by session duration (e.g., sessions longer than 15 mins)
long_session_idx = sessions_df['session_duration_seconds'] > 900
sessions_df.loc[long_session_idx, 'is_high_value_session'] = np.random.binomial(1, base_prob * 2, size=long_session_idx.sum())

# Fill remaining sessions with base probability, ensuring no overlap for probability assignment
# (i.e., if a session was already set to 1 by VIP or long_session rule, it stays 1)
remaining_idx_to_fill = sessions_df['is_high_value_session'] == 0
sessions_df.loc[remaining_idx_to_fill, 'is_high_value_session'] = np.random.binomial(1, base_prob, size=remaining_idx_to_fill.sum())

# Adjust to ensure target proportion (10-20%)
target_high_value_count = int(sessions_df.shape[0] * np.random.uniform(0.10, 0.20))
current_high_value_count = sessions_df['is_high_value_session'].sum()

if current_high_value_count < target_high_value_count:
    diff = target_high_value_count - current_high_value_count
    low_value_indices = sessions_df[sessions_df['is_high_value_session'] == 0].index
    if len(low_value_indices) > 0:
        indices_to_change = np.random.choice(low_value_indices, min(diff, len(low_value_indices)), replace=False)
        sessions_df.loc[indices_to_change, 'is_high_value_session'] = 1
elif current_high_value_count > target_high_value_count:
    diff = current_high_value_count - target_high_value_count
    high_value_indices = sessions_df[sessions_df['is_high_value_session'] == 1].index
    if len(high_value_indices) > 0:
        indices_to_change = np.random.choice(high_value_indices, min(diff, len(high_value_indices)), replace=False)
        sessions_df.loc[indices_to_change, 'is_high_value_session'] = 0

# Drop temporary user_segment used for target generation
sessions_df = sessions_df.drop(columns=['user_segment'])


# 1.3 events_df
num_events = np.random.randint(10000, 15001)
event_ids = np.arange(num_events)
# Sample session IDs, potentially biasing towards existing sessions for more realistic event distribution
event_session_ids = np.random.choice(sessions_df['session_id'], num_events, 
                                     p=np.bincount(sessions_df['session_id'], minlength=len(sessions_df)) / len(sessions_df['session_id']))

event_types = np.random.choice(['page_view', 'add_to_cart', 'checkout', 'search', 'filter'], num_events, p=[0.6, 0.15, 0.05, 0.1, 0.1])
event_values = np.where(
    np.isin(event_types, ['add_to_cart', 'checkout']),
    np.random.uniform(0, 100, num_events),
    np.nan
)

events_df = pd.DataFrame({
    'event_id': event_ids,
    'session_id': event_session_ids,
    'event_type': event_types,
    'event_value': event_values
})

# Merge session data for event_timestamp generation
events_df = events_df.merge(sessions_df[['session_id', 'session_start_date', 'session_duration_seconds']], on='session_id', how='left')

# Generate event_timestamp within session duration
events_df['event_timestamp_offset'] = np.random.uniform(0, 1, events_df.shape[0]) * events_df['session_duration_seconds']
events_df['event_timestamp'] = events_df['session_start_date'] + pd.to_timedelta(events_df['event_timestamp_offset'], unit='S')
events_df['event_timestamp'] = events_df['event_timestamp'].dt.floor('S') # Round to nearest second

# Clean up temporary columns from events_df
events_df = events_df.drop(columns=['session_start_date', 'session_duration_seconds', 'event_timestamp_offset'])


print(f"Generated {len(users_df)} users, {len(sessions_df)} sessions, {len(events_df)} events.")
print(f"High value sessions: {sessions_df['is_high_value_session'].sum()} ({sessions_df['is_high_value_session'].mean():.2%})")

# --- 2. Load into SQLite & SQL Feature Engineering ---

conn = sqlite3.connect(':memory:')

users_df.to_sql('users', conn, index=False, if_exists='replace')
sessions_df.to_sql('sessions', conn, index=False, if_exists='replace')
events_df.to_sql('events', conn, index=False, if_exists='replace')

sql_query = """
SELECT
    s.session_id,
    s.user_id,
    s.session_start_date,
    s.session_duration_seconds,
    u.user_segment,
    u.device_type,
    s.is_high_value_session,
    COUNT(e.event_id) AS num_events_in_session,
    SUM(CASE WHEN e.event_type = 'page_view' THEN 1 ELSE 0 END) AS num_page_views,
    SUM(CASE WHEN e.event_type = 'add_to_cart' THEN 1 ELSE 0 END) AS num_add_to_carts,
    SUM(CASE WHEN e.event_type = 'checkout' THEN 1 ELSE 0 END) AS num_checkouts,
    SUM(e.event_value) AS total_event_value,
    -- Calculate time to first event in seconds, NULL if no events
    CAST((julianday(MIN(e.event_timestamp)) - julianday(s.session_start_date)) * 24 * 60 * 60 AS INTEGER) AS time_to_first_event_seconds,
    -- Calculate time to last event in seconds, NULL if no events
    CAST((julianday(MAX(e.event_timestamp)) - julianday(s.session_start_date)) * 24 * 60 * 60 AS INTEGER) AS time_to_last_event_seconds
FROM
    sessions s
LEFT JOIN
    users u ON s.user_id = u.user_id
LEFT JOIN
    events e ON s.session_id = e.session_id
GROUP BY
    s.session_id, s.user_id, s.session_start_date, s.session_duration_seconds,
    u.user_segment, u.device_type, s.is_high_value_session
ORDER BY s.session_id;
"""

session_features_df = pd.read_sql_query(sql_query, conn)
conn.close()

print(f"\nSQL Feature Engineering completed. Shape of session_features_df: {session_features_df.shape}")
print("Sample of session_features_df after SQL aggregation:")
print(session_features_df.head())


# --- 3. Pandas Feature Engineering & Binary Target Creation ---

# Convert date columns back to datetime objects if needed (read_sql might convert to string)
session_features_df['session_start_date'] = pd.to_datetime(session_features_df['session_start_date'])

# Handle NaN values from LEFT JOIN (sessions with no events)
session_features_df['num_events_in_session'] = session_features_df['num_events_in_session'].fillna(0).astype(int)
session_features_df['num_page_views'] = session_features_df['num_page_views'].fillna(0).astype(int)
session_features_df['num_add_to_carts'] = session_features_df['num_add_to_carts'].fillna(0).astype(int)
session_features_df['num_checkouts'] = session_features_df['num_checkouts'].fillna(0).astype(int)
session_features_df['total_event_value'] = session_features_df['total_event_value'].fillna(0.0)

# Fill time-related NaNs for sessions with no events
# time_to_first_event_seconds: If no events, assume it never happened, use session duration as sentinel
session_features_df['time_to_first_event_seconds'] = session_features_df['time_to_first_event_seconds'].fillna(session_features_df['session_duration_seconds'])
# time_to_last_event_seconds: If no events, assume 0.0 (event at start of session conceptually for a non-event session)
session_features_df['time_to_last_event_seconds'] = session_features_df['time_to_last_event_seconds'].fillna(0.0)

# Calculate new features
session_features_df['event_density_per_second'] = session_features_df['num_events_in_session'] / (session_features_df['session_duration_seconds'] + 1) # +1 to prevent division by zero
session_features_df['checkout_rate_in_session'] = session_features_df['num_checkouts'] / (session_features_df['num_add_to_carts'] + 1) # +1 to prevent division by zero

# Define features X and target y
numerical_features = [
    'session_duration_seconds',
    'num_events_in_session',
    'num_page_views',
    'num_add_to_carts',
    'num_checkouts',
    'total_event_value',
    'time_to_first_event_seconds',
    'time_to_last_event_seconds',
    'event_density_per_second',
    'checkout_rate_in_session'
]
categorical_features = ['user_segment', 'device_type']

X = session_features_df[numerical_features + categorical_features]
y = session_features_df['is_high_value_session']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

print(f"\nData split into training ({len(X_train)} samples) and testing ({len(X_test)} samples).")
print(f"Target distribution in training set:\n{y_train.value_counts(normalize=True)}")
print(f"Target distribution in test set:\n{y_test.value_counts(normalize=True)}")


# --- 4. Data Visualization ---

print("\nGenerating visualizations...")

# Violin plot of session_duration_seconds for high-value vs. low-value sessions
plt.figure(figsize=(10, 6))
sns.violinplot(x='is_high_value_session', y='session_duration_seconds', data=session_features_df)
plt.title('Session Duration Distribution by High-Value Status')
plt.xlabel('Is High Value Session (0=No, 1=Yes)')
plt.ylabel('Session Duration (Seconds)')
plt.show()

# Stacked bar chart of high-value proportion across user segments
segment_proportions = session_features_df.groupby('user_segment')['is_high_value_session'].value_counts(normalize=True).unstack()
segment_proportions.plot(kind='bar', stacked=True, figsize=(10, 6), cmap='viridis')
plt.title('Proportion of High-Value Sessions by User Segment')
plt.xlabel('User Segment')
plt.ylabel('Proportion')
plt.legend(title='Is High Value Session', labels=['Low Value (0)', 'High Value (1)'])
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# --- 5. ML Pipeline & Evaluation ---

print("\nBuilding and evaluating ML Pipeline...")

# Create preprocessing pipelines for numerical and categorical features
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = OneHotEncoder(handle_unknown='ignore')

# Create a column transformer to apply different transformations to different columns
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Create the full pipeline with HistGradientBoostingClassifier
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', HistGradientBoostingClassifier(random_state=42))
])

# Train the pipeline
model_pipeline.fit(X_train, y_train)

# Predict probabilities for the positive class (class 1) on the test set
y_pred_proba = model_pipeline.predict_proba(X_test)[:, 1]
# Predict hard labels for classification report
y_pred = model_pipeline.predict(X_test) 

# Calculate and print ROC AUC Score
roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"\nROC AUC Score on Test Set: {roc_auc:.4f}")

# Print Classification Report
print("\nClassification Report on Test Set:")
print(classification_report(y_test, y_pred))

print("\nScript execution complete.")