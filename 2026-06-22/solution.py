import pandas as pd
import numpy as np
import datetime
import sqlite3
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress warnings for cleaner output in a single script context
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# --- 1. Synthetic Data Generation ---
print("--- 1. Generating Synthetic Data ---")

np.random.seed(42)
num_users = np.random.randint(1000, 1500)
num_pages = np.random.randint(50, 100)
num_sessions = np.random.randint(20000, 30000)

# Users DataFrame
signup_dates = pd.to_datetime(pd.Timestamp.now() - pd.to_timedelta(np.random.randint(3*365, 5*365, num_users), unit='D'))
device_types = np.random.choice(['Mobile', 'Desktop', 'Tablet'], num_users, p=[0.6, 0.3, 0.1])
browsers = np.random.choice(['Chrome', 'Firefox', 'Safari', 'Edge'], num_users, p=[0.5, 0.25, 0.15, 0.1])
users_df = pd.DataFrame({
    'user_id': np.arange(1, num_users + 1),
    'signup_date': signup_dates,
    'device_type': device_types,
    'browser': browsers
})

# Pages DataFrame
page_types = ['Landing', 'Product', 'Informational', 'ContactUs', 'Blog']
pages_df = pd.DataFrame({
    'page_id': np.arange(1, num_pages + 1),
    'page_name': [f'Page_{i}' for i in range(1, num_pages + 1)],
    'page_type': np.random.choice(page_types, num_pages)
})

# Sessions DataFrame
session_user_ids = np.random.choice(users_df['user_id'], num_sessions)
session_landing_page_ids = np.random.choice(pages_df['page_id'], num_sessions)

sessions_raw_data = []
for i in range(num_sessions):
    user_id = session_user_ids[i]
    
    # Get user and page details for session simulation
    user_row = users_df[users_df['user_id'] == user_id].iloc[0]
    user_signup_date = user_row['signup_date']
    device_type = user_row['device_type']
    browser = user_row['browser']

    page_type = pages_df[pages_df['page_id'] == session_landing_page_ids[i]]['page_type'].iloc[0]
    
    # Ensure session starts after signup and within a reasonable timeframe (up to 3 years ago)
    start_datetime_candidate = pd.Timestamp.now() - pd.to_timedelta(np.random.randint(1, 365*3), unit='D') 
    while start_datetime_candidate <= user_signup_date:
        start_datetime_candidate = pd.Timestamp.now() - pd.to_timedelta(np.random.randint(1, 365*3), unit='D')
    
    had_conversion = np.random.rand() < 0.12 # Base conversion rate ~12%
    
    session_duration_seconds = 0.0 # Initialize

    # Longer duration for converted sessions
    if had_conversion:
        session_duration_seconds = np.random.uniform(120, 1200)
    else:
        session_duration_seconds = np.random.uniform(1, 600)
        # Introduce some very short sessions that are likely bounces if not converted
        if np.random.rand() < 0.3: # 30% chance for very short non-converted session
            session_duration_seconds = np.random.uniform(1, 30)
            
    # Add some noise based on device/browser/page for realism and correlation
    if device_type == 'Mobile' and not had_conversion:
        session_duration_seconds *= np.random.uniform(0.7, 1.2)
    if browser == 'Safari' and not had_conversion:
        session_duration_seconds *= np.random.uniform(0.8, 1.1)
    if page_type == 'ContactUs' and not had_conversion:
        session_duration_seconds *= np.random.uniform(0.5, 1.0)
    
    sessions_raw_data.append({
        'session_id': i + 1,
        'user_id': user_id,
        'start_datetime': start_datetime_candidate,
        'landing_page_id': session_landing_page_ids[i],
        'session_duration_seconds': max(1.0, round(session_duration_seconds, 2)), # Ensure min 1 second
        'had_conversion': int(had_conversion)
    })

sessions_df = pd.DataFrame(sessions_raw_data)
sessions_df['start_datetime'] = sessions_df['start_datetime'].dt.floor('S') # Remove microseconds for consistency

# Sort sessions by user_id then start_datetime for time-series operations
sessions_df = sessions_df.sort_values(by=['user_id', 'start_datetime']).reset_index(drop=True)

print(f"Generated {len(users_df)} users, {len(pages_df)} pages, {len(sessions_df)} sessions.")
print("Users head:\n", users_df.head())
print("Sessions head:\n", sessions_df.head())
print("Pages head:\n", pages_df.head())

# --- 2. Load into SQLite & SQL Feature Engineering ---
print("\n--- 2. Loading into SQLite & SQL Feature Engineering ---")

conn = sqlite3.connect(':memory:')
users_df.to_sql('users', conn, index=False, if_exists='replace', datetime_format='ISO')
pages_df.to_sql('pages', conn, index=False, if_exists='replace')
sessions_df.to_sql('sessions', conn, index=False, if_exists='replace', datetime_format='ISO')

# Define GLOBAL_PREDICTION_CUTOFF_DATE
latest_session_dt = pd.read_sql("SELECT MAX(start_datetime) FROM sessions", conn).iloc[0, 0]
GLOBAL_PREDICTION_CUTOFF_DATE = pd.to_datetime(latest_session_dt) - pd.Timedelta(days=7)

print(f"Global Prediction Cutoff Date: {GLOBAL_PREDICTION_CUTOFF_DATE}")

sql_query = f"""
WITH UserHistoricalSessions AS (
    -- Sessions occurring on or before the cutoff date
    SELECT
        s.user_id,
        s.start_datetime,
        s.session_duration_seconds,
        s.had_conversion
    FROM sessions s
    WHERE s.start_datetime <= '{GLOBAL_PREDICTION_CUTOFF_DATE.isoformat()}'
),
HistoricalAggregates AS (
    -- Aggregates for each user from their historical sessions within the last 30 days before cutoff
    -- Also calculates days since the last session before/on cutoff
    SELECT
        uhs.user_id,
        COALESCE(AVG(CASE WHEN julianday('{GLOBAL_PREDICTION_CUTOFF_DATE.isoformat()}') - julianday(uhs.start_datetime) <= 30 THEN uhs.session_duration_seconds ELSE NULL END), 0.0) AS avg_session_duration_prev_30d,
        COALESCE(COUNT(CASE WHEN julianday('{GLOBAL_PREDICTION_CUTOFF_DATE.isoformat()}') - julianday(uhs.start_datetime) <= 30 THEN 1 ELSE NULL END), 0) AS num_sessions_prev_30d,
        COALESCE(AVG(CASE WHEN julianday('{GLOBAL_PREDICTION_CUTOFF_DATE.isoformat()}') - julianday(uhs.start_datetime) <= 30 THEN uhs.had_conversion ELSE NULL END), 0.0) AS conversion_rate_prev_30d,
        -- Calculate days between cutoff and the most recent historical session
        julianday('{GLOBAL_PREDICTION_CUTOFF_DATE.isoformat()}') - julianday(MAX(uhs.start_datetime)) AS days_since_last_session_raw
    FROM UserHistoricalSessions uhs
    GROUP BY uhs.user_id
)
SELECT
    s.session_id,
    s.user_id,
    s.start_datetime,
    s.session_duration_seconds,
    s.had_conversion,
    u.signup_date,
    u.device_type,
    u.browser,
    p.page_name,
    p.page_type,
    -- Join historical aggregates, handling NULLs if no historical data exists for a user
    COALESCE(ha.avg_session_duration_prev_30d, 0.0) AS avg_session_duration_prev_30d,
    COALESCE(ha.num_sessions_prev_30d, 0) AS num_sessions_prev_30d,
    COALESCE(ha.conversion_rate_prev_30d, 0.0) AS conversion_rate_prev_30d,
    -- If no historical sessions, set days_since_last_session_at_cutoff to 9999
    CASE
        WHEN ha.user_id IS NULL OR ha.days_since_last_session_raw IS NULL THEN 9999
        ELSE CAST(ha.days_since_last_session_raw AS INTEGER)
    END AS days_since_last_session_at_cutoff
FROM sessions s
JOIN users u ON s.user_id = u.user_id
JOIN pages p ON s.landing_page_id = p.page_id
LEFT JOIN HistoricalAggregates ha ON s.user_id = ha.user_id
WHERE s.start_datetime > '{GLOBAL_PREDICTION_CUTOFF_DATE.isoformat()}'
ORDER BY s.user_id, s.start_datetime;
"""

session_features_df = pd.read_sql_query(sql_query, conn)
conn.close()

print(f"Fetched {len(session_features_df)} sessions for prediction after cutoff.")
print("Session Features Head:\n", session_features_df.head())
print("Session Features Info:\n")
session_features_df.info()


# --- 3. Pandas Feature Engineering & Binary Target Creation ---
print("\n--- 3. Pandas Feature Engineering & Binary Target Creation ---")

# Convert datetime columns
session_features_df['signup_date'] = pd.to_datetime(session_features_df['signup_date'])
session_features_df['start_datetime'] = pd.to_datetime(session_features_df['start_datetime'])

# Fill NaNs for numerical historical aggregated features
# Using direct assignment to avoid FutureWarning for inplace=True
numerical_hist_cols = ['avg_session_duration_prev_30d', 'conversion_rate_prev_30d']
for col in numerical_hist_cols:
    session_features_df[col] = session_features_df[col].fillna(0.0)

session_features_df['num_sessions_prev_30d'] = session_features_df['num_sessions_prev_30d'].fillna(0).astype(int)
session_features_df['days_since_last_session_at_cutoff'] = session_features_df['days_since_last_session_at_cutoff'].fillna(9999).astype(int)


# Calculate user_tenure_at_session_start_days
session_features_df['user_tenure_at_session_start_days'] = (session_features_df['start_datetime'] - session_features_df['signup_date']).dt.days

# Create the Binary Target `is_bounce`
bounce_duration_threshold = session_features_df['session_duration_seconds'].quantile(0.20)
print(f"20th percentile of session duration: {bounce_duration_threshold:.2f} seconds.")

# A session is a bounce if duration is low AND no conversion
session_features_df['is_bounce'] = ((session_features_df['session_duration_seconds'] <= bounce_duration_threshold) & (session_features_df['had_conversion'] == 0)).astype(int)

# Drop target-defining columns and identifier columns to prevent leakage
X = session_features_df.drop(columns=['session_duration_seconds', 'had_conversion', 'is_bounce', 'session_id', 'user_id', 'signup_date', 'start_datetime'])
y = session_features_df['is_bounce']

print("\nFeatures (X) head after target creation and leakage prevention:\n", X.head())
print("\nTarget (y) value counts:\n", y.value_counts())

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

print(f"\nTraining set size: {len(X_train)} samples")
print(f"Test set size: {len(X_test)} samples")
print(f"Bounce rate in training set: {y_train.mean():.2f}")
print(f"Bounce rate in test set: {y_test.mean():.2f}")


# --- 4. Data Visualization ---
print("\n--- 4. Data Visualization ---")
plt.figure(figsize=(14, 6))

# Plot 1: Violin plot of user_tenure_at_session_start_days vs is_bounce
plt.subplot(1, 2, 1)
sns.violinplot(x='is_bounce', y='user_tenure_at_session_start_days', data=session_features_df)
plt.title('User Tenure at Session Start by Bounce Status')
plt.xlabel('Is Bounce (0: No, 1: Yes)')
plt.ylabel('User Tenure (Days)')
plt.xticks([0, 1], ['Not Bounce', 'Bounce'])

# Plot 2: Stacked bar chart of is_bounce proportions across device_type
plt.subplot(1, 2, 2)
device_bounce_proportion = session_features_df.groupby('device_type')['is_bounce'].value_counts(normalize=True).unstack().fillna(0)
device_bounce_proportion.plot(kind='bar', stacked=True, ax=plt.gca(), cmap='viridis')
plt.title('Bounce Proportion by Device Type')
plt.xlabel('Device Type')
plt.ylabel('Proportion')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Is Bounce', labels=['Not Bounce', 'Bounce'])
plt.tight_layout()
plt.show()

# --- 5. ML Pipeline & Evaluation ---
print("\n--- 5. Machine Learning Pipeline & Evaluation ---")

# Identify numerical and categorical features
numerical_features = [
    'avg_session_duration_prev_30d',
    'num_sessions_prev_30d',
    'conversion_rate_prev_30d',
    'days_since_last_session_at_cutoff',
    'user_tenure_at_session_start_days'
]
categorical_features = ['device_type', 'browser', 'page_name', 'page_type']

# Create preprocessing pipelines for numerical and categorical features
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)) # CRITICAL FIX: sparse_output=False
])

# Create a preprocessor using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='drop' # Drop any columns not specified
)

# Create the full machine learning pipeline
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', HistGradientBoostingClassifier(random_state=42, class_weight='balanced'))
])

# Train the model
print("Training the HistGradientBoostingClassifier pipeline...")
model_pipeline.fit(X_train, y_train)
print("Training complete.")

# Predict probabilities for the positive class (class 1: Bounce) on the test set
y_pred_proba = model_pipeline.predict_proba(X_test)[:, 1]
# Predict hard labels for the classification report
y_pred = model_pipeline.predict(X_test) 

# Evaluate the model
roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"\nROC AUC Score on Test Set: {roc_auc:.4f}")

print("\nClassification Report on Test Set:")
print(classification_report(y_test, y_pred, target_names=['Not Bounce', 'Bounce']))