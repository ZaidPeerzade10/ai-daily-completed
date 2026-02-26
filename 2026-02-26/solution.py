import pandas as pd
import numpy as np
import datetime
import sqlite3
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import random
import warnings

# Suppress specific future warnings from pandas/seaborn
warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")
warnings.filterwarnings("ignore", category=FutureWarning, module="seaborn")
warnings.filterwarnings("ignore", category=UserWarning, module="seaborn")

# --- 1. Generate Synthetic Data (Pandas/Numpy) ---

print("--- Generating Synthetic Data ---")

# Define number of users
num_users = np.random.randint(500, 701)

# Generate signup_dates over the last 2-3 years
end_date = datetime.datetime.now()
start_date = end_date - datetime.timedelta(days=3 * 365)
date_range = (end_date - start_date).days
signup_dates = [start_date + datetime.timedelta(days=np.random.randint(0, date_range)) for _ in range(num_users)]

# users_df
users_df = pd.DataFrame({
    'user_id': range(1, num_users + 1),
    'signup_date': signup_dates,
    'assigned_onboarding_variant': np.random.choice(['Control', 'Variant_A', 'Variant_B'], num_users, p=[0.4, 0.3, 0.3]),
    'referral_source': np.random.choice(['Organic', 'Paid_Ad', 'Referral', 'Social'], num_users, p=[0.4, 0.3, 0.2, 0.1])
})
users_df['signup_date'] = pd.to_datetime(users_df['signup_date']).dt.normalize()

# onboarding_events_df (Target: 3000-5000 rows)
onboarding_events_data = []
event_id_counter = 1
event_types = ['step_1_completed', 'profile_filled', 'tutorial_viewed', 'payment_info_added', 'email_verified', 'initial_action']
critical_steps = ['profile_filled', 'payment_info_added']

for index, user in users_df.iterrows():
    user_id = user['user_id']
    signup_date = user['signup_date']
    variant = user['assigned_onboarding_variant']

    # Simulate number of onboarding events per user (average 5-8 events)
    num_events_for_user = np.random.randint(3, 10) # Adjusted to hit target 3000-5000 events

    for _ in range(num_events_for_user):
        event_date = signup_date + datetime.timedelta(days=np.random.randint(0, 7)) # Within 7 days
        event_type = np.random.choice(event_types)
        duration_seconds = np.random.randint(10, 301)

        # Simulate variant impact on event completion rates and duration
        if variant == 'Variant_A':
            # Variant A: Higher chance of critical steps, higher tutorial duration
            if np.random.rand() < 0.6: # 60% chance to have a critical step (vs 40-50% for control)
                event_type = np.random.choice(critical_steps, p=[0.7, 0.3])
            if event_type == 'tutorial_viewed':
                duration_seconds = np.random.randint(60, 401) # Longer tutorial view
        elif variant == 'Variant_B':
            # Variant B: Lower chance of critical steps, shorter tutorial duration
            if np.random.rand() < 0.3: # 30% chance (vs 40-50% for control)
                event_type = np.random.choice(critical_steps, p=[0.6, 0.4])
            if event_type == 'tutorial_viewed':
                duration_seconds = np.random.randint(5, 151) # Shorter tutorial view

        onboarding_events_data.append({
            'event_id': event_id_counter,
            'user_id': user_id,
            'event_date': event_date,
            'event_type': event_type,
            'duration_seconds': duration_seconds
        })
        event_id_counter += 1

onboarding_events_df = pd.DataFrame(onboarding_events_data)
onboarding_events_df['event_date'] = pd.to_datetime(onboarding_events_df['event_date']).dt.normalize()

# future_activity_df (Target: 2000-3000 rows)
future_activity_data = []
activity_id_counter = 1
activity_types = ['login', 'feature_use_A', 'feature_use_B', 'content_view', 'purchase']

# Determine which users are more likely to be retained based on simulated onboarding
user_retention_scores = {}
for user_id in users_df['user_id']:
    user_events = onboarding_events_df[onboarding_events_df['user_id'] == user_id]
    num_critical = user_events[user_events['event_type'].isin(critical_steps)].shape[0]
    tutorial_duration = user_events[user_events['event_type'] == 'tutorial_viewed']['duration_seconds'].sum()
    
    base_prob = 0.4 # Base retention probability
    prob_modifier = 0
    if num_critical >= 2: prob_modifier += 0.2
    elif num_critical == 1: prob_modifier += 0.1
    if tutorial_duration > 300: prob_modifier += 0.15
    elif tutorial_duration > 100: prob_modifier += 0.05
    
    # Adjust for variant effects on retention directly
    variant = users_df[users_df['user_id'] == user_id]['assigned_onboarding_variant'].iloc[0]
    if variant == 'Variant_A': prob_modifier += 0.1
    elif variant == 'Variant_B': prob_modifier -= 0.1
    
    user_retention_scores[user_id] = min(0.9, max(0.1, base_prob + prob_modifier))

for user_id in users_df['user_id']:
    signup_date = users_df[users_df['user_id'] == user_id]['signup_date'].iloc[0]
    
    retention_prob = user_retention_scores.get(user_id, 0.4) # Default if no score calculated

    if np.random.rand() < retention_prob: # User is likely to have future activity
        # Number of future activities for retained users (average 3-6 activities)
        num_activities_for_user = np.random.randint(3, 7) # Adjusted to hit target 2000-3000 activities
        
        for _ in range(num_activities_for_user):
            activity_date = signup_date + datetime.timedelta(days=np.random.randint(30, 91)) # 30 to 90 days after signup
            activity_type = np.random.choice(activity_types)
            
            future_activity_data.append({
                'activity_id': activity_id_counter,
                'user_id': user_id,
                'activity_date': activity_date,
                'activity_type': activity_type
            })
            activity_id_counter += 1

future_activity_df = pd.DataFrame(future_activity_data)
future_activity_df['activity_date'] = pd.to_datetime(future_activity_df['activity_date']).dt.normalize()

print(f"Generated {len(users_df)} users.")
print(f"Generated {len(onboarding_events_df)} onboarding events (Target: 3000-5000).")
print(f"Generated {len(future_activity_df)} future activities (Target: 2000-3000).")

# --- 2. Load into SQLite & SQL Feature Engineering (Onboarding Behavior) ---

print("\n--- Performing SQL Feature Engineering ---")

# Create an in-memory SQLite database
conn = sqlite3.connect(':memory:')

# Load dataframes into SQLite tables
users_df.to_sql('users', conn, index=False, if_exists='replace')
onboarding_events_df.to_sql('onboarding_events', conn, index=False, if_exists='replace')

# Convert date columns to TEXT for SQLite to handle properly in SQL queries
# This is important for strftime and JULIANDAY functions
users_df['signup_date_str'] = users_df['signup_date'].dt.strftime('%Y-%m-%d')
onboarding_events_df['event_date_str'] = onboarding_events_df['event_date'].dt.strftime('%Y-%m-%d')

users_df[['user_id', 'signup_date_str', 'assigned_onboarding_variant', 'referral_source']].to_sql('users', conn, index=False, if_exists='replace')
onboarding_events_df[['event_id', 'user_id', 'event_date_str', 'event_type', 'duration_seconds']].to_sql('onboarding_events', conn, index=False, if_exists='replace')


# SQL query for aggregation
sql_query = """
SELECT
    u.user_id,
    u.signup_date_str AS signup_date,
    u.assigned_onboarding_variant,
    u.referral_source,
    COALESCE(COUNT(oe.event_id), 0) AS num_onboarding_events,
    COALESCE(SUM(oe.duration_seconds), 0) AS total_onboarding_duration,
    COALESCE(AVG(oe.duration_seconds), 0.0) AS avg_step_duration,
    COALESCE(SUM(CASE WHEN oe.event_type IN ('profile_filled', 'payment_info_added') THEN 1 ELSE 0 END), 0) AS num_critical_steps_completed,
    COALESCE(CAST(SUM(CASE WHEN oe.event_type IN ('profile_filled', 'payment_info_added') THEN 1 ELSE 0 END) AS REAL) / 2.0, 0.0) AS onboarding_completion_rate,
    CAST(JULIANDAY(MIN(oe.event_date_str)) - JULIANDAY(u.signup_date_str) AS INTEGER) AS days_to_first_onboarding_event
FROM
    users u
LEFT JOIN
    onboarding_events oe ON u.user_id = oe.user_id
    AND oe.event_date_str BETWEEN u.signup_date_str AND DATE(u.signup_date_str, '+7 days')
GROUP BY
    u.user_id, u.signup_date_str, u.assigned_onboarding_variant, u.referral_source
ORDER BY
    u.user_id;
"""

# Fetch results into a pandas DataFrame
user_onboarding_features_df = pd.read_sql_query(sql_query, conn)

# Close the connection
conn.close()

print("SQL Feature Engineering complete. Preview of features:")
print(user_onboarding_features_df.head())
print(f"Shape of user_onboarding_features_df: {user_onboarding_features_df.shape}")


# --- 3. Pandas Feature Engineering & Binary Target Creation ---

print("\n--- Pandas Feature Engineering & Target Creation ---")

# Handle NaN values
# num_onboarding_events, total_onboarding_duration, num_critical_steps_completed are already COALESCE'd to 0 by SQL
# avg_step_duration, onboarding_completion_rate are already COALESCE'd to 0.0 by SQL
# days_to_first_onboarding_event will be NULL for users with no events, fill with sentinel value
user_onboarding_features_df['days_to_first_onboarding_event'] = user_onboarding_features_df['days_to_first_onboarding_event'].fillna(9999)

# Convert signup_date to datetime objects
user_onboarding_features_df['signup_date'] = pd.to_datetime(user_onboarding_features_df['signup_date']).dt.normalize()

# Create the Binary Target `is_retained_90_days`
# A user is considered retained (1) if they have *any* activity between signup_date + 30 days and signup_date + 90 days.
retained_users_check_df = future_activity_df.merge(
    users_df[['user_id', 'signup_date']], 
    on='user_id', how='left'
)

# Filter for activities within the 30-90 day window
retained_users_check_df['activity_date'] = pd.to_datetime(retained_users_check_df['activity_date']).dt.normalize()
retained_users_check_df['signup_date'] = pd.to_datetime(retained_users_check_df['signup_date']).dt.normalize()

retained_filtered = retained_users_check_df[
    (retained_users_check_df['activity_date'] >= retained_users_check_df['signup_date'] + pd.Timedelta(days=30)) &
    (retained_users_check_df['activity_date'] <= retained_users_check_df['signup_date'] + pd.Timedelta(days=90))
]

# Get unique user_ids who are retained
retained_user_ids = retained_filtered['user_id'].unique()

# Add the binary target column to the main features DataFrame
user_onboarding_features_df['is_retained_90_days'] = user_onboarding_features_df['user_id'].isin(retained_user_ids).astype(int)

print("Target creation complete. Retention rate:")
print(user_onboarding_features_df['is_retained_90_days'].value_counts(normalize=True))

# Define features X and target y
numerical_features = [
    'num_onboarding_events', 
    'total_onboarding_duration', 
    'avg_step_duration', 
    'num_critical_steps_completed', 
    'days_to_first_onboarding_event', 
    'onboarding_completion_rate'
]
categorical_features = [
    'assigned_onboarding_variant', 
    'referral_source'
]

X = user_onboarding_features_df[numerical_features + categorical_features]
y = user_onboarding_features_df['is_retained_90_days']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")


# --- 4. Data Visualization ---

print("\n--- Generating Data Visualizations ---")

plt.style.use('seaborn-v0_8-darkgrid')
plt.figure(figsize=(14, 6))

# Plot 1: Stacked bar chart for retention by onboarding variant
plt.subplot(1, 2, 1)
retention_by_variant = user_onboarding_features_df.groupby('assigned_onboarding_variant')['is_retained_90_days'].value_counts(normalize=True).unstack().fillna(0)
retention_by_variant.plot(kind='bar', stacked=True, ax=plt.gca(), color=['lightcoral', 'lightgreen'])
plt.title('90-Day Retention Rate by Onboarding Variant')
plt.ylabel('Proportion')
plt.xlabel('Onboarding Variant')
plt.xticks(rotation=45)
plt.legend(title='Retained (90 days)', labels=['Not Retained (0)', 'Retained (1)'])

# Plot 2: Violin plot for total_onboarding_duration vs. retention
plt.subplot(1, 2, 2)
sns.violinplot(x='is_retained_90_days', y='total_onboarding_duration', data=user_onboarding_features_df, palette='viridis', ax=plt.gca())
plt.title('Total Onboarding Duration Distribution by 90-Day Retention')
plt.xlabel('90-Day Retained (0=No, 1=Yes)')
plt.ylabel('Total Onboarding Duration (seconds)')
plt.tight_layout()
plt.show()

print("Visualizations displayed.")


# --- 5. ML Pipeline & Evaluation (Binary Classification) ---

print("\n--- Building and Evaluating ML Pipeline ---")

# Create a ColumnTransformer for preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ]), numerical_features),
        ('cat', Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ]), categorical_features)
    ])

# Create the ML Pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(random_state=42, solver='liblinear', class_weight='balanced'))
])

# Train the pipeline
pipeline.fit(X_train, y_train)

# Predict probabilities on the test set
y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
y_pred = pipeline.predict(X_test) # For classification report

# Evaluate the model
roc_auc = roc_auc_score(y_test, y_pred_proba)
class_report = classification_report(y_test, y_pred)

print(f"\n--- Model Evaluation Results ---")
print(f"ROC AUC Score on Test Set: {roc_auc:.4f}")
print("\nClassification Report on Test Set:\n", class_report)

print("\nML Pipeline execution complete.")