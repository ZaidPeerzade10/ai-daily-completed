import pandas as pd
import numpy as np
import datetime
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, classification_report

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')

# --- 1. Generate Synthetic Data (Pandas/Numpy) ---
np.random.seed(42)

# Define date ranges for synthetic data
end_date = datetime.date.today()
start_date = end_date - datetime.timedelta(days=5*365) # Last 5 years

# Helper function to generate random dates
def random_dates(start, end, n):
    delta = (end - start).days
    return [start + datetime.timedelta(days=np.random.randint(delta)) for _ in range(n)]

# 1.1 users_df
num_users = np.random.randint(500, 701)
users_df = pd.DataFrame({
    'user_id': np.arange(1, num_users + 1),
    'signup_date': random_dates(start_date, end_date - datetime.timedelta(days=90), num_users), # signup must be far enough in past for usage
    'industry': np.random.choice(['Tech', 'Healthcare', 'Education', 'Finance', 'Manufacturing', 'Retail'], num_users),
    'company_size': np.random.choice(['Small', 'Medium', 'Large'], num_users),
    'current_plan': 'Free' # All users initially start on 'Free' plan
})
users_df['signup_date'] = pd.to_datetime(users_df['signup_date'])


# 1.2 feature_usage_df
num_usage_events_base = np.random.randint(5000, 8001)
feature_names = ['Basic_Dashboard', 'Report_Gen', 'Data_Export_Limited', 'Share_Doc', 'Team_Collab', 'Search', 'Notifications']
premium_adjacent_features = ['Report_Gen', 'Data_Export_Limited', 'Team_Collab']

# Identify potential converters (e.g., 25-30% of users)
converter_user_ids = np.random.choice(users_df['user_id'], size=int(0.28 * num_users), replace=False)

usage_data = []
for _ in range(num_usage_events_base):
    user_id = np.random.choice(users_df['user_id'])
    signup_date = users_df[users_df['user_id'] == user_id]['signup_date'].iloc[0]

    # Ensure usage_date is after signup_date
    usage_date_range_end = end_date - datetime.timedelta(days=30) # Usage shouldn't go right up to today
    usage_date_delta = (usage_date_range_end - signup_date.date()).days
    if usage_date_delta <= 0: # If signup is too recent, skip or adjust
        continue
    usage_date = signup_date + pd.to_timedelta(np.random.randint(1, usage_date_delta), unit='D')

    feature_name = np.random.choice(feature_names)
    duration_minutes = np.random.uniform(5, 120)

    # Bias usage for potential converters
    if user_id in converter_user_ids:
        # Increase duration and frequency
        duration_minutes *= np.random.uniform(1.2, 2.0)
        if np.random.rand() < 0.6: # 60% chance for a premium-adjacent feature
            feature_name = np.random.choice(premium_adjacent_features)
        # Add more usage events for converters (simulating higher activity)
        if np.random.rand() < 0.2: # Add an extra event for this user
            usage_data.append({
                'user_id': user_id,
                'usage_date': usage_date,
                'feature_name': np.random.choice(feature_names),
                'duration_minutes': np.random.uniform(5, 120) * np.random.uniform(1.2, 2.0)
            })

    usage_data.append({
        'user_id': user_id,
        'usage_date': usage_date,
        'feature_name': feature_name,
        'duration_minutes': duration_minutes
    })

feature_usage_df = pd.DataFrame(usage_data)
feature_usage_df['usage_date'] = pd.to_datetime(feature_usage_df['usage_date'])
feature_usage_df = feature_usage_df.sort_values(by=['user_id', 'usage_date']).reset_index(drop=True)
feature_usage_df['usage_id'] = np.arange(1, len(feature_usage_df) + 1) # Assign unique IDs


# Determine global_analysis_date and feature_cutoff_date based on current feature_usage_df
# This will be fixed for the ML task.
max_usage_date = feature_usage_df['usage_date'].max()
global_analysis_date = max_usage_date + pd.Timedelta(days=60)
feature_cutoff_date = global_analysis_date - pd.Timedelta(days=180) # 6 months prior to global analysis

print(f"Global Analysis Date: {global_analysis_date.strftime('%Y-%m-%d')}")
print(f"Feature Cutoff Date: {feature_cutoff_date.strftime('%Y-%m-%d')}")


# 1.3 premium_conversions_df - CRITICAL IMPROVEMENT FOR CLASS IMBALANCE
num_conversions = np.random.randint(100, 201)
conversion_data = []

# Ensure a good portion of conversions fall into the 90-day prediction window
# The window is from feature_cutoff_date to feature_cutoff_date + 90 days
prediction_window_start_target = feature_cutoff_date
prediction_window_end_target = feature_cutoff_date + pd.Timedelta(days=90)

# Select a subset of potential converters who will convert *within the target window*
num_conversions_in_window = int(0.5 * num_conversions) # ~50% of conversions fall in prediction window
users_for_in_window_conversion = np.random.choice(
    converter_user_ids, size=min(num_conversions_in_window, len(converter_user_ids)), replace=False
)

for user_id in users_for_in_window_conversion:
    signup_date = users_df[users_df['user_id'] == user_id]['signup_date'].iloc[0]
    
    # Ensure conversion date is after signup AND within the target prediction window
    earliest_valid_conversion_date = max(signup_date, prediction_window_start_target)
    
    # If earliest_valid_conversion_date is after prediction_window_end_target, this user can't convert in window.
    if earliest_valid_conversion_date >= prediction_window_end_target:
        continue

    conversion_date = earliest_valid_conversion_date + pd.to_timedelta(
        np.random.randint(0, (prediction_window_end_target - earliest_valid_conversion_date).days + 1), unit='D'
    )
    
    conversion_data.append({
        'user_id': user_id,
        'conversion_date': conversion_date
    })

# Add remaining conversions for other converter_user_ids (these might fall outside the window)
remaining_converters_ids = [uid for uid in converter_user_ids if uid not in [c['user_id'] for c in conversion_data]]
for user_id in remaining_converters_ids:
    if len(conversion_data) >= num_conversions:
        break
    signup_date = users_df[users_df['user_id'] == user_id]['signup_date'].iloc[0]
    
    # Conversion date broadly, but still after signup_date and not too far in the future
    conversion_date = signup_date + pd.to_timedelta(np.random.randint(1, (global_analysis_date - signup_date).days), unit='D')
    
    conversion_data.append({
        'user_id': user_id,
        'conversion_date': conversion_date
    })

# If we still haven't reached num_conversions, add some random non-converters
# This ensures we have the desired number of premium_conversions but dilutes the "smartly placed" ones
while len(conversion_data) < num_conversions:
    user_id = np.random.choice(users_df['user_id'])
    if user_id in [c['user_id'] for c in conversion_data]: # Avoid duplicate conversions for same user
        continue
    
    signup_date = users_df[users_df['user_id'] == user_id]['signup_date'].iloc[0]
    # Conversion date can be anywhere after signup, even outside the prediction window
    conversion_date = signup_date + pd.to_timedelta(np.random.randint(1, (global_analysis_date - signup_date).days + 1), unit='D')
    
    conversion_data.append({
        'user_id': user_id,
        'conversion_date': conversion_date
    })

premium_conversions_df = pd.DataFrame(conversion_data).head(num_conversions) # Ensure exactly num_conversions
premium_conversions_df['conversion_date'] = pd.to_datetime(premium_conversions_df['conversion_date'])
premium_conversions_df['conversion_id'] = np.arange(1, len(premium_conversions_df) + 1) # Assign unique IDs

print(f"Generated {len(users_df)} users, {len(feature_usage_df)} usage events, {len(premium_conversions_df)} conversions.")


# --- 2. Load into SQLite & SQL Feature Engineering ---
conn = sqlite3.connect(':memory:')

users_df.to_sql('users', conn, if_exists='replace', index=False)
feature_usage_df.to_sql('feature_usage', conn, if_exists='replace', index=False,
                        dtype={'usage_date': 'TEXT'}) # Store dates as TEXT for SQLite

# Convert pandas datetime objects to string for SQL query for comparison
feature_cutoff_date_str = feature_cutoff_date.strftime('%Y-%m-%d %H:%M:%S')

sql_query = f"""
SELECT
    u.user_id,
    u.signup_date,
    u.industry,
    u.company_size,
    COALESCE(SUM(CASE WHEN fu.usage_date < '{feature_cutoff_date_str}' THEN fu.duration_minutes ELSE 0 END), 0.0) AS total_usage_duration_pre_cutoff,
    COALESCE(COUNT(CASE WHEN fu.usage_date < '{feature_cutoff_date_str}' THEN fu.usage_id ELSE NULL END), 0) AS num_usage_events_pre_cutoff,
    COALESCE(AVG(CASE WHEN fu.usage_date < '{feature_cutoff_date_str}' THEN fu.duration_minutes ELSE NULL END), 0.0) AS avg_duration_per_event_pre_cutoff,
    COALESCE(COUNT(DISTINCT CASE WHEN fu.usage_date < '{feature_cutoff_date_str}' THEN fu.feature_name ELSE NULL END), 0) AS num_unique_features_used_pre_cutoff,
    COALESCE(COUNT(CASE WHEN fu.feature_name = 'Report_Gen' AND fu.usage_date < '{feature_cutoff_date_str}' THEN fu.usage_id ELSE NULL END), 0) AS count_report_gen_pre_cutoff,
    COALESCE(COUNT(CASE WHEN fu.feature_name = 'Data_Export_Limited' AND fu.usage_date < '{feature_cutoff_date_str}' THEN fu.usage_id ELSE NULL END), 0) AS count_data_export_pre_cutoff,
    -- Calculate days since last usage before cutoff
    CASE
        WHEN MAX(CASE WHEN fu.usage_date < '{feature_cutoff_date_str}' THEN fu.usage_date ELSE NULL END) IS NOT NULL
        THEN JULIANDAY('{feature_cutoff_date_str}') - JULIANDAY(MAX(CASE WHEN fu.usage_date < '{feature_cutoff_date_str}' THEN fu.usage_date ELSE NULL END))
        ELSE NULL
    END AS days_since_last_usage_pre_cutoff
FROM
    users AS u
LEFT JOIN
    feature_usage AS fu ON u.user_id = fu.user_id
GROUP BY
    u.user_id, u.signup_date, u.industry, u.company_size
ORDER BY
    u.user_id;
"""

user_conversion_features_df = pd.read_sql_query(sql_query, conn)
conn.close()

# --- 3. Pandas Feature Engineering & Binary Target Creation ---
user_conversion_features_df['signup_date'] = pd.to_datetime(user_conversion_features_df['signup_date'])
feature_cutoff_date = pd.to_datetime(feature_cutoff_date) # Ensure it's datetime object

# Handle NaN values
user_conversion_features_df['total_usage_duration_pre_cutoff'].fillna(0, inplace=True)
user_conversion_features_df['num_usage_events_pre_cutoff'].fillna(0, inplace=True)
user_conversion_features_df['count_report_gen_pre_cutoff'].fillna(0, inplace=True)
user_conversion_features_df['count_data_export_pre_cutoff'].fillna(0, inplace=True)
user_conversion_features_df['num_unique_features_used_pre_cutoff'].fillna(0, inplace=True)
user_conversion_features_df['avg_duration_per_event_pre_cutoff'].fillna(0.0, inplace=True)

# Calculate user_account_age_at_cutoff_days
user_conversion_features_df['user_account_age_at_cutoff_days'] = (
    feature_cutoff_date - user_conversion_features_df['signup_date']
).dt.days

# Fill NaN for days_since_last_usage_pre_cutoff
# Sentinel value: user_account_age_at_cutoff_days + 30 (or a large value)
# If account age is negative, it means signup_date is after feature_cutoff_date, which shouldn't happen with generation logic
# But for safety, set to 0 minimum.
user_conversion_features_df['user_account_age_at_cutoff_days'] = user_conversion_features_df['user_account_age_at_cutoff_days'].apply(lambda x: max(x, 0))

large_sentinel_value = user_conversion_features_df['user_account_age_at_cutoff_days'] + 30
user_conversion_features_df['days_since_last_usage_pre_cutoff'].fillna(large_sentinel_value, inplace=True)

# Calculate new features
user_conversion_features_df['usage_frequency_pre_cutoff'] = user_conversion_features_df['num_usage_events_pre_cutoff'] / (user_conversion_features_df['user_account_age_at_cutoff_days'] + 1)
user_conversion_features_df['avg_daily_usage_duration_pre_cutoff'] = user_conversion_features_df['total_usage_duration_pre_cutoff'] / (user_conversion_features_df['user_account_age_at_cutoff_days'] + 1)

# Create the Binary Target `converted_to_premium_in_next_90_days`
prediction_window_start = feature_cutoff_date
prediction_window_end = feature_cutoff_date + pd.Timedelta(days=90)

# Merge with premium_conversions_df to identify conversions within the window
conversions_in_window = premium_conversions_df[
    (premium_conversions_df['conversion_date'] >= prediction_window_start) &
    (premium_conversions_df['conversion_date'] < prediction_window_end)
]

user_conversion_features_df['converted_to_premium_in_next_90_days'] = user_conversion_features_df['user_id'].isin(
    conversions_in_window['user_id']
).astype(int)

# Check target distribution
print("\nTarget distribution:")
print(user_conversion_features_df['converted_to_premium_in_next_90_days'].value_counts(normalize=True))

# Define features X and target y
categorical_features = ['industry', 'company_size']
numerical_features = [
    'total_usage_duration_pre_cutoff', 'num_usage_events_pre_cutoff',
    'avg_duration_per_event_pre_cutoff', 'num_unique_features_used_pre_cutoff',
    'count_report_gen_pre_cutoff', 'count_data_export_pre_cutoff',
    'days_since_last_usage_pre_cutoff', 'user_account_age_at_cutoff_days',
    'usage_frequency_pre_cutoff', 'avg_daily_usage_duration_pre_cutoff'
]

X = user_conversion_features_df[numerical_features + categorical_features]
y = user_conversion_features_df['converted_to_premium_in_next_90_days']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

print(f"\nTrain set size: {len(X_train)} users, Test set size: {len(X_test)} users")
print("Train target distribution:\n", y_train.value_counts(normalize=True))
print("Test target distribution:\n", y_test.value_counts(normalize=True))


# --- 4. Data Visualization ---
plt.style.use('seaborn-v0_8-darkgrid')
plt.figure(figsize=(14, 6))

# Plot 1: Violin plot for total_usage_duration_pre_cutoff
plt.subplot(1, 2, 1)
sns.violinplot(
    x='converted_to_premium_in_next_90_days',
    y='total_usage_duration_pre_cutoff',
    data=user_conversion_features_df,
    inner='quartile',
    palette='viridis'
)
plt.title('Total Usage Duration Pre-Cutoff by Conversion Status')
plt.xlabel('Converted to Premium (0=No, 1=Yes)')
plt.ylabel('Total Usage Duration (minutes)')
plt.yscale('log') # Log scale helps with skewed data
plt.tight_layout()

# Plot 2: Stacked bar chart for conversion proportion across industries
plt.subplot(1, 2, 2)
industry_conversion_pivot = user_conversion_features_df.groupby('industry')['converted_to_premium_in_next_90_days'].value_counts(normalize=True).unstack().fillna(0)
industry_conversion_pivot.plot(kind='bar', stacked=True, color=['lightcoral', 'lightseagreen'], ax=plt.gca())
plt.title('Proportion of Premium Conversion by Industry')
plt.xlabel('Industry')
plt.ylabel('Proportion')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Converted', labels=['No', 'Yes'])
plt.tight_layout()
plt.show()


# --- 5. ML Pipeline & Evaluation (Binary Classification) ---

# Preprocessing steps
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Create a preprocessor using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='passthrough' # Keep other columns (if any)
)

# Create the full pipeline
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', HistGradientBoostingClassifier(random_state=42))
])

# Train the model
model_pipeline.fit(X_train, y_train)

# Predict probabilities on the test set
y_pred_proba = model_pipeline.predict_proba(X_test)[:, 1]
y_pred = model_pipeline.predict(X_test) # For classification report

# Evaluate the model
roc_auc = roc_auc_score(y_test, y_pred_proba)
class_report = classification_report(y_test, y_pred, target_names=['Not Converted', 'Converted'])

print("\n--- Model Evaluation ---")
print(f"ROC AUC Score on Test Set: {roc_auc:.4f}")
print("\nClassification Report on Test Set:")
print(class_report)

# Print a summary of predictions to verify class balance in results
print("\nTest set predictions value counts:")
print(pd.Series(y_pred).value_counts())
print("\nTest set true labels value counts:")
print(y_test.value_counts())