import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
import warnings

# Suppress specific warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# --- 1. Generate Synthetic Data (Pandas/Numpy) ---
print("--- Generating Synthetic Data ---")

# Users DataFrame
num_users = np.random.randint(500, 701)
current_date = datetime.now()
signup_dates = [current_date - timedelta(days=np.random.randint(0, 5*365)) for _ in range(num_users)]
countries = ['US', 'UK', 'DE', 'FR']
subscription_plans = ['Free', 'Premium_Monthly', 'Premium_Annual']

users_df = pd.DataFrame({
    'user_id': np.arange(1, num_users + 1),
    'signup_date': signup_dates,
    'country': np.random.choice(countries, num_users, p=[0.4, 0.25, 0.2, 0.15]),
    'subscription_plan': np.random.choice(subscription_plans, num_users, p=[0.3, 0.4, 0.3])
})

# Identify users who will be high engagement later (for initial bias)
# Let's say 20-30% of users are likely to be high engagement
high_engagement_user_ids = np.random.choice(users_df['user_id'], int(np.random.uniform(0.2, 0.3) * num_users), replace=False)

# Events DataFrame
num_events = np.random.randint(15000, 25001)
event_types = ['app_open', 'post_view', 'post_like', 'comment', 'share', 'profile_update', 'settings_change']
events_data = []

# Map user_id to signup_date and subscription_plan for event generation
user_info_map = users_df.set_index('user_id')[['signup_date', 'subscription_plan']].to_dict('index')

for i in range(1, num_events + 1):
    user_id = np.random.choice(users_df['user_id'])
    user_signup_date = user_info_map[user_id]['signup_date']
    user_plan = user_info_map[user_id]['subscription_plan']
    
    # Ensure event_timestamp is after signup_date
    # Generate event timestamp randomly, but at least 1 day after signup
    time_offset_days = np.random.randint(1, 365 * 5 + 1) # Events can occur from 1 day up to 5 years after signup
    event_timestamp = user_signup_date + timedelta(days=time_offset_days, seconds=np.random.randint(0, 86400))
    
    event_type = np.random.choice(event_types, p=[0.25, 0.3, 0.15, 0.1, 0.05, 0.1, 0.05])
    duration_seconds = 0
    
    # Base duration for specific event types
    if event_type == 'app_open':
        duration_seconds = np.random.randint(1, 601) # 1 to 600 seconds
    elif event_type == 'post_view':
        duration_seconds = np.random.randint(1, 11) # 1 to 10 seconds (very short)
    elif event_type in ['comment', 'share']:
        duration_seconds = np.random.randint(1, 61) # 1 to 60 seconds

    # Simulate realistic engagement patterns:
    
    # Bias for Premium users
    if user_plan != 'Free':
        if np.random.rand() < 0.3: # 30% chance to bias event type for premium users
            event_type = np.random.choice(['comment', 'share', 'post_like'], p=[0.4, 0.3, 0.3])
            if event_type == 'comment':
                duration_seconds = np.random.randint(1, 46) # Slightly longer/more frequent for premium
            elif event_type == 'share':
                duration_seconds = np.random.randint(1, 31)
            elif event_type == 'post_like':
                duration_seconds = 0

    # Bias for users who are likely to be high engagement later (initial behavior)
    if user_id in high_engagement_user_ids:
        if np.random.rand() < 0.4: # 40% chance to bias event type for high engagement users
            event_type = np.random.choice(['post_like', 'comment', 'share', 'profile_update'], p=[0.3, 0.3, 0.2, 0.2])
            if event_type == 'comment':
                duration_seconds = np.random.randint(1, 61)
            elif event_type == 'share':
                duration_seconds = np.random.randint(1, 41)
            elif event_type == 'profile_update':
                 duration_seconds = np.random.randint(1, 91)
            elif event_type == 'post_like':
                duration_seconds = 0

    # Bias for Free plan users
    if user_plan == 'Free':
        if np.random.rand() < 0.25: # 25% chance to bias event type for free users
            event_type = np.random.choice(['app_open', 'post_view', 'settings_change'], p=[0.6, 0.3, 0.1])
            if event_type == 'app_open':
                duration_seconds = np.random.randint(1, 601)
            elif event_type == 'post_view':
                duration_seconds = np.random.randint(1, 11)
            elif event_type == 'settings_change':
                duration_seconds = np.random.randint(1, 31)


    events_data.append({
        'event_id': i,
        'user_id': user_id,
        'event_timestamp': event_timestamp,
        'event_type': event_type,
        'duration_seconds': duration_seconds
    })

events_df = pd.DataFrame(events_data)

# Sort events_df
events_df = events_df.sort_values(by=['user_id', 'event_timestamp']).reset_index(drop=True)

print(f"Generated {len(users_df)} users and {len(events_df)} events.")
print("Users head:\n", users_df.head())
print("\nEvents head:\n", events_df.head())

# --- 2. Load into SQLite & SQL Feature Engineering (Early User Engagement) ---
print("\n--- Loading data into SQLite and SQL Feature Engineering ---")

conn = sqlite3.connect(':memory:')
# Convert datetime columns to string format for SQLite
users_df['signup_date'] = users_df['signup_date'].dt.strftime('%Y-%m-%d %H:%M:%S')
events_df['event_timestamp'] = events_df['event_timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')

users_df.to_sql('users', conn, index=False, if_exists='replace')
events_df.to_sql('events', conn, index=False, if_exists='replace')

sql_query = """
SELECT
    u.user_id,
    u.signup_date,
    u.country,
    u.subscription_plan,
    COALESCE(SUM(CASE WHEN e.event_id IS NOT NULL THEN 1 ELSE 0 END), 0) AS num_events_first_14d,
    COALESCE(SUM(CASE WHEN e.event_type = 'app_open' THEN e.duration_seconds ELSE 0 END), 0) AS total_app_open_duration_first_14d,
    COALESCE(SUM(CASE WHEN e.event_type = 'post_like' THEN 1 ELSE 0 END), 0) AS num_likes_first_14d,
    COALESCE(SUM(CASE WHEN e.event_type = 'comment' THEN 1 ELSE 0 END), 0) AS num_comments_first_14d,
    COALESCE(SUM(CASE WHEN e.event_type = 'share' THEN 1 ELSE 0 END), 0) AS num_shares_first_14d,
    COALESCE(COUNT(DISTINCT STRFTIME('%Y-%m-%d', e.event_timestamp)), 0) AS days_with_activity_first_14d,
    COALESCE(MAX(CASE WHEN e.event_type = 'profile_update' THEN 1 ELSE 0 END), 0) AS has_profile_update_first_14d
FROM
    users u
LEFT JOIN
    events e ON u.user_id = e.user_id
    AND e.event_timestamp BETWEEN u.signup_date AND DATE(u.signup_date, '+14 days')
GROUP BY
    u.user_id, u.signup_date, u.country, u.subscription_plan
ORDER BY
    u.user_id;
"""

user_early_features_df = pd.read_sql_query(sql_query, conn)
print("\nEarly user features head:\n", user_early_features_df.head())

# Close SQLite connection
conn.close()

# --- 3. Pandas Feature Engineering & Multi-Class Target Creation (Future Engagement Tier) ---
print("\n--- Pandas Feature Engineering and Target Creation ---")

# Handle NaN values (if any, though SQL COALESCE should largely prevent this for aggregated columns)
fill_cols = [
    'num_events_first_14d', 'total_app_open_duration_first_14d',
    'num_likes_first_14d', 'num_comments_first_14d', 'num_shares_first_14d',
    'days_with_activity_first_14d', 'has_profile_update_first_14d'
]
for col in fill_cols:
    if col in user_early_features_df.columns:
        user_early_features_df[col] = user_early_features_df[col].fillna(0).astype(int)

# Convert signup_date to datetime
user_early_features_df['signup_date'] = pd.to_datetime(user_early_features_df['signup_date'])

# Calculate account_age_at_cutoff_days
user_early_features_df['account_age_at_cutoff_days'] = 14 # By definition of the window

# Calculate event_frequency_first_14d
user_early_features_df['event_frequency_first_14d'] = user_early_features_df['num_events_first_14d'] / 14.0
user_early_features_df['event_frequency_first_14d'] = user_early_features_df['event_frequency_first_14d'].fillna(0)

# Calculate engagement_action_ratio_first_14d
user_early_features_df['engagement_action_ratio_first_14d'] = (
    user_early_features_df['num_likes_first_14d'] +
    user_early_features_df['num_comments_first_14d'] +
    user_early_features_df['num_shares_first_14d']
) / (user_early_features_df['num_events_first_14d'] + 1) # Add 1 to prevent division by zero
user_early_features_df['engagement_action_ratio_first_14d'] = user_early_features_df['engagement_action_ratio_first_14d'].fillna(0)

# Create the Multi-Class Target `future_engagement_tier`
# Restore original events_df datetime types for calculation
events_df['event_timestamp'] = pd.to_datetime(events_df['event_timestamp'])
users_df['signup_date'] = pd.to_datetime(users_df['signup_date'])

events_df_with_signup = pd.merge(events_df, users_df[['user_id', 'signup_date']], on='user_id', how='left')

# Calculate cutoff date for each user
events_df_with_signup['cutoff_date'] = events_df_with_signup['signup_date'] + timedelta(days=14)

# Filter events strictly after the 14-day cutoff
future_events_df = events_df_with_signup[events_df_with_signup['event_timestamp'] > events_df_with_signup['cutoff_date']]

total_events_after_14d = future_events_df.groupby('user_id').size().reset_index(name='total_events_after_14d')

# Merge with user_early_features_df
user_early_features_df = pd.merge(
    user_early_features_df,
    total_events_after_14d,
    on='user_id',
    how='left'
)
user_early_features_df['total_events_after_14d'] = user_early_features_df['total_events_after_14d'].fillna(0).astype(int)

# Define engagement tiers
# Calculate percentiles only on non-zero future events
non_zero_future_events = user_early_features_df[user_early_features_df['total_events_after_14d'] > 0]['total_events_after_14d']

p25, p50, p75 = 0, 0, 0 # Initialize with 0 in case non_zero_future_events is empty
if not non_zero_future_events.empty:
    # Ensure percentiles are distinct to avoid issues with pd.cut bins if data is sparse
    unique_non_zero_events = non_zero_future_events.unique()
    if len(unique_non_zero_events) >= 3:
        p25, p50, p75 = np.percentile(non_zero_future_events, [25, 50, 75])
    elif len(unique_non_zero_events) == 2:
        p25 = min(unique_non_zero_events)
        p50 = np.median(unique_non_zero_events)
        p75 = max(unique_non_zero_events)
    elif len(unique_non_zero_events) == 1:
        p25 = p50 = p75 = unique_non_zero_events[0]
    # Smallest possible non-zero values for percentiles if data is very sparse
    p25 = max(1, p25) 
    p50 = max(p25, p50)
    p75 = max(p50, p75)


def assign_engagement_tier(row):
    if row['total_events_after_14d'] == 0:
        return 'Inactive'
    # Use strict greater than for lower bound, less than or equal for upper bound of each percentile bin
    elif row['total_events_after_14d'] > 0 and row['total_events_after_14d'] <= p25:
        return 'Low_Engagement'
    elif row['total_events_after_14d'] > p25 and row['total_events_after_14d'] <= p50:
        return 'Medium_Engagement'
    elif row['total_events_after_14d'] > p50 and row['total_events_after_14d'] <= p75:
        return 'High_Engagement'
    else: # Must be > p75
        return 'Very_High_Engagement'

user_early_features_df['future_engagement_tier'] = user_early_features_df.apply(assign_engagement_tier, axis=1)

print("\nUser early features with target head:\n", user_early_features_df.head())
print("\nFuture Engagement Tier Distribution:\n", user_early_features_df['future_engagement_tier'].value_counts())

# Define features (X) and target (y)
numerical_features = [
    'num_events_first_14d', 'total_app_open_duration_first_14d',
    'num_likes_first_14d', 'num_comments_first_14d', 'num_shares_first_14d',
    'days_with_activity_first_14d', 'account_age_at_cutoff_days',
    'event_frequency_first_14d', 'engagement_action_ratio_first_14d'
]
categorical_features = ['country', 'subscription_plan']
binary_features = ['has_profile_update_first_14d'] # Treat as categorical for one-hot encoding

X = user_early_features_df[numerical_features + categorical_features + binary_features]
y = user_early_features_df['future_engagement_tier']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
print(f"\nTrain set size: {len(X_train)} | Test set size: {len(X_test)}")
print("Target distribution in train set:\n", y_train.value_counts(normalize=True))
print("Target distribution in test set:\n", y_test.value_counts(normalize=True))

# --- 4. Data Visualization ---
print("\n--- Data Visualization ---")
plt.style.use('seaborn-v0_8-darkgrid')

# Define a consistent order for tiers
tier_order = ['Inactive', 'Low_Engagement', 'Medium_Engagement', 'High_Engagement', 'Very_High_Engagement']

# 4.1 Violin plot for num_likes_first_14d by future_engagement_tier
plt.figure(figsize=(12, 7))
sns.violinplot(
    x='future_engagement_tier',
    y='num_likes_first_14d',
    data=user_early_features_df,
    order=tier_order,
    palette='viridis'
)
plt.title('Distribution of Number of Likes in First 14 Days by Future Engagement Tier')
plt.xlabel('Future Engagement Tier')
plt.ylabel('Number of Likes (First 14 Days)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# 4.2 Stacked bar chart for subscription_plan vs. future_engagement_tier
plt.figure(figsize=(12, 7))
subscription_tier_pivot = user_early_features_df.groupby('subscription_plan')['future_engagement_tier'].value_counts(normalize=True).unstack().fillna(0)

# Fix: Reindex with fill_value=0 to handle missing columns gracefully, ensuring order
# Filter tier_order to only include tiers actually present in the pivot table for reindexing
present_tiers_in_pivot = [tier for tier in tier_order if tier in subscription_tier_pivot.columns]
subscription_tier_pivot = subscription_tier_pivot.reindex(columns=present_tiers_in_pivot, fill_value=0)

subscription_tier_pivot.plot(kind='bar', stacked=True, colormap='viridis')
plt.title('Proportion of Future Engagement Tiers by Subscription Plan')
plt.xlabel('Subscription Plan')
plt.ylabel('Proportion')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Engagement Tier', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# --- 5. ML Pipeline & Evaluation (Multi-Class) ---
print("\n--- Building and Evaluating ML Pipeline ---")

# Define preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ]), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features + binary_features)
    ],
    remainder='passthrough' # Keep other columns if any, though not expected here
)

# Create the ML Pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', HistGradientBoostingClassifier(random_state=42))
])

# Train the pipeline
pipeline.fit(X_train, y_train)

# Predict on the test set
y_pred = pipeline.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy on Test Set: {accuracy:.4f}")

print("\nClassification Report:\n")
# The `labels` argument in classification_report can be used to explicitly include all possible class labels,
# even if some are not present in y_test or y_pred, which can happen with small test sets and `stratify`.
# This ensures a consistent report structure.
# Get all unique labels from y_train (which should contain all classes due to stratify)
all_possible_labels = sorted(y_train.unique().tolist(), key=lambda x: tier_order.index(x))
print(classification_report(y_test, y_pred, labels=all_possible_labels, target_names=all_possible_labels, zero_division=0))

print("\n--- Script Finished Successfully ---")