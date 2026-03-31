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
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
import warnings

# Suppress specific warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning, module='seaborn')
warnings.filterwarnings('ignore', category=FutureWarning)

# --- 1. Generate Synthetic Data (Pandas/Numpy) ---
np.random.seed(42)

# Users DataFrame
num_users = np.random.randint(500, 701)
users_df = pd.DataFrame({
    'user_id': np.arange(1, num_users + 1),
    'signup_date': pd.to_datetime(pd.Timestamp.now() - pd.to_timedelta(np.random.randint(0, 5*365, num_users), unit='days'))
})

countries = ['US', 'UK', 'DE', 'FR']
users_df['country'] = np.random.choice(countries, num_users, p=[0.4, 0.25, 0.2, 0.15])

subscription_plans = ['Free', 'Premium_Monthly', 'Premium_Annual']
# Bias: more free users, then premium monthly, then annual
users_df['subscription_plan'] = np.random.choice(subscription_plans, num_users, p=[0.55, 0.3, 0.15])

# Simulate base engagement propensity to influence event generation
users_df['engagement_propensity'] = 1
users_df.loc[users_df['subscription_plan'] == 'Premium_Monthly', 'engagement_propensity'] = 1.5
users_df.loc[users_df['subscription_plan'] == 'Premium_Annual', 'engagement_propensity'] = 2.5

# Identify a subset of users who will be 'High_Engagement' later on for early event bias
# Let's say 20% of users will have higher future engagement, boost their propensity
high_engagement_user_ids = np.random.choice(users_df['user_id'], int(num_users * 0.2), replace=False)
users_df['is_high_future_engager'] = users_df['user_id'].isin(high_engagement_user_ids).astype(int)
users_df.loc[users_df['is_high_future_engager'] == 1, 'engagement_propensity'] *= 1.8


# Events DataFrame
all_events = []
event_id_counter = 1
min_total_events = 15000
max_total_events = 25000

# To control the total number of events more effectively, we can estimate average events per user
# and cap the total generation if it goes too high.
avg_events_per_user_base = (min_total_events + max_total_events) / 2 / num_users

for idx, user in users_df.iterrows():
    user_id = user['user_id']
    signup_date = user['signup_date']
    propensity = user['engagement_propensity']
    subscription = user['subscription_plan']
    is_high_future_engager = user['is_high_future_engager']

    # Determine number of events for this user based on propensity, with randomness
    # Higher propensity users get more events
    num_user_events = int(np.random.normal(avg_events_per_user_base * propensity, avg_events_per_user_base / 2))
    num_user_events = max(3, num_user_events) # At least 3 events per user

    # Generate events for this user
    for _ in range(num_user_events):
        # Cap max total events for efficiency
        if event_id_counter > max_total_events:
            break

        # Ensure event_timestamp is after signup_date, spread over up to 5 years from signup
        event_time_offset_days = np.random.randint(1, 365 * 5 + 1)
        event_timestamp = signup_date + pd.to_timedelta(event_time_offset_days, unit='days')
        
        event_types = ['app_open', 'post_view', 'post_like', 'comment', 'share', 'profile_update', 'settings_change']
        
        # Base probabilities
        event_probs = {
            'app_open': 0.25, 'post_view': 0.30, 'post_like': 0.15,
            'comment': 0.08, 'share': 0.05, 'profile_update': 0.10, 'settings_change': 0.07
        }

        # Adjust probabilities based on subscription plan
        if subscription == 'Premium_Monthly' or subscription == 'Premium_Annual':
            event_probs['app_open'] *= 0.8
            event_probs['post_view'] *= 0.9
            event_probs['post_like'] *= 1.5
            event_probs['comment'] *= 2.0
            event_probs['share'] *= 2.0
            event_probs['profile_update'] *= 1.2
            event_probs['settings_change'] *= 1.1
        elif subscription == 'Free':
            event_probs['app_open'] *= 1.5
            event_probs['post_view'] *= 1.2
            event_probs['post_like'] *= 0.5
            event_probs['comment'] *= 0.3
            event_probs['share'] *= 0.2
            event_probs['profile_update'] *= 0.8
            event_probs['settings_change'] *= 0.9

        # Further adjust for users destined for high future engagement
        if is_high_future_engager:
            event_probs['post_like'] *= 1.8
            event_probs['comment'] *= 1.8
            event_probs['share'] *= 1.8
            event_probs['profile_update'] *= 1.5
        
        # Normalize probabilities
        total_prob = sum(event_probs.values())
        for k in event_probs:
            event_probs[k] /= total_prob
            
        event_type = np.random.choice(list(event_types), p=list(event_probs.values()))

        duration_seconds = 0
        if event_type == 'app_open':
            duration_seconds = np.random.randint(30, 601) # 30 seconds to 10 minutes
        elif event_type == 'post_view':
            duration_seconds = np.random.randint(1, 15) # Very short duration
        else: # Most other engagement actions are instantaneous or very short
            duration_seconds = np.random.randint(0, 5)

        all_events.append({
            'event_id': event_id_counter,
            'user_id': user_id,
            'event_timestamp': event_timestamp,
            'event_type': event_type,
            'duration_seconds': duration_seconds
        })
        event_id_counter += 1
    
    # If we've generated enough events globally, stop adding more
    if event_id_counter > max_total_events:
        break

events_df = pd.DataFrame(all_events)

# Ensure event count is within bounds, adjust if needed (e.g., resample or regenerate)
if len(events_df) < min_total_events:
    print(f"Warning: Generated only {len(events_df)} events, less than minimum {min_total_events}. Consider adjusting generation parameters.")
elif len(events_df) > max_total_events:
    events_df = events_df.sample(max_total_events, random_state=42).reset_index(drop=True)
    print(f"Warning: Generated too many events, sampled down to {len(events_df)} events.")


# Sort events_df
events_df = events_df.sort_values(by=['user_id', 'event_timestamp']).reset_index(drop=True)

# Clean up temporary columns from users_df
users_df = users_df.drop(columns=['engagement_propensity', 'is_high_future_engager'])


print(f"Generated {len(users_df)} users and {len(events_df)} events.")
print("Users DataFrame head:")
print(users_df.head())
print("\nEvents DataFrame head:")
print(events_df.head())
print("\nEvents DataFrame dtypes:")
print(events_df.dtypes)
print("\nUsers DataFrame dtypes:")
print(users_df.dtypes)

# --- 2. Load into SQLite & SQL Feature Engineering ---
conn = sqlite3.connect(':memory:')

# Convert datetime columns to string for SQLite insertion
users_df['signup_date'] = users_df['signup_date'].dt.strftime('%Y-%m-%d %H:%M:%S')
events_df['event_timestamp'] = events_df['event_timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')

users_df.to_sql('users', conn, if_exists='replace', index=False)
events_df.to_sql('events', conn, if_exists='replace', index=False)

# SQL Query for early user engagement features
sql_query = """
SELECT
    u.user_id,
    u.signup_date,
    u.country,
    u.subscription_plan,
    COALESCE(COUNT(e.event_id), 0) AS num_events_first_14d,
    COALESCE(SUM(CASE WHEN e.event_type = 'app_open' THEN e.duration_seconds ELSE 0 END), 0) AS total_app_open_duration_first_14d,
    COALESCE(SUM(CASE WHEN e.event_type = 'post_like' THEN 1 ELSE 0 END), 0) AS num_likes_first_14d,
    COALESCE(SUM(CASE WHEN e.event_type = 'comment' THEN 1 ELSE 0 END), 0) AS num_comments_first_14d,
    COALESCE(SUM(CASE WHEN e.event_type = 'share' THEN 1 ELSE 0 END), 0) AS num_shares_first_14d,
    COALESCE(COUNT(DISTINCT strftime('%Y-%m-%d', e.event_timestamp)), 0) AS days_with_activity_first_14d,
    COALESCE(MAX(CASE WHEN e.event_type = 'profile_update' THEN 1 ELSE 0 END), 0) AS has_profile_update_first_14d
FROM
    users AS u
LEFT JOIN
    events AS e
ON
    u.user_id = e.user_id AND
    e.event_timestamp >= u.signup_date AND
    e.event_timestamp <= DATE(u.signup_date, '+14 days')
GROUP BY
    u.user_id, u.signup_date, u.country, u.subscription_plan
ORDER BY
    u.user_id;
"""

user_early_features_df = pd.read_sql_query(sql_query, conn)
print("\nUser Early Features DataFrame head (from SQL):")
print(user_early_features_df.head())
print("\nUser Early Features DataFrame dtypes:")
print(user_early_features_df.dtypes)

# Close the SQLite connection
conn.close()

# --- 3. Pandas Feature Engineering & Multi-Class Target Creation ---

# Handle NaN values (COALESCE in SQL should have handled most, but for safety)
# This will also ensure correct type conversion for int columns if they became float due to NaNs
for col in ['num_events_first_14d', 'total_app_open_duration_first_14d', 'num_likes_first_14d',
            'num_comments_first_14d', 'num_shares_first_14d', 'days_with_activity_first_14d',
            'has_profile_update_first_14d']:
    user_early_features_df[col] = user_early_features_df[col].fillna(0).astype(int)

# Convert signup_date to datetime objects
user_early_features_df['signup_date'] = pd.to_datetime(user_early_features_df['signup_date'])

# Calculate account_age_at_cutoff_days
user_early_features_df['account_age_at_cutoff_days'] = 14 # Always 14 days for this window

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
# Restore original events_df timestamps to datetime for calculations (if modified for SQLite)
# If events_df was not modified in place, this step is redundant but safe.
events_df['event_timestamp'] = pd.to_datetime(events_df['event_timestamp'])
# Also ensure original users_df signup_date is datetime type
users_df['signup_date'] = pd.to_datetime(users_df['signup_date'])

# Calculate early_behavior_cutoff_date for each user
users_df['early_behavior_cutoff_date'] = users_df['signup_date'] + pd.to_timedelta(14, unit='days')

# Calculate total_events_after_14d for each user
# Merge users_df with events_df to get each user's cutoff date for filtering
events_with_cutoff = events_df.merge(
    users_df[['user_id', 'early_behavior_cutoff_date']],
    on='user_id',
    how='left'
)
# Filter events that occurred strictly AFTER the 14-day cutoff
future_events = events_with_cutoff[events_with_cutoff['event_timestamp'] > events_with_cutoff['early_behavior_cutoff_date']]

total_events_after_14d = future_events.groupby('user_id').size().reset_index(name='total_events_after_14d')

# Merge with user_early_features_df
user_early_features_df = user_early_features_df.merge(
    total_events_after_14d, on='user_id', how='left'
)
user_early_features_df['total_events_after_14d'] = user_early_features_df['total_events_after_14d'].fillna(0).astype(int)

# Define target tiers
user_early_features_df['future_engagement_tier'] = 'Inactive' # Default all to 'Inactive' first

# Get non-zero events for percentile calculation
active_events = user_early_features_df.loc[user_early_features_df['total_events_after_14d'] > 0, 'total_events_after_14d']

if not active_events.empty:
    p25, p50, p75 = active_events.quantile([0.25, 0.50, 0.75]).values
    print(f"\nPercentiles for total_events_after_14d (non-zero): 25th={p25:.2f}, 50th={p50:.2f}, 75th={p75:.2f}")

    # Apply tiers using np.select for active users
    conditions = [
        (user_early_features_df['total_events_after_14d'] > 0) & (user_early_features_df['total_events_after_14d'] <= p25),
        (user_early_features_df['total_events_after_14d'] > p25) & (user_early_features_df['total_events_after_14d'] <= p50),
        (user_early_features_df['total_events_after_14d'] > p50) & (user_early_features_df['total_events_after_14d'] <= p75),
        (user_early_features_df['total_events_after_14d'] > p75)
    ]
    choices = ['Low_Engagement', 'Medium_Engagement', 'High_Engagement', 'Very_High_Engagement']
    
    # Use np.select to set engagement tiers for active users, keeping 'Inactive' for others
    user_early_features_df['future_engagement_tier'] = np.select(
        conditions, choices, default=user_early_features_df['future_engagement_tier'] # Keep existing 'Inactive'
    )
else:
    print("\nWarning: No active users found after 14 days. All users will be classified as 'Inactive'.")


# Reorder categories for better visualization and consistent order
tier_order = ['Inactive', 'Low_Engagement', 'Medium_Engagement', 'High_Engagement', 'Very_High_Engagement']
user_early_features_df['future_engagement_tier'] = pd.Categorical(
    user_early_features_df['future_engagement_tier'],
    categories=tier_order,
    ordered=True
)

print("\nDataFrame with Features and Target head:")
print(user_early_features_df.head())
print("\nTarget distribution:")
print(user_early_features_df['future_engagement_tier'].value_counts())


# Define features X and target y
numerical_features = [
    'num_events_first_14d', 'total_app_open_duration_first_14d',
    'num_likes_first_14d', 'num_comments_first_14d', 'num_shares_first_14d',
    'days_with_activity_first_14d', 'account_age_at_cutoff_days',
    'event_frequency_first_14d', 'engagement_action_ratio_first_14d'
]
categorical_features = ['country', 'subscription_plan', 'has_profile_update_first_14d']

X = user_early_features_df[numerical_features + categorical_features]
y = user_early_features_df['future_engagement_tier']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"\nTrain set size: {len(X_train)} users")
print(f"Test set size: {len(X_test)} users")
print("\nX_train head:")
print(X_train.head())
print("\ny_train value counts:")
print(y_train.value_counts())

# --- 4. Data Visualization ---

plt.style.use('seaborn-v0_8-darkgrid')
plt.figure(figsize=(16, 7))

# Violin Plot: Distribution of num_likes_first_14d by future_engagement_tier
plt.subplot(1, 2, 1)
sns.violinplot(
    x='future_engagement_tier',
    y='num_likes_first_14d',
    data=user_early_features_df,
    palette='viridis'
)
plt.title('Distribution of Likes in First 14 Days by Future Engagement Tier', fontsize=14)
plt.xlabel('Future Engagement Tier', fontsize=12)
plt.ylabel('Number of Likes (First 14 Days)', fontsize=12)
plt.xticks(rotation=45, ha='right')


# Stacked Bar Chart: Proportion of future_engagement_tier across subscription_plan
plt.subplot(1, 2, 2)
# Calculate proportions
plan_tier_proportions = user_early_features_df.groupby('subscription_plan')['future_engagement_tier'].value_counts(normalize=True).unstack(fill_value=0)
# Ensure all tier columns are present in the correct order for consistent plotting
plan_tier_proportions = plan_tier_proportions.reindex(columns=tier_order, fill_value=0)

plan_tier_proportions.plot(kind='bar', stacked=True, ax=plt.gca(), cmap='viridis')
plt.title('Future Engagement Tier Proportions by Subscription Plan', fontsize=14)
plt.xlabel('Subscription Plan', fontsize=12)
plt.ylabel('Proportion', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.legend(title='Engagement Tier', bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.show()

# --- 5. ML Pipeline & Evaluation (Multi-Class) ---

# Preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ]), numerical_features),
        ('cat', Pipeline([
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ]), categorical_features)
    ])

# Create the ML pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', HistGradientBoostingClassifier(random_state=42))
])

# Train the pipeline
pipeline.fit(X_train, y_train)

# Make predictions
y_pred = pipeline.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy on Test Set: {accuracy:.4f}")

print("\nClassification Report on Test Set:")
print(classification_report(y_test, y_pred, zero_division=0))

print("\nScript completed successfully!")