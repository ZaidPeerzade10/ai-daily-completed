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
from sklearn.metrics import classification_report
import warnings

# Suppress specific future warnings from pandas or sklearn if they clutter output
warnings.simplefilter(action='ignore', category=FutureWarning)

# --- 1. Synthetic Data Generation (Pandas/Numpy) ---

np.random.seed(42)

# Global parameters for data generation
NUM_USERS = np.random.randint(1000, 1501)
NUM_ACTIVITIES = np.random.randint(20000, 30001)

# Dates
current_time = pd.Timestamp.now()
# Latest activity date overall will be 1 week ago
max_activity_date_overall = current_time - pd.Timedelta(weeks=1) 

# Fix for the previous error: pd.Timedelta does not support 'years' keyword.
# Using days for timedelta to represent years.
FIVE_YEARS_AGO = max_activity_date_overall - pd.Timedelta(days=5 * 365)
THREE_YEARS_AGO = max_activity_date_overall - pd.Timedelta(days=3 * 365)

# 1.1 users_df
user_ids = np.arange(1, NUM_USERS + 1)
# Generate signup dates between 3 and 5 years ago relative to max_activity_date_overall
signup_dates = pd.to_datetime(np.random.uniform(THREE_YEARS_AGO.timestamp(), FIVE_YEARS_AGO.timestamp(), NUM_USERS), unit='s')
countries = np.random.choice(['USA', 'CAN', 'MEX'], NUM_USERS, p=[0.6, 0.2, 0.2])
device_types = np.random.choice(['Mobile', 'Desktop'], NUM_USERS, p=[0.7, 0.3])
age_groups = np.random.choice(['18-24', '25-44', '45+'], NUM_USERS, p=[0.3, 0.5, 0.2])

users_df = pd.DataFrame({
    'user_id': user_ids,
    'signup_date': signup_dates,
    'country': countries,
    'device_type': device_types,
    'age_group': age_groups
})

# 1.2 activity_df
# Simulate varying activity levels:
# Create a skewed distribution for user activity where some users are much more active.
active_users_pool = np.random.choice(user_ids, size=int(NUM_USERS * 0.1), replace=False) # 10% very active users
less_active_users_pool = np.random.choice(list(set(user_ids) - set(active_users_pool)), size=int(NUM_USERS * 0.9), replace=False) # Remaining 90%
activity_user_ids = np.random.choice(
    np.concatenate([active_users_pool, less_active_users_pool]),
    size=NUM_ACTIVITIES,
    p=np.concatenate([np.repeat(2/len(active_users_pool), len(active_users_pool)), # Give active users higher prob
                      np.repeat(0.5/len(less_active_users_pool), len(less_active_users_pool))]) # Less active users lower prob
)

# Generate activity dates, ensuring they are after signup_date and skewed towards recent activity.
# Create a temporary DataFrame to link activities to signup dates and device types
pre_activity_df = pd.DataFrame({
    'activity_id': np.arange(1, NUM_ACTIVITIES + 1),
    'user_id': activity_user_ids,
    'activity_type': np.random.choice(['Page View', 'Login', 'Add to Cart', 'Search', 'Comment'], NUM_ACTIVITIES, p=[0.4, 0.2, 0.15, 0.15, 0.1])
})

pre_activity_df = pre_activity_df.merge(users_df[['user_id', 'signup_date', 'device_type']], on='user_id', how='left')
pre_activity_df['signup_date'] = pre_activity_df['signup_date'].dt.tz_localize(None) # Remove timezone for comparison if present

# Generate random days relative to signup, skewed towards more recent activity for a user
# Max possible duration for an activity since signup up to max_activity_date_overall
max_duration_days = (max_activity_date_overall - pre_activity_df['signup_date'].min()).days
random_duration_factors = np.random.rand(NUM_ACTIVITIES) # Uniform random numbers
# Skew towards more recent activity (closer to max_activity_date_overall) by using a power.
# r^2 skews towards earlier, r^0.5 skews towards later. We want activity dropping off, so more recent should be common.
# Let's generate days *relative to signup_date* instead, then cap it at max_activity_date_overall.
# This makes it such that activity generally occurs, but drops off near the end.
# A small `r**2` value will result in larger (earlier) activity dates (relative to current).
# To make activity more common recently, we need smaller timedelta from max_activity_date_overall.
# Let's generate a uniform random fraction (0 to 1) for the activity window for each user.
# Then, for each user's activities, we take a 'skewed' fraction of their available time.
# Available time for user: max_activity_date_overall - user_signup_date
# For activity_date, make it user_signup_date + (random_fraction^0.5 * available_time)

# Calculate available time for each activity based on user's signup_date
available_time = (max_activity_date_overall - pre_activity_df['signup_date']).dt.total_seconds()
# Generate a random factor (0 to 1) for each activity, and skew it to push activities towards recent
skew_factor = np.random.rand(NUM_ACTIVITIES)**0.5 # Values closer to 1 are more common -> more recent activity
activity_seconds_since_signup = skew_factor * available_time

pre_activity_df['activity_date'] = pre_activity_df['signup_date'] + pd.to_timedelta(activity_seconds_since_signup, unit='s')

# Ensure no activity goes beyond max_activity_date_overall
pre_activity_df['activity_date'] = pre_activity_df['activity_date'].clip(upper=max_activity_date_overall)

# Simulate Mobile users having more 'Page View' activities
# Identify mobile users with non-'Page View' activities
mobile_non_pv_indices = pre_activity_df[(pre_activity_df['device_type'] == 'Mobile') & (pre_activity_df['activity_type'] != 'Page View')].index
# Randomly change a proportion of these to 'Page View'
num_to_change = int(len(mobile_non_pv_indices) * 0.3) # Change 30% of eligible activities
change_indices = np.random.choice(mobile_non_pv_indices, size=num_to_change, replace=False)
pre_activity_df.loc[change_indices, 'activity_type'] = 'Page View'


activity_df = pre_activity_df[['activity_id', 'user_id', 'activity_date', 'activity_type']]
activity_df = activity_df.sort_values(by=['user_id', 'activity_date']).reset_index(drop=True)

print(f"Generated {len(users_df)} users and {len(activity_df)} activities.")
print(f"Users head:\n{users_df.head()}")
print(f"\nActivity head:\n{activity_df.head()}")

# --- 2. Load into SQLite & SQL Feature Engineering ---

# Create an in-memory SQLite database
conn = sqlite3.connect(':memory:')

# Load DataFrames into SQLite
# SQLite does not have a native datetime type, use TEXT for dates
users_df.to_sql('users', conn, index=False, if_exists='replace', dtype={'signup_date': 'TEXT'})
activity_df.to_sql('activity', conn, index=False, if_exists='replace', dtype={'activity_date': 'TEXT'})

# Define GLOBAL_PREDICTION_CUTOFF_DATE
# It's 7 days prior to the latest activity_date in your generated activity_df
GLOBAL_PREDICTION_CUTOFF_DATE = activity_df['activity_date'].max() - pd.Timedelta(days=7)
GLOBAL_PREDICTION_CUTOFF_DATE_STR = GLOBAL_PREDICTION_CUTOFF_DATE.strftime('%Y-%m-%d %H:%M:%S')

print(f"\nGLOBAL_PREDICTION_CUTOFF_DATE: {GLOBAL_PREDICTION_CUTOFF_DATE}")

# SQL Query for feature engineering
# Aggregates activity within the 30 days preceding GLOBAL_PREDICTION_CUTOFF_DATE
# Includes static user attributes and calculates days since last activity before cutoff.
sql_query = f"""
WITH UserActivity AS (
    SELECT
        u.user_id,
        u.signup_date,
        u.country,
        u.device_type,
        u.age_group,
        a.activity_date,
        a.activity_type
    FROM
        users u
    LEFT JOIN
        activity a ON u.user_id = a.user_id
),
AggregatedFeatures AS (
    SELECT
        user_id,
        COUNT(CASE WHEN activity_type = 'Login' THEN 1 END) AS num_logins_prev_30d,
        COUNT(CASE WHEN activity_type = 'Page View' THEN 1 END) AS num_page_views_prev_30d,
        COUNT(activity_type) AS total_activity_count_prev_30d,
        COUNT(DISTINCT activity_type) AS num_unique_activity_types_prev_30d
    FROM
        UserActivity
    WHERE
        activity_date BETWEEN datetime('{GLOBAL_PREDICTION_CUTOFF_DATE_STR}', '-30 days') AND datetime('{GLOBAL_PREDICTION_CUTOFF_DATE_STR}')
    GROUP BY
        user_id
),
LastActivityBeforeCutoff AS (
    SELECT
        user_id,
        MAX(activity_date) AS last_activity_before_cutoff
    FROM
        UserActivity
    WHERE
        activity_date <= datetime('{GLOBAL_PREDICTION_CUTOFF_DATE_STR}')
    GROUP BY
        user_id
)
SELECT
    u.user_id,
    u.signup_date,
    u.country,
    u.device_type,
    u.age_group,
    '{GLOBAL_PREDICTION_CUTOFF_DATE_STR}' AS current_cutoff_date,
    COALESCE(af.num_logins_prev_30d, 0) AS num_logins_prev_30d,
    COALESCE(af.num_page_views_prev_30d, 0) AS num_page_views_prev_30d,
    COALESCE(af.total_activity_count_prev_30d, 0) AS total_activity_count_prev_30d,
    COALESCE(
        CAST(julianday('{GLOBAL_PREDICTION_CUTOFF_DATE_STR}') - julianday(labc.last_activity_before_cutoff) AS INTEGER),
        9999
    ) AS days_since_last_activity_at_cutoff,
    COALESCE(af.num_unique_activity_types_prev_30d, 0) AS num_unique_activity_types_prev_30d
FROM
    users u
LEFT JOIN
    AggregatedFeatures af ON u.user_id = af.user_id
LEFT JOIN
    LastActivityBeforeCutoff labc ON u.user_id = labc.user_id
ORDER BY
    u.user_id;
"""

# Fetch results into a pandas DataFrame
user_features_df = pd.read_sql_query(sql_query, conn)

# Close the connection
conn.close()

print(f"\nUser features DataFrame head (from SQL):\n{user_features_df.head()}")
print(f"User features DataFrame shape: {user_features_df.shape}")

# --- 3. Pandas Feature Engineering & Multi-class Target Creation ---

# Convert date columns to datetime objects
user_features_df['signup_date'] = pd.to_datetime(user_features_df['signup_date'])
user_features_df['current_cutoff_date'] = pd.to_datetime(user_features_df['current_cutoff_date'])

# Handle NaN values for numerical aggregated features.
# 'days_since_last_activity_at_cutoff' is already handled by COALESCE in SQL with 9999.
for col in ['num_logins_prev_30d', 'num_page_views_prev_30d', 'total_activity_count_prev_30d', 'num_unique_activity_types_prev_30d']:
    user_features_df[col] = user_features_df[col].fillna(0).astype(int)

# Calculate user_tenure_at_cutoff_days
user_features_df['user_tenure_at_cutoff_days'] = (user_features_df['current_cutoff_date'] - user_features_df['signup_date']).dt.days

# Create the Multi-class Target `next_7d_engagement_category`
# Define the 7-day future window for target calculation
target_start_date = GLOBAL_PREDICTION_CUTOFF_DATE # Start *after* the cutoff date
target_end_date = GLOBAL_PREDICTION_CUTOFF_DATE + pd.Timedelta(days=7)

# Filter activities within the target window
next_7d_activity = activity_df[
    (activity_df['activity_date'] > target_start_date) & 
    (activity_df['activity_date'] <= target_end_date)
]

# Sum activities for each user in the next 7 days
next_7d_activity_count = next_7d_activity.groupby('user_id').size().reset_index(name='next_7d_activity_count')

# Merge this `next_7d_activity_count` into `user_features_df`
user_features_df = user_features_df.merge(next_7d_activity_count, on='user_id', how='left')
# Fill NaN with 0 for users with no activity in the next 7 days
user_features_df['next_7d_activity_count'] = user_features_df['next_7d_activity_count'].fillna(0).astype(int)

# Categorize `next_7d_activity_count` into 'Low', 'Medium', 'High'
# Adjust thresholds as suggested to ensure a reasonable balance across classes.
# Original: 'Low': <= 5, 'Medium': 5 < x <= 20, 'High': > 20
# For pd.cut with right=True, bins should be [lower_bound, upper_bound].
# To include 0 in 'Low' (<=5), the first bin needs to start below 0.
bins = [-1, 5, 20, np.inf] 
labels = ['Low', 'Medium', 'High']
user_features_df['next_7d_engagement_category'] = pd.cut(
    user_features_df['next_7d_activity_count'], 
    bins=bins, 
    labels=labels, 
    right=True,        # Interval (a, b] meaning values equal to 'b' go into this bin
    include_lowest=True # Includes the lowest boundary value (-1 in this case)
)

print(f"\nUser features DataFrame head (after Pandas FE and target creation):\n{user_features_df.head()}")
print(f"\nTarget distribution:\n{user_features_df['next_7d_engagement_category'].value_counts()}")

# Define features X and target y
numerical_features = [
    'num_logins_prev_30d',
    'num_page_views_prev_30d',
    'total_activity_count_prev_30d',
    'days_since_last_activity_at_cutoff',
    'num_unique_activity_types_prev_30d',
    'user_tenure_at_cutoff_days'
]
categorical_features = [
    'country',
    'device_type',
    'age_group'
]

X = user_features_df[numerical_features + categorical_features]
y = user_features_df['next_7d_engagement_category']

# Split into training and testing sets (70/30 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

print(f"\nX_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
print(f"\nTraining target distribution:\n{y_train.value_counts(normalize=True)}")
print(f"Test target distribution:\n{y_test.value_counts(normalize=True)}")


# --- 4. Data Visualization (Matplotlib/Seaborn) ---

plt.style.use('ggplot')
plt.figure(figsize=(15, 6))

# Plot 1: Violin plot of total_activity_count_prev_30d by next_7d_engagement_category
plt.subplot(1, 2, 1)
sns.violinplot(
    data=user_features_df, 
    x='next_7d_engagement_category', 
    y='total_activity_count_prev_30d', 
    order=labels, # Ensure categories are ordered as Low, Medium, High
    palette='viridis'
)
plt.title('Previous 30-Day Activity Count by Future Engagement Category')
plt.xlabel('Next 7-Day Engagement Category')
plt.ylabel('Total Activity Count (Previous 30 Days)')
plt.ylim(bottom=0) # Ensure y-axis starts at 0 for counts

# Plot 2: Stacked bar chart of next_7d_engagement_category proportions by device_type
plt.subplot(1, 2, 2)
# Calculate proportions of engagement categories for each device type
device_engagement_prop = user_features_df.groupby('device_type')['next_7d_engagement_category'].value_counts(normalize=True).unstack()
device_engagement_prop = device_engagement_prop[labels] # Ensure consistent order of categories

device_engagement_prop.plot(kind='bar', stacked=True, ax=plt.gca(), cmap='viridis')
plt.title('Future Engagement Category Proportion by Device Type')
plt.xlabel('Device Type')
plt.ylabel('Proportion')
plt.xticks(rotation=0)
plt.legend(title='Engagement Category')

plt.tight_layout()
plt.show()

# --- 5. ML Pipeline & Evaluation (Multi-class Classification) ---

# Create preprocessing steps for numerical and categorical features
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')), # Fill missing numerical values with mean
    ('scaler', StandardScaler())                 # Scale numerical features
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore')) # One-hot encode categorical features
])

# Create a preprocessor using ColumnTransformer to apply different transformations to different columns
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Create the full machine learning pipeline
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('classifier', HistGradientBoostingClassifier(random_state=42))]) # Use HistGradientBoostingClassifier

# Train the pipeline on the training data
print("\nTraining the ML pipeline...")
pipeline.fit(X_train, y_train)
print("Pipeline training complete.")

# Make predictions on the test set
y_pred = pipeline.predict(X_test)

# Evaluate the model using classification_report
print("\n--- Model Evaluation ---")
# The target_names should match the labels used for pd.cut
print(classification_report(y_test, y_pred, target_names=labels))