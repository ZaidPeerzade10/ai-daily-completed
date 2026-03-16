import pandas as pd
import numpy as np
import datetime
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer # Corrected import based on feedback
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, classification_report

# Set a random seed for reproducibility
np.random.seed(42)

# --- 1. Generate Synthetic Data (Pandas/Numpy) ---

print("--- Step 1: Generating Synthetic Data ---")

# 1.1 users_df
num_users = np.random.randint(500, 701)
user_ids = np.arange(1, num_users + 1)
signup_dates = pd.to_datetime('now') - pd.to_timedelta(np.random.randint(0, 5 * 365, num_users), unit='D')
acquisition_channels = np.random.choice(['Organic', 'Paid_Social', 'Referral', 'Direct'], num_users, p=[0.4, 0.3, 0.2, 0.1])
device_types = np.random.choice(['Mobile', 'Desktop', 'Tablet'], num_users, p=[0.6, 0.3, 0.1])

# Simulate retention patterns for initial data generation
# Users who are 'retained' will have more engagement in initial sessions and in the retention window
retention_rate = 0.3 # Roughly 30% of users will be 'simulated' as retained
is_retained_simulated = np.random.choice([0, 1], num_users, p=[1-retention_rate, retention_rate])

users_df = pd.DataFrame({
    'user_id': user_ids,
    'signup_date': signup_dates,
    'acquisition_channel': acquisition_channels,
    'device_type': device_types,
    'is_retained_simulated': is_retained_simulated
})

print(f"Generated users_df with {len(users_df)} rows.")

# 1.2 sessions_df
num_sessions = np.random.randint(8000, 12001)
session_ids = np.arange(1, num_sessions + 1)

session_data = []
for _ in range(num_sessions):
    user = users_df.sample(1).iloc[0]
    user_id = user['user_id']
    signup_date = user['signup_date']
    is_retained = user['is_retained_simulated']

    # Bias session start time: earlier for non-retained, spread out for retained
    max_days_after_signup = 120 if is_retained else 60
    session_start_time = signup_date + pd.to_timedelta(np.random.randint(1, max_days_after_signup), unit='D') + pd.to_timedelta(np.random.randint(0, 24*60*60), unit='s')

    # Bias session duration and page views
    if is_retained:
        session_duration = np.random.randint(90, 1800) # Longer sessions
        num_page_views = np.random.randint(5, 50) # More page views
    else:
        session_duration = np.random.randint(30, 900) # Shorter sessions
        num_page_views = np.random.randint(1, 25) # Fewer page views

    session_data.append({
        'session_id': session_ids[_],
        'user_id': user_id,
        'session_start_time': session_start_time,
        'session_duration_seconds': session_duration,
        'num_page_views': num_page_views
    })

sessions_df = pd.DataFrame(session_data)
# Add signup_date to sessions_df for easy filtering in SQL later
sessions_df = sessions_df.merge(users_df[['user_id', 'signup_date']], on='user_id', how='left')

# Sort sessions_df as required
sessions_df = sessions_df.sort_values(by=['user_id', 'session_start_time']).reset_index(drop=True)

print(f"Generated sessions_df with {len(sessions_df)} rows.")

# 1.3 page_views_df
num_page_views = np.random.randint(25000, 40001)
page_view_ids = np.arange(1, num_page_views + 1)
page_url_categories = ['Homepage', 'Product_Page', 'Category_Page', 'Cart_Page', 'Checkout_Page', 'Help_Page', 'Blog']

page_view_data = []
for _ in range(num_page_views):
    session = sessions_df.sample(1).iloc[0]
    session_id = session['session_id']
    user_id = session['user_id']
    is_retained = users_df[users_df['user_id'] == user_id]['is_retained_simulated'].iloc[0]

    # Bias page view categories
    if is_retained:
        # Higher chance of product/cart pages for retained users
        category_weights = [0.1, 0.3, 0.15, 0.15, 0.1, 0.1, 0.1]
    else:
        # Higher chance of homepage/help page for non-retained users
        category_weights = [0.25, 0.15, 0.15, 0.05, 0.05, 0.25, 0.1]
    
    page_url_category = np.random.choice(page_url_categories, p=category_weights)
    view_time = np.random.randint(5, 300)

    page_view_data.append({
        'page_view_id': page_view_ids[_],
        'session_id': session_id,
        'page_url_category': page_url_category,
        'view_time_seconds': view_time
    })

page_views_df = pd.DataFrame(page_view_data)

# Sort page_views_df as required
page_views_df = page_views_df.sort_values(by=['session_id', 'view_time_seconds']).reset_index(drop=True)

print(f"Generated page_views_df with {len(page_views_df)} rows.")

# Drop the temporary 'is_retained_simulated' column from users_df
users_df = users_df.drop(columns=['is_retained_simulated'])
print("Synthetic data generation complete.")

# --- 2. Load into SQLite & SQL Feature Engineering ---

print("\n--- Step 2: Loading data into SQLite and SQL Feature Engineering ---")

conn = sqlite3.connect(':memory:')

# Convert datetime objects to string format compatible with SQLite for easy filtering
users_df['signup_date'] = users_df['signup_date'].dt.strftime('%Y-%m-%d %H:%M:%S')
sessions_df['session_start_time'] = sessions_df['session_start_time'].dt.strftime('%Y-%m-%d %H:%M:%S')

users_df.to_sql('users', conn, if_exists='replace', index=False)
sessions_df.to_sql('sessions', conn, if_exists='replace', index=False)
page_views_df.to_sql('page_views', conn, if_exists='replace', index=False)

# SQL Query for initial 7-day behavior
sql_query = """
WITH UserTimeWindows AS (
    SELECT
        user_id,
        signup_date,
        acquisition_channel,
        device_type,
        STRFTIME('%Y-%m-%d %H:%M:%S', JULIANDAY(signup_date) + 7) AS initial_behavior_cutoff_date
    FROM users
),
AggSessions AS (
    SELECT
        s.user_id,
        COUNT(DISTINCT s.session_id) AS num_sessions_first_7d,
        SUM(s.session_duration_seconds) AS total_session_duration_first_7d,
        AVG(CAST(s.session_duration_seconds AS REAL)) AS avg_session_duration_first_7d,
        COUNT(DISTINCT STRFTIME('%Y-%m-%d', s.session_start_time)) AS days_with_activity_first_7d,
        AVG(CAST(s.num_page_views AS REAL)) AS avg_page_views_per_session_first_7d
    FROM sessions s
    JOIN UserTimeWindows utw ON s.user_id = utw.user_id
    WHERE s.session_start_time <= utw.initial_behavior_cutoff_date
    GROUP BY s.user_id
),
AggPageViews AS (
    SELECT
        s.user_id,
        COUNT(pv.page_view_id) AS total_page_views_first_7d,
        SUM(CASE WHEN pv.page_url_category = 'Product_Page' THEN 1 ELSE 0 END) AS num_product_page_views_first_7d,
        SUM(CASE WHEN pv.page_url_category = 'Cart_Page' THEN 1 ELSE 0 END) AS num_cart_page_views_first_7d
    FROM page_views pv
    JOIN sessions s ON pv.session_id = s.session_id
    JOIN UserTimeWindows utw ON s.user_id = utw.user_id
    WHERE s.session_start_time <= utw.initial_behavior_cutoff_date
    GROUP BY s.user_id
)
SELECT
    utw.user_id,
    utw.signup_date,
    utw.acquisition_channel,
    utw.device_type,
    asess.num_sessions_first_7d,
    asess.total_session_duration_first_7d,
    asess.avg_session_duration_first_7d,
    apv.total_page_views_first_7d,
    apv.num_product_page_views_first_7d,
    apv.num_cart_page_views_first_7d,
    asess.days_with_activity_first_7d,
    asess.avg_page_views_per_session_first_7d
FROM UserTimeWindows utw
LEFT JOIN AggSessions asess ON utw.user_id = asess.user_id
LEFT JOIN AggPageViews apv ON utw.user_id = apv.user_id;
"""

user_initial_features_df = pd.read_sql_query(sql_query, conn)
conn.close()

print(f"SQL feature engineering complete. Resulting DataFrame has {len(user_initial_features_df)} rows.")

# --- 3. Pandas Feature Engineering & Binary Target Creation ---

print("\n--- Step 3: Pandas Feature Engineering & Binary Target Creation ---")

# Handle NaN values
fill_zero_cols = [
    'num_sessions_first_7d', 'total_session_duration_first_7d',
    'total_page_views_first_7d', 'num_product_page_views_first_7d',
    'num_cart_page_views_first_7d', 'days_with_activity_first_7d'
]
user_initial_features_df[fill_zero_cols] = user_initial_features_df[fill_zero_cols].fillna(0).astype(int)

fill_zero_float_cols = ['avg_session_duration_first_7d', 'avg_page_views_per_session_first_7d']
user_initial_features_df[fill_zero_float_cols] = user_initial_features_df[fill_zero_float_cols].fillna(0.0)

# Convert signup_date to datetime objects
user_initial_features_df['signup_date'] = pd.to_datetime(user_initial_features_df['signup_date'])

# Calculate account_age_at_cutoff_days
user_initial_features_df['account_age_at_cutoff_days'] = 7

# Calculate engagement_ratio_first_7d
user_initial_features_df['engagement_ratio_first_7d'] = (
    user_initial_features_df['num_product_page_views_first_7d'] + user_initial_features_df['num_cart_page_views_first_7d']
) / (user_initial_features_df['total_page_views_first_7d'] + 1)

# Calculate session_frequency_first_7d
user_initial_features_df['session_frequency_first_7d'] = user_initial_features_df['num_sessions_first_7d'] / 7.0

# Create the Binary Target `is_retained_after_30_days`
# Re-load original sessions_df with datetime objects for this step
sessions_df_original = pd.DataFrame(session_data) # Use the raw generated data to avoid string conversion issues
sessions_df_original = sessions_df_original.merge(users_df[['user_id', 'signup_date']], on='user_id', how='left')

# Calculate retention window for each session based on its user's signup date
sessions_df_original['retention_start_date'] = sessions_df_original['signup_date'] + pd.Timedelta(days=30)
sessions_df_original['retention_end_date'] = sessions_df_original['signup_date'] + pd.Timedelta(days=90)

# Filter sessions that fall within the retention window
retained_sessions = sessions_df_original[
    (sessions_df_original['session_start_time'] >= sessions_df_original['retention_start_date']) &
    (sessions_df_original['session_start_time'] < sessions_df_original['retention_end_date'])
]

# Get unique user_ids who had at least one session in the retention window
retained_users_ids = retained_sessions['user_id'].unique()

# Assign the binary target
user_initial_features_df['is_retained_after_30_days'] = user_initial_features_df['user_id'].isin(retained_users_ids).astype(int)

print(f"Target variable 'is_retained_after_30_days' created. Retention rate: {user_initial_features_df['is_retained_after_30_days'].mean():.2%}")

# Define features X and target y
numerical_features = [
    'num_sessions_first_7d', 'total_session_duration_first_7d', 'avg_session_duration_first_7d',
    'total_page_views_first_7d', 'num_product_page_views_first_7d', 'num_cart_page_views_first_7d',
    'days_with_activity_first_7d', 'avg_page_views_per_session_first_7d',
    'account_age_at_cutoff_days', 'engagement_ratio_first_7d', 'session_frequency_first_7d'
]
categorical_features = ['acquisition_channel', 'device_type']

X = user_initial_features_df[numerical_features + categorical_features]
y = user_initial_features_df['is_retained_after_30_days']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

print(f"Data split into training ({len(X_train)} samples) and testing ({len(X_test)} samples) sets.")
print("Pandas feature engineering and target creation complete.")


# --- 4. Data Visualization ---

print("\n--- Step 4: Generating Data Visualizations ---")

plt.figure(figsize=(14, 6))

# Plot 1: Violin plot for total_session_duration_first_7d vs. retention
plt.subplot(1, 2, 1)
sns.violinplot(x='is_retained_after_30_days', y='total_session_duration_first_7d', data=user_initial_features_df)
plt.title('Total Session Duration in First 7 Days by Retention Status')
plt.xlabel('Retained After 30 Days (0=No, 1=Yes)')
plt.ylabel('Total Session Duration (seconds)')
plt.xticks([0, 1], ['Not Retained', 'Retained'])

# Plot 2: Stacked bar chart for acquisition_channel vs. retention
plt.subplot(1, 2, 2)
retention_by_channel = user_initial_features_df.groupby('acquisition_channel')['is_retained_after_30_days'].value_counts(normalize=True).unstack().fillna(0)
retention_by_channel.plot(kind='bar', stacked=True, ax=plt.gca(), cmap='viridis')
plt.title('Retention Proportion by Acquisition Channel')
plt.xlabel('Acquisition Channel')
plt.ylabel('Proportion')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Retained', labels=['Not Retained', 'Retained'])
plt.tight_layout()
plt.show()

print("Visualizations displayed.")

# --- 5. ML Pipeline & Evaluation ---

print("\n--- Step 5: Building and Evaluating ML Pipeline ---")

# Create preprocessing pipelines for numerical and categorical features
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Create a column transformer to apply different transformations to different columns
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Create the full pipeline
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', HistGradientBoostingClassifier(random_state=42))
])

# Train the pipeline
print("Training the ML pipeline...")
model_pipeline.fit(X_train, y_train)
print("Training complete.")

# Predict probabilities on the test set
y_pred_proba = model_pipeline.predict_proba(X_test)[:, 1]
y_pred = model_pipeline.predict(X_test) # For classification report

# Evaluate the model
roc_auc = roc_auc_score(y_test, y_pred_proba)
class_report = classification_report(y_test, y_pred)

print(f"\n--- Model Evaluation on Test Set ---")
print(f"ROC AUC Score: {roc_auc:.4f}")
print("\nClassification Report:")
print(class_report)

print("\nML pipeline and evaluation complete. Script execution finished.")