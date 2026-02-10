import pandas as pd
import numpy as np
import datetime
import sqlite3
import matplotlib.pyplot as plt
import io # Required for Matplotlib to save figures without display
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, classification_report

# Set Matplotlib backend to 'Agg' for non-interactive environments (like script execution without display)
plt.switch_backend('Agg')

# --- 1. Generate Synthetic Data (Pandas/Numpy) ---

# Random seed for reproducibility
np.random.seed(42)

# --- users_df ---
num_users = np.random.randint(500, 701)
users_df = pd.DataFrame({
    'user_id': np.arange(1, num_users + 1),
    'signup_date': pd.to_datetime(np.random.choice(pd.date_range(start=datetime.date.today() - datetime.timedelta(days=3*365), end=datetime.date.today()), size=num_users)),
    'age': np.random.randint(18, 71, size=num_users),
    'gender': np.random.choice(['Male', 'Female', 'Other'], size=num_users, p=[0.45, 0.5, 0.05]),
    'region': np.random.choice(['North', 'South', 'East', 'West', 'Central'], size=num_users, p=[0.25, 0.2, 0.2, 0.2, 0.15])
})

# --- ads_df ---
num_ads = np.random.randint(100, 151)
ad_categories = ['Fashion', 'Tech', 'Travel', 'Food', 'Finance', 'Education', 'Health']
ad_types = ['Banner', 'Video', 'Text', 'Pop-up']
target_age_groups = ['18-24', '25-34', '35-44', '45-54', '55+']
ads_df = pd.DataFrame({
    'ad_id': np.arange(1, num_ads + 1),
    'ad_category': np.random.choice(ad_categories, size=num_ads),
    'ad_type': np.random.choice(ad_types, size=num_ads),
    'target_audience_age_group': np.random.choice(target_age_groups, size=num_ads, p=[0.2, 0.25, 0.2, 0.15, 0.2])
})

# --- impressions_df ---
num_impressions = np.random.randint(5000, 8001)
impressions_df = pd.DataFrame({
    'impression_id': np.arange(1, num_impressions + 1),
    'user_id': np.random.choice(users_df['user_id'], size=num_impressions),
    'ad_id': np.random.choice(ads_df['ad_id'], size=num_impressions),
    'device_type': np.random.choice(['Mobile', 'Desktop', 'Tablet'], size=num_impressions, p=[0.6, 0.3, 0.1])
})

# Merge user and ad data for click simulation
impressions_merged_df = impressions_df.merge(users_df, on='user_id', how='left')
impressions_merged_df = impressions_merged_df.merge(ads_df, on='ad_id', how='left')

# Ensure impression_date is after signup_date
# Generate random days after signup for each impression
max_days_after_signup = 365 * 2 # up to 2 years after signup
impressions_merged_df['days_since_signup'] = np.random.randint(1, max_days_after_signup, size=num_impressions)
impressions_merged_df['impression_date'] = impressions_merged_df['signup_date'] + pd.to_timedelta(impressions_merged_df['days_since_signup'], unit='D')
impressions_merged_df = impressions_merged_df.drop(columns=['signup_date', 'days_since_signup']) # Drop temporary signup date, will rejoin later

# Simulate realistic CTR patterns
base_ctr = 0.12 # Overall approximate 10-15% click rate
click_probabilities = np.full(num_impressions, base_ctr)

# Helper to parse age group
def parse_age_group(age_group_str):
    if '+' in age_group_str:
        return int(age_group_str.replace('+', '')), 100 # Assume 100 as max age
    parts = age_group_str.split('-')
    return int(parts[0]), int(parts[1])

# Apply biases using vectorized operations
min_ages, max_ages = zip(*impressions_merged_df['target_audience_age_group'].apply(parse_age_group))
min_ages = np.array(min_ages)
max_ages = np.array(max_ages)

user_age = impressions_merged_df['age'].values
ad_category = impressions_merged_df['ad_category'].values
region = impressions_merged_df['region'].values
device_type = impressions_merged_df['device_type'].values

# Age match bias
age_match_mask = (user_age >= min_ages) & (user_age <= max_ages)
click_probabilities[age_match_mask] *= 1.5

# Ad category / region bias
fashion_south_mask = (ad_category == 'Fashion') & (region == 'South')
click_probabilities[fashion_south_mask] *= 1.3

tech_east_mask = (ad_category == 'Tech') & (region == 'East')
click_probabilities[tech_east_mask] *= 1.4

# Device type bias
mobile_mask = (device_type == 'Mobile')
click_probabilities[mobile_mask] *= 1.2

tablet_mask = (device_type == 'Tablet')
click_probabilities[tablet_mask] *= 0.8

# Clip probabilities to [0, 1]
click_probabilities = np.clip(click_probabilities, 0.01, 0.99)

# Generate clicks based on probabilities
impressions_df['was_clicked'] = (np.random.rand(num_impressions) < click_probabilities).astype(int)

# Rejoin only necessary columns for impressions_df
impressions_df = impressions_df[['impression_id', 'user_id', 'ad_id', 'device_type', 'was_clicked']].merge(
    impressions_merged_df[['impression_id', 'impression_date']], on='impression_id', how='left'
)

# Sort impressions_df by user_id then impression_date for SQL window functions
impressions_df = impressions_df.sort_values(by=['user_id', 'impression_date']).reset_index(drop=True)

# Print initial DataFrame info
print(f"Generated {len(users_df)} users, {len(ads_df)} ads, {len(impressions_df)} impressions.")
print(f"Overall CTR in synthetic data: {impressions_df['was_clicked'].mean():.2%}\n")

# --- 2. Load into SQLite & SQL Feature Engineering ---
conn = sqlite3.connect(':memory:')

users_df.to_sql('users', conn, index=False, if_exists='replace')
ads_df.to_sql('ads', conn, index=False, if_exists='replace')
# Convert datetime to string for SQLite storage for impressions_df before loading
impressions_df_sql = impressions_df.copy()
impressions_df_sql['impression_date'] = impressions_df_sql['impression_date'].dt.strftime('%Y-%m-%d %H:%M:%S')
impressions_df_sql.to_sql('impressions', conn, index=False, if_exists='replace')

# The SQL query for feature engineering
sql_query = """
WITH CombinedData AS (
    SELECT
        i.impression_id,
        i.user_id,
        i.ad_id,
        i.impression_date,
        i.device_type,
        i.was_clicked,
        u.signup_date,
        u.age,
        u.gender,
        u.region,
        a.ad_category,
        a.ad_type,
        a.target_audience_age_group
    FROM
        impressions i
    JOIN
        users u ON i.user_id = u.user_id
    JOIN
        ads a ON i.ad_id = a.ad_id
),
LaggedImpressions AS (
    SELECT
        *,
        LAG(impression_date, 1, NULL) OVER (PARTITION BY user_id ORDER BY impression_date) AS prev_user_impression_date_str
    FROM CombinedData
)
SELECT
    impression_id,
    user_id,
    ad_id,
    impression_date,
    device_type,
    was_clicked,
    signup_date,
    age,
    gender,
    region,
    ad_category,
    ad_type,
    target_audience_age_group,

    -- User's past interactions
    COALESCE(CAST(COUNT(impression_id) OVER (PARTITION BY user_id ORDER BY impression_date ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS REAL), 0) AS user_past_total_impressions,
    COALESCE(CAST(SUM(was_clicked) OVER (PARTITION BY user_id ORDER BY impression_date ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS REAL), 0) AS user_past_total_clicks,
    
    -- Ad's past interactions
    COALESCE(CAST(COUNT(impression_id) OVER (PARTITION BY ad_id ORDER BY impression_date ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS REAL), 0) AS ad_past_total_impressions,
    COALESCE(CAST(SUM(was_clicked) OVER (PARTITION BY ad_id ORDER BY impression_date ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS REAL), 0) AS ad_past_total_clicks,
    
    -- Days since last user impression
    CASE
        WHEN prev_user_impression_date_str IS NULL THEN NULL
        ELSE JULIANDAY(impression_date) - JULIANDAY(prev_user_impression_date_str)
    END AS days_since_last_user_impression

FROM LaggedImpressions
ORDER BY user_id, impression_date;
"""

impression_features_df = pd.read_sql_query(sql_query, conn)
conn.close()

print("SQL Feature Engineering complete. First 5 rows of `impression_features_df`:")
print(impression_features_df.head())
print(f"Shape after SQL features: {impression_features_df.shape}\n")

# --- 3. Pandas Feature Engineering & Binary Target Creation ---

# Handle NaN values for prior interaction features
fill_cols_zero_int = ['user_past_total_impressions', 'user_past_total_clicks', 'ad_past_total_impressions', 'ad_past_total_clicks']
impression_features_df[fill_cols_zero_int] = impression_features_df[fill_cols_zero_int].fillna(0)

# Calculate CTRs in pandas to handle division by zero
impression_features_df['user_past_ctr'] = np.where(
    impression_features_df['user_past_total_impressions'] > 0,
    impression_features_df['user_past_total_clicks'] / impression_features_df['user_past_total_impressions'],
    0.0
)
impression_features_df['ad_past_ctr'] = np.where(
    impression_features_df['ad_past_total_impressions'] > 0,
    impression_features_df['ad_past_total_clicks'] / impression_features_df['ad_past_total_impressions'],
    0.0
)

# Fill days_since_last_user_impression with a large sentinel value for first impressions
impression_features_df['days_since_last_user_impression'] = impression_features_df['days_since_last_user_impression'].fillna(9999)

# Convert date columns to datetime objects
impression_features_df['signup_date'] = pd.to_datetime(impression_features_df['signup_date'])
impression_features_df['impression_date'] = pd.to_datetime(impression_features_df['impression_date'])

# Calculate user_account_age_at_impression_days
impression_features_df['user_account_age_at_impression_days'] = (
    impression_features_df['impression_date'] - impression_features_df['signup_date']
).dt.days

# Create user_ad_age_match
def check_age_match(row):
    user_age = row['age']
    target_group = row['target_audience_age_group']
    min_age, max_age = parse_age_group(target_group)
    return 1 if (user_age >= min_age) and (user_age <= max_age) else 0

impression_features_df['user_ad_age_match'] = impression_features_df.apply(check_age_match, axis=1)

# Define features X and target y
numerical_features = [
    'age',
    'user_account_age_at_impression_days',
    'user_past_total_impressions',
    'user_past_total_clicks',
    'user_past_ctr',
    'days_since_last_user_impression',
    'ad_past_total_impressions',
    'ad_past_total_clicks',
    'ad_past_ctr'
]
categorical_features = [
    'gender',
    'region',
    'ad_category',
    'ad_type',
    'device_type',
    'target_audience_age_group',
    'user_ad_age_match' # This is treated as categorical for one-hot encoding
]

X = impression_features_df[numerical_features + categorical_features]
y = impression_features_df['was_clicked']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
print(f"Train set CTR: {y_train.mean():.2%}, Test set CTR: {y_test.mean():.2%}\n")

# --- 4. Data Visualization ---

print("Generating plots (output to stdout as text for typical script execution).")

# Plot 1: CTR by Device Type
ctr_by_device = impression_features_df.groupby('device_type')['was_clicked'].mean()
fig1, ax1 = plt.subplots(figsize=(8, 5))
ctr_by_device.plot(kind='bar', ax=ax1, color='skyblue')
ax1.set_title('CTR by Device Type')
ax1.set_ylabel('Click-Through Rate')
ax1.set_xlabel('Device Type')
ax1.tick_params(axis='x', rotation=45)
fig1.tight_layout()
# Instead of showing, save to a buffer and print a placeholder.
buf1 = io.BytesIO()
fig1.savefig(buf1, format='png')
plt.close(fig1)
print("Plot 1: CTR by Device Type - Bar chart generated.")
# For typical stdout output, we'd just confirm generation. If this were a notebook, we'd show it.

# Plot 2: Stacked bar chart of was_clicked distribution across ad_category
fig2, ax2 = plt.subplots(figsize=(10, 6))
ad_category_clicks = pd.crosstab(impression_features_df['ad_category'], impression_features_df['was_clicked'])
ad_category_clicks.plot(kind='bar', stacked=True, ax=ax2, colormap='Paired')
ax2.set_title('Distribution of Clicks (0/1) by Ad Category')
ax2.set_ylabel('Number of Impressions')
ax2.set_xlabel('Ad Category')
ax2.legend(title='Was Clicked', labels=['No Click', 'Click'])
ax2.tick_params(axis='x', rotation=45)
fig2.tight_layout()
buf2 = io.BytesIO()
fig2.savefig(buf2, format='png')
plt.close(fig2)
print("Plot 2: Distribution of Clicks by Ad Category - Stacked bar chart generated.\n")


# --- 5. ML Pipeline & Evaluation (Binary Classification) ---

# Preprocessing steps for numerical features
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# Preprocessing steps for categorical features
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Create a preprocessor using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='drop' # Drop any columns not specified
)

# Create the full pipeline
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', HistGradientBoostingClassifier(random_state=42))
])

# Train the pipeline
print("Training the ML pipeline...")
model_pipeline.fit(X_train, y_train)
print("Training complete.\n")

# Predict probabilities on the test set
y_pred_proba = model_pipeline.predict_proba(X_test)[:, 1] # Probability of the positive class (1)
y_pred = model_pipeline.predict(X_test) # Predict class labels

# Evaluate the model
roc_auc = roc_auc_score(y_test, y_pred_proba)
class_report = classification_report(y_test, y_pred)

print(f"ROC AUC Score on Test Set: {roc_auc:.4f}")
print("\nClassification Report on Test Set:")
print(class_report)