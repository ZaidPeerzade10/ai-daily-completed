import pandas as pd
import numpy as np
import datetime
import random
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, classification_report
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# --- 1. Generate Synthetic Data (Pandas/Numpy) ---

# Define parameters
NUM_USERS = np.random.randint(500, 701)
NUM_BROWSING_EVENTS_BASE = np.random.randint(15000, 25001)
TARGET_PURCHASE_RATE = np.random.uniform(0.20, 0.30) # Overall first-time purchase rate 20-30%

# Users DataFrame
user_ids = np.arange(1, NUM_USERS + 1)
signup_dates = pd.to_datetime('2021-01-01') + pd.to_timedelta(np.random.randint(0, 3 * 365, NUM_USERS), unit='D')
regions = np.random.choice(['North', 'South', 'East', 'West', 'Central'], NUM_USERS, p=[0.2, 0.2, 0.2, 0.2, 0.2])
referral_sources = np.random.choice(['Organic', 'Paid Search', 'Social Media', 'Referral', 'Email'], NUM_USERS, p=[0.3, 0.25, 0.2, 0.15, 0.1])
age_groups = np.random.choice(['18-24', '25-34', '35-49', '50+'], NUM_USERS, p=[0.2, 0.35, 0.3, 0.15])

users_df = pd.DataFrame({
    'user_id': user_ids,
    'signup_date': signup_dates,
    'region': regions,
    'referral_source': referral_sources,
    'age_group': age_groups
})

# Determine which users will purchase within 30 days to bias data generation
num_purchasers = int(NUM_USERS * TARGET_PURCHASE_RATE)

# Assign base purchase propensity
users_df['purchase_propensity'] = 0.4 # Base propensity
users_df.loc[users_df['referral_source'] == 'Paid Search', 'purchase_propensity'] += 0.2
users_df.loc[users_df['referral_source'] == 'Referral', 'purchase_propensity'] += 0.15
users_df.loc[users_df['age_group'] == '25-34', 'purchase_propensity'] += 0.1
users_df.loc[users_df['age_group'] == '18-24', 'purchase_propensity'] -= 0.1 # Lower for young

users_df['purchase_propensity'] = np.clip(users_df['purchase_propensity'], 0.1, 0.9)

# Explicitly mark users for purchase to hit the target rate
users_df = users_df.sort_values(by='purchase_propensity', ascending=False).reset_index(drop=True)
users_df['will_purchase_within_30d'] = 0
users_df.loc[users_df.index < num_purchasers, 'will_purchase_within_30d'] = 1
users_df = users_df.sample(frac=1, random_state=42).reset_index(drop=True) # Shuffle back


# Browsing Events DataFrame
event_types = ['view_product', 'add_to_cart', 'view_category', 'search', 'homepage_visit', 'checkout_page']
product_ids = np.arange(1001, 1101) # 100 unique product IDs

all_events_data = []
current_event_id = 1
for _, user_row in users_df.iterrows():
    user_id = user_row['user_id']
    signup_date = user_row['signup_date']
    will_purchase = user_row['will_purchase_within_30d']

    # Bias event counts and type for users who will purchase
    num_events_for_user = int(np.random.normal(loc=25, scale=8)) # Base events
    if will_purchase:
        num_events_for_user = int(num_events_for_user * np.random.uniform(1.5, 2.5)) # More events for purchasers
    num_events_for_user = max(3, num_events_for_user) # Minimum events
    
    # Events spread over a period (mostly in first 30-60 days)
    event_days_after_signup = np.random.exponential(scale=15, size=num_events_for_user)
    event_days_after_signup = event_days_after_signup[event_days_after_signup < 90] # Cap at 90 days
    
    # Ensure at least one event if the user is destined to purchase, especially in first 7 days
    if will_purchase and not any(d <= 7 for d in event_days_after_signup):
        event_days_after_signup = np.append(event_days_after_signup, np.random.randint(1, 8))
    
    if len(event_days_after_signup) == 0: continue

    event_timestamps = [signup_date + pd.to_timedelta(d, unit='D') + pd.to_timedelta(np.random.randint(0, 24*60*60), unit='s')
                        for d in event_days_after_signup]
    event_timestamps.sort()

    for ts in event_timestamps:
        e_type = np.random.choice(event_types, p=[0.3, 0.1, 0.2, 0.15, 0.2, 0.05])
        
        # Stronger bias in first 7 days for purchasers
        if will_purchase and ts <= signup_date + pd.to_timedelta(7, unit='D'):
            e_type = np.random.choice(['view_product', 'add_to_cart', 'search', 'homepage_visit'], p=[0.45, 0.25, 0.15, 0.15]) # More intent-driven events
            
        prod_id = np.random.choice(product_ids) if e_type in ['view_product', 'add_to_cart'] else None
        
        all_events_data.append({
            'event_id': current_event_id,
            'user_id': user_id,
            'event_timestamp': ts,
            'event_type': e_type,
            'product_id': prod_id
        })
        current_event_id += 1

browsing_events_df = pd.DataFrame(all_events_data)
browsing_events_df['event_timestamp'] = pd.to_datetime(browsing_events_df['event_timestamp'])
browsing_events_df = browsing_events_df.sort_values(by=['user_id', 'event_timestamp']).reset_index(drop=True)


# Purchases DataFrame
purchase_data = []
current_purchase_id = 1

for _, user_row in users_df.iterrows():
    user_id = user_row['user_id']
    signup_date = user_row['signup_date']
    will_purchase = user_row['will_purchase_within_30d']

    if will_purchase:
        # Generate exactly one purchase within 30 days for these users
        purchase_date = signup_date + pd.to_timedelta(np.random.randint(1, 31), unit='D') + pd.to_timedelta(np.random.randint(0, 24*60*60), unit='s')
        total_amount = np.random.uniform(10.0, 500.0)
        purchase_data.append({
            'purchase_id': current_purchase_id,
            'user_id': user_id,
            'purchase_date': purchase_date,
            'total_amount': total_amount
        })
        current_purchase_id += 1
    else:
        # For non-purchasers, there's a small chance of a purchase *after* 30 days
        if np.random.rand() < 0.05: # 5% chance to purchase after 30 days
            purchase_date = signup_date + pd.to_timedelta(np.random.randint(31, 180), unit='D') + pd.to_timedelta(np.random.randint(0, 24*60*60), unit='s')
            total_amount = np.random.uniform(10.0, 500.0)
            purchase_data.append({
                'purchase_id': current_purchase_id,
                'user_id': user_id,
                'purchase_date': purchase_date,
                'total_amount': total_amount
            })
            current_purchase_id += 1

purchases_df = pd.DataFrame(purchase_data)
purchases_df['purchase_date'] = pd.to_datetime(purchases_df['purchase_date'])
purchases_df = purchases_df.sort_values(by=['user_id', 'purchase_date']).reset_index(drop=True)

print(f"Generated {len(users_df)} users.")
print(f"Generated {len(browsing_events_df)} browsing events.")
print(f"Generated {len(purchases_df)} purchases (some might be outside 30-day target window).")


# --- 2. Load into SQLite & SQL Feature Engineering ---

# Connect to an in-memory SQLite database
conn = sqlite3.connect(':memory:')

# Load DataFrames into SQLite tables
users_df.to_sql('users', conn, if_exists='replace', index=False, dtype={'signup_date': 'TEXT'})
browsing_events_df.to_sql('browsing_events', conn, if_exists='replace', index=False, dtype={'event_timestamp': 'TEXT'})
purchases_df.to_sql('purchases', conn, if_exists='replace', index=False, dtype={'purchase_date': 'TEXT'})

# SQL Query for early browsing behavior features
sql_query = """
WITH UserEvents7d AS (
    SELECT
        b.user_id,
        b.event_timestamp,
        b.event_type,
        u.signup_date,
        DATE(u.signup_date, '+7 days') AS early_behavior_cutoff_date
    FROM browsing_events b
    JOIN users u ON b.user_id = u.user_id
    WHERE b.event_timestamp >= u.signup_date
      AND b.event_timestamp <= DATE(u.signup_date, '+7 days')
),
RankedEvents AS (
    SELECT
        user_id,
        event_timestamp,
        LAG(event_timestamp, 1, event_timestamp) OVER (PARTITION BY user_id ORDER BY event_timestamp) AS prev_event_timestamp
    FROM UserEvents7d
),
TimeDiffs AS (
    SELECT
        user_id,
        (julianday(event_timestamp) - julianday(prev_event_timestamp)) * 24 * 60 * 60 AS time_diff_seconds
    FROM RankedEvents
    WHERE event_timestamp != prev_event_timestamp -- Exclude the first event for each user
)
SELECT
    u.user_id,
    u.signup_date,
    u.region,
    u.referral_source,
    u.age_group,
    -- Basic counts
    COALESCE(SUM(CASE WHEN ue7d.event_id IS NOT NULL THEN 1 ELSE 0 END), 0) AS num_events_first_7d,
    COALESCE(SUM(CASE WHEN ue7d.event_type = 'view_product' THEN 1 ELSE 0 END), 0) AS num_product_views_first_7d,
    COALESCE(SUM(CASE WHEN ue7d.event_type = 'add_to_cart' THEN 1 ELSE 0 END), 0) AS num_add_to_cart_first_7d,
    COALESCE(SUM(CASE WHEN ue7d.event_type = 'search' THEN 1 ELSE 0 END), 0) AS num_searches_first_7d,
    -- Distinct days with activity
    COALESCE(COUNT(DISTINCT STRFTIME('%Y-%m-%d', ue7d.event_timestamp)), 0) AS days_with_activity_first_7d,
    -- Average time between events
    COALESCE(AVG(td.time_diff_seconds), 0.0) AS avg_time_between_events_first_7d
FROM users u
LEFT JOIN (SELECT *, CAST(event_id AS INTEGER) AS event_id FROM browsing_events) ue7d ON u.user_id = ue7d.user_id
    AND ue7d.event_timestamp >= u.signup_date
    AND ue7d.event_timestamp <= DATE(u.signup_date, '+7 days')
LEFT JOIN TimeDiffs td ON u.user_id = td.user_id
GROUP BY
    u.user_id, u.signup_date, u.region, u.referral_source, u.age_group
ORDER BY u.user_id;
"""

user_early_features_df = pd.read_sql_query(sql_query, conn)
conn.close() # Close connection after fetching

print("\n--- SQL Feature Engineering Results (First 5 Rows) ---")
print(user_early_features_df.head())
print(f"Shape of user_early_features_df: {user_early_features_df.shape}")

# --- 3. Pandas Feature Engineering & Binary Target Creation ---

# Handle NaN values from SQL query (if any remain, e.g., avg_time_between_events_first_7d for users with 0 or 1 event)
numerical_cols = [
    'num_events_first_7d', 'num_product_views_first_7d', 'num_add_to_cart_first_7d',
    'num_searches_first_7d', 'days_with_activity_first_7d', 'avg_time_between_events_first_7d'
]
for col in numerical_cols:
    user_early_features_df[col] = user_early_features_df[col].fillna(0.0 if col == 'avg_time_between_events_first_7d' else 0)

# Convert signup_date to datetime objects
user_early_features_df['signup_date'] = pd.to_datetime(user_early_features_df['signup_date'])

# Calculate days_since_signup_at_cutoff (always 7 for this specific cutoff)
user_early_features_df['days_since_signup_at_cutoff'] = 7

# Calculate engagement_rate_first_7d
user_early_features_df['engagement_rate_first_7d'] = (
    user_early_features_df['num_product_views_first_7d'] + user_early_features_df['num_add_to_cart_first_7d']
) / (user_early_features_df['num_events_first_7d'] + 1) # Add 1 to denominator to avoid division by zero
user_early_features_df['engagement_rate_first_7d'] = user_early_features_df['engagement_rate_first_7d'].fillna(0.0) # Fill any NaN after division


# Create the Binary Target `made_first_purchase_within_30d`
# Ensure purchases_df purchase_date and users_df signup_date are datetime objects
purchases_df['purchase_date'] = pd.to_datetime(purchases_df['purchase_date'])
# users_df['signup_date'] is already datetime from initial generation

# Merge signup_date into purchases_df for easy comparison
purchases_with_signup = pd.merge(purchases_df, users_df[['user_id', 'signup_date']], on='user_id', how='left')

# Filter for purchases within 30 days of signup
purchases_within_30d = purchases_with_signup[
    (purchases_with_signup['purchase_date'] >= purchases_with_signup['signup_date']) &
    (purchases_with_signup['purchase_date'] <= purchases_with_signup['signup_date'] + pd.to_timedelta(30, unit='D'))
]

# Aggregate to get unique users who purchased within 30 days
users_purchased_30d = purchases_within_30d['user_id'].unique()

# Create target column in user_early_features_df
user_early_features_df['made_first_purchase_within_30d'] = user_early_features_df['user_id'].isin(users_purchased_30d).astype(int)

print("\n--- Pandas Feature Engineering & Target Creation Results (First 5 Rows) ---")
print(user_early_features_df.head())

# Check target distribution
purchase_rate = user_early_features_df['made_first_purchase_within_30d'].mean() * 100
print(f"\nOverall first-time purchase rate within 30 days: {purchase_rate:.2f}% (Target: 20-30%)")

# Define features X and target y
numerical_features = [
    'num_events_first_7d', 'num_product_views_first_7d', 'num_add_to_cart_first_7d',
    'num_searches_first_7d', 'days_with_activity_first_7d', 'days_since_signup_at_cutoff',
    'avg_time_between_events_first_7d', 'engagement_rate_first_7d'
]
categorical_features = ['region', 'referral_source', 'age_group']

X = user_early_features_df[numerical_features + categorical_features]
y = user_early_features_df['made_first_purchase_within_30d']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

print(f"\nX_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
print(f"Purchase rate in y_train: {y_train.mean():.2f}")
print(f"Purchase rate in y_test: {y_test.mean():.2f}")


# --- 4. Data Visualization ---
plt.style.use('seaborn-v0_8-darkgrid')

# Plot 1: Violin plot of num_add_to_cart_first_7d vs. purchase status
plt.figure(figsize=(10, 6))
sns.violinplot(x='made_first_purchase_within_30d', y='num_add_to_cart_first_7d', data=user_early_features_df, palette='viridis')
plt.title('Distribution of "Add to Cart" Events in First 7 Days by Purchase Status', fontsize=14)
plt.xlabel('Made First Purchase within 30 Days (0=No, 1=Yes)', fontsize=12)
plt.ylabel('Number of "Add to Cart" Events', fontsize=12)
plt.xticks([0, 1], ['No Purchase', 'Purchased'])
plt.tight_layout()
plt.show()

# Plot 2: Stacked bar chart of purchase proportion across referral_source
referral_purchase_counts = user_early_features_df.groupby(['referral_source', 'made_first_purchase_within_30d']).size().unstack(fill_value=0)
referral_purchase_proportions = referral_purchase_counts.apply(lambda x: x / x.sum(), axis=1)

plt.figure(figsize=(12, 7))
referral_purchase_proportions.plot(kind='bar', stacked=True, color=['lightcoral', 'lightgreen'])
plt.title('Proportion of First Purchase within 30 Days by Referral Source', fontsize=14)
plt.xlabel('Referral Source', fontsize=12)
plt.ylabel('Proportion', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.legend(title='Purchased within 30 Days', labels=['No', 'Yes'], bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()


# --- 5. ML Pipeline & Evaluation (Binary Classification) ---

# Create preprocessing pipeline for numerical and categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')), # Handles potential NaNs, though filled earlier
            ('scaler', StandardScaler())
        ]), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Create the full machine learning pipeline
ml_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', HistGradientBoostingClassifier(random_state=42))
])

# Train the pipeline
print("\n--- Training ML Pipeline ---")
ml_pipeline.fit(X_train, y_train)
print("Pipeline training complete.")

# Predict probabilities and classes on the test set
y_pred_proba = ml_pipeline.predict_proba(X_test)[:, 1]
y_pred = ml_pipeline.predict(X_test)

# Evaluate the model
roc_auc = roc_auc_score(y_test, y_pred_proba)
classification_rep = classification_report(y_test, y_pred)

print(f"\n--- Model Evaluation on Test Set ---")
print(f"ROC AUC Score: {roc_auc:.4f}")
print("\nClassification Report:")
print(classification_rep)