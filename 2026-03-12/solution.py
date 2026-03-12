import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer # Corrected import from sklearn.impute
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, classification_report

# Use 'Agg' backend for non-interactive plots, suitable for script execution
plt.switch_backend('Agg') 


# --- 1. Generate Synthetic Data ---

# Seed for reproducibility
np.random.seed(42)
random.seed(42)

# 1.1 users_df
num_users = random.randint(500, 700)
signup_dates = [datetime.now() - timedelta(days=np.random.randint(0, 5*365)) for _ in range(num_users)]
users_df = pd.DataFrame({
    'user_id': range(1, num_users + 1),
    'signup_date': signup_dates,
    'age': np.random.randint(18, 71, num_users),
    'region': np.random.choice(['North', 'South', 'East', 'West', 'Central'], num_users),
    'email_engagement_score': np.random.rand(num_users),
    'has_premium_plan': np.random.choice([0, 1], num_users, p=[0.7, 0.3])
})

# 1.2 offers_df
num_offers = random.randint(50, 100)
offer_start_dates = [datetime.now() - timedelta(days=np.random.randint(0, 3*365)) for _ in range(num_offers)]
offers_df = pd.DataFrame({
    'offer_id': range(101, 101 + num_offers),
    'offer_type': np.random.choice(['Discount', 'Free_Shipping', 'Bonus_Points', 'BOGO'], num_offers, p=[0.4, 0.3, 0.2, 0.1]),
    'discount_percent': np.random.uniform(5.0, 50.0, num_offers),
    'min_purchase_value': np.random.uniform(0.0, 200.0, num_offers),
    'offer_start_date': offer_start_dates
})
offers_df['offer_end_date'] = offers_df['offer_start_date'] + pd.to_timedelta(np.random.randint(1, 31, num_offers), unit='D')

# 1.3 offer_interactions_df (initial generation without `was_redeemed` for sequential assignment)
num_interactions = random.randint(8000, 12000)
interaction_data = []

user_ids = users_df['user_id'].values
offer_ids = offers_df['offer_id'].values

# Create a mapping for quick lookup
user_map = users_df.set_index('user_id')
offer_map = offers_df.set_index('offer_id')

# Generate raw interaction events
for _ in range(num_interactions * 2): # Generate more to account for invalid dates and duplicates
    user_id = np.random.choice(user_ids)
    offer_id = np.random.choice(offer_ids)

    user_signup_date = user_map.loc[user_id, 'signup_date']
    offer_start = offer_map.loc[offer_id, 'offer_start_date']
    offer_end = offer_map.loc[offer_id, 'offer_end_date']

    # Ensure interaction_date is valid (after user signup, within offer active period)
    valid_start_date = max(user_signup_date, offer_start)
    
    if valid_start_date >= offer_end:
        continue # Offer ended before user signed up or before it started, skip

    days_delta = (offer_end - valid_start_date).days
    if days_delta < 0: # Should not happen with above check, but good for robustness
        continue
    elif days_delta == 0: # Interaction must be on the start/end date
        interaction_date = valid_start_date
    else: # Pick a date within the valid range
        interaction_date = valid_start_date + timedelta(days=np.random.randint(0, days_delta + 1))
    
    interaction_data.append({
        'user_id': user_id,
        'offer_id': offer_id,
        'interaction_date': interaction_date
    })

# Convert to DataFrame and drop duplicates
temp_interactions_df = pd.DataFrame(interaction_data).drop_duplicates(subset=['user_id', 'offer_id', 'interaction_date'])
temp_interactions_df = temp_interactions_df.head(num_interactions) # Cap to desired number of interactions


# Merge with user and offer details for `was_redeemed` generation logic
temp_interactions_df = temp_interactions_df.merge(users_df, on='user_id', how='left')
temp_interactions_df = temp_interactions_df.merge(offers_df, on='offer_id', how='left')

# Sort for sequential `was_redeemed` generation (crucial for "users who previously redeemed" bias)
temp_interactions_df.sort_values(by=['user_id', 'interaction_date'], inplace=True)
temp_interactions_df.reset_index(drop=True, inplace=True)

# Simulate realistic redemption patterns for `was_redeemed`
# Initialize a dictionary to store prior redemptions count for each user
user_redemption_counts = {}

overall_base_redemption_rate = 0.08 # Target overall redemption rate 5-15%

def calculate_redemption_probability(row, prior_redeemed_count):
    p = overall_base_redemption_rate

    # User-centric biases
    p += row['email_engagement_score'] * 0.1 # Higher engagement -> higher chance
    p += row['has_premium_plan'] * 0.05 # Premium users more likely
    
    # Offer-centric biases
    p += row['discount_percent'] * 0.002 # Higher discount -> higher chance
    if row['offer_type'] == 'Discount':
        p += 0.05
    elif row['offer_type'] == 'Bonus_Points':
        p += 0.03
    
    # Time-centric biases within the offer campaign
    days_into_offer = (row['interaction_date'] - row['offer_start_date']).days
    offer_duration = (row['offer_end_date'] - row['offer_start_date']).days
    
    # Early in offer campaign
    if days_into_offer < 5:
        p += 0.02
    # Late in offer campaign (urgency)
    if offer_duration > 0 and (offer_duration - days_into_offer) < 3:
        p += 0.03

    # Prior redemption history bias (CRITICAL for realism)
    # If a user has redeemed before, increase their likelihood to redeem again
    if prior_redeemed_count > 0:
        p += prior_redeemed_count * 0.02 # Each prior redemption slightly boosts future likelihood

    # Clip probability to a valid range
    return np.clip(p, 0.01, 0.9) # Ensure probability is not too low or too high

was_redeemed_list = []
for index, row in temp_interactions_df.iterrows():
    user_id = row['user_id']
    
    # Get user's prior redemption count (initialized to 0 if no prior redemptions)
    prior_count = user_redemption_counts.get(user_id, 0)
    
    prob = calculate_redemption_probability(row, prior_count)
    
    redeemed = int(np.random.rand() < prob)
    was_redeemed_list.append(redeemed)
    
    # Update prior redemptions count if the current offer was redeemed
    if redeemed == 1:
        user_redemption_counts[user_id] = prior_count + 1

temp_interactions_df['was_redeemed'] = was_redeemed_list

# Final `offer_interactions_df` with required columns
offer_interactions_df = temp_interactions_df[[
    'user_id', 'offer_id', 'interaction_date', 'was_redeemed'
]].copy()
offer_interactions_df['interaction_id'] = range(1, len(offer_interactions_df) + 1)
offer_interactions_df = offer_interactions_df[[
    'interaction_id', 'user_id', 'offer_id', 'interaction_date', 'was_redeemed'
]]

# Sort as requested for sequential processing consistency
offer_interactions_df.sort_values(by=['user_id', 'interaction_date'], inplace=True)
offer_interactions_df.reset_index(drop=True, inplace=True)


print("--- Data Generation Summary ---")
print(f"Users: {len(users_df)}")
print(f"Offers: {len(offers_df)}")
print(f"Interactions: {len(offer_interactions_df)}")
print(f"Overall Redemption Rate: {offer_interactions_df['was_redeemed'].mean():.2%}\n")


# --- 2. Load into SQLite & SQL Feature Engineering ---

conn = sqlite3.connect(':memory:')

# Convert date columns to string for SQLite to ensure proper storage and comparison
users_df['signup_date'] = users_df['signup_date'].dt.strftime('%Y-%m-%d')
offers_df['offer_start_date'] = offers_df['offer_start_date'].dt.strftime('%Y-%m-%d')
offers_df['offer_end_date'] = offers_df['offer_end_date'].dt.strftime('%Y-%m-%d')
offer_interactions_df['interaction_date'] = offer_interactions_df['interaction_date'].dt.strftime('%Y-%m-%d')

users_df.to_sql('users', conn, index=False, if_exists='replace')
offers_df.to_sql('offers', conn, index=False, if_exists='replace')
offer_interactions_df.to_sql('offer_interactions', conn, index=False, if_exists='replace')

sql_query = """
WITH RankedInteractions AS (
    SELECT
        oi.interaction_id,
        oi.user_id,
        oi.offer_id,
        oi.interaction_date,
        oi.was_redeemed,
        u.signup_date,
        u.age,
        u.region,
        u.email_engagement_score,
        u.has_premium_plan,
        o.offer_type,
        o.discount_percent,
        o.min_purchase_value,
        o.offer_start_date,
        o.offer_end_date
    FROM
        offer_interactions oi
    JOIN
        users u ON oi.user_id = u.user_id
    JOIN
        offers o ON oi.offer_id = o.offer_id
    ORDER BY
        oi.user_id, oi.interaction_date -- Pre-sort for window functions consistency
)
SELECT
    ri.interaction_id,
    ri.user_id,
    ri.offer_id,
    ri.interaction_date,
    ri.was_redeemed, -- This is our target variable
    ri.age,
    ri.region,
    ri.email_engagement_score,
    ri.has_premium_plan,
    ri.offer_type,
    ri.discount_percent,
    ri.min_purchase_value,
    ri.signup_date,
    ri.offer_start_date,
    ri.offer_end_date,

    -- User's prior offer interactions (any offer_id)
    COALESCE(SUM(1) OVER (
        PARTITION BY ri.user_id
        ORDER BY ri.interaction_date
        ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
    ), 0) AS user_prior_offers_received,

    -- User's prior offers redeemed
    COALESCE(SUM(CASE WHEN ri.was_redeemed = 1 THEN 1 ELSE 0 END) OVER (
        PARTITION BY ri.user_id
        ORDER BY ri.interaction_date
        ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
    ), 0) AS user_prior_offers_redeemed,
    
    -- User's prior redemption rate
    -- Handle division by zero using NULLIF and COALESCE
    COALESCE(
        CAST(SUM(CASE WHEN ri.was_redeemed = 1 THEN 1 ELSE 0 END) OVER (
            PARTITION BY ri.user_id
            ORDER BY ri.interaction_date
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
        ) AS REAL) /
        NULLIF(CAST(SUM(1) OVER (
            PARTITION BY ri.user_id
            ORDER BY ri.interaction_date
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
        ) AS REAL), 0),
        0.0
    ) AS user_prior_redemption_rate,
    
    -- Days since last user redemption
    -- If no prior redemption, use days from signup_date to interaction_date
    COALESCE(
        JULIANDAY(ri.interaction_date) - JULIANDAY(MAX(CASE WHEN ri.was_redeemed = 1 THEN ri.interaction_date END) OVER (
            PARTITION BY ri.user_id
            ORDER BY ri.interaction_date
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
        )),
        JULIANDAY(ri.interaction_date) - JULIANDAY(ri.signup_date)
    ) AS days_since_last_user_redemption,
    
    -- Offer's prior interactions (for the specific offer_id, across all users)
    COALESCE(SUM(1) OVER (
        PARTITION BY ri.offer_id
        ORDER BY ri.interaction_date
        ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
    ), 0) AS offer_prior_interactions_all_users,
    
    -- Offer's prior redemptions (for the specific offer_id, across all users)
    COALESCE(SUM(CASE WHEN ri.was_redeemed = 1 THEN 1 ELSE 0 END) OVER (
        PARTITION BY ri.offer_id
        ORDER BY ri.interaction_date
        ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
    ), 0) AS offer_prior_redemptions_all_users,

    -- Offer's prior redemption rate (for the specific offer_id, across all users)
    COALESCE(
        CAST(SUM(CASE WHEN ri.was_redeemed = 1 THEN 1 ELSE 0 END) OVER (
            PARTITION BY ri.offer_id
            ORDER BY ri.interaction_date
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
        ) AS REAL) /
        NULLIF(CAST(SUM(1) OVER (
            PARTITION BY ri.offer_id
            ORDER BY ri.interaction_date
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
        ) AS REAL), 0),
        0.0
    ) AS offer_prior_redemption_rate_all_users

FROM
    RankedInteractions ri
ORDER BY
    ri.user_id, ri.interaction_date;
"""

offer_prediction_df = pd.read_sql_query(sql_query, conn)
conn.close()

print("--- SQL Feature Engineering Complete ---")
print(f"DataFrame shape after SQL: {offer_prediction_df.shape}\n")


# --- 3. Pandas Feature Engineering & Binary Target Creation ---

# Handle NaN values for prior aggregates/rates (should mostly be handled by COALESCE in SQL but good to re-check)
# Fill prior counts with 0
for col in ['user_prior_offers_received', 'user_prior_offers_redeemed',
            'offer_prior_interactions_all_users', 'offer_prior_redemptions_all_users']:
    offer_prediction_df[col] = offer_prediction_df[col].fillna(0).astype(int)

# Fill prior redemption rates with 0.0
for col in ['user_prior_redemption_rate', 'offer_prior_redemption_rate_all_users']:
    offer_prediction_df[col] = offer_prediction_df[col].fillna(0.0)

# Convert date columns to datetime objects
date_cols = ['signup_date', 'offer_start_date', 'offer_end_date', 'interaction_date']
for col in date_cols:
    offer_prediction_df[col] = pd.to_datetime(offer_prediction_df[col])

# Calculate `days_since_signup_at_interaction`
offer_prediction_df['days_since_signup_at_interaction'] = \
    (offer_prediction_df['interaction_date'] - offer_prediction_df['signup_date']).dt.days

# Fill `days_since_last_user_redemption` if it's NaN (meaning no prior redemption or first interaction)
# SQL query should handle this by falling back to `days_since_signup_at_interaction`, but Pandas fillna is a safety net
offer_prediction_df['days_since_last_user_redemption'] = offer_prediction_df['days_since_last_user_redemption'].fillna(
    offer_prediction_df['days_since_signup_at_interaction']
)

# Calculate `days_into_offer_campaign`
offer_prediction_df['days_into_offer_campaign'] = \
    (offer_prediction_df['interaction_date'] - offer_prediction_df['offer_start_date']).dt.days

# Calculate `offer_total_duration_days`
offer_prediction_df['offer_total_duration_days'] = \
    (offer_prediction_df['offer_end_date'] - offer_prediction_df['offer_start_date']).dt.days

# Define numerical and categorical features
numerical_features = [
    'age', 'email_engagement_score', 'discount_percent', 'min_purchase_value',
    'user_prior_offers_received', 'user_prior_offers_redeemed', 'user_prior_redemption_rate',
    'days_since_last_user_redemption', 'offer_prior_interactions_all_users',
    'offer_prior_redemptions_all_users', 'offer_prior_redemption_rate_all_users',
    'days_since_signup_at_interaction', 'days_into_offer_campaign', 'offer_total_duration_days'
]
categorical_features = ['region', 'has_premium_plan', 'offer_type'] 

# Create feature matrix X and target vector y
X = offer_prediction_df[numerical_features + categorical_features].copy()
y = offer_prediction_df['was_redeemed'].copy()

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y # Stratify to maintain class balance
)

print("--- Pandas Feature Engineering Complete ---")
print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}\n")


# --- 4. Data Visualization ---

print("--- Generating Visualizations ---")

# Plot 1: Violin plot for user_prior_redemption_rate vs was_redeemed
plt.figure(figsize=(10, 6))
sns.violinplot(x='was_redeemed', y='user_prior_redemption_rate', data=offer_prediction_df)
plt.title('Distribution of User Prior Redemption Rate by Redemption Status')
plt.xlabel('Was Redeemed (0=No, 1=Yes)')
plt.ylabel('User Prior Redemption Rate')
plt.xticks([0, 1], ['Not Redeemed', 'Redeemed'])
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('user_prior_redemption_rate_violin.png')


# Plot 2: Stacked bar chart for proportion of was_redeemed across offer_type
offer_type_redeemed_prop = offer_prediction_df.groupby('offer_type')['was_redeemed'].value_counts(normalize=True).unstack().fillna(0)

plt.figure(figsize=(10, 6))
offer_type_redeemed_prop.plot(kind='bar', stacked=True, color=['lightcoral', 'lightseagreen'])
plt.title('Proportion of Redemption Status by Offer Type')
plt.xlabel('Offer Type')
plt.ylabel('Proportion')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Was Redeemed', labels=['No', 'Yes'], bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('offer_type_redemption_stacked_bar.png')
print("Plots saved as user_prior_redemption_rate_violin.png and offer_type_redemption_stacked_bar.png\n")


# --- 5. ML Pipeline & Evaluation (Binary Classification) ---

print("--- Building and Training ML Pipeline ---")

# Preprocessing for numerical features
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# Preprocessing for categorical features
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Create a preprocessor using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='passthrough' 
)

# Create the full pipeline with HistGradientBoostingClassifier
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', HistGradientBoostingClassifier(random_state=42))
])

# Train the pipeline
pipeline.fit(X_train, y_train)
print("ML Pipeline trained successfully.\n")

# Predict probabilities for the positive class (class 1) on the test set
y_pred_proba = pipeline.predict_proba(X_test)[:, 1] 
# Predict class labels
y_pred = pipeline.predict(X_test)

# Evaluate the model
roc_auc = roc_auc_score(y_test, y_pred_proba)
class_report = classification_report(y_test, y_pred)

print("--- Model Evaluation ---")
print(f"ROC AUC Score on Test Set: {roc_auc:.4f}")
print("\nClassification Report on Test Set:\n", class_report)

print("\n--- Script Finished Successfully ---")