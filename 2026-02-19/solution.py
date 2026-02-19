import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer # CRITICAL FIX: Changed from sklearn.preprocessing
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, classification_report


# --- 1. Generate Synthetic Data (Pandas/Numpy) ---

print("--- 1. Generating Synthetic Data ---")

# Seed for reproducibility
np.random.seed(42)

# 1.1 users_df
num_users = np.random.randint(500, 701)
signup_start_date = datetime.now() - timedelta(days=5 * 365)
users_data = {
    'user_id': np.arange(num_users),
    'signup_date': [signup_start_date + timedelta(days=int(d)) for d in np.random.randint(0, 5 * 365, num_users)],
    'age': np.random.randint(18, 71, num_users),
    'region': np.random.choice(['North', 'South', 'East', 'West', 'Central'], num_users),
    'browsing_frequency_level': np.random.choice(['Low', 'Medium', 'High'], num_users, p=[0.35, 0.4, 0.25])
}
users_df = pd.DataFrame(users_data)
print(f"Generated users_df with {len(users_df)} rows.")

# 1.2 offers_df
num_offers = np.random.randint(50, 101)
offers_data = {
    'offer_id': np.arange(num_offers),
    'offer_type': np.random.choice(['Discount_10', 'Free_Shipping', 'Bundle_Deal', 'Gift_Card', 'BOGO', 'Seasonal_Sale'], num_offers),
    'category_focus': np.random.choice(['Electronics', 'Books', 'Clothing', 'HomeGoods', 'Services', 'Gaming', 'Travel'], num_offers),
    'discount_percentage': np.round(np.random.uniform(5.0, 30.0, num_offers), 1)
}
offers_df = pd.DataFrame(offers_data)
print(f"Generated offers_df with {len(offers_df)} rows.")

# 1.3 campaign_exposures_df
num_exposures = np.random.randint(5000, 8001)
exposures_data = {
    'exposure_id': np.arange(num_exposures),
    'user_id': np.random.choice(users_df['user_id'], num_exposures),
    'offer_id': np.random.choice(offers_df['offer_id'], num_exposures),
}
campaign_exposures_df = pd.DataFrame(exposures_data)

# Merge with user signup dates and offer details to simulate realistic exposure dates and conversion patterns
campaign_exposures_df = campaign_exposures_df.merge(users_df[['user_id', 'signup_date', 'age', 'browsing_frequency_level']], on='user_id', how='left')
campaign_exposures_df = campaign_exposures_df.merge(offers_df[['offer_id', 'offer_type', 'category_focus']], on='offer_id', how='left')

# Generate exposure dates after signup_date
def generate_exposure_date(row):
    # Ensure exposure_date is after signup_date and not in the future
    start_offset = np.random.randint(0, 365 * 2) # up to 2 years after signup
    exposure_date_candidate = row['signup_date'] + timedelta(days=start_offset)
    
    # Cap exposure date at today's date
    if exposure_date_candidate > datetime.now():
        exposure_date_candidate = datetime.now() - timedelta(days=np.random.randint(1, 30)) # ensure it's slightly in the past

    # Ensure it's strictly after signup date, if by chance it's the same day due to random offset
    if exposure_date_candidate <= row['signup_date']:
        exposure_date_candidate = row['signup_date'] + timedelta(days=np.random.randint(1, 30))
    
    return exposure_date_candidate

campaign_exposures_df['exposure_date'] = campaign_exposures_df.apply(generate_exposure_date, axis=1)

# Simulate realistic conversion patterns
campaign_exposures_df['was_converted'] = 0
base_conversion_rate = 0.07 # Overall 7% conversion

for index, row in campaign_exposures_df.iterrows():
    conversion_prob = base_conversion_rate

    # Bias: Users with 'High' browsing_frequency_level have a higher chance
    if row['browsing_frequency_level'] == 'High':
        conversion_prob *= 1.8
    elif row['browsing_frequency_level'] == 'Medium':
        conversion_prob *= 1.2
    else: # Low
        conversion_prob *= 0.7

    # Bias: Some offer_types have higher conversion rates
    if row['offer_type'] in ['Discount_10', 'Free_Shipping']:
        conversion_prob *= 1.5
    elif row['offer_type'] == 'Bundle_Deal':
        conversion_prob *= 1.1
    elif row['offer_type'] in ['Gift_Card', 'BOGO']:
        conversion_prob *= 1.3
    else: # Seasonal_Sale
        conversion_prob *= 0.9

    # Subtle correlation: younger users (< 35) for 'Electronics'/'Gaming', older for 'HomeGoods'/'Services'
    if row['age'] < 35 and row['category_focus'] in ['Electronics', 'Gaming']:
        conversion_prob *= 1.4
    elif row['age'] >= 35 and row['category_focus'] in ['HomeGoods', 'Services']:
        conversion_prob *= 1.3
    elif row['age'] >= 50 and row['category_focus'] == 'Travel':
        conversion_prob *= 1.2
    
    # Ensure probabilities are within [0, 1]
    conversion_prob = np.clip(conversion_prob, 0.01, 0.5)

    if np.random.rand() < conversion_prob:
        campaign_exposures_df.loc[index, 'was_converted'] = 1

print(f"Generated campaign_exposures_df with {len(campaign_exposures_df)} rows. Overall conversion rate: {campaign_exposures_df['was_converted'].mean():.2f}")

# Sort `campaign_exposures_df` by `user_id` then `exposure_date`
campaign_exposures_df = campaign_exposures_df.sort_values(by=['user_id', 'exposure_date']).reset_index(drop=True)

# Drop temporary merge columns
campaign_exposures_df = campaign_exposures_df.drop(columns=['signup_date', 'age', 'browsing_frequency_level', 'offer_type', 'category_focus'])


# --- 2. Load into SQLite & SQL Feature Engineering (Event-Level Context) ---

print("\n--- 2. Loading to SQLite and SQL Feature Engineering ---")

# Connect to in-memory SQLite database
conn = sqlite3.connect(':memory:')

# Convert DataFrames to SQLite tables
users_df.to_sql('users', conn, index=False, if_exists='replace')
offers_df.to_sql('offers', conn, index=False, if_exists='replace')
campaign_exposures_df.to_sql('exposures', conn, index=False, if_exists='replace')

# SQL Query for feature engineering
# Uses julianday for date arithmetic and window functions for prior context
sql_query = """
SELECT
    e.exposure_id,
    e.user_id,
    e.offer_id,
    e.exposure_date,
    e.was_converted,
    u.age,
    u.region,
    u.browsing_frequency_level,
    u.signup_date,
    o.offer_type,
    o.category_focus,
    o.discount_percentage,

    -- User-level prior features
    COALESCE(CAST(SUM(1) OVER (
        PARTITION BY e.user_id
        ORDER BY julianday(e.exposure_date)
        ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
    ) AS INTEGER), 0) AS user_prior_total_exposures,

    COALESCE(CAST(SUM(e.was_converted) OVER (
        PARTITION BY e.user_id
        ORDER BY julianday(e.exposure_date)
        ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
    ) AS INTEGER), 0) AS user_prior_converted_exposures,

    CAST(COALESCE(SUM(e.was_converted) OVER (
        PARTITION BY e.user_id
        ORDER BY julianday(e.exposure_date)
        ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
    ), 0) AS REAL) /
    NULLIF(COALESCE(SUM(1) OVER (
        PARTITION BY e.user_id
        ORDER BY julianday(e.exposure_date)
        ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
    ), 0), 0) AS user_prior_conversion_rate,

    COALESCE(
        julianday(e.exposure_date) - julianday(LAG(e.exposure_date, 1) OVER (
            PARTITION BY e.user_id
            ORDER BY julianday(e.exposure_date)
        )),
        julianday(e.exposure_date) - julianday(u.signup_date)
    ) AS days_since_last_user_exposure,

    -- Offer-level prior features
    COALESCE(CAST(SUM(1) OVER (
        PARTITION BY e.offer_id
        ORDER BY julianday(e.exposure_date)
        ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
    ) AS INTEGER), 0) AS offer_prior_total_exposures,

    COALESCE(CAST(SUM(e.was_converted) OVER (
        PARTITION BY e.offer_id
        ORDER BY julianday(e.exposure_date)
        ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
    ) AS INTEGER), 0) AS offer_prior_converted_exposures,

    CAST(COALESCE(SUM(e.was_converted) OVER (
        PARTITION BY e.offer_id
        ORDER BY julianday(e.exposure_date)
        ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
    ), 0) AS REAL) /
    NULLIF(COALESCE(SUM(1) OVER (
        PARTITION BY e.offer_id
        ORDER BY julianday(e.exposure_date)
        ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
    ), 0), 0) AS offer_prior_conversion_rate

FROM
    exposures e
JOIN
    users u ON e.user_id = u.user_id
JOIN
    offers o ON e.offer_id = o.offer_id
ORDER BY
    e.user_id, julianday(e.exposure_date)
"""

# Fetch results into pandas DataFrame
campaign_features_df = pd.read_sql_query(sql_query, conn)
conn.close() # Close connection after fetching data
print(f"SQL query executed. Resulting DataFrame has {len(campaign_features_df)} rows and {len(campaign_features_df.columns)} columns.")


# --- 3. Pandas Feature Engineering & Binary Target Creation ---

print("\n--- 3. Pandas Feature Engineering & Binary Target Creation ---")

# Handle NaN values from SQL query
# Prior counts should be 0 for first events, prior rates 0.0
# days_since_last_user_exposure should be handled by SQL, but fill if any edge case NaN remains
campaign_features_df['user_prior_total_exposures'] = campaign_features_df['user_prior_total_exposures'].fillna(0).astype(int)
campaign_features_df['user_prior_converted_exposures'] = campaign_features_df['user_prior_converted_exposures'].fillna(0).astype(int)
campaign_features_df['user_prior_conversion_rate'] = campaign_features_df['user_prior_conversion_rate'].fillna(0.0)
campaign_features_df['offer_prior_total_exposures'] = campaign_features_df['offer_prior_total_exposures'].fillna(0).astype(int)
campaign_features_df['offer_prior_converted_exposures'] = campaign_features_df['offer_prior_converted_exposures'].fillna(0).astype(int)
campaign_features_df['offer_prior_conversion_rate'] = campaign_features_df['offer_prior_conversion_rate'].fillna(0.0)

# `days_since_last_user_exposure` should be handled by SQL's COALESCE, but as a safeguard
# if there's any NaN, it means the first exposure for a user, fill with a sentinel value.
campaign_features_df['days_since_last_user_exposure'] = campaign_features_df['days_since_last_user_exposure'].fillna(9999).astype(int)

# Convert date columns to datetime objects
campaign_features_df['signup_date'] = pd.to_datetime(campaign_features_df['signup_date'])
campaign_features_df['exposure_date'] = pd.to_datetime(campaign_features_df['exposure_date'])

# Calculate user_account_age_at_exposure_days
campaign_features_df['user_account_age_at_exposure_days'] = (
    campaign_features_df['exposure_date'] - campaign_features_df['signup_date']
).dt.days
campaign_features_df['user_account_age_at_exposure_days'] = campaign_features_df['user_account_age_at_exposure_days'].fillna(0).astype(int)
# Ensure account age is not negative (shouldn't happen with correct data generation, but safety)
campaign_features_df['user_account_age_at_exposure_days'] = campaign_features_df['user_account_age_at_exposure_days'].clip(lower=0)


# Create user_had_prior_conversion
campaign_features_df['user_had_prior_conversion'] = (campaign_features_df['user_prior_converted_exposures'] > 0).astype(int)

# Define features X and target y
numerical_features = [
    'age', 'discount_percentage', 'user_account_age_at_exposure_days',
    'user_prior_total_exposures', 'user_prior_converted_exposures', 'user_prior_conversion_rate',
    'days_since_last_user_exposure', 'offer_prior_total_exposures', 'offer_prior_converted_exposures',
    'offer_prior_conversion_rate'
]
categorical_features = [
    'region', 'browsing_frequency_level', 'offer_type', 'category_focus', 'user_had_prior_conversion'
]

X = campaign_features_df[numerical_features + categorical_features]
y = campaign_features_df['was_converted']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
print(f"Data split into training (n={len(X_train)}) and testing (n={len(X_test)}) sets.")


# --- 4. Data Visualization ---

print("\n--- 4. Generating Data Visualizations ---")

plt.figure(figsize=(14, 6))

# Plot 1: Violin plot of discount_percentage vs was_converted
plt.subplot(1, 2, 1)
sns.violinplot(x='was_converted', y='discount_percentage', data=campaign_features_df)
plt.title('Discount Percentage Distribution by Conversion Status')
plt.xlabel('Was Converted (0=No, 1=Yes)')
plt.ylabel('Discount Percentage')

# Plot 2: Stacked bar chart of was_converted proportions across offer_type
plt.subplot(1, 2, 2)
# Calculate proportions
offer_conversion_proportions = campaign_features_df.groupby('offer_type')['was_converted'].value_counts(normalize=True).unstack().fillna(0)
offer_conversion_proportions.plot(kind='bar', stacked=True, ax=plt.gca(), cmap='viridis')
plt.title('Conversion Proportion by Offer Type')
plt.xlabel('Offer Type')
plt.ylabel('Proportion')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Was Converted', labels=['No', 'Yes'])
plt.tight_layout()
plt.show()


# --- 5. ML Pipeline & Evaluation (Binary Classification) ---

print("\n--- 5. Building and Evaluating ML Pipeline ---")

# Create preprocessing pipelines for numerical and categorical features
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Create a column transformer to apply different transformations to different columns
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Create the full pipeline
model_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                 ('classifier', HistGradientBoostingClassifier(random_state=42))])

# Train the pipeline
print("Training the HistGradientBoostingClassifier...")
model_pipeline.fit(X_train, y_train)
print("Training complete.")

# Predict probabilities on the test set
y_pred_proba = model_pipeline.predict_proba(X_test)[:, 1]

# Predict class labels on the test set
y_pred = model_pipeline.predict(X_test)

# Calculate and print ROC AUC score
roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"\nROC AUC Score on Test Set: {roc_auc:.4f}")

# Print classification report
print("\nClassification Report on Test Set:")
print(classification_report(y_test, y_pred))

print("\n--- Script Finished ---")