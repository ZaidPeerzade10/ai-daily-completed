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

# Ensure reproducibility
np.random.seed(42)
pd.set_option('display.max_columns', None)

# 1. Generate Synthetic Data
print("1. Generating Synthetic Data...")

# Users DataFrame
num_users = np.random.randint(500, 701)
user_ids = np.arange(1, num_users + 1)
signup_dates = pd.to_datetime('2018-01-01') + pd.to_timedelta(np.random.randint(0, 5 * 365, num_users), unit='D')
regions = np.random.choice(['North', 'South', 'East', 'West'], num_users)
subscription_levels = np.random.choice(['Free', 'Basic', 'Premium'], num_users, p=[0.5, 0.3, 0.2])

users_df = pd.DataFrame({
    'user_id': user_ids,
    'signup_date': signup_dates,
    'region': regions,
    'subscription_level': subscription_levels
})

# Products DataFrame
num_products = np.random.randint(100, 151)
product_ids = np.arange(1, num_products + 1)
categories = np.random.choice(['Electronics', 'Books', 'Apparel', 'HomeGoods', 'Food'], num_products)
prices = np.random.uniform(10.0, 1000.0, num_products)
avg_ratings = np.random.uniform(2.5, 5.0, num_products)
launch_dates = pd.to_datetime('2020-01-01') + pd.to_timedelta(np.random.randint(0, 3 * 365, num_products), unit='D')

products_df = pd.DataFrame({
    'product_id': product_ids,
    'category': categories,
    'price': prices,
    'avg_rating': avg_ratings,
    'launch_date': launch_dates
})

# Interactions DataFrame
num_interactions = np.random.randint(5000, 8001)
interaction_data = []

# To link user and product attributes to interactions during generation
users_info = users_df.set_index('user_id')
products_info = products_df.set_index('product_id')

# Generate raw interactions ensuring date constraints
for i in range(num_interactions):
    user_id = np.random.choice(users_df['user_id'])
    product_id = np.random.choice(products_df['product_id'])

    user_signup_date = users_info.loc[user_id, 'signup_date']
    product_launch_date = products_info.loc[product_id, 'launch_date']

    earliest_interaction_date = max(user_signup_date, product_launch_date) + pd.Timedelta(days=1)
    latest_interaction_date = pd.Timestamp.now() - pd.Timedelta(days=1) # Ensure not in future

    if earliest_interaction_date > latest_interaction_date:
        # If the product/user combination means a very recent launch/signup,
        # interaction date might not be possible in the past. Use earliest possible.
        interaction_date = earliest_interaction_date
    else:
        days_diff = (latest_interaction_date - earliest_interaction_date).days
        interaction_date = earliest_interaction_date + pd.Timedelta(days=np.random.randint(0, max(1, days_diff + 1)))

    interaction_type = np.random.choice(['view', 'add_to_cart', 'purchase', 'review'], p=[0.6, 0.2, 0.15, 0.05])
    
    interaction_data.append({
        'interaction_id': i + 1, # Unique ID
        'user_id': user_id,
        'product_id': product_id,
        'interaction_date': interaction_date,
        'interaction_type': interaction_type,
        'is_positive_interaction': 0 # Placeholder, will be biased later
    })

interactions_df = pd.DataFrame(interaction_data)

# Sort interactions_df by user_id then interaction_date for sequential biasing
interactions_df = interactions_df.sort_values(by=['user_id', 'interaction_date']).reset_index(drop=True)

# Apply realistic patterns for is_positive_interaction sequentially
print("Applying sequential biases to interactions...")
overall_positive_rate_target = (0.10, 0.20) 

# Initialize user_propensity for each user to simulate historical interaction rate
user_propensity = {uid: 0.5 for uid in users_df['user_id']}

current_positive_interactions_count = 0
current_total_interactions_count = 0

for idx, row in interactions_df.iterrows():
    user_id = row['user_id']
    product_id = row['product_id']
    
    user_sub_level = users_info.loc[user_id, 'subscription_level']
    product_avg_rating = products_info.loc[product_id, 'avg_rating']
    product_price = products_info.loc[product_id, 'price']

    # Base probability for a positive interaction
    base_prob = 0.10 # Lower base as biases will increase it

    # Bias 1: Users with 'Premium' subscription_level have a higher chance
    if user_sub_level == 'Premium':
        base_prob += 0.15 # Stronger boost
    elif user_sub_level == 'Basic':
        base_prob += 0.05
    
    # Bias 2: Products with higher avg_rating or lower price tend to have more positive interactions
    # avg_rating bias: scale 2.5-5.0 to 0-0.2
    base_prob += (product_avg_rating - 2.5) / 2.5 * 0.20 
    
    # price bias: scale 10-1000 inversely to 0-0.2
    base_prob += (1000.0 - product_price) / 990.0 * 0.20

    # Bias 3: Correlate with user's simulated past overall positive interaction rate
    propensity_factor = user_propensity[user_id] 
    final_prob = base_prob * propensity_factor * 1.5 # Propensity heavily influences final probability

    # Clamp probability to a reasonable range
    final_prob = np.clip(final_prob, 0.01, 0.95) 

    # Decide if interaction is positive
    is_positive = 1 if np.random.rand() < final_prob else 0
    
    # Update user_propensity for future interactions of this user (learning effect)
    if is_positive == 1:
        user_propensity[user_id] = min(0.9, user_propensity[user_id] + 0.08) # Increase propensity
    else:
        user_propensity[user_id] = max(0.1, user_propensity[user_id] - 0.03) # Decrease propensity
    
    interactions_df.at[idx, 'is_positive_interaction'] = is_positive
    
    current_positive_interactions_count += is_positive
    current_total_interactions_count += 1

overall_positive_rate = current_positive_interactions_count / current_total_interactions_count
print(f"Overall positive interaction rate: {overall_positive_rate:.2f} (Target: {overall_positive_rate_target[0]:.2f}-{overall_positive_rate_target[1]:.2f})")
print("Synthetic data generation complete.")


# 2. Load into SQLite & SQL Feature Engineering
print("\n2. Loading data into SQLite and performing SQL Feature Engineering...")

conn = sqlite3.connect(':memory:')

users_df.to_sql('users', conn, index=False, if_exists='replace')
products_df.to_sql('products', conn, index=False, if_exists='replace')
interactions_df.to_sql('interactions', conn, index=False, if_exists='replace')

# SQL query for feature engineering
# Dates are stored as strings, need to use JULIANDAY for calculations
sql_query = """
SELECT
    i.interaction_id,
    i.user_id,
    i.product_id,
    i.interaction_date,
    i.is_positive_interaction,
    u.region,
    u.subscription_level,
    u.signup_date,
    p.category,
    p.price,
    p.avg_rating,
    p.launch_date,

    -- User sequential features
    COALESCE(SUM(1) OVER (
        PARTITION BY i.user_id
        ORDER BY i.interaction_date
        ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
    ), 0) AS user_prior_total_interactions,

    COALESCE(SUM(CASE WHEN i.is_positive_interaction = 1 THEN 1 ELSE 0 END) OVER (
        PARTITION BY i.user_id
        ORDER BY i.interaction_date
        ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
    ), 0) AS user_prior_positive_interactions,

    COALESCE(
        CAST(SUM(CASE WHEN i.is_positive_interaction = 1 THEN 1 ELSE 0 END) OVER (
            PARTITION BY i.user_id
            ORDER BY i.interaction_date
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
        ) AS REAL) /
        NULLIF(SUM(1) OVER (
            PARTITION BY i.user_id
            ORDER BY i.interaction_date
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
        ), 0),
    0.0) AS user_prior_positive_interaction_rate,

    -- days_since_last_user_interaction
    -- For the first interaction, use (interaction_date - signup_date)
    CAST(JULIANDAY(i.interaction_date) - COALESCE(
        LAG(JULIANDAY(i.interaction_date), 1) OVER (PARTITION BY i.user_id ORDER BY i.interaction_date),
        JULIANDAY(u.signup_date)
    ) AS INTEGER) AS days_since_last_user_interaction,

    -- Product sequential features
    COALESCE(SUM(1) OVER (
        PARTITION BY i.product_id
        ORDER BY i.interaction_date
        ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
    ), 0) AS product_prior_total_interactions,

    COALESCE(SUM(CASE WHEN i.is_positive_interaction = 1 THEN 1 ELSE 0 END) OVER (
        PARTITION BY i.product_id
        ORDER BY i.interaction_date
        ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
    ), 0) AS product_prior_positive_interactions,

    COALESCE(
        CAST(SUM(CASE WHEN i.is_positive_interaction = 1 THEN 1 ELSE 0 END) OVER (
            PARTITION BY i.product_id
            ORDER BY i.interaction_date
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
        ) AS REAL) /
        NULLIF(SUM(1) OVER (
            PARTITION BY i.product_id
            ORDER BY i.interaction_date
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
        ), 0),
    0.0) AS product_prior_positive_interaction_rate

FROM
    interactions AS i
JOIN
    users AS u ON i.user_id = u.user_id
JOIN
    products AS p ON i.product_id = p.product_id
ORDER BY
    i.user_id, i.interaction_date
"""

user_product_features_df = pd.read_sql_query(sql_query, conn)
conn.close()
print("SQL feature engineering complete. Head of resulting DataFrame:")
print(user_product_features_df.head())


# 3. Pandas Feature Engineering & Binary Target Creation
print("\n3. Performing Pandas Feature Engineering...")

# Convert date columns to datetime objects
date_cols = ['signup_date', 'launch_date', 'interaction_date']
for col in date_cols:
    user_product_features_df[col] = pd.to_datetime(user_product_features_df[col])

# Calculate age features
user_product_features_df['user_account_age_at_interaction_days'] = (
    user_product_features_df['interaction_date'] - user_product_features_df['signup_date']
).dt.days

user_product_features_df['product_age_at_interaction_days'] = (
    user_product_features_df['interaction_date'] - user_product_features_df['launch_date']
).dt.days

# Create user_had_prior_positive_interaction
user_product_features_df['user_had_prior_positive_interaction'] = (
    user_product_features_df['user_prior_positive_interactions'] > 0
).astype(int)

# Define features (X) and target (y)
numerical_features = [
    'price', 'avg_rating', 'user_account_age_at_interaction_days',
    'product_age_at_interaction_days', 'user_prior_total_interactions',
    'user_prior_positive_interactions', 'user_prior_positive_interaction_rate',
    'days_since_last_user_interaction', 'product_prior_total_interactions',
    'product_prior_positive_interactions', 'product_prior_positive_interaction_rate'
]
categorical_features = [
    'region', 'subscription_level', 'category', 'user_had_prior_positive_interaction'
]

# Ensure all selected features are present in the DataFrame
all_features = numerical_features + categorical_features
X = user_product_features_df[all_features]
y = user_product_features_df['is_positive_interaction']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
print(f"Dataset split into training ({len(X_train)} samples) and testing ({len(X_test)} samples).")
print(f"Positive interaction rate in training: {y_train.mean():.2f}")
print(f"Positive interaction rate in testing: {y_test.mean():.2f}")


# 4. Data Visualization
print("\n4. Generating Data Visualizations...")

plt.figure(figsize=(14, 6))

# Violin plot for avg_rating vs. is_positive_interaction
plt.subplot(1, 2, 1)
sns.violinplot(x='is_positive_interaction', y='avg_rating', data=user_product_features_df, palette='muted')
plt.title('Distribution of Average Rating by Positive Interaction Outcome')
plt.xlabel('Is Positive Interaction (0=No, 1=Yes)')
plt.ylabel('Average Rating')

# Stacked bar chart for category vs. is_positive_interaction
plt.subplot(1, 2, 2)
category_positive_counts = user_product_features_df.groupby('category')['is_positive_interaction'].value_counts(normalize=True).unstack().fillna(0)
category_positive_counts.plot(kind='bar', stacked=True, ax=plt.gca(), cmap='viridis')
plt.title('Proportion of Positive Interactions by Product Category')
plt.xlabel('Product Category')
plt.ylabel('Proportion')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Is Positive Interaction', labels=['Negative (0)', 'Positive (1)'])
plt.tight_layout()
plt.show()
print("Visualizations displayed.")


# 5. ML Pipeline & Evaluation (Binary Classification)
print("\n5. Building and Evaluating ML Pipeline...")

# Create preprocessing pipelines for numerical and categorical features
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')), # Handles any remaining NaNs
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
    ],
    remainder='drop' # Drop any columns not specified
)

# Create the full pipeline with preprocessor and classifier
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', HistGradientBoostingClassifier(random_state=42))
])

# Train the pipeline
print("Training the HistGradientBoostingClassifier pipeline...")
pipeline.fit(X_train, y_train)
print("Training complete.")

# Predict probabilities for the positive class on the test set
y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

# For classification report, we need binary predictions. Use a default threshold of 0.5
y_pred = (y_pred_proba > 0.5).astype(int)

# Evaluate the model
print("\nModel Evaluation on Test Set:")
roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"ROC AUC Score: {roc_auc:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nPipeline execution complete.")