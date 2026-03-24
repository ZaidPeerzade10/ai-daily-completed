import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime, timedelta
import random

# For plotting
import matplotlib.pyplot as plt
import seaborn as sns

# For ML
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer # Corrected import based on feedback
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, classification_report

# --- 1. Generate Synthetic Data ---
np.random.seed(42)
random.seed(42)

# Customers DataFrame
num_customers = np.random.randint(500, 701)
customer_ids = np.arange(1, num_customers + 1)
signup_dates = pd.to_datetime(pd.to_datetime('now') - pd.to_timedelta(np.random.randint(0, 5*365, num_customers), unit='D'))
regions = np.random.choice(['North', 'South', 'East', 'West'], num_customers)
subscription_tiers = np.random.choice(['Basic', 'Premium', 'Pro'], num_customers, p=[0.5, 0.3, 0.2])

customers_df = pd.DataFrame({
    'customer_id': customer_ids,
    'signup_date': signup_dates,
    'region': regions,
    'subscription_tier': subscription_tiers
})

# Interactions DataFrame
all_interactions = []
num_interactions = np.random.randint(10000, 15001)

# Generate interactions, ensuring dates are after signup and applying biases
for _ in range(num_interactions):
    customer = customers_df.sample(1).iloc[0]
    cid = customer['customer_id']
    sdate = customer['signup_date']
    tier = customer['subscription_tier']

    # interaction_date must be after signup_date
    int_date_offset = np.random.randint(1, 3*365 + 1) # up to 3 years after signup
    int_date = sdate + pd.to_timedelta(int_date_offset, unit='D')
    
    int_type = np.random.choice(['Support_Call', 'Chat', 'FAQ_Visit', 'Forum_Post'], p=[0.25, 0.25, 0.3, 0.2])
    duration = np.random.randint(5, 61) # 5-60 minutes

    successful_res = np.nan
    if int_type in ['Support_Call', 'Chat']:
        # Base success rate 70%
        success_rate = 0.7
        if tier == 'Premium':
            success_rate = 0.85 # Premium users have higher success rates
        elif tier == 'Pro':
            success_rate = 0.9
        
        successful_res = 1 if np.random.rand() < success_rate else 0

    all_interactions.append({
        'interaction_id': len(all_interactions) + 1,
        'customer_id': cid,
        'interaction_date': int_date,
        'interaction_type': int_type,
        'duration_minutes': duration,
        'successful_resolution': successful_res
    })

interactions_df = pd.DataFrame(all_interactions)
# Ensure successful_resolution is int where not NaN, then convert back to float for NaN
interactions_df['successful_resolution'] = interactions_df['successful_resolution'].fillna(-1).astype(int)
interactions_df['successful_resolution'] = interactions_df['successful_resolution'].replace(-1, np.nan)

# Sort for easier sequential processing
interactions_df = interactions_df.sort_values(by=['customer_id', 'interaction_date']).reset_index(drop=True)

# Feedback DataFrame
all_feedback = []
num_feedback = np.random.randint(1500, 2501)

# Generate feedback entries
for _ in range(num_feedback):
    customer = customers_df.sample(1).iloc[0]
    cid = customer['customer_id']
    sdate = customer['signup_date']

    # feedback_date must be after signup_date
    fb_date_offset = np.random.randint(1, 3*365 + 1) # up to 3 years after signup
    fb_date = sdate + pd.to_timedelta(fb_date_offset, unit='D')
    
    # Initialize sentiment score to a neutral/positive range
    sentiment_score = np.random.randint(3, 6) # 3, 4, 5 initially

    all_feedback.append({
        'feedback_id': len(all_feedback) + 1,
        'customer_id': cid,
        'feedback_date': fb_date,
        'sentiment_score': sentiment_score
    })

feedback_df = pd.DataFrame(all_feedback)

# Apply sentiment bias: lower scores (1-2) are more likely for customers with multiple
# 'Support_Call' or 'Chat' interactions that have successful_resolution=0 in a recent period.
# Define "recent period" as 30 days prior to feedback_date.
feedback_df_merged_for_bias = feedback_df.merge(customers_df[['customer_id', 'signup_date']], on='customer_id', how='left')

# Iterate through feedback to apply bias
for idx, feedback_row in feedback_df_merged_for_bias.iterrows():
    cid = feedback_row['customer_id']
    fb_date = feedback_row['feedback_date']
    
    recent_interactions = interactions_df[
        (interactions_df['customer_id'] == cid) &
        (interactions_df['interaction_date'] < fb_date) &
        (interactions_df['interaction_date'] >= (fb_date - pd.to_timedelta(30, unit='D'))) &
        (interactions_df['interaction_type'].isin(['Support_Call', 'Chat'])) &
        (interactions_df['successful_resolution'] == 0)
    ]
    
    if len(recent_interactions) >= 2: # Multiple failed resolutions in recent period
        # 70% chance of score 1, 30% chance of score 2
        feedback_df.loc[idx, 'sentiment_score'] = np.random.choice([1, 2], p=[0.7, 0.3])
    else:
        # Small chance of negative sentiment even without recent failures
        if np.random.rand() < 0.05: # 5% chance of negative sentiment
             feedback_df.loc[idx, 'sentiment_score'] = np.random.choice([1, 2])
        # Otherwise, the initial sentiment_score (3-5) is kept

feedback_df = feedback_df.sort_values(by=['customer_id', 'feedback_date']).reset_index(drop=True)

print(f"Generated {len(customers_df)} customers.")
print(f"Generated {len(interactions_df)} interactions.")
print(f"Generated {len(feedback_df)} feedback entries.")

# --- 2. Load into SQLite & SQL Feature Engineering ---
conn = sqlite3.connect(':memory:')

# SQLite doesn't natively support pandas Timestamp, convert to string
customers_df['signup_date'] = customers_df['signup_date'].dt.strftime('%Y-%m-%d %H:%M:%S')
interactions_df['interaction_date'] = interactions_df['interaction_date'].dt.strftime('%Y-%m-%d %H:%M:%S')
feedback_df['feedback_date'] = feedback_df['feedback_date'].dt.strftime('%Y-%m-%d %H:%M:%S')

customers_df.to_sql('customers', conn, index=False)
interactions_df.to_sql('interactions', conn, index=False)
feedback_df.to_sql('feedback', conn, index=False)

sql_query = """
WITH CustomerEarlyInteractions AS (
    SELECT
        i.customer_id,
        MIN(i.interaction_date) AS first_interaction_date_in_window,
        COUNT(i.interaction_id) AS num_interactions_first_30d,
        SUM(i.duration_minutes) AS total_duration_first_30d,
        AVG(i.duration_minutes) AS avg_duration_first_30d,
        SUM(CASE WHEN i.interaction_type IN ('Support_Call', 'Chat') THEN 1 ELSE 0 END) AS num_support_contacts_first_30d,
        SUM(CASE WHEN i.interaction_type IN ('Support_Call', 'Chat') AND i.successful_resolution = 0 THEN 1 ELSE 0 END) AS num_failed_resolutions_first_30d
    FROM
        interactions i
    INNER JOIN
        customers c ON i.customer_id = c.customer_id
    WHERE
        i.interaction_date BETWEEN c.signup_date AND DATE(c.signup_date, '+30 days')
    GROUP BY
        i.customer_id
)
SELECT
    c.customer_id,
    c.signup_date,
    c.region,
    c.subscription_tier,
    COALESCE(cei.num_interactions_first_30d, 0) AS num_interactions_first_30d,
    COALESCE(cei.total_duration_first_30d, 0) AS total_duration_first_30d,
    COALESCE(cei.avg_duration_first_30d, 0.0) AS avg_duration_first_30d,
    COALESCE(cei.num_support_contacts_first_30d, 0) AS num_support_contacts_first_30d,
    COALESCE(cei.num_failed_resolutions_first_30d, 0) AS num_failed_resolutions_first_30d,
    CASE
        WHEN cei.first_interaction_date_in_window IS NOT NULL THEN
            CAST(julianday(cei.first_interaction_date_in_window) - julianday(c.signup_date) AS INTEGER)
        ELSE NULL
    END AS days_since_first_interaction_first_30d
FROM
    customers c
LEFT JOIN
    CustomerEarlyInteractions cei ON c.customer_id = cei.customer_id
;
"""

customer_initial_features_df = pd.read_sql_query(sql_query, conn)
conn.close()

print(f"\nSQL Feature Engineering results (first 5 rows):\n{customer_initial_features_df.head()}")
print(f"Shape of initial features DataFrame: {customer_initial_features_df.shape}")

# --- 3. Pandas Feature Engineering & Binary Target Creation ---
# Handle NaN values
customer_initial_features_df['num_interactions_first_30d'] = customer_initial_features_df['num_interactions_first_30d'].fillna(0).astype(int)
customer_initial_features_df['total_duration_first_30d'] = customer_initial_features_df['total_duration_first_30d'].fillna(0).astype(int)
customer_initial_features_df['avg_duration_first_30d'] = customer_initial_features_df['avg_duration_first_30d'].fillna(0.0)
customer_initial_features_df['num_support_contacts_first_30d'] = customer_initial_features_df['num_support_contacts_first_30d'].fillna(0).astype(int)
customer_initial_features_df['num_failed_resolutions_first_30d'] = customer_initial_features_df['num_failed_resolutions_first_30d'].fillna(0).astype(int)
# Fill days_since_first_interaction_first_30d with 30 (representing no interaction within 30 days)
customer_initial_features_df['days_since_first_interaction_first_30d'] = customer_initial_features_df['days_since_first_interaction_first_30d'].fillna(30).astype(int)


# Convert signup_date to datetime
customer_initial_features_df['signup_date'] = pd.to_datetime(customer_initial_features_df['signup_date'])

# Calculate derived features
# Avoid division by zero: if denominator is 0, set rate to 0.0
customer_initial_features_df['support_contact_rate_first_30d'] = customer_initial_features_df.apply(
    lambda row: row['num_support_contacts_first_30d'] / row['num_interactions_first_30d'] if row['num_interactions_first_30d'] > 0 else 0.0,
    axis=1
)
customer_initial_features_df['failed_resolution_rate_first_30d'] = customer_initial_features_df.apply(
    lambda row: row['num_failed_resolutions_first_30d'] / row['num_support_contacts_first_30d'] if row['num_support_contacts_first_30d'] > 0 else 0.0,
    axis=1
)

# Create the Binary Target `is_negative_future_sentiment`
# Define future sentiment window start date for each customer
customer_initial_features_df['future_sentiment_window_start_date'] = customer_initial_features_df['signup_date'] + pd.to_timedelta(60, unit='D')

# Ensure feedback_df dates are datetime objects for comparison
feedback_df['feedback_date'] = pd.to_datetime(feedback_df['feedback_date'])

# Identify negative feedback in the future window
negative_future_feedback = feedback_df[feedback_df['sentiment_score'].isin([1, 2])].copy()
negative_future_feedback = negative_future_feedback.merge(
    customer_initial_features_df[['customer_id', 'future_sentiment_window_start_date']],
    on='customer_id',
    how='inner'
)
negative_future_feedback = negative_future_feedback[
    negative_future_feedback['feedback_date'] > negative_future_feedback['future_sentiment_window_start_date']
]

# For each customer, check if there's *any* negative future feedback
customer_has_negative_future_sentiment = negative_future_feedback.groupby('customer_id').size().reset_index(name='negative_feedback_count')
customer_has_negative_future_sentiment['is_negative_future_sentiment'] = 1

# Merge target back to main DataFrame
customer_initial_features_df = customer_initial_features_df.merge(
    customer_has_negative_future_sentiment[['customer_id', 'is_negative_future_sentiment']],
    on='customer_id',
    how='left'
)
customer_initial_features_df['is_negative_future_sentiment'] = customer_initial_features_df['is_negative_future_sentiment'].fillna(0).astype(int)

# Drop temporary columns
customer_initial_features_df = customer_initial_features_df.drop(columns=['signup_date', 'future_sentiment_window_start_date'])

print(f"\nDataFrame after Pandas Feature Engineering and Target Creation (first 5 rows):\n{customer_initial_features_df.head()}")
print(f"Shape of final features DataFrame: {customer_initial_features_df.shape}")
print(f"Target distribution:\n{customer_initial_features_df['is_negative_future_sentiment'].value_counts(normalize=True)}")

# Define features (X) and target (y)
numerical_features = [
    'num_interactions_first_30d',
    'total_duration_first_30d',
    'avg_duration_first_30d',
    'num_support_contacts_first_30d',
    'num_failed_resolutions_first_30d',
    'days_since_first_interaction_first_30d',
    'support_contact_rate_first_30d',
    'failed_resolution_rate_first_30d'
]
categorical_features = ['region', 'subscription_tier']

X = customer_initial_features_df[numerical_features + categorical_features]
y = customer_initial_features_df['is_negative_future_sentiment']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

print(f"\nTrain set size: {X_train.shape[0]} samples")
print(f"Test set size: {X_test.shape[0]} samples")
print(f"Train target distribution:\n{y_train.value_counts(normalize=True)}")
print(f"Test target distribution:\n{y_test.value_counts(normalize=True)}")


# --- 4. Data Visualization ---
print("\nGenerating visualizations...")
plt.style.use('seaborn-v0_8-darkgrid')

# Plot 1: Violin plot of avg_duration_first_30d vs. future sentiment
plt.figure(figsize=(8, 6))
sns.violinplot(x='is_negative_future_sentiment', y='avg_duration_first_30d', data=customer_initial_features_df)
plt.title('Distribution of Avg Interaction Duration (First 30 Days) by Future Sentiment')
plt.xlabel('Is Negative Future Sentiment (0: No, 1: Yes)')
plt.ylabel('Average Interaction Duration (Minutes)')
plt.xticks([0, 1], ['No Negative Sentiment', 'Negative Sentiment'])
plt.tight_layout()
plt.savefig('avg_duration_vs_sentiment.png') # Save plot to file
plt.close() # Close the plot to prevent it from displaying

# Plot 2: Stacked bar chart of subscription_tier vs. future sentiment proportion
plt.figure(figsize=(10, 7))
tier_sentiment_proportions = customer_initial_features_df.groupby('subscription_tier')['is_negative_future_sentiment'].value_counts(normalize=True).unstack()
tier_sentiment_proportions.plot(kind='bar', stacked=True, color=['lightgreen', 'salmon'])
plt.title('Proportion of Future Sentiment by Subscription Tier')
plt.xlabel('Subscription Tier')
plt.ylabel('Proportion')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Is Negative Sentiment', labels=['No', 'Yes'], loc='upper left')
plt.tight_layout()
plt.savefig('tier_vs_sentiment_stacked_bar.png') # Save plot to file
plt.close() # Close the plot

print("Visualizations saved as 'avg_duration_vs_sentiment.png' and 'tier_vs_sentiment_stacked_bar.png'.")

# --- 5. ML Pipeline & Evaluation ---
print("\nBuilding and training ML pipeline...")

# Create preprocessing pipelines for numerical and categorical features
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Create a preprocessor using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Create the full pipeline
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', HistGradientBoostingClassifier(random_state=42))
])

# Train the pipeline
model_pipeline.fit(X_train, y_train)

# Predict probabilities on the test set
y_pred_proba = model_pipeline.predict_proba(X_test)[:, 1]

# Predict class labels for classification report
y_pred = (y_pred_proba > 0.5).astype(int) # Using 0.5 as threshold

# Evaluate the model
roc_auc = roc_auc_score(y_test, y_pred_proba)
classification_rep = classification_report(y_test, y_pred, target_names=['No Negative Sentiment', 'Negative Sentiment'])

print(f"\n--- Model Evaluation ---")
print(f"ROC AUC Score: {roc_auc:.4f}")
print("\nClassification Report:\n", classification_rep)

print("\nPipeline execution complete.")