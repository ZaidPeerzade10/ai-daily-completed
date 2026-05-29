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

# Suppress FutureWarnings related to inplace=True (e.g., in older pandas versions, though current code avoids it)
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning, module='seaborn') # Suppress specific seaborn warnings if any

# 1. Generate Synthetic Data

# Customers DataFrame
num_customers = np.random.randint(1000, 1500)
signup_start_date = pd.Timestamp.now() - pd.DateOffset(years=5)
signup_end_date = pd.Timestamp.now() - pd.DateOffset(years=2) # Ensure customers have some history
customers_df = pd.DataFrame({
    'customer_id': np.arange(num_customers),
    'signup_date': pd.to_datetime(np.random.uniform(signup_start_date.timestamp(), signup_end_date.timestamp(), num_customers), unit='s').date,
    'region': np.random.choice(['North', 'South', 'East', 'West', 'Central'], num_customers),
    'customer_segment': np.random.choice(['New', 'Regular', 'VIP', 'Churn Risk'], num_customers, p=[0.2, 0.5, 0.15, 0.15])
})
customers_df['signup_date'] = pd.to_datetime(customers_df['signup_date']) # Convert back to datetime objects for calculations

# Agents DataFrame
num_agents = np.random.randint(100, 200)
agents_df = pd.DataFrame({
    'agent_id': np.arange(num_agents),
    'department': np.random.choice(['Billing', 'Tech Support', 'Sales', 'Customer Service', 'Onboarding'], num_agents),
    'agent_seniority': np.random.choice(['Junior', 'Mid', 'Senior'], num_agents, p=[0.25, 0.5, 0.25]),
    'average_handle_time_minutes': np.random.uniform(10.0, 60.0, num_agents),
    'past_satisfaction_rating': np.random.uniform(2.5, 5.0, num_agents) # Agents typically have decent ratings
})

# Interactions DataFrame
num_interactions = np.random.randint(20000, 30000)
customer_ids = np.random.choice(customers_df['customer_id'], num_interactions, replace=True)
agent_ids = np.random.choice(agents_df['agent_id'], num_interactions, replace=True)

interactions_df = pd.DataFrame({
    'interaction_id': np.arange(num_interactions),
    'customer_id': customer_ids,
    'agent_id': agent_ids,
    'channel': np.random.choice(['Phone', 'Chat', 'Email', 'Social Media'], num_interactions, p=[0.4, 0.3, 0.2, 0.1]),
    'issue_type': np.random.choice(['Billing Inquiry', 'Technical Issue', 'Account Update', 'Product Info', 'Complaint', 'Feature Request'], num_interactions, p=[0.25, 0.30, 0.2, 0.15, 0.05, 0.05]),
    'interaction_duration_minutes': np.random.uniform(5.0, 90.0, num_interactions)
})

# Merge to get signup_date for interaction_date generation and other attributes for correlation
interactions_df = interactions_df.merge(customers_df[['customer_id', 'signup_date', 'customer_segment']], on='customer_id', how='left')
interactions_df = interactions_df.merge(agents_df[['agent_id', 'agent_seniority', 'past_satisfaction_rating']], on='agent_id', how='left')

# Generate interaction_date always after signup_date
# Generate random offsets from signup date (minimum 1 day to maximum 3 years)
random_days_offset = np.random.randint(1, 365 * 3, num_interactions)
interactions_df['interaction_date'] = interactions_df['signup_date'] + pd.to_timedelta(random_days_offset, unit='D')

# Ensure interaction_date is not in the future (relative to now)
current_time = pd.Timestamp.now()
interactions_df['interaction_date'] = interactions_df['interaction_date'].apply(
    lambda x: x if x < current_time else current_time - pd.Timedelta(days=np.random.randint(1, 30))
)

# Simulate realistic patterns for post_interaction_satisfaction_score
satisfaction_scores = []
for index, row in interactions_df.iterrows():
    # Base imbalance: more 4s and 5s
    score = np.random.choice([1, 2, 3, 4, 5], p=[0.05, 0.1, 0.2, 0.35, 0.3])

    # Agent seniority correlation
    if row['agent_seniority'] == 'Senior':
        score = min(5, score + np.random.choice([0, 1], p=[0.3, 0.7]))
    elif row['agent_seniority'] == 'Junior':
        score = max(1, score - np.random.choice([0, 1], p=[0.7, 0.3]))

    # Agent past satisfaction rating correlation
    if row['past_satisfaction_rating'] > 4.5:
        score = min(5, score + np.random.choice([0, 1], p=[0.4, 0.6]))
    elif row['past_satisfaction_rating'] < 3.0:
        score = max(1, score - np.random.choice([0, 1], p=[0.4, 0.6]))

    # VIP customer segment correlation
    if row['customer_segment'] == 'VIP':
        score = min(5, score + np.random.choice([0, 1], p=[0.5, 0.5]))
    elif row['customer_segment'] == 'Churn Risk':
        score = max(1, score - np.random.choice([0, 1], p=[0.5, 0.5]))

    # Issue type correlation
    if row['issue_type'] == 'Technical Issue':
        score = max(1, score - np.random.choice([0, 1], p=[0.6, 0.4]))
    elif row['issue_type'] == 'Complaint':
        score = max(1, score - np.random.choice([1, 2], p=[0.5, 0.5]))

    satisfaction_scores.append(score)

interactions_df['post_interaction_satisfaction_score'] = satisfaction_scores

# Clean up temp columns used for generation
interactions_df = interactions_df.drop(columns=['signup_date', 'customer_segment', 'agent_seniority', 'past_satisfaction_rating'])

# Sort interactions by date
interactions_df = interactions_df.sort_values(by='interaction_date').reset_index(drop=True)

print("--- Synthetic Data Generation Complete ---")
print("Customers DataFrame head:\n", customers_df.head())
print("\nAgents DataFrame head:\n", agents_df.head())
print("\nInteractions DataFrame head:\n", interactions_df.head())
print("\nSatisfaction Score Distribution:\n", interactions_df['post_interaction_satisfaction_score'].value_counts(normalize=True).sort_index())

# 2. Load into SQLite & SQL Feature Engineering (Time-Windowed Aggregations)

conn = sqlite3.connect(':memory:')

# Convert datetime objects to string format compatible with SQLite
customers_df['signup_date'] = customers_df['signup_date'].dt.strftime('%Y-%m-%d %H:%M:%S')
interactions_df['interaction_date'] = interactions_df['interaction_date'].dt.strftime('%Y-%m-%d %H:%M:%S')

customers_df.to_sql('customers', conn, index=False, if_exists='replace')
agents_df.to_sql('agents', conn, index=False, if_exists='replace')
interactions_df.to_sql('interactions', conn, index=False, if_exists='replace')

# Define GLOBAL_PREDICTION_CUTOFF_DATE
latest_interaction_date_str = interactions_df['interaction_date'].max()
latest_interaction_datetime = pd.to_datetime(latest_interaction_date_str)
GLOBAL_PREDICTION_CUTOFF_DATE = (latest_interaction_datetime - pd.Timedelta(weeks=2)).strftime('%Y-%m-%d %H:%M:%S')

print(f"\nGlobal Prediction Cutoff Date: {GLOBAL_PREDICTION_CUTOFF_DATE}")

sql_query = f"""
WITH CustomerHistoricalData AS (
    SELECT
        c.customer_id,
        -- Average satisfaction for this customer in the 90 days prior to or on cutoff
        COALESCE(AVG(CASE WHEN julianday(i_hist.interaction_date) <= julianday('{GLOBAL_PREDICTION_CUTOFF_DATE}')
                      AND julianday(i_hist.interaction_date) > julianday('{GLOBAL_PREDICTION_CUTOFF_DATE}') - 90
                 THEN i_hist.post_interaction_satisfaction_score ELSE NULL END), 0.0) AS avg_satisfaction_customer_prev_90d,
        -- Count of interactions for this customer in the 90 days prior to or on cutoff
        COALESCE(COUNT(CASE WHEN julianday(i_hist.interaction_date) <= julianday('{GLOBAL_PREDICTION_CUTOFF_DATE}')
                       AND julianday(i_hist.interaction_date) > julianday('{GLOBAL_PREDICTION_CUTOFF_DATE}') - 90
                  THEN i_hist.interaction_id ELSE NULL END), 0) AS num_interactions_customer_prev_90d,
        -- Days since last interaction for this customer before or on cutoff
        COALESCE(CAST(JULIANDAY('{GLOBAL_PREDICTION_CUTOFF_DATE}') - JULIANDAY(MAX(CASE WHEN julianday(i_hist.interaction_date) <= julianday('{GLOBAL_PREDICTION_CUTOFF_DATE}')
                                                                             THEN i_hist.interaction_date ELSE NULL END)) AS INTEGER), 9999) AS days_since_last_interaction_customer_at_cutoff
    FROM
        customers c
    LEFT JOIN
        interactions i_hist ON c.customer_id = i_hist.customer_id
    GROUP BY
        c.customer_id
)
SELECT
    i.interaction_id,
    i.customer_id,
    i.agent_id,
    i.interaction_date,
    i.channel,
    i.issue_type,
    i.interaction_duration_minutes,
    i.post_interaction_satisfaction_score, -- Target variable for the current interaction
    c.signup_date,
    c.region,
    c.customer_segment,
    a.department,
    a.agent_seniority,
    a.average_handle_time_minutes,
    a.past_satisfaction_rating,
    -- Historical customer aggregates (up to cutoff date, using COALESCE for initial NULLs from CTE)
    chd.avg_satisfaction_customer_prev_90d,
    chd.num_interactions_customer_prev_90d,
    chd.days_since_last_interaction_customer_at_cutoff,
    -- Time-based features from current interaction
    CAST(strftime('%w', i.interaction_date) AS INTEGER) AS day_of_week, -- 0 for Sunday, 6 for Saturday
    CAST(strftime('%H', i.interaction_date) AS INTEGER) AS hour_of_day,
    CAST(strftime('%m', i.interaction_date) AS INTEGER) AS month_of_year
FROM
    interactions i
JOIN
    customers c ON i.customer_id = c.customer_id
JOIN
    agents a ON i.agent_id = a.agent_id
LEFT JOIN
    CustomerHistoricalData chd ON i.customer_id = chd.customer_id
WHERE
    julianday(i.interaction_date) > julianday('{GLOBAL_PREDICTION_CUTOFF_DATE}')
ORDER BY
    i.interaction_date;
"""

interaction_features_df = pd.read_sql_query(sql_query, conn)
conn.close()

print("\n--- SQL Feature Engineering Complete ---")
print("Features DataFrame head (interactions after cutoff):\n", interaction_features_df.head())
print("Features DataFrame info:\n", interaction_features_df.info())
print("Number of interactions after cutoff for prediction:", len(interaction_features_df))

# 3. Pandas Feature Engineering & Multi-class Target Creation

# Convert relevant date/datetime columns
interaction_features_df['signup_date'] = pd.to_datetime(interaction_features_df['signup_date'])
interaction_features_df['interaction_date'] = pd.to_datetime(interaction_features_df['interaction_date'])

# Handle NaN values for numerical historical features (should be mostly handled by COALESCE in SQL, but good to ensure)
interaction_features_df['avg_satisfaction_customer_prev_90d'] = interaction_features_df['avg_satisfaction_customer_prev_90d'].fillna(0.0)
interaction_features_df['num_interactions_customer_prev_90d'] = interaction_features_df['num_interactions_customer_prev_90d'].fillna(0)
interaction_features_df['days_since_last_interaction_customer_at_cutoff'] = interaction_features_df['days_since_last_interaction_customer_at_cutoff'].fillna(9999)

# Calculate customer_tenure_at_interaction_days
interaction_features_df['customer_tenure_at_interaction_days'] = (
    interaction_features_df['interaction_date'] - interaction_features_df['signup_date']
).dt.days

# Create the Multi-class Target `satisfaction_category`
def categorize_satisfaction(score):
    if score <= 2:
        return 'Low'
    elif score == 3:
        return 'Medium'
    else: # score >= 4
        return 'High'

interaction_features_df['satisfaction_category'] = interaction_features_df['post_interaction_satisfaction_score'].apply(categorize_satisfaction)

# Define features X and target y
numerical_features = [
    'interaction_duration_minutes', 'average_handle_time_minutes', 'past_satisfaction_rating',
    'avg_satisfaction_customer_prev_90d', 'num_interactions_customer_prev_90d',
    'days_since_last_interaction_customer_at_cutoff', 'day_of_week', 'hour_of_day',
    'month_of_year', 'customer_tenure_at_interaction_days'
]
categorical_features = [
    'region', 'customer_segment', 'channel', 'issue_type', 'department', 'agent_seniority'
]

# Ensure all feature columns exist in the DataFrame
all_features = numerical_features + categorical_features
missing_features = [f for f in all_features if f not in interaction_features_df.columns]
if missing_features:
    raise ValueError(f"Missing features in the DataFrame: {missing_features}")

X = interaction_features_df[all_features]
y = interaction_features_df['satisfaction_category']

# Convert y to pandas Series with categorical dtype for robust handling in scikit-learn and value_counts
y = y.astype('category')

print("\n--- Pandas Feature Engineering & Target Creation Complete ---")
print("Target Variable Distribution:\n", y.value_counts(normalize=True))

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

print(f"\nTrain set size: {len(X_train)} samples")
print(f"Test set size: {len(X_test)} samples")
print("Train target distribution:\n", y_train.value_counts(normalize=True))
print("Test target distribution:\n", y_test.value_counts(normalize=True))


# 4. Data Visualization

print("\n--- Generating Visualizations ---")
plt.style.use('seaborn-v0_8-darkgrid')

# Violin plot: interaction_duration_minutes vs. satisfaction_category
plt.figure(figsize=(10, 6))
sns.violinplot(x='satisfaction_category', y='interaction_duration_minutes', data=interaction_features_df,
               order=['Low', 'Medium', 'High'], palette='viridis')
plt.title('Distribution of Interaction Duration by Satisfaction Category')
plt.xlabel('Satisfaction Category')
plt.ylabel('Interaction Duration (minutes)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Stacked bar chart: proportion of satisfaction_category for different channel values
channel_satisfaction_proportions = interaction_features_df.groupby('channel')['satisfaction_category'].value_counts(normalize=True).unstack(fill_value=0)
# Ensure columns are in the desired order
channel_satisfaction_proportions = channel_satisfaction_proportions[['Low', 'Medium', 'High']]

plt.figure(figsize=(12, 7))
channel_satisfaction_proportions.plot(kind='bar', stacked=True, colormap='viridis')
plt.title('Proportion of Satisfaction Categories by Channel')
plt.xlabel('Channel')
plt.ylabel('Proportion')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Satisfaction Category', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


# 5. ML Pipeline & Evaluation (Multi-class Classification)

print("\n--- Building and Training ML Pipeline ---")

# Preprocessing steps for numerical and categorical features
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

# Train the model
model_pipeline.fit(X_train, y_train)

print("\n--- Evaluating ML Pipeline ---")

# Make predictions on the test set
y_pred = model_pipeline.predict(X_test)

# Print classification report
print("\nClassification Report on Test Set:")
print(classification_report(y_test, y_pred, target_names=['Low', 'Medium', 'High']))

print("\n--- ML Pipeline Execution Complete ---")