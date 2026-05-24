import pandas as pd
import numpy as np
import datetime
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer # Corrected import for SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, classification_report

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')

# --- 1. Generate Synthetic Data (Pandas/Numpy) ---

print("--- Generating Synthetic Data ---")

# Parameters for data generation
NUM_CUSTOMERS = np.random.randint(1000, 1500)
NUM_CONTENT = np.random.randint(100, 200)
NUM_VIEWS = np.random.randint(20000, 30000)
CHURN_RATE = 0.17 # ~15-20%
DATE_RANGE_YEARS_SIGNUP = (3, 5)
DATE_RANGE_MONTHS_CHURN = 12
PREMIUM_PLAN_FACTOR = 1.5 # Premium users view 1.5x more
CHURN_DROP_OFF_FACTOR = 0.3 # Viewing activity drops to 30% before churn
CHURN_DROP_OFF_DAYS = 60 # Number of days before churn where activity drops

# Seed for reproducibility
np.random.seed(42)

# --- customers_df ---
customer_ids = np.arange(1, NUM_CUSTOMERS + 1)
signup_dates = pd.to_datetime(np.random.choice(pd.date_range(
    start=datetime.date.today() - pd.Timedelta(days=DATE_RANGE_YEARS_SIGNUP[1]*365),
    end=datetime.date.today() - pd.Timedelta(days=DATE_RANGE_YEARS_SIGNUP[0]*365)
), NUM_CUSTOMERS))

subscription_plans = np.random.choice(['Basic', 'Premium', 'Family'], NUM_CUSTOMERS, p=[0.5, 0.3, 0.2])
regions = np.random.choice(['North', 'South', 'East', 'West'], NUM_CUSTOMERS)
age_groups = np.random.choice(['18-24', '25-44', '45+'], NUM_CUSTOMERS, p=[0.25, 0.5, 0.25])

customers_df = pd.DataFrame({
    'customer_id': customer_ids,
    'signup_date': signup_dates,
    'subscription_plan': subscription_plans,
    'region': regions,
    'age_group': age_groups
})

# Simulate churn_date for a subset
churn_mask = np.random.rand(NUM_CUSTOMERS) < CHURN_RATE
churn_dates = pd.Series(pd.NaT, index=customers_df.index)
for idx in customers_df[churn_mask].index:
    min_churn_date = customers_df.loc[idx, 'signup_date'] + pd.Timedelta(days=30) # Must churn at least 30 days after signup
    max_churn_date = pd.to_datetime(datetime.date.today())
    
    # Ensure churn_date is within the last 12 months and after signup_date
    latest_possible_churn = max_churn_date
    earliest_possible_churn = max(min_churn_date, max_churn_date - pd.Timedelta(days=DATE_RANGE_MONTHS_CHURN*30))
    
    if earliest_possible_churn < latest_possible_churn:
        churn_dates.loc[idx] = pd.to_datetime(np.random.choice(
            pd.date_range(start=earliest_possible_churn, end=latest_possible_churn), 1
        )[0])
    else:
        # If no valid range for churn, mark as not churned
        churn_mask[idx] = False 
customers_df['churn_date'] = churn_dates

print(f"Generated {len(customers_df)} customers, {customers_df['churn_date'].notna().sum()} churned.")

# --- content_df ---
content_ids = np.arange(1, NUM_CONTENT + 1)
titles = [f'Content Title {i}' for i in content_ids]
genres = np.random.choice(['Action', 'Comedy', 'Drama', 'Sci-Fi', 'Documentary', 'Thriller', 'Animation'], NUM_CONTENT)
avg_ratings = np.random.uniform(1.0, 5.0, NUM_CONTENT)

content_df = pd.DataFrame({
    'content_id': content_ids,
    'title': titles,
    'genre': genres,
    'avg_rating': avg_ratings
})
print(f"Generated {len(content_df)} content items.")

# --- viewing_history_df ---
all_view_data = []
for _ in range(NUM_VIEWS):
    customer_idx = np.random.randint(0, NUM_CUSTOMERS)
    customer_id = customers_df.loc[customer_idx, 'customer_id']
    signup_date = customers_df.loc[customer_idx, 'signup_date']
    churn_date = customers_df.loc[customer_idx, 'churn_date']

    # Determine view date, ensuring it's after signup and before churn
    start_view_date = signup_date
    end_view_date = churn_date if pd.notna(churn_date) else pd.to_datetime(datetime.date.today())

    if start_view_date >= end_view_date:
        continue # Customer signed up and churned too quickly, or signed up too recently

    view_date_range = pd.date_range(start=start_view_date, end=end_view_date, freq='H')
    if len(view_date_range) == 0:
        continue
    view_date = np.random.choice(view_date_range)

    duration = np.random.uniform(5.0, 180.0)

    # Apply realistic patterns
    # Premium plan users have higher duration
    if customers_df.loc[customer_idx, 'subscription_plan'] == 'Premium':
        duration *= PREMIUM_PLAN_FACTOR
        duration = min(duration, 240.0) # Cap duration

    # Churn drop-off
    if pd.notna(churn_date) and (churn_date - view_date).days <= CHURN_DROP_OFF_DAYS:
        duration *= CHURN_DROP_OFF_FACTOR
        duration = max(duration, 5.0) # Ensure minimum duration

    all_view_data.append({
        'customer_id': customer_id,
        'content_id': np.random.choice(content_ids),
        'view_date': view_date,
        'duration_minutes': duration
    })

viewing_history_df = pd.DataFrame(all_view_data)
viewing_history_df['view_id'] = np.arange(1, len(viewing_history_df) + 1)
viewing_history_df = viewing_history_df[['view_id', 'customer_id', 'content_id', 'view_date', 'duration_minutes']]

# Sort for realistic time-series data
viewing_history_df = viewing_history_df.sort_values(by=['customer_id', 'view_date']).reset_index(drop=True)
print(f"Generated {len(viewing_history_df)} viewing history records.")

# --- 2. Load into SQLite & SQL Feature Engineering ---

print("\n--- Loading Data into SQLite and SQL Feature Engineering ---")

# Create an in-memory SQLite database
conn = sqlite3.connect(':memory:')

# Load DataFrames into SQLite tables
customers_df.to_sql('customers', conn, index=False, if_exists='replace', dtype={'signup_date': 'TEXT', 'churn_date': 'TEXT'})
content_df.to_sql('content', conn, index=False, if_exists='replace')
viewing_history_df.to_sql('viewing_history', conn, index=False, if_exists='replace', dtype={'view_date': 'TEXT'})

# Convert date columns in SQLite to appropriate format
cursor = conn.cursor()
cursor.execute("UPDATE customers SET signup_date = datetime(signup_date)")
cursor.execute("UPDATE customers SET churn_date = datetime(churn_date)")
cursor.execute("UPDATE viewing_history SET view_date = datetime(view_date)")
conn.commit()

# Define GLOBAL_PREDICTION_CUTOFF_DATE
latest_view_date_str = pd.read_sql("SELECT MAX(view_date) FROM viewing_history", conn).iloc[0, 0]
latest_view_date = pd.to_datetime(latest_view_date_str)
GLOBAL_PREDICTION_CUTOFF_DATE = latest_view_date - pd.Timedelta(days=30)
GLOBAL_PREDICTION_CUTOFF_DATE_STR = GLOBAL_PREDICTION_CUTOFF_DATE.strftime('%Y-%m-%d %H:%M:%S')

print(f"Global Prediction Cutoff Date: {GLOBAL_PREDICTION_CUTOFF_DATE_STR}")

# SQL Query for feature engineering
sql_query = f"""
WITH CustomerViewingAgg AS (
    SELECT
        vh.customer_id,
        SUM(vh.duration_minutes) AS total_view_duration_prev_30d,
        COUNT(vh.view_id) AS num_views_prev_30d,
        COUNT(DISTINCT vh.content_id) AS num_unique_content_prev_30d,
        COUNT(DISTINCT c.genre) AS num_unique_genres_prev_30d,
        MAX(vh.view_date) AS last_view_date_at_cutoff
    FROM
        viewing_history vh
    LEFT JOIN
        content c ON vh.content_id = c.content_id
    WHERE
        vh.view_date BETWEEN datetime('{GLOBAL_PREDICTION_CUTOFF_DATE_STR}', '-30 days') AND datetime('{GLOBAL_PREDICTION_CUTOFF_DATE_STR}')
    GROUP BY
        vh.customer_id
),
LastViewBeforeCutoff AS (
    SELECT
        vh.customer_id,
        MAX(vh.view_date) AS most_recent_view_before_cutoff
    FROM
        viewing_history vh
    WHERE
        vh.view_date <= datetime('{GLOBAL_PREDICTION_CUTOFF_DATE_STR}')
    GROUP BY
        vh.customer_id
)
SELECT
    c.customer_id,
    c.signup_date,
    c.subscription_plan,
    c.region,
    c.age_group,
    '{GLOBAL_PREDICTION_CUTOFF_DATE_STR}' AS current_cutoff_date,
    COALESCE(cva.total_view_duration_prev_30d, 0.0) AS total_view_duration_prev_30d,
    COALESCE(cva.num_views_prev_30d, 0) AS num_views_prev_30d,
    COALESCE(cva.num_unique_content_prev_30d, 0) AS num_unique_content_prev_30d,
    COALESCE(cva.num_unique_genres_prev_30d, 0) AS num_unique_genres_prev_30d,
    COALESCE(
        CAST(julianday('{GLOBAL_PREDICTION_CUTOFF_DATE_STR}') - julianday(lvbc.most_recent_view_before_cutoff) AS INTEGER),
        9999
    ) AS days_since_last_view_at_cutoff
FROM
    customers c
LEFT JOIN
    CustomerViewingAgg cva ON c.customer_id = cva.customer_id
LEFT JOIN
    LastViewBeforeCutoff lvbc ON c.customer_id = lvbc.customer_id;
"""

customer_features_df = pd.read_sql_query(sql_query, conn)
print(f"Generated {len(customer_features_df)} customer feature records from SQL.")

# Close SQLite connection
conn.close()

# --- 3. Pandas Feature Engineering & Binary Target Creation ---

print("\n--- Pandas Feature Engineering & Binary Target Creation ---")

# Convert date columns to datetime objects
customer_features_df['signup_date'] = pd.to_datetime(customer_features_df['signup_date'])
customer_features_df['current_cutoff_date'] = pd.to_datetime(customer_features_df['current_cutoff_date'])

# Handle NaN values (already done by COALESCE in SQL for aggregates, but good to double-check)
# For `days_since_last_view_at_cutoff` 9999 is used to indicate no views before cutoff.
# For other aggregated numerical features, COALESCE handles 0.0 or 0.

# Calculate customer_tenure_at_cutoff_days
customer_features_df['customer_tenure_at_cutoff_days'] = (
    customer_features_df['current_cutoff_date'] - customer_features_df['signup_date']
).dt.days

# Calculate avg_view_duration_per_view_prev_30d
customer_features_df['avg_view_duration_per_view_prev_30d'] = (
    customer_features_df['total_view_duration_prev_30d'] / customer_features_df['num_views_prev_30d']
)
# Fill NaN (division by zero) and inf values with 0
customer_features_df['avg_view_duration_per_view_prev_30d'].replace([np.inf, -np.inf], 0, inplace=True)
customer_features_df['avg_view_duration_per_view_prev_30d'].fillna(0, inplace=True)

# Create the Binary Target `will_churn_in_next_30_days`
# Merge original churn_date
customer_features_df = pd.merge(
    customer_features_df,
    customers_df[['customer_id', 'churn_date']],
    on='customer_id',
    how='left'
)

prediction_window_start = customer_features_df['current_cutoff_date'] + pd.Timedelta(days=1)
prediction_window_end = customer_features_df['current_cutoff_date'] + pd.Timedelta(days=30)

customer_features_df['will_churn_in_next_30_days'] = 0
churn_condition = (
    customer_features_df['churn_date'].notna() &
    (customer_features_df['churn_date'] >= prediction_window_start) &
    (customer_features_df['churn_date'] <= prediction_window_end)
)
customer_features_df.loc[churn_condition, 'will_churn_in_next_30_days'] = 1

print(f"Target distribution:\n{customer_features_df['will_churn_in_next_30_days'].value_counts()}")

# Define features X and target y
numerical_features = [
    'total_view_duration_prev_30d',
    'num_views_prev_30d',
    'num_unique_content_prev_30d',
    'num_unique_genres_prev_30d',
    'days_since_last_view_at_cutoff',
    'customer_tenure_at_cutoff_days',
    'avg_view_duration_per_view_prev_30d'
]
categorical_features = [
    'subscription_plan',
    'region',
    'age_group'
]

X = customer_features_df[numerical_features + categorical_features]
y = customer_features_df['will_churn_in_next_30_days']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

# --- 4. Data Visualization ---

print("\n--- Generating Data Visualizations ---")

plt.style.use('ggplot')
plt.figure(figsize=(14, 6))

# Violin plot for total_view_duration_prev_30d
plt.subplot(1, 2, 1)
sns.violinplot(x='will_churn_in_next_30_days', y='total_view_duration_prev_30d', data=customer_features_df)
plt.title('Total View Duration (Prev 30d) by Churn Status')
plt.xlabel('Will Churn in Next 30 Days (0=No, 1=Yes)')
plt.ylabel('Total View Duration (minutes)')
plt.xticks([0, 1], ['No Churn', 'Churn'])

# Stacked bar chart for subscription_plan vs churn
plt.subplot(1, 2, 2)
churn_by_plan = customer_features_df.groupby('subscription_plan')['will_churn_in_next_30_days'].value_counts(normalize=True).unstack()
churn_by_plan.plot(kind='bar', stacked=True, ax=plt.gca(), cmap='coolwarm')
plt.title('Churn Proportion by Subscription Plan')
plt.xlabel('Subscription Plan')
plt.ylabel('Proportion')
plt.xticks(rotation=45)
plt.legend(title='Will Churn', labels=['No Churn', 'Churn'])
plt.tight_layout()
plt.show()

# --- 5. ML Pipeline & Evaluation (Binary Classification) ---

print("\n--- Building and Evaluating ML Pipeline ---")

# Create a ColumnTransformer for preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ]), numerical_features),
        ('cat', Pipeline([
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ]), categorical_features)
    ],
    remainder='passthrough' # Keep other columns if any, though not expected here
)

# Create the ML Pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', HistGradientBoostingClassifier(random_state=42, class_weight='balanced'))
])

# Train the pipeline
pipeline.fit(X_train, y_train)
print("ML Pipeline trained successfully.")

# Predict probabilities on the test set
y_pred_proba = pipeline.predict_proba(X_test)[:, 1] # Probability of the positive class (churn=1)

# Predict classes for classification report
y_pred = pipeline.predict(X_test)

# Calculate and print ROC AUC Score
roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"\nROC AUC Score on Test Set: {roc_auc:.4f}")

# Calculate and print Classification Report
print("\nClassification Report on Test Set:")
print(classification_report(y_test, y_pred))

print("\n--- Pipeline execution complete ---")