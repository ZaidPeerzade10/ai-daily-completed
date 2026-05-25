import pandas as pd
import numpy as np
import sqlite3
from datetime import timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, classification_report
import random

# --- 1. Generate Synthetic Data (Pandas/Numpy) ---

# Set a fixed reference date for consistency in data generation
REFERENCE_DATE = pd.Timestamp('2023-10-31')

# 1.1 customers_df
num_customers = np.random.randint(1000, 1500)
customer_ids = np.arange(1, num_customers + 1)
signup_dates = [REFERENCE_DATE - pd.Timedelta(days=np.random.randint(365 * 3, 365 * 5)) for _ in range(num_customers)]
subscription_plans = np.random.choice(['Basic', 'Premium', 'Family'], num_customers, p=[0.4, 0.4, 0.2])
regions = np.random.choice(['North', 'South', 'East', 'West'], num_customers)
age_groups = np.random.choice(['18-24', '25-44', '45+'], num_customers, p=[0.2, 0.5, 0.3])

customers_df = pd.DataFrame({
    'customer_id': customer_ids,
    'signup_date': signup_dates,
    'subscription_plan': subscription_plans,
    'region': regions,
    'age_group': age_groups,
})

# Simulate churn: 15-20% of customers churn
churn_rate_actual = np.random.uniform(0.15, 0.20)
num_churners = int(num_customers * churn_rate_actual)
churner_indices = np.random.choice(customers_df.index, num_churners, replace=False)

churn_dates_series = pd.Series([pd.NaT] * num_customers, index=customers_df.index)
for idx in churner_indices:
    signup = customers_df.loc[idx, 'signup_date']
    # Churn date must be after signup and within the last 12 months (relative to REFERENCE_DATE)
    # Ensure a valid time range for churn date generation
    min_churn_date = max(signup, REFERENCE_DATE - pd.Timedelta(days=365))
    max_churn_date = REFERENCE_DATE

    if min_churn_date < max_churn_date:
        # Generate random timestamp within the valid range
        random_timestamp_seconds = np.random.randint(min_churn_date.timestamp(), max_churn_date.timestamp())
        churn_dates_series[idx] = pd.Timestamp(random_timestamp_seconds, unit='s')
    else:
        # If no valid range, this customer won't churn in the simulated period
        pass # churn_dates_series[idx] remains NaT

customers_df['churn_date'] = churn_dates_series
customers_df_original_churn_dates = customers_df[['customer_id', 'churn_date']].copy() # Keep for later target calculation

print("--- Generated `customers_df` head ---")
print(customers_df.head())
print(f"Total customers: {len(customers_df)}, Churners: {customers_df['churn_date'].count()}")
print("-" * 40)

# 1.2 content_df
num_content = np.random.randint(100, 200)
content_ids = np.arange(1, num_content + 1)
titles = [f'Content {i}' for i in content_ids]
genres = np.random.choice(['Action', 'Comedy', 'Drama', 'Sci-Fi', 'Documentary', 'Thriller', 'Romance'], num_content)
avg_ratings = np.round(np.random.uniform(1.0, 5.0, num_content), 1)

content_df = pd.DataFrame({
    'content_id': content_ids,
    'title': titles,
    'genre': genres,
    'avg_rating': avg_ratings
})
print("\n--- Generated `content_df` head ---")
print(content_df.head())
print("-" * 40)

# 1.3 viewing_history_df - Simulate realistic patterns
all_viewing_history = []
view_id_counter = 1
min_views_per_month = {'Basic': 5, 'Premium': 15, 'Family': 10}
max_views_per_month = {'Basic': 15, 'Premium': 30, 'Family': 25}
avg_duration_base = {'Basic': 30, 'Premium': 60, 'Family': 45} # minutes

for idx, customer in customers_df.iterrows():
    customer_id = customer['customer_id']
    signup_date = customer['signup_date']
    churn_date = customer['churn_date']
    subscription_plan = customer['subscription_plan']

    # Determine the end date for viewing activity for this customer
    # It's either REFERENCE_DATE or the churn_date, whichever comes first
    activity_end_date = REFERENCE_DATE
    if pd.notna(churn_date):
        activity_end_date = min(REFERENCE_DATE, churn_date)

    # Generate views between signup_date and activity_end_date
    current_month_start = signup_date.replace(day=1) # Start from the beginning of the month of signup
    while current_month_start <= activity_end_date:
        # Ensure activity does not extend past activity_end_date
        month_end_for_views = min(current_month_start + pd.Timedelta(days=29), activity_end_date)
        if month_end_for_views < current_month_start: # If month end is before start, no activity this month
            current_month_start += pd.Timedelta(days=30) # Move to next month
            continue

        num_views_month = np.random.randint(min_views_per_month[subscription_plan], max_views_per_month[subscription_plan] + 1)

        # Simulate drop-off for churners in the 1-2 months leading up to churn_date
        is_churner_pre_churn_window = pd.notna(churn_date) and current_month_start > (churn_date - pd.Timedelta(days=60)) and current_month_start < churn_date
        if is_churner_pre_churn_window:
            # Reduce views and duration significantly
            num_views_month = int(num_views_month * np.random.uniform(0.1, 0.4)) # 10-40% of normal
            avg_duration_factor = np.random.uniform(0.3, 0.6) # 30-60% of normal
        else:
            avg_duration_factor = np.random.uniform(0.8, 1.2) # Normal fluctuation

        for _ in range(num_views_month):
            if month_end_for_views >= current_month_start:
                # Generate random day within the valid month range
                view_date_candidate = current_month_start + pd.Timedelta(days=random.randint(0, (month_end_for_views - current_month_start).days))
                # Add random time of day
                view_date_candidate = view_date_candidate.replace(hour=random.randint(0,23), minute=random.randint(0,59), second=random.randint(0,59))

                content_id = np.random.choice(content_df['content_id'])
                duration = max(5.0, np.round(avg_duration_base[subscription_plan] * avg_duration_factor * np.random.uniform(0.7, 1.3), 1))
                duration = min(duration, 180.0) # Cap duration

                all_viewing_history.append({
                    'view_id': view_id_counter,
                    'customer_id': customer_id,
                    'content_id': content_id,
                    'view_date': view_date_candidate,
                    'duration_minutes': duration
                })
                view_id_counter += 1
        
        current_month_start += pd.Timedelta(days=30) # Move to next month (approx)

viewing_history_df = pd.DataFrame(all_viewing_history)

# Ensure enough records are generated as per requirements (20k-30k)
# If fewer, this simple script proceeds, but in a real setting, regenerate or adjust parameters.
if len(viewing_history_df) < 20000:
    print(f"Warning: Only {len(viewing_history_df)} viewing records generated. Expected 20k-30k. Consider adjusting parameters.")

viewing_history_df = viewing_history_df.sort_values(by=['customer_id', 'view_date']).reset_index(drop=True)

print("\n--- Generated `viewing_history_df` head ---")
print(viewing_history_df.head())
print(f"Total viewing records: {len(viewing_history_df)}")
print("-" * 40)

# --- 2. Load into SQLite & SQL Feature Engineering ---

conn = sqlite3.connect(':memory:')

# Convert date columns to string for SQLite compatibility and consistency
# Note: For `churn_date`, pd.NaT is converted to None automatically by to_sql if the column dtype is object (mixed)
# Explicit conversion to string handles NaT by not including it.
customers_df['signup_date'] = customers_df['signup_date'].dt.strftime('%Y-%m-%d')
customers_df['churn_date'] = customers_df['churn_date'].apply(lambda x: x.strftime('%Y-%m-%d') if pd.notna(x) else None)
viewing_history_df['view_date'] = viewing_history_df['view_date'].dt.strftime('%Y-%m-%d %H:%M:%S')

customers_df.to_sql('customers', conn, index=False, if_exists='replace')
content_df.to_sql('content', conn, index=False, if_exists='replace')
viewing_history_df.to_sql('viewing_history', conn, index=False, if_exists='replace')

# Define GLOBAL_PREDICTION_CUTOFF_DATE
# Safely get the max view date, handle empty viewing history if it happens
if not viewing_history_df.empty:
    latest_view_date_str = viewing_history_df['view_date'].max()
    latest_view_date_pd = pd.to_datetime(latest_view_date_str)
    GLOBAL_PREDICTION_CUTOFF_DATE = latest_view_date_pd - pd.Timedelta(days=30) # 1 month prior approx
else:
    GLOBAL_PREDICTION_CUTOFF_DATE = REFERENCE_DATE - pd.Timedelta(days=30)
    print("Warning: Viewing history is empty. GLOBAL_PREDICTION_CUTOFF_DATE set relative to REFERENCE_DATE.")

GLOBAL_PREDICTION_CUTOFF_DATE_STR = GLOBAL_PREDICTION_CUTOFF_DATE.strftime('%Y-%m-%d')

print(f"\nGLOBAL_PREDICTION_CUTOFF_DATE: {GLOBAL_PREDICTION_CUTOFF_DATE_STR}")
print("-" * 40)

# SQL Query for feature engineering
sql_query = f"""
SELECT
    c.customer_id,
    c.signup_date,
    c.subscription_plan,
    c.region,
    c.age_group,
    '{GLOBAL_PREDICTION_CUTOFF_DATE_STR}' AS current_cutoff_date,
    COALESCE(SUM(CASE WHEN STRFTIME('%Y-%m-%d', v.view_date) BETWEEN DATE('{GLOBAL_PREDICTION_CUTOFF_DATE_STR}', '-30 days') AND DATE('{GLOBAL_PREDICTION_CUTOFF_DATE_STR}') THEN v.duration_minutes ELSE 0 END), 0.0) AS total_view_duration_prev_30d,
    COALESCE(COUNT(CASE WHEN STRFTIME('%Y-%m-%d', v.view_date) BETWEEN DATE('{GLOBAL_PREDICTION_CUTOFF_DATE_STR}', '-30 days') AND DATE('{GLOBAL_PREDICTION_CUTOFF_DATE_STR}') THEN v.view_id END), 0) AS num_views_prev_30d,
    COALESCE(COUNT(DISTINCT CASE WHEN STRFTIME('%Y-%m-%d', v.view_date) BETWEEN DATE('{GLOBAL_PREDICTION_CUTOFF_DATE_STR}', '-30 days') AND DATE('{GLOBAL_PREDICTION_CUTOFF_DATE_STR}') THEN v.content_id END), 0) AS num_unique_content_prev_30d,
    COALESCE(COUNT(DISTINCT CASE WHEN STRFTIME('%Y-%m-%d', v.view_date) BETWEEN DATE('{GLOBAL_PREDICTION_CUTOFF_DATE_STR}', '-30 days') AND DATE('{GLOBAL_PREDICTION_CUTOFF_DATE_STR}') THEN ct.genre END), 0) AS num_unique_genres_prev_30d,
    COALESCE(
        JULIANDAY('{GLOBAL_PREDICTION_CUTOFF_DATE_STR}') - JULIANDAY(MAX(CASE WHEN STRFTIME('%Y-%m-%d', v.view_date) <= DATE('{GLOBAL_PREDICTION_CUTOFF_DATE_STR}') THEN STRFTIME('%Y-%m-%d', v.view_date) ELSE NULL END)),
        9999
    ) AS days_since_last_view_at_cutoff
FROM
    customers c
LEFT JOIN
    viewing_history v ON c.customer_id = v.customer_id
LEFT JOIN
    content ct ON v.content_id = ct.content_id
GROUP BY
    c.customer_id, c.signup_date, c.subscription_plan, c.region, c.age_group
ORDER BY
    c.customer_id;
"""

customer_features_df = pd.read_sql_query(sql_query, conn)
conn.close()

print("\n--- SQL Feature Engineered `customer_features_df` head ---")
print(customer_features_df.head())
print(f"Shape of `customer_features_df`: {customer_features_df.shape}")
print("-" * 40)

# --- 3. Pandas Feature Engineering & Binary Target Creation ---

# Convert date columns
customer_features_df['signup_date'] = pd.to_datetime(customer_features_df['signup_date'])
customer_features_df['current_cutoff_date'] = pd.to_datetime(customer_features_df['current_cutoff_date'])

# Handle NaN values (SQL COALESCE already handles most, but for safety)
numerical_agg_cols = [
    'total_view_duration_prev_30d', 'num_views_prev_30d',
    'num_unique_content_prev_30d', 'num_unique_genres_prev_30d'
]
customer_features_df[numerical_agg_cols] = customer_features_df[numerical_agg_cols].fillna(0)
customer_features_df['days_since_last_view_at_cutoff'] = customer_features_df['days_since_last_view_at_cutoff'].fillna(9999)

# Calculate customer_tenure_at_cutoff_days
customer_features_df['customer_tenure_at_cutoff_days'] = (customer_features_df['current_cutoff_date'] - customer_features_df['signup_date']).dt.days

# Calculate avg_view_duration_per_view_prev_30d
customer_features_df['avg_view_duration_per_view_prev_30d'] = customer_features_df['total_view_duration_prev_30d'] / customer_features_df['num_views_prev_30d']
customer_features_df['avg_view_duration_per_view_prev_30d'] = customer_features_df['avg_view_duration_per_view_prev_30d'].replace([np.inf, -np.inf], np.nan).fillna(0)


# Create the Binary Target `will_churn_in_next_30_days`
# Merge original churn_dates to determine the target
customer_features_df = customer_features_df.merge(customers_df_original_churn_dates, on='customer_id', how='left')

# Define the 30-day period *immediately following* the cutoff date
churn_start_window = customer_features_df['current_cutoff_date']
churn_end_window = customer_features_df['current_cutoff_date'] + pd.Timedelta(days=30)

# Check if churn_date falls within this window
customer_features_df['will_churn_in_next_30_days'] = (
    (customer_features_df['churn_date'] >= churn_start_window) &
    (customer_features_df['churn_date'] < churn_end_window)
).astype(int)

print("\n--- `customer_features_df` with new features and target head ---")
print(customer_features_df.head())
print(f"Churners in next 30 days: {customer_features_df['will_churn_in_next_30_days'].sum()} ({customer_features_df['will_churn_in_next_30_days'].mean()*100:.2f}%)")
print("-" * 40)

# Define features (X) and target (y)
numerical_features = [
    'total_view_duration_prev_30d', 'num_views_prev_30d',
    'num_unique_content_prev_30d', 'num_unique_genres_prev_30d',
    'days_since_last_view_at_cutoff', 'customer_tenure_at_cutoff_days',
    'avg_view_duration_per_view_prev_30d'
]
categorical_features = ['subscription_plan', 'region', 'age_group']

X = customer_features_df[numerical_features + categorical_features]
y = customer_features_df['will_churn_in_next_30_days']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

print(f"\nShape of X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"Shape of X_test: {X_test.shape}, y_test: {y_test.shape}")
print(f"Churn rate in y_train: {y_train.mean()*100:.2f}%")
print(f"Churn rate in y_test: {y_test.mean()*100:.2f}%")
print("-" * 40)


# --- 4. Data Visualization (Matplotlib/Seaborn) ---

plt.figure(figsize=(14, 6))

# Plot 1: Violin plot for total_view_duration_prev_30d
plt.subplot(1, 2, 1)
sns.violinplot(x='will_churn_in_next_30_days', y='total_view_duration_prev_30d', data=customer_features_df)
plt.title('Total View Duration in Previous 30 Days by Churn Status')
plt.xlabel('Will Churn in Next 30 Days (0=No, 1=Yes)')
plt.ylabel('Total View Duration (minutes)')

# Plot 2: Stacked bar chart for subscription_plan vs churn
plt.subplot(1, 2, 2)
# Ensure columns are numeric for calculation by `value_counts`
churn_by_plan = customer_features_df.groupby('subscription_plan')['will_churn_in_next_30_days'].value_counts(normalize=True).unstack()
if 0 not in churn_by_plan.columns: # Handle cases where a group has no non-churners
    churn_by_plan[0] = 0
if 1 not in churn_by_plan.columns: # Handle cases where a group has no churners
    churn_by_plan[1] = 0
churn_by_plan = churn_by_plan[[0,1]] # Ensure order

churn_by_plan.plot(kind='bar', stacked=True, ax=plt.gca(), cmap='viridis')
plt.title('Churn Proportion by Subscription Plan')
plt.xlabel('Subscription Plan')
plt.ylabel('Proportion')
plt.xticks(rotation=45)
plt.legend(title='Will Churn', labels=['No', 'Yes'])

plt.tight_layout()
plt.show()

print("\n--- Data Visualization completed ---")
print("-" * 40)


# --- 5. ML Pipeline & Evaluation (Binary Classification) ---

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
model_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                 ('classifier', HistGradientBoostingClassifier(random_state=42, class_weight='balanced'))])

# Train the model
model_pipeline.fit(X_train, y_train)

# Predict probabilities on the test set for ROC AUC
y_pred_proba = model_pipeline.predict_proba(X_test)[:, 1]
# Predict class labels on the test set for classification report
y_pred = model_pipeline.predict(X_test) 

print("\n--- Model Training & Prediction Completed ---")
print("-" * 40)

# Evaluate the model
roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"ROC AUC Score on Test Set: {roc_auc:.4f}")

print("\nClassification Report on Test Set:")
print(classification_report(y_test, y_pred))
print("-" * 40)