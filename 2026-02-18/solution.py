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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# --- 1. Generate Synthetic Data (Pandas/Numpy) ---

print("--- 1. Generating Synthetic Data ---")

# Seed for reproducibility
np.random.seed(42)

# Customers DataFrame
num_customers = np.random.randint(500, 701)
customer_ids = np.arange(1, num_customers + 1)

start_date_signup = pd.Timestamp.today() - pd.DateOffset(years=5)
end_date_signup = pd.Timestamp.today() - pd.DateOffset(months=6) # Ensure some recent activity

signup_dates = pd.to_datetime(start_date_signup + (end_date_signup - start_date_signup) * np.random.rand(num_customers))

industries = ['SaaS', 'E-commerce', 'Fintech', 'Healthcare', 'Manufacturing']
subscription_tiers = ['Bronze', 'Silver', 'Gold']

customers_df = pd.DataFrame({
    'customer_id': customer_ids,
    'signup_date': signup_dates,
    'industry': np.random.choice(industries, num_customers),
    'subscription_tier': np.random.choice(subscription_tiers, num_customers, p=[0.4, 0.4, 0.2])
})
customers_df['signup_date'] = customers_df['signup_date'].dt.normalize() # Ensure no time component

print(f"Generated {len(customers_df)} customers.")


# Usage Logs DataFrame
num_usage_logs = np.random.randint(5000, 8001)
log_ids = np.arange(1, num_usage_logs + 1)

usage_logs_data = []
feature_used_options = ['Login', 'Dashboard_View', 'Report_Download', 'Data_Export', 'Billing_Access', 'Search', 'Settings']

for _ in range(num_usage_logs):
    customer_id = np.random.choice(customers_df['customer_id'])
    signup_date = customers_df[customers_df['customer_id'] == customer_id]['signup_date'].iloc[0]
    
    # log_date must be after signup_date, and not in the far future
    days_since_signup = np.random.randint(1, 365 * 4) # Up to 4 years after signup
    log_date = signup_date + pd.Timedelta(days=days_since_signup)
    
    # Ensure log_date is not in the future beyond a reasonable analysis point
    if log_date > pd.Timestamp.today():
        log_date = pd.Timestamp.today() - pd.Timedelta(days=np.random.randint(1, 30))
    
    feature_used = np.random.choice(feature_used_options)
    success_status = np.random.choice([0, 1], p=[0.05, 0.95]) # Base 5% failure rate

    usage_logs_data.append([customer_id, log_date, feature_used, success_status])

usage_logs_df = pd.DataFrame(usage_logs_data, columns=['customer_id', 'log_date', 'feature_used', 'success_status'])
usage_logs_df['log_id'] = log_ids
usage_logs_df = usage_logs_df[['log_id', 'customer_id', 'log_date', 'feature_used', 'success_status']]
usage_logs_df['log_date'] = usage_logs_df['log_date'].dt.normalize() # Ensure no time component

print(f"Generated {len(usage_logs_df)} usage logs.")


# Support Tickets DataFrame
num_support_tickets = np.random.randint(800, 1201)
ticket_ids = np.arange(1, num_support_tickets + 1)

support_tickets_data = []
ticket_categories = ['Bug', 'Feature_Request', 'Billing', 'Technical_Support', 'Onboarding', 'General_Inquiry']
ticket_severities = ['Low', 'Medium', 'High']

# Probabilities for ticket categories based on subscription tier
category_p_tier = {
    'Bronze': {'Bug': 0.25, 'Feature_Request': 0.1, 'Billing': 0.2, 'Technical_Support': 0.2, 'Onboarding': 0.15, 'General_Inquiry': 0.1},
    'Silver': {'Bug': 0.2, 'Feature_Request': 0.2, 'Billing': 0.15, 'Technical_Support': 0.15, 'Onboarding': 0.1, 'General_Inquiry': 0.2},
    'Gold': {'Bug': 0.1, 'Feature_Request': 0.4, 'Billing': 0.1, 'Technical_Support': 0.1, 'Onboarding': 0.05, 'General_Inquiry': 0.25}
}

for _ in range(num_support_tickets):
    customer_id = np.random.choice(customers_df['customer_id'])
    customer_info = customers_df[customers_df['customer_id'] == customer_id].iloc[0]
    signup_date = customer_info['signup_date']
    tier = customer_info['subscription_tier']

    # ticket_open_date must be after signup_date
    days_since_signup = np.random.randint(1, 365 * 4) # Up to 4 years after signup
    ticket_open_date = signup_date + pd.Timedelta(days=days_since_signup)
    
    if ticket_open_date > pd.Timestamp.today():
        ticket_open_date = pd.Timestamp.today() - pd.Timedelta(days=np.random.randint(1, 30))

    category_probs = [category_p_tier[tier][cat] for cat in ticket_categories]
    ticket_category = np.random.choice(ticket_categories, p=category_probs)
    
    ticket_severity = np.random.choice(ticket_severities, p=[0.6, 0.3, 0.1]) # Base severity distribution

    support_tickets_data.append([customer_id, ticket_open_date, ticket_category, ticket_severity])

support_tickets_df = pd.DataFrame(support_tickets_data, columns=['customer_id', 'ticket_open_date', 'ticket_category', 'ticket_severity'])
support_tickets_df['ticket_id'] = ticket_ids
support_tickets_df = support_tickets_df[['ticket_id', 'customer_id', 'ticket_open_date', 'ticket_category', 'ticket_severity']]
support_tickets_df['ticket_open_date'] = support_tickets_df['ticket_open_date'].dt.normalize() # Ensure no time component

print(f"Generated {len(support_tickets_df)} support tickets.")


# --- Simulate realistic patterns (correlation for failed usage before bug/technical tickets) ---
# This needs to modify existing data, so it's done after initial generation.
print("Applying synthetic correlation patterns...")

bug_tech_tickets = support_tickets_df[
    support_tickets_df['ticket_category'].isin(['Bug', 'Technical_Support'])
].copy()

# Customers who opened such tickets are more likely to have failed logs before the ticket.
for _, ticket_row in bug_tech_tickets.iterrows():
    customer_id = ticket_row['customer_id']
    ticket_date = ticket_row['ticket_open_date']
    
    # Define a window before the ticket date (e.g., 7 days)
    window_start = ticket_date - pd.Timedelta(days=7)
    
    # Find relevant usage logs for this customer within this window
    relevant_logs_idx = usage_logs_df[
        (usage_logs_df['customer_id'] == customer_id) &
        (usage_logs_df['log_date'] >= window_start) &
        (usage_logs_df['log_date'] < ticket_date) &
        (usage_logs_df['feature_used'].isin(['Report_Download', 'Data_Export', 'Login'])) # Features likely to cause issues
    ].index
    
    # For these logs, increase failure rate
    if not relevant_logs_idx.empty:
        num_logs_to_fail = min(len(relevant_logs_idx), np.random.randint(1, 5)) # Fail 1-4 logs in the window
        
        # Randomly select a few logs from the relevant ones to change to failure
        logs_to_fail_indices = np.random.choice(logs_to_fail_idx, size=num_logs_to_fail, replace=False)
        usage_logs_df.loc[logs_to_fail_indices, 'success_status'] = 0

print("Synthetic data generation complete and patterns applied.")


# --- 2. Load into SQLite & SQL Feature Engineering ---
print("\n--- 2. Loading into SQLite and SQL Feature Engineering ---")

conn = sqlite3.connect(':memory:')

# Convert date columns to string for SQLite compatibility
customers_df['signup_date_str'] = customers_df['signup_date'].dt.strftime('%Y-%m-%d')
usage_logs_df['log_date_str'] = usage_logs_df['log_date'].dt.strftime('%Y-%m-%d')
support_tickets_df['ticket_open_date_str'] = support_tickets_df['ticket_open_date'].dt.strftime('%Y-%m-%d')

customers_df.to_sql('customers', conn, if_exists='replace', index=False)
usage_logs_df.to_sql('usage_logs', conn, if_exists='replace', index=False)
support_tickets_df.to_sql('support_tickets', conn, if_exists='replace', index=False)

# Determine global analysis dates using pandas before SQL, as it's easier
global_analysis_date_pd = usage_logs_df['log_date'].max() + pd.Timedelta(days=30)
feature_cutoff_date_pd = global_analysis_date_pd - pd.Timedelta(days=30)

# Convert to string for SQL query
global_analysis_date_str = global_analysis_date_pd.strftime('%Y-%m-%d')
feature_cutoff_date_str = feature_cutoff_date_pd.strftime('%Y-%m-%d')

print(f"Global Analysis Date: {global_analysis_date_str}")
print(f"Feature Cutoff Date: {feature_cutoff_date_str}")

sql_query = f"""
SELECT
    c.customer_id,
    c.industry,
    c.subscription_tier,
    c.signup_date_str AS signup_date,
    COALESCE(ul.total_usage_logs_pre_cutoff, 0) AS total_usage_logs_pre_cutoff,
    COALESCE(ul.num_failed_attempts_pre_cutoff, 0) AS num_failed_attempts_pre_cutoff,
    COALESCE(ul.avg_usage_success_rate_pre_cutoff, 1.0) AS avg_usage_success_rate_pre_cutoff,
    (JULIANDAY('{feature_cutoff_date_str}') - JULIANDAY(ul.last_failed_usage_date)) AS days_since_last_failed_usage_pre_cutoff,
    COALESCE(st.num_prior_tickets_pre_cutoff, 0) AS num_prior_tickets_pre_cutoff,
    COALESCE(st.num_high_severity_tickets_pre_cutoff, 0) AS num_high_severity_tickets_pre_cutoff,
    (JULIANDAY('{feature_cutoff_date_str}') - JULIANDAY(st.last_ticket_date)) AS days_since_last_ticket_pre_cutoff
FROM
    customers AS c
LEFT JOIN
    (
        SELECT
            customer_id,
            COUNT(log_id) AS total_usage_logs_pre_cutoff,
            SUM(CASE WHEN success_status = 0 THEN 1 ELSE 0 END) AS num_failed_attempts_pre_cutoff,
            AVG(success_status) AS avg_usage_success_rate_pre_cutoff,
            MAX(CASE WHEN success_status = 0 THEN log_date_str ELSE NULL END) AS last_failed_usage_date
        FROM
            usage_logs
        WHERE
            log_date_str < '{feature_cutoff_date_str}'
        GROUP BY
            customer_id
    ) AS ul
ON
    c.customer_id = ul.customer_id
LEFT JOIN
    (
        SELECT
            customer_id,
            COUNT(ticket_id) AS num_prior_tickets_pre_cutoff,
            SUM(CASE WHEN ticket_severity = 'High' THEN 1 ELSE 0 END) AS num_high_severity_tickets_pre_cutoff,
            MAX(ticket_open_date_str) AS last_ticket_date
        FROM
            support_tickets
        WHERE
            ticket_open_date_str < '{feature_cutoff_date_str}'
        GROUP BY
            customer_id
    ) AS st
ON
    c.customer_id = st.customer_id;
"""

customer_features_df = pd.read_sql_query(sql_query, conn)
conn.close()

print(f"SQL feature engineering complete. Shape: {customer_features_df.shape}")


# --- 3. Pandas Feature Engineering & Multi-Class Target Creation ---
print("\n--- 3. Pandas Feature Engineering & Multi-Class Target Creation ---")

# Handle NaN values from SQL query results
# Fill counts/sums with 0
customer_features_df['total_usage_logs_pre_cutoff'].fillna(0, inplace=True)
customer_features_df['num_failed_attempts_pre_cutoff'].fillna(0, inplace=True)
customer_features_df['num_prior_tickets_pre_cutoff'].fillna(0, inplace=True)
customer_features_df['num_high_severity_tickets_pre_cutoff'].fillna(0, inplace=True)

# Fill avg_usage_success_rate_pre_cutoff with 1.0 (perfect success if no logs)
customer_features_df['avg_usage_success_rate_pre_cutoff'].fillna(1.0, inplace=True)

# Fill days_since_last_failed_usage_pre_cutoff and days_since_last_ticket_pre_cutoff with sentinel value
# Convert to numeric first to allow filling with int/float
customer_features_df['days_since_last_failed_usage_pre_cutoff'] = pd.to_numeric(customer_features_df['days_since_last_failed_usage_pre_cutoff'], errors='coerce')
customer_features_df['days_since_last_ticket_pre_cutoff'] = pd.to_numeric(customer_features_df['days_since_last_ticket_pre_cutoff'], errors='coerce')

sentinel_days = 9999
customer_features_df['days_since_last_failed_usage_pre_cutoff'].fillna(sentinel_days, inplace=True)
customer_features_df['days_since_last_ticket_pre_cutoff'].fillna(sentinel_days, inplace=True)


# Convert signup_date to datetime objects
customer_features_df['signup_date'] = pd.to_datetime(customer_features_df['signup_date'])

# Calculate account_age_at_cutoff_days
customer_features_df['account_age_at_cutoff_days'] = (feature_cutoff_date_pd - customer_features_df['signup_date']).dt.days


# Create the Multi-Class Target `main_pain_point_category`
# Filter support tickets for the future period
future_tickets_df = support_tickets_df[
    (support_tickets_df['ticket_open_date'] >= feature_cutoff_date_pd) &
    (support_tickets_df['ticket_open_date'] < global_analysis_date_pd)
].copy()

# For each customer, find the most frequent ticket_category
# Use a custom aggregation to get the most frequent category
def get_most_frequent(series):
    if series.empty:
        return 'No_Future_Tickets'
    return series.mode()[0] # .mode() handles ties by returning all, [0] picks the first

future_pain_points = future_tickets_df.groupby('customer_id')['ticket_category'].apply(get_most_frequent).reset_index()
future_pain_points.rename(columns={'ticket_category': 'main_pain_point_category'}, inplace=True)

# Merge back to the main features DataFrame
customer_features_df = pd.merge(
    customer_features_df,
    future_pain_points,
    on='customer_id',
    how='left'
)

# Assign 'No_Future_Tickets' to customers with no tickets in the future period
customer_features_df['main_pain_point_category'].fillna('No_Future_Tickets', inplace=True)

print(f"Pandas feature engineering complete. Shape: {customer_features_df.shape}")
print("Target variable distribution:")
print(customer_features_df['main_pain_point_category'].value_counts())

# Define features X and target y
X = customer_features_df.drop(columns=['customer_id', 'signup_date', 'main_pain_point_category'])
y = customer_features_df['main_pain_point_category']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"\nTraining set shape: {X_train.shape}, Test set shape: {X_test.shape}")
print(f"Target distribution in training set:\n{y_train.value_counts(normalize=True)}")
print(f"Target distribution in test set:\n{y_test.value_counts(normalize=True)}")


# --- 4. Data Visualization ---
print("\n--- 4. Data Visualization ---")

plt.style.use('seaborn-v0_8-darkgrid')
plt.figure(figsize=(15, 6))

# Plot 1: Violin plot of avg_usage_success_rate_pre_cutoff for each main_pain_point_category
plt.subplot(1, 2, 1)
sns.violinplot(
    x='main_pain_point_category',
    y='avg_usage_success_rate_pre_cutoff',
    data=customer_features_df,
    palette='viridis'
)
plt.title('Avg Usage Success Rate vs. Future Pain Point Category')
plt.xlabel('Main Future Pain Point Category')
plt.ylabel('Average Usage Success Rate (Pre-Cutoff)')
plt.xticks(rotation=45, ha='right')


# Plot 2: Stacked bar chart of main_pain_point_category across different subscription_tier values
plt.subplot(1, 2, 2)
# Calculate proportions for stacked bar chart
tier_category_counts = customer_features_df.groupby(['subscription_tier', 'main_pain_point_category']).size().unstack(fill_value=0)
tier_category_proportions = tier_category_counts.div(tier_category_counts.sum(axis=1), axis=0)
tier_category_proportions.plot(kind='bar', stacked=True, ax=plt.gca(), cmap='tab20')
plt.title('Future Pain Point Category Distribution by Subscription Tier')
plt.xlabel('Subscription Tier')
plt.ylabel('Proportion of Customers')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Future Pain Point', bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.show()

print("Visualizations generated: violin plot and stacked bar chart.")


# --- 5. ML Pipeline & Evaluation (Multi-Class) ---
print("\n--- 5. Machine Learning Pipeline & Evaluation ---")

# Define numerical and categorical features
numerical_features = [
    'total_usage_logs_pre_cutoff',
    'num_failed_attempts_pre_cutoff',
    'avg_usage_success_rate_pre_cutoff',
    'days_since_last_failed_usage_pre_cutoff',
    'num_prior_tickets_pre_cutoff',
    'num_high_severity_tickets_pre_cutoff',
    'days_since_last_ticket_pre_cutoff',
    'account_age_at_cutoff_days'
]
categorical_features = ['industry', 'subscription_tier']

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
    ],
    remainder='passthrough' # Keep other columns if any, or drop ('drop')
)

# Create the full ML pipeline
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42, n_estimators=100, class_weight='balanced'))
])

# Train the pipeline
print("Training the Random Forest Classifier pipeline...")
model_pipeline.fit(X_train, y_train)
print("Training complete.")

# Predict on the test set
y_pred = model_pipeline.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f"\nModel Accuracy on Test Set: {accuracy:.4f}")
print("\nClassification Report on Test Set:\n", class_report)

print("\n--- Script execution complete ---")