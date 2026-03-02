import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, precision_score, recall_score, f1_score

# --- 1. Simulate and Generate Initial Datasets ---

# Constants for simulation
NUM_CUSTOMERS = 5000
SIMULATION_START_DATE = datetime(2022, 1, 1)
SIMULATION_END_DATE = datetime(2023, 11, 30) # End date for general event generation
CHURN_RATE = 0.25 # Percentage of customers who will churn
AVG_EVENTS_PER_MONTH = 10
EVENT_TYPES = ['login', 'feature_A_usage', 'feature_B_usage', 'support_contact', 'settings_update']
REGIONS = ['North', 'South', 'East', 'West']
ACCOUNT_TYPES = ['Individual', 'Business']
SUBSCRIPTION_PLANS = ['Basic', 'Premium', 'Enterprise']

# Generate Customer Profiles
np.random.seed(42) # for reproducibility

customer_ids = [f'CUST_{i:05d}' for i in range(NUM_CUSTOMERS)]
signup_dates = [SIMULATION_START_DATE + timedelta(days=np.random.randint(0, (SIMULATION_END_DATE - SIMULATION_START_DATE).days)) for _ in range(NUM_CUSTOMERS)]
regions = np.random.choice(REGIONS, NUM_CUSTOMERS)
account_types = np.random.choice(ACCOUNT_TYPES, NUM_CUSTOMERS)
subscription_plans = np.random.choice(SUBSCRIPTION_PLANS, NUM_CUSTOMERS, p=[0.6, 0.3, 0.1])

customers_df = pd.DataFrame({
    'customer_id': customer_ids,
    'signup_date': signup_dates,
    'region': regions,
    'account_type': account_types,
    'subscription_plan': subscription_plans
})

# Simulate churn_date:
# For non-churned customers, set churn_date far in the future or NaN
# For churned customers, set a realistic churn_date after signup
churned_customers_count = int(NUM_CUSTOMERS * CHURN_RATE)
churned_indices = np.random.choice(customers_df.index, churned_customers_count, replace=False)

customers_df['churn_date'] = pd.NaT # Initialize with Not a Time
for idx in churned_indices:
    # Churn date must be after signup date and within simulation end date
    churn_possible_start = customers_df.loc[idx, 'signup_date'] + timedelta(days=30) # At least 30 days active
    churn_possible_end = SIMULATION_END_DATE - timedelta(days=30) # Churn should not be too close to end

    if churn_possible_start < churn_possible_end: # Ensure valid range for churn date
        customers_df.loc[idx, 'churn_date'] = churn_possible_start + timedelta(days=np.random.randint(0, (churn_possible_end - churn_possible_start).days))
    else:
        # If no valid churn window, consider them non-churned for this simulation
        customers_df.loc[idx, 'churn_date'] = pd.NaT


# Generate Usage Events
all_events = []
for _, row in customers_df.iterrows():
    customer_id = row['customer_id']
    signup_date = row['signup_date']
    churn_date = row['churn_date']

    # Determine the end date for event generation for this customer
    # If churned, events stop on or before churn_date
    # If not churned, events continue up to SIMULATION_END_DATE
    event_generation_end = churn_date if pd.notna(churn_date) else SIMULATION_END_DATE

    # Events should start after signup_date
    start_event_date = signup_date

    # Ensure valid period for event generation
    if start_event_date >= event_generation_end:
        continue # No events if signup is too late or churns immediately

    current_date = start_event_date
    while current_date <= event_generation_end:
        # Simulate varying activity levels (e.g., more events early, then tapering)
        num_events_this_day = np.random.poisson(AVG_EVENTS_PER_MONTH / 30) # Average daily events

        for _ in range(num_events_this_day):
            event_timestamp = current_date + timedelta(seconds=np.random.randint(0, 86400)) # Random time within the day
            event_type = np.random.choice(EVENT_TYPES, p=[0.4, 0.2, 0.2, 0.1, 0.1]) # Different probabilities
            event_value = np.random.rand() * 100 if 'usage' in event_type else 0 # Value for usage events

            # Ensure event is not *after* churn_date if customer is churned
            if pd.notna(churn_date) and event_timestamp > churn_date:
                continue

            all_events.append({
                'customer_id': customer_id,
                'event_timestamp': event_timestamp,
                'event_type': event_type,
                'event_value': event_value
            })
        current_date += timedelta(days=1)

usage_events_df = pd.DataFrame(all_events)
# Sort to ensure proper chronological order for later use
usage_events_df = usage_events_df.sort_values(by=['customer_id', 'event_timestamp']).reset_index(drop=True)

print(f"Simulated {NUM_CUSTOMERS} customers and {len(usage_events_df)} usage events.")
print(f"Number of churned customers in simulation: {customers_df['churn_date'].count()}")
print("-" * 50)

# --- 2. Define Key Dates and Time Windows ---
global_analysis_date = datetime(2023, 9, 15) # "Today" for analysis
feature_window_duration = timedelta(days=60) # Look back 60 days for features
feature_cutoff_date = global_analysis_date - feature_window_duration # Features up to this date
churn_observation_period_duration = timedelta(days=90) # Observe churn for 90 days after feature_cutoff
churn_observation_end_date = feature_cutoff_date + churn_observation_period_duration

print(f"Global Analysis Date: {global_analysis_date.date()}")
print(f"Feature Window Duration: {feature_window_duration.days} days")
print(f"Feature Cutoff Date: {feature_cutoff_date.date()}")
print(f"Churn Observation Period Duration: {churn_observation_period_duration.days} days")
print(f"Churn Observation End Date: {churn_observation_end_date.date()}")
print("-" * 50)

# --- 3. Feature Engineering - Static Profile & Time-Agnostic Features ---
# Start with customer profiles for our feature set
features_df = customers_df[['customer_id', 'signup_date', 'region', 'account_type', 'subscription_plan', 'churn_date']].copy()

# Account Age at feature_cutoff_date
features_df['account_age_at_feature_cutoff'] = (feature_cutoff_date - features_df['signup_date']).dt.days
# Handle cases where signup is after cutoff (should be rare, but possible with simulation dates)
features_df.loc[features_df['account_age_at_feature_cutoff'] < 0, 'account_age_at_feature_cutoff'] = 0

print("Generated static and account age features.")
print("-" * 50)

# --- 4. Advanced Feature Engineering - Time-Windowed Usage Patterns ---

# Filter usage data to only include events before or on the feature_cutoff_date
# And within the feature window
filtered_usage_df = usage_events_df[
    (usage_events_df['event_timestamp'] <= feature_cutoff_date) &
    (usage_events_df['event_timestamp'] > (feature_cutoff_date - feature_window_duration))
].copy()

if filtered_usage_df.empty:
    print("Warning: No usage events found within the feature window. This might lead to issues.")
    # Create an empty aggregation DataFrame with all customer IDs
    usage_features = pd.DataFrame({'customer_id': customers_df['customer_id'].unique()})
    # Add dummy columns for all expected aggregated features, filled with NaNs
    dummy_cols = [
        'total_events_last_60_days', 'avg_daily_events_last_60_days',
        'distinct_event_types_last_60_days', 'days_since_last_activity',
        'login_count_last_60_days', 'feature_A_usage_count_last_60_days',
        'feature_B_usage_count_last_60_days', 'support_contact_count_last_60_days',
        'avg_event_value_last_60_days', 'std_event_value_last_60_days',
        'sum_event_value_last_60_days'
    ]
    for col in dummy_cols:
        usage_features[col] = np.nan
else:
    # Aggregate within feature window
    usage_features = filtered_usage_df.groupby('customer_id').agg(
        total_events_last_60_days=('event_timestamp', 'count'),
        latest_event_timestamp=('event_timestamp', 'max'),
        distinct_event_types_last_60_days=('event_type', 'nunique'),
        # Specific event type counts
        login_count_last_60_days=('event_type', lambda x: (x == 'login').sum()),
        feature_A_usage_count_last_60_days=('event_type', lambda x: (x == 'feature_A_usage').sum()),
        feature_B_usage_count_last_60_days=('event_type', lambda x: (x == 'feature_B_usage').sum()),
        support_contact_count_last_60_days=('event_type', lambda x: (x == 'support_contact').sum()),
        # Aggregations on event_value
        avg_event_value_last_60_days=('event_value', 'mean'),
        std_event_value_last_60_days=('event_value', 'std'),
        sum_event_value_last_60_days=('event_value', 'sum')
    ).reset_index()

    # Calculate derived metrics
    usage_features['avg_daily_events_last_60_days'] = usage_features['total_events_last_60_days'] / feature_window_duration.days
    usage_features['days_since_last_activity'] = (feature_cutoff_date - usage_features['latest_event_timestamp']).dt.days

    # Drop intermediate latest_event_timestamp
    usage_features = usage_features.drop(columns=['latest_event_timestamp'])


# Merge usage features with the main features_df
features_df = pd.merge(features_df, usage_features, on='customer_id', how='left')

# Fill NaN values for customers with no activity in the feature window
# For counts/sums, fill with 0
for col in ['total_events_last_60_days', 'distinct_event_types_last_60_days',
            'login_count_last_60_days', 'feature_A_usage_count_last_60_days',
            'feature_B_usage_count_last_60_days', 'support_contact_count_last_60_days',
            'sum_event_value_last_60_days']:
    features_df[col] = features_df[col].fillna(0)

# For averages/std, fill with 0
features_df['avg_daily_events_last_60_days'] = features_df['avg_daily_events_last_60_days'].fillna(0)
features_df['avg_event_value_last_60_days'] = features_df['avg_event_value_last_60_days'].fillna(0)
features_df['std_event_value_last_60_days'] = features_df['std_event_value_last_60_days'].fillna(0)

# For days_since_last_activity, if no activity, assume maximum possible in the feature window
# (i.e., customer had no activity in the last 60 days, so the last activity was 60+ days ago).
# A common proxy for this is feature_window_duration.days.
features_df['days_since_last_activity'] = features_df['days_since_last_activity'].fillna(feature_window_duration.days)


# Ratio features
features_df['support_contact_ratio_last_60_days'] = (
    features_df['support_contact_count_last_60_days'] / features_df['total_events_last_60_days']
).fillna(0) # Fill division by zero with 0

print("Generated time-windowed usage features.")
print(f"Shape of features_df after feature engineering: {features_df.shape}")
print("-" * 50)

# --- 5. Target Variable Creation (`is_churned`) ---

# Create the binary target variable 'is_churned'
features_df['is_churned'] = 0 # Default to not churned

# A customer is churned if their churn_date is AFTER feature_cutoff_date AND
# ON OR BEFORE churn_observation_end_date
features_df.loc[
    (features_df['churn_date'].notna()) &
    (features_df['churn_date'] > feature_cutoff_date) &
    (features_df['churn_date'] <= churn_observation_end_date),
    'is_churned'
] = 1

# Exclude customers who churned BEFORE or ON the feature_cutoff_date
# These customers already churned before our observation window for features closed,
# so they are not relevant for predicting future churn from this feature set.
initial_customers_count = len(features_df)
features_df = features_df[
    (features_df['churn_date'].isna()) | # Not churned yet (or churned much later)
    (features_df['churn_date'] > feature_cutoff_date) # Churned after feature cutoff
].copy()

# Drop original churn_date and signup_date as they are either used or would cause leakage
features_df = features_df.drop(columns=['churn_date', 'signup_date'])

print(f"Target variable 'is_churned' created.")
print(f"Excluded {initial_customers_count - len(features_df)} customers who churned before or on feature_cutoff_date.")
print(f"Remaining customers for modeling: {len(features_df)}")
print(f"Churned customers in target dataset: {features_df['is_churned'].sum()} ({features_df['is_churned'].mean():.2%} of total)")
print("-" * 50)

# --- 6. Data Preparation for Modeling ---

# Separate target variable
X = features_df.drop(columns=['customer_id', 'is_churned'])
y = features_df['is_churned']

# Identify numerical and categorical features
numerical_features = X.select_dtypes(include=np.number).columns.tolist()
categorical_features = X.select_dtypes(include='object').columns.tolist()

# Preprocessing pipelines for numerical and categorical features
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')), # Impute NaNs that might arise from std_event_value for single events
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')), # Should not be needed if simulation is robust
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Create a preprocessor using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Split dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Data split into training ({len(X_train)} samples) and test ({len(X_test)} samples) sets.")
print(f"Numerical features: {numerical_features}")
print(f"Categorical features: {categorical_features}")
print("-" * 50)

# --- 7. Model Training and Evaluation ---

# Model selection and pipeline
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42, n_estimators=100, class_weight='balanced'))
])

print("Training the Random Forest Classifier...")
model_pipeline.fit(X_train, y_train)
print("Training complete.")

# Make predictions on the test set
y_pred = model_pipeline.predict(X_test)
y_prob = model_pipeline.predict_proba(X_test)[:, 1] # Probability of churn

# Evaluation
print("\n--- Model Evaluation on Test Set ---")
print("Classification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(f"True Negatives: {cm[0,0]}, False Positives: {cm[0,1]}")
print(f"False Negatives: {cm[1,0]}, True Positives: {cm[1,1]}")

roc_auc = roc_auc_score(y_test, y_prob)
print(f"\nROC AUC Score: {roc_auc:.4f}")

precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print(f"Precision (churn class): {precision:.4f}")
print(f"Recall (churn class): {recall:.4f}")
print(f"F1-Score (churn class): {f1:.4f}")
print("-" * 50)

# --- 8. Model Deployment and Monitoring (High-Level) ---
print("\n--- High-Level Deployment and Monitoring Considerations ---")
print("This trained model can be deployed as an API endpoint to score new customers periodically.")
print(f"When predicting for a new customer (e.g., CUST_NEW), the same feature engineering pipeline (steps 3-4) would be applied:")
print(f"  1. Define a current 'global_analysis_date' (e.g., today's date).")
print(f"  2. Calculate 'feature_cutoff_date' (today - {feature_window_duration.days} days).")
print(f"  3. Extract static features and calculate 'account_age_at_feature_cutoff'.")
print(f"  4. Aggregate usage events for CUST_NEW within the last {feature_window_duration.days} days leading up to 'feature_cutoff_date'.")
print(f"  5. Use the preprocessor and trained classifier to get a churn probability for CUST_NEW.")
print("Continuous monitoring of model performance (e.g., ROC AUC, precision/recall for actual churn vs. predicted churn) is crucial.")
print("Data drift (changes in feature distributions) and concept drift (changes in churn behavior patterns) necessitate periodic retraining of the model with fresh data.")
print("This script provides a solid foundation for a comprehensive churn prediction system.")