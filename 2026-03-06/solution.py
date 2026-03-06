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
import random
import warnings

# Suppress specific warnings from sklearn and pandas
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

def generate_synthetic_data(num_users_range=(500, 700), num_usage_range=(5000, 8000), num_sub_events_range=(1500, 2500)):
    """
    Generates synthetic user, usage, and subscription event dataframes.
    Includes realistic patterns for subscription renewals, usage, and cancellations.
    """
    # Define reference dates
    global_analysis_date = pd.Timestamp.now() + pd.Timedelta(days=90) # Future date for analysis
    feature_cutoff_date = global_analysis_date - pd.Timedelta(days=90) # Look-back window end
    synthetic_data_end_date = feature_cutoff_date - pd.Timedelta(days=30) # Data generation mostly ends here
    signup_start_date = synthetic_data_end_date - pd.Timedelta(days=5*365) # 5 years back

    print(f"Synthetic Data Generation Parameters:")
    print(f"  Global Analysis Date: {global_analysis_date.date()}")
    print(f"  Feature Cutoff Date: {feature_cutoff_date.date()}")
    print(f"  Synthetic Data End Date: {synthetic_data_end_date.date()}")
    print(f"  Signup Start Date: {signup_start_date.date()}")

    # 1. Users DataFrame
    num_users = random.randint(*num_users_range)
    users_df = pd.DataFrame({
        'user_id': range(1, num_users + 1),
        'signup_date': pd.to_datetime(np.random.uniform(signup_start_date.timestamp(), synthetic_data_end_date.timestamp(), num_users), unit='s').date,
        'initial_subscription_plan': np.random.choice(['Starter', 'Pro', 'Enterprise'], num_users, p=[0.5, 0.4, 0.1]),
        'region': np.random.choice(['North', 'South', 'East', 'West'], num_users, p=[0.25, 0.25, 0.25, 0.25]),
        'has_opted_for_annual_billing': np.random.choice([0, 1], num_users, p=[0.6, 0.4])
    })
    users_df['signup_date'] = pd.to_datetime(users_df['signup_date'])
    print(f"Generated {len(users_df)} users.")

    # Prepare for efficient date sampling
    user_signup_dates = users_df.set_index('user_id')['signup_date'].dt.date.to_dict()

    # 2. Usage Logs DataFrame
    num_usage_logs = random.randint(*num_usage_range)
    usage_logs_data = {
        'log_id': range(1, num_usage_logs + 1),
        'user_id': np.random.choice(users_df['user_id'], num_usage_logs),
        'feature_accessed': np.random.choice(['Dashboard_View', 'Report_Gen', 'Data_Export', 'API_Access', 'Support_Chat'], num_usage_logs, p=[0.4, 0.2, 0.15, 0.15, 0.1]),
        'session_duration_minutes': np.random.uniform(1.0, 120.0, num_usage_logs),
        'log_date': [None] * num_usage_logs
    }
    usage_logs_df = pd.DataFrame(usage_logs_data)

    # Ensure log_date is after signup_date and before synthetic_data_end_date
    def get_random_date_after_signup(user_id, end_date):
        signup_dt = user_signup_dates[user_id]
        if signup_dt >= end_date.date(): # If signup is too recent, push it back slightly to allow logs
            signup_dt = end_date.date() - pd.Timedelta(days=1)
        return pd.to_datetime(np.random.uniform(pd.to_datetime(signup_dt).timestamp(), end_date.timestamp()))

    usage_logs_df['log_date'] = usage_logs_df['user_id'].apply(lambda uid: get_random_date_after_signup(uid, synthetic_data_end_date))
    print(f"Generated {len(usage_logs_df)} usage logs.")

    # 3. Subscription Events DataFrame
    num_sub_events = random.randint(*num_sub_events_range)
    event_types = ['Subscription_Start', 'Renewal_Success', 'Renewal_Failed_Payment', 'Cancellation', 'Upgrade', 'Downgrade']
    subscription_events_data = {
        'event_id': range(1, num_sub_events + 1),
        'user_id': np.random.choice(users_df['user_id'], num_sub_events),
        'event_type': np.random.choice(event_types, num_sub_events, p=[0.25, 0.3, 0.1, 0.1, 0.15, 0.1]),
        'plan_effected': np.random.choice(['Starter', 'Pro', 'Enterprise'], num_sub_events, p=[0.5, 0.4, 0.1]),
        'amount_charged': np.random.uniform(10.0, 500.0, num_sub_events),
        'event_date': [None] * num_sub_events
    }
    subscription_events_df = pd.DataFrame(subscription_events_data)

    # Ensure event_date is after signup_date and before synthetic_data_end_date
    subscription_events_df['event_date'] = subscription_events_df['user_id'].apply(lambda uid: get_random_date_after_signup(uid, synthetic_data_end_date))
    
    # Set amount_charged to NaN for non-charge events
    non_charge_events = ['Cancellation', 'Upgrade', 'Downgrade']
    subscription_events_df.loc[subscription_events_df['event_type'].isin(non_charge_events), 'amount_charged'] = np.nan

    # Initial subscription plan logic for 'Subscription_Start' events
    # Ensure the first 'Subscription_Start' event matches the initial_subscription_plan
    for user_id in users_df['user_id'].unique():
        user_events = subscription_events_df[subscription_events_df['user_id'] == user_id].sort_values('event_date')
        first_start_event = user_events[user_events['event_type'] == 'Subscription_Start']
        if not first_start_event.empty:
            initial_plan = users_df[users_df['user_id'] == user_id]['initial_subscription_plan'].iloc[0]
            subscription_events_df.loc[first_start_event.index[0], 'plan_effected'] = initial_plan
            # For simplicity, if the first event is not 'Subscription_Start', just make sure one exists
        else: # Add a Subscription_Start event if none exists for the user
            signup_dt = users_df[users_df['user_id'] == user_id]['signup_date'].iloc[0]
            initial_plan = users_df[users_df['user_id'] == user_id]['initial_subscription_plan'].iloc[0]
            new_event = {
                'event_id': subscription_events_df['event_id'].max() + 1 if not subscription_events_df.empty else 1,
                'user_id': user_id,
                'event_type': 'Subscription_Start',
                'plan_effected': initial_plan,
                'amount_charged': np.random.uniform(50.0, 300.0),
                'event_date': signup_dt + pd.Timedelta(days=random.randint(0, 30)) # Shortly after signup
            }
            subscription_events_df = pd.concat([subscription_events_df, pd.DataFrame([new_event])], ignore_index=True)
            subscription_events_df['event_id'] = range(1, len(subscription_events_df) + 1) # Re-index event_id

    # Simulate realistic patterns (remaining parts)
    # Bias Renewal_Success for annual billers (after synthetic_data_end_date, within the renewal window)
    renewal_candidates = users_df[users_df['has_opted_for_annual_billing'] == 1]['user_id'].tolist()
    
    # Introduce renewal success events *after* feature_cutoff_date, *before* global_analysis_date
    # These events ensure there are positive labels for the target variable
    num_injected_renewals = int(num_users * 0.2) # ~20% of users renew in this future window
    users_for_renewal_injection = random.sample(renewal_candidates, min(len(renewal_candidates), num_injected_renewals))

    injected_events = []
    for user_id in users_for_renewal_injection:
        last_event = subscription_events_df[(subscription_events_df['user_id'] == user_id) & (subscription_events_df['event_date'] <= synthetic_data_end_date)] \
                     .sort_values('event_date', ascending=False).head(1)
        if not last_event.empty and last_event['event_type'].iloc[0] not in ['Cancellation', 'Renewal_Failed_Payment']:
            event_date = pd.to_datetime(np.random.uniform(feature_cutoff_date.timestamp(), global_analysis_date.timestamp()))
            plan = last_event['plan_effected'].iloc[0] if not last_event.empty else users_df[users_df['user_id'] == user_id]['initial_subscription_plan'].iloc[0]
            amount = np.random.uniform(10.0, 500.0)
            injected_events.append({
                'event_id': 0, # Placeholder, will re-index
                'user_id': user_id,
                'event_type': 'Renewal_Success',
                'plan_effected': plan,
                'amount_charged': amount,
                'event_date': event_date
            })
    if injected_events:
        injected_df = pd.DataFrame(injected_events)
        subscription_events_df = pd.concat([subscription_events_df, injected_df], ignore_index=True)
        subscription_events_df['event_id'] = range(1, len(subscription_events_df) + 1) # Re-index event_id

    # Decline usage before Renewal_Failed_Payment or Cancellation
    for _, event_row in subscription_events_df[
        subscription_events_df['event_type'].isin(['Renewal_Failed_Payment', 'Cancellation']) &
        (subscription_events_df['event_date'] <= synthetic_data_end_date) # Only apply to historical events
    ].iterrows():
        user_id = event_row['user_id']
        event_date = event_row['event_date']
        
        # Identify usage logs 30-60 days leading up to the event
        start_period = event_date - pd.Timedelta(days=60)
        end_period = event_date - pd.Timedelta(days=30)
        
        relevant_logs_idx = usage_logs_df[
            (usage_logs_df['user_id'] == user_id) &
            (usage_logs_df['log_date'] >= start_period) &
            (usage_logs_df['log_date'] < end_period)
        ].index
        
        if not relevant_logs_idx.empty:
            # Reduce session duration for some logs
            usage_logs_df.loc[relevant_logs_idx, 'session_duration_minutes'] *= np.random.uniform(0.3, 0.8)
            # Potentially remove or change high-value features for a subset
            high_value_features = ['Report_Gen', 'Data_Export', 'API_Access']
            for hvf in high_value_features:
                hvf_logs_idx = usage_logs_df.loc[relevant_logs_idx][usage_logs_df['feature_accessed'] == hvf].index
                if not hvf_logs_idx.empty:
                    # Randomly change some of them to 'Dashboard_View' or reduce their count
                    num_to_change = int(len(hvf_logs_idx) * np.random.uniform(0.3, 0.7))
                    change_idx = np.random.choice(hvf_logs_idx, num_to_change, replace=False)
                    usage_logs_df.loc[change_idx, 'feature_accessed'] = 'Dashboard_View' # Downgrade feature access

    # 'Enterprise' plan users -> higher 'Data_Export' and 'API_Access'
    enterprise_users = users_df[users_df['initial_subscription_plan'] == 'Enterprise']['user_id'].tolist()
    if enterprise_users:
        enterprise_usage_logs_idx = usage_logs_df[
            (usage_logs_df['user_id'].isin(enterprise_users)) &
            (~usage_logs_df['feature_accessed'].isin(['Data_Export', 'API_Access'])) # Find non-high-value features
        ].index
        
        # For a portion of these, change to 'Data_Export' or 'API_Access'
        num_to_promote = int(len(enterprise_usage_logs_idx) * 0.15) # Promote 15% of other accesses
        if num_to_promote > 0:
            promote_idx = np.random.choice(enterprise_usage_logs_idx, num_to_promote, replace=False)
            usage_logs_df.loc[promote_idx, 'feature_accessed'] = np.random.choice(['Data_Export', 'API_Access'], num_to_promote)

    # Ensure log_date and event_date are never before signup_date, and within synthetic data range
    users_df_with_signup = users_df[['user_id', 'signup_date']].set_index('user_id')

    # Clip log dates
    usage_logs_df = usage_logs_df.set_index('user_id')
    usage_logs_df['signup_date_user'] = users_df_with_signup['signup_date']
    usage_logs_df['log_date'] = usage_logs_df.apply(
        lambda row: max(row['log_date'], row['signup_date_user'] + pd.Timedelta(days=1)), axis=1
    )
    usage_logs_df['log_date'] = usage_logs_df['log_date'].clip(upper=synthetic_data_end_date)
    usage_logs_df = usage_logs_df.drop(columns='signup_date_user').reset_index()

    # Clip event dates
    subscription_events_df = subscription_events_df.set_index('user_id')
    subscription_events_df['signup_date_user'] = users_df_with_signup['signup_date']
    subscription_events_df['event_date'] = subscription_events_df.apply(
        lambda row: max(row['event_date'], row['signup_date_user'] + pd.Timedelta(days=1)), axis=1
    )
    # Clip only historical events. Injected renewal events must remain in the future window.
    historical_event_mask = subscription_events_df['event_date'] <= synthetic_data_end_date
    subscription_events_df.loc[historical_event_mask, 'event_date'] = \
        subscription_events_df.loc[historical_event_mask, 'event_date'].clip(upper=synthetic_data_end_date)
    
    subscription_events_df = subscription_events_df.drop(columns='signup_date_user').reset_index()

    print(f"Generated {len(subscription_events_df)} subscription events (including injected renewals).")

    return users_df, usage_logs_df, subscription_events_df, global_analysis_date, feature_cutoff_date

def create_sqlite_db_and_features(users_df, usage_logs_df, subscription_events_df, feature_cutoff_date):
    """
    Creates an in-memory SQLite database, loads data, and performs SQL feature engineering.
    """
    conn = sqlite3.connect(':memory:')

    users_df.to_sql('users', conn, index=False, if_exists='replace')
    usage_logs_df.to_sql('usage_logs', conn, index=False, if_exists='replace')
    subscription_events_df.to_sql('subscription_events', conn, index=False, if_exists='replace')

    feature_cutoff_date_str = feature_cutoff_date.strftime('%Y-%m-%d %H:%M:%S')

    # SQL Query for feature engineering
    sql_query = f"""
    WITH UserUsage AS (
        SELECT
            user_id,
            SUM(session_duration_minutes) AS total_usage_duration_pre_cutoff,
            COUNT(log_id) AS num_usage_events_pre_cutoff,
            AVG(session_duration_minutes) AS avg_session_duration_pre_cutoff,
            SUM(CASE WHEN feature_accessed = 'API_Access' THEN 1 ELSE 0 END) AS num_api_access_events_pre_cutoff,
            MAX(log_date) AS last_log_date_pre_cutoff
        FROM usage_logs
        WHERE log_date <= '{feature_cutoff_date_str}'
        GROUP BY user_id
    ),
    UserSubscription AS (
        SELECT
            user_id,
            SUM(CASE WHEN event_type = 'Renewal_Failed_Payment' THEN 1 ELSE 0 END) AS num_renewal_failures_pre_cutoff,
            MAX(event_date) AS last_event_date_pre_cutoff
        FROM subscription_events
        WHERE event_date <= '{feature_cutoff_date_str}'
        GROUP BY user_id
    ),
    LatestSubscriptionPlan AS (
        SELECT
            user_id,
            plan_effected AS current_plan_at_cutoff
        FROM (
            SELECT
                user_id,
                plan_effected,
                event_date,
                ROW_NUMBER() OVER(PARTITION BY user_id ORDER BY event_date DESC, event_id DESC) as rn
            FROM subscription_events
            WHERE event_date <= '{feature_cutoff_date_str}'
        )
        WHERE rn = 1
    )
    SELECT
        u.user_id,
        u.signup_date,
        u.region,
        u.has_opted_for_annual_billing,
        u.initial_subscription_plan, -- Explicitly selected as per reviewer feedback
        COALESCE(lsp.current_plan_at_cutoff, u.initial_subscription_plan) AS current_plan_at_cutoff,
        COALESCE(uu.total_usage_duration_pre_cutoff, 0.0) AS total_usage_duration_pre_cutoff,
        COALESCE(uu.num_usage_events_pre_cutoff, 0) AS num_usage_events_pre_cutoff,
        COALESCE(uu.avg_session_duration_pre_cutoff, 0.0) AS avg_session_duration_pre_cutoff,
        COALESCE(uu.num_api_access_events_pre_cutoff, 0) AS num_api_access_events_pre_cutoff,
        COALESCE(us.num_renewal_failures_pre_cutoff, 0) AS num_renewal_failures_pre_cutoff,
        -- Calculate days_since_last_usage_pre_cutoff
        CASE
            WHEN uu.last_log_date_pre_cutoff IS NOT NULL
            THEN JULIANDAY('{feature_cutoff_date_str}') - JULIANDAY(uu.last_log_date_pre_cutoff)
            ELSE NULL
        END AS days_since_last_usage_pre_cutoff,
        -- Calculate days_since_last_subscription_event_pre_cutoff
        CASE
            WHEN us.last_event_date_pre_cutoff IS NOT NULL
            THEN JULIANDAY('{feature_cutoff_date_str}') - JULIANDAY(us.last_event_date_pre_cutoff)
            ELSE NULL
        END AS days_since_last_subscription_event_pre_cutoff
    FROM users AS u
    LEFT JOIN UserUsage AS uu ON u.user_id = uu.user_id
    LEFT JOIN UserSubscription AS us ON u.user_id = us.user_id
    LEFT JOIN LatestSubscriptionPlan AS lsp ON u.user_id = lsp.user_id;
    """
    user_renewal_features_df = pd.read_sql_query(sql_query, conn)
    conn.close()
    print(f"SQL feature engineering completed. Resulting DataFrame has {len(user_renewal_features_df)} rows.")
    return user_renewal_features_df

def pandas_feature_engineering_and_target_creation(user_renewal_features_df, subscription_events_df, feature_cutoff_date, global_analysis_date):
    """
    Performs additional Pandas feature engineering and creates the binary target variable.
    """
    # Convert dates to datetime objects
    user_renewal_features_df['signup_date'] = pd.to_datetime(user_renewal_features_df['signup_date'])
    subscription_events_df['event_date'] = pd.to_datetime(subscription_events_df['event_date'])

    # Handle NaN values for numerical features
    user_renewal_features_df['total_usage_duration_pre_cutoff'] = user_renewal_features_df['total_usage_duration_pre_cutoff'].fillna(0.0)
    user_renewal_features_df['num_usage_events_pre_cutoff'] = user_renewal_features_df['num_usage_events_pre_cutoff'].fillna(0)
    user_renewal_features_df['avg_session_duration_pre_cutoff'] = user_renewal_features_df['avg_session_duration_pre_cutoff'].fillna(0.0)
    user_renewal_features_df['num_api_access_events_pre_cutoff'] = user_renewal_features_df['num_api_access_events_pre_cutoff'].fillna(0)
    user_renewal_features_df['num_renewal_failures_pre_cutoff'] = user_renewal_features_df['num_renewal_failures_pre_cutoff'].fillna(0)

    # Fill NaN for days_since_last_usage/event with a large sentinel value
    sentinel_days = 9999
    user_renewal_features_df['days_since_last_usage_pre_cutoff'] = user_renewal_features_df['days_since_last_usage_pre_cutoff'].fillna(sentinel_days)
    user_renewal_features_df['days_since_last_subscription_event_pre_cutoff'] = user_renewal_features_df['days_since_last_subscription_event_pre_cutoff'].fillna(sentinel_days)

    # Calculate `account_age_at_cutoff_days`
    user_renewal_features_df['account_age_at_cutoff_days'] = (feature_cutoff_date - user_renewal_features_df['signup_date']).dt.days.clip(lower=0)

    # Calculate `usage_frequency_pre_cutoff`
    user_renewal_features_df['usage_frequency_pre_cutoff'] = user_renewal_features_df['num_usage_events_pre_cutoff'] / (user_renewal_features_df['account_age_at_cutoff_days'] + 1)

    # Calculate `avg_daily_duration_pre_cutoff`
    user_renewal_features_df['avg_daily_duration_pre_cutoff'] = user_renewal_features_df['total_usage_duration_pre_cutoff'] / (user_renewal_features_df['account_age_at_cutoff_days'] + 1)

    # Create the Binary Target `will_renew_in_next_90_days`
    renewal_window_start = feature_cutoff_date
    renewal_window_end = global_analysis_date

    # Determine user's subscription status at feature_cutoff_date
    latest_events_pre_cutoff = subscription_events_df[subscription_events_df['event_date'] <= feature_cutoff_date] \
        .sort_values(['user_id', 'event_date', 'event_id'], ascending=[True, False, False]) \
        .drop_duplicates('user_id')

    # Users are 'active' if their last event before cutoff was not a cancellation or failed payment
    active_at_cutoff_users = latest_events_pre_cutoff[
        ~latest_events_pre_cutoff['event_type'].isin(['Cancellation', 'Renewal_Failed_Payment'])
    ]['user_id'].unique()

    # Initialize target column
    user_renewal_features_df['will_renew_in_next_90_days'] = 0

    # For users active at cutoff, check for Renewal_Success in the next 90 days (renewal window)
    renewal_events_in_window = subscription_events_df[
        (subscription_events_df['event_date'] > renewal_window_start) &
        (subscription_events_df['event_date'] <= renewal_window_end) &
        (subscription_events_df['event_type'] == 'Renewal_Success')
    ]

    users_renewing_in_window = renewal_events_in_window['user_id'].unique()

    # A user renews (target = 1) if they were active at cutoff AND had a Renewal_Success event in the window
    renewing_users = np.intersect1d(active_at_cutoff_users, users_renewing_in_window)
    user_renewal_features_df.loc[user_renewal_features_df['user_id'].isin(renewing_users), 'will_renew_in_next_90_days'] = 1
    
    # Define features X and target y
    numerical_features = [
        'account_age_at_cutoff_days',
        'total_usage_duration_pre_cutoff',
        'num_usage_events_pre_cutoff',
        'avg_session_duration_pre_cutoff',
        'num_api_access_events_pre_cutoff',
        'num_renewal_failures_pre_cutoff',
        'days_since_last_usage_pre_cutoff',
        'days_since_last_subscription_event_pre_cutoff',
        'usage_frequency_pre_cutoff',
        'avg_daily_duration_pre_cutoff'
    ]
    categorical_features = [
        'region',
        'initial_subscription_plan',
        'has_opted_for_annual_billing',
        'current_plan_at_cutoff'
    ]

    # Combine all features
    all_features = numerical_features + categorical_features
    X = user_renewal_features_df[all_features]
    y = user_renewal_features_df['will_renew_in_next_90_days']

    print(f"\nPandas feature engineering completed.")
    print(f"Target variable 'will_renew_in_next_90_days' distribution:")
    print(y.value_counts(normalize=True))
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    print(f"\nData split into training ({len(X_train)} samples) and testing ({len(X_test)} samples).")
    
    return X_train, X_test, y_train, y_test, numerical_features, categorical_features, user_renewal_features_df

def visualize_data(user_renewal_features_df):
    """
    Creates two plots for data visualization.
    """
    print("\nGenerating data visualizations...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    plt.style.use('seaborn-v0_8-darkgrid')

    # Violin plot: avg_session_duration_pre_cutoff vs. will_renew_in_next_90_days
    sns.violinplot(
        x='will_renew_in_next_90_days',
        y='avg_session_duration_pre_cutoff',
        data=user_renewal_features_df,
        ax=axes[0],
        palette={0: 'salmon', 1: 'lightgreen'}
    )
    axes[0].set_title('Avg Session Duration Pre-Cutoff by Renewal Status')
    axes[0].set_xlabel('Will Renew in Next 90 Days')
    axes[0].set_ylabel('Average Session Duration (minutes)')
    axes[0].set_xticks([0, 1])
    axes[0].set_xticklabels(['No Renewal', 'Renewal'])

    # Stacked bar chart: proportion of renewals across different current_plan_at_cutoff values
    plan_renewal_counts = user_renewal_features_df.groupby(['current_plan_at_cutoff', 'will_renew_in_next_90_days']).size().unstack(fill_value=0)
    plan_renewal_proportions = plan_renewal_counts.apply(lambda x: x / x.sum(), axis=1)
    
    plan_renewal_proportions.plot(
        kind='bar',
        stacked=True,
        ax=axes[1],
        color=['salmon', 'lightgreen'],
        edgecolor='black'
    )
    axes[1].set_title('Renewal Proportion by Current Plan at Cutoff')
    axes[1].set_xlabel('Current Plan at Cutoff')
    axes[1].set_ylabel('Proportion')
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].legend(title='Will Renew', labels=['No Renewal', 'Renewal'])

    plt.tight_layout()
    plt.show()
    print("Visualizations displayed.")

def build_train_evaluate_ml_pipeline(X_train, X_test, y_train, y_test, numerical_features, categorical_features):
    """
    Builds, trains, and evaluates an ML pipeline for binary classification.
    """
    print("\nBuilding and training ML pipeline...")

    # Create preprocessing pipelines for numerical and categorical features
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Create a preprocessor using ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough' # Keep other columns if any, though not expected here
    )

    # Create the full pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', HistGradientBoostingClassifier(random_state=42))
    ])

    # Train the pipeline
    pipeline.fit(X_train, y_train)
    print("ML pipeline trained successfully.")

    # Predict probabilities on the test set
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1] # Probability of the positive class (1)

    # Evaluate the model
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    print(f"\nTest ROC AUC Score: {roc_auc:.4f}")

    # Predict class labels for classification report (using default threshold of 0.5)
    y_pred = pipeline.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("ML pipeline evaluation complete.")

if __name__ == "__main__":
    print("Starting the Data Science / ML Engineering pipeline...\n")

    # 1. Generate Synthetic Data
    users_df, usage_logs_df, subscription_events_df, global_analysis_date, feature_cutoff_date = generate_synthetic_data()

    # 2. Load into SQLite & SQL Feature Engineering
    user_renewal_features_df = create_sqlite_db_and_features(users_df, usage_logs_df, subscription_events_df, feature_cutoff_date)

    # 3. Pandas Feature Engineering & Binary Target Creation
    X_train, X_test, y_train, y_test, numerical_features, categorical_features, processed_df_for_viz = \
        pandas_feature_engineering_and_target_creation(user_renewal_features_df.copy(), subscription_events_df.copy(), feature_cutoff_date, global_analysis_date)
    
    # 4. Data Visualization
    visualize_data(processed_df_for_viz)

    # 5. ML Pipeline & Evaluation
    build_train_evaluate_ml_pipeline(X_train, X_test, y_train, y_test, numerical_features, categorical_features)

    print("\nData Science / ML Engineering pipeline completed.")