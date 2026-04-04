import pandas as pd
import numpy as np
import datetime
import sqlite3
import random
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer # Corrected import path from sklearn.preprocessing to sklearn.impute
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, classification_report

def generate_synthetic_data(num_subscribers_min, num_subscribers_max, num_usage_min, num_usage_max):
    """
    Generates synthetic subscriber and usage dataframes with biased renewal patterns.
    """
    num_subscribers = random.randint(num_subscribers_min, num_subscribers_max)

    # 1. Subscribers DataFrame
    subscriber_ids = np.arange(1, num_subscribers + 1)
    signup_dates = [datetime.date.today() - relativedelta(days=random.randint(30, 730)) for _ in range(num_subscribers)]
    plan_types = random.choices(['Basic', 'Standard', 'Premium'], weights=[0.4, 0.35, 0.25], k=num_subscribers)
    regions = random.choices(['North', 'South', 'East', 'West'], weights=[0.25, 0.25, 0.3, 0.2], k=num_subscribers)

    subscribers_df = pd.DataFrame({
        'subscriber_id': subscriber_ids,
        'signup_date': signup_dates,
        'plan_type': plan_types,
        'region': regions
    })

    # Simulate realistic renewal patterns (40-60% overall)
    # Bias `is_renewed` based on plan_type and region
    renewal_probabilities = []
    for i in range(num_subscribers):
        prob = 0.45 # Base probability
        if subscribers_df.loc[i, 'plan_type'] == 'Premium':
            prob += 0.20 # Premium users are more likely to renew
        elif subscribers_df.loc[i, 'plan_type'] == 'Standard':
            prob += 0.05
        if subscribers_df.loc[i, 'region'] == 'East':
            prob += 0.10 # Users from 'East' might have higher renewal rates
        elif subscribers_df.loc[i, 'region'] == 'South':
            prob -= 0.05 # Users from 'South' might have lower renewal rates
        renewal_probabilities.append(np.clip(prob, 0.1, 0.9)) # Clip probabilities to sensible range

    subscribers_df['is_renewed'] = [1 if random.random() < p else 0 for p in renewal_probabilities]

    # Adjust overall renewal rate to be within 40-60% if it drifts too much
    current_renewal_rate = subscribers_df['is_renewed'].mean()
    if not (0.4 <= current_renewal_rate <= 0.6):
        print(f"Initial renewal rate ({current_renewal_rate:.2f}) outside [0.4, 0.6] range. Adjusting...")
        num_target_renewed = int(num_subscribers * random.uniform(0.4, 0.6))
        
        # Sort by the initial calculated renewal probability to maintain bias, then assign `is_renewed`
        subscribers_df['renewal_prob'] = renewal_probabilities
        subscribers_df_sorted = subscribers_df.sort_values(by='renewal_prob', ascending=False).reset_index(drop=True)
        subscribers_df_sorted['is_renewed'] = 0 # Set all to 0 initially
        subscribers_df_sorted.loc[:num_target_renewed-1, 'is_renewed'] = 1 # Set top N to 1
        subscribers_df = subscribers_df_sorted.sample(frac=1, random_state=42).drop(columns='renewal_prob').reset_index(drop=True) # Shuffle back

    print(f"Generated {num_subscribers} subscribers with {subscribers_df['is_renewed'].mean():.2f} renewal rate.")

    # 2. Usage DataFrame
    usage_data = []
    usage_id_counter = 1
    num_total_usage_events_target = random.randint(num_usage_min, num_usage_max)

    # Define base activity types and weights
    activity_types = ['stream_content', 'download_item', 'support_chat', 'settings_change']
    base_activity_weights = {
        'stream_content': 0.5,
        'download_item': 0.2,
        'support_chat': 0.1,
        'settings_change': 0.2
    }

    # Generate usage events for each subscriber, biasing based on 'is_renewed' status
    for _, sub_row in subscribers_df.iterrows():
        subscriber_id = sub_row['subscriber_id']
        signup_date = sub_row['signup_date']
        is_renewed = sub_row['is_renewed']

        # Determine number of events for this subscriber based on renewal status
        # Renewed users typically have more events, especially in the first 30 days
        # Non-renewed users have fewer events overall and more support chats
        if is_renewed == 1:
            num_events_for_sub = random.randint(20, 80) # More events for renewed
            early_engagement_ratio = 0.8 # Higher probability of activity in first 30 days
            support_chat_prob_modifier = 0.3 # Less support chat for renewed
        else:
            num_events_for_sub = random.randint(5, 30) # Fewer events for non-renewed
            early_engagement_ratio = 0.3 # Lower probability of activity in first 30 days
            support_chat_prob_modifier = 2.0 # More support chat for non-renewed

        for _ in range(num_events_for_sub):
            if usage_id_counter > num_total_usage_events_target + 500: # Cap generation to avoid excessive size
                break

            # Distribute events over time, biased towards early engagement for renewed users
            if random.random() < early_engagement_ratio:
                days_after_signup = random.randint(0, 29) # Event within first 30 days
            else:
                days_after_signup = random.randint(30, 365) # Event after first 30 days, up to a year
            
            event_date = signup_date + relativedelta(days=days_after_signup)
            event_timestamp = datetime.datetime.combine(event_date, datetime.time(random.randint(0, 23), random.randint(0, 59), random.randint(0, 59)))

            # Adjust activity type weights based on renewal status
            activity_weights_adjusted = base_activity_weights.copy()
            if is_renewed == 1:
                activity_weights_adjusted['stream_content'] *= 1.2 # Higher stream duration
                activity_weights_adjusted['download_item'] *= 1.2 # More downloads
                activity_weights_adjusted['support_chat'] *= support_chat_prob_modifier # Reduced support chat
            else:
                activity_weights_adjusted['stream_content'] *= 0.8 # Lower stream duration
                activity_weights_adjusted['download_item'] *= 0.8 # Fewer downloads
                activity_weights_adjusted['support_chat'] *= support_chat_prob_modifier # Increased support chat
            
            # Normalize weights to sum to 1
            total_weight = sum(activity_weights_adjusted.values())
            normalized_weights = [v / total_weight for v in activity_weights_adjusted.values()]

            activity_type = random.choices(activity_types, weights=normalized_weights, k=1)[0]
            
            duration_minutes = 0
            if activity_type in ['stream_content', 'download_item']:
                duration_minutes = random.randint(1, 240) # Max 4 hours for streaming/downloading

            usage_data.append([usage_id_counter, subscriber_id, event_timestamp, activity_type, duration_minutes])
            usage_id_counter += 1
        
        if usage_id_counter > num_total_usage_events_target + 500:
            break

    usage_df = pd.DataFrame(usage_data, columns=['usage_id', 'subscriber_id', 'event_timestamp', 'activity_type', 'duration_minutes'])
    
    # Trim the usage_df to the target size if it grew too large
    if len(usage_df) > num_total_usage_events_target:
        usage_df = usage_df.sample(n=num_total_usage_events_target, random_state=42).reset_index(drop=True)
    
    usage_df['usage_id'] = np.arange(1, len(usage_df) + 1) # Ensure usage_ids are unique and sequential after trimming
    usage_df = usage_df.sort_values(by=['subscriber_id', 'event_timestamp']).reset_index(drop=True)

    print(f"Generated {len(usage_df)} usage events.")
    return subscribers_df, usage_df

def main():
    # 1. Generate Synthetic Data
    print("--- 1. Generating Synthetic Data ---")
    subscribers_df, usage_df = generate_synthetic_data(500, 700, 15000, 25000)

    # 2. Load into SQLite & SQL Feature Engineering
    print("\n--- 2. Loading into SQLite & SQL Feature Engineering ---")
    conn = sqlite3.connect(':memory:')
    subscribers_df.to_sql('subscribers', conn, index=False, if_exists='replace')
    usage_df.to_sql('usage', conn, index=False, if_exists='replace')

    sql_query = """
    SELECT
        s.subscriber_id,
        s.signup_date,
        s.plan_type,
        s.region,
        s.is_renewed,
        COALESCE(SUM(CASE WHEN u.event_timestamp IS NOT NULL THEN 1 ELSE 0 END), 0) AS num_activities_first_30d,
        COALESCE(SUM(CASE WHEN u.activity_type = 'stream_content' THEN u.duration_minutes ELSE 0 END), 0) AS total_stream_duration_first_30d,
        COALESCE(SUM(CASE WHEN u.activity_type = 'download_item' THEN 1 ELSE 0 END), 0) AS num_downloads_first_30d,
        COALESCE(SUM(CASE WHEN u.activity_type = 'support_chat' THEN 1 ELSE 0 END), 0) AS num_support_chats_first_30d,
        COALESCE(COUNT(DISTINCT STRFTIME('%Y-%m-%d', u.event_timestamp)), 0) AS days_with_activity_first_30d
    FROM
        subscribers s
    LEFT JOIN
        usage u ON s.subscriber_id = u.subscriber_id
               AND JULIANDAY(u.event_timestamp) >= JULIANDAY(s.signup_date)
               AND JULIANDAY(u.event_timestamp) <= JULIANDAY(DATE(s.signup_date, '+30 days'))
    GROUP BY
        s.subscriber_id, s.signup_date, s.plan_type, s.region, s.is_renewed
    ORDER BY
        s.subscriber_id;
    """

    subscriber_early_features_df = pd.read_sql_query(sql_query, conn)
    conn.close()
    print("SQL Feature Engineering complete. Resulting DataFrame head:")
    print(subscriber_early_features_df.head())
    print(f"Shape of subscriber_early_features_df: {subscriber_early_features_df.shape}")


    # 3. Pandas Feature Engineering & Binary Target Creation
    print("\n--- 3. Pandas Feature Engineering ---")
    # Handle NaN values: COALESCE in SQL should handle most, but ensure types are correct
    numerical_cols_to_fill = [
        'num_activities_first_30d',
        'total_stream_duration_first_30d',
        'num_downloads_first_30d',
        'num_support_chats_first_30d',
        'days_with_activity_first_30d'
    ]
    for col in numerical_cols_to_fill:
        if col in subscriber_early_features_df.columns:
            # Ensure these columns are numerical, fillna(0) for safety
            subscriber_early_features_df[col] = pd.to_numeric(subscriber_early_features_df[col], errors='coerce').fillna(0).astype(int)
    
    # Convert signup_date to datetime
    subscriber_early_features_df['signup_date'] = pd.to_datetime(subscriber_early_features_df['signup_date'])

    # Calculate activity_frequency_first_30d
    subscriber_early_features_df['activity_frequency_first_30d'] = subscriber_early_features_df['num_activities_first_30d'] / 30.0
    subscriber_early_features_df['activity_frequency_first_30d'] = subscriber_early_features_df['activity_frequency_first_30d'].fillna(0.0)

    # Calculate engagement_score_composite
    subscriber_early_features_df['engagement_score_composite'] = (
        subscriber_early_features_df['total_stream_duration_first_30d'] * 0.5 +
        subscriber_early_features_df['num_downloads_first_30d'] * 10 -
        subscriber_early_features_df['num_support_chats_first_30d'] * 20
    )

    # Define features X and target y
    numerical_features = [
        'num_activities_first_30d',
        'total_stream_duration_first_30d',
        'num_downloads_first_30d',
        'num_support_chats_first_30d',
        'days_with_activity_first_30d',
        'activity_frequency_first_30d',
        'engagement_score_composite'
    ]
    categorical_features = ['plan_type', 'region']
    target = 'is_renewed'

    X = subscriber_early_features_df[numerical_features + categorical_features]
    y = subscriber_early_features_df[target]

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    print(f"Data split into training ({len(X_train)} samples) and testing ({len(X_test)} samples).")
    print(f"Original renewal rate: {y.mean():.2f}, Train renewal rate: {y_train.mean():.2f}, Test renewal rate: {y_test.mean():.2f}")


    # 4. Data Visualization
    print("\n--- 4. Data Visualization ---")
    plt.figure(figsize=(14, 6))

    # Violin plot for total_stream_duration_first_30d vs is_renewed
    plt.subplot(1, 2, 1)
    sns.violinplot(x='is_renewed', y='total_stream_duration_first_30d', data=subscriber_early_features_df)
    plt.title('Stream Duration in First 30 Days by Renewal Status')
    plt.xlabel('Is Renewed (0=No, 1=Yes)')
    plt.ylabel('Total Stream Duration (minutes)')

    # Stacked bar chart for is_renewed proportions across plan_type
    plt.subplot(1, 2, 2)
    plan_renewal_props = subscriber_early_features_df.groupby('plan_type')['is_renewed'].value_counts(normalize=True).unstack()
    plan_renewal_props.plot(kind='bar', stacked=True, ax=plt.gca(), cmap='viridis')
    plt.title('Renewal Proportions by Plan Type')
    plt.xlabel('Plan Type')
    plt.ylabel('Proportion')
    plt.xticks(rotation=45)
    plt.legend(title='Is Renewed', labels=['No', 'Yes'])
    plt.tight_layout()
    plt.show()
    print("Visualizations displayed.")

    # 5. ML Pipeline & Evaluation
    print("\n--- 5. ML Pipeline & Evaluation ---")

    # Create preprocessing pipelines for numerical and categorical features
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Create a ColumnTransformer to apply different transformations to different columns
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop' # Drop any columns not specified
    )

    # Create the full pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', HistGradientBoostingClassifier(random_state=42))
    ])

    # Train the pipeline
    print("Training the ML pipeline...")
    pipeline.fit(X_train, y_train)
    print("Training complete.")

    # Predict probabilities for the positive class (class 1) on the test set
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred = pipeline.predict(X_test) # For classification report thresholds

    # Evaluate the model
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    print(f"\nTest Set ROC AUC Score: {roc_auc:.4f}")

    print("\nClassification Report (Test Set):")
    print(classification_report(y_test, y_pred))
    print("ML Pipeline and Evaluation complete.")

if __name__ == "__main__":
    main()