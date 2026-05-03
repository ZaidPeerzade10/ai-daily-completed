import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, classification_report

# Set a random seed for reproducibility
np.random.seed(42)

# --- 1. Synthetic Data Generation ---
NUM_USERS = 1000
MIN_REG_DATE = pd.to_datetime('2022-01-01')
MAX_OVERALL_ACTIVITY_DATE = pd.to_datetime('2023-11-01') # Ensures activity can go past cutoff for non-churners

# Define GLOBAL_PREDICTION_CUTOFF_DATE
GLOBAL_PREDICTION_CUTOFF_DATE = pd.to_datetime('2023-10-01')
PREDICTION_WINDOW_START = GLOBAL_PREDICTION_CUTOFF_DATE
PREDICTION_WINDOW_END = GLOBAL_PREDICTION_CUTOFF_DATE + pd.Timedelta(days=30)

# Generate users_df
users_data = {
    'user_id': range(1, NUM_USERS + 1),
    # Ensure registration date is at least 90 days before max activity date to allow for some history
    'registration_date': [MIN_REG_DATE + pd.Timedelta(days=np.random.randint(0, (MAX_OVERALL_ACTIVITY_DATE - MIN_REG_DATE).days - 90)) for _ in range(NUM_USERS)],
    'subscription_type': np.random.choice(['Free', 'Standard', 'Premium'], NUM_USERS, p=[0.5, 0.3, 0.2])
}
users_df = pd.DataFrame(users_data)

# Simulate churn dates with emphasis on the prediction window to avoid extreme imbalance
overall_churn_rate = 0.30 # Total percentage of users who will ever churn
churn_in_window_target_proportion = 0.12 # Target proportion of users churning *within* the 30-day prediction window

churn_dates = []
churners_in_window_count = 0
for i in range(NUM_USERS):
    reg_date = users_df.loc[i, 'registration_date']
    
    # First, attempt to place churners into the prediction window
    if np.random.rand() < churn_in_window_target_proportion:
        churn_offset = np.random.randint(0, 30) # Churn between cutoff and cutoff+29 days
        churn_date_candidate = PREDICTION_WINDOW_START + pd.Timedelta(days=churn_offset)
        
        # Ensure churn_date is after registration date
        if churn_date_candidate > reg_date:
            churn_dates.append(churn_date_candidate)
            churners_in_window_count += 1
            continue # Move to next user
    
    # For users not assigned to churn in the window, decide if they are general churners or non-churners
    if np.random.rand() < overall_churn_rate:
        # General churners (outside the prediction window)
        # Churn date either before cutoff or well after prediction window
        is_early_churner = np.random.rand() < 0.7 # More likely to churn before cutoff
        if is_early_churner:
            # Churn before the cutoff, but after registration
            churn_offset = np.random.randint(max(30, (GLOBAL_PREDICTION_CUTOFF_DATE - reg_date).days - 180), (GLOBAL_PREDICTION_CUTOFF_DATE - reg_date).days - 1)
            if churn_offset < 30: churn_offset = np.random.randint(30, 90) # Ensure at least 30 days after reg
            churn_date_candidate = reg_date + pd.Timedelta(days=churn_offset)
            if churn_date_candidate < reg_date: churn_date_candidate = reg_date + pd.Timedelta(days=30) # Fallback
            if churn_date_candidate < PREDICTION_WINDOW_START:
                 churn_dates.append(churn_date_candidate)
            else: # Fallback if churn date accidentally in window or too late
                churn_dates.append(pd.NaT) # Make them non-churners
        else:
            # Churn well after prediction window
            churn_date_candidate = PREDICTION_WINDOW_END + pd.Timedelta(days=np.random.randint(60, 365))
            if churn_date_candidate > reg_date:
                churn_dates.append(churn_date_candidate)
            else:
                churn_dates.append(pd.NaT) # Fallback if churn date before reg
    else:
        # Non-churner
        churn_dates.append(pd.NaT)

users_df['churn_date'] = churn_dates
users_df['churn_date'] = users_df['churn_date'].replace({pd.NaT: None}) # For SQLite compatibility later
users_df['registration_date'] = users_df['registration_date'].dt.date # Store as date for SQLite simplicity for initial load
# Convert back to datetime for pandas operations later
users_df['registration_date'] = pd.to_datetime(users_df['registration_date'])
users_df['churn_date'] = pd.to_datetime(users_df['churn_date'])


print(f"Total users: {NUM_USERS}")
print(f"Users churning specifically in prediction window: {churners_in_window_count}")
print(f"Proportion of users churning in prediction window: {churners_in_window_count/NUM_USERS:.2%}")


# Generate activity_df
activity_data = []
ACTIVITY_TYPES = ['login', 'page_view', 'support_ticket', 'feature_use']
ACTIVITY_FREQS = {'login': 0.7, 'page_view': 0.8, 'support_ticket': 0.05, 'feature_use': 0.5} # Relative freqs

for user_id in users_df['user_id']:
    user_row = users_df[users_df['user_id'] == user_id].iloc[0]
    reg_date = user_row['registration_date']
    churn_date = user_row['churn_date']

    # Define the period for activity generation
    # Max activity date for non-churners or users churning far in future
    end_activity_date = MAX_OVERALL_ACTIVITY_DATE 

    # Determine specific activity end and pre-churn drop period for this user
    pre_churn_activity_period = None
    if pd.notna(churn_date):
        # Activity significantly drops 7-20 days before churn
        pre_churn_activity_period = churn_date - pd.Timedelta(days=np.random.randint(7, 20)) 
        # Activity largely ceases shortly after churn date
        user_specific_activity_end = min(end_activity_date, churn_date + pd.Timedelta(days=np.random.randint(0, 5)))
    else:
        user_specific_activity_end = end_activity_date

    # Ensure activity generation starts after registration
    start_activity_date = reg_date

    current_date = start_activity_date
    while current_date < user_specific_activity_end:
        daily_activity_count = np.random.randint(1, 10) # Base activity

        # Adjust activity frequency based on churn proximity
        if pd.notna(churn_date) and pre_churn_activity_period and current_date > pre_churn_activity_period:
            daily_activity_count = np.random.randint(0, 2) # Significantly reduced activity before churn
        elif pd.notna(churn_date) and current_date >= churn_date:
            daily_activity_count = np.random.randint(0, 1) # Almost no activity after churn

        for _ in range(daily_activity_count):
            if current_date < MAX_OVERALL_ACTIVITY_DATE: # Cap activity timestamps
                activity_time = current_date + pd.Timedelta(seconds=np.random.randint(0, 86400))
                activity_type = np.random.choice(ACTIVITY_TYPES, p=list(ACTIVITY_FREQS.values())/np.sum(list(ACTIVITY_FREQS.values())))
                activity_data.append([user_id, activity_time, activity_type])
        current_date += pd.Timedelta(days=1)

activity_df = pd.DataFrame(activity_data, columns=['user_id', 'activity_timestamp', 'activity_type'])
activity_df = activity_df.sort_values(by=['user_id', 'activity_timestamp']).reset_index(drop=True)

# Convert all date/time columns to datetime objects for consistency
activity_df['activity_timestamp'] = pd.to_datetime(activity_df['activity_timestamp'])


print(f"\nGenerated {len(users_df)} users.")
print(f"Generated {len(activity_df)} activity records.")
print("\nUsers head:")
print(users_df.head())
print("\nActivity head:")
print(activity_df.head())

# --- 2. SQL Feature Engineering (SQLite) ---
conn = sqlite3.connect(':memory:')
users_df_sql = users_df.copy()
activity_df_sql = activity_df.copy()

# Convert datetime columns to string for SQLite to avoid potential issues with specific datetime formats
users_df_sql['registration_date'] = users_df_sql['registration_date'].dt.strftime('%Y-%m-%d %H:%M:%S')
users_df_sql['churn_date'] = users_df_sql['churn_date'].dt.strftime('%Y-%m-%d %H:%M:%S').replace({pd.NaT: None}) # handle NaT for SQLite
activity_df_sql['activity_timestamp'] = activity_df_sql['activity_timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')

users_df_sql.to_sql('users', conn, index=False, if_exists='replace')
activity_df_sql.to_sql('activity', conn, index=False, if_exists='replace')

# Define cutoff date for SQL query
cutoff_str = GLOBAL_PREDICTION_CUTOFF_DATE.strftime('%Y-%m-%d %H:%M:%S')
thirty_days_ago_str = (GLOBAL_PREDICTION_CUTOFF_DATE - pd.Timedelta(days=30)).strftime('%Y-%m-%d %H:%M:%S')

sql_query = f"""
WITH user_activity_summary AS (
    SELECT
        a.user_id,
        MAX(a.activity_timestamp) AS last_activity_overall,
        SUM(CASE WHEN a.activity_timestamp >= '{thirty_days_ago_str}' AND a.activity_timestamp < '{cutoff_str}' THEN 1 ELSE 0 END) AS total_activity_count_prev_30d,
        SUM(CASE WHEN a.activity_timestamp >= '{thirty_days_ago_str}' AND a.activity_timestamp < '{cutoff_str}' AND a.activity_type = 'login' THEN 1 ELSE 0 END) AS num_logins_prev_30d,
        SUM(CASE WHEN a.activity_timestamp >= '{thirty_days_ago_str}' AND a.activity_timestamp < '{cutoff_str}' AND a.activity_type = 'support_ticket' THEN 1 ELSE 0 END) AS num_support_tickets_prev_30d
    FROM activity AS a
    WHERE a.activity_timestamp <= '{cutoff_str}' -- Only consider activity up to cutoff date
    GROUP BY a.user_id
)
SELECT
    u.user_id,
    u.registration_date,
    u.subscription_type,
    u.churn_date,
    COALESCE(uas.num_logins_prev_30d, 0) AS num_logins_prev_30d,
    COALESCE(uas.num_support_tickets_prev_30d, 0) AS num_support_tickets_prev_30d,
    COALESCE(uas.total_activity_count_prev_30d, 0) AS total_activity_count_prev_30d,
    -- Calculate days since last activity at cutoff. If no activity ever, use a large number.
    CASE
        WHEN uas.last_activity_overall IS NULL THEN 9999 -- Large number for users with no activity ever
        ELSE JULIANDAY('{cutoff_str}') - JULIANDAY(uas.last_activity_overall)
    END AS days_since_last_activity_at_cutoff
FROM users AS u
LEFT JOIN user_activity_summary AS uas ON u.user_id = uas.user_id;
"""

features_df = pd.read_sql_query(sql_query, conn)
conn.close()

# Convert date columns back to datetime objects for pandas operations
features_df['registration_date'] = pd.to_datetime(features_df['registration_date'])
features_df['churn_date'] = pd.to_datetime(features_df['churn_date'])

print("\nFeatures DataFrame head after SQL query:")
print(features_df.head())
print(f"Features DataFrame shape: {features_df.shape}")


# --- 3. Pandas Feature Engineering, Target Creation & Data Split ---

# Engineer additional features
features_df['user_age_at_cutoff_days'] = (GLOBAL_PREDICTION_CUTOFF_DATE - features_df['registration_date']).dt.days
features_df['activity_frequency_prev_30d'] = features_df['total_activity_count_prev_30d'] / 30.0 # Normalized daily frequency

# Create the binary target variable
# Churn if churn_date falls within [GLOBAL_PREDICTION_CUTOFF_DATE, GLOBAL_PREDICTION_CUTOFF_DATE + 30 days)
features_df['will_churn_in_next_30_days'] = (
    (features_df['churn_date'] >= PREDICTION_WINDOW_START) &
    (features_df['churn_date'] < PREDICTION_WINDOW_END)
).astype(int) # Convert boolean to 0/1 integer

# Prepare features (X) and target (y)
X = features_df.drop(columns=['user_id', 'registration_date', 'churn_date', 'will_churn_in_next_30_days'])
y = features_df['will_churn_in_next_30_days']

print("\nTarget variable distribution:")
print(y.value_counts())
print(f"Churn rate in prediction window: {y.mean():.2%}")


# Stratified train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

print(f"\nX_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
print(f"y_train churn distribution:\n{y_train.value_counts(normalize=True)}")
print(f"y_test churn distribution:\n{y_test.value_counts(normalize=True)}")


# --- 4. Data Visualization (Matplotlib/Seaborn) ---
plt.style.use('seaborn-v0_8-darkgrid')
plt.figure(figsize=(16, 6))

# Violin plot for days_since_last_activity_at_cutoff
plt.subplot(1, 2, 1)
sns.violinplot(
    x='will_churn_in_next_30_days',
    y='days_since_last_activity_at_cutoff',
    data=features_df,
    palette='viridis'
)
plt.title('Days Since Last Activity at Cutoff by Churn Status')
plt.xlabel('Will Churn in Next 30 Days (0=No, 1=Yes)')
plt.ylabel('Days Since Last Activity')
plt.xticks([0, 1], ['No Churn', 'Will Churn'])

# Stacked bar chart for churn proportion across subscription_type
plt.subplot(1, 2, 2)
churn_by_subscription = features_df.groupby('subscription_type')['will_churn_in_next_30_days'].value_counts(normalize=True).unstack().fillna(0)
churn_by_subscription = churn_by_subscription.rename(columns={0: 'No Churn', 1: 'Will Churn'})
churn_by_subscription.plot(kind='bar', stacked=True, ax=plt.gca(), cmap='coolwarm')
plt.title('Churn Proportion by Subscription Type')
plt.xlabel('Subscription Type')
plt.ylabel('Proportion')
plt.xticks(rotation=45)
plt.legend(title='Churn Status')

plt.tight_layout()
plt.show()


# --- 5. ML Pipeline & Evaluation (Scikit-learn) ---

# Define numerical and categorical features
numerical_features = ['num_logins_prev_30d', 'num_support_tickets_prev_30d',
                      'total_activity_count_prev_30d', 'days_since_last_activity_at_cutoff',
                      'user_age_at_cutoff_days', 'activity_frequency_prev_30d']
categorical_features = ['subscription_type']

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

# Create the full machine learning pipeline
# HistGradientBoostingClassifier handles missing values internally, so SimpleImputer might be redundant for it,
# but it's good practice for other models. Scaling is still beneficial.
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('classifier', HistGradientBoostingClassifier(random_state=42))])

# Train the model
print("\nTraining the HistGradientBoostingClassifier pipeline...")
pipeline.fit(X_train, y_train)
print("Training complete.")

# Predict probabilities on the test set
y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
y_pred = pipeline.predict(X_test)

# Evaluate performance
roc_auc = roc_auc_score(y_test, y_pred_proba)
# zero_division=0 handles cases where a class has no predictions or true instances, preventing warnings
report = classification_report(y_test, y_pred, target_names=['No Churn', 'Churn'], zero_division=0) 

print(f"\n--- Model Evaluation ---")
print(f"ROC AUC Score: {roc_auc:.4f}")
print("\nClassification Report:")
print(report)