import pandas as pd
import numpy as np
import sqlite3
from datetime import timedelta, datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, classification_report

# --- 1. Generate Synthetic Data ---
np.random.seed(42)

# Users DataFrame
num_users = np.random.randint(700, 1001)
users_df = pd.DataFrame({
    'user_id': range(1, num_users + 1),
    'signup_date': pd.to_datetime(np.random.choice(pd.date_range(end=datetime.now() - timedelta(days=90), periods=365*2), size=num_users)),
    'device_type': np.random.choice(['iOS', 'Android', 'Web'], size=num_users, p=[0.4, 0.4, 0.2]),
    'region': np.random.choice(['North', 'South', 'East', 'West'], size=num_users, p=[0.25, 0.25, 0.25, 0.25])
})

# Simulate engagement potential (latent variable to bias sessions)
users_df['engagement_potential'] = np.random.rand(num_users) # Base potential
# Boost potential for certain device types and regions
users_df.loc[users_df['device_type'] == 'iOS', 'engagement_potential'] *= 1.2
users_df.loc[users_df['region'] == 'West', 'engagement_potential'] *= 1.1
users_df['engagement_potential'] = np.clip(users_df['engagement_potential'], 0.1, 1.5) # Clip to reasonable range

# Sessions DataFrame
num_sessions = np.random.randint(20000, 30001)
sessions_data = []

# Distribute sessions among users based on their engagement potential
user_weights = users_df['engagement_potential'] / users_df['engagement_potential'].sum()
user_ids_for_sessions = np.random.choice(users_df['user_id'], size=num_sessions, p=user_weights)

for i in range(num_sessions):
    user_id = user_ids_for_sessions[i]
    user_info = users_df[users_df['user_id'] == user_id].iloc[0]
    signup_date = user_info['signup_date']
    engagement_potential = user_info['engagement_potential']

    # Session start time after signup date, within a reasonable future window (e.g., up to 180 days after signup)
    session_start_time = signup_date + timedelta(days=np.random.randint(0, 180)) + timedelta(hours=np.random.randint(0, 24), minutes=np.random.randint(0, 60))

    # Bias session duration and features interacted based on engagement potential
    base_duration = np.random.randint(1, 60)
    # Higher potential leads to longer sessions, min 1, max 90
    session_duration_minutes = int(base_duration * (0.8 + engagement_potential * 0.7)) 
    session_duration_minutes = np.clip(session_duration_minutes, 1, 90)

    base_features = np.random.randint(0, 10)
    # Higher potential leads to more features interacted, min 0, max 15
    num_features_interacted = int(base_features * (0.8 + engagement_potential * 0.7))
    num_features_interacted = np.clip(num_features_interacted, 0, 15)

    sessions_data.append({
        'session_id': i + 1,
        'user_id': user_id,
        'session_start_time': session_start_time,
        'session_duration_minutes': session_duration_minutes,
        'num_features_interacted': num_features_interacted
    })

sessions_df = pd.DataFrame(sessions_data)

# Ensure session_start_time is always after signup_date (filter out any rare edge cases from random generation)
# Merge signup_date into sessions_df temporarily for filtering, then drop it.
sessions_df = pd.merge(sessions_df, users_df[['user_id', 'signup_date']], on='user_id', how='left', suffixes=('', '_user_signup_temp'))
sessions_df = sessions_df[sessions_df['session_start_time'] >= sessions_df['signup_date_user_signup_temp']]
sessions_df = sessions_df.drop(columns=['signup_date_user_signup_temp'])
sessions_df.reset_index(drop=True, inplace=True)

# Sort sessions_df by user_id then session_start_time
sessions_df = sessions_df.sort_values(by=['user_id', 'session_start_time']).reset_index(drop=True)

print("--- Synthetic Data Generation Complete ---")
print(f"Generated {len(users_df)} users and {len(sessions_df)} sessions.")
print("Users DataFrame head:\n", users_df.head())
print("Sessions DataFrame head:\n", sessions_df.head())


# --- 2. Load into SQLite & SQL Feature Engineering (First 7 Days) ---
conn = sqlite3.connect(':memory:')

# Load DataFrames into SQLite tables
users_df.to_sql('users', conn, if_exists='replace', index=False)
# Specify 'TIMESTAMP' for session_start_time to ensure proper date arithmetic in SQLite
sessions_df.to_sql('sessions', conn, if_exists='replace', index=False,
                   dtype={'session_start_time': 'TIMESTAMP'})

sql_query = """
SELECT
    u.user_id,
    u.signup_date,
    u.device_type,
    u.region,
    COALESCE(COUNT(s.session_id), 0) AS num_sessions_first_7d,
    COALESCE(SUM(s.session_duration_minutes), 0) AS total_duration_first_7d,
    COALESCE(AVG(s.session_duration_minutes), 0.0) AS avg_session_duration_first_7d,
    COALESCE(COUNT(DISTINCT STRFTIME('%Y-%m-%d', s.session_start_time)), 0) AS num_unique_days_active_first_7d,
    COALESCE(AVG(s.num_features_interacted), 0.0) AS avg_features_per_session_first_7d
FROM
    users AS u
LEFT JOIN
    sessions AS s ON u.user_id = s.user_id
    AND s.session_start_time <= DATE(u.signup_date, '+7 days')
GROUP BY
    u.user_id, u.signup_date, u.device_type, u.region
ORDER BY
    u.user_id;
"""

user_early_features_df = pd.read_sql_query(sql_query, conn)
conn.close()

print("\n--- SQL Feature Engineering (First 7 Days) Complete ---")
print("User Early Features DataFrame head:\n", user_early_features_df.head())


# --- 3. Pandas Feature Engineering & Binary Target Creation (Future Engagement) ---

# Handle NaN values: COALESCE in SQL should mostly prevent NaNs for aggregates,
# but ensure types and default values for robustness.
user_early_features_df['num_sessions_first_7d'] = user_early_features_df['num_sessions_first_7d'].fillna(0).astype(int)
user_early_features_df['total_duration_first_7d'] = user_early_features_df['total_duration_first_7d'].fillna(0).astype(int)
user_early_features_df['num_unique_days_active_first_7d'] = user_early_features_df['num_unique_days_active_first_7d'].fillna(0).astype(int)
user_early_features_df['avg_session_duration_first_7d'] = user_early_features_df['avg_session_duration_first_7d'].fillna(0.0)
user_early_features_df['avg_features_per_session_first_7d'] = user_early_features_df['avg_features_per_session_first_7d'].fillna(0.0)

# Convert signup_date to datetime objects
user_early_features_df['signup_date'] = pd.to_datetime(user_early_features_df['signup_date'])

# Calculate session_frequency_first_7d
user_early_features_df['session_frequency_first_7d'] = user_early_features_df['num_sessions_first_7d'] / 7.0
user_early_features_df['session_frequency_first_7d'] = user_early_features_df['session_frequency_first_7d'].fillna(0.0)

# Create the Binary Target `is_high_engagement_next_30d`
# Ensure session_start_time in original sessions_df is datetime
sessions_df['session_start_time'] = pd.to_datetime(sessions_df['session_start_time'])

# Prepare data for target creation: merge signup_date into sessions_df for calculations
sessions_with_signup = pd.merge(sessions_df, users_df[['user_id', 'signup_date']], on='user_id', how='left')

# Define the 30-day future engagement window (from day 7 to day 37 after signup)
sessions_with_signup['early_behavior_cutoff_date'] = sessions_with_signup['signup_date'] + timedelta(days=7)
sessions_with_signup['future_engagement_cutoff_date'] = sessions_with_signup['signup_date'] + timedelta(days=37)

# Filter sessions for the future engagement window
future_sessions = sessions_with_signup[
    (sessions_with_signup['session_start_time'] > sessions_with_signup['early_behavior_cutoff_date']) &
    (sessions_with_signup['session_start_time'] <= sessions_with_signup['future_engagement_cutoff_date'])
]

# Aggregate total duration for future sessions for each user
total_duration_next_30d = future_sessions.groupby('user_id')['session_duration_minutes'].sum().reset_index()
total_duration_next_30d.rename(columns={'session_duration_minutes': 'total_duration_next_30d'}, inplace=True)

# Merge this aggregate with user_early_features_df
user_early_features_df = pd.merge(user_early_features_df, total_duration_next_30d, on='user_id', how='left')
user_early_features_df['total_duration_next_30d'] = user_early_features_df['total_duration_next_30d'].fillna(0)

# Define 'High Engagement' users based on 75th percentile of NON-ZERO future engagement
non_zero_durations = user_early_features_df[user_early_features_df['total_duration_next_30d'] > 0]['total_duration_next_30d']
if not non_zero_durations.empty:
    engagement_threshold = non_zero_durations.quantile(0.75)
else:
    engagement_threshold = 0 # Default if no future engagement observed for any user

user_early_features_df['is_high_engagement_next_30d'] = (user_early_features_df['total_duration_next_30d'] > engagement_threshold).astype(int)

# Define features X and target y
numerical_features = [
    'num_sessions_first_7d', 'total_duration_first_7d', 'avg_session_duration_first_7d',
    'num_unique_days_active_first_7d', 'avg_features_per_session_first_7d',
    'session_frequency_first_7d'
]
categorical_features = ['device_type', 'region']

X = user_early_features_df[numerical_features + categorical_features]
y = user_early_features_df['is_high_engagement_next_30d']

# Split into training and testing sets (70/30 split, stratified on y for class balance)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

print("\n--- Pandas Feature Engineering & Binary Target Creation Complete ---")
print("User Early Features DataFrame with Target head:\n", user_early_features_df.head())
print(f"\nTarget (is_high_engagement_next_30d) distribution:\n{y.value_counts(normalize=True)}")
print(f"Engagement threshold for defining high engagement: {engagement_threshold:.2f} total minutes")
print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")


# --- 4. Data Visualization ---
print("\n--- Generating Data Visualizations ---")

# Violin plot: total_duration_first_7d vs. is_high_engagement_next_30d
plt.figure(figsize=(10, 6))
sns.violinplot(x='is_high_engagement_next_30d', y='total_duration_first_7d', data=user_early_features_df)
plt.title('Distribution of Total Duration in First 7 Days by Future Engagement', fontsize=14)
plt.xlabel('Is High Engagement Next 30 Days (0: Low/Medium, 1: High)', fontsize=12)
plt.ylabel('Total Duration in First 7 Days (minutes)', fontsize=12)
plt.xticks([0, 1], ['Low/Medium Engagement', 'High Engagement'], fontsize=10)
plt.yticks(fontsize=10)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Stacked bar chart: proportion of engagement by device_type
device_engagement_pivot = user_early_features_df.groupby('device_type')['is_high_engagement_next_30d'].value_counts(normalize=True).unstack().fillna(0)
device_engagement_pivot.plot(kind='bar', stacked=True, figsize=(10, 6), colormap='viridis')
plt.title('Proportion of High vs. Low/Medium Engagement by Device Type', fontsize=14)
plt.xlabel('Device Type', fontsize=12)
plt.ylabel('Proportion', fontsize=12)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(fontsize=10)
plt.legend(title='Is High Engagement Next 30 Days', labels=['Low/Medium', 'High'], loc='upper left', bbox_to_anchor=(1,1))
plt.tight_layout()
plt.show()


# --- 5. ML Pipeline & Evaluation (Binary Classification) ---
print("\n--- Building and Evaluating ML Pipeline ---")

# Preprocessing steps for numerical and categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='mean')), # Handle potential NaNs in numerical features
            ('scaler', StandardScaler())                 # Scale numerical features
        ]), numerical_features),
        ('cat', Pipeline([
            ('onehot', OneHotEncoder(handle_unknown='ignore')) # One-hot encode categorical features
        ]), categorical_features)
    ])

# Create the ML pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', HistGradientBoostingClassifier(random_state=42)) # Using HistGradientBoostingClassifier
])

# Train the pipeline on the training data
pipeline.fit(X_train, y_train)

# Predict probabilities for the positive class (class 1) on the test set
y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

# Predict class labels on the test set
y_pred = pipeline.predict(X_test)

# Evaluate the model
roc_auc = roc_auc_score(y_test, y_pred_proba)
classification_rep = classification_report(y_test, y_pred)

print(f"\nROC AUC Score on Test Set: {roc_auc:.4f}")
print("\nClassification Report on Test Set:")
print(classification_rep)

print("\n--- Script Finished Successfully ---")