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
import warnings

# Suppress warnings for cleaner output in production-like scripts
warnings.filterwarnings('ignore')

# --- 1. Generate Synthetic Data ---

print("--- Generating Synthetic Data ---")

# Define today's date for relative date generation
TODAY = pd.Timestamp.now().normalize()

# Define New Feature Launch Date (make it recent enough for adoption window to be relevant)
# This places the launch date 3 to 6 months in the past relative to TODAY
NEW_FEATURE_LAUNCH_DATE = TODAY - pd.Timedelta(days=np.random.randint(90, 180))
adoption_window_days = 60
ADOPTION_END_DATE = NEW_FEATURE_LAUNCH_DATE + pd.Timedelta(days=adoption_window_days)

# --- users_df ---
N_USERS = np.random.randint(800, 1201)
print(f"Generating {N_USERS} users...")

user_ids = np.arange(1, N_USERS + 1)
# Sign-up dates over the last 3 years, up to today
signup_dates = [TODAY - pd.Timedelta(days=np.random.randint(0, 3*365)) for _ in range(N_USERS)]

segments = np.random.choice(['New User', 'Explorer', 'Power User'], size=N_USERS, p=[0.3, 0.4, 0.3])
device_types = np.random.choice(['Mobile', 'Desktop', 'Tablet'], size=N_USERS, p=[0.5, 0.4, 0.1])
ages = np.random.randint(18, 66, size=N_USERS) # 18-65

previous_feature_engagement_scores = np.zeros(N_USERS)
for i in range(N_USERS):
    if segments[i] == 'Power User':
        previous_feature_engagement_scores[i] = np.random.uniform(5.0, 10.0) # Higher for Power Users
    else:
        previous_feature_engagement_scores[i] = np.random.uniform(0.0, 7.0) # Lower for others

users_df = pd.DataFrame({
    'user_id': user_ids,
    'signup_date': signup_dates,
    'segment': segments,
    'device_type': device_types,
    'age': ages,
    'previous_feature_engagement_score': previous_feature_engagement_scores
})

# Convert signup_date to datetime objects for consistency
users_df['signup_date'] = pd.to_datetime(users_df['signup_date'])


# --- feature_events_df ---
# The total number of events will be dynamic based on the generation loop,
# ensuring biases are applied. We'll set a rough target.
N_EVENTS_TARGET = np.random.randint(20000, 30001) 
print(f"Targeting approx. {N_EVENTS_TARGET} events...")

feature_names = ['Search', 'Upload', 'Share', 'Settings']
new_feature_name = 'New_Analytics_Dashboard'

event_timestamps = []
event_feature_names = []
event_durations = []
actual_event_user_ids = []

# Map user attributes for efficient access
user_attr_map = users_df.set_index('user_id').to_dict('index')

# Weighted event generation for *existing* features
for user_id in users_df['user_id']:
    user_data = user_attr_map[user_id]
    
    signup_date = user_data['signup_date']
    engagement_score = user_data['previous_feature_engagement_score']
    device_type = user_data['device_type']
    
    # Base number of events per user, adjusted by engagement/device
    num_user_events_base = int(np.random.normal(N_EVENTS_TARGET / N_USERS * 0.7, 5)) 
    
    if engagement_score > 7:
        num_user_events_base = int(num_user_events_base * 1.5) # High engagement users get more events
    if device_type == 'Desktop':
        num_user_events_base = int(num_user_events_base * 1.2) # Desktop users also get more events
    
    num_user_events_base = max(num_user_events_base, 2) # Ensure at least a few events

    for _ in range(num_user_events_base):
        # Event timestamp must be after signup date and before TODAY
        time_diff = (TODAY - signup_date).total_seconds()
        if time_diff <= 0: # If signup_date is today or in the future (unlikely with current gen but good check)
            continue
        
        random_seconds_after_signup = np.random.randint(1, time_diff + 1)
        timestamp = signup_date + pd.Timedelta(seconds=random_seconds_after_signup)
        
        # Bias feature names
        if device_type == 'Mobile':
            feature = np.random.choice(feature_names, p=[0.5, 0.2, 0.15, 0.15]) # More search for mobile
        else:
            feature = np.random.choice(feature_names, p=[0.3, 0.3, 0.2, 0.2]) # Balanced for others

        duration = np.random.randint(0, 601) # 0-600 seconds
        
        event_timestamps.append(timestamp)
        event_feature_names.append(feature)
        event_durations.append(duration)
        actual_event_user_ids.append(user_id)

# Now, add events for the New_Analytics_Dashboard, explicitly ensuring a healthy proportion of adopters
adopter_ratio = 0.25 # Target 25% of users to adopt
num_adopters = int(N_USERS * adopter_ratio)
adopter_user_ids = np.random.choice(users_df['user_id'], size=num_adopters, replace=False)

print(f"Targeting {num_adopters} users for adoption within {adoption_window_days} days of launch ({NEW_FEATURE_LAUNCH_DATE.strftime('%Y-%m-%d')} to {ADOPTION_END_DATE.strftime('%Y-%m-%d')}).")

for user_id in users_df['user_id']:
    user_data = user_attr_map[user_id]
    signup_date = user_data['signup_date']
    engagement_score = user_data['previous_feature_engagement_score']
    device_type = user_data['device_type']

    # Determine valid start date for new feature events (after signup AND after launch)
    valid_new_feature_start = max(signup_date, NEW_FEATURE_LAUNCH_DATE)
    
    # Adopters: Ensure at least one 'New_Analytics_Dashboard' event within the adoption window
    if user_id in adopter_user_ids:
        # Check if the user's signup date allows them to adopt within the window
        if valid_new_feature_start < ADOPTION_END_DATE:
            event_time_start = valid_new_feature_start
            event_time_end = ADOPTION_END_DATE
            
            # Generate at least one event within this specific adoption window
            if (event_time_end - event_time_start).total_seconds() > 0:
                timestamp_delta_seconds = np.random.randint(0, (event_time_end - event_time_start).total_seconds() + 1)
                timestamp = event_time_start + pd.Timedelta(seconds=timestamp_delta_seconds)
                
                event_timestamps.append(timestamp)
                event_feature_names.append(new_feature_name)
                event_durations.append(np.random.randint(0, 601))
                actual_event_user_ids.append(user_id)

    # Generate additional 'New_Analytics_Dashboard' events for all users (including non-adopters) 
    # after the launch date, up to TODAY.
    # Bias: Desktop users with higher engagement get more new feature events.
    base_new_feature_events_count = 0
    if device_type == 'Desktop' and engagement_score > 7:
        base_new_feature_events_count = np.random.randint(2, 6)
    elif device_type == 'Desktop' or engagement_score > 5:
        base_new_feature_events_count = np.random.randint(0, 3)

    for _ in range(base_new_feature_events_count):
        event_time_start = valid_new_feature_start
        event_time_end = TODAY # New feature events can happen up to today

        if (event_time_end - event_time_start).total_seconds() > 0:
            timestamp_delta_seconds = np.random.randint(0, (event_time_end - event_time_start).total_seconds() + 1)
            timestamp = event_time_start + pd.Timedelta(seconds=timestamp_delta_seconds)
            
            event_timestamps.append(timestamp)
            event_feature_names.append(new_feature_name)
            event_durations.append(np.random.randint(0, 601))
            actual_event_user_ids.append(user_id)

# Consolidate feature events into a DataFrame
feature_events_df = pd.DataFrame({
    'event_id': np.arange(1, len(event_timestamps) + 1),
    'user_id': actual_event_user_ids,
    'event_timestamp': event_timestamps,
    'feature_name': event_feature_names,
    'duration_seconds': event_durations
})

# Sort feature_events_df
feature_events_df = feature_events_df.sort_values(by=['user_id', 'event_timestamp']).reset_index(drop=True)

print(f"Generated {len(feature_events_df)} feature events.")
print(f"Users DataFrame head:\n{users_df.head()}")
print(f"\nFeature Events DataFrame head:\n{feature_events_df.head()}")
print(f"\nNew Feature Launch Date: {NEW_FEATURE_LAUNCH_DATE.strftime('%Y-%m-%d')}")
print(f"Adoption Window End Date: {ADOPTION_END_DATE.strftime('%Y-%m-%d')}")


# --- 2. Load into SQLite & SQL Feature Engineering ---

print("\n--- Loading Data into SQLite and Performing SQL Feature Engineering ---")

conn = sqlite3.connect(':memory:')
users_df.to_sql('users', conn, index=False, if_exists='replace')
feature_events_df.to_sql('feature_events', conn, index=False, if_exists='replace')

sql_query = f"""
SELECT
    u.user_id,
    u.signup_date,
    u.segment,
    u.device_type,
    u.age,
    u.previous_feature_engagement_score,
    COALESCE(COUNT(fe.event_id), 0) AS num_events_first_14d,
    COALESCE(SUM(fe.duration_seconds), 0) AS total_duration_first_14d,
    COALESCE(COUNT(DISTINCT fe.feature_name), 0) AS num_unique_features_first_14d
FROM
    users AS u
LEFT JOIN
    feature_events AS fe ON u.user_id = fe.user_id
    AND julianday(fe.event_timestamp) <= julianday(u.signup_date) + 14
    AND fe.feature_name != '{new_feature_name}'
GROUP BY
    u.user_id, u.signup_date, u.segment, u.device_type, u.age, u.previous_feature_engagement_score
ORDER BY
    u.user_id;
"""

user_early_features_df = pd.read_sql_query(sql_query, conn)
conn.close()

print(f"Early features DataFrame head:\n{user_early_features_df.head()}")
print(f"Early features DataFrame shape: {user_early_features_df.shape}")


# --- 3. Pandas Feature Engineering & Binary Target Creation ---

print("\n--- Performing Pandas Feature Engineering and Creating Binary Target ---")

# Handle NaN values (COALESCE in SQL should prevent most, but defensive coding)
user_early_features_df['num_events_first_14d'] = user_early_features_df['num_events_first_14d'].fillna(0)
user_early_features_df['total_duration_first_14d'] = user_early_features_df['total_duration_first_14d'].fillna(0)
user_early_features_df['num_unique_features_first_14d'] = user_early_features_df['num_unique_features_first_14d'].fillna(0)

# Convert signup_date to datetime objects
user_early_features_df['signup_date'] = pd.to_datetime(user_early_features_df['signup_date'])

# Calculate days_from_signup_to_feature_launch
user_early_features_df['days_from_signup_to_feature_launch'] = \
    (NEW_FEATURE_LAUNCH_DATE - user_early_features_df['signup_date']).dt.days

# Calculate engagement_per_event_first_14d
user_early_features_df['engagement_per_event_first_14d'] = \
    user_early_features_df['total_duration_first_14d'] / (user_early_features_df['num_events_first_14d'] + 1)
# Fill NaN or inf with 0 (e.g., if total_duration_first_14d is 0 and num_events_first_14d is 0, result is 0/1 = 0)
user_early_features_df['engagement_per_event_first_14d'] = \
    user_early_features_df['engagement_per_event_first_14d'].replace([np.inf, -np.inf], np.nan).fillna(0)


# Create the Binary Target `will_adopt_new_feature`
# Filter for 'New_Analytics_Dashboard' events within the defined adoption window
adoption_events_df = feature_events_df[
    (feature_events_df['feature_name'] == new_feature_name) &
    (feature_events_df['event_timestamp'] >= NEW_FEATURE_LAUNCH_DATE) &
    (feature_events_df['event_timestamp'] <= ADOPTION_END_DATE)
]

# Identify unique user_ids who adopted
adopters_ids = adoption_events_df['user_id'].unique()

# Create the target column, 1 if user_id is in adopters_ids, else 0
user_early_features_df['will_adopt_new_feature'] = user_early_features_df['user_id'].isin(adopters_ids).astype(int)

print(f"\nAdoption rate: {user_early_features_df['will_adopt_new_feature'].mean():.2%} ({user_early_features_df['will_adopt_new_feature'].sum()} adopters)")
print(f"Target column distribution:\n{user_early_features_df['will_adopt_new_feature'].value_counts()}")

# Define features X and target y
numerical_features = [
    'num_events_first_14d', 'total_duration_first_14d', 'num_unique_features_first_14d',
    'age', 'previous_feature_engagement_score', 'days_from_signup_to_feature_launch',
    'engagement_per_event_first_14d'
]
categorical_features = ['segment', 'device_type']
target = 'will_adopt_new_feature'

X = user_early_features_df[numerical_features + categorical_features]
y = user_early_features_df[target]

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"\nX_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
print(f"Training target distribution:\n{y_train.value_counts(normalize=True)}")
print(f"Testing target distribution:\n{y_test.value_counts(normalize=True)}")


# --- 4. Data Visualization ---

print("\n--- Generating Data Visualizations ---")

plt.style.use('ggplot')
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Violin plot for previous_feature_engagement_score vs adoption
sns.violinplot(
    x='will_adopt_new_feature',
    y='previous_feature_engagement_score',
    data=user_early_features_df,
    ax=axes[0],
    palette='viridis'
)
axes[0].set_title('Previous Engagement Score by New Feature Adoption')
axes[0].set_xlabel('Will Adopt New Feature (0=No, 1=Yes)')
axes[0].set_ylabel('Previous Feature Engagement Score')

# Stacked bar chart for segment vs adoption
segment_adoption_proportions = user_early_features_df.groupby('segment')['will_adopt_new_feature'].value_counts(normalize=True).unstack().fillna(0)
segment_adoption_proportions.plot(
    kind='bar',
    stacked=True,
    ax=axes[1],
    color=['lightcoral', 'lightgreen']
)
axes[1].set_title('New Feature Adoption Proportion by Segment')
axes[1].set_xlabel('Segment')
axes[1].set_ylabel('Proportion')
axes[1].legend(title='Will Adopt', labels=['No', 'Yes'])
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()


# --- 5. ML Pipeline & Evaluation ---

print("\n--- Building ML Pipeline and Evaluating Model ---")

# Create preprocessing pipelines for numerical and categorical features
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')), # Impute before scaling
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore')) # Handle unseen categories gracefully
])

# Create a column transformer to apply different transformations to different columns
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Create the full pipeline with preprocessor and classifier
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', HistGradientBoostingClassifier(random_state=42)) # Using HistGradientBoosting for good performance
])

# Train the pipeline
pipeline.fit(X_train, y_train)

# Predict probabilities for the positive class (class 1) on the test set
y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

# Predict classes on the test set (for classification report)
y_pred = pipeline.predict(X_test)


# --- Evaluation ---

print("\n--- Model Evaluation Results ---")

# ROC AUC Score
roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"ROC AUC Score: {roc_auc:.4f}")

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nScript execution complete.")