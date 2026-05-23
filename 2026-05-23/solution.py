import pandas as pd
import numpy as np
import datetime
import sqlite3
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
import warnings

# Suppress specific warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
warnings.filterwarnings('ignore', category=FutureWarning, module='seaborn')
warnings.filterwarnings('ignore', category=FutureWarning, module='matplotlib')
warnings.filterwarnings('ignore', category=pd.io.sql.PerformanceWarning) # For to_sql large data

# --- 1. Generate Synthetic Data ---

print("--- 1. Generating Synthetic Data ---")

# Define parameters for data generation
num_users_min, num_users_max = 1000, 1500
num_features_min, num_features_max = 50, 100
num_interactions_min, num_interactions_max = 20000, 30000

num_users = np.random.randint(num_users_min, num_users_max + 1)
num_features = np.random.randint(num_features_min, num_features_max + 1)
num_interactions = np.random.randint(num_interactions_min, num_interactions_max + 1)

# Generate Users DataFrame
signup_start_date = pd.Timestamp.now() - pd.DateOffset(years=5)
signup_end_date = pd.Timestamp.now() - pd.DateOffset(years=3)
users_df = pd.DataFrame({
    'user_id': np.arange(num_users),
    'signup_date': pd.to_datetime(np.random.uniform(signup_start_date.timestamp(), signup_end_date.timestamp(), num_users), unit='s').date,
    'country': np.random.choice(['USA', 'CAN', 'GBR', 'DEU', 'FRA', 'AUS', 'JPN'], num_users),
    'device_type': np.random.choice(['Mobile', 'Desktop', 'Tablet'], num_users, p=[0.65, 0.30, 0.05]),
    'user_segment': np.random.choice(['Early Adopter', 'Mainstream', 'Late Adopter'], num_users, p=[0.15, 0.70, 0.15])
})

# Generate Features DataFrame
feature_release_start_date = pd.Timestamp.now() - pd.DateOffset(years=3)
feature_release_end_date = pd.Timestamp.now() - pd.DateOffset(years=1) # Ensure some features are recent
features_df = pd.DataFrame({
    'feature_id': np.arange(num_features),
    'feature_name': [f'Feature_{i}' for i in np.arange(num_features)],
    'release_date': pd.to_datetime(np.random.uniform(feature_release_start_date.timestamp(), feature_release_end_date.timestamp(), num_features), unit='s').date,
    'feature_category': np.random.choice(['Communication', 'Content Creation', 'Discovery', 'Analytics', 'Productivity', 'Social'], num_features)
})

# Select TARGET_FEATURE_ID and GLOBAL_PREDICTION_CUTOFF_DATE
# Choose a feature that was released in the last 1-1.5 years to ensure recent context
recent_features = features_df[features_df['release_date'] > (pd.Timestamp.now() - pd.DateOffset(years=1, months=6)).date()]
if not recent_features.empty:
    TARGET_FEATURE_ID = recent_features.sample(1, random_state=42)['feature_id'].iloc[0]
else: # Fallback if no sufficiently recent features exist (e.g., small num_features or very old dataset)
    TARGET_FEATURE_ID = features_df.sample(1, random_state=42)['feature_id'].iloc[0] # Pick any one

GLOBAL_PREDICTION_CUTOFF_DATE = features_df[features_df['feature_id'] == TARGET_FEATURE_ID]['release_date'].iloc[0]
TARGET_FEATURE_CATEGORY = features_df[features_df['feature_id'] == TARGET_FEATURE_ID]['feature_category'].iloc[0]

print(f"Selected TARGET_FEATURE_ID: {TARGET_FEATURE_ID}")
print(f"GLOBAL_PREDICTION_CUTOFF_DATE: {GLOBAL_PREDICTION_CUTOFF_DATE}")
print(f"TARGET_FEATURE_CATEGORY: {TARGET_FEATURE_CATEGORY}")

# Generate User Feature Interactions DataFrame
interaction_data = []
user_ids = users_df['user_id'].values
feature_ids = features_df['feature_id'].values
interaction_types = ['Viewed', 'Clicked', 'Used_Once', 'Used_Multiple']
interaction_type_probs = [0.5, 0.3, 0.15, 0.05] # Higher chance of passive interaction types

# Pre-calculate user and feature info for faster lookup
user_signup_dates = users_df.set_index('user_id')['signup_date'].apply(pd.to_datetime)
feature_release_dates = features_df.set_index('feature_id')['release_date'].apply(pd.to_datetime)
user_segments = users_df.set_index('user_id')['user_segment']
feature_categories = features_df.set_index('feature_id')['feature_category']

# Max timestamp for any interaction (up to the present moment)
max_overall_interaction_dt = pd.Timestamp.now()

# Simulate interactions with some biases
for i in range(num_interactions):
    u_id = np.random.choice(user_ids)
    f_id = np.random.choice(feature_ids)

    user_signup_dt = user_signup_dates.loc[u_id]
    feature_release_dt = feature_release_dates.loc[f_id]

    min_interaction_dt = max(user_signup_dt, feature_release_dt)

    if min_interaction_dt >= max_overall_interaction_dt:
        continue # Skip if no valid interaction window up to now

    # Determine a base duration for interaction (in days from min_interaction_dt)
    days_since_min_dt = (max_overall_interaction_dt - min_interaction_dt).days
    if days_since_min_dt <= 0:
        continue # Not enough time for interaction

    # Apply 'Early Adopter' bias: interact sooner after release
    # For Early Adopters, interaction_timestamp is skewed towards closer to release_date
    if user_segments.loc[u_id] == 'Early Adopter':
        interaction_duration_days = np.random.lognormal(mean=1.5, sigma=0.8) # Skewed towards smaller values
    else:
        interaction_duration_days = np.random.uniform(1, days_since_min_dt + 1)
    
    interaction_dt = min_interaction_dt + pd.Timedelta(days=min(interaction_duration_days, days_since_min_dt))
    
    # Cap interaction_dt at max_overall_interaction_dt if it somehow exceeds (due to lognormal tail)
    if interaction_dt > max_overall_interaction_dt:
        interaction_dt = max_overall_interaction_dt
        
    # Simulate category affinity: Users who are 'Early Adopter' or have interacted a lot in a category
    # are more likely to perform 'Used_Once'/'Used_Multiple' for features in that category.
    current_interaction_type = np.random.choice(interaction_types, p=interaction_type_probs)
    if user_segments.loc[u_id] == 'Early Adopter' or feature_categories.loc[f_id] == TARGET_FEATURE_CATEGORY:
        if random.random() < 0.2: # 20% chance to upgrade interaction type if biased
             current_interaction_type = np.random.choice(['Used_Once', 'Used_Multiple'], p=[0.6, 0.4])

    interaction_data.append({
        'user_id': u_id,
        'feature_id': f_id,
        'interaction_timestamp': interaction_dt.strftime('%Y-%m-%d %H:%M:%S'),
        'interaction_type': current_interaction_type
    })

user_feature_interactions_df = pd.DataFrame(interaction_data)
user_feature_interactions_df['interaction_id'] = np.arange(len(user_feature_interactions_df))

# Sort as required
user_feature_interactions_df = user_feature_interactions_df.sort_values(by=['user_id', 'interaction_timestamp']).reset_index(drop=True)

print(f"Generated {len(users_df)} users, {len(features_df)} features, {len(user_feature_interactions_df)} interactions.")
print("users_df head:\n", users_df.head())
print("features_df head:\n", features_df.head())
print("user_feature_interactions_df head:\n", user_feature_interactions_df.head())


# --- 2. Load into SQLite & SQL Feature Engineering ---

print("\n--- 2. Loading into SQLite & SQL Feature Engineering ---")

conn = sqlite3.connect(':memory:')

# Load DataFrames into SQLite tables
# Removed datetime_format as per feedback to ensure compatibility
users_df.to_sql('users', conn, index=False, if_exists='replace')
features_df.to_sql('features', conn, index=False, if_exists='replace')
user_feature_interactions_df.to_sql('interactions', conn, index=False, if_exists='replace')

sql_query = f"""
WITH TargetFeatureInfo AS (
    SELECT
        feature_id AS target_feature_id,
        feature_category AS target_feature_category,
        release_date AS global_prediction_cutoff_date
    FROM features
    WHERE feature_id = {TARGET_FEATURE_ID}
),
InteractionsBeforeCutoff AS (
    SELECT
        i.user_id,
        i.feature_id,
        i.interaction_timestamp,
        f.feature_category,
        i.interaction_id
    FROM interactions i
    JOIN features f ON i.feature_id = f.feature_id
    WHERE julianday(i.interaction_timestamp) <= julianday((SELECT global_prediction_cutoff_date FROM TargetFeatureInfo))
),
Prev30DayInteractions AS (
    SELECT
        user_id,
        feature_id,
        interaction_timestamp,
        feature_category,
        interaction_id
    FROM InteractionsBeforeCutoff
    WHERE julianday(interaction_timestamp) > julianday((SELECT global_prediction_cutoff_date FROM TargetFeatureInfo)) - 30
),
AggregatedFeatures AS (
    SELECT
        u.user_id,
        COUNT(p30.interaction_id) AS num_total_interactions_prev_30d,
        COUNT(DISTINCT p30.feature_id) AS num_distinct_features_used_prev_30d,
        SUM(CASE WHEN p30.feature_category = (SELECT target_feature_category FROM TargetFeatureInfo) THEN 1 ELSE 0 END) AS num_interactions_target_category_prev_30d
    FROM users u
    LEFT JOIN Prev30DayInteractions p30 ON u.user_id = p30.user_id
    GROUP BY u.user_id
),
LastInteractionOverall AS (
    SELECT
        user_id,
        MAX(interaction_timestamp) AS last_overall_interaction_before_cutoff
    FROM InteractionsBeforeCutoff
    GROUP BY user_id
)
SELECT
    u.user_id,
    u.signup_date,
    u.country,
    u.device_type,
    u.user_segment,
    (SELECT global_prediction_cutoff_date FROM TargetFeatureInfo) AS current_cutoff_date,
    (SELECT target_feature_category FROM TargetFeatureInfo) AS target_feature_category,
    COALESCE(af.num_total_interactions_prev_30d, 0) AS num_total_interactions_prev_30d,
    COALESCE(af.num_distinct_features_used_prev_30d, 0) AS num_distinct_features_used_prev_30d,
    COALESCE(af.num_interactions_target_category_prev_30d, 0) AS num_interactions_target_category_prev_30d,
    CASE
        WHEN lio.last_overall_interaction_before_cutoff IS NOT NULL THEN
            CAST(julianday((SELECT global_prediction_cutoff_date FROM TargetFeatureInfo)) - julianday(lio.last_overall_interaction_before_cutoff) AS INTEGER)
        ELSE 9999
    END AS days_since_last_interaction_at_cutoff
FROM users u
LEFT JOIN AggregatedFeatures af ON u.user_id = af.user_id
LEFT JOIN LastInteractionOverall lio ON u.user_id = lio.user_id;
"""

user_features_df = pd.read_sql_query(sql_query, conn)
conn.close()

print("SQL Feature Engineering complete. user_features_df head:\n", user_features_df.head())


# --- 3. Pandas Feature Engineering & Binary Target Creation ---

print("\n--- 3. Pandas Feature Engineering & Binary Target Creation ---")

# Convert date columns to datetime objects
user_features_df['signup_date'] = pd.to_datetime(user_features_df['signup_date'])
user_features_df['current_cutoff_date'] = pd.to_datetime(user_features_df['current_cutoff_date'])

# Handle NaNs: fill numerical aggregated features with 0, days_since_last_interaction_at_cutoff with 9999
numerical_agg_cols = [
    'num_total_interactions_prev_30d',
    'num_distinct_features_used_prev_30d',
    'num_interactions_target_category_prev_30d'
]
user_features_df[numerical_agg_cols] = user_features_df[numerical_agg_cols].fillna(0).astype(int)
# days_since_last_interaction_at_cutoff should already be handled by SQL's COALESCE/CASE statement.
# If for some unforeseen reason NaNs creep in, this handles them.
user_features_df['days_since_last_interaction_at_cutoff'] = user_features_df['days_since_last_interaction_at_cutoff'].fillna(9999).astype(int)

# Calculate user_tenure_at_cutoff_days
user_features_df['user_tenure_at_cutoff_days'] = (user_features_df['current_cutoff_date'] - user_features_df['signup_date']).dt.days

# Create the Binary Target `will_adopt_target_feature_in_7d`
adoption_window_start = pd.to_datetime(GLOBAL_PREDICTION_CUTOFF_DATE)
adoption_window_end = adoption_window_start + pd.Timedelta(days=7)

# Filter interactions for the target feature within the adoption window with specific interaction types
target_adoptions = user_feature_interactions_df[
    (user_feature_interactions_df['feature_id'] == TARGET_FEATURE_ID) &
    (pd.to_datetime(user_feature_interactions_df['interaction_timestamp']) > adoption_window_start) & # Exclusive start
    (pd.to_datetime(user_feature_interactions_df['interaction_timestamp']) <= adoption_window_end) &  # Inclusive end
    (user_feature_interactions_df['interaction_type'].isin(['Used_Once', 'Used_Multiple']))
]

# Get unique user_ids who adopted the target feature
adopting_users = target_adoptions['user_id'].unique()

# Create the target column
user_features_df['will_adopt_target_feature_in_7d'] = user_features_df['user_id'].isin(adopting_users).astype(int)

# Define features X and target y
numerical_features = [
    'num_total_interactions_prev_30d',
    'num_distinct_features_used_prev_30d',
    'num_interactions_target_category_prev_30d',
    'days_since_last_interaction_at_cutoff',
    'user_tenure_at_cutoff_days'
]
categorical_features = [
    'country',
    'device_type',
    'user_segment',
    'target_feature_category'
]

X = user_features_df[numerical_features + categorical_features]
y = user_features_df['will_adopt_target_feature_in_7d']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

print(f"Shape of X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"Shape of X_test: {X_test.shape}, y_test: {y_test.shape}")
print("Target distribution in training set:\n", y_train.value_counts(normalize=True))
print("Target distribution in test set:\n", y_test.value_counts(normalize=True))


# --- 4. Data Visualization ---

print("\n--- 4. Data Visualization ---")

plt.style.use('seaborn-v0_8-darkgrid')
plt.figure(figsize=(15, 6))

# Plot 1: Violin plot for num_interactions_target_category_prev_30d vs adoption
plt.subplot(1, 2, 1)
# Add a small constant to allow log scale for 0 values, or filter them for visualization
plot_data_log = user_features_df.copy()
plot_data_log['num_interactions_target_category_prev_30d_log'] = plot_data_log['num_interactions_target_category_prev_30d'].apply(lambda x: x if x > 0 else np.nan) # Handle 0 for log plot

sns.violinplot(x='will_adopt_target_feature_in_7d', y='num_interactions_target_category_prev_30d_log', data=plot_data_log, palette='viridis', inner='quartile', hue='will_adopt_target_feature_in_7d', legend=False)
plt.yscale('log') # Use log scale due to potential skew
plt.title('Distribution of Target Category Interactions by Adoption')
plt.xlabel('Will Adopt Target Feature in 7 Days (0=No, 1=Yes)')
plt.ylabel('Num Interactions in Target Category (Prev 30 Days, Log Scale, non-zero)')
plt.grid(True, which="both", ls="-", alpha=0.2)


# Plot 2: Stacked bar chart for user_segment vs adoption proportion
plt.subplot(1, 2, 2)
user_segment_adoption = pd.crosstab(user_features_df['user_segment'], user_features_df['will_adopt_target_feature_in_7d'], normalize='index')
user_segment_adoption.plot(kind='bar', stacked=True, ax=plt.gca(), cmap='coolwarm')
plt.title('Proportion of Adoption by User Segment')
plt.xlabel('User Segment')
plt.ylabel('Proportion')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Will Adopt', labels=['No', 'Yes'], loc='upper left', bbox_to_anchor=(1, 1))

plt.tight_layout()
plt.show()


# --- 5. ML Pipeline & Evaluation ---

print("\n--- 5. ML Pipeline & Evaluation ---")

# Create preprocessing pipelines for numerical and categorical features
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')), # Handles potential NaNs, though filled earlier
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
    remainder='passthrough' # Keep other columns if any (not expected here)
)

# Create the full pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', HistGradientBoostingClassifier(random_state=42, class_weight='balanced')) # Balanced for potential class imbalance
])

# Train the pipeline
print("Training ML model...")
model.fit(X_train, y_train)
print("Model training complete.")

# Predict probabilities for the positive class (class 1) on the test set
y_pred_proba = model.predict_proba(X_test)[:, 1]
# Predict classes for the classification report
y_pred = model.predict(X_test)

# Evaluate the model
roc_auc = roc_auc_score(y_test, y_pred_proba)
class_report = classification_report(y_test, y_pred)

print(f"\nROC AUC Score on Test Set: {roc_auc:.4f}")
print("\nClassification Report on Test Set:")
print(class_report)