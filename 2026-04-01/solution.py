import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime, timedelta
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, classification_report

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# --- 1. Generate Synthetic Data ---

print("1. Generating Synthetic Data...")

# Leads DataFrame
num_leads = random.randint(500, 700)
lead_ids = np.arange(1, num_leads + 1)
signup_dates = [datetime.now() - timedelta(days=random.randint(0, 3*365)) for _ in range(num_leads)]
sources = ['Website_Organic', 'Paid_Ad', 'Referral', 'Partnership', 'Social_Media']
industries = ['Tech', 'Finance', 'Healthcare', 'Retail', 'Education']

leads_df = pd.DataFrame({
    'lead_id': lead_ids,
    'signup_date': signup_dates,
    'source': np.random.choice(sources, num_leads, p=[0.3, 0.25, 0.2, 0.15, 0.1]),
    'industry': np.random.choice(industries, num_leads, p=[0.3, 0.2, 0.2, 0.15, 0.15])
})

# Simulate conversion based on biases
# Base conversion rate around 10-20%
base_conversion_prob = 0.15
leads_df['is_converted_sim'] = np.random.rand(num_leads) < base_conversion_prob

# Bias conversion for specific sources/industries
leads_df.loc[leads_df['source'] == 'Referral', 'is_converted_sim'] = leads_df.loc[leads_df['source'] == 'Referral', 'is_converted_sim'].apply(lambda x: x or (random.random() < 0.3)) # Higher chance
leads_df.loc[leads_df['source'] == 'Paid_Ad', 'is_converted_sim'] = leads_df.loc[leads_df['source'] == 'Paid_Ad', 'is_converted_sim'].apply(lambda x: x and (random.random() < 0.7)) # Lower chance
leads_df.loc[leads_df['industry'] == 'Tech', 'is_converted_sim'] = leads_df.loc[leads_df['industry'] == 'Tech', 'is_converted_sim'].apply(lambda x: x or (random.random() < 0.25)) # Higher chance

# Activities DataFrame
all_activities = []
activity_id_counter = 1

activity_types = ['website_visit', 'email_open', 'form_submission', 'demo_request', 'resource_download', 'ad_click']

for _, lead in leads_df.iterrows():
    num_activities_for_lead = random.randint(10, 30) # Average activities per lead

    # If lead is simulated to convert, ensure conversion activities and higher engagement
    if lead['is_converted_sim']:
        # Ensure at least one 'demo_request' or 'form_submission' activity within 90 days for converted leads
        conversion_activity_timestamp = lead['signup_date'] + timedelta(days=random.randint(5, 80), hours=random.randint(1,23), minutes=random.randint(1,59))
        conversion_activity_type = random.choice(['demo_request', 'form_submission'])
        all_activities.append({
            'activity_id': activity_id_counter,
            'lead_id': lead['lead_id'],
            'activity_timestamp': conversion_activity_timestamp,
            'activity_type': conversion_activity_type,
            'duration_seconds': random.randint(120, 600) if conversion_activity_type == 'demo_request' else 0
        })
        activity_id_counter += 1
        num_activities_for_lead -= 1 # Account for this activity

        # Bias early activities for converted leads
        num_early_eng_activities = random.randint(1, 3) # More resource downloads/form submissions early
        for _ in range(num_early_eng_activities):
            early_activity_timestamp = lead['signup_date'] + timedelta(days=random.randint(1, 7), hours=random.randint(1,23))
            early_activity_type = random.choice(['form_submission', 'resource_download'])
            all_activities.append({
                'activity_id': activity_id_counter,
                'lead_id': lead['lead_id'],
                'activity_timestamp': early_activity_timestamp,
                'activity_type': early_activity_type,
                'duration_seconds': 0
            })
            activity_id_counter += 1
            num_activities_for_lead -= 1

    # Generate remaining activities
    for _ in range(num_activities_for_lead):
        # Activity timestamp must be after signup_date
        activity_timestamp = lead['signup_date'] + timedelta(days=random.randint(0, 120), hours=random.randint(1,23), minutes=random.randint(1,59))
        
        # Ensure it's strictly after signup_date for initial activities
        while activity_timestamp <= lead['signup_date']:
            activity_timestamp = lead['signup_date'] + timedelta(days=random.randint(0, 120), hours=random.randint(1,23), minutes=random.randint(1,59))

        activity_type = random.choice(activity_types)
        duration_seconds = 0
        if activity_type in ['website_visit', 'demo_request']:
            # Converted leads generally have higher durations
            duration_seconds = random.randint(60, 450) if lead['is_converted_sim'] else random.randint(10, 200)

        all_activities.append({
            'activity_id': activity_id_counter,
            'lead_id': lead['lead_id'],
            'activity_timestamp': activity_timestamp,
            'activity_type': activity_type,
            'duration_seconds': duration_seconds
        })
        activity_id_counter += 1

activities_df = pd.DataFrame(all_activities)

# Ensure activities_df has enough rows
min_activities = 10000
if len(activities_df) < min_activities:
    print(f"Warning: Generated {len(activities_df)} activities, less than {min_activities}. Adding more.")
    # Add more generic activities if needed to meet minimum count
    additional_activities_needed = min_activities - len(activities_df)
    additional_activities = []
    for _ in range(additional_activities_needed):
        lead = leads_df.sample(1).iloc[0]
        activity_timestamp = lead['signup_date'] + timedelta(days=random.randint(0, 120), hours=random.randint(1,23), minutes=random.randint(1,59))
        while activity_timestamp <= lead['signup_date']:
            activity_timestamp = lead['signup_date'] + timedelta(days=random.randint(0, 120), hours=random.randint(1,23), minutes=random.randint(1,59))

        activity_type = random.choice(activity_types)
        duration_seconds = 0
        if activity_type in ['website_visit', 'demo_request']:
            duration_seconds = random.randint(10, 200) # Generic lower duration
        
        additional_activities.append({
            'activity_id': activity_id_counter,
            'lead_id': lead['lead_id'],
            'activity_timestamp': activity_timestamp,
            'activity_type': activity_type,
            'duration_seconds': duration_seconds
        })
        activity_id_counter += 1
    activities_df = pd.concat([activities_df, pd.DataFrame(additional_activities)], ignore_index=True)

activities_df['activity_id'] = np.arange(1, len(activities_df) + 1) # Re-index activity_id to ensure uniqueness and sequence
activities_df = activities_df.sort_values(by=['lead_id', 'activity_timestamp']).reset_index(drop=True)

# Convert dates to string for SQLite
leads_df['signup_date'] = leads_df['signup_date'].dt.strftime('%Y-%m-%d %H:%M:%S')
activities_df['activity_timestamp'] = activities_df['activity_timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')

print(f"Generated {len(leads_df)} leads and {len(activities_df)} activities.")

# --- 2. Load into SQLite & SQL Feature Engineering ---

print("\n2. Loading data into SQLite and performing SQL Feature Engineering...")

conn = sqlite3.connect(':memory:') # In-memory SQLite database

leads_df.to_sql('leads', conn, if_exists='replace', index=False)
activities_df.to_sql('activities', conn, if_exists='replace', index=False)

# SQL query to aggregate early lead behavior (first 10 days post-signup)
sql_query = """
SELECT
    l.lead_id,
    l.signup_date,
    l.source,
    l.industry,
    COALESCE(SUM(CASE WHEN a.activity_id IS NOT NULL THEN 1 ELSE 0 END), 0) AS num_activities_first_10d,
    COALESCE(SUM(a.duration_seconds), 0) AS total_engagement_duration_first_10d,
    COALESCE(SUM(CASE WHEN a.activity_type = 'website_visit' THEN 1 ELSE 0 END), 0) AS num_website_visits_first_10d,
    COALESCE(SUM(CASE WHEN a.activity_type = 'form_submission' THEN 1 ELSE 0 END), 0) AS num_form_submissions_first_10d,
    COALESCE(SUM(CASE WHEN a.activity_type = 'demo_request' THEN 1 ELSE 0 END), 0) AS num_demo_requests_first_10d,
    COALESCE(COUNT(DISTINCT DATE(a.activity_timestamp)), 0) AS days_with_activity_first_10d,
    COALESCE(MAX(CASE WHEN a.activity_type = 'form_submission' THEN 1 ELSE 0 END), 0) AS has_submitted_form_first_10d
FROM
    leads AS l
LEFT JOIN
    activities AS a ON l.lead_id = a.lead_id
    AND julianday(a.activity_timestamp) <= julianday(DATE(l.signup_date, '+10 days'))
GROUP BY
    l.lead_id, l.signup_date, l.source, l.industry
ORDER BY
    l.lead_id;
"""

lead_early_features_df = pd.read_sql_query(sql_query, conn)
conn.close() # Close the SQLite connection

print(f"SQL Feature Engineering complete. Resulting DataFrame has {len(lead_early_features_df)} rows.")
print("Sample of lead_early_features_df head:")
print(lead_early_features_df.head())

# --- 3. Pandas Feature Engineering & Binary Target Creation ---

print("\n3. Performing Pandas Feature Engineering and creating Binary Target...")

# Handle NaN values (from leads with no activities in first 10 days) - Already handled by COALESCE in SQL
# However, explicit fillna here ensures robustness for any unexpected NaNs
numeric_cols_to_fill = [
    'num_activities_first_10d', 'total_engagement_duration_first_10d',
    'num_website_visits_first_10d', 'num_form_submissions_first_10d',
    'num_demo_requests_first_10d', 'days_with_activity_first_10d',
    'has_submitted_form_first_10d'
]
for col in numeric_cols_to_fill:
    if col in lead_early_features_df.columns:
        lead_early_features_df[col] = pd.to_numeric(lead_early_features_df[col], errors='coerce').fillna(0).astype(int)

# Convert signup_date to datetime objects
lead_early_features_df['signup_date'] = pd.to_datetime(lead_early_features_df['signup_date'])

# Calculate account_age_at_cutoff_days (always 10 for this setup)
lead_early_features_df['account_age_at_cutoff_days'] = 10 # Since cutoff is always +10 days

# Calculate engagement_action_ratio_first_10d
lead_early_features_df['engagement_action_ratio_first_10d'] = (
    lead_early_features_df['num_form_submissions_first_10d'] +
    lead_early_features_df['num_demo_requests_first_10d']
) / (lead_early_features_df['num_activities_first_10d'] + 1) # Add 1 to prevent division by zero
lead_early_features_df['engagement_action_ratio_first_10d'] = lead_early_features_df['engagement_action_ratio_first_10d'].fillna(0)

# Calculate activity_frequency_first_10d
lead_early_features_df['activity_frequency_first_10d'] = lead_early_features_df['num_activities_first_10d'] / 10.0
lead_early_features_df['activity_frequency_first_10d'] = lead_early_features_df['activity_frequency_first_10d'].fillna(0)


# Create the Binary Target `is_converted`
# Convert original activities_df dates to datetime for calculation
activities_df['activity_timestamp'] = pd.to_datetime(activities_df['activity_timestamp'])
leads_df['signup_date'] = pd.to_datetime(leads_df['signup_date']) # Re-convert leads_df signup_date for merge

# Define conversion window
conversion_window_df = leads_df[['lead_id', 'signup_date']].copy()
conversion_window_df['conversion_start_date'] = conversion_window_df['signup_date']
conversion_window_df['conversion_end_date'] = conversion_window_df['signup_date'] + timedelta(days=90)

# Filter activities for conversion events within the 90-day window
conversion_activities = activities_df[
    activities_df['activity_type'].isin(['demo_request', 'form_submission'])
].copy()

# Merge with conversion window to filter by lead-specific date ranges
converted_leads_temp = pd.merge(
    conversion_activities,
    conversion_window_df,
    on='lead_id',
    how='inner'
)

# Check if activity falls within the 90-day window
converted_leads_temp = converted_leads_temp[
    (converted_leads_temp['activity_timestamp'] >= converted_leads_temp['conversion_start_date']) &
    (converted_leads_temp['activity_timestamp'] <= converted_leads_temp['conversion_end_date'])
]

# Get unique lead_ids that converted
converted_lead_ids = converted_leads_temp['lead_id'].unique()

# Create the final `is_converted` target DataFrame
target_df = pd.DataFrame({'lead_id': leads_df['lead_id'], 'is_converted': 0})
target_df.loc[target_df['lead_id'].isin(converted_lead_ids), 'is_converted'] = 1

# Merge the target with the main features DataFrame
final_df = pd.merge(lead_early_features_df, target_df, on='lead_id', how='left')
# Fill any NaN in `is_converted` with 0 (shouldn't happen with `how='left'` from all leads)
final_df['is_converted'] = final_df['is_converted'].fillna(0).astype(int)


print(f"Final DataFrame for ML has {len(final_df)} rows and {len(final_df.columns)} columns.")
print(f"Overall conversion rate: {final_df['is_converted'].mean():.2%}")
print("Sample of final_df head with new features and target:")
print(final_df.head())

# Define features (X) and target (y)
X = final_df.drop(columns=['lead_id', 'signup_date', 'is_converted'])
y = final_df['is_converted']

# Identify numerical and categorical features
numerical_features = [
    'num_activities_first_10d', 'total_engagement_duration_first_10d',
    'num_website_visits_first_10d', 'num_form_submissions_first_10d',
    'num_demo_requests_first_10d', 'days_with_activity_first_10d',
    'account_age_at_cutoff_days', 'engagement_action_ratio_first_10d',
    'activity_frequency_first_10d'
]
categorical_features = ['source', 'industry', 'has_submitted_form_first_10d']

# Ensure 'has_submitted_form_first_10d' is treated as categorical by OneHotEncoder
# It's currently int, which StandardScaler would process. Convert to object for OHE.
X['has_submitted_form_first_10d'] = X['has_submitted_form_first_10d'].astype(str)


# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
print(f"Data split into training ({len(X_train)} samples) and testing ({len(X_test)} samples).")
print(f"Training set conversion rate: {y_train.mean():.2%}")
print(f"Testing set conversion rate: {y_test.mean():.2%}")


# --- 4. Data Visualization ---

print("\n4. Generating Data Visualizations...")

plt.style.use('seaborn-v0_8-darkgrid')

# Plot 1: Violin plot for total_engagement_duration_first_10d vs. is_converted
plt.figure(figsize=(10, 6))
sns.violinplot(x='is_converted', y='total_engagement_duration_first_10d', data=final_df)
plt.title('Total Engagement Duration in First 10 Days by Conversion Status')
plt.xlabel('Is Converted (0=No, 1=Yes)')
plt.ylabel('Total Engagement Duration (Seconds)')
plt.show()

# Plot 2: Stacked bar chart for proportion of is_converted across different sources
plt.figure(figsize=(12, 7))
source_conversion_pivot = final_df.groupby('source')['is_converted'].value_counts(normalize=True).unstack().fillna(0)
source_conversion_pivot.plot(kind='bar', stacked=True, color=['salmon', 'lightgreen'])
plt.title('Conversion Proportion by Lead Source')
plt.xlabel('Lead Source')
plt.ylabel('Proportion')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Is Converted', labels=['No', 'Yes'])
plt.tight_layout()
plt.show()


# --- 5. ML Pipeline & Evaluation ---

print("\n5. Building ML Pipeline and Evaluating...")

# Preprocessing steps
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
    remainder='passthrough' # Keep other columns (if any)
)

# Create the ML pipeline
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', HistGradientBoostingClassifier(random_state=42))
])

# Train the pipeline
print("Training the ML pipeline...")
model_pipeline.fit(X_train, y_train)
print("Pipeline training complete.")

# Predict probabilities on the test set
y_pred_proba = model_pipeline.predict_proba(X_test)[:, 1]

# Calculate ROC AUC score
roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"\nROC AUC Score on Test Set: {roc_auc:.4f}")

# Predict classes for classification report (using default threshold of 0.5)
y_pred = (y_pred_proba >= 0.5).astype(int)

# Generate and print classification report
print("\nClassification Report on Test Set:")
print(classification_report(y_test, y_pred))

print("\nML Pipeline execution finished.")