import pandas as pd
import numpy as np
import datetime
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer # Corrected import from sklearn.impute
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report

# Suppress pandas FutureWarnings and SettingWithCopyWarning for cleaner output
pd.options.mode.chained_assignment = None
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning, module='seaborn') # Suppress seaborn specific warnings

print("--- 1. Generating Synthetic Data ---")

# --- 1. Generate Synthetic Data (Pandas/Numpy) ---

# Define global date ranges
GLOBAL_START_DATE = datetime.datetime.now() - datetime.timedelta(days=5 * 365) # 5 years ago
GLOBAL_ANALYSIS_DATE_DEFAULT = datetime.datetime.now() # Default, will be updated later

# Users DataFrame
num_users = np.random.randint(500, 701)
user_ids = np.arange(1, num_users + 1)
signup_dates = [GLOBAL_START_DATE + datetime.timedelta(days=np.random.randint(0, 5 * 365)) for _ in range(num_users)]
age_groups = ['18-24', '25-40', '41-60', '60+']
preferred_genres = ['Tech', 'Science', 'History', 'Lifestyle', 'News']
users_df = pd.DataFrame({
    'user_id': user_ids,
    'signup_date': pd.to_datetime(signup_dates),
    'age_group': np.random.choice(age_groups, num_users),
    'preferred_genre': np.random.choice(preferred_genres, num_users)
})
print(f"Generated {len(users_df)} users.")

# Content DataFrame
num_content = np.random.randint(200, 301)
content_ids = np.arange(1, num_content + 1)
genres = preferred_genres # Same genres
difficulties = ['Beginner', 'Intermediate', 'Advanced']
avg_read_time_minutes = np.random.uniform(5, 60, num_content)
upload_dates = [GLOBAL_START_DATE + datetime.timedelta(days=np.random.randint(0, 4 * 365)) for _ in range(num_content)] # Last 4 years
content_df = pd.DataFrame({
    'content_id': content_ids,
    'genre': np.random.choice(genres, num_content),
    'difficulty': np.random.choice(difficulties, num_content),
    'avg_read_time_minutes': avg_read_time_minutes,
    'upload_date': pd.to_datetime(upload_dates)
})
print(f"Generated {len(content_df)} content items.")

# Recommendations DataFrame
num_recs = np.random.randint(10000, 15001)
rec_ids = np.arange(1, num_recs + 1)
rec_user_ids = np.random.choice(users_df['user_id'], num_recs)
rec_content_ids = np.random.choice(content_df['content_id'], num_recs)

recommendations_df = pd.DataFrame({
    'rec_id': rec_ids,
    'user_id': rec_user_ids,
    'content_id': rec_content_ids
})

# Merge to get signup_date and upload_date for each recommendation
recommendations_df = recommendations_df.merge(users_df[['user_id', 'signup_date', 'age_group', 'preferred_genre']], on='user_id')
recommendations_df = recommendations_df.merge(content_df[['content_id', 'genre', 'difficulty', 'upload_date']], on='content_id')

# Generate rec_date ensuring it's after both signup_date and upload_date
min_valid_rec_start_dates = np.maximum(recommendations_df['signup_date'], recommendations_df['upload_date'])
# Add a random timedelta between 1 day and 300 days (arbitrary but ensures future date)
random_timedelta_days = np.random.randint(1, 301, num_recs)
recommendations_df['rec_date'] = min_valid_rec_start_dates + pd.to_timedelta(random_timedelta_days, unit='D')
recommendations_df['rec_date'] = recommendations_df['rec_date'].dt.normalize() # Normalize to remove time component

# Simulate realistic engagement patterns for 'was_clicked'
# Overall click rate 5-10%
base_click_prob = 0.05
click_probs = np.full(num_recs, base_click_prob)

# Bias 1: Users more likely to click on content matching their preferred_genre
genre_match_mask = recommendations_df['preferred_genre'] == recommendations_df['genre']
click_probs[genre_match_mask] += 0.08 # +8% chance

# Bias 2: Content with difficulty='Beginner' might have higher initial click rates
beginner_mask = recommendations_df['difficulty'] == 'Beginner'
click_probs[beginner_mask] += 0.04 # +4% chance

# Bias 3: More recently uploaded content generally gets more clicks
# Calculate age of content at recommendation time
content_age_at_rec_days = (recommendations_df['rec_date'] - recommendations_df['upload_date']).dt.days
# Add a bias for newer content (e.g., within 90 days)
recent_upload_mask = content_age_at_rec_days <= 90
click_probs[recent_upload_mask] += 0.03 # +3% chance

# Bias 4: Some age_groups might prefer certain genres or difficulty levels
# Example: '60+' prefer 'History'
age_genre_bias_mask = (recommendations_df['age_group'] == '60+') & (recommendations_df['genre'] == 'History')
click_probs[age_genre_bias_mask] += 0.05 # +5% chance

# Example: '18-24' prefer 'Tech' and 'Beginner' difficulty
age_tech_beginner_bias_mask = (recommendations_df['age_group'] == '18-24') & \
                              (recommendations_df['genre'] == 'Tech') & \
                              (recommendations_df['difficulty'] == 'Beginner')
click_probs[age_tech_beginner_bias_mask] += 0.07 # +7% chance

# Cap probabilities at a reasonable max (e.g., 0.5 to avoid too many clicks)
click_probs = np.clip(click_probs, 0.01, 0.5)

recommendations_df['was_clicked'] = (np.random.rand(num_recs) < click_probs).astype(int)

# Drop temporary merge columns
recommendations_df = recommendations_df.drop(columns=['signup_date', 'age_group', 'preferred_genre', 'genre', 'difficulty', 'upload_date'])

# Sort recommendations_df
recommendations_df = recommendations_df.sort_values(by=['user_id', 'rec_date']).reset_index(drop=True)
print(f"Generated {len(recommendations_df)} recommendations with a {recommendations_df['was_clicked'].mean():.2%} click rate.")
print("--- Synthetic data generation complete. ---")


print("\n--- 2. Loading into SQLite & SQL Feature Engineering ---")

# --- 2. Load into SQLite & SQL Feature Engineering ---
conn = sqlite3.connect(':memory:')

# Load DataFrames into SQLite tables
users_df.to_sql('users', conn, index=False, if_exists='replace')
content_df.to_sql('content', conn, index=False, if_exists='replace')
recommendations_df.to_sql('recommendations', conn, index=False, if_exists='replace')

# Create a temporary table for difficulty mapping
conn.execute("""
CREATE TEMPORARY TABLE IF NOT EXISTS difficulty_mapping (
    difficulty TEXT PRIMARY KEY,
    score INTEGER
);
""")
conn.execute("INSERT OR REPLACE INTO difficulty_mapping (difficulty, score) VALUES ('Beginner', 1);")
conn.execute("INSERT OR REPLACE INTO difficulty_mapping (difficulty, score) VALUES ('Intermediate', 2);")
conn.execute("INSERT OR REPLACE INTO difficulty_mapping (difficulty, score) VALUES ('Advanced', 3);")
conn.commit()


sql_query = """
WITH UserRecs AS (
    SELECT
        r.user_id,
        r.rec_id,
        r.rec_date,
        r.was_clicked,
        c.content_id,
        c.genre,
        c.difficulty,
        c.avg_read_time_minutes,
        u.signup_date,
        DATE(u.signup_date, '+30 days') AS initial_window_cutoff_date
    FROM
        recommendations AS r
    JOIN
        users AS u ON r.user_id = u.user_id
    JOIN
        content AS c ON r.content_id = c.content_id
),
First30DayRecs AS (
    SELECT
        ur.user_id,
        ur.rec_id,
        ur.rec_date,
        ur.was_clicked,
        ur.genre,
        ur.difficulty,
        ur.avg_read_time_minutes,
        ur.signup_date,
        dm.score AS difficulty_score
    FROM
        UserRecs AS ur
    LEFT JOIN
        difficulty_mapping AS dm ON ur.difficulty = dm.difficulty
    WHERE
        ur.rec_date BETWEEN ur.signup_date AND ur.initial_window_cutoff_date
)
SELECT
    u.user_id,
    u.signup_date,
    u.age_group,
    u.preferred_genre,
    COALESCE(COUNT(f.rec_id), 0) AS num_recs_first_30d,
    COALESCE(SUM(CASE WHEN f.was_clicked = 1 THEN 1 ELSE 0 END), 0) AS num_clicks_first_30d,
    COALESCE(AVG(CASE WHEN f.was_clicked = 1 THEN f.avg_read_time_minutes ELSE NULL END), 0.0) AS avg_clicked_read_time_first_30d,
    COALESCE(COUNT(DISTINCT CASE WHEN f.was_clicked = 1 THEN f.genre ELSE NULL END), 0) AS num_unique_genres_clicked_first_30d,
    COALESCE(AVG(CASE WHEN f.was_clicked = 1 THEN f.difficulty_score ELSE NULL END), 0.0) AS avg_difficulty_score_first_30d,
    CASE
        WHEN MIN(f.rec_date) IS NOT NULL THEN JULIANDAY(MIN(f.rec_date)) - JULIANDAY(u.signup_date)
        ELSE NULL
    END AS days_since_first_rec_first_30d
FROM
    users AS u
LEFT JOIN
    First30DayRecs AS f ON u.user_id = f.user_id
GROUP BY
    u.user_id, u.signup_date, u.age_group, u.preferred_genre
ORDER BY
    u.user_id;
"""

user_initial_features_df = pd.read_sql_query(sql_query, conn)
print(f"Generated {len(user_initial_features_df)} user initial feature rows from SQL.")
print("--- SQL Feature Engineering complete. ---")

print("\n--- 3. Pandas Feature Engineering & Multi-Class Target Creation ---")

# --- 3. Pandas Feature Engineering & Multi-Class Target Creation ---

# Handle NaN values
user_initial_features_df['num_recs_first_30d'] = user_initial_features_df['num_recs_first_30d'].fillna(0).astype(int)
user_initial_features_df['num_clicks_first_30d'] = user_initial_features_df['num_clicks_first_30d'].fillna(0).astype(int)
user_initial_features_df['num_unique_genres_clicked_first_30d'] = user_initial_features_df['num_unique_genres_clicked_first_30d'].fillna(0).astype(int)
user_initial_features_df['avg_clicked_read_time_first_30d'] = user_initial_features_df['avg_clicked_read_time_first_30d'].fillna(0.0)
user_initial_features_df['avg_difficulty_score_first_30d'] = user_initial_features_df['avg_difficulty_score_first_30d'].fillna(0.0)
user_initial_features_df['days_since_first_rec_first_30d'] = user_initial_features_df['days_since_first_rec_first_30d'].fillna(30.0)

# Convert signup_date to datetime objects
user_initial_features_df['signup_date'] = pd.to_datetime(user_initial_features_df['signup_date'])

# Calculate global_analysis_date
# This is max rec_date from original recommendations + 60 days
GLOBAL_ANALYSIS_DATE = recommendations_df['rec_date'].max() + pd.Timedelta(days=60)
print(f"Global analysis date set to: {GLOBAL_ANALYSIS_DATE.strftime('%Y-%m-%d')}")

# Calculate user_account_age_at_analysis_days
user_initial_features_df['user_account_age_at_analysis_days'] = \
    (GLOBAL_ANALYSIS_DATE - user_initial_features_df['signup_date']).dt.days

# Calculate click_rate_first_30d
user_initial_features_df['click_rate_first_30d'] = \
    user_initial_features_df['num_clicks_first_30d'] / \
    user_initial_features_df['num_recs_first_30d'].replace({0: 1.0}) # Replace 0 with 1.0 to avoid division by zero

# Create the Multi-Class Target `future_engagement_tier`
# Define the cutoff date for future clicks
user_initial_features_df['initial_window_cutoff_date'] = user_initial_features_df['signup_date'] + pd.Timedelta(days=30)

# Merge back user signup dates to original recommendations_df
recs_with_signup = recommendations_df.merge(users_df[['user_id', 'signup_date']], on='user_id', how='left')

# Filter for future clicks
future_clicks_df = recs_with_signup[
    (recs_with_signup['rec_date'] > recs_with_signup['signup_date'] + pd.Timedelta(days=30)) &
    (recs_with_signup['rec_date'] <= GLOBAL_ANALYSIS_DATE)
]

total_future_clicks = future_clicks_df.groupby('user_id')['was_clicked'].sum().reset_index()
total_future_clicks.rename(columns={'was_clicked': 'total_future_clicks'}, inplace=True)

user_initial_features_df = user_initial_features_df.merge(total_future_clicks, on='user_id', how='left')
user_initial_features_df['total_future_clicks'] = user_initial_features_df['total_future_clicks'].fillna(0).astype(int)

# Calculate percentiles for non-zero total_future_clicks
non_zero_future_clicks = user_initial_features_df[user_initial_features_df['total_future_clicks'] > 0]['total_future_clicks']
if not non_zero_future_clicks.empty:
    p33 = non_zero_future_clicks.quantile(0.33)
    p66 = non_zero_future_clicks.quantile(0.66)
else:
    # Fallback for very sparse clicks, or if all future clicks are 0 (unlikely with synthetic data setup)
    p33 = 1
    p66 = 2
    print("Warning: All future clicks are zero or very few, using fallback percentiles.")


def assign_engagement_tier(clicks):
    if clicks == 0:
        return 'Low_Engagement'
    elif clicks <= p33:
        return 'Medium_Engagement'
    elif clicks <= p66:
        return 'High_Engagement'
    else:
        return 'Very_High_Engagement'

user_initial_features_df['future_engagement_tier'] = user_initial_features_df['total_future_clicks'].apply(assign_engagement_tier)

# Define features X and target y
numerical_features = [
    'user_account_age_at_analysis_days',
    'num_recs_first_30d',
    'num_clicks_first_30d',
    'avg_clicked_read_time_first_30d',
    'num_unique_genres_clicked_first_30d',
    'avg_difficulty_score_first_30d',
    'days_since_first_rec_first_30d',
    'click_rate_first_30d'
]
categorical_features = ['age_group', 'preferred_genre']

X = user_initial_features_df[numerical_features + categorical_features]
y = user_initial_features_df['future_engagement_tier']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
print("Future Engagement Tier distribution in training set:")
print(y_train.value_counts(normalize=True))
print("--- Pandas Feature Engineering & Target Creation complete. ---")


print("\n--- 4. Data Visualization ---")

# --- 4. Data Visualization ---
plt.style.use('seaborn-v0_8-darkgrid')
plt.figure(figsize=(16, 7))

# Plot 1: Violin plot of click_rate_first_30d vs. future_engagement_tier
plt.subplot(1, 2, 1)
sns.violinplot(x='future_engagement_tier', y='click_rate_first_30d', data=user_initial_features_df,
               order=['Low_Engagement', 'Medium_Engagement', 'High_Engagement', 'Very_High_Engagement'],
               palette='viridis')
plt.title('Distribution of Initial 30-Day Click Rate by Future Engagement Tier')
plt.xlabel('Future Engagement Tier')
plt.ylabel('Click Rate (First 30 Days)')
plt.ylim(0, user_initial_features_df['click_rate_first_30d'].max() * 1.1)


# Plot 2: Stacked bar chart of future_engagement_tier across age_group
plt.subplot(1, 2, 2)
age_group_engagement_proportion = user_initial_features_df.groupby('age_group')['future_engagement_tier'].value_counts(normalize=True).unstack()
# Ensure all target classes are present for consistent plotting, fill with 0 if not
for tier in ['Low_Engagement', 'Medium_Engagement', 'High_Engagement', 'Very_High_Engagement']:
    if tier not in age_group_engagement_proportion.columns:
        age_group_engagement_proportion[tier] = 0
age_group_engagement_proportion = age_group_engagement_proportion[['Low_Engagement', 'Medium_Engagement', 'High_Engagement', 'Very_High_Engagement']]

age_group_engagement_proportion.plot(kind='bar', stacked=True, ax=plt.gca(), cmap='viridis')
plt.title('Proportion of Future Engagement Tier by Age Group')
plt.xlabel('Age Group')
plt.ylabel('Proportion')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Engagement Tier', bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.show()
print("--- Data Visualization complete. ---")

print("\n--- 5. ML Pipeline & Evaluation ---")

# --- 5. ML Pipeline & Evaluation ---

# Preprocessing steps
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')), # Impute before OHE for potentially missing categories
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Create the ML pipeline
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('classifier', HistGradientBoostingClassifier(random_state=42))])

# Train the pipeline
pipeline.fit(X_train, y_train)
print("Model training complete.")

# Predict on the test set
y_pred = pipeline.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f"\nModel Accuracy on Test Set: {accuracy:.4f}")
print("\nClassification Report on Test Set:")
print(class_report)

print("--- ML Pipeline & Evaluation complete. ---")

# Close SQLite connection
conn.close()
print("\n--- Script Finished ---")