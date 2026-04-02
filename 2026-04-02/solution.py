import pandas as pd
import numpy as np
import datetime
from datetime import timedelta
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer # Corrected import path
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

print("Starting content interaction prediction pipeline...")

# 1. Generate Synthetic Data
print("1. Generating synthetic datasets...")

# --- Users DataFrame ---
num_users = np.random.randint(500, 701)
signup_dates = [pd.to_datetime('now') - timedelta(days=np.random.randint(0, 365*5)) for _ in range(num_users)]
age_groups = np.random.choice(['18-24', '25-34', '35-49', '50+'], size=num_users, p=[0.25, 0.35, 0.25, 0.15])
premium_status = np.random.choice(['Free', 'Basic', 'Premium'], size=num_users, p=[0.5, 0.3, 0.2])

users_df = pd.DataFrame({
    'user_id': range(num_users),
    'signup_date': signup_dates,
    'age_group': age_groups,
    'premium_status': premium_status
})

# --- Content Items DataFrame ---
num_content_items = np.random.randint(100, 151)
content_categories = ['Video', 'Article', 'Quiz', 'Forum', 'Ebook']
difficulties = ['Beginner', 'Intermediate', 'Advanced']

content_items_df = pd.DataFrame({
    'content_id': range(num_content_items),
    'category': np.random.choice(content_categories, size=num_content_items, p=[0.3, 0.3, 0.15, 0.15, 0.1]),
    'difficulty': np.random.choice(difficulties, size=num_content_items, p=[0.4, 0.4, 0.2]),
    'avg_rating': np.round(np.random.uniform(2.0, 5.0, num_content_items), 1)
})

# --- Interactions DataFrame ---
num_interactions_target = np.random.randint(10000, 15001)
interaction_types = ['view', 'like', 'share', 'comment', 'complete']
interaction_type_probs = [0.6, 0.2, 0.1, 0.05, 0.05] # 'comment' and 'complete' are rarer

interactions_data = []
interaction_id_counter = 0

# Simulate realistic patterns:
# - Users often interact with multiple items of the same category in a session.
# - Premium users interact with more diverse content and potentially more 'Advanced' difficulty items.
# - timestamps are after signup_date.
for _, user in users_df.iterrows():
    user_id = user['user_id']
    signup_date = user['signup_date']
    premium_status = user['premium_status']

    num_user_interactions = np.random.randint(15, 50) # Each user has a min/max number of interactions

    current_timestamp = signup_date + timedelta(days=np.random.randint(1, 30)) # First interaction after signup
    
    last_category = None
    for _ in range(num_user_interactions):
        # Decide if we stick to the last category or pick a new one
        if last_category is not None and np.random.rand() < 0.7: # 70% chance to stick to last category
            possible_content = content_items_df[content_items_df['category'] == last_category]
        else:
            possible_content = content_items_df

        # Premium user bias: more advanced content, more diverse (handled by not sticking to category as much)
        if premium_status == 'Premium':
            if np.random.rand() < 0.4: # 40% chance for advanced content
                premium_possible_content = possible_content[possible_content['difficulty'] == 'Advanced']
                if not premium_possible_content.empty:
                    possible_content = premium_possible_content
            # To ensure diversity, sometimes force a new category for premium users more often
            if np.random.rand() < 0.3 and last_category is not None: # 30% chance to break out of current category for premium
                possible_content = content_items_df[content_items_df['category'] != last_category]


        if possible_content.empty: # Fallback if filtering makes it empty
            possible_content = content_items_df

        selected_content = possible_content.sample(1).iloc[0]
        content_id = selected_content['content_id']
        current_category = selected_content['category']
        last_category = current_category # Update last category

        interaction_type = np.random.choice(interaction_types, p=interaction_type_probs)

        interactions_data.append({
            'interaction_id': interaction_id_counter,
            'user_id': user_id,
            'content_id': content_id,
            'timestamp': current_timestamp,
            'interaction_type': interaction_type
        })
        interaction_id_counter += 1

        # Increment timestamp for next interaction
        time_jump = np.random.randint(10, 60*24/2) # 10 minutes to half a day for next interaction
        current_timestamp += timedelta(minutes=time_jump)

        # Ensure we don't exceed target number of interactions significantly if not enough users
        if interaction_id_counter >= num_interactions_target * 1.1: # Allow some overshoot
            break
    if interaction_id_counter >= num_interactions_target * 1.1:
        break


interactions_df = pd.DataFrame(interactions_data)
# Filter to target size and sort
interactions_df = interactions_df.iloc[:num_interactions_target].sort_values(by=['user_id', 'timestamp']).reset_index(drop=True)
interactions_df['interaction_id'] = interactions_df.index # Re-assign interaction_id after sorting/filtering

print(f"Generated {len(users_df)} users, {len(content_items_df)} content items, {len(interactions_df)} interactions.")

# 2. Load into SQLite & SQL Feature Engineering
print("2. Loading data into SQLite and performing SQL feature engineering...")

conn = sqlite3.connect(':memory:')

users_df.to_sql('users', conn, index=False, if_exists='replace')
content_items_df.to_sql('content_items', conn, index=False, if_exists='replace')
# Convert timestamp to string for SQLite storage
interactions_df['timestamp_str'] = interactions_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
interactions_df.to_sql('interactions', conn, index=False, if_exists='replace')

# SQL Query for feature engineering and target creation
sql_query = """
WITH RankedInteractions AS (
    SELECT
        i.interaction_id,
        i.user_id,
        i.content_id,
        i.timestamp_str AS timestamp,
        i.interaction_type,
        ci.category,
        ci.difficulty,
        ci.avg_rating,
        u.signup_date,
        u.age_group,
        u.premium_status,
        -- Get the next content category for the same user
        LEAD(ci.category, 1) OVER (PARTITION BY u.user_id ORDER BY i.timestamp_str) AS next_content_category,
        -- Prior number of interactions for the user
        COUNT(i.interaction_id) OVER (PARTITION BY u.user_id ORDER BY i.timestamp_str ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS user_prior_num_interactions,
        -- Days since last user interaction (or signup_date if first interaction)
        julianday(i.timestamp_str) - julianday(LAG(i.timestamp_str, 1, u.signup_date) OVER (PARTITION BY u.user_id ORDER BY i.timestamp_str)) AS days_since_last_user_interaction,
        -- Prior unique content categories (string aggregated)
        GROUP_CONCAT(ci.category) OVER (PARTITION BY u.user_id ORDER BY i.timestamp_str ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS user_prior_categories_concat,
        -- Prior video views
        SUM(CASE WHEN ci.category = 'Video' AND i.interaction_type = 'view' THEN 1 ELSE 0 END) OVER (PARTITION BY u.user_id ORDER BY i.timestamp_str ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS user_prior_num_video_views,
        -- Prior article views
        SUM(CASE WHEN ci.category = 'Article' AND i.interaction_type = 'view' THEN 1 ELSE 0 END) OVER (PARTITION BY u.user_id ORDER BY i.timestamp_str ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS user_prior_num_article_views
    FROM
        interactions i
    JOIN
        users u ON i.user_id = u.user_id
    JOIN
        content_items ci ON i.content_id = ci.content_id
)
SELECT
    interaction_id,
    user_id,
    timestamp,
    interaction_type,
    content_id,
    category, -- current content category
    difficulty,
    avg_rating,
    signup_date,
    age_group,
    premium_status,
    user_prior_num_interactions,
    days_since_last_user_interaction,
    user_prior_categories_concat,
    user_prior_num_video_views,
    user_prior_num_article_views,
    next_content_category
FROM
    RankedInteractions
WHERE
    next_content_category IS NOT NULL
ORDER BY
    user_id, timestamp
;
"""

interaction_features_df = pd.read_sql_query(sql_query, conn)
conn.close()

print(f"SQL feature engineering complete. Resulting DataFrame has {len(interaction_features_df)} rows.")

# 3. Pandas Post-Processing, Additional Feature Engineering, and Data Split
print("3. Performing Pandas feature engineering and splitting data...")

# Handle NaN values for prior features
interaction_features_df['user_prior_num_interactions'] = interaction_features_df['user_prior_num_interactions'].fillna(0).astype(int)
interaction_features_df['user_prior_num_video_views'] = interaction_features_df['user_prior_num_video_views'].fillna(0).astype(int)
interaction_features_df['user_prior_num_article_views'] = interaction_features_df['user_prior_num_article_views'].fillna(0).astype(int)

# Convert dates
interaction_features_df['timestamp'] = pd.to_datetime(interaction_features_df['timestamp'])
interaction_features_df['signup_date'] = pd.to_datetime(interaction_features_df['signup_date'])

# Calculate days_since_signup_at_interaction
interaction_features_df['days_since_signup_at_interaction'] = (interaction_features_df['timestamp'] - interaction_features_df['signup_date']).dt.days

# Fill NaN for days_since_last_user_interaction (should be handled by SQL, but as a safeguard)
# For the very first interaction (where days_since_last_user_interaction might still be NaN if SQL didn't catch it for some reason)
# it means it's the first interaction since signup.
interaction_features_df['days_since_last_user_interaction'] = interaction_features_df['days_since_last_user_interaction'].fillna(
    interaction_features_df['days_since_signup_at_interaction']
)
interaction_features_df['days_since_last_user_interaction'] = interaction_features_df['days_since_last_user_interaction'].apply(lambda x: max(x, 0)) # Ensure non-negative

# Calculate user_prior_num_unique_content_categories from concatenated string
def count_unique_categories(category_string):
    if pd.isna(category_string) or category_string == '':
        return 0
    return len(set(category_string.split(',')))

interaction_features_df['user_prior_num_unique_content_categories'] = interaction_features_df['user_prior_categories_concat'].apply(count_unique_categories)
interaction_features_df.drop(columns=['user_prior_categories_concat'], inplace=True)


# Calculate interaction_frequency_prior
interaction_features_df['interaction_frequency_prior'] = interaction_features_df['user_prior_num_interactions'] / (interaction_features_df['days_since_signup_at_interaction'] + 1)
interaction_features_df['interaction_frequency_prior'].replace([np.inf, -np.inf], 0, inplace=True)
interaction_features_df['interaction_frequency_prior'] = interaction_features_df['interaction_frequency_prior'].fillna(0)

# Define features (X) and target (y)
numerical_features = [
    'avg_rating',
    'user_prior_num_interactions',
    'days_since_last_user_interaction',
    'user_prior_num_unique_content_categories',
    'user_prior_num_video_views',
    'user_prior_num_article_views',
    'days_since_signup_at_interaction',
    'interaction_frequency_prior'
]
categorical_features = [
    'interaction_type',
    'category', # current content category
    'difficulty',
    'age_group',
    'premium_status'
]

X = interaction_features_df[numerical_features + categorical_features]
y = interaction_features_df['next_content_category']

# Ensure all categories are present in target, filter out rare ones if necessary for stratification
# For simplicity, we assume target classes are well-distributed enough after filtering out NULLs
# If classes are too imbalanced, additional handling like oversampling or grouping might be needed.
# For now, let's check counts and proceed.
# print("Target class distribution before split:")
# print(y.value_counts())

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

print(f"Data split: {len(X_train)} training samples, {len(X_test)} testing samples.")

# 4. Data Visualization
print("4. Generating data visualizations...")

# Set up matplotlib for better plots
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams.update({'font.size': 10, 'figure.figsize': (10, 6)})

# Plot 1: Violin plot of days_since_last_user_interaction for top 5 next_content_category
top_5_categories = y.value_counts().head(5).index
df_plot_1 = interaction_features_df[interaction_features_df['next_content_category'].isin(top_5_categories)]

plt.figure(figsize=(12, 7))
sns.violinplot(x='next_content_category', y='days_since_last_user_interaction', data=df_plot_1, inner='quartile', palette='viridis')
plt.title('Distribution of Days Since Last Interaction by Top 5 Next Content Categories')
plt.xlabel('Next Content Category')
plt.ylabel('Days Since Last User Interaction')
plt.tight_layout()
plt.show()

# Plot 2: Stacked bar chart of next_content_category proportions across premium_status
df_plot_2 = interaction_features_df.groupby(['premium_status', 'next_content_category']).size().unstack(fill_value=0)
df_plot_2_prop = df_plot_2.apply(lambda x: x / x.sum(), axis=1) # Proportions within each premium_status

plt.figure(figsize=(12, 7))
df_plot_2_prop.plot(kind='bar', stacked=True, colormap='tab20')
plt.title('Proportion of Next Content Categories by Premium Status')
plt.xlabel('Premium Status')
plt.ylabel('Proportion')
plt.xticks(rotation=45)
plt.legend(title='Next Category', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

print("Visualizations complete.")

# 5. ML Pipeline & Evaluation
print("5. Building, training, and evaluating the Machine Learning Pipeline...")

# Create preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ]), numerical_features),
        ('cat', Pipeline([
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ]), categorical_features)
    ],
    remainder='drop' # Drop any columns not specified
)

# Create the full pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', HistGradientBoostingClassifier(random_state=42))
])

# Train the pipeline
print("Training the model...")
pipeline.fit(X_train, y_train)
print("Model training complete.")

# Make predictions
print("Making predictions on the test set...")
y_pred = pipeline.predict(X_test)
print("Predictions complete.")

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print("\n--- Model Evaluation ---")
print(f"Accuracy Score: {accuracy:.4f}")
print("\nClassification Report:\n", class_report)
print("Pipeline execution finished successfully.")