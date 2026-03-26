import pandas as pd
import numpy as np
import datetime
import random
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, classification_report

# --- 1. Generate Synthetic Data (Pandas/Numpy) ---
print("--- Generating Synthetic Data ---")

# Seed for reproducibility
np.random.seed(42)
random.seed(42)

# Users DataFrame
num_users = random.randint(500, 700)
signup_dates = [datetime.datetime.now() - datetime.timedelta(days=random.randint(1, 5*365)) for _ in range(num_users)]
regions = ['North', 'South', 'East', 'West', 'Central']
device_preferences = ['Mobile', 'Desktop', 'Tablet']

users_df = pd.DataFrame({
    'user_id': range(num_users),
    'signup_date': signup_dates,
    'region': np.random.choice(regions, num_users),
    'device_preference': np.random.choice(device_preferences, num_users)
})

# Sessions DataFrame
num_sessions = random.randint(8000, 12000)
session_users = np.random.choice(users_df['user_id'], num_sessions)

sessions_data = []
for i in range(num_sessions):
    user_id = session_users[i]
    user_signup_date = users_df[users_df['user_id'] == user_id]['signup_date'].iloc[0]
    
    # Session start time must be AFTER signup date
    # Ensure min_time_delta is positive to guarantee session after signup
    min_time_delta = datetime.timedelta(days=1) 
    max_days_after_signup = (datetime.datetime.now() - user_signup_date).days
    
    if max_days_after_signup <= 0: # If signup is in the future or today
        time_after_signup = min_time_delta
    else:
        time_after_signup = datetime.timedelta(days=random.randint(1, max_days_after_signup))
    
    session_start_time = user_signup_date + time_after_signup
    
    session_duration_seconds = random.randint(30, 1800)
    num_page_views = random.randint(1, 50)
    
    sessions_data.append({
        'session_id': i,
        'user_id': user_id,
        'session_start_time': session_start_time,
        'session_duration_seconds': session_duration_seconds,
        'num_page_views': num_page_views
    })

sessions_df = pd.DataFrame(sessions_data)

# Merge with user data to get region, device_preference, signup_date for conversion biasing
sessions_df = pd.merge(sessions_df, users_df[['user_id', 'signup_date', 'region', 'device_preference']], on='user_id', how='left', suffixes=('_session', '_user'))
sessions_df.rename(columns={'signup_date_user': 'signup_date'}, inplace=True) # Keep original signup_date for user

# Calculate days since signup at session start for biasing (FIXED: correct datetime subtraction)
sessions_df['days_since_signup_at_session_for_bias'] = (sessions_df['session_start_time'] - sessions_df['signup_date']).dt.days

# Simulate realistic conversion patterns (initial estimation for page generation)
sessions_df['conversion_probability'] = 0.05 # Base conversion rate (around 5-10% overall target)

# Bias by session duration
sessions_df['conversion_probability'] += (sessions_df['session_duration_seconds'] / sessions_df['session_duration_seconds'].max()) * 0.05
# Bias by page views
sessions_df['conversion_probability'] += (sessions_df['num_page_views'] / sessions_df['num_page_views'].max()) * 0.05
# Bias by region
region_bias = {'North': 0.02, 'South': -0.01, 'East': 0.01, 'West': 0.03, 'Central': -0.02}
sessions_df['conversion_probability'] += sessions_df['region'].map(region_bias).fillna(0) # Fillna for safety
# Bias by device preference
device_bias = {'Mobile': -0.01, 'Desktop': 0.03, 'Tablet': 0.00}
sessions_df['conversion_probability'] += sessions_df['device_preference'].map(device_bias).fillna(0) # Fillna for safety
# Bias by days since signup (more established users)
sessions_df['conversion_probability'] += (sessions_df['days_since_signup_at_session_for_bias'] / sessions_df['days_since_signup_at_session_for_bias'].max()) * 0.03

# Clip probabilities to sensible range to control overall conversion rate
sessions_df['conversion_probability'] = sessions_df['conversion_probability'].clip(0.01, 0.20) # Ensure rate is between 1% and 20% for individual sessions

# Assign initial is_converted based on probability
sessions_df['is_converted'] = (np.random.rand(len(sessions_df)) < sessions_df['conversion_probability']).astype(int)

# Page Views DataFrame
num_page_views_total = random.randint(25000, 40000)
page_view_data = []

page_types = ['Homepage', 'Product_Page', 'Category_Page', 'Cart_Page', 'Checkout_Page', 'Purchase_Success', 'Help_Page']

# Base weights for page types
base_weights = {'Homepage': 0.3, 'Product_Page': 0.25, 'Category_Page': 0.2, 'Cart_Page': 0.05, 'Checkout_Page': 0.03, 'Purchase_Success': 0.005, 'Help_Page': 0.165}
# Additional weights for conversion-related pages if session is 'converted'
conversion_weights_bias = {'Homepage': 0.02, 'Product_Page': 0.15, 'Category_Page': 0.05, 'Cart_Page': 0.15, 'Checkout_Page': 0.15, 'Purchase_Success': 0.25, 'Help_Page': 0.05} # Significant boost for Purchase_Success

# Distribute page views among sessions, weighted by num_page_views
# Use .values.sum() to handle potential empty dataframe or all zero cases
total_num_page_views = sessions_df['num_page_views'].values.sum()
if total_num_page_views > 0:
    session_id_distribution = np.random.choice(sessions_df['session_id'], size=num_page_views_total, 
                                               p=sessions_df['num_page_views'].values / total_num_page_views)
else: # Fallback if no page views are specified in sessions_df
    session_id_distribution = np.random.choice(sessions_df['session_id'], size=num_page_views_total)

session_map = sessions_df.set_index('session_id')

# Track which converted sessions have received a Purchase_Success page
converted_sessions_with_ps = set()

for i in range(num_page_views_total):
    session_id = session_id_distribution[i]
    session_info = session_map.loc[session_id]
    
    session_start = pd.to_datetime(session_info['session_start_time']) # Ensure datetime type
    session_duration = session_info['session_duration_seconds']
    
    time_offset = random.randint(0, session_duration)
    timestamp = session_start + datetime.timedelta(seconds=time_offset)
    
    view_time_seconds = random.randint(5, 300)
    
    current_weights = np.array([base_weights[pt] for pt in page_types])
    
    if session_info['is_converted'] == 1:
        # Apply conversion bias to weights
        bias_factors = np.array([conversion_weights_bias[pt] for pt in page_types])
        current_weights = current_weights + bias_factors
        
        # Ensure 'Purchase_Success' if a converted session hasn't had one yet (for the first few page views)
        if session_id not in converted_sessions_with_ps and random.random() < 0.3: # Try to inject early
             page_type_choice = 'Purchase_Success'
             converted_sessions_with_ps.add(session_id)
        else:
            page_type_choice = np.random.choice(page_types, p=current_weights / current_weights.sum())
    else:
        page_type_choice = np.random.choice(page_types, p=current_weights / current_weights.sum())
    
    page_view_data.append({
        'page_view_id': i,
        'session_id': session_id,
        'page_type': page_type_choice,
        'view_time_seconds': view_time_seconds,
        'timestamp': timestamp
    })

page_views_df = pd.DataFrame(page_view_data)

# Re-evaluate is_converted based on actual page views (Purchase_Success presence)
# This is the definitive 'is_converted'
purchase_success_sessions = page_views_df[page_views_df['page_type'] == 'Purchase_Success']['session_id'].unique()
sessions_df['is_converted'] = sessions_df['session_id'].isin(purchase_success_sessions).astype(int)

actual_conversion_rate = sessions_df['is_converted'].mean()
print(f"Final actual conversion rate (after page view generation): {actual_conversion_rate:.2%}")

# Clean up temporary bias column
sessions_df.drop(columns=['conversion_probability', 'days_since_signup_at_session_for_bias'], inplace=True)

# Sort DataFrames
sessions_df = sessions_df.sort_values(by=['user_id', 'session_start_time']).reset_index(drop=True)
page_views_df = page_views_df.sort_values(by=['session_id', 'timestamp']).reset_index(drop=True)

# Format datetime columns for SQLite
for df in [users_df, sessions_df, page_views_df]:
    for col in df.select_dtypes(include=['datetime64[ns]']).columns:
        df[col] = df[col].dt.strftime('%Y-%m-%d %H:%M:%S')

print("\nusers_df head:")
print(users_df.head())
print("\nsessions_df head:")
print(sessions_df.head())
print("\npage_views_df head:")
print(page_views_df.head())


# --- 2. Load into SQLite & SQL Feature Engineering ---
print("\n--- Loading Data into SQLite and SQL Feature Engineering ---")

conn = sqlite3.connect(':memory:')

users_df.to_sql('users', conn, index=False, if_exists='replace')
sessions_df.to_sql('sessions', conn, index=False, if_exists='replace')
page_views_df.to_sql('page_views', conn, index=False, if_exists='replace')

sql_query = """
SELECT
    s.session_id,
    s.user_id,
    s.session_start_time,
    s.session_duration_seconds,
    s.num_page_views,
    s.is_converted,
    u.region,
    u.device_preference,
    u.signup_date,
    COALESCE(pv_agg.num_product_page_views, 0) AS num_product_page_views,
    COALESCE(pv_agg.total_time_on_product_pages_seconds, 0) AS total_time_on_product_pages_seconds,
    COALESCE(pv_agg.num_cart_page_views, 0) AS num_cart_page_views,
    COALESCE(pv_agg.num_checkout_page_views, 0) AS num_checkout_page_views,
    CASE WHEN COALESCE(pv_agg.num_checkout_page_views, 0) > 0 THEN 1 ELSE 0 END AS has_viewed_checkout,
    COALESCE(pv_agg.total_page_view_duration_sum_seconds, 0) AS total_page_view_duration_sum_seconds,
    COALESCE(pv_agg.avg_view_time_per_page_in_session, 0.0) AS avg_view_time_per_page_in_session
FROM
    sessions s
LEFT JOIN
    users u ON s.user_id = u.user_id
LEFT JOIN (
    SELECT
        session_id,
        COUNT(CASE WHEN page_type = 'Product_Page' THEN 1 ELSE NULL END) AS num_product_page_views,
        SUM(CASE WHEN page_type = 'Product_Page' THEN view_time_seconds ELSE 0 END) AS total_time_on_product_pages_seconds,
        COUNT(CASE WHEN page_type = 'Cart_Page' THEN 1 ELSE NULL END) AS num_cart_page_views,
        COUNT(CASE WHEN page_type = 'Checkout_Page' THEN 1 ELSE NULL END) AS num_checkout_page_views,
        SUM(view_time_seconds) AS total_page_view_duration_sum_seconds,
        AVG(view_time_seconds) AS avg_view_time_per_page_in_session
    FROM
        page_views
    GROUP BY
        session_id
) AS pv_agg ON s.session_id = pv_agg.session_id
ORDER BY
    s.session_id;
"""

session_features_df = pd.read_sql_query(sql_query, conn)
conn.close()

print("\nSession Features DataFrame head (from SQL):")
print(session_features_df.head())
print(f"Shape of session_features_df: {session_features_df.shape}")


# --- 3. Pandas Feature Engineering & Binary Target Creation ---
print("\n--- Pandas Feature Engineering ---")

# Handle NaN values (mostly from COALESCE in SQL, but double-check)
numerical_agg_cols = [
    'num_product_page_views', 'total_time_on_product_pages_seconds',
    'num_cart_page_views', 'num_checkout_page_views',
    'total_page_view_duration_sum_seconds', 'avg_view_time_per_page_in_session'
]
session_features_df[numerical_agg_cols] = session_features_df[numerical_agg_cols].fillna(0)
# 'has_viewed_checkout' is already properly 0/1 from SQL query CASE WHEN

# Convert date columns to datetime objects
session_features_df['signup_date'] = pd.to_datetime(session_features_df['signup_date'])
session_features_df['session_start_time'] = pd.to_datetime(session_features_df['session_start_time'])

# Calculate days_since_signup_at_session (FIXED: correct datetime subtraction)
session_features_df['days_since_signup_at_session'] = (
    session_features_df['session_start_time'] - session_features_df['signup_date']
).dt.days

# Calculate engagement_score_composite
session_features_df['engagement_score_composite'] = (
    session_features_df['session_duration_seconds'] +
    session_features_df['total_page_view_duration_sum_seconds'] +
    (session_features_df['num_product_page_views'] * 10) +
    (session_features_df['num_cart_page_views'] * 20)
)

print("\nSession Features DataFrame head (after Pandas FE):")
print(session_features_df.head())
print(f"Actual overall conversion rate: {session_features_df['is_converted'].mean():.2%}")

# Define features X and target y
numerical_features = [
    'session_duration_seconds', 'num_page_views',
    'num_product_page_views', 'total_time_on_product_pages_seconds',
    'num_cart_page_views', 'num_checkout_page_views',
    'total_page_view_duration_sum_seconds', 'avg_view_time_per_page_in_session',
    'days_since_signup_at_session', 'engagement_score_composite', # days_since_signup_at_session is numeric
    'has_viewed_checkout' # This is already 0/1, so treat as numerical
]
categorical_features = ['region', 'device_preference'] 

X = session_features_df[numerical_features + categorical_features]
y = session_features_df['is_converted']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

print(f"\nX_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
print(f"Conversion rate in y_train: {y_train.mean():.2%}")
print(f"Conversion rate in y_test: {y_test.mean():.2%}")


# --- 4. Data Visualization ---
print("\n--- Generating Visualizations ---")

plt.figure(figsize=(14, 6))

# Plot 1: Violin plot of session_duration_seconds by is_converted
plt.subplot(1, 2, 1)
sns.violinplot(x='is_converted', y='session_duration_seconds', data=session_features_df, palette='viridis')
plt.title('Session Duration by Conversion Status')
plt.xlabel('Is Converted (0=No, 1=Yes)')
plt.ylabel('Session Duration (Seconds)')
plt.xticks([0, 1], ['Not Converted', 'Converted'])

# Plot 2: Stacked bar chart of conversion proportion by region
plt.subplot(1, 2, 2)
region_conversion_proportions = session_features_df.groupby('region')['is_converted'].value_counts(normalize=True).unstack().fillna(0)
region_conversion_proportions.plot(kind='bar', stacked=True, ax=plt.gca(), cmap='coolwarm')
plt.title('Conversion Proportion by Region')
plt.xlabel('Region')
plt.ylabel('Proportion')
plt.legend(title='Is Converted', labels=['No', 'Yes'])
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


# --- 5. ML Pipeline & Evaluation ---
print("\n--- Building and Evaluating ML Pipeline ---")

# Define numerical and categorical features for the ColumnTransformer
# 'has_viewed_checkout' is 0/1 and already in numerical_features list, so passed to numerical_transformer
numerical_features_for_pipeline = numerical_features 

# Preprocessing pipelines for numerical and categorical features
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
        ('num', numerical_transformer, numerical_features_for_pipeline),
        ('cat', categorical_transformer, categorical_features)
    ])

# Create the full pipeline
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', HistGradientBoostingClassifier(random_state=42))
])

# Train the pipeline
print("Training the model pipeline...")
model_pipeline.fit(X_train, y_train)
print("Model training complete.")

# Predict probabilities on the test set
y_pred_proba = model_pipeline.predict_proba(X_test)[:, 1]

# Predict class labels on the test set for classification report
y_pred = model_pipeline.predict(X_test)

# Calculate and print ROC AUC score
roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"\nROC AUC Score on Test Set: {roc_auc:.4f}")

# Generate and print classification report
print("\nClassification Report on Test Set:")
print(classification_report(y_test, y_pred))

print("\n--- Script Finished ---")