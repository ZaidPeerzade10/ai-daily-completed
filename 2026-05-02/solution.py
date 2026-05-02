import pandas as pd
import numpy as np
import datetime
import sqlite3
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, classification_report

# Ensure plots are displayed without requiring user interaction
plt.switch_backend('Agg')

# --- 1. Generate Synthetic Data (Pandas/Numpy) ---

print("--- Generating Synthetic Data ---")

N_USERS = random.randint(1000, 1500)
N_ASSIGNMENTS = random.randint(2000, 3000)
N_EVENTS = random.randint(30000, 50000)

# User data
user_ids = np.arange(N_USERS)
signup_start_date = datetime.datetime.now() - datetime.timedelta(days=3 * 365)
signup_dates = [signup_start_date + datetime.timedelta(days=random.randint(0, 3 * 365)) for _ in range(N_USERS)]
countries = ['USA', 'Canada', 'UK', 'Germany', 'France', 'Australia']
device_types = ['Mobile', 'Desktop', 'Tablet']
age_groups = ['18-24', '25-34', '35-44', '45+']

users_df = pd.DataFrame({
    'user_id': user_ids,
    'signup_date': signup_dates,
    'country': np.random.choice(countries, N_USERS),
    'device_type': np.random.choice(device_types, N_USERS),
    'age_group': np.random.choice(age_groups, N_USERS)
})

# AB test assignments
assignment_ids = np.arange(N_ASSIGNMENTS)
assignment_user_ids = np.random.choice(user_ids, N_ASSIGNMENTS, replace=True)
test_names = ['HomepageRedesign', 'CheckoutFlow', 'SearchAlgorithm', 'PricingPage']
variant_names = ['Control', 'VariantA', 'VariantB']

ab_test_assignments_df = pd.DataFrame({
    'assignment_id': assignment_ids,
    'user_id': assignment_user_ids,
    'test_name': np.random.choice(test_names, N_ASSIGNMENTS),
    'variant_name': np.random.choice(variant_names, N_ASSIGNMENTS)
})

# Ensure assignment_date is after signup_date
ab_test_assignments_df = ab_test_assignments_df.merge(
    users_df[['user_id', 'signup_date']], on='user_id', how='left'
)
ab_test_assignments_df['assignment_date'] = ab_test_assignments_df.apply(
    lambda row: row['signup_date'] + datetime.timedelta(days=random.randint(7, 365)), axis=1
)
ab_test_assignments_df = ab_test_assignments_df.drop(columns=['signup_date'])


# User events
event_ids = np.arange(N_EVENTS)
event_user_ids = np.random.choice(user_ids, N_EVENTS, replace=True)
event_types = ['page_view', 'click', 'add_to_cart', 'page_view', 'click'] # 'purchase' added later for control

user_events_df = pd.DataFrame({
    'event_id': event_ids,
    'user_id': event_user_ids,
    'event_type': np.random.choice(event_types, N_EVENTS),
    'revenue': 0.0
})

# Ensure event_timestamp is after signup_date
user_events_df = user_events_df.merge(users_df[['user_id', 'signup_date']], on='user_id', how='left')
user_events_df['event_timestamp'] = user_events_df.apply(
    lambda row: row['signup_date'] + datetime.timedelta(
        days=random.randint(0, 3 * 365 + 100),
        hours=random.randint(0, 23),
        minutes=random.randint(0, 59),
        seconds=random.randint(0, 59)
    ), axis=1
)
user_events_df = user_events_df.drop(columns=['signup_date'])


# Simulate realistic conversion patterns
# Base conversion rate
base_conversion_rate = 0.06

# Variant effects (e.g., +2-5% increase for some, - for others)
variant_effects = {
    ('HomepageRedesign', 'Control'): 0.0,
    ('HomepageRedesign', 'VariantA'): 0.03, # +3%
    ('HomepageRedesign', 'VariantB'): -0.01, # -1%
    ('CheckoutFlow', 'Control'): 0.0,
    ('CheckoutFlow', 'VariantA'): 0.02,
    ('CheckoutFlow', 'VariantB'): 0.01,
    ('SearchAlgorithm', 'Control'): 0.0,
    ('SearchAlgorithm', 'VariantA'): 0.04,
    ('SearchAlgorithm', 'VariantB'): -0.02,
    ('PricingPage', 'Control'): 0.0,
    ('PricingPage', 'VariantA'): 0.01,
    ('PricingPage', 'VariantB'): -0.005,
}

# Demographic effects
demographic_effects = {
    'Desktop': 0.01,
    'USA': 0.005,
    '18-24': -0.005,
    '45+': 0.01,
}

# List to store new purchase events
purchase_events = []
purchase_event_id_counter = user_events_df['event_id'].max() + 1

# Convert users_df signup_date to datetime for merging
users_df['signup_date'] = pd.to_datetime(users_df['signup_date'])
ab_test_assignments_df['assignment_date'] = pd.to_datetime(ab_test_assignments_df['assignment_date'])

# Merge user demographics into assignments for calculating conversion probability
assignment_with_demographics = ab_test_assignments_df.merge(
    users_df[['user_id', 'country', 'device_type', 'age_group']], on='user_id', how='left'
)

for _, row in assignment_with_demographics.iterrows():
    user_id = row['user_id']
    assignment_date = row['assignment_date']
    test_name = row['test_name']
    variant_name = row['variant_name']
    country = row['country']
    device_type = row['device_type']
    age_group = row['age_group']

    current_prob = base_conversion_rate

    # Apply variant effect
    current_prob += variant_effects.get((test_name, variant_name), 0.0)

    # Apply demographic effects
    current_prob += demographic_effects.get(device_type, 0.0)
    current_prob += demographic_effects.get(country, 0.0)
    current_prob += demographic_effects.get(age_group, 0.0)
    
    current_prob = max(0.01, min(0.3, current_prob)) # Cap probabilities

    if random.random() < current_prob:
        # User converts
        purchase_timestamp = assignment_date + datetime.timedelta(
            days=random.randint(0, 6),
            hours=random.randint(0, 23),
            minutes=random.randint(0, 59),
            seconds=random.randint(0, 59)
        )
        
        purchase_events.append({
            'event_id': purchase_event_id_counter,
            'user_id': user_id,
            'event_timestamp': purchase_timestamp,
            'event_type': 'purchase',
            'revenue': round(random.uniform(5.0, 500.0), 2)
        })
        purchase_event_id_counter += 1

# Append generated purchase events to user_events_df
if purchase_events:
    purchase_events_df = pd.DataFrame(purchase_events)
    user_events_df = pd.concat([user_events_df, purchase_events_df], ignore_index=True)

# Sort user_events_df
user_events_df['event_timestamp'] = pd.to_datetime(user_events_df['event_timestamp'])
user_events_df = user_events_df.sort_values(by=['user_id', 'event_timestamp']).reset_index(drop=True)

print(f"Generated {len(users_df)} users, {len(ab_test_assignments_df)} assignments, {len(user_events_df)} events.")
print(f"Total purchases simulated: {len(purchase_events)}")

# Convert date columns to appropriate types
users_df['signup_date'] = users_df['signup_date'].dt.strftime('%Y-%m-%d')
ab_test_assignments_df['assignment_date'] = ab_test_assignments_df['assignment_date'].dt.strftime('%Y-%m-%d')
user_events_df['event_timestamp'] = user_events_df['event_timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')

# --- 2. Load into SQLite & SQL Feature Engineering ---

print("\n--- Performing SQL Feature Engineering ---")

conn = sqlite3.connect(':memory:')

users_df.to_sql('users', conn, if_exists='replace', index=False)
ab_test_assignments_df.to_sql('assignments', conn, if_exists='replace', index=False)
user_events_df.to_sql('events', conn, if_exists='replace', index=False)

sql_query = """
SELECT
    a.assignment_id,
    a.user_id,
    a.test_name,
    a.variant_name,
    a.assignment_date,
    u.country,
    u.device_type,
    u.age_group,
    SUM(CASE WHEN e.event_type = 'page_view' THEN 1 ELSE 0 END) AS num_page_views_7d,
    SUM(CASE WHEN e.event_type = 'click' THEN 1 ELSE 0 END) AS num_clicks_7d,
    SUM(CASE WHEN e.event_type = 'add_to_cart' THEN 1 ELSE 0 END) AS num_add_to_carts_7d,
    SUM(CASE WHEN e.event_type = 'purchase' THEN 1 ELSE 0 END) AS total_purchases_7d,
    SUM(e.revenue) AS total_revenue_7d,
    COUNT(e.event_id) AS total_events_7d
FROM
    assignments AS a
JOIN
    users AS u ON a.user_id = u.user_id
LEFT JOIN
    events AS e ON a.user_id = e.user_id
    AND julianday(e.event_timestamp) >= julianday(a.assignment_date)
    AND julianday(e.event_timestamp) < julianday(a.assignment_date) + 7
GROUP BY
    a.assignment_id, a.user_id, a.test_name, a.variant_name, a.assignment_date,
    u.country, u.device_type, u.age_group
ORDER BY
    a.assignment_id;
"""

ab_features_df = pd.read_sql_query(sql_query, conn)
conn.close()

print(f"Fetched {len(ab_features_df)} rows from SQL query.")
print("Columns from SQL query:")
print(ab_features_df.columns.tolist())

# --- 3. Pandas Feature Engineering & Binary Target Creation ---

print("\n--- Performing Pandas Feature Engineering ---")

# Merge signup_date for 'days_since_signup_at_assignment'
ab_features_df = ab_features_df.merge(
    users_df[['user_id', 'signup_date']], on='user_id', how='left'
)

# Convert date columns to datetime objects
ab_features_df['signup_date'] = pd.to_datetime(ab_features_df['signup_date'])
ab_features_df['assignment_date'] = pd.to_datetime(ab_features_df['assignment_date'])

# Fill NaN values for aggregated numerical features with 0
numerical_agg_cols = [
    'num_page_views_7d', 'num_clicks_7d', 'num_add_to_carts_7d',
    'total_purchases_7d', 'total_revenue_7d', 'total_events_7d'
]
for col in numerical_agg_cols:
    ab_features_df[col] = ab_features_df[col].fillna(0).astype(int if 'total_revenue' not in col else float) # Use int for counts, float for revenue

# Calculate days_since_signup_at_assignment
ab_features_df['days_since_signup_at_assignment'] = (
    ab_features_df['assignment_date'] - ab_features_df['signup_date']
).dt.days.fillna(0).astype(int)

# Calculate ratio features, handle division by zero
epsilon = 1e-6 # Small constant to avoid division by zero

ab_features_df['click_through_rate_7d'] = ab_features_df['num_clicks_7d'] / (ab_features_df['num_page_views_7d'] + epsilon)
ab_features_df['add_to_cart_rate_7d'] = ab_features_df['num_add_to_carts_7d'] / (ab_features_df['num_page_views_7d'] + epsilon)
ab_features_df['revenue_per_event_7d'] = ab_features_df['total_revenue_7d'] / (ab_features_df['total_events_7d'] + epsilon)

# Fill NaN or inf values resulting from division with 0
ab_features_df = ab_features_df.replace([np.inf, -np.inf], np.nan).fillna(0)


# Create the Binary Target `is_purchased_7d`
ab_features_df['is_purchased_7d'] = (ab_features_df['total_purchases_7d'] > 0).astype(int)

print(f"Total entries in final feature set: {len(ab_features_df)}")
print(f"Conversion rate (is_purchased_7d): {ab_features_df['is_purchased_7d'].mean():.2%}")

# Define features (X) and target (y)
numerical_features = [
    'num_page_views_7d', 'num_clicks_7d', 'num_add_to_carts_7d',
    'total_revenue_7d', 'total_events_7d', 'days_since_signup_at_assignment',
    'click_through_rate_7d', 'add_to_cart_rate_7d', 'revenue_per_event_7d'
]
categorical_features = [
    'test_name', 'variant_name', 'country', 'device_type', 'age_group'
]

X = ab_features_df[numerical_features + categorical_features]
y = ab_features_df['is_purchased_7d']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"Training set size: {len(X_train)} samples")
print(f"Test set size: {len(X_test)} samples")


# --- 4. Data Visualization ---

print("\n--- Generating Data Visualizations ---")

# Set a style for the plots
sns.set_style("whitegrid")

# Plot 1: Violin plot of total_revenue_7d for purchasers vs. non-purchasers
plt.figure(figsize=(10, 6))
sns.violinplot(
    x='is_purchased_7d',
    y='total_revenue_7d',
    data=ab_features_df,
    inner='quartile',
    palette='pastel'
)
plt.title('Distribution of Total Revenue in 7 Days by Purchase Status')
plt.xlabel('Purchased (0: No, 1: Yes)')
plt.ylabel('Total Revenue in 7 Days')
plt.xticks(ticks=[0, 1], labels=['Non-Purchaser', 'Purchaser'])
plt.yscale('log') # Use log scale for revenue distribution
plt.tight_layout()
plt.savefig('total_revenue_by_purchase_status_violin.png')
plt.close()
print("Saved violin plot to 'total_revenue_by_purchase_status_violin.png'")


# Plot 2: Stacked bar chart for conversion rate by variant_name within a specific test_name
target_test = 'HomepageRedesign' # Choose one test name for demonstration

# Filter data for the target test
filtered_df = ab_features_df[ab_features_df['test_name'] == target_test]

if not filtered_df.empty:
    # Calculate proportions
    variant_conversion_proportions = filtered_df.groupby('variant_name')['is_purchased_7d'].value_counts(normalize=True).unstack(fill_value=0)

    # Ensure both 0 and 1 columns exist, fill if not
    if 0 not in variant_conversion_proportions.columns:
        variant_conversion_proportions[0] = 0
    if 1 not in variant_conversion_proportions.columns:
        variant_conversion_proportions[1] = 0

    # Sort columns for consistent stacking (0=No Purchase, 1=Purchase)
    variant_conversion_proportions = variant_conversion_proportions[[0, 1]]

    plt.figure(figsize=(10, 6))
    variant_conversion_proportions.plot(
        kind='bar',
        stacked=True,
        color=['lightcoral', 'lightgreen'],
        ax=plt.gca()
    )
    plt.title(f'Proportion of Purchase Status by Variant for {target_test}')
    plt.xlabel('Variant Name')
    plt.ylabel('Proportion')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Purchased in 7d', labels=['No', 'Yes'], bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('conversion_rate_by_variant_stacked_bar.png')
    plt.close()
    print(f"Saved stacked bar chart for '{target_test}' to 'conversion_rate_by_variant_stacked_bar.png'")
else:
    print(f"No data for test_name '{target_test}' to generate stacked bar chart.")


# --- 5. ML Pipeline & Evaluation (Binary Classification) ---

print("\n--- Building and Evaluating ML Pipeline ---")

# Preprocessing steps for numerical and categorical features
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
    ],
    remainder='passthrough' # Keep any other columns not specified
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

# Predict hard labels for the classification report (using default threshold 0.5)
y_pred = pipeline.predict(X_test)

# Evaluate the model
roc_auc = roc_auc_score(y_test, y_pred_proba)
class_report = classification_report(y_test, y_pred)

print("\n--- Model Evaluation Results ---")
print(f"ROC AUC Score: {roc_auc:.4f}")
print("\nClassification Report:\n", class_report)

print("\n--- Script Finished ---")