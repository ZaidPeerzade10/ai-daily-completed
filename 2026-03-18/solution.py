import pandas as pd
import numpy as np
import sqlite3
import datetime
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Ensure plots don't try to open separate windows in some environments
plt.switch_backend('Agg')

# --- 1. Generate Synthetic Data (Pandas/Numpy) ---

np.random.seed(42) # for reproducibility

# User data
N_users = np.random.randint(500, 701)
users_data = {
    'user_id': np.arange(N_users),
    'signup_date': pd.to_datetime('today') - pd.to_timedelta(np.random.rand(N_users) * 5 * 365, unit='D'),
    'segment': np.random.choice(['Budget', 'Standard', 'Premium'], N_users, p=[0.35, 0.45, 0.20]),
    'avg_annual_income': np.random.randint(30000, 200001, N_users)
}
users_df = pd.DataFrame(users_data)
# Ensure dates are normalized (no time component)
users_df['signup_date'] = users_df['signup_date'].dt.normalize()

# Product data
N_products = np.random.randint(100, 151)
categories = ['Electronics', 'Books', 'Home Goods', 'Apparel', 'Services']
category_weights = [0.2, 0.25, 0.2, 0.15, 0.2] # Distribution of categories
products_data = {
    'product_id': np.arange(N_products),
    'category': np.random.choice(categories, N_products, p=category_weights),
    'release_date': pd.to_datetime('today') - pd.to_timedelta(np.random.rand(N_products) * 4 * 365, unit='D')
}
products_df = pd.DataFrame(products_data)
products_df['release_date'] = products_df['release_date'].dt.normalize()

# Adjust unit_price based on category
def get_unit_price(category):
    if category == 'Electronics':
        return np.random.uniform(150.0, 900.0) # Higher price range
    elif category == 'Books':
        return np.random.uniform(10.0, 50.0)
    elif category == 'Home Goods':
        return np.random.uniform(30.0, 200.0)
    elif category == 'Apparel':
        return np.random.uniform(20.0, 150.0)
    elif category == 'Services':
        return np.random.uniform(50.0, 500.0)
    return np.random.uniform(10.0, 1000.0)

products_df['unit_price'] = products_df['category'].apply(get_unit_price)

# Purchases data - complex generation for realism
purchases_list = []
purchase_id_counter = 0

# Assign each user a base number of purchases, then add more for premium/high income
user_purchase_counts = pd.Series(np.random.randint(2, 15, N_users), index=users_df['user_id'])
user_purchase_counts = user_purchase_counts.astype(float) 

# Boost purchase counts for 'Premium' segment and higher income
user_purchase_counts = user_purchase_counts.mul(users_df.set_index('user_id')['segment'].map({'Budget': 1.0, 'Standard': 1.2, 'Premium': 1.5}))
user_purchase_counts = user_purchase_counts.mul(users_df.set_index('user_id')['avg_annual_income'] / users_df['avg_annual_income'].mean() * 0.8 + 0.6) # Income multiplier
user_purchase_counts = user_purchase_counts.round().astype(int)
user_purchase_counts = user_purchase_counts.clip(lower=2) # Ensure at least 2 purchases for target definition

# Target total purchases
N_purchases_target_range = np.random.randint(8000, 12001)

# Adjust if generated too few or too many purchases
total_generated = user_purchase_counts.sum()
if total_generated < N_purchases_target_range * 0.8 or total_generated > N_purchases_target_range * 1.2:
    factor = N_purchases_target_range / total_generated
    user_purchase_counts = (user_purchase_counts * factor).round().astype(int).clip(lower=2)

print(f"Generating purchases across {N_users} users with a target of {N_purchases_target_range} purchases...")

for user_idx, user_row in users_df.iterrows():
    user_id = user_row['user_id']
    signup_date = user_row['signup_date']
    user_avg_income_ratio = user_row['avg_annual_income'] / users_df['avg_annual_income'].mean()
    user_segment = user_row['segment']
    
    num_purchases = user_purchase_counts.loc[user_id]
    
    last_purchase_date = signup_date
    prior_categories = set()

    for _ in range(num_purchases):
        available_products = products_df.copy()
        
        # If user has prior purchases, bias towards their prior categories
        if prior_categories and np.random.rand() < 0.7: # 70% chance to repeat category
            product_category = np.random.choice(list(prior_categories))
            available_products = available_products[available_products['category'] == product_category]
            if available_products.empty: # Fallback if no products in prior category
                available_products = products_df.copy()
        
        if available_products.empty: # Fallback if still empty
             available_products = products_df.copy()

        # Select a product - prioritize products with higher unit_price for Premium users
        if user_segment == 'Premium' or user_avg_income_ratio > 1.2:
            product = available_products.sample(n=1, weights=available_products['unit_price']**1.5, random_state=np.random.randint(0, 1000)).iloc[0]
        else:
            product = available_products.sample(n=1, random_state=np.random.randint(0, 1000)).iloc[0]
            
        product_id = product['product_id']
        product_release_date = product['release_date']
        unit_price = product['unit_price']

        # Determine purchase date
        min_purchase_date = max(signup_date, product_release_date, last_purchase_date + pd.Timedelta(days=1))
        today_date = pd.to_datetime('today').normalize()
        
        # Ensure purchase date is not in the future relative to "today" for simulation purposes
        if min_purchase_date > today_date:
            min_purchase_date = today_date - pd.Timedelta(days=np.random.randint(0, 30)) 
            min_purchase_date = max(min_purchase_date, signup_date, product_release_date) # ensure not before signup/release
            
        days_spread = (today_date - min_purchase_date).days
        
        if days_spread < 1: 
            purchase_date = min_purchase_date
        else:
            random_days_delta = np.random.randint(0, days_spread + 1)
            purchase_date = min_purchase_date + pd.Timedelta(days=random_days_delta)
        
        purchase_date = purchase_date.normalize() # Ensure no time component

        # Quantity bias for Premium/higher income
        quantity = np.random.randint(1, 4)
        if user_segment == 'Premium' or user_avg_income_ratio > 1.2:
            quantity = np.random.randint(1, 6) # Higher quantity for premium

        # Calculate amount
        amount = quantity * unit_price
        
        # Further bias amount for Premium/high income
        if user_segment == 'Premium':
            amount *= np.random.uniform(1.1, 1.5) # Premium users spend more
        elif user_segment == 'Standard':
            amount *= np.random.uniform(1.0, 1.2)
        amount *= (user_avg_income_ratio * 0.5 + 0.75) # Income has a multiplier effect

        purchases_list.append({
            'purchase_id': purchase_id_counter,
            'user_id': user_id,
            'product_id': product_id,
            'purchase_date': purchase_date.normalize(),
            'quantity': quantity,
            'amount': amount
        })
        purchase_id_counter += 1
        last_purchase_date = purchase_date.normalize()
        prior_categories.add(product['category'])


purchases_df = pd.DataFrame(purchases_list)
purchases_df['amount'] = purchases_df['amount'].round(2)

# Sort purchases for sequential processing in SQL
purchases_df = purchases_df.sort_values(by=['user_id', 'purchase_date']).reset_index(drop=True)

print(f"Generated {len(purchases_df)} purchases.")
print(f"Users: {len(users_df)}, Products: {len(products_df)}, Purchases: {len(purchases_df)}")


# --- 2. Load into SQLite & SQL Feature Engineering ---
conn = sqlite3.connect(':memory:')

users_df.to_sql('users', conn, if_exists='replace', index=False, dtype={'signup_date': 'TEXT'})
products_df.to_sql('products', conn, if_exists='replace', index=False, dtype={'release_date': 'TEXT'})
purchases_df.to_sql('purchases', conn, if_exists='replace', index=False, dtype={'purchase_date': 'TEXT'})

# SQL Query for feature engineering and target creation
sql_query = """
WITH ranked_purchases AS (
    SELECT
        p.purchase_id,
        p.user_id,
        p.purchase_date,
        p.amount,
        p.quantity,
        p.product_id,
        pr.category,
        pr.unit_price,
        u.signup_date,
        u.segment,
        u.avg_annual_income,
        
        -- Lagged features for previous purchase date
        LAG(p.purchase_date, 1) OVER (PARTITION BY u.user_id ORDER BY p.purchase_date) AS prev_purchase_date,

        -- Aggregated features over prior purchases using window functions
        COUNT(p.purchase_id) OVER (PARTITION BY u.user_id ORDER BY p.purchase_date ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS user_prior_num_purchases,
        SUM(p.amount) OVER (PARTITION BY u.user_id ORDER BY p.purchase_date ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS user_prior_total_spend,
        AVG(p.amount) OVER (PARTITION BY u.user_id ORDER BY p.purchase_date ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS user_avg_prior_spend,
        
        -- Regression Target
        LEAD(p.amount, 1) OVER (PARTITION BY u.user_id ORDER BY p.purchase_date) AS next_purchase_amount
    FROM
        purchases p
    JOIN
        users u ON p.user_id = u.user_id
    JOIN
        products pr ON p.product_id = pr.product_id
),
final_features AS (
    SELECT
        rp.*,
        -- Calculate days_since_last_user_purchase:
        -- If it's not the first purchase (prev_purchase_date exists), use difference from prev_purchase_date.
        -- Otherwise (first purchase), use difference from signup_date.
        COALESCE(
            JULIANDAY(rp.purchase_date) - JULIANDAY(rp.prev_purchase_date),
            JULIANDAY(rp.purchase_date) - JULIANDAY(rp.signup_date)
        ) AS days_since_last_user_purchase,
        
        -- Calculate user_num_unique_categories_prior using a subquery for robustness across SQLite versions
        (SELECT COUNT(DISTINCT pr2.category)
         FROM purchases p2
         JOIN products pr2 ON p2.product_id = pr2.product_id
         WHERE p2.user_id = rp.user_id AND p2.purchase_date < rp.purchase_date) AS user_num_unique_categories_prior
    FROM
        ranked_purchases rp
)
SELECT
    ff.purchase_id,
    ff.user_id,
    ff.purchase_date,
    ff.amount,
    ff.quantity,
    ff.product_id,
    ff.category,
    ff.unit_price,
    ff.signup_date,
    ff.segment,
    ff.avg_annual_income,
    ff.user_prior_num_purchases,
    ff.user_prior_total_spend,
    ff.user_avg_prior_spend,
    ff.days_since_last_user_purchase,
    ff.user_num_unique_categories_prior,
    ff.next_purchase_amount
FROM
    final_features ff
WHERE
    ff.next_purchase_amount IS NOT NULL
ORDER BY
    ff.user_id, ff.purchase_date;
"""

purchase_features_df = pd.read_sql_query(sql_query, conn)
conn.close()

# --- 3. Pandas Feature Engineering & Regression Target Creation ---

# Handle NaN values
# Fill counts/sums with 0 for first purchases (where prior values would be NULL)
purchase_features_df['user_prior_num_purchases'] = purchase_features_df['user_prior_num_purchases'].fillna(0).astype(int)
purchase_features_df['user_prior_total_spend'] = purchase_features_df['user_prior_total_spend'].fillna(0.0)
purchase_features_df['user_avg_prior_spend'] = purchase_features_df['user_avg_prior_spend'].fillna(0.0)
purchase_features_df['user_num_unique_categories_prior'] = purchase_features_df['user_num_unique_categories_prior'].fillna(0).astype(int)

# Convert date columns to datetime objects
purchase_features_df['signup_date'] = pd.to_datetime(purchase_features_df['signup_date'])
purchase_features_df['purchase_date'] = pd.to_datetime(purchase_features_df['purchase_date'])

# Calculate days_since_signup_at_purchase
purchase_features_df['days_since_signup_at_purchase'] = (purchase_features_df['purchase_date'] - purchase_features_df['signup_date']).dt.days

# Fill remaining NaN in days_since_last_user_purchase (if any, e.g. for first purchases in edge cases not perfectly handled by SQL)
purchase_features_df['days_since_last_user_purchase'] = purchase_features_df['days_since_last_user_purchase'].fillna(
    purchase_features_df['days_since_signup_at_purchase']
)
# Ensure days_since_last_user_purchase is non-negative
purchase_features_df['days_since_last_user_purchase'] = purchase_features_df['days_since_last_user_purchase'].clip(lower=0)


# Calculate spend_ratio_to_avg_prior
# If user_avg_prior_spend is 0 (first purchase), the ratio is considered 1.0 (current amount is 1x prior avg)
purchase_features_df['spend_ratio_to_avg_prior'] = purchase_features_df.apply(
    lambda row: row['amount'] / row['user_avg_prior_spend'] if row['user_avg_prior_spend'] > 0 else 1.0, axis=1
)
# Handle potential inf or NaN from division by zero, or cases where amount was 0 for first purchase
purchase_features_df['spend_ratio_to_avg_prior'] = purchase_features_df['spend_ratio_to_avg_prior'].replace([np.inf, -np.inf], 0).fillna(0)


# Define features X and target y
numerical_features = [
    'amount', 'quantity', 'unit_price', 'avg_annual_income',
    'user_prior_num_purchases', 'user_prior_total_spend', 'user_avg_prior_spend',
    'days_since_last_user_purchase', 'user_num_unique_categories_prior',
    'days_since_signup_at_purchase', 'spend_ratio_to_avg_prior'
]
categorical_features = ['segment', 'category']
target = 'next_purchase_amount'

X = purchase_features_df[numerical_features + categorical_features]
y = purchase_features_df[target]

print(f"\nTotal records with target variable: {len(y)}")

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print(f"Shape of X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"Shape of X_test: {X_test.shape}, y_test: {y_test.shape}")


# --- 4. Data Visualization ---

print("\nGenerating visualizations...")
# Scatter plot: user_avg_prior_spend vs. next_purchase_amount
plt.figure(figsize=(10, 6))
sns.regplot(x='user_avg_prior_spend', y='next_purchase_amount', data=purchase_features_df,
            scatter_kws={'alpha':0.3, 's':10}, line_kws={'color':'red'})
plt.title('User Average Prior Spend vs. Next Purchase Amount')
plt.xlabel('User Average Prior Spend')
plt.ylabel('Next Purchase Amount')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('avg_prior_spend_vs_next_amount_regplot.png')
plt.close() # Close plot to free memory

# Box plot: next_purchase_amount across segments
plt.figure(figsize=(10, 6))
sns.boxplot(x='segment', y='next_purchase_amount', data=purchase_features_df, order=['Budget', 'Standard', 'Premium'])
plt.title('Distribution of Next Purchase Amount by User Segment')
plt.xlabel('User Segment')
plt.ylabel('Next Purchase Amount')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('next_purchase_amount_by_segment_boxplot.png')
plt.close() # Close plot to free memory
print("Visualizations saved as PNG files: 'avg_prior_spend_vs_next_amount_regplot.png' and 'next_purchase_amount_by_segment_boxplot.png'")


# --- 5. ML Pipeline & Evaluation (Regression) ---

# Preprocessing steps
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')), # Impute any potential NaNs from data gen/SQL
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Create a preprocessor using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Create the full pipeline
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', HistGradientBoostingRegressor(random_state=42))
])

# Train the model
print("\nTraining HistGradientBoostingRegressor model...")
model_pipeline.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model_pipeline.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\n--- Model Evaluation ---")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"R-squared (R2 Score): {r2:.2f}")

print("\nScript execution complete.")