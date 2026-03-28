import pandas as pd
import numpy as np
import sqlite3
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer # Corrected import for SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# --- 1. Generate Synthetic Data ---
print("--- 1. Generating Synthetic Data ---")

# Properties DataFrame
num_properties = np.random.randint(300, 501)
property_ids = np.arange(1, num_properties + 1)

built_years = np.random.randint(1900, 2021, num_properties)
square_footage = np.random.randint(800, 5001, num_properties)
num_bedrooms = np.random.randint(1, 7, num_properties)
num_bathrooms_choices = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
num_bathrooms = np.random.choice(num_bathrooms_choices, num_properties)
property_types = np.random.choice(['House', 'Condo', 'Townhouse'], num_properties, p=[0.6, 0.3, 0.1])
neighborhoods = np.random.choice(['Downtown', 'Suburban_East', 'Suburban_West', 'Rural', 'Uptown'], num_properties, p=[0.2, 0.3, 0.25, 0.15, 0.1])

properties_df = pd.DataFrame({
    'property_id': property_ids,
    'built_year': built_years,
    'square_footage': square_footage,
    'num_bedrooms': num_bedrooms,
    'num_bathrooms': num_bathrooms,
    'property_type': property_types,
    'neighborhood': neighborhoods
})

# Transactions DataFrame
num_transactions = np.random.randint(5000, 8001)
transaction_ids = np.arange(1, num_transactions + 1)

transactions_data = []
min_sale_date = datetime.date.today() - datetime.timedelta(days=20 * 365) # Last 20 years

# Create transactions ensuring strictly increasing sale_dates and multiple sales
property_id_counts = {}
for _ in range(num_transactions):
    # Select a property, biased towards properties with fewer sales currently
    available_properties = [pid for pid in property_ids if property_id_counts.get(pid, 0) < 5] # Cap sales per property
    if not available_properties:
        available_properties = property_ids # If all properties have >5 sales, allow more

    pid = np.random.choice(available_properties)
    prop_info = properties_df[properties_df['property_id'] == pid].iloc[0]

    # Determine last sale date for this property
    last_sale_date_str = None
    if pid in property_id_counts:
        # Get the last sale date for this property from transactions_data list
        # Filter for current property_id and get max sale_date
        prop_transactions = [t for t in transactions_data if t[1] == pid]
        if prop_transactions:
            last_sale_date_str = max(prop_transactions, key=lambda x: x[2])[2] # x[2] is sale_date

    if last_sale_date_str:
        last_sale_date = datetime.datetime.strptime(last_sale_date_str, '%Y-%m-%d').date()
    else:
        # First sale for this property, must be after built_year
        built_year_date = datetime.date(int(prop_info['built_year']), 1, 1)
        # Ensure first sale is not too far in the past if built_year is very old
        start_bound = max(built_year_date, min_sale_date)
        last_sale_date = start_bound - datetime.timedelta(days=1) # So first new sale_date is after this

    # Generate a new sale_date strictly after the last one
    days_to_add = np.random.randint(30, 365 * 5) # Sale can be 1 month to 5 years later
    new_sale_date = last_sale_date + datetime.timedelta(days=days_to_add)

    # Ensure sale_date is not in the future
    if new_sale_date > datetime.date.today():
        new_sale_date = datetime.date.today() - datetime.timedelta(days=np.random.randint(1, 30)) # Make it recent past

    # Ensure sale_date is after built_year (redundant due to last_sale_date logic, but good sanity)
    if new_sale_date < datetime.date(int(prop_info['built_year']), 1, 1):
        new_sale_date = datetime.date(int(prop_info['built_year']), 1, 1) + datetime.timedelta(days=np.random.randint(30, 365))

    # Calculate base price influenced by attributes
    base_price = (prop_info['square_footage'] * 150 +
                  prop_info['num_bedrooms'] * 10000 +
                  prop_info['num_bathrooms'] * 5000 +
                  (2023 - prop_info['built_year']) * 500) # Newer properties slightly higher

    # Neighborhood influence
    if prop_info['neighborhood'] == 'Downtown':
        base_price *= 1.5
    elif prop_info['neighborhood'] == 'Uptown':
        base_price *= 1.3
    elif prop_info['neighborhood'] == 'Suburban_East':
        base_price *= 1.1
    elif prop_info['neighborhood'] == 'Suburban_West':
        base_price *= 1.05
    else: # Rural
        base_price *= 0.8

    # Time-based appreciation (linear for simplicity, could be more complex)
    appreciation_factor = 1 + (new_sale_date - min_sale_date).days / (365 * 20) * 0.5 # Up to 50% appreciation over 20 years
    sale_price = base_price * appreciation_factor
    sale_price = max(100000, min(2000000, sale_price + np.random.normal(0, sale_price * 0.05))) # Add some noise

    transactions_data.append((len(transactions_data) + 1, pid, new_sale_date.strftime('%Y-%m-%d'), sale_price))
    property_id_counts[pid] = property_id_counts.get(pid, 0) + 1


transactions_df = pd.DataFrame(transactions_data, columns=['transaction_id', 'property_id', 'sale_date', 'sale_price'])
transactions_df['sale_date'] = pd.to_datetime(transactions_df['sale_date'])

# Sort transactions_df as required
transactions_df = transactions_df.sort_values(by=['property_id', 'sale_date']).reset_index(drop=True)

# Ensure at least two sales for a significant portion of properties
# If a property only has one sale, duplicate it with a slightly later date and higher price
# This loop needs to be robust, let's do it after initial generation for simplicity.
# Check properties with only one sale
one_sale_props = transactions_df['property_id'].value_counts()
one_sale_props = one_sale_props[one_sale_props == 1].index

new_transactions = []
for pid in one_sale_props:
    original_sale = transactions_df[transactions_df['property_id'] == pid].iloc[0]
    new_date = original_sale['sale_date'] + pd.Timedelta(days=np.random.randint(90, 730)) # 3 months to 2 years later
    if new_date > pd.to_datetime(datetime.date.today()):
        new_date = pd.to_datetime(datetime.date.today() - datetime.timedelta(days=30)) # ensure not future
    new_price = original_sale['sale_price'] * (1 + np.random.uniform(0.01, 0.15)) # 1-15% appreciation
    new_transaction_id = transactions_df['transaction_id'].max() + 1 + len(new_transactions)
    new_transactions.append({
        'transaction_id': new_transaction_id,
        'property_id': pid,
        'sale_date': new_date,
        'sale_price': new_price
    })

if new_transactions:
    transactions_df = pd.concat([transactions_df, pd.DataFrame(new_transactions)], ignore_index=True)
    transactions_df = transactions_df.sort_values(by=['property_id', 'sale_date']).reset_index(drop=True)

# Update transaction_ids to be unique and sequential after potential additions
transactions_df['transaction_id'] = np.arange(1, len(transactions_df) + 1)

print(f"Generated {len(properties_df)} properties and {len(transactions_df)} transactions.")
print("Sample properties_df head:")
print(properties_df.head())
print("\nSample transactions_df head:")
print(transactions_df.head())

# --- 2. Load into SQLite & SQL Feature Engineering ---
print("\n--- 2. Loading into SQLite & SQL Feature Engineering ---")

conn = sqlite3.connect(':memory:')
properties_df.to_sql('properties', conn, index=False, if_exists='replace')
transactions_df['sale_date'] = transactions_df['sale_date'].dt.strftime('%Y-%m-%d') # Convert to string for SQLite
transactions_df.to_sql('transactions', conn, index=False, if_exists='replace')

sql_query = """
WITH RankedTransactions AS (
    SELECT
        t.transaction_id,
        t.property_id,
        t.sale_date,
        t.sale_price,
        p.built_year,
        p.square_footage,
        p.num_bedrooms,
        p.num_bathrooms,
        p.property_type,
        p.neighborhood,
        LAG(t.sale_date, 1) OVER (PARTITION BY t.property_id ORDER BY t.sale_date) AS prev_sale_date,
        LAG(t.sale_price, 1) OVER (PARTITION BY t.property_id ORDER BY t.sale_date) AS prev_sale_price_for_prop,
        LEAD(t.sale_price, 1) OVER (PARTITION BY t.property_id ORDER BY t.sale_date) AS next_sale_price,
        ROW_NUMBER() OVER (PARTITION BY t.property_id ORDER BY t.sale_date) as rn
    FROM
        transactions t
    JOIN
        properties p ON t.property_id = p.property_id
)
SELECT
    rt.transaction_id,
    rt.property_id,
    rt.sale_date,
    rt.sale_price,
    rt.built_year,
    rt.square_footage,
    rt.num_bedrooms,
    rt.num_bathrooms,
    rt.property_type,
    rt.neighborhood,
    -- Target Variable
    rt.next_sale_price,

    -- Property-specific sequential features
    rt.rn - 1 AS property_prior_sales_count,
    COALESCE(
        (SELECT AVG(t2.sale_price)
         FROM transactions t2
         WHERE t2.property_id = rt.property_id
           AND t2.sale_date < rt.sale_date),
        0.0
    ) AS property_avg_prior_sale_price,
    COALESCE(
        julianday(rt.sale_date) - julianday(rt.prev_sale_date),
        julianday(rt.sale_date) - julianday(DATE(rt.built_year || '-01-01'))
    ) AS days_since_last_property_sale,

    -- Neighborhood-specific prior market activity
    COALESCE(
        (SELECT AVG(t3.sale_price)
         FROM transactions t3
         JOIN properties p3 ON t3.property_id = p3.property_id
         WHERE p3.neighborhood = rt.neighborhood
           AND t3.sale_date < rt.sale_date
           AND t3.property_id != rt.property_id), -- Exclude current property's sales
        0.0
    ) AS neighborhood_avg_price_prior_to_sale,
    COALESCE(
        (SELECT COUNT(t3.transaction_id)
         FROM transactions t3
         JOIN properties p3 ON t3.property_id = p3.property_id
         WHERE p3.neighborhood = rt.neighborhood
           AND t3.sale_date < rt.sale_date
           AND t3.property_id != rt.property_id), -- Exclude current property's sales
        0
    ) AS neighborhood_num_sales_prior_to_sale
FROM
    RankedTransactions rt
WHERE
    rt.next_sale_price IS NOT NULL
ORDER BY
    rt.property_id, rt.sale_date;
"""

property_features_df = pd.read_sql_query(sql_query, conn)
conn.close()

print(f"Generated {len(property_features_df)} feature rows from SQL.")
print("Sample property_features_df head after SQL engineering:")
print(property_features_df.head())

# --- 3. Pandas Feature Engineering & Regression Target Creation ---
print("\n--- 3. Pandas Feature Engineering & Regression Target Creation ---")

# Handle NaN values (mostly done by COALESCE in SQL, but good to ensure)
property_features_df['property_prior_sales_count'] = property_features_df['property_prior_sales_count'].fillna(0).astype(int)
property_features_df['property_avg_prior_sale_price'] = property_features_df['property_avg_prior_sale_price'].fillna(0.0)
property_features_df['neighborhood_avg_price_prior_to_sale'] = property_features_df['neighborhood_avg_price_prior_to_sale'].fillna(0.0)
property_features_df['neighborhood_num_sales_prior_to_sale'] = property_features_df['neighborhood_num_sales_prior_to_sale'].fillna(0).astype(int)

# Convert date columns
property_features_df['sale_date'] = pd.to_datetime(property_features_df['sale_date'])

# Calculate property_age_at_sale_days
property_features_df['built_date'] = pd.to_datetime(property_features_df['built_year'].astype(str) + '-01-01')
property_features_df['property_age_at_sale_days'] = (property_features_df['sale_date'] - property_features_df['built_date']).dt.days

# Fill any remaining NaNs in days_since_last_property_sale with property_age_at_sale_days for first sales
property_features_df['days_since_last_property_sale'] = property_features_df['days_since_last_property_sale'].fillna(
    property_features_df['property_age_at_sale_days']
)

# Calculate price_per_sqft_at_sale
property_features_df['price_per_sqft_at_sale'] = property_features_df['sale_price'] / property_features_df['square_footage']

# Calculate price_deviation_from_neighborhood_avg
# If neighborhood_avg_price_prior_to_sale is 0, use the property's sale_price as deviation from a 0 base, or a global average.
# For simplicity, we'll use 0.0, meaning the deviation IS the sale_price if neighborhood average is 0.
property_features_df['price_deviation_from_neighborhood_avg'] = property_features_df['sale_price'] - property_features_df['neighborhood_avg_price_prior_to_sale']
# If neighborhood_avg_price_prior_to_sale is 0, and we want deviation from global avg for those,
# we would need to calculate a global average. For this exercise, 0.0 as fillna makes the deviation `sale_price - 0`.

# Define features (X) and target (y)
numerical_features = [
    'sale_price', 'built_year', 'square_footage', 'num_bedrooms', 'num_bathrooms',
    'property_prior_sales_count', 'property_avg_prior_sale_price', 'days_since_last_property_sale',
    'neighborhood_avg_price_prior_to_sale', 'neighborhood_num_sales_prior_to_sale',
    'property_age_at_sale_days', 'price_per_sqft_at_sale', 'price_deviation_from_neighborhood_avg'
]
categorical_features = ['property_type', 'neighborhood']

X = property_features_df[numerical_features + categorical_features]
y = property_features_df['next_sale_price']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print(f"Dataset split: {len(X_train)} training samples, {len(X_test)} testing samples.")
print("\nSample X_train head:")
print(X_train.head())
print("\nSample y_train head:")
print(y_train.head())


# --- 4. Data Visualization ---
print("\n--- 4. Data Visualization ---")
plt.style.use('seaborn-v0_8-darkgrid')

plt.figure(figsize=(14, 6))

# Scatter plot: sale_price (current) vs. next_sale_price
plt.subplot(1, 2, 1)
sns.regplot(x='sale_price', y='next_sale_price', data=property_features_df, scatter_kws={'alpha':0.3})
plt.title('Current Sale Price vs. Next Sale Price')
plt.xlabel('Current Sale Price ($)')
plt.ylabel('Next Sale Price ($)')
plt.ticklabel_format(style='plain', axis='both')

# Box plot: distribution of next_sale_price across different neighborhoods
plt.subplot(1, 2, 2)
sns.boxplot(x='neighborhood', y='next_sale_price', data=property_features_df)
plt.title('Next Sale Price Distribution by Neighborhood')
plt.xlabel('Neighborhood')
plt.ylabel('Next Sale Price ($)')
plt.xticks(rotation=45, ha='right')
plt.ticklabel_format(style='plain', axis='y')
plt.tight_layout()
plt.show()


# --- 5. ML Pipeline & Evaluation (Regression) ---
print("\n--- 5. ML Pipeline & Evaluation (Regression) ---")

# Create preprocessing pipelines for numerical and categorical features
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')), # Corrected from sklearn.preprocessing
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

# Create the full pipeline
model_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                 ('regressor', HistGradientBoostingRegressor(random_state=42))])

# Train the pipeline
print("Training the model pipeline...")
model_pipeline.fit(X_train, y_train)
print("Model training complete.")

# Predict on the test set
y_pred = model_pipeline.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nModel Evaluation on Test Set:")
print(f"Mean Absolute Error (MAE): ${mae:,.2f}")
print(f"R-squared (R2 Score): {r2:.4f}")

# Optional: Plot predicted vs actual
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.title('Actual vs. Predicted Next Sale Price (Test Set)')
plt.xlabel('Actual Next Sale Price ($)')
plt.ylabel('Predicted Next Sale Price ($)')
plt.ticklabel_format(style='plain', axis='both')
plt.tight_layout()
plt.show()

print("\n--- Script Finished ---")