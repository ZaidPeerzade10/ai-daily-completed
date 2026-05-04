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
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import warnings

# Suppress specific warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

def generate_synthetic_data():
    """
    Generates synthetic customer and transaction dataframes.
    """
    np.random.seed(42)

    # 1. Generate customers_df
    num_customers = np.random.randint(1000, 1500)
    customer_ids = np.arange(1, num_customers + 1)
    
    signup_start = pd.Timestamp.now() - pd.DateOffset(years=5)
    signup_end = pd.Timestamp.now() - pd.DateOffset(months=12) # Ensure ample transaction history
    signup_dates = pd.to_datetime(signup_start + (signup_end - signup_start) * np.random.rand(num_customers))

    regions = np.random.choice(['North', 'South', 'East', 'West'], num_customers, p=[0.25, 0.25, 0.25, 0.25])
    ages = np.random.randint(18, 71, num_customers)
    initial_channels = np.random.choice(['Web', 'Mobile App', 'Referral', 'Email'], num_customers, p=[0.4, 0.3, 0.2, 0.1])

    customers_df = pd.DataFrame({
        'customer_id': customer_ids,
        'signup_date': signup_dates,
        'region': regions,
        'age': ages,
        'initial_channel': initial_channels
    })

    # 2. Generate transactions_df
    num_transactions = np.random.randint(20000, 30000)
    transaction_ids = np.arange(1, num_transactions + 1)
    
    # Sample customer IDs ensuring each customer has at least one transaction
    transaction_customer_ids = np.random.choice(customer_ids, num_transactions - num_customers, replace=True)
    transaction_customer_ids = np.concatenate((customer_ids, transaction_customer_ids))
    np.random.shuffle(transaction_customer_ids)

    # Determine current_max_date for transactions
    current_max_date = pd.Timestamp.now() - pd.DateOffset(weeks=4)

    transactions_data = []
    for i in range(num_transactions):
        cust_id = transaction_customer_ids[i]
        customer_signup_date = customers_df[customers_df['customer_id'] == cust_id]['signup_date'].iloc[0]

        # Ensure transaction_date is after signup_date
        date_range_start = customer_signup_date + pd.Timedelta(days=1)
        if date_range_start >= current_max_date: # If signup is too recent, skip transaction or adjust
            date_range_start = current_max_date - pd.Timedelta(days=30) # fallback
            if date_range_start < customer_signup_date: date_range_start = customer_signup_date + pd.Timedelta(days=1)
        
        transaction_date = pd.to_datetime(date_range_start + (current_max_date - date_range_start) * np.random.rand())
        
        amount = np.random.uniform(5.0, 1000.0)
        product_category = np.random.choice(['Electronics', 'Books', 'Groceries', 'Services', 'Apparel'], 1, 
                                            p=[0.2, 0.2, 0.2, 0.2, 0.2])[0]

        transactions_data.append({
            'transaction_id': transaction_ids[i],
            'customer_id': cust_id,
            'transaction_date': transaction_date,
            'amount': amount,
            'product_category': product_category
        })
    transactions_df = pd.DataFrame(transactions_data)
    
    # Apply realistic patterns
    # Older customers (age > 45) might have slightly higher average amount
    transactions_df = transactions_df.merge(customers_df[['customer_id', 'age']], on='customer_id', how='left')
    transactions_df.loc[transactions_df['age'] > 45, 'amount'] = transactions_df.loc[transactions_df['age'] > 45, 'amount'] * np.random.uniform(1.05, 1.2, transactions_df.loc[transactions_df['age'] > 45].shape[0])
    
    # 'Mobile App' users might have more frequent but smaller amount transactions
    mobile_users = customers_df[customers_df['initial_channel'] == 'Mobile App']['customer_id'].unique()
    transactions_df.loc[transactions_df['customer_id'].isin(mobile_users), 'amount'] = transactions_df.loc[transactions_df['customer_id'].isin(mobile_users), 'amount'] * np.random.uniform(0.8, 0.95, transactions_df.loc[transactions_df['customer_id'].isin(mobile_users)].shape[0])
    # To simulate frequency for mobile users, we might add more transactions for them, but for simplicity, we'll just adjust amounts.

    # 'Electronics' and 'Services' categories might have higher amount on average
    transactions_df.loc[transactions_df['product_category'].isin(['Electronics', 'Services']), 'amount'] = transactions_df.loc[transactions_df['product_category'].isin(['Electronics', 'Services']), 'amount'] * np.random.uniform(1.1, 1.3, transactions_df.loc[transactions_df['product_category'].isin(['Electronics', 'Services'])].shape[0])

    # A small percentage (e.g., 5-10%) of customers could have significantly higher amounts
    high_value_customer_ids = np.random.choice(customer_ids, int(num_customers * 0.07), replace=False)
    transactions_df.loc[transactions_df['customer_id'].isin(high_value_customer_ids), 'amount'] = transactions_df.loc[transactions_df['customer_id'].isin(high_value_customer_ids), 'amount'] * np.random.uniform(2.0, 5.0, transactions_df.loc[transactions_df['customer_id'].isin(high_value_customer_ids)].shape[0])

    transactions_df.drop(columns=['age'], inplace=True) # remove temporary age column
    
    transactions_df['amount'] = transactions_df['amount'].round(2)
    
    # Sort transactions_df
    transactions_df = transactions_df.sort_values(by=['customer_id', 'transaction_date']).reset_index(drop=True)

    print(f"Generated {len(customers_df)} customers and {len(transactions_df)} transactions.")
    return customers_df, transactions_df

def run_pipeline():
    # 1. Generate Synthetic Data
    customers_df, transactions_df = generate_synthetic_data()

    # 2. Load into SQLite & SQL Feature Engineering
    conn = sqlite3.connect(':memory:')
    customers_df.to_sql('customers', conn, index=False, if_exists='replace')
    transactions_df.to_sql('transactions', conn, index=False, if_exists='replace', 
                           dtype={'transaction_date': 'TEXT'}) # Store dates as TEXT for SQLite

    # Convert transaction_date back to datetime for pandas operations later if needed
    # (But for SQL, it's better to store as TEXT and use SQLite's date functions)
    transactions_df['transaction_date'] = pd.to_datetime(transactions_df['transaction_date'])

    # Define GLOBAL_PREDICTION_CUTOFF_DATE
    # Corrected: use pd.DateOffset for month arithmetic
    GLOBAL_PREDICTION_CUTOFF_DATE = transactions_df['transaction_date'].max() - pd.DateOffset(months=6)
    GLOBAL_PREDICTION_CUTOFF_DATE_STR = GLOBAL_PREDICTION_CUTOFF_DATE.strftime('%Y-%m-%d %H:%M:%S')
    PREV_6M_START_DATE_STR = (GLOBAL_PREDICTION_CUTOFF_DATE - pd.DateOffset(months=6)).strftime('%Y-%m-%d %H:%M:%S')
    TARGET_6M_END_DATE = GLOBAL_PREDICTION_CUTOFF_DATE + pd.DateOffset(months=6)
    TARGET_6M_END_DATE_STR = TARGET_6M_END_DATE.strftime('%Y-%m-%d %H:%M:%S')

    print(f"\nGLOBAL_PREDICTION_CUTOFF_DATE: {GLOBAL_PREDICTION_CUTOFF_DATE}")
    print(f"Historical 6-month window starts: {PREV_6M_START_DATE_STR}")
    print(f"Target 6-month window ends: {TARGET_6M_END_DATE_STR}")

    # SQL Query for historical feature engineering
    sql_query = f"""
    SELECT
        c.customer_id,
        c.signup_date,
        c.region,
        c.age,
        c.initial_channel,
        '{GLOBAL_PREDICTION_CUTOFF_DATE_STR}' AS current_cutoff_date,
        COALESCE(SUM(CASE WHEN t.transaction_date BETWEEN '{PREV_6M_START_DATE_STR}' AND '{GLOBAL_PREDICTION_CUTOFF_DATE_STR}' THEN t.amount ELSE 0 END), 0) AS total_spend_prev_6m,
        COALESCE(COUNT(CASE WHEN t.transaction_date BETWEEN '{PREV_6M_START_DATE_STR}' AND '{GLOBAL_PREDICTION_CUTOFF_DATE_STR}' THEN t.transaction_id ELSE NULL END), 0) AS num_transactions_prev_6m,
        COALESCE(AVG(CASE WHEN t.transaction_date BETWEEN '{PREV_6M_START_DATE_STR}' AND '{GLOBAL_PREDICTION_CUTOFF_DATE_STR}' THEN t.amount ELSE NULL END), 0) AS avg_transaction_value_prev_6m,
        CASE
            WHEN MAX(CASE WHEN t.transaction_date <= '{GLOBAL_PREDICTION_CUTOFF_DATE_STR}' THEN t.transaction_date ELSE NULL END) IS NOT NULL
            THEN CAST(JULIANDAY('{GLOBAL_PREDICTION_CUTOFF_DATE_STR}') - JULIANDAY(MAX(CASE WHEN t.transaction_date <= '{GLOBAL_PREDICTION_CUTOFF_DATE_STR}' THEN t.transaction_date ELSE NULL END)) AS INTEGER)
            ELSE 9999
        END AS days_since_last_transaction_at_cutoff,
        COALESCE(COUNT(DISTINCT CASE WHEN t.transaction_date BETWEEN '{PREV_6M_START_DATE_STR}' AND '{GLOBAL_PREDICTION_CUTOFF_DATE_STR}' THEN t.product_category ELSE NULL END), 0) AS num_unique_categories_prev_6m
    FROM
        customers c
    LEFT JOIN
        transactions t ON c.customer_id = t.customer_id
    GROUP BY
        c.customer_id, c.signup_date, c.region, c.age, c.initial_channel
    ORDER BY
        c.customer_id;
    """

    customer_features_df = pd.read_sql_query(sql_query, conn)
    conn.close()

    print("\n--- SQL Feature Engineering Results (first 5 rows) ---")
    print(customer_features_df.head())
    print(f"Shape of customer_features_df: {customer_features_df.shape}")

    # 3. Pandas Feature Engineering & Regression Target Creation
    customer_features_df['signup_date'] = pd.to_datetime(customer_features_df['signup_date'])
    customer_features_df['current_cutoff_date'] = pd.to_datetime(customer_features_df['current_cutoff_date'])

    # Handle NaN values (mostly done by COALESCE in SQL, but good to double check)
    customer_features_df['total_spend_prev_6m'] = customer_features_df['total_spend_prev_6m'].fillna(0)
    customer_features_df['num_transactions_prev_6m'] = customer_features_df['num_transactions_prev_6m'].fillna(0)
    customer_features_df['avg_transaction_value_prev_6m'] = customer_features_df['avg_transaction_value_prev_6m'].fillna(0)
    customer_features_df['num_unique_categories_prev_6m'] = customer_features_df['num_unique_categories_prev_6m'].fillna(0)
    # days_since_last_transaction_at_cutoff is handled by SQL CASE WHEN/ELSE

    # Calculate customer_age_at_cutoff_days
    customer_features_df['customer_age_at_cutoff_days'] = (customer_features_df['current_cutoff_date'] - customer_features_df['signup_date']).dt.days

    # Calculate avg_daily_spend_prev_6m
    customer_features_df['avg_daily_spend_prev_6m'] = customer_features_df['total_spend_prev_6m'] / 180.0
    customer_features_df['avg_daily_spend_prev_6m'] = customer_features_df['avg_daily_spend_prev_6m'].replace([np.inf, -np.inf], np.nan).fillna(0)

    # Create the Regression Target `next_6m_spend`
    # Filter transactions for the target window
    target_transactions_df = transactions_df[
        (transactions_df['transaction_date'] > GLOBAL_PREDICTION_CUTOFF_DATE) &
        (transactions_df['transaction_date'] <= TARGET_6M_END_DATE)
    ].copy()

    next_6m_spend = target_transactions_df.groupby('customer_id')['amount'].sum().reset_index()
    next_6m_spend.rename(columns={'amount': 'next_6m_spend'}, inplace=True)

    customer_features_df = pd.merge(customer_features_df, next_6m_spend, on='customer_id', how='left')
    customer_features_df['next_6m_spend'] = customer_features_df['next_6m_spend'].fillna(0)

    print("\n--- Pandas Feature Engineering & Target Creation (first 5 rows) ---")
    print(customer_features_df.head())
    print(f"Target variable statistics:\n{customer_features_df['next_6m_spend'].describe()}")

    # Define features X and target y
    numerical_features = [
        'age', 'total_spend_prev_6m', 'num_transactions_prev_6m',
        'avg_transaction_value_prev_6m', 'days_since_last_transaction_at_cutoff',
        'num_unique_categories_prev_6m', 'customer_age_at_cutoff_days',
        'avg_daily_spend_prev_6m'
    ]
    categorical_features = ['region', 'initial_channel']

    X = customer_features_df[numerical_features + categorical_features]
    y = customer_features_df['next_6m_spend']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    print(f"\nTraining data shape: {X_train.shape}, {y_train.shape}")
    print(f"Testing data shape: {X_test.shape}, {y_test.shape}")

    # 4. Data Visualization
    print("\n--- Generating visualizations ---")
    plt.figure(figsize=(14, 6))

    # Scatter plot: total_spend_prev_6m vs. next_6m_spend
    plt.subplot(1, 2, 1)
    # Applying log transformation for better visualization of skewed data
    sns.scatterplot(x=np.log1p(customer_features_df['total_spend_prev_6m']), 
                    y=np.log1p(customer_features_df['next_6m_spend']), alpha=0.6)
    plt.title('Log(Total Spend Prev 6m) vs. Log(Next 6m Spend)')
    plt.xlabel('Log(Total Spend Previous 6 Months + 1)')
    plt.ylabel('Log(Next 6 Months Spend + 1)')
    plt.grid(True, linestyle='--', alpha=0.7)

    # Box plot: next_6m_spend across initial_channel
    plt.subplot(1, 2, 2)
    sns.boxplot(x='initial_channel', y=np.log1p(customer_features_df['next_6m_spend']), data=customer_features_df)
    plt.title('Log(Next 6 Months Spend) by Initial Channel')
    plt.xlabel('Initial Channel')
    plt.ylabel('Log(Next 6 Months Spend + 1)')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()

    # 5. ML Pipeline & Evaluation
    print("\n--- Building and training ML pipeline ---")

    # Create preprocessing pipelines for numerical and categorical features
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
        remainder='passthrough' # Keep other columns (shouldn't be any in this case)
    )

    # Create the full pipeline
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', HistGradientBoostingRegressor(random_state=42))
    ])

    # Train the pipeline
    model_pipeline.fit(X_train, y_train)

    # Predict on the test set
    y_pred = model_pipeline.predict(X_test)

    # Ensure predictions are non-negative
    y_pred[y_pred < 0] = 0

    # Evaluate the model
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\n--- Model Evaluation Results ---")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"R-squared (R2): {r2:.2f}")

if __name__ == "__main__":
    run_pipeline()