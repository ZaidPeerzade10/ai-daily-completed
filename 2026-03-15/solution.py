import pandas as pd
import numpy as np
import datetime
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')

def generate_synthetic_data():
    """
    Generates synthetic customer, purchase, and browsing activity dataframes.
    Simulates realistic patterns including date dependencies and biases.
    """
    np.random.seed(42)
    
    # --- 1. Customers DataFrame ---
    num_customers = np.random.randint(500, 701)
    customer_ids = np.arange(1, num_customers + 1)
    
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=5*365) # Last 5 years
    
    signup_dates = [start_date + datetime.timedelta(days=np.random.randint(0, (end_date - start_date).days)) 
                    for _ in range(num_customers)]
    
    acquisition_channels = ['Organic', 'Paid_Social', 'Referral', 'Email']
    demographic_segments = ['Young_Adult', 'Middle_Age', 'Senior']
    
    customers_df = pd.DataFrame({
        'customer_id': customer_ids,
        'signup_date': signup_dates,
        'acquisition_channel': np.random.choice(acquisition_channels, num_customers, p=[0.3, 0.35, 0.2, 0.15]),
        'demographic_segment': np.random.choice(demographic_segments, num_customers, p=[0.4, 0.4, 0.2])
    })
    
    # Identify a pool of "high LTV" customers for biased data generation
    high_ltv_customer_pool = np.random.choice(customer_ids, size=int(num_customers * 0.15), replace=False)
    
    # --- 2. Purchases DataFrame ---
    num_purchases = np.random.randint(5000, 8001)
    purchase_ids = np.arange(1, num_purchases + 1)
    purchase_customer_ids = np.random.choice(customer_ids, num_purchases)
    
    # Temporarily merge with customers_df to get signup_date and channel for bias application
    temp_purchases_df = pd.DataFrame({'customer_id': purchase_customer_ids}).merge(
        customers_df[['customer_id', 'signup_date', 'acquisition_channel']], on='customer_id', how='left'
    )
    
    purchase_dates = []
    amounts = []
    product_categories = ['Electronics', 'Books', 'Groceries', 'Apparel', 'Services']
    
    for _, row in temp_purchases_df.iterrows():
        cid = row['customer_id']
        sdate = row['signup_date']
        channel = row['acquisition_channel']
        
        # Ensure purchase_date is after signup_date and not in the future
        days_after_signup = np.random.randint(1, (end_date - sdate).days + 1 if (end_date - sdate).days > 0 else 1)
        p_date = sdate + datetime.timedelta(days=days_after_signup)
        
        # Base amount
        amount = np.random.uniform(10.0, 1500.0)
        
        # Apply biases
        if channel == 'Referral': # Referral customers might have higher amounts
            amount *= np.random.uniform(1.1, 1.5)
        if cid in high_ltv_customer_pool: # High LTV customers might have higher amounts
            amount *= np.random.uniform(1.2, 1.8)
            
        amounts.append(amount)
        purchase_dates.append(p_date)

    purchases_df = pd.DataFrame({
        'purchase_id': purchase_ids,
        'customer_id': temp_purchases_df['customer_id'],
        'purchase_date': purchase_dates,
        'amount': amounts,
        'product_category': np.random.choice(product_categories, num_purchases, p=[0.2, 0.2, 0.3, 0.15, 0.15])
    })
    
    # --- 3. Browsing DataFrame ---
    num_browsing = np.random.randint(10000, 15001)
    browse_ids = np.arange(1, num_browsing + 1)
    browse_customer_ids = np.random.choice(customer_ids, num_browsing)
    
    # Temporarily merge with customers_df to get signup_date and channel for bias application
    temp_browsing_df = pd.DataFrame({'customer_id': browse_customer_ids}).merge(
        customers_df[['customer_id', 'signup_date', 'acquisition_channel']], on='customer_id', how='left'
    )
    
    browse_dates = []
    page_view_types = ['Product_Page', 'Category_Page', 'Homepage', 'Checkout_Page', 'Help_Page']
    time_on_page_seconds = []
    
    for _, row in temp_browsing_df.iterrows():
        cid = row['customer_id']
        sdate = row['signup_date']
        channel = row['acquisition_channel']
        
        # Ensure browse_date is after signup_date and not in the future
        days_after_signup = np.random.randint(1, (end_date - sdate).days + 1 if (end_date - sdate).days > 0 else 1)
        b_date = sdate + datetime.timedelta(days=days_after_signup, seconds=np.random.randint(0, 86400)) # Add random time
        
        time_on_page = np.random.randint(5, 301)
        
        # Paid_Social bias: higher initial browsing activity
        if channel == 'Paid_Social':
            time_on_page *= np.random.uniform(1.1, 1.3)
            time_on_page = min(300, int(time_on_page)) # Cap at max 300
        
        time_on_page_seconds.append(time_on_page)
        browse_dates.append(b_date)

    browsing_df = pd.DataFrame({
        'browse_id': browse_ids,
        'customer_id': temp_browsing_df['customer_id'],
        'browse_date': browse_dates,
        'page_view_type': np.random.choice(page_view_types, num_browsing, p=[0.4, 0.2, 0.2, 0.1, 0.1]),
        'time_on_page_seconds': time_on_page_seconds
    })

    # High LTV bias for browsing: More 'Product_Page'/'Checkout_Page' views
    for cid in high_ltv_customer_pool:
        customer_browsing_indices = browsing_df[browsing_df['customer_id'] == cid].index
        if not customer_browsing_indices.empty:
            # For 20% of their browsing events, force to Product_Page or Checkout_Page
            num_to_modify = int(len(customer_browsing_indices) * 0.2)
            if num_to_modify > 0:
                indices_to_modify = np.random.choice(customer_browsing_indices, num_to_modify, replace=False)
                browsing_df.loc[indices_to_modify, 'page_view_type'] = np.random.choice(['Product_Page', 'Checkout_Page'], num_to_modify)

    # Sort dataframes
    purchases_df = purchases_df.sort_values(by=['customer_id', 'purchase_date']).reset_index(drop=True)
    browsing_df = browsing_df.sort_values(by=['customer_id', 'browse_date']).reset_index(drop=True)
    
    # Convert date columns to string format for SQLite
    customers_df['signup_date'] = customers_df['signup_date'].dt.strftime('%Y-%m-%d')
    purchases_df['purchase_date'] = purchases_df['purchase_date'].dt.strftime('%Y-%m-%d')
    browsing_df['browse_date'] = browsing_df['browse_date'].dt.strftime('%Y-%m-%d %H:%M:%S')

    print("Synthetic Data Generation Complete:")
    print(f"  Customers: {len(customers_df)} rows")
    print(f"  Purchases: {len(purchases_df)} rows")
    print(f"  Browsing: {len(browsing_df)} rows")
    
    return customers_df, purchases_df, browsing_df

def sql_feature_engineering(customers_df, purchases_df, browsing_df):
    """
    Loads data into an in-memory SQLite DB and performs SQL feature engineering
    for early customer behavior (first 60 days).
    """
    conn = sqlite3.connect(':memory:')
    
    customers_df.to_sql('customers', conn, index=False, if_exists='replace')
    purchases_df.to_sql('purchases', conn, index=False, if_exists='replace')
    browsing_df.to_sql('browsing', conn, index=False, if_exists='replace')
    
    sql_query = """
    SELECT
        c.customer_id,
        c.signup_date,
        c.acquisition_channel,
        c.demographic_segment,
        
        -- Purchase features
        COALESCE(p_agg.num_purchases_first_60d, 0) AS num_purchases_first_60d,
        COALESCE(p_agg.total_spend_first_60d, 0.0) AS total_spend_first_60d,
        COALESCE(p_agg.avg_purchase_amount_first_60d, 0.0) AS avg_purchase_amount_first_60d,
        COALESCE(p_agg.num_unique_product_categories_first_60d, 0) AS num_unique_product_categories_first_60d,
        
        -- Days since first purchase
        CASE
            WHEN p_agg.first_purchase_date_first_60d IS NOT NULL
            THEN CAST(julianday(p_agg.first_purchase_date_first_60d) - julianday(c.signup_date) AS INTEGER)
            ELSE NULL
        END AS days_since_first_purchase_first_60d,

        -- Browsing features
        COALESCE(b_agg.num_browsing_events_first_60d, 0) AS num_browsing_events_first_60d,
        COALESCE(b_agg.total_browse_duration_first_60d, 0) AS total_browse_duration_first_60d,
        COALESCE(b_agg.has_browsed_checkout_page_first_60d, 0) AS has_browsed_checkout_page_first_60d

    FROM
        customers c
    LEFT JOIN (
        SELECT
            p.customer_id,
            COUNT(p.purchase_id) AS num_purchases_first_60d,
            SUM(p.amount) AS total_spend_first_60d,
            AVG(p.amount) AS avg_purchase_amount_first_60d,
            COUNT(DISTINCT p.product_category) AS num_unique_product_categories_first_60d,
            MIN(p.purchase_date) AS first_purchase_date_first_60d
        FROM
            purchases p
        INNER JOIN
            customers c_sub ON p.customer_id = c_sub.customer_id
        WHERE
            p.purchase_date BETWEEN c_sub.signup_date AND DATE(c_sub.signup_date, '+60 days')
        GROUP BY
            p.customer_id
    ) AS p_agg ON c.customer_id = p_agg.customer_id
    LEFT JOIN (
        SELECT
            b.customer_id,
            COUNT(b.browse_id) AS num_browsing_events_first_60d,
            SUM(b.time_on_page_seconds) AS total_browse_duration_first_60d,
            MAX(CASE WHEN b.page_view_type = 'Checkout_Page' THEN 1 ELSE 0 END) AS has_browsed_checkout_page_first_60d
        FROM
            browsing b
        INNER JOIN
            customers c_sub ON b.customer_id = c_sub.customer_id
        WHERE
            -- Use DATE(..., '+60 days', '+23 hours', ...) to cover full 60th day for timestamps
            b.browse_date BETWEEN c_sub.signup_date AND DATE(c_sub.signup_date, '+60 days', '+23 hours', '+59 minutes', '+59 seconds')
        GROUP BY
            b.customer_id
    ) AS b_agg ON c.customer_id = b_agg.customer_id;
    """
    
    customer_features_df = pd.read_sql_query(sql_query, conn)
    
    conn.close()
    
    print("\nSQL Feature Engineering Complete. Sample `customer_features_df` head:")
    print(customer_features_df.head())
    
    return customer_features_df

def pandas_feature_engineering_and_target_creation(customer_features_df, purchases_df_original, customers_df_original):
    """
    Performs additional Pandas feature engineering and creates the multi-class LTV target.
    """
    
    # Handle NaN values from SQL query
    customer_features_df['num_purchases_first_60d'] = customer_features_df['num_purchases_first_60d'].fillna(0).astype(int)
    customer_features_df['total_spend_first_60d'] = customer_features_df['total_spend_first_60d'].fillna(0.0)
    customer_features_df['avg_purchase_amount_first_60d'] = customer_features_df['avg_purchase_amount_first_60d'].fillna(0.0)
    customer_features_df['num_browsing_events_first_60d'] = customer_features_df['num_browsing_events_first_60d'].fillna(0).astype(int)
    customer_features_df['total_browse_duration_first_60d'] = customer_features_df['total_browse_duration_first_60d'].fillna(0).astype(int)
    customer_features_df['num_unique_product_categories_first_60d'] = customer_features_df['num_unique_product_categories_first_60d'].fillna(0).astype(int)
    customer_features_df['has_browsed_checkout_page_first_60d'] = customer_features_df['has_browsed_checkout_page_first_60d'].fillna(0).astype(int)
    
    # For days_since_first_purchase_first_60d, fill NaN with 60 (representing no purchase within 60 days)
    customer_features_df['days_since_first_purchase_first_60d'] = customer_features_df['days_since_first_purchase_first_60d'].fillna(60).astype(int)
    
    # Convert signup_date to datetime objects for pandas calculations
    customer_features_df['signup_date'] = pd.to_datetime(customer_features_df['signup_date'])
    
    # Calculate account_age_at_cutoff_days (always 60 for this window definition)
    customer_features_df['account_age_at_cutoff_days'] = 60
    
    # Calculate purchase_frequency_first_60d
    customer_features_df['purchase_frequency_first_60d'] = customer_features_df['num_purchases_first_60d'] / 60.0
    customer_features_df['purchase_frequency_first_60d'] = customer_features_df['purchase_frequency_first_60d'].fillna(0.0)
    
    # Calculate browse_to_purchase_ratio_first_60d
    customer_features_df['browse_to_purchase_ratio_first_60d'] = customer_features_df['num_browsing_events_first_60d'] / (customer_features_df['num_purchases_first_60d'] + 1)
    
    # --- Create the Multi-Class Target `future_ltv_tier` ---
    
    # Ensure original dataframes have datetime objects for date calculations
    customers_df_original['signup_date'] = pd.to_datetime(customers_df_original['signup_date'])
    purchases_df_original['purchase_date'] = pd.to_datetime(purchases_df_original['purchase_date'])

    # Merge purchases with customer signup dates to determine future purchases
    purchases_with_signup_info = purchases_df_original.merge(
        customers_df_original[['customer_id', 'signup_date']], on='customer_id', how='left'
    )

    # Calculate the early_window_cutoff_date for each purchase
    purchases_with_signup_info['early_window_cutoff_date'] = purchases_with_signup_info['signup_date'] + pd.Timedelta(days=60)

    # Filter for purchases occurring *after* the 60-day window
    future_purchases = purchases_with_signup_info[
        purchases_with_signup_info['purchase_date'] > purchases_with_signup_info['early_window_cutoff_date']
    ]

    # Calculate total_future_spend for each customer
    total_future_spend = future_purchases.groupby('customer_id')['amount'].sum().reset_index()
    total_future_spend.rename(columns={'amount': 'total_future_spend'}, inplace=True)

    # Merge total_future_spend with customer_features_df
    customer_features_df = customer_features_df.merge(total_future_spend, on='customer_id', how='left')
    customer_features_df['total_future_spend'] = customer_features_df['total_future_spend'].fillna(0.0)

    # Calculate percentiles for non-zero total_future_spend
    non_zero_future_spend = customer_features_df[customer_features_df['total_future_spend'] > 0]['total_future_spend']
    if not non_zero_future_spend.empty:
        p33 = non_zero_future_spend.quantile(0.33)
        p66 = non_zero_future_spend.quantile(0.66)
    else: # Fallback if all future spends are zero
        p33, p66 = 0, 0
    
    # Define segments for future_ltv_tier
    def get_ltv_tier(spend):
        if spend == 0:
            return 'Low_LTV'
        elif spend <= p33:
            return 'Medium_LTV'
        elif spend <= p66:
            return 'High_LTV'
        else:
            return 'Very_High_LTV'
            
    customer_features_df['future_ltv_tier'] = customer_features_df['total_future_spend'].apply(get_ltv_tier)
    
    # Define features X and target y
    numerical_features = [
        'total_spend_first_60d', 'num_purchases_first_60d', 'avg_purchase_amount_first_60d',
        'num_browsing_events_first_60d', 'total_browse_duration_first_60d',
        'num_unique_product_categories_first_60d', 'days_since_first_purchase_first_60d',
        'account_age_at_cutoff_days', 'purchase_frequency_first_60d', 'browse_to_purchase_ratio_first_60d'
    ]
    # 'has_browsed_checkout_page_first_60d' is binary but specified as categorical
    categorical_features = [
        'acquisition_channel', 'demographic_segment', 'has_browsed_checkout_page_first_60d'
    ]

    # Use a copy of relevant columns for X to avoid SettingWithCopyWarning
    X = customer_features_df[numerical_features + categorical_features].copy()
    y = customer_features_df['future_ltv_tier']
    
    # Encode target variable for stratification in train_test_split
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    X_train, X_test, y_train_encoded, y_test_encoded = train_test_split(
        X, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
    )
    
    # Inverse transform y_train_encoded and y_test_encoded back to string labels for the model
    # (HistGradientBoostingClassifier can handle string labels directly)
    y_train = pd.Series(le.inverse_transform(y_train_encoded), index=X_train.index)
    y_test = pd.Series(le.inverse_transform(y_test_encoded), index=X_test.index)

    print("\nPandas Feature Engineering & Target Creation Complete.")
    print(f"LTV Tier distribution:\n{y.value_counts()}")
    print("\nSample `customer_features_df` with target head:")
    print(customer_features_df[['customer_id', 'total_future_spend', 'future_ltv_tier', 'acquisition_channel']].head())
    
    return X_train, X_test, y_train, y_test, numerical_features, categorical_features, customer_features_df, le


def visualize_data(customer_features_df):
    """
    Generates and displays two plots to visualize relationships with future LTV tier.
    """
    plt.figure(figsize=(14, 6))

    # Violin plot for total_spend_first_60d vs future_ltv_tier
    plt.subplot(1, 2, 1)
    sns.violinplot(x='future_ltv_tier', y='total_spend_first_60d', data=customer_features_df, 
                   order=['Low_LTV', 'Medium_LTV', 'High_LTV', 'Very_High_LTV'])
    plt.title('Total Spend in First 60 Days by Future LTV Tier')
    plt.xlabel('Future LTV Tier')
    plt.ylabel('Total Spend (First 60 Days)')
    plt.yscale('log') # Use log scale to better visualize variations for potentially skewed data

    # Stacked bar chart for acquisition_channel vs future_ltv_tier
    plt.subplot(1, 2, 2)
    # Calculate proportions
    ltv_channel_pivot = customer_features_df.groupby('acquisition_channel')['future_ltv_tier'].value_counts(normalize=True).unstack().fillna(0)
    # Ensure consistent order of LTV tiers for plotting
    ltv_channel_pivot = ltv_channel_pivot[['Low_LTV', 'Medium_LTV', 'High_LTV', 'Very_High_LTV']]
    ltv_channel_pivot.plot(kind='bar', stacked=True, ax=plt.gca())
    plt.title('Proportion of Future LTV Tier by Acquisition Channel')
    plt.xlabel('Acquisition Channel')
    plt.ylabel('Proportion of Customers')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Future LTV Tier', bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.show()
    print("\nData Visualizations Displayed.")

def ml_pipeline_and_evaluation(X_train, X_test, y_train, y_test, numerical_features, categorical_features):
    """
    Builds and evaluates an ML pipeline for multi-class classification of LTV tiers.
    """
    
    # Preprocessing pipelines for numerical and categorical features
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    
    # OneHotEncoder for categorical features (including the binary 'has_browsed_checkout_page_first_60d')
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Create a column transformer to apply different transformations to different columns
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough' # In case there are other columns not explicitly handled, pass them through
    )
    
    # Create the full pipeline with preprocessor and classifier
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', HistGradientBoostingClassifier(random_state=42))
    ])
    
    # Train the pipeline
    print("\nTraining ML Pipeline...")
    model_pipeline.fit(X_train, y_train)
    print("ML Pipeline Training Complete.")
    
    # Predict on the test set
    y_pred = model_pipeline.predict(X_test)
    
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0) # zero_division=0 handles cases where a class has no samples in predictions/true
    
    print("\nML Model Evaluation:")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(report)

if __name__ == "__main__":
    # Task 1: Generate Synthetic Data
    customers_df, purchases_df_original, browsing_df_original = generate_synthetic_data()
    
    # Task 2: Load into SQLite & SQL Feature Engineering
    customer_features_df = sql_feature_engineering(
        customers_df.copy(),  # Use copies to avoid modifying original DFs passed to other functions
        purchases_df_original.copy(), 
        browsing_df_original.copy()
    )
    
    # Task 3: Pandas Feature Engineering & Multi-Class Target Creation
    X_train, X_test, y_train, y_test, numerical_features, categorical_features, customer_features_df_with_target, label_encoder_y = pandas_feature_engineering_and_target_creation(
        customer_features_df.copy(), 
        purchases_df_original.copy(), 
        customers_df.copy()
    )
    
    # Task 4: Data Visualization
    visualize_data(customer_features_df_with_target)
    
    # Task 5: ML Pipeline & Evaluation
    ml_pipeline_and_evaluation(X_train, X_test, y_train, y_test, numerical_features, categorical_features)