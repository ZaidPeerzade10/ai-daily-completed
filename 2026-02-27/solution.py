import numpy as np
import pandas as pd
import datetime
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer # Corrected import based on feedback
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, classification_report
import warnings

# Suppress specific warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning) # For seaborn

def generate_synthetic_data():
    """Generates synthetic user, ad, and impression dataframes."""

    print("1. Generating Synthetic Data...")
    
    # --- Users DataFrame ---
    num_users = np.random.randint(500, 701)
    users_df = pd.DataFrame({
        'user_id': np.arange(1, num_users + 1),
        'signup_date': pd.to_datetime(np.random.choice(pd.date_range(end=datetime.date.today(), periods=5*365), num_users)),
        'age': np.random.randint(18, 71, num_users),
        'gender': np.random.choice(['Male', 'Female', 'Other'], num_users, p=[0.45, 0.45, 0.1]),
        'device_type': np.random.choice(['Mobile', 'Desktop', 'Tablet'], num_users, p=[0.6, 0.3, 0.1]),
        'ad_blocker_enabled': np.random.choice([0, 1], num_users, p=[0.7, 0.3]),
        'user_base_click_propensity': np.random.uniform(0.8, 1.2, num_users) # For sequential bias simulation
    })
    print(f"Generated {len(users_df)} users.")

    # --- Ads DataFrame ---
    num_ads = np.random.randint(50, 101)
    ads_df = pd.DataFrame({
        'ad_id': np.arange(101, 101 + num_ads),
        'advertiser_category': np.random.choice(['Finance', 'Gaming', 'Retail', 'Travel', 'Education', 'Automotive'], num_ads),
        'ad_format': np.random.choice(['Banner', 'Video', 'Native', 'Pop-up'], num_ads, p=[0.4, 0.3, 0.2, 0.1])
    })
    # Ensure target_audience_age_max > target_audience_age_min
    ads_df['target_audience_age_min'] = np.random.randint(18, 51, num_ads)
    ads_df['target_audience_age_max'] = ads_df['target_audience_age_min'] + np.random.randint(10, 31, num_ads)
    ads_df['target_audience_age_max'] = ads_df['target_audience_age_max'].clip(upper=70) # Max age is 70
    print(f"Generated {len(ads_df)} ads.")

    # --- Impressions DataFrame ---
    num_impressions = np.random.randint(5000, 8001)
    impressions_df = pd.DataFrame({
        'impression_id': np.arange(10001, 10001 + num_impressions),
        'user_id': np.random.choice(users_df['user_id'], num_impressions),
        'ad_id': np.random.choice(ads_df['ad_id'], num_impressions),
    })

    # Merge to get user and ad details for impression date and CTR simulation
    impressions_df = impressions_df.merge(users_df[['user_id', 'signup_date', 'age', 'ad_blocker_enabled', 'user_base_click_propensity']], on='user_id', how='left')
    impressions_df = impressions_df.merge(ads_df[['ad_id', 'advertiser_category', 'ad_format', 'target_audience_age_min', 'target_audience_age_max']], on='ad_id', how='left')

    # Generate impression_date (always after signup_date)
    impressions_df['impression_date'] = impressions_df.apply(
        lambda row: pd.to_datetime(row['signup_date']) + pd.to_timedelta(np.random.randint(1, 365*3), unit='D'), axis=1
    )
    impressions_df['impression_date'] = impressions_df['impression_date'].clip(upper=pd.to_datetime(datetime.date.today()))

    # Sort impressions for sequential processing of CTR simulation
    impressions_df.sort_values(by=['user_id', 'impression_date'], inplace=True)

    # Simulate was_clicked with biases (overall 2-5% CTR)
    base_ctr = 0.035 # Average 3.5%
    
    impressions_df['was_clicked'] = 0 # Initialize

    # Calculate click probability for each impression
    impressions_df['click_prob'] = base_ctr * impressions_df['user_base_click_propensity']

    # Ad blocker bias
    impressions_df.loc[impressions_df['ad_blocker_enabled'] == 1, 'click_prob'] *= 0.2

    # Target audience age bias
    in_target_audience = (impressions_df['age'] >= impressions_df['target_audience_age_min']) & \
                         (impressions_df['age'] <= impressions_df['target_audience_age_max'])
    impressions_df.loc[in_target_audience, 'click_prob'] *= 1.8

    # Advertiser category bias
    impressions_df.loc[impressions_df['advertiser_category'] == 'Gaming', 'click_prob'] *= 1.7
    impressions_df.loc[impressions_df['advertiser_category'] == 'Finance', 'click_prob'] *= 0.6

    # Ad format bias
    impressions_df.loc[impressions_df['ad_format'] == 'Video', 'click_prob'] *= 1.5
    impressions_df.loc[impressions_df['ad_format'] == 'Banner', 'click_prob'] *= 0.8
    impressions_df.loc[impressions_df['ad_format'] == 'Pop-up', 'click_prob'] *= 0.1 # Very low CTR for pop-ups

    # Clip probabilities to sensible range
    impressions_df['click_prob'] = impressions_df['click_prob'].clip(lower=0.001, upper=0.15)

    # Assign clicks based on probability
    impressions_df['was_clicked'] = (np.random.rand(len(impressions_df)) < impressions_df['click_prob']).astype(int)

    actual_ctr = impressions_df['was_clicked'].mean()
    print(f"Generated {len(impressions_df)} impressions. Actual CTR: {actual_ctr:.4f}")

    # Clean up temporary columns before returning
    impressions_df.drop(columns=['signup_date', 'age', 'ad_blocker_enabled', 'user_base_click_propensity',
                                 'advertiser_category', 'ad_format', 'target_audience_age_min',
                                 'target_audience_age_max', 'click_prob'], inplace=True)
    users_df.drop(columns=['user_base_click_propensity'], inplace=True)
    
    # Final sort for impressions_df before SQL
    impressions_df.sort_values(by=['user_id', 'impression_date'], inplace=True)

    return users_df, ads_df, impressions_df

def load_to_sqlite_and_feature_engineer(users_df, ads_df, impressions_df):
    """Loads data into SQLite and performs SQL-based feature engineering."""

    print("\n2. Loading into SQLite and SQL Feature Engineering...")

    conn = sqlite3.connect(':memory:') # In-memory database

    users_df.to_sql('users', conn, index=False, if_exists='replace')
    ads_df.to_sql('ads', conn, index=False, if_exists='replace')
    impressions_df.to_sql('impressions', conn, index=False, if_exists='replace', dtype={'impression_date': 'TEXT'})

    print("Data loaded into SQLite tables: users, ads, impressions.")

    # SQL query for feature engineering
    sql_query = """
    WITH RankedImpressions AS (
        SELECT
            imp.impression_id,
            imp.user_id,
            imp.ad_id,
            imp.impression_date,
            imp.was_clicked,
            usr.signup_date,
            usr.age,
            usr.gender,
            usr.device_type,
            usr.ad_blocker_enabled,
            ad.advertiser_category,
            ad.ad_format,
            ad.target_audience_age_min,
            ad.target_audience_age_max,
            -- For User-level sequential features
            COUNT(imp.impression_id) OVER (PARTITION BY imp.user_id ORDER BY imp.impression_date, imp.impression_id ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS user_prior_impressions_raw,
            SUM(imp.was_clicked) OVER (PARTITION BY imp.user_id ORDER BY imp.impression_date, imp.impression_id ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS user_prior_clicks_raw,
            LAG(CASE WHEN imp.was_clicked = 1 THEN imp.impression_date END, 1, usr.signup_date) 
                IGNORE NULLS OVER (PARTITION BY imp.user_id ORDER BY imp.impression_date, imp.impression_id) AS user_last_click_date,
            -- For Ad-level sequential features
            COUNT(imp.impression_id) OVER (PARTITION BY imp.ad_id ORDER BY imp.impression_date, imp.impression_id ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS ad_prior_impressions_raw,
            SUM(imp.was_clicked) OVER (PARTITION BY imp.ad_id ORDER BY imp.impression_date, imp.impression_id ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS ad_prior_clicks_raw
        FROM
            impressions imp
        JOIN
            users usr ON imp.user_id = usr.user_id
        JOIN
            ads ad ON imp.ad_id = ad.ad_id
    )
    SELECT
        impression_id,
        user_id,
        ad_id,
        impression_date,
        was_clicked,
        age,
        gender,
        device_type,
        ad_blocker_enabled,
        advertiser_category,
        ad_format,
        target_audience_age_min,
        target_audience_age_max,
        signup_date,
        COALESCE(user_prior_impressions_raw, 0) AS user_prior_impressions,
        COALESCE(user_prior_clicks_raw, 0) AS user_prior_clicks,
        CAST(COALESCE(user_prior_clicks_raw, 0) AS REAL) / NULLIF(COALESCE(user_prior_impressions_raw, 0), 0) AS user_prior_ctr,
        CAST(JULIANDAY(impression_date) - JULIANDAY(COALESCE(user_last_click_date, signup_date)) AS INTEGER) AS days_since_last_user_click,
        COALESCE(ad_prior_impressions_raw, 0) AS ad_prior_impressions,
        COALESCE(ad_prior_clicks_raw, 0) AS ad_prior_clicks,
        CAST(COALESCE(ad_prior_clicks_raw, 0) AS REAL) / NULLIF(COALESCE(ad_prior_impressions_raw, 0), 0) AS ad_prior_ctr
    FROM
        RankedImpressions
    ORDER BY
        user_id, impression_date, impression_id;
    """
    
    ad_features_df = pd.read_sql_query(sql_query, conn)
    conn.close()
    
    print(f"SQL Feature Engineering complete. Resulting DataFrame has {len(ad_features_df)} rows and {len(ad_features_df.columns)} columns.")
    print("Example of engineered features:")
    print(ad_features_df[['user_id', 'impression_date', 'user_prior_impressions', 'user_prior_clicks', 'user_prior_ctr', 'days_since_last_user_click', 'was_clicked']].head(10).to_string())

    return ad_features_df

def pandas_feature_engineering_and_split(ad_features_df):
    """Performs Pandas-based feature engineering and splits data for ML."""

    print("\n3. Pandas Feature Engineering & Binary Target Creation...")

    # Handle NaN values (fill with 0 for counts, 0.0 for CTRs)
    ad_features_df['user_prior_impressions'].fillna(0, inplace=True)
    ad_features_df['user_prior_clicks'].fillna(0, inplace=True)
    ad_features_df['ad_prior_impressions'].fillna(0, inplace=True)
    ad_features_df['ad_prior_clicks'].fillna(0, inplace=True)
    
    ad_features_df['user_prior_ctr'].fillna(0.0, inplace=True)
    ad_features_df['ad_prior_ctr'].fillna(0.0, inplace=True)
    
    # Ensure days_since_last_user_click is numeric and fill NaNs if any (e.g., first impression for a user)
    ad_features_df['days_since_last_user_click'] = pd.to_numeric(ad_features_df['days_since_last_user_click'], errors='coerce')
    ad_features_df['days_since_last_user_click'].fillna(9999, inplace=True) # Sentinel for no prior click

    # Convert dates to datetime objects
    ad_features_df['signup_date'] = pd.to_datetime(ad_features_df['signup_date'])
    ad_features_df['impression_date'] = pd.to_datetime(ad_features_df['impression_date'])

    # Calculate user_account_age_at_impression_days
    ad_features_df['user_account_age_at_impression_days'] = (ad_features_df['impression_date'] - ad_features_df['signup_date']).dt.days

    # Create is_user_in_target_audience
    ad_features_df['is_user_in_target_audience'] = (
        (ad_features_df['age'] >= ad_features_df['target_audience_age_min']) &
        (ad_features_df['age'] <= ad_features_df['target_audience_age_max'])
    ).astype(int)

    # Define features (X) and target (y)
    numerical_features = [
        'age', 'user_account_age_at_impression_days', 'user_prior_impressions',
        'user_prior_clicks', 'user_prior_ctr', 'days_since_last_user_click',
        'ad_prior_impressions', 'ad_prior_clicks', 'ad_prior_ctr',
        'target_audience_age_min', 'target_audience_age_max'
    ]
    categorical_features = [
        'gender', 'device_type', 'ad_blocker_enabled', 'advertiser_category',
        'ad_format', 'is_user_in_target_audience' # is_user_in_target_audience is binary but treated as categorical for OneHotEncoding
    ]

    # Convert ad_blocker_enabled to object type for categorical handling in ColumnTransformer
    ad_features_df['ad_blocker_enabled'] = ad_features_df['ad_blocker_enabled'].astype(object)
    ad_features_df['is_user_in_target_audience'] = ad_features_df['is_user_in_target_audience'].astype(object)

    X = ad_features_df[numerical_features + categorical_features]
    y = ad_features_df['was_clicked']

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    print(f"Total dataset shape: {X.shape}")
    print(f"Training set shape: {X_train.shape}, Test set shape: {X_test.shape}")
    print(f"Original CTR: {y.mean():.4f}, Training CTR: {y_train.mean():.4f}, Test CTR: {y_test.mean():.4f}")

    return X_train, X_test, y_train, y_test, numerical_features, categorical_features, ad_features_df # Return ad_features_df for visualization

def visualize_data(ad_features_df):
    """Creates visualizations to inspect relationships with was_clicked."""

    print("\n4. Data Visualization...")

    plt.figure(figsize=(16, 6))

    # Violin plot for user_prior_ctr vs was_clicked
    plt.subplot(1, 2, 1)
    sns.violinplot(x='was_clicked', y='user_prior_ctr', data=ad_features_df)
    plt.title('Distribution of User Prior CTR by Click Status')
    plt.xlabel('Was Clicked (0=No, 1=Yes)')
    plt.ylabel('User Prior CTR')

    # Stacked bar chart for ad_format vs was_clicked proportion
    plt.subplot(1, 2, 2)
    ad_format_counts = ad_features_df.groupby(['ad_format', 'was_clicked']).size().unstack(fill_value=0)
    ad_format_proportions = ad_format_counts.div(ad_format_counts.sum(axis=1), axis=0)
    ad_format_proportions.plot(kind='bar', stacked=True, ax=plt.gca(), cmap='viridis')
    plt.title('Proportion of Clicks by Ad Format')
    plt.xlabel('Ad Format')
    plt.ylabel('Proportion')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Was Clicked', labels=['No Click', 'Click'])

    plt.tight_layout()
    plt.show()

def build_and_evaluate_ml_pipeline(X_train, X_test, y_train, y_test, numerical_features, categorical_features):
    """Builds, trains, and evaluates an ML pipeline for CTR prediction."""

    print("\n5. ML Pipeline & Evaluation (Binary Classification)...")

    # Preprocessing steps
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
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # Create the ML pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', HistGradientBoostingClassifier(random_state=42))
    ])

    # Train the pipeline
    print("Training the ML pipeline...")
    pipeline.fit(X_train, y_train)
    print("Training complete.")

    # Predict probabilities on the test set
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred = pipeline.predict(X_test) # For classification report

    # Evaluate the model
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    clf_report = classification_report(y_test, y_pred)

    print(f"\n--- Model Evaluation ---")
    print(f"ROC AUC Score: {roc_auc:.4f}")
    print(f"\nClassification Report:\n{clf_report}")

if __name__ == "__main__":
    # 1. Generate Synthetic Data
    users_df, ads_df, impressions_df = generate_synthetic_data()

    # 2. Load into SQLite & SQL Feature Engineering
    ad_features_df = load_to_sqlite_and_feature_engineer(users_df, ads_df, impressions_df)

    # 3. Pandas Feature Engineering & Binary Target Creation
    X_train, X_test, y_train, y_test, numerical_features, categorical_features, full_df_for_viz = \
        pandas_feature_engineering_and_split(ad_features_df.copy()) # Pass a copy for visualization to avoid modifications

    # 4. Data Visualization
    visualize_data(full_df_for_viz)

    # 5. ML Pipeline & Evaluation
    build_and_evaluate_ml_pipeline(X_train, X_test, y_train, y_test, numerical_features, categorical_features)