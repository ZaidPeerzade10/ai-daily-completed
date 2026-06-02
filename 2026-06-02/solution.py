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
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, classification_report
import warnings

# Suppress specific warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning, module='seaborn')
warnings.filterwarnings('ignore', category=FutureWarning)

def generate_synthetic_data():
    """
    Generates synthetic employee and work activity dataframes.
    Simulates realistic patterns including attrition triggers and activity drop-off.
    """
    print("1. Generating Synthetic Data...")

    # --- Employee Data ---
    departments = ['Sales', 'Engineering', 'HR', 'Marketing', 'Operations', 'Finance']
    num_employees = np.random.randint(1000, 1500)

    employees_data = {
        'employee_id': np.arange(1, num_employees + 1),
        'hire_date': pd.to_datetime(pd.Timestamp.now() - pd.Timedelta(days=np.random.randint(365 * 5, 365 * 10, num_employees))),
        'department': np.random.choice(departments, num_employees),
        'salary': np.random.uniform(50000, 200000, num_employees).round(2),
        'performance_rating': np.random.randint(1, 6, num_employees), # 1-5
        'satisfaction_score': np.random.uniform(1.0, 5.0, num_employees).round(1), # 1.0-5.0
    }
    employees_df = pd.DataFrame(employees_data)

    # Simulate last_promotion_date
    employees_df['last_promotion_date'] = pd.NaT
    promoted_indices = np.random.choice(employees_df.index, int(num_employees * 0.7), replace=False) # 70% promoted
    for idx in promoted_indices:
        hire_date = employees_df.loc[idx, 'hire_date']
        # Ensure promotion is after hire date, but not too recent
        if pd.Timestamp.now() - hire_date > pd.Timedelta(days=365): # at least 1 year tenure
            promotion_date = hire_date + pd.Timedelta(days=np.random.randint(365, (pd.Timestamp.now() - hire_date).days))
            employees_df.loc[idx, 'last_promotion_date'] = promotion_date
        
    # --- Attrition Data ---
    employees_df['attrition_date'] = pd.NaT
    attrition_rate = 0.18 # Around 15-20% attrition

    # Simulate attrition based on factors
    for idx in employees_df.index:
        will_attrit = False
        hire_date = employees_df.loc[idx, 'hire_date']
        
        # Base attrition likelihood
        if np.random.rand() < attrition_rate:
            will_attrit = True

        # Higher likelihood for low satisfaction/performance
        if employees_df.loc[idx, 'satisfaction_score'] < 2.5 and np.random.rand() < 0.6: # 60% extra chance
            will_attrit = True
        if employees_df.loc[idx, 'performance_rating'] <= 2 and np.random.rand() < 0.5: # 50% extra chance
            will_attrit = True
        
        # Higher likelihood for certain departments
        if employees_df.loc[idx, 'department'] in ['Sales', 'Operations'] and np.random.rand() < 0.4: # 40% extra chance
            will_attrit = True

        if will_attrit:
            # Attrition must be after hire_date and within the last 18 months from now
            # And also after any last_promotion_date if it exists
            min_attrition_date = employees_df.loc[idx, 'hire_date']
            if pd.notna(employees_df.loc[idx, 'last_promotion_date']):
                min_attrition_date = max(min_attrition_date, employees_df.loc[idx, 'last_promotion_date'])
            
            # Ensure attrition_date is within the last 18 months and after hire/promotion
            
            # Start of the 18-month window from now, but not earlier than min_attrition_date
            potential_attrition_start = max(min_attrition_date, pd.Timestamp.now() - pd.Timedelta(days=18*30))
            # End of the window: up to a month ago to ensure it's in the past and not too recent
            potential_attrition_end = pd.Timestamp.now() - pd.Timedelta(days=30) 

            if potential_attrition_start < potential_attrition_end:
                days_range = (potential_attrition_end - potential_attrition_start).days
                if days_range > 0:
                    employees_df.loc[idx, 'attrition_date'] = potential_attrition_start + pd.Timedelta(days=np.random.randint(days_range))
                else: # Edge case: if range is 0 or negative, set to end date
                     employees_df.loc[idx, 'attrition_date'] = potential_attrition_end


    # --- Work Activity Data ---
    num_activities = np.random.randint(20000, 30000)
    activity_data = {
        'activity_id': np.arange(1, num_activities + 1),
        'employee_id': np.random.choice(employees_df['employee_id'], num_activities, replace=True),
        'hours_worked': np.random.uniform(4.0, 12.0, num_activities).round(1),
        'project_count': np.random.randint(1, 6, num_activities),
    }
    work_activity_df = pd.DataFrame(activity_data)

    # Assign activity_date ensuring it's after hire_date and before attrition_date
    work_activity_df = work_activity_df.merge(
        employees_df[['employee_id', 'hire_date', 'attrition_date']],
        on='employee_id', how='left'
    )

    activity_dates = []
    for idx, row in work_activity_df.iterrows():
        hire_date = row['hire_date']
        attrition_date = row['attrition_date']

        start_date = hire_date
        end_date = attrition_date if pd.notna(attrition_date) else pd.Timestamp.now()

        # Ensure valid date range for activity generation
        if start_date >= end_date:
            # If start_date is after or same as end_date, adjust start_date
            # A common reason is `hire_date` very close to `attrition_date` or `now()`
            # Set activity date to be just before end_date or just after hire_date
            if (end_date - hire_date).days > 1: # if there's at least 2 days in range
                activity_dates.append(end_date - pd.Timedelta(days=np.random.randint(1, (end_date - hire_date).days)))
            else: # Fallback for very narrow or invalid ranges
                activity_dates.append(hire_date + pd.Timedelta(days=1)) # Just one day after hire
        else:
            days_range = (end_date - start_date).days
            activity_dates.append(start_date + pd.Timedelta(days=np.random.randint(days_range) if days_range > 0 else 0))

    work_activity_df['activity_date'] = activity_dates

    # Simulate activity drop-off for attriting employees
    for idx, row in employees_df[employees_df['attrition_date'].notna()].iterrows():
        attrition_date = row['attrition_date']
        employee_id = row['employee_id']

        # Identify activities in the 1-2 months leading up to attrition
        pre_attrition_start = attrition_date - pd.Timedelta(days=60) # 2 months
        pre_attrition_end = attrition_date - pd.Timedelta(days=1) # day before attrition

        mask = (work_activity_df['employee_id'] == employee_id) & \
               (work_activity_df['activity_date'] >= pre_attrition_start) & \
               (work_activity_df['activity_date'] <= pre_attrition_end)

        # Reduce hours_worked and project_count for these activities
        if not work_activity_df.loc[mask].empty:
            work_activity_df.loc[mask, 'hours_worked'] = work_activity_df.loc[mask, 'hours_worked'] * np.random.uniform(0.5, 0.9, size=mask.sum())
            work_activity_df.loc[mask, 'project_count'] = np.maximum(1, (work_activity_df.loc[mask, 'project_count'] * np.random.uniform(0.5, 0.9, size=mask.sum())).astype(int))

    work_activity_df = work_activity_df.drop(columns=['hire_date', 'attrition_date_y']).rename(columns={'attrition_date_x': 'attrition_date'})
    work_activity_df = work_activity_df.sort_values(by=['employee_id', 'activity_date']).reset_index(drop=True)

    print(f"Generated {len(employees_df)} employees and {len(work_activity_df)} work activities.")
    print(f"Attrition rate: {employees_df['attrition_date'].count() / len(employees_df):.2%}")

    return employees_df, work_activity_df

def sql_feature_engineering(employees_df, work_activity_df):
    """
    Loads data into an in-memory SQLite DB and performs SQL feature engineering.
    """
    print("\n2. Loading data into SQLite & SQL Feature Engineering...")

    conn = sqlite3.connect(':memory:')
    employees_df.to_sql('employees', conn, index=False, if_exists='replace')
    work_activity_df.to_sql('work_activity', conn, index=False, if_exists='replace')

    # Define GLOBAL_PREDICTION_CUTOFF_DATE
    latest_activity_date_str = pd.read_sql_query("SELECT MAX(activity_date) FROM work_activity", conn).iloc[0, 0]
    latest_activity_date = pd.to_datetime(latest_activity_date_str)
    global_prediction_cutoff_date = latest_activity_date - pd.Timedelta(days=6*30) # 6 months prior
    print(f"Global Prediction Cutoff Date: {global_prediction_cutoff_date.strftime('%Y-%m-%d')}")

    # SQL query for feature engineering
    sql_query = f"""
    WITH EmployeeActivity AS (
        SELECT
            wa.employee_id,
            wa.activity_date,
            wa.hours_worked,
            wa.project_count
        FROM
            work_activity wa
        WHERE
            wa.activity_date <= '{global_prediction_cutoff_date.strftime('%Y-%m-%d %H:%M:%S')}'
    ),
    RecentActivity AS (
        SELECT
            ea.employee_id,
            AVG(ea.hours_worked) AS avg_hours_worked_prev_90d,
            COUNT(ea.activity_date) AS num_activities_prev_90d,
            COUNT(DISTINCT ea.project_count) AS num_distinct_projects_prev_90d,
            MAX(ea.activity_date) AS last_activity_date_at_cutoff
        FROM
            EmployeeActivity ea
        WHERE
            ea.activity_date >= DATE('{global_prediction_cutoff_date.strftime('%Y-%m-%d %H:%M:%S')}', '-90 days')
            AND ea.activity_date <= '{global_prediction_cutoff_date.strftime('%Y-%m-%d %H:%M:%S')}'
        GROUP BY
            ea.employee_id
    )
    SELECT
        e.employee_id,
        e.hire_date,
        e.department,
        e.salary,
        e.performance_rating,
        e.satisfaction_score,
        e.last_promotion_date,
        e.attrition_date,
        '{global_prediction_cutoff_date.strftime('%Y-%m-%d %H:%M:%S')}' AS current_cutoff_date,
        COALESCE(ra.avg_hours_worked_prev_90d, 0.0) AS avg_hours_worked_prev_90d,
        COALESCE(ra.num_activities_prev_90d, 0) AS num_activities_prev_90d,
        COALESCE(ra.num_distinct_projects_prev_90d, 0) AS num_distinct_projects_prev_90d,
        CAST(
            JULIANDAY('{global_prediction_cutoff_date.strftime('%Y-%m-%d %H:%M:%S')}') - 
            JULIANDAY(COALESCE(ra.last_activity_date_at_cutoff, 
                               DATE('{global_prediction_cutoff_date.strftime('%Y-%m-%d %H:%M:%S')}', '-9999 days'))) 
            AS INTEGER
        ) AS days_since_last_activity_at_cutoff
    FROM
        employees e
    LEFT JOIN
        RecentActivity ra ON e.employee_id = ra.employee_id
    WHERE
        e.hire_date <= '{global_prediction_cutoff_date.strftime('%Y-%m-%d %H:%M:%S')}'
        AND (e.attrition_date IS NULL OR e.attrition_date > '{global_prediction_cutoff_date.strftime('%Y-%m-%d %H:%M:%S')}')
    ;
    """
    employee_features_df = pd.read_sql_query(sql_query, conn)
    conn.close()

    print(f"Generated {len(employee_features_df)} employee features records active at cutoff.")
    return employee_features_df, global_prediction_cutoff_date

def pandas_feature_engineering_and_target(employee_features_df, global_prediction_cutoff_date):
    """
    Performs additional Pandas feature engineering and creates the binary target.
    """
    print("\n3. Pandas Feature Engineering & Binary Target Creation...")

    # Convert date columns
    date_cols = ['hire_date', 'current_cutoff_date', 'last_promotion_date', 'attrition_date']
    for col in date_cols:
        employee_features_df[col] = pd.to_datetime(employee_features_df[col])

    # Handle NaN values for aggregated features
    # days_since_last_activity_at_cutoff is already handled by SQL COALESCE with 9999 default
    employee_features_df['avg_hours_worked_prev_90d'] = employee_features_df['avg_hours_worked_prev_90d'].fillna(0.0)
    employee_features_df['num_activities_prev_90d'] = employee_features_df['num_activities_prev_90d'].fillna(0)
    employee_features_df['num_distinct_projects_prev_90d'] = employee_features_df['num_distinct_projects_prev_90d'].fillna(0)

    # Calculate employee_tenure_at_cutoff_days
    employee_features_df['employee_tenure_at_cutoff_days'] = \
        (employee_features_df['current_cutoff_date'] - employee_features_df['hire_date']).dt.days

    # Calculate days_since_last_promotion_at_cutoff
    employee_features_df['days_since_last_promotion_at_cutoff'] = \
        (employee_features_df['current_cutoff_date'] - employee_features_df['last_promotion_date']).dt.days

    # If no promotion or promotion after cutoff, use tenure
    promotion_nan_or_future_mask = employee_features_df['last_promotion_date'].isna() | \
                                   (employee_features_df['last_promotion_date'] > employee_features_df['current_cutoff_date'])
    employee_features_df.loc[promotion_nan_or_future_mask, 'days_since_last_promotion_at_cutoff'] = \
        employee_features_df.loc[promotion_nan_or_future_mask, 'employee_tenure_at_cutoff_days']

    # Create the Binary Target 'will_attrit_in_next_6_months'
    attrition_window_start = employee_features_df['current_cutoff_date']
    attrition_window_end = employee_features_df['current_cutoff_date'] + pd.Timedelta(days=6*30) # 6 months

    employee_features_df['will_attrit_in_next_6_months'] = (
        (employee_features_df['attrition_date'] > attrition_window_start) &
        (employee_features_df['attrition_date'] <= attrition_window_end)
    ).astype(int)

    # Define features X and target y
    numerical_features = [
        'salary', 'performance_rating', 'satisfaction_score',
        'avg_hours_worked_prev_90d', 'num_activities_prev_90d',
        'num_distinct_projects_prev_90d', 'days_since_last_activity_at_cutoff',
        'employee_tenure_at_cutoff_days', 'days_since_last_promotion_at_cutoff'
    ]
    categorical_features = ['department']

    features = numerical_features + categorical_features
    target = 'will_attrit_in_next_6_months'

    X = employee_features_df[features]
    y = employee_features_df[target]

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    print(f"Total employees for modeling: {len(X)}")
    print(f"Attrition in next 6 months rate: {y.mean():.2%}")
    print(f"Training set size: {len(X_train)}, Test set size: {len(X_test)}")
    print(f"Train target distribution:\n{y_train.value_counts(normalize=True)}")
    print(f"Test target distribution:\n{y_test.value_counts(normalize=True)}")

    return X_train, X_test, y_train, y_test, numerical_features, categorical_features, employee_features_df

def visualize_data(employee_features_df):
    """
    Creates data visualizations to inspect relationships.
    """
    print("\n4. Data Visualization...")

    plt.style.use('seaborn-v0_8-darkgrid')
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Violin plot for satisfaction_score vs. attrition
    sns.violinplot(
        x='will_attrit_in_next_6_months',
        y='satisfaction_score',
        data=employee_features_df,
        ax=axes[0],
        palette='viridis'
    )
    axes[0].set_title('Satisfaction Score Distribution by Attrition Status', fontsize=14)
    axes[0].set_xlabel('Will Attrit in Next 6 Months (0=No, 1=Yes)', fontsize=12)
    axes[0].set_ylabel('Satisfaction Score', fontsize=12)
    axes[0].set_ylim(0.5, 5.5)

    # Plot 2: Stacked bar chart for attrition proportion by department
    attrition_by_dept = employee_features_df.groupby('department')['will_attrit_in_next_6_months'].value_counts(normalize=True).unstack().fillna(0)
    attrition_by_dept.plot(
        kind='bar',
        stacked=True,
        ax=axes[1],
        color=['lightcoral', 'skyblue'] # No, Yes
    )
    axes[1].set_title('Attrition Proportion by Department', fontsize=14)
    axes[1].set_xlabel('Department', fontsize=12)
    axes[1].set_ylabel('Proportion', fontsize=12)
    axes[1].legend(title='Will Attrit', labels=['No', 'Yes'])
    axes[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show()
    print("Visualizations displayed.")

def build_and_evaluate_ml_pipeline(X_train, X_test, y_train, y_test, numerical_features, categorical_features):
    """
    Builds, trains, and evaluates an ML pipeline for binary classification.
    """
    print("\n5. ML Pipeline & Evaluation...")

    # Create preprocessing steps
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
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
        ('classifier', HistGradientBoostingClassifier(random_state=42, class_weight='balanced'))
    ])

    # Train the pipeline
    print("Training ML pipeline...")
    model_pipeline.fit(X_train, y_train)
    print("Training complete.")

    # Predict probabilities on the test set
    y_pred_proba = model_pipeline.predict_proba(X_test)[:, 1]

    # Evaluate using ROC AUC Score
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    print(f"\nROC AUC Score on Test Set: {roc_auc:.4f}")

    # For classification report, convert probabilities to binary predictions (using 0.5 threshold)
    y_pred = (y_pred_proba >= 0.5).astype(int)
    print("\nClassification Report on Test Set:")
    print(classification_report(y_test, y_pred))

    print("ML pipeline built, trained, and evaluated successfully.")


if __name__ == "__main__":
    # 1. Generate Synthetic Data
    employees_df, work_activity_df = generate_synthetic_data()

    # 2. Load into SQLite & SQL Feature Engineering
    employee_features_df, global_prediction_cutoff_date = sql_feature_engineering(employees_df, work_activity_df)

    # 3. Pandas Feature Engineering & Binary Target Creation
    X_train, X_test, y_train, y_test, numerical_features, categorical_features, full_df_for_viz = \
        pandas_feature_engineering_and_target(employee_features_df, global_prediction_cutoff_date)

    # 4. Data Visualization
    visualize_data(full_df_for_viz)

    # 5. ML Pipeline & Evaluation
    build_and_evaluate_ml_pipeline(X_train, X_test, y_train, y_test, numerical_features, categorical_features)