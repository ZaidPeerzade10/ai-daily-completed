import pandas as pd
import numpy as np
import datetime
import sqlite3
import warnings
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    confusion_matrix,
    average_precision_score,
)
from sklearn.utils import resample

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# --- 1. Synthetic Data Generation and Initial Target Definition ---

np.random.seed(42)
num_leads = 2000
num_interactions = 10000

# Global Prediction Cutoff Date
GLOBAL_PREDICTION_CUTOFF_DATE = pd.to_datetime("2023-10-31")

# Sales Leads Data
lead_ids = np.arange(1, num_leads + 1)
ages = np.random.randint(20, 60, num_leads).astype(float)
ages[np.random.choice(num_leads, int(num_leads * 0.05), replace=False)] = np.nan # Introduce NaNs
company_sizes = np.random.choice([10, 50, 100, 500, 1000], num_leads)
industries = np.random.choice(
    ["Tech", "Finance", "Healthcare", "Retail", "Manufacturing"], num_leads
)
regions = np.random.choice(["North", "South", "East", "West"], num_leads)
lead_sources = np.random.choice(
    ["Webinar", "Referral", "Cold Call", "Paid Ad"], num_leads
)

# Simulate conversion dates
# Only a fraction of leads convert
conversion_prob = 0.15
actual_conversion_dates = []
for _ in range(num_leads):
    if np.random.rand() < conversion_prob:
        # Convert within a reasonable window after cutoff, for the target
        # or before, for historical interactions
        conversion_offset = np.random.randint(-120, 90) # Days relative to cutoff
        conv_date = GLOBAL_PREDICTION_CUTOFF_DATE + pd.Timedelta(days=int(conversion_offset))
        actual_conversion_dates.append(conv_date)
    else:
        actual_conversion_dates.append(pd.NaT) # No conversion

sales_leads_df = pd.DataFrame(
    {
        "lead_id": lead_ids,
        "age": ages,
        "company_size": company_sizes,
        "industry": industries,
        "region": regions,
        "lead_source": lead_sources,
        "_actual_conversion_date": actual_conversion_dates,
    }
)

# Lead Interactions Data
interaction_lead_ids = np.random.choice(lead_ids, num_interactions)
interaction_dates = [
    GLOBAL_PREDICTION_CUTOFF_DATE - pd.Timedelta(days=np.random.randint(1, 365))
    for _ in range(num_interactions)
] # Interactions up to 1 year before cutoff
interaction_types = np.random.choice(
    ["Demo Request", "Email Open", "Website Visit", "Download Guide"],
    num_interactions,
    p=[0.1, 0.4, 0.3, 0.2],
)

lead_interactions_df = pd.DataFrame(
    {
        "lead_id": interaction_lead_ids,
        "interaction_date": interaction_dates,
        "interaction_type": interaction_types,
    }
)

print(f"--- Synthetic Data Generated ---")
print(f"Sales Leads (first 5 rows):\n{sales_leads_df.head()}")
print(f"Lead Interactions (first 5 rows):\n{lead_interactions_df.head()}")
print(f"Prediction Cutoff Date: {GLOBAL_PREDICTION_CUTOFF_DATE}\n")

# --- 2. SQL-based Time-Series Feature Engineering ---

# Create an in-memory SQLite database
conn = sqlite3.connect(":memory:")

# Convert DataFrames to SQL tables
sales_leads_df.to_sql("sales_leads", conn, index=False, if_exists="replace")
lead_interactions_df.to_sql("lead_interactions", conn, index=False, if_exists="replace")

# SQL query for time-series feature engineering
sql_query = f"""
SELECT
    sl.lead_id,
    COUNT(li.interaction_id) AS num_interactions_total,
    MAX(julianday('{GLOBAL_PREDICTION_CUTOFF_DATE.strftime('%Y-%m-%d')}') - julianday(li.interaction_date)) AS days_since_last_interaction,

    SUM(CASE WHEN li.interaction_type = 'Demo Request' AND (julianday('{GLOBAL_PREDICTION_CUTOFF_DATE.strftime('%Y-%m-%d')}') - julianday(li.interaction_date)) <= 7 THEN 1 ELSE 0 END) AS num_demo_requests_prev_7d,
    SUM(CASE WHEN li.interaction_type = 'Email Open' AND (julianday('{GLOBAL_PREDICTION_CUTOFF_DATE.strftime('%Y-%m-%d')}') - julianday(li.interaction_date)) <= 7 THEN 1 ELSE 0 END) AS num_email_opens_prev_7d,
    SUM(CASE WHEN li.interaction_type = 'Website Visit' AND (julianday('{GLOBAL_PREDICTION_CUTOFF_DATE.strftime('%Y-%m-%d')}') - julianday(li.interaction_date)) <= 7 THEN 1 ELSE 0 END) AS num_website_visits_prev_7d,

    SUM(CASE WHEN li.interaction_type = 'Demo Request' AND (julianday('{GLOBAL_PREDICTION_CUTOFF_DATE.strftime('%Y-%m-%d')}') - julianday(li.interaction_date)) <= 30 THEN 1 ELSE 0 END) AS num_demo_requests_prev_30d,
    SUM(CASE WHEN li.interaction_type = 'Email Open' AND (julianday('{GLOBAL_PREDICTION_CUTOFF_DATE.strftime('%Y-%m-%d')}') - julianday(li.interaction_date)) <= 30 THEN 1 ELSE 0 END) AS num_email_opens_prev_30d,
    SUM(CASE WHEN li.interaction_type = 'Website Visit' AND (julianday('{GLOBAL_PREDICTION_CUTOFF_DATE.strftime('%Y-%m-%d')}') - julianday(li.interaction_date)) <= 30 THEN 1 ELSE 0 END) AS num_website_visits_prev_30d,

    SUM(CASE WHEN li.interaction_type = 'Demo Request' AND (julianday('{GLOBAL_PREDICTION_CUTOFF_DATE.strftime('%Y-%m-%d')}') - julianday(li.interaction_date)) <= 60 THEN 1 ELSE 0 END) AS num_demo_requests_prev_60d,
    SUM(CASE WHEN li.interaction_type = 'Email Open' AND (julianday('{GLOBAL_PREDICTION_CUTOFF_DATE.strftime('%Y-%m-%d')}') - julianday(li.interaction_date)) <= 60 THEN 1 ELSE 0 END) AS num_email_opens_prev_60d,
    SUM(CASE WHEN li.interaction_type = 'Website Visit' AND (julianday('{GLOBAL_PREDICTION_CUTOFF_DATE.strftime('%Y-%m-%d')}') - julianday(li.interaction_date)) <= 60 THEN 1 ELSE 0 END) AS num_website_visits_prev_60d
FROM
    sales_leads sl
LEFT JOIN (
    SELECT
        ROW_NUMBER() OVER (ORDER BY interaction_date) AS interaction_id, -- dummy ID for COUNT
        lead_id,
        interaction_date,
        interaction_type
    FROM
        lead_interactions
    WHERE
        julianday(interaction_date) <= julianday('{GLOBAL_PREDICTION_CUTOFF_DATE.strftime('%Y-%m-%d')}')
) li ON sl.lead_id = li.lead_id
GROUP BY
    sl.lead_id
"""

# Execute the query and fetch results
historical_features_df = pd.read_sql_query(sql_query, conn)
conn.close()

# Merge features back to the main sales_leads_df
# Use `how='left'` to ensure all leads are present, even those without interactions
features_df = sales_leads_df.merge(historical_features_df, on="lead_id", how="left")

# Fill NaNs for leads with no interactions (these features would be NULL from LEFT JOIN)
# For interaction counts, 0 means no interactions
interaction_count_cols = [col for col in features_df.columns if col.startswith('num_')]
features_df[interaction_count_cols] = features_df[interaction_count_cols].fillna(0)

# For days_since_last_interaction, a very large number indicates no recent interaction or no interaction at all
features_df['days_since_last_interaction'] = features_df['days_since_last_interaction'].fillna(365 * 10) # 10 years

print(f"--- SQL-based Time-Series Feature Engineering Complete ---")
print(f"Features after SQL aggregation (first 5 rows):\n{features_df.head()}")
print(f"Missing values after initial imputation (num_*, days_since_last_interaction):\n"
      f"{features_df[interaction_count_cols + ['days_since_last_interaction']].isnull().sum()}\n")


# --- 3. Feature Preprocessing and Transformation ---

# Define feature types
numerical_features_median_impute = ["age", "company_size"]
numerical_features_zero_impute = [col for col in features_df.columns if col.startswith('num_')]
numerical_features_large_impute = ["days_since_last_interaction"]
categorical_features = ["industry", "region", "lead_source"]

# Create preprocessing pipelines for different feature types
numeric_median_pipeline = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

numeric_zero_pipeline = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
        ("scaler", StandardScaler()),
    ]
)

numeric_large_pipeline = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="constant", fill_value=365 * 10)), # Use a consistent large number
        ("scaler", StandardScaler()),
    ]
)

categorical_pipeline = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)

# Combine pipelines using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ("num_median", numeric_median_pipeline, numerical_features_median_impute),
        ("num_zero", numeric_zero_pipeline, numerical_features_zero_impute),
        ("num_large", numeric_large_pipeline, numerical_features_large_impute),
        ("cat", categorical_pipeline, categorical_features),
    ],
    remainder="drop",  # Drop lead_id and _actual_conversion_date, etc.
)

print(f"--- Feature Preprocessing Pipeline Defined ---")


# --- 4. Final Target Variable Creation and Data Splitting ---

# Finalize the target variable
conversion_window_start = GLOBAL_PREDICTION_CUTOFF_DATE
conversion_window_end = GLOBAL_PREDICTION_CUTOFF_DATE + pd.Timedelta(days=30)

features_df["will_convert_next_30d"] = (
    (features_df["_actual_conversion_date"] >= conversion_window_start)
    & (features_df["_actual_conversion_date"] < conversion_window_end)
).astype(int)

# Separate features (X) and target (y)
X = features_df.drop(
    columns=["lead_id", "_actual_conversion_date", "will_convert_next_30d"]
)
y = features_df["will_convert_next_30d"]

# Split data into training and testing sets with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"--- Target Variable Finalized and Data Split ---")
print(f"Original target distribution:\n{y.value_counts(normalize=True)}")
print(f"Training target distribution:\n{y_train.value_counts(normalize=True)}")
print(f"Test target distribution:\n{y_test.value_counts(normalize=True)}\n")


# --- 5. Model Selection, Training, and Hyperparameter Optimization with Imbalance Handling ---

# Define models with class imbalance handling
models = {
    "Logistic Regression": LogisticRegression(random_state=42, solver="liblinear", class_weight="balanced"),
    "Random Forest": RandomForestClassifier(random_state=42, class_weight="balanced"),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42), # GBC does not have class_weight directly. Can use sample_weight during fit, or resample.
}

# Define hyperparameter grids
param_grids = {
    "Logistic Regression": {
        "classifier__C": [0.01, 0.1, 1, 10],
    },
    "Random Forest": {
        "classifier__n_estimators": [50, 100, 200],
        "classifier__max_depth": [5, 10, None],
    },
    "Gradient Boosting": {
        "classifier__n_estimators": [50, 100, 200],
        "classifier__learning_rate": [0.01, 0.1, 0.2],
    },
}

best_models = {}
print("--- Training Models and Performing Hyperparameter Optimization ---")
for name, model in models.items():
    print(f"Training {name}...")
    pipeline = Pipeline(
        steps=[("preprocessor", preprocessor), ("classifier", model)]
    )

    grid_search = GridSearchCV(
        pipeline,
        param_grids[name],
        cv=3,
        scoring="roc_auc",  # ROC AUC is robust for imbalanced data
        n_jobs=-1,
        verbose=0,
    )
    grid_search.fit(X_train, y_train)
    best_models[name] = grid_search.best_estimator_
    print(f"Best parameters for {name}: {grid_search.best_params_}")
    print(f"Best ROC AUC score for {name} on CV: {grid_search.best_score_:.4f}\n")


# --- 6. Model Evaluation and Interpretation ---

print("--- Model Evaluation on Test Set ---")
for name, model in best_models.items():
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    print(f"\n--- {name} ---")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print(f"ROC AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")
    print(f"Average Precision Score: {average_precision_score(y_test, y_pred_proba):.4f}")

# Feature Importance for the best performing model (e.g., Random Forest or Gradient Boosting)
# We need to get the feature names after preprocessing
print("\n--- Feature Importance Analysis ---")
best_model_name = max(best_models, key=lambda name: roc_auc_score(y_test, best_models[name].predict_proba(X_test)[:, 1]))
best_model_pipeline = best_models[best_model_name]
print(f"Analyzing feature importance for: {best_model_name}")

# Get feature names after one-hot encoding and other transformations
ohe_feature_names = best_model_pipeline.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_features)
all_feature_names = numerical_features_median_impute + numerical_features_zero_impute + numerical_features_large_impute + list(ohe_feature_names)

# Handle cases where the model might not have feature_importances_ (e.g., Logistic Regression)
if hasattr(best_model_pipeline.named_steps['classifier'], 'feature_importances_'):
    importances = best_model_pipeline.named_steps['classifier'].feature_importances_
    feature_importances = pd.Series(importances, index=all_feature_names)
    print("Top 10 Feature Importances:")
    print(feature_importances.nlargest(10))
elif hasattr(best_model_pipeline.named_steps['classifier'], 'coef_'):
    # For Logistic Regression, coefficients are used (absolute value for magnitude)
    coefs = best_model_pipeline.named_steps['classifier'].coef_[0]
    feature_importances = pd.Series(np.abs(coefs), index=all_feature_names)
    print("Top 10 Feature Coefficients (Absolute Value for Magnitude):")
    print(feature_importances.nlargest(10))
else:
    print("Feature importances/coefficients not available for the best model.")

print("\n--- Pipeline Execution Complete ---")