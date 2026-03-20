import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, log_loss, confusion_matrix
import joblib  # For model persistence
import random  # For random choices in simulation
from pandasql import sqldf  # For SQL feature engineering

# Set a random seed for reproducibility
np.random.seed(42)
random.seed(42)

# 1. Data Simulation and Initial Setup
# Simulate customers
num_customers = 5000  # Reduced for faster execution in self-contained script
customers_data = {
    'customer_id': range(1, num_customers + 1),
    'age': np.random.randint(18, 70, num_customers),
    'loyalty_status': np.random.choice(['Bronze', 'Silver', 'Gold', 'Platinum'], num_customers, p=[0.4, 0.3, 0.2, 0.1]),
    'signup_date': [datetime.now() - timedelta(days=np.random.randint(1, 1000)) for _ in range(num_customers)]
}
customers_df = pd.DataFrame(customers_data)

# Simulate campaigns
num_campaigns = 250  # Reduced for faster execution
campaigns_data = {
    'campaign_id': range(1, num_campaigns + 1),
    'campaign_type': np.random.choice(['Promotional', 'Newsletter', 'Personalized Offer', 'Educational'], num_campaigns, p=[0.4, 0.3, 0.2, 0.1]),
    'segment': np.random.choice(['New User', 'Engaged', 'Churn Risk', 'High Value'], num_campaigns, p=[0.25, 0.25, 0.25, 0.25]),
    'send_date': [datetime.now() - timedelta(days=np.random.randint(1, 365)) + timedelta(days=i * 1) for i in range(num_campaigns)]  # Spread campaigns over a year, daily
}
campaigns_df = pd.DataFrame(campaigns_data)
campaigns_df = campaigns_df.sort_values(by='send_date').reset_index(drop=True)

# Generate potential email events first, then simulate clicks chronologically
# Create all possible customer-campaign combinations where campaign send_date > customer signup_date
potential_events = []
# Simulate a reasonable number of actual email send events (customer receives an email from a campaign)
for _ in range(30000):
    customer_idx = np.random.randint(0, num_customers)
    customer = customers_df.iloc[customer_idx]

    # Select a campaign that occurred after the customer's signup date
    eligible_campaigns = campaigns_df[campaigns_df['send_date'] > customer['signup_date']]
    if eligible_campaigns.empty:
        continue  # Skip if no eligible campaigns for this customer

    campaign_idx = np.random.randint(0, len(eligible_campaigns))
    campaign = eligible_campaigns.iloc[campaign_idx]

    potential_events.append({
        'customer_id': customer['customer_id'],
        'campaign_id': campaign['campaign_id'],
        'send_date': campaign['send_date']  # Temporarily include for sorting
    })

email_events_base_df = pd.DataFrame(potential_events)
# Remove duplicate customer-campaign pairs to avoid redundant events for the same email
email_events_base_df = email_events_base_df.drop_duplicates(subset=['customer_id', 'campaign_id']).reset_index(drop=True)

# Sort events chronologically to simulate actual interaction flow
email_events_base_df = email_events_base_df.sort_values(by=['customer_id', 'send_date']).reset_index(drop=True)

# Initial customer engagement scores (base CTR probability) for dynamic click simulation
# This score will evolve with customer interactions, creating a strong sequential signal.
customer_engagement_scores = {cid: np.random.uniform(0.02, 0.08) for cid in customers_df['customer_id']}

all_email_events_with_clicks = []
event_id_counter = 1

print("Simulating email click events chronologically to build a realistic signal...")
for idx, row in email_events_base_df.iterrows():
    customer_id = row['customer_id']
    campaign_id = row['campaign_id']

    customer_info = customers_df[customers_df['customer_id'] == customer_id].iloc[0]
    campaign_info = campaigns_df[campaigns_df['campaign_id'] == campaign_id].iloc[0]

    # Calculate probability of click based on various factors
    p_click = customer_engagement_scores[customer_id]

    # Loyalty status influence
    if customer_info['loyalty_status'] == 'Platinum': p_click += 0.05
    elif customer_info['loyalty_status'] == 'Gold': p_click += 0.03

    # Campaign type influence
    if campaign_info['campaign_type'] == 'Personalized Offer': p_click += 0.04
    elif campaign_info['campaign_type'] == 'Educational': p_click += 0.02
    elif campaign_info['campaign_type'] == 'Promotional': p_click -= 0.01  # Slightly lower for promos

    # Age influence (e.g., older customers might be slightly more stable clickers, younger more volatile)
    # Let's say age > 40 gives a slight boost, age < 25 a slight penalty
    if customer_info['age'] > 40: p_click += 0.005
    elif customer_info['age'] < 25: p_click -= 0.005

    # Clip probability to a reasonable range
    p_click = np.clip(p_click, 0.01, 0.20)  # Ensure p_click is between 1% and 20%

    is_clicked = int(np.random.rand() < p_click)

    # Update customer engagement score based on click outcome for future emails
    if is_clicked:
        customer_engagement_scores[customer_id] = min(customer_engagement_scores[customer_id] + 0.01, 0.15)
    else:
        customer_engagement_scores[customer_id] = max(customer_engagement_scores[customer_id] - 0.005, 0.01)

    all_email_events_with_clicks.append({
        'event_id': event_id_counter,
        'customer_id': customer_id,
        'campaign_id': campaign_id,
        'is_clicked': is_clicked
    })
    event_id_counter += 1

email_events_df = pd.DataFrame(all_email_events_with_clicks)
print(f"Finished simulating {len(email_events_df)} email click events.")

# 2. Advanced SQL Feature Engineering for Sequential Data
# Using pandasql for the initial join and projection, and then Pandas for complex sequential features.
# datetime columns are converted to strings for pandasql compatibility with SQL queries.

customers_df['signup_date_str'] = customers_df['signup_date'].dt.strftime('%Y-%m-%d')
campaigns_df['send_date_str'] = campaigns_df['send_date'].dt.strftime('%Y-%m-%d')

query = """
SELECT
    ee.event_id,
    ee.customer_id,
    ee.campaign_id,
    ee.is_clicked,
    c.age,
    c.loyalty_status,
    c.signup_date_str AS signup_date_sql, -- Using alias for clarity in SQL output
    camp.campaign_type,
    camp.segment,
    camp.send_date_str AS send_date_sql -- Using alias for clarity in SQL output
FROM
    email_events_df AS ee
LEFT JOIN
    customers_df AS c ON ee.customer_id = c.customer_id
LEFT JOIN
    campaigns_df AS camp ON ee.campaign_id = camp.campaign_id
ORDER BY
    ee.customer_id, camp.send_date_str
"""
print("--- SQL Feature Engineering (Initial Join) ---")
# Use globals() for sqldf to find the dataframes in the current scope
feature_df = sqldf(query, globals())
print(f"Shape after initial SQL join: {feature_df.shape}")

# Convert date strings back to datetime objects
feature_df['signup_date'] = pd.to_datetime(feature_df['signup_date_sql'])
feature_df['send_date'] = pd.to_datetime(feature_df['send_date_sql'])
# Drop the temporary string columns used for SQL
feature_df = feature_df.drop(columns=['signup_date_sql', 'send_date_sql'])

# Ensure chronological sort is maintained for sequential feature engineering in Pandas
feature_df = feature_df.sort_values(by=['customer_id', 'send_date']).reset_index(drop=True)

# 3. Data Loading and Python-based Feature Refinement
print("\n--- Python-based Feature Refinement ---")

# Engineer sequential features in Pandas
# customer_prior_total_emails_sent: cumcount() is 0-indexed, so it represents the number of emails *before* the current one.
feature_df['customer_prior_total_emails_sent'] = feature_df.groupby('customer_id').cumcount().astype(float)
# customer_prior_emails_clicked: cumsum() + shift(1) correctly gets the sum of clicks *before* the current event.
feature_df['customer_prior_emails_clicked'] = feature_df.groupby('customer_id')['is_clicked'].cumsum().shift(1).fillna(0).astype(float)

# days_since_last_customer_email_send: Calculate difference from previous send_date for the same customer
feature_df['days_since_last_customer_email_send'] = feature_df.groupby('customer_id')['send_date'].diff().dt.days

# Engineer derived features
# customer_prior_click_rate: Handle division by zero using np.where
feature_df['customer_prior_click_rate'] = np.where(
    feature_df['customer_prior_total_emails_sent'] > 0,
    feature_df['customer_prior_emails_clicked'] / feature_df['customer_prior_total_emails_sent'],
    0.0  # Fill with 0.0 where no prior emails (denominator is 0)
)
feature_df['days_since_signup_at_send'] = (feature_df['send_date'] - feature_df['signup_date']).dt.days

# Handle NaN values for engineered features as per hint:
# For days_since_last_customer_email_send, replace NaN (which occurs for the first email for a customer)
# with the days between their signup date and the current email's send date.
feature_df['days_since_last_customer_email_send'] = feature_df['days_since_last_customer_email_send'].fillna(feature_df['days_since_signup_at_send'])

# Impute age (if any NaNs, though simulation prevents it here)
if feature_df['age'].isnull().any():
    feature_df['age'] = feature_df['age'].fillna(feature_df['age'].median())

# Display basic stats and sample data
print(f"Total events: {len(feature_df)}")
print(f"Overall Click-Through Rate (CTR): {feature_df['is_clicked'].mean():.4f}")
print("Sample of engineered features and target (first 10 rows):")
print(feature_df[['customer_id', 'send_date', 'is_clicked', 'customer_prior_total_emails_sent',
                  'customer_prior_emails_clicked', 'customer_prior_click_rate',
                  'days_since_last_customer_email_send', 'days_since_signup_at_send']].head(10).to_string())


# 4. Feature Selection, Categorization, and Data Splitting
print("\n--- Feature Selection and Data Splitting ---")

# Define features and target
TARGET = 'is_clicked'
FEATURES = [
    'age',
    'loyalty_status',
    'campaign_type',
    'segment',
    'customer_prior_total_emails_sent',
    'customer_prior_emails_clicked',
    'customer_prior_click_rate',
    'days_since_last_customer_email_send',
    'days_since_signup_at_send'
]

numerical_features = [
    'age',
    'customer_prior_total_emails_sent',
    'customer_prior_emails_clicked',
    'customer_prior_click_rate',
    'days_since_last_customer_email_send',
    'days_since_signup_at_send'
]
categorical_features = [
    'loyalty_status',
    'campaign_type',
    'segment'
]

X = feature_df[FEATURES + ['send_date']]  # Keep send_date for chronological split
y = feature_df[TARGET]

# Chronological split: Sort by send_date to ensure older events are in train, newer in test.
X_sorted = X.sort_values(by='send_date')
y_sorted = y.loc[X_sorted.index]

# Define split points (e.g., 70% train, 15% validation, 15% test)
train_split_idx = int(len(X_sorted) * 0.70)
val_split_idx = int(len(X_sorted) * 0.85)

X_train_full = X_sorted.iloc[:train_split_idx]
y_train = y_sorted.iloc[:train_split_idx]

X_val_full = X_sorted.iloc[train_split_idx:val_split_idx]
y_val = y_sorted.iloc[train_split_idx:val_split_idx]

X_test_full = X_sorted.iloc[val_split_idx:]
y_test = y_sorted.iloc[val_split_idx:]

# Drop 'send_date' from feature sets after splitting
X_train = X_train_full.drop(columns=['send_date'])
X_val = X_val_full.drop(columns=['send_date'])
X_test = X_test_full.drop(columns=['send_date'])

print(f"Train data size: {len(X_train)} (CTR: {y_train.mean():.4f})")
print(f"Validation data size: {len(X_val)} (CTR: {y_val.mean():.4f})")
print(f"Test data size: {len(X_test)} (CTR: {y_test.mean():.4f})")
print(f"Train send date range: {X_train_full['send_date'].min().date()} to {X_train_full['send_date'].max().date()}")
print(f"Validation send date range: {X_val_full['send_date'].min().date()} to {X_val_full['send_date'].max().date()}")
print(f"Test send date range: {X_test_full['send_date'].min().date()} to {X_test_full['send_date'].max().date()}")


# 5. Machine Learning Pipeline Construction and Training
print("\n--- Machine Learning Pipeline Construction and Training ---")

# Preprocessing steps for numerical and categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Model selection: XGBoost Classifier, suitable for tabular data and handles class imbalance
# Calculate scale_pos_weight for handling class imbalance (sum(negative instances) / sum(positive instances))
neg_count = y_train.value_counts().get(0, 0)
pos_count = y_train.value_counts().get(1, 0)
scale_pos_weight_value = neg_count / pos_count if pos_count > 0 else 1.0 # Default to 1 if no positive samples

print(f"Calculated scale_pos_weight: {scale_pos_weight_value:.2f}")

# Define the XGBoost model with optimized hyperparameters and class imbalance handling
xgb_model = XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    use_label_encoder=False,  # Suppress the warning
    random_state=42,
    n_estimators=200,         # Number of boosting rounds
    learning_rate=0.05,       # Step size shrinkage
    max_depth=7,              # Maximum depth of a tree
    subsample=0.7,            # Subsample ratio of the training instance
    colsample_bytree=0.7,     # Subsample ratio of columns when constructing each tree
    gamma=0.2,                # Minimum loss reduction required to make a further partition
    scale_pos_weight=scale_pos_weight_value,  # Crucial for imbalanced datasets
    n_jobs=-1                 # Use all available cores
)

# Create the full scikit-learn pipeline
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('classifier', xgb_model)])

# Train the pipeline on the training data
print("Training the model...")
pipeline.fit(X_train, y_train)
print("Model training complete.")


# 6. Model Evaluation and Hyperparameter Tuning
print("\n--- Model Evaluation ---")

def evaluate_model(model, X_data, y_true, data_name="Validation"):
    """Evaluates the model and prints key classification metrics."""
    y_pred_proba = model.predict_proba(X_data)[:, 1]

    # For imbalanced data, tuning the classification threshold can be beneficial.
    # Here, we use a standard threshold of 0.5 for general evaluation.
    y_pred = (y_pred_proba > 0.5).astype(int)

    auc_roc = roc_auc_score(y_true, y_pred_proba)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    logloss = log_loss(y_true, y_pred_proba)

    print(f"\n{data_name} Set Evaluation:")
    print(f"  AUC-ROC: {auc_roc:.4f}")
    print(f"  Precision (Class 1 - Clicks): {precision:.4f}")
    print(f"  Recall (Class 1 - Clicks): {recall:.4f}")
    print(f"  F1-Score (Class 1 - Clicks): {f1:.4f}")
    print(f"  Log Loss: {logloss:.4f}")
    print(f"  Confusion Matrix:\n{confusion_matrix(y_true, y_pred)}")

# Evaluate on the validation set
evaluate_model(pipeline, X_val, y_val, "Validation")

# Hyperparameter tuning (Conceptual outline)
# In a full project, this step would involve using GridSearchCV or RandomizedSearchCV
# to find optimal hyperparameters for the pipeline on the training and validation sets.
# For this self-contained script, we'll use the initially defined pipeline as the 'best' one
# to manage execution time and complexity.
best_pipeline = pipeline

# Final, unbiased evaluation on the held-out test set
evaluate_model(best_pipeline, X_test, y_test, "Test")


# 7. Model Persistence and Deployment Considerations
print("\n--- Model Persistence and Deployment Considerations ---")

# Save the fully trained and optimized machine learning pipeline
model_filename = 'email_ctr_prediction_pipeline.joblib'
joblib.dump(best_pipeline, model_filename)
print(f"Trained model pipeline saved to {model_filename}")

print("\nDeployment Considerations:")
print("1.  **Feature Engineering at Scale:** For real-time prediction requests, the sequential features "
      "(`customer_prior_total_emails_sent`, `customer_prior_emails_clicked`, `days_since_last_customer_email_send`) "
      "need to be calculated efficiently in a production environment. This often involves a real-time feature store "
      "or streaming data pipelines that keep customer interaction histories updated for quick lookups.")
print("2.  **Latency Requirements:** Predictions for new emails might be required in milliseconds. The preprocessing "
      "steps and the model itself must be optimized for low latency inference. Batch predictions can be used for "
      "scheduled email campaigns, while real-time predictions are needed for trigger-based or personalized emails.")
print("3.  **Data Ingestion and Freshness:** New customer profiles, campaign details, and most importantly, new "
      "email interaction logs (including `is_clicked` outcomes) must continuously update the underlying data "
      "sources. Stale features will lead to degraded model performance.")
print("4.  **Monitoring:** Implement robust monitoring for model performance (e.g., AUC-ROC, Precision, Recall, "
      "actual CTR lift compared to predictions) and data drift (changes in feature distributions, target distribution). "
      "Alerting mechanisms should be in place for significant drops in performance or shifts in data.")
print("5.  **Retraining Schedules:** Establish a clear retraining strategy (e.g., daily, weekly, monthly, or "
      "event-driven based on drift detection) to adapt the model to new trends, customer behaviors, and campaign types. "
      "This often involves an automated MLOps pipeline.")
print("6.  **A/B Testing Integration:** To rigorously validate the model's impact, integrate it into an A/B testing "
      "framework. Compare model-driven targeting against existing rules-based approaches or random targeting to "
      "quantify the business value (e.g., increased CTR, conversion rates).")
print("7.  **Explainability:** For models used in marketing, understanding feature importance and individual prediction "
      "explanations (e.g., using SHAP or LIME) can provide valuable insights to marketing teams, helping them refine "
      "campaign strategies and content.")
print("8.  **Infrastructure:** Deploy the model as a scalable microservice (e.g., using a web framework like Flask/FastAPI, "
      "containerized with Docker, orchestrated with Kubernetes) to serve predictions via an API, capable of handling "
      "the expected request volume.")