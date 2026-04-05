import pandas as pd
import numpy as np
from datetime import timedelta
import random

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve, RocCurveDisplay
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')

# --- 1. Generate Synthetic Data ---

np.random.seed(42)
random.seed(42)

NUM_USERS = 1000
START_DATE = pd.Timestamp('2023-08-01')
END_DATE = pd.Timestamp('2023-11-30') # Data range for events and tickets
GLOBAL_ANALYSIS_CUTOFF_DATE = pd.Timestamp('2023-10-01') # The "now" point for analysis

# Users DataFrame
user_ids = [f'user_{i:04d}' for i in range(NUM_USERS)]
signup_dates = pd.to_datetime(START_DATE + pd.to_timedelta(np.random.randint(0, (GLOBAL_ANALYSIS_CUTOFF_DATE - START_DATE).days, NUM_USERS), unit='D'))
countries = np.random.choice(['USA', 'CAN', 'GBR', 'DEU', 'FRA'], NUM_USERS)
user_types = np.random.choice(['free', 'premium', 'enterprise'], NUM_USERS, p=[0.6, 0.3, 0.1])

users_df = pd.DataFrame({
    'user_id': user_ids,
    'signup_date': signup_dates,
    'country': countries,
    'user_type': user_types
})

# App Events DataFrame
app_events_data = []
event_types = ['login', 'view_dashboard', 'feature_X_use', 'feature_Y_use', 'error_event']
# Identify a subset of users who will be 'error-prone'
error_prone_users = np.random.choice(user_ids, int(NUM_USERS * 0.2), replace=False) # 20% of users

for user_id in user_ids:
    user_signup_date = users_df[users_df['user_id'] == user_id]['signup_date'].iloc[0]
    
    # Generate events throughout the defined data range (START_DATE to END_DATE)
    # Ensure events don't occur before signup_date
    actual_event_start_date = max(user_signup_date, START_DATE)
    
    num_base_events = np.random.randint(20, 100) 
    for _ in range(num_base_events):
        event_timestamp = actual_event_start_date + pd.to_timedelta(np.random.randint(0, (END_DATE - actual_event_start_date).days + 1), unit='D') \
                          + pd.to_timedelta(np.random.randint(0, 24*60*60-1), unit='S')
        event_type = np.random.choice(event_types, p=[0.2, 0.3, 0.2, 0.2, 0.1])
        
        duration = 0
        if event_type != 'error_event':
            duration = np.random.randint(5, 600) # 5 seconds to 10 minutes
        
        app_events_data.append([user_id, event_timestamp, event_type, duration])
        
    # For error-prone users, add more 'error_event's specifically in the 30 days before cutoff
    if user_id in error_prone_users:
        num_extra_errors = np.random.randint(5, 20)
        for _ in range(num_extra_errors):
            event_timestamp = GLOBAL_ANALYSIS_CUTOFF_DATE - pd.to_timedelta(np.random.randint(1, 31), unit='D') \
                              + pd.to_timedelta(np.random.randint(0, 24*60*60-1), unit='S')
            
            # Only add if the event timestamp is not before user's signup_date
            if event_timestamp >= user_signup_date:
                app_events_data.append([user_id, event_timestamp, 'error_event', 0])

app_events_df = pd.DataFrame(app_events_data, columns=['user_id', 'event_timestamp', 'event_type', 'duration_seconds'])
app_events_df = app_events_df.sort_values(['user_id', 'event_timestamp']).reset_index(drop=True)

# Support Tickets DataFrame
support_tickets_data = []

# Define the future window for ticket prediction
TICKET_FORECAST_START = GLOBAL_ANALYSIS_CUTOFF_DATE
TICKET_FORECAST_END = GLOBAL_ANALYSIS_CUTOFF_DATE + pd.Timedelta(days=30)

# Simulate ticket patterns: error-prone users are more likely to generate tickets in the next 30 days
for user_id in user_ids:
    num_tickets = 0
    if user_id in error_prone_users:
        # Error-prone users have a higher chance of future tickets
        if np.random.rand() < 0.6: # 60% chance to have 1-3 tickets
            num_tickets = np.random.randint(1, 4) 
    else:
        # Non-error-prone users have a smaller chance
        if np.random.rand() < 0.05: # 5% chance to have 1 ticket
            num_tickets = 1

    for _ in range(num_tickets):
        ticket_timestamp = TICKET_FORECAST_START + pd.to_timedelta(np.random.randint(0, (TICKET_FORECAST_END - TICKET_FORECAST_START).days), unit='D') \
                           + pd.to_timedelta(np.random.randint(0, 24*60*60-1), unit='S')
        support_tickets_data.append([user_id, ticket_timestamp])

# Add some historical tickets for realism, not specifically tied to the future prediction window
# These tickets will be outside the TICKET_FORECAST_START/END
for user_id in np.random.choice(user_ids, int(NUM_USERS * 0.1), replace=False):
    if np.random.rand() < 0.5: # 50% chance for a historical ticket
        ticket_timestamp = START_DATE + pd.to_timedelta(np.random.randint(0, (GLOBAL_ANALYSIS_CUTOFF_DATE - START_DATE).days), unit='D') \
                           + pd.to_timedelta(np.random.randint(0, 24*60*60-1), unit='S')
        support_tickets_data.append([user_id, ticket_timestamp])

support_tickets_df = pd.DataFrame(support_tickets_data, columns=['user_id', 'ticket_timestamp'])
# Ensure all user_ids in support_tickets_df exist in users_df (should be true by design, but good check)
support_tickets_df = support_tickets_df[support_tickets_df['user_id'].isin(users_df['user_id'])].copy()
support_tickets_df = support_tickets_df.sort_values(['user_id', 'ticket_timestamp']).reset_index(drop=True)


print("--- Synthetic Data Generated ---")
print("Users DF Head:\n", users_df.head())
print("\nApp Events DF Head:\n", app_events_df.head())
print("\nSupport Tickets DF Head:\n", support_tickets_df.head())
print(f"\nTotal users: {len(users_df)}")
print(f"Total app events: {len(app_events_df)}")
print(f"Total support tickets: {len(support_tickets_df)}")


# --- 2. Define Analysis Window and SQL-like Feature Engineering ---

LOOKBACK_START_DATE = GLOBAL_ANALYSIS_CUTOFF_DATE - pd.Timedelta(days=30)
LOOKBACK_END_DATE = GLOBAL_ANALYSIS_CUTOFF_DATE # Exclusive end date

print(f"\n--- Feature Engineering for Analysis Window: {LOOKBACK_START_DATE} to {LOOKBACK_END_DATE} (exclusive) ---")

# Filter app events for the lookback period
recent_app_events_df = app_events_df[
    (app_events_df['event_timestamp'] >= LOOKBACK_START_DATE) &
    (app_events_df['event_timestamp'] < LOOKBACK_END_DATE)
].copy()

# Group by user_id to extract features
user_features_list = []

for user_id in users_df['user_id'].unique(): # Iterate through all users
    user_events = recent_app_events_df[recent_app_events_df['user_id'] == user_id]
    
    features = {'user_id': user_id}
    
    if not user_events.empty:
        # Total counts of various event_types
        event_counts = user_events['event_type'].value_counts()
        for event_type_val in event_types: # Ensure all event types are represented, even if count is 0
            features[f'{event_type_val}_count_last_30d'] = event_counts.get(event_type_val, 0)
        
        # Number of unique days with activity
        features['days_with_activity_last_30d'] = user_events['event_timestamp'].dt.date.nunique()
        
        # Binary flags for specific event types occurring
        features['has_error_event_last_30d'] = 1 if 'error_event' in user_events['event_type'].unique() else 0
        features['has_login_last_30d'] = 1 if 'login' in user_events['event_type'].unique() else 0

        # Average duration_seconds for events where duration is positive
        positive_duration_events = user_events[user_events['duration_seconds'] > 0]
        if not positive_duration_events.empty:
            features['avg_event_duration_last_30d'] = positive_duration_events['duration_seconds'].mean()
        else:
            features['avg_event_duration_last_30d'] = 0 # If no positive duration events, default to 0
    else:
        # Default features for users with no activity in the period
        for event_type_val in event_types:
            features[f'{event_type_val}_count_last_30d'] = 0
        features['days_with_activity_last_30d'] = 0
        features['has_error_event_last_30d'] = 0
        features['has_login_last_30d'] = 0
        features['avg_event_duration_last_30d'] = 0
    
    user_features_list.append(features)

user_event_features_df = pd.DataFrame(user_features_list)

print("\nUser Event Features DF Head:\n", user_event_features_df.head())
print("\nUser Event Features DF Info:\n", user_event_features_df.info())


# --- 3. Merge User Profiles, Engineer Additional Features, and Create Target Variable ---

# Combine event features with users_df
# Use a left merge to keep all users from users_df
df = pd.merge(users_df, user_event_features_df, on='user_id', how='left')

# Fill any remaining NaNs, which should primarily be for numerical features for users
# that had no app activity and might have been missed by explicit 0 initialization (though less likely now)
df = df.fillna(0) 

# Engineer additional features from users_df (e.g., tenure)
df['user_tenure_days'] = (GLOBAL_ANALYSIS_CUTOFF_DATE - df['signup_date']).dt.days

# Create the binary target variable
print(f"\n--- Creating Target Variable for Forecast Window: {TICKET_FORECAST_START} to {TICKET_FORECAST_END} (exclusive) ---")

forecast_support_tickets_df = support_tickets_df[
    (support_tickets_df['ticket_timestamp'] >= TICKET_FORECAST_START) &
    (support_tickets_df['ticket_timestamp'] < TICKET_FORECAST_END)
].copy()

# Identify users with at least one ticket in the forecast window
users_with_tickets = forecast_support_tickets_df['user_id'].unique()
df['has_support_ticket_next_30d'] = df['user_id'].isin(users_with_tickets).astype(int)

print("\nFinal Feature DF Head with Target:\n", df.head())
print("\nFinal Feature DF Info:\n", df.info())


# --- 4. Exploratory Data Analysis (EDA) and Data Preparation ---

print("\n--- Exploratory Data Analysis (EDA) ---")

# Check class imbalance
target_counts = df['has_support_ticket_next_30d'].value_counts(normalize=True)
print("\nTarget variable (has_support_ticket_next_30d) distribution:\n", target_counts)
if target_counts.iloc[0] > 0.75 or target_counts.iloc[1] > 0.75:
    print("\nNote: Significant class imbalance detected. `class_weight='balanced'` is used in the model.")

# Define features and target
X = df.drop(columns=['user_id', 'signup_date', 'has_support_ticket_next_30d'])
y = df['has_support_ticket_next_30d']

# Identify numerical and categorical features for the ColumnTransformer
numerical_features = X.select_dtypes(include=np.number).columns.tolist()
categorical_features = X.select_dtypes(include='object').columns.tolist()

print(f"\nNumerical Features: {numerical_features}")
print(f"Categorical Features: {categorical_features}")

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

print(f"\nTrain set shape: {X_train.shape}, Target shape: {y_train.shape}")
print(f"Test set shape: {X_test.shape}, Target shape: {y_test.shape}")

# Column Transformer setup for preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='passthrough' # Any columns not specified will be passed through (shouldn't be any here)
)


# --- 5. Model Training ---

print("\n--- Model Training ---")

# Define a machine learning pipeline
# RandomForestClassifier is chosen for its robustness and ability to provide feature importances.
# `class_weight='balanced'` is used to address potential class imbalance.
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42, class_weight='balanced', n_estimators=100))
])

print("\nTraining RandomForestClassifier pipeline...")
pipeline.fit(X_train, y_train)
print("Training complete.")


# --- 6. Model Evaluation and Interpretation ---

print("\n--- Model Evaluation ---")

y_pred = pipeline.predict(X_test)
y_proba = pipeline.predict_proba(X_test)[:, 1] # Probability of the positive class (Has Ticket)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, target_names=['No Ticket', 'Has Ticket']))

roc_auc = roc_auc_score(y_test, y_proba)
print(f"\nROC-AUC Score: {roc_auc:.4f}")

# Confusion Matrix
plt.figure(figsize=(6, 5))
disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=['No Ticket', 'Has Ticket'], cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()

# ROC Curve
plt.figure(figsize=(6, 5))
RocCurveDisplay.from_estimator(pipeline, X_test, y_test)
plt.title('ROC Curve')
plt.plot([0, 1], [0, 1], 'k--', label='Random Guess') # Add random guess line
plt.legend()
plt.show()

# Feature Importances (for RandomForestClassifier)
if isinstance(pipeline.named_steps['classifier'], RandomForestClassifier):
    print("\n--- Feature Importances ---")
    
    # Get feature names after preprocessing
    transformed_feature_names = numerical_features
    # Add one-hot encoded categorical feature names
    ohe_feature_names = pipeline.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(categorical_features)
    transformed_feature_names.extend(ohe_feature_names)
    
    importances = pipeline.named_steps['classifier'].feature_importances_
    
    feature_importances = pd.Series(importances, index=transformed_feature_names)
    
    # Sort and plot top N features
    top_n = 20
    feature_importances = feature_importances.sort_values(ascending=False)
    print(f"Top {top_n} Features by Importance:\n", feature_importances.head(top_n))

    plt.figure(figsize=(10, 8))
    sns.barplot(x=feature_importances.head(top_n), y=feature_importances.head(top_n).index)
    plt.title(f'Top {top_n} Feature Importances')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.show()

print("\n--- Sample Predictions (Test Set) ---")
sample_predictions_df = X_test.copy()
sample_predictions_df['Actual Ticket'] = y_test
sample_predictions_df['Predicted Ticket'] = y_pred
sample_predictions_df['Probability (Has Ticket)'] = y_proba.round(3)
print(sample_predictions_df[['Actual Ticket', 'Predicted Ticket', 'Probability (Has Ticket)']].head(10))

print("\n--- Pipeline Execution Complete ---")