import pandas as pd
import numpy as np
import sqlite3
import random
import datetime

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, SimpleImputer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, classification_report

# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)

# --- 1. Synthetic Data Generation ---

N_CUSTOMERS = 1000
N_BOOKINGS = 10000
N_ACTIVITIES = 30000

print("--- Generating Synthetic Data ---")

# Define global 'now' timestamp for data generation
now = pd.Timestamp.now()

# 1.1 Generate Customers and their Start Dates
customer_data = []
for i in range(N_CUSTOMERS):
    customer_id = f'CUST{i:04d}'
    # Customer start date 4 to 2 years ago
    customer_start_date = now - pd.Timedelta(days=np.random.randint(365 * 2, 365 * 4))
    customer_data.append({'customer_id': customer_id, 'customer_start_date': customer_start_date})
customers_df = pd.DataFrame(customer_data)

# Map customer_id to customer_start_date for easy lookup
customer_start_dates = customers_df.set_index('customer_id')['customer_start_date'].to_dict()

# 1.2 Generate Bookings Data
bookings_data = []
customer_ids_in_bookings = []
for i in range(N_BOOKINGS):
    booking_id = f'BOOK{i:05d}'
    customer_id = np.random.choice(customers_df['customer_id'])
    customer_ids_in_bookings.append(customer_id)

    # reservation_datetime must be after customer_start_date and up to now
    start_date = customer_start_dates[customer_id]
    reservation_datetime = start_date + pd.Timedelta(seconds=np.random.randint(0, int((now - start_date).total_seconds())))

    num_guests = np.random.randint(1, 9)
    booking_channel = np.random.choice(['Online', 'Phone', 'Walk-in', 'App'], p=[0.5, 0.3, 0.1, 0.1])
    special_requests_flag = np.random.rand() < 0.2
    customer_segment = np.random.choice(['Bronze', 'Silver', 'Gold', 'Platinum'], p=[0.4, 0.3, 0.2, 0.1])

    # Simulate realistic no-show patterns
    no_show_prob = 0.18 # Base no-show rate
    if customer_segment == 'Platinum':
        no_show_prob *= 0.5 # Platinum customers less likely to no-show
    elif customer_segment == 'Gold':
        no_show_prob *= 0.8
    if booking_channel == 'Online':
        no_show_prob *= 1.2 # Online bookings slightly higher no-show
    elif booking_channel == 'Walk-in':
        no_show_prob *= 0.5 # Walk-ins rarely no-show (already there)

    is_no_show = np.random.rand() < no_show_prob

    bookings_data.append({
        'booking_id': booking_id,
        'customer_id': customer_id,
        'reservation_datetime': reservation_datetime,
        'num_guests': num_guests,
        'booking_channel': booking_channel,
        'special_requests_flag': special_requests_flag,
        'customer_segment': customer_segment,
        'is_no_show': is_no_show
    })

bookings_df = pd.DataFrame(bookings_data)
bookings_df['reservation_datetime'] = pd.to_datetime(bookings_df['reservation_datetime'])
bookings_df['special_requests_flag'] = bookings_df['special_requests_flag'].astype(int) # For consistent handling

# Get the earliest reservation datetime for each customer
min_reservation_datetime_per_customer = bookings_df.groupby('customer_id')['reservation_datetime'].min()

# 1.3 Generate Customer Activity Data
activity_data = []
activity_types = ['reservation_made', 'reservation_cancelled', 'website_visit', 'loyalty_points_redeemed', 'takeout_order']

for i in range(N_ACTIVITIES):
    activity_id = f'ACT{i:05d}'
    customer_id = np.random.choice(customers_df['customer_id']) # Pick from all customers
    
    # Ensure activity_datetime is before the customer's earliest reservation (if any)
    # or before now if no reservations for that customer.
    # It must also be after customer_start_date.
    
    earliest_res_for_customer = min_reservation_datetime_per_customer.get(customer_id, now + pd.Timedelta(days=365)) # +1 year if no bookings
    
    # Max allowed activity date is min_reservation_datetime - 1 day, or now if min_reservation_datetime is in future
    max_activity_date = min(now, earliest_res_for_customer - pd.Timedelta(days=1))
    
    # Ensure max_activity_date is not before customer_start_date
    start_date = customer_start_dates[customer_id]
    if max_activity_date <= start_date:
        # If no valid range, make activity_datetime up to current_date - 1 day
        max_activity_date = now - pd.Timedelta(days=1)
        if max_activity_date <= start_date:
            start_date = max_activity_date - pd.Timedelta(days=30) # fallback, generate within last month
            if start_date <= customer_start_dates[customer_id]: # if this makes start_date too early
                start_date = customer_start_dates[customer_id]

    activity_datetime = start_date + pd.Timedelta(seconds=np.random.randint(0, int((max_activity_date - start_date).total_seconds())))

    activity_type = np.random.choice(activity_types)

    # Simulate patterns
    if activity_type == 'reservation_cancelled':
        # People who cancel are less likely to no-show for future bookings, but this activity should be historical
        pass 
    if activity_type == 'loyalty_points_redeemed':
        # Loyalty point redeemers are generally more engaged and less likely to no-show
        pass

    activity_data.append({
        'activity_id': activity_id,
        'customer_id': customer_id,
        'activity_datetime': activity_datetime,
        'activity_type': activity_type
    })

customer_activity_df = pd.DataFrame(activity_data)
customer_activity_df['activity_datetime'] = pd.to_datetime(customer_activity_df['activity_datetime'])
customer_activity_df = customer_activity_df.sort_values(by=['customer_id', 'activity_datetime']).reset_index(drop=True)

print(f"Bookings generated: {len(bookings_df)} rows")
print(f"Customer Activities generated: {len(customer_activity_df)} rows")
print(f"No-show rate: {bookings_df['is_no_show'].mean():.2f}")
print("Example Bookings Head:")
print(bookings_df.head())
print("\nExample Customer Activities Head:")
print(customer_activity_df.head())


# --- 2. Load into SQLite & SQL Feature Engineering ---

print("\n--- Loading Data into SQLite and SQL Feature Engineering ---")

# Create an in-memory SQLite database
conn = sqlite3.connect(':memory:')

# Load DataFrames into SQLite tables
bookings_df.to_sql('bookings', conn, index=False, if_exists='replace')
customer_activity_df.to_sql('customer_activities', conn, index=False, if_exists='replace')

# Define GLOBAL_PREDICTION_CUTOFF_DATE
GLOBAL_PREDICTION_CUTOFF_DATE = bookings_df['reservation_datetime'].max() - pd.Timedelta(weeks=4)
GLOBAL_PREDICTION_CUTOFF_DATE_STR = GLOBAL_PREDICTION_CUTOFF_DATE.strftime('%Y-%m-%d %H:%M:%S')

print(f"Global Prediction Cutoff Date: {GLOBAL_PREDICTION_CUTOFF_DATE}")

# SQL Query for feature engineering, ensuring no future leakage beyond GLOBAL_PREDICTION_CUTOFF_DATE
# And selecting only bookings AFTER the cutoff date for prediction
sql_query = f"""
WITH CustomerHistory AS (
    SELECT
        ca.customer_id,
        -- Count previous bookings (reservation_made activities) in the 12 months before or on cutoff
        SUM(CASE WHEN ca.activity_type = 'reservation_made'
                  AND julianday(ca.activity_datetime) >= julianday('{GLOBAL_PREDICTION_CUTOFF_DATE_STR}') - 365
                  AND julianday(ca.activity_datetime) <= julianday('{GLOBAL_PREDICTION_CUTOFF_DATE_STR}')
             THEN 1 ELSE 0 END) AS num_prev_bookings_customer_12m,
        -- Count previous cancellations in the 12 months before or on cutoff
        SUM(CASE WHEN ca.activity_type = 'reservation_cancelled'
                  AND julianday(ca.activity_datetime) >= julianday('{GLOBAL_PREDICTION_CUTOFF_DATE_STR}') - 365
                  AND julianday(ca.activity_datetime) <= julianday('{GLOBAL_PREDICTION_CUTOFF_DATE_STR}')
             THEN 1 ELSE 0 END) AS num_prev_cancellations_customer_12m,
        -- Most recent activity before or on cutoff
        MAX(CASE WHEN julianday(ca.activity_datetime) <= julianday('{GLOBAL_PREDICTION_CUTOFF_DATE_STR}')
                 THEN ca.activity_datetime ELSE NULL END) AS last_activity_datetime_at_cutoff,
        -- Count loyalty points redeemed in the 12 months before or on cutoff
        SUM(CASE WHEN ca.activity_type = 'loyalty_points_redeemed'
                  AND julianday(ca.activity_datetime) >= julianday('{GLOBAL_PREDICTION_CUTOFF_DATE_STR}') - 365
                  AND julianday(ca.activity_datetime) <= julianday('{GLOBAL_PREDICTION_CUTOFF_DATE_STR}')
             THEN 1 ELSE 0 END) AS num_loyalty_redeemed_customer_12m
    FROM
        customer_activities ca
    GROUP BY
        ca.customer_id
)
SELECT
    b.booking_id,
    b.customer_id,
    b.reservation_datetime,
    b.num_guests,
    b.booking_channel,
    b.special_requests_flag,
    b.customer_segment,
    b.is_no_show,
    '{GLOBAL_PREDICTION_CUTOFF_DATE_STR}' AS current_cutoff_date, -- Add cutoff date for consistency
    COALESCE(ch.num_prev_bookings_customer_12m, 0) AS num_prev_bookings_customer_12m,
    COALESCE(ch.num_prev_cancellations_customer_12m, 0) AS num_prev_cancellations_customer_12m,
    COALESCE(
        julianday('{GLOBAL_PREDICTION_CUTOFF_DATE_STR}') - julianday(ch.last_activity_datetime_at_cutoff),
        9999
    ) AS days_since_last_activity_at_cutoff,
    COALESCE(ch.num_loyalty_redeemed_customer_12m, 0) AS num_loyalty_redeemed_customer_12m
FROM
    bookings b
LEFT JOIN
    CustomerHistory ch ON b.customer_id = ch.customer_id
WHERE
    julianday(b.reservation_datetime) > julianday('{GLOBAL_PREDICTION_CUTOFF_DATE_STR}')
ORDER BY b.booking_id;
"""

booking_features_df = pd.read_sql_query(sql_query, conn)
conn.close()

print(f"Bookings after cutoff for prediction: {len(booking_features_df)} rows")
print("SQL Feature Engineering Results Head:")
print(booking_features_df.head())


# --- 3. Pandas Feature Engineering & Binary Target Creation ---

print("\n--- Pandas Feature Engineering & Data Preparation ---")

# Convert datetime columns
booking_features_df['reservation_datetime'] = pd.to_datetime(booking_features_df['reservation_datetime'])
booking_features_df['current_cutoff_date'] = pd.to_datetime(booking_features_df['current_cutoff_date'])

# Calculate `time_until_reservation_days_at_cutoff`
booking_features_df['time_until_reservation_days_at_cutoff'] = \
    (booking_features_df['reservation_datetime'] - booking_features_df['current_cutoff_date']).dt.days

# Extract temporal features from reservation_datetime
booking_features_df['hour_of_day'] = booking_features_df['reservation_datetime'].dt.hour
booking_features_df['day_of_week'] = booking_features_df['reservation_datetime'].dt.dayofweek
booking_features_df['month_of_year'] = booking_features_df['reservation_datetime'].dt.month

# Define features (X) and target (y)
numerical_features = [
    'num_guests',
    'num_prev_bookings_customer_12m',
    'num_prev_cancellations_customer_12m',
    'days_since_last_activity_at_cutoff',
    'num_loyalty_redeemed_customer_12m',
    'time_until_reservation_days_at_cutoff',
    'hour_of_day',
    'day_of_week',
    'month_of_year'
]
categorical_features = [
    'booking_channel',
    'special_requests_flag', # Treat as categorical (0/1) for OneHotEncoder
    'customer_segment'
]

X = booking_features_df[numerical_features + categorical_features]
y = booking_features_df['is_no_show']

print(f"Total samples for ML: {len(X)}")
print(f"Target distribution (No-Show): {y.value_counts(normalize=True).loc[1]:.2f}")

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
print(f"No-show rate in training: {y_train.value_counts(normalize=True).loc[1]:.2f}")
print(f"No-show rate in test: {y_test.value_counts(normalize=True).loc[1]:.2f}")


# --- 4. Data Visualization ---

print("\n--- Generating Visualizations ---")
plt.style.use('seaborn-v0_8-darkgrid')

# Stacked bar chart for no-show proportion across booking_channel
plt.figure(figsize=(10, 6))
booking_channel_no_show = bookings_df.groupby('booking_channel')['is_no_show'].value_counts(normalize=True).unstack()
booking_channel_no_show.plot(kind='bar', stacked=True, color=['#66c2a5', '#fc8d62']) # Colorblind-friendly palette
plt.title('Proportion of No-Show by Booking Channel')
plt.xlabel('Booking Channel')
plt.ylabel('Proportion')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Is No-Show', labels=['Show', 'No-Show'])
plt.tight_layout()
plt.show()

# Violin plot for num_guests distribution vs. no-show
plt.figure(figsize=(8, 6))
sns.violinplot(x='is_no_show', y='num_guests', data=bookings_df, palette=['#a6cee3', '#fb9a99'])
plt.title('Distribution of Number of Guests for Show vs. No-Show Bookings')
plt.xlabel('Is No-Show (0: Show, 1: No-Show)')
plt.ylabel('Number of Guests')
plt.tight_layout()
plt.show()


# --- 5. ML Pipeline & Evaluation ---

print("\n--- Building and Evaluating ML Pipeline ---")

# Preprocessing steps for numerical and categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')), # Handles potential NaNs, though SQL/Pandas should have covered it
            ('scaler', StandardScaler())
        ]), numerical_features),
        ('cat', Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ]), categorical_features)
    ])

# Create the ML pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', HistGradientBoostingClassifier(random_state=42))
])

# Train the pipeline
pipeline.fit(X_train, y_train)

# Predict probabilities on the test set
y_pred_proba = pipeline.predict_proba(X_test)[:, 1] # Probability of the positive class (no-show)

# Evaluate ROC AUC Score
roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"\nROC AUC Score on Test Set: {roc_auc:.4f}")

# Convert probabilities to binary predictions for classification report (using default 0.5 threshold)
y_pred = (y_pred_proba > 0.5).astype(int)
print("\nClassification Report on Test Set:")
print(classification_report(y_test, y_pred))

print("\n--- Pipeline Execution Complete ---")