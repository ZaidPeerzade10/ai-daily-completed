import pandas as pd
import numpy as np
import datetime
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')

# --- Step 1: Synthetic Data Generation and Cutoff Definition ---

print("Step 1: Synthetic Data Generation and Cutoff Definition")

# Define global parameters
N_RESTAURANTS = 50
START_DATE = datetime.date(2022, 1, 1)
END_DATE = datetime.date(2023, 12, 31) # Data up to this date
GLOBAL_PREDICTION_CUTOFF_DATE = datetime.date(2023, 11, 30) # Features up to this date
PREDICTION_WINDOW_DAYS = 7

# Booking demand category thresholds (as proportion of capacity)
LOW_DEMAND_THRESHOLD = 0.3
MEDIUM_DEMAND_THRESHOLD = 0.7

# 1.1 Generate restaurants_df
np.random.seed(42)
restaurants_data = {
    'restaurant_id': [f'R{i:03d}' for i in range(N_RESTAURANTS)],
    'cuisine': np.random.choice(['Italian', 'Mexican', 'Asian', 'American', 'French'], N_RESTAURANTS),
    'location': np.random.choice(['Downtown', 'Uptown', 'Suburban', 'Waterfront'], N_RESTAURANTS),
    'capacity': np.random.randint(20, 150, N_RESTAURANTS),
    'rating': np.round(np.random.uniform(3.0, 5.0, N_RESTAURANTS), 1)
}
restaurants_df = pd.DataFrame(restaurants_data)
print(f"Generated {len(restaurants_df)} restaurants.")
print("Restaurants DataFrame head:")
print(restaurants_df.head())

# 1.2 Generate daily_bookings_df
date_range = pd.date_range(start=START_DATE, end=END_DATE, freq='D')
all_bookings = []

for _, restaurant in restaurants_df.iterrows():
    restaurant_id = restaurant['restaurant_id']
    capacity = restaurant['capacity']
    rating = restaurant['rating']
    
    # Simulate demand with some seasonality and randomness
    for date in date_range:
        day_of_week = date.weekday() # 0=Monday, 6=Sunday
        
        # Base demand influenced by rating and capacity
        base_demand = capacity * (rating / 5.0) * np.random.uniform(0.5, 1.0)
        
        # Weekend boost
        if day_of_week >= 5: # Saturday or Sunday
            base_demand *= np.random.uniform(1.1, 1.5)
        
        # Monthly seasonality (peak in middle of month, dip at ends)
        day_of_month = date.day
        monthly_factor = 1 + np.sin(np.pi * day_of_month / 30) * 0.2 # Roughly cyclical
        base_demand *= monthly_factor
        
        # Random fluctuation
        total_guests_booked = int(np.clip(base_demand * np.random.uniform(0.8, 1.2), 0, capacity * 1.2))
        
        all_bookings.append({
            'restaurant_id': restaurant_id,
            'date': date,
            'total_guests_booked': total_guests_booked
        })

daily_bookings_df = pd.DataFrame(all_bookings)
daily_bookings_df['date'] = pd.to_datetime(daily_bookings_df['date'])
print(f"\nGenerated {len(daily_bookings_df)} daily bookings records.")
print("Daily Bookings DataFrame head:")
print(daily_bookings_df.head())

print(f"\nGlobal Prediction Cutoff Date: {GLOBAL_PREDICTION_CUTOFF_DATE}")

# --- Step 2: Base Feature and Target Engineering (Pandas/Numpy) ---

print("\nStep 2: Base Feature and Target Engineering (Pandas/Numpy)")

# 2.1 Future Prediction Dates
prediction_dates = pd.date_range(start=pd.to_datetime(GLOBAL_PREDICTION_CUTOFF_DATE) + pd.Timedelta(days=1),
                                 periods=PREDICTION_WINDOW_DAYS, freq='D')
future_prediction_df = pd.DataFrame({
    'restaurant_id': np.tile(restaurants_df['restaurant_id'].values, PREDICTION_WINDOW_DAYS),
    'date': np.repeat(prediction_dates, N_RESTAURANTS)
})
print(f"Generated {len(future_prediction_df)} future prediction date records.")
print("Future Prediction Dates DataFrame head:")
print(future_prediction_df.head())

# Combine all dates for feature generation
all_dates_df = pd.concat([
    daily_bookings_df[['restaurant_id', 'date']],
    future_prediction_df[['restaurant_id', 'date']]
]).drop_duplicates().reset_index(drop=True)

# 2.2 Date-based Features
all_dates_df['day_of_week'] = all_dates_df['date'].dt.dayofweek # 0=Monday, 6=Sunday
all_dates_df['day_name'] = all_dates_df['date'].dt.day_name()
all_dates_df['day_of_year'] = all_dates_df['date'].dt.dayofyear
all_dates_df['month'] = all_dates_df['date'].dt.month
all_dates_df['year'] = all_dates_df['date'].dt.year
all_dates_df['is_weekend'] = all_dates_df['day_of_week'].isin([5, 6]).astype(int)

# 2.3 Initial Merging
# Merge static restaurant attributes
full_df = pd.merge(all_dates_df, restaurants_df, on='restaurant_id', how='left')
# Merge daily booking data (will be NaN for future dates)
full_df = pd.merge(full_df, daily_bookings_df, on=['restaurant_id', 'date'], how='left')

print("\nFull DataFrame after initial merge and date features (head):")
print(full_df.head())
print("Full DataFrame after initial merge and date features (tail, showing future dates):")
print(full_df.tail())

# 2.4 Historical Target Creation
def categorize_demand(row):
    if pd.isna(row['total_guests_booked']):
        return np.nan # For future dates or missing data
    
    guests_ratio = row['total_guests_booked'] / row['capacity']
    if guests_ratio < LOW_DEMAND_THRESHOLD:
        return 'Low'
    elif guests_ratio < MEDIUM_DEMAND_THRESHOLD:
        return 'Medium'
    else:
        return 'High'

full_df['booking_demand_category'] = full_df.apply(categorize_demand, axis=1)

historical_full_df = full_df[full_df['date'] <= pd.to_datetime(GLOBAL_PREDICTION_CUTOFF_DATE)].copy()

print("\nHistorical Full DataFrame with target variable (head):")
print(historical_full_df[['restaurant_id', 'date', 'total_guests_booked', 'capacity', 'booking_demand_category']].head())
print("\nTarget category distribution (Historical):")
print(historical_full_df['booking_demand_category'].value_counts(dropna=False))


# --- Step 3: Advanced Feature Engineering with SQLite Aggregations ---

print("\nStep 3: Advanced Feature Engineering with SQLite Aggregations")

# Filter daily_bookings_df to strictly up to GLOBAL_PREDICTION_CUTOFF_DATE for feature generation
historical_bookings_for_features_df = daily_bookings_df[
    daily_bookings_df['date'] <= pd.to_datetime(GLOBAL_PREDICTION_CUTOFF_DATE)
].copy()

# Add capacity to this filtered historical bookings df for ratio calculations in SQL
historical_bookings_for_features_df = pd.merge(
    historical_bookings_for_features_df, 
    restaurants_df[['restaurant_id', 'capacity']], 
    on='restaurant_id', 
    how='left'
)

# Also add 'is_weekend' for conditional aggregations
historical_bookings_for_features_df['is_weekend'] = historical_bookings_for_features_df['date'].dt.dayofweek.isin([5, 6]).astype(int)

# Create demand category for SQL features
historical_bookings_for_features_df['is_high_demand'] = historical_bookings_for_features_df.apply(
    lambda row: 1 if (row['total_guests_booked'] / row['capacity']) >= MEDIUM_DEMAND_THRESHOLD else 0, axis=1
)


# 3.1 Database Setup
conn = sqlite3.connect(':memory:')
restaurants_df.to_sql('restaurants', conn, index=False, if_exists='replace')
historical_bookings_for_features_df.to_sql('daily_bookings', conn, index=False, if_exists='replace')
print("\nLoaded historical data into in-memory SQLite database.")

# Helper function for date arithmetic in SQL
cutoff_julianday = pd.to_datetime(GLOBAL_PREDICTION_CUTOFF_DATE).toordinal() + 1721424.5 # julianday for cutoff

# 3.2 Restaurant-Specific Aggregations
print("\nCalculating restaurant-specific aggregated features...")

restaurant_features_query = f"""
WITH BookingAggregates AS (
    SELECT
        db.restaurant_id,
        db.total_guests_booked,
        db.capacity,
        db.is_weekend,
        db.is_high_demand,
        JULIANDAY(db.date) AS julian_date
    FROM
        daily_bookings db
    WHERE
        db.date <= '{GLOBAL_PREDICTION_CUTOFF_DATE}'
)
SELECT
    r.restaurant_id,
    -- Average guests last 7 days
    AVG(CASE WHEN ba.julian_date >= {cutoff_julianday} - 7 AND ba.julian_date <= {cutoff_julianday} THEN ba.total_guests_booked END) AS avg_guests_last_7_days,
    -- Average guests last 14 days
    AVG(CASE WHEN ba.julian_date >= {cutoff_julianday} - 14 AND ba.julian_date <= {cutoff_julianday} THEN ba.total_guests_booked END) AS avg_guests_last_14_days,
    -- Average guests last 30 days
    AVG(CASE WHEN ba.julian_date >= {cutoff_julianday} - 30 AND ba.julian_date <= {cutoff_julianday} THEN ba.total_guests_booked END) AS avg_guests_last_30_days,
    
    -- Average weekend guests last 30 days
    AVG(CASE WHEN ba.is_weekend = 1 AND ba.julian_date >= {cutoff_julianday} - 30 AND ba.julian_date <= {cutoff_julianday} THEN ba.total_guests_booked END) AS avg_weekend_guests_last_30_days,
    -- Average weekday guests last 30 days
    AVG(CASE WHEN ba.is_weekend = 0 AND ba.julian_date >= {cutoff_julianday} - 30 AND ba.julian_date <= {cutoff_julianday} THEN ba.total_guests_booked END) AS avg_weekday_guests_last_30_days,
    
    -- Proportion of high demand days last 30 days
    CAST(SUM(CASE WHEN ba.is_high_demand = 1 AND ba.julian_date >= {cutoff_julianday} - 30 AND ba.julian_date <= {cutoff_julianday} THEN 1 ELSE 0 END) AS REAL) / 
    NULLIF(COUNT(CASE WHEN ba.julian_date >= {cutoff_julianday} - 30 AND ba.julian_date <= {cutoff_julianday} THEN 1 END), 0) AS prop_high_demand_last_30_days,
    
    -- Booking trend: Avg guests last 7 days vs prior 7 days
    (AVG(CASE WHEN ba.julian_date >= {cutoff_julianday} - 7 AND ba.julian_date <= {cutoff_julianday} THEN ba.total_guests_booked END) -
     AVG(CASE WHEN ba.julian_date >= {cutoff_julianday} - 14 AND ba.julian_date < {cutoff_julianday} - 7 THEN ba.total_guests_booked END)) AS booking_trend_7_14_days
FROM
    restaurants r
LEFT JOIN
    BookingAggregates ba ON r.restaurant_id = ba.restaurant_id
GROUP BY
    r.restaurant_id;
"""
restaurant_agg_features_df = pd.read_sql_query(restaurant_features_query, conn)
print("Restaurant-specific aggregated features head:")
print(restaurant_agg_features_df.head())

# 3.3 Market-Level Aggregations
print("\nCalculating market-level aggregated features...")
market_features_query = f"""
WITH BookingAggregates AS (
    SELECT
        db.restaurant_id,
        r.cuisine,
        r.location,
        db.total_guests_booked,
        JULIANDAY(db.date) AS julian_date
    FROM
        daily_bookings db
    JOIN
        restaurants r ON db.restaurant_id = r.restaurant_id
    WHERE
        db.date <= '{GLOBAL_PREDICTION_CUTOFF_DATE}'
)
SELECT
    r.restaurant_id,
    -- Market average guests by cuisine last 7 days
    (SELECT AVG(ba_sub.total_guests_booked) FROM BookingAggregates ba_sub 
     WHERE ba_sub.cuisine = r.cuisine AND ba_sub.julian_date >= {cutoff_julianday} - 7 AND ba_sub.julian_date <= {cutoff_julianday}) AS market_avg_cuisine_last_7_days,
    -- Market average guests by location last 7 days
    (SELECT AVG(ba_sub.total_guests_booked) FROM BookingAggregates ba_sub 
     WHERE ba_sub.location = r.location AND ba_sub.julian_date >= {cutoff_julianday} - 7 AND ba_sub.julian_date <= {cutoff_julianday}) AS market_avg_location_last_7_days,
    -- Overall market average guests last 7 days
    (SELECT AVG(ba_sub.total_guests_booked) FROM BookingAggregates ba_sub 
     WHERE ba_sub.julian_date >= {cutoff_julianday} - 7 AND ba_sub.julian_date <= {cutoff_julianday}) AS market_overall_avg_last_7_days
FROM
    restaurants r
GROUP BY
    r.restaurant_id;
"""
market_agg_features_df = pd.read_sql_query(market_features_query, conn)
print("Market-level aggregated features head:")
print(market_agg_features_df.head())

conn.close()

# Merge all aggregated features
all_agg_features_df = pd.merge(restaurant_agg_features_df, market_agg_features_df, on='restaurant_id', how='left')
print("\nAll aggregated features DataFrame head:")
print(all_agg_features_df.head())

# --- Step 4: Dataset Assembly and Exploratory Data Analysis (EDA) ---

print("\nStep 4: Dataset Assembly and Exploratory Data Analysis (EDA)")

# 4.1 Feature Consolidation
# Start with the full_df (which contains all dates and initial features)
# Filter it for the columns we need, dropping `total_guests_booked` as it's the raw data,
# and `booking_demand_category` will be our target or derived from actuals.
master_df = full_df.drop(columns=['total_guests_booked', 'booking_demand_category']).copy()

# Merge with all aggregated features (these are constant for all dates post-cutoff)
master_df = pd.merge(master_df, all_agg_features_df, on='restaurant_id', how='left')

# Restore actual total_guests_booked and booking_demand_category for the *entire* range
# This includes the "future actuals" for evaluation.
master_df = pd.merge(master_df, daily_bookings_df[['restaurant_id', 'date', 'total_guests_booked']], on=['restaurant_id', 'date'], how='left')
master_df['booking_demand_category_actual'] = master_df.apply(categorize_demand, axis=1)

print("\nMaster DataFrame with all features and actual target (head):")
print(master_df.head())
print("\nMaster DataFrame with all features and actual target (tail, showing future dates):")
print(master_df.tail())

# Impute NaN aggregated features with 0 (or mean/median if appropriate)
agg_cols = all_agg_features_df.columns.drop('restaurant_id')
master_df[agg_cols] = master_df[agg_cols].fillna(0)

print(f"\nMaster DataFrame shape: {master_df.shape}")
print("Missing values in Master DataFrame after aggregation merge and initial fill:")
print(master_df.isnull().sum()[master_df.isnull().sum() > 0]) # `total_guests_booked` for future dates will be NaN as expected

# 4.2 Data Visualization (Matplotlib/Seaborn)
print("\nPerforming EDA visualizations...")

plt.figure(figsize=(18, 12))
plt.suptitle('Exploratory Data Analysis', fontsize=16)

# Distribution of Capacity
plt.subplot(2, 3, 1)
sns.histplot(master_df['capacity'], kde=True)
plt.title('Distribution of Restaurant Capacity')

# Distribution of Rating
plt.subplot(2, 3, 2)
sns.histplot(master_df['rating'], kde=True)
plt.title('Distribution of Restaurant Rating')

# Frequency of Target Variable
plt.subplot(2, 3, 3)
sns.countplot(x='booking_demand_category_actual', data=master_df[master_df['date'] <= pd.to_datetime(GLOBAL_PREDICTION_CUTOFF_DATE)], order=['Low', 'Medium', 'High'])
plt.title('Historical Booking Demand Category Distribution')

# Time-series plot for an example restaurant
plt.subplot(2, 1, 2)
sample_restaurant_id = restaurants_df['restaurant_id'].sample(1).iloc[0]
sample_df = master_df[master_df['restaurant_id'] == sample_restaurant_id].sort_values('date')
sns.lineplot(x='date', y='total_guests_booked', data=sample_df)
plt.axvline(pd.to_datetime(GLOBAL_PREDICTION_CUTOFF_DATE), color='r', linestyle='--', label='Prediction Cutoff')
plt.title(f'Daily Guests Booked for Restaurant {sample_restaurant_id} Over Time')
plt.xlabel('Date')
plt.ylabel('Total Guests Booked')
plt.legend()
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

plt.figure(figsize=(16, 6))

# Boxplot of capacity by cuisine
plt.subplot(1, 2, 1)
sns.boxplot(x='cuisine', y='capacity', data=master_df.drop_duplicates(subset=['restaurant_id']))
plt.title('Capacity by Cuisine Type')

# Bar plot of booking demand by day of week
plt.subplot(1, 2, 2)
sns.countplot(x='day_name', hue='booking_demand_category_actual', 
              data=master_df[master_df['date'] <= pd.to_datetime(GLOBAL_PREDICTION_CUTOFF_DATE)],
              order=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
              hue_order=['Low', 'Medium', 'High'])
plt.title('Booking Demand Category by Day of Week (Historical)')
plt.tight_layout()
plt.show()


# --- Step 5: Data Preprocessing and Model Training (Scikit-learn) ---

print("\nStep 5: Data Preprocessing and Model Training (Scikit-learn)")

# Separate data into Training Set and Prediction Set
train_df = master_df[master_df['date'] <= pd.to_datetime(GLOBAL_PREDICTION_CUTOFF_DATE)].copy()
predict_df = master_df[master_df['date'] > pd.to_datetime(GLOBAL_PREDICTION_CUTOFF_DATE)].copy()

# Drop rows with NaN target values in training data
train_df.dropna(subset=['booking_demand_category_actual'], inplace=True)

# Define features (X) and target (y)
X_train = train_df.drop(columns=['total_guests_booked', 'booking_demand_category_actual', 'date', 'restaurant_id'])
y_train = train_df['booking_demand_category_actual']

X_predict = predict_df.drop(columns=['total_guests_booked', 'booking_demand_category_actual', 'date', 'restaurant_id'])
y_actual_future = predict_df['booking_demand_category_actual'] # For evaluation later

print(f"\nTraining set shape: {X_train.shape}")
print(f"Prediction set shape: {X_predict.shape}")
print(f"Number of NaN targets in training set (should be 0): {y_train.isnull().sum()}")

# Identify categorical and numerical features
categorical_features = ['cuisine', 'location', 'day_name']
numerical_features = X_train.select_dtypes(include=np.number).columns.tolist()
# Remove features that are representations of other features or IDs
numerical_features = [f for f in numerical_features if f not in ['day_of_week', 'day_of_year', 'month', 'year']]


# Preprocessing Pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='passthrough'
)

# Model Selection
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')

# Create the full pipeline
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('classifier', model)])

print("\nPreprocessing and Model Pipeline defined.")
print("Numerical features:", numerical_features)
print("Categorical features:", categorical_features)

# Model Training with Cross-Validation
print("\nTraining model with 3-fold cross-validation...")
cv_scores = cross_val_score(pipeline, X_train, y_train, cv=3, scoring='f1_weighted', n_jobs=-1)

print(f"Cross-validation F1-weighted scores: {cv_scores}")
print(f"Mean CV F1-weighted score: {np.mean(cv_scores):.4f}")
print(f"Standard deviation of CV F1-weighted score: {np.std(cv_scores):.4f}")

# Train the model on the full training data
pipeline.fit(X_train, y_train)
print("\nModel training complete on full training data.")

# --- Step 6: Prediction and Evaluation (Scikit-learn/Matplotlib/Seaborn) ---

print("\nStep 6: Prediction and Evaluation (Scikit-learn/Matplotlib/Seaborn)")

# 6.1 Generate Predictions
y_pred_future = pipeline.predict(X_predict)

print(f"\nGenerated {len(y_pred_future)} predictions for the next {PREDICTION_WINDOW_DAYS} days.")

# Add predictions to the prediction DataFrame
predict_df['predicted_demand_category'] = y_pred_future

# 6.2 Model Evaluation
print("\n--- Model Evaluation for Future Prediction Window ---")
print("Classification Report:")
print(classification_report(y_actual_future, y_pred_future))

accuracy = accuracy_score(y_actual_future, y_pred_future)
print(f"Overall Accuracy on future predictions: {accuracy:.4f}")

# Confusion Matrix
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_actual_future, y_pred_future, labels=['Low', 'Medium', 'High'])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Low', 'Medium', 'High'], yticklabels=['Low', 'Medium', 'High'])
plt.title('Confusion Matrix for Future Predictions')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# 6.3 Visualize Predictions
print("\nVisualizing predictions for a sample restaurant in the prediction window...")

# Select a random restaurant for visualization
sample_pred_restaurant_id = restaurants_df['restaurant_id'].sample(1).iloc[0]
sample_pred_df = predict_df[predict_df['restaurant_id'] == sample_pred_restaurant_id].sort_values('date')

plt.figure(figsize=(12, 6))
sns.lineplot(x='date', y='total_guests_booked', data=sample_pred_df, label='Actual Guests Booked', color='gray', linestyle='--')
sns.scatterplot(x='date', y='total_guests_booked', hue='predicted_demand_category', 
                data=sample_pred_df, s=100, zorder=5, 
                palette={'Low': 'blue', 'Medium': 'orange', 'High': 'red'},
                marker='o', legend='full',
                style='predicted_demand_category')

# Plot actual vs predicted category
for _, row in sample_pred_df.iterrows():
    plt.text(row['date'], row['total_guests_booked'] + 5, 
             f"Actual: {row['booking_demand_category_actual']}\nPred: {row['predicted_demand_category']}",
             horizontalalignment='center', fontsize=8, color='black', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

plt.title(f'Actual vs. Predicted Booking Demand for Restaurant {sample_pred_restaurant_id} (Next {PREDICTION_WINDOW_DAYS} Days)')
plt.xlabel('Date')
plt.ylabel('Total Guests Booked')
plt.xticks(rotation=45)
plt.grid(True)
plt.legend(title='Predicted Category')
plt.tight_layout()
plt.show()

print("\n--- Pipeline Execution Complete ---")