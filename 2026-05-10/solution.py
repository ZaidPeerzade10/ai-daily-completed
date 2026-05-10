import pandas as pd
import numpy as np
import sqlite3
import datetime
from datetime import timedelta, date

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report

# Set random seed for reproducibility
np.random.seed(42)
pd.set_option('display.max_columns', None)

# 1. Generate Synthetic Data (Pandas/Numpy)
print("1. Generating Synthetic Data...")

# --- Flights Data ---
N_FLIGHTS = np.random.randint(10000, 15001)
airlines = ['AA', 'DL', 'UA', 'WN', 'AS', 'B6', 'F9', 'NK'] # B6=JetBlue, F9=Frontier, NK=Spirit
type_a_airlines = ['AA', 'DL', 'UA'] # Assume these have slightly better performance
airports = ['JFK', 'LAX', 'ORD', 'DFW', 'ATL', 'SFO', 'DEN', 'CLT', 'MIA', 'SEA']
weather_conditions = ['Clear', 'Rain', 'Snow', 'Fog', 'Thunderstorm']

flight_ids = np.arange(N_FLIGHTS)
chosen_airlines = np.random.choice(airlines, N_FLIGHTS)

# Generate origin and destination ensuring they are different
origin_airports = np.random.choice(airports, N_FLIGHTS)
destination_airports = np.array([np.random.choice([a for a in airports if a != o]) for o in origin_airports])

# Scheduled departure over the last year
end_date = pd.to_datetime('2023-10-26') # Consistent end date for reproducibility
start_date = end_date - pd.Timedelta(days=365)
scheduled_departure_dates = start_date + (end_date - start_date) * np.random.rand(N_FLIGHTS)

scheduled_duration_minutes = np.random.randint(60, 361, N_FLIGHTS) # 1 to 6 hours

# Base actual delay minutes (can be negative for early)
actual_delay_minutes = np.random.randint(-20, 31, N_FLIGHTS) # -20 to 30 minutes

# Introduce significant delays (10-15% of flights)
significant_delay_indices = np.random.choice(N_FLIGHTS, int(N_FLIGHTS * np.random.uniform(0.10, 0.15)), replace=False)
actual_delay_minutes[significant_delay_indices] += np.random.randint(40, 151, len(significant_delay_indices)) # Add 40-150 mins

# Simulate weather-related delays (placeholder, actual join happens later)
# We'll apply this logic during synthetic data generation to simulate,
# and later the ML model will see this via the actual joined weather.
# For now, let's inject delays based on origin and month
month_of_departure = pd.to_datetime(scheduled_departure_dates).month
is_winter_month = ((month_of_departure >= 12) | (month_of_departure <= 2))

# ORD in winter often has delays
ord_winter_flights = (origin_airports == 'ORD') & is_winter_month
actual_delay_minutes[ord_winter_flights] += np.random.randint(0, 61, ord_winter_flights.sum()) # Add 0-60 mins

# TypeA airlines have slightly better performance (reduce delay)
type_a_flights = np.isin(chosen_airlines, type_a_airlines)
actual_delay_minutes[type_a_flights] -= np.random.randint(0, 11, type_a_flights.sum()) # Reduce by 0-10 mins

# Ensure delays are not excessively negative
actual_delay_minutes[actual_delay_minutes < -30] = -30

flights_df = pd.DataFrame({
    'flight_id': flight_ids,
    'airline': chosen_airlines,
    'origin_airport': origin_airports,
    'destination_airport': destination_airports,
    'scheduled_departure': scheduled_departure_dates,
    'scheduled_duration_minutes': scheduled_duration_minutes,
    'actual_delay_minutes': actual_delay_minutes
})

# Sort flights_df by scheduled_departure
flights_df = flights_df.sort_values(by='scheduled_departure').reset_index(drop=True)

# --- Airport Weather Data ---
N_WEATHER_RECORDS = np.random.randint(2000, 3001)
weather_dates = []
airport_codes_weather = []
for _ in range(N_WEATHER_RECORDS):
    # Randomly pick a date within the last year
    random_date = start_date + (end_date - start_date) * np.random.rand()
    weather_dates.append(random_date.date())
    airport_codes_weather.append(np.random.choice(airports))

# Ensure daily granularity per airport, fill missing days for some airports to simulate
unique_airport_dates = pd.DataFrame({'airport_code': airport_codes_weather, 'weather_date': weather_dates}).drop_duplicates()
unique_airport_dates['weather_condition'] = np.random.choice(weather_conditions, len(unique_airport_dates), p=[0.5, 0.2, 0.1, 0.1, 0.1])

# Introduce more 'Fog'/'Snow' at ORD/JFK/BOS for some winter dates
winter_months = [12, 1, 2]
for airport in ['ORD', 'JFK', 'BOS']: # BOS not in main airports, but for weather
    winter_days_indices = (unique_airport_dates['airport_code'] == airport) & (pd.to_datetime(unique_airport_dates['weather_date']).month.isin(winter_months))
    if winter_days_indices.any():
        unique_airport_dates.loc[winter_days_indices, 'weather_condition'] = np.random.choice(
            ['Clear', 'Snow', 'Fog', 'Rain'], winter_days_indices.sum(), p=[0.3, 0.3, 0.2, 0.2]
        )

airport_weather_df = unique_airport_dates.reset_index(drop=True)

print(f"Generated {len(flights_df)} flight records and {len(airport_weather_df)} weather records.")
print("\nFlights DataFrame Head:")
print(flights_df.head())
print("\nAirport Weather DataFrame Head:")
print(airport_weather_df.head())

# 2. Load into SQLite & SQL Feature Engineering (Time-Windowed Aggregations)
print("\n2. Loading data into SQLite and performing SQL Feature Engineering...")

conn = sqlite3.connect(':memory:')
flights_df.to_sql('flights', conn, index=False, if_exists='replace')
airport_weather_df.to_sql('airport_weather', conn, index=False, if_exists='replace')

sql_query = """
WITH FlightDetails AS (
    SELECT
        f.flight_id,
        f.airline,
        f.origin_airport,
        f.destination_airport,
        f.scheduled_departure,
        f.scheduled_duration_minutes,
        f.actual_delay_minutes,
        DATE(f.scheduled_departure) AS departure_date,
        STRFTIME('%w', f.scheduled_departure) AS day_of_week, -- 0=Sun, 6=Sat
        STRFTIME('%H', f.scheduled_departure) AS hour_of_day,
        STRFTIME('%m', f.scheduled_departure) AS month_of_year,
        julianday(f.scheduled_departure) AS sd_julian
    FROM flights f
),
WeatherJoined AS (
    SELECT
        fd.*,
        COALESCE(aw.weather_condition, 'Unknown') AS departure_weather_condition
    FROM FlightDetails fd
    LEFT JOIN airport_weather aw
        ON fd.origin_airport = aw.airport_code
        AND fd.departure_date = aw.weather_date
),
-- Historical aggregates for origin airport within 30 days preceding each flight
OriginAggregates AS (
    SELECT
        f_main.flight_id,
        COALESCE(AVG(f_hist.actual_delay_minutes), 0.0) AS avg_origin_delay_prev_30d,
        COALESCE(COUNT(f_hist.flight_id), 0) AS num_flights_origin_prev_30d
    FROM FlightDetails f_main
    LEFT JOIN flights f_hist
        ON f_main.origin_airport = f_hist.origin_airport
        AND f_hist.sd_julian < f_main.sd_julian -- Strictly before current flight's time
        AND f_hist.sd_julian >= f_main.sd_julian - 30 -- Within the last 30 days
    GROUP BY f_main.flight_id
),
-- Historical aggregates for airline within 30 days preceding each flight
AirlineAggregates AS (
    SELECT
        f_main.flight_id,
        COALESCE(AVG(f_hist.actual_delay_minutes), 0.0) AS avg_airline_delay_prev_30d
    FROM FlightDetails f_main
    LEFT JOIN flights f_hist
        ON f_main.airline = f_hist.airline
        AND f_hist.sd_julian < f_main.sd_julian -- Strictly before current flight's time
        AND f_hist.sd_julian >= f_main.sd_julian - 30 -- Within the last 30 days
    GROUP BY f_main.flight_id
)
SELECT
    wj.flight_id,
    wj.airline,
    wj.origin_airport,
    wj.destination_airport,
    wj.scheduled_departure,
    wj.scheduled_duration_minutes,
    wj.actual_delay_minutes,
    CAST(wj.day_of_week AS INTEGER) AS day_of_week,
    CAST(wj.hour_of_day AS INTEGER) AS hour_of_day,
    CAST(wj.month_of_year AS INTEGER) AS month_of_year,
    wj.departure_weather_condition,
    oa.avg_origin_delay_prev_30d,
    oa.num_flights_origin_prev_30d,
    aa.avg_airline_delay_prev_30d
FROM WeatherJoined wj
LEFT JOIN OriginAggregates oa ON wj.flight_id = oa.flight_id
LEFT JOIN AirlineAggregates aa ON wj.flight_id = aa.flight_id
ORDER BY wj.scheduled_departure;
"""

flight_features_df = pd.read_sql_query(sql_query, conn)
conn.close()

print("\nSQL Feature Engineering Results Head:")
print(flight_features_df.head())
print(f"DataFrame after SQL query has {len(flight_features_df)} rows.")

# 3. Pandas Feature Engineering & Multi-class Target Creation
print("\n3. Performing Pandas Feature Engineering and creating target variable...")

# Convert scheduled_departure to datetime objects
flight_features_df['scheduled_departure'] = pd.to_datetime(flight_features_df['scheduled_departure'])

# Fill NaN values in numerical aggregated features with 0 (COALESCE in SQL should handle most)
# Re-filling here as a safety measure for potential edge cases
numerical_agg_cols = [
    'avg_airline_delay_prev_30d',
    'num_flights_origin_prev_30d',
    'avg_origin_delay_prev_30d'
]
for col in numerical_agg_cols:
    if col in flight_features_df.columns:
        flight_features_df[col] = flight_features_df[col].fillna(0.0)

# Create the Multi-class Target `delay_category`
def categorize_delay(minutes):
    if minutes <= 15:
        return 'On Time'
    elif 15 < minutes <= 60:
        return 'Slight Delay'
    else: # minutes > 60
        return 'Significant Delay'

flight_features_df['delay_category'] = flight_features_df['actual_delay_minutes'].apply(categorize_delay)

# Define features X and target y
numerical_features = [
    'scheduled_duration_minutes',
    'avg_airline_delay_prev_30d',
    'num_flights_origin_prev_30d',
    'avg_origin_delay_prev_30d',
    'day_of_week',
    'hour_of_day',
    'month_of_year'
]
categorical_features = [
    'airline',
    'origin_airport',
    'destination_airport',
    'departure_weather_condition'
]

# Ensure all feature columns exist
all_features = numerical_features + categorical_features
X = flight_features_df[all_features]
y = flight_features_df['delay_category']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print("\nDelay Category Distribution in Training Set:")
print(y_train.value_counts(normalize=True))
print("\nDelay Category Distribution in Test Set:")
print(y_test.value_counts(normalize=True))
print(f"Training set size: {len(X_train)}, Test set size: {len(X_test)}")


# 4. Data Visualization
print("\n4. Generating Data Visualizations...")

plt.figure(figsize=(15, 6))

# Plot 1: Violin plot of scheduled_duration_minutes by delay_category
plt.subplot(1, 2, 1)
sns.violinplot(x='delay_category', y='scheduled_duration_minutes', data=flight_features_df,
               order=['On Time', 'Slight Delay', 'Significant Delay'], palette='viridis')
plt.title('Distribution of Scheduled Duration by Delay Category')
plt.xlabel('Delay Category')
plt.ylabel('Scheduled Duration (minutes)')

# Plot 2: Stacked bar chart of delay_category proportions across departure_weather_condition
plt.subplot(1, 2, 2)
# Calculate proportions
weather_delay_proportions = flight_features_df.groupby('departure_weather_condition')['delay_category'].value_counts(normalize=True).unstack().fillna(0)
# Ensure all categories are present and in desired order
desired_order = ['On Time', 'Slight Delay', 'Significant Delay']
for col in desired_order:
    if col not in weather_delay_proportions.columns:
        weather_delay_proportions[col] = 0
weather_delay_proportions = weather_delay_proportions[desired_order]

weather_delay_proportions.plot(kind='bar', stacked=True, ax=plt.gca(), cmap='viridis')
plt.title('Proportion of Delay Categories by Departure Weather Condition')
plt.xlabel('Departure Weather Condition')
plt.ylabel('Proportion')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Delay Category', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

print("Visualizations displayed (violin plot and stacked bar chart).")


# 5. ML Pipeline & Evaluation (Multi-class Classification)
print("\n5. Building and Evaluating ML Pipeline...")

# Create preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ]), numerical_features),
        ('cat', Pipeline([
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ]), categorical_features)
    ],
    remainder='passthrough' # Keep other columns (like flight_id if present)
)

# Create the ML pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', HistGradientBoostingClassifier(random_state=42))
])

# Train the pipeline
print("Training the model...")
pipeline.fit(X_train, y_train)
print("Model training complete.")

# Predict on the test set
print("Making predictions on the test set...")
y_pred = pipeline.predict(X_test)
print("Predictions complete.")

# Calculate and print the classification report
print("\nClassification Report for the Test Set:")
print(classification_report(y_test, y_pred, target_names=['On Time', 'Slight Delay', 'Significant Delay']))

print("\n--- Pipeline Execution Complete ---")