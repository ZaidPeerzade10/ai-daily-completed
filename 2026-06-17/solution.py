import pandas as pd
import numpy as np
import io
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# --- 1. Data Ingestion, Initial Preprocessing, and Target Variable Engineering ---

# Simulate Data
np.random.seed(42)

# Road Segments Data
num_segments = 20
road_segments_data = {
    'segment_ID': [f'SEG_{i:03d}' for i in range(num_segments)],
    'length_km': np.random.uniform(0.5, 10, num_segments).round(1),
    'lanes': np.random.choice([2, 3, 4, 6], num_segments),
    'speed_limit_kmh': np.random.choice([50, 60, 80, 100, 120], num_segments),
    'road_type': np.random.choice(['Urban', 'Suburban', 'Highway'], num_segments)
}
road_segments_df = pd.DataFrame(road_segments_data)

# Traffic Sensors Data (1-3 sensors per segment)
sensors_data = []
sensor_id_counter = 1
for seg_id in road_segments_df['segment_ID']:
    num_sensors = np.random.randint(1, 4)
    for _ in range(num_sensors):
        sensors_data.append({'sensor_ID': f'SEN_{sensor_id_counter:04d}', 'segment_ID': seg_id})
        sensor_id_counter += 1
traffic_sensors_df = pd.DataFrame(sensors_data)

# Traffic Readings Data (using a short duration for demo performance)
start_date = datetime(2023, 1, 1, 0, 0, 0)
end_date = datetime(2023, 1, 5, 0, 0, 0) # Only 4 days of data for faster execution
interval_minutes = 30 # Readings every 30 minutes

all_readings = []
current_date = start_date
while current_date < end_date:
    for idx, sensor_row in traffic_sensors_df.iterrows():
        sensor_id = sensor_row['sensor_ID']
        segment_id = sensor_row['segment_ID']
        
        # Get static attributes for congestion calculation
        segment_attrs = road_segments_df[road_segments_df['segment_ID'] == segment_id].iloc[0]
        speed_limit = segment_attrs['speed_limit_kmh']
        lanes = segment_attrs['lanes']

        # Simulate time-of-day/day-of-week patterns
        hour = current_date.hour
        day_of_week = current_date.weekday()

        base_vehicle_count = np.random.randint(50, 200)
        base_observed_speed = np.random.randint(30, 100)

        # Introduce patterns:
        if 7 <= hour <= 9 or 16 <= hour <= 18: # Peak hours
            base_vehicle_count = base_vehicle_count * 1.5 + np.random.randint(0, 50)
            base_observed_speed = base_observed_speed * 0.7 - np.random.randint(0, 10)
        elif 0 <= hour <= 5: # Night hours
            base_vehicle_count = base_vehicle_count * 0.5 - np.random.randint(0, 20)
            base_observed_speed = base_observed_speed * 1.2 + np.random.randint(0, 20)
        
        if day_of_week >= 5: # Weekend
            base_vehicle_count *= 0.8
            base_observed_speed *= 1.1

        vehicle_count = max(0, int(base_vehicle_count + np.random.normal(0, 20)))
        # Observed speed cannot exceed speed limit by much, or be zero
        observed_speed = max(1, min(speed_limit + 10, int(base_observed_speed + np.random.normal(0, 10))))

        # Calculate actual_congestion_index: Higher is more congested
        # Factor in vehicle density and deviation from speed limit
        congestion_index = (vehicle_count / lanes) * (speed_limit / max(1, observed_speed))
        congestion_index = congestion_index * np.random.uniform(0.8, 1.2) # Add some noise
        congestion_index = max(0, congestion_index)

        all_readings.append({
            'sensor_ID': sensor_id,
            'timestamp': current_date,
            'vehicle_count': vehicle_count,
            'observed_speed_kmh': observed_speed,
            'actual_congestion_index': congestion_index
        })
    current_date += timedelta(minutes=interval_minutes)

traffic_readings_df = pd.DataFrame(all_readings)

# Convert timestamp to datetime objects
traffic_readings_df['timestamp'] = pd.to_datetime(traffic_readings_df['timestamp'])

# Target Variable Creation
congestion_index_threshold_low = traffic_readings_df['actual_congestion_index'].quantile(0.33)
congestion_index_threshold_high = traffic_readings_df['actual_congestion_index'].quantile(0.67)

def classify_congestion(index):
    if index <= congestion_index_threshold_low:
        return 'Low'
    elif index <= congestion_index_threshold_high:
        return 'Medium'
    else:
        return 'High'

traffic_readings_df['traffic_congestion_level'] = traffic_readings_df['actual_congestion_index'].apply(classify_congestion)

print("--- Initial Dataframes Head & Target Distribution ---")
print("Road Segments (first 3 rows):\n", road_segments_df.head(3))
print("\nTraffic Sensors (first 3 rows):\n", traffic_sensors_df.head(3))
print("\nTraffic Readings (first 3 rows):\n", traffic_readings_df.head(3))
print("\nTarget Variable Distribution:\n", traffic_readings_df['traffic_congestion_level'].value_counts())
print(f"Congestion Index Thresholds: Low <= {congestion_index_threshold_low:.2f}, Medium <= {congestion_index_threshold_high:.2f}")


# --- 2. Feature Engineering - Static Road Attributes and Real-time Readings ---

# Merge static road attributes with sensor info
sensors_with_segments_df = pd.merge(traffic_sensors_df, road_segments_df, on='segment_ID', how='left')

# Join all information into a master DataFrame
master_df = pd.merge(traffic_readings_df, sensors_with_segments_df, on='sensor_ID', how='left')
master_df = master_df.sort_values(by=['segment_ID', 'timestamp']).reset_index(drop=True)

print("\n--- Master DataFrame Head (after joining static & real-time features) ---")
print(master_df.head())


# --- 3. Feature Engineering - Time-Series Historical Aggregates (Data Leakage Prevention Focus) ---

# Create Temporal Features from the current timestamp
master_df['hour_of_day'] = master_df['timestamp'].dt.hour
master_df['day_of_week'] = master_df['timestamp'].dt.dayofweek
master_df['day_of_month'] = master_df['timestamp'].dt.day
master_df['month'] = master_df['timestamp'].dt.month
master_df['is_weekend'] = master_df['day_of_week'].isin([5, 6]).astype(int)

# Function to generate historical features using explicit time-based filtering in Pandas.
# This ensures strict data leakage prevention.
def calculate_historical_features(row, df_history, windows):
    segment_id = row['segment_ID']
    current_timestamp = row['timestamp']
    
    # Filter historical data for the specific segment and STRICTLY before the current timestamp
    segment_history = df_history[
        (df_history['segment_ID'] == segment_id) & 
        (df_history['timestamp'] < current_timestamp)
    ]
    
    features = {}
    for window_hours in windows:
        window_start_time = current_timestamp - timedelta(hours=window_hours)
        
        # Filter data within the defined historical window
        window_data = segment_history[segment_history['timestamp'] >= window_start_time]
        
        if not window_data.empty:
            agg_data_vc = window_data['vehicle_count'].agg(['mean', 'median', 'std', 'min', 'max'])
            agg_data_os = window_data['observed_speed_kmh'].agg(['mean', 'median', 'std', 'min', 'max'])
            
            features.update({
                f'hist_avg_vehicle_count_{window_hours}h': agg_data_vc['mean'],
                f'hist_med_vehicle_count_{window_hours}h': agg_data_vc['median'],
                f'hist_std_vehicle_count_{window_hours}h': agg_data_vc['std'],
                f'hist_min_vehicle_count_{window_hours}h': agg_data_vc['min'],
                f'hist_max_vehicle_count_{window_hours}h': agg_data_vc['max'],
                f'hist_avg_observed_speed_{window_hours}h': agg_data_os['mean'],
                f'hist_med_observed_speed_{window_hours}h': agg_data_os['median'],
                f'hist_std_observed_speed_{window_hours}h': agg_data_os['std'],
                f'hist_min_observed_speed_{window_hours}h': agg_data_os['min'],
                f'hist_max_observed_speed_{window_hours}h': agg_data_os['max'],
            })
        else:
            # Handle missing aggregates by filling with NaN for later imputation
            # This ensures consistency even if no historical data exists for a window
            for var_prefix in ['hist_avg_vehicle_count', 'hist_med_vehicle_count', 'hist_std_vehicle_count',
                               'hist_min_vehicle_count', 'hist_max_vehicle_count',
                               'hist_avg_observed_speed', 'hist_med_observed_speed', 'hist_std_observed_speed',
                               'hist_min_observed_speed', 'hist_max_observed_speed']:
                features[f'{var_prefix}_{window_hours}h'] = np.nan

    return pd.Series(features)

# Define aggregation windows in hours
windows_hours = [1, 4, 24, 24*7] # 1 hour, 4 hours, 24 hours, 7 days

print(f"\n--- Generating Historical Features (DF size: {len(master_df)} rows) ---")
print("This step iterates through each row and applies historical lookups. It may take a few seconds...")

historical_features_df = master_df.apply(
    calculate_historical_features, 
    axis=1, 
    df_history=master_df, # The full master_df is used as the source for historical data
    windows=windows_hours
)
master_df = pd.concat([master_df, historical_features_df], axis=1)

print("\n--- Master DataFrame Head (after historical features) ---")
print(master_df.head())


# --- 4. Dataset Consolidation and Preprocessing for Machine Learning ---

# Define features (X) and target (y)
X = master_df.drop(columns=[
    'sensor_ID', 'segment_ID', 'actual_congestion_index', 'traffic_congestion_level', 'timestamp'
])
y = master_df['traffic_congestion_level']

# Identify categorical and numerical features for preprocessing
categorical_features = ['road_type', 'hour_of_day', 'day_of_week', 'day_of_month', 'month']
# All other numeric columns (length, lanes, speed_limit, vehicle_count, observed_speed, historical aggregates, is_weekend)
numerical_features = X.select_dtypes(include=np.number).columns.tolist()
# Ensure categorical features are not accidentally in numerical_features if they were stored as numbers
numerical_features = [f for f in numerical_features if f not in categorical_features]

# Preprocessing Pipeline
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean').set_output(transform="pandas")), # Impute missing numericals (e.g., initial NaNs in historical features)
    ('scaler', StandardScaler().set_output(transform="pandas")) # Then scale
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent').set_output(transform="pandas")), # Impute missing categories
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False).set_output(transform="pandas")) # One-hot encode
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='passthrough' # Keep other columns (like 'is_weekend' if not explicitly handled)
).set_output(transform="pandas")


print(f"\n--- Features identified for Preprocessing ---")
print(f"Numerical: {numerical_features}")
print(f"Categorical: {categorical_features}")

# --- 5. Time-Series Train-Test Split ---

# Define cutoff date for time-series split
# For our 4-day dataset (Jan 1 to Jan 4), let's use Jan 3rd 00:00:00 as cutoff
cutoff_date = pd.to_datetime('2023-01-03 00:00:00')

X_train = X[master_df['timestamp'] < cutoff_date]
y_train = y[master_df['timestamp'] < cutoff_date]
X_test = X[master_df['timestamp'] >= cutoff_date]
y_test = y[master_df['timestamp'] >= cutoff_date]

print(f"\n--- Train-Test Split (Time-Series) ---")
print(f"Training data range: {master_df[master_df['timestamp'] < cutoff_date]['timestamp'].min()} to {master_df[master_df['timestamp'] < cutoff_date]['timestamp'].max()}")
print(f"Test data range: {master_df[master_df['timestamp'] >= cutoff_date]['timestamp'].min()} to {master_df[master_df['timestamp'] >= cutoff_date]['timestamp'].max()}")
print(f"Train samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")
print(f"Train target distribution:\n{y_train.value_counts(normalize=True)}")
print(f"Test target distribution:\n{y_test.value_counts(normalize=True)}")


# --- 6. Model Selection, Training, and Evaluation ---

# Model selection: RandomForestClassifier
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42, n_estimators=100))
])

print("\n--- Training Model ---")
model.fit(X_train, y_train)
print("Model training complete.")

print("\n--- Evaluating Model on Test Set ---")
y_pred = model.predict(X_test)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

accuracy = accuracy_score(y_test, y_pred)
print(f"\nOverall Accuracy: {accuracy:.2f}")

# Example of predicting for a new data point
if not X_test.empty:
    sample_new_data = X_test.iloc[[0]].copy()
    actual_congestion = y_test.iloc[0]
    
    predicted_congestion = model.predict(sample_new_data)
    
    print(f"\n--- Example Prediction for a test point ---")
    print(f"Actual Congestion Level: {actual_congestion}")
    print(f"Predicted Congestion Level: {predicted_congestion[0]}")
else:
    print("\nNo test data available for example prediction.")