import pandas as pd
import numpy as np
import datetime
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, classification_report

# Ensure reproducibility
np.random.seed(42)

# --- 1. Generate Synthetic Data ---

# --- machines_df ---
N_MACHINES = np.random.randint(500, 701)
machine_ids = np.arange(N_MACHINES)

machine_types = np.random.choice(['TypeA', 'TypeB', 'TypeC'], size=N_MACHINES, p=[0.3, 0.4, 0.3])
locations = np.random.choice(['North', 'South', 'East', 'West'], size=N_MACHINES, p=[0.25, 0.25, 0.25, 0.25])

today = pd.to_datetime(datetime.date.today())
installation_start_date = today - pd.Timedelta(days=3 * 365) # Last 3 years
installation_dates = installation_start_date + pd.to_timedelta(np.random.randint(0, (today - installation_start_date).days + 1, size=N_MACHINES), unit='D')

machines_df = pd.DataFrame({
    'machine_id': machine_ids,
    'machine_type': machine_types,
    'location': locations,
    'installation_date': installation_dates
})

# Introduce major_error_date for 20-30% of machines
num_error_machines = int(N_MACHINES * np.random.uniform(0.2, 0.3))
error_machine_ids = np.random.choice(machine_ids, size=num_error_machines, replace=False)

major_error_dates = {}
last_6_months_start = today - pd.Timedelta(days=180)

for mid in error_machine_ids:
    install_date = machines_df.loc[machines_df['machine_id'] == mid, 'installation_date'].iloc[0]
    # Error date must be after installation and within last 6 months (or later than install if install is very recent)
    valid_start_date_for_error = max(install_date, last_6_months_start)
    if valid_start_date_for_error < today:
        error_date_offset_days = np.random.randint(0, (today - valid_start_date_for_error).days + 1)
        major_error_dates[mid] = valid_start_date_for_error + pd.to_timedelta(error_date_offset_days, unit='D')
    else: # If install date is too recent or in future, no error simulated for criteria
        major_error_dates[mid] = pd.NaT 

# Add major_error_date to machines_df
machines_df['major_error_date'] = machines_df['machine_id'].map(major_error_dates)
machines_df['major_error_date'] = pd.to_datetime(machines_df['major_error_date'])


# --- telemetry_df ---
N_TELEMETRY = np.random.randint(15000, 25001)
telemetry_ids = np.arange(N_TELEMETRY)
telemetry_machine_ids = np.random.choice(machine_ids, size=N_TELEMETRY, replace=True)

# Generate timestamps ensuring they are after installation_date
temp_telemetry_df = pd.DataFrame({'machine_id': telemetry_machine_ids, 'telemetry_id': telemetry_ids})
temp_telemetry_df = temp_telemetry_df.merge(machines_df[['machine_id', 'installation_date', 'machine_type', 'major_error_date']], on='machine_id', how='left')

# Generate timestamps: 0 to (today - installation_date) days
time_offsets = []
for inst_date in temp_telemetry_df['installation_date']:
    days_diff = (today - inst_date).days
    time_offsets.append(np.random.randint(0, days_diff + 1) if days_diff >= 0 else 0)
timestamps = temp_telemetry_df['installation_date'] + pd.to_timedelta(time_offsets, unit='D')
temp_telemetry_df['timestamp'] = timestamps

# Base sensor readings
temperature = np.random.uniform(20, 100, size=N_TELEMETRY)
vibration = np.random.uniform(0, 10, size=N_TELEMETRY)

# Apply machine_type specific baselines
temperature[temp_telemetry_df['machine_type'] == 'TypeC'] *= np.random.uniform(1.05, 1.15) # TypeC higher temp
vibration[temp_telemetry_df['machine_type'] == 'TypeA'] *= np.random.uniform(1.05, 1.15) # TypeA higher vibration

# Introduce increasing trend before major_error_date
has_error_mask = temp_telemetry_df['major_error_date'].notna()
error_telemetry_df = temp_telemetry_df[has_error_mask].copy()

if not error_telemetry_df.empty:
    days_to_error = (error_telemetry_df['major_error_date'] - error_telemetry_df['timestamp']).dt.days
    pre_error_mask = (days_to_error >= 7) & (days_to_error <= 14)

    # Apply increase to temperature and vibration for these specific readings
    original_indices = error_telemetry_df[pre_error_mask].index.to_numpy()
    
    if original_indices.size > 0:
        increase_factor_temp = np.random.uniform(1.05, 1.15)
        increase_factor_vibe = np.random.uniform(1.05, 1.15)

        temperature[original_indices] *= increase_factor_temp
        vibration[original_indices] *= increase_factor_vibe

# Ensure temperature/vibration stay within reasonable bounds after adjustments
temperature = np.clip(temperature, 15, 120)
vibration = np.clip(vibration, 0, 15)

telemetry_df = pd.DataFrame({
    'telemetry_id': temp_telemetry_df['telemetry_id'],
    'machine_id': temp_telemetry_df['machine_id'],
    'timestamp': temp_telemetry_df['timestamp'],
    'temperature': temperature,
    'vibration': vibration
})

telemetry_df = telemetry_df.sort_values(by=['machine_id', 'timestamp']).reset_index(drop=True)


# --- maintenance_df ---
N_MAINTENANCE = np.random.randint(2000, 3001)
maintenance_ids = np.arange(N_MAINTENANCE)
maintenance_machine_ids = np.random.choice(machine_ids, size=N_MAINTENANCE, replace=True)

temp_maintenance_df = pd.DataFrame({'machine_id': maintenance_machine_ids, 'maintenance_id': maintenance_ids})
temp_maintenance_df = temp_maintenance_df.merge(machines_df[['machine_id', 'installation_date', 'major_error_date']], on='machine_id', how='left')

# Generate maintenance_date after installation_date
maint_time_offsets = []
for inst_date in temp_maintenance_df['installation_date']:
    days_diff = (today - inst_date).days
    maint_time_offsets.append(np.random.randint(0, days_diff + 1) if days_diff >= 0 else 0)
maintenance_dates = temp_maintenance_df['installation_date'] + pd.to_timedelta(maint_time_offsets, unit='D')
temp_maintenance_df['maintenance_date'] = maintenance_dates

maintenance_types = np.random.choice(['Routine Check', 'Lubrication', 'Component Repair', 'Software Update'],
                                     size=N_MAINTENANCE, p=[0.4, 0.3, 0.2, 0.1])
temp_maintenance_df['maintenance_type'] = maintenance_types

maintenance_df = temp_maintenance_df[['maintenance_id', 'machine_id', 'maintenance_date', 'maintenance_type']]

# Simulate 'Component Repair' after major errors
additional_maintenance_records = []
existing_maintenance_ids = maintenance_ids.max() + 1 if maintenance_ids.size > 0 else 0

for mid in error_machine_ids:
    error_date = machines_df.loc[machines_df['machine_id'] == mid, 'major_error_date'].iloc[0]
    if pd.isna(error_date):
        continue

    # For 50% of machines with an error, add a 'Component Repair' 0-7 days after the error
    if np.random.rand() < 0.5:
        repair_date = error_date + pd.to_timedelta(np.random.randint(0, 8), unit='D')
        additional_maintenance_records.append({
            'maintenance_id': existing_maintenance_ids,
            'machine_id': mid,
            'maintenance_date': repair_date,
            'maintenance_type': 'Component Repair'
        })
        existing_maintenance_ids += 1

if additional_maintenance_records:
    additional_maintenance_df = pd.DataFrame(additional_maintenance_records)
    maintenance_df = pd.concat([maintenance_df, additional_maintenance_df], ignore_index=True)

maintenance_df = maintenance_df.sort_values(by=['machine_id', 'maintenance_date']).reset_index(drop=True)

print(f"Generated {len(machines_df)} machine records.")
print(f"Generated {len(telemetry_df)} telemetry records.")
print(f"Generated {len(maintenance_df)} maintenance records.")
print(f"Machines with simulated major error: {machines_df['major_error_date'].count()}")


# --- 2. Load into SQLite & SQL Feature Engineering ---

conn = sqlite3.connect(':memory:')

# Convert datetime columns to string for SQLite compatibility and proper storage
# SQLite does not have a native datetime type, stores as TEXT, INTEGER, or REAL
machines_df['installation_date'] = machines_df['installation_date'].dt.strftime('%Y-%m-%d %H:%M:%S')
machines_df['major_error_date'] = machines_df['major_error_date'].dt.strftime('%Y-%m-%d %H:%M:%S').fillna('') # Fill NaT with empty string
telemetry_df['timestamp'] = telemetry_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
maintenance_df['maintenance_date'] = maintenance_df['maintenance_date'].dt.strftime('%Y-%m-%d %H:%M:%S')


machines_df.to_sql('machines', conn, index=False, if_exists='replace')
telemetry_df.to_sql('telemetry', conn, index=False, if_exists='replace')
maintenance_df.to_sql('maintenance', conn, index=False, if_exists='replace')


# SQL query for time-windowed aggregations
sql_query = """
WITH LatestTelemetry AS (
    SELECT
        machine_id,
        MAX(timestamp) AS latest_telemetry_date
    FROM telemetry
    GROUP BY machine_id
),
TelemetryAgg AS (
    SELECT
        t.machine_id,
        AVG(t.temperature) AS avg_temp_prev_30d,
        MAX(t.vibration) AS max_vibration_prev_30d,
        COUNT(t.telemetry_id) AS num_telemetry_readings_prev_30d
    FROM telemetry t
    JOIN LatestTelemetry lt ON t.machine_id = lt.machine_id
    WHERE julianday(t.timestamp) BETWEEN julianday(lt.latest_telemetry_date) - 30 AND julianday(lt.latest_telemetry_date)
    GROUP BY t.machine_id
),
MaintenanceAgg AS (
    SELECT
        m.machine_id,
        COUNT(m.maintenance_id) AS num_maintenances_prev_30d,
        SUM(CASE WHEN m.maintenance_type = 'Component Repair' THEN 1 ELSE 0 END) AS num_repairs_prev_30d
    FROM maintenance m
    JOIN LatestTelemetry lt ON m.machine_id = lt.machine_id
    WHERE julianday(m.maintenance_date) BETWEEN julianday(lt.latest_telemetry_date) - 30 AND julianday(lt.latest_telemetry_date)
    GROUP BY m.machine_id
)
SELECT
    m.machine_id,
    m.machine_type,
    m.location,
    m.installation_date,
    lt.latest_telemetry_date AS current_observation_date,
    COALESCE(ta.avg_temp_prev_30d, 0.0) AS avg_temp_prev_30d,
    COALESCE(ta.max_vibration_prev_30d, 0.0) AS max_vibration_prev_30d,
    COALESCE(ta.num_telemetry_readings_prev_30d, 0) AS num_telemetry_readings_prev_30d,
    COALESCE(ma.num_maintenances_prev_30d, 0) AS num_maintenances_prev_30d,
    COALESCE(ma.num_repairs_prev_30d, 0) AS num_repairs_prev_30d
FROM machines m
LEFT JOIN LatestTelemetry lt ON m.machine_id = lt.machine_id
LEFT JOIN TelemetryAgg ta ON m.machine_id = ta.machine_id
LEFT JOIN MaintenanceAgg ma ON m.machine_id = ma.machine_id
ORDER BY m.machine_id;
"""

machine_features_df = pd.read_sql_query(sql_query, conn)

# Close the connection
conn.close()

print("\nSQL Feature Engineering Result (first 5 rows):")
print(machine_features_df.head())
print(f"Generated {len(machine_features_df)} machine feature records.")


# --- 3. Pandas Feature Engineering & Binary Target Creation ---

# Convert date columns to datetime objects
machine_features_df['installation_date'] = pd.to_datetime(machine_features_df['installation_date'])
machine_features_df['current_observation_date'] = pd.to_datetime(machine_features_df['current_observation_date'])

# Handle potential NaNs in 'current_observation_date' if some machines have no telemetry
# For machines without any telemetry, 'current_observation_date' would be NaT.
# We'll forward fill to 'installation_date' so we can calculate 'days_since_installation_at_obs',
# although these machines likely won't have other telemetry-derived features.
machine_features_df['current_observation_date'] = machine_features_df['current_observation_date'].fillna(
    machine_features_df['installation_date']
)

# Calculate days_since_installation_at_obs
machine_features_df['days_since_installation_at_obs'] = (
    (machine_features_df['current_observation_date'] - machine_features_df['installation_date']).dt.days
)
machine_features_df['days_since_installation_at_obs'] = machine_features_df['days_since_installation_at_obs'].fillna(0).clip(lower=0)


# Calculate telemetry_frequency_prev_30d
# Avoid division by zero by ensuring 30.0 for the denominator. If num_readings is 0, frequency is 0.
machine_features_df['telemetry_frequency_prev_30d'] = (
    machine_features_df['num_telemetry_readings_prev_30d'] / 30.0
)
machine_features_df['telemetry_frequency_prev_30d'] = machine_features_df['telemetry_frequency_prev_30d'].replace([np.inf, -np.inf], np.nan).fillna(0)


# Create the Binary Target 'major_error_in_next_7_days'
# Merge major_error_date from original machines_df
machines_df['major_error_date'] = pd.to_datetime(machines_df['major_error_date'], errors='coerce')
machine_features_df = machine_features_df.merge(
    machines_df[['machine_id', 'major_error_date']],
    on='machine_id',
    how='left'
)

# Define the 7-day prediction window immediately following current_observation_date
prediction_window_start = machine_features_df['current_observation_date']
prediction_window_end = prediction_window_start + pd.Timedelta(days=7)

# Check if major_error_date falls within this window
machine_features_df['major_error_in_next_7_days'] = (
    (machine_features_df['major_error_date'] >= prediction_window_start) &
    (machine_features_df['major_error_date'] < prediction_window_end)
).astype(int) # Convert boolean to 0 or 1

# Drop the temporary major_error_date column
machine_features_df = machine_features_df.drop(columns=['major_error_date'])

print("\nPandas Feature Engineering & Target Creation Result (first 5 rows with target):")
print(machine_features_df.head())
print(f"Target distribution: \n{machine_features_df['major_error_in_next_7_days'].value_counts(normalize=True)}")

# Define features (X) and target (y)
numerical_features = [
    'avg_temp_prev_30d', 'max_vibration_prev_30d',
    'num_telemetry_readings_prev_30d', 'num_maintenances_prev_30d',
    'num_repairs_prev_30d', 'days_since_installation_at_obs',
    'telemetry_frequency_prev_30d'
]
categorical_features = ['machine_type', 'location']
all_features = numerical_features + categorical_features

X = machine_features_df[all_features]
y = machine_features_df['major_error_in_next_7_days']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"\nTrain set size: {len(X_train)} samples")
print(f"Test set size: {len(X_test)} samples")
print(f"Train target distribution: \n{y_train.value_counts(normalize=True)}")
print(f"Test target distribution: \n{y_test.value_counts(normalize=True)}")


# --- 4. Data Visualization ---

# Plot 1: Violin plot for max_vibration_prev_30d vs major_error_in_next_7_days
plt.figure(figsize=(10, 6))
sns.violinplot(x='major_error_in_next_7_days', y='max_vibration_prev_30d', data=machine_features_df)
plt.title('Distribution of Max Vibration (Prev 30d) by Major Error Status')
plt.xlabel('Major Error in Next 7 Days (0=No Error, 1=Error)')
plt.ylabel('Max Vibration (Prev 30d)')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Plot 2: Stacked bar chart for machine_type vs major_error_in_next_7_days
plt.figure(figsize=(10, 6))
# Calculate proportions
error_proportions = machine_features_df.groupby('machine_type')['major_error_in_next_7_days'].value_counts(normalize=True).unstack().fillna(0)
error_proportions.plot(kind='bar', stacked=True, color=['skyblue', 'salmon'])
plt.title('Proportion of Major Error by Machine Type')
plt.xlabel('Machine Type')
plt.ylabel('Proportion')
plt.xticks(rotation=45)
plt.legend(title='Major Error in Next 7 Days', labels=['No Error', 'Error'])
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


# --- 5. ML Pipeline & Evaluation ---

# Create preprocessing pipelines for numerical and categorical features
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Create a preprocessor using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='passthrough' # Keep other columns if any, though not expected here
)

# Create the full pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', HistGradientBoostingClassifier(random_state=42))
])

# Train the pipeline
pipeline.fit(X_train, y_train)

# Predict probabilities for the positive class (class 1) on the test set
y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

# Calculate ROC AUC Score
roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"\nROC AUC Score on Test Set: {roc_auc:.4f}")

# Convert probabilities to binary predictions for classification report (using 0.5 threshold)
y_pred = (y_pred_proba > 0.5).astype(int)

# Print classification report
print("\nClassification Report on Test Set:")
print(classification_report(y_test, y_pred))