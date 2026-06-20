import pandas as pd
import numpy as np
import sqlite3
import datetime
from datetime import timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer # FIX: Correct import for SimpleImputer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, classification_report

# Set a random seed for reproducibility
np.random.seed(42)

# --- 1. Generate Synthetic Data ---

print("1. Generating Synthetic Data...")

# Patients DataFrame
N_PATIENTS = np.random.randint(1000, 1501)
patients_data = {
    'patient_id': np.arange(N_PATIENTS),
    'age': np.random.randint(18, 91, N_PATIENTS),
    'gender': np.random.choice(['Male', 'Female', 'Other'], N_PATIENTS, p=[0.45, 0.45, 0.1]),
    'existing_condition': np.random.choice(['None', 'Chronic', 'Acute'], N_PATIENTS, p=[0.6, 0.3, 0.1]),
    'signup_date': pd.to_datetime(pd.Timestamp.now() - pd.to_timedelta(np.random.randint(365 * 5, 365 * 10, N_PATIENTS), unit='D'))
}
patients_df = pd.DataFrame(patients_data)

# Doctors DataFrame
N_DOCTORS = np.random.randint(50, 101)
doctors_data = {
    'doctor_id': np.arange(N_DOCTORS),
    'specialty': np.random.choice(['Cardiology', 'Pediatrics', 'Dermatology', 'General', 'Orthopedics', 'Neurology', 'Oncology'], N_DOCTORS),
    'doctor_experience_years': np.random.randint(3, 31, N_DOCTORS),
    'doctor_rating': np.random.uniform(3.0, 5.0, N_DOCTORS).round(1)
}
doctors_df = pd.DataFrame(doctors_data)

# Appointments DataFrame
N_APPOINTMENTS = np.random.randint(30000, 50001)

# Merge patient and doctor info for easier simulation of _attended
temp_appointments_df = pd.DataFrame({
    'appointment_id': np.arange(N_APPOINTMENTS),
    'patient_id': np.random.choice(patients_df['patient_id'], N_APPOINTMENTS),
    'doctor_id': np.random.choice(doctors_df['doctor_id'], N_APPOINTMENTS)
})

# Add signup_date from patients_df
temp_appointments_df = temp_appointments_df.merge(patients_df[['patient_id', 'signup_date', 'existing_condition']], on='patient_id', how='left')
# Add doctor_rating from doctors_df
temp_appointments_df = temp_appointments_df.merge(doctors_df[['doctor_id', 'doctor_rating']], on='doctor_id', how='left')

# Generate appointment_datetime ensuring it's after signup_date
# Generate random offsets from signup_date, then add them
temp_appointments_df['appointment_datetime_offset'] = np.random.randint(1, 365 * 5, N_APPOINTMENTS) # Appointments up to 5 years after signup
temp_appointments_df['appointment_datetime'] = temp_appointments_df['signup_date'] + pd.to_timedelta(temp_appointments_df['appointment_datetime_offset'], unit='D')
# Ensure no appointment is significantly in the future relative to "now"
latest_possible_appt_date = pd.Timestamp.now() + pd.Timedelta(days=10) # A small buffer to allow recent appointments
temp_appointments_df['appointment_datetime'] = temp_appointments_df['appointment_datetime'].clip(upper=latest_possible_appt_date)


# Simulate _attended (binary: 1 for attended, 0 for no-show)
# Base no-show probability
base_no_show_prob = 0.15

# Adjust probability based on factors
temp_appointments_df['no_show_prob'] = base_no_show_prob
temp_appointments_df.loc[temp_appointments_df['existing_condition'] == 'Chronic', 'no_show_prob'] += 0.05
temp_appointments_df.loc[temp_appointments_df['existing_condition'] == 'Acute', 'no_show_prob'] += 0.02
temp_appointments_df.loc[temp_appointments_df['doctor_rating'] < 3.5, 'no_show_prob'] += 0.07
temp_appointments_df.loc[temp_appointments_df['doctor_rating'] < 4.0, 'no_show_prob'] += 0.03
temp_appointments_df['no_show_prob'] = temp_appointments_df['no_show_prob'].clip(0.05, 0.45) # Clip probabilities

# Sort for cumulative no-show calculation
temp_appointments_df = temp_appointments_df.sort_values(by=['patient_id', 'appointment_datetime']).reset_index(drop=True)

# Add past no-show influence (this requires a loop or cumulative operation)
# This is a potentially slow part for very large N_APPOINTMENTS but ensures realistic simulation
temp_appointments_df['past_no_shows_patient'] = 0
temp_appointments_df['_attended_simulated'] = -1

current_patient_id = -1
cumulative_no_shows = 0

for idx, row in temp_appointments_df.iterrows():
    if row['patient_id'] != current_patient_id:
        current_patient_id = row['patient_id']
        cumulative_no_shows = 0
    
    current_prob = row['no_show_prob']
    if cumulative_no_shows > 0:
        current_prob += 0.10 # Increase no-show prob if past no-shows
    
    # Ensure prob stays within reasonable bounds
    current_prob = min(current_prob, 0.5) 
    
    attended = 1 if np.random.rand() > current_prob else 0
    if attended == 0:
        cumulative_no_shows += 1
    
    temp_appointments_df.loc[idx, '_attended_simulated'] = attended
    temp_appointments_df.loc[idx, 'past_no_shows_patient'] = cumulative_no_shows - (1 - attended) # cumulative before *this* appt

# Final appointments_df
appointments_df = temp_appointments_df.drop(columns=['signup_date', 'existing_condition', 'doctor_rating', 'appointment_datetime_offset', 'no_show_prob', 'past_no_shows_patient'])
appointments_df = appointments_df.rename(columns={'_attended_simulated': '_attended'})
appointments_df['_attended'] = appointments_df['_attended'].astype(int)

# Sort `appointments_df` by `patient_id` then `appointment_datetime`.
appointments_df = appointments_df.sort_values(by=['patient_id', 'appointment_datetime']).reset_index(drop=True)

print(f"Generated {len(patients_df)} patients, {len(doctors_df)} doctors, {len(appointments_df)} appointments.")
print(f"Overall no-show rate: {100 * (1 - appointments_df['_attended']).mean():.2f}%")

# Convert datetimes to string for SQLite
patients_df['signup_date_str'] = patients_df['signup_date'].dt.strftime('%Y-%m-%d %H:%M:%S')
appointments_df['appointment_datetime_str'] = appointments_df['appointment_datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')

# --- 2. Load into SQLite & SQL Feature Engineering ---

print("\n2. Loading data into SQLite and performing SQL Feature Engineering...")

conn = sqlite3.connect(':memory:')

patients_df.to_sql('patients', conn, if_exists='replace', index=False)
doctors_df.to_sql('doctors', conn, if_exists='replace', index=False)
appointments_df.to_sql('appointments', conn, if_exists='replace', index=False)

# Define GLOBAL_PREDICTION_CUTOFF_DATE
latest_appointment_datetime = pd.to_datetime(appointments_df['appointment_datetime']).max()
GLOBAL_PREDICTION_CUTOFF_DATE = latest_appointment_datetime - timedelta(days=7)
GLOBAL_PREDICTION_CUTOFF_DATE_STR = GLOBAL_PREDICTION_CUTOFF_DATE.strftime('%Y-%m-%d %H:%M:%S')

print(f"Global prediction cutoff date: {GLOBAL_PREDICTION_CUTOFF_DATE_STR}")

# Function to execute SQL query and handle potential empty results
def execute_sql_query(conn, cutoff_date_str):
    sql_query = f"""
    WITH HistoricalAppointments AS (
        SELECT
            ap.appointment_id,
            ap.patient_id,
            ap.doctor_id,
            ap.appointment_datetime_str AS appointment_datetime,
            ap._attended,
            pa.signup_date_str AS signup_date
        FROM appointments ap
        JOIN patients pa ON ap.patient_id = pa.patient_id
        WHERE ap.appointment_datetime_str <= '{cutoff_date_str}'
    ),
    PatientHistoricalAggregates AS (
        SELECT
            patient_id,
            SUM(CASE WHEN julianday('{cutoff_date_str}') - julianday(appointment_datetime) <= 90 THEN 1 ELSE 0 END) AS num_past_appointments_patient_prev_90d,
            CAST(SUM(CASE WHEN julianday('{cutoff_date_str}') - julianday(appointment_datetime) <= 90 AND _attended = 0 THEN 1 ELSE 0 END) AS REAL) /
            NULLIF(SUM(CASE WHEN julianday('{cutoff_date_str}') - julianday(appointment_datetime) <= 90 THEN 1 ELSE 0 END), 0) AS patient_no_show_rate_prev_90d,
            MAX(appointment_datetime) AS last_appointment_datetime_at_cutoff,
            MIN(signup_date) AS patient_signup_date -- Need signup date for tenure calculation
        FROM HistoricalAppointments
        GROUP BY patient_id
    ),
    DoctorHistoricalAggregates AS (
        SELECT
            doctor_id,
            CAST(SUM(CASE WHEN julianday('{cutoff_date_str}') - julianday(appointment_datetime) <= 90 AND _attended = 0 THEN 1 ELSE 0 END) AS REAL) /
            NULLIF(SUM(CASE WHEN julianday('{cutoff_date_str}') - julianday(appointment_datetime) <= 90 THEN 1 ELSE 0 END), 0) AS doctor_avg_no_show_rate_prev_90d
        FROM HistoricalAppointments
        GROUP BY doctor_id
    )
    SELECT
        ap.appointment_id,
        ap.patient_id,
        ap.doctor_id,
        ap.appointment_datetime_str AS appointment_datetime,
        ap._attended,
        pa.age,
        pa.gender,
        pa.existing_condition,
        doc.specialty,
        doc.doctor_experience_years,
        doc.doctor_rating,
        -- Historical features for patient
        COALESCE(pha.num_past_appointments_patient_prev_90d, 0) AS num_past_appointments_patient_prev_90d,
        COALESCE(pha.patient_no_show_rate_prev_90d, 0.0) AS patient_no_show_rate_prev_90d,
        COALESCE(
            CAST(julianday('{cutoff_date_str}') - julianday(pha.last_appointment_datetime_at_cutoff) AS INTEGER),
            9999
        ) AS days_since_last_appointment_at_cutoff,
        -- Historical features for doctor
        COALESCE(dha.doctor_avg_no_show_rate_prev_90d, 0.0) AS doctor_avg_no_show_rate_prev_90d,
        -- Time-based features for current appointment
        STRFTIME('%w', ap.appointment_datetime_str) AS appointment_day_of_week, -- 0 for Sunday, 6 for Saturday
        STRFTIME('%H', ap.appointment_datetime_str) AS appointment_hour_of_day,
        pha.patient_signup_date AS signup_date -- Patient signup date at cutoff
    FROM appointments ap
    JOIN patients pa ON ap.patient_id = pa.patient_id
    JOIN doctors doc ON ap.doctor_id = doc.doctor_id
    LEFT JOIN PatientHistoricalAggregates pha ON ap.patient_id = pha.patient_id
    LEFT JOIN DoctorHistoricalAggregates dha ON ap.doctor_id = dha.doctor_id
    WHERE ap.appointment_datetime_str > '{cutoff_date_str}'
    ORDER BY ap.patient_id, ap.appointment_datetime_str;
    """
    return pd.read_sql_query(sql_query, conn)

appointment_features_df = execute_sql_query(conn, GLOBAL_PREDICTION_CUTOFF_DATE_STR)

if appointment_features_df.empty:
    print("WARNING: No future appointments found after initial cutoff date. Adjusting cutoff date to ensure data for model training.")
    # If no data, shift cutoff back to ensure there is data for prediction
    # Use 70th percentile of appointment_datetime as cutoff to ensure historical and future data
    GLOBAL_PREDICTION_CUTOFF_DATE = pd.to_datetime(appointments_df['appointment_datetime']).quantile(0.7) 
    GLOBAL_PREDICTION_CUTOFF_DATE_STR = GLOBAL_PREDICTION_CUTOFF_DATE.strftime('%Y-%m-%d %H:%M:%S')
    
    print(f"Adjusted global prediction cutoff date to: {GLOBAL_PREDICTION_CUTOFF_DATE_STR}")
    appointment_features_df = execute_sql_query(conn, GLOBAL_PREDICTION_CUTOFF_DATE_STR)
    print(f"Re-extracted {len(appointment_features_df)} future appointments after cutoff adjustment.")

conn.close()

if appointment_features_df.empty:
    print("Error: Still no future appointments after cutoff date even after adjustment. Cannot proceed.")
    exit()

print(f"Extracted {len(appointment_features_df)} future appointments for prediction.")

# --- 3. Pandas Feature Engineering & Binary Target Creation ---

print("\n3. Performing Pandas Feature Engineering...")

# Convert datetime columns
appointment_features_df['appointment_datetime'] = pd.to_datetime(appointment_features_df['appointment_datetime'])
appointment_features_df['signup_date'] = pd.to_datetime(appointment_features_df['signup_date'])
GLOBAL_PREDICTION_CUTOFF_DATE_PANDAS = pd.to_datetime(GLOBAL_PREDICTION_CUTOFF_DATE_STR) # Ensure pandas datetime object

# Handle NaN values as specified (SQL handles some, but defensive checks here)
appointment_features_df['num_past_appointments_patient_prev_90d'] = appointment_features_df['num_past_appointments_patient_prev_90d'].fillna(0).astype(int)
appointment_features_df['patient_no_show_rate_prev_90d'] = appointment_features_df['patient_no_show_rate_prev_90d'].fillna(0.0)
appointment_features_df['days_since_last_appointment_at_cutoff'] = appointment_features_df['days_since_last_appointment_at_cutoff'].fillna(9999).astype(int)
appointment_features_df['doctor_avg_no_show_rate_prev_90d'] = appointment_features_df['doctor_avg_no_show_rate_prev_90d'].fillna(0.0)

# Convert day and hour to int
appointment_features_df['appointment_day_of_week'] = appointment_features_df['appointment_day_of_week'].astype(int)
appointment_features_df['appointment_hour_of_day'] = appointment_features_df['appointment_hour_of_day'].astype(int)

# Fill static attributes with means if any NaNs (shouldn't happen with joins but for robustness)
for col in ['age', 'doctor_experience_years', 'doctor_rating']:
    if appointment_features_df[col].isnull().any():
        appointment_features_df[col] = appointment_features_df[col].fillna(appointment_features_df[col].mean())

# Calculate patient_tenure_at_cutoff_days
appointment_features_df['patient_tenure_at_cutoff_days'] = (GLOBAL_PREDICTION_CUTOFF_DATE_PANDAS - appointment_features_df['signup_date']).dt.days
appointment_features_df['patient_tenure_at_cutoff_days'] = appointment_features_df['patient_tenure_at_cutoff_days'].fillna(0).astype(int).clip(lower=0) # Ensure no negative tenure

# Create the Binary Target `will_no_show`
appointment_features_df['will_no_show'] = (appointment_features_df['_attended'] == 0).astype(int)

# Define features X and target y
numerical_features = [
    'age',
    'doctor_experience_years',
    'doctor_rating',
    'num_past_appointments_patient_prev_90d',
    'patient_no_show_rate_prev_90d',
    'days_since_last_appointment_at_cutoff',
    'doctor_avg_no_show_rate_prev_90d',
    'appointment_day_of_week',
    'appointment_hour_of_day',
    'patient_tenure_at_cutoff_days'
]
categorical_features = [
    'gender',
    'existing_condition',
    'specialty'
]

X = appointment_features_df[numerical_features + categorical_features]
y = appointment_features_df['will_no_show']

# Check if there's enough data for splitting
if len(X) < 2 or len(y.unique()) < 2:
    print("Error: Not enough data or classes for train-test split. Cannot proceed with ML pipeline.")
    exit()

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

print(f"Training set size: {len(X_train)} appointments")
print(f"Test set size: {len(X_test)} appointments")
print(f"No-show rate in training set: {100 * y_train.mean():.2f}%")
print(f"No-show rate in test set: {100 * y_test.mean():.2f}%")


# --- 4. Data Visualization ---

print("\n4. Generating Data Visualizations...")

plt.figure(figsize=(14, 6))

# Violin plot for days_since_last_appointment_at_cutoff
plt.subplot(1, 2, 1)
sns.violinplot(x='will_no_show', y='days_since_last_appointment_at_cutoff', data=appointment_features_df, palette='viridis')
plt.title('Days Since Last Appointment by No-Show Status')
plt.xlabel('Will No-Show (0: Attended, 1: No-Show)')
plt.ylabel('Days Since Last Appointment at Cutoff')
plt.xticks([0, 1], ['Attended', 'No-Show'])

# Stacked bar chart for specialty vs. no-show proportion
plt.subplot(1, 2, 2)
specialty_counts = appointment_features_df.groupby(['specialty', 'will_no_show']).size().unstack(fill_value=0)
specialty_proportions = specialty_counts.apply(lambda x: x / x.sum(), axis=1)
specialty_proportions.plot(kind='bar', stacked=True, ax=plt.gca(), cmap='coolwarm')
plt.title('No-Show Proportion by Specialty')
plt.xlabel('Specialty')
plt.ylabel('Proportion')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Will No-Show', labels=['Attended', 'No-Show'])
plt.tight_layout()
plt.show()


# --- 5. ML Pipeline & Evaluation ---

print("\n5. Building, Training, and Evaluating ML Pipeline...")

# Create preprocessing pipelines for numerical and categorical features
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Create a column transformer to apply different transformations to different columns
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='passthrough' # Keep other columns (e.g., IDs) if any, untransformed
)

# Create the full pipeline
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', HistGradientBoostingClassifier(random_state=42, class_weight='balanced'))
])

# Train the model
print("Training the model...")
model_pipeline.fit(X_train, y_train)
print("Model training complete.")

# Predict probabilities on the test set
y_pred_proba = model_pipeline.predict_proba(X_test)[:, 1]

# Predict classes for classification report (using default threshold 0.5)
y_pred_class = model_pipeline.predict(X_test)

# Evaluate the model
roc_auc = roc_auc_score(y_test, y_pred_proba)
class_report = classification_report(y_test, y_pred_class, target_names=['Attended', 'No-Show'])

print(f"\nROC AUC Score on Test Set: {roc_auc:.4f}")
print("\nClassification Report on Test Set:")
print(class_report)

print("\nScript finished successfully.")