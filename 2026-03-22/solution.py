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

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')

# --- 1. Generate Synthetic Data ---

print("--- Generating Synthetic Data ---")

# Define ranges for data generation
N_PATIENTS = np.random.randint(1000, 1501)
N_APPOINTMENTS = np.random.randint(15000, 20001)

# Patients Data
genders = ['Male', 'Female', 'Non-binary']
insurance_types = ['Private', 'Public', 'None']

patients_df = pd.DataFrame({
    'patient_id': np.arange(1, N_PATIENTS + 1),
    'signup_date': pd.to_datetime('now') - pd.to_timedelta(np.random.randint(1, 7*365, N_PATIENTS), unit='D'),
    'age': np.random.randint(18, 91, N_PATIENTS),
    'gender': np.random.choice(genders, N_PATIENTS),
    'insurance_type': np.random.choice(insurance_types, N_PATIENTS, p=[0.5, 0.35, 0.15]),
    'num_existing_conditions': np.random.randint(0, 6, N_PATIENTS)
})

# Add a 'no_show_propensity' for each patient to simulate inherent likelihood
# Skewed towards lower values, but some patients have higher propensity
patients_df['no_show_propensity'] = np.random.beta(a=2, b=8, size=N_PATIENTS) * 0.1 # Base 0-0.1 for individual patient bias

print(f"Generated {N_PATIENTS} patients.")

# Appointments Data
clinic_locations = ['Downtown', 'Suburban_East', 'Suburban_West', 'Rural_Clinic']
specialties = ['General Practice', 'Cardiology', 'Dermatology', 'Pediatrics', 'Ophthalmology', 'Orthopedics']

appointment_data = []
patient_ids = patients_df['patient_id'].values
patient_signup_dates = patients_df.set_index('patient_id')['signup_date']
patient_num_conditions = patients_df.set_index('patient_id')['num_existing_conditions']
patient_insurance_types = patients_df.set_index('patient_id')['insurance_type']
patient_no_show_propensity = patients_df.set_index('patient_id')['no_show_propensity']


# Base no-show probability and bias factors (tuned for ~10-20% overall rate)
BASE_NO_SHOW_PROB = 0.08

BIAS_INSURANCE = {'Private': 0, 'Public': 0.04, 'None': 0.08}
BIAS_CONDITIONS = {0: 0, 1: 0.01, 2: 0.02, 3: 0.04, 4: 0.06, 5: 0.08}
BIAS_CLINIC = {'Downtown': 0.03, 'Suburban_East': 0.01, 'Suburban_West': 0.02, 'Rural_Clinic': 0.05}
BIAS_SPECIALTY = {'General Practice': 0, 'Cardiology': 0.02, 'Dermatology': 0.01, 'Pediatrics': 0.03, 'Ophthalmology': 0.01, 'Orthopedics': 0.02}
BIAS_DAY_OF_WEEK = {0: 0.03, 1: 0, 2: 0, 3: 0, 4: 0.02, 5: 0.05, 6: 0.04} # Mon, Fri, Sat, Sun higher
BIAS_TIME_OF_DAY = {'Morning': 0.01, 'Afternoon': 0, 'Evening': 0.02} # Evening slightly higher

for i in range(N_APPOINTMENTS):
    appt_id = i + 1
    patient_id = np.random.choice(patient_ids)
    
    signup_date = patient_signup_dates[patient_id]
    num_existing_conditions = patient_num_conditions[patient_id]
    insurance_type = patient_insurance_types[patient_id]
    patient_propensity = patient_no_show_propensity[patient_id]

    # Ensure scheduled_datetime is after signup_date and not in the far future
    scheduled_datetime_min = signup_date + pd.Timedelta(days=1)
    # Max scheduled_datetime can be up to 'now' so appt_datetime can be 'now' + 90 days.
    # To keep appt_datetime within a reasonable historical range (e.g., last 5 years),
    # ensure scheduled_datetime is generated within last 5 years and after signup.
    
    # Random point in time for scheduled, ensuring it's after signup_date but not too recent
    # so that `appt_datetime` often falls within the last 5 years
    five_years_ago = pd.to_datetime('now') - pd.Timedelta(days=5*365)
    
    # Scheduled datetime must be after patient signup date AND after 5 years ago (for data freshness)
    # AND before today (to allow appt_datetime to be in the near future)
    start_scheduled_range = max(scheduled_datetime_min, five_years_ago)
    end_scheduled_range = pd.to_datetime('now') - pd.Timedelta(days=1) # At least 1 day before now for some scheduling buffer

    if start_scheduled_range >= end_scheduled_range: # Handle cases where signup is very recent
        scheduled_datetime = end_scheduled_range
    else:
        scheduled_datetime = start_scheduled_range + pd.to_timedelta(
            np.random.randint(1, (end_scheduled_range - start_scheduled_range).days + 1), unit='D'
        )
    
    # appt_datetime must be after scheduled_datetime and within 90 days
    scheduled_days_in_advance = np.random.randint(1, 91)
    appt_datetime = scheduled_datetime + pd.to_timedelta(scheduled_days_in_advance, unit='D')

    clinic_loc = np.random.choice(clinic_locations)
    specialty = np.random.choice(specialties)

    # Calculate no-show probability with biases
    current_prob = BASE_NO_SHOW_PROB + patient_propensity

    current_prob += BIAS_INSURANCE.get(insurance_type, 0)
    current_prob += BIAS_CONDITIONS.get(num_existing_conditions, 0)
    current_prob += BIAS_CLINIC.get(clinic_loc, 0)
    current_prob += BIAS_SPECIALTY.get(specialty, 0)

    # Bias based on scheduled_days_in_advance (U-shaped: very short or very long)
    if scheduled_days_in_advance < 3:
        current_prob += 0.04
    elif scheduled_days_in_advance < 7:
        current_prob += 0.02
    elif scheduled_days_in_advance > 60:
        current_prob += 0.03
    elif scheduled_days_in_advance > 30:
        current_prob += 0.01

    # Bias based on day of week and time of day
    appt_day_of_week = appt_datetime.dayofweek # Monday=0, Sunday=6
    current_prob += BIAS_DAY_OF_WEEK.get(appt_day_of_week, 0)

    appt_hour = appt_datetime.hour
    if 6 <= appt_hour <= 11:
        time_of_day_category = 'Morning'
    elif 12 <= appt_hour <= 17:
        time_of_day_category = 'Afternoon'
    else:
        time_of_day_category = 'Evening'
    current_prob += BIAS_TIME_OF_DAY.get(time_of_day_category, 0)

    # Clip probability between 0 and 0.6 (realistic max for single appointment)
    current_prob = np.clip(current_prob, 0.01, 0.6) # Ensure always a small chance, and capped at 60%
    was_no_show = 1 if np.random.rand() < current_prob else 0

    appointment_data.append({
        'appt_id': appt_id,
        'patient_id': patient_id,
        'scheduled_datetime': scheduled_datetime,
        'appt_datetime': appt_datetime,
        'clinic_location': clinic_loc,
        'specialty': specialty,
        'was_no_show': was_no_show
    })

appointments_df = pd.DataFrame(appointment_data)
appointments_df = appointments_df.sort_values(by=['patient_id', 'appt_datetime']).reset_index(drop=True)

print(f"Generated {N_APPOINTMENTS} appointments.")
print(f"Overall No-Show Rate: {appointments_df['was_no_show'].mean():.2%}")
print("--- Data Generation Complete ---")

# --- 2. Load into SQLite & SQL Feature Engineering ---

print("\n--- Loading data into SQLite and SQL Feature Engineering ---")

conn = sqlite3.connect(':memory:')
patients_df.to_sql('patients', conn, index=False, if_exists='replace')
appointments_df.to_sql('appointments', conn, index=False, if_exists='replace')

sql_query = """
WITH PriorAppts AS (
    SELECT
        a.appt_id,
        a.patient_id,
        a.scheduled_datetime,
        a.appt_datetime,
        a.clinic_location,
        a.specialty,
        a.was_no_show,
        p.signup_date,
        p.age,
        p.gender,
        p.insurance_type,
        p.num_existing_conditions,
        -- Calculate prior appt counts and no-show counts
        SUM(CASE WHEN a.appt_id IS NOT NULL THEN 1 ELSE 0 END) OVER (
            PARTITION BY a.patient_id
            ORDER BY a.appt_datetime
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
        ) AS patient_prior_appts_count_raw,
        SUM(CASE WHEN a.was_no_show = 1 THEN 1 ELSE 0 END) OVER (
            PARTITION BY a.patient_id
            ORDER BY a.appt_datetime
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
        ) AS patient_prior_no_shows_count_raw,
        -- Get previous appt_datetime for 'days_since_last_patient_appt'
        LAG(julianday(a.appt_datetime), 1, julianday(p.signup_date)) OVER (
            PARTITION BY a.patient_id
            ORDER BY a.appt_datetime
        ) AS prev_appt_julianday
    FROM
        appointments a
    JOIN
        patients p ON a.patient_id = p.patient_id
)
SELECT
    pa.appt_id,
    pa.patient_id,
    pa.appt_datetime,
    pa.was_no_show,
    pa.age,
    pa.gender,
    pa.insurance_type,
    pa.num_existing_conditions,
    pa.clinic_location,
    pa.specialty,
    pa.signup_date,

    -- Sequential features
    COALESCE(pa.patient_prior_appts_count_raw, 0) AS patient_prior_appts_count,
    COALESCE(pa.patient_prior_no_shows_count_raw, 0) AS patient_prior_no_shows_count,
    CASE
        WHEN pa.patient_prior_appts_count_raw IS NULL OR pa.patient_prior_appts_count_raw = 0 THEN 0.0
        ELSE CAST(pa.patient_prior_no_shows_count_raw AS REAL) / pa.patient_prior_appts_count_raw
    END AS patient_prior_no_show_rate,
    
    -- days_since_last_patient_appt (uses signup_date if no prior appt)
    JULIANDAY(pa.appt_datetime) - pa.prev_appt_julianday AS days_since_last_patient_appt,

    -- Current appointment and scheduling features
    JULIANDAY(pa.appt_datetime) - JULIANDAY(pa.scheduled_datetime) AS scheduled_days_in_advance,
    
    -- Day of week for appt_datetime (0=Sunday, 1=Monday, ..., 6=Saturday)
    CASE CAST(STRFTIME('%w', pa.appt_datetime) AS INTEGER)
        WHEN 0 THEN 'Sunday'
        WHEN 1 THEN 'Monday'
        WHEN 2 THEN 'Tuesday'
        WHEN 3 THEN 'Wednesday'
        WHEN 4 THEN 'Thursday'
        WHEN 5 THEN 'Friday'
        WHEN 6 THEN 'Saturday'
    END AS appt_day_of_week,
    
    -- Time of day category for appt_datetime
    CASE
        WHEN CAST(STRFTIME('%H', pa.appt_datetime) AS INTEGER) BETWEEN 6 AND 11 THEN 'Morning'
        WHEN CAST(STRFTIME('%H', pa.appt_datetime) AS INTEGER) BETWEEN 12 AND 17 THEN 'Afternoon'
        ELSE 'Evening'
    END AS appt_time_of_day_category
FROM
    PriorAppts pa
ORDER BY
    pa.patient_id, pa.appt_datetime;
"""

appointment_features_df = pd.read_sql_query(sql_query, conn)
conn.close()

print(f"Generated {len(appointment_features_df)} records with SQL features.")
print("--- SQL Feature Engineering Complete ---")

# --- 3. Pandas Feature Engineering & Binary Target Creation ---

print("\n--- Pandas Feature Engineering & Data Preparation ---")

# Handle NaN values. SQL query should handle most NaNs for prior counts/rate and days_since.
# `scheduled_days_in_advance` could theoretically be < 0 if appt_datetime is before scheduled_datetime,
# though generation aims to prevent this. Let's ensure it's non-negative.
appointment_features_df['scheduled_days_in_advance'] = appointment_features_df['scheduled_days_in_advance'].clip(lower=0)

# `days_since_last_patient_appt` should be days from signup_date for first appts due to SQL LAG default.
# If any NaN values remain (e.g., if signup_date was also NULL or issue, which it shouldn't be here):
appointment_features_df['days_since_last_patient_appt'] = appointment_features_df['days_since_last_patient_appt'].fillna(9999) # Large sentinel value

# Convert datetime columns
appointment_features_df['appt_datetime'] = pd.to_datetime(appointment_features_df['appt_datetime'])
appointment_features_df['signup_date'] = pd.to_datetime(appointment_features_df['signup_date'])

# Create `is_first_appointment` flag
appointment_features_df['is_first_appointment'] = (appointment_features_df['patient_prior_appts_count'] == 0).astype(int)

# Define features (X) and target (y)
numerical_features = [
    'age',
    'num_existing_conditions',
    'scheduled_days_in_advance',
    'patient_prior_appts_count',
    'patient_prior_no_shows_count',
    'patient_prior_no_show_rate',
    'days_since_last_patient_appt',
    'is_first_appointment'
]
categorical_features = [
    'gender',
    'insurance_type',
    'clinic_location',
    'specialty',
    'appt_day_of_week',
    'appt_time_of_day_category'
]

X = appointment_features_df[numerical_features + categorical_features]
y = appointment_features_df['was_no_show']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
print("--- Pandas Feature Engineering & Data Preparation Complete ---")

# --- 4. Data Visualization ---

print("\n--- Generating Data Visualizations ---")

plt.figure(figsize=(14, 6))

# Plot 1: Violin plot for scheduled_days_in_advance vs. was_no_show
plt.subplot(1, 2, 1)
sns.violinplot(x='was_no_show', y='scheduled_days_in_advance', data=appointment_features_df, palette='viridis')
plt.title('Distribution of Scheduled Days in Advance by No-Show Status')
plt.xlabel('Was No-Show (0=No, 1=Yes)')
plt.ylabel('Scheduled Days In Advance')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Plot 2: Stacked bar chart for specialty vs. was_no_show proportion
plt.subplot(1, 2, 2)
specialty_no_show_proportions = appointment_features_df.groupby('specialty')['was_no_show'].value_counts(normalize=True).unstack().fillna(0)
specialty_no_show_proportions.plot(kind='bar', stacked=True, ax=plt.gca(), cmap='coolwarm')
plt.title('Proportion of No-Shows by Specialty')
plt.xlabel('Specialty')
plt.ylabel('Proportion')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Was No-Show', labels=['Show', 'No-Show'])
plt.tight_layout()
plt.show()

print("--- Data Visualizations Complete ---")

# --- 5. ML Pipeline & Evaluation ---

print("\n--- Building ML Pipeline and Evaluation ---")

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
    remainder='passthrough' # Keep other columns not explicitly transformed
)

# Create the full pipeline
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', HistGradientBoostingClassifier(random_state=42))
])

# Train the model
print("Training the model...")
model_pipeline.fit(X_train, y_train)
print("Model training complete.")

# Predict probabilities on the test set
y_pred_proba = model_pipeline.predict_proba(X_test)[:, 1]
y_pred = model_pipeline.predict(X_test)

# Evaluate the model
roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"\nROC AUC Score on Test Set: {roc_auc:.4f}")

print("\nClassification Report on Test Set:")
print(classification_report(y_test, y_pred))

print("--- ML Pipeline and Evaluation Complete ---")