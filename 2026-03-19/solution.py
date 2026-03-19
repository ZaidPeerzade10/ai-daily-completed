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

# Ensure reproducibility
np.random.seed(42)

# --- 1. Generate Synthetic Data (Pandas/Numpy) ---

# Student data
num_students = np.random.randint(500, 700)
student_ids = np.arange(1000, 1000 + num_students)
enrollment_start_date = datetime.date(2021, 1, 1)
enrollment_end_date = datetime.date(2023, 12, 31)
date_range = (enrollment_end_date - enrollment_start_date).days
enrollment_dates = [enrollment_start_date + datetime.timedelta(days=int(d)) for d in np.random.randint(0, date_range, num_students)]
majors = ['Computer Science', 'Mathematics', 'Biology', 'History', 'Art', 'Engineering', 'Psychology']
academic_levels = ['Freshman', 'Sophomore', 'Junior', 'Senior']
prior_gpas = np.random.uniform(2.0, 4.0, num_students)

students_df = pd.DataFrame({
    'student_id': student_ids,
    'enrollment_date': enrollment_dates,
    'major': np.random.choice(majors, num_students),
    'academic_level': np.random.choice(academic_levels, num_students),
    'prior_gpa': prior_gpas
})

# Simulate early dropout status for initial biasing
# Higher dropout risk for certain majors and lower GPA
dropout_prone_majors = ['History', 'Art']
dropout_prone_levels = ['Freshman']
dropout_base_rate = 0.1 # 10% base dropout rate

students_df['will_dropout_early_simulated'] = 0
for i, row in students_df.iterrows():
    dropout_prob = dropout_base_rate
    if row['major'] in dropout_prone_majors:
        dropout_prob += 0.1
    if row['academic_level'] in dropout_prone_levels:
        dropout_prob += 0.05
    if row['prior_gpa'] < 2.5:
        dropout_prob += 0.15
    elif row['prior_gpa'] < 3.0:
        dropout_prob += 0.05
    
    if np.random.rand() < dropout_prob:
        students_df.loc[i, 'will_dropout_early_simulated'] = 1

# Dropout events
num_potential_dropouts = students_df['will_dropout_early_simulated'].sum()
num_dropouts = int(num_potential_dropouts * np.random.uniform(0.7, 1.0)) # A portion of simulated dropouts actually drop out
dropout_students = students_df[students_df['will_dropout_early_simulated'] == 1].sample(n=num_dropouts, replace=False)

dropout_events_data = []
dropout_id_counter = 1
for _, student_row in dropout_students.iterrows():
    enrollment_date = student_row['enrollment_date']
    # Dropout within 60 days for 'early dropout' target
    dropout_date_offset = np.random.randint(15, 60) # Simulate dropout within this window
    dropout_date = enrollment_date + datetime.timedelta(days=dropout_date_offset)
    dropout_events_data.append({
        'dropout_id': dropout_id_counter,
        'student_id': student_row['student_id'],
        'dropout_date': dropout_date
    })
    dropout_id_counter += 1

dropout_events_df = pd.DataFrame(dropout_events_data)


# LMS Activities
num_activities_total = np.random.randint(5000, 8000)
activity_types = ['Login', 'Module_View', 'Assignment_Accessed', 'Quiz_Started', 'Forum_Post', 'Video_Watched']
lms_activities_data = []
activity_id_counter = 1

# Merge dropout status for biasing activity generation
students_with_dropout_status = students_df.merge(
    dropout_events_df[['student_id', 'dropout_date']], 
    on='student_id', 
    how='left'
)
# Define an early dropout cutoff: 60 days after enrollment for the target variable (not for activity generation)
students_with_dropout_status['is_early_dropout_for_bias'] = (
    students_with_dropout_status['dropout_date'].notna() &
    (students_with_dropout_status['dropout_date'] - students_with_dropout_status['enrollment_date']).dt.days <= 60
)

for student_id in students_with_dropout_status['student_id'].unique():
    student_row = students_with_dropout_status[students_with_dropout_status['student_id'] == student_id].iloc[0]
    enrollment_date = student_row['enrollment_date']
    is_early_dropout = student_row['is_early_dropout_for_bias']
    
    # Base number of activities for a student
    num_student_activities = np.random.randint(5, 20) 
    if is_early_dropout:
        num_student_activities = np.random.randint(2, 10) # Fewer activities for simulated dropouts
    
    # Simulate activity distribution and duration based on dropout status
    for _ in range(num_student_activities):
        activity_offset = np.random.randint(1, 31) # Within first 30 days of enrollment
        activity_date = enrollment_date + datetime.timedelta(days=activity_offset)
        
        chosen_activity_type = np.random.choice(activity_types)
        duration = 0.0

        if chosen_activity_type == 'Login':
            duration = np.random.uniform(1.0, 5.0)
        elif chosen_activity_type in ['Module_View', 'Video_Watched']:
            duration = np.random.uniform(5.0, 30.0)
        elif chosen_activity_type in ['Assignment_Accessed', 'Quiz_Started']:
            duration = np.random.uniform(2.0, 15.0)
        elif chosen_activity_type == 'Forum_Post':
            duration = np.random.uniform(3.0, 10.0)
        
        # Bias for dropouts: lower duration for 'valuable' activities, less likely to have them
        if is_early_dropout:
            duration *= np.random.uniform(0.2, 0.8) # Reduce duration
            if chosen_activity_type in ['Assignment_Accessed', 'Quiz_Started', 'Forum_Post'] and np.random.rand() < 0.5:
                # Reduce chances of these 'valuable' activities for dropouts
                chosen_activity_type = np.random.choice(['Login', 'Module_View'])
                
        lms_activities_data.append({
            'activity_id': activity_id_counter,
            'student_id': student_id,
            'activity_date': activity_date,
            'activity_type': chosen_activity_type,
            'duration_minutes': max(0.5, duration) # Ensure minimum duration
        })
        activity_id_counter += 1

lms_activities_df = pd.DataFrame(lms_activities_data)

# Sort lms_activities_df as requested
lms_activities_df = lms_activities_df.sort_values(by=['student_id', 'activity_date']).reset_index(drop=True)

# Convert dates to datetime objects for consistency
students_df['enrollment_date'] = pd.to_datetime(students_df['enrollment_date'])
lms_activities_df['activity_date'] = pd.to_datetime(lms_activities_df['activity_date'])
dropout_events_df['dropout_date'] = pd.to_datetime(dropout_events_df['dropout_date'])

print("--- Synthetic Data Generated ---")
print(f"Students DataFrame Shape: {students_df.shape}")
print(f"LMS Activities DataFrame Shape: {lms_activities_df.shape}")
print(f"Dropout Events DataFrame Shape: {dropout_events_df.shape}")
print("\nStudents Sample:")
print(students_df.head())
print("\nLMS Activities Sample:")
print(lms_activities_df.head())
print("\nDropout Events Sample:")
print(dropout_events_df.head())


# --- 2. Load into SQLite & SQL Feature Engineering ---
conn = sqlite3.connect(':memory:')

# Load DataFrames into SQLite
students_df.to_sql('students', conn, index=False, if_exists='replace')
lms_activities_df.to_sql('lms_activities', conn, index=False, if_exists='replace')

# SQL Query for early engagement features (first 14 days)
# Corrected WHERE clause placement as per feedback: filtering moved to ON clause of LEFT JOIN
sql_query = """
SELECT
    s.student_id,
    s.enrollment_date,
    s.major,
    s.academic_level,
    s.prior_gpa,
    COALESCE(SUM(CASE WHEN a.activity_type = 'Login' THEN 1 ELSE 0 END), 0) AS num_logins_first_14d,
    COALESCE(SUM(CASE WHEN a.activity_type = 'Assignment_Accessed' THEN 1 ELSE 0 END), 0) AS num_assignment_views_first_14d,
    COALESCE(SUM(a.duration_minutes), 0.0) AS total_activity_duration_first_14d,
    COALESCE(COUNT(DISTINCT a.activity_date), 0) AS days_with_activity_first_14d,
    COALESCE(AVG(a.duration_minutes), 0.0) AS avg_activity_duration_first_14d,
    COALESCE(MAX(CASE WHEN a.activity_type = 'Forum_Post' THEN 1 ELSE 0 END), 0) AS has_posted_to_forum_first_14d,
    (MIN(julianday(a.activity_date)) - julianday(s.enrollment_date)) AS days_since_first_activity_first_14d
FROM
    students s
LEFT JOIN
    lms_activities a ON s.student_id = a.student_id
    AND julianday(a.activity_date) - julianday(s.enrollment_date) >= 0
    AND julianday(a.activity_date) - julianday(s.enrollment_date) <= 14
GROUP BY
    s.student_id, s.enrollment_date, s.major, s.academic_level, s.prior_gpa
ORDER BY
    s.student_id;
"""

student_engagement_features_df = pd.read_sql_query(sql_query, conn)
conn.close()

print("\n--- SQL Feature Engineering Complete ---")
print(f"Student Engagement Features DataFrame Shape: {student_engagement_features_df.shape}")
print("Student Engagement Features Sample:")
print(student_engagement_features_df.head())


# --- 3. Pandas Feature Engineering & Binary Target Creation ---

# Handle NaN values from SQL aggregation
fill_0_cols = [
    'num_logins_first_14d', 'num_assignment_views_first_14d',
    'total_activity_duration_first_14d', 'days_with_activity_first_14d',
    'has_posted_to_forum_first_14d'
]
student_engagement_features_df[fill_0_cols] = student_engagement_features_df[fill_0_cols].fillna(0)
student_engagement_features_df['avg_activity_duration_first_14d'] = student_engagement_features_df['avg_activity_duration_first_14d'].fillna(0.0)
student_engagement_features_df['days_since_first_activity_first_14d'] = student_engagement_features_df['days_since_first_activity_first_14d'].fillna(14.0)

# Convert enrollment_date to datetime objects
student_engagement_features_df['enrollment_date'] = pd.to_datetime(student_engagement_features_df['enrollment_date'])

# Calculate derived features
student_engagement_features_df['activity_frequency_first_14d'] = student_engagement_features_df['days_with_activity_first_14d'] / 14.0
student_engagement_features_df['activity_frequency_first_14d'] = student_engagement_features_df['activity_frequency_first_14d'].fillna(0.0)

# Simplified engagement_ratio_first_14d as per feedback
student_engagement_features_df['engagement_ratio_first_14d'] = (
    student_engagement_features_df['num_assignment_views_first_14d'] + 
    student_engagement_features_df['num_logins_first_14d']
) / (student_engagement_features_df['num_logins_first_14d'] + 1)
# No NaNs expected here due to +1, but fill for robustness.
student_engagement_features_df['engagement_ratio_first_14d'] = student_engagement_features_df['engagement_ratio_first_14d'].fillna(0.0)

# Create the Binary Target `will_dropout_early`
# Merge dropout events
df_merged_target = student_engagement_features_df.merge(
    dropout_events_df[['student_id', 'dropout_date']],
    on='student_id',
    how='left'
)

# Define early dropout within 60 days of enrollment
df_merged_target['will_dropout_early'] = 0
early_dropout_condition = (
    df_merged_target['dropout_date'].notna() &
    (df_merged_target['dropout_date'] >= df_merged_target['enrollment_date']) &
    (df_merged_target['dropout_date'] <= df_merged_target['enrollment_date'] + pd.Timedelta(days=60))
)
df_merged_target.loc[early_dropout_condition, 'will_dropout_early'] = 1

# Drop the temporary dropout_date column from the merged dataframe
df_final = df_merged_target.drop(columns=['dropout_date'])

print("\n--- Pandas Feature Engineering & Target Creation Complete ---")
print(f"Final DataFrame Shape: {df_final.shape}")
print("Final DataFrame Sample:")
print(df_final.head())
print("\nTarget Variable Distribution:")
print(df_final['will_dropout_early'].value_counts(normalize=True))

# Define features (X) and target (y)
numerical_features = [
    'prior_gpa',
    'num_logins_first_14d',
    'num_assignment_views_first_14d',
    'total_activity_duration_first_14d',
    'days_with_activity_first_14d',
    'avg_activity_duration_first_14d',
    'days_since_first_activity_first_14d',
    'activity_frequency_first_14d',
    'engagement_ratio_first_14d'
]
categorical_features = ['major', 'academic_level', 'has_posted_to_forum_first_14d'] # has_posted_to_forum_first_14d is binary but can be treated as categorical by OneHotEncoder

X = df_final[numerical_features + categorical_features]
y = df_final['will_dropout_early']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

print(f"\nTraining set shape: {X_train.shape}, Test set shape: {X_test.shape}")
print(f"Training target distribution:\n{y_train.value_counts(normalize=True)}")
print(f"Test target distribution:\n{y_test.value_counts(normalize=True)}")


# --- 4. Data Visualization ---
print("\n--- Generating Visualizations ---")
plt.figure(figsize=(14, 6))

# Violin plot for total_activity_duration_first_14d vs. will_dropout_early
plt.subplot(1, 2, 1)
sns.violinplot(x='will_dropout_early', y='total_activity_duration_first_14d', data=df_final)
plt.title('Total Activity Duration (First 14 Days) by Dropout Status')
plt.xlabel('Will Dropout Early (0=No, 1=Yes)')
plt.ylabel('Total Activity Duration (Minutes)')

# Stacked bar chart for proportion of will_dropout_early across different major values
plt.subplot(1, 2, 2)
major_dropout_proportion = df_final.groupby('major')['will_dropout_early'].value_counts(normalize=True).unstack().fillna(0)
major_dropout_proportion.plot(kind='bar', stacked=True, ax=plt.gca(), cmap='viridis')
plt.title('Proportion of Early Dropout by Major')
plt.xlabel('Major')
plt.ylabel('Proportion')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Will Dropout Early', labels=['No', 'Yes'])

plt.tight_layout()
plt.show()


# --- 5. ML Pipeline & Evaluation ---
print("\n--- Building and Evaluating ML Pipeline ---")

# Preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ]), numerical_features),
        ('cat', Pipeline([
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ]), categorical_features)
    ])

# Create the full pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', HistGradientBoostingClassifier(random_state=42))
])

# Train the pipeline
pipeline.fit(X_train, y_train)

# Predict probabilities for the positive class on the test set
y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
# For classification report, we need hard predictions
y_pred = pipeline.predict(X_test) 

# Evaluate the model
roc_auc = roc_auc_score(y_test, y_pred_proba)
class_report = classification_report(y_test, y_pred)

print(f"\nROC AUC Score on Test Set: {roc_auc:.4f}")
print("\nClassification Report on Test Set:")
print(class_report)

print("\n--- Script Finished Successfully ---")