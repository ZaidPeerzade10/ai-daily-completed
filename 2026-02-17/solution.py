import pandas as pd
import numpy as np
import datetime
import sqlite3
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer # CORRECTED IMPORT PATH
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, classification_report
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# 1. Generate Synthetic Data (Pandas/Numpy)
print("1. Generating Synthetic Data...")

# Seed for reproducibility
np.random.seed(42)

# --- students_df ---
num_students = np.random.randint(500, 701)
student_ids = np.arange(1, num_students + 1)
signup_dates = pd.to_datetime('today') - pd.to_timedelta(np.random.randint(0, 3 * 365, size=num_students), unit='days')
education_levels = np.random.choice(['High School', 'Undergrad', 'Postgrad'], size=num_students, p=[0.25, 0.5, 0.25])
countries = np.random.choice(['USA', 'Canada', 'UK', 'India', 'Australia'], size=num_students)

students_df = pd.DataFrame({
    'student_id': student_ids,
    'signup_date': signup_dates,
    'education_level': education_levels,
    'country': countries
})

print(f"  Generated students_df with {len(students_df)} rows.")

# --- courses_df ---
num_courses = np.random.randint(50, 101)
course_ids = np.arange(1001, 1001 + num_courses)
course_names = [f"Course {i} {np.random.choice(['Fundamentals', 'Advanced', 'Intro to', 'Mastering'])}" for i in range(num_courses)]
difficulties = np.random.choice(['Beginner', 'Intermediate', 'Advanced'], size=num_courses, p=[0.4, 0.4, 0.2])
categories = np.random.choice(['Programming', 'Data Science', 'Marketing', 'Design', 'Business'], size=num_courses)
expected_duration_days = np.random.randint(30, 181, size=num_courses)

courses_df = pd.DataFrame({
    'course_id': course_ids,
    'course_name': course_names,
    'difficulty': difficulties,
    'category': categories,
    'expected_duration_days': expected_duration_days
})
print(f"  Generated courses_df with {len(courses_df)} rows.")

# --- enrollments_df ---
num_enrollments = np.random.randint(800, 1201)
enrollment_ids = np.arange(20001, 20001 + num_enrollments)

# Sample student and course IDs
sampled_student_ids = np.random.choice(students_df['student_id'], size=num_enrollments)
sampled_course_ids = np.random.choice(courses_df['course_id'], size=num_enrollments)

# Create a temporary df to link signup_date and difficulty for enrollment_date and completion bias
temp_enroll_df = pd.DataFrame({
    'student_id': sampled_student_ids,
    'course_id': sampled_course_ids
}).merge(students_df[['student_id', 'signup_date', 'education_level']], on='student_id', how='left') \
  .merge(courses_df[['course_id', 'difficulty']], on='course_id', how='left')

# Generate enrollment_date (after signup_date)
# Ensure enrollment_date is after signup_date
enrollment_dates = temp_enroll_df['signup_date'] + pd.to_timedelta(np.random.randint(1, 365, size=num_enrollments), unit='days')

# Determine is_completed_course with bias
is_completed_course = np.zeros(num_enrollments, dtype=int)
for i in range(num_enrollments):
    edu_level = temp_enroll_df.loc[i, 'education_level']
    difficulty = temp_enroll_df.loc[i, 'difficulty']

    base_prob_completion = 0.5 # Baseline
    
    # Adjust for education level
    if edu_level == 'High School':
        base_prob_completion -= 0.15
    elif edu_level == 'Undergrad':
        base_prob_completion += 0.05
    elif edu_level == 'Postgrad':
        base_prob_completion += 0.15
        
    # Adjust for difficulty
    if difficulty == 'Beginner':
        base_prob_completion += 0.1
    elif difficulty == 'Intermediate':
        pass
    elif difficulty == 'Advanced':
        base_prob_completion -= 0.15
        
    # Ensure probability is within [0, 1]
    base_prob_completion = max(0.1, min(0.9, base_prob_completion))
    
    if np.random.rand() < base_prob_completion:
        is_completed_course[i] = 1

enrollments_df = pd.DataFrame({
    'enrollment_id': enrollment_ids,
    'student_id': sampled_student_ids,
    'course_id': sampled_course_ids,
    'enrollment_date': enrollment_dates,
    'is_completed_course': is_completed_course
})

# Add course duration to enrollments_df for activity log generation
enrollments_df = enrollments_df.merge(courses_df[['course_id', 'expected_duration_days']], on='course_id', how='left')
print(f"  Generated enrollments_df with {len(enrollments_df)} rows.")

# --- activity_logs_df ---
activity_data = []
activity_type_options = ['lecture_view', 'quiz_attempt', 'assignment_submit', 'forum_post', 'reading']
submission_types = ['quiz_attempt', 'assignment_submit']

# To ensure the total number of activities is within the desired range,
# we'll generate activities per enrollment and then potentially sample from them or augment.
# For now, generate a variable number per enrollment.
total_activities_generated = 0
for _, enrollment in enrollments_df.iterrows():
    enrollment_id = enrollment['enrollment_id']
    enrollment_date = enrollment['enrollment_date']
    is_completed = enrollment['is_completed_course']
    expected_duration = enrollment['expected_duration_days']

    # Determine number of activities based on completion status
    if is_completed == 1:
        n_activities_for_enrollment = np.random.randint(8, 25) # More activities
        activity_duration_factor = np.random.uniform(0.7, 1.1) # Spread over longer duration (up to 110% of expected)
        submission_prob = 0.3 # Higher prob of submission types
    else:
        n_activities_for_enrollment = np.random.randint(1, 10) # Fewer activities
        activity_duration_factor = np.random.uniform(0.1, 0.6) # Stop earlier (10%-60% of expected)
        submission_prob = 0.05 # Lower prob of submission types

    # Calculate actual activity end date offset for this enrollment
    # Ensure it's at least 1 day if activities are to be generated, up to expected duration
    activity_end_offset_days = max(1, min(expected_duration, int(expected_duration * activity_duration_factor)))

    for _ in range(n_activities_for_enrollment):
        activity_date_offset = np.random.randint(0, activity_end_offset_days)
        current_activity_date = enrollment_date + pd.to_timedelta(activity_date_offset, unit='days')

        activity_type = np.random.choice(activity_type_options)
        if np.random.rand() < submission_prob:
            activity_type = np.random.choice(submission_types)

        time_spent = np.random.uniform(5, 90)

        activity_data.append({
            'enrollment_id': enrollment_id,
            'activity_date': current_activity_date,
            'activity_type': activity_type,
            'time_spent_minutes': time_spent
        })
        total_activities_generated += 1

# Create activity_logs_df from generated data
activity_logs_df = pd.DataFrame(activity_data)
# Filter for desired range if too many activities were generated
target_num_activities = np.random.randint(5000, 8001)
if total_activities_generated > target_num_activities:
    activity_logs_df = activity_logs_df.sample(n=target_num_activities, random_state=42).reset_index(drop=True)
elif total_activities_generated < target_num_activities:
    # If not enough, replicate some activities (simple augmentation for demo purposes)
    replication_factor = int(target_num_activities / total_activities_generated)
    remainder = target_num_activities % total_activities_generated
    
    if replication_factor > 0:
        activity_logs_df = pd.concat([activity_logs_df] * replication_factor, ignore_index=True)
    if remainder > 0:
        activity_logs_df = pd.concat([activity_logs_df, activity_logs_df.sample(n=remainder, random_state=42)], ignore_index=True)


activity_logs_df['activity_log_id'] = np.arange(1, len(activity_logs_df) + 1)
activity_logs_df = activity_logs_df[['activity_log_id', 'enrollment_id', 'activity_date', 'activity_type', 'time_spent_minutes']] # Reorder columns
print(f"  Generated activity_logs_df with {len(activity_logs_df)} rows.")

# Clean up temporary duration column from enrollments_df
enrollments_df = enrollments_df.drop(columns=['expected_duration_days'])


# 2. Load into SQLite & SQL Feature Engineering (Early Engagement)
print("\n2. Loading data into SQLite and performing SQL Feature Engineering...")

# Create an in-memory SQLite database
conn = sqlite3.connect(':memory:')

# Load DataFrames into SQLite tables
students_df.to_sql('students', conn, index=False, if_exists='replace')
courses_df.to_sql('courses', conn, index=False, if_exists='replace')
enrollments_df.to_sql('enrollments', conn, index=False, if_exists='replace')
activity_logs_df.to_sql('activity_logs', conn, index=False, if_exists='replace')

# Determine global_analysis_date (using pandas for date arithmetic then converting to string)
global_analysis_date_pd = activity_logs_df['activity_date'].max() + pd.Timedelta(days=60)
global_analysis_date_str = global_analysis_date_pd.strftime('%Y-%m-%d')
print(f"  Global analysis date set to: {global_analysis_date_str}")

early_engagement_window_days = 30
print(f"  Early engagement window: {early_engagement_window_days} days.")

sql_query = f"""
WITH EarlyActivities AS (
    SELECT
        al.enrollment_id,
        al.activity_log_id,
        al.activity_date,
        al.activity_type,
        al.time_spent_minutes,
        e.enrollment_date -- Include enrollment_date for date diff calculation
    FROM
        activity_logs al
    JOIN
        enrollments e ON al.enrollment_id = e.enrollment_id
    WHERE
        al.activity_date BETWEEN e.enrollment_date AND DATE(e.enrollment_date, '+' || {early_engagement_window_days} || ' days')
),
AggregatedEarlyFeatures AS (
    SELECT
        ea.enrollment_id,
        COUNT(ea.activity_log_id) AS early_total_activities,
        SUM(ea.time_spent_minutes) AS early_total_time_spent,
        SUM(CASE WHEN ea.activity_type = 'quiz_attempt' THEN 1 ELSE 0 END) AS early_num_quiz_attempts,
        SUM(CASE WHEN ea.activity_type = 'assignment_submit' THEN 1 ELSE 0 END) AS early_num_assignments_submitted,
        -- Calculate days from enrollment to first activity within the window
        MIN(JULIANDAY(ea.activity_date) - JULIANDAY(ea.enrollment_date)) AS days_from_enroll_to_first_activity,
        -- Calculate early activity frequency, cast count to REAL for float division
        CAST(COUNT(ea.activity_log_id) AS REAL) / {early_engagement_window_days} AS early_activity_frequency
    FROM
        EarlyActivities ea
    GROUP BY
        ea.enrollment_id
)
SELECT
    e.enrollment_id,
    e.student_id,
    e.course_id,
    s.education_level,
    s.country,
    c.difficulty,
    c.category,
    c.expected_duration_days,
    e.enrollment_date,
    e.is_completed_course,
    COALESCE(aef.early_total_activities, 0) AS early_total_activities,
    COALESCE(aef.early_total_time_spent, 0.0) AS early_total_time_spent,
    COALESCE(aef.early_num_quiz_attempts, 0) AS early_num_quiz_attempts,
    COALESCE(aef.early_num_assignments_submitted, 0) AS early_num_assignments_submitted,
    aef.days_from_enroll_to_first_activity, -- NULL if no activities, will be handled by pandas
    COALESCE(aef.early_activity_frequency, 0.0) AS early_activity_frequency
FROM
    enrollments e
LEFT JOIN
    AggregatedEarlyFeatures aef ON e.enrollment_id = aef.enrollment_id
JOIN
    students s ON e.student_id = s.student_id
JOIN
    courses c ON e.course_id = c.course_id;
"""

enrollment_features_df = pd.read_sql_query(sql_query, conn)
conn.close() # Close SQLite connection

print(f"  SQL query executed. Fetched {len(enrollment_features_df)} enrollment features.")
print("  First 5 rows of enrollment_features_df:")
print(enrollment_features_df.head())


# 3. Pandas Feature Engineering & Binary Target Creation
print("\n3. Performing Pandas Feature Engineering and Data Splitting...")

# Handle NaN values from LEFT JOIN (for enrollments with no early activities)
enrollment_features_df['early_total_activities'] = enrollment_features_df['early_total_activities'].fillna(0).astype(int)
enrollment_features_df['early_total_time_spent'] = enrollment_features_df['early_total_time_spent'].fillna(0.0)
enrollment_features_df['early_num_quiz_attempts'] = enrollment_features_df['early_num_quiz_attempts'].fillna(0).astype(int)
enrollment_features_df['early_num_assignments_submitted'] = enrollment_features_df['early_num_assignments_submitted'].fillna(0).astype(int)
enrollment_features_df['early_activity_frequency'] = enrollment_features_df['early_activity_frequency'].fillna(0.0)

# For days_from_enroll_to_first_activity, fill with a large sentinel value
sentinel_days = early_engagement_window_days + 40
enrollment_features_df['days_from_enroll_to_first_activity'] = enrollment_features_df['days_from_enroll_to_first_activity'].fillna(sentinel_days)

# Convert enrollment_date to datetime objects (already handled by pd.read_sql_query if SQLite stores as string)
# Explicit conversion for safety
enrollment_features_df['enrollment_date'] = pd.to_datetime(enrollment_features_df['enrollment_date'])

# Calculate enrollment_age_at_cutoff_days: The number of days between enrollment_date and (enrollment_date + early_engagement_window_days)
# This is a fixed value for all enrollments, equal to the window size.
enrollment_features_df['enrollment_age_at_cutoff_days'] = early_engagement_window_days


# Define features X and target y
X = enrollment_features_df.drop(columns=['enrollment_id', 'student_id', 'course_id', 'enrollment_date', 'is_completed_course'])
y = enrollment_features_df['is_completed_course']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
print(f"  Data split into training ({len(X_train)} samples) and testing ({len(X_test)} samples).")
print(f"  Target variable (is_completed_course) distribution in train: \n{y_train.value_counts(normalize=True)}")
print(f"  Target variable (is_completed_course) distribution in test: \n{y_test.value_counts(normalize=True)}")


# 4. Data Visualization
print("\n4. Generating Data Visualizations...")

plt.figure(figsize=(14, 6))

# Violin plot for early_total_time_spent
plt.subplot(1, 2, 1)
data_for_violin = [
    enrollment_features_df[enrollment_features_df['is_completed_course'] == 0]['early_total_time_spent'],
    enrollment_features_df[enrollment_features_df['is_completed_course'] == 1]['early_total_time_spent']
]
plt.violinplot(data_for_violin, showmeans=True)
plt.xticks([1, 2], ['Not Completed (0)', 'Completed (1)'])
plt.title('Distribution of Early Total Time Spent by Course Completion')
plt.ylabel('Early Total Time Spent (minutes)')
plt.xlabel('Course Completion Status')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Stacked bar chart for difficulty vs. completion
plt.subplot(1, 2, 2)
difficulty_completion_counts = pd.crosstab(enrollment_features_df['difficulty'], enrollment_features_df['is_completed_course'])
difficulty_completion_proportions = difficulty_completion_counts.div(difficulty_completion_counts.sum(axis=1), axis=0)

difficulty_completion_proportions.plot(kind='bar', stacked=True, ax=plt.gca(), cmap='viridis')
plt.title('Proportion of Course Completion by Difficulty Level')
plt.xlabel('Course Difficulty')
plt.ylabel('Proportion')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Completed', labels=['No', 'Yes'])
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()
print("  Plots displayed: Violin plot for early time spent, Stacked bar for difficulty vs. completion.")


# 5. ML Pipeline & Evaluation (Binary Classification)
print("\n5. Building ML Pipeline, Training, and Evaluating...")

# Identify numerical and categorical features
numerical_features = [
    'expected_duration_days',
    'enrollment_age_at_cutoff_days',
    'early_total_activities',
    'early_total_time_spent',
    'early_num_quiz_attempts',
    'early_num_assignments_submitted',
    'days_from_enroll_to_first_activity',
    'early_activity_frequency'
]

categorical_features = [
    'education_level',
    'country',
    'difficulty',
    'category'
]

# Create preprocessing pipelines for numerical and categorical features
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Create a column transformer to apply different transformations to different columns
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='passthrough' # Keep other columns if any, though none are expected here
)

# Create the full pipeline with preprocessing and the classifier
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', HistGradientBoostingClassifier(random_state=42))
])

# Train the pipeline
print("  Training the HistGradientBoostingClassifier model...")
model_pipeline.fit(X_train, y_train)
print("  Model training complete.")

# Predict probabilities on the test set for ROC AUC score
y_pred_proba = model_pipeline.predict_proba(X_test)[:, 1]

# Predict class labels on the test set for classification report
y_pred = model_pipeline.predict(X_test)

# Calculate and print evaluation metrics
roc_auc = roc_auc_score(y_test, y_pred_proba)
class_report = classification_report(y_test, y_pred)

print(f"\n--- Model Evaluation on Test Set ---")
print(f"ROC AUC Score: {roc_auc:.4f}")
print("\nClassification Report:")
print(class_report)

print("\nScript execution complete.")