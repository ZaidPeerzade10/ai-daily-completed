import pandas as pd
import numpy as np
import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report

# --- 1. Synthetic Data Generation ---

# Set a seed for reproducibility
np.random.seed(42)

def generate_random_dates(start_date, end_date, n):
    """Generates an array of random dates between start_date and end_date."""
    start_ts = datetime.datetime.timestamp(start_date)
    end_ts = datetime.datetime.timestamp(end_date)
    random_timestamps = np.random.uniform(start_ts, end_ts, n)
    return pd.to_datetime(random_timestamps, unit='s').date

def create_courses_df(num_courses=150):
    """Generates the courses DataFrame."""
    course_ids = np.arange(1, num_courses + 1)
    release_dates = generate_random_dates(
        datetime.date(2021, 1, 1), datetime.date(2024, 1, 1), num_courses
    )
    categories = np.random.choice(
        ['Programming', 'Data Science', 'Marketing', 'Design', 'Business', 'Photography'],
        num_courses,
        p=[0.3, 0.25, 0.15, 0.1, 0.1, 0.1]
    )
    instructor_experience_years = np.random.randint(1, 21, num_courses)
    difficulty_levels = np.random.choice(
        ['Beginner', 'Intermediate', 'Advanced'],
        num_courses,
        p=[0.4, 0.4, 0.2]
    )
    prices = np.random.uniform(20, 500, num_courses).round(2)

    # Simulate inherent popularity: higher factor means more enrollments
    popularity_factors = np.random.beta(a=2, b=5, size=num_courses) * 10 + 1 # Skewed towards lower popularity initially

    courses_df = pd.DataFrame({
        'course_id': course_ids,
        'release_date': release_dates,
        'category': categories,
        'instructor_experience_years': instructor_experience_years,
        'difficulty_level': difficulty_levels,
        'price': prices,
        'popularity_factor': popularity_factors # For internal simulation
    })
    return courses_df

def create_users_df(num_users=600):
    """Generates the users DataFrame."""
    user_ids = np.arange(1, num_users + 1)
    signup_dates = generate_random_dates(
        datetime.date(2019, 1, 1), datetime.date(2024, 1, 1), num_users
    )
    regions = np.random.choice(['North', 'South', 'East', 'West', 'Central'], num_users)

    users_df = pd.DataFrame({
        'user_id': user_ids,
        'signup_date': signup_dates,
        'region': regions
    })
    return users_df

def create_enrollments_df(courses_df, users_df, num_enrollments=12000):
    """Generates the enrollments DataFrame with realistic patterns."""
    enrollment_data = []

    # Prepare data for sampling
    course_ids = courses_df['course_id'].values
    user_ids = users_df['user_id'].values
    release_dates_map = courses_df.set_index('course_id')['release_date'].to_dict()
    signup_dates_map = users_df.set_index('user_id')['signup_date'].to_dict()
    popularity_factors_map = courses_df.set_index('course_id')['popularity_factor'].to_dict()
    difficulty_levels_map = courses_df.set_index('course_id')['difficulty_level'].to_dict()
    instructor_exp_map = courses_df.set_index('course_id')['instructor_experience_years'].to_dict()
    category_map = courses_df.set_index('course_id')['category'].to_dict()

    # Bias course selection for enrollments based on popularity, experience, and category
    course_weights = (
        courses_df['popularity_factor'] *
        (courses_df['instructor_experience_years'] / 20) *
        courses_df['category'].map({'Programming': 1.5, 'Data Science': 1.4, 'Marketing': 1.1, 'Design': 1.0, 'Business': 1.0, 'Photography': 0.9})
    )
    course_weights = course_weights / course_weights.sum()

    for i in range(num_enrollments):
        user_id = np.random.choice(user_ids)
        course_id = np.random.choice(course_ids, p=course_weights)

        user_signup_date = signup_dates_map[user_id]
        course_release_date = release_dates_map[course_id]

        # Ensure enrollment_date is after both signup_date and release_date
        min_enrollment_date = max(user_signup_date, course_release_date)
        if min_enrollment_date > datetime.date.today(): # Course released too recently or user signed up in future
            continue # Skip this enrollment if it's impossible

        # Random date between min_enrollment_date and today (or just past 30 days of release for early enrollments focus)
        # To ensure we have early enrollments, we bias enrollment dates towards earlier dates
        time_since_min_enrollment = (datetime.date.today() - min_enrollment_date).days
        if time_since_min_enrollment <= 0:
            continue # Again, skip if min date is today or in future

        random_offset_days = np.random.gamma(shape=2, scale=time_since_min_enrollment/5) # Skew towards earlier dates
        enrollment_date = min_enrollment_date + datetime.timedelta(days=min(int(random_offset_days), time_since_min_enrollment))


        # Simulate completion percentage and time spent hours
        difficulty = difficulty_levels_map[course_id]
        base_completion = np.random.uniform(30, 90) # Most enrollments have some progress
        base_time_spent = np.random.uniform(5, 100)

        if difficulty == 'Beginner':
            completion_modifier = np.random.uniform(0.9, 1.1)
            time_modifier = np.random.uniform(0.8, 1.0)
        elif difficulty == 'Intermediate':
            completion_modifier = np.random.uniform(0.8, 1.0)
            time_modifier = np.random.uniform(0.9, 1.1)
        else: # Advanced
            completion_modifier = np.random.uniform(0.6, 0.9) # Harder courses, lower completion for some
            time_modifier = np.random.uniform(1.0, 1.5) # But those who commit, spend more time

        completion_percentage = int(np.clip(base_completion * completion_modifier + np.random.randint(-10, 10), 0, 100))
        # time_spent_hours correlates with completion
        time_spent_hours = np.clip(
            (base_time_spent * time_modifier * (completion_percentage / 100)**1.5) + np.random.uniform(0, 20), # Exponential growth with completion
            0.5, 200
        ).round(2)


        enrollment_data.append({
            'enrollment_id': i + 1,
            'user_id': user_id,
            'course_id': course_id,
            'enrollment_date': enrollment_date,
            'completion_percentage': completion_percentage,
            'time_spent_hours': time_spent_hours
        })

    enrollments_df = pd.DataFrame(enrollment_data)

    # --- CRITICAL ADJUSTMENT for Low_LTV tier generation ---
    # Ensure some courses have 0 enrollments
    if not enrollments_df.empty:
        low_ltv_course_count = max(1, int(len(courses_df) * 0.05)) # ~5% of courses
        # Select courses with fewer existing enrollments to become Low_LTV
        course_enroll_counts = enrollments_df['course_id'].value_counts()
        courses_to_make_low_ltv = course_enroll_counts.nsmallest(low_ltv_course_count).index.tolist()
        # If there are courses with no enrollments yet, include some of them
        courses_without_enrollments = list(set(courses_df['course_id']) - set(course_enroll_counts.index))
        if len(courses_to_make_low_ltv) < low_ltv_course_count and courses_without_enrollments:
            courses_to_make_low_ltv.extend(np.random.choice(courses_without_enrollments, min(low_ltv_course_count - len(courses_to_make_low_ltv), len(courses_without_enrollments)), replace=False).tolist())
        courses_to_make_low_ltv = list(set(courses_to_make_low_ltv)) # Remove duplicates

        enrollments_df = enrollments_df[~enrollments_df['course_id'].isin(courses_to_make_low_ltv)]

    # Sort as requested
    enrollments_df = enrollments_df.sort_values(by=['course_id', 'enrollment_date']).reset_index(drop=True)

    return enrollments_df

print("--- Generating Synthetic Data ---")
courses_df = create_courses_df(num_courses=np.random.randint(100, 201))
users_df = create_users_df(num_users=np.random.randint(500, 701))
enrollments_df = create_enrollments_df(
    courses_df, users_df, num_enrollments=np.random.randint(10000, 15001)
)

print(f"Courses generated: {len(courses_df)}")
print(f"Users generated: {len(users_df)}")
print(f"Enrollments generated: {len(enrollments_df)}")

# --- 2. SQL-like Early Performance Metric Aggregation (First 30 Days) ---

print("\n--- Aggregating Early Performance Metrics (First 30 Days) ---")

# Ensure release_date is datetime for timedelta operations
courses_df['release_date'] = pd.to_datetime(courses_df['release_date'])
enrollments_df['enrollment_date'] = pd.to_datetime(enrollments_df['enrollment_date'])

# Calculate initial_popularity_cutoff_date for each course
courses_df['initial_popularity_cutoff_date'] = courses_df['release_date'] + pd.Timedelta(days=30)

# Merge courses with enrollments to filter
merged_early_df = pd.merge(
    enrollments_df,
    courses_df[['course_id', 'release_date', 'initial_popularity_cutoff_date']],
    on='course_id',
    how='left'
)

# Filter for enrollments within the first 30 days
early_enrollments_filtered = merged_early_df[
    (merged_early_df['enrollment_date'] >= merged_early_df['release_date']) &
    (merged_early_df['enrollment_date'] <= merged_early_df['initial_popularity_cutoff_date'])
].copy() # Use .copy() to avoid SettingWithCopyWarning

# Aggregate early metrics
early_metrics = early_enrollments_filtered.groupby('course_id').agg(
    num_enrollments_30d=('enrollment_id', 'count'),
    num_unique_users_30d=('user_id', lambda x: x.nunique()),
    avg_completion_30d=('completion_percentage', 'mean'),
    avg_time_spent_30d=('time_spent_hours', 'mean'),
    days_to_first_enrollment=('enrollment_date', lambda x: (x.min() - courses_df.loc[x.index[0], 'release_date']).days)
).reset_index()

# Merge early metrics back to courses_df
# Use .copy() to ensure we're working on a new DataFrame
ml_df = courses_df.drop(columns=['popularity_factor', 'initial_popularity_cutoff_date']).copy()
ml_df = pd.merge(ml_df, early_metrics, on='course_id', how='left')

# Fill NaN values for courses with no early enrollments
ml_df['num_enrollments_30d'] = ml_df['num_enrollments_30d'].fillna(0).astype(int)
ml_df['num_unique_users_30d'] = ml_df['num_unique_users_30d'].fillna(0).astype(int)
ml_df['avg_completion_30d'] = ml_df['avg_completion_30d'].fillna(0.0)
ml_df['avg_time_spent_30d'] = ml_df['avg_time_spent_30d'].fillna(0.0)
# If no enrollments in 30 days, days_to_first_enrollment is considered 31 (i.e., outside the window)
ml_df['days_to_first_enrollment'] = ml_df['days_to_first_enrollment'].fillna(31).astype(int)


print("Early metrics aggregated for courses. Example:")
print(ml_df[['course_id', 'release_date', 'num_enrollments_30d', 'days_to_first_enrollment', 'avg_completion_30d']].head())


# --- 3. Pandas-based Target Variable and Additional Feature Engineering ---

print("\n--- Creating Popularity Tier Target Variable ---")

# Calculate total_enrollments_all_time for each course
total_enrollments_all_time = enrollments_df.groupby('course_id')['enrollment_id'].count().reset_index(name='total_enrollments_all_time')

# Merge total enrollments into the main DataFrame
ml_df = pd.merge(ml_df, total_enrollments_all_time, on='course_id', how='left')
ml_df['total_enrollments_all_time'] = ml_df['total_enrollments_all_time'].fillna(0).astype(int)

# Define popularity tiers based on non-zero total_enrollments_all_time
non_zero_enrollments = ml_df[ml_df['total_enrollments_all_time'] > 0]['total_enrollments_all_time']

# Check if there are enough non-zero enrollments to calculate percentiles
if len(non_zero_enrollments) >= 3: # Need at least 3 points for 3 percentiles
    low_threshold, high_threshold = non_zero_enrollments.quantile([0.33, 0.66])
else: # Fallback for very small datasets or if most courses have 0 enrollments
    low_threshold = 1
    high_threshold = 2
    if len(non_zero_enrollments) > 0:
        low_threshold = non_zero_enrollments.min()
        high_threshold = non_zero_enrollments.max() if len(non_zero_enrollments) > 1 else non_zero_enrollments.min()

print(f"Popularity thresholds (non-zero enrollments): 33rd percentile={low_threshold:.0f}, 66th percentile={high_threshold:.0f}")

conditions = [
    (ml_df['total_enrollments_all_time'] == 0),
    (ml_df['total_enrollments_all_time'] > 0) & (ml_df['total_enrollments_all_time'] <= low_threshold),
    (ml_df['total_enrollments_all_time'] > low_threshold) & (ml_df['total_enrollments_all_time'] <= high_threshold),
    (ml_df['total_enrollments_all_time'] > high_threshold)
]
choices = ['Low_LTV', 'Medium', 'High', 'Very_High']
ml_df['popularity_tier'] = np.select(conditions, choices, default='Unknown')

print("\nPopularity tier distribution:")
print(ml_df['popularity_tier'].value_counts())

# Drop the temporary 'total_enrollments_all_time' column and 'release_date' as it's not a direct feature
ml_df = ml_df.drop(columns=['total_enrollments_all_time', 'release_date'])

print("\nFinal DataFrame structure for ML:")
print(ml_df.head())
print(ml_df.info())


# --- 4. Machine Learning Model Training and Evaluation ---

print("\n--- Training Machine Learning Model ---")

# Separate features (X) and target (y)
X = ml_df.drop(columns=['course_id', 'popularity_tier'])
y = ml_df['popularity_tier']

# Define categorical and numerical features
categorical_features = ['category', 'difficulty_level']
numerical_features = [col for col in X.columns if col not in categorical_features]

# Create preprocessor using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Create a pipeline with preprocessing and classifier
# Use HistGradientBoostingClassifier as it handles various feature types well
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', HistGradientBoostingClassifier(random_state=42))
])

# Split data into training and testing sets, stratifying by popularity_tier
# Check if there's at least one sample per class in y for stratification
if y.nunique() > 1 and all(y.value_counts() > 1):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
else:
    print("Warning: Cannot stratify due to insufficient samples per class. Splitting without stratification.")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Train the model
model_pipeline.fit(X_train, y_train)

# Make predictions
y_pred = model_pipeline.predict(X_test)

# Evaluate the model
print("\n--- Model Evaluation ---")
print("Classification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

print("\nPipeline execution complete.")