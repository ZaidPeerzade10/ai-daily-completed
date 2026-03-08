import pandas as pd
import numpy as np
import datetime
import random
import sqlite3
import re
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')

# --- 1. Generate Synthetic Data ---

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# 1.1 developers_df
num_developers = random.randint(500, 700)
dev_ids = np.arange(1, num_developers + 1)
teams = ['Frontend', 'Backend', 'Mobile', 'DevOps', 'QA', 'UI/UX']
developers_df = pd.DataFrame({
    'dev_id': dev_ids,
    'team': np.random.choice(teams, num_developers),
    'experience_years': np.random.randint(1, 16, num_developers),
    'avg_bugs_resolved_per_month': np.round(np.random.uniform(1.0, 15.0, num_developers), 1)
})

# 1.2 bug_reports_df
num_bug_reports = random.randint(5000, 8000)
bug_descriptions_pool = [
    "Application crashes on startup.", "Login button is not working.", "Data loss after saving a file.",
    "UI elements are misaligned on mobile devices.", "Backend API endpoint returns 500 error.",
    "Performance bottleneck in database query.", "New feature X has integration issues.",
    "User profile picture fails to upload.", "Critical security vulnerability found.",
    "Minor UI glitch in dropdown menu.", "Translation error on checkout page.",
    "Unresponsive UI when loading large datasets.", "Memory leak detected in module Y.",
    "Third-party library conflict causing issues.", "Urgent fix needed for production server.",
    "Incorrect calculation in financial report.", "Search functionality returns irrelevant results.",
    "Complex workflow breaks at step 3.", "Disk space issue on build server.",
    "Notification system not sending alerts.", "A small bug on the landing page, priority low.",
    "Typo in the documentation.", "Minor styling issue in the footer.", "Unexpected behavior with caching.",
    "API gateway intermittently failing.", "Crash report shows null pointer exception.",
    "Data integrity issue after batch processing.", "Slow response time for dashboard loading.",
    "High severity bug, causing major disruption.", "Urgent: payment processing failed for critical users.",
    "Complex authentication flow is broken.", "Integration with external service has a major bug."
]

severities = ['Minor', 'Major', 'Critical']
bug_reports_df = pd.DataFrame({
    'bug_id': np.arange(1, num_bug_reports + 1),
    'reporter_dev_id': np.random.choice(dev_ids, num_bug_reports),
    'report_date': pd.to_datetime(pd.Timestamp.now() - pd.to_timedelta(np.random.randint(1, 3 * 365, num_bug_reports), unit='D')),
    'bug_description': np.random.choice(bug_descriptions_pool, num_bug_reports),
    'severity': np.random.choice(severities, num_bug_reports, p=[0.5, 0.35, 0.15]), # Bias severity distribution
    'estimated_fix_hours': np.random.randint(1, 101, num_bug_reports) # Initial random hours
})

# Simulate realistic patterns for estimated_fix_hours and priority_level
bug_reports_df['priority_level'] = 'Medium' # Default

# Conditions for 'Critical' severity and 'High' priority
critical_keywords = ['crash', 'urgent', 'data loss', 'critical', 'production down', 'security vulnerability', 'major disruption', 'payment failed']
critical_pattern = r'\b(?:' + '|'.join(critical_keywords) + r')\b'

# Apply patterns to adjust severity, estimated_fix_hours, and priority_level
for i, row in bug_reports_df.iterrows():
    desc = row['bug_description'].lower()
    
    # Adjust severity based on keywords
    if re.search(critical_pattern, desc):
        bug_reports_df.loc[i, 'severity'] = 'Critical'
    elif 'complex' in desc or 'integration' in desc or 'performance' in desc:
        if bug_reports_df.loc[i, 'severity'] == 'Minor': # Only upgrade if it's minor
            bug_reports_df.loc[i, 'severity'] = 'Major'
            
    # Adjust estimated_fix_hours based on severity and keywords
    if bug_reports_df.loc[i, 'severity'] == 'Critical':
        bug_reports_df.loc[i, 'estimated_fix_hours'] = max(row['estimated_fix_hours'], np.random.randint(50, 101))
    elif bug_reports_df.loc[i, 'severity'] == 'Major':
        bug_reports_df.loc[i, 'estimated_fix_hours'] = max(row['estimated_fix_hours'], np.random.randint(20, 70))
    
    if 'complex' in desc or 'integration' in desc:
        bug_reports_df.loc[i, 'estimated_fix_hours'] = max(row['estimated_fix_hours'], np.random.randint(30, 80))

    # Determine priority_level based on severity, estimated_fix_hours and keywords
    if bug_reports_df.loc[i, 'severity'] == 'Critical':
        bug_reports_df.loc[i, 'priority_level'] = 'High'
    elif bug_reports_df.loc[i, 'severity'] == 'Major':
        if bug_reports_df.loc[i, 'estimated_fix_hours'] >= 60 or re.search(critical_pattern, desc):
            bug_reports_df.loc[i, 'priority_level'] = 'High'
        elif bug_reports_df.loc[i, 'estimated_fix_hours'] >= 30:
            bug_reports_df.loc[i, 'priority_level'] = 'Medium'
        else:
            bug_reports_df.loc[i, 'priority_level'] = np.random.choice(['Low', 'Medium'], p=[0.2, 0.8])
    elif bug_reports_df.loc[i, 'severity'] == 'Minor':
        if bug_reports_df.loc[i, 'estimated_fix_hours'] >= 20:
            bug_reports_df.loc[i, 'priority_level'] = 'Medium'
        else:
            bug_reports_df.loc[i, 'priority_level'] = 'Low'

    # Introduce some randomness to priority to make it less deterministic
    if np.random.rand() < 0.05: # 5% chance to slightly shift priority
        current_priority = bug_reports_df.loc[i, 'priority_level']
        if current_priority == 'Low': bug_reports_df.loc[i, 'priority_level'] = np.random.choice(['Low', 'Medium'], p=[0.9, 0.1])
        elif current_priority == 'Medium': bug_reports_df.loc[i, 'priority_level'] = np.random.choice(['Low', 'Medium', 'High'], p=[0.1, 0.8, 0.1])
        elif current_priority == 'High': bug_reports_df.loc[i, 'priority_level'] = np.random.choice(['Medium', 'High'], p=[0.1, 0.9])

print("--- Synthetic Data Generation Complete ---")
print(f"Developers DF shape: {developers_df.shape}")
print(f"Bug Reports DF shape: {bug_reports_df.shape}")
print("\nPriority Level Distribution (Raw):")
print(bug_reports_df['priority_level'].value_counts(normalize=True))

# --- 2. Load into SQLite & SQL Feature Engineering ---
conn = sqlite3.connect(':memory:')
developers_df.to_sql('developers', conn, index=False, if_exists='replace')
bug_reports_df.to_sql('bug_reports', conn, index=False, if_exists='replace')

sql_query = """
SELECT
    br.bug_id,
    br.reporter_dev_id,
    br.report_date,
    br.bug_description,
    br.severity,
    br.estimated_fix_hours,
    br.priority_level,
    d.team AS reporter_team,
    d.experience_years AS reporter_experience_years,
    d.avg_bugs_resolved_per_month AS reporter_avg_bugs_resolved_per_month
FROM bug_reports br
INNER JOIN developers d ON br.reporter_dev_id = d.dev_id;
"""

bug_features_df = pd.read_sql(sql_query, conn)
conn.close()

print("\n--- SQLite & SQL Feature Engineering Complete ---")
print(f"Combined Bug Features DF shape: {bug_features_df.shape}")
print(bug_features_df.head())

# --- 3. Pandas Feature Engineering & Multi-Class Target Creation ---

# Handle NaN values (though with inner join and valid IDs, shouldn't be many)
bug_features_df['estimated_fix_hours'].fillna(bug_features_df['estimated_fix_hours'].median(), inplace=True)
bug_features_df['reporter_experience_years'].fillna(bug_features_df['reporter_experience_years'].median(), inplace=True)
bug_features_df['reporter_avg_bugs_resolved_per_month'].fillna(bug_features_df['reporter_avg_bugs_resolved_per_month'].median(), inplace=True)

# Convert report_date to datetime objects
bug_features_df['report_date'] = pd.to_datetime(bug_features_df['report_date'])

# Calculate bug_age_at_analysis_days
analysis_date = bug_features_df['report_date'].max() + pd.Timedelta(days=30)
bug_features_df['bug_age_at_analysis_days'] = (analysis_date - bug_features_df['report_date']).dt.days

# Text Features from bug_description
bug_features_df['description_length'] = bug_features_df['bug_description'].str.len()

# has_critical_keyword
all_critical_keywords_pattern = r'\b(critical|crash|urgent|data loss|production down|security vulnerability|major disruption|payment failed)\b'
bug_features_df['has_critical_keyword'] = bug_features_df['bug_description'].str.contains(
    all_critical_keywords_pattern, case=False, na=False
).astype(int)

# num_tech_keywords
tech_keywords = ['error', 'bug', 'issue', 'feature', 'database', 'frontend', 'backend', 'api', 'module', 'system', 'server', 'client', 'integration', 'network', 'deployment', 'performance']
tech_keywords_pattern = r'\b(?:' + '|'.join(tech_keywords) + r')\b'
bug_features_df['num_tech_keywords'] = bug_features_df['bug_description'].str.lower().apply(
    lambda x: len(re.findall(tech_keywords_pattern, x))
)

# Define features X and target y
numerical_features = [
    'estimated_fix_hours', 'bug_age_at_analysis_days', 'reporter_experience_years',
    'reporter_avg_bugs_resolved_per_month', 'description_length',
    'has_critical_keyword', 'num_tech_keywords'
]
categorical_features = ['severity', 'reporter_team']
text_feature = ['bug_description'] # Keep it as a list for ColumnTransformer

X = bug_features_df[numerical_features + categorical_features + text_feature]
y = bug_features_df['priority_level']

# Convert y to a categorical type and ensure order for consistent stratification and plotting
priority_order = ['Low', 'Medium', 'High']
y = pd.Categorical(y, categories=priority_order, ordered=True)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print("\n--- Pandas Feature Engineering Complete ---")
print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
print("\nTraining set priority distribution:")
print(y_train.value_counts(normalize=True))
print("\nTest set priority distribution:")
print(y_test.value_counts(normalize=True))

# --- 4. Data Visualization ---

print("\n--- Generating Visualizations ---")
plt.figure(figsize=(15, 6))

# Violin plot for estimated_fix_hours vs priority_level
plt.subplot(1, 2, 1)
sns.violinplot(x='priority_level', y='estimated_fix_hours', data=bug_features_df, order=priority_order)
plt.title('Distribution of Estimated Fix Hours by Priority Level')
plt.xlabel('Priority Level')
plt.ylabel('Estimated Fix Hours')

# Stacked bar chart for priority_level across different severity values
plt.subplot(1, 2, 2)
# Ensure severity has a consistent order for plotting
severity_order = ['Minor', 'Major', 'Critical']
bug_features_df['severity'] = pd.Categorical(bug_features_df['severity'], categories=severity_order, ordered=True)

crosstab_severity_priority = pd.crosstab(
    bug_features_df['severity'], 
    bug_features_df['priority_level'], 
    normalize='index'
).loc[severity_order, priority_order] # Ensure consistent order

crosstab_severity_priority.plot(kind='bar', stacked=True, ax=plt.gca(), cmap='viridis')
plt.title('Proportion of Priority Level by Severity')
plt.xlabel('Severity')
plt.ylabel('Proportion')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Priority Level', bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.show()

# --- 5. ML Pipeline & Evaluation ---

print("\n--- Building and Training ML Pipeline ---")

# Preprocessing steps
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = OneHotEncoder(handle_unknown='ignore')

# TfidfVectorizer needs to handle potential NaNs in text by filling them
# It's better to ensure this happens before the ColumnTransformer for the text column
X_train['bug_description'].fillna('', inplace=True)
X_test['bug_description'].fillna('', inplace=True)

text_transformer = TfidfVectorizer(max_features=1000, stop_words='english')

# Create a preprocessor using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features),
        ('text', text_transformer, text_feature)
    ],
    remainder='passthrough' # Keep other columns if any, though not expected here
)

# Create the full pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'))
])

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

print("\n--- Model Evaluation ---")
print(f"Accuracy Score: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=priority_order))

print("\n--- Pipeline Execution Complete ---")