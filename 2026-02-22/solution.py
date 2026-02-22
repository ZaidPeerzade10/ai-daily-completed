import pandas as pd
import numpy as np
import random
import datetime
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')

# --- 1. Generate Synthetic Data (Pandas/Numpy) ---

print("--- 1. Generating Synthetic Data ---")

# Define number of rows
N_CUSTOMERS = random.randint(500, 700)
N_MESSAGES = random.randint(5000, 8000)

# Customer data
customer_ids = np.arange(1, N_CUSTOMERS + 1)
signup_dates = [datetime.date.today() - datetime.timedelta(days=random.randint(1, 365*5)) for _ in range(N_CUSTOMERS)]
regions = ['North', 'South', 'East', 'West', 'Central']
account_tiers = ['Bronze', 'Silver', 'Gold', 'Platinum']

customers_df = pd.DataFrame({
    'customer_id': customer_ids,
    'signup_date': signup_dates,
    'region': np.random.choice(regions, N_CUSTOMERS),
    'account_tier': np.random.choice(account_tiers, N_CUSTOMERS, p=[0.4, 0.3, 0.2, 0.1])
})

# Message data
message_ids = np.arange(1, N_MESSAGES + 1)
message_customer_ids = np.random.choice(customers_df['customer_id'], N_MESSAGES)

messages_df = pd.DataFrame({
    'message_id': message_ids,
    'customer_id': message_customer_ids
})

# Merge signup_date to messages_df temporarily to ensure message_date > signup_date
messages_df = messages_df.merge(customers_df[['customer_id', 'signup_date']], on='customer_id', how='left')

# Define message intent categories, keywords, and response time biases
intent_categories = {
    'Billing_Issue': {
        'keywords': ['bill', 'invoice', 'charge', 'payment', 'refund', 'transaction', 'cost'],
        'response_time_range': (0.5, 24) # Shorter
    },
    'Technical_Support': {
        'keywords': ['error', 'bug', 'crash', 'login', 'issue', 'problem', 'fix', 'technical', 'broken'],
        'response_time_range': (0.5, 24) # Shorter
    },
    'Feature_Request': {
        'keywords': ['feature', 'request', 'suggestion', 'add', 'new functionality', 'idea', 'improve'],
        'response_time_range': (12, 72) # Longer
    },
    'General_Inquiry': {
        'keywords': ['hello', 'question', 'info', 'support', 'help', 'ask', 'query', 'general'],
        'response_time_range': (12, 72) # Longer
    }
}

message_texts = []
actual_response_times = []
message_dates = []

for index, row in messages_df.iterrows():
    signup_date = row['signup_date']
    
    # Generate message_date always after signup_date
    days_after_signup = random.randint(1, 365*5) # Up to 5 years after signup
    message_date = signup_date + datetime.timedelta(days=days_after_signup)
    
    # Choose a random intent category
    intent = np.random.choice(list(intent_categories.keys()), p=[0.3, 0.3, 0.2, 0.2])
    
    # Generate message text based on intent
    keyword = random.choice(intent_categories[intent]['keywords'])
    generic_phrases = [
        "I have a quick question about", "Can you help me with", "Regarding an issue with",
        "Just wanted to inquire about", "Looking for assistance on", "Curious about", "Facing a problem with"
    ]
    message_text = f"{random.choice(generic_phrases)} my {keyword}."
    if random.random() < 0.3: # Add a question mark sometimes
        message_text += "?"
    
    # Generate response time based on intent bias
    min_time, max_time = intent_categories[intent]['response_time_range']
    response_time = round(random.uniform(min_time, max_time), 2)
    
    message_dates.append(message_date)
    message_texts.append(message_text)
    actual_response_times.append(response_time)

messages_df['message_date'] = message_dates
messages_df['message_text'] = message_texts
messages_df['actual_response_time_hours'] = actual_response_times

# Drop the temporary signup_date column
messages_df = messages_df.drop(columns=['signup_date'])

# Sort messages_df by customer_id then message_date for sequential features
messages_df = messages_df.sort_values(by=['customer_id', 'message_date']).reset_index(drop=True)

print(f"Generated {len(customers_df)} customers and {len(messages_df)} messages.")
print("Customers DataFrame head:")
print(customers_df.head())
print("\nMessages DataFrame head:")
print(messages_df.head())

# --- 2. Load into SQLite & SQL Feature Engineering ---

print("\n--- 2. Loading into SQLite & SQL Feature Engineering ---")

conn = sqlite3.connect(':memory:')

customers_df.to_sql('customers', conn, index=False, if_exists='replace')
messages_df.to_sql('messages', conn, index=False, if_exists='replace')

sql_query = """
SELECT
    m.message_id,
    m.customer_id,
    m.message_date,
    m.message_text,
    c.region,
    c.account_tier,
    c.signup_date,
    m.actual_response_time_hours,
    -- user_prior_message_count: Count of all previous messages by the same user
    COUNT(m.message_id) OVER (
        PARTITION BY m.customer_id
        ORDER BY m.message_date
        ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
    ) AS user_prior_message_count,
    -- user_avg_prior_response_time_hours: Average actual_response_time_hours of all previous messages
    COALESCE(AVG(m.actual_response_time_hours) OVER (
        PARTITION BY m.customer_id
        ORDER BY m.message_date
        ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
    ), 0.0) AS user_avg_prior_response_time_hours,
    -- days_since_last_user_message: Days between current message_date and user's most recent prior message_date.
    -- If first message, days between signup_date and message_date.
    COALESCE(
        CAST((julianday(m.message_date) - julianday(LAG(m.message_date) OVER (
            PARTITION BY m.customer_id
            ORDER BY m.message_date
        ))) AS INTEGER),
        CAST((julianday(m.message_date) - julianday(c.signup_date)) AS INTEGER)
    ) AS days_since_last_user_message
FROM
    messages AS m
JOIN
    customers AS c ON m.customer_id = c.customer_id
ORDER BY
    m.customer_id, m.message_date;
"""

message_features_df = pd.read_sql_query(sql_query, conn)
conn.close()

print("SQL Feature Engineering complete. Resulting DataFrame head:")
print(message_features_df.head())
print(f"DataFrame shape after SQL: {message_features_df.shape}")


# --- 3. Pandas Feature Engineering & Multi-Class Target Creation ---

print("\n--- 3. Pandas Feature Engineering & Multi-Class Target Creation ---")

# Handle NaN values (SQL should have handled most, but as a safeguard)
message_features_df['user_prior_message_count'] = message_features_df['user_prior_message_count'].fillna(0).astype(int)
message_features_df['user_avg_prior_response_time_hours'] = message_features_df['user_avg_prior_response_time_hours'].fillna(0.0)
# SQL's COALESCE handles days_since_last_user_message for first messages, so NaNs should not be present.
# If any still exist (e.g., edge case), fill with a large sentinel value
message_features_df['days_since_last_user_message'] = message_features_df['days_since_last_user_message'].fillna(9999).astype(int)


# Convert date columns to datetime objects
message_features_df['signup_date'] = pd.to_datetime(message_features_df['signup_date'])
message_features_df['message_date'] = pd.to_datetime(message_features_df['message_date'])

# Calculate user_account_age_at_message_days
message_features_df['user_account_age_at_message_days'] = (
    message_features_df['message_date'] - message_features_df['signup_date']
).dt.days

# Text Features from message_text
message_features_df['message_length'] = message_features_df['message_text'].apply(len)
message_features_df['has_question_mark'] = message_features_df['message_text'].apply(lambda x: 1 if '?' in x else 0)

billing_keywords = ['bill', 'invoice', 'charge', 'payment', 'refund', 'transaction', 'cost']
tech_keywords = ['error', 'bug', 'crash', 'login', 'issue', 'problem', 'fix', 'technical', 'broken']

message_features_df['num_keywords_billing'] = message_features_df['message_text'].apply(
    lambda x: sum(1 for kw in billing_keywords if kw in x.lower())
)
message_features_df['num_keywords_tech'] = message_features_df['message_text'].apply(
    lambda x: sum(1 for kw in tech_keywords if kw in x.lower())
)

# Create the Multi-Class Target message_intent_category
bins = [-np.inf, 12, 48, np.inf]
labels = ['Urgent_Support', 'Standard_Support', 'Low_Priority']
message_features_df['message_intent_category'] = pd.cut(
    message_features_df['actual_response_time_hours'],
    bins=bins,
    labels=labels,
    right=False # Include 12 in Standard, not Urgent
)

# Define features X and target y
numerical_features = [
    'user_account_age_at_message_days',
    'user_prior_message_count',
    'user_avg_prior_response_time_hours',
    'days_since_last_user_message',
    'message_length',
    'has_question_mark',
    'num_keywords_billing',
    'num_keywords_tech'
]
categorical_features = ['region', 'account_tier']
text_feature = 'message_text'

features_to_use = numerical_features + categorical_features + [text_feature]
X = message_features_df[features_to_use]
y = message_features_df['message_intent_category']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
print("\nTarget distribution in training set:")
print(y_train.value_counts(normalize=True))
print("\nTarget distribution in test set:")
print(y_test.value_counts(normalize=True))

print("\nDataFrame with new features and target head:")
print(message_features_df[['message_id', 'user_account_age_at_message_days', 'days_since_last_user_message', 'message_length', 'message_intent_category']].head())


# --- 4. Data Visualization ---

print("\n--- 4. Data Visualization ---")

plt.style.use('seaborn-v0_8-darkgrid')
plt.figure(figsize=(18, 7))

# Violin plot: message_length vs. message_intent_category
plt.subplot(1, 2, 1)
sns.violinplot(x='message_intent_category', y='message_length', data=message_features_df, palette='viridis')
plt.title('Distribution of Message Length by Intent Category')
plt.xlabel('Message Intent Category')
plt.ylabel('Message Length')

# Stacked bar chart: message_intent_category across different account_tier
plt.subplot(1, 2, 2)
(message_features_df.groupby('account_tier')['message_intent_category']
 .value_counts(normalize=True)
 .unstack()
 .plot(kind='bar', stacked=True, ax=plt.gca(), cmap='plasma'))
plt.title('Message Intent Category Distribution by Account Tier')
plt.xlabel('Account Tier')
plt.ylabel('Proportion')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Intent Category', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()


# --- 5. ML Pipeline & Evaluation (Multi-Class) ---

print("\n--- 5. ML Pipeline & Evaluation (Multi-Class) ---")

# Preprocessing for numerical features
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# Preprocessing for categorical features
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Preprocessing for text features
text_transformer = Pipeline(steps=[
    ('tfidf', TfidfVectorizer(max_features=500))
])

# Create a ColumnTransformer to apply different transformations to different columns
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features),
        ('text', text_transformer, text_feature)
    ],
    remainder='drop' # Drop columns not specified
)

# Create the full pipeline
# HistGradientBoostingClassifier is a good choice for speed and performance
# and handles missing values, though we've imputed them.
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', HistGradientBoostingClassifier(random_state=42))
])

# Train the model
print("Training the ML pipeline...")
model_pipeline.fit(X_train, y_train)
print("Training complete.")

# Make predictions on the test set
y_pred = model_pipeline.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f"\nModel Accuracy on Test Set: {accuracy:.4f}")
print("\nClassification Report:\n", class_report)

print("\n--- Script Finished ---")