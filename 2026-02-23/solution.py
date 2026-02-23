import pandas as pd
import numpy as np
import datetime
import random
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report

# Set Matplotlib backend to 'Agg' to prevent display issues in non-interactive environments
# and suppress plot windows from popping up
plt.switch_backend('Agg')

# --- 1. Generate Synthetic Data (Pandas/Numpy) ---

# Random seed for reproducibility
np.random.seed(42)
random.seed(42)

# 1.1 customers_df
num_customers = random.randint(500, 700)
customer_ids = np.arange(num_customers)
signup_dates = pd.to_datetime('today') - pd.to_timedelta(np.random.randint(0, 5 * 365, num_customers), unit='D')
regions = np.random.choice(['North', 'South', 'East', 'West', 'Central'], num_customers)
account_tiers = np.random.choice(['Bronze', 'Silver', 'Gold', 'Platinum'], num_customers, p=[0.4, 0.3, 0.2, 0.1])

customers_df = pd.DataFrame({
    'customer_id': customer_ids,
    'signup_date': signup_dates,
    'region': regions,
    'account_tier': account_tiers
})

# 1.2 messages_df
num_messages = random.randint(5000, 8000)
message_ids = np.arange(num_messages)

# Sample customer IDs for messages (with replacement)
message_customer_ids = np.random.choice(customers_df['customer_id'], num_messages)

# Map signup dates to message customer IDs
customer_signup_map = customers_df.set_index('customer_id')['signup_date'].to_dict()
message_signup_dates = pd.Series(message_customer_ids).map(customer_signup_map)

# Generate message dates ensuring they are after signup dates
message_dates = []
for signup_dt in message_signup_dates:
    days_after_signup = np.random.randint(1, 365 * 3) # Message can be up to 3 years after signup
    message_dates.append(signup_dt + pd.to_timedelta(days_after_signup, unit='D'))
message_dates = pd.Series(message_dates)


# Synthetic message text and response time generation based on hidden intent
message_templates = {
    'Billing_Issue': [
        "I have a problem with my {action} on the last {product}.",
        "My {item} is incorrect. Could you check the {type}?",
        "There's an issue with my {product} {transaction}.",
        "I need help with my {status} {payment}.",
        "I was charged twice for the {service}. Please investigate this {issue}."
    ],
    'Technical_Support': [
        "My {device} is not {action}. I'm getting an {error} message.",
        "I can't {verb} into my {system}. It keeps {status}.",
        "The {software} {status} after the last update. It keeps {action}.",
        "I'm experiencing a {type} {issue} with my {product}.",
        "My {app} has a {bug}. It {action} every time I {verb}."
    ],
    'Feature_Request': [
        "I'd like to {action} a new {feature} for {product}.",
        "It would be great if you could {verb} {functionality} into {app}.",
        "Consider adding a {type} {feature} to your {service}.",
        "Is it possible to {action} {item} to {platform}?",
        "I have an {idea} for a new {capability} in your {software}."
    ],
    'General_Inquiry': [
        "I have a {type} question about {product}.",
        "Could you tell me more about your {service} {feature}?",
        "What's the {status} on {item}?",
        "I need some {info} regarding {topic}.",
        "Can you clarify the {detail} about {product}?"
    ]
}

intent_keywords_pool = {
    'Billing_Issue': {
        'action': ['bill', 'invoice', 'charge', 'payment', 'subscription'],
        'product': ['service', 'account', 'plan', 'order'],
        'item': ['bill', 'invoice', 'charge'],
        'type': ['statement', 'amount'],
        'transaction': ['payment', 'charge', 'billing'],
        'status': ['pending', 'overdue'],
        'payment': ['method', 'transaction'],
        'service': ['subscription', 'premium feature'],
        'issue': ['discrepancy', 'error']
    },
    'Technical_Support': {
        'device': ['app', 'website', 'login', 'account', 'system'],
        'action': ['working', 'loading', 'connecting', 'saving'],
        'error': ['error', 'bug', 'crash', 'timeout'],
        'verb': ['login', 'access', 'upload', 'submit'],
        'system': ['platform', 'dashboard', 'portal'],
        'status': ['freezing', 'crashing', 'disconnecting'],
        'software': ['app', 'website', 'tool'],
        'type': ['critical', 'minor'],
        'issue': ['glitch', 'problem'],
        'product': ['service', 'application'],
        'app': ['application', 'software'],
        'bug': ['defect', 'malfunction']
    },
    'Feature_Request': {
        'action': ['suggest', 'request', 'propose', 'ask for'],
        'feature': ['feature', 'option', 'tool', 'setting'],
        'product': ['app', 'service', 'platform'],
        'verb': ['integrate', 'add', 'implement', 'include'],
        'functionality': ['new functionality', 'an export option', 'a custom report'],
        'app': ['application', 'software'],
        'type': ['new', 'advanced', 'custom'],
        'item': ['this feature', 'a dashboard', 'an integration'],
        'platform': ['your platform', 'the system'],
        'idea': ['great idea', 'suggestion'],
        'capability': ['capability', 'function']
    },
    'General_Inquiry': {
        'type': ['general', 'quick', 'simple', 'specific'],
        'product': ['service', 'account', 'plan', 'feature'],
        'service': ['your service', 'the platform'],
        'feature': ['details', 'pricing', 'availability'],
        'status': ['status', 'update'],
        'item': ['my request', 'this issue'],
        'info': ['information', 'details'],
        'topic': ['account management', 'upcoming features', 'support hours'],
        'detail': ['terms', 'process']
    }
}

message_texts = []
response_times = []
intent_categories = []
for _ in range(num_messages):
    intent = np.random.choice(list(message_templates.keys()), p=[0.25, 0.35, 0.20, 0.20])
    intent_categories.append(intent)

    template = random.choice(message_templates[intent])
    
    # Randomly select keywords for the template
    replacements = {}
    for placeholder in intent_keywords_pool[intent]:
        replacements[placeholder] = random.choice(intent_keywords_pool[intent][placeholder])
    
    message_text = template.format(**replacements)
    message_texts.append(message_text)

    # Bias response times
    if intent in ['Billing_Issue', 'Technical_Support']:
        response_times.append(np.random.uniform(0.5, 24.0)) # Shorter
    else: # Feature_Request, General_Inquiry
        response_times.append(np.random.uniform(12.0, 72.0)) # Longer

messages_df = pd.DataFrame({
    'message_id': message_ids,
    'customer_id': message_customer_ids,
    'message_date': message_dates,
    'message_text': message_texts,
    'actual_response_time_hours': response_times,
    'hidden_intent': intent_categories # Keep hidden intent for verification/analysis
})

# Sort messages_df by customer_id then message_date
messages_df = messages_df.sort_values(by=['customer_id', 'message_date']).reset_index(drop=True)

print(f"Generated {len(customers_df)} customer records.")
print(f"Generated {len(messages_df)} message records.")
print("\nCustomers DataFrame head:")
print(customers_df.head())
print("\nMessages DataFrame head:")
print(messages_df.head())


# --- 2. Load into SQLite & SQL Feature Engineering ---
conn = sqlite3.connect(':memory:')

customers_df.to_sql('customers', conn, index=False, if_exists='replace')
messages_df.to_sql('messages', conn, index=False, if_exists='replace',
                   dtype={'message_date': 'TEXT', 'signup_date': 'TEXT'}) # Store dates as TEXT for SQLite

sql_query = """
SELECT
    m.message_id,
    m.customer_id,
    m.message_date,
    m.message_text,
    c.region,
    c.account_tier,
    c.signup_date,
    -- user_prior_message_count: count of all *previous* messages by the same user
    COUNT(m.message_id) OVER (PARTITION BY m.customer_id ORDER BY m.message_date ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS user_prior_message_count,
    -- user_avg_prior_response_time_hours: average response time of all *previous* messages by the same user
    COALESCE(AVG(m.actual_response_time_hours) OVER (PARTITION BY m.customer_id ORDER BY m.message_date ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING), 0.0) AS user_avg_prior_response_time_hours,
    -- days_since_last_user_message: days between current message_date and user's most recent prior message_date.
    -- If first message, use days between signup_date and message_date.
    CAST(
        JULIANDAY(m.message_date) - JULIANDAY(
            COALESCE(
                LAG(m.message_date, 1) OVER (PARTITION BY m.customer_id ORDER BY m.message_date),
                c.signup_date
            )
        )
    AS REAL) AS days_since_last_user_message,
    m.actual_response_time_hours
FROM
    messages m
JOIN
    customers c ON m.customer_id = c.customer_id
ORDER BY
    m.customer_id, m.message_date;
"""

message_features_df = pd.read_sql_query(sql_query, conn)
conn.close()

print("\nSQL Feature Engineered DataFrame head:")
print(message_features_df.head())
print("\nSQL Feature Engineered DataFrame describe for new features:")
print(message_features_df[['user_prior_message_count', 'user_avg_prior_response_time_hours', 'days_since_last_user_message']].describe())


# --- 3. Pandas Feature Engineering & Multi-Class Target Creation ---

# Handle NaN values (SQL query should have handled most, but reinforce)
message_features_df['user_prior_message_count'] = message_features_df['user_prior_message_count'].fillna(0).astype(int)
message_features_df['user_avg_prior_response_time_hours'] = message_features_df['user_avg_prior_response_time_hours'].fillna(0.0).astype(float)
# days_since_last_user_message is handled by COALESCE in SQL, ensuring it's never NULL for first messages.

# Convert dates to datetime objects
message_features_df['signup_date'] = pd.to_datetime(message_features_df['signup_date'])
message_features_df['message_date'] = pd.to_datetime(message_features_df['message_date'])

# user_account_age_at_message_days
message_features_df['user_account_age_at_message_days'] = (
    message_features_df['message_date'] - message_features_df['signup_date']
).dt.days.astype(float)

# Text Features from message_text
message_features_df['message_length'] = message_features_df['message_text'].apply(len)
message_features_df['has_question_mark'] = message_features_df['message_text'].apply(lambda x: 1 if '?' in x else 0)

billing_keywords = ['bill', 'invoice', 'charge', 'payment', 'subscription']
tech_keywords = ['error', 'bug', 'crash', 'login', 'website', 'app', 'system']

message_features_df['num_keywords_billing'] = message_features_df['message_text'].apply(
    lambda x: sum(1 for keyword in billing_keywords if keyword in x.lower())
)
message_features_df['num_keywords_tech'] = message_features_df['message_text'].apply(
    lambda x: sum(1 for keyword in tech_keywords if keyword in x.lower())
)

# Create the Multi-Class Target message_intent_category
conditions = [
    message_features_df['actual_response_time_hours'] < 12,
    (message_features_df['actual_response_time_hours'] >= 12) & (message_features_df['actual_response_time_hours'] <= 48),
    message_features_df['actual_response_time_hours'] > 48
]
choices = ['Urgent_Support', 'Standard_Support', 'Low_Priority']
message_features_df['message_intent_category'] = np.select(conditions, choices, default='Unknown')

print("\nPandas Feature Engineered DataFrame head (with new features and target):")
print(message_features_df[['message_id', 'user_account_age_at_message_days', 'message_length', 'has_question_mark',
                           'num_keywords_billing', 'num_keywords_tech', 'message_intent_category']].head())
print("\nMessage Intent Category distribution:")
print(message_features_df['message_intent_category'].value_counts())

# Define features X and target y
numerical_features = ['user_account_age_at_message_days', 'user_prior_message_count', 'user_avg_prior_response_time_hours',
                      'days_since_last_user_message', 'message_length', 'has_question_mark',
                      'num_keywords_billing', 'num_keywords_tech']
categorical_features = ['region', 'account_tier']
text_feature = 'message_text'

X = message_features_df[numerical_features + categorical_features + [text_feature]]
y = message_features_df['message_intent_category']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

print(f"\nX_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")


# --- 4. Data Visualization ---

print("\nGenerating visualizations...")

# 4.1 Violin plot of message_length by message_intent_category
plt.figure(figsize=(10, 6))
sns.violinplot(x='message_intent_category', y='message_length', data=message_features_df, palette='viridis')
plt.title('Distribution of Message Length by Intent Category')
plt.xlabel('Message Intent Category')
plt.ylabel('Message Length')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('message_length_by_intent_violin_plot.png')
print("Plot saved: message_length_by_intent_violin_plot.png")
plt.close() # Close plot to free memory

# 4.2 Stacked bar chart of message_intent_category across different account_tier values
account_tier_intent_counts = message_features_df.groupby(['account_tier', 'message_intent_category']).size().unstack(fill_value=0)
account_tier_intent_percentages = account_tier_intent_counts.apply(lambda x: x / x.sum(), axis=1)

plt.figure(figsize=(12, 7))
account_tier_intent_percentages.plot(kind='bar', stacked=True, colormap='viridis')
plt.title('Distribution of Message Intent Category by Account Tier')
plt.xlabel('Account Tier')
plt.ylabel('Proportion of Messages')
plt.xticks(rotation=45)
plt.legend(title='Message Intent Category', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('message_intent_by_account_tier_stacked_bar.png')
print("Plot saved: message_intent_by_account_tier_stacked_bar.png")
plt.close() # Close plot to free memory


# --- 5. ML Pipeline & Evaluation (Multi-Class) ---

print("\nBuilding and training ML pipeline...")

# Create preprocessing pipelines for numerical, categorical, and text features
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

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
    remainder='drop' # Drop any columns not specified
)

# Create the full pipeline with preprocessor and classifier
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', HistGradientBoostingClassifier(random_state=42))
])

# Train the pipeline
model_pipeline.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model_pipeline.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print("\n--- ML Model Evaluation ---")
print(f"Accuracy Score: {accuracy:.4f}")
print("\nClassification Report:")
print(class_report)

print("\nScript finished successfully.")