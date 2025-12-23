import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

# 1. Generate a synthetic dataset of 500-1000 short text documents
def generate_synthetic_data(num_documents=750):
    categories = {
        'Technology': {
            'keywords': ['tech', 'software', 'ai', 'computer', 'developer', 'startup', 'innovation', 'data', 'cloud', 'security'],
            'generic_phrases': ['a new report', 'latest news from', 'future of', 'exploring the', 'cutting edge', 'modern solutions']
        },
        'Sports': {
            'keywords': ['game', 'score', 'team', 'match', 'player', 'athlete', 'win', 'championship', 'stadium', 'ball'],
            'generic_phrases': ['today\'s highlights', 'after a tough', 'great performance by', 'upcoming season', 'final result was']
        },
        'Cooking': {
            'keywords': ['recipe', 'cook', 'ingredient', 'bake', 'food', 'kitchen', 'meal', 'flavor', 'delicious', 'prepare'],
            'generic_phrases': ['easy steps to', 'a perfect dish', 'for your next dinner', 'try this amazing', 'healthy and tasty']
        }
    }
    
    generic_words = ['the', 'a', 'is', 'it', 'and', 'but', 'for', 'with', 'we', 'this', 'that', 'new', 'great', 'some', 'many', 'of', 'in', 'on', 'at', 'to', 'from']

    documents = []
    labels = []

    for _ in range(num_documents):
        category_name = random.choice(list(categories.keys()))
        category_info = categories[category_name]
        
        doc_parts = []
        
        # Start with a generic phrase
        if random.random() < 0.7:
            doc_parts.append(random.choice(category_info['generic_phrases']))
        
        # Add 1 to 3 category-specific keywords
        num_keywords = random.randint(1, 3)
        chosen_keywords = random.sample(category_info['keywords'], num_keywords)
        doc_parts.extend(chosen_keywords)

        # Add some generic words
        num_generic = random.randint(3, 7)
        doc_parts.extend(random.sample(generic_words, num_generic))

        random.shuffle(doc_parts) # Shuffle to make it less structured
        document = ' '.join(doc_parts).strip()
        documents.append(document)
        labels.append(category_name)
        
    return documents, labels

print("--- Generating Synthetic Dataset ---")
X, y = generate_synthetic_data(num_documents=750)
print(f"Generated {len(X)} documents across {len(set(y))} categories.")
print(f"Example document: '{X[0]}' (Category: {y[0]})")
print(f"Example document: '{X[len(X)//3]}' (Category: {y[len(X)//3]})")
print(f"Example document: '{X[len(X)//3 * 2]}' (Category: {y[len(X)//3 * 2]})")
print("-" * 40)

# 2. Split the dataset into training and testing sets
print("--- Splitting Data ---")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
print(f"Training set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")
print("-" * 40)

# 3. Construct an sklearn.pipeline.Pipeline
print("--- Constructing Pipeline ---")
text_classification_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', max_df=0.85, min_df=5)),
    ('classifier', LogisticRegression(random_state=42, solver='liblinear', max_iter=1000))
])
print("Pipeline steps: TfidfVectorizer -> LogisticRegression")
print("-" * 40)

# 4. Train the pipeline on the training data and make predictions on the test data
print("--- Training Pipeline ---")
text_classification_pipeline.fit(X_train, y_train)
print("Pipeline training complete.")

print("--- Making Predictions on Test Data ---")
y_pred = text_classification_pipeline.predict(X_test)
print("Predictions generated.")
print("-" * 40)

# 5. Print the sklearn.metrics.classification_report for the test set predictions
print("--- Classification Report ---")
print(classification_report(y_test, y_pred))
print("-" * 40)

# 6. Extract and print feature importance
print("--- Feature Importance (Top 5 words per class) ---")

# Access the trained vectorizer and classifier
tfidf_vectorizer = text_classification_pipeline.named_steps['tfidf']
logistic_regression_model = text_classification_pipeline.named_steps['classifier']

# Get feature names (words)
feature_names = tfidf_vectorizer.get_feature_names_out()

# Get coefficients for each class
# For multi-class (OvR), coef_[i] represents coefficients for class i vs. all others
coefficients = logistic_regression_model.coef_
class_labels = logistic_regression_model.classes_

for i, class_label in enumerate(class_labels):
    print(f"\nCategory: '{class_label}'")
    
    # Get the coefficients for the current class
    class_coefficients = coefficients[i]
    
    # Get indices of top 5 highest positive coefficients
    top_5_indices = class_coefficients.argsort()[-5:][::-1] # [::-1] to get descending order
    
    # Map indices to feature names and print
    top_5_words = [feature_names[idx] for idx in top_5_indices]
    print(f"  Top 5 important words: {', '.join(top_5_words)}")
    
    # Interpretation
    if class_label == 'Technology':
        print("  Interpretation: These words strongly suggest topics related to software development, data science, and innovation.")
    elif class_label == 'Sports':
        print("  Interpretation: These words are highly indicative of competitive events, team activities, and athletic performance.")
    elif class_label == 'Cooking':
        print("  Interpretation: These words clearly point to culinary activities, ingredients, and meal preparation.")
    else:
        print("  Interpretation: These words are highly characteristic of this category.")

print("-" * 40)