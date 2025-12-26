import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Suppress warnings for cleaner output, especially related to matplotlib/seaborn versions
warnings.filterwarnings('ignore')

# --- Task 1: Generate a pandas DataFrame with 800 samples ---
N_SAMPLES = 800

# 1.1 Numerical features: amount, age, duration_months
# amount: random positive floats with right-skew and clear high outliers
# Majority from exponential distribution for skewness
amount_base = np.random.exponential(scale=150, size=int(N_SAMPLES * 0.95))
# A small percentage of significantly larger values for clear outliers
amount_outliers = np.random.uniform(low=1000, high=8000, size=N_SAMPLES - len(amount_base))
amount = np.concatenate((amount_base, amount_outliers))
np.random.shuffle(amount) # Shuffle to mix base values and outliers

age = np.random.randint(18, 66, N_SAMPLES) # Ages between 18 and 65
duration_months = np.random.randint(1, 121, N_SAMPLES) # Durations between 1 and 120 months

# 1.2 Categorical feature: region with varying proportions
regions = ['North', 'South', 'East', 'West']
region_proportions = [0.25, 0.35, 0.20, 0.20] # Varying proportions
region = np.random.choice(regions, N_SAMPLES, p=region_proportions)

# Create DataFrame
df = pd.DataFrame({
    'amount': amount,
    'age': age,
    'duration_months': duration_months,
    'region': region
})

print("--- Task 1: Generated DataFrame Head and Info ---")
print("DataFrame Head:")
print(df.head())
print("\nDataFrame Info:")
df.info()
print("-" * 50 + "\n")

# --- Task 2: Calculate and display comprehensive descriptive statistics ---
print("--- Task 2: Descriptive Statistics Grouped by Region ---")
# Use a list of aggregation functions for multi-column, multi-stat summary
grouped_stats = df.groupby('region')[['amount', 'age', 'duration_months']].agg(
    ['mean', 'median', 'std', 'min', 'max',
     ('25th_percentile', lambda x: x.quantile(0.25)), # Custom named lambda for 25th percentile
     ('75th_percentile', lambda x: x.quantile(0.75))]  # Custom named lambda for 75th percentile
)
print(grouped_stats)
print("-" * 50 + "\n")

# --- Task 3: Create subplots to visualize feature distributions across regions ---
print("--- Task 3: Visualizing Feature Distributions Across Regions ---")
plt.figure(figsize=(16, 6))

# Subplot 1: Amount distribution by Region
plt.subplot(1, 2, 1)
sns.boxplot(x='region', y='amount', data=df)
plt.title('Amount Distribution by Region', fontsize=14)
plt.xlabel('Region', fontsize=12)
plt.ylabel('Amount', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Subplot 2: Duration (Months) distribution by Region
plt.subplot(1, 2, 2)
sns.violinplot(x='region', y='duration_months', data=df, inner='quartile') # Using violinplot with quartiles
plt.title('Duration (Months) Distribution by Region', fontsize=14)
plt.xlabel('Region', fontsize=12)
plt.ylabel('Duration (Months)', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to make space for suptitle
plt.suptitle('Feature Distributions Across Different Regions', fontsize=16, weight='bold')
plt.show()
print("Displayed Box/Violin Plots for Amount and Duration by Region.\n")


# --- Task 4: Apply log1p transformation to 'amount' and visualize its effect ---
print("--- Task 4: Visualizing Original vs. Log1p-Transformed Amount Distribution ---")
df['log_amount'] = np.log1p(df['amount'])

plt.figure(figsize=(16, 6))

# Subplot 1: Original Amount distribution
plt.subplot(1, 2, 1)
sns.histplot(df['amount'], kde=True, bins=50, color='skyblue')
plt.title('Original Amount Distribution (Highly Skewed)', fontsize=14)
plt.xlabel('Amount', fontsize=12)
plt.ylabel('Frequency / Density', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Subplot 2: Log1p-Transformed Amount distribution
plt.subplot(1, 2, 2)
sns.histplot(df['log_amount'], kde=True, bins=50, color='lightcoral')
plt.title('Log1p-Transformed Amount Distribution (Normalized)', fontsize=14)
plt.xlabel('Log1p(Amount)', fontsize=12)
plt.ylabel('Frequency / Density', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to make space for suptitle
plt.suptitle('Effect of Log1p Transformation on the Amount Feature', fontsize=16, weight='bold')
plt.show()
print("Displayed Histograms for Original and Log1p-Transformed Amount.\n")


# --- Task 5: Compute and visualize pairwise correlation matrix ---
print("--- Task 5: Visualizing Pairwise Numerical Feature Correlation ---")
# Select numerical features, using the log1p-transformed 'amount'
numerical_features_for_corr = ['log_amount', 'age', 'duration_months']
correlation_matrix = df[numerical_features_for_corr].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, linecolor='black')
plt.title('Pairwise Correlation Matrix of Numerical Features (using Log1p Amount)', fontsize=14, weight='bold')
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(rotation=0, fontsize=10)
plt.colorbar(label='Correlation Coefficient')
plt.tight_layout()
plt.show()
print("Displayed Heatmap of Pairwise Correlation Matrix.")
print("-" * 50)