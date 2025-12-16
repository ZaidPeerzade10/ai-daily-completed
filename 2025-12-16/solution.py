import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score

def main():
    # 1. Generate a pandas DataFrame with synthetic time-series data
    start_date = '2020-01-01'
    n_days = 3 * 365  # 3 years of daily data
    
    dates = pd.date_range(start=start_date, periods=n_days, freq='D')
    
    # Synthetic 'value' with linear trend, seasonality, and noise
    trend = np.linspace(0, 100, n_days)
    seasonality = 15 * np.sin(np.arange(n_days) / 30 * 2 * np.pi) + 10 * np.sin(np.arange(n_days) / 365 * 2 * np.pi)
    noise = np.random.normal(0, 5, n_days)
    
    value = trend + seasonality + noise
    
    # Additional numerical features
    feature_A = np.random.rand(n_days) * 50
    feature_B = np.random.rand(n_days) * 100 + 20
    
    df = pd.DataFrame({
        'date': dates,
        'value': value,
        'feature_A': feature_A,
        'feature_B': feature_B
    })
    
    # Ensure DataFrame is sorted by date for time-series operations
    df.sort_values(by='date', inplace=True)
    
    print("--- Original DataFrame Head ---")
    print(df.head())
    print("\nDataFrame Info:")
    df.info()

    # 2. Create new features using pandas operations
    # lag_1_value: The value from the previous day
    df['lag_1_value'] = df['value'].shift(1)
    
    # rolling_7d_mean_feature_A: A 7-day trailing rolling mean of feature_A
    df['rolling_7d_mean_feature_A'] = df['feature_A'].rolling(window=7, min_periods=1).mean() # min_periods=1 to allow calculation for first 6 days

    # day_of_week_num: Numerical day of the week (0-6, Monday=0, Sunday=6)
    df['day_of_week_num'] = df['date'].dt.dayofweek
    
    # month_num: Numerical month (1-12)
    df['month_num'] = df['date'].dt.month

    print("\n--- DataFrame with Engineered Features Head ---")
    print(df.head(10)) # Print more rows to show lag and rolling features taking effect

    # 3. Handle any NaN values introduced by lag/rolling features
    initial_rows_before_dropna = df.shape[0]
    df.dropna(inplace=True)
    rows_dropped = initial_rows_before_dropna - df.shape[0]
    print(f"\nDropped {rows_dropped} rows containing NaN values.")
    print(f"DataFrame shape after dropping NaNs: {df.shape}")
    print("\n--- DataFrame after Dropping NaNs Head ---")
    print(df.head())
    
    # Define features and target
    target_column = 'value'
    feature_columns = [
        'feature_A', 
        'feature_B', 
        'lag_1_value', 
        'rolling_7d_mean_feature_A', 
        'day_of_week_num', 
        'month_num'
    ]
    
    X = df[feature_columns]
    y = df[target_column]

    # 4. Split the dataset into training and testing sets based on time
    # Use the first 80% of data for training and the remaining 20% for testing
    split_index = int(len(df) * 0.8)
    
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
    
    print(f"\nTraining set size: {len(X_train)} samples")
    print(f"Testing set size: {len(X_test)} samples")

    # 5. Construct an sklearn.pipeline.Pipeline
    # All numerical features (original + engineered) need scaling
    numerical_features_for_scaling = feature_columns # All engineered features are numerical

    pipeline = Pipeline([
        ('scaler', StandardScaler()),  # Scale numerical features
        ('regressor', Ridge(random_state=42)) # Ridge regressor
    ])

    # 6. Train the pipeline and evaluate its performance
    print("\n--- Training the Pipeline ---")
    pipeline.fit(X_train, y_train)
    print("Pipeline training complete.")

    print("\n--- Evaluating Model Performance ---")
    y_pred = pipeline.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Mean Absolute Error (MAE) on test set: {mae:.2f}")
    print(f"R-squared (R2) score on test set: {r2:.2f}")

if __name__ == "__main__":
    main()