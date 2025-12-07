import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def generate_and_analyze_sales_data():
    """
    Generates synthetic sales data, engineers time-based features,
    aggregates data, and visualizes trends.
    """
    print("--- Starting Data Generation and Analysis Script ---")

    # 1. Generate the Base Time-Series DataFrame
    start_date = '2020-01-01'
    end_date = '2022-12-31' # Approximately 3 years
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    df = pd.DataFrame({'timestamp': date_range})

    # Generate synthetic 'value' (sales data)
    num_days = len(df)
    
    # Base value
    base_sales = 1000

    # Linear trend
    trend = np.linspace(0, 500, num_days)

    # Yearly seasonality (using sine wave)
    # Day of year ranges from 1 to 365/366. Cycle length 365.25 days.
    day_of_year_radians = (df['timestamp'].dt.dayofyear / 365.25) * 2 * np.pi
    yearly_seasonality = 200 * np.sin(day_of_year_radians) + 50 * np.cos(day_of_year_radians * 2) # Add a secondary harmonic

    # Weekly seasonality (using sine wave)
    # Weekday ranges from 0 (Monday) to 6 (Sunday). Cycle length 7 days.
    day_of_week_radians = (df['timestamp'].dt.weekday / 7) * 2 * np.pi
    weekly_seasonality = 100 * np.sin(day_of_week_radians + np.pi/2) # Peak around Friday/Saturday

    # Random noise
    noise = np.random.normal(0, 75, num_days)

    # Combine components to create sales value
    # Ensure values are non-negative
    df['value'] = base_sales + trend + yearly_seasonality + weekly_seasonality + noise
    df['value'] = df['value'].apply(lambda x: max(50, x)) # Ensure a minimum sales value

    # 2. Engineer Time-Based Features
    df['month'] = df['timestamp'].dt.month
    df['day_of_week'] = df['timestamp'].dt.day_name()
    df['is_weekend'] = df['timestamp'].dt.weekday >= 5 # Monday=0, Sunday=6

    # Define the order for days of the week for correct plotting and aggregation
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    df['day_of_week'] = pd.Categorical(df['day_of_week'], categories=day_order, ordered=True)

    # 5. Display Key Data Structures (head of the DataFrame)
    print("\n--- Head of DataFrame with Engineered Features ---")
    print(df.head())

    # 3. Aggregate Data to Identify Trends
    # Calculate average value by month
    monthly_avg_value = df.groupby('month')['value'].mean().reset_index()

    # Calculate average value by day of week
    daily_avg_value = df.groupby('day_of_week')['value'].mean().reset_index()

    # 5. Display Key Data Structures (aggregated dataframes)
    print("\n--- Average Value by Month ---")
    print(monthly_avg_value)

    print("\n--- Average Value by Day of Week ---")
    print(daily_avg_value)

    # 4. Visualize Monthly and Weekly Trends
    plt.style.use('seaborn-v0_8-darkgrid')

    # Figure 1: Average value trend across months
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=monthly_avg_value, x='month', y='value', marker='o')
    plt.title('Average Value Trend Across Months')
    plt.xlabel('Month')
    plt.ylabel('Average Value')
    plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    plt.grid(True)
    plt.tight_layout()

    # Figure 2: Average value for each day of the week
    plt.figure(figsize=(10, 6))
    sns.barplot(data=daily_avg_value, x='day_of_week', y='value', palette='viridis')
    plt.title('Average Value by Day of Week')
    plt.xlabel('Day of Week')
    plt.ylabel('Average Value')
    plt.grid(axis='y')
    plt.tight_layout()

    plt.show()

    print("\n--- Script Finished ---")

if __name__ == "__main__":
    generate_and_analyze_sales_data()