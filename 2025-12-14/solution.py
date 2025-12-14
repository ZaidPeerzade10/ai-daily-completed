import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_synthetic_data():
    """
    Generates a pandas DataFrame with synthetic transaction data.
    """
    num_transactions = 10000
    num_customers = np.random.randint(5, 11)  # 5 to 10 unique customers
    num_product_categories = np.random.randint(3, 6) # 3 to 5 unique categories

    customer_ids = [f'CUST_{i:03d}' for i in range(1, num_customers + 1)]
    product_categories = [f'Category_{chr(65+i)}' for i in range(num_product_categories)]

    # Generate dates spanning 6-12 months
    end_date = datetime.now()
    start_date = end_date - timedelta(days=np.random.randint(180, 366)) # 6 to 12 months

    # Generate transaction dates (as a list to avoid TypeError with generator)
    date_range_days = (end_date - start_date).days
    transaction_dates_list = [
        start_date + timedelta(days=np.random.randint(0, date_range_days))
        for _ in range(num_transactions)
    ]

    data = {
        'customer_id': np.random.choice(customer_ids, num_transactions),
        'transaction_date': transaction_dates_list,
        'amount': np.random.uniform(5.0, 500.0, num_transactions),
        'product_category': np.random.choice(product_categories, num_transactions)
    }

    df = pd.DataFrame(data)
    # Ensure transaction_date is datetime type
    df['transaction_date'] = pd.to_datetime(df['transaction_date'])

    return df

def main():
    # 1. Generate synthetic transaction data
    df = generate_synthetic_data()
    print("--- Original DataFrame Head ---")
    print(df.head())
    print(f"\nTotal transactions: {len(df)}")
    print(f"Unique customers: {df['customer_id'].nunique()}")
    print(f"Date range: {df['transaction_date'].min().date()} to {df['transaction_date'].max().date()}")

    # 2. Prepare Data for Window Functions
    # Sort by customer_id and transaction_date for correct window function application
    df = df.sort_values(by=['customer_id', 'transaction_date']).reset_index(drop=True)

    # 3. Calculate `customer_30d_avg_spend` Feature
    # For each customer, calculate the average amount over the past 30 days, inclusive of current date.
    # The 'on' parameter specifies the column to use for the time window.
    # 'closed=None' (default for 'on') means the right boundary is inclusive.
    df['customer_30d_avg_spend'] = df.groupby('customer_id')['amount'].transform(
        lambda x: x.rolling(window='30D', on=df.loc[x.index, 'transaction_date']).mean()
    )

    # 4. Calculate `customer_cumulative_transactions` Feature
    # Running total count of transactions for each customer.
    df['customer_cumulative_transactions'] = df.groupby('customer_id').cumcount() + 1

    print("\n--- DataFrame with New Features (Head) ---")
    print(df.head())

    # Extract month for aggregation
    df['transaction_month'] = df['transaction_date'].dt.to_period('M')

    # 5. Aggregate Monthly Customer Spending
    # Total amount spent by each customer_id for each month.
    monthly_customer_spending = df.groupby(['customer_id', 'transaction_month'])['amount'].sum().reset_index()
    monthly_customer_spending = monthly_customer_spending.sort_values(by=['customer_id', 'transaction_month'])

    print("\n--- Monthly Total Spending per Customer (Head) ---")
    print(monthly_customer_spending.head())

    # 6. Aggregate Top Product Category Per Month
    # Find the product_category with the highest total amount spent across all customers for each month.
    monthly_category_spending = df.groupby(['transaction_month', 'product_category'])['amount'].sum().reset_index()

    # Get the product category with the max amount for each month
    top_product_category_per_month = monthly_category_spending.loc[
        monthly_category_spending.groupby('transaction_month')['amount'].idxmax()
    ].sort_values(by='transaction_month')

    print("\n--- Top Product Category by Total Amount per Month ---")
    print(top_product_category_per_month)

if __name__ == "__main__":
    main()