import sqlite3
import pandas as pd
import numpy as np
import datetime
import random

def run_sql_analytics():
    """
    Creates an in-memory SQLite database, populates it with synthetic transaction data,
    performs SQL analytics using window functions, and displays results in a pandas DataFrame.
    """

    # 1. Create an in-memory SQLite database
    conn = sqlite3.connect(':memory:')
    cursor = conn.cursor()

    print("1. In-memory SQLite database created.")

    # 2. Create the transactions table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS transactions (
            transaction_id INTEGER PRIMARY KEY AUTOINCREMENT,
            customer_id INTEGER,
            product_id INTEGER,
            transaction_date TEXT, -- YYYY-MM-DD format
            amount REAL
        )
    ''')
    conn.commit()
    print("2. 'transactions' table created.")

    # 3. Insert synthetic data into the transactions table
    num_customers = 5
    num_products = 3
    num_transactions = random.randint(25, 35) # 20-30 transactions

    customer_ids = list(range(1, num_customers + 1))
    product_ids = list(range(1, num_products + 1))

    transactions_data = []
    start_date = datetime.date(2023, 10, 1)
    end_date = datetime.date(2024, 1, 31)
    
    date_range = [start_date + datetime.timedelta(days=x) for x in range((end_date - start_date).days + 1)]

    for i in range(1, num_transactions + 1):
        customer_id = random.choice(customer_ids)
        product_id = random.choice(product_ids)
        transaction_date = random.choice(date_range).strftime('%Y-%m-%d')
        amount = round(random.uniform(10.0, 500.0), 2)
        transactions_data.append((customer_id, product_id, transaction_date, amount))

    cursor.executemany('''
        INSERT INTO transactions (customer_id, product_id, transaction_date, amount)
        VALUES (?, ?, ?, ?)
    ''', transactions_data)
    conn.commit()
    print(f"3. Inserted {len(transactions_data)} synthetic transactions.")

    # 4. Write a single SQL query with window functions
    sql_query = """
    SELECT
        transaction_id,
        customer_id,
        product_id,
        transaction_date,
        amount,
        SUM(amount) OVER (
            PARTITION BY customer_id, STRFTIME('%Y-%m', transaction_date)
        ) AS customer_monthly_total,
        AVG(amount) OVER (
            PARTITION BY customer_id, STRFTIME('%Y-%m', transaction_date)
        ) AS customer_monthly_avg_transaction,
        SUM(amount) OVER (
            PARTITION BY customer_id
            ORDER BY transaction_date, transaction_id -- Adding transaction_id for stable order in case of same date
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) AS customer_cumulative_total
    FROM
        transactions
    ORDER BY
        customer_id, transaction_date, transaction_id;
    """
    print("\n4. Executing SQL query with window functions...")

    # 5. Retrieve the results of this SQL query
    cursor.execute(sql_query)
    results = cursor.fetchall()
    column_names = [description[0] for description in cursor.description]
    print("SQL query executed and results fetched.")

    # 6. Retrieve results into a pandas DataFrame
    df = pd.DataFrame(results, columns=column_names)
    print("\n6. Results loaded into pandas DataFrame.")

    # 7. Display selected DataFrame columns
    print("\n7. Head of the DataFrame with calculated window functions:")
    display_cols = [
        'transaction_date', 'customer_id', 'amount',
        'customer_monthly_total', 'customer_monthly_avg_transaction',
        'customer_cumulative_total'
    ]
    print(df[display_cols].head(10).to_string(index=False))

    # Optional: print some unique months and customer-monthly totals to verify
    print("\nVerification Example (first 5 unique customer_id-month combinations):")
    df['year_month'] = df['transaction_date'].apply(lambda x: x[:7])
    
    unique_combinations = df[['customer_id', 'year_month']].drop_duplicates().head(5)

    for _, row in unique_combinations.iterrows():
        customer = row['customer_id']
        month = row['year_month']
        
        # Get the first entry for this customer and month, which will have the correct monthly total
        filtered_df = df[(df['customer_id'] == customer) & (df['year_month'] == month)]
        if not filtered_df.empty:
            monthly_total = filtered_df['customer_monthly_total'].iloc[0]
            monthly_avg = filtered_df['customer_monthly_avg_transaction'].iloc[0]
            actual_sum = filtered_df['amount'].sum()
            actual_avg = filtered_df['amount'].mean()
            print(f"  Customer {customer}, Month {month}:")
            print(f"    Calculated Monthly Total: {monthly_total:.2f} (Actual Sum: {actual_sum:.2f})")
            print(f"    Calculated Monthly Avg: {monthly_avg:.2f} (Actual Avg: {actual_avg:.2f})")
    
    df.drop(columns=['year_month'], inplace=True) # Clean up temporary column


    # 8. Close database connection
    conn.close()
    print("\n8. SQLite database connection closed.")

if __name__ == "__main__":
    run_sql_analytics()