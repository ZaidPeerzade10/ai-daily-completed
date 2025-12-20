import sqlite3
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

def generate_synthetic_data():
    """Generates synthetic data for customers and orders."""
    # Customer data
    num_customers = 7
    regions = ['North', 'South', 'East', 'West', 'Central']
    customer_names = [f'Customer_{i}' for i in range(1, num_customers + 1)]
    customer_data = []
    for i in range(num_customers):
        customer_data.append({
            'customer_id': i + 1,
            'name': customer_names[i],
            'region': random.choice(regions)
        })

    # Order data
    num_orders = 25
    start_date = datetime(2023, 1, 1)
    order_data = []
    for i in range(num_orders):
        order_date = start_date + timedelta(days=random.randint(0, 89)) # Orders over ~3 months
        order_data.append({
            'order_id': i + 1,
            'customer_id': random.randint(1, num_customers),
            'order_date': order_date.strftime('%Y-%m-%d'),
            'total_amount': round(random.uniform(20.0, 1500.0), 2)
        })
    return customer_data, order_data

def main():
    """
    Main function to set up DB, insert data, run query, and display results.
    """
    # 1. Initialize Database Connection (in-memory)
    conn = sqlite3.connect(':memory:')
    cursor = conn.cursor()
    print("1. In-memory SQLite database connection established.")

    # 2. Create Tables
    cursor.execute('''
        CREATE TABLE customers (
            customer_id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            region TEXT NOT NULL
        );
    ''')
    cursor.execute('''
        CREATE TABLE orders (
            order_id INTEGER PRIMARY KEY,
            customer_id INTEGER NOT NULL,
            order_date TEXT NOT NULL,
            total_amount REAL NOT NULL,
            FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
        );
    ''')
    print("2. 'customers' and 'orders' tables created.")

    # 3. Insert Synthetic Data
    customer_data, order_data = generate_synthetic_data()

    # Insert customer data
    cursor.executemany(
        "INSERT INTO customers (customer_id, name, region) VALUES (?, ?, ?)",
        [(c['customer_id'], c['name'], c['region']) for c in customer_data]
    )
    # Insert order data
    cursor.executemany(
        "INSERT INTO orders (order_id, customer_id, order_date, total_amount) VALUES (?, ?, ?, ?)",
        [(o['order_id'], o['customer_id'], o['order_date'], o['total_amount']) for o in order_data]
    )
    conn.commit()
    print(f"3. Inserted {len(customer_data)} customers and {len(order_data)} orders.")

    # 4. Write a single SQL query
    # Find customers whose total_spent is greater than the average_order_value_per_region
    analytics_query = """
    WITH CustomerSpend AS (
        SELECT
            c.customer_id,
            c.name,
            c.region,
            SUM(o.total_amount) AS total_spent
        FROM
            customers c
        JOIN
            orders o ON c.customer_id = o.customer_id
        GROUP BY
            c.customer_id, c.name, c.region
    ),
    RegionalAverageOrderValue AS (
        SELECT
            c.region,
            AVG(o.total_amount) AS average_order_value_per_region
        FROM
            customers c
        JOIN
            orders o ON c.customer_id = o.customer_id
        GROUP BY
            c.region
    )
    SELECT
        cs.customer_id,
        cs.name,
        cs.region,
        cs.total_spent
    FROM
        CustomerSpend cs
    JOIN
        RegionalAverageOrderValue raov ON cs.region = raov.region
    WHERE
        cs.total_spent > raov.average_order_value_per_region
    ORDER BY
        cs.total_spent DESC;
    """
    print("\n4. Executing SQL analytics query...")

    # 5. Retrieve the results into a pandas DataFrame and display
    df_results = pd.read_sql_query(analytics_query, conn)
    print("\n5. Results for customers with total_spent > average_order_value_per_region:")
    print(df_results.head(10)) # Display top 10 results

    # Close the database connection
    conn.close()
    print("\nDatabase connection closed.")

if __name__ == "__main__":
    main()