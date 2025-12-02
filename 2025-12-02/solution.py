import sqlite3
import pandas as pd
import numpy as np # Included as per the general requirements, though not explicitly used for this task's core logic.

# 1. Establish In-Memory SQLite Database Connection
# An in-memory database exists only for the duration of the connection.
conn = sqlite3.connect(':memory:')
cursor = conn.cursor()

print("--- Initializing Database ---")

# 2. Create two tables: customers and orders
# customers table
cursor.execute('''
    CREATE TABLE customers (
        customer_id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        city TEXT
    );
''')
print("Table 'customers' created.")

# orders table with a foreign key constraint
cursor.execute('''
    CREATE TABLE orders (
        order_id INTEGER PRIMARY KEY,
        customer_id INTEGER,
        amount REAL NOT NULL,
        order_date TEXT,
        FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
    );
''')
print("Table 'orders' created with foreign key to 'customers'.")
conn.commit() # Commit the table creation

# 3. Insert sample data into both tables
print("\n--- Inserting Sample Data ---")

# Sample customers data (at least 5 distinct customers)
customers_data = [
    (1, 'Alice Smith', 'New York'),
    (2, 'Bob Johnson', 'Los Angeles'),
    (3, 'Charlie Brown', 'Chicago'),
    (4, 'Diana Prince', 'New York'),
    (5, 'Eve Adams', 'Houston'),
    (6, 'Frank White', 'Miami') # Added an extra customer for more variety
]
cursor.executemany("INSERT INTO customers (customer_id, name, city) VALUES (?, ?, ?)", customers_data)
print(f"Inserted {len(customers_data)} customer records.")

# Sample orders data (10-15 orders, ensuring some customers have multiple orders)
orders_data = [
    (101, 1, 150.00, '2023-01-10'),
    (102, 2, 250.50, '2023-01-15'),
    (103, 1, 75.25, '2023-01-20'), # Alice Smith's second order
    (104, 3, 300.00, '2023-02-01'),
    (105, 2, 120.75, '2023-02-05'), # Bob Johnson's second order
    (106, 4, 500.00, '2023-02-10'),
    (107, 1, 99.99, '2023-02-12'), # Alice Smith's third order
    (108, 5, 45.00, '2023-02-15'),
    (109, 3, 180.00, '2023-02-20'), # Charlie Brown's second order
    (110, 2, 35.50, '2023-02-25'), # Bob Johnson's third order
    (111, 4, 110.00, '2023-03-01'), # Diana Prince's second order
    (112, 6, 600.00, '2023-03-05'), # Frank White's order
    (113, 5, 85.00, '2023-03-10'), # Eve Adams' second order
    (114, 1, 200.00, '2023-03-15'), # Alice Smith's fourth order
    (115, 3, 125.00, '2023-03-20') # Charlie Brown's third order
]
cursor.executemany("INSERT INTO orders (order_id, customer_id, amount, order_date) VALUES (?, ?, ?, ?)", orders_data)
print(f"Inserted {len(orders_data)} order records.")
conn.commit() # Commit the data insertion

# 4. Using a single SQL query, calculate total purchase amount for each customer
#    Retrieving customer's name and their total_revenue.
print("\n--- Executing SQL Analytics Query ---")
sql_analytics_query = """
SELECT
    c.name,
    SUM(o.amount) AS total_revenue
FROM
    customers c
JOIN
    orders o ON c.customer_id = o.customer_id
GROUP BY
    c.name
ORDER BY
    total_revenue DESC;
"""

# 5. Retrieve these aggregated results directly into a pandas DataFrame
df_customer_revenue = pd.read_sql_query(sql_analytics_query, conn)

print("\n--- All Customer Total Revenues ---")
print(df_customer_revenue.to_string(index=False)) # .to_string(index=False) for cleaner output without pandas index

# 6. Display the top 3 customers by their total_revenue
print("\n--- Top 3 Customers by Total Revenue ---")
top_3_customers = df_customer_revenue.head(3)
print(top_3_customers.to_string(index=False))

# Close the database connection
conn.close()
print("\n--- Database connection closed. ---")