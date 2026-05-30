Here are the implementation steps for developing the machine learning pipeline:

1.  **Synthetic Data Generation and Database Setup:**
    *   Generate two pandas DataFrames: `products_df` (1000-1500 rows) with `product_id`, `category`, `brand`, `base_price`, `launch_date`, and `sales_df` (20000-30000 rows) with `sale_id`, `product_id`, `sale_date`, `quantity_sold`.
    *   Ensure `sale_date` for each sale is after its corresponding `launch_date` and up to approximately two weeks prior to the current date.
    *   Implement realistic sales patterns: some products with high demand, some low, and some with mild seasonality. Vary `quantity_sold` appropriately. Simulate higher sales for certain categories/brands and a gradual decrease in sales for older products.
    *   Sort `sales_df` by `product_id` then `sale_date`.
    *   Initialize an in-memory SQLite database connection.
    *   Load `products_df` into a SQL table named `products` and `sales_df` into a table named `sales`.

2.  **SQL-Based Historical Feature Engineering:**
    *   Define `GLOBAL_PREDICTION_CUTOFF_DATE` as two months prior to the latest `sale_date` in the generated `sales_df`.
    *   Write and execute a single SQL query to perform the following for *each product*, joining `products` and `sales` tables:
        *   Include static product attributes: `product_id`, `category`, `brand`, `base_price`, `launch_date`.
        *   Calculate `current_cutoff_date` (the `GLOBAL_PREDICTION_CUTOFF_DATE` itself).
        *   Aggregate sales data for the 30 days *immediately preceding* `GLOBAL_PREDICTION_CUTOFF_DATE`:
            *   `avg_qty_sold_prev_30d`: Average quantity sold.
            *   `total_qty_sold_prev_30d`: Sum of quantity sold.
            *   `num_sales_prev_30d`: Count of sales.
        *   Calculate `days_since_last_sale_at_cutoff`: Number of days between `GLOBAL_PREDICTION_CUTOFF_DATE` and the most recent `sale_date` *on or before* the cutoff. Return a large default value (e.g., 9999) if no sales occurred before the cutoff for a product.
    *   Ensure the query uses `LEFT JOIN` from `products` to `sales` to include all products, even those with no sales in the specified window, returning 0/0.0 for aggregated metrics where no sales occurred.

3.  **Pandas Feature Engineering and Target Creation:**
    *   Fetch the results of the SQL query into a pandas DataFrame, `product_features_df`.
    *   Convert `launch_date` and `current_cutoff_date` columns to datetime objects.
    *   Handle missing values in `product_features_df`: Fill numerical aggregated features (e.g., `avg_qty_sold_prev_30d`, `total_qty_sold_prev_30d`, `num_sales_prev_30d`) with 0 or 0.0. Fill `days_since_last_sale_at_cutoff` with 9999. Fill `base_price` NaNs with the median `base_price`.
    *   Calculate `product_age_at_cutoff_days`: The difference in days between `current_cutoff_date` and `launch_date`.
    *   **Create the `next_14d_demand_category` target:** For each `product_id`, calculate the sum of `quantity_sold` from `sales_df` for all sales occurring *after* `current_cutoff_date` and *on or before* `current_cutoff_date + pd.Timedelta(days=14)`. This sum is `next_14d_total_qty_sold`.
    *   Merge `next_14d_total_qty_sold` into `product_features_df`, filling `NaN`s with 0 for products with no sales in the target window.
    *   Categorize `next_14d_total_qty_sold` into 'Low' (<=10), 'Medium' (10-50), and 'High' (>50) for the `next_14d_demand_category` target. Adjust thresholds as needed for class balance.
    *   Define feature matrix `X` (numerical: `base_price`, `avg_qty_sold_prev_30d`, `total_qty_sold_prev_30d`, `num_sales_prev_30d`, `days_since_last_sale_at_cutoff`, `product_age_at_cutoff_days`; categorical: `category`, `brand`) and target vector `y` (`next_14d_demand_category`).
    *   Split `X` and `y` into training and testing sets (e.g., 70/30) using `train_test_split`, ensuring `random_state=42` and `stratify=y` for balanced class distribution.

4.  **Exploratory Data Visualization:**
    *   Generate a violin plot (or box plot) showing the distribution of `total_qty_sold_prev_30d` for each of the `next_14d_demand_category` classes ('Low', 'Medium', 'High'). Add appropriate labels and a descriptive title.
    *   Create a stacked bar chart illustrating the proportional breakdown of `next_14d_demand_category` (across 'Low', 'Medium', 'High') for each unique `category` in `products_df`. Ensure clear labels and a title.

5.  **Machine Learning Pipeline, Training, and Evaluation:**
    *   Construct an `sklearn.pipeline.Pipeline` starting with a `sklearn.compose.ColumnTransformer` for preprocessing:
        *   For numerical features, apply a `SimpleImputer` with a 'mean' strategy, followed by a `StandardScaler`.
        *   For categorical features, apply a `OneHotEncoder` with `handle_unknown='ignore'`.
    *   Append `sklearn.ensemble.HistGradientBoostingClassifier` as the final estimator to the pipeline, setting `random_state=42`.
    *   Train the complete pipeline using the `X_train` and `y_train` datasets.
    *   Make predictions on the `X_test` dataset.
    *   Print the `sklearn.metrics.classification_report` to evaluate the model's performance on the test set, providing detailed metrics for each demand category.