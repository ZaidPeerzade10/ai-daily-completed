Here are the implementation steps for the given task:

1.  **Generate Synthetic Time-Series Data:**
    *   Create a pandas DataFrame with a `date` column, starting from '2020-01-01' and extending for 2 to 3 years of daily data.
    *   Populate a `value` column with synthetic time-series data: incorporate a linear trend, add daily seasonality using a sine wave function (e.g., `np.sin`), and introduce random noise.
    *   Add two additional numerical features, `feature_A` and `feature_B`, populating them with random numerical values.
    *   Ensure the DataFrame is explicitly sorted by the `date` column to prepare for time-series operations.

2.  **Engineer New Features using Pandas:**
    *   Create a new column `lag_1_value` by shifting the `value` column one period (day) back.
    *   Calculate `rolling_7d_mean_feature_A` as the 7-day rolling mean of `feature_A`, ensuring the rolling window is properly defined (e.g., centered or trailing).
    *   Extract the numerical day of the week (0 for Monday, 6 for Sunday, or similar) from the `date` column and store it as `day_of_week_num`.
    *   Extract the numerical month (1-12) from the `date` column and store it as `month_num`.

3.  **Handle Missing Values and Identify Features/Target:**
    *   Drop any rows containing `NaN` values that were introduced at the beginning of the DataFrame by the lag and rolling window feature engineering operations.
    *   Define the target variable (`y`) as the `value` column from the cleaned DataFrame.
    *   Identify all numerical feature columns (`X`) for the model. This includes `feature_A`, `feature_B`, `lag_1_value`, `rolling_7d_mean_feature_A`, `day_of_week_num`, and `month_num`. Exclude the original `date` and `value` columns from `X`.

4.  **Split Data into Training and Testing Sets:**
    *   Split the DataFrame chronologically into training and testing sets. For instance, allocate the first 80% of the data for training and the remaining 20% for testing.
    *   Separate the features (`X_train`, `X_test`) and the target (`y_train`, `y_test`) for both the training and testing portions.

5.  **Construct Scikit-learn Pipeline:**
    *   Create an `sklearn.pipeline.Pipeline` that encapsulates the data preprocessing and the model.
    *   The first step in the pipeline should be a `StandardScaler` to scale all numerical features.
    *   The second step should be a `Ridge` regressor, which will serve as the final predictive model.

6.  **Train and Evaluate the Model:**
    *   Fit the constructed `sklearn.pipeline.Pipeline` using the training features (`X_train`) and the training target (`y_train`).
    *   Generate predictions on the test set features (`X_test`) using the trained pipeline.
    *   Calculate and report the Mean Absolute Error (MAE) and the R-squared score (`r2_score`) by comparing the pipeline's predictions against the actual test target values (`y_test`).