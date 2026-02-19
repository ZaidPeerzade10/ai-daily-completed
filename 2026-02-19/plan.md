Here are the implementation steps for the marketing campaign conversion prediction task:

1.  **Generate Synthetic Data:**
    *   Create `users_df` with 500-700 rows, including `user_id` (unique integers), `signup_date` (random dates over the last 5 years), `age` (18-70), `region` (e.g., 'North', 'South', 'East', 'West'), and `browsing_frequency_level` ('Low', 'Medium', 'High').
    *   Create `offers_df` with 50-100 rows, including `offer_id` (unique integers), `offer_type` (e.g., 'Discount_10', 'Free_Shipping', 'Bundle_Deal', 'Gift_Card'), `category_focus` (e.g., 'Electronics', 'Books', 'Clothing', 'HomeGoods', 'Services'), and `discount_percentage` (random floats 5.0-30.0).
    *   Create `campaign_exposures_df` with 5000-8000 rows. Populate `exposure_id` (unique integers), `user_id` (randomly sampled from `users_df`), and `offer_id` (randomly sampled from `offers_df`).
    *   Crucially, set `exposure_date` for each event to be *after* the respective user's `signup_date`.
    *   Simulate `was_converted` (binary 0/1) with an overall conversion rate of 5-10%, ensuring:
        *   Users with 'High' `browsing_frequency_level` have a higher probability of conversion.
        *   Specific `offer_type`s (e.g., 'Discount_10', 'Free_Shipping') have generally higher conversion rates.
        *   A subtle age-category correlation: younger users (e.g., < 35) are more likely to convert for 'Electronics' or 'Gaming' offers, while older users might prefer 'HomeGoods' or 'Services'.
    *   Sort `campaign_exposures_df` first by `user_id` and then by `exposure_date`.

2.  **Load Data into SQLite and Perform SQL Feature Engineering:**
    *   Establish an in-memory SQLite database connection.
    *   Load `users_df`, `offers_df`, and `campaign_exposures_df` into three separate tables named `users`, `offers`, and `exposures` respectively.
    *   Construct a single SQL query to join these three tables and calculate the following features for each exposure event using window functions:
        *   `user_prior_total_exposures`: Count of all preceding exposures for the same user.
        *   `user_prior_converted_exposures`: Count of preceding converted exposures for the same user.
        *   `user_prior_conversion_rate`: Ratio of prior converted to prior total exposures for the user (0.0 if no prior exposures).
        *   `days_since_last_user_exposure`: Days between the current and the user's most recent prior exposure. For a user's first exposure, calculate days between `signup_date` and `exposure_date`. Use `LAG()` combined with `COALESCE` for this.
        *   `offer_prior_total_exposures`: Count of all preceding exposures for the same offer (across all users).
        *   `offer_prior_converted_exposures`: Count of preceding converted exposures for the same offer.
        *   `offer_prior_conversion_rate`: Ratio of prior converted to prior total exposures for the offer (0.0 if no prior exposures).
    *   Include `exposure_id`, `user_id`, `offer_id`, `exposure_date`, `was_converted`, and relevant static attributes from `users` (`age`, `region`, `browsing_frequency_level`, `signup_date`) and `offers` (`offer_type`, `category_focus`, `discount_percentage`) in the final query result.

3.  **Pandas Post-SQL Processing and Data Preparation:**
    *   Fetch the results of the SQL query into a pandas DataFrame, `campaign_features_df`.
    *   Handle `NaN` values for the newly engineered prior features: fill `user_prior_total_exposures`, `user_prior_converted_exposures`, `offer_prior_total_exposures`, `offer_prior_converted_exposures` with 0. Fill `user_prior_conversion_rate` and `offer_prior_conversion_rate` with 0.0. Verify `days_since_last_user_exposure` is correctly handled for first exposures (as per SQL, if any `NaN`s remain, fill with a large sentinel value like 9999).
    *   Convert `signup_date` and `exposure_date` columns to pandas datetime objects.
    *   Calculate `user_account_age_at_exposure_days`: The difference in days between `exposure_date` and `signup_date` for each event.
    *   Create a binary feature `user_had_prior_conversion` (1 if `user_prior_converted_exposures > 0`, else 0).
    *   Define the feature set `X` (including numerical features like `age`, `discount_percentage`, `user_account_age_at_exposure_days`, all prior counts, rates, and `days_since_last_user_exposure`; and categorical features like `region`, `browsing_frequency_level`, `offer_type`, `category_focus`, `user_had_prior_conversion`) and the target variable `y` (`was_converted`).
    *   Split `X` and `y` into training and testing sets (e.g., 70/30 split) using `sklearn.model_selection.train_test_split`, ensuring `random_state=42` and `stratify` on `y` for class imbalance.

4.  **Data Visualization:**
    *   Generate a violin plot (or box plot) to compare the distribution of `discount_percentage` for events where `was_converted=0` versus `was_converted=1`. Label axes and add a descriptive title.
    *   Create a stacked bar chart to visualize the proportion of `was_converted` (0 or 1) across different `offer_type` values. Ensure clear labels, a legend, and a title.

5.  **Build and Evaluate Machine Learning Pipeline:**
    *   Construct an `sklearn.pipeline.Pipeline` that incorporates a `sklearn.compose.ColumnTransformer` for preprocessing:
        *   For numerical features (e.g., `age`, `discount_percentage`, `user_account_age_at_exposure_days`, `user_prior_total_exposures`, `user_prior_converted_exposures`, `user_prior_conversion_rate`, `days_since_last_user_exposure`, `offer_prior_total_exposures`, `offer_prior_converted_exposures`, `offer_prior_conversion_rate`), apply `sklearn.preprocessing.SimpleImputer(strategy='mean')` followed by `sklearn.preprocessing.StandardScaler`.
        *   For categorical features (e.g., `region`, `browsing_frequency_level`, `offer_type`, `category_focus`, `user_had_prior_conversion`), apply `sklearn.preprocessing.OneHotEncoder(handle_unknown='ignore')`.
    *   Append `sklearn.ensemble.HistGradientBoostingClassifier` as the final estimator in the pipeline, setting `random_state=42`.
    *   Train the complete pipeline using the `X_train` and `y_train` datasets.
    *   Predict the probabilities for the positive class (class 1) on the `X_test` dataset.
    *   Calculate and print the `sklearn.metrics.roc_auc_score` and a detailed `sklearn.metrics.classification_report` using the test set predictions and true labels.