Here are 5 clear implementation steps for a Python ML engineer to follow:

1.  **Data Acquisition and Preparation:** Generate a synthetic binary classification dataset using `sklearn.datasets.make_classification` with parameters like 1000 samples, 10 features (2 informative), and 2 classes. Subsequently, split this dataset into training and testing sets (e.g., 80% train, 20% test) using `sklearn.model_selection.train_test_split`, ensuring reproducibility by setting a `random_state`.

2.  **Model Instantiation and Training:** Initialize a `LogisticRegression` model from `sklearn.linear_model`. Train this model using the features and labels from your designated training set.

3.  **Prediction on Test Data:** Utilize the trained `LogisticRegression` model to generate two sets of outputs on the unseen test features: predicted class labels (0 or 1) using the `predict` method, and predicted probabilities for the positive class (class 1) using the `predict_proba` method.

4.  **Performance Metric Calculation and Display:** Calculate the following key evaluation metrics using the true test labels and your model's predictions/probabilities: Accuracy, Precision, Recall, F1-score, and ROC AUC score. Import these metric functions from `sklearn.metrics`. Print each of these calculated metrics clearly to the console.

5.  **ROC Curve Visualization:** Compute the False Positive Rates (FPR) and True Positive Rates (TPR) required for the Receiver Operating Characteristic (ROC) curve using the true test labels and the predicted positive class probabilities. Plot the ROC curve using `matplotlib.pyplot`, ensuring proper labeling of the X-axis ("False Positive Rate") and Y-axis ("True Positive Rate"), a descriptive title (e.g., "Receiver Operating Characteristic (ROC) Curve"), and include the previously calculated ROC AUC score within the plot's legend.