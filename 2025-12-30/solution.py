import numpy as np
import pandas as pd # Included as per general requirements, though not strictly used in this specific script
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# 1. Generate synthetic regression dataset and introduce non-linear relationship
# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Generate original dataset
X_original, y_original = make_regression(
    n_samples=1000,
    n_features=3,
    n_informative=3,
    noise=5,
    random_state=42
)

# Introduce a non-linear relationship: y = y_original + 2 * X[:, 0]^2 + noise
y = y_original + 2 * X_original[:, 0]**2 + np.random.normal(0, 0.5, size=1000)

print("Dataset generated:")
print(f"X shape: {X_original.shape}")
print(f"y shape: {y.shape}")
print(f"First 5 samples of X:\n{X_original[:5]}")
print(f"First 5 samples of y:\n{y[:5]}\n")

# 2. Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_original, y, test_size=0.2, random_state=42
)

print(f"Training data shape: X_train={X_train.shape}, y_train={y_train.shape}")
print(f"Testing data shape: X_test={X_test.shape}, y_test={y_test.shape}\n")

# 3. Apply StandardScaler to the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Features scaled using StandardScaler.\n")

# 4. Build a simple sequential neural network model
model = Sequential([
    Dense(32, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dense(1, activation='linear') # Output layer for regression
])

print("Neural Network Model Summary:")
model.summary()
print("\n")

# 5. Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

print("Model compiled with Adam optimizer and Mean Squared Error loss.\n")

# 6. Train the model
history = model.fit(
    X_train_scaled, y_train,
    epochs=100,
    batch_size=32,
    verbose=0 # Suppress verbose output during training
)

print(f"Model trained for {len(history.history['loss'])} epochs.\n")

# 7. Evaluate the trained model's performance on the scaled test set
loss = model.evaluate(X_test_scaled, y_test, verbose=0)
print(f"Mean Squared Error on the test set: {loss:.4f}\n")

# 8. Visualize the model's predictions against the actual test target values
y_pred = model.predict(X_test_scaled).flatten() # Flatten predictions for plotting

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect Prediction (y=x)')
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Model Predictions vs. Actual Values on Test Set")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()