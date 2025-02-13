# Imports
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load CSV files
neutral_df = pd.read_csv("neutral_bicep_final.csv")
correct_df = pd.read_csv("correct_bicep_final.csv")
wrong_df = pd.read_csv("wrong_bicep_final.csv")

# Parameters
X = []
y = []
no_of_timesteps = 20

# Data processing function
def process_data(df, label):
    datasets = df.iloc[:, :].values  # Extract all columns as numpy array
    n_samples = len(datasets)
    for i in range(no_of_timesteps, n_samples):
        X.append(datasets[i-no_of_timesteps:i, :])
        y.append(label)

# Process each dataset
process_data(neutral_df, 0)  # Neutral posture labeled as 0
process_data(correct_df, 1)   # Correct overhead press labeled as 1
process_data(wrong_df, 2)   # Wrong overhead press labeled as 2

# Convert to numpy arrays
X, y = np.array(X), np.array(y)
print("Shape of X:", X.shape)  # (Samples, Timesteps, Features)
print("Shape of y:", y.shape)  # (Samples,)

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the model
model = Sequential()
model.add(LSTM(units=64, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.3))
model.add(LSTM(units=64))
model.add(Dropout(0.3))
model.add(Dense(units=3, activation="softmax"))

# Compile the model
model.compile(optimizer="adam", metrics=["accuracy"], loss="sparse_categorical_crossentropy")

# EarlyStopping callback
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# Train the model
history = model.fit(
    X_train, y_train,
    epochs=15,  # Train for up to 15 epochs
    batch_size=28,
    validation_data=(X_val, y_val),
    callbacks=[early_stopping],
    verbose=1
)

# Evaluate the model
val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")
print(f"Validation Loss: {val_loss:.4f}")

# Confusion Matrix
y_pred = model.predict(X_val, verbose=0).argmax(axis=1)
cm = confusion_matrix(y_val, y_pred)

# Save the model
model.save("lstm_Bicep_curls_model.keras", save_format='keras')
print("Model saved to 'lstm_Bicep_curls_model.keras'")

# Plot Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Neutral', 'Correct', 'Wrong'], yticklabels=['Neutral', 'Correct', 'Wrong'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

# Plot Accuracy and Loss
plt.figure(figsize=(12, 6))
plt.plot(history.history['accuracy'], label="Training Accuracy")
plt.plot(history.history['val_accuracy'], label="Validation Accuracy")
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label="Training Loss")
plt.plot(history.history['val_loss'], label="Validation Loss")
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.show()
