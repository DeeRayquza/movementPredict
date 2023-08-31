import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, TimeDistributed, Dropout
from tensorflow.keras.callbacks import TensorBoard
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve, auc
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error

data_path = r'D:\MSc Project\Data\Old Data\MEG\s25\training_data.csv'

# DATA PREPROCESSING

# Define the headers
headers = ['Time', 'Alpha Power', 'Beta Power', 'Gamma Power', 'Trial', 'Event']

# Read the CSV file with headers
data_frame = pd.read_csv(data_path, header=None, names=headers)

# Update 'Event' column based on 'Time' column
for index, row in data_frame.iterrows():
    if row['Time'] == 0:
        data_frame.at[index, 'Cue'] = 1
    else:
        data_frame.at[index, 'Cue'] = 0

data_frame['Event'] = data_frame['Event'].apply(lambda x: 1 if x > 0 else 0)

#Class Imbalance

for i in range(len(data_frame)):
    if data_frame.at[i, 'Event'] == 1:
        if i - 1 >= 0:
            data_frame.at[i - 1, 'Event'] = 1
        if i - 2 >= 0:
            data_frame.at[i - 2, 'Event'] = 1
        if i - 3 >= 0:
            data_frame.at[i - 3, 'Event'] = 1

# Separate features and target
X = data_frame[['Time', 'Alpha Power', 'Beta Power', 'Gamma Power', 'Trial', 'Cue']].values
y = data_frame['Event'].values

# Segment data into trials
unique_trials = np.unique(X[:, -1])
segmented_data = []
for trial in unique_trials:
    trial_indices = np.where(X[:, -1] == trial)
    trial_X = X[trial_indices]
    trial_y = y[trial_indices]
    segmented_data.append((trial_X, trial_y))

# Split data into training and validation sets
train_data, val_data = train_test_split(segmented_data, test_size=0.2, random_state=42)

# Standardize features using StandardScaler
scaler = StandardScaler()
train_data_scaled = [(scaler.fit_transform(X), y) for X, y in train_data]
val_data_scaled = [(scaler.transform(X), y) for X, y in val_data]

log_dir = "logs/"  # Change this to your desired log directory
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

# Reshape X to have a fixed sequence length
sequence_length = 101  # Adjust as needed based on experimentation
num_features = X.shape[1] - 1  # Exclude the 'Trial' column

X_reshaped = []
y_reshaped = []

for i in range(0, len(X) - sequence_length + 1, sequence_length):
    X_reshaped.append(X[i:i + sequence_length, :-1])  # Exclude the 'Trial' column
    y_reshaped.append(y[i:i + sequence_length])

X_reshaped = np.array(X_reshaped)
y_reshaped = np.array(y_reshaped)

print("New shape of X:", X_reshaped.shape)
print("New shape of y:", y_reshaped.shape)

# Split data into training, validation, and testing sets
train_split = 0.6
val_split = 0.2
test_split = 0.2

# Calculate split indices
train_index = int(len(X_reshaped) * train_split)
val_index = train_index + int(len(X_reshaped) * val_split)

X_train, y_train = X_reshaped[:train_index], y_reshaped[:train_index]
X_val, y_val = X_reshaped[train_index:val_index], y_reshaped[train_index:val_index]
X_test, y_test = X_reshaped[val_index:], y_reshaped[val_index:]

# Define the model architecture
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(sequence_length, num_features)),
    LSTM(32, return_sequences=True),
    TimeDistributed(Dense(1, activation='sigmoid'))
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model using the training data
history = model.fit(X_train, y_train, epochs=10, batch_size=32,
                    validation_data=(X_val, y_val), callbacks=[tensorboard_callback])


# Evaluate the model on the testing set
loss, accuracy = model.evaluate(X_test, y_test, batch_size=32)
print("Test loss:", loss)
print("Test accuracy:", accuracy * 100, "%")

#model.save('saved_model', save_format='tf')

# Generate predictions on the testing set
predicted_outputs = model.predict(X_test)

# Calculate predictions rounded to binary values
binary_predicted_outputs = np.round(predicted_outputs)

# Calculate evaluation metrics
accuracy = accuracy_score(y_test.flatten(), binary_predicted_outputs.flatten())
precision = precision_score(y_test.flatten(), binary_predicted_outputs.flatten())
recall = recall_score(y_test.flatten(), binary_predicted_outputs.flatten())
f1 = f1_score(y_test.flatten(), binary_predicted_outputs.flatten())
mse = mean_squared_error(y_test.flatten(), predicted_outputs.flatten())
cross_entropy = -np.mean(y_test.flatten() * np.log(predicted_outputs.flatten()) + (1 - y_test.flatten()) * np.log(1 - predicted_outputs.flatten()))

# Print the metrics
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)
print("Mean Squared Error:", mse)
print("Cross-Entropy:", cross_entropy)

'''
# Access the training history from the 'history' object
train_loss = history.history['loss']
val_loss = history.history['val_loss']
train_accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']

# Create a figure with subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# Plot loss
ax1.plot(train_loss, label='Train Loss')
ax1.plot(val_loss, label='Validation Loss')
ax1.set_title('Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.legend()

# Plot accuracy
ax2.plot(train_accuracy, label='Train Accuracy')
ax2.plot(val_accuracy, label='Validation Accuracy')
ax2.set_title('Accuracy')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.legend()

# Adjust spacing between subplots
plt.tight_layout()

# Show the plots
plt.show()

train_mse = history.history['mean_squared_error']  # Replace with the actual key used in history
val_mse = history.history['val_mean_squared_error']  # Replace with the actual key used in history
train_cross_entropy = history.history['cross_entropy']  # Replace with the actual key used in history
val_cross_entropy = history.history['val_cross_entropy']  # Replace with the actual key used in history

# Create a figure with subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# Plot MSE
ax1.plot(train_mse, label='Train MSE')
ax1.plot(val_mse, label='Validation MSE')
ax1.set_title('Mean Squared Error (MSE)')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('MSE')
ax1.legend()

# Plot cross-entropy
ax2.plot(train_cross_entropy, label='Train Cross-Entropy')
ax2.plot(val_cross_entropy, label='Validation Cross-Entropy')
ax2.set_title('Cross-Entropy')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Cross-Entropy')
ax2.legend()

# Adjust spacing between subplots
plt.tight_layout()

# Show the plots
plt.show()

'''