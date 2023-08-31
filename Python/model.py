import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, TimeDistributed
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K
from sklearn.utils.class_weight import compute_class_weight
from keras.layers import Input, Masking, Concatenate
from tensorflow.keras.models import Model
from imblearn.over_sampling import SMOTE

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

# Separate features and target
X = data_frame[['Time', 'Alpha Power', 'Beta Power', 'Gamma Power', 'Trial','Cue']].values
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

# Apply SMOTE to balance the class distribution in the training data
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Calculate class weights based on class imbalance
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train.flatten())  # Flatten y_train
class_weights = {0: class_weights[0], 1: class_weights[1]}

# Define custom loss that penalizes time differences between events
def custom_loss(y_true, y_pred):
    
    cue_presence = y_true[:, -1]
    event_presence = y_true[:, -2]
    
    # Masking to focus on events that occur after the cue
    masked_event_presence = event_presence * cue_presence
    
    masked_event_presence = K.expand_dims(masked_event_presence, axis=-1)  # Reshape for BCE loss
    
    # Binary cross-entropy loss for emphasizing event presence
    bce_loss = K.binary_crossentropy(K.cast(masked_event_presence, dtype='float32'), y_pred)
    
    # Create a time step factor to reduce loss for initial time steps
    num_time_steps = tf.shape(y_true)[1]
    time_step_factor = tf.range(1, num_time_steps + 1, dtype=K.floatx())
    time_step_factor = 1.0 / tf.sqrt(time_step_factor)  # Apply a scaling function
    
    # Calculate a penalty for events predicted before the cue
    event_before_cue_penalty = K.binary_crossentropy(K.cast(event_presence, dtype='float32'), y_pred)
    event_before_cue_penalty = K.expand_dims(event_before_cue_penalty, axis=-1)  # Reshape for consistency
    
    # Calculate the weighted loss for each time step
    #weighted_loss_per_step = bce_loss * time_step_factor + event_before_cue_penalty
    weighted_loss_per_step = bce_loss * time_step_factor
    # Flatten the tensors and compute the weighted sum of losses
    weighted_loss = K.sum(weighted_loss_per_step)

    return weighted_loss
    #return bce_loss

# Define the model architecture
input_layer = Input(shape=(sequence_length, num_features))

# Masking layer to ignore entries before the cue
#masked_input = Masking(mask_value=0)(input_layer)

# LSTM layer to capture temporal patterns, with return_sequences=True
lstm_output = LSTM(64, return_sequences=True)(input_layer)

# Fully connected layer for predicting event presence at each time step
output = TimeDistributed(Dense(1, activation='sigmoid'))(lstm_output)

# Create the model
model = Model(inputs=input_layer, outputs=output)

# model = Sequential([
#     LSTM(64, return_sequences=True, input_shape=(sequence_length, num_features)),
#     LSTM(32, return_sequences=True),
#     TimeDistributed(Dense(1, activation='sigmoid'))
# ])

# Compile the model
model.compile(optimizer='adam', loss=custom_loss, metrics=['accuracy'])

# Train the model using the training data
history = model.fit(X_train_resampled, y_train_resampled, epochs=10, batch_size=32,
                    validation_data=(X_val, y_val))

# Evaluate the model on the testing set
loss, accuracy = model.evaluate(X_test, y_test, batch_size=32)
print("Test loss:", loss)
print("Test accuracy:", accuracy * 100, "%")

# Generate predictions on the testing set
predicted_outputs = model.predict(X_test)

# Scatter PLOT

y_test_flattened = y_test.flatten()
predicted_outputs_flattened = predicted_outputs.flatten()

# Create a scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(y_test_flattened, predicted_outputs_flattened, alpha=0.5)
plt.xlabel('True Labels (y_test)')
plt.ylabel('Predicted Outputs')
plt.title('Scatter Plot: True Labels vs. Predicted Outputs')
plt.grid(True)
plt.show()

'''
# Assuming 'predicted_outputs' and 'y_test' are your arrays
predicted_probabilities_size = len(predicted_outputs.flatten())
actual_output_size = len(y_test.flatten())

print("Size of Predicted Probabilities:", predicted_probabilities_size)
print("Size of Actual Output:", actual_output_size)

# Threshold for converting predicted probabilities to binary predictions
threshold = 0.1
predicted_binary = (predicted_outputs >= threshold).astype(int)

data = {
    'Predicted Probabilities': predicted_outputs.flatten(),
    'Predicted Binary': predicted_binary.flatten(),
    'Actual Output': y_test.flatten()
}

temp = pd.DataFrame(data)
temp.to_csv('temp.csv', index=False)

# Create a figure and axis

# Specify the iteration index for which you want to visualize the output
iteration_index = 0  # Replace with the desired iteration index

# Extract the actual and predicted outputs for the specified iteration
actual_output_iteration = y_test[iteration_index]
predicted_output_iteration = predicted_binary[iteration_index]

# Create a line graph for the specified iteration
plt.plot(range(len(predicted_output_iteration)), predicted_output_iteration, color='red', label='Predicted Output')
plt.plot(range(len(actual_output_iteration)), actual_output_iteration, color='blue', label='Actual Output')
plt.xlabel('Sample Index')
plt.ylabel('Output')
plt.legend()
plt.title(f'Actual vs Predicted Output for Iteration {iteration_index}')
plt.show()
'''