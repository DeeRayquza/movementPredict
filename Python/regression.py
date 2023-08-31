import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


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

# Separate features and target for regression
X_regression = data_frame[['Time', 'Alpha Power', 'Beta Power', 'Gamma Power', 'Trial', 'Cue']].values
y_event_intensity = data_frame['Event'].values  # Assuming you have a column 'EventIntensity'

# Split data into trials
unique_trials = np.unique(X[:, -1])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_regression, y_event_intensity, test_size=0.2, random_state=42)

# Initialize the regression model
regression_model = LinearRegression()

# Train the model
regression_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = regression_model.predict(X_test)

# Print coefficients and intercept
coefficients = regression_model.coef_
intercept = regression_model.intercept_
print("Coefficients:", coefficients)
print("Intercept:", intercept)

# Print Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# Use the trained model to make predictions on new data (example using first trial)
new_trial_data = X_regression[X_regression[:, -1] == unique_trials[0]]
predicted_intensity = regression_model.predict(new_trial_data)

# Print predictions for the new trial
print("Predicted Intensity for New Trial:", predicted_intensity)

# Scatter plot of predicted vs. actual event intensities
plt.scatter(y_test, y_pred, color='blue', alpha=0.5)
plt.xlabel('Actual Event Intensity')
plt.ylabel('Predicted Event Intensity')
plt.title('Actual vs. Predicted Event Intensity')
plt.show()

# Use the trained model to make predictions on new data (example using first trial)
new_trial_data = X_regression[X_regression[:, -1] == unique_trials[0]]
predicted_intensity = regression_model.predict(new_trial_data)

# Print predictions for the new trial
print("Predicted Intensity for New Trial:", predicted_intensity)