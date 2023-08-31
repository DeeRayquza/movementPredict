import pandas as pd

csv_path = r'D:\MSc Project\Data\Old Data\MEG\s08\mean_power_data_python.csv'
data = pd.read_csv(csv_path, header=None, names=['Time', 'Mean_Alpha_Power', 'Mean_Beta_Power', 'Mean_Gamma_Power', 'Trial_Number'])

# Filter data for the specified time range
filtered_data = data[(data['Time'] >= -0.5) & (data['Time'] <= 0)]
# Calculate the mean values for alpha, beta, and gamma power for each trial
mean_values = filtered_data.groupby('Trial_Number')[['Mean_Alpha_Power', 'Mean_Beta_Power', 'Mean_Gamma_Power']].mean()


# Loop through each row and calculate and store the percent change
for index, row in data.iterrows():
    trial_number = row['Trial_Number']
    corresponding_alpha_mean = mean_values.loc[trial_number, 'Mean_Alpha_Power']
    corresponding_beta_mean = mean_values.loc[trial_number, 'Mean_Beta_Power']
    corresponding_gamma_mean = mean_values.loc[trial_number, 'Mean_Gamma_Power']
    
    alpha_percent_change = ((row['Mean_Alpha_Power'] / corresponding_alpha_mean)-1) * 100
    beta_percent_change = ((row['Mean_Beta_Power'] / corresponding_beta_mean)-1) * 100
    gamma_percent_change = ((row['Mean_Gamma_Power'] / corresponding_gamma_mean)-1) * 100
    
    data.at[index, 'alpha_percent_change'] = alpha_percent_change
    data.at[index, 'beta_percent_change'] = beta_percent_change
    data.at[index, 'gamma_percent_change'] = gamma_percent_change

new_columns = ['Time', 'alpha_percent_change', 'beta_percent_change', 'gamma_percent_change', 'Trial_Number']
new_data = data[new_columns]
new_path = r'D:\MSc Project\Data\Old Data\MEG\s08\mean_power_data_python_bc.csv'
new_data.to_csv(new_path, index=False, header=False)

#print(data)
#print(mean_values)
