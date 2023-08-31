function export_power_data()
    % Load the data
    data = load('04924_FT_TFR_all_trials.mat');
    TFRwave = data.TFRwave;
    
    times = TFRwave.time;
    
    % Define the time and frequency ranges
    time_range = [-1, 5];
    freq_range = [0, 40];
    
    % Find the indices corresponding to the specified time and frequency ranges
    time_indices = find(times >= time_range(1) & times <= time_range(2));
    
    alpha_freq_indices = find(TFRwave.freq >= 8 & TFRwave.freq <= 12);
    beta_freq_indices = find(TFRwave.freq >= 13 & TFRwave.freq <= 30);
    gamma_freq_indices = find(TFRwave.freq > 30);
    
    % Initialize data arrays
    export_data = [];
    
    for trial_num = 1:size(TFRwave.powspctrm, 1)
        % Extract power data for the current trial
        trial_data = squeeze(TFRwave.powspctrm(trial_num, :, :, :));
        
        % Calculate mean power in alpha, beta, and gamma bands
        mean_power_alpha = squeeze(mean(trial_data(:, alpha_freq_indices, :), [1 2]));
        mean_power_beta = squeeze(mean(trial_data(:, beta_freq_indices, :), [1 2]));
        mean_power_gamma = squeeze(mean(trial_data(:, gamma_freq_indices, :), [1 2]));

        % Concatenate data for the current trial
        trial_data_export = [times(time_indices)', mean_power_alpha, mean_power_beta, mean_power_gamma, trial_num * ones(length(time_indices), 1)];
        
        % Append to the export_data array
        export_data = [export_data; trial_data_export];
    end
    
    % Write data to CSV file
    csvwrite('mean_power_data_python.csv', export_data);
end
