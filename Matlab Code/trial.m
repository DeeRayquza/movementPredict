function trial(plot_type, trial_num)
    % Load the data
    data = load('4672_FT_TFR_all_trials_bc.mat');
    TFRwave = data.TFRwave;
    
    freqs = TFRwave.freq;
    times = TFRwave.time;
    
    % Define the time and frequency ranges
    time_range = [-1, 5];
    freq_range = [0, 40];
    
    % Find the indices corresponding to the specified time and frequency ranges
    time_indices = find(times >= time_range(1) & times <= time_range(2));
    freq_indices = find(freqs >= freq_range(1) & freqs <= freq_range(2));

    % Extract data for the specified trial, time, and frequency ranges
    wave_data = TFRwave.powspctrm(trial_num, :, freq_indices, time_indices);

    alpha_freq_indices = find(freqs >= 8 & freqs <= 12);
    beta_freq_indices = find(freqs >= 13 & freqs <= 30);
    gamma_freq_indices = find(freqs > 30);
    
    % Plotting
    figure;
    switch plot_type
        case 'tfr'
            mean_power = squeeze(mean(wave_data, 1));
            reshaped_mean_power = reshape(mean_power, [], length(time_indices));
            imagesc(times(time_indices), freqs(freq_indices), reshaped_mean_power);
            set(gca, 'YDir', 'normal'); % Flip the y-axis direction
            xlabel('Time (s)');
            ylabel('Frequency (Hz)');
            title(['Time-Frequency Representation - Trial ', num2str(trial_num)]);
            colorbar;
            
        case 'line_chart'
            mean_power = squeeze(mean(mean(wave_data, 2), 3));
            plot(times(time_indices), mean_power);
            xlabel('Time (s)');
            ylabel('Mean Power');
            title(['Mean Power across Frequencies - Trial ', num2str(trial_num)]);
            
        case 'freq_bands'
            mean_power_alpha = squeeze(mean(wave_data(:, alpha_freq_indices, :), 2));
            mean_power_beta = squeeze(mean(wave_data(:, beta_freq_indices, :), 2));
            mean_power_gamma = squeeze(mean(wave_data(:, gamma_freq_indices, :), 2));
            
            mean_power_alpha = reshape(mean_power_alpha, [], length(time_indices));
            mean_power_beta = reshape(mean_power_beta, [], length(time_indices));
            mean_power_gamma = reshape(mean_power_gamma, [], length(time_indices));
            
            plot_freq_bands(mean_power_alpha, mean_power_beta, mean_power_gamma, times(time_indices), trial_num);
        
        case 'mean_freq_bands'
            plot_mean_freq_bands(TFRwave, trial_num, alpha_freq_indices, beta_freq_indices, gamma_freq_indices);
%{        
        case 'export_python'
            mean_power_alpha = squeeze(mean(wave_data(:, alpha_freq_indices, :), [1 2]));
            mean_power_beta = squeeze(mean(wave_data(:, beta_freq_indices, :), [1 2]));
            mean_power_gamma = squeeze(mean(wave_data(:, gamma_freq_indices, :), [1 2]));
        
            export_data = [times(time_indices)', mean_power_alpha, mean_power_beta, mean_power_gamma];
            csvwrite('mean_power_data_python.csv', export_data);
%}  
  end
end

function plot_freq_bands(mean_power_alpha, mean_power_beta, mean_power_gamma, time_vals, trial_num)
    figure;
    plot(time_vals, mean_power_alpha, 'r', 'LineWidth', 2);
    hold on;
    plot(time_vals, mean_power_beta, 'g', 'LineWidth', 2);
    plot(time_vals, mean_power_gamma, 'b', 'LineWidth', 2);

    xlabel('Time (s)');
    ylabel('Mean Power');
    title(['Mean Power in Alpha, Beta, and Gamma Bands - Trial ', num2str(trial_num)]);
    legend('Alpha (8-12 Hz)', 'Beta (13-30 Hz)', 'Gamma (>30 Hz)');
    grid on;
end

function plot_mean_freq_bands(TFRwave, trial_num, alpha_freq_indices, beta_freq_indices, gamma_freq_indices)
    trial_data = squeeze(TFRwave.powspctrm(trial_num, :, :, :));

    mean_power_alpha = squeeze(mean(trial_data(:, alpha_freq_indices, :), [1 2]));
    mean_power_beta = squeeze(mean(trial_data(:, beta_freq_indices, :), [1 2]));
    mean_power_gamma = squeeze(mean(trial_data(:, gamma_freq_indices, :), [1 2]));

    plot(TFRwave.time, mean_power_alpha, 'r', 'LineWidth', 2);
    hold on;
    plot(TFRwave.time, mean_power_beta, 'g', 'LineWidth', 2);
    plot(TFRwave.time, mean_power_gamma, 'b', 'LineWidth', 2);

    xlabel('Time (s)');
    ylabel('Mean Power');
    title(['Mean Power in Alpha, Beta, and Gamma Bands - Trial ', num2str(trial_num)]);
    legend('Alpha (8-12 Hz)', 'Beta (13-30 Hz)', 'Gamma (>30 Hz)');
    grid on;
end
