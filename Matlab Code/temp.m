function temp(what)
    %data = load('4672_FT_TFR_all_trials.mat');
    data = load('4672_FT_TFRmorlet.mat');
    freqs = data.TFRwave.freq;
    times = data.TFRwave.time;
    
    % Choose the time and frequency range you are interested in
    time_range = [-1, 5]; % Define your desired time range
    freq_range = [0, 40]; % Define your desired frequency range

    % Find the indices corresponding to your desired time and frequency range
    time_indices = find(times >= time_range(1) & times <= time_range(2));
    freq_indices = find(freqs >= freq_range(1) & freqs <= freq_range(2));

    wave_data = data.TFRwave.powspctrm(:, freq_indices, time_indices);

    alpha_freq_indices = find(freqs >= 8 & freqs <= 12);
    beta_freq_indices = find(freqs >= 13 & freqs <= 30);
    gamma_freq_indices = find(freqs > 30);
    
    switch what
        case 'tfr'
            % Plotting TFR
            figure;
            imagesc(times(time_indices), freqs(freq_indices), squeeze(mean(wave_data, 1)));
            set(gca, 'YDir', 'normal'); % Flip the y-axis direction
            xlabel('Time (s)');
            ylabel('Frequency (Hz)');
            title('Time-Frequency Representation');
            colorbar;
            
        case 'line_chart'
            mean_power = squeeze(mean(wave_data, 1));

            % Plotting Line Chart
            figure;
            plot(times(time_indices), mean_power);
            xlabel('Time (s)');
            ylabel('Mean Power');
            title('Mean Power across Frequencies');
            
        case 'freq_bands'
            
            mean_power_alpha = squeeze(mean(wave_data(:, alpha_freq_indices, :), 2));
            mean_power_beta = squeeze(mean(wave_data(:, beta_freq_indices, :), 2));
            mean_power_gamma = squeeze(mean(wave_data(:, gamma_freq_indices, :), 2));

            % Plotting
            figure;
            plot(times(time_indices), mean_power_alpha, 'r', 'LineWidth', 2);
            hold on;
            plot(times(time_indices), mean_power_beta, 'g', 'LineWidth', 2);
            plot(times(time_indices), mean_power_gamma, 'b', 'LineWidth', 2);
            
            xlabel('Time (s)');
            ylabel('Mean Power');
            title('Mean Power in Alpha, Beta, and Gamma Bands');
            legend('Alpha (8-12 Hz)', 'Beta (13-30 Hz)', 'Gamma (>30 Hz)');
            grid on;

        case 'mean_freq_bands'
            mean_power_alpha = squeeze(mean(wave_data(:, alpha_freq_indices, :), [1 2]));
            mean_power_beta = squeeze(mean(wave_data(:, beta_freq_indices, :), [1 2]));
            mean_power_gamma = squeeze(mean(wave_data(:, gamma_freq_indices, :), [1 2]));

            % Plotting
            figure;
            plot(times(time_indices), mean_power_alpha, 'r', 'LineWidth', 2);
            hold on;
            plot(times(time_indices), mean_power_beta, 'g', 'LineWidth', 2);
            plot(times(time_indices), mean_power_gamma, 'b', 'LineWidth', 2);
            
            xlabel('Time (s)');
            ylabel('Mean Power');
            title('Mean Power in Alpha, Beta, and Gamma Bands');
            legend('Alpha (8-12 Hz)', 'Beta (13-30 Hz)', 'Gamma (>30 Hz)');
            grid on;

        case 'export_python'
            mean_power_alpha = squeeze(mean(wave_data(:, alpha_freq_indices, :), [1 2]));
            mean_power_beta = squeeze(mean(wave_data(:, beta_freq_indices, :), [1 2]));
            mean_power_gamma = squeeze(mean(wave_data(:, gamma_freq_indices, :), [1 2]));

            % Export data for Python plotting
            export_data = [times(time_indices)', mean_power_alpha, mean_power_beta, mean_power_gamma];
            csvwrite('mean_power_data_python.csv', export_data);
            
        otherwise
            disp('Invalid option. Please choose a valid option: tfr, line_chart, freq_bands, mean_freq_bands, export_python');
    end
end
