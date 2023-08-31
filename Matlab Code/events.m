data = load('D:\MSc Project\Data\Old Data\Behavioural\s25\behDataMEG.mat');
csv_filename = 'D:\MSc Project\Data\Old Data\MEG\s25\mean_power_data_python_bc.csv';

timing_data = data.A.timing;

% Creating a column of zeros with the same number of rows as timing_data
zero_column = zeros(size(timing_data, 1), 1);

% Concatenating the zero column to the left of the timing_data matrix
timing = [zero_column, timing_data];

% Load the CSV file
events_data = csvread(csv_filename);

events_data(:,1) = events_data(:,1) * 1000;

events_label = [0, 1, 2 , 3, 4, 5, 6];
events_index = 6;
last_processed_index = 1; % Initialize with the starting index

for row = 1:size(timing, 1)
    for col = 1:size(timing, 2)
        label = events_label(col);
        timing_value  = timing(row, col);

        for rowIndex = last_processed_index:size(events_data, 1)
            event_time = events_data(rowIndex, 1);
            trial_number = events_data(rowIndex, 5);

            if event_time  >= timing_value && trial_number == row
                events_data(rowIndex, events_index) = label;
                last_processed_index = rowIndex; % Update the last processed index
                break; % No need to continue checking once the condition is met
            end

        end
    
    end
   
end    

csv_filename_modified = 'D:\MSc Project\Data\Old Data\MEG\s25\training_data.csv';
csvwrite(csv_filename_modified, events_data);

