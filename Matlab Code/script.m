
function script(what)

% Loading toolboxes
addpath(genpath('D:\Matlab\fieldtrip-20230422')); % Fieldtrip
addpath(genpath('D:\Matlab\userfun')); % Util
addpath(genpath('D:\MSc Project\Code')); % Scripts
%addpath(genpath('D:\MSc Project\Data\Old Data\MEG\s25')); % Data

switch(what)

    case 'convert'
        % Loading Data
        ft_hastoolbox('spm8',1);
        addpath(genpath('D:\Matlab\spm12')); %SPM to convert the data
        data_path = spm_eeg_load('fcPRdmg4672_blk02_ICA_corrected.mat');
        
        % Convert Data from SPM to FieldTrip
        data = spm2fieldtrip(data_path);

        save('4672_FT', 'data', '-v7.3');
        clear all;

    case 'tfr'

        %compute TFR
        cfg = [];
        cfg.channel    = 'MEG';
        cfg.method     = 'wavelet';
        cfg.width      = 7;
        cfg.output     = 'pow';
        cfg.foi        = 7:1:40;
        cfg.toi        = -0.5:0.05:4.5;

        load '04924_FT.mat';
        %data = load('4670_FT.mat');
        TFRwave = ft_freqanalysis(cfg, data);

        save('04924_FT_TFRmorlet', 'TFRwave', '-v7.3');

        %TFR plot
        cfg = [];
        cfg.baseline     = [-0.5 -0.1];
        cfg.baselinetype = 'relchange';
        cfg.maskstyle    = 'saturation';
        cfg.xlim         = [-0.5 4.5];
        cfg.zlim         = [-0.25 0.25];
        cfg.channel      = 'all';
        cfg.layout       = 'CTF151_helmet.mat';
        figure;
        ft_singleplotTFR(cfg, TFRwave);

        % Topoplot
        cfg = [];
        cfg.baseline     = [-0.5 -0.1];
        cfg.baselinetype = 'relchange';
        cfg.xlim         = [-0.5 1.8];
        cfg.ylim         = [13 30];
        cfg.zlim         = [-0.1 0.1];
        cfg.marker       = 'on';
        cfg.layout       = 'CTF275_helmet.mat'; % at UCL: CTF system; at CHBH: Neuromag TRIUX -> https://www.fieldtriptoolbox.org/template/layout/
        cfg.colorbar     = 'yes';
        figure;
        ft_topoplotTFR(cfg, TFRwave);


        case 'tfr_trials'
            
            
            % Load converted data
            load '04924_FT.mat'; % Update the file name to match your actual file name
        
            % Configuration for TFR
            cfg = [];
            cfg.channel    = 'MEG';
            cfg.method     = 'wavelet';
            cfg.width      = 7;
            cfg.output     = 'pow';
            cfg.foi        = 3:1:40;
            cfg.toi        = -0.5:0.05:4.5;
        
            % Specify 'all' trials to process
            cfg.trials = 'all';
            cfg.keeptrials = 'yes';

            % Perform TFR analysis on all trials
            TFRwave = ft_freqanalysis(cfg, data);
           

            % Save TFR results
            save('04924_FT_TFR_all_trials', 'TFRwave', '-v7.3');
       
            % TFR plot
            cfg = [];
            cfg.baseline     = [-0.5 -0.1];
            cfg.baselinetype = 'relchange';
            cfg.maskstyle    = 'saturation';
            cfg.xlim         = [-0.5 4.5];
           
            cfg.channel      = 'all';
            cfg.layout       = 'CTF151_helmet.mat';
            figure;
            ft_singleplotTFR(cfg, TFRwave);
        
            % Topoplot
            cfg = [];
            cfg.baseline     = [-0.5 -0.1];
            cfg.baselinetype = 'relchange';
            cfg.xlim         = [1.8 4.5];
            cfg.ylim         = [13 30];
            cfg.marker       = 'on';
            cfg.layout       = 'CTF275_helmet.mat';
            cfg.colorbar     = 'yes';
            figure;
            ft_topoplotTFR(cfg, TFRwave);
          

        case 'prcChange'
            % Load TFR data for all trials
            load('4672_FT_TFR_all_trials.mat'); % Update the filename accordingly
            
            bslStart = -0.5;
            bslEnd = 0;
            tWinStart = -0.5;
            tWinEnd = 4.5;
        
            bslTrial = (find(TFRwave.time == bslStart) : find(TFRwave.time == bslEnd));
            timeTrial = (find(TFRwave.time == tWinStart) : find(TFRwave.time == tWinEnd));
            freq = 1:40; % e.g., All Bands
            CoI = 1:numel(TFRwave.label); % All channels of interest
            
            % Compute baseline-corrected power for all trials, specific channels, and frequencies of interest
            tfrData = TFRwave.powspctrm(:, CoI, freq, timeTrial);
            tfrDataBsl = mean(TFRwave.powspctrm(:, CoI, freq, bslTrial), 4); % Average baseline power across time
            tfrDataBslMean = mean(tfrDataBsl, 3); % Mean power in the baseline period for all channels and frequencies
            tfrDataBslCorr = ((tfrData ./ tfrDataBslMean) - 1) * 100; % Percent power change from baseline period
          
            % Save Baseline Corrected results
            save('4672_baseline_corr', 'tfrDataBslCorr', '-v7.3');
end        
