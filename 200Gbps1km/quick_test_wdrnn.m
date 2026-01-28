%% Quick Test WDRNN (Based on Roll-8)
% Runs WDRNN on first 3 ROP points (1.6, 2.6, 3.6 dBm)

clear;
close all;

%% Add Paths
addpath('fns');
addpath(fullfile('fns','fns2'));
addpath('algos');

%% Parameters
bbb = 100;
Ft = bbb*1e9;                       
Osamp_factor = 2;
NumSymbols = 2^17;
NumPreamble = 0;            
M = 4;

% Seed (Consistent with Tx generation)
s = RandStream.create('mt19937ar', 'seed', 529551);
prevStream = RandStream.setGlobalStream(s);

%% Tx Signal Generation
if exist('xsym.mat', 'file')
    load('xsym.mat', 'xsym');
else
    error('xsym.mat not found! Please ensure the file is in the directory.');
end
xsym = double(xsym);
xm = pammod(xsym, M, 0, 'gray'); 
xs = xm(:);

%% Pulse Shaping (Reference)
rolloff = 0.1;
N = 128;
sqrt_ht = rcosdesign(rolloff, N/Osamp_factor, Osamp_factor, 'normal');
sqrt_ht = sqrt_ht ./ max(sqrt_ht);
x_upsamp = upsample(xs, Osamp_factor);
x_shape = conv(sqrt_ht, x_upsamp);

%% Data List (First 3 points: 1.6, 2.6, 3.6 dBm)
base_path = 'C:\Users\27456\Desktop\毕业设计';
file_names = { ...
    'PS_PAM4_2_RoF_0.1_Cal_45GHz_AWG_210mV_ROP_1.6dBm.mat', ...
    'PS_PAM4_2_RoF_0.1_Cal_45GHz_AWG_210mV_ROP_2.6dBm.mat', ...
    'PS_PAM4_2_RoF_0.1_Cal_45GHz_AWG_210mV_ROP_3.6dBm.mat' ...
};
rop_dBm = [1.6, 2.6, 3.6];

%% WDRNN Parameters (From Roll-8)
params.RNN_InputLength = 61;
params.RNN_HiddenSize = 22;   % Consistent with Roll-8
params.RNN_LR = 0.001;
params.RNN_Epochs = 50;       
params.RNN_k = 25; 
params.RNN_DelayCandidates = -20:20;
params.RNN_OffsetCandidates = [1 2];

% Training Length
NumPreamble_TDE = 30000; 

%% Processing Loop
BER_list = zeros(length(file_names), 1);

disp('================================================');
disp('Starting WDRNN Quick Test (3 Data Points)');
disp('================================================');

for n1 = 1:length(file_names)
    fname = file_names{n1};
    fpath = fullfile(base_path, fname);
    
    fprintf('Processing: %s (ROP=%.1fdBm)\n', fname, rop_dBm(n1));
    
    if ~exist(fpath, 'file')
        warning('File not found: %s. Skipping...', fpath);
        BER_list(n1) = NaN;
        continue;
    end
    load(fpath, 'rx_original');

    % Process Only Segment 1
    ii = 1;
    xp = rx_original(ii,:); 
    xp = xp(:);
    xp = xp - mean(xp);
    
    % Resample
    ReData = resample(xp, 2*Ft/1e9, 256); 
    
    % Synchronization (TEFEMMM2)
    th = 0.2;
    [TE, FE] = TEFEMMM2(ReData, 1024, th);
    TE = TE + 1;
    ysync = ReData(1024 + 20 + TE : end); 
    
    % Filter Delay Compensation
    ysync_after = ysync(1 + N/2 : end);
    % Safe truncate
    len_needed = length(x_shape) - N;
    if length(ysync_after) < len_needed
        ysync_after = [ysync_after; zeros(len_needed-length(ysync_after),1)];
    end
    yt_filter = ysync_after(1 : len_needed);

    xTx = xs;
    xRx = yt_filter;
    
    % Run WDRNN
    try
        tic;
        [ye, ~, valid_idx, best_delay, best_offset] = WDRNN_Implementation( ...
            xRx, xTx, NumPreamble_TDE, params.RNN_InputLength, params.RNN_HiddenSize, ...
            params.RNN_LR, params.RNN_Epochs, params.RNN_k, params.RNN_DelayCandidates, params.RNN_OffsetCandidates);
        elapsed = toc;
        
        idxTx = valid_idx(:);
        ye_use = ye(idxTx);
        
        stats = eval_equalizer_pam4(ye_use, idxTx, xsym, xm, NumPreamble_TDE, M);
        BER_list(n1) = stats.BER;
        
        fprintf('  [WDRNN] Time: %.2fs | BER: %.2e\n', elapsed, stats.BER);
    catch ME
        fprintf('  [WDRNN] FAILED: %s\n', ME.message);
        BER_list(n1) = NaN;
    end
    disp('------------------------------------------------');
end

%% Simple Plot
figure('Name', 'Quick Test: WDRNN Results', 'Color', 'w');
semilogy(rop_dBm, BER_list, '-o', 'LineWidth', 1.5, 'MarkerSize', 7, 'MarkerFaceColor', 'b');
grid on; grid minor;
xlabel('Received Optical Power (dBm)');
ylabel('Bit Error Rate (BER)');
title('WDRNN Quick Test (1.6, 2.6, 3.6 dBm)');
yline(3.8e-3, '--k', 'HD-FEC (3.8e-3)');
set(gca, 'YScale', 'log');
