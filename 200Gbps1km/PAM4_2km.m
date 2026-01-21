%% PAM4 2km - Roll-7 Algorithms Comparison (Single Segment)
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
NumSym_total = NumSymbols + NumPreamble;
M = 4;                                                                                      

% Seed (Consistent with Tx generation)
s = RandStream.create('mt19937ar', 'seed', 529551);
prevStream = RandStream.setGlobalStream(s);

%% Tx Signal Generation (Load pre-generated Tx)
% Load 'xsym' from file (0,1,2,3)
if exist('xsym.mat', 'file')
    load('xsym.mat', 'xsym');
else
    error('xsym.mat not found! Please ensure the file provided by teacher is in the directory.');
end

% Modulate to (-3, -1, 1, 3)
% Ensure xsym is double for pammod
xsym = double(xsym);
RowSym = 0:M-1;
xm = pammod(xsym, M, 0, 'gray'); 
xs = xm(:); % Transmitted sequence (Tx)

%% Pulse Shaping (Reference for Length)
rolloff = 0.1;
N = 128;
sqrt_ht = rcosdesign(rolloff, N/Osamp_factor, Osamp_factor, 'normal');
sqrt_ht = sqrt_ht ./ max(sqrt_ht);
x_upsamp = upsample(xs, Osamp_factor);
x_shape = conv(sqrt_ht, x_upsamp); % Length reference

%% Equalizer Parameters (From roll_7.m)
params.N1 = 61;                     % FFE Taps
params.N2 = 7;                      % Volterra Taps
params.WL = 3;                      
params.D1 = 11;                     % DFE Taps
params.D2 = 0;
params.WD = 1;
params.K_Lin = 18;                  % Cluster Centers
params.K_Vol = 90;
params.Lambda = 0.9999;
params.scale = M/2;

% NN Parameters (Aligned with Paper)
params.FNN_InputLength = 61;
params.FNN_HiddenSize = 20;  % Paper Optimal: 20 (was 128)
params.FNN_LR = 0.001;
params.FNN_Epochs = 50; 
params.FNN_DelayCandidates = -20:20;
params.FNN_OffsetCandidates = [1 2];

params.RNN_InputLength = 61;
params.RNN_HiddenSize = 20;  % Paper Optimal: 20 (was 16)
params.RNN_LR = 0.001;
params.RNN_Epochs = 50;      % Increased to 50 for better convergence
params.RNN_k = 25; 
params.RNN_DelayCandidates = -20:20;
params.RNN_OffsetCandidates = [1 2];

%% Algo List
algo_list = { ...
    'FFE', ...
    'VNLE', ...
    'LE_FFE_DFE', ...
    'DP_VFFE_VDFE', ...
    'CLUT_VDFE', ...
    'FNN', ...
    'RNN', ...
    'WDRNN' ...
};

%% Data Loading & Processing
data_path = 'C:\Users\27456\Desktop\毕业设计\PS_PAM4_2_RoF_0.1_Cal_45GHz_AWG_210mV_ROP_5.6dBm.mat';
disp(['Loading Data: ', data_path]);
load(data_path, 'rx_original');

% Process ONLY the first segment (Index 1)
ii = 1; 
disp('------------------------------------------------');
disp(['Processing Data Segment ', num2str(ii), ' ONLY']);

xp = rx_original(ii,:); 
xp = xp(:);
xp = xp - mean(xp);

% Resample
ReData = resample(xp, 2*Ft/1e9, 256); 

%% Synchronization (Original TEFEMMM2)
th = 0.2;
[TE, FE] = TEFEMMM2(ReData, 1024, th);
TE = TE + 1;
ysync = ReData(1024 + 20 + TE : end); 

% Filter Delay Compensation
ysync_after = ysync(1 + N/2 : end);
yt_filter = ysync_after(1 : length(x_shape) - N); % Truncate to match Tx length

%% Equalization Loop
xTx = xs;
xRx = yt_filter;
NumPreamble_TDE_Base = 10000; % Base training length

BER_Result = zeros(length(algo_list), 1);

for a = 1:length(algo_list)
    algo_id = algo_list{a};
    disp(['Running Algorithm: ', algo_id, ' ...']);
    
    % Adjust Training Length for NN
    if ismember(upper(algo_id), {'FNN', 'RNN', 'WDRNN'})
        NumPreamble_TDE = 30000; % More data for NN
    else
        NumPreamble_TDE = NumPreamble_TDE_Base;
    end
    
    try
        tic;
        [ye_use, idxTx] = run_equalizer(algo_id, xRx, xTx, xsym, NumPreamble_TDE, M, params);
        elapsed = toc;
        
        % Evaluate
        stats = eval_equalizer_pam4(ye_use, idxTx, xsym, xm, NumPreamble_TDE, M);
        BER_Result(a) = stats.BER;
        
        fprintf('  > %s Completed in %.2fs | BER: %.2e\n', algo_id, elapsed, stats.BER);
        
    catch ME
        fprintf('  > %s FAILED: %s\n', algo_id, ME.message);
        BER_Result(a) = NaN;
    end
end

%% Plotting Results (Linear Scale)
figure('Name', 'BER Comparison (2km, 5.6dBm)');
b = bar(categorical(algo_list), BER_Result);
ylabel('BER');
title('BER Performance Comparison (2km, 5.6dBm)');
grid on;

% Add Value Labels
xtips = b.XEndPoints;
ytips = b.YEndPoints;
labels = string(compose('%.1e', BER_Result));
text(xtips, ytips, labels, 'HorizontalAlignment','center', 'VerticalAlignment','bottom');

% Threshold Line (HD-FEC)
yline(3.8e-3, '--r', 'HD-FEC (3.8e-3)', 'LineWidth', 1.5);

disp('------------------------------------------------');
disp('Done.');

%% ---------------- Local Functions ----------------
function [ye_use, idxTx, best_delay, best_offset] = run_equalizer(algo_id, xRx, xTx, xsym, NumPreamble_TDE, M, params)
    clear ye ye_valid valid_idx net

    best_delay = NaN;
    best_offset = NaN;

    switch upper(algo_id)
        case 'FFE'
            [~, ye] = FFE_2pscenter(xRx, xTx, NumPreamble_TDE, params.N1, params.Lambda);
            is_nn = false;
        case 'VNLE'
            [~, ye] = VNLE2_2pscenter(xRx, xTx, NumPreamble_TDE, params.N1, params.N2, params.Lambda, params.WL);
            is_nn = false;
        case 'LE_FFE_DFE'
            [~, ~, ye] = LE_FFE2ps_centerDFE_new(xRx, xTx, NumPreamble_TDE, params.N1, params.D1, params.Lambda, M, params.scale);
            is_nn = false;
        case 'DP_VFFE_VDFE'
            [~, ~, ye] = DP_VFFE2pscenter_VDFE(xRx, xTx, NumPreamble_TDE, params.N1, params.N2, params.D1, params.D2, params.Lambda, params.WL, params.WD, M, params.scale);
            is_nn = false;
        case 'CLUT_VDFE'
            [~, ye] = CLUT_VDFE_Implementation(xRx, xTx, NumPreamble_TDE, params.N1, params.N2, params.D1, params.D2, params.WL, params.WD, M, params.K_Lin, params.K_Vol, params.Lambda);
            is_nn = false;
        case 'FNN'
            [ye_valid, ~, valid_idx, best_delay, best_offset] = FNN_FS2pscenter( ...
                xRx, xTx, NumPreamble_TDE, params.FNN_InputLength, params.FNN_HiddenSize, ...
                params.FNN_LR, params.FNN_Epochs, params.FNN_DelayCandidates, params.FNN_OffsetCandidates);
            is_nn = true;
        case 'RNN'
            [ye, ~, valid_idx, best_delay, best_offset] = RNN_Implementation( ...
                xRx, xTx, NumPreamble_TDE, params.RNN_InputLength, params.RNN_HiddenSize, ...
                params.RNN_LR, params.RNN_Epochs, params.RNN_k, params.RNN_DelayCandidates, params.RNN_OffsetCandidates);
            is_nn = true;
        case 'WDRNN'
            [ye, ~, valid_idx, best_delay, best_offset] = WDRNN_Implementation( ...
                xRx, xTx, NumPreamble_TDE, params.RNN_InputLength, params.RNN_HiddenSize, ...
                params.RNN_LR, params.RNN_Epochs, params.RNN_k, params.RNN_DelayCandidates, params.RNN_OffsetCandidates);
            is_nn = true;
        otherwise
            error('Unknown algorithm: %s', algo_id);
    end

    if is_nn
        if ~exist('valid_idx', 'var') || isempty(valid_idx)
            error('valid_idx missing for NN algorithm: %s', algo_id);
        end
        idxTx = valid_idx(:);
        if exist('ye_valid', 'var') && ~isempty(ye_valid)
            ye_use = ye_valid(:);
        else
            ye_use = ye(idxTx);
        end
    else
        if ~exist('ye', 'var') || isempty(ye)
            error('ye missing for classical algorithm: %s', algo_id);
        end
        % Classical Sync Alignment
        [off, d0] = align_offset_delay_by_ser(ye, xsym, NumPreamble_TDE, M, -60:60);
        
        if length(ye) > 1.5 * length(xsym)
            ye_use = ye(off:2:end);
        else
            ye_use = ye(:);
        end
        idxTx = (1:length(ye_use)).' + d0;
    end
end
