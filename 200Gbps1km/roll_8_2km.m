%% Roll-8 Comparison: BER vs ROP (2km Data)
% Includes: FFE, VNLE, LE_FFE_DFE, DP_VFFE_VDFE, CLUT_VDFE, FNN, RNN(AR), WDRNN
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

%% Tx Signal Generation (Load pre-generated Tx)
% Load 'xsym' from file (0,1,2,3)
if exist('xsym.mat', 'file')
    load('xsym.mat', 'xsym');
else
    error('xsym.mat not found! Please ensure the file is in the directory.');
end

% Modulate to (-3, -1, 1, 3)
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

%% Data List (ROP Files)
base_path = 'C:\Users\27456\Desktop\毕业设计';
file_names = { ...
    'PS_PAM4_2_RoF_0.1_Cal_45GHz_AWG_210mV_ROP_1.6dBm.mat', ...
    'PS_PAM4_2_RoF_0.1_Cal_45GHz_AWG_210mV_ROP_2.6dBm.mat', ...
    'PS_PAM4_2_RoF_0.1_Cal_45GHz_AWG_210mV_ROP_3.6dBm.mat', ...
    'PS_PAM4_2_RoF_0.1_Cal_45GHz_AWG_210mV_ROP_4.6dBm.mat', ...
    'PS_PAM4_2_RoF_0.1_Cal_45GHz_AWG_210mV_ROP_5.6dBm.mat' ...
};

file_list = fullfile(base_path, file_names);

% Parse dBm for plotting
rop_dBm = [1.6, 2.6, 3.6, 4.6, 5.6];

%% Equalizer Parameters
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

% NN Parameters (Paper Aligned)
params.FNN_InputLength = 61;
params.FNN_HiddenSize = 20;   % Paper: 20
params.FNN_LR = 0.001;
params.FNN_Epochs = 50; 
params.FNN_DelayCandidates = -20:20;
params.FNN_OffsetCandidates = [1 2];

params.RNN_InputLength = 61;
params.RNN_HiddenSize = 20;   % Paper: 20
params.RNN_LR = 0.001;
params.RNN_Epochs = 50;       % Optimized: 50
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

%% Processing Loop
BERall = zeros(length(file_list), length(algo_list));

for n1 = 1:length(file_list)
    disp(['================================================']);
    disp(['Processing File: ', file_names{n1}, ' (ROP=', num2str(rop_dBm(n1)), 'dBm)']);
    
    if ~exist(file_list{n1}, 'file')
        error('File not found: %s', file_list{n1});
    end
    load(file_list{n1}, 'rx_original');

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
    NumPreamble_TDE_Base = 10000; 

    for a = 1:length(algo_list)
        algo_id = algo_list{a};
        
        % Training Length
        if ismember(upper(algo_id), {'FNN', 'RNN', 'WDRNN'})
            NumPreamble_TDE = 30000; 
        else
            NumPreamble_TDE = NumPreamble_TDE_Base;
        end
        
        try
            tic;
            [ye_use, idxTx] = run_equalizer(algo_id, xRx, xTx, xsym, NumPreamble_TDE, M, params);
            elapsed = toc;
            
            stats = eval_equalizer_pam4(ye_use, idxTx, xsym, xm, NumPreamble_TDE, M);
            BERall(n1, a) = stats.BER;
            
            fprintf('  [%s] %.2fs | BER: %.2e\n', algo_id, elapsed, stats.BER);
        catch ME
            fprintf('  [%s] FAILED: %s\n', algo_id, ME.message);
            BERall(n1, a) = NaN;
        end
    end
end

%% Plot: BER vs ROP (dBm)
figure('Name', 'BER vs ROP (2km)');

% Color Palette & Markers
colors = [
    0 0.4470 0.7410;      % 1:FFE (Blue)
    0.8500 0.3250 0.0980; % 2:VNLE (Red)
    0.9290 0.6940 0.1250; % 3:DFE (Yellow)
    0.4940 0.1840 0.5560; % 4:VDFE (Purple)
    0.4660 0.6740 0.1880; % 5:CLUT (Green)
    0.3010 0.7450 0.9330; % 6:FNN (Cyan)
    0.5 0.5 0.5;          % 7:RNN (Gray - AR)
    0 0 0                 % 8:WDRNN (Black - WD)
];
markers = {'o-', 's-', 'd-', '^-', 'v-', '>-', 'p-', 'h-'};
mk_size = 8;

h_plots = gobjects(length(algo_list), 1);
min_val = 1e-6; 

hold on;
for a = 1:length(algo_list)
    y_data = BERall(:, a);
    y_data(y_data == 0) = min_val; 
    y_log = log10(max(y_data, 1e-7)); 
    
    h_plots(a) = plot(rop_dBm, y_log, markers{a}, ...
        'Color', colors(a,:), 'LineWidth', 1.5, ...
        'MarkerSize', mk_size, 'MarkerFaceColor', colors(a,:));
end

% Limits Lines
yline(log10(3.8e-3), '--k', 'HD-FEC (3.8e-3)', 'LineWidth', 1.2, 'FontSize', 10);
yline(log10(2.4e-2), ':k', 'SD-FEC (2.4e-2)', 'LineWidth', 1.2, 'FontSize', 10);

grid on;
xlabel('Received Optical Power (dBm)', 'FontSize', 12, 'FontName', 'Arial');
ylabel('log10(BER)', 'FontSize', 12, 'FontName', 'Arial'); 
title('BER Performance vs ROP (200Gbps 2km)', 'FontSize', 14, 'FontName', 'Arial');
legend(h_plots, algo_list, 'Location', 'southwest', 'FontSize', 10, 'Interpreter', 'none');
ylim([-5, -0.5]); % Adjust based on data range

box on;
hold off;

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
        [off, d0] = align_offset_delay_by_ser(ye, xsym, NumPreamble_TDE, M, -60:60);
        if length(ye) > 1.5 * length(xsym)
            ye_use = ye(off:2:end);
        else
            ye_use = ye(:);
        end
        idxTx = (1:length(ye_use)).' + d0;
    end
end
