%% Quick Optimization Test: WDRNN Only (4.6dBm & 5.6dBm)
clear;
close all;

%% Add Paths
addpath('fns');
addpath('algos');

%% Parameters
bbb = 100;
Ft = bbb*1e9;                       
Osamp_factor = 2;
M = 4;

% Seed
s = RandStream.create('mt19937ar', 'seed', 529551);
RandStream.setGlobalStream(s);

%% Load Tx
if exist('xsym.mat', 'file')
    load('xsym.mat', 'xsym');
else
    error('xsym.mat not found!');
end
xsym = double(xsym);
xm = pammod(xsym, M, 0, 'gray'); 
xs = xm(:); 

% Pulse Shaping Reference
rolloff = 0.1;
N = 128;
sqrt_ht = rcosdesign(rolloff, N/Osamp_factor, Osamp_factor, 'normal');
sqrt_ht = sqrt_ht ./ max(sqrt_ht);
x_upsamp = upsample(xs, Osamp_factor);
x_shape = conv(sqrt_ht, x_upsamp); 

%% Target Files (4.6dBm & 5.6dBm)
base_path = 'C:\Users\27456\Desktop\毕业设计';
file_names = { ...
    'PS_PAM4_2_RoF_0.1_Cal_45GHz_AWG_210mV_ROP_4.6dBm.mat', ...
    'PS_PAM4_2_RoF_0.1_Cal_45GHz_AWG_210mV_ROP_5.6dBm.mat' ...
};
rop_vals = [4.6, 5.6];

%% Algos to Compare
algo_list = {'WDRNN'};

%% Params
% WDRNN Params (Aligned with Roll 8)
params.RNN_InputLength = 61;
params.RNN_HiddenSize = 20;   
params.RNN_LR = 0.001;
params.RNN_Epochs = 50;       
params.RNN_k = 25; 
params.RNN_DelayCandidates = -20:20;
params.RNN_OffsetCandidates = [1 2];

%% Loop
fprintf('----------------------------------------------------------------\n');
fprintf('Quick Optimization Run (WDRNN Only): %s\n', datestr(now));
fprintf('----------------------------------------------------------------\n');

for n = 1:length(file_names)
    fname = file_names{n};
    fpath = fullfile(base_path, fname);
    
    fprintf('\n>>> Processing: %s (ROP=%.1fdBm)\n', fname, rop_vals(n));
    
    if ~exist(fpath, 'file')
        warning('File not found: %s', fpath);
        continue;
    end
    load(fpath, 'rx_original');
    
    % Preprocessing
    xp = rx_original(1,:); xp = xp(:) - mean(xp);
    ReData = resample(xp, 2*Ft/1e9, 256);
    [TE, ~] = TEFEMMM2(ReData, 1024, 0.2);
    ysync = ReData(1024 + 20 + TE + 1 : end);
    ysync = ysync(1 + N/2 : end);
    len_needed = length(x_shape) - N;
    if length(ysync) < len_needed, ysync = [ysync; zeros(len_needed-length(ysync),1)]; end
    xRx = ysync(1:len_needed);
    xTx = xs;
    
    NumPreamble = 30000; % Default for RNN/WDRNN
    
    for a = 1:length(algo_list)
        algo = algo_list{a};
        
        try
            tic;
            [ye, idxTx] = run_equalizer(algo, xRx, xTx, xsym, NumPreamble, M, params);
            t = toc;
            stats = eval_equalizer_pam4(ye, idxTx, xsym, xm, NumPreamble, M);
            fprintf('  %-15s | Time: %6.2fs | BER: %.2e | SNR: %.2fdB\n', algo, t, stats.BER, stats.SNRdB);
        catch ME
            fprintf('  %-15s | FAILED: %s\n', algo, ME.message);
        end
    end
end
fprintf('\nDone.\n');

%% Helper (Inline)
function [ye_use, idxTx] = run_equalizer(algo_id, xRx, xTx, xsym, NumPreamble, M, params)
    switch upper(algo_id)
        case 'WDRNN'
             [ye, ~, valid_idx] = WDRNN_Implementation(xRx, xTx, NumPreamble, ...
                 params.RNN_InputLength, params.RNN_HiddenSize, params.RNN_LR, ...
                 params.RNN_Epochs, params.RNN_k, params.RNN_DelayCandidates, ...
                 params.RNN_OffsetCandidates);
             ye_use = ye(valid_idx); 
             idxTx = valid_idx;
        otherwise
            error('Unknown algorithm: %s', algo_id);
    end
end