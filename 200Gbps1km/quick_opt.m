%% Quick Optimization Test Framework (CLUT vs WDRNN)
% Targets: 4.6dBm (Battleground) and 5.6dBm (High Performance)
clear; close all;

%% Setup
addpath('fns'); addpath(fullfile('fns','fns2')); addpath('algos');

% Load Tx
if ~exist('xsym.mat', 'file'), error('Missing xsym.mat'); end
load('xsym.mat', 'xsym');
M=4; xsym=double(xsym); xm=pammod(xsym,M,0,'gray'); xs=xm(:);

% Pulse Shape Ref
rolloff=0.1; N=128; Osamp_factor=2;
sqrt_ht = rcosdesign(rolloff, N/Osamp_factor, Osamp_factor, 'normal');
sqrt_ht = sqrt_ht./max(sqrt_ht);
x_shape = conv(sqrt_ht, upsample(xs, Osamp_factor));

% Target Files
base_path = 'C:\Users\27456\Desktop\毕业设计';
test_files = {
    'PS_PAM4_2_RoF_0.1_Cal_45GHz_AWG_210mV_ROP_4.6dBm.mat', 4.6;
    'PS_PAM4_2_RoF_0.1_Cal_45GHz_AWG_210mV_ROP_5.6dBm.mat', 5.6
};

% Algorithm Params
params.N1=61; params.N2=7; params.WL=3; params.D1=11; params.D2=0; params.WD=1;
params.K_Lin=18; params.K_Vol=90; params.Lambda=0.9999; params.scale=M/2;

% RNN Params (To be optimized)
params.RNN_InputLength = 61;   % Try increasing?
params.RNN_HiddenSize = 32;    % Increased for stability
params.RNN_LR = 0.001;
params.RNN_Epochs = 100;       % Increased for convergence
params.RNN_k = 25;             % Feedback length
params.RNN_DelayCandidates = -20:20;
params.RNN_OffsetCandidates = [1 2];

%% Loop
fprintf('%-10s | %-10s | %-10s | %-10s\n', 'ROP', 'Algo', 'BER', 'Time');
fprintf('---------------------------------------------------\n');

for i = 1:size(test_files, 1)
    fname = test_files{i,1};
    rop = test_files{i,2};
    fpath = fullfile(base_path, fname);
    
    load(fpath, 'rx_original');
    
    % Prep Data (Seg 1)
    xp = rx_original(1,:)'; 
    xp = xp - mean(xp);
    ReData = resample(xp, 200e9/1e9, 256); % Assuming Ft=100G -> 200G/s
    
    % Sync
    [TE,~] = TEFEMMM2(ReData, 1024, 0.2);
    ysync = ReData(1024+20+TE+1 : end);
    ysync = ysync(1+N/2 : end);
    len_need = length(x_shape)-N;
    if length(ysync)<len_need, ysync=[ysync; zeros(len_need-length(ysync),1)]; end
    xRx = ysync(1:len_need);
    xTx = xs;
    
    % --- Run CLUT (Benchmark) ---
    tic;
    [~, ye] = CLUT_VDFE_Implementation(xRx, xTx, 10000, params.N1, params.N2, params.D1, params.D2, params.WL, params.WD, M, params.K_Lin, params.K_Vol, params.Lambda);
    t_clut = toc;
    [~, d0] = align_offset_delay_by_ser(ye, xsym, 10000, M, -60:60);
    ye_clut = ye(d0 + (1:length(ye)/2)'); % Approx indexing
    % Re-eval strictly
    stats_clut = eval_equalizer_pam4(ye, (1:length(ye)/2)'+d0, xsym, xm, 10000, M);
    fprintf('%-10.1f | %-10s | %-10.2e | %-10.2fs\n', rop, 'CLUT', stats_clut.BER, t_clut);
    
    % --- Run WDRNN (Target) ---
    tic;
    % Using params.RNN_... as WDRNN shares them in main script
    [ye_wd, ~, idxTx] = WDRNN_Implementation(xRx, xTx, 30000, ...
        params.RNN_InputLength, params.RNN_HiddenSize, params.RNN_LR, ...
        params.RNN_Epochs, params.RNN_k, params.RNN_DelayCandidates, params.RNN_OffsetCandidates);
    t_wd = toc;
    
    % Standard Eval
    stats_wd = eval_equalizer_pam4(ye_wd, idxTx, xsym, xm, 30000, M);
    
    fprintf('%-10.1f | %-10s | %-10.2e | %-10.2fs\n', rop, 'WDRNN', stats_wd.BER, t_wd);
    fprintf('---------------------------------------------------\n');
end
