%% FNN Comparison Test: Legacy vs New FS-FNN vs FFE
clear;
close all;

%% add paths
addpath('fns');
addpath(fullfile('fns','fns2'));

%% parameters
Ft = 200e9;
Osamp_factor = 2;
NumSymbols = 2^18; 
M = 4;

s = RandStream.create('mt19937ar', 'seed', 529558);
prevStream = RandStream.setGlobalStream(s);

%% PAM4 modulate
[xsym, xm] = PAMSource(M, NumSymbols);
xsym = xsym(:);
xm = xm(:);
xs = xm;

%% pulse shaping
rolloff = 0.1;
N = 128;
sqrt_ht = rcosdesign(rolloff, N/Osamp_factor, Osamp_factor, 'normal');
sqrt_ht = sqrt_ht ./ max(sqrt_ht);

x_upsamp = upsample(xs, Osamp_factor);
x_shape = conv(sqrt_ht, x_upsamp);
x_shape = x_shape ./ sqrt(mean(abs(x_shape).^2));

%% Test File: rop3dBm_1.mat (High SNR for clear comparison)
file_name = 'rop3dBm_1.mat';
disp(['Loading ', file_name, '...']);
if ~exist(file_name, 'file')
    error(['File not found: ', file_name]);
end
load(file_name, 'ReData');
ReData = -ReData;

%% Synchronization
th = 0.3;
[TE, FE] = TEFEMMM2(ReData, 1024, th);
ysync = ReData(1024 + 20 + TE : end);
ysync = ysync(1 : length(x_shape));

%% Match Filtering
yt_filter = ysync(1 + N/2 : length(ysync) - N/2);
xTx = xs;
xRx = yt_filter;

NumPreamble_TDE = 10000; 

%% --- 1. Baseline: FFE ---
disp('------------------------------------------------');
disp('Running FFE (Baseline)...');
N1 = 111;
Lambda = 0.9999;
tic;
[hffe, ye_ffe] = FFE_2pscenter(xRx, xTx, NumPreamble_TDE, N1, Lambda);
time_ffe = toc;

% FFE Alignment
[off_ffe, d0_ffe] = align_offset_delay_by_ser(ye_ffe, xsym, NumPreamble_TDE, M, -60:60);
if length(ye_ffe) > 1.5 * length(xsym)
    ye_use_ffe = ye_ffe(off_ffe:2:end);
else
    ye_use_ffe = ye_ffe(:);
end
idxTx_ffe = (1:length(ye_use_ffe)).' + d0_ffe;

stats_ffe = eval_equalizer_pam4(ye_use_ffe, idxTx_ffe, xsym, xm, NumPreamble_TDE, M);
disp(['FFE BER: ', num2str(stats_ffe.BER)]);


%% --- 2. Legacy FNN (Standard) ---
disp('------------------------------------------------');
disp('Running Legacy FNN (Standard)...');
InputLength = 111;
HiddenSize = 32; % Using the "Optimized" params from before
LearningRate = 0.001;
MaxEpochs = 50;

tic;
[ye_old, ~, idx_old] = FNN_Implementation(xRx, xTx, NumPreamble_TDE, ...
    InputLength, HiddenSize, LearningRate, MaxEpochs);
time_old = toc;

% Ensure idx is valid
idx_old = idx_old(:);
ye_old = ye_old(:);

stats_old = eval_equalizer_pam4(ye_old(idx_old), idx_old, xsym, xm, NumPreamble_TDE, M);
disp(['Legacy FNN BER: ', num2str(stats_old.BER)]);


%% --- 3. New FS-FNN (2sps) ---
disp('------------------------------------------------');
disp('Running New FS-FNN (2sps)...');
TapLen = 111;
HiddenSize_FS = 64; % Slightly larger for FS
LearningRate_FS = 0.001;
MaxEpochs_FS = 60;
DelayCandidates = -60:60;
PhaseCandidates = [0 1];

tic;
[ye_new, ~, idx_new, best_d, best_p] = FNN_FS2pscenter( ...
    xRx, xTx, NumPreamble_TDE, TapLen, HiddenSize_FS, LearningRate_FS, MaxEpochs_FS, DelayCandidates, PhaseCandidates);
time_new = toc;

disp(['New FS-FNN Found Delay: ', num2str(best_d), ', Phase: ', num2str(best_p)]);

idx_new = idx_new(:);
ye_new = ye_new(:);

stats_new = safe_eval_equalizer_pam4(ye_new, idx_new, xsym, xm, NumPreamble_TDE, M);
disp(['New FS-FNN BER: ', num2str(stats_new.BER)]);

disp('------------------------------------------------');
