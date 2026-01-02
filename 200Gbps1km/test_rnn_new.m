%% Quick Validation: Linear vs RNNs (Standard JLT Config)
clear;
close all;

%% add paths
addpath('fns');
addpath(fullfile('fns','fns2'));

%% parameters
NumSymbols = 2^18; 
M = 4;
NumPreamble_TDE = 10000;

s = RandStream.create('mt19937ar', 'seed', 529558);
prevStream = RandStream.setGlobalStream(s);

%% PAM4 modulate
[xsym, xm] = PAMSource(M, NumSymbols);
xs = xm;

%% pulse shaping
rolloff = 0.1;
Osamp_factor = 2;
N = 128;
sqrt_ht = rcosdesign(rolloff, N/Osamp_factor, Osamp_factor, 'normal');
sqrt_ht = sqrt_ht ./ max(sqrt_ht);
x_upsamp = upsample(xs, Osamp_factor);
x_shape = conv(sqrt_ht, x_upsamp);
x_shape = x_shape ./ sqrt(mean(abs(x_shape).^2));

%% Load Data (Use 2dBm for clear SNR distinction)
file_name = 'rop2dBm_1.mat';
disp(['Loading ', file_name, '...']);
if ~exist(file_name, 'file'), error('File not found'); end
load(file_name, 'ReData');
ReData = -ReData;

%% Sync
th = 0.3;
[TE, FE] = TEFEMMM2(ReData, 1024, th);
ysync = ReData(1024 + 20 + TE : end);
ysync = ysync(1 : length(x_shape));
yt_filter = ysync(1 + N/2 : length(ysync) - N/2);

xTx = xs;
xRx = yt_filter;

%% --- 0. Linear FFE+DFE (Baseline) ---
disp('------------------------------------------------');
disp('Running LE_FFE_DFE (Baseline)...');
tic;
% N1=111, D1=25 (Standard configs)
params.N1 = 111; params.D1 = 25; params.Lambda = 0.9999; params.scale = M/2;
[~, ~, ye_lin] = LE_FFE2ps_centerDFE_new(xRx, xTx, NumPreamble_TDE, params.N1, params.D1, params.Lambda, M, params.scale);
t_lin = toc;
% For linear, we need to align manually if the function doesn't return indices
% But LE_FFE2ps_centerDFE_new usually returns aligned ye.
% Let's use eval helper which handles alignment if needed or we use a safe wrapper.
% The standard run_equalizer logic in roll_7 does manual alignment for linear.
[off, d0] = align_offset_delay_by_ser(ye_lin, xsym, NumPreamble_TDE, M, -60:60);
if length(ye_lin) > 1.5 * length(xsym)
    ye_lin_use = ye_lin(off:2:end);
else
    ye_lin_use = ye_lin(:);
end
idx_lin = (1:length(ye_lin_use)).' + d0;

stats_lin = eval_equalizer_pam4(ye_lin_use, idx_lin, xsym, xm, NumPreamble_TDE, M);
disp(['Linear Time: ', num2str(t_lin), 's']);
disp(['Linear BER:  ', num2str(stats_lin.BER)]);

%% --- 0.5. FNN (No Feedback) ---
disp('------------------------------------------------');
disp('Running FNN (No Feedback)...');
tic;
[ye_fnn, ~, idx_fnn, d_fnn, p_fnn] = FNN_FS2pscenter(xRx, xTx, NumPreamble_TDE, 101, 64, 0.001, 30, -20:20, [1 2]);
t_fnn = toc;
stats_fnn = eval_equalizer_pam4(ye_fnn(idx_fnn), idx_fnn, xsym, xm, NumPreamble_TDE, M);
disp(['FNN Time: ', num2str(t_fnn), 's']);
disp(['FNN BER:  ', num2str(stats_fnn.BER)]);

%% --- 1. WD-RNN (Hard Feedback) ---
disp('------------------------------------------------');
disp('Running WD-RNN (Hard Feedback)...');
tic;
% Residual Mode: Input=101, Hidden=64, k=2, Noise=0.05
[ye_wd, ~, idx_wd, d_wd, off_wd] = RNN_WD_Implementation(xRx, xTx, NumPreamble_TDE, 101, 64, 0.001, 30, 2, -20:20, [1 2], 0.05);
t_wd = toc;
stats_wd = eval_equalizer_pam4(ye_wd(idx_wd), idx_wd, xsym, xm, NumPreamble_TDE, M);
disp(['WD-RNN Time: ', num2str(t_wd), 's']);
disp(['WD-RNN BER:  ', num2str(stats_wd.BER)]);

%% --- 2. AR-RNN (Soft Feedback) ---
disp('------------------------------------------------');
disp('Running AR-RNN (Soft Feedback)...');
tic;
% Increased Capacity: Input=101, Hidden=64, k=15
[ye_ar, ~, idx_ar, d_ar, off_ar] = RNN_AR_Implementation(xRx, xTx, NumPreamble_TDE, 101, 64, 0.001, 30, 15, -20:20, [1 2]);
t_ar = toc;
stats_ar = eval_equalizer_pam4(ye_ar(idx_ar), idx_ar, xsym, xm, NumPreamble_TDE, M);
disp(['AR-RNN Time: ', num2str(t_ar), 's']);
disp(['AR-RNN BER:  ', num2str(stats_ar.BER)]);

disp('------------------------------------------------');
disp('Summary (BER @ 2dBm):');
disp(['Linear FFE+DFE: ', num2str(stats_lin.BER)]);
disp(['FNN (No FB):    ', num2str(stats_fnn.BER)]);
disp(['RNN_WD (Hard):  ', num2str(stats_wd.BER)]);
disp(['RNN_AR (Soft):  ', num2str(stats_ar.BER)]);
disp('------------------------------------------------');