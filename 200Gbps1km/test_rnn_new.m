%% Quick Validation: WD-RNN vs AR-RNN vs FNN
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

%% Load Data
file_name = 'rop3dBm_1.mat';
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

%% --- 1. FS-FNN (Baseline) ---
disp('------------------------------------------------');
disp('Running FS-FNN (New Baseline)...');
tic;
[ye_fnn, ~, idx_fnn, d_fnn, p_fnn] = FNN_FS2pscenter(xRx, xTx, NumPreamble_TDE, 111, 32, 0.001, 30, -30:30, [0 1]);
t_fnn = toc;
stats_fnn = safe_eval_equalizer_pam4(ye_fnn(idx_fnn), idx_fnn, xsym, xm, NumPreamble_TDE, M);
disp(['FNN Time: ', num2str(t_fnn), 's, Delay: ', num2str(d_fnn)]);
disp(['FNN BER:  ', num2str(stats_fnn.BER)]);

%% --- 2. WD-RNN (Hard Feedback) ---
disp('------------------------------------------------');
disp('Running WD-RNN (Hard Feedback)...');
tic;
% InputLength=101, Hidden=32, k=25
[ye_wd, ~, idx_wd, d_wd, off_wd] = RNN_WD_Implementation(xRx, xTx, NumPreamble_TDE, 101, 32, 0.001, 30, 25, -30:30, [1 2]);
t_wd = toc;
stats_wd = safe_eval_equalizer_pam4(ye_wd(idx_wd), idx_wd, xsym, xm, NumPreamble_TDE, M);
disp(['WD-RNN Time: ', num2str(t_wd), 's, Delay: ', num2str(d_wd)]);
disp(['WD-RNN BER:  ', num2str(stats_wd.BER)]);

%% --- 3. AR-RNN (Soft Feedback) ---
disp('------------------------------------------------');
disp('Running AR-RNN (Soft Feedback)...');
tic;
% InputLength=61, Hidden=64, k=5 (Soft feedback needs shorter history usually)
[ye_ar, ~, idx_ar, d_ar, off_ar] = RNN_AR_Implementation(xRx, xTx, NumPreamble_TDE, 61, 64, 0.001, 20, 5, -30:30, [1 2]);
t_ar = toc;
stats_ar = safe_eval_equalizer_pam4(ye_ar(idx_ar), idx_ar, xsym, xm, NumPreamble_TDE, M);
disp(['AR-RNN Time: ', num2str(t_ar), 's, Delay: ', num2str(d_ar)]);
disp(['AR-RNN BER:  ', num2str(stats_ar.BER)]);

disp('------------------------------------------------');